#!/usr/bin/env python
"""Re-score ORCA-bench runs whose verifier failed, or all of them.

Upstream's `check_prediction.py` is a subprocess that makes one judge call per
rubric, and on a saturated endpoint it can outlive the harness's timeout. The
runner then records `score=0.0` with `detail.score_error` -- a row without an
`error` on its prediction, so resume never retries it and the zero sits in the
mean. The report itself is intact, so the fix is to score it again, which
costs only the judge. This is the path for re-grading
without re-running the method.

    scripts/rescore_orca.py runs/orca-bench__* --dry-run   # which rows, and why
    scripts/rescore_orca.py runs/orca-bench__*             # re-score failed rows
    scripts/rescore_orca.py runs/orca-bench__* --all       # re-score every row

Two things this has to get right. `OrcaBench.load` under the upstream interface
*unlinks* every task's report file, so that a stale file is never graded as a
new run's answer -- which is exactly the file this script needs, so the tasks
are loaded through an instance that does not know the instruction template and
scoring goes through one that does. And `.env` must be loaded before the judge
subprocess is built, or it inherits a stale ambient key and every task comes
back 0.0 with a 401 that reads like a broken verifier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sheldon.core.io import load_dotenv  # noqa: E402

# The harness's loader, not python-dotenv's: `.env` beats the ambient
# environment here. With the library default the shell's exported key won,
# and that key was over its budget at the proxy -- the verifier caught the
# 429 itself and wrote reward 0.0 with the error inside `nested`, so a row
# that had scored 0.667 was quietly re-scored to zero with no failure marked.
load_dotenv()

from sheldon.benchmarks.orca_bench import OrcaBench  # noqa: E402
from sheldon.core.config import Config  # noqa: E402
from sheldon.core.io import read_jsonl, rewrite_jsonl, write_json  # noqa: E402
from sheldon.core.types import Prediction  # noqa: E402
from sheldon.runner import summarize  # noqa: E402


def verifier_error(score_row: dict[str, Any] | None) -> str | None:
    """The failure behind a row's score, whichever layer recorded it.

    Three places, none of them the score itself: the harness's `score_error`
    (the subprocess raised, usually a timeout), its `verifier_failed` (no
    reward file), and upstream's own `nested.error` -- the verifier catches
    its judge's exception and still writes a reward of 0.0, so a 429 or a
    401 looks exactly like a report that earned nothing.
    """
    if score_row is None:
        return "no score row"
    detail = score_row.get("detail") or {}
    if detail.get("score_error"):
        return "score_error: " + str(detail["score_error"])[-80:]
    if detail.get("verifier_failed"):
        return "verifier_failed: " + str(detail["verifier_failed"])[-80:]
    nested = detail.get("nested")
    if isinstance(nested, dict) and nested.get("error"):
        return "verifier error: " + str(nested["error"])[:80]
    if detail.get("grader") == "upstream" and not (score_row.get("metrics") or {}):
        return "upstream grader returned no metrics"
    return None


def needs_rescore(score_row: dict[str, Any] | None) -> str | None:
    """Why a row should be scored again, or None if it stands."""
    return verifier_error(score_row)


async def rescore(
    run: Path, *, write: bool, everything: bool, jobs: int, overrides: dict[str, str] | None = None
) -> dict[str, Any] | None:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))["config"]
    if config["benchmark"] not in ("orca-bench", "orca"):
        return None
    opts = dict((config.get("params") or {}).get("benchmark_opts") or {})
    opts.update(overrides or {})
    if not (opts.get("verifier") and opts.get("rubrics_dir")):
        raise SystemExit(f"{run.name}: not graded by upstream's verifier; nothing to re-score")

    scorer = OrcaBench(**opts)
    # Loads the same rows and gold, and never touches a report file: without
    # the template `upstream_instruction` returns None before its unlink.
    loader = OrcaBench(**{**opts, "instruction_template": None})
    tasks = {t.id: t for t in loader.load(config.get("split", "public"))}

    preds = {r["task_id"]: r for r in read_jsonl(run / "predictions.jsonl")}
    scores_path = run / "scores.jsonl"
    old = {r["task_id"]: r for r in read_jsonl(scores_path)} if scores_path.exists() else {}

    todo: list[tuple[str, str]] = []
    for task_id, pred in preds.items():
        if pred.get("error"):
            continue  # never answered; resume's job, not the judge's
        if task_id not in tasks:
            continue
        why = "requested" if everything else needs_rescore(old.get(task_id))
        if why:
            todo.append((task_id, why))

    sem = asyncio.Semaphore(max(1, jobs))
    new: dict[str, dict[str, Any]] = {}

    async def one(task_id: str) -> None:
        async with sem:
            score = await scorer.score(tasks[task_id], Prediction.from_dict(preds[task_id]), None)
            new[task_id] = score.to_dict()

    if write and todo:
        await asyncio.gather(*(one(task_id) for task_id, _ in todo))
        rows = [new.get(tid, old.get(tid)) for tid in preds if new.get(tid) or old.get(tid)]
        rewrite_jsonl(scores_path, rows)
        write_json(run / "summary.json", summarize(Config(**config), run))

    return {"run": run.name, "todo": todo, "new": new, "old": old}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--all", action="store_true", help="re-score every answered task")
    ap.add_argument("--jobs", type=int, default=2, help="judge subprocesses at once")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--opt", action="append", default=[], metavar="K=V",
                    help="override a benchmark_opts entry for scoring, e.g. judge_model=gpt-5.4 "
                         "(repeatable); the run's own config supplies the rest")
    args = ap.parse_args()

    for run in sorted(args.runs):
        if not (run / "config.json").exists():
            continue
        out = asyncio.run(rescore(
            run, write=not args.dry_run, everything=args.all, jobs=args.jobs,
            overrides=dict(kv.split("=", 1) for kv in args.opt),
        ))
        if out is None:
            continue
        print(f"\n{out['run']}: {len(out['todo'])} row(s) to re-score")
        for task_id, why in out["todo"]:
            before = (out["old"].get(task_id) or {}).get("score")
            after = (out["new"].get(task_id) or {}).get("score")
            after_s = f"{after:.3f}" if isinstance(after, (int, float)) else "-"
            still = verifier_error(out["new"].get(task_id)) if task_id in out["new"] else None
            print(f"  {task_id:<28} {before!s:>6} -> {after_s:>6}  {why[:70]}"
                  + (f"  STILL FAILING ({still[:60]})" if still else ""))
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
