#!/usr/bin/env python
"""Re-score med-dx runs under EviDx's Level 3 metrics.

Predictions and retrievals are reused; only the judge is paid. A run's
`scores.jsonl` is replaced (the old file is kept beside it with a timestamp)
and `summary.json` is rebuilt from the new rows.

Acquisition needs every tool output the run saw, which the runner keeps under
`runs/<id>/retrievals/<task>/r*.json`; runs from before that directory existed
score acquisition on the initial narrative alone, and the row says so.

    python scripts/rescore_med_dx.py runs/med-dx__* --evidx-root scratch/EviDx
    python scripts/rescore_med_dx.py runs/med-dx__* --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sheldon.core.io import load_dotenv  # noqa: E402

# Before any client exists: the ambient key on this box is dead and only .env works.
load_dotenv()

from sheldon.benchmarks.med_dx import MedDx  # noqa: E402
from sheldon.core.budget import Budget  # noqa: E402
from sheldon.core.cache import DiskCache  # noqa: E402
from sheldon.core.config import Config  # noqa: E402
from sheldon.core.io import read_jsonl, rewrite_jsonl, write_json  # noqa: E402
from sheldon.core.llm import LLM  # noqa: E402
from sheldon.core.trace import NullTrace  # noqa: E402
from sheldon.core.types import Prediction, RunContext  # noqa: E402
from sheldon.runner import summarize  # noqa: E402


def replayed_outputs(run: Path, task_id: str) -> list[str] | None:
    """Tool outputs in call order, or None when the run kept no retrievals."""
    folder = run / "retrievals" / task_id
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("r*.json"), key=lambda p: int(p.stem[1:]))
    out: list[str] = []
    for f in files:
        entry = json.loads(f.read_text(encoding="utf-8"))
        if (
            entry.get("tool") == "recall"
        ):  # reads the index; adds nothing the run did not already see
            continue
        out.append(str(entry.get("summary", "")))
    return out


async def rescore(
    run: Path, *, evidx_root: str | None, judge_model: str | None, jobs: int, write: bool
) -> dict[str, Any]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))["config"]
    if config["benchmark"] not in ("med-dx", "med", "medical"):
        raise SystemExit(f"{run.name}: not a med-dx run")
    opts = dict((config.get("params") or {}).get("benchmark_opts") or {})
    if evidx_root:
        opts["evidx_root"] = evidx_root
    if judge_model:
        opts["judge_model"] = judge_model
    bench = MedDx(**opts)
    problems = bench.check()
    if problems:
        raise SystemExit(f"{run.name}: {'; '.join(problems)}")
    tasks = {t.id: t for t in bench.load(config.get("split", "DiagnosisArena"))}
    preds = {r["task_id"]: r for r in read_jsonl(run / "predictions.jsonl")}
    cfg = Config(**config)
    llm = LLM(
        provider=cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
        thinking=cfg.thinking,
        cache=DiskCache(enabled=True),
        concurrency=max(jobs, 1),
    )
    sem = asyncio.Semaphore(max(1, jobs))
    rows: dict[str, dict[str, Any]] = {}
    no_retrievals: list[str] = []

    async def one(task_id: str) -> None:
        task = tasks[task_id]
        outputs = replayed_outputs(run, task_id)
        if outputs is None:
            no_retrievals.append(task_id)
            outputs = []
        task.payload["_tool_outputs"] = outputs
        ctx = RunContext(
            llm=llm,
            budget=Budget(max_rounds=1, max_usd=None),
            trace=NullTrace(),
            benchmark=bench,
            tools=None,
            params=cfg.params,
        )
        async with sem:
            score = await bench.score(task, Prediction.from_dict(preds[task_id]), ctx)
        row = score.to_dict()
        if task_id in no_retrievals:
            row.setdefault("detail", {})["no_retrievals"] = True
        row["rescored"] = "evidx-level3"
        rows[task_id] = row

    todo = [tid for tid, p in preds.items() if not p.get("error") and tid in tasks]
    await asyncio.gather(*(one(tid) for tid in todo))
    old = (
        {r["task_id"]: r for r in read_jsonl(run / "scores.jsonl")}
        if (run / "scores.jsonl").exists()
        else {}
    )
    if write:
        # Copy, not rename: `rewrite_jsonl` replaces an existing file and
        # refuses to create one, so the old rows must still be there when it runs.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        scores = run / "scores.jsonl"
        if scores.exists():
            shutil.copy2(scores, run / f"scores.jsonl.pre-level3-{stamp}")
        else:
            scores.write_text("", encoding="utf-8")
        rewrite_jsonl(
            scores,
            [rows.get(tid, old.get(tid)) for tid in preds if rows.get(tid) or old.get(tid)],
        )
        write_json(run / "summary.json", summarize(cfg, run))
    await llm.aclose()
    n = len(rows)
    mean = lambda k: (sum(float(r["metrics"].get(k, 0.0)) for r in rows.values()) / n) if n else 0.0  # noqa: E731
    return {
        "run": run.name,
        "n": n,
        "no_retrievals": len(no_retrievals),
        "open": mean("open"),
        "open_strict": mean("open_strict"),
        "mc": mean("mc"),
        "acq": mean("acq"),
        "cog": mean("cog"),
        "old_score": (sum(float(r.get("score", 0.0)) for r in old.values()) / len(old))
        if old
        else None,
        "judge_usd": llm.billed.cost_usd if hasattr(llm, "billed") else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument(
        "--evidx-root",
        default=None,
        help="override benchmark_opts.evidx_root (old runs point at another user's checkout)",
    )
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--jobs", type=int, default=4, help="judge calls in flight")
    ap.add_argument("--dry-run", action="store_true", help="score, print, write nothing")
    args = ap.parse_args()
    for run in args.runs:
        res = asyncio.run(
            rescore(
                run,
                evidx_root=args.evidx_root,
                judge_model=args.judge_model,
                jobs=args.jobs,
                write=not args.dry_run,
            )
        )
        print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
