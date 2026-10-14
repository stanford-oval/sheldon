#!/usr/bin/env python
"""Re-score finished `dockstring` runs on the best molecule they measured.

Scoring moved from "the molecule handed in" to "the best molecule measured",
which is what the paper and every optimizer baseline report. Runs finished
before that change carry the old number, and a table mixing the two compares
nothing. This recovers each task's measurements from its trace — every
`dock_molecule` result is in there verbatim — and rewrites `scores.jsonl` and
`summary.json` through the benchmark's own scorer, so old and new runs are
judged by identical code.

    scripts/rescore_dockstring.py runs/dockstring__* --dry-run   # look first
    scripts/rescore_dockstring.py runs/dockstring__*             # then write

A run whose method never docked anything (the `codex` engine cannot reach
`ctx.tools`) re-scores to exactly what it scored before, which is the check that
this is a re-scoring and not a re-interpretation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sheldon.benchmarks.dockstring import Dockstring  # noqa: E402
from sheldon.core.io import read_jsonl, rewrite_jsonl, write_json  # noqa: E402
from sheldon.core.types import Prediction  # noqa: E402

#: The metrics line `dock_molecule` writes into every trace.
_MEASUREMENT = re.compile(
    r"vina=(-?[\d.]+) kcal/mol  qed=([\d.]+)  s=(-?[\d.]+)  \[(\w+)\]"
)


def measurements(trace: Path) -> dict[str, dict[str, Any]]:
    """Every molecule this task docked, recovered from its trace."""
    out: dict[str, dict[str, Any]] = {}
    if not trace.exists():
        return out
    for row in read_jsonl(trace):
        if row.get("kind") != "tool" or row.get("probe") != "dock_molecule":
            continue
        if not row.get("ok"):
            continue
        lines = (row.get("summary") or "").split("\n")
        if len(lines) < 2:
            continue
        found = _MEASUREMENT.fullmatch(lines[1])
        if not found:
            continue
        vina, qed, objective, source = found.groups()
        out[lines[0]] = {
            "smiles": lines[0],
            "vina": float(vina),
            "qed": float(qed),
            "objective": float(objective),
            "source": source,
        }
    return out


async def rescore(run: Path, write: bool) -> dict[str, Any] | None:
    config = json.loads((run / "config.json").read_text())["config"]
    if config["benchmark"] != "dockstring":
        return None
    opts = dict((config.get("params") or {}).get("benchmark_opts") or {})
    bench = Dockstring(**opts)

    tasks = {t.id: t for t in bench.load(config.get("split", "eval"))}
    rows: list[dict[str, Any]] = []
    for pred_row in read_jsonl(run / "predictions.jsonl"):
        task = tasks.get(pred_row["task_id"])
        if task is None:
            continue
        # Seed the benchmark's registry with what the run actually measured.
        bench._measured[task.id] = measurements(run / "traces" / f"{task.id}.jsonl")
        score = await bench.score(task, Prediction.from_dict(pred_row), None)
        rows.append(score.to_dict())

    if write and rows:
        rewrite_jsonl(run / "scores.jsonl", rows)
        summary = json.loads((run / "summary.json").read_text())
        scored = [r["score"] for r in rows]
        summary["score"] = round(sum(scored) / len(scored), 4)
        write_json(run / "summary.json", summary)
    return {"run": run.name, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for run in sorted(args.runs):
        if not (run / "config.json").exists():
            continue
        out = asyncio.run(rescore(run, write=not args.dry_run))
        if out is None:
            continue
        print(f"\n{out['run']}")
        print(f"  {'target':<8}{'scored':>9}{'submitted':>11}{'loss':>7}{'bar':>7}"
              f"{'pct':>10}{'docked':>8}  pass  on")
        for row in sorted(out["rows"], key=lambda r: -r["metrics"].get("objective", -99)):
            m, d = row["metrics"], row["detail"]
            if not m.get("valid"):
                print(f"  {row['task_id']:<8}{'invalid':>9}   {d.get('reason', '')[:40]}")
                continue
            print(f"  {row['task_id']:<8}{m['objective']:>9.2f}{m['submitted_objective']:>11.2f}"
                  f"{m['finalize_loss']:>7.2f}{d['bar']:>7.2f}{m['percentile']:>10.3f}"
                  f"{m['molecules_measured']:>8.0f}  {int(row['score']):>4}  {d['scored_on']}")
        scored = [r["score"] for r in out["rows"]]
        if scored:
            print(f"  {'mean':<8}{sum(scored) / len(scored):>9.3f}"
                  f"{'  (pass rate)':>24}")
    if args.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
