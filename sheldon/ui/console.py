"""A browsable view of `runs/` — every run, every task, every model call.

`sheldon show` answers "what happened in this one task". This answers the
question that actually comes up: *why did this method lose the task that one
won?* So runs are addressable side by side and a task id is the thing they are
compared on, not a detail inside one of them.

Self-contained HTML with the data embedded, so it works from a file:// URL with
no server and no network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheldon.core.io import read_jsonl, runs_root

#: Guard against embedding a sweep's worth of traces into one page. Traces run
#: about 12 KB per task with prompts on, so a wide sweep is easily hundreds of
#: megabytes; the newest runs are kept and the rest are listed as omitted.
DEFAULT_MAX_MB = 12.0

_BASELINE_ORDER = {"direct": 0, "best-of-n": 1}


def collect(
    root: Path | None = None,
    *,
    benchmark: str | None = None,
    tag: str | None = None,
    max_mb: float = DEFAULT_MAX_MB,
    embed: bool = True,
) -> dict[str, Any]:
    """Gather runs, with their traces if they are to be embedded.

    `embed=False` builds the index only — every run, every task, scores and
    timings, but no trace bodies and no belief snapshots. Those are fetched per
    task when one is opened. The whole index is about 9 MB where the traces are
    about 149 MB, so this is the difference between a page that holds every run
    and one that holds eight tasks of one.

    It costs a server, which is why embedding stays the default: an embedded
    page opens from a `file://` URL and can be handed to someone, and a fetching
    one cannot.
    """
    root = root or runs_root()
    runs: list[dict[str, Any]] = []
    omitted: list[str] = []

    wanted: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in sorted(
        root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not summary.get("n"):
            continue
        if benchmark and summary.get("benchmark") != benchmark:
            continue
        if tag is not None and summary.get("tag", "") != tag:
            continue
        wanted.append((summary_path.parent, summary))

    # Every run is listed whatever the budget, because a run you cannot see is
    # a run you cannot compare against — and the scores and task lists are
    # small. The budget buys *traces*, shared newest-first so one sweep with
    # full prompts cannot swallow the page; whatever a run leaves unspent rolls
    # forward to the next.
    budget = int(max_mb * 1024 * 1024)
    share = budget // max(1, len(wanted))
    allowance = 0
    for d, summary in wanted:
        run = {k: summary.get(k) for k in (
            "run_id", "benchmark", "method", "split", "tag", "model", "n", "score",
            "stderr", "errors", "mean_rounds", "cost_per_task", "cost_usd", "wall_s",
            "latency_mean_s", "latency_p90_s", "calls_per_task", "concurrency",
            "started_at", "finished_at", "git_sha",
        )}
        if embed:
            allowance += share
            run["tasks"], spent = _tasks(d, allowance)
            allowance -= spent
            if any(not t["events"] for t in run["tasks"]):
                omitted.append(d.name)
        else:
            run["tasks"], _ = _tasks(d, 0)
            for t in run["tasks"]:
                # null, not []: an empty array is truthy in JavaScript, so the
                # page would read "already loaded" and never fetch.
                t["events"] = None
        runs.append(run)

    runs.sort(key=lambda r: (
        r.get("benchmark") or "", _BASELINE_ORDER.get(r.get("method"), 99),
        r.get("method") or "", r.get("tag") or "",
    ))
    return {
        "generated": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "root": str(root),
        "runs": runs,
        #: False when trace bodies are fetched per task instead of embedded.
        "embedded": embed,
        #: runs whose traces were partly or wholly left out for size. The runs
        #: themselves are always present.
        "omitted": omitted,
    }


def _tasks(d: Path, budget: int) -> tuple[list[dict[str, Any]], int]:
    scores = {s["task_id"]: s for s in _safe_rows(d / "scores.jsonl")}
    preds = {p["task_id"]: p for p in _safe_rows(d / "predictions.jsonl")}

    out: list[dict[str, Any]] = []
    spent = 0
    # Worst first: the failures are what a trace viewer is opened for.
    for tid in sorted(scores, key=lambda t: (scores[t].get("score", 0), t)):
        s, p = scores[tid], preds.get(tid, {})
        events: list[dict[str, Any]] = []
        trace = d / "traces" / f"{tid}.jsonl"
        if trace.exists() and spent < budget:
            events = list(_safe_rows(trace))
            spent += trace.stat().st_size
        # Both bodies travel together: a task is opened, or it is not.
        belief = p.get("belief") if events else None
        out.append({
            "id": tid,
            "score": s.get("score", 0.0),
            "metrics": s.get("metrics") or {},
            "rounds": p.get("rounds"),
            "wall_s": p.get("wall_s"),
            "error": p.get("error"),
            "stop_reason": (p.get("extra") or {}).get("stop_reason"),
            "belief": belief,
            "events": events,
        })
    return out, spent


def _safe_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path.exists() else []


def build(data: dict[str, Any]) -> str:
    """The page, with the data baked in."""
    template = (Path(__file__).with_name("console.html")).read_text(encoding="utf-8")
    # ASCII-escaped on purpose: traces are model-written text full of em dashes
    # and multiplication signs, and `\uXXXX` decodes correctly whatever the
    # document's encoding turns out to be.
    payload = json.dumps(data, default=str, ensure_ascii=True)
    # A literal </script> inside embedded JSON ends the block early, and a trace
    # of model-written code is exactly where one would turn up.
    payload = payload.replace("</", "<\\/")
    return template.replace("__DATA__", payload)


def write(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(data), encoding="utf-8")
    return path



def task_payload(run_id: str, task_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """One task's trace and belief snapshot, read on demand.

    `runs/<run_id>/traces/<task_id>.jsonl` is already keyed by exactly this
    lookup, which is why no database is involved. Both ids are checked against
    the runs directory rather than trusted: they arrive from a URL, and
    `../` in either would otherwise read anything the process can.
    """
    base = (root or runs_root()).resolve()
    run_dir = (base / run_id).resolve()
    if not run_dir.is_dir() or run_dir.parent != base:
        return None
    trace = (run_dir / "traces" / f"{task_id}.jsonl").resolve()
    if trace.parent != (run_dir / "traces").resolve():
        return None

    events = _safe_rows(trace)
    belief = None
    for row in _safe_rows(run_dir / "predictions.jsonl"):
        if row.get("task_id") == task_id:
            belief = row.get("belief")
            break
    return {"id": task_id, "events": events, "belief": belief}
