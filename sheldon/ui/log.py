"""runs/ -> a browsable log of every model call.

`sheldon ui` answers "what did this method do on this task" — hypotheses,
observations, stage events. This answers the question underneath it: **what was actually
said**. Every `llm` event in a trace carries the whole exchange — the system
prompt, the full message list, the tools that were on offer, the reply verbatim,
the tool calls with their arguments, and the cache key — and none of that is
readable anywhere today without `jq` over a JSONL file.

That gap is where the expensive mistakes live. A stage that "logged 8
observations" tells you a call happened; only the call itself tells you it was
asked the wrong question.

Same split as `sheldon ui`: the index is small and always complete, and the call
bodies are fetched per task when opened. One task can be megabytes of
prompts, so embedding every run's calls would be a page nobody can load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sheldon.core.io import read_jsonl, runs_root

#: Fields of an `llm` trace event that make up one call's record.
_CALL_FIELDS = (
    "tag", "model", "cached", "key", "stop_reason", "text", "usage",
    "tool_use", "system", "messages", "tools_offered",
)


def collect(
    root: Path | None = None,
    *,
    benchmark: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """The index: every run, its tasks, and how many calls each holds.

    Deliberately does not read message bodies. Counting calls needs one pass
    over each trace, which is cheap; rendering them does not belong here.
    """
    root = Path(root or runs_root())
    runs: list[dict[str, Any]] = []
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
        directory = summary_path.parent
        run = {k: summary.get(k) for k in (
            "run_id", "benchmark", "method", "split", "tag", "model", "n", "score",
        )}
        run["dir"] = directory.name
        run["tasks"] = _task_index(directory)
        run["calls"] = sum(t["calls"] for t in run["tasks"])
        runs.append(run)

    runs.sort(key=lambda r: (
        r.get("benchmark") or "", r.get("method") or "", r.get("tag") or ""
    ))
    return {"root": str(root), "runs": runs}


def _task_index(directory: Path) -> list[dict[str, Any]]:
    """One row per task, built without opening a single trace.

    Traces run to tens of megabytes per task, so anything that reads
    them to build the index makes the command unusable — the first version
    counted `llm` events by scanning every line and never finished. Everything
    needed is already summarised: `usage.calls` in predictions.jsonl *is* the
    call count, cost sits beside it, and the trace directory listing says which
    tasks have a log at all. Bodies are read when a task is opened.
    """
    scores = {
        row["task_id"]: row.get("score")
        for row in _rows(directory / "scores.jsonl")
        if row.get("task_id")
    }
    traced = {path.stem for path in (directory / "traces").glob("*.jsonl")}
    tasks: list[dict[str, Any]] = []
    for row in _rows(directory / "predictions.jsonl"):
        task_id = row.get("task_id")
        if not task_id or task_id not in traced:
            continue
        usage = row.get("usage") or {}
        tasks.append({
            "task_id": task_id,
            "calls": int(usage.get("calls") or 0),
            "usd": round(float(usage.get("cost_usd") or 0.0), 4),
            "score": scores.get(task_id),
            "error": row.get("error"),
        })
    # Worst first: the task you want to read is almost never the one that
    # worked. Errored and unscored tasks sort to the top, since a task that
    # never answered is the most likely reason anyone opened this page.
    tasks.sort(key=lambda t: (
        t["error"] is None, t["score"] is None,
        t["score"] if t["score"] is not None else 0, t["task_id"],
    ))
    return tasks


def _rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file, tolerating a truncated last line.

    A run killed mid-write leaves one, and refusing to render the other 200
    calls because of it would make this useless exactly when it is needed.
    """
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for row in read_jsonl(path):
            out.append(row)
    except json.JSONDecodeError:
        pass
    return out


def calls_payload(
    run_dir: str, task_id: str, root: Path | None = None
) -> dict[str, Any] | None:
    """Every model call for one task, in order, with both sides in full."""
    root = Path(root or runs_root())
    trace = root / run_dir / "traces" / f"{task_id}.jsonl"
    if not trace.exists() or ".." in run_dir or ".." in task_id:
        return None

    calls: list[dict[str, Any]] = []
    for event in _rows(trace):
        if event.get("kind") != "llm":
            continue
        call = {k: event.get(k) for k in _CALL_FIELDS}
        call["t"] = event.get("t")
        call["index"] = len(calls) + 1
        calls.append(call)
    return {"run": run_dir, "task_id": task_id, "calls": calls}


def build(data: dict[str, Any]) -> str:
    """The page, with the index baked in."""
    template = (Path(__file__).with_name("log.html")).read_text(encoding="utf-8")
    # ASCII-escaped and `</`-broken for the same two reasons as `sheldon ui`: a
    # `file://` page has no charset header to fall back on, and a literal
    # </script> inside model-written text would end the data block early.
    payload = json.dumps(data, default=str, ensure_ascii=True).replace("</", "<\\/")
    return template.replace("__DATA__", payload)


def write(path: Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(data), encoding="utf-8")
    return path
