"""Aggregate finished runs into the table you actually look at.

Rows are methods, columns are benchmarks, cells are the primary score. Baselines
come first so a method is never read without its control next to it.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from sheldon.core.io import read_jsonl, runs_root

# Baselines sort to the top; everything else follows alphabetically.
_BASELINE_ORDER = {"direct": 0, "best-of-n": 1}


def collect(root: Path | None = None, *, benchmark: str | None = None,
            method: str | None = None, tag: str | None = None) -> list[dict[str, Any]]:
    root = root or runs_root()
    rows: list[dict[str, Any]] = []
    for summary in sorted(root.glob("*/summary.json")):
        try:
            row = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if benchmark and row.get("benchmark") != benchmark:
            continue
        if method and row.get("method") != method:
            continue
        if tag is not None and row.get("tag", "") != tag:
            continue
        if row.get("n"):
            rows.append(row)
    return rows


def pivot(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[tuple[str, str], dict]]:
    bms = sorted({r["benchmark"] for r in rows})
    methods = sorted(
        {r["method"] for r in rows}, key=lambda m: (_BASELINE_ORDER.get(m, 99), m)
    )
    cells: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["method"], r["benchmark"])
        # If a method ran twice on a benchmark, the larger n wins the cell.
        if key not in cells or r["n"] > cells[key]["n"]:
            cells[key] = r
    return bms, methods, cells


def markdown(rows: list[dict[str, Any]], *, show_cost: bool = True) -> str:
    if not rows:
        return "_no completed runs found — try `sheldon run` first_"
    bms, methods, cells = pivot(rows)

    header = ["method", *bms]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for m in methods:
        row = [f"`{m}`"]
        for b in bms:
            c = cells.get((m, b))
            row.append(
                f"{c['score']:.3f} ±{c['stderr']:.3f} (n={c['n']})" if c else "—"
            )
        lines.append("| " + " | ".join(row) + " |")

    if show_cost:
        lines += [
            "",
            "| method | benchmark | $/task | total $ | billed $ | s/task | p90 s | wall | "
            "calls/task | rounds | errors | cached |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for m in methods:
            for b in bms:
                c = cells.get((m, b))
                if not c:
                    continue
                hits = (c.get("cache") or {}).get("hits", 0)
                jobs = c.get("concurrency")
                wall = fmt_duration(c.get("wall_s", 0.0))
                if jobs:
                    wall += f" (j={jobs})"
                lines.append(
                    f"| `{m}` | {b} | ${c['cost_per_task']:.4f} | "
                    f"${c.get('cost_usd', 0.0):.2f} | "
                    f"${c.get('billed_usd', c.get('cost_usd', 0.0)):.2f} | "
                    f"{c.get('latency_mean_s', 0.0):.0f}s | "
                    f"{c.get('latency_p90_s', 0.0):.0f}s | {wall} | "
                    f"{c.get('calls_per_task', 0.0):.1f} | "
                    f"{c['mean_rounds']:.1f} | {c['errors']} | {hits} |"
                )
        lines += [
            "",
            "_`s/task` is one task's latency inside the method; `wall` is the whole "
            "cell end to end at `j` concurrent tasks. `total $` is what the run costs "
            "to produce and is what compares across methods; `billed $` is what was "
            "actually spent this time, and is lower wherever the cache was reused._",
        ]
    return "\n".join(lines)


def fmt_duration(seconds: float) -> str:
    """Compact and readable at every scale a run actually takes."""
    seconds = float(seconds or 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    fields = ["benchmark", "method", "split", "model", "tag", "n", "score", "stderr",
              "mean_rounds", "errors", "cost_per_task", "cost_usd", "billed_usd",
              "calls_per_task",
              "latency_mean_s", "latency_p50_s", "latency_p90_s", "wall_s", "concurrency",
              "started_at", "finished_at", "git_sha", "run_id"]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["benchmark"], r["method"])):
        w.writerow(r)
    return buf.getvalue()


def failures(run_id: str, root: Path | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """The lowest-scoring tasks in a run, with a pointer to each trace.

    The fastest way from "the number is bad" to "here is the round where it
    went wrong".
    """
    d = (root or runs_root()) / run_id
    scores = sorted(read_jsonl(d / "scores.jsonl"), key=lambda s: s["score"])
    preds = {p["task_id"]: p for p in read_jsonl(d / "predictions.jsonl")}
    out = []
    for s in scores[:limit]:
        p = preds.get(s["task_id"], {})
        out.append(
            {
                "task_id": s["task_id"],
                "score": s["score"],
                "error": p.get("error"),
                "rounds": p.get("rounds"),
                "wall_s": p.get("wall_s"),
                "stop_reason": (p.get("extra") or {}).get("stop_reason"),
                "detail": s.get("detail"),
                "trace": str(d / "traces" / f"{s['task_id']}.jsonl"),
            }
        )
    return out


def render_trace(run_id: str, task_id: str | None = None, root: Path | None = None,
                 width: int = 2000) -> str:
    """A trace as something you can read top to bottom.

    The JSONL is the record; this is the view. Prompts and replies carry real
    newlines, which makes one event per line unreadable exactly when you most
    need to read it.
    """
    d = (root or runs_root()) / run_id
    traces = sorted((d / "traces").glob("*.jsonl"))
    if task_id:
        traces = [t for t in traces if t.stem == task_id] or traces[:0]
    if not traces:
        return f"_no trace for {run_id}{'/' + task_id if task_id else ''}_"

    out: list[str] = []
    for path in traces:
        out.append(f"{'=' * 70}\n{path.stem}\n{'=' * 70}")
        for e in read_jsonl(path):
            kind, t = e.get("kind", "?"), e.get("t", 0)
            head = f"\n[{t:>7.2f}s] {kind}"
            if kind == "llm":
                cached = " (cached)" if e.get("cached") else ""
                out.append(f"{head}  tag={e.get('tag')} stop={e.get('stop_reason')}{cached}")
                if e.get("tools_offered"):
                    out.append(f"  tools: {', '.join(e['tools_offered'])}")
                if e.get("system"):
                    out.append(_block("SYSTEM", e["system"], width))
                for m in e.get("messages") or []:
                    out.append(_block(m.get("role", "?").upper(), _content(m.get("content")), width))
                for tu in e.get("thinking") or []:
                    out.append(_block("THINKING", tu, width))
                if e.get("text"):
                    out.append(_block("ASSISTANT", e["text"], width))
                for call in e.get("tool_use") or []:
                    out.append(_block(f"CALL {call.get('name')}",
                                      json.dumps(call.get("input"), indent=2), width))
            elif kind == "tool":
                out.append(f"{head}  {e.get('probe')} ok={e.get('ok')}")
                out.append(_block("RESULT", str(e.get("summary", "")), width))
            else:
                rest = {k: v for k, v in e.items() if k not in ("kind", "t")}
                out.append(f"{head}  {json.dumps(rest, default=str)[:width]}")
    return "\n".join(out)


def _content(content: Any) -> str:
    """Message content is a string or a list of blocks; render either."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if not isinstance(b, dict):
            parts.append(str(b))
        elif b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            parts.append(f"[tool_use {b.get('name')}] {json.dumps(b.get('input'), default=str)}")
        elif b.get("type") == "tool_result":
            err = " (error)" if b.get("is_error") else ""
            parts.append(f"[tool_result{err}] {_content(b.get('content'))}")
        else:
            parts.append(json.dumps(b, default=str))
    return "\n".join(parts)


def _block(label: str, body: str, width: int) -> str:
    body = (body or "").rstrip()
    if len(body) > width:
        body = body[:width] + f"\n  ... [{len(body) - width} more chars]"
    indented = "\n".join("  " + line for line in body.splitlines()) or "  (empty)"
    return f"  --- {label} ---\n{indented}"
