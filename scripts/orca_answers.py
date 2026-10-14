#!/usr/bin/env python3
"""Which rubrics a task is graded against, decided the way upstream decides it.

ORCA-bench does not grade a task against the one fault it was written for.
`generate_answers.py` in the upstream repo picks the *plausible* faults: every
event on the reported local day that fired before the query time and switched
a flag on, whose rubric documents a user-facing symptom, put in front of an
LLM together with the complaint and asked which could explain it. The verifier
then scores the report against each plausible rubric and averages, so a report
naming only one of two plausible causes caps at half credit, and one naming a
fault outside the set is charged with fabrication.

This script is that step, ported so the decision is made by the same prompt,
the same schema, the same window rule, and the same model call (upstream's own
`async_call_llm_judge`, imported from the verifier we already ship), over the
rubrics we have. Two things differ from upstream and are deliberate:

- Upstream reads event times from `data/events/<id>.json`, written by the
  scheduler when a flag fired. We do not have those; each rubric carries the
  same instant as `incident_time`, and its filename is the event id, so the
  rubric set *is* the event log for the faults that have a rubric -- which is
  the only set upstream keeps anyway (a candidate without a rubric is dropped).
- The answer lives in `data/orca-bench/answers.json`, keyed by task id, and
  the benchmark reads it only at scoring time. It must never go into the task
  row: `Task.blind()` keeps `meta`, and a list of plausible flags in `meta` is
  the answer key one attribute access away from any method.

    scripts/orca_answers.py            # every task in data/orca-bench/public.jsonl
    scripts/orca_answers.py --task orca-2f4a5d35-easy --force

Re-running skips tasks already answered unless `--force`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sheldon.core.io import data_root, load_dotenv  # noqa: E402
from sheldon.core.llm import normalize_base_url  # noqa: E402

#: Upstream's `task_spec.LOCAL_TZ`: the schedule runs on New York time and
#: "the reported day" is a day in that zone.
LOCAL_TZ = ZoneInfo("America/New_York")

#: Upstream's `generate_answers.EXCLUDED_FEATURE_FLAGS`: too generic to stand
#: as the answer when a more specific fault is also plausible; kept when sole.
EXCLUDED_FEATURE_FLAGS = frozenset({"imageSlowLoad"})

#: Upstream's defaults for this step: `--model openai-gpt-5.4`, effort unset.
DEFAULT_MODEL = "gpt-5.4"

# Verbatim from upstream's generate_answers.py.
ANSWER_PROMPT_TEMPLATE = """\
You are an expert SRE deciding which of several recent feature-flag-driven \
incidents could plausibly explain a user-reported issue. The reported issue \
is short and may be flag-agnostic; multiple incidents may match.

## Reported issue

{issue}

## Candidate incidents

For each candidate below, decide whether its user-visible effect could \
plausibly explain the reported issue. Base your decision on the rubric's \
description, mechanism, and (where present) the user-facing frontend \
symptoms. Be inclusive: if the symptoms could reasonably be confused with \
the reported issue, mark the candidate plausible.

You must return one verdict per candidate, in the same order as listed. \
Each verdict has:
  - ``event_id``: copy verbatim from the candidate.
  - ``plausible``: true if the candidate could explain the reported issue.
  - ``reason``: one short sentence justifying your verdict.

The order of verdicts must match the order of candidates. Do not omit any.

{candidate_blocks}
"""


def build_answer_prompt(issue: str, candidates: list[dict]) -> str:
    blocks = [
        f"### Candidate {i + 1}: `{c['event_id']}`\n\n{c['rubric_md']}"
        for i, c in enumerate(candidates)
    ]
    return ANSWER_PROMPT_TEMPLATE.format(issue=issue, candidate_blocks="\n\n".join(blocks))


def build_output_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "plausible_events": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "plausible": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["event_id", "plausible", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plausible_events"],
        "additionalProperties": False,
    }


def start_of_local_day(dt: datetime) -> datetime:
    local = dt.astimezone(LOCAL_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_local_day(dt: datetime) -> datetime:
    return start_of_local_day(dt) + timedelta(days=1)


def load_events(rubrics_dir: Path) -> dict[str, dict]:
    """The event log, reconstructed from the rubrics: id, flag, value, time."""
    events: dict[str, dict] = {}
    for path in sorted(rubrics_dir.glob("*.json")):
        rubric = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem
        flag = rubric.get("feature_flag") or ""
        # `d5-ic2-adFailure-on` -> value `on`; `...-paymentFailure-10%` -> `10%`.
        value = stem.rsplit("-", 1)[-1] if flag and f"-{flag}-" in f"-{stem}-" else "on"
        events[stem] = {
            "id": stem,
            "flag": flag,
            "value": value,
            "wall_clock_utc": rubric.get("incident_time") or "",
            "rubric": rubric,
        }
    return events


def candidates_in_window(events: dict[str, dict], lo: datetime, hi: datetime) -> list[dict]:
    out = []
    for ev in events.values():
        if ev["value"] == "off" or not ev["wall_clock_utc"]:
            continue
        ts = datetime.fromisoformat(ev["wall_clock_utc"])
        if lo <= ts < hi:
            out.append(ev)
    out.sort(key=lambda e: e["wall_clock_utc"])
    return out


def drop_excluded(events: list[dict]) -> list[dict]:
    kept = [e for e in events if e["root_cause"] not in EXCLUDED_FEATURE_FLAGS]
    if kept and len(kept) < len(events):
        return kept
    return events


def _verifier(path: Path):
    """Upstream's `check_prediction.py`, imported from wherever it is shipped,
    for `format_rubric` and `async_call_llm_judge` -- the same rendering and
    the same call upstream's answer step makes."""
    spec = importlib.util.spec_from_file_location("check_prediction", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def answer(
    row: dict, events: dict[str, dict], verifier, client, model: str, effort: str | None
) -> dict:
    task_id = row["id"]
    reported = datetime.fromisoformat(row["reported_time"])
    # Upstream: the window is the reported *local day*, closed at the query
    # time. Our query time is the reported time, so the window is the part of
    # that day before the report.
    lo, hi = start_of_local_day(reported), min(reported, end_of_local_day(reported))
    candidates = [
        ev for ev in candidates_in_window(events, lo, hi)
        if (ev["rubric"].get("symptoms") or {}).get("frontend")
    ]
    out: dict = {
        "task_id": task_id,
        "authored": str(row.get("rubric_file") or "").removesuffix(".json"),
        "window": [lo.isoformat(), hi.isoformat()],
        "model": model,
        "effort": effort,
        "candidates": [],
        "events": [],
    }
    if not candidates:
        return out
    blocks = [
        {"event_id": ev["id"], "rubric_md": verifier.format_rubric(ev["rubric"], include_frontend=True)}
        for ev in candidates
    ]
    prompt = build_answer_prompt(row.get("report") or "", blocks)
    raw, _summary, _usage = await verifier.async_call_llm_judge(
        client, prompt, model=model, reasoning_effort=effort,
        output_schema=build_output_schema(len(candidates)),
    )
    verdicts = {v["event_id"]: v for v in json.loads(raw).get("plausible_events") or []}
    by_id = {ev["id"]: ev for ev in candidates}
    triples = [
        {"event_id": eid, "root_cause": by_id[eid]["flag"], "event_time": by_id[eid]["wall_clock_utc"]}
        for eid, v in verdicts.items()
        if v.get("plausible") and eid in by_id and by_id[eid]["flag"]
    ]
    # Authored first, the rest chronological -- upstream's `_project_to_events`.
    authored = next((t for t in triples if t["event_id"] == out["authored"]), None)
    others = sorted((t for t in triples if t is not authored), key=lambda t: t["event_time"])
    projected = [authored, *others] if authored else others
    out["events"] = drop_excluded(projected)
    out["candidates"] = [
        {
            "event_id": ev["id"], "flag": ev["flag"], "wall_clock_utc": ev["wall_clock_utc"],
            "plausible": bool(verdicts.get(ev["id"], {}).get("plausible")),
            "reason": verdicts.get(ev["id"], {}).get("reason", ""),
        }
        for ev in candidates
    ]
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="orca-bench/public.jsonl", help="task rows, relative to the data root")
    ap.add_argument("--rubrics", default="orca-bench/upstream/share-2w/json")
    ap.add_argument("--verifier", default="orca-bench/upstream/check_prediction.py")
    ap.add_argument("--out", default="orca-bench/answers.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=None, choices=[None, "low", "medium", "high"])
    ap.add_argument("--task", action="append", default=[], help="only these task ids (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-decide tasks already answered")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    load_dotenv()
    # Upstream builds its client from OPENAI_BASE_URL verbatim; ours may name
    # the chat endpoint, which their client would not strip.
    if os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = normalize_base_url(os.environ["OPENAI_BASE_URL"])

    root = data_root()
    rows = [json.loads(line) for line in (root / args.tasks).read_text().splitlines() if line.strip()]
    if args.task:
        rows = [r for r in rows if r["id"] in set(args.task)]
    out_path = root / args.out
    answers: dict[str, dict] = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [r for r in rows if args.force or r["id"] not in answers]
    events = load_events(root / args.rubrics)
    verifier = _verifier(root / args.verifier)
    from openai import AsyncOpenAI  # noqa: PLC0415 - the client upstream's call takes

    client = AsyncOpenAI(max_retries=4, timeout=900)
    print(f"{len(events)} rubric events, {len(rows)} tasks, {len(todo)} to decide", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)

    async def one(row: dict) -> None:
        async with sem:
            try:
                res = await answer(row, events, verifier, client, args.model, args.effort)
            except Exception as e:  # noqa: BLE001 - one failure must not lose the batch
                print(f"{row['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                return
            answers[row["id"]] = res
            flags = [e["root_cause"] for e in res["events"]]
            print(f"{row['id']}: {len(res['candidates'])} candidate(s) -> {flags}", file=sys.stderr)
            # Written after every task so a killed run keeps what it decided.
            out_path.write_text(json.dumps(answers, indent=1, sort_keys=True))

    await asyncio.gather(*(one(r) for r in todo))
    print(f"wrote {out_path} ({len(answers)} tasks)", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
