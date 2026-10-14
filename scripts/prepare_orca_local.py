#!/usr/bin/env python3
"""Build an ORCA-bench task set from upstream's incident rubrics.

**This is not the published ORCA-bench dataset**, and numbers from it are not
comparable to the paper's leaderboard. The official 1,079-task set lives behind
the Harbor Hub and a gated HuggingFace registry:

    https://hub.harborframework.com/datasets/orca-bench/orca-bench
    https://huggingface.co/datasets/orca-bench/orca-bench-harbor-tasks

Both return 401 without credentials, and the published snapshot image contains
only telemetry — no task records at all. What upstream's repo *does* ship in the
clear is `share-2w/json/`: SRE-curated incident rubrics with the ground-truth
feature flag, a mechanism write-up, and the metric/log/trace symptoms. That is
everything a task needs except the user-facing report, which upstream generates
with an LLM. So does this.

    scripts/prepare_orca_local.py --rubrics path/to/ORCA-bench/share-2w/json

Difficulty follows upstream's definitions:

    medium  one short vague clause; no product, page, error code or message
    easy    names the product/page, the concrete symptom, and the error string

Tasks are tagged `source: local-rubrics` so they can never be mistaken for the
published set in a report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from sheldon.core.io import load_dotenv, write_json  # noqa: F401
from sheldon.core.llm import LLM

SYSTEM = (
    "You write the first line of an incident: the user-facing complaint that reaches "
    "an on-call engineer before anyone knows what broke. You are given the true cause "
    "for context. Never reveal it. Never name the feature flag, the failing service, "
    "the mechanism, or any internal component. A report that hints at the answer makes "
    "the task worthless."
)

PROMPT = """A microservice demo shop is having an incident.

What actually happened (do NOT reveal any of this):
{description}

Observable user-facing symptoms you may draw on:
{symptoms}

Write two independent reports of this incident, as a customer or support ticket
would phrase it.

- "medium": one short, vague clause. No product names, page names, error codes,
  messages, users, regions or locales. Example: "product page is having issues".
- "easy": name the product or page, the concrete symptom, and the specific error
  or message a user sees. Example: "Checkout for the National Park Foundation
  Explorascope fails with 'could not retrieve quote' for EU customers".

Both must describe the SAME underlying incident."""


class Reports(BaseModel):
    medium: str = Field(description="one short vague clause, no specifics")
    easy: str = Field(description="names the page, the symptom and the error string")


def _reported(incident: str | None, seed: str) -> str:
    """When the incident was *noticed*, deterministically after it began."""
    if not incident:
        return ""
    delay = 5 + int(hashlib.sha256(seed.encode()).hexdigest()[:4], 16) % 41
    return (datetime.fromisoformat(incident) + timedelta(minutes=delay)).isoformat()


def symptom_lines(rubric: dict) -> list[str]:
    """Flatten upstream's symptom block into the strings our scorer compares."""
    out: list[str] = []
    sym = rubric.get("symptoms") or {}
    for store, entries in sym.items() if isinstance(sym, dict) else []:
        for e in entries or []:
            if not isinstance(e, dict):
                out.append(f"{store}: {e}")
                continue
            name = e.get("metric_family") or e.get("name") or e.get("pattern") or ""
            labels = e.get("defining_labels") or {}
            label_s = " ".join(f"{k}={v}" for k, v in labels.items())
            layer = e.get("signal_layer", "")
            out.append(f"{store}: {name} {label_s} [{layer}]".strip())
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rubrics", required=True, type=Path,
                    help="upstream share-2w/json directory")
    ap.add_argument("--out", type=Path, default=Path("data/orca-bench"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_dotenv()
    files = sorted(args.rubrics.glob("*.json"))[: args.limit]
    if not files:
        raise SystemExit(f"no rubrics under {args.rubrics}")

    llm = LLM(model=args.model) if args.model else LLM()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for path in files:
        rubric = json.loads(path.read_text(encoding="utf-8"))
        flag = rubric.get("feature_flag") or path.stem
        symptoms = symptom_lines(rubric)
        res = await llm.complete(
            PROMPT.format(
                description=rubric.get("description", ""),
                symptoms="\n".join(f"- {s}" for s in symptoms[:12]) or "(none recorded)",
            ),
            system=SYSTEM,
            schema=Reports,
            tag="orca-prep",
        )
        reports: Reports = res.parsed or Reports(medium="", easy="")
        for difficulty in ("medium", "easy"):
            report = getattr(reports, difficulty).strip()
            if not report:
                continue
            # An opaque id. `d1-c1-adHighCpu-on-easy` names its own answer, and
            # `Task.id` has to survive blinding for scoring — so the id itself
            # was handing the flag to any stage that looked at it.
            slug = hashlib.sha256(path.stem.encode()).hexdigest()[:8]
            rows.append({
                "id": f"orca-{slug}-{difficulty}",
                "report": report,
                "difficulty": difficulty,
                "root_cause": flag,
                "symptoms": symptoms,
                # The mechanism is gold — it names the flag, the file and the
                # causal chain. `Task.blind()` copies `meta` verbatim, so it goes
                # at the top level where the loader can route it into `answer`.
                "mechanism": (rubric.get("mechanism") or "")[:2000],
                # `rubric_file` names the flag too, so it is gold, not meta.
                "rubric_file": path.name,
                # Upstream varies time-to-detection deliberately, and says so in
                # the prompt: the reported time is not when the incident began.
                # Handing over the exact onset would point straight at the
                # answer, so the report lands a deterministic 5-45 minutes late
                # and the agent has to find onset itself.
                "reported_time": _reported(rubric.get("incident_time"), path.stem),
                "meta": {
                    "source": "local-rubrics",
                    "incident_time": rubric.get("incident_time"),
                },
            })
        print(f"  {path.stem}: {flag}", flush=True)

    out = args.out / "public.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    await llm.aclose()
    llm.cache.close()
    print(f"\n{out}  ({len(rows)} tasks from {len(files)} incidents)")
    print(f"cost ${llm.total.cost_usd:.3f}")
    print("NOT the published ORCA-bench set — meta.source = local-rubrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
