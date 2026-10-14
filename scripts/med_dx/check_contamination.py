#!/usr/bin/env python
"""Did synthesis add anything the source case did not contain?

`check_fidelity.py` asks two narrower questions -- does the golden evidence
survive, and is the *gold* answer visible. Both miss the general failure: the
synthesiser inferring something the source never said. A speculated diagnosis
that happens to be *wrong* is still contamination, still invisible to a search
for the gold string, and still changes what the agent is investigating.

That cannot be checked by pattern matching, because the synthesiser is supposed
to reword: prose becomes schema fields, and "a 6-month history of progressive
dysarthria" legitimately survives as review_of_systems entries sharing almost no
surface form. Only a reader with both texts can separate restructuring from
addition. So this is an LLM call per case with the source and the synthesised
EHR side by side.

Output is structured rather than a verdict, because the point is repair: each
addition is returned with its section, its text, and a severity, so a follow-up
pass can strip the contamination without rebuilding the environment.

  severity      meaning
  ------------  ------------------------------------------------------------
  conclusion    a diagnosis, impression or unifying explanation of the case.
                Worst: it answers the question the agent is meant to work on,
                whether or not it matches the gold.
  inference     a clinical claim derived from findings but not stated in the
                source (staging, causal attribution, "consistent with X").
  fabrication   a concrete fact with no basis in the source at all -- a value,
                a date, a finding that simply is not there.
  benign        rewording, normalisation, unit conversion, restructuring.
                Reported for completeness, not a defect.

    ./check_contamination.py --in ehr-JAMA.jsonl --out contamination-JAMA.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter
from pathlib import Path

SYSTEM = """You audit a structured clinical record against the raw case text it was built from.

Your ONLY question: does the record assert anything the raw text does not support?

The record is a faithful restructuring if it reorganises, reworders, normalises units,
splits prose into fields, or drops detail. None of that is an addition. Dropping
information is NOT your concern here.

An ADDITION is a statement in the record that a careful reader could not derive
verbatim from the raw text. Classify each one:

- "conclusion": a diagnosis, impression, or unifying explanation for THIS presentation.
  Any disease name offered as the answer, whether or not it is the correct answer.
  Phrases like "most likely diagnosis", "consistent with", "compatible with", "impression".
  A genuine PRIOR condition explicitly stated in the raw text as past history is NOT this.
- "inference": a clinical claim derived from the findings but not stated -- staging,
  causal attribution, severity grading, "suggests", "indicates".
- "fabrication": a concrete fact with no basis in the raw text at all: a lab value,
  a date, a measurement, a finding that simply does not appear there.
- "benign": rewording, unit conversion, field reorganisation, expansion of an
  abbreviation. Report at most 3 of these; they are not defects.

Answer with JSON only:
{"clean": true|false,
 "additions": [{"severity": "conclusion|inference|fabrication|benign",
                "section": "<the record field it appears in, best guess>",
                "text": "<the added statement, quoted from the record>",
                "why": "<one clause: why the raw text does not support it>"}]}

"clean" is true only if there are no conclusion, inference or fabrication additions."""

USER = """--- RAW CASE TEXT ---
{source}

--- STRUCTURED RECORD BUILT FROM IT ---
{record}"""


def render(ehr: dict, extra: dict) -> str:
    """The record as the auditor should see it: every field, including the
    narratives, since those are what the agent reads before any tool call."""
    body = {k: v for k, v in (ehr or {}).items() if v}
    for k in ("chief_complaint", "presentation"):
        if extra.get(k):
            body[k] = extra[k]
    return json.dumps(body, ensure_ascii=False, indent=1)


def audit(client, model: str, row: dict) -> dict:
    rec = render(row.get("ehr") or {}, row)
    try:
        r = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER.format(
                          source=row["question"][:24000], record=rec[:24000])}],
        )
        body = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", body, re.S)
        out = json.loads(m.group(0)) if m else {}
    except Exception as exc:
        return {"id": row["id"], "error": f"{type(exc).__name__}: {exc}", "additions": []}
    adds = [a for a in (out.get("additions") or []) if isinstance(a, dict)]
    bad = [a for a in adds if a.get("severity") in ("conclusion", "inference", "fabrication")]
    return {"id": row["id"], "dataset": row.get("dataset"), "gold": row.get("gold_text"),
            "clean": not bad, "additions": adds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=os.environ.get("GOS_MODEL", "gpt-5.6-terra"))
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    rows = [json.loads(l) for l in Path(a.inp).open(encoding="utf-8")]
    rows = [r for r in rows if r.get("ehr")]
    if a.limit:
        rows = rows[: a.limit]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(audit, client, a.model, r): r for r in rows}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            results.append(fut.result())
            print(f"  [{i:>3}/{len(rows)}]", end="\r", flush=True)
    results.sort(key=lambda r: r["id"])

    sev = Counter(a_["severity"] for r in results for a_ in r["additions"]
                  if a_.get("severity"))
    dirty = [r for r in results if not r.get("clean", True)]
    errs = [r for r in results if r.get("error")]

    print(f"  {len(results)} cases audited{' ' * 20}")
    print(f"    clean            : {len(results) - len(dirty) - len(errs)}/{len(results)}")
    print(f"    contaminated     : {len(dirty)}")
    if errs:
        print(f"    audit errors     : {len(errs)}")
    print(f"    additions by type: {dict(sev)}")

    shown = 0
    for r in dirty:
        for ad in r["additions"]:
            if ad.get("severity") in ("conclusion", "fabrication", "inference") and shown < a.show:
                print(f"\n    [{ad['severity']}] {r['id']}  (gold: {str(r.get('gold'))[:40]})")
                print(f"      section: {ad.get('section')}")
                print(f"      added  : {str(ad.get('text'))[:160]}")
                print(f"      why    : {str(ad.get('why'))[:140]}")
                shown += 1

    if a.out:
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n    written: {a.out}")


if __name__ == "__main__":
    main()
