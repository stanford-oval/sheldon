#!/usr/bin/env python
"""Does the synthesised environment preserve the case, and does it leak the answer?

Layer 4 puts an LLM between the source case and the agent. Two things can go
wrong, and both are invisible downstream:

**Loss.** Diagnosis-critical detail can be dropped. Med-Evidence-2.6k gives the
exact `original_text` spans that matter per case, typed Inclusion / Exclusion /
Differentiation, so survival is measurable rather than assumed.

**Leakage.** The synthesiser can put the answer where the agent will see it. A
smoke test found exactly this: for a case whose source never contains the word
"epidermolytic", the synthesiser wrote "Most likely diagnosis: epidermolytic
ichthyosis" into `past_medical_history` -- inferred, not copied, and in a
section that is visible before any tool call. EviDx's synthesis prompt has no
instruction against naming the diagnosis.

Evidence survival uses **EviDx's own matcher** (`check_acquisition_advanced` in
their `evaluate_evidence_recall.py`): normalised exact substring first, then an
LLM semantic judge with their prompt. A token-overlap heuristic was tried first
and was badly wrong -- it scored 13 spans lost where 10 were present, because
the synthesiser turns prose into schema fields and "a 6-month history of
progressive dysarthria and dysphagia" survives as review_of_systems entries with
none of the original sentence structure. Their public annotations carry only
`original_text` and `type`, so their `Rule-Feature` tier does not apply.

Leakage is checked at three depths, because an exact-string test misses the
case above (gold is "Annular epidermolytic ichthyosis", the leak says plain
"epidermolytic ichthyosis"):

  exact     the gold string appears verbatim
  partial   a majority of the gold's content words appear together
  marker    text like "most likely diagnosis:" appears in a visible section,
            which is a diagnosis being asserted regardless of which one

    ./check_fidelity.py --in ehr-DiagnosisArena-30.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter
from pathlib import Path

STOP = {"the", "a", "an", "of", "and", "or", "with", "due", "to", "in", "on",
        "for", "from", "type", "disease", "syndrome", "s"}

#: Sections an agent sees before any tool call, in either narrative mode.
VISIBLE_KEYS = ["demographics", "past_medical_history", "surgical_history",
                "social_history", "family_history", "allergies",
                "current_medications", "review_of_systems", "physical_exam"]

DX_MARKERS = [r"most likely diagnosis", r"\bdiagnosis\s*[:\-]", r"diagnosed with",
              r"consistent with a diagnosis", r"final diagnosis"]


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in STOP and len(w) > 2}


def normalize_text(text: str) -> str:
    """Theirs: collapse whitespace and case before an exact-substring test."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


ACQ_PROMPT = """
### Task
Determine if the specific **Medical Evidence** described below is present in the provided **Text Data** (which contains patient history, tool outputs, etc.).
Be flexible with phrasing, formatting, and abbreviations (e.g., "NG tube" = "nasogastric tube").

### Target Evidence
- Raw Text: "{original_text}"

### Text Data (What the Agent saw)
{context}

### Question
Does the Text Data contain the information described in the Target Evidence?
Answer JSON: {{"present": true, "reason": "short explanation"}} or {{"present": false}}
"""


def acquired(client, model, span: str, context: str) -> tuple[bool, str]:
    """EviDx's cascade: normalised exact match, then an LLM semantic judge.

    Their third tier keys on `feature_name`, which the public annotations do not
    carry, so it is skipped rather than approximated.
    """
    if normalize_text(span) in normalize_text(context):
        return True, "Rule-Exact"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ACQ_PROMPT.format(
                original_text=span, context=context[:20000])}],
            response_format={"type": "json_object"},
        )
        body = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", body, re.S)
        ok = bool(json.loads(m.group(0)).get("present")) if m else False
        return ok, ("LLM-Semantic" if ok else "Missed")
    except Exception:
        return False, "Error"


def flatten(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(flatten(x) for x in v)
    if isinstance(v, dict):
        return " ".join(flatten(x) for x in v.values())
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--show", type=int, default=3, help="examples per finding")
    ap.add_argument("--model", default=os.environ.get("GOS_MODEL", "gpt-5.6-terra"))
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])

    rows = [json.loads(l) for l in Path(a.inp).open(encoding="utf-8")]
    rows = [r for r in rows if r.get("ehr")]
    if not rows:
        print("  no synthesised rows"); return

    ev_total = ev_kept = 0
    by_type: Counter = Counter()
    kept_by_type: Counter = Counter()
    method_counts: Counter = Counter()
    lost_examples = []
    jobs: list = []

    leak_exact = leak_partial = leak_marker = 0
    leak_examples = []

    for r in rows:
        ehr = r["ehr"]
        whole = flatten(ehr).lower()
        visible = " ".join(flatten(ehr.get(k)) for k in VISIBLE_KEYS)
        visible += " " + (r.get("chief_complaint") or "") + " " + (r.get("presentation") or "")
        visible_l = visible.lower()

        for ev in r.get("golden_evidence") or []:
            span = str(ev.get("original_text", "")).strip()
            if span:
                jobs.append((r["id"], ev.get("type", "?"), span, whole))

        # --- leakage: is the answer visible before any tool call? ---
        gold = (r.get("gold_text") or "").strip()
        gw = words(gold)
        e = gold.lower() in visible_l
        p = bool(gw) and len(gw & words(visible_l)) / len(gw) >= 0.6
        m = any(re.search(pat, visible_l) for pat in DX_MARKERS)
        leak_exact += e; leak_partial += (p and not e); leak_marker += m
        if (e or p or m) and len(leak_examples) < a.show:
            snip = ""
            for pat in DX_MARKERS:
                mm = re.search(pat, visible_l)
                if mm:
                    snip = visible[max(0, mm.start() - 30):mm.start() + 130]
                    break
            leak_examples.append((r["id"], "exact" if e else ("partial" if p else "marker"),
                                  gold[:50], snip.replace("\n", " ")))

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(lambda j: acquired(client, a.model, j[2], j[3]), jobs))
    for (cid, t, span, _), (ok, how) in zip(jobs, results):
        ev_total += 1
        by_type[t] += 1
        method_counts[how] += 1
        if ok:
            ev_kept += 1
            kept_by_type[t] += 1
        elif len(lost_examples) < a.show:
            lost_examples.append((cid, t, span[:100]))

    n = len(rows)
    print(f"  {n} synthesised cases\n")
    print("  === EVIDENCE SURVIVAL (does the environment still contain what matters?) ===")
    print(f"    {ev_kept}/{ev_total} golden spans found in the EHR  ({ev_kept/max(ev_total,1):.1%})")
    for t in by_type:
        print(f"      {t:<16} {kept_by_type[t]}/{by_type[t]}  ({kept_by_type[t]/by_type[t]:.0%})")
    print(f"      matched by: {dict(method_counts)}")
    for i, (cid, t, s) in enumerate(lost_examples):
        if i == 0:
            print("    lost, e.g.:")
        print(f"      [{t}] {cid}: {s}")

    print("\n  === ANSWER LEAKAGE (is the diagnosis visible before any tool call?) ===")
    print(f"    exact gold string visible : {leak_exact}/{n}")
    print(f"    partial gold visible      : {leak_partial}/{n}")
    print(f"    a diagnosis asserted      : {leak_marker}/{n}")
    for cid, kind, gold, snip in leak_examples:
        print(f"      [{kind}] {cid}  gold={gold!r}")
        if snip:
            print(f"              ...{snip}...")


if __name__ == "__main__":
    main()
