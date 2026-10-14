#!/usr/bin/env python
"""Label what each case actually asks for, before anything is graded.

EviDx classifies every case into a `question_focus` because these datasets are
not uniformly diagnostic. Measured on our 100-case samples:

    DiagnosisArena   100 diagnosis
    MedXpertQA-Diag   94 diagnosis, 3 workup_associated_anomaly, 2 risk_factor,
                       1 complication
    JAMA              52 diagnosis, 45 management_next_step, 3 interpretation

JAMA is the one that matters: nearly half of it asks what to do next, not what
the disease is. Grading those against a diagnosis rubric measures the wrong
thing, and a loader that assumes every case is diagnostic -- as GoS's does --
will do exactly that.

Classification is by question stem first, which is deterministic and free, with
an LLM fallback only where the stem is genuinely ambiguous. A judged label on a
question we could have matched exactly is variance we do not need. The main
ambiguity is "what is the most likely cause of...", which is a diagnosis
question when the options are diseases and a mechanism question when they are
pathophysiological processes; those go to the LLM with the options in view.

Categories follow EviDx's own set.

    ./classify_focus.py --in corpus-JAMA.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

CATEGORIES = [
    "diagnosis", "management_next_step", "treatment", "diagnostic_test",
    "workup_associated_anomaly", "mechanism", "risk_factor", "complication",
    "interpretation", "other",
]

#: Ordered: the first match wins, so put the specific before the general.
STEM_PATTERNS: list[tuple[str, str]] = [
    (r"what (is|are) (your |the )?(most likely |most probable |most accurate |likely )?(primary )?diagnos", "diagnosis"),
    (r"which .{0,40}(is|are) .{0,30}most likely diagnos", "diagnosis"),
    (r"which one of the following diagnos", "diagnosis"),
    (r"what would you do next|what is the (best |most appropriate )?next step|next best step", "management_next_step"),
    (r"how (do|would) you interpret", "interpretation"),
    (r"which .{0,40}(treatment|therapy|management)|how should .{0,30}be (treated|managed)", "treatment"),
    (r"which .{0,50}(test|study|imaging|investigation) .{0,30}(next|order|obtain|perform)|what test", "diagnostic_test"),
    (r"which .{0,40}(condition|anomal|abnormalit) .{0,30}(investigat|associat|screen)", "workup_associated_anomaly"),
    (r"which .{0,40}(mechanism|pathophysiolog)", "mechanism"),
    (r"which .{0,40}risk factor", "risk_factor"),
    (r"which .{0,40}complication", "complication"),
]

LLM_SYSTEM = (
    "You classify what a clinical question is asking for. Answer with exactly one "
    "of these labels and nothing else: " + ", ".join(CATEGORIES) + ".\n\n"
    "Classify from the question wording together with the answer options, because "
    "the wording alone is often ambiguous. In particular, 'what is the most likely "
    "cause of...' is a `diagnosis` question when the options are diseases or "
    "conditions, and a `mechanism` question when they are pathophysiological "
    "processes. Do not use the case narrative."
)


def last_question(text: str) -> str:
    """The trailing interrogative is what the case is actually asking.

    The option block is stripped first: MedXpertQA appends "Answer Choices: (A)
    ... (J) ...", and option text containing a question mark otherwise wins the
    "last interrogative" race and the case gets classified from a distractor.
    """
    body = re.split(r"Answer Choices:", text)[0]
    qs = re.findall(r"([^.?!\n]{5,120}\?)", body)
    if qs:
        return qs[-1].strip()
    return body.strip()[-160:]


def by_stem(q: str) -> str | None:
    low = q.lower()
    for pat, label in STEM_PATTERNS:
        if re.search(pat, low):
            return label
    return None


def by_llm(q: str, model: str, options: dict | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": LLM_SYSTEM},
                  {"role": "user", "content": q + (
                      "\n\nOptions:\n" + "\n".join(f"({k}) {v}" for k, v in options.items())
                      if options else "")}],
    )
    out = (r.choices[0].message.content or "").strip().lower()
    for c in CATEGORIES:
        if c in out:
            return c
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", default=os.environ.get("GOS_MODEL", "gpt-5.6-terra"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-llm", action="store_true", help="stems only; leave the rest 'other'")
    a = ap.parse_args()

    p = Path(a.inp)
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    for r in rows:
        r["_q"] = last_question(r["question"])
        r["question_focus"] = by_stem(r["_q"])

    todo = [r for r in rows if r["question_focus"] is None]
    if todo and not a.no_llm:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for r, lab in zip(todo, pool.map(lambda x: by_llm(x["_q"], a.model, x.get("options")), todo)):
                r["question_focus"] = lab
    for r in rows:
        r["question_focus"] = r.get("question_focus") or "other"
        r.pop("_q", None)

    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    c = Counter(r["question_focus"] for r in rows)
    print(f"  {p.name}  ({len(rows)} cases, {len(todo)} needed the LLM)")
    for lab, n in c.most_common():
        print(f"    {n:>4}  {lab}")


if __name__ == "__main__":
    main()
