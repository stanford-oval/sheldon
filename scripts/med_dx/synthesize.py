#!/usr/bin/env python
"""Build EviDx's EHR environments, using EviDx's own synthesiser.

The synthesis *is* the environment -- what it produces is what an agent can
query -- so this calls `EviDx.agent.Synth.NarrativeSynthesizer` directly rather
than reimplementing it. An earlier version of this file paraphrased their
prompt, which is precisely the drift that makes a number non-comparable.
`NarrativeSynthesizer(llm_client, model_name)` takes a plain OpenAI client and
has no coupling to their pipeline, so it drops straight in.

**The second narrative is derived, not prompted.** EviDx starts an agent with a
one-sentence chief complaint and gates everything else behind tools. That is
harsher than GoS's split, which shows the full presentation and gates only the
diagnostic workup, and it changes what a baseline number means -- so we want the
option of either. Rather than ask the model for a second narrative (another
generation to go wrong, and a divergence from their prompt), `presentation` is
rendered deterministically from the EHR's pre-diagnostic sections. That costs
nothing, cannot hallucinate, and is guaranteed to agree with what the Examiner
tools would return for the same sections.

    ./synthesize.py --in corpus-DiagnosisArena.jsonl --limit 30
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import re
import sys
from pathlib import Path

EVIDX = Path(os.environ.get("EVIDX_ROOT", "EviDx"))
sys.path.insert(0, str(EVIDX / "agent"))

#: Sections that exist before any diagnostic test is ordered. Labs, imaging and
#: other procedures are excluded on purpose: they are what the agent must go and
#: ask for, in both narrative modes.
PRESENTATION_SECTIONS = [
    ("demographics", "Demographics"),
    ("past_medical_history", "Past medical history"),
    ("surgical_history", "Surgical history"),
    ("social_history", "Social history"),
    ("family_history", "Family history"),
    ("allergies", "Allergies"),
    ("current_medications", "Current medications"),
    ("review_of_systems", "Review of systems"),
    ("physical_exam", "Physical examination"),
]


def strip_options(text: str, options: dict | None) -> str:
    """Remove answer-option text that runs into the case narrative.

    JAMA's cases append the option strings with no delimiter before the stem:

        ...neovascularization in the superior cornea (Figure 1B).
        Subconjunctival injection of anti-VEGF medication
        Refraction with rigid gas permeable contact lens fitting
        What Would You Do Next?

    The synthesiser reasonably reads those as narrative and files them under
    `other_procedures`, so the record then states that two of the four candidate
    answers were already performed. On a management question that eliminates or
    misleads. Verified on JAMA-907 and JAMA-817.

    MedXpertQA delimits its options with "Answer Choices:", and the audit found
    no bleed there, so only text before such a marker is cleaned -- this is a
    no-op for datasets that label their options.
    """
    if not options:
        return text
    head, sep, tail = text.partition("Answer Choices:")
    for opt in options.values():
        o = str(opt or "").strip()
        if len(o) > 15 and o in head:
            head = head.replace(o, " ")
    head = re.sub(r"[ \t]{2,}", " ", head)
    return head + sep + tail


def _flatten(v) -> str:
    """Render a schema value as prose without inventing anything."""
    if v is None or v == "" or v == [] or v == {}:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, dict):
                f = str(item.get("finding", "")).strip()
                d = str(item.get("details", "")).strip()
                parts.append(f"{f} ({d})" if f and d else f or d)
            elif item:
                parts.append(str(item).strip())
        return "; ".join(p for p in parts if p)
    if isinstance(v, dict):
        parts = []
        for k, sub in v.items():
            s = _flatten(sub)
            if s:
                parts.append(f"{k.replace('_', ' ')}: {s}")
        return "; ".join(parts)
    return str(v).strip()


def render_presentation(ehr: dict) -> str:
    """The presenting history and exam, from the EHR itself.

    Deterministic by design. The alternative -- asking the model for a second
    narrative -- adds a generation that can drop or invent findings, and would
    let the two modes disagree about the same case.
    """
    out = []
    for key, label in PRESENTATION_SECTIONS:
        s = _flatten(ehr.get(key))
        if s:
            out.append(f"{label}: {s}")
    return "\n".join(out)



#: Appended to EviDx's own prompts rather than edited into their repo, so the
#: divergence is visible here and their files stay pristine.
#:
#: Measured on 30 DiagnosisArena cases synthesised with their prompts unmodified:
#: 13/30 put a diagnosis into a section the agent sees before any tool call, 2/30
#: the exact gold string. These are not copied from the source -- for one case the
#: raw text never contains the word "epidermolytic" and the synthesiser wrote
#: "Most likely diagnosis: epidermolytic ichthyosis" into past_medical_history.
#: Their prompt has no rule against naming the diagnosis; an environment that
#: answers its own question cannot measure a method.
NO_LEAK_RULE = """

---

**RULE 0 — DO NOT STATE THE ANSWER (overrides everything below):**

The EHR you produce is an environment an agent will investigate. It records
**observations**, never **conclusions about this presentation**.

* NEVER write a diagnosis, impression, or unifying explanation for the current
  presentation into ANY field. No "most likely diagnosis", "consistent with",
  "diagnosed with", "impression:", "compatible with", or a disease name offered
  as the explanation of these findings.
* NEVER infer. If a disease name does not appear in the raw text, it must not
  appear in your output. If it does appear only as the case's answer, omit it.
* Record the *findings* that support a conclusion, not the conclusion. A genetic
  variant, a stain result, a lesion morphology are findings — keep them, in
  their own fields, verbatim. The disease they imply is the answer — omit it.
* A genuine PRIOR condition from the patient's past history is legitimate and
  should be kept, but only as history, never framed as the explanation of the
  current presentation.
* `initial_narrative_summary` is a chief complaint: symptoms and duration only.

This rule is not a style preference. Naming the answer makes the case
unanswerable as a task.
"""


class _CompatClient:
    """Pass EviDx's calls through, minus parameters this endpoint rejects.

    Their synthesiser hardcodes `temperature=0.0` and `extra_body={"enable_thinking":
    False}`. This gateway rejects both for gpt-5.6-terra -- the temperature error is
    "Only the default (1) value is supported", the same quirk GoS's own config notes.
    Their code has a fallback for `extra_body` but not for temperature, so the draft
    call fails, `synthesize_case` takes an early-return path that yields five values
    where the success path yields six, and the caller dies on the unpack.

    Shimming the client keeps their prompts and control flow untouched, which is the
    point of using their synthesiser at all.

    Consequence worth recording: synthesis cannot run at temperature 0 here, so the
    environments are not reproducible from the model alone. They are generated once
    and cached to disk; that file, not the prompt, is the fixed environment.
    """

    _DROP = ("temperature", "extra_body", "top_p")

    def __init__(self, inner):
        self._inner = inner
        self.chat = self  # their code calls client.chat.completions.create
        self.completions = self

    def create(self, **kw):
        return self._inner.chat.completions.create(
            **{k: v for k, v in kw.items() if k not in self._DROP})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=os.environ.get("SYNTH_MODEL", "gpt-5.6-terra"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=32,
                    help="cases in flight; these are independent API calls")
    a = ap.parse_args()

    import Synth.refiner_prompts as rp

    # Patch the guard into both prompts before the classes bind them. The
    # Refiner is the audit pass, so it has to check for leakage as well --
    # otherwise it re-adds what the synthesiser was told to leave out.
    import Synth.synthesizer_prompts as sp
    from openai import OpenAI

    sp.SYNTHESIZER_SYSTEM_PROMPT = sp.SYNTHESIZER_SYSTEM_PROMPT + NO_LEAK_RULE
    rp.REFINER_SYSTEM_PROMPT = rp.REFINER_SYSTEM_PROMPT + NO_LEAK_RULE + """
**AUDIT FOR LEAKAGE TOO:** as well as restoring dropped information, remove any
diagnosis, impression or inferred disease name for the current presentation that
the draft contains. Removing a leaked conclusion is not information loss; the
findings behind it must stay.
"""

    from Synth.synthesizer import NarrativeSynthesizer

    client = _CompatClient(OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                  base_url=os.environ["OPENAI_BASE_URL"]))

    src = Path(a.inp)
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out or src.with_name(src.stem + "-ehr.jsonl"))

    def one(r):
        # Their synthesiser prints progress per case; with a thread pool that
        # interleaves into noise, so capture it and report per case ourselves.
        try:
            synth = NarrativeSynthesizer(client, a.model)
            source = strip_options(r["question"], r.get("options"))
            got = synth.synthesize_case(source, dataset_hints=r["dataset"])
            # Their failure path returns five values and their success path six,
            # so unpack defensively rather than turning their bug into ours.
            got = tuple(got) + ("",) * (6 - len(got))
            narrative, ehr, question, choices, trace, focus = got[:6]
        except Exception as exc:  # a failed case is recorded, not fatal
            return {**r, "ehr": {}, "synth_error": f"{type(exc).__name__}: {exc}"}
        return {
            **r,
            "ehr": ehr or {},
            "chief_complaint": narrative or "",
            "presentation": render_presentation(ehr or {}),
            "synth_clinical_question": question or "",
            "synth_answer_choices": choices or {},
            "synth_question_focus": focus or "",
            "synth_model": a.model,
        }

    # as_completed, not map: map yields in submission order, so one slow case
    # hides every result behind it and the run looks stalled when it is not.
    # Their code prints per case; silence it around the whole pool rather than
    # per worker -- redirect_stdout is process-wide, so doing it inside a thread
    # captures every other thread's output too, including our own progress line.
    done = []
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(one, r): r for r in rows}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            done.append(res)
            mark = "ok  " if res.get("ehr") else "FAIL"
            note = res.get("synth_error", "")
            print(f"  [{i:>3}/{len(rows)}] {mark} {res['id']} {note}", file=real_stdout, flush=True)
    sys.stdout = real_stdout
    done.sort(key=lambda r: r["id"])

    with out.open("w", encoding="utf-8") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fails = sum(1 for r in done if not r.get("ehr"))
    print(f"\n  {len(done)} cases -> {out.name}  ({fails} failed)")


if __name__ == "__main__":
    main()
