#!/usr/bin/env python
"""The queryable patient environment: EviDx's Examiner over our EHRs.

The Examiner is imported, not ported. It is 230 lines of dict lookups over the
synthesised EHR with no external dependencies, and `EXAMINER_TOOL_DEFINITIONS`
is already in OpenAI tool-call shape, so reimplementing it would only introduce
drift. Eight tools:

    get_available_data_menu     what this record contains -- counts and names
    get_patient_demographics    get_vital_signs        get_history
    review_system               perform_physical_exam
    get_lab_results             get_imaging_reports

The menu is the piece that matters. GoS's DiagnosisArena interface tells the
agent only that "pre-prepared auxiliary evidence texts" exist, so it must guess
what was ordered, and an empty return conflates "not ordered", "not in this
record" and "the retriever missed your phrasing". Worse, absence is strongly
informative here -- test sets barely overlap across cases, so a miss rules out
whole diagnostic families, and unlimited free probing turns that into a side
channel. The menu hands the same information to everyone in one call and makes
coverage measurable instead.

**Narrative mode is a flag, not a fork.** `chief_complaint` is EviDx's
one-sentence opening with everything else gated. `presentation` shows the
history and examination and gates only the diagnostic workup, which is GoS's
split and keeps continuity with our earlier numbers. Both were emitted at
synthesis, so switching costs nothing.

Coverage is tracked the way EviDx defines it: the fraction of menu-listed
sections actually queried. It is recorded rather than enforced -- their Observer
harness uses it to veto termination, and that is their method, not the
environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVIDX = Path(os.environ.get("EVIDX_ROOT", "EviDx"))
sys.path.insert(0, str(EVIDX / "agent"))

from Exam.examiner import Examiner  # noqa: E402
from Exam.examiner_prompts import EXAMINER_USAGE_INSTRUCTION  # noqa: E402
from Exam.examiner_tools import EXAMINER_TOOL_DEFINITIONS  # noqa: E402

NARRATIVE_MODES = ("chief_complaint", "presentation")

#: Sections the menu advertises, for coverage. Labs and imaging are counted as
#: one section each rather than per item: an agent that reads three of fourteen
#: labs has queried the section, and per-item coverage would mostly measure how
#: many labs a case happens to have.
COVERAGE_SECTIONS = ["demographics", "vital_signs", "history", "review_of_systems",
                     "physical_exam", "lab_results", "imaging_reports"]

_TOOL_SECTION = {
    "get_patient_demographics": "demographics",
    "get_vital_signs": "vital_signs",
    "get_history": "history",
    "review_system": "review_of_systems",
    "perform_physical_exam": "physical_exam",
    "get_lab_results": "lab_results",
    "get_imaging_reports": "imaging_reports",
}


class MedEnv:
    """One case, queryable. Stateless apart from the trace it records."""

    def __init__(self, row: dict, narrative: str = "presentation"):
        if narrative not in NARRATIVE_MODES:
            raise ValueError(f"narrative must be one of {NARRATIVE_MODES}")
        self.row = row
        self.narrative = narrative
        self.ehr = row.get("ehr") or {}
        self._ex = Examiner()
        self._ex.load_ehr(self.ehr)
        self.trace: list[dict] = []
        self._queried: set[str] = set()

    # -- what the agent starts with -------------------------------------- #

    @property
    def opening(self) -> str:
        text = self.row.get(self.narrative) or self.row.get("chief_complaint") or ""
        return text.strip()

    @property
    def question(self) -> str:
        return (self.row.get("synth_clinical_question")
                or self.row.get("question", "").rsplit("===", 1)[-1]).strip()

    def tools(self) -> list[dict]:
        return EXAMINER_TOOL_DEFINITIONS

    def usage_instruction(self) -> str:
        return EXAMINER_USAGE_INSTRUCTION

    # -- querying --------------------------------------------------------- #

    def call(self, name: str, arguments: dict | None = None) -> str:
        args = arguments or {}
        try:
            out = self._ex.execute_tool(name, args)
        except Exception as exc:  # a bad call is an observation, not a crash
            out = f"Tool error: {type(exc).__name__}: {exc}"
        sec = _TOOL_SECTION.get(name)
        if sec and not out.lower().startswith(("no ", "system error", "tool error")):
            self._queried.add(sec)
        self.trace.append({"tool": name, "arguments": args, "result": out})
        return out

    # -- what was seen ----------------------------------------------------- #

    def available_sections(self) -> list[str]:
        """Sections this record actually has, so coverage is not penalised for
        asking about labs on a case that has none."""
        present = []
        for s in COVERAGE_SECTIONS:
            if s == "vital_signs":
                v = (self.ehr.get("physical_exam") or {}).get("vital_signs")
            elif s == "history":
                v = any(self.ehr.get(k) for k in
                        ("past_medical_history", "surgical_history", "social_history",
                         "family_history", "allergies", "current_medications"))
            else:
                v = self.ehr.get(s)
            if v:
                present.append(s)
        return present

    def coverage(self) -> float:
        avail = set(self.available_sections())
        if not avail:
            return 0.0
        return len(self._queried & avail) / len(avail)

    def summary(self) -> dict:
        return {"id": self.row.get("id"), "dataset": self.row.get("dataset"),
                "narrative": self.narrative, "calls": len(self.trace),
                "coverage": round(self.coverage(), 3),
                "sections_available": self.available_sections(),
                "sections_queried": sorted(self._queried)}


def load(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect one case's environment.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--case", default="", help="case id; default the first")
    ap.add_argument("--narrative", choices=NARRATIVE_MODES, default="presentation")
    a = ap.parse_args()

    rows = load(a.inp)
    row = next((r for r in rows if r["id"] == a.case), rows[0])
    env = MedEnv(row, a.narrative)

    print(f"=== {row['id']}  ({a.narrative}) ===\n")
    print("OPENING:"); print(" ", env.opening[:600].replace("\n", "\n  "))
    print("\nQUESTION:"); print(" ", env.question[:200])
    print("\nMENU:"); print("  " + env.call("get_available_data_menu").replace("\n", "\n  "))
    print("\nLABS:"); print("  " + env.call("get_lab_results")[:400].replace("\n", "\n  "))
    print(f"\nSUMMARY: {json.dumps(env.summary())}")
    print(f"GOLD: {row.get('gold_text')}")
