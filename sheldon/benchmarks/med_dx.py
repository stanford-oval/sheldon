"""Active clinical diagnosis over EviDx's three medical benchmarks.

DiagnosisArena, JAMA Clinical Challenge and MedXpertQA-Diag, each case turned
into a queryable patient record: the agent sees a presenting narrative and must
ask for history, examination, labs and imaging through tools.

This exists as a benchmark rather than a standalone runner so that `direct`,
`sheldon` and every engine work on it unchanged, and so budget, tracing, caching,
resume and reporting come for free. A bespoke runner reimplements all of that
and gets it worse.

**Where the data comes from.** `scripts/prepare_med_dx.py` builds the corpus:
case text and gold from Med-Evidence-2.6k (which is also what EviDx runs on),
multiple-choice options joined from the original datasets, and a structured EHR
synthesised by EviDx's own `NarrativeSynthesizer`. Measured on that corpus:
99.2% of annotated diagnosis-critical evidence survives synthesis.

**Two departures from EviDx, both measured.**

A leakage guard was added to the synthesis prompts. Without it 13 of 30 records
stated a diagnosis in a section visible before any tool call, and the
synthesiser was inferring rather than copying -- one case's source contains
"epidermolytic" zero times and the record opened with "Most likely diagnosis:
epidermolytic ichthyosis". Their prompts have no rule against naming the answer.
After the guard: zero such additions across 300 cases, with evidence survival
unchanged.

Answer options are stripped from JAMA case text before synthesis. 34 of 100 JAMA
cases had option strings running into the narrative with no delimiter, and the
synthesiser filed them under `other_procedures` -- so the record stated that
candidate answers had already been performed.

The cost of both is that the environment is ours, not theirs, so numbers here do
not belong in EviDx's published table.

**Scoring is EviDx's Level 3, through our LLM.** The paper's evaluation
pyramid has three levels. Level 1 (MCP syntax and schema errors) and Level 2
(uncertainty convergence over the Diagnostician's self-reported entropy, and
Observer interventions) are properties of their agent harness and have no
counterpart here: our engines validate tool calls before they run, `direct` has
no belief distribution to take an entropy of, and there is no Observer. Level 3
is implemented in full, with their code where it can be imported and their
prompts verbatim where it cannot:

- `open`: their `Judge.evaluate_openqa` protocol -- the same system prompt, the
  same user content (question, inferred intent, options, gold key and text,
  the answer core), relation entailed / partial / unrelated / contradicts.
  `open` is what their code names `openqa_score` (entailed or partial);
  `open_strict` is entailed only. The headline `score` is `open_strict`: a
  broader category is not the diagnosis, and the partial-inclusive number is
  in every row for anyone who wants to compare against a table that used it.
- `mc`: their dual-track design -- open answer first, then the same report is
  mapped to one option under their EXAM MODE prompt, graded deterministically
  with their `_normalize_choice`. Their exam turn is taken by the Diagnostician
  with Consultant tools; ours is one tool-free call by the run's model, and
  `detail.mc_model` records which. `mc` is therefore not identical to theirs.
- `acq` and `cog`: Reference Evidence Recall over the Med-Evidence-2.6k
  annotations, per item: acquisition by exact match on the initial narrative
  plus every tool output, then their LLM-semantic prompt; cognition by their
  prompt over the agent's intermediate reasoning (`Prediction.belief`, where a
  method records one) and the final report. Fractions of the case's items.

The annotations are gold-side data and live on the benchmark, not on the task:
`Task.blind()` keeps `meta`, and the quotes say exactly which findings matter.

**The menu is the point of the tool interface.** `get_available_data_menu` lists
what a record contains -- names and counts, no results -- so an agent queries
deliberately. Without it, absence becomes a side channel: test sets barely
overlap across cases, so an empty return rules out whole diagnostic families,
and unlimited free probing turns the shape of the workup into an oracle.

    sheldon run -b med-dx -s DiagnosisArena -m direct,sheldon --tag med1
    sheldon run -b med-dx -s JAMA -p benchmark_opts.narrative=chief_complaint
    sheldon run -b med-dx -s DiagnosisArena -m sheldon -p benchmark_opts.evidence_strength=0.7

`evidence_strength` is the ceiling on what a critic's reading of the record may
do to a hypothesis. The probes themselves never refute, and the shipped default
of 0.0 also stopped the falsifier: its verdict strength is capped at the
ceiling of the probes it consulted, so every refutation on this benchmark landed
with a delta of zero (98 of 107 in the recorded 50-case sheldon run; the rest were
renormalisation noise) and `integrate` bearings were capped to nothing. The
default is kept at 0.0 so existing run ids still mean what they meant; 0.7 is
the harness's ordinary evidential ceiling (`Probe.max_strength`), the one an
ORCA shell probe already gets.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from sheldon.benchmarks.base import Benchmark
from sheldon.core.io import read_jsonl
from sheldon.core.probes import Probe, ProbeResult
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task

#: EviDx's Examiner is imported rather than reimplemented: it is dict lookups
#: over the synthesised record with no dependencies, and reimplementing it would
#: only introduce drift between what we measure and what they built.
#: Where the EviDx checkout lives: `benchmark_opts.evidx_root`, else
#: `EVIDX_ROOT`, else `./EviDx`.
_EVIDX_DEFAULT = os.environ.get("EVIDX_ROOT", "EviDx")

NARRATIVE_MODES = ("chief_complaint", "presentation")


def evidence_strength(opts: dict[str, Any]) -> float:
    """The ceiling a record-backed refutation or bearing may reach; see module doc."""
    raw = opts.get("evidence_strength", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"med-dx: evidence_strength must be a number, got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"med-dx: evidence_strength must be in [0, 1], got {value}")
    return value


def _judge_module(root: str):
    """EviDx's Judge (normalisers, intent inference) and its two system prompts.

    Imported, not copied, for the same reason the Examiner is: the module is
    pure string handling with no client at import time, so the only thing a
    copy could add is drift. The call itself goes through `ctx.llm`.
    """
    agent = str(Path(root) / "agent")
    if agent not in sys.path:
        sys.path.insert(0, agent)
    from Judge import judge_prompts  # noqa: PLC0415
    from Judge.judge import Judge  # noqa: PLC0415

    return Judge, judge_prompts


#: Their `evaluate_evidence_recall.py` builds an OpenAI client at import time,
#: which is the one thing a benchmark module here may not do, so its two
#: prompts are carried verbatim. Placeholders renamed to `str.format` fields
#: only; the wording is theirs.
ACQUISITION_PROMPT = """
### Task
Determine if the specific **Medical Evidence** described below is present in the provided **Text Data** (which contains patient history, tool outputs, etc.).
Be flexible with phrasing, formatting, and abbreviations (e.g., "NG tube" = "nasogastric tube").
### Target Evidence
- Raw Text: "{original_text}"
- Clinical Concept: "{feature_name}"
### Text Data (What the Agent saw)
{context}
### Question
Does the Text Data contain the information described in the Target Evidence?
Answer JSON: {{"present": true, "reason": "short explanation"}} or {{"present": false}}
"""

COGNITION_PROMPT = """
### Role
You are a strict Medical Logic Evaluator.
### Task
Determine if the AI Physician explicitly **considered and used** a specific piece of "Golden Evidence" during its diagnostic reasoning process.
### Golden Evidence
- Raw Text: "{original_text}"
- Concept: "{feature_name}"
- Role: {ev_type}
### Agent's Thinking Process
{thoughts}
### Agent's Final Report
{report}
### Evaluation Criteria
1. **Hit**: The agent explicitly mentions the evidence (or semantic equivalent) AND uses it to support/refute a hypothesis.
2. **Miss**: The agent ignores this evidence or only mentions it without linking it to the diagnosis.
### Output Format
Return ONLY a JSON object:
{{
  "recalled": true,
  "reason": "Brief explanation."
}}
"""

#: Their EXAM MODE turn (`agent/main.py`), minus the line offering Consultant
#: tools -- this turn has none -- and with the `submit_mcq_answer` call
#: replaced by the same JSON as a structured reply.
EXAM_MODE_PROMPT = """EXAM MODE.

You have already produced an OpenQA clinical report.
Now map the report to the SINGLE BEST answer choice.

--- TASK DEFINITION (MUST FOLLOW) ---
QuestionFocus = {question_focus}
ClinicalQuestion = {clinical_question}

Rules:
- Your answer MUST address the ClinicalQuestion, NOT necessarily the primary diagnosis.
- If QuestionFocus is "workup_associated_anomaly" / "associated_condition":
  you MUST choose the option that is the associated condition to investigate (e.g., VSD),
  NOT the already-established diagnosis (e.g., EA/TEF), unless the question explicitly asks diagnosis.
- When ready, answer with JSON: {{"answer":"A","reasoning":"..."}}

--- QUESTION (RAW) ---
{raw}

--- YOUR OPENQA REPORT ---
{report}

--- OPTIONS ---
{options}"""


class _OpenQACandidate(BaseModel):
    choice: str = "Unclear"
    confidence: float = 0.0
    why: str = ""


class OpenQAJudgement(BaseModel):
    """The JSON their open-QA judge prompt asks for."""

    short_answer: str = ""
    relation_to_gt: str = "unrelated"
    confidence: float = 0.0
    why: str = ""
    evidence_spans: list[str] = []
    best_matching_choice: str = "Unclear"
    candidates: list[_OpenQACandidate] = []


class McqAnswer(BaseModel):
    answer: str = ""
    reasoning: str = ""


class EvidencePresent(BaseModel):
    present: bool = False
    reason: str = ""


class EvidenceRecalled(BaseModel):
    recalled: bool = False
    reason: str = ""


def _normalize_text(text: str) -> str:
    """Their `normalize_text`: whitespace-collapsed, lower-cased."""
    return re.sub(r"\s+", " ", str(text).lower().strip()) if text else ""


def _tool_outputs(task: Task, ctx: RunContext | None) -> list[str]:
    """Every tool output the run saw, in order: the live record when scoring
    inside a run, or what a re-scorer read back from `runs/<id>/retrievals/`."""
    tools = getattr(ctx, "tools", None) if ctx is not None else None
    record = getattr(tools, "record", None)
    if record:
        return [str(r.result.summary) for r in record]
    return [str(x) for x in (task.payload.get("_tool_outputs") or [])]


def _belief_thoughts(belief: dict[str, Any], limit: int = 12000) -> str:
    """The "thinking process" the cognition judge reads, from `Prediction.belief`.

    The belief method's snapshot lists every hypothesis -- per-observation
    reasons, full history -- ahead of the observation ledger, so a JSON dump
    cut at `limit` ends before the first observation on any non-trivial run
    and the judge is asked whether the agent used evidence it is never shown.
    That shape is rendered instead: the observations first, then one line per
    hypothesis with what supports and refutes it and why, and the cut comes
    last. The old belief_loop shape keeps the dump, so recorded runs grade as
    they did; `observation_space` is the ledger only the new method writes.
    """
    if not isinstance(belief, dict) or "observation_space" not in belief:
        return json.dumps(belief, ensure_ascii=False)[:limit]

    def cite(ids: list[str], why: dict[str, str]) -> str:
        return "; ".join(f"{o} ({why[o]})" if why.get(o) else o for o in ids) or "none"

    lines = ["Observations:", *(str(o) for o in belief.get("observations") or [])]
    lines += ["", f"Hypotheses (leader: {belief.get('leader') or 'none'}):"]
    for h in belief.get("hypotheses") or []:
        status = str(h.get("status", ""))
        reason = h.get("lowered_reason") or h.get("retired_reason")
        if status == "retired" and reason:
            status += f": {reason}"
        elif status == "refined" and h.get("successor"):
            status += f" into {h['successor']}"
        why = h.get("relationship_reason") or h.get("why") or {}
        lines.append(
            f"[{h.get('id')}] {h.get('statement', '')}  ({status}, score {h.get('score', 0)}) "
            f"supports: {cite(h.get('supports') or [], why)}; "
            f"refutes: {cite(h.get('refutes') or [], why)}"
        )
    return "\n".join(lines)[:limit]


def _examiner_module(root: str):
    agent = str(Path(root) / "agent")
    if agent not in sys.path:
        sys.path.insert(0, agent)
    from Exam.examiner import Examiner  # noqa: PLC0415
    from Exam.examiner_tools import EXAMINER_TOOL_DEFINITIONS  # noqa: PLC0415

    return Examiner, EXAMINER_TOOL_DEFINITIONS


#: One EviDx tool description is written as an instruction rather than a
#: description: `get_available_data_menu` opens "CRITICAL FIRST STEP" and says
#: "Always call this first". That is a call ordering handed to whichever method
#: is being measured, and their `EXAMINER_USAGE_INSTRUCTION` goes further --
#: "You MUST call `get_available_data_menu` as your very first action", then a
#: hypothesis-driven search protocol. Neither is given here: an investigation
#: strategy supplied by the environment is the environment doing the method's
#: work, and the point of the baseline is to find out what a method does on its
#: own. The informational half is kept, so the menu is still discoverable.
#:
#: Rewritten here rather than in EviDx, which stays vendored unmodified.
_NEUTRAL_DESCRIPTION = {
    "get_available_data_menu": (
        "Returns a summary list (index) of what data exists in this patient's "
        "record, so test names do not have to be guessed."
    ),
}


class ExaminerProbe(Probe):
    """One Examiner tool, over one patient record.

    Never refutes: reading a record is retrieval, not a verdict on a hypothesis.
    A probe that claimed refutation here would let the belief loop treat "the
    record does not mention X" as evidence against X, which is exactly the
    inference the menu exists to prevent.

    `max_strength` is a separate question -- how far the *critic's* reading of
    what the probe returned may move a hypothesis. A biopsy report that says
    "diffuse" when the claim predicts "lobular" is evidence against the claim
    even though no probe asserted so. At 0.0 the falsifier's every verdict is
    zeroed by the consulted-probe cap in `falsifier.py`; the benchmark option
    `evidence_strength` sets it, and stays at 0.0 by default for the sake of
    existing run ids.
    """

    decisive = False
    max_strength = 0.0

    def __init__(self, examiner: Any, definition: dict, max_strength: float = 0.0) -> None:
        fn = definition["function"]
        self.name = fn["name"]
        self.description = _NEUTRAL_DESCRIPTION.get(self.name, fn["description"])
        self._schema = fn.get("parameters") or {"type": "object", "properties": {}}
        self._examiner = examiner
        self.max_strength = max_strength  # type: ignore[misc]

    @property
    def input_schema(self) -> dict[str, Any]:
        props = dict(self._schema.get("properties") or {})
        return {
            "type": "object",
            "properties": props,
            "required": list(self._schema.get("required") or []),
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        try:
            out = self._examiner.execute_tool(self.name, kwargs)
        except Exception as exc:  # a bad call is an observation, not a crash
            return ProbeResult(summary=f"Tool error: {type(exc).__name__}: {exc}", ok=False)
        text = str(out or "")
        empty = text.lower().startswith(("no ", "system error", "tool error"))
        # The whole result is the point here, so do not let the default
        # five-row render silently truncate a lab panel.
        return ProbeResult(summary=text, ok=not empty, limit=8000)


@benchmarks.register("med-dx", "med", "medical")
class MedDx(Benchmark):
    name = "med-dx"
    splits = ("DiagnosisArena", "JAMA", "MedXpertQA-Diag")
    blurb = "active clinical diagnosis over a queryable patient record (arXiv:2608.24570)"
    answer_kind = "diagnosis"
    #: The Examiner is the only access to the record, so an engine that brings
    #: its own shell still needs it -- see `Benchmark.probes_essential`.
    probes_essential = True

    #: Vocabulary for `sheldon explain`. Without it the analysis still runs, but
    #: asks about "a specific entity rather than a category" where it should ask
    #: about an organism, a stage, a tissue of origin -- which is the difference
    #: between "the gold names something absent" and knowing *what kind* of thing
    #: is absent. See `sheldon/explain.py`.
    explain_profile: ClassVar[dict[str, str]] = {
        "answer_noun": "diagnosis",
        "requirement_kinds": (
            "a tissue of origin, an organism species, a disease stage or grade, a "
            "laterality, a subtype qualifier, a named mechanism, a confirming test result"
        ),
        "proximity_wording": (
            '- "parent"   : a broader category that strictly CONTAINS the gold (gold '
            '"M. kansasii infection", held "mycobacterial infection"). Refining it -- '
            "adding the subtype, organism or stage -- reaches the gold.\n"
            '- "sibling"  : the same disease family, wrong member (gold "Crohn disease", '
            'held "ulcerative colitis"). Reaching gold needs a switch, not a refinement.\n'
            '- "adjacent" : the right organ system or broad mechanism, a different '
            "disease process.\n"
            '- "unrelated": nothing held bears on the gold.'
        ),
    }

    def explain_record(self, task: Task) -> str:
        """The narrative plus the whole structured record.

        The default would show only `task.question`, which here is the presenting
        narrative alone -- everything behind the Examiner would then read as
        absent, and every gated lab would be scored as an unreachable gold.
        """
        ehr = (task.payload or {}).get("ehr")
        parts = [f"=== PRESENTED TO THE METHOD ===\n{task.question}"]
        if ehr:
            parts.append("=== COMPLETE PATIENT RECORD (all fields, all values) ===\n"
                         + json.dumps(ehr, ensure_ascii=False, indent=1))
        return "\n\n".join(parts)
    answer_format = (
        "Give the single most likely answer to the clinical question as a concise "
        "clinical statement on its own line -- a diagnosis, or the specific action "
        "asked for. Do not hedge across alternatives."
    )

    #: Reference evidence per task id, filled by `load()`. Gold-side: never on the task.
    _golden: dict[str, list[dict[str, Any]]]

    def __init__(self, root: str | Path | None = None, **opts: Any) -> None:
        super().__init__(root, **opts)
        self._golden = {}

    # -- loading -----------------------------------------------------------

    def load(
        self, split: str = "DiagnosisArena", limit: int | None = None, seed: int = 0
    ) -> list[Task]:
        narrative = str(self.opts.get("narrative", "presentation"))
        if narrative not in NARRATIVE_MODES:
            raise ValueError(f"{self.name}: narrative must be one of {NARRATIVE_MODES}")

        path = self.root / f"ehr-{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: {path} not found. Run scripts/prepare_med_dx.py first."
            )

        tasks: list[Task] = []
        for row in read_jsonl(path):
            if not row.get("ehr"):
                continue
            opening = (row.get(narrative) or row.get("chief_complaint") or "").strip()
            question = (
                row.get("synth_clinical_question") or row.get("question", "").rsplit("===", 1)[-1]
            ).strip()
            tasks.append(
                Task(
                    id=str(row["id"]),
                    benchmark=self.name,
                    question=f"{opening}\n\nClinical question: {question}",
                    payload={
                        "ehr": row["ehr"],
                        "options": row.get("options") or {},
                        "narrative": narrative,
                        "clinical_question": question,
                    },
                    answer=row.get("gold_text", ""),
                    meta={
                        "dataset": row.get("dataset", split),
                        "answer_idx": row.get("answer_idx", ""),
                        "question_focus": row.get("question_focus", ""),
                    },
                )
            )
            self._golden[str(row["id"])] = list(row.get("golden_evidence") or [])
        return self.subsample(tasks, limit, seed)

    # -- tools -------------------------------------------------------------

    def probes(self, task: Task) -> list[Probe]:
        """The Examiner over this task's record. No Python sandbox: there is
        nothing to compute here, and offering one invites the model to write
        code against a record it can only read."""
        Examiner, definitions = _examiner_module(str(self.opts.get("evidx_root", _EVIDX_DEFAULT)))
        ex = Examiner()
        ex.load_ehr(task.payload.get("ehr") or {})
        ceiling = evidence_strength(self.opts)
        return [ExaminerProbe(ex, d, max_strength=ceiling) for d in definitions]

    # -- scoring: EviDx Level 3 -----------------------------------------------

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        answer = str(getattr(pred, "answer", "") or "").strip()
        gold = str(task.answer or "").strip()
        options = {str(k): str(v) for k, v in (task.payload.get("options") or {}).items()}
        gold_key = str(task.meta.get("answer_idx", "")).strip()
        question = str(task.payload.get("clinical_question") or task.question)
        judge, prompts = _judge_module(str(self.opts.get("evidx_root", _EVIDX_DEFAULT)))
        outputs = _tool_outputs(task, ctx)
        golden = self._golden.get(task.id, [])

        metrics: dict[str, float] = {
            "open": 0.0,
            "open_strict": 0.0,
            "mc": 0.0,
            "acq": 0.0,
            "cog": 0.0,
            "evidence_n": float(len(golden)),
            "tool_calls": float(len(outputs)),
        }
        detail: dict[str, Any] = {
            "gold": gold,
            "answer": answer,
            "answer_idx": gold_key,
            "question_focus": task.meta.get("question_focus", ""),
            "dataset": task.meta.get("dataset", ""),
        }

        if answer:
            open_eval = await self._judge_open(
                ctx, judge, prompts, question, options, answer, gold_key
            )
            metrics["open"] = 1.0 if open_eval["openqa_score"] else 0.0
            metrics["open_strict"] = 1.0 if open_eval["openqa_score_strict"] else 0.0
            detail.update(
                relation=open_eval["relation_to_gt"],
                grade=open_eval["openqa_grade"],
                short_answer=open_eval.get("short_answer", ""),
                why=open_eval.get("why", ""),
                best_matching_choice=open_eval.get("best_matching_choice", "Unclear"),
                question_intent=open_eval.get("question_intent", ""),
            )
            if options and gold_key:
                mc_eval = await self._exam_mc(ctx, judge, task, answer, options, gold_key)
                metrics["mc"] = 1.0 if mc_eval["is_correct"] else 0.0
                detail.update(mc_choice=mc_eval["student_choice"], mc_model=mc_eval["model"])
        else:
            detail["reason"] = "empty answer"

        if golden and self.opts.get("evidence_recall", True):
            ev = await self._evidence_recall(ctx, task, pred, answer, outputs, golden)
            metrics["acq"] = ev["acquisition_score"]
            metrics["cog"] = ev["recall_score"]
            detail["evidence"] = ev["details"]

        return Score(task_id=task.id, score=metrics["open_strict"], metrics=metrics, detail=detail)

    async def _judge_open(
        self,
        ctx: RunContext,
        judge: Any,
        prompts: Any,
        question: str,
        options: dict[str, str],
        answer: str,
        gold_key: str,
    ) -> dict[str, Any]:
        """`Judge.evaluate_openqa`, with the model call routed through `ctx.llm`.

        The user content is built exactly as theirs is (`tests/test_med_dx_official.py`
        holds the two side by side). Our methods return one answer, not their
        finish-tool payload, so the answer core carries `best_answer_text` and
        the focus stays "unknown" -- which is also what their code does with a
        plain-string report, and it does not enter the score unless
        `require_focus_match_openqa` is on, which it is not by default.
        """
        gt_norm = judge._normalize_choice(gold_key, options)
        gt_text = options.get(gt_norm, "")
        intent = judge._infer_intent_from_question(question)
        answer_core = {"question_focus": "unknown", "best_answer_text": answer}
        user_content = f"""
--- QUESTION ---
{question}

--- QUESTION_INTENT ---
{intent}

--- OPTIONS ---
{judge._format_options(options)}

--- GROUND TRUTH KEY ---
{gold_key}

--- GROUND TRUTH TEXT ---
{gt_text}

--- STUDENT OPENQA ANSWER CORE ---
{json.dumps(answer_core, ensure_ascii=False, indent=2)}
"""
        res = await ctx.llm.complete(
            user_content,
            system=prompts.JUDGE_OPENQA_SYSTEM_PROMPT,
            model=self.opts.get("judge_model"),
            schema=OpenQAJudgement,
            tag="judge",
        )
        out = res.parsed.model_dump() if res.parsed is not None else _parse(res.text)
        relation = judge._normalize_relation(out.get("relation_to_gt", "unrelated"))
        strict = relation == "entailed"
        weak = relation in ("entailed", "partial")
        if relation == "contradicts":
            grade = "contradicts"
        elif strict:
            grade = "correct"
        elif weak:
            grade = "weak_correct"
        else:
            grade = "incorrect"
        out.update(
            relation_to_gt=relation,
            ground_truth_norm=gt_norm,
            ground_truth_text=gt_text,
            best_matching_choice=judge._normalize_choice(
                out.get("best_matching_choice", "Unclear"), options
            ),
            question_intent=intent,
            openqa_score_strict=strict,
            openqa_score_weak=weak,
            openqa_score=weak,
            openqa_grade=grade,
        )
        return out

    async def _exam_mc(
        self,
        ctx: RunContext,
        judge: Any,
        task: Task,
        answer: str,
        options: dict[str, str],
        gold_key: str,
    ) -> dict[str, Any]:
        """Their EXAM MODE turn: map the open report to one option, graded as
        `Judge.evaluate_mcq` grades a structured `submit_mcq_answer` payload."""
        body = "\n".join(f"({k}) {v}" for k, v in options.items())
        prompt = EXAM_MODE_PROMPT.format(
            question_focus=task.meta.get("question_focus", "") or "diagnosis",
            clinical_question=task.payload.get("clinical_question", ""),
            raw=task.question,
            report=answer,
            options=body,
        )
        res = await ctx.llm.complete(prompt, schema=McqAnswer, tag="mc")
        raw = res.parsed.answer if res.parsed is not None else _parse(res.text).get("answer", "")
        choice = judge._normalize_choice(raw, options)
        gt = judge._normalize_choice(gold_key, options)
        return {
            "student_choice": choice,
            "is_correct": choice != "Unclear" and gt != "Unclear" and choice == gt,
            "model": res.model,
        }

    async def _evidence_recall(
        self,
        ctx: RunContext,
        task: Task,
        pred: Prediction,
        answer: str,
        outputs: list[str],
        golden: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """`evaluate_single_case` from their `evaluate_evidence_recall.py`.

        Acquisition context is the initial narrative plus every tool output, as
        theirs is, so a clue stated in the presenting narrative counts as
        acquired without a tool call. The "thinking process" their cognition
        judge reads is the Diagnostician's per-step reasoning; the nearest thing
        a method here records is `Prediction.belief`, and a method that records
        nothing is judged on its final report alone.
        """
        context = "\n\n".join(
            [
                f"=== INITIAL NARRATIVE ===\n{task.question}",
                "=== TOOL OUTPUTS ===\n" + "\n".join(outputs),
            ]
        )
        norm_ctx = _normalize_text(context)
        thoughts = (
            _belief_thoughts(pred.belief)
            if getattr(pred, "belief", None)
            else "(no intermediate reasoning was recorded for this method)"
        )
        judge_model = self.opts.get("judge_model")

        async def acquired(ev: dict[str, Any]) -> dict[str, Any]:
            text = str(ev.get("original_text", ""))
            feature = str(ev.get("feature_name", ""))
            if _normalize_text(text) and _normalize_text(text) in norm_ctx:
                return {"acquired": True, "method": "Rule-Exact"}
            if feature and (len(feature.split()) > 1 or len(feature) > 6):
                if _normalize_text(feature) in norm_ctx:
                    return {"acquired": True, "method": "Rule-Feature"}
            res = await ctx.llm.complete(
                ACQUISITION_PROMPT.format(
                    original_text=text, feature_name=feature, context=context[:20000]
                ),
                model=judge_model,
                schema=EvidencePresent,
                tag="judge",
            )
            if res.parsed is not None:
                present = bool(res.parsed.present)
            else:
                present = bool(_parse(res.text).get("present"))
            return {"acquired": present, "method": "LLM-Semantic" if present else "Missed"}

        async def used(ev: dict[str, Any]) -> dict[str, Any]:
            res = await ctx.llm.complete(
                COGNITION_PROMPT.format(
                    original_text=str(ev.get("original_text", "")),
                    feature_name=str(ev.get("feature_name", "")),
                    ev_type=str(ev.get("type", "Evidence")),
                    thoughts=thoughts,
                    report=answer or "(no answer)",
                ),
                model=judge_model,
                schema=EvidenceRecalled,
                tag="judge",
            )
            if res.parsed is not None:
                return {"recalled": bool(res.parsed.recalled), "reason": res.parsed.reason}
            out = _parse(res.text)
            return {"recalled": bool(out.get("recalled")), "reason": str(out.get("reason", ""))}

        acq = await asyncio.gather(*(acquired(ev) for ev in golden))
        cog = await asyncio.gather(*(used(ev) for ev in golden))
        details = [
            {
                "original_text": ev.get("original_text"),
                "type": ev.get("type"),
                "status_acquired": a["acquired"],
                "acq_method": a["method"],
                "status_used": c["recalled"],
                "judge_reason": c.get("reason", ""),
            }
            for ev, a, c in zip(golden, acq, cog, strict=True)
        ]
        n = len(golden)
        n_acq = sum(1 for a in acq if a["acquired"])
        n_cog = sum(1 for c in cog if c["recalled"])
        return {
            "total_evidence": n,
            "acquired_count": n_acq,
            "cognition_hit_count": n_cog,
            "acquisition_score": n_acq / n if n else 0.0,
            "recall_score": n_cog / n if n else 0.0,
            "details": details,
        }

    # -- setup -------------------------------------------------------------

    def check(self) -> list[str]:
        problems = []
        if not self.root.exists():
            return [f"data directory not found: {self.root} (run scripts/prepare_med_dx.py)"]
        missing = [s for s in self.splits if not (self.root / f"ehr-{s}.jsonl").exists()]
        if missing:
            problems.append(f"missing splits: {', '.join(missing)}")
        root = Path(str(self.opts.get("evidx_root", _EVIDX_DEFAULT)))
        try:
            found = (root / "agent" / "Exam" / "examiner.py").exists()
        except PermissionError:
            # Another user's checkout: unreadable is "not found" for us, and
            # `sheldon ls` must say so rather than crash on every benchmark.
            found = False
        if not found:
            problems.append(
                f"EviDx Examiner not found under {root} (set benchmark_opts.evidx_root)"
            )
        elif not (root / "agent" / "Judge" / "judge.py").exists():
            problems.append(f"EviDx Judge not found under {root}; scoring imports it")
        return problems


def _parse(text: str) -> dict:
    import re

    s = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        return {}
