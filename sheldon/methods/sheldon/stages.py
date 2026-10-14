"""The stages that decide: plan, explore, select, comparative analysis, exploit,
finalize.

Each stage is two functions: `*_step(run)` builds the call, and `apply_*(run,
out, retrievals)` folds its answer into the state. The mode decides how the
call is made -- on its own (`staged`) or as a turn of the round's
conversation (`session`) -- and nothing here knows which.

CandidateAnalysis is a stage of its own, in `candidate_analysis.py`: it runs as sub-agents in
both modes.
"""

from __future__ import annotations

from typing import Any

from sheldon.core.probes import Retrieval
from sheldon.core.types import Usage
from sheldon.methods.base import answer_format_hint
from sheldon.methods.retrievals import has_retrievals, render_retrievals
from sheldon.methods.sheldon.prompts import render
from sheldon.methods.sheldon.run import Run, Step
from sheldon.methods.sheldon.schemas import (
    SUBMIT_EVIDENCE,
    SUBMIT_EXPLORATION,
    SUBMIT_FINAL,
    SUBMIT_MERGES,
    SUBMIT_REVISION,
    SUBMIT_SELECTION,
    EvidenceOut,
    ExploreOut,
    FinalOut,
    MergeOut,
    ObservationDraft,
    PlanOut,
    ReviseOut,
    SelectOut,
)
from sheldon.methods.sheldon.state import PENDING, SHELVED, TESTED

# -- plan ---------------------------------------------------------------------


async def plan(run: Run) -> Usage:
    """Split the statement into the problem and the form of the answer. A split
    whose situation is empty or implausibly short is rejected and the whole
    statement stands."""
    if not run.param("plan", True):
        return Usage()
    ctx = run.ctx
    prompt = render("plan", question=run.task.question)
    res = await ctx.llm.complete(prompt.user, system=prompt.system, schema=PlanOut, tag="plan")
    ctx.budget.charge(res.usage)
    out: PlanOut | None = res.parsed
    floor = min(int(run.param("plan_min_chars", 120)), len(run.task.question))
    ok = bool(out and len(out.situation.strip()) >= floor)
    if ok:
        run.plan = out
    ctx.trace.log("plan", accepted=ok, requirements=list(out.requirements)[:12] if out else [])
    return res.usage


# -- explore ------------------------------------------------------------------


def _stagnating(run: Run) -> bool:
    return run.state.stagnating(int(run.param("stagnation_rounds", 2)))


def eligible(run: Run) -> list:
    return run.state.eligible(run.round, int(run.param("set_aside_cooldown", 3)), _stagnating(run))


def region_limit(run: Run) -> int | None:
    """`max_regions` per space, or None for no cap. Stamped on the state so
    every place that files an item under a region sees the same cap."""
    raw = run.param("max_regions", 10)
    limit = None if raw in (None, 0, "", False) else int(raw)
    run.state.region_limit = limit
    return limit


def merge_needed(run: Run) -> list[str]:
    """The spaces whose map is full, so explore could not open a region."""
    region_limit(run)
    return [space for space in ("h", "o") if run.state.full(space)]


def merge_step(run: Run, spaces: list[str]) -> Step:
    """One call before explore, with the model's whole attention on the map:
    which regions have turned out to mean the same thing, or gone nowhere,
    and what to fold them into. The model decides what merges; the harness
    only applies it and keeps the history."""
    s = run.state
    ctx = dict(
        run.base_context(), spaces=spaces, limit=s.region_limit,
        h_regions=s.render_regions("h"), o_regions=s.render_regions("o"),
        hypotheses=s.render_hypotheses(), observations=s.render_observations(),
    )
    return Step("merge_regions", ctx, MergeOut, SUBMIT_MERGES, "merge",
                steps=int(run.param("merge_probe_calls", 0)))


def apply_merge(run: Run, out: MergeOut | None, spaces: list[str]) -> dict[str, Any]:
    s, r = run.state, run.round
    merged, refused = [], []
    for d in (out.merges if out else []):
        ok, why = s.merge_regions(d.space, d.members, d.name, d.axis, d.description, d.why, r)
        (merged if ok else refused).append(f"{d.space}: {why}")
    still_full = [space for space in spaces if s.full(space)]
    result = dict(round=r, merged=merged, refused=refused, still_full=still_full)
    run.ctx.trace.log("merge.done", **result)
    return result


def explore_step(run: Run) -> Step:
    s, first = run.state, run.round == 1
    region_limit(run)
    n = int(run.param("n_initial", 5) if first else run.param("n_explore", 3))
    ctx = dict(
        run.base_context(), first=first, n=n, limit=s.region_limit,
        full=[space for space in ("h", "o") if s.full(space)],
        h_regions=s.render_regions("h"), o_regions=s.render_regions("o"),
        hypotheses=s.render_hypotheses(), observations=s.render_observations(),
        evidence=s.render_evidence(s.tested_by_round.get(run.round - 1, [])),
        eligible="\n".join(s.render_hypothesis(h) for h in eligible(run)),
        distinct="\n".join(s.render_hypothesis(h) for h in s.distinct_on_parts()
                           if h not in eligible(run)),
    )
    return Step("explore", ctx, ExploreOut, SUBMIT_EXPLORATION, "explore",
                steps=int(run.param("explore_probe_calls", 8)))


def apply_explore(run: Run, out: ExploreOut | None, _retrievals: list[Retrieval]) -> dict[str, Any]:
    s, r = run.state, run.round
    out = out or ExploreOut()
    regions: dict[str, list[str]] = {"added": [], "folded": [], "full": []}
    replaced = []
    for space, drafts in (("h", out.hypothesis_regions), ("o", out.observation_regions)):
        for d in drafts:
            if d.replaces_axis.strip() and d.axis.strip():
                if s.replace_axis(space, d.replaces_axis, d.axis):
                    replaced.append(f"{d.replaces_axis}->{d.axis}")
            verdict = s.add_region(space, d.axis, d.name, d.description, r)
            if verdict in regions:
                regions[verdict].append(f"{space}: {d.name}")
    added, dupes = [], []
    for d in out.hypotheses:
        h = s.add(d.statement, d.rationale, d.region, r)
        (added if h else dupes).append(h.id if h else d.statement[:80])
    allowed = {h.id for h in eligible(run)}
    revived, refused = [], []
    for hid in out.revive:
        h = s.hypotheses.get(hid.strip())
        if h is not None and h.id in allowed:
            s.revive(h, r)
            revived.append(h.id)
        else:
            refused.append(hid)
    result = dict(round=r, added=added, duplicates=dupes, revived=revived, revive_refused=refused,
                  regions_added=regions["added"], regions_folded=regions["folded"],
                  regions_refused_full=regions["full"], axes_replaced=replaced,
                  unplaced={space: s.unplaced(space) for space in ("h", "o")})
    run.ctx.trace.log("explore.done", **result)
    return result


# -- select -------------------------------------------------------------------


def select_needed(run: Run) -> bool:
    """Round 1 tests everything explore opened; later rounds choose only when
    there are more candidates than slots. Kept hypotheses never take a slot."""
    return run.round > 1 and len(run.state.of(PENDING)) > int(run.param("select_k", 5))


def select_step(run: Run) -> Step:
    s = run.state
    ctx = dict(
        run.base_context(), k=int(run.param("select_k", 5)),
        h_regions=s.render_regions("h"),
        candidates=s.render_hypotheses((PENDING,)), kept=s.render_hypotheses((TESTED,)),
        observations=s.render_observations(),
    )
    return Step("select", ctx, SelectOut, SUBMIT_SELECTION, "select", steps=int(run.param("select_probe_calls", 2)))


def apply_select(run: Run, out: SelectOut | None) -> list[str]:
    """Keep what was chosen, capped; shelve every other candidate. A choice that
    names no candidate falls back to the newest ones."""
    s, k = run.state, int(run.param("select_k", 5))
    pending = s.of(PENDING)
    ids = {h.id for h in pending}
    chosen = [hid for hid in dict.fromkeys(out.chosen if out else []) if hid in ids][:k]
    by = "model"
    if not chosen:
        chosen, by = [h.id for h in sorted(pending, key=lambda h: -h.born)][:k], "fallback"
    for h in pending:
        if h.id not in chosen:
            s.shelve(h, f"not selected in round {run.round}", run.round)
    run.ctx.trace.log("select.done", round=run.round, chosen=chosen, by=by,
                      reasoning=(out.reasoning if out else "")[:600])
    return chosen


# -- comparative analysis ----------------------------------------------------------------


def resumed_targets(run: Run) -> list[str]:
    """What the first round after a resume tests when nothing is pending.

    A finished run leaves no pending claim -- every one is tested or shelved --
    so a continuation depends on explore proposing something new, and explore is
    looking at the table it wrote last time: its proposals can all fold as
    duplicates and the round end `nothing_to_test`. The claims that earned their
    place are the ranked leaders, and a local loop starts from the hypothesis's
    own artifact, so re-testing them continues the work rather than restating
    it. Only the first resumed round: after that a dry round means
    what it has always meant.
    """
    if not run.resumed or run.round != run.resumed + 1:
        return []
    k = int(run.param("select_k", 5))
    chosen = [h.id for h in run.state.ranked()[:k] if h.artifact]
    run.ctx.trace.log("resume.targets", round=run.round, chosen=chosen)
    return chosen


def comparative_analysis_step(run: Run) -> Step:
    s = run.state
    ctx = dict(
        run.base_context(), o_regions=s.render_regions("o"),
        hypotheses=s.render_hypotheses((TESTED,)), observations=s.render_observations(),
    )
    return Step("comparative_analysis", ctx, EvidenceOut, SUBMIT_EVIDENCE, "comparative_analysis",
                steps=int(run.param("comparative_analysis_probe_calls", 10)))


def file_observations(run: Run, drafts: list[ObservationDraft], retrievals: list[Retrieval],
                      origin: str, default_hypothesis: str = "") -> list[str]:
    """File a stage's observations; one that cites nothing is taken to rest on
    the stage's own retrievals, and a verdict naming no hypothesis is on
    `default_hypothesis` where there is one."""
    own = [r.id for r in retrievals]
    filed = []
    for d in drafts:
        verdicts = []
        for v in d.verdicts:
            hid = v.hypothesis.strip() or default_hypothesis
            verdicts.append({"hypothesis": hid, "relation": v.relation, "why": v.why})
        o = run.state.observe(d.content, d.region, [x for x in d.sources if x.strip()] or own,
                              run.round, origin, verdicts)
        if o is not None:
            filed.append(o.id)
    return filed


def apply_comparative_analysis(run: Run, out: EvidenceOut | None, retrievals: list[Retrieval]) -> list[str]:
    filed = file_observations(run, (out.observations if out else []), retrievals, "global_test")
    run.ctx.trace.log("comparative_analysis.done", round=run.round, observations=filed,
                      summary=(out.summary if out else "")[:600])
    return filed


# -- exploit -------------------------------------------------------------------


def exploit_step(run: Run) -> Step:
    s = run.state
    ctx = dict(run.base_context(), hypotheses=s.render_hypotheses((TESTED,)),
               observations=s.render_observations(),
               evidence=s.render_evidence([h.id for h in s.of(TESTED)]))
    return Step("exploit", ctx, ReviseOut, SUBMIT_REVISION, "exploit", steps=int(run.param("exploit_probe_calls", 4)))


def apply_exploit(run: Run, out: ReviseOut | None) -> dict[str, list[str]]:
    """One decision per tested hypothesis; one the stage left out is kept.
    A refinement is a sibling and the parent is shelved behind it, so it can
    still be recovered and still ranks at the end."""
    s, r = run.state, run.round
    done: dict[str, list[str]] = {"keep": [], "repair": [], "refine": [], "shelve": []}
    decided = set()
    for op in (out.ops if out else []):
        h = s.hypotheses.get(op.hypothesis.strip())
        if h is None or h.status != TESTED or h.id in decided:
            continue
        decided.add(h.id)
        kind = op.op
        if kind == "repair" and not run.executable():
            kind = "keep"
        if kind == "repair":
            s.request_repair(h, op.rationale or "repair requested", r)
        elif kind == "refine":
            child = s.refine(h, op.statement, op.rationale, r)
            leader = s.leader()
            if child is None:
                kind = "keep"
            elif leader is None or leader.id != h.id:
                # The leader stays live beside its refinement, which has to beat it.
                s.shelve(h, f"refined into {child.id}", r)
        elif kind == "shelve" and s.leader() is not None and s.leader().id == h.id:
            # The run's best stays live: shelving it leaves the next round
            # nothing to build on.
            kind = "keep"
            run.ctx.trace.log("exploit.kept_leader", round=r, hypothesis=h.id, rationale=op.rationale[:300])
        elif kind == "shelve":
            s.shelve(h, op.rationale or "shelved by revise", r)
        done[kind].append(h.id)
    done["keep"] += [h.id for h in s.of(TESTED) if h.id not in decided]
    run.ctx.trace.log("exploit.done", round=r, **done)
    return done


# -- finalize -----------------------------------------------------------------


def final_step(run: Run) -> Step:
    s = run.state
    question = run.situation()
    deliverable, hint = run.deliverable(), answer_format_hint(run.ctx)
    if deliverable and deliverable.strip() != hint.strip():
        question = f"{question}\n\n{deliverable}"
    record = run.record()
    ranked = s.ranked()
    ctx = dict(
        run.base_context(), question=question, answer_hint=hint, requirements=run.requirements(),
        hypotheses="\n".join(s.render_hypothesis(h) for h in ranked) or s.render_hypotheses((PENDING, TESTED, SHELVED)),
        observations=s.render_observations(),
        retrievals=render_retrievals(record) if has_retrievals(record) else "",
    )
    return Step("finalize", ctx, FinalOut, SUBMIT_FINAL, "finalize",
                steps=int(run.param("finalize_probe_calls", 6)), required=True)
