"""Session mode: a round's deciding stages as turns of one conversation.

    system   the investigator's standing instructions
    user     explore            -> tool turns ... -> submit_exploration
    user     select (round 2+)  -> tool turns ... -> submit_selection
             [candidate analysis runs as sub-agents, outside the conversation]
    user     comparative analysis -> tool turns ... -> submit_evidence
    user     exploit            -> tool turns ... -> submit_revision

The explore turn opens the conversation, so it is the stage's full prompt with
the whole state. Every later turn is a `session_*` prompt: the stage's
instructions and only what is new or changed since the model last saw it --
compared per item on its rendered text -- because the rest is already in the
conversation. A new round opens a new conversation.

The conversation is driven through `LLM.converse` directly, with the run's
probes, because it has to outlive a single engine call. That is why an engine
that brings its own tools (codex) cannot host it -- it can still run the local
loops, via `candidate_analysis_engine`.
"""

from __future__ import annotations

from typing import Any

from sheldon.core.types import Usage
from sheldon.engines.base import tool_definitions
from sheldon.engines.toolloop.engine import FINISH, _executor
from sheldon.methods.sheldon import candidate_analysis, stages
from sheldon.methods.sheldon.prompts import render
from sheldon.methods.sheldon.run import Asked, Run, Step, staged
from sheldon.methods.sheldon.schemas import (
    SUBMIT_EVIDENCE,
    SUBMIT_EXPLORATION,
    SUBMIT_REVISION,
    SUBMIT_SELECTION,
    EvidenceOut,
    ReviseOut,
    SelectOut,
)
from sheldon.methods.sheldon.state import PENDING, TESTED
from sheldon.methods.stage import from_call, isolated, own_retrievals

#: Every submit tool a session turn can end with, by name. Under
#: `session_all_tools` each turn offers all of them, so the tool list -- part
#: of the prompt prefix the provider caches -- is the same on every turn. With
#: one submit tool per stage the list changed at every stage switch and the
#: whole conversation was billed uncached.
SESSION_SUBMIT_TOOLS = sorted(
    [SUBMIT_EXPLORATION, SUBMIT_SELECTION, SUBMIT_EVIDENCE, SUBMIT_REVISION], key=lambda t: t["name"]
)


class Session:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.messages: list[dict[str, Any]] = []
        self.system = render("session", round=run.round, first=run.round == 1).system
        self.turns = 0
        #: what the conversation holds of each item, as last rendered into it
        self.shown_h: dict[str, str] = {}
        self.shown_o: dict[str, str] = {}

    def seen_all(self) -> None:
        """After a turn that rendered the whole state."""
        st = self.run.state
        self.shown_h = {h.id: st.render_hypothesis(h) for h in st.hypotheses.values()}
        self.shown_o = {o.id: st.render_observation(o) for o in st.observations.values()}

    def delta(self) -> dict[str, str]:
        """The hypotheses and observations that are new or changed since the
        conversation last saw them, rendered, and marked as seen."""
        st = self.run.state
        out = {"hypotheses": [], "observations": []}
        for h in st.hypotheses.values():
            text = st.render_hypothesis(h)
            if self.shown_h.get(h.id) != text:
                out["hypotheses"].append(text)
                self.shown_h[h.id] = text
        for o in st.observations.values():
            text = st.render_observation(o)
            if self.shown_o.get(o.id) != text:
                out["observations"].append(text)
                self.shown_o[o.id] = text
        return {k: "\n".join(v) for k, v in out.items()}

    # -- the later turns: instructions plus the delta ------------------------------

    def select_step(self) -> Step:
        run = self.run
        ctx = dict(k=int(run.param("select_k", 5)), candidates=[h.id for h in run.state.of(PENDING)],
                   kept=[h.id for h in run.state.of(TESTED)], **self.delta())
        return Step("session_select", ctx, SelectOut, SUBMIT_SELECTION, "select",
                    steps=int(run.param("select_probe_calls", 2)))

    def comparative_analysis_step(self) -> Step:
        run = self.run
        ctx = dict(tested=[h.id for h in run.state.of(TESTED)], **self.delta())
        return Step("session_comparative_analysis", ctx, EvidenceOut, SUBMIT_EVIDENCE, "comparative_analysis",
                    steps=int(run.param("comparative_analysis_probe_calls", 10)))

    def exploit_step(self) -> Step:
        run = self.run
        tested = [h.id for h in run.state.of(TESTED)]
        ctx = dict(tested=tested, executable=run.executable(),
                   evidence=run.state.render_evidence(tested), **self.delta())
        return Step("session_exploit", ctx, ReviseOut, SUBMIT_REVISION, "exploit",
                    steps=int(run.param("exploit_probe_calls", 4)))

    async def turn(self, step: Step) -> Asked:
        run, ctx = self.run, self.run.ctx
        if not ctx.budget.ok():
            ctx.trace.log("budget.stop", where=step.tag, reason=ctx.budget.spend_reason())
            return Asked(None, Usage())
        self.turns += 1
        prompt = render(step.template, **dict(step.context, tools=True))
        text = f"{prompt.system}\n\n{prompt.user}" if prompt.system else prompt.user
        sub = isolated(ctx)
        start = len(run.record())
        steps = int(run.param(f"{step.tag}_probe_calls", step.steps))
        submit, execute = [step.tool], _executor(sub, steps)
        if run.param("session_all_tools", False):
            submit = SESSION_SUBMIT_TOOLS
            others = {t["name"] for t in SESSION_SUBMIT_TOOLS} - {step.tool["name"]}
            base_execute = execute

            async def execute(name: str, arguments: dict[str, Any]) -> str:
                if name in others:
                    return f"This turn is {step.tag}; submit it with {step.tool['name']}, not {name}."
                return await base_execute(name, arguments)

        convo = await ctx.llm.converse(
            text, tools=tool_definitions(sub, submit), execute=execute,
            system=self.system, max_iterations=steps + 1, finish=FINISH,
            stop_on={step.tool["name"]}, should_continue=lambda _i: ctx.budget.ok(),
            on_usage=ctx.budget.charge, history=self.messages, tag=f"session.{step.tag}",
        )
        self.messages = convo.messages
        usage = convo.usage
        out = from_call(convo, step.tool["name"], step.schema)
        if out is None and ctx.budget.ok():
            # The turn ended without its submit call: ask for the structure,
            # with the stage's prompt and what it wrote, outside the transcript.
            notes = (convo.text or "").strip()
            res = await ctx.llm.complete(
                f"{text}\n\n---\n\nYour notes so far:\n{notes}" if notes else text,
                system=self.system, schema=step.schema, tag=f"session.{step.tag}",
            )
            ctx.budget.charge(res.usage)
            usage += res.usage
            out = res.parsed
            ctx.trace.log("session.recovered", tag=step.tag, stopped_by=convo.stopped_by)
        ctx.trace.log("session.turn", round=run.round, stage=step.tag, turn=self.turns,
                      messages=len(self.messages), stopped_by=convo.stopped_by)
        return Asked(out, usage, own_retrievals(sub, run.record(), start), convo.stopped_by)


async def round_session(run: Run) -> tuple[Usage, bool]:
    engine = run.ctx.get_engine()
    if getattr(engine, "brings_own_tools", False) or not getattr(engine, "uses_tools", True):
        raise ValueError(
            f"sheldon mode=session needs a native tool-loop engine, not {engine.name!r}; "
            "use candidate_analysis_engine to put the local loops on it instead"
        )
    session = Session(run)
    usage = Usage()
    # A full map is settled before explore proposes anything: one call with
    # the model's whole attention on which regions to fold, outside the
    # session's conversation in both modes, so the merge is one decision.
    for spaces in ([stages.merge_needed(run)] if stages.merge_needed(run) else []):
        asked = await staged(run, stages.merge_step(run, spaces))
        usage += asked.usage
        stages.apply_merge(run, asked.parsed, spaces)
    step = stages.explore_step(run)
    session.seen_all()  # the explore prompt carries the whole state
    asked = await session.turn(step)
    usage += asked.usage
    stages.apply_explore(run, asked.parsed, asked.retrievals)
    if stages.select_needed(run):
        asked = await session.turn(session.select_step())
        usage += asked.usage
        chosen = stages.apply_select(run, asked.parsed)
    else:
        chosen = [h.id for h in run.state.of(PENDING)]
    chosen = chosen or stages.resumed_targets(run)
    if not chosen:
        return usage, False
    usage += await candidate_analysis.run_all(run, chosen)
    asked = await session.turn(session.comparative_analysis_step())
    usage += asked.usage
    stages.apply_comparative_analysis(run, asked.parsed, asked.retrievals)
    asked = await session.turn(session.exploit_step())
    usage += asked.usage
    stages.apply_exploit(run, asked.parsed)
    return usage, True
