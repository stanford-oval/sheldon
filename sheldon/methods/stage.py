"""How a stage talks to the model: its prompt, its tool, and the ask itself.

A method's prompts are Jinja templates with two sections, `[system]` and
`[user]`, split in the template *source* before any variable is substituted --
a task statement that happens to contain a line reading `[system]` must not be
able to move the rest of the prompt into the system turn. Editing a template
is automatically a cache miss (the request is the cache key), so no cache
needs clearing after a prompt edit.

A stage that can search has to hand its answer back through a `submit_*` tool
call, because the engine owns the tool loop; `submit_tool` generates that tool
from the stage's pydantic schema so the two cannot drift. `ask` is the one path
every stage takes: offer the submit tool, run the engine, and recover the
answer as structured output if the model spent its steps before calling one.
`will_search` is the one predicate deciding whether a stage gets tools at all,
shared by `ask` and by the templates, so a prompt never promises tools the
engine is not going to offer, and `stage_engine` is the one place that says
which engine a stage runs on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel

from sheldon.core.probes import Retrieval
from sheldon.core.types import RunContext, Usage

_SECTIONS = ("system", "user")


@dataclass
class Prompt:
    system: str
    user: str


def split_source(text: str) -> dict[str, str]:
    """Split template source at lines that are exactly `[system]` or `[user]`.

    Text before the first marker (a template comment, say) is ignored. Done on
    the source, never on rendered text: a substituted value is data and must
    not be able to open a section.
    """
    parts: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        marker = line.strip().lower()
        if marker.startswith("[") and marker.endswith("]") and marker[1:-1] in _SECTIONS:
            current = marker[1:-1]
            parts.setdefault(current, [])
            continue
        if current is not None:
            parts[current].append(line)
    if "user" not in parts:
        raise ValueError("a prompt needs a [user] section")
    return {key: "\n".join(lines) for key, lines in parts.items()}


def submit_tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """A `submit_*` tool whose arguments are exactly `model`.

    Generated from the schema rather than written by hand: a second copy of
    the same shape is the one that goes stale.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {"name": name, "description": description, "input_schema": schema}


def from_call(result: Any, name: str, schema: type[BaseModel]) -> Any | None:
    """Pull a stage's answer out of its `submit_*` call, if it made one."""
    for call in reversed(result.tool_calls or []):
        if call.name == name and isinstance(call.input, dict):
            try:
                return schema.model_validate(call.input)
            except Exception:  # noqa: BLE001 - a malformed payload is a miss, not a crash
                return None
    return None


#: `param(key, default)` -- how a stage reads the method's knobs. Threaded
#: through rather than read off `ctx.params`, so a method built directly in a
#: test with `Sheldon(select_k=1)` sees its own parameters.
ParamFn = Callable[[str, Any], Any]

def stage_engine(ctx: RunContext, param: ParamFn, stage: str) -> Any:
    """The engine one stage runs on -- the run's, unless it has been moved.

    A stage is one "ask the model, let it act" step, so it takes the run's
    engine by default and nothing in the loop has to mention engines at all.
    `<stage>_engine` moves a single stage onto another one, with
    `<stage>_engine_params` for that engine's own knobs: `implement_engine:
    codex` builds the artifact with an external coding agent while the rest of
    the loop stays on the run's engine and the run's model. It is the fourth
    axis applied one stage at a time, not a new mechanism -- the stage is still
    offered `ctx.tools` and nothing else, and an engine that brings its own
    tools brings them here too, which is most of the reason to move a stage.
    Tag the arm: a run whose builder is codex is not comparable to one whose
    builder is the run's model, for the same reason a codex run never is.

    Built once per task and kept in `ctx.scratch` -- which `isolated` shares,
    since `replace` copies the reference -- so every attempt on every hypothesis
    is the same agent. Keyed on the name as well as the stage: the params are
    fixed for a run, but a key that is only the stage hands back whatever was
    built there first, which is an engine nobody asked for. `runner` has already
    refused the run if the engine is not installed, rather than letting a
    missing binary arrive as an empty stage.
    """
    name = param(f"{stage}_engine", None)
    if not name:
        return ctx.get_engine()
    key = f"engine.{stage}.{name}"
    engine = ctx.scratch.get(key)
    if engine is None:
        from sheldon.engines import get as get_engine  # noqa: PLC0415

        # Same rule as the run's own engine: the run's effort is the stage
        # engine's `reasoning` default, and `<stage>_engine_params.reasoning`
        # still overrides. A stage moved onto codex would otherwise take that
        # engine's hard-coded `low` while the rest of the loop ran at the run's
        # effort -- the one call in the loop doing the building, quietly the
        # cheapest one.
        stage_params = {"reasoning": ctx.llm.effort,
                        **(param(f"{stage}_engine_params", None) or {})}
        engine = get_engine(str(name), **stage_params)
        ctx.scratch[key] = engine
    return engine


def will_search(
    ctx: RunContext, param: ParamFn, stage: str, allow_search: bool | None = None
) -> bool:
    """Whether a stage is going to be offered tools.

    Three things have to hold: the params allow it (`<stage>_probes`, falling
    back to `stage_probes`, unless the caller decides outright), the engine
    uses tools at all (`simple` does not), and there is something to offer --
    the benchmark's probes, or an engine that brings its own (`codex` is
    offered nothing and can still search). The engine asked is this stage's,
    so a stage moved onto `codex` is told it can search even where the run's
    engine would have had nothing to offer it.
    """
    may = (
        bool(param(f"{stage}_probes", param("stage_probes", True)))
        if allow_search is None
        else bool(allow_search)
    )
    engine = stage_engine(ctx, param, stage)
    if not getattr(engine, "uses_tools", True):
        return False
    own = bool(getattr(engine, "brings_own_tools", False))
    return may and (bool(ctx.tools) or own)


def isolated(ctx: RunContext, exclude: set[str] | None = None) -> RunContext:
    """A context whose tool set is a `ProbeSet.view()` of the shared one.

    `exclude` withholds named probes from this stage alone -- the objective
    probe from `implement` under `implement_evaluator: false`.

    Concurrent stages share the run's record, so `record[start:]` is not a
    stage's own work when another stage was retrieving at the same time. A view
    has its own `history`, which is exactly this stage's dispatches; the
    probes, the record and the retrievals directory are shared, so the stage is
    offered the same toolbox and its retrievals still outlive it.
    """
    if ctx.tools is None:
        return ctx
    return replace(ctx, tools=ctx.tools.view(exclude=exclude))


def own_retrievals(view_ctx: RunContext, record: list[Retrieval], start: int) -> list[Retrieval]:
    """The retrievals a stage made itself, out of everything appended to the
    shared record since it began. Matched by result identity, which is what
    `dispatch` puts in both the view's history and the record."""
    if view_ctx.tools is None:
        return []
    mine = {id(res) for _probe, res in view_ctx.tools.history}
    return [r for r in record[start:] if id(r.result) in mine]


@dataclass
class Asked:
    """What one stage call produced."""

    parsed: Any
    usage: Usage
    #: True only when the whole thing was a single cached completion; the
    #: engine path does not report per-turn cache hits.
    cached: bool
    #: whether the stage was run with tools
    searched: bool
    #: how the engine or completion ended -- `budget` when it never ran
    stopped_by: str | None = None


async def ask(
    ctx: RunContext,
    *,
    param: ParamFn,
    schema: type[BaseModel],
    tool: dict[str, Any],
    tag: str,
    prompt: Prompt,
    default_steps: int = 6,
    required: bool = False,
    allow_search: bool | None = None,
) -> Asked:
    """Run one stage, letting it search first if there is anything to search.

    Every stage that decides something is allowed to look before it decides:
    the agent that searched should be the agent that answers. The engine owns
    the tool loop, so the stage hands its answer back through `tool`; if it
    spends its steps before calling it, the answer is recovered as structured
    output from the prompt plus whatever it wrote, rather than lost.

    `required` marks a stage that must produce something whatever the budget
    says. Routing a stage through the engine puts it behind the budget gate,
    and for the stage that states the answer that would turn a spent budget
    into a blank submission. `max_calls` and `max_usd` belong to the harness,
    not the model, so the closing call is made either way. A stage that is not
    required makes no call at all once the budget is spent.
    """
    stage = tag.split(".")[0]
    searching = will_search(ctx, param, stage, allow_search)
    if required and not ctx.budget.ok():
        searching = False
    # `<stage>_probe_calls` is the ceiling the engine enforces. A sub-stage can
    # carry its own: `<stage>_<sub>_probe_calls` for a call tagged
    # `<stage>.<sub>...`, falling back to the stage's.
    sub_stage = "_".join(tag.split(".")[:2])
    allowed = int(param(f"{sub_stage}_probe_calls", param(f"{stage}_probe_calls", default_steps)))
    bare = prompt

    if not searching:
        if not required and not ctx.budget.ok():
            ctx.trace.log("budget.stop", where=tag, reason=ctx.budget.spend_reason())
            return Asked(None, Usage(), False, False, "budget")
        res = await ctx.llm.complete(bare.user, system=bare.system, schema=schema, tag=tag)
        ctx.budget.charge(res.usage)
        return Asked(res.parsed, res.usage, res.cached, False, res.stop_reason)

    result = await stage_engine(ctx, param, stage).run(
        ctx,
        prompt.user,
        system=prompt.system,
        extra_tools=[tool],
        stop_on={tool["name"]},
        max_steps=allowed,
        tag=tag,
    )
    usage = result.usage
    out = from_call(result, tool["name"], schema)
    if out is None and (required or ctx.budget.ok()):
        # The stage did not call its tool. Ask for the structure it should have
        # submitted, with the original prompt and its own notes in front of it
        # -- handing it only the notes would ask it to answer without the task.
        notes = (result.text or "").strip()
        recovery = bare.user
        if notes:
            recovery = f"{bare.user}\n\n---\n\nYour notes so far:\n{notes}"
        res = await ctx.llm.complete(recovery, system=bare.system, schema=schema, tag=tag)
        ctx.budget.charge(res.usage)
        usage += res.usage
        out = res.parsed
        ctx.trace.log(f"{stage}.recovered", stopped_by=result.stopped_by, tag=tag)
    return Asked(out, usage, False, True, result.stopped_by)
