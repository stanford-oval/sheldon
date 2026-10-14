"""Native tool-calling loop. The default.

Uses the provider's own tool protocol via `LLM.converse`, so the model emits
structured tool calls rather than text we have to parse. Every turn passes
through the disk cache and the per-task budget.
"""

from __future__ import annotations

from typing import Any

from sheldon.core.registry import engines
from sheldon.core.types import RunContext
from sheldon.engines.base import Engine, EngineResult, tool_definitions

# Said only when the tool budget runs out before the model has answered. It has
# to work for every caller — a baseline answering a task, the falsifier
# reporting a verdict — so it names no benchmark and no output format.
FINISH = (
    "You have used all the tool calls available for this step. Do not request "
    "another tool. Give your final response now, based on what you have already "
    "worked out. An answer you are unsure of is worth more than no answer."
)


@engines.register("toolloop", "tools")
class ToolLoopEngine(Engine):
    name = "toolloop"
    blurb = "native tool-calling loop (default)"

    async def run(
        self,
        ctx: RunContext,
        prompt: str,
        *,
        system: str | None = None,
        extra_tools: list[dict[str, Any]] | None = None,
        stop_on: set[str] | None = None,
        max_steps: int | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> EngineResult:
        # `engine_params.model` puts this engine on a model of its own, the way
        # `codex` takes one: `implement_engine: toolloop` with a smaller model is
        # then the same arm as codex under that model, with only the engine
        # changed. Unset, the caller's model (else the run's) is used as before.
        model = model or self.param("model")
        tools = tool_definitions(ctx, extra_tools)
        if not tools:
            res = await ctx.llm.complete(prompt, system=system, model=model, tag=tag)
            ctx.budget.charge(res.usage)
            return EngineResult(
                text=res.text, usage=res.usage, stopped_by=res.stop_reason, iterations=1
            )

        steps = max_steps if max_steps is not None else int(self.param("max_steps", 4))
        convo = await ctx.llm.converse(
            prompt,
            tools=tools,
            execute=_executor(ctx, steps),
            system=system,
            model=model,
            max_iterations=steps + 1,
            finish=self.param("finish_prompt", FINISH),
            stop_on=stop_on,
            should_continue=lambda _i: ctx.budget.ok(),
            on_usage=ctx.budget.charge,
            tag=tag,
        )
        ctx.trace.log(
            "engine.done", engine=self.name, tag=tag,
            stopped_by=convo.stopped_by, iterations=convo.iterations,
            tool_calls=len(convo.tool_calls), chars=len(convo.text or ""),
        )
        return EngineResult(
            text=convo.text,
            usage=convo.usage,
            tool_calls=convo.tool_calls,
            stopped_by=convo.stopped_by,
            iterations=convo.iterations,
        )


def _executor(ctx: RunContext, steps: int | None = None) -> Any:
    """Dispatch one tool call, and say how many are left.

    The caller's prompt states the allowance once, before the loop starts, and
    then cannot say anything again -- so a stage that has spent five of six
    calls learns that only by having counted. The engine knows, so it tells it,
    on the result of every call. Cheap, and it goes in the cache key the same
    way the rest of the turn does, which is correct: a run with a different
    allowance is a different run.
    """
    used = 0

    async def execute(name: str, arguments: dict[str, Any]) -> str:
        nonlocal used
        used += 1
        result = await ctx.tools.dispatch(name, arguments)
        ctx.trace.log("tool", probe=name, ok=result.ok, summary=result.summary)
        text = result.render()
        if steps:
            left = max(0, steps - used)
            note = (
                f"[{used} of {steps} tool calls used; {left} left]"
                if left
                else f"[{used} of {steps} tool calls used; this was the last one]"
            )
            text = f"{text}\n\n{note}"
        return text

    return execute
