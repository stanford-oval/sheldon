"""One completion, no tools.

The floor. Useful as an ablation — "what does this method achieve if it cannot
act?" — and as the only engine that works against an endpoint with no
tool-calling support at all.
"""

from __future__ import annotations

from typing import Any

from sheldon.core.registry import engines
from sheldon.core.types import RunContext
from sheldon.engines.base import Engine, EngineResult


@engines.register("simple")
class SimpleEngine(Engine):
    name = "simple"
    blurb = "a single completion, tools ignored"
    uses_tools = False

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
        res = await ctx.llm.complete(prompt, system=system, model=model, tag=tag)
        ctx.budget.charge(res.usage)
        return EngineResult(
            text=res.text, usage=res.usage, stopped_by=res.stop_reason, iterations=1
        )
