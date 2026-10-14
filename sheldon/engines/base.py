"""Engines: *how* one reasoning step gets executed.

Three axes, kept apart on purpose:

    benchmark   what the task is, and which tools exist for it
    method      what strategy is applied (direct, best-of-n, the belief loop)
    engine      how a single "ask the model, let it use the tools" step runs

The engine is the swappable one here. A plain completion, a native tool-calling
loop, a text-protocol ReAct agent, and an external coding agent are all valid
ways to fulfil the same request, and which one you use should not require
touching a method or a benchmark.

The contract is narrow so it survives adding engines later:

    await engine.run(ctx, prompt, system=..., extra_tools=[...], stop_on={...})
        -> EngineResult(text, usage, tool_calls, ...)

Tools always come from `ctx.tools` — the benchmark's probe set, identical for
every method. An engine may decline to use them (`simple` does), but it may
never reach for anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from sheldon.core.llm import ToolCall
from sheldon.core.types import RunContext, Usage


@dataclass
class EngineResult:
    """What one engine step produced."""

    text: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stopped_by: str | None = None
    iterations: int = 0

    def calls_to(self, name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == name]


class Engine(ABC):
    name: ClassVar[str] = ""
    blurb: ClassVar[str] = ""
    #: whether this engine will use `ctx.tools` at all
    uses_tools: ClassVar[bool] = True
    #: True for an engine that arrives with a shell and a search of its own.
    #: The harness then offers it nothing generic: handing `codex` a `run_shell`
    #: beside the shell it already has measures nothing, and the probes this
    #: repo supplies are meant to give other engines the capability codex came
    #: with, not to give codex a second copy.
    brings_own_tools: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        self.params = params

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @abstractmethod
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
        """Answer `prompt`, optionally using tools.

        `extra_tools` are caller-supplied definitions that are not probes — the
        falsifier's `submit_verdict` is the only one today. `stop_on` names
        tools whose invocation ends the step.
        """

    def check(self) -> list[str]:
        """Problems that would stop this engine from running. Empty means ready."""
        return []

    def describe(self) -> str:
        problems = self.check()
        status = "ready" if not problems else problems[0]
        return f"{self.name:<12} {self.blurb}  [{status}]"


def tool_definitions(ctx: RunContext, extra_tools: list[dict[str, Any]] | None) -> list[dict]:
    """The probe set's definitions plus any caller-supplied ones."""
    base = ctx.tools.tool_definitions() if ctx.tools else []
    return [*base, *(extra_tools or [])]
