"""What a method has to provide: `run(task, ctx) -> Prediction`.

Methods receive a *blinded* task (no gold answer). Anything they want to know
about the benchmark they get through `ctx.benchmark` — the answer format, the
number of attempts allowed, and the label-free verifier.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from sheldon.benchmarks.base import Benchmark
from sheldon.core.types import Prediction, RunContext, Task, Usage


class Method(ABC):
    name: ClassVar[str] = ""
    blurb: ClassVar[str] = ""
    #: Bump when a change to this method's *code* changes what it produces.
    #:
    #: `config_hash` is built from the config, so a behavioural default that
    #: lives in code — `self.param("revise", True)` — is invisible to it. Adding
    #: revision changed every sheldon result and moved the hash not at all, which
    #: means the same command would have resumed into the old directory and
    #: reported stale numbers as new. That is the failure this exists to stop.
    #:
    #: "1" means "as originally written" and is excluded from the hash, so
    #: bumping one method never invalidates the others.
    version: ClassVar[str] = "1"

    def __init__(self, **params: Any) -> None:
        self.params = params

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @abstractmethod
    async def run(self, task: Task, ctx: RunContext) -> Prediction: ...

    def describe(self) -> str:
        return f"{self.name:<14} {self.blurb}"


class timer:
    """`with timer() as t: ...` then `t.elapsed`."""

    def __enter__(self) -> timer:
        self._t0 = time.monotonic()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.monotonic() - self._t0


def answer_format_hint(ctx: RunContext) -> str:
    """What an answer looks like here, as the benchmark describes it.

    The benchmark states it, so the method layer holds no table of answer kinds
    and a new benchmark needs no edit here.
    """
    return getattr(ctx.benchmark, "answer_format", Benchmark.answer_format)


# -- shared tool plumbing -------------------------------------------------
#
# Every method reaches its tools the same way, through `ctx.tools`. Keeping this
# in one place is what keeps the comparison honest: a baseline that cannot run
# code is not a baseline for a method that can.

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


async def ask(
    ctx: RunContext,
    prompt: str,
    *,
    system: str | None = None,
    tag: str = "",
    max_tool_calls: int | None = None,
) -> tuple[str, Usage]:
    """Ask the model one question, letting it use this benchmark's tools.

    How that actually happens is the engine's business — a plain completion, a
    native tool loop, a ReAct transcript, an external agent. Methods should not
    care, which is the point of routing every ask through here.
    """
    result = await ctx.get_engine().run(
        ctx,
        prompt,
        system=system,
        max_steps=max_tool_calls,
        tag=tag,
    )
    return result.text, result.usage


def answer_from_text(ctx: RunContext, task: Task, text: str) -> Any:
    """Turn a model's final text into an answer.

    If the benchmark executes candidates and the model wrote one, run it — see
    `Benchmark.realize`. Transcribing a large answer by hand is a failure mode
    that has nothing to do with whether the rule was found.

    Every method gets this shortcut or none does: the benchmark's
    `answer_format` offers it to whoever is answering. The benchmark also
    decides what counts as a candidate -- `realize` returns None for one it
    cannot use, and this only runs where a benchmark says its candidates are
    executable.
    """
    if getattr(ctx.benchmark, "executable_candidates", False):
        for block in _CODE_BLOCK.findall(text or ""):
            try:
                value = ctx.benchmark.realize(task, {"program": block})
            except Exception:  # noqa: BLE001 - a block that will not run is not an answer
                value = None
            if value is not None:
                ctx.trace.log("answer.realized", source="code_block")
                return value
    return ctx.benchmark.parse_answer(text)
