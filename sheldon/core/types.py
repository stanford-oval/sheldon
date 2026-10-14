"""The narrow interface every benchmark and every method meets.

Keeping this file small is the whole point: a new benchmark should only have to
produce `Task`s and consume `Prediction`s, and a new method should only have to
turn a `Task` into a `Prediction`. Nothing else in the repo should need to know
which benchmark or which method is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sheldon.benchmarks.base import Benchmark
    from sheldon.core.budget import Budget
    from sheldon.core.llm import LLM
    from sheldon.core.trace import Trace


@dataclass(frozen=True)
class Task:
    """One evaluable unit, benchmark-agnostic.

    `question` is what a method can put in front of a model without knowing the
    benchmark. `payload` carries the raw benchmark fields for methods that want
    to be benchmark-aware (worked examples, candidate URLs, source documents).
    `answer` is the gold label and is *not* passed to methods at run time — the
    runner strips it before handing the task over.
    """

    id: str
    benchmark: str
    question: str
    payload: dict[str, Any] = field(default_factory=dict)
    answer: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def blind(self) -> Task:
        """A copy with the gold answer removed, safe to hand to a method."""
        return Task(
            id=self.id,
            benchmark=self.benchmark,
            question=self.question,
            payload=self.payload,
            answer=None,
            meta=self.meta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "benchmark": self.benchmark,
            "question": self.question,
            "payload": self.payload,
            "answer": self.answer,
            "meta": self.meta,
        }


@dataclass
class Usage:
    """Token and cost accounting, summed across every call a method makes."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def __iadd__(self, other: Usage) -> Usage:
        return self + other

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class Prediction:
    """What a method returns. `attempts` is ordered best-first.

    Benchmarks that allow k guesses score `attempts[:k]`;
    single-answer benchmarks score `answer`, which defaults to `attempts[0]`.
    """

    task_id: str
    answer: Any = None
    attempts: list[Any] = field(default_factory=list)
    belief: dict[str, Any] | None = None
    rounds: int = 0
    usage: Usage = field(default_factory=Usage)
    wall_s: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer is None and self.attempts:
            self.answer = self.attempts[0]
        elif self.answer is not None and not self.attempts:
            self.attempts = [self.answer]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "attempts": self.attempts,
            "belief": self.belief,
            "rounds": self.rounds,
            "usage": self.usage.to_dict(),
            "wall_s": round(self.wall_s, 3),
            "error": self.error,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prediction:
        u = d.get("usage") or {}
        return cls(
            task_id=d["task_id"],
            answer=d.get("answer"),
            attempts=d.get("attempts") or [],
            belief=d.get("belief"),
            rounds=d.get("rounds", 0),
            usage=Usage(**u) if u else Usage(),
            wall_s=d.get("wall_s", 0.0),
            error=d.get("error"),
            extra=d.get("extra") or {},
        )


@dataclass
class Score:
    """A benchmark's verdict on one prediction.

    `score` is the single number that goes in the headline table. Anything else
    worth tracking (partial credit, judge rationale, per-cell accuracy) goes in
    `metrics` / `detail` so it survives into the run artifacts.
    """

    task_id: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "metrics": self.metrics,
            "detail": self.detail,
        }


@dataclass
class Falsification:
    """The result of trying to break a candidate answer.

    This is the return type of `Verifier.check`. `refuted=True` means a concrete
    failure case was found; `counterexamples` should carry enough for the method
    to actually revise (the failing input, what was expected, what was produced).
    """

    refuted: bool
    counterexamples: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    checked: int = 0
    usage: Usage = field(default_factory=Usage)
    #: how far this refutation may move credence, already capped by whichever
    #: probe produced it. None means "decide from the counterexample kinds".
    strength: float | None = None
    #: names of the probes that were actually invoked
    probes_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "refuted": self.refuted,
            "counterexamples": self.counterexamples,
            "detail": self.detail,
            "checked": self.checked,
            "strength": self.strength,
            "probes_used": self.probes_used,
        }


@runtime_checkable
class Verifier(Protocol):
    """A label-free way to attack a candidate answer.

    A verifier must never read `Task.answer` — it exists so a method can hunt
    for failure cases at eval time without leaking the gold label: running a
    candidate program against the worked examples, say, or re-checking a claim
    against retrieved evidence.

    Benchmarks that have no cheap label-free check simply return `None` from
    `Benchmark.verifier()`, and the method falls back to an LLM critic.
    """

    async def check(self, candidate: Any) -> Falsification: ...


@dataclass
class RunContext:
    """Everything a method is allowed to touch while solving one task.

    `tools` is the same probe set for every method on a given benchmark. That
    equality is the point: if one method can execute code and another cannot,
    the comparison measures tool access, not method quality.
    """

    llm: LLM
    budget: Budget
    trace: Trace
    benchmark: Benchmark
    tools: Any = None  # ProbeSet; typed loosely to avoid a core import cycle
    engine: Any = None  # Engine; how one "ask the model, let it act" step runs
    #: `Sandbox` or None: what an agent's commands may see, from
    #: `Benchmark.sandbox`. On the context, not on a probe, because an engine
    #: that brings its own shell is offered no probes and must still honour it.
    sandbox: Any = None
    #: Where this task's per-round checkpoints go, or None to write none. A
    #: method that can be interrupted mid-task saves its state here after each
    #: round and picks it up from here, which `runs/` resume cannot do: that
    #: works per task and re-pays a whole task to reach a state the trace
    #: already shows.
    checkpoint_dir: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    #: Whatever a method derived once about this task, for its own later stages.
    #: Per-task because the context is; a method instance is shared across tasks
    #: running concurrently and cannot hold anything task-specific itself.
    scratch: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def get_engine(self) -> Any:
        """The engine for this run, falling back to the default.

        Resolved here rather than at each call site so a context built by hand
        — in a test, or a one-off script — still runs.
        """
        if self.engine is None:
            from sheldon.engines import get as get_engine  # noqa: PLC0415

            self.engine = get_engine()
        return self.engine
