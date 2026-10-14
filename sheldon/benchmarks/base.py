"""What a benchmark has to provide.

Five things, and only the first two are mandatory:

  load()         produce Tasks
  score()        judge a Prediction
  parse_answer() turn raw model text into this benchmark's answer type
  verifier()     a label-free way to attack a candidate  <- the falsifier's hook
  check()        say what's missing when the data isn't there

A benchmark must never leak `Task.answer` into anything a method can reach.
The runner blinds tasks before handing them over; `verifier()` is held to the
same rule by convention, and `tests/test_no_label_leak.py` checks it.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from sheldon.core.io import data_root
from sheldon.core.isolation import Sandbox
from sheldon.core.probes import Probe, PythonProbe, VerifierProbe
from sheldon.core.types import Prediction, RunContext, Score, Task, Verifier


class Benchmark(ABC):
    name: ClassVar[str] = ""
    splits: ClassVar[tuple[str, ...]] = ("train", "eval")
    #: how many guesses count as correct
    n_attempts: ClassVar[int] = 1
    #: what kind of thing an answer is, for a method that needs to branch on it
    answer_kind: ClassVar[str] = "text"
    #: One line telling the model what an answer looks like here. It belongs to
    #: the benchmark because it is a fact about the benchmark: a method that
    #: switches on `answer_kind` to produce it has to know every benchmark that
    #: exists, and will quietly hand the wrong hint to the ones it does not.
    answer_format: ClassVar[str] = (
        "Give the final answer on its own line, as briefly as the question allows."
    )
    #: True when a candidate can be written as a program and run -- `realize`
    #: turns one into an answer. The loop asks for `candidate_program` only where
    #: something can execute it.
    executable_candidates: ClassVar[bool] = False
    #: The contract such a program must satisfy, in one sentence, for the prompt
    #: that asks for it. One benchmark's entry point is not another's -- and a
    #: candidate need not be a program at all -- so the method layer cannot
    #: state it without knowing which benchmark it is on.
    candidate_contract: ClassVar[str] = (
        "runnable as written, with no placeholders, in whatever form this "
        "benchmark executes"
    )
    #: The probe that measures the objective on an optimization benchmark, or
    #: empty. Named by the benchmark so a method can find or withhold it
    #: without knowing any benchmark's tool names.
    objective_probe: ClassVar[str] = ""
    #: The key under which that probe's evidence carries the objective's value,
    #: so a method can read a measurement off a retrieval without naming the
    #: benchmark. Empty where there is no single number.
    objective_key: ClassVar[str] = ""
    #: True when this benchmark's probes are the *subject*, not a convenience.
    #:
    #: An engine that brings its own shell is offered no probes by default,
    #: because handing codex a second `run_shell` measures nothing. That reasoning
    #: fails wherever the probes are the only way to reach the thing being
    #: measured: an oracle that is the objective, or a tool that is the only
    #: access to the task's material. There the default hands the engine an empty
    #: toolbox and the run still finishes, reporting a number for a method that
    #: could not see the task.
    #:
    #: `-p offer_tools=false` still overrides this, which is the control arm that
    #: measures codex with nothing but its own tools.
    probes_essential: ClassVar[bool] = False
    #: one-line description shown by `sheldon ls`
    blurb: ClassVar[str] = ""

    def __init__(self, root: str | Path | None = None, **opts: Any) -> None:
        self.root = Path(root) if root else data_root() / self.name
        self.opts = opts

    # -- required ---------------------------------------------------------

    @abstractmethod
    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        """Return tasks for `split`, deterministically subsampled to `limit`."""

    @abstractmethod
    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        """Judge one prediction. May call `ctx.llm` for judge-based scoring."""

    # -- optional ---------------------------------------------------------

    def parse_answer(self, text: str) -> Any:
        """Turn a model's raw text into this benchmark's answer type."""
        return text.strip()

    def verifier(self, task: Task) -> Verifier | None:
        """A label-free check the falsifier can run. None if there isn't one."""
        return None

    def python_context(self, task: Task) -> dict[str, Any]:
        """Public task data to preload into the sandbox as variables.

        Anything already visible in `task.question` is fair game — putting it
        here just saves the model from retyping it. Never the gold answer.
        """
        return {}

    def situation(self, task: Task) -> str:
        """The problem, as a stage that is not writing the answer needs it.

        `Task.question` is what the *answering* agent gets, and on some
        benchmarks that includes a full specification of the artifact: section
        headings, citation formats, an output template, a path to write to.

        A stage whose job is to investigate should not be handed all of that,
        or it starts writing the deliverable instead.

        Defaults to the whole question, so a benchmark that has no such split
        behaves exactly as before and a new one needs to do nothing.
        """
        return task.question

    def deliverable(self, task: Task) -> str:
        """The full specification of the final artifact, for the stage that
        writes it. Defaults to the one-line `answer_format`, which is all most
        benchmarks have to say."""
        return self.answer_format

    def probes(self, task: Task) -> list[Probe]:
        """The tools every method gets on this task.

        Defaults to a Python sandbox, preloaded with `python_context`. Override
        to add domain tools — SQL over the task's table, search over its source
        documents, the web — and call `super().probes(task)` to keep the
        sandbox in the set.

        Deliberately *not* here: `verifier()`. An oracle that checks a candidate
        for you is a benchmark-specific crutch, and handing it to one method and
        not another is the confound this whole layer exists to avoid. Benchmarks
        that want to offer it must opt in explicitly (see `oracle_probes`), and
        then it goes to every method equally.

        Probes are built from the task the method sees, and must never read
        `Task.answer`.
        """
        if not self.opts.get("sandbox", True):
            return []
        return [
            PythonProbe(
                timeout=float(self.opts.get("exec_timeout", 10.0)),
                context=self.python_context(task),
            )
        ]

    def workspace(self, task: Task, path: Path) -> None:
        """Seed the per-task working directory, or leave it empty.

        Every method gets a shell opening in `path`, so this is what the shell
        starts out able to see: on a benchmark whose task *is* a program, the
        starting program belongs here as a file. It is the cheap way to work on
        an artifact that would otherwise cross the tool boundary as a string --
        a 17,550-character seed costs ~4,400 output tokens every time it is
        passed whole, which is enough to make rewriting from scratch the
        cheaper move and is how a run comes to ignore what it was handed.

        Written fresh per task under the run directory, never into `data/`.
        """
        return None

    def sandbox(self, task: Task) -> Sandbox | None:
        """What an agent's commands may see on this task, or None for the host.

        Separate from `probes` on purpose. An engine that brings its own shell
        (`codex`) is offered no probes, so a restriction carried only by a probe
        would never reach it; the runner puts this declaration on the run context
        for every engine instead. A benchmark whose grading material sits on the
        same disk as the agent's workspace should return one. See
        `sheldon.core.isolation`.
        """
        return None

    def oracle_probes(self, task: Task) -> list[Probe]:
        """The verifier, as a decisive probe. Off unless `benchmark_opts.oracle`.

        Kept reachable so "how much is the oracle worth?" stays a measurable
        ablation rather than a baked-in assumption.
        """
        if not self.opts.get("oracle", False):
            return []
        verifier = self.verifier(task)
        return [VerifierProbe(verifier)] if verifier is not None else []

    def probe_state(self, task: Task) -> Any:
        """Whatever this task's probes hold that a resumed run must not lose.

        Probes ration what matters -- calls to an oracle or an evaluator --
        each count held per task in a closure the runner builds once. A run
        resumed mid-task without this would be handed
        a fresh allowance and its number would not compare with one that ran
        straight through. Return something JSON-serialisable, or None where a
        benchmark rations nothing.
        """
        return None

    def objective_exhausted(self, task: Task) -> bool:
        """True once the objective probe's per-task allowance is spent.

        A loop that keeps opening rounds after that pays for stages whose every
        measurement is refused. False wherever nothing is rationed.
        """
        return False

    def restore_probe_state(self, task: Task, state: Any) -> None:
        """Put back what `probe_state` returned.

        Concrete and empty on purpose: a benchmark that rations nothing has
        nothing to restore, and making this abstract would force every
        benchmark to write the same empty method.
        """
        return None

    def realize(self, task: Task, candidate: dict[str, Any]) -> Any:
        """Turn a candidate artifact into a concrete answer for this task.

        The companion to `verifier`: once a candidate rule survives being
        tested, this is what converts it into something scoreable -- running
        a winning program on the held-out input, say, which is both free and
        exact. Return None if the candidate can't be
        realized (or if the benchmark has no such notion).
        """
        return None

    def check(self) -> list[str]:
        """Problems that would stop a run. Empty list means good to go."""
        if not self.root.exists():
            return [f"data directory not found: {self.root}"]
        return []

    def describe(self) -> str:
        status = "ready" if not self.check() else "; ".join(self.check())
        return f"{self.name:<18} {self.blurb}  [{status}]"

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def subsample(tasks: list[Task], limit: int | None, seed: int = 0) -> list[Task]:
        """Deterministic subsample. Same seed and limit always give the same
        tasks, so two methods are compared on identical work."""
        if limit is None or limit >= len(tasks):
            return tasks
        rng = random.Random(seed)
        return sorted(rng.sample(tasks, limit), key=lambda t: t.id)
