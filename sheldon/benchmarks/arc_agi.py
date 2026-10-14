"""ARC-AGI-1 and ARC-AGI-2.

Fully wired: real loader, real scorer, and a real label-free verifier. ARC is
the benchmark where the falsifier has actual teeth — a candidate rule can be
written as a Python `transform(grid)` and executed against the demonstration
pairs, which finds concrete failure cases for free and without touching the
held-out answer.

Data layout (see data/README.md for fetch commands):

    data/arc-agi-1/training/*.json
    data/arc-agi-1/evaluation/*.json
    data/arc-agi-2/training/*.json
    data/arc-agi-2/evaluation/*.json

Each file: {"train": [{"input": grid, "output": grid}, ...], "test": [...]}
where a grid is a list of rows of ints 0-9.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sheldon.benchmarks.base import Benchmark
from sheldon.core import isolation
from sheldon.core.isolation import Sandbox
from sheldon.core.probes import Probe
from sheldon.core.registry import benchmarks
from sheldon.core.sandbox import run_transform
from sheldon.core.types import Falsification, Prediction, RunContext, Score, Task, Verifier

Grid = list[list[int]]

_SPLIT_DIRS = {"train": "training", "eval": "evaluation", "evaluation": "evaluation"}


def render_grid(grid: Grid) -> str:
    """Rows of digits, one per line. Compact and unambiguous for 0-9 values."""
    return "\n".join("".join(str(c) for c in row) for row in grid)


def parse_grid(text: str) -> Grid | None:
    """Pull a grid out of model text.

    Accepts a JSON array of arrays anywhere in the text, or a run of lines that
    are all digits and all the same length. Returns None if neither is found.
    """
    for match in re.finditer(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL):
        try:
            grid = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if _is_grid(grid):
            return grid

    best: Grid | None = None
    run: list[list[int]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and all(c.isdigit() for c in stripped):
            row = [int(c) for c in stripped]
            if run and len(row) != len(run[0]):
                if best is None or len(run) > len(best):
                    best = run
                run = []
            run.append(row)
        else:
            if run and (best is None or len(run) > len(best)):
                best = run
            run = []
    if run and (best is None or len(run) > len(best)):
        best = run
    return best


def _is_grid(obj: Any) -> bool:
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(r, list) and r for r in obj)
        and all(isinstance(c, int) and 0 <= c <= 9 for r in obj for c in r)
        and len({len(r) for r in obj}) == 1
    )


def grids_equal(a: Any, b: Any) -> bool:
    return _is_grid(a) and _is_grid(b) and a == b


class ArcVerifier:
    """Runs a candidate `transform` against the demonstration pairs.

    Candidate forms accepted:
      {"program": "<python defining transform(grid) -> grid>"}  -> executed
      anything else                                             -> not refutable

    The demonstration pairs are public input, so this leaks nothing. Execution
    follows `benchmark_opts.sandbox` — this runs model-written Python.
    """

    def __init__(self, task: Task, allow_code_exec: bool = False, timeout: float = 10.0) -> None:
        self.pairs = task.payload["train"]
        self.allow_code_exec = allow_code_exec
        self.timeout = timeout

    async def check(self, candidate: Any) -> Falsification:
        program = candidate.get("program") if isinstance(candidate, dict) else None
        if not program:
            return Falsification(
                refuted=False,
                detail="no executable candidate (expected {'program': ...})",
            )
        if not self.allow_code_exec:
            return Falsification(
                refuted=False,
                detail="code execution disabled; set params.allow_code_exec=true to enable",
            )

        inputs = [p["input"] for p in self.pairs]
        res = run_transform(program, inputs, entry="transform", timeout=self.timeout)
        if not res.get("ok"):
            return Falsification(
                refuted=True,
                counterexamples=[{"kind": "program_error", "error": res.get("error", "")}],
                detail=f"program failed to run: {res.get('error', '')}",
                checked=0,
            )

        bad: list[dict[str, Any]] = []
        for r in res["results"]:
            i = r["i"]
            expected = self.pairs[i]["output"]
            if not r["ok"]:
                bad.append(
                    {
                        "kind": "raised",
                        "pair": i,
                        "input": render_grid(inputs[i]),
                        "expected": render_grid(expected),
                        "error": r.get("error", ""),
                    }
                )
            elif not grids_equal(r["value"], expected):
                got = r["value"]
                bad.append(
                    {
                        "kind": "mismatch",
                        "pair": i,
                        "input": render_grid(inputs[i]),
                        "expected": render_grid(expected),
                        "got": render_grid(got) if _is_grid(got) else repr(got)[:400],
                    }
                )

        n = len(self.pairs)
        return Falsification(
            refuted=bool(bad),
            counterexamples=bad,
            detail=f"{n - len(bad)}/{n} demonstration pairs reproduced",
            checked=n,
        )


class _ArcBase(Benchmark):
    splits = ("train", "eval")
    n_attempts = 2
    answer_kind = "grid"
    answer_format = (
        "Give the output grid as rows of digits, one row per line. Or give a "
        "```python``` block defining `transform(grid) -> grid` and it will be run "
        "on the test input for you -- either is accepted, and the program avoids "
        "transcribing the grid by hand."
    )
    executable_candidates = True
    candidate_contract = (
        "Python source defining `transform(grid) -> grid`, executed against every "
        "demonstration pair, so it must run as written with no placeholders"
    )

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        d = self.root / _SPLIT_DIRS.get(split, split)
        if not d.exists():
            raise FileNotFoundError(f"{self.name}: no such split directory: {d}")

        tasks: list[Task] = []
        for path in sorted(d.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            demos = raw["train"]
            for i, test in enumerate(raw["test"]):
                tid = path.stem if len(raw["test"]) == 1 else f"{path.stem}#{i}"
                tasks.append(
                    Task(
                        id=tid,
                        benchmark=self.name,
                        question=self._render(demos, test["input"]),
                        payload={"train": demos, "test_input": test["input"]},
                        answer=test.get("output"),
                        meta={"file": path.name, "test_index": i, "n_demos": len(demos)},
                    )
                )
        return self.subsample(tasks, limit, seed)

    @staticmethod
    def _render(demos: list[dict[str, Grid]], test_input: Grid) -> str:
        parts = [
            "You are given input/output grid pairs that all follow one hidden rule.",
            "Grids are rows of digits 0-9, one row per line.",
            "",
        ]
        for i, pair in enumerate(demos, 1):
            parts += [
                f"## Example {i}",
                "Input:",
                render_grid(pair["input"]),
                "Output:",
                render_grid(pair["output"]),
                "",
            ]
        parts += ["## Test", "Input:", render_grid(test_input), ""]
        return "\n".join(parts)

    def parse_answer(self, text: str) -> Any:
        return parse_grid(text)

    def verifier(self, task: Task) -> Verifier | None:
        return ArcVerifier(
            task,
            allow_code_exec=bool(self.opts.get("sandbox", True)),
            timeout=float(self.opts.get("exec_timeout", 10.0)),
        )

    def python_context(self, task: Task) -> dict[str, Any]:
        """Preload the public grids. Both are already in `task.question`; this
        just spares the model from retyping them into every snippet."""
        return {"demos": task.payload["train"], "test_input": task.payload["test_input"]}

    def probes(self, task: Task) -> list[Probe]:
        """The sandbox, and the demonstration-pair oracle only if asked for.

        Withholding the oracle by default is the whole point: a method should
        have to work out for itself that the way to test a rule is to run it
        against the worked examples. Handing it a `check_against_examples`
        button makes the benchmark measure who was given the button.
        """
        return [*super().probes(task), *self.oracle_probes(task)]

    def sandbox(self, task: Task) -> Sandbox | None:
        """What an agent with a shell may see, under `benchmark_opts.isolate`.

        The whole task is in the prompt, so the answer is nothing from disk --
        but the disk holds it anyway: `data/arc-agi-2/evaluation/<id>.json`
        carries the test output in plain JSON, and so does any other copy of
        the dataset on the host. So the sandbox mounts nothing of the repository at all;
        the engine adds its own working directory. `run_python` is unaffected
        (it is not a shell); this exists for codex, which brings one. Asking
        for isolation without bubblewrap is an error, never a fallback.
        """
        if not self.opts.get("isolate"):
            return None
        if not isolation.available():
            raise RuntimeError(
                "benchmark_opts.isolate is set but bubblewrap (`bwrap`) is not installed; "
                "refusing to give the agent the host filesystem"
            )
        return Sandbox.of(read_only=[], writable=[], env={})

    def realize(self, task: Task, candidate: dict[str, Any]) -> Any:
        """Run a candidate program on the held-out test input.

        This is why writing the hypothesis as code pays for itself: a rule that
        reproduces every demonstration pair yields the test answer for free,
        with no further model call and no chance of a transcription slip.
        """
        program = candidate.get("program") if isinstance(candidate, dict) else None
        if not program or not self.opts.get("sandbox", True):
            return None
        res = run_transform(
            program,
            [task.payload["test_input"]],
            entry="transform",
            timeout=float(self.opts.get("exec_timeout", 10.0)),
        )
        if not res.get("ok") or not res.get("results"):
            return None
        first = res["results"][0]
        return first["value"] if first.get("ok") and _is_grid(first.get("value")) else None

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        gold = task.answer
        attempts = (pred.attempts or [pred.answer])[: self.n_attempts]
        hits = [grids_equal(a, gold) for a in attempts]
        correct = float(any(hits))
        return Score(
            task_id=task.id,
            score=correct,
            metrics={
                "pass@1": float(bool(hits and hits[0])),
                f"pass@{self.n_attempts}": correct,
                "parsed": float(any(_is_grid(a) for a in attempts)),
                "n_attempts": float(len(attempts)),
            },
            detail={
                "gold_shape": [len(gold), len(gold[0])] if _is_grid(gold) else None,
                "pred_shape": (
                    [len(attempts[0]), len(attempts[0][0])]
                    if attempts and _is_grid(attempts[0])
                    else None
                ),
            },
        )

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems + [f"see data/README.md for how to populate {self.name}"]
        if not any((self.root / _SPLIT_DIRS[s]).exists() for s in self.splits):
            expected = sorted({_SPLIT_DIRS[s] for s in self.splits})
            return [f"no split directories under {self.root} (expected one of {expected})"]
        return []


@benchmarks.register("arc-agi-1", "arc1")
class ArcAgi1(_ArcBase):
    name = "arc-agi-1"
    blurb = "ARC-AGI-1 grid induction, 2 attempts, exact match"


@benchmarks.register("arc-agi-2", "arc2")
class ArcAgi2(_ArcBase):
    name = "arc-agi-2"
    blurb = "ARC-AGI-2 grid induction, 2 attempts, exact match"
