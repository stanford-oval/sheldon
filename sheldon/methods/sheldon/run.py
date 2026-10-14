"""What one task's stages share, and the one way a stage is asked."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel

from sheldon.core.probes import ProbeSet, Retrieval
from sheldon.core.types import RunContext, Task, Usage
from sheldon.methods.sheldon.prompts import render
from sheldon.methods.sheldon.schemas import PlanOut
from sheldon.methods.sheldon.state import State
from sheldon.methods.stage import ParamFn, ask, isolated, own_retrievals, will_search


@dataclass
class Step:
    """One stage call, before it is made: the same object drives both modes."""

    template: str
    context: dict[str, Any]
    schema: type[BaseModel]
    tool: dict[str, Any]
    tag: str
    steps: int = 6
    required: bool = False


@dataclass
class Asked:
    parsed: Any
    usage: Usage
    #: the retrievals this call made itself
    retrievals: list[Retrieval] = field(default_factory=list)
    stopped_by: str | None = None


@dataclass
class Run:
    ctx: RunContext
    task: Task
    param: ParamFn
    state: State = field(default_factory=State)
    round: int = 0
    #: the round a `resume_from` state left off at, 0 for a run of its own
    resumed: int = 0
    plan: PlanOut | None = None
    #: the hypotheses chosen for this round's local tests
    chosen: list[str] = field(default_factory=list)

    def situation(self) -> str:
        if self.plan is not None and self.plan.situation.strip():
            return self.plan.situation
        getter = getattr(self.ctx.benchmark, "situation", None)
        return getter(self.task) if getter else self.task.question

    def deliverable(self) -> str:
        if self.plan is not None and self.plan.deliverable.strip():
            return self.plan.deliverable
        getter = getattr(self.ctx.benchmark, "deliverable", None)
        return getter(self.task) if getter else ""

    def requirements(self) -> list[str]:
        return [r for r in (self.plan.requirements if self.plan else []) if r.strip()]

    def has_tools(self) -> bool:
        own = bool(getattr(self.ctx.get_engine(), "brings_own_tools", False))
        return bool(self.ctx.tools) or own

    def executable(self) -> str:
        """The benchmark's contract for a runnable candidate, or empty where
        candidates are not executable or nothing on offer can run one."""
        bench = self.ctx.benchmark
        if not getattr(bench, "executable_candidates", False):
            return ""
        runners = {"run_python", "run_shell", self.objective()}
        tools = self.ctx.tools or ProbeSet()
        if not self.has_tools() or not any(n and n in tools for n in runners):
            return ""
        return str(getattr(bench, "candidate_contract", ""))

    def objective(self) -> str:
        return str(getattr(self.ctx.benchmark, "objective_probe", "") or "")

    def scored(self) -> bool:
        """Whether there is a scoring function to hill-climb on. Candidates need
        not be programs: any candidate the objective probe measures counts."""
        key = str(getattr(self.ctx.benchmark, "objective_key", "") or "")
        return bool(self.has_tools() and self.objective() and key
                    and self.objective() in (self.ctx.tools or ProbeSet()))

    def record(self) -> list[Retrieval]:
        return self.ctx.tools.record if self.ctx.tools is not None else []

    def workspace(self) -> str:
        for probe in (self.ctx.tools.probes.values() if self.ctx.tools is not None else []):
            cwd = getattr(probe, "cwd", None)
            if cwd:
                return str(cwd)
        return ""

    def score_of(self, r: Retrieval) -> float | None:
        """The objective value a retrieval measured, if it is an accepted one."""
        if getattr(r.probe, "name", "") != self.objective():
            return None
        key = str(getattr(self.ctx.benchmark, "objective_key", "") or "")
        values = [
            float(e[key]) for e in (r.result.evidence or [])
            if isinstance(e, dict) and not e.get("error") and not e.get("exploit")
            and isinstance(e.get(key), (int, float))
        ]
        return max(values) if values else None

    def seed_file(self) -> str:
        """The task's starting program: the file the benchmark seeded read-only
        into the workspace, if there is exactly one."""
        from pathlib import Path  # noqa: PLC0415

        ws = self.workspace()
        if not ws or not Path(ws).is_dir():
            return ""
        seeded = sorted(p for p in Path(ws).iterdir() if p.is_file() and not p.stat().st_mode & 0o222)
        return str(seeded[0]) if len(seeded) == 1 else ""

    def seed_text(self) -> str:
        s = self.state
        if not s.seed_path:
            return ""
        score = f"scores {s.seed_score:.6g}" if s.seed_score is not None else "could not be scored"
        return f"The starting program `{s.seed_path}` {score}."

    def base_context(self) -> dict[str, Any]:
        return {"situation": self.situation(), "tools": self.has_tools(), "executable": self.executable(),
                "seed": self.seed_text()}


async def staged(run: Run, step: Step, tools: Any = None) -> Asked:
    """Make one stage call on its own: a fresh prompt, the stage's own probe
    view, and the retrievals it made read back off the shared record. `tools`
    is a view the caller built, where a stage needs one of its own."""
    ctx = run.ctx
    sub = replace(ctx, tools=tools) if tools is not None else isolated(ctx)
    start = len(run.record())
    searching = will_search(ctx, run.param, step.tag.split(".")[0])
    prompt = render(step.template, **{**step.context, "tools": searching})
    got = await ask(
        sub, param=run.param, schema=step.schema, tool=step.tool, tag=step.tag,
        prompt=prompt, default_steps=step.steps, required=step.required,
    )
    return Asked(got.parsed, got.usage, own_retrievals(sub, run.record(), start), got.stopped_by)
