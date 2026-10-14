"""Local testing: build, evaluate and evolve one hypothesis, a few times over.

Implementation and local testing are one loop here. Each iteration is one
sub-agent call (on `candidate_analysis_engine`, codex in the intended arm) that builds or
improves the artifact, runs it, and gathers evidence for and against the
claim. The harness, not the agent, decides whether to go again:

    with a score   stop at `candidate_analysis_max_evals` (3) evaluator calls, or when
                   `candidate_analysis_patience` measured iterations in a row fail to beat
                   the best so far by `candidate_analysis_min_gain_pct` percent of the seed
                   program's score. The best so far starts at the best of the
                   hypothesis's lineage, so a refinement has to beat its parent
    without one    stop when the agent says `done`, or an iteration files
                   nothing new, or at `candidate_analysis_max_evals` iterations

Scores are read off the evaluator's own retrievals, never off the agent's
report. A claim revision is accepted only with a basis -- evidence gathered
against the claim, or a small change of approach expected to help -- and a
rationale; it becomes a sibling and the remaining iterations continue on it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sheldon.core.probes import Probe, ProbeResult, ProbeSet
from sheldon.core.types import Usage
from sheldon.methods.sheldon.run import Run, Step, staged
from sheldon.methods.sheldon.schemas import SUBMIT_LOCAL, LocalOut
from sheldon.methods.sheldon.stages import file_observations
from sheldon.methods.sheldon.state import Hypothesis


async def run_all(run: Run, hids: list[str]) -> Usage:
    """Every chosen hypothesis at once; each writes only its own files."""
    hs = [run.state.hypotheses[h] for h in hids if h in run.state.hypotheses]
    if run.param("candidate_analysis_concurrent", True):
        results = await asyncio.gather(*(one(run, h) for h in hs))
    else:
        results = [await one(run, h) for h in hs]
    usage = Usage()
    for u in results:
        usage += u
    return usage


def _artifact_path(run: Run, h: Hypothesis, i: int) -> str:
    """A fresh file per iteration, so the best one is never overwritten."""
    ws = run.workspace()
    if not ws:
        return ""
    starting = sorted(p for p in Path(ws).iterdir() if p.is_file() and not p.stat().st_mode & 0o222) \
        if Path(ws).is_dir() else []
    suffix = starting[0].suffix if starting else ".py"
    return str(Path(ws) / f"hd_{h.id}_{i}{suffix}")


class _Capped(Probe):
    """The objective probe with this loop's evaluation budget in front of it.

    Checked at the tool, not between iterations: one iteration can call the
    evaluator many times. Refused calls
    run nothing and are not counted; every call that reaches the evaluator is,
    a crash included."""

    def __init__(self, inner: Probe, budget: dict[str, int]) -> None:
        self.inner, self.budget = inner, budget
        self.name = inner.name  # type: ignore[misc]
        self.description = inner.description  # type: ignore[misc]

    @property
    def input_schema(self) -> dict[str, Any]:  # type: ignore[override]
        return self.inner.input_schema

    async def run(self, **kwargs: Any) -> ProbeResult:
        # One charge per distinct program: an agent that gave up waiting and
        # asked again is asking for
        # the same measurement, which the benchmark serves from its cache.
        key = _program_key(kwargs)
        if key not in self.budget["charged"]:
            if self.budget["left"] <= 0:
                return ProbeResult(
                    summary=f"evaluation budget for this hypothesis is spent ({self.budget['cap']} runs); "
                    "submit what you have", ok=False)
            self.budget["left"] -= 1
            self.budget["charged"].add(key)
        # Kept, and shielded from the caller: an agent whose own tool call
        # timed out leaves the measurement running, and the harness waits for
        # it at the end of the iteration rather than losing the score.
        task = asyncio.ensure_future(self.inner.run(**kwargs))
        self.budget["calls"].append((dict(kwargs), task))
        return await asyncio.shield(task)


def _program_key(kwargs: dict[str, Any]) -> str:
    """What a call measures: the file's contents where it names one, else its
    arguments. A candidate that is not a program names neither, and
    keying on the program alone made every such call one free measurement."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    path = str(kwargs.get("file_path") or "")
    text = str(kwargs.get("program") or "")
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = path
    elif not text:
        text = json.dumps({k: v for k, v in kwargs.items() if k != "refutes"}, sort_keys=True, default=str)
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def _capped_view(run: Run, budget: dict[str, Any]) -> ProbeSet | None:
    tools = getattr(run.ctx, "tools", None)
    if tools is None or run.objective() not in tools.probes:
        return None
    view = tools.view()
    view.probes = {**tools.probes, run.objective(): _Capped(tools.probes[run.objective()], budget)}
    return view


def _consolidate(run: Run, root: Hypothesis, head: Hypothesis) -> None:
    """One live hypothesis per loop. A loop that revised its claim leaves a
    line of siblings; only the best-scoring one stays live (the latest, where
    none scored) and the rest are shelved behind it, recoverable. Revise used
    to be handed every step of every line -- 18 hypotheses from five loops,
    half of them "X does not help" -- and kept eleven."""
    s = run.state
    line, cur = [], head
    while cur is not None and cur.id not in {x.id for x in line}:
        line.append(cur)
        if cur.id == root.id:
            break
        cur = s.hypotheses.get(cur.parent)
    if len(line) < 2:
        return
    scored = [x for x in line if x.best is not None]
    keep = max(scored, key=lambda x: x.best) if scored else head
    for x in line:
        if x.id != keep.id and x.status != "shelved":
            s.shelve(x, f"superseded in its local loop by {keep.id}", run.round)
    if keep.status == "shelved":
        s.tested(keep, run.round)


async def measure_seed(run: Run) -> None:
    """Score the starting program once, before anything is built, so every
    gain is measured against it. Recorded like any retrieval."""
    s = run.state
    s.seed_path = run.seed_file()
    if not s.seed_path or not run.scored() or run.ctx.tools is None:
        return
    await run.ctx.tools.dispatch(run.objective(), {"file_path": s.seed_path})
    r = run.record()[-1] if run.record() else None
    s.seed_score = run.score_of(r) if r is not None else None
    run.ctx.trace.log("seed", path=s.seed_path, score=s.seed_score, retrieval=r.id if r else None)


def min_gain(run: Run, reference: float | None) -> float:
    """`candidate_analysis_min_gain_pct` percent of the seed's score -- or of the first
    score measured, where there is no seed -- as an absolute threshold."""
    base = run.state.seed_score if run.state.seed_score is not None else reference
    return abs(base or 0.0) * float(run.param("candidate_analysis_min_gain_pct", 0.5)) / 100


async def _late_scores(run: Run, budget: dict[str, Any], start: int, file_path: str
                       ) -> list[tuple[float, dict[str, float], str]]:
    """Scores of this iteration's evaluations, waiting for any the agent left
    running when it returned. Bounded by the engine's probe timeout."""
    calls = budget["calls"][start:]
    pending = [t for _k, t in calls if not t.done()]
    if pending:
        limit = float((run.param("candidate_analysis_engine_params", None) or {}).get("probe_timeout", 1200))
        await asyncio.wait(pending, timeout=limit)
        run.ctx.trace.log("candidate_analysis.waited", round=run.round, pending=len(pending),
                          finished=sum(1 for t in pending if t.done()))
    key = str(getattr(run.ctx.benchmark, "objective_key", "") or "")
    out = []
    for kwargs, t in calls:
        if not t.done() or t.cancelled() or t.exception() is not None:
            continue
        named = str(kwargs.get("file_path") or "")
        if named and file_path and _name(named) != _name(file_path):
            continue
        value, metrics = _numbers(t.result().evidence, key)
        if value is not None:
            out.append((value, metrics, _candidate(kwargs), named))
    return out


def _name(path: str) -> str:
    return str(path or "").rsplit("/", 1)[-1]


def _candidate(kwargs: dict[str, Any]) -> str:
    """A measured candidate that is not a file, as the call that measured it
    named it (`name=value`), so the hypothesis can carry it as its artifact."""
    return ", ".join(f"{k}={v}" for k, v in sorted((kwargs or {}).items())
                     if k not in ("refutes", "file_path", "program") and v not in (None, ""))


def _numbers(evidence: list[Any], key: str) -> tuple[float | None, dict[str, float]]:
    """The objective value in an evaluation's evidence, and every other number
    it reported."""
    for e in evidence or []:
        if isinstance(e, dict) and not e.get("error") and not e.get("exploit") \
                and isinstance(e.get(key), (int, float)):
            metrics = {k: float(v) for k, v in e.items()
                       if k != key and isinstance(v, (int, float)) and not isinstance(v, bool)}
            return float(e[key]), metrics
    return None, {}


def _scores(run: Run, retrievals: list[Any], file_path: str) -> list[tuple[float, dict[str, float], str, str]]:
    """The accepted measurements this iteration made, of its own file where the
    call names one, each with the metrics that came with it, the candidate it
    measured and the file the call named."""
    key = str(getattr(getattr(run.ctx, "benchmark", None), "objective_key", "") or "")
    out = []
    for r in retrievals:
        if run.score_of(r) is None:
            continue
        named = str((r.arguments or {}).get("file_path") or "")
        if named and file_path and _name(named) != _name(file_path):
            continue
        value, metrics = _numbers(getattr(getattr(r, "result", None), "evidence", None), key)
        out.append((value if value is not None else run.score_of(r), metrics, _candidate(r.arguments or {}), named))
    return out


async def one(run: Run, h: Hypothesis) -> Usage:
    ctx, s = run.ctx, run.state
    scored = run.scored()
    cap = max(1, int(run.param("candidate_analysis_max_evals", 3)))
    patience = max(1, int(run.param("candidate_analysis_patience", 2)))
    usage = Usage()
    previous: list[dict[str, Any]] = []
    evals, stale, stop = 0, 0, "max_iterations"
    current = h
    repair = h.repair
    budget: dict[str, Any] = {"cap": cap, "left": cap, "charged": set(), "calls": []}
    # What this loop has to beat: the best of the claim and everything it was
    # refined from, so a revision is compared with its parent, not with nothing.
    best = s.lineage_best(h)
    for i in range(1, cap + 1):
        if not ctx.budget.ok():
            stop = "budget"
            break
        target = _artifact_path(run, current, i) if run.executable() else ""
        # Build on this claim's own line: the last iteration's file, else its
        # own or its nearest ancestor's artifact, else the seed. Never the
        # run's leader: a loop pointed at the best program stops testing its
        # own claim, and the claims converge on one program.
        if previous and previous[-1]["file_path"]:
            base, base_from = previous[-1]["file_path"], "your previous iteration"
        else:
            base, owner = s.lineage_artifact(current)
            base_from = ("this hypothesis's best program" if owner == current.id
                         else f"the program of {owner}, the hypothesis it refines" if owner else "")
            if not base:
                base, base_from = s.seed_path, "the starting program" if s.seed_path else ""
        step = Step(
            "candidate_analysis",
            dict(run.base_context(), scored=scored, evaluator=run.objective(), iteration=i,
                 max_iterations=cap, artifact_path=target, base_artifact=base, repair=repair,
                 hypothesis_id=current.id, hypothesis=s.render_hypothesis(current),
                 base_from=base_from, o_regions=s.render_regions("o"), previous=previous),
            LocalOut, SUBMIT_LOCAL, f"candidate_analysis.{current.id}.{i}",
            steps=int(run.param("candidate_analysis_probe_calls", 12)),
        )
        view = _capped_view(run, budget)
        calls_before = len(budget["calls"])
        asked = await staged(run, step, tools=view)
        usage += asked.usage
        out: LocalOut = asked.parsed or LocalOut()
        file_path = (out.file_path or "").strip() or (target if target and Path(target).is_file() else "")
        found = _scores(run, asked.retrievals, file_path)
        if view is not None:
            found = found or await _late_scores(run, budget, calls_before, file_path)
        # Every run of the evaluator counts toward the cap, a crash included:
        # the cap is on runs, and a crash is only unscored, not free.
        evals = cap - budget["left"] if view is not None else evals + sum(
            1 for r in asked.retrievals if getattr(getattr(r, "probe", None), "name", "") == run.objective())
        top = max(found, key=lambda x: x[0]) if found else None
        score, metrics = (top[0], top[1]) if top else (None, {})
        # A program is its file; any other candidate is what the call measured.
        # An agent that reports no path, having evaluated a file other than
        # the one it was pointed at, still measured a file: the call names it.
        measured = (file_path or (top[3] if top else "")) if run.executable() else (top[2] if top else "")

        # The score belongs to the claim the artifact was built under. The
        # revision comes after the measurement and is for the next iteration:
        # crediting the child with it puts a refuting score on a claim that
        # predicted the opposite.
        before = best
        if score is not None:
            own_best = current.best
            current.scores.append(score)
            if own_best is None or score >= own_best:
                # Never blank an artifact: a best score with no file behind it
                # drops the hypothesis out of `leader()` and out of finalize.
                current.artifact = measured or current.artifact
                current.metrics = metrics
            best = score if best is None else max(best, score)
        elif file_path and not scored:
            current.artifact = file_path
        # Gathered while testing the claim as it stood, so filed against it.
        filed = file_observations(run, out.observations, asked.retrievals, "local", current.id)
        # A revised claim is a sibling and the rest of the loop works on it --
        # but only with a basis and a reason; anything else is a summary of
        # results written in the claim's place, and is dropped.
        if out.statement.strip():
            accepted = bool(out.revision_basis and out.rationale.strip() and out.predicts_improvement)
            child = s.refine(current, out.statement, out.rationale, run.round) if accepted else None
            ctx.trace.log("candidate_analysis.revision", round=run.round, hypothesis=current.id, iteration=i,
                          accepted=child is not None, basis=out.revision_basis,
                          child=child.id if child else None, statement=out.statement[:500],
                          rationale=out.rationale[:500])
            if child is not None:
                s.tested(current, run.round)
                current = child
        current.notes = out.notes.strip()[:600]
        s.tested(current, run.round)
        repair = ""
        previous.append({"iteration": i, "score": score, "file_path": file_path or measured,
                         "notes": out.notes.strip()[:400]})
        ctx.trace.log("candidate_analysis.iteration", round=run.round, hypothesis=current.id, iteration=i, score=score,
                      file_path=file_path, observations=filed, evals=evals, stopped_by=asked.stopped_by)

        if scored:
            if evals >= cap:
                stop = "max_evals"
                break
            # An iteration that measured nothing (a crash, a skipped run) is
            # neither a gain nor a plateau; the iteration cap still bounds it.
            if score is not None:
                stale = 0 if before is None or score - before > min_gain(run, before or score) else stale + 1
            if stale >= patience:
                stop = "diminishing_returns"
                break
        elif out.done or not filed:
            stop = "done" if out.done else "nothing_new"
            break
    _consolidate(run, h, current)
    ctx.trace.log("candidate_analysis.done", round=run.round, hypothesis=h.id, final=current.id, stop=stop,
                  iterations=len(previous), evals=evals, best=best, seed=s.seed_score)
    return usage
