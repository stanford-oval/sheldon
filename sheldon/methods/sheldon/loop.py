"""The method: plan, then rounds of Explore -> CandidateAnalysis ->
ComparativeAnalysis -> Exploit.

    round 1   explore opens the map and n hypotheses; all are tested locally
    round 2+  explore extends the map and proposes (or revives from the shelf);
              select picks at most `select_k` of the new, refined, repaired and
              revived; kept hypotheses carry over untested and take no slot
    each      candidate analysis per chosen hypothesis -> comparative analysis -> exploit
    end       budget, the round cap, or a round with nothing to test
    finalize  compose the answer from everything, whatever the budget says

`-p mode=staged` asks each stage on its own; `-p mode=session` runs a round's
deciding stages as turns of one conversation (`session.py`).
"""

from __future__ import annotations

from typing import Any

from sheldon.core.registry import methods
from sheldon.core.types import Prediction, RunContext, Task, Usage
from sheldon.methods.base import Method, answer_from_text, timer
from sheldon.methods.sheldon import candidate_analysis, resume, stages
from sheldon.methods.sheldon.run import Run, staged
from sheldon.methods.sheldon.state import PENDING


def _realize(run: Run, file_path: str) -> Any:
    if not file_path or not run.executable():
        return None
    try:
        path = file_path if file_path.startswith("/") or not run.workspace() else f"{run.workspace()}/{file_path}"
        program = open(path, encoding="utf-8").read()
        return run.ctx.benchmark.realize(run.task, {"program": program})
    except Exception as e:  # noqa: BLE001 - a broken artifact is an outcome, not a crash
        run.ctx.trace.log("realize.error", file_path=file_path, error=f"{type(e).__name__}: {e}")
        return None


async def finalize(run: Run) -> tuple[list[Any], Usage]:
    asked = await staged(run, stages.final_step(run))
    out = asked.parsed
    value = _realize(run, out.file_path.strip()) if out else None
    source = "artifact" if value is not None else "answer"
    leader = run.state.leader()
    if value is None and leader is not None:
        # The measured file, not a retyping of it: a program written into the
        # answer from memory scores below the file it was copying.
        value = _realize(run, leader.artifact)
        source = f"leader:{leader.id}" if value is not None else source
    if value is None:
        value = answer_from_text(run.ctx, run.task, out.answer if out else "")
    attempts = [value] if value is not None else []
    k = max(1, int(getattr(run.ctx.benchmark, "n_attempts", 1)))
    # The best artifacts on record back up a closing call that produced nothing usable.
    for h in run.state.ranked():
        if len(attempts) >= k:
            break
        extra = _realize(run, h.artifact)
        if extra is not None and all(extra != a for a in attempts):
            attempts.append(extra)
    run.ctx.trace.log("finalize", source=source, answer=str(value) if value is not None else None,
                      file_path=(out.file_path if out else ""), stopped_by=asked.stopped_by)
    return attempts[:k], asked.usage


async def round_staged(run: Run) -> tuple[Usage, bool]:
    """One round with every stage asked on its own. Returns (usage, tested any)."""
    usage = Usage()
    # A full map is settled before explore proposes anything: one call with
    # the model's whole attention on which regions to fold, outside the
    # session's conversation in both modes, so the merge is one decision.
    for spaces in ([stages.merge_needed(run)] if stages.merge_needed(run) else []):
        asked = await staged(run, stages.merge_step(run, spaces))
        usage += asked.usage
        stages.apply_merge(run, asked.parsed, spaces)
    asked = await staged(run, stages.explore_step(run))
    usage += asked.usage
    stages.apply_explore(run, asked.parsed, asked.retrievals)
    if stages.select_needed(run):
        asked = await staged(run, stages.select_step(run))
        usage += asked.usage
        chosen = stages.apply_select(run, asked.parsed)
    else:
        chosen = [h.id for h in run.state.of(PENDING)]
    chosen = chosen or stages.resumed_targets(run)
    if not chosen:
        return usage, False
    usage += await candidate_analysis.run_all(run, chosen)
    asked = await staged(run, stages.comparative_analysis_step(run))
    usage += asked.usage
    stages.apply_comparative_analysis(run, asked.parsed, asked.retrievals)
    asked = await staged(run, stages.exploit_step(run))
    usage += asked.usage
    stages.apply_exploit(run, asked.parsed)
    return usage, True


@methods.register("sheldon")
class Sheldon(Method):
    name = "sheldon"
    blurb = "explore -> candidate analysis -> comparative analysis -> exploit, over hypothesis and observation spaces"
    version = "hd-8d"

    async def run(self, task: Task, ctx: RunContext) -> Prediction:
        mode = str(self.param("mode", "staged"))
        if mode not in ("staged", "session"):
            raise ValueError(f"sheldon: unknown mode {mode!r}; use staged or session")
        run = Run(ctx=ctx, task=task, param=self.param)
        usage = Usage()
        stop = "max_rounds"
        with timer() as t:
            # A resumed run opens on another run's belief state, so it must not
            # re-plan or re-measure the seed: both are already in the state it
            # just restored, and measuring the seed again spends an objective
            # call to learn a number it has.
            done = resume.apply(run)
            if not done:
                usage += await stages.plan(run)
                await candidate_analysis.measure_seed(run)
            cap = ctx.budget.max_rounds or int(self.param("rounds", 6))
            round_costs: list[float] = []
            for round_i in range(done + 1, done + cap + 1):
                why = ctx.budget.reason()
                if why is not None:
                    stop = f"budget:{why.split(':')[0]}"
                    break
                # A spent objective budget leaves
                # a round nothing to measure; opening one pays for refusals.
                exhausted = getattr(ctx.benchmark, "objective_exhausted", None)
                if exhausted is not None and exhausted(task):
                    stop = "objective_exhausted"
                    break
                # A round is not opened that the money left cannot mostly pay
                # for: every loop in such a round is cut off, and an engine
                # costed after the fact takes the run past its cap.
                left = ctx.budget.usd_left()
                if round_costs and left is not None:
                    per_round = sum(round_costs) / len(round_costs)
                    if left < float(self.param("round_reserve", 0.5)) * per_round:
                        stop = "budget:round_reserve"
                        ctx.trace.log("budget.stop", where="round", usd_left=left, usd_per_round=per_round)
                        break
                opened_at = ctx.budget.spent.cost_usd
                ctx.budget.start_round()
                run.round = round_i
                if mode == "session":
                    from sheldon.methods.sheldon import session  # noqa: PLC0415

                    u, tested = await session.round_session(run)
                else:
                    u, tested = await round_staged(run)
                usage += u
                round_costs.append(ctx.budget.spent.cost_usd - opened_at)
                run.state.close_round()
                ctx.trace.log("round.done", round=round_i, best=run.state.best(),
                              tested=[h.id for h in run.state.of("tested")],
                              shelved=[h.id for h in run.state.of("shelved")],
                              observations=len(run.state.observations))
                if not tested:
                    stop = "nothing_to_test"
                    break
                if not ctx.budget.ok():
                    stop = f"budget:{ctx.budget.spend_reason()}"
                    break
            attempts, u = await finalize(run)
            usage += u
        snapshot = {"round": run.round, "stop_reason": stop, **run.state.snapshot()}
        ctx.trace.log("sheldon.final", **snapshot)
        return Prediction(
            task_id=task.id, answer=attempts[0] if attempts else None, attempts=attempts,
            belief=snapshot, rounds=run.round, usage=usage, wall_s=t.elapsed,
            extra={"stop_reason": stop},
        )
