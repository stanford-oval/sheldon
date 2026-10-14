"""Run one (benchmark, method) cell and write its artifacts.

Layout, one directory per cell, named by config hash so re-running the same
config resumes rather than forking:

    runs/<benchmark>__<method>__<split>__<hash>/
        config.json          resolved config + git sha + dirty flag
        predictions.jsonl    one row per task
        scores.jsonl         one row per task
        summary.json         the row that shows up in `sheldon report`
        traces/<task>.jsonl  every call, hypothesis, and belief update

Resume is on by default: tasks that *succeeded* are skipped, and a task whose
row carries an `error` is run again — a timeout is not a result, and leaving it
in place would make one bad minute permanent. Between that and the LLM cache,
re-running after a scorer change costs approximately nothing, which is the whole
point, and re-running after a flaky one costs only the tasks that broke.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sheldon.benchmarks import get as get_benchmark
from sheldon.core.budget import Budget
from sheldon.core.cache import DiskCache
from sheldon.core.config import Config
from sheldon.core.io import (
    append_jsonl,
    git_dirty,
    git_sha,
    read_jsonl,
    rewrite_jsonl,
    runs_root,
    write_json,
)
from sheldon.core.llm import LLM
from sheldon.core.probes import ProbeSet, ShellProbe, build_probes
from sheldon.core.trace import Trace
from sheldon.core.types import Prediction, RunContext, Score, Task, Usage
from sheldon.engines import get as get_engine
from sheldon.methods import get as get_method


def run_dir(cfg: Config) -> Path:
    return runs_root() / cfg.run_id


async def run_cells(
    cells: list[Config],
    *,
    echo: bool = False,
    on_cell: Any = None,
    llm_factory: Any = None,
) -> list[dict[str, Any]]:
    """Run several cells at once against one shared set of limits.

    Cells are independent by construction — a different method, or a different
    benchmark, over its own tasks — so running them one after another only ever
    idled the endpoint. What they must not do is multiply the load: three cells
    at `-j 8` is 24 tasks in flight, which is how a clean sweep turns into a
    wall of 429s. So `concurrency` is enforced across all of them at once, not
    within each, and the same goes for calls in flight.

    Each cell keeps its own `LLM` so its usage and cache counters stay its own;
    they share the semaphores and the on-disk cache, which is content-addressed
    and safe for concurrent readers.
    """
    if not cells:
        return []
    jobs = max(1, min(c.concurrency for c in cells))
    calls = max((c.call_concurrency for c in cells), default=0) or jobs * 4
    task_sem = asyncio.Semaphore(jobs)
    call_sem = asyncio.Semaphore(max(calls, jobs))

    async def one(cfg: Config) -> dict[str, Any]:
        summary = await run_experiment(
            cfg,
            echo=echo,
            llm=llm_factory(cfg) if llm_factory else None,
            task_sem=task_sem,
            call_sem=call_sem,
        )
        if on_cell:
            on_cell(cfg, summary, None)
        return summary

    async def guarded(cfg: Config) -> dict[str, Any] | None:
        try:
            return await one(cfg)
        except Exception as e:  # one bad cell must not take the sweep with it
            if on_cell:
                on_cell(cfg, None, e)
            return None

    return [r for r in await asyncio.gather(*(guarded(c) for c in cells)) if r is not None]


def _check_stage_engines(cfg: Config) -> None:
    """Refuse the run if a stage's own engine is not installed.

    A method may route one stage through a different engine -- `implement_engine:
    codex` -- and that engine is checked here beside the run's, before anything
    is paid for. Left to the stage, a missing binary arrives as an unavailable
    engine halfway through the first task: empty text, the structured-output
    fallback quietly answers instead, and the arm reports a codex builder it
    never had. Generic on purpose, so the runner learns no stage names.
    """
    for key, name in sorted(cfg.params.items()):
        if not key.endswith("_engine") or not isinstance(name, str) or not name:
            continue
        stage = key[: -len("_engine")]
        params = {"reasoning": cfg.effort, **(cfg.params.get(f"{stage}_engine_params") or {})}
        problems = get_engine(name, **params).check()
        if problems:
            raise RuntimeError(
                f"engine {name!r} for stage {stage!r} is not ready: " + "; ".join(problems)
            )


def _checkpointed_round(directory: Path, cfg: Config) -> int | None:
    """The newest round a method checkpointed for this task, or None. Read
    off the files so the runner need not know which method wrote them; off
    when the run does not resume, or resumes tasks but not rounds."""
    if not cfg.resume or not (cfg.params or {}).get("resume_rounds", True):
        return None
    if not directory.is_dir():
        return None
    rounds = []
    for f in directory.glob("round_*.json"):
        try:
            rounds.append(int(f.stem.split("_", 1)[1]))
        except ValueError:
            continue
    return max(rounds) if rounds else None

async def run_experiment(
    cfg: Config,
    *,
    echo: bool = False,
    llm: LLM | None = None,
    on_task: Any = None,
    task_sem: asyncio.Semaphore | None = None,
    call_sem: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """Execute one cell. Returns the summary dict that `report` reads."""
    bench = get_benchmark(cfg.benchmark, **cfg.params.get("benchmark_opts", {}))
    problems = bench.check()
    if problems:
        raise RuntimeError(f"{cfg.benchmark} is not ready: " + "; ".join(problems))

    method = get_method(cfg.method, **cfg.params)
    # One effort for every model call: the run's `effort` (config or `--effort`)
    # reaches the engine as its `reasoning` default, and an explicit
    # `engine_params.reasoning` still wins. Without this the codex engine fell
    # back to a hard-coded `low` while the harness ran at the config's effort.
    engine_params = {"reasoning": cfg.effort, **(cfg.params.get("engine_params") or {})}
    engine = get_engine(cfg.engine, **engine_params)
    problems = engine.check()
    if problems:
        raise RuntimeError(f"engine {cfg.engine!r} is not ready: " + '; '.join(problems))
    _check_stage_engines(cfg)
    tasks = bench.load(cfg.split, limit=cfg.limit, seed=cfg.seed)
    if cfg.tasks:
        # Named ids bypass the limit entirely: `--task` exists to re-run one
        # trajectory after changing something, and subsampling around it would
        # defeat that.
        wanted = [t.strip() for t in cfg.tasks.split(",") if t.strip()]
        tasks = bench.load(cfg.split, limit=None, seed=cfg.seed)
        by_id = {t.id: t for t in tasks}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise RuntimeError(f"{cfg.benchmark}: no such task(s): {missing}")
        tasks = [by_id[w] for w in wanted]

    out = run_dir(cfg)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json",
        {"config": cfg.to_dict(), "git_sha": git_sha(), "git_dirty": git_dirty(),
         "n_tasks": len(tasks)},
    )

    done: set[str] = set()
    if cfg.resume and (out / "predictions.jsonl").exists():
        rows = list(read_jsonl(out / "predictions.jsonl"))
        done = {r["task_id"] for r in rows if not r.get("error")}
        retry = {r["task_id"] for r in rows} - done
        # A scoring failure leaves the prediction clean, so the check above
        # calls the task done and it is never retried — but its score row is a
        # 0.0 that no method earned. Under a rate-limited endpoint this is the
        # common case, not the rare one: the answer costs a whole method run
        # and the judge call that grades it is one more request to be throttled.
        # Both files are rewritten together below, so the kept prediction is
        # discarded with it and the task is simply run again.
        if (out / "scores.jsonl").exists():
            retry |= {r["task_id"] for r in read_jsonl(out / "scores.jsonl")
                      if (r.get("detail") or {}).get("score_error")}
        retry &= {r["task_id"] for r in rows}
        done -= retry
        if retry:
            # A row with an `error` is a task that never produced an answer —
            # a timeout, a provider hiccup, a bug — and none of those are a
            # verdict on the method. Counting it as done makes a five-second
            # blip permanent: the same command skips it forever, and its 0.0 in
            # scores.jsonl goes on dragging the headline number down as though
            # the method had answered and answered wrongly.
            #
            # So the stale rows come out of both files before the task goes back
            # on the list. They have to leave together — a prediction retried
            # against a score row that outlived it would double-count the task.
            _drop(out, retry)
            print(f"  retrying {len(retry)} failed task(s): {', '.join(sorted(retry))}")
    todo = [t for t in tasks if t.id not in done]

    owned_llm = llm is None
    llm = llm or LLM(
        provider=cfg.provider,
        base_url=cfg.base_url,
        model=cfg.model,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
        thinking=cfg.thinking,
        cache=DiskCache(enabled=cfg.cache),
        concurrency=max(cfg.concurrency, 4),
        sem=call_sem,
        trace_prompts=cfg.trace_prompts,
    )

    started = time.time()
    # How much a run records is a property of the run, so the config governs it
    # even when the caller brought its own LLM.
    llm.trace_prompts = cfg.trace_prompts

    template = Budget(
        max_rounds=cfg.max_rounds, max_calls=cfg.max_calls, max_usd=cfg.max_usd,
        max_seconds=cfg.max_seconds,
    )
    sem = task_sem or asyncio.Semaphore(cfg.concurrency)
    lock = asyncio.Lock()

    async def one(task: Task) -> None:
        async with sem:
            trace = Trace(
                task.id, out / "traces" / f"{_safe(task.id)}.jsonl", echo=echo,
                # A task resumed from a round keeps the rounds it already ran.
                keep_through_round=_checkpointed_round(
                    out / "checkpoints" / _safe(task.id), cfg
                ),
            )
            blind = task.blind()
            # One tool set per task, identical for every method: the benchmark's
            # own probes plus whatever the config named. Built from the blinded
            # task so a probe can never reach the gold answer.
            # An engine that arrives with its own shell and search is offered
            # nothing by default: giving `codex` a `run_shell` beside the one it
            # already has measures nothing, and these tools exist to give the
            # other engines the capabilities codex came with.
            #
            # Default, not rule. Some tools are not a second copy of something
            # codex has -- an oracle that *is* the objective, where withholding
            # it leaves nothing to optimise. `-p offer_tools=true`
            # hands them over, and the run is then tagged separately, because a
            # codex arm with our tools and one without are different experiments.
            # Built before the toolset, because a shell probe has to carry it.
            # `ctx.sandbox` alone reaches only the engines that bring their own
            # commands; a `run_shell` in `ctx.tools` runs whatever the host can
            # see unless it is handed the declaration as well, which is how an
            # isolated run comes to read the very files isolation is for.
            sandbox = bench.sandbox(blind)
            offer = cfg.params.get("offer_tools")
            if offer is None:
                offer = bench.probes_essential
            if engine.brings_own_tools and not offer:
                tools = ProbeSet()
            else:
                tools = ProbeSet(bench.probes(blind))
                # Built per task, never once per run: a shell probe is given this
                # task's workspace, sandbox and retrievals directory below, and a
                # shared one would keep the first task's for every task after it.
                for probe in build_probes(cfg.params.get("probes")):
                    tools.add(probe)
                # Retrievals go to a directory beside the trace, and any shell in
                # the set is told where. That is the whole of `recall`: with a
                # shell there is no need for a bespoke way back into what the run
                # already found, and `grep -l` over the directory beats reading
                # an entry back by an id you had to know in advance.
                tools.record_dir = out / "retrievals" / _safe(task.id)
                # One working directory per task, seeded by the benchmark, and
                # the directory every shell opens in. A shell with no cwd runs
                # in the repository root, which is both the wrong place to work
                # and a place that holds other runs, other tasks and -- on some
                # benchmarks -- the answer.
                workspace = out / "workspace" / _safe(task.id)
                workspace.mkdir(parents=True, exist_ok=True)
                bench.workspace(blind, workspace)
                for probe in tools.probes.values():
                    # A probe that takes a path resolves it against the same
                    # directory the shell opens in, and nothing else -- the
                    # evaluator imports what it is handed, so any other path
                    # would be a read of that file through its error channel.
                    if hasattr(probe, "workspace") and not probe.workspace:
                        probe.workspace = str(workspace)
                    if isinstance(probe, ShellProbe):
                        if probe.sandbox is None:
                            probe.sandbox = sandbox
                        if not probe.cwd:
                            probe.cwd = str(workspace)
                        probe.env.setdefault("SHELDON_WORKSPACE", str(workspace))
                        if not probe.record_dir:
                            probe.record_dir = str(tools.record_dir)
                            probe.env.setdefault("SHELDON_RETRIEVALS", probe.record_dir)
            trace.log("tools", available=sorted(tools.probes))
            # Whatever the engine is offered, it is held to the benchmark's
            # isolation: codex gets no probes above, and the sandbox is the one
            # thing standing between its shell and the answer key on disk.
            if sandbox is not None:
                trace.log("sandbox", read_only=list(sandbox.read_only),
                          writable=list(sandbox.writable))
            ctx = RunContext(
                llm=llm,
                budget=template.fresh(),
                trace=trace,
                benchmark=bench,
                tools=tools,
                engine=engine,
                params=cfg.params,
                sandbox=sandbox,
                # Beside the trace and the retrievals, so a task interrupted
                # mid-run can be picked up where it stopped instead of being
                # re-paid in full. `runs/` resume is per task and cannot.
                checkpoint_dir=out / "checkpoints" / _safe(task.id),
            )
            t0 = time.perf_counter()
            try:
                # Binds this task's trace for everything below, so the shared
                # LLM can log each call to the right file without every
                # signature in between having to carry a trace.
                with trace:
                    pred = await method.run(blind, ctx)
            except Exception as e:
                trace.log("error", error=f"{type(e).__name__}: {e}",
                          tb=traceback.format_exc()[-2000:])
                pred = Prediction(task_id=task.id, error=f"{type(e).__name__}: {e}",
                                  usage=ctx.budget.spent)
            # Methods time themselves, but a method that raised never got to,
            # and a slow failure is exactly the one worth seeing in the table.
            if not pred.wall_s:
                pred.wall_s = time.perf_counter() - t0
            try:
                score = await bench.score(task, pred, ctx)
            except Exception as e:
                trace.log("score.error", error=f"{type(e).__name__}: {e}")
                score = Score(task_id=task.id, score=0.0, detail={"score_error": str(e)})

            async with lock:
                append_jsonl(out / "predictions.jsonl", pred.to_dict())
                append_jsonl(out / "scores.jsonl", score.to_dict())
            if on_task:
                on_task(task, pred, score)

    if todo:
        await asyncio.gather(*(one(t) for t in todo))

    summary = summarize(cfg, out, llm, started=started, finished=time.time())
    write_json(out / "summary.json", summary)
    if owned_llm:
        await llm.aclose()
        llm.cache.close()
    return summary


def _drop(out: Path, task_ids: set[str]) -> None:
    """Remove every row for these tasks, so a retry lands on a clean slate."""
    for name in ("predictions.jsonl", "scores.jsonl"):
        path = out / name
        if path.exists():
            rewrite_jsonl(path, [r for r in read_jsonl(path) if r["task_id"] not in task_ids])


def summarize(
    cfg: Config,
    out: Path,
    llm: LLM | None = None,
    *,
    started: float | None = None,
    finished: float | None = None,
) -> dict[str, Any]:
    """Fold scores.jsonl + predictions.jsonl into the reportable row.

    Two different clocks, and the difference matters. `wall_s` is how long the
    cell took end to end at `concurrency` jobs — what you waited. `latency_*`
    are per-task, measured inside the method — what one task costs in time,
    independent of how many ran at once. Quoting the first as if it were the
    second makes a method look faster by having been run on a wider machine.

    `started`/`finished` come from the live run. Re-summarizing a finished run
    has neither, so the span of the run directory's own files stands in.
    """
    scores = list(read_jsonl(out / "scores.jsonl")) if (out / "scores.jsonl").exists() else []
    preds = (
        list(read_jsonl(out / "predictions.jsonl"))
        if (out / "predictions.jsonl").exists()
        else []
    )

    values = [s["score"] for s in scores]
    metrics: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        for k, v in (s.get("metrics") or {}).items():
            metrics[k].append(float(v))

    usage = Usage()
    rounds: list[float] = []
    latencies: list[float] = []
    errors = 0
    for p in preds:
        u = p.get("usage") or {}
        usage += Usage(**u) if u else Usage()
        rounds.append(float(p.get("rounds", 0)))
        if p.get("wall_s"):
            latencies.append(float(p["wall_s"]))
        if p.get("error"):
            errors += 1

    if started is not None and finished is not None:
        wall = finished - started
    else:
        wall = _dir_span(out)

    n = len(values)
    mean = statistics.fmean(values) if values else 0.0
    return {
        "run_id": cfg.run_id,
        "benchmark": cfg.benchmark,
        "method": cfg.method,
        "split": cfg.split,
        "model": cfg.model,
        "tag": cfg.tag,
        "n": n,
        "score": round(mean, 4),
        "stderr": round(_stderr(values), 4),
        "metrics": {k: round(statistics.fmean(v), 4) for k, v in sorted(metrics.items())},
        "errors": errors,
        "mean_rounds": round(statistics.fmean(rounds), 2) if rounds else 0.0,
        "usage": usage.to_dict(),
        "cost_per_task": round(usage.cost_usd / n, 5) if n else 0.0,
        "cost_usd": round(usage.cost_usd, 4),
        # What the provider was actually charged this time. Below `cost_usd`
        # whenever the cache was used; the two differing is normal, and only
        # `cost_usd` is comparable across runs.
        "billed_usd": round((llm.billed.cost_usd if llm else usage.cost_usd), 4),
        "calls_per_task": round(usage.calls / n, 2) if n else 0.0,
        "wall_s": round(wall, 1),
        "concurrency": cfg.concurrency,
        "latency_mean_s": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "latency_p50_s": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p90_s": round(_pct(latencies, 0.9), 1) if latencies else 0.0,
        "started_at": _iso(started),
        "finished_at": _iso(finished),
        "cache": llm.cache.stats() if llm else {},
        "git_sha": git_sha(),
        "config_hash": cfg.config_hash,
    }


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / (len(values) ** 0.5)


def _safe(task_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.#" else "_" for c in task_id)[:120]


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. n is small here; interpolation would imply a
    precision the sample does not have."""
    if not values:
        return 0.0
    ordered = sorted(values)
    i = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[i]


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")


def _dir_span(out: Path) -> float:
    """When a run was being written, for runs summarized after the fact."""
    stamps = [p.stat().st_mtime for p in out.rglob("*") if p.is_file()]
    return max(stamps) - min(stamps) if len(stamps) > 1 else 0.0
