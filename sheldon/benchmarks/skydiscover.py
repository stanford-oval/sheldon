"""SkyDiscover — program optimisation against a scored evaluator.

    github.com/skydiscover-ai/skydiscover/tree/main/benchmarks   (Apache-2.0)

Upstream is an evolutionary-search framework (AdaEvolve, OpenEvolve, GEPA, ...)
and `benchmarks/` is the ~200 tasks it is measured on: the AlphaEvolve math
problems (circle packing, Heilbronn, autocorrelation inequalities, matmul), the
ADRS systems tasks from Berkeley (cloud broadcast, MoE load balancing, model
placement, column reordering, transaction scheduling), a QNN topology task, GPU
kernels, competitive programming and more. Every task has the same shape:

    initial_program.py     a working but weak solution, with EVOLVE-BLOCK markers
    config.yaml            the system prompt that states the problem, and settings
    evaluator/             a Docker image whose `evaluate.sh <program> <mode>`
                           prints `{"status", "combined_score", "metrics", ...}`
    evaluator.py           or a host-side `evaluate(program_path) -> dict`

and the job is to rewrite the program so `combined_score` goes up. Most of the
math tasks normalise it so 1.0 is the AlphaEvolve result and more beats it.

This is an *optimization* benchmark in the same sense as `dockstring`, and the
same rule applies: the evaluator is the objective, not an oracle to withhold,
so `evaluate_program` goes to every method as an ordinary probe, and calls to
it are unlimited by default (`benchmark_opts.max_evaluations`). Every evaluation
costs a model turn to ask for, so the run's own spend cap already bounds a
task, and a separate ration of the objective only shapes the search in ways
nobody chose: a run can spend its whole ration before its first round closes
and reason blind thereafter.
`max_evaluations: 100` is upstream's setting -- `max_iterations` in every task
config, one program generated and evaluated per iteration -- and is what to use
when comparing against its published numbers. `0` is the no-evaluator ablation,
which prices what a model can write from the statement alone.

Like `dockstring`, a task scores on the best program the method *measured*,
not the one it handed in (`benchmark_opts.score_on: submitted` restores the
stricter reading), and both numbers stay in the metrics with the gap between
them, because a `finalize` that submits a worse program than it found is a
defect worth seeing. The score-time evaluation is a fresh run of the evaluator
in upstream's `test` mode -- it never draws on, or tops up, the method's budget.

The headline `score` is `combined_score` itself, which is what upstream reports.
It is comparable within a task and not across them, so the table also carries
`initial_score` (the starting program, measured once per task at scoring time)
and `beats_initial`, which is the one number that means the same thing on every
task: did the method improve on what it was handed.

A program that does not finish the task is rejected, mechanically. PRISM
averages the pressure over only the configurations a program did not raise on,
so a program that places a few of them outscores an honest one; txn reports
`validity`. Where an
evaluator reports how much of the task a program completed (`success_rate`,
`validity`), a program below 1.0 is returned to the method as rejected -- with
the rule it broke -- and never counts as `best_found`; scoring applies the same
rule to the program submitted. A model judge reading each program is not a
substitute: a rule that holds only when the reader agrees is not a rule. The
other known holes --
zero edge costs on cloudcast, eplb's workload file on disk -- are closed by the
sandbox rather than by a reader: the file is not mounted, and an unmounted file
cannot be read.

Evaluators run in Docker by default with no network -- the candidate is a
model-written program and the container is the only thing between it and the
host. `benchmark_opts.runtime: host` runs a task's `evaluator.py` in a
subprocess of this interpreter instead, for machines without Docker; it needs
the task's requirements installed here and offers no isolation.

Data comes from `scripts/prepare_skydiscover.py --download`, which copies the
task directories under `data/skydiscover/` and writes the `tasks.jsonl` index
`load()` reads. GPU tasks (`gpu_mode`, `kernelbench`), the Frontier-CS judge,
ALE-Bench, image generation and prompt optimisation each need infrastructure
this harness does not provide, and the prepare script leaves them out unless
asked; `sheldon check skydiscover` says what is present.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from sheldon.benchmarks.base import Benchmark
from sheldon.core import isolation
from sheldon.core.io import read_jsonl
from sheldon.core.probes import CallableProbe, Probe, ProbeResult
from sheldon.core.registry import benchmarks
from sheldon.core.types import Falsification, Prediction, RunContext, Score, Task, Verifier

#: The families the prepare script takes by default: everything whose
#: evaluator is a self-contained CPU image and needs no external service.
DEFAULT_FAMILIES = ("math", "ADRS", "qnn_circuit_topology")

#: How long one `docker build` may take. Images install scipy, jax, torch.
BUILD_TIMEOUT = 1800.0


def _digest(program: str) -> str:
    """Identity of a program: its text, with trailing whitespace stripped.

    Two submissions that differ only in trailing whitespace are the same
    program, and the budget must not charge twice for it.
    """
    return hashlib.sha256(program.rstrip().encode("utf-8")).hexdigest()[:16]


# -- running an evaluator ----------------------------------------------------


#: A per-instance metric vector longer than this is summarised rather than
#: listed: the point is to show which instance is weak, and thirty numbers in
#: every probe summary is a wall, not a signal.
_VECTOR_LIMIT = 16


def _flatten(key: str, seq: Any) -> dict[str, float]:
    """A numeric sequence, as scalar metrics the model will actually see.

    An evaluator that reports per-instance numbers -- llm_sql's `hit_rates`,
    one per dataset -- returned them as a list, and a list is not a float, so
    it was filed as an artifact and the probe never showed it. The run could
    see its combined score and its runtime and not which of the five datasets
    it was losing on, while the system it is compared against had that vector
    in front of it every iteration. Indexed scalars keep every consumer's
    float contract and put the vector back in the summary.
    """
    values = []
    for v in seq:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return {}
        values.append(float(v))
    if not values:
        return {}
    if len(values) > _VECTOR_LIMIT:
        return {f"{key}_n": float(len(values)), f"{key}_min": min(values),
                f"{key}_mean": sum(values) / len(values), f"{key}_max": max(values)}
    return {f"{key}_{i}": v for i, v in enumerate(values)}


def _normalize(data: dict[str, Any], stderr: str = "") -> dict[str, Any]:
    """Upstream's container JSON, in one shape, whichever protocol produced it.

    The container protocol is `{"status", "combined_score", "metrics",
    "artifacts"}`; a host `evaluate()` returns a flat dict where anything
    non-numeric is an artifact and `error` marks failure. Both collapse to
    the same record here.
    """
    if "status" in data or "metrics" in data:
        metrics = {
            k: float(v)
            for k, v in (data.get("metrics") or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        for k, v in (data.get("metrics") or {}).items():
            if isinstance(v, (list, tuple)):
                metrics.update(_flatten(k, v))
        artifacts = {k: str(v) for k, v in (data.get("artifacts") or {}).items()}
        status = str(data.get("status", "success"))
        score = data.get("combined_score", metrics.get("combined_score", 0.0))
    else:
        metrics, artifacts = {}, {}
        for k, v in data.items():
            if isinstance(v, bool):
                metrics[k] = float(v)
            elif isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, str):
                artifacts[k] = v
            elif isinstance(v, (list, tuple)):
                # Indexed scalars where the entries are numbers, and the JSON
                # beside them either way: the artifact is what `score` records,
                # the metrics are what the probe shows.
                metrics.update(_flatten(k, v))
                artifacts[k] = json.dumps(list(v), default=str)
            elif isinstance(v, dict):
                artifacts[k] = json.dumps(v, default=str)
        status = "error" if "error" in artifacts else "success"
        score = metrics.get("combined_score", 0.0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    if score != score:  # NaN is not a score
        score = 0.0
    metrics["combined_score"] = score
    error = artifacts.get("error") or ("" if status == "success" else status)
    if stderr and "stderr" not in artifacts:
        artifacts["stderr"] = stderr[-4000:]
    return {
        "status": status,
        "combined_score": score,
        "metrics": metrics,
        "artifacts": artifacts,
        "error": error or None,
    }


async def _exclusive(lock_path: str, run: Any, slots: int = 1) -> Any:
    """`await run()` holding one of `slots` exclusive locks shared by every
    process on the host: `lock_path` itself, then `lock_path.1`, ... Taken by
    polling rather than a blocking `flock` in a thread: a waiter that is
    cancelled must not go on to acquire a lock that nothing will ever
    release."""
    if not lock_path:
        return await run()
    import fcntl  # noqa: PLC0415 - POSIX only, and only where the option is on

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    paths = [lock_path] + [f"{lock_path}.{i}" for i in range(1, max(1, slots))]
    handles = [open(p, "a+") for p in paths]  # noqa: SIM115 - closed below
    held = None
    try:
        while held is None:
            for fd in handles:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = fd
                    break
                except BlockingIOError:
                    continue
            if held is None:
                await asyncio.sleep(0.5)
        return await run()
    finally:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
        for fd in handles:
            fd.close()


def _infrastructure(result: dict[str, Any]) -> bool:
    """True when the evaluator itself could not run, as opposed to the program
    failing inside it. The former is not a measurement."""
    return bool(result.get("infrastructure"))


#: Metrics an evaluator reports for how much of the task a program completed.
#: A program below 1.0 on any of them is scored over the part it finished, and
#: that number is not the task's score.
COMPLETION_KEYS = ("success_rate", "validity")


def _incomplete(metrics: dict[str, Any]) -> tuple[str, str] | None:
    """`(rule, mechanism)` where the program did not finish the task, else None."""
    for key in COMPLETION_KEYS:
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v < 1.0:
            return (
                "a program must complete every instance the evaluator poses; a score over "
                "the instances it finished is not the task's score",
                f"{key} = {float(v):.4g}: the evaluator scored only what the program completed",
            )
    return None


def _exploit(measurement: dict[str, Any]) -> bool:
    """Whether this measurement was rejected as not the task's score."""
    return bool((measurement.get("rejected") or {}).get("exploit"))


def _reject(entry: dict[str, Any] | None, result: dict[str, Any]) -> dict[str, Any] | None:
    """Rule a measurement out where the program did not finish the task, and
    stamp the ruling onto the measured entry so `best_found` skips it. A
    measurement that errored is already no score. Returns the ruling, or None."""
    if result.get("error"):
        return None
    found = _incomplete(result.get("metrics") or {})
    if found is None:
        return None
    rule, mechanism = found
    ruling = {"exploit": True, "mechanism": mechanism, "rule": rule, "by": "completeness"}
    if entry is not None:
        entry["rejected"] = ruling
    return ruling


def _failure(reason: str, **artifacts: str) -> dict[str, Any]:
    return {
        "status": "error",
        "combined_score": 0.0,
        "metrics": {"combined_score": 0.0},
        "artifacts": artifacts,
        "error": reason,
    }


def _parse_stdout(stdout: str, stderr: str) -> dict[str, Any]:
    """The last JSON object on stdout; evaluators print progress before it."""
    text = stdout.strip()
    if not text:
        return _failure("evaluator printed nothing", stderr=stderr[-4000:])
    for candidate in (text, text.splitlines()[-1]):
        data: Any = None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            # Upstream's qnn evaluator ends with `print(result)` -- a Python
            # dict repr, single quotes and all. Read it as one rather than
            # scoring every program on that task 0 for a print statement.
            try:
                data = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(data, dict):
            return _normalize(data, stderr)
    return _failure("evaluator output was not JSON", stdout=text[-2000:], stderr=stderr[-4000:])


def _kill_container(name: str) -> None:
    """Best-effort kill of a named container; `--rm` then removes it."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


class DockerEvaluator:
    """Upstream's containerised protocol: build once, `docker run` per program.

    Upstream keeps one container alive and `docker exec`s into it. A fresh
    `--rm` container per evaluation costs half a second and needs no lifecycle
    -- nothing to stop when a run is interrupted, and nothing one candidate
    can leave behind for the next. The image build is memoised per process,
    and Docker's layer cache makes the second process's build a no-op.
    """

    _built: dict[Path, str] = {}

    def __init__(
        self, evaluator_dir: Path, image: str, timeout: float, suffix: str = ".py",
        env: dict[str, str] | None = None,
    ) -> None:
        self.dir = Path(evaluator_dir)
        self.image = image
        self.timeout = float(timeout)
        self.suffix = suffix
        self.env = dict(env or {})

    def build(self) -> str:
        tag = self._built.get(self.dir)
        if tag:
            return tag
        proc = subprocess.run(
            ["docker", "build", "-q", "-t", self.image, str(self.dir)],
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"docker build failed for {self.dir}:\n{proc.stderr[-4000:]}")
        self._built[self.dir] = self.image
        return self.image

    def _run(self, program: str, mode: str) -> dict[str, Any]:
        try:
            self.build()
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            return {**_failure(f"{type(e).__name__}: {e}"), "infrastructure": True}
        with tempfile.TemporaryDirectory(prefix="sheldon-sky-") as tmp:
            path = Path(tmp) / f"candidate{self.suffix}"
            path.write_text(program, encoding="utf-8")
            os.chmod(path, 0o644)
            inside = f"/tmp/candidate{self.suffix}"
            # Named, so a timeout can kill the container and not just the client.
            # `subprocess.run(timeout=...)` kills `docker run`, and the daemon
            # keeps the container going regardless: two hung candidates ran at
            # ~100% CPU for five hours after their runs had finished, and that
            # load is what pushed other tasks' evaluations past their limits.
            name = f"sheldon-sky-{os.getpid()}-{uuid.uuid4().hex[:12]}"
            cmd = [
                "docker", "run", "--rm", "--name", name, "--network", "none",
                "-v", f"{path}:{inside}:ro",
                *(arg for k, v in sorted(self.env.items()) for arg in ("-e", f"{k}={v}")),
                self.image, inside, mode,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                _kill_container(name)
                return _failure(f"evaluation timed out after {self.timeout:g}s")
            except FileNotFoundError:
                return {**_failure("docker is not installed"), "infrastructure": True}
            if proc.returncode != 0 and not proc.stdout.strip():
                return _failure(
                    f"evaluator exited with code {proc.returncode}", stderr=proc.stderr[-4000:]
                )
            return _parse_stdout(proc.stdout, proc.stderr)


    async def evaluate(self, program: str, mode: str = "train") -> dict[str, Any]:
        return await asyncio.to_thread(self._run, program, mode)


_HOST_RUNNER = r"""
import importlib.util, json, os, sys
evaluator_path, program_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(evaluator_path))
spec = importlib.util.spec_from_file_location("_sheldon_sky_evaluator", evaluator_path)
module = importlib.util.module_from_spec(spec)
real_stdout = sys.stdout
sys.stdout = sys.stderr  # an evaluator's prints must not land in the JSON
try:
    spec.loader.exec_module(module)
    result = module.evaluate(program_path)
except BaseException as e:  # noqa: BLE001 - the whole point is to report it
    import traceback
    result = {"combined_score": 0.0, "error": f"{type(e).__name__}: {e}",
              "traceback": traceback.format_exc()}
finally:
    sys.stdout = real_stdout
if not isinstance(result, dict):
    result = {"combined_score": 0.0, "error": f"evaluate() returned {type(result).__name__}"}
print(json.dumps(result, default=str))
"""


class HostEvaluator:
    """Upstream's plain-Python protocol, in a subprocess of this interpreter.

    No isolation: the candidate runs as this user with this environment. It
    exists for machines without Docker and for tests, and `runtime: host` has
    to be asked for.
    """

    def __init__(
        self, evaluator_py: Path, timeout: float, suffix: str = ".py",
        env: dict[str, str] | None = None,
    ) -> None:
        # Absolute: both runners start in the evaluator's own directory, where a
        # relative path to it no longer resolves.
        self.path = Path(evaluator_py).resolve()
        self.timeout = float(timeout)
        self.suffix = suffix
        self.env = dict(env or {})

    def _run(self, program: str, mode: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="sheldon-sky-") as tmp:
            path = Path(tmp) / f"candidate{self.suffix}"
            path.write_text(program, encoding="utf-8")
            env = {**os.environ, **self.env, "PYTHONHASHSEED": "0", "SKYDISCOVER_EVAL_MODE": mode}
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _HOST_RUNNER, str(self.path), str(path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(self.path.parent),
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return _failure(f"evaluation timed out after {self.timeout:g}s")
            if proc.returncode != 0 and not proc.stdout.strip():
                return _failure(
                    f"evaluator exited with code {proc.returncode}", stderr=proc.stderr[-4000:]
                )
            return _parse_stdout(proc.stdout, proc.stderr)


    async def evaluate(self, program: str, mode: str = "train") -> dict[str, Any]:
        return await asyncio.to_thread(self._run, program, mode)


# -- candidates --------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*([\w+.-]*)[^\n]*\n(.*?)```", re.DOTALL)


def parse_program(text: str) -> str | None:
    """The program in a reply: the last fenced code block, else the whole text
    if it looks like code.

    The last block rather than the first because a model that quotes the
    starting program before writing its own puts the answer second; and the
    whole block, never a diff -- upstream's diff protocol is a search-loop
    convenience and there is nothing here to apply one to.
    """
    if not text:
        return None
    blocks = [body for _lang, body in _FENCE.findall(text)]
    if blocks:
        chosen = blocks[-1].strip("\n")
        return chosen if chosen.strip() else None
    stripped = text.strip()
    if re.search(r"^\s*(def|class|import|from)\s", stripped, re.MULTILINE):
        return stripped
    return None


def _candidate_program(candidate: Any) -> str | None:
    if isinstance(candidate, str):
        return parse_program(candidate)
    if isinstance(candidate, dict):
        for key in ("program", "code", "answer"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                got = parse_program(value)
                if got:
                    return got
    return None


class ProgramVerifier:
    """Attack a candidate by running it: refuted if it errors, or if it claims
    a score it does not reach. Only reachable through `oracle_probes`, which
    is opt-in, since it evaluates outside the method's budget."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    async def check(self, candidate: Any) -> Falsification:
        program = _candidate_program(candidate)
        if not program:
            return Falsification(
                refuted=True,
                counterexamples=[{"kind": "no_program", "detail": "candidate carries no program"}],
                detail="no program to run",
                checked=0,
            )
        result = await self.evaluator.evaluate(program, mode="train")
        if result["error"]:
            return Falsification(
                refuted=True,
                counterexamples=[{"kind": "error", "detail": result["error"]}],
                detail=result["error"],
                checked=1,
            )
        claimed = candidate.get("score") if isinstance(candidate, dict) else None
        got = result["combined_score"]
        try:
            claimed_f = float(claimed) if claimed is not None else None
        except (TypeError, ValueError):
            claimed_f = None
        if claimed_f is not None and got < claimed_f:
            return Falsification(
                refuted=True,
                counterexamples=[{"kind": "score", "claimed": claimed_f, "measured": got}],
                detail=f"claimed combined_score {claimed_f}, measured {got}",
                checked=1,
            )
        return Falsification(refuted=False, detail=f"combined_score={got}", checked=1)


# -- the benchmark -------------------------------------------------------------


@benchmarks.register("skydiscover", "sky")
class SkyDiscover(Benchmark):
    name = "skydiscover"
    #: `eval` is every prepared task; a family name selects one directory.
    splits = ("eval", "math", "adrs", "qnn")
    answer_kind = "program"
    answer_format = (
        "Give the complete program in one fenced code block -- the whole file, runnable as "
        "written, with the EVOLVE-BLOCK markers kept and the entry-point function the "
        "evaluator calls still defined. Not a diff."
    )
    executable_candidates = True
    candidate_contract = (
        "the complete program file, runnable as written by the task's evaluator, keeping "
        "the entry point the evaluator calls and the EVOLVE-BLOCK markers"
    )
    probes_essential = True
    objective_probe = "evaluate_program"
    objective_key = "combined_score"
    blurb = "rewrite a program to raise an evaluator's score (SkyDiscover: AlphaEvolve math, ADRS)"

    def __init__(self, root: Any = None, **opts: Any) -> None:
        super().__init__(root, **opts)
        # None means unlimited: the evaluator is the objective here, and every
        # evaluation costs a model turn to ask for, so the run's own spend cap is
        # what bounds a task. `0` is still the no-evaluator ablation -- what a
        # model can write from the statement alone -- and an integer caps it,
        # which is what upstream does at `max_iterations: 100`.
        raw = opts.get("max_evaluations")
        self.max_evaluations = None if raw is None else int(raw)
        self.runtime = str(opts.get("runtime", "docker"))
        self.score_on = str(opts.get("score_on", "best"))
        self.show_evaluator = bool(opts.get("show_evaluator", False))
        self.timeout_override = opts.get("eval_timeout")
        #: One evaluation at a time on this host, across processes. Where the
        #: score includes wall clock, two evaluations running at once measure
        #: each other: a program timed alone reads several times slower beside
        #: others. A path, or true for the default one.
        #: An integer is how many may run at once: the lock's cost is that the
        #: evaluator's own loading and scoring queue behind it too.
        #: The default lock lives on local disk: "this host" is the point, and
        #: on a network filesystem a lock is shared by every machine that
        #: mounts it, and can be held from another host or left stale.
        lock = opts.get("eval_exclusive")
        self.eval_slots = int(lock) if isinstance(lock, int) and not isinstance(lock, bool) else 1
        self.eval_lock = (
            str(lock) if isinstance(lock, str) and lock
            else f"/tmp/sheldon-skydiscover-eval-{os.getuid()}.lock" if lock else ""
        )
        #: Settings a task's evaluator reads from its environment -- a thread
        #: count, a per-program time limit. Here rather than exported in the
        #: shell because they change what a program scores: `benchmark_opts`
        #: is in `config_hash` and the ambient environment is not.
        self.evaluator_env = {str(k): str(v) for k, v in (opts.get("evaluator_env") or {}).items()}
        #: A file whose text replaces the task's problem statement, relative
        #: to the task's directory unless absolute. For reproducing another
        #: system's run of the same task under *its* statement, where the
        #: text is theirs and is fetched at prepare time rather than kept
        #: here. In `benchmark_opts`, so in `config_hash`: a different
        #: statement is a different experiment.
        self.statement_file = str(opts.get("statement_file") or "").strip()
        if opts.get("exploit_judge"):
            raise ValueError(
                "benchmark_opts.exploit_judge was removed: a program that does not complete "
                "the task is rejected mechanically now (see the module docstring), and the "
                "sandbox keeps the evaluator's data out of reach. Delete the option."
            )
        #: Built on first use, so a run that never judges never makes a client;
        #: tests put a stub here.
        #: task id -> {program digest: measurement}. The probe's budget dict is
        #: this, so scoring sees every program the method evaluated.
        self._measured: dict[str, dict[str, dict[str, Any]]] = {}
        #: task id -> the starting program's own measurement, taken once.
        self._initial: dict[str, dict[str, Any]] = {}
        self._index: list[dict[str, Any]] | None = None

    # -- data -----------------------------------------------------------------

    def _rows(self) -> list[dict[str, Any]]:
        if self._index is None:
            path = self.root / "tasks.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"{self.name}: {path} not found. Run scripts/prepare_skydiscover.py --download"
                )
            self._index = list(read_jsonl(path))
        return self._index

    def _selected(self, split: str) -> list[dict[str, Any]]:
        rows = self._rows()
        chosen = self.opts.get("tasks")
        if isinstance(chosen, str):
            chosen = [t.strip() for t in chosen.split(",") if t.strip()]
        if chosen:
            wanted = {str(t) for t in chosen}
            return [r for r in rows if r["id"] in wanted or r["family"] in wanted]
        if split == "eval":
            return rows
        return [r for r in rows if r["family"].lower() == split.lower()]

    def _dir(self, row: dict[str, Any]) -> Path:
        return self.root / row["dir"]

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        tasks: list[Task] = []
        for row in self._selected(split):
            task_dir = self._dir(row)
            program = (task_dir / row["initial_program"]).read_text(encoding="utf-8")
            evaluator_src = ""
            if self.show_evaluator:
                src = task_dir / row["evaluator"]
                if src.is_dir():
                    src = src / "evaluator.py"
                if src.exists():
                    evaluator_src = src.read_text(encoding="utf-8")
            payload = {
                "family": row["family"],
                "task": row["task"],
                "language": row.get("language", "python"),
                "suffix": row.get("suffix", ".py"),
                "system_message": self._statement(row, task_dir) + self._time_limit(row),
                "initial_program": program,
                "evaluator_source": evaluator_src,
                "timeout": float(row.get("timeout") or 600.0),
            }
            tasks.append(
                Task(
                    id=row["id"],
                    benchmark=self.name,
                    question=self._render(row, program, evaluator_src),
                    payload=payload,
                    # There is no gold program. What scoring needs and the
                    # method must not see is nothing -- the evaluator is offered
                    # to every method -- so `answer` carries only provenance.
                    answer={"source": row.get("source", "")},
                    meta={"family": row["family"], "task": row["task"]},
                )
            )
        return self.subsample(tasks, limit, seed)

    def _render(self, row: dict[str, Any], program: str, evaluator_src: str) -> str:
        budget = (
            "You may call `evaluate_program` as often as it is worth calling; a program "
            "you have already evaluated is free to evaluate again. What ends the task is the "
            "run's own budget, so spend measurements on questions worth answering."
            if self.max_evaluations is None
            else (
                f"You may call `evaluate_program` {self.max_evaluations} times on this task; a "
                "program you have already evaluated is free to evaluate again."
                if self.max_evaluations
                else "No evaluator is available on this task -- you get one shot, unmeasured."
            )
        )
        lang = row.get("language", "python")
        parts = [
            f"# {row['family']} / {row['task']}",
            "",
            "## The problem",
            "",
            (self._statement(row, self._dir(row)).strip() + self._time_limit(row))
            or "(the task states no problem text)",
            "",
            "## The starting program",
            "",
            "This runs and scores, and the score is what you have to beat. Keep its "
            "entry point and the `EVOLVE-BLOCK` markers; rewrite anything between them.",
            "",
            f"```{lang}",
            program.rstrip("\n"),
            "```",
            "",
            "## How it is scored",
            "",
            "The evaluator runs the program and reports `combined_score`, higher is better, "
            "plus the task's own metrics. A program that errors, times out or violates the "
            "task's constraints scores 0. " + budget,
        ]
        parts += [
            "",
            "A program that does not complete every instance the evaluator poses is rejected "
            "and does not count: a score over the instances it finished is not the task's "
            "score. A score that comes from changing what the evaluator measures -- "
            "fabricating the task's own quantities, reading data the problem does not give "
            "you, or tampering with the evaluation -- is not a solution.",
        ]
        if self.opts.get("sandbox_python") and self.opts.get("isolate"):
            # Said, because nothing else would: a shell that had no numpy is a
            # shell nobody tries a second time, and the one attempt a run made
            # there died on the import.
            parts += [
                "",
                "## Running code yourself",
                "",
                "`python` in your shell is the evaluator's own interpreter, with the packages "
                "the evaluator runs your program under. The data it scores on is not available "
                "to you: run a program on input you construct, to learn whether it runs, how long "
                "it takes and what its output looks like, before you spend an evaluation on it.",
            ]
        if evaluator_src:
            script = self._evaluator_script(row)
            parts += ["", "## The evaluator", ""]
            if script is not None:
                parts += [
                    f"The scoring code, also on disk at `{script}`. It is read-only: you can "
                    "read it and run it, and you cannot change what scores you. It is here so "
                    "you know exactly how a program is measured and on what. A program that "
                    "reads the evaluator's own files or data when it runs, or changes what is "
                    "measured instead of solving the problem, is an exploit and scores nothing.",
                    "",
                ]
            parts += ["```python", evaluator_src.rstrip("\n"), "```"]
        parts += [
            "",
            "## Answer",
            "",
            "The complete program, as one fenced code block. It must be the program you "
            "want scored.",
        ]
        return "\n".join(parts)

    def _time_limit(self, row: dict[str, Any]) -> str:
        """The evaluator's time limit, as a line of the statement, under
        `benchmark_opts.state_timeout`.

        Upstream's statements never mention it, so an agent learns the limit
        only by losing evaluations to it. On txn_scheduling, where runtime is
        not scored and a longer search usually finds a shorter makespan, that
        pushes programs to the edge, and evaluations are lost to the limit. A program
        told the limit can stop its own search in time. In `benchmark_opts`,
        so in `config_hash`: the statement is part of the task.
        """
        if not self.opts.get("state_timeout"):
            return ""
        limit = float(self.timeout_override or row.get("timeout") or 600.0)
        return (
            f"\n\nEach evaluation runs the program once and stops it after {limit:g} seconds "
            "of wall-clock time; a program that has not returned by then scores 0."
        )

    def _evaluator_script(self, row: dict[str, Any]) -> Path | None:
        """The file `show_evaluator` shows, or None where there is none."""
        src = self._dir(row) / row["evaluator"]
        if src.is_dir():
            src = src / "evaluator.py"
        return src if src.is_file() else None

    def _statement(self, row: dict[str, Any], task_dir: Path) -> str:
        """The task's problem text, or the text of `statement_file` in its
        place. A missing file is an error and never a fallback: a run that
        silently used the default statement would be filed under the other
        one's name."""
        if not self.statement_file:
            return row.get("system_message", "")
        path = Path(self.statement_file).expanduser()
        if not path.is_absolute():
            path = task_dir / path
        if not path.is_file():
            raise FileNotFoundError(
                f"benchmark_opts.statement_file: no file at {path} for task {row['id']}"
            )
        return path.read_text(encoding="utf-8")

    # -- evaluation -----------------------------------------------------------

    def _evaluator(self, task: Task) -> Any:
        row = next(r for r in self._rows() if r["id"] == task.id)
        task_dir = self._dir(row)
        timeout = float(self.timeout_override or task.payload["timeout"])
        suffix = task.payload.get("suffix", ".py")
        evaluator = task_dir / row["evaluator"]
        if self.runtime == "host" or not evaluator.is_dir():
            script = evaluator / "evaluator.py" if evaluator.is_dir() else evaluator
            return HostEvaluator(script, timeout=timeout, suffix=suffix, env=self.evaluator_env)
        image = "sheldon-skydiscover-" + re.sub(r"[^a-z0-9_.-]+", "-", task.id.lower())
        return DockerEvaluator(
            evaluator, image=image, timeout=timeout, suffix=suffix, env=self.evaluator_env
        )

    def probes(self, task: Task) -> list[Probe]:
        """The evaluator, to every method alike, under one per-task budget.

        Same construction as `dockstring.dock_molecule`: the slot is reserved
        synchronously before the first `await`, so a within-round fan-out
        cannot overrun the cap, and an in-flight evaluation is shared so two
        attacks on the same program pay once.
        """
        evaluator = self._evaluator(task)
        limit = self.max_evaluations
        seen: dict[str, dict[str, Any]] = self._measured.setdefault(task.id, {})
        pending: dict[str, asyncio.Task[dict[str, Any]]] = {}

        def clean_best(exclude: str = "") -> float | None:
            scores = [r["combined_score"] for k, r in seen.items()
                      if k != exclude and not r.get("error") and not _exploit(r)]
            return max(scores) if scores else None

        def counter() -> str:
            """`7/40` under a cap, `7` without one."""
            return f"{len(seen)}/{limit}" if limit is not None else str(len(seen))

        def read_file(file_path: str) -> tuple[str, str]:
            """The program at a workspace path, or why it could not be read.

            Confined to the task's workspace: the evaluator imports what it is
            given, and a traceback quotes the line that failed, so a path to
            any other file on disk would be a read of that file through the
            error channel. The workspace is set by the runner once the
            directory exists; before that there is nothing to resolve against.
            """
            root = getattr(probe, "workspace", None)
            if not root:
                return "", "file_path cannot be used here: this run has no workspace"
            base = Path(root).resolve()
            path = Path(file_path).expanduser()
            path = (path if path.is_absolute() else base / path).resolve()
            if base != path and base not in path.parents:
                return "", f"file_path must be inside the workspace {base}"
            if not path.is_file():
                return "", f"no file at {path}"
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return "", f"could not read {path}: {e}"
            if not text.strip():
                return "", f"the file at {path} is empty"
            return text, ""

        async def evaluate_program(
            program: str = "", file_path: str = "", refutes: bool = False
        ) -> ProbeResult:
            program = program if isinstance(program, str) else str(program or "")
            file_path = str(file_path or "").strip()
            if file_path:
                # The file wins where both are given: what was built is what
                # gets measured, and the measurement is attributed to it.
                program, why = read_file(file_path)
                if why:
                    return ProbeResult(summary=why, ok=False)
            if not program.strip():
                return ProbeResult(summary="no program given: pass file_path or program", ok=False)
            key = _digest(program)
            where = {"file_path": file_path} if file_path else {}
            if key in seen:
                result = seen[key]
            else:
                inflight = pending.get(key)
                if inflight is None:
                    if limit is not None and len(seen) + len(pending) >= limit:
                        return ProbeResult(
                            summary=f"evaluation budget exhausted ({limit} programs evaluated)",
                            ok=False,
                        )
                    inflight = asyncio.ensure_future(
                        _exclusive(self.eval_lock, lambda p=program: evaluator.evaluate(p, mode="train"),
                                   self.eval_slots)
                    )
                    pending[key] = inflight
                    # Cleared by whoever started it, when it finishes -- not in
                    # each waiter's `finally`, where the first one out removes an
                    # entry its siblings are still sharing and the next caller
                    # pays to measure the same program again.
                    inflight.add_done_callback(lambda _t, k=key: pending.pop(k, None))
                try:
                    # Shielded: a caller that goes away must not cancel the
                    # measurement every other stage is waiting on. The codex
                    # bridge cancels the coroutine when its client times out
                    # (`probe_server._call`), and unshielded that kills
                    # every other stage sharing the same digest.
                    result = await asyncio.shield(inflight)
                except asyncio.CancelledError:
                    # This caller was cancelled: let it unwind. The evaluation
                    # survives for whoever else is waiting.
                    if not inflight.cancelled():
                        raise
                    return ProbeResult(
                        summary="the evaluation was cancelled before it finished; "
                                "run it again",
                        ok=False,
                    )
                # A program that failed still spent the call: the evaluator ran
                # and said so. An evaluator that could not run at all did not,
                # and the model should get the reason back, not a smaller budget.
                if not _infrastructure(result):
                    seen[key] = {**result, "program": program, "digest": key}
            if result["error"]:
                extra = result["artifacts"].get("stderr") or result["artifacts"].get("traceback") or ""
                return ProbeResult(
                    summary=f"combined_score=0.0  error: {result['error']}"
                    + (f"\n{extra[-1500:]}" if extra else "")
                    + f"\n{counter()} programs evaluated",
                    ok=key in seen,
                    refutes=bool(refutes),
                    evidence=[{"digest": key, "error": result["error"], **where}],
                )
            verdict = _reject(seen.get(key), result)
            best = max((r["combined_score"] for r in seen.values() if not _exploit(r)),
                       default=result["combined_score"])
            metrics = {k: v for k, v in result["metrics"].items() if k != "combined_score"}
            if verdict is not None:
                shown = clean_best()
                return ProbeResult(
                    summary=(
                        f"REJECTED: combined_score={result['combined_score']:.6g} does not count "
                        "toward this task.\n"
                        f"what the program did: {verdict['mechanism']}\n"
                        f"rule every program must meet: {verdict['rule']}\n"
                        f"{counter()} programs evaluated; best combined_score so far "
                        + (f"{shown:.6g}" if shown is not None else "none")
                    ),
                    refutes=bool(refutes),
                    evidence=[{
                        "digest": key,
                        "combined_score": result["combined_score"],
                        **metrics,
                        **where,
                        "exploit": True,
                        "exploit_mechanism": verdict["mechanism"],
                        "exploit_rule": verdict["rule"],
                    }],
                )
            return ProbeResult(
                summary=(
                    f"combined_score={result['combined_score']:.6g}"
                    + (f"  {json.dumps(metrics, sort_keys=True)}" if metrics else "")
                    + f"\n{counter()} programs evaluated; best combined_score so far {best:.6g}"
                ),
                refutes=bool(refutes),
                evidence=[{"digest": key, "combined_score": result["combined_score"],
                           **metrics, **where}],
            )

        # With an interpreter on the shell's PATH (`sandbox_python`), the
        # harness's own `run_python` is withheld: it is the harness's Python,
        # with a 10 s cap and none of the task's packages, and a builder that
        # reaches for it files a `ModuleNotFoundError` as a fact about the
        # task. The shell runs the real one.
        row = next((r for r in self._rows() if r["id"] == task.id), None)
        probes: list[Probe] = [
            p for p in super().probes(task)
            if not (p.name == "run_python" and self._sandbox_python(row) is not None)
        ]
        if limit is None or limit > 0:
            probe = CallableProbe(
                    name="evaluate_program",
                    description=(
                        "Run the task's evaluator on a complete program and return its "
                        "combined_score (higher is better) and metrics. This is the real "
                        "measurement the task is scored on. Give `file_path` for a program "
                        "you wrote to a file in your workspace -- what you built is then "
                        "what gets measured -- or `program` for the whole source as text. "
                        + (
                            "Re-evaluating a program you have already run is free. "
                            if limit is None
                            else f"Budgeted: {limit} distinct programs for this task; "
                            "re-evaluating one you have already run is free. "
                        )
                        + "A program that does not complete every instance is rejected and "
                        "does not count. Either way it is the whole program, not a diff."
                    ),
                    fn=evaluate_program,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": (
                                    "Path of the program file to evaluate, inside your "
                                    "workspace. Preferred: the file you built is what is measured."
                                ),
                            },
                            "program": {
                                "type": "string",
                                "description": (
                                    "The complete program source, as text, for a program you "
                                    "have not written to a file."
                                ),
                            },
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                    # The evaluator is the objective. A measurement that contradicts
                    # a claim about it is as settled as this benchmark gets.
                    max_strength=0.95,
                )
            #: set by the runner once the task's workspace exists; `file_path`
            #: resolves against it and nowhere else.
            probe.workspace = None
            probes.append(probe)
        return [*probes, *self.oracle_probes(task)]

    def workspace(self, task: Task, path: Path) -> None:
        """The starting program, as a file, in the directory the shell opens in.

        It is already in the prompt, so this adds no information -- it changes
        the price. Passing a 17,550-character seed through `evaluate_program`
        costs ~4,400 output tokens every time, which made rewriting from
        scratch the cheaper move: on llm_sql every one of 38 measured programs
        was a 478-3,389 character replacement, none carried the seed's
        algorithm, and the run finished at 0.093 against a seed worth 0.673. As
        a file it can be read, copied and edited in place, and only the result
        crosses the boundary.
        """
        row = next((r for r in self._rows() if r["id"] == task.id), None)
        if row is None:
            return
        src = self._dir(row) / row["initial_program"]
        if src.exists():
            dst = path / row["initial_program"]
            # Read-only: a builder told to edit in place can write its own
            # candidate over the seed, and every later stage then reads a
            # hypothesis's artifact as the task's starting point. The
            # file is unlinked first because a re-seed on resume finds the
            # read-only copy from last time.
            if dst.exists():
                dst.chmod(0o644)
                dst.unlink()
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            dst.chmod(0o444)
        if self.opts.get("seed_modules"):
            for module in self._seed_modules(row):
                dst = path / module.name
                if dst.exists():
                    dst.chmod(0o644)
                    dst.unlink()
                shutil.copyfile(module, dst)

    def _seed_modules(self, row: dict[str, Any]) -> list[Path]:
        """The evaluator-directory modules the starting program imports.

        txn_scheduling's seed does `from txn_simulator import Workload` and
        `from workloads import WORKLOAD_1`, both resolved from the evaluator's
        directory at scoring time, and the sandbox mounts neither, so without
        them a program could only be run through the evaluator. What is
        copied is what the program already loads and calls to compute its own
        objective -- the simulator and the fixed instance it optimises -- not
        `evaluator.py` and not data the program never reads. A module the seed
        does not import is never copied, which keeps llm_sql's datasets and
        eplb's workload file out.

        Copied writable, unlike the seed. Methods take the one read-only file
        in the workspace as the starting program (`Run.seed_file`), and
        several read-only files would make that rule find none. A
        copy the agent edits misleads only its own runs -- the evaluator
        scores a copy of the program's text from a temporary directory with
        its own directory first on `sys.path`.

        Opt-in (`benchmark_opts.seed_modules`)
        because it changes what an agent can do, and only `benchmark_opts` is
        in `config_hash`.
        """
        evaluator = self._dir(row) / row["evaluator"]
        if not evaluator.is_dir():
            return []
        seed = self._dir(row) / row["initial_program"]
        try:
            tree = ast.parse(seed.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module.split(".")[0])
        seed_name = Path(row["initial_program"]).stem
        return sorted(
            evaluator / f"{n}.py" for n in names
            if n not in {"evaluator", seed_name} and (evaluator / f"{n}.py").is_file()
        )

    def sandbox(self, task: Task) -> isolation.Sandbox | None:
        """What an agent with a shell may see, under `benchmark_opts.isolate`.

        There is no gold program here, so `Task.blind()` has nothing to hide --
        and the disk still decides the task. The evaluator directory holds the
        scoring code that `show_evaluator` withholds on purpose, and beside it
        `evaluator/datasets/*.csv` is the workload the statement calls hidden.
        A shell in the repository root reads both, and then the job is to
        compute an ordering on the evaluation data rather than to find an
        algorithm that generalises -- the same score for a different task.

        So: the starting program read-only, because it is in the prompt
        already and a file is the cheaper way to work on 17,550 characters;
        the evaluator only where `show_evaluator` has already put its source
        in the prompt; nothing else of the repository. Writable paths are the
        engine's own, added by whatever runs the command.

        `run_python` is untouched, as on `arc_agi` -- it is not a shell, and
        the declaration rides on the probes that run commands.
        """
        if not self.opts.get("isolate"):
            return None
        if not isolation.available():
            raise RuntimeError(
                "benchmark_opts.isolate is set but bubblewrap (`bwrap`) is not installed; "
                "refusing to give the agent the host filesystem"
            )
        row = next((r for r in self._rows() if r["id"] == task.id), None)
        read_only: list[Path] = []
        if row is not None:
            task_dir = self._dir(row)
            seed = task_dir / row["initial_program"]
            if seed.exists():
                read_only.append(seed)
            if self.show_evaluator:
                # The script and nothing beside it. The evaluator's directory
                # also holds what it scores *on* -- eplb's `expert-load.json`,
                # llm_sql's datasets -- and the prompt shows the code, not the
                # data. Mounting the directory would hand over both, and on
                # eplb reading the workload ahead is the known exploit.
                src = self._evaluator_script(row)
                if src is not None:
                    read_only.append(src)
        env: dict[str, str] = {}
        interpreter = self._sandbox_python(row)
        if interpreter is not None:
            venv, home = interpreter
            read_only += [venv, home]
            env = {
                "PATH": f"{venv / 'bin'}:{isolation.SAFE_PATH}",
                "VIRTUAL_ENV": str(venv),
                # The mount is read-only and HOME is a private /tmp.
                "NUMBA_CACHE_DIR": "/tmp/numba",
                "MPLCONFIGDIR": "/tmp/mpl",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        return isolation.Sandbox.of(read_only=read_only, writable=[], env=env,
                                    nice=int(self.opts.get("shell_nice") or 0))

    def _sandbox_python(self, row: dict[str, Any] | None) -> tuple[Path, Path] | None:
        """The interpreter `benchmark_opts.sandbox_python` puts on the shell's
        PATH: a virtualenv, and the installation its `python` links to.

        The isolated shell sees `/usr` and nothing else, so on a host whose
        system Python has no numpy an agent cannot run a line of the kind of
        program the task asks for. On the denoising task every one of 73
        programs a run wrote was a `replace` on the file it was handed, and
        the one attempt to import scipy died on the import: a new algorithm
        had to be written blind, with a 45 s evaluation as the only way to
        learn whether it parsed. The venv is the evaluator's own, so what
        runs here runs there. Only the environment is mounted -- not the task
        directory beside it, which holds the dataset the split is made from.

        A path relative to the task directory, or absolute. An option rather
        than a default because it changes what an agent can do, and only
        `benchmark_opts` is in `config_hash`.
        """
        raw = self.opts.get("sandbox_python")
        if not raw or row is None:
            return None
        venv = Path(str(raw)).expanduser()
        venv = (venv if venv.is_absolute() else self._dir(row) / venv).resolve()
        python = venv / "bin" / "python"
        if not python.exists():
            raise RuntimeError(
                f"benchmark_opts.sandbox_python: no interpreter at {python}; "
                "build the task's environment first"
            )
        # <install>/bin/python3.x -> <install>, which holds the stdlib the
        # venv's pyvenv.cfg points back to.
        home = python.resolve().parent.parent
        if str(home) in ("/usr", "/"):
            return venv, venv  # a system interpreter is already in the sandbox
        return venv, home

    def verifier(self, task: Task) -> Verifier | None:
        return ProgramVerifier(self._evaluator(task))

    def probe_state(self, task: Task) -> Any:
        """The programs already evaluated for this task, so a resumed run does
        not get a second allowance of `max_evaluations`."""
        return self._measured.get(task.id) or {}

    def restore_probe_state(self, task: Task, state: Any) -> None:
        if isinstance(state, dict):
            # The probe's budget dict *is* this, so the closure the runner
            # already built sees the restored entries: a re-evaluated program
            # is free and the count picks up where it stopped.
            self._measured.setdefault(task.id, {}).update(
                {k: v for k, v in state.items() if isinstance(v, dict)}
            )

    def python_context(self, task: Task) -> dict[str, Any]:
        return {
            "initial_program": task.payload["initial_program"],
            "task_name": f"{task.payload['family']}/{task.payload['task']}",
        }

    # -- answers ----------------------------------------------------------------

    def parse_answer(self, text: str) -> Any:
        return parse_program(text)

    def realize(self, task: Task, candidate: dict[str, Any]) -> Any:
        return _candidate_program(candidate)

    # -- scoring ----------------------------------------------------------------

    def best_found(self, task_id: str) -> dict[str, Any] | None:
        measured = [
            r for r in (self._measured.get(task_id) or {}).values()
            if not r.get("error") and not _exploit(r)
        ]
        return max(measured, key=lambda r: r["combined_score"], default=None)

    async def _initial_score(self, task: Task, evaluator: Any) -> dict[str, Any]:
        """The starting program's own measurement, taken once per task.

        Only a measurement that ran is cached. A seed evaluation that timed out
        is not a score of zero -- it is no score -- and caching it would pin
        every later scoring of the task to the failure. Seen on llm_sql, where
        three concurrent runs pushed a two-minute evaluation past its 360 s
        limit and the task was recorded as beating a seed worth 0.0.
        """
        cached = self._initial.get(task.id)
        if cached is not None:
            return cached
        result = await evaluator.evaluate(task.payload["initial_program"], mode="test")
        if not result["error"]:
            self._initial[task.id] = result
        return result

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        raw = pred.answer if isinstance(pred.answer, str) else str(pred.answer or "")
        program = parse_program(raw)
        evaluator = self._evaluator(task)
        initial = await self._initial_score(task, evaluator)
        # A seed that could not be measured has no score to compare against,
        # so every seed-relative metric is left out rather than computed
        # against 0.0 -- which would count any valid program as an improvement.
        seed: dict[str, float] = (
            {} if initial["error"] else {"initial_score": float(initial["combined_score"])}
        )

        measured = float(len(self._measured.get(task.id) or {}))
        if not program:
            return Score(
                task_id=task.id,
                score=0.0,
                metrics={"valid": 0.0, **seed, "programs_evaluated": measured},
                detail={"reason": "no program in the answer", "initial_error": initial["error"]},
            )

        submitted = await evaluator.evaluate(program, mode="test")
        submitted_score = float(submitted["combined_score"])

        best = self.best_found(task.id)
        # The same completeness rule the probe applies, on the program
        # submitted. Scoring never writes into the measured set.
        verdict = _reject(None, submitted)
        submitted_exploit = verdict is not None
        effective = 0.0 if submitted_exploit else submitted_score

        if self.score_on != "submitted" and best is not None and best["combined_score"] > effective:
            scored, source = best, "best_found"
        elif submitted_exploit:
            scored, source = {**submitted, "combined_score": 0.0, "metrics": {}}, "submitted_exploit"
        else:
            scored, source = submitted, "submitted"
        s = float(scored["combined_score"])

        metrics: dict[str, float] = {
            "valid": float(not submitted["error"]),
            "combined_score": s,
            "submitted_score": submitted_score,
            "finalize_loss": max(0.0, s - effective),
            "programs_evaluated": measured,
        }
        exploits = [r for r in (self._measured.get(task.id) or {}).values() if _exploit(r)]
        metrics["exploits_rejected"] = float(len(exploits))
        metrics["submitted_exploit"] = float(submitted_exploit)
        if seed:
            metrics["initial_score"] = seed["initial_score"]
            metrics["gain"] = s - seed["initial_score"]
            metrics["beats_initial"] = float(s > seed["initial_score"])
        for k, v in scored["metrics"].items():
            if k != "combined_score":
                metrics[f"task_{k}"] = float(v)

        return Score(
            task_id=task.id,
            score=s,
            metrics=metrics,
            detail={
                "scored_on": source,
                "submitted_error": submitted["error"],
                "submitted_artifacts": {k: v[-2000:] for k, v in submitted["artifacts"].items()},
                "best_found_digest": best["digest"] if best else None,
                "submitted_digest": _digest(program),
                "initial_error": initial["error"],
                "submitted_verdict": verdict,
                "exploits": [
                    {"digest": r["digest"], "combined_score": r["combined_score"],
                     "mechanism": r["rejected"]["mechanism"], "rule": r["rejected"]["rule"]}
                    for r in exploits
                ],
            },
        )

    # -- readiness ----------------------------------------------------------------

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        if not (self.root / "tasks.jsonl").exists():
            return [f"{self.root}/tasks.jsonl missing: run scripts/prepare_skydiscover.py --download"]
        try:
            rows = self._rows()
        except Exception as e:  # noqa: BLE001
            return [f"cannot read tasks.jsonl: {e}"]
        missing = [r["id"] for r in rows if not (self._dir(r) / r["initial_program"]).exists()]
        if missing:
            problems.append(f"task directories missing for: {', '.join(missing[:5])}")
        if self.runtime == "docker" and any((self._dir(r) / r["evaluator"]).is_dir() for r in rows):
            if shutil.which("docker") is None:
                problems.append("docker not found; set benchmark_opts.runtime: host or install it")
        return problems

    def describe(self) -> str:
        problems = self.check()
        if problems:
            return f"{self.name:<18} {self.blurb}  [{problems[0].splitlines()[0]}]"
        rows = self._rows()
        families = sorted({r["family"] for r in rows})
        return f"{self.name:<18} {self.blurb}  [ready, {len(rows)} tasks: {', '.join(families)}]"
