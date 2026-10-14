"""ORCA-bench — root cause analysis on a production-fidelity oncall system.

    "ORCA-bench: How Ready Are Language Model Agents for Oncall?",
    arXiv:2607.28545. Dataset CC BY 4.0,
    hub.harborframework.com/datasets/orca-bench/orca-bench

An agent gets an ambiguous user-facing incident report and has to find the root
cause by digging through six days of metrics, logs, traces, and source code from
an OpenTelemetry-instrumented microservice system. 1,079 tasks (755 public),
graded against ground-truth symptoms signed off by SREs.

RCA is hypothesis work with the falsification step already built into the job:
you suspect a service, you go find the trace that would exonerate it, you move
on. That maps onto the loop more directly than anything else here — each
telemetry store is a probe, and "query the thing that would prove me wrong" is
exactly the falsifier's move.

Upstream ships telemetry as a static snapshot rather than a live cluster, so
this adapter reads a *dump directory* and needs no running services. Populate it
with `scripts/prepare_orca_bench.py`; `check()` tells you what is missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from sheldon.benchmarks.base import Benchmark
from sheldon.core import isolation
from sheldon.core.io import REPO_ROOT, data_root, read_jsonl
from sheldon.core.isolation import Sandbox
from sheldon.core.probes import Probe, ShellProbe
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task, Usage

#: None means "the run's model"; set benchmark_opts.judge_model to pin one.
#: Used by the local judge only.
JUDGE_MODEL: str | None = None

#: Upstream's verifier defaults: `check_prediction.py --model gpt-5.4 --effort
#: high` is baked into every published task, so the same judge grades every
#: submission. `benchmark_opts.judge_model` / `judge_effort` override.
UPSTREAM_JUDGE_MODEL = "gpt-5.4"
UPSTREAM_JUDGE_EFFORT = "high"

#: Telemetry stores in the dump, and the subdirectory each lives in.
STORES = {
    "metrics": "metrics",
    "logs": "logs",
    "traces": "traces",
    "source": "source",
}

#: One plantable hypothesis per fault flag in upstream's registry
#: (`flagd/demo.flagd.json`), as (service, surface, statement). Used only when
#: `benchmark_opts.seed_hypothesis` is set — see `seed_hypothesis`. The service
#: is what keeps an "incorrect" seed honestly incorrect: a wrong flag on the
#: right service is a near-miss, not a wrong hypothesis, so `_incorrect_flag`
#: never picks one whose service hosts any fault active in the scenario. The
#: surface names which user-visible symptom the fault produces, and is what
#: makes a `far` seed far: `_far_flag` requires it disjoint from every active
#: fault's surface, so the seed cannot even account for the reported symptom.
SEED_HYPOTHESES: dict[str, tuple[str, str, str]] = {
    "adFailure": (
        "ad",
        "ads",
        "the ad service is dropping requests — the `adFailure` fault flag is on, so a "
        "slice of `getAds` calls error out and the storefront renders without ads",
    ),
    "adHighCpu": (
        "ad",
        "latency",
        "the ad service is CPU-bound — the `adHighCpu` fault flag is on, spinning up "
        "busy-loop workers that saturate the ad JVM and slow every call through it",
    ),
    "adManualGc": (
        "ad",
        "latency",
        "the ad service is stalling on garbage collection — the `adManualGc` fault flag "
        "is on, forcing full manual GCs that pause the ad JVM",
    ),
    "cartFailure": (
        "cart",
        "checkout",
        "the cart service is failing — the `cartFailure` fault flag is on, so cart "
        "operations error and checkout breaks",
    ),
    "emailMemoryLeak": (
        "email",
        "email",
        "the email service is leaking memory — the `emailMemoryLeak` fault flag is on, "
        "so order-confirmation sends degrade as the process swells",
    ),
    "failedReadinessProbe": (
        "cart",
        "checkout",
        "the cart service is flapping out of rotation — the `failedReadinessProbe` "
        "fault flag is on, so its readiness probe fails and traffic to it cuts out",
    ),
    "imageSlowLoad": (
        "frontend",
        "latency",
        "the frontend is serving images slowly — the `imageSlowLoad` fault flag is on, "
        "delaying product images and dragging page loads with them",
    ),
    "kafkaQueueProblems": (
        "kafka",
        "order-pipeline",
        "the Kafka queue is backed up — the `kafkaQueueProblems` fault flag is on, "
        "flooding the queue while delaying the consumer, so a lag spike builds",
    ),
    "llmInaccurateResponse": (
        "llm",
        "product-content",
        "the product-summary LLM is answering wrongly — the `llmInaccurateResponse` "
        "fault flag is on, returning an inaccurate summary for one product",
    ),
    "llmRateLimitError": (
        "llm",
        "product-content",
        "the product-summary LLM is being throttled — the `llmRateLimitError` fault "
        "flag is on, returning intermittent rate-limit errors",
    ),
    "loadGeneratorFloodHomepage": (
        "load-generator",
        "latency",
        "the load generator is flooding the homepage — the `loadGeneratorFloodHomepage` "
        "fault flag is on, hammering the frontend with synthetic requests",
    ),
    "paymentFailure": (
        "payment",
        "checkout",
        "the payment service is failing charges — the `paymentFailure` fault flag is "
        "on, erroring a fraction of charge requests so checkouts fail",
    ),
    "paymentUnreachable": (
        "payment",
        "checkout",
        "the payment service is unreachable — the `paymentUnreachable` fault flag is "
        "on, so calls to it fail outright and checkout cannot complete",
    ),
    "productCatalogFailure": (
        "product-catalog",
        "browse",
        "the product catalog is failing on one product — the `productCatalogFailure` "
        "fault flag is on, erroring GetProduct for a specific product ID",
    ),
    "recommendationCacheFailure": (
        "recommendation",
        "recommendations",
        "the recommendation service's cache is broken — the `recommendationCacheFailure` "
        "fault flag is on, so recommendation calls miss their cache and degrade",
    ),
}


def _scenario(rubric_file: str) -> str:
    """`d1-c1-adHighCpu-on.json` -> `d1-c1`. The scenario is the unit — see
    `_rubrics_for`."""
    return "-".join(rubric_file.split("-")[:2])


def _scenario_flags(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Which faults are live in each scenario, per the whole artifact."""
    out: dict[str, set[str]] = {}
    for row in rows:
        flag = str(row.get("root_cause") or "")
        if flag:
            out.setdefault(_scenario(str(row.get("rubric_file") or "")), set()).add(flag)
    return out


def _incorrect_flag(task_id: str, active: set[str], scenario_flags: dict[str, set[str]]) -> str:
    """A fault that is plausible for this system and wrong for this task.

    Plausible: it is some other task's gold cause, so it demonstrably produces
    a real incident here. Wrong: it is not active in this task's scenario —
    scenario-mates are in the rubric set and naming one is partially *right* —
    and its service hosts no active fault, so the seed cannot degenerate into a
    near-miss that leads the agent to the right component for the wrong reason.

    The pick is a stable hash of the task id: the same task is seeded with the
    same wrong hypothesis in every run, so the arm reproduces and its cache hits.
    """
    banned_services = {SEED_HYPOTHESES[f][0] for f in active if f in SEED_HYPOTHESES}
    pool = sorted(
        flag
        for flag in set().union(*scenario_flags.values())
        if flag not in active
        and flag in SEED_HYPOTHESES
        and SEED_HYPOTHESES[flag][0] not in banned_services
    )
    if not pool:
        raise ValueError(
            f"{task_id}: no plausibly-incorrect seed exists — every other fault "
            "in the artifact shares a service with one active in this scenario"
        )
    return pool[int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % len(pool)]


def _far_flag(task_id: str, active: set[str]) -> str:
    """A fault that cannot even account for the reported symptom.

    Everything `_incorrect_flag` requires, plus surface disjointness: the seed's
    user-visible symptom must differ from that of *every* fault active in the
    scenario, so a "storefront is slow" incident can be seeded with a leaking
    email service but never with another latency fault. The pool is the whole
    flag registry rather than the artifact's gold causes — plausibility is not
    a requirement here, failing to explain the report is — which is also what
    keeps the pool non-empty in fault-heavy scenarios like `d6-big`.
    """
    banned_services = {SEED_HYPOTHESES[f][0] for f in active if f in SEED_HYPOTHESES}
    banned_surfaces = {SEED_HYPOTHESES[f][1] for f in active if f in SEED_HYPOTHESES}
    pool = sorted(
        flag
        for flag, (service, surface, _statement) in SEED_HYPOTHESES.items()
        if flag not in active and service not in banned_services and surface not in banned_surfaces
    )
    if not pool:
        raise ValueError(
            f"{task_id}: no far seed exists — every registry fault shares a service "
            "or a symptom surface with one active in this scenario"
        )
    return pool[int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % len(pool)]


def _concurrent_flag(task_id: str, gold: str, active: set[str]) -> str | None:
    """A scenario-mate: a fault that is genuinely ON, and not the answer.

    The one seed the telemetry *supports*: its flagd impressions, logs and
    metric shifts are all really there, because the fault is really running —
    it just does not cause the reported symptom. This is also codex's own
    dominant failure mode unseeded (on the control run it blamed the concurrent
    `cartFailure` hang for a payment incident, and `adFailure` for a load-
    generator flood), so the arm reproduces the confusion codex actually has,
    rather than one invented for it. Refuting it takes relevance reasoning —
    the fault exists but does not explain this symptom's path or timing — where
    `incorrect`/`far` fall to a single absence check.

    None when the scenario has no second fault: a single-fault scenario has no
    concurrently-active fault to hand over, and the task is simply not in this
    arm (see `load`). Note the scoring asymmetry: scenario-mates are in the
    rubric set, so a report anchoring on the mate is *partially credited* —
    read this arm through paired deltas and `gold_flag_mentioned`, not through
    the raw mean.
    """
    pool = sorted(active - {gold})
    if not pool:
        return None
    return pool[int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % len(pool)]


def seed_hypothesis(
    row: dict[str, Any],
    kind: str,
    scenario_flags: dict[str, set[str]],
    framing: str = "hedged",
) -> dict[str, str] | None:
    """The hypothesis `benchmark_opts.seed_hypothesis` plants in the prompt.

    `correct` is a **deliberate label leak** — the gold fault, handed to the
    method as a teammate's hypothesis. That is the point of the arm (how
    much of the score is *finding* the cause, versus grounding a report in
    telemetry once you have it), which is why the switch lives in
    `benchmark_opts`: it is in `config_hash`, each arm gets its own run
    directory, and it can never be a code default the hash cannot see. Seeded
    runs are compared only against each other, never in the headline table.

    `incorrect` is a plausible wrong fault (`_incorrect_flag`); `far` is a wrong
    fault that does not even match the reported symptom (`_far_flag`);
    `concurrent` is a fault that is really running and really evidenced, just
    not the cause of the reported symptom (`_concurrent_flag`; None for a task
    in a single-fault scenario, which this arm excludes). How hard the handoff
    pushes is a separate axis — `benchmark_opts.seed_framing`, `hedged` (an
    unverified hunch) or `assertive` (a signed-off diagnosis) — because
    distance and pressure are different mechanisms, and folding them into one
    switch would leave a result that cannot be attributed to either.

    Deterministic on the row alone, so `scripts/analyze_seeding.py` can
    re-derive the seed from the artifact instead of needing it stored.
    """
    gold = str(row.get("root_cause") or "")
    active = scenario_flags.get(_scenario(str(row.get("rubric_file") or "")), set())
    active = active | ({gold} if gold else set())
    task_id = str(row["id"])
    if kind == "correct":
        flag = gold
    elif kind == "far":
        flag = _far_flag(task_id, active)
    elif kind == "concurrent":
        maybe = _concurrent_flag(task_id, gold, active)
        if maybe is None:
            return None
        flag = maybe
    else:
        flag = _incorrect_flag(task_id, active, scenario_flags)
    _service, _surface, statement = SEED_HYPOTHESES.get(flag) or (
        "",
        "",
        f"the `{flag}` fault flag is on and is behind this incident",
    )
    return {"kind": kind, "flag": flag, "statement": statement, "framing": framing}


@benchmarks.register("orca-bench", "orca")
class OrcaBench(Benchmark):
    name = "orca-bench"
    splits = ("public",)
    blurb = "oncall root cause analysis over a telemetry snapshot (arXiv:2607.28545)"
    # Assigned twice until now -- `root_cause` above, `rca_report` below -- so the
    # first was dead. The report shape is the answer key here: asking for "the
    # root cause" turns the task into matching a description to a name, and every
    # fault's name and description sit in `flagd/demo.flagd.json`.
    answer_kind = "rca_report"
    answer_format = (
        "Give an incident RCA report with four sections -- Summary, Timeline, "
        "5 Whys, Remediation -- with every claim cited to telemetry you retrieved "
        "(the PromQL, the trace IDs, the Lucene query)."
    )

    def __init__(self, root: Any = None, **opts: Any) -> None:
        super().__init__(root, **opts)
        self.snapshot = self._path("snapshot") or self.root / "snapshot"

    # -- data -------------------------------------------------------------

    def load(self, split: str = "public", limit: int | None = None, seed: int = 0) -> list[Task]:
        path = self.root / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: {path} not found. Run scripts/prepare_orca_bench.py first."
            )
        rows = list(read_jsonl(path))
        kind = str(self.opts.get("seed_hypothesis") or "")
        if kind and kind not in ("correct", "incorrect", "far", "concurrent"):
            raise ValueError(
                f"{self.name}: seed_hypothesis must be 'correct', 'incorrect', 'far' "
                f"or 'concurrent', got {kind!r}"
            )
        framing = str(self.opts.get("seed_framing") or "hedged")
        if framing not in ("hedged", "assertive"):
            raise ValueError(
                f"{self.name}: seed_framing must be 'hedged' or 'assertive', got {framing!r}"
            )
        if not kind and "seed_framing" in self.opts:
            # A framing with nothing to frame silently measures the default.
            raise ValueError(f"{self.name}: seed_framing is set but seed_hypothesis is not")
        # From the whole artifact, before subsampling — a `limit` must not
        # shrink the exclusion set an incorrect seed is checked against.
        scenario_flags = _scenario_flags(rows) if kind else {}
        tasks = []
        for row in rows:
            seeded = seed_hypothesis(row, kind, scenario_flags, framing) if kind else None
            if kind and seeded is None:
                # A single-fault scenario has no concurrently-active fault to
                # hand over. The task is *out of the arm*, not silently
                # unseeded — one unseeded task inside a seeded run would move
                # the mean for a reason no column records.
                continue
            meta = _safe_meta(row)
            if seeded:
                # Already in the question, so recording it here adds no new
                # leak — and it is how `score()` and the analysis find it.
                meta["seed_hypothesis"] = seeded
            tasks.append(
                Task(
                    id=str(row["id"]),
                    benchmark=self.name,
                    question=(
                        self.upstream_instruction(row)
                        if str(self.opts.get("interface", "probes")) == "upstream"
                        else None
                    )
                    or self._render(row, seeded),
                    payload={
                        "report": row.get("report", ""),
                        "difficulty": row.get("difficulty", ""),
                    },
                    # Gold: the SRE-curated symptoms, root cause and mechanism.
                    # The mechanism belongs here and nowhere else — it names the
                    # flag, the file and the whole causal chain, so anywhere that
                    # survives `Task.blind()` is a label leak.
                    answer={
                        "root_cause": row.get("root_cause", ""),
                        "symptoms": row.get("symptoms") or [],
                        "mechanism": _mechanism(row),
                        # Which rubric grades this task. Gold, not meta: the
                        # filename names the flag.
                        "rubric_file": row.get("rubric_file", ""),
                    },
                    meta=meta,
                )
            )
        return self.subsample(tasks, limit, seed)

    @staticmethod
    def _render(row: dict[str, Any], seeded: dict[str, str] | None = None) -> str:
        """The task as upstream poses it: an incident report, and a demand for a
        report back.

        The output shape is load-bearing rather than cosmetic. Asking "name the
        root cause" is a question the flag registry in `flagd/demo.flagd.json`
        already answers — every fault's name and description are in the source,
        so the task collapses into matching a description to the report.
        Upstream hands agents that same file and is unaffected, because a
        Timeline needs real onset times and a 5 Whys chain needs queries that
        return real series. You cannot fabricate either from a list of names.

        A seeded run adds a handoff note right after the incident: the planted
        hypothesis, framed per `seed_framing`. `hedged` is a teammate's
        unverified hunch; `assertive` is a signed-off diagnosis from someone
        confident they found it. Both deliberately say nothing about what to do
        with it — instructing the agent to verify-or-discard would measure
        recovery under coaching, when the experiment asks what it does uncoached
        — and within an arm the framing text is identical for every task, so
        only the hypothesis varies.
        """
        reported = row.get("reported_time") or ""
        sections = [
            "You are an expert site reliability engineer.",
            "## Task\n\nThe following issue was reported"
            + (f" at {reported}" if reported else "")
            + f":\n\n{row.get('report', '')}",
            "The time an issue was reported is not necessarily when the incident "
            "began. Pinpoint the root cause events and when they occurred. There may "
            "be multiple root causes, or none.",
            "## Evidence\n\nThe telemetry is where the answer is: query the metrics, "
            "logs and traces, and read the service source where it helps. Every claim "
            "you make has to be grounded in something you actually retrieved.",
            "## Your report\n\nWrite an incident RCA report with four sections.",
            "**Summary** — 2-3 sentences: what happened, when, and the impact. You "
            "may name the cause in a phrase, but leave the causal chain to the 5 Whys.",
            "**Timeline** — one line per event, in order: the UTC time, the services "
            "involved, and what happened. Only what went wrong; quantify where you "
            "can. The first entry is the earliest anomaly visible in the telemetry.",
            "**5 Whys** — start from a precise observable symptom, then drill one "
            "chain. Each answer targets the previous one and must cite what you "
            "retrieved: the PromQL for a metric (with its before and after values and "
            "times), representative trace IDs for a cascade, the Lucene query for a "
            "log range. Stop at a cause someone could act on. Then state the root "
            "cause in a sentence, and note branches you did not follow.",
            "**Remediation** — for each action, which step in the chain it targets and "
            "whether it is a root-cause, propagation, or detection fix.",
        ]
        if seeded and seeded.get("framing") == "assertive":
            sections.insert(
                2,
                "## Handoff\n\nThe engineer who triaged this incident before you "
                "completed their investigation and signed off on the diagnosis:\n\n"
                f"> {seeded['statement']}\n\n"
                "They were confident this is the issue.",
            )
        elif seeded:
            sections.insert(
                2,
                "## Handoff\n\nThe engineer who triaged this incident before you "
                "left a note with their working hypothesis, which they had not "
                f"had time to verify:\n\n> {seeded['statement']}",
            )
        return "\n\n".join(sections)

    # -- tools ------------------------------------------------------------

    def report_path(self, task_id: str) -> Path:
        """Where the agent is told to write its report, under `interface: upstream`.

        Upstream's contract is a file: "write a structured incident RCA report
        to `/app/report.md`". Taking its instruction verbatim while still
        scoring the returned text grades a cover note: an agent that obeys
        the instruction writes the report and returns a short summary, and the
        summary is not the report.

        Under /tmp on purpose: codex's workspace-write sandbox permits writes
        only to its workdir, /tmp and $TMPDIR, and `/app` exists solely inside
        upstream's container. Per task, because tasks sharing one path would
        overwrite each other.
        """
        # An *output* path, so relative means the checkout -- not `SHELDON_DATA`,
        # which is where inputs live. Resolving it as an input silently sent
        # reports to `data/runs/...`, nothing was written where the scorer
        # looks, and `submitted_report` fell back to the returned text: the run
        # graded a different artifact than the upstream contract specifies, with
        # no error anywhere.
        raw = self.opts.get("report_dir")
        if raw:
            base = Path(str(raw)).expanduser()
            if not base.is_absolute():
                base = REPO_ROOT / base
        else:
            base = REPO_ROOT / "runs" / "orca-reports"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", task_id)[:120] or "task"
        return base / safe / "report.md"

    def upstream_instruction(self, row: dict[str, Any]) -> str | None:
        """Upstream's `instruction.md.template`, filled in, or None.

        Our own `_render` is a faithful paraphrase, but a paraphrase is still an
        edit: it rewords the citation rules and the section contracts that the
        rubric grades against. When the point is to match upstream's evaluation
        setting, the instruction has to be upstream's file, not our reading of
        it. Only two substitutions are made -- the template's hardcoded
        `/app/opentelemetry-demo` becomes wherever the source actually is here,
        and the placeholders are filled.
        """
        path = self._path("instruction_template")
        if not path or not Path(path).exists():
            return None
        text = Path(path).read_text(encoding="utf-8")
        reported = str(row.get("reported_time") or "")
        text = (
            text.replace("{current}", str(self.opts.get("current_time") or reported))
            .replace("{reported}", reported)
            .replace("{user_facing_issue}", str(row.get("report") or ""))
        )
        source = self._path("source_dir")
        if source:
            text = text.replace("/app/opentelemetry-demo", str(source))
        report = self.report_path(str(row.get("id") or row.get("task_id") or "task"))
        report.parent.mkdir(parents=True, exist_ok=True)
        # Stale file from a previous run would be read as this run's answer.
        report.unlink(missing_ok=True)
        text = text.replace("/app/report.md", str(report))
        return text

    def upstream_shell(self, task: Task) -> list[Probe]:
        """Upstream's interface: a shell, a Grafana URL, and the source tree.

        `instruction.md.template` names exactly one tool -- "Grafana HTTP API
        (available at $GRAFANA_URL ...)" -- and points at the source at
        `/app/opentelemetry-demo`. Nothing structured, no query helpers: working
        out the API is part of the task. Our `query_logs` / `query_metrics` /
        `query_traces` are this repo's invention, and on a benchmark where the
        evidence is entirely behind tools, inventing the interface means
        measuring a different problem than the one upstream graded.

        So in this mode the agent gets what upstream gives it and nothing else.
        """
        if not self.opts.get("grafana_url"):
            return []
        env = self._shell_env()
        # Upstream's agent reaches the backends by container hostname --
        # `prometheus:9090`, `jaeger:16686`, `opensearch:9200` -- because it runs
        # inside the compose network. A restore on a host exposes the same
        # services on mapped ports instead, and Grafana's own proxy only answers
        # for Prometheus here (its Jaeger datasource points at `/jaeger/ui`, a UI
        # path). Publishing the reachable URLs is not a new tool: it is the same
        # three services upstream offers, addressed the way this machine
        # addresses them. Without it the agent meets a dead route and reads the
        # silence as "no telemetry".
        return [
            ShellProbe(
                cwd=self._path("source_dir"),
                env=env,
                timeout=float(self.opts.get("shell_timeout", 120.0)),
                max_output=int(self.opts.get("shell_max_output", 40000)),
                sandbox=self.sandbox(task),
            )
        ]

    def _shell_env(self) -> dict[str, str]:
        """The environment the agent's shell is given: the Grafana URL and login,
        and the reachable backend URLs where configured."""
        env: dict[str, str] = {}
        if self.opts.get("grafana_url"):
            env["GRAFANA_URL"] = str(self.opts["grafana_url"])
            env["GRAFANA_USER"] = str(self.opts.get("grafana_user", "admin"))
            env["GRAFANA_PASSWORD"] = str(self.opts.get("grafana_password", "admin"))
        # Upstream's agent reaches the backends by container hostname --
        # `prometheus:9090`, `jaeger:16686`, `opensearch:9200` -- because it runs
        # inside the compose network. A restore on a host exposes the same
        # services on mapped ports instead. Publishing the reachable URLs is not a
        # new tool: it is the same three services upstream offers, addressed the
        # way this machine addresses them.
        for key in ("prometheus_url", "opensearch_url", "jaeger_url"):
            value = self.opts.get(key)
            if value:
                env[key.upper()] = str(value)
        return env

    def sandbox(self, task: Task) -> Sandbox | None:
        """What the agent may see under `benchmark_opts.isolate`: the demo
        source, read-only, and this task's own report directory, writable.

        Everything that grades the task sits beside the source on disk -- the
        manifest, `answers.json`, the rubrics, the fault schedule -- and so do
        earlier runs' reports and traces. Codex agents read all of those when
        they could. The runner hands this to every engine on the run context,
        because codex is offered no probes; the shell probe carries the same
        declaration. Asking for isolation without bubblewrap is an error.
        """
        if not self.opts.get("isolate"):
            return None
        if not isolation.available():
            raise RuntimeError(
                "benchmark_opts.isolate is set but bubblewrap (`bwrap`) is not installed; "
                "refusing to give the agent the host filesystem"
            )
        report_dir = self.report_path(task.id).parent
        report_dir.mkdir(parents=True, exist_ok=True)
        return Sandbox.of(
            read_only=[self._path("source_dir")], writable=[report_dir], env=self._shell_env()
        )

    def _path(self, opt: str) -> Path | None:
        """A configured path, resolved against the data root when relative.

        An absolute path in a config is reproducible on exactly one machine.
        A relative path resolves against `SHELDON_DATA` (default `data/`), so
        `orca-bench/upstream/check_prediction.py` means the same thing on any
        checkout that ran the prepare step.
        """
        raw = self.opts.get(opt)
        if not raw:
            return None
        path = Path(str(raw)).expanduser()
        return path if path.is_absolute() else (data_root() / path)

    def probes(self, task: Task) -> list[Probe]:
        """A shell, the source, and the endpoints. Nothing invented here.

        Upstream drops its agent in a container with the source tree on disk and
        a Grafana HTTP API behind an environment variable; `instruction.md.template`
        names exactly one tool, and working out the API is part of the task. Its
        own analysis confirms there is nothing else: `format_tool_calls.py`
        classifies `exec_command` and `bash_command` strings and knows no
        structured tool at all.

        This repo used to answer that with `query_metrics`, `query_logs` and
        `query_traces` -- typed wrappers taking a PromQL or Lucene string and
        returning parsed rows -- plus a `search_source` grep. They made this
        harness the author of the interface rather than upstream, and so measured
        an easier problem than the rubrics grade: no endpoint to discover, no API
        to work out, no pipeline to compose. Numbers produced through them were
        never comparable to a published ORCA result.

        What is left is what upstream offers.
        """
        shell = self.upstream_shell(task)
        if shell:
            return shell
        # Nothing configured to reach. A probe that greps the snapshot directory
        # is not a fallback: the metrics are Prometheus TSDB blocks and the logs
        # and traces are OpenSearch repository snapshots, readable only by the
        # services that wrote them.
        return []

    # -- scoring ----------------------------------------------------------

    async def _score_upstream(self, task: Task, pred: Prediction) -> Score | None:
        """Grade with upstream's own verifier, if pointed at it.

        `check_prediction.py` is a standalone script taking a report and a set
        of rubrics, so the real grading is reachable without Harbor. It scores
        0-3 per rubric — 2 means the report "identifies some of this rubric's
        ground-truth metrics, logs, or traces and moves toward this root cause"
        — and drops every rubric's score by one for a materially wrong claim.
        Naming several plausible causes is explicitly *not* such a claim, which
        is how it forecloses the flag-guessing our own judge rewards.

        Returns None when unconfigured, so this stays opt-in.
        """
        script = self._path("verifier")
        rubrics = self._path("rubrics_dir")
        if not script or not rubrics:
            return None

        import subprocess  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        report = self.submitted_report(task, pred)
        events = self._answer_events(task)
        with tempfile.TemporaryDirectory(prefix="sheldon-orca-") as tmp:
            d = Path(tmp)
            (d / "report.md").write_text(report, encoding="utf-8")
            # `tests/expected.json` as upstream's builder writes it: the same
            # events the rubric directory holds, and the query time.
            (d / "expected.json").write_text(
                json.dumps(
                    {
                        "events": [
                            {"root_cause": e["root_cause"], "event_time": e["event_time"]}
                            for e in events
                        ],
                        "current": (task.meta or {}).get("reported_time", ""),
                    }
                ),
                encoding="utf-8",
            )
            # One rubric per plausible event — see `_answer_events`.
            rub = d / "rubrics"
            rub.mkdir()
            for src in self._rubrics_for(Path(str(rubrics)), task):
                (rub / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

            # Upstream builds its own `AsyncOpenAI` from `OPENAI_BASE_URL`
            # verbatim. Ours points at `/v1/chat/completions`, which our client
            # strips and theirs does not — a 404 that looks like a missing model.
            env = {**os.environ}
            if env.get("OPENAI_BASE_URL"):
                from sheldon.core.llm import normalize_base_url  # noqa: PLC0415

                env["OPENAI_BASE_URL"] = normalize_base_url(env["OPENAI_BASE_URL"])
            proc = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    str(script),
                    "--expected",
                    str(d / "expected.json"),
                    "--predictions",
                    str(d / "report.md"),
                    "--rubrics-dir",
                    str(rub),
                    "--reward",
                    str(d / "reward.json"),
                    "--details",
                    str(d / "details.json"),
                    "--model",
                    str(self.opts.get("judge_model") or UPSTREAM_JUDGE_MODEL),
                    "--effort",
                    str(self.opts.get("judge_effort") or UPSTREAM_JUDGE_EFFORT),
                ],
                capture_output=True,
                text=True,
                # One judge call per rubric, serially, on whatever the endpoint
                # is doing at the time. 600s was enough alone and not beside
                # two other sweeps: the direct arm's first task of the codex
                # comparison hit it and was recorded as 0.0 with `score_error`,
                # a row resume never retries. `scripts/rescore_orca.py` is the
                # repair; this is the prevention.
                timeout=1800,
                check=False,
                env=env,
            )
            reward_file = d / "reward.json"
            if not reward_file.exists():
                return Score(
                    task_id=task.id,
                    score=0.0,
                    detail={
                        "verifier_failed": proc.stderr[-400:] or proc.stdout[-400:],
                    },
                )
            rewards = json.loads(reward_file.read_text())
            details = {}
            if (d / "details.json").exists():
                details = json.loads((d / "details.json").read_text())

        return Score(
            task_id=task.id,
            score=float(rewards.get("reward", 0.0)),
            metrics={k: float(v) for k, v in rewards.items() if isinstance(v, int | float)},
            detail={
                "grader": "upstream",
                "difficulty": task.payload.get("difficulty", ""),
                "nested": details.get("nested", details),
            },
        )

    def _answer_events(self, task: Task) -> list[dict[str, Any]]:
        """The plausible root causes this task is graded against, as upstream
        decides them: `[{"event_id", "root_cause", "event_time"}, ...]`.

        Upstream does not grade a task against the fault it was written for.
        Its `generate_answers.py` takes every flag switched on during the
        reported local day before the query time, whose rubric documents a
        user-facing symptom, and asks an LLM which of them could explain the
        complaint; the verifier averages over that set. `scripts/orca_answers.py`
        is that step ported, and its output -- `answers.json` under the data
        root, read here and nowhere else -- is the only place the set lives.
        It is not in the task row on purpose: `Task.blind()` keeps `meta`, and
        a list of plausible flags there is the answer key one attribute access
        from any method.

        Two earlier rules were both wrong. The task's own rubric alone charged
        a concurrently-active fault as fabrication on every task of a 22-task
        run; every rubric in the scenario graded faults that began *after* the
        report, and a run was credited for a fault upstream would never have
        asked about.
        """
        path = self._path("answers") or (data_root() / "orca-bench" / "answers.json")
        if not Path(path).is_file():
            raise RuntimeError(
                f"no answer file at {path}: run scripts/orca_answers.py to decide, the "
                "way upstream does, which rubrics each task is graded against"
            )
        answers = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = answers.get(task.id)
        if entry is None:
            raise RuntimeError(
                f"{task.id} has no entry in {path}: run scripts/orca_answers.py --task {task.id}"
            )
        return [
            {
                "event_id": str(e.get("event_id") or ""),
                "root_cause": str(e.get("root_cause") or ""),
                "event_time": str(e.get("event_time") or ""),
            }
            for e in entry.get("events") or []
        ]

    def _rubrics_for(self, root: Path, task: Task) -> list[Path]:
        """One rubric per plausible event -- `tests/rubrics/<event_id>.json`
        in an upstream task. An event without a rubric is skipped, as
        upstream's builder skips it."""
        return [
            root / f"{e['event_id']}.json"
            for e in self._answer_events(task)
            if e["event_id"] and (root / f"{e['event_id']}.json").is_file()
        ]

    def submitted_report(self, task: Task, pred: Prediction) -> str:
        """What the agent actually handed in.

        Under the upstream interface that is the *file*, not the reply: the
        instruction says to write one, and a well-behaved agent then returns
        only a summary of what it wrote. Falls back to the returned text when
        the file is missing, so a run that never wrote one is still scored on
        whatever it said rather than on silence.
        """
        answer = pred.answer if isinstance(pred.answer, str) else json.dumps(pred.answer or "")
        if str(self.opts.get("interface", "probes")) != "upstream":
            return answer
        path = self.report_path(task.id)
        try:
            written = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            written = ""
        return written or answer

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        upstream = await self._score_upstream(task, pred)
        if upstream is not None:
            return self._seeded(task, pred, upstream)

        gold = task.answer or {}
        proposed = self.submitted_report(task, pred)
        if not proposed.strip():
            return self._seeded(
                task, pred, Score(task_id=task.id, score=0.0, detail={"reason": "empty diagnosis"})
            )

        verdict, usage = await self._judge(ctx, task, proposed, gold)
        correct = float(bool(verdict.get("correct")))
        symptoms = gold.get("symptoms") or []
        matched = [i for i in (verdict.get("matched_symptoms") or []) if 0 <= i < len(symptoms)]
        return self._seeded(
            task,
            pred,
            Score(
                task_id=task.id,
                score=correct,
                metrics={
                    "rca_accuracy": correct,
                    "symptom_recall": len(matched) / len(symptoms) if symptoms else 0.0,
                },
                detail={
                    "difficulty": task.payload.get("difficulty", ""),
                    "reason": verdict.get("reason", ""),
                    "judge_usage": usage.to_dict(),
                },
            ),
        )

    def _seeded(self, task: Task, pred: Prediction, score: Score) -> Score:
        """Attach seed-echo metrics when the run was seeded (see `load`).

        `*_mentioned` is a substring check and nothing more: a report that names
        the seeded flag *to rule it out* still mentions it, exactly as
        `hallucinate_any` is "named a flag" rather than "fabricated". The
        quality claim stays with the rubric score; these two exist so recovery
        is readable per task without re-deriving the seed — and `detail` keeps
        the seed itself, so `scores.jsonl` says which hypothesis this task was
        handed and in which arm.
        """
        seeded = (task.meta or {}).get("seed_hypothesis")
        if not seeded:
            return score
        report = self.submitted_report(task, pred)
        low = report.lower()
        gold = str((task.answer or {}).get("root_cause") or "")
        score.metrics["seed_flag_mentioned"] = float(str(seeded["flag"]).lower() in low)
        if gold:
            score.metrics["gold_flag_mentioned"] = float(gold.lower() in low)
        score.detail["seed_hypothesis"] = dict(seeded)
        return score

    async def _judge(
        self, ctx: RunContext, task: Task, proposed: str, gold: dict[str, Any]
    ) -> tuple[dict[str, Any], Usage]:
        symptoms = gold.get("symptoms") or []
        symptom_block = "\n".join(f"[{i}] {s}" for i, s in enumerate(symptoms)) or "(none listed)"
        prompt = (
            "Grade a root cause analysis against the ground truth an SRE signed off on.\n\n"
            f"## Incident report\n{task.payload.get('report', '')}\n\n"
            f"## Ground-truth root cause\n{gold.get('root_cause', '')}\n\n"
            f"## Ground-truth symptoms\n{symptom_block}\n\n"
            f"## Proposed analysis\n{proposed}\n\n"
            "Mark it correct only if it identifies the same failing component *and* the same "
            "mechanism. Naming a downstream service that merely shows symptoms is not correct, "
            "however well argued. Then list which ground-truth symptoms it accounts for.\n\n"
            'Reply as JSON: {"correct": bool, "matched_symptoms": [int], "reason": "one sentence"}'
        )
        res = await ctx.llm.complete(
            prompt, model=self.opts.get("judge_model", JUDGE_MODEL), tag="judge"
        )
        match = re.search(r"\{.*\}", res.text, re.DOTALL)
        try:
            return json.loads(match.group(0)) if match else {}, res.usage
        except json.JSONDecodeError:
            return {"correct": False, "reason": "judge returned unparseable output"}, res.usage

    # -- readiness --------------------------------------------------------

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems + ["run scripts/prepare_orca_bench.py to populate it"]
        missing = [s for s in self.splits if not (self.root / f"{s}.jsonl").exists()]
        if missing:
            return [f"missing task file(s): {missing} (run scripts/prepare_orca_bench.py)"]
        if self.opts.get("prometheus_url") and self.opts.get("opensearch_url"):
            return []
        if not self.snapshot.exists():
            return [
                f"telemetry snapshot not found at {self.snapshot}",
                "without it the probes return nothing and a run measures guessing, not RCA",
            ]
        if not self.available_stores():
            return [
                f"snapshot at {self.snapshot} contains none of {sorted(STORES)}",
                "with no evidence at all a run measures guessing, not RCA",
            ]
        return []

    def available_stores(self) -> list[str]:
        """Which telemetry stores are actually on disk.

        A partial snapshot is a real situation, not a broken one: upstream ships
        metrics as Prometheus TSDB blocks and logs/traces as OpenSearch
        repository snapshots, both of which need their service running to query,
        while source code is plain files. Refusing to run on anything less than
        all four would mean never running at all. So the benchmark offers a probe
        per store it has, and the run records which — `sheldon ls` prints them, and
        the trace's `tools` event names the set the method was actually given.
        """
        return [n for n, d in sorted(STORES.items()) if (self.snapshot / d).exists()]


def _mechanism(row: dict[str, Any]) -> str:
    """The write-up of how the fault propagates, wherever it was written."""
    return str(row.get("mechanism") or (row.get("meta") or {}).get("mechanism") or "")


def _safe_meta(row: dict[str, Any]) -> dict[str, Any]:
    """`meta` as the method may see it.

    `Task.blind()` drops `answer` and copies `meta` verbatim, so anything gold
    left in `meta` reaches the method regardless. A local task set built from
    upstream's rubrics carried the mechanism here for provenance — which is the
    answer, in full, one attribute access away.
    """
    meta = dict(row.get("meta") or {})
    meta.pop("mechanism", None)
    # The reported time is in the prompt already; the probes need it too, to
    # window on what the agent knows rather than on the true onset.
    if row.get("reported_time"):
        meta["reported_time"] = row["reported_time"]
    return meta
