<div align="center">

<img src="assets/logo.svg" alt="SHELDON" width="96">

# SHELDON

### Reasoning through Dual Search over Hypothesis and Observation Spaces

[Harshit Joshi](https://harshitjoshi.com)\* &nbsp;·&nbsp; Priyank Shethia\* &nbsp;·&nbsp; [Monica S. Lam](https://suif.stanford.edu/~lam)

Stanford University &nbsp;·&nbsp; \*equal contribution

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2610.18025-b31b1b.svg)](https://arxiv.org/abs/2610.18025)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://sheldon.genie.stanford.edu/sheldon.pdf)
[![Project page](https://img.shields.io/badge/Project-page-8A2BE2.svg)](https://sheldon.genie.stanford.edu)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](pyproject.toml)

<br>

<img src="https://sheldon.genie.stanford.edu/intro_fig.png" alt="SHELDON searches a hypothesis space and an observation space jointly" width="100%">

<sub><i>(a) SHELDON searches a hypothesis space and an observation space jointly; edges in the hypothesis space are refinements. (b) New hypotheses drive improvements on a system-optimization task. (c) Scores on DiagnosisArena, ORCA-Bench, ARC-AGI-2, and QNN circuit topology.</i></sub>

</div>

<br>

## Abstract

Test-time scaling has enabled large language models to solve increasingly
complex reasoning, discovery, and optimization tasks, but existing approaches
often suffer from fixed search strategies, or hypothesis-centric reasoning that
under-explores informative evidence.

Inspired by the hypothetico-deductive method and dual-space models of scientific
discovery, we introduce SHELDON, a scientific reasoning harness that jointly
models hypothesis and observation spaces. By treating observations as
first-class objects, SHELDON can identify gaps in its current understanding,
uncover new hypotheses, and design investigations that better distinguish among
alternatives. Across diagnosis, abstract reasoning, and optimization benchmarks,
SHELDON outperforms evaluated baselines by up to 26.7 points. On system root
cause analysis tasks, SHELDON discovers 29.8 percentage points more ground-truth
causes and investigates 34.4 percentage points more of the discovered causes
than other baselines, demonstrating that joint hypothesis-observation search
leads to broader and more effective scientific reasoning at test time.

<br>

## Results

All methods use GPT-5.6-terra with high reasoning effort; SHELDON delegates
code implementation to GPT-5.6-luna. Best score per column in **bold**.

### Diagnosis, root-cause analysis, abstract reasoning, circuit discovery

| Method | Family | DiagnosisArena | ORCA-Bench | ARC-AGI-2 | QNN topology |
|:--|:--|--:|--:|--:|--:|
| ReAct | Reasoning agent | 72.00 | 40.00 | 25.00 | 0.80 |
| Codex | Coding agent | 76.00 | 40.00 | 24.44 | 0.82 |
| EvoX | Meta-evolution | — | — | 47.50 | 0.85 |
| RLM | Recursive agent | 78.00 | 40.00 | 15.00 | 0.48 |
| GoS | Reasoning agent | 68.00 | 30.00 | — | — |
| **SHELDON** | Scientific reasoning | **84.00** | **66.67** | **59.17** | **1.00** |

### Molecular optimization (DOCKSTRING)

Best molecule each method docked within 1,000 docking calls, per protein
target. *Library best* is the top of the 260k-compound DOCKSTRING database.

| Method | ADRB1 | CDK2 | PPARA | PPARG | F2 | PGR |
|:--|--:|--:|--:|--:|--:|--:|
| Codex | 5.97 | 6.92 | 7.46 | 6.38 | 9.31 | 7.40 |
| ReAct | 8.49 | 8.32 | 8.98 | 8.40 | 7.79 | 7.67 |
| EvoX | 8.63 | 7.19 | 8.62 | 7.41 | 8.36 | 9.17 |
| CORAL | 11.01 | 9.32 | 9.64 | **9.69** | 9.10 | **10.65** |
| **SHELDON** | **11.26** | **10.26** | **9.85** | 9.47 | **9.85** | 9.99 |
| *Library best* | *11.33* | *9.53* | *9.91* | *9.53* | *9.31* | *9.74* |

### System optimization (ADRS)

Mean / best over three runs per method on the same hardware. Higher is better
except Cloudcast, where the score is a cost. *Human best* is the task's
published reference solution.

| Method | EPLB | PRISM | Cloudcast ↓ | Txn scheduling | LLM-SQL |
|:--|--:|--:|--:|--:|--:|
| Codex | **0.1441** / **0.1441** | 25.60 / 25.65 | 677.10 / 618.10 | 3069 / 3558 | 0.6877 / 0.7109 |
| ReAct | 0.1440 / 0.1440 | 26.24 / **26.26** | 621.11 / 618.10 | 3932 / 4000 | 0.7001 / 0.7069 |
| EvoX | 0.1318 / 0.1433 | **26.26** / **26.26** | 618.10 / 618.08 | 4009 / 4184 | 0.7149 / 0.7196 |
| ScientistOne | 0.1319 / 0.1440 | **26.26** / **26.26** | 624.10 / 618.10 | 2533 / 3962 | 0.7034 / 0.7137 |
| Arbor | 0.1433 / 0.1435 | 26.25 / **26.26** | 618.80 / 618.10 | 3668 / 3759 | 0.7130 / 0.7218 |
| CORAL | 0.1436 / 0.1436 | **26.26** / **26.26** | 619.00 / 619.00 | 4077 / 4201 | 0.7115 / 0.7163 |
| **SHELDON** | **0.1441** / **0.1441** | **26.26** / **26.26** | **618.09** / **618.07** | **4198** / **4274** | **0.7204** / **0.7245** |
| *Human best* | *— / 0.1265* | *— / 21.89* | *— / 626.24* | *— / 2724.8* | *— / 0.6920* |

### Ablations

| Variant | DiagnosisArena | ORCA-Bench | ARC-AGI-2 | QNN topology |
|:--|--:|--:|--:|--:|
| **SHELDON (full)** | **84.00** | **66.67** | **59.17** | **1.00** |
| without comparative analysis | 78.00 | 56.67 | 50.00 | 0.883 |
| without observation space | 70.00 | 53.33 | 50.00 | 0.867 |
| without continued exploration | 82.00 | 53.33 | 57.70 | 0.733 |

Removing the observation space costs the most, and most consistently. Full
tables, per-task costs, and the coverage analysis are in the
[paper](https://sheldon.genie.stanford.edu/sheldon.pdf).

<br>

## Getting started

### Install

```bash
git clone https://github.com/stanford-oval/sheldon.git
cd sheldon
make setup            # uv venv + editable install
cp .env.example .env  # fill in one provider
```

`.env` takes either an OpenAI-compatible endpoint (`OPENAI_BASE_URL`,
`OPENAI_API_KEY`) or `ANTHROPIC_API_KEY`. Model names are whatever your
endpoint serves; set `model:` in `configs/default.yaml` or pass `--model`.

Check the install without an API key:

```bash
make smoke            # the full pipeline, offline, against a scripted model
sheldon ls            # registered benchmarks, methods and engines
sheldon check         # which benchmark data is present
```

### Run

The method is registered as `sheldon`. A first run on ARC-AGI-2:

```bash
scripts/fetch_arc.sh
sheldon run -b arc-agi-2 -m sheldon -n 10 --tag pilot
sheldon report
```

### Configs

A run is described by a YAML config. `configs/default.yaml` is the baseline;
an experiment config extends it and overrides what it changes, and
`configs/experiments/` holds the ones behind the tables above:

```bash
sheldon run -c configs/experiments/087_arc_agi_2_sheldon_session.yaml
sheldon run -c configs/experiments/087_dockstring_sheldon_terra.yaml
sheldon run -c configs/experiments/082_skydiscover_llm_sql_sheldon_terra.yaml
```

A config has four parts. Everything in the first three is hashed into the run
id, so a changed setting is a different run; the execution block is not.

```yaml
extends: ../default.yaml

# what: the cell
benchmark: arc-agi-2
split: eval            # or a benchmark-specific split, listed in its module's `splits`
limit: 20              # tasks per cell; null = the whole split
seed: 0                # subsample seed, so every method sees the same tasks
method: sheldon        # sheldon | direct | best-of-n

# how: the model and the engine
engine: toolloop       # toolloop | simple | react | codex | rlm
provider: auto         # auto | openai | anthropic
model: gpt-5.6-terra
effort: high           # low | medium | high | xhigh | max
max_tokens: 16000
thinking: true

# how much: per-task budget; the loop stops at whichever binds first
max_rounds: 4          # null lifts a cap
max_calls: 40
max_usd: 1.0
max_seconds: null

# execution: changes speed, never the numbers
concurrency: 8         # tasks in flight
cache: true            # model calls cached on disk
resume: true           # finished tasks are skipped on re-run
tag: exp087
notes: one line for the report

params:                # method knobs; see the table below
  mode: session
  rounds: 6
```

Every field can be set from the command line, which wins over the file:
`--model`, `--effort`, `--engine`, `--rounds`, `--max-usd`, `--max-calls`,
`--task <id,id>`, `--tag`, `--no-cache`. A method parameter is `-p key=value`
with a JSON value, nested with dots:

```bash
sheldon run -c configs/experiments/087_arc_agi_2_sheldon_session.yaml \
  --model gpt-5.6-luna --max-usd 4 \
  -p mode=staged -p select_k=3 -p benchmark_opts.isolate=true
sheldon run -c ... --dry-run     # print the resolved config and exit
```

### Hyperparameters

The method's knobs live under `params:`. Defaults are the code's; the ones
the experiment configs change most are at the top.

| Parameter | Default | What it does |
|:--|:--|:--|
| `mode` | `staged` | `staged` asks each stage in a fresh call over freshly rendered state; `session` runs a round's deciding stages as turns of one conversation (needs a native tool-loop engine) |
| `rounds` | `6` | rounds per task, within `max_rounds` |
| `n_initial` | `5` | hypotheses explore opens with in round 1 |
| `n_explore` | `3` | hypotheses explore adds in each later round |
| `select_k` | `5` | hypotheses tested per round; kept ones take no slot |
| `max_regions` | `10` | regions per space (hypothesis map and observation map each); when a map is full, a merge call before explore asks the model which regions to fold together, and the merged region keeps its history. `0` lifts the cap |
| `set_aside_cooldown` | `3` | rounds a shelved hypothesis waits before explore may revive it |
| `stagnation_rounds` | `2` | rounds without a new best score or observation before the whole shelf unlocks |
| `plan` | `true` | one call before round 1 separates the problem from the specification of the answer |
| `candidate_analysis_max_evals` | `3` | evaluator runs per hypothesis per round, on benchmarks with a scoring function |
| `candidate_analysis_patience` | `2` | measured iterations in a row that fail to improve before the local loop stops |
| `candidate_analysis_min_gain_pct` | `0.5` | the improvement, in percent of the seed's score, that counts as progress |
| `candidate_analysis_concurrent` | `true` | run a round's local loops at once |
| `candidate_analysis_engine` | the run's engine | the engine that builds and tests artifacts, with its own `candidate_analysis_engine_params` |
| `stage_probes` | `true` | let the deciding stages search before deciding; `<stage>_probes` overrides one |
| `<stage>_probe_calls` | explore `8`, select `2`, candidate_analysis `12`, comparative_analysis `10`, exploit `4`, finalize `6` | tool-call ceiling for one call of that stage |
| `probes` | `[web_search, web_fetch, run_shell]` | the generic tools every stage can reach, beside the benchmark's own; `[]` leaves only the benchmark's |
| `benchmark_opts` | `{}` | per-benchmark switches, such as `isolate`, `max_evaluations`, `max_docking_calls` |
| `offer_tools` | per benchmark | whether an engine that brings its own tools is also offered ours |

`direct` and `best-of-n` take `n` (default `5`) and `test_rule`, and nothing
else.

### Engines

An engine is how one "ask the model, let it act" step runs. The benchmark
decides which tools exist and the method decides the strategy; the engine is
the piece you swap to change what executes a step, without touching either.
`sheldon ls` lists them.

| Engine | What it is | When to use it |
|:--|:--|:--|
| `toolloop` | the provider's native tool-calling loop (default) | any endpoint with tool support; every call is cached and budgeted |
| `simple` | one completion, tools ignored | the no-tools ablation, or an endpoint with no tool support |
| `react` | Thought / Action / Observation as text, parsed by the harness | a proxy that mangles tool schemas, or a paper's ReAct baseline |
| `codex` | the Codex CLI as the step, with our probes served to it over MCP | a coding agent in the loop; needs `codex` on `PATH` |
| `rlm` | a Recursive Language Model: a REPL the model programs, with sub-calls on its own context | the RLM baseline; needs the `rlm` package in the venv |

Pick one with `engine:` in the config or `--engine` on the command line, and
tune it under `params.engine_params`:

```yaml
engine: codex
params:
  engine_params:
    mode: research        # research: workspace-write, network, web search; reason: read-only
    reasoning: high       # defaults to the run's `effort`
    timeout: 2400         # seconds per codex call
    isolate: true         # run under bubblewrap, like the benchmark's own shell
    cwd: scratch/codex-work
    usd_watchdog: true    # kill a call at the remaining budget; codex is told the limit
    serve_probes: true    # false: codex with nothing but its own tools
```

```yaml
engine: rlm
params:
  engine_params:
    max_iterations: 8
    max_depth: 1
    timeout: 1500
    isolate: true
    rates: [2.50, 15.00]  # $/M tokens in, out: the endpoint returns no price
```

`toolloop` and `react` take `max_steps` (default `4`), the tool calls per
step before the model is asked to answer; `toolloop` also takes `model`, which
puts that engine on a model of its own.

**Two engines per run.** The deciding stages and the stage that builds and
tests artifacts can run on different engines. This is how the results above
were produced: the stages that explore, select and revise run on `toolloop`
with the run's model, and the local build-and-test loop runs on a second
engine or a cheaper model.

```yaml
engine: toolloop
model: gpt-5.6-terra
params:
  candidate_analysis_engine: codex            # or toolloop with another model
  candidate_analysis_engine_params:
    reasoning: high
    timeout: 2400
    isolate: true
```

```yaml
params:
  candidate_analysis_engine: toolloop
  candidate_analysis_engine_params:
    model: gpt-5.6-luna
```

**Comparing engines honestly.** `codex` and `rlm` bring their own shell and
file access on top of whatever the harness offers, and their model calls do
not pass through the harness's cache, so a run on either is not comparable to
one on `toolloop` and gets its own row. `probes: []` with the benchmark's
`isolate` is the usual setting for both.

### Benchmarks

| Name | Task | Get the data |
|:--|:--|:--|
| `med-dx` | DiagnosisArena: diagnose a patient by asking for history, examination, labs and imaging | `scripts/prepare_med_dx.py` |
| `orca-bench` | ORCA-Bench: find the root cause of a production incident | `scripts/prepare_orca_bench.py` |
| `arc-agi-1`, `arc-agi-2` | induce a grid rule from worked examples | `scripts/fetch_arc.sh` |
| `skydiscover` | rewrite a program so an evaluator scores it higher: the ADRS systems tasks and QNN circuit topology | `scripts/prepare_skydiscover.py --download --datasets` |
| `dockstring` | design a molecule that binds a protein target and stays drug-like | `uv pip install ".[dockstring]"` then `scripts/prepare_dockstring.py --download` |

Data lives under `data/<benchmark>/`, or wherever `SHELDON_DATA` points.

### Read a run

Each run writes `runs/<benchmark>__<method>__<split>__<tag>__<hash>/` with its
config, predictions, scores and a full trace per task.

```bash
sheldon report                   # the table: scores, then what they cost
sheldon failures <run_id>        # the worst tasks, with trace paths
sheldon show <run_id> <task_id>  # one trace, rendered top to bottom
sheldon ui --open                # every run and trace, with the hypothesis state per task
sheldon log --serve              # every model call, prompt and reply in full
```

Model calls are cached on disk and runs resume, so re-running a config costs
nothing for work already done.

<details>
<summary><b>Repository layout</b></summary>
<br>

```
sheldon/
  core/           types, config, budget, LLM + cache, trace, sandbox, isolation
  benchmarks/     one module per benchmark; base.py is the contract
  methods/        sheldon/ (the method and its prompts), stage.py
  engines/        how one model step runs: toolloop, simple, react, codex, rlm
  runner.py       async execution, resume, artifacts
  ui/             the trace console and the model-call log
  cli.py          ls / check / smoke / run / report / show / ui / log / failures
configs/          default.yaml and experiments/
scripts/          fetch and prepare benchmark data, re-score finished runs
```

</details>

> **Caution.** The Python sandbox and shell execute model-written code in a
> subprocess with a timeout. That is isolation, not a security boundary. Set
> `benchmark_opts.isolate: true` to run commands under bubblewrap where a
> benchmark supports it.

<br>

## Citation

```bibtex
@article{joshi2026sheldon,
  title   = {{SHELDON}: Reasoning through Dual Search over Hypothesis and Observation Spaces},
  author  = {Joshi, Harshit and Shethia, Priyank and Lam, Monica S.},
  journal = {arXiv preprint arXiv:2610.18025},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.18025}
}
```

Released under the [MIT License](LICENSE).
