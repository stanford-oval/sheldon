"""Codex CLI as an engine — an external coding agent doing the reasoning step.

Written against the working invocation in `graph-search-sliders/src/codex_exec.py`,
which is the reference for the details that are easy to get wrong:

  - the prompt goes in on **stdin** (`-`), never as an argv element; a long
    prompt overflows the ~128KB argv limit and dies as "Argument list too long"
  - `--cd` must be **absolute**
  - `--output-last-message` is how you get the final text back; parsing the
    JSON event stream is guesswork by comparison
  - `--output-schema` constrains that final message to JSON, which is what makes
    the falsifier's `submit_verdict` work here at all
  - `web_search` is rejected at `model_reasoning_effort=minimal` (Azure 400), so
    a research-mode call floors the effort at `low`
  - `VIRTUAL_ENV` must be dropped, or Codex inherits this repo's venv

**Not verified end to end from here** — no Codex binary was available. `check()`
looks for it; treat the first real run as a debugging session.

Two structural caveats for comparisons:

  - **Codex brings its own tools, on top of ours.** `ctx.tools` is served to it
    over MCP (see `probe_server.py`), so it is offered the identical probe set
    every other method gets — the same objects, so a shared per-task budget stays
    shared. What no configuration removes is its own shell and file access, so
    the invariant holds in one direction only: codex is never offered *less*.
    Still tag it separately. `-p engine_params.serve_probes=false` is the control
    that measures codex with nothing but its own tools.
  - **Its calls never touch our cache.** Cost is recovered afterwards from the
    session rollout so the budget is still charged, but a cache hit is
    impossible and re-running is never free.
  - **`max_usd` cannot stop an exec from the outside** -- cost is read back
    when the exec returns, so a single call can overrun the task budget by
    however much it likes. With
    `engine_params.usd_watchdog: true`, `_spawn` polls the session rollout
    while codex runs and kills the process group at the remaining budget, the
    way graph-search-sliders' `usd_cap` does; codex is told the limit in the
    prompt so it can pace itself. Opt-in, because turning it on changes what a
    codex run produces and the arm has to say so.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from sheldon.core import isolation
from sheldon.core.io import REPO_ROOT
from sheldon.core.llm import ToolCall
from sheldon.core.registry import engines
from sheldon.core.types import RunContext, Usage
from sheldon.engines.base import Engine, EngineResult
from sheldon.engines.codex.probe_server import ProbeBridge

#: Read-only, no network. For pure reasoning steps.
MODE_REASON = "reason"
#: Workspace-write plus network and Codex's own web_search.
MODE_RESEARCH = "research"

_TOKENS_RE = re.compile(r"tokens\s+used\s*\n\s*([\d,]+)", re.IGNORECASE)

#: `_spawn` return code for "killed at the task's spend cap". Distinct from a
#: timeout (-1) because it is the harness's budget doing its job, not a hang.
BUDGET_EXIT = -2

#: Seconds a shim call waits on the probe bridge. Raise it with
#: `engine_params.probe_timeout` wherever a probe runs longer than this; a
#: probe that outlives it still runs and is still charged, and only the answer
#: is lost.
SHIM_TIMEOUT = 300.0

#: What `codex-loop`'s Stop hook reads from the workspace. Same name on both
#: sides by construction: the hook defines it too, and a rename here without
#: one there is a loop that silently runs on the hook's defaults.
LOOP_FILE = "sheldon_loop.json"


@engines.register("codex")
class CodexEngine(Engine):
    name = "codex"
    blurb = "external Codex CLI (brings its own tools)"
    brings_own_tools = True

    async def run(
        self,
        ctx: RunContext,
        prompt: str,
        *,
        system: str | None = None,
        extra_tools: list[dict[str, Any]] | None = None,
        stop_on: set[str] | None = None,
        max_steps: int | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> EngineResult:
        problems = self.check()
        if problems:
            ctx.trace.log("engine.unavailable", engine=self.name, problems=problems)
            return EngineResult(text="", stopped_by="unavailable")

        # When the caller wants a specific tool called (the falsifier's
        # submit_verdict), constrain the final message to that tool's schema.
        # Codex cannot call our probes, but it can be made to answer in shape.
        schema, schema_name = _terminal_schema(extra_tools, stop_on)

        # Serve `ctx.tools` over MCP so codex is offered the same probes as
        # every other method. Without it a codex run measures a different agent
        # *and* a different toolbox, and the two cannot be told apart.
        serve = bool(ctx.tools) and bool(self.param("serve_probes", True))
        bridge = (
            ProbeBridge(ctx.tools, asyncio.get_running_loop(), trace=ctx.trace,
                        timeout=float(self.param("probe_timeout", SHIM_TIMEOUT)))
            if serve else None
        )

        # `codex-loop` runs the belief loop inside this one exec: a Stop hook
        # keeps the turn going stage by stage, and the falsifier is a spawned
        # role. The method leaves everything the hook needs in `ctx.scratch`;
        # this engine only attaches it. A side channel, and deliberately the
        # one engine that already is one -- codex brings its own tools, so a
        # hook on its turn loop is a property of this agent, not a probe.
        loop = (getattr(ctx, "scratch", None) or {}).get("codex_loop")

        with tempfile.TemporaryDirectory(prefix="sheldon-codex-") as tmp:
            work = self._workspace(ctx, tmp, tag)
            task_dir = Path(tmp)
            prompt_file = task_dir / "prompt.txt"
            prompt_file.write_text(f"{system}\n\n{prompt}" if system else prompt, encoding="utf-8")
            out_file = task_dir / "out.txt"
            if loop:
                # The hook reads this from codex's cwd: the situation and the
                # deliverable already separated, and every knob, so nothing
                # about the loop is decided by the hook's own environment.
                (work / LOOP_FILE).write_text(
                    json.dumps({k: v for k, v in loop.items()
                                if k not in ("hook", "python", "role_file", "role_name")}),
                    encoding="utf-8",
                )
                ctx.trace.log(
                    "engine.loop", engine=self.name, hook=loop["hook"],
                    state_dir=loop["state_dir"], falsifier_mode=loop.get("falsifier_mode"),
                )
            mcp_url = bridge.start() if bridge is not None else None
            transport = str(self.param("probe_transport", "python"))
            # MCP is offered but codex will not call it without dropping its
            # sandbox (see `_argv`). The python transport reaches the identical
            # bridge through codex's *shell*, which is ungated in
            # workspace-write mode -- same objects, same budgets, sandbox kept.
            if mcp_url and transport in ("python", "both"):
                (work / "sheldon_probes.py").write_text(
                    self._shim_for(ctx.tools, mcp_url), encoding="utf-8"
                )
                prompt = f"{prompt}\n\n{_shim_prompt(ctx.tools)}"
                prompt_file.write_text(
                    f"{system}\n\n{prompt}" if system else prompt, encoding="utf-8"
                )
            usd_cap = self._usd_cap(ctx)
            if usd_cap is not None:
                prompt = f"{prompt}\n\n{_budget_prompt(usd_cap)}"
                prompt_file.write_text(
                    f"{system}\n\n{prompt}" if system else prompt, encoding="utf-8"
                )
            if mcp_url:
                ctx.trace.log(
                    "engine.probes",
                    engine=self.name,
                    url=mcp_url,
                    transport=transport,
                    offered=sorted(ctx.tools.probes),
                )
            argv = self._argv(
                task_dir,
                work,
                model,
                schema,
                out_file,
                mcp_url if transport in ("mcp", "both") else None,
                loop=loop,
            )
            sandbox = self._sandbox(ctx)
            if sandbox is not None:
                if loop:
                    raise RuntimeError(
                        "codex-loop runs a Stop hook from the repo, which an isolated "
                        "codex cannot see; run it without benchmark_opts.isolate"
                    )
                argv = self._isolate(argv, sandbox, work, task_dir, _task_workspace(ctx))
                ctx.trace.log(
                    "engine.isolated", engine=self.name,
                    read_only=list(sandbox.read_only), writable=list(sandbox.writable),
                    workspace=str(_task_workspace(ctx) or ""),
                )
            since = time.time() - 5.0

            try:
                code, stderr = await self._spawn(
                    argv, prompt_file, task_dir, work, usd_cap=usd_cap, since=since,
                    timeout=float(loop["timeout"]) if loop and loop.get("timeout") else None,
                    extra_env=(
                        {"SHELDON_CODEX_STATE_DIR": str(loop["state_dir"])} if loop else None
                    ),
                )
            finally:
                if bridge is not None:
                    bridge.stop()

            # Read inside the block. `out_file` lives in the temporary
            # directory, so reading after the dedent finds it already deleted —
            # which silently returned empty text from every codex run ever made,
            # exit code 0 and all.
            text = out_file.read_text(encoding="utf-8").strip() if out_file.exists() else ""
            usage = self._recover_usage(work, since, task_dir)
            self._keep_transcript(ctx, task_dir, tag)

        ctx.budget.charge(usage)

        if code == BUDGET_EXIT:
            # Not an error: the harness stopped it. Whatever it produced stands,
            # and on a benchmark scored from the environment the result is
            # untouched by the kill.
            ctx.trace.log(
                "engine.budget_stop",
                engine=self.name,
                usd=round(usage.cost_usd, 4),
                cap=round(usd_cap or 0.0, 4),
                why=stderr,
                probe_calls=len(bridge.calls) if bridge is not None else 0,
            )
            return EngineResult(text=text, usage=usage, stopped_by="budget:usd", iterations=1)

        if not text:
            # A paid call that produced nothing is a failure whatever the exit
            # code says, and saying so is the difference between a broken engine
            # and a method that looks like it declined to answer.
            ctx.trace.log(
                "engine.error",
                engine=self.name,
                code=code,
                usd=round(usage.cost_usd, 4),
                stderr=stderr[-600:],
                probe_calls=len(bridge.calls) if bridge is not None else 0,
            )
            return EngineResult(text="", usage=usage, stopped_by=f"exit{code}:no_output")

        calls: list[ToolCall] = []
        stopped_by = "end_turn"
        if schema_name and text:
            try:
                calls.append(ToolCall(name=schema_name, input=json.loads(text)))
                stopped_by = schema_name
            except json.JSONDecodeError:
                ctx.trace.log("engine.schema_miss", engine=self.name, chars=len(text))

        probe_calls = list(bridge.calls) if bridge is not None else []
        ctx.trace.log(
            "engine.codex",
            chars=len(text),
            usd=round(usage.cost_usd, 4),
            probe_calls=len(probe_calls),
            probes_used=sorted(set(probe_calls)),
        )
        return EngineResult(
            text=text, usage=usage, tool_calls=calls, stopped_by=stopped_by, iterations=1
        )

    # -- invocation --------------------------------------------------------

    def _workspace(self, ctx: RunContext, tmp: str, tag: str) -> Path:
        """A working directory of this call's own, even under a configured `cwd`.

        Concurrent tasks used to share the configured directory outright, and
        each wrote its own `sheldon_probes.py` there — same path, different
        bridge URL — so whichever landed second silently rerouted the other
        task's probe calls to its own bridge: retrievals recorded against the
        wrong task, and the wrong task's budget charged. A subdirectory per
        engine call keeps what `cwd` is for (disposable, inspectable after the
        run) and also makes `_recover_usage` exact, since rollouts are matched
        by cwd. Named after the task's trace so a workspace can be found from a
        trace; the uuid keeps two same-tag calls within one task apart.
        """
        base = self.param("cwd")
        if not base:
            return Path(tmp).resolve()
        # Relative to the checkout, not to the caller's shell: a config naming
        # `scratch/codex-work` has to mean the same directory however the run was
        # launched, and `../scratch/...` would put it outside the repo entirely.
        base = Path(base)
        if not base.is_absolute():
            base = REPO_ROOT / base
        stem = getattr(ctx.trace, "path", None)
        stem = stem.stem if stem is not None else "task"
        work = (Path(base) / f"{stem}--{tag or 'step'}--{uuid.uuid4().hex[:6]}").resolve()
        work.mkdir(parents=True, exist_ok=True)
        return work

    #: The run's effort names as codex spells them. Codex has no `max`; its
    #: top is `xhigh`.
    EFFORTS = {"minimal": "minimal", "low": "low", "medium": "medium",
               "high": "high", "xhigh": "xhigh", "max": "xhigh"}

    def _effort(self, mode: str) -> str:
        """The reasoning effort codex is told to use.

        `engine_params.reasoning` when set; otherwise the run's `effort`, which
        the runner passes in as `reasoning` so one setting governs the loop's
        own calls and the agent alike. The old default was a silent `low` while
        the loop ran at `high`, and nothing said so: `--effort` changed half a
        run. `minimal` is floored at `low` in research mode, where Azure rejects
        web_search at minimal.
        """
        raw = str(self.param("reasoning", "low") or "low").lower()
        if raw not in self.EFFORTS:
            raise ValueError(f"unknown reasoning effort {raw!r}; one of {sorted(self.EFFORTS)}")
        effort = self.EFFORTS[raw]
        if mode == MODE_RESEARCH and effort == "minimal":
            effort = "low"
        return effort

    def _argv(
        self,
        task_dir: Path,
        work: Path,
        model: str | None,
        schema: dict | None,
        out: Path,
        mcp_url: str | None = None,
        loop: dict[str, Any] | None = None,
    ) -> list[str]:
        binary = self.param("binary", "codex")
        mode = self.param("mode", MODE_RESEARCH)
        effort = self._effort(mode)

        argv = [binary, "exec", "--skip-git-repo-check", "--cd", str(work)]
        for override in self.param("provider_config", None) or _provider_config():
            argv += ["-c", override]
        if effort:
            argv += ["-c", f"model_reasoning_effort={effort}"]
        if mode == MODE_RESEARCH:
            argv += [
                "-c",
                "sandbox_workspace_write.network_access=true",
                "-c",
                "tools.web_search=true",
                "-c",
                "shell_environment_policy.inherit=all",
            ]
        # Codex `exec` denies every MCP tool call under all four approval
        # policies -- `never`, `on-request`, `untrusted`, `--full-auto` -- with
        # "user cancelled MCP tool call", and never sends the request. Verified
        # against a trivial stdio server too, so it is codex's gate rather than
        # anything about our bridge. The only setting that lets one through also
        # drops the filesystem sandbox, so serving probes to codex and sandboxing
        # codex are mutually exclusive in this version. Off by default: turning
        # it on is a decision about the machine the run is on, and `cwd` should
        # point somewhere disposable when it is.
        if self.param("bypass_sandbox", False):
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        elif mode == MODE_RESEARCH:
            argv += ["--sandbox", "workspace-write"]
        else:
            argv += ["--sandbox", "read-only"]

        if mcp_url:
            # The benchmark's probes, served live from this process. Without
            # this codex answers with its own shell against a toolbox nobody
            # else was offered, which is a different experiment. Note that codex
            # will not actually *call* them unless `bypass_sandbox` is set --
            # see above; the tools are offered and listed either way.
            name = self.param("mcp_name", "sheldon")
            argv += [
                "-c",
                f"mcp_servers.{name}.url={json.dumps(mcp_url)}",
                "-c",
                f"mcp_servers.{name}.startup_timeout_sec=30",
                "-c",
                # The same limit as the bridge and the shim: a fixed 300 s here
                # gave up on every evaluation that waited its turn behind
                # another (`eval_exclusive`) while the bridge was still waiting.
                f"mcp_servers.{name}.tool_timeout_sec={int(float(self.param('probe_timeout', SHIM_TIMEOUT)))}",
            ]

        if loop:
            # The Stop hook that runs the loop, and the role the falsify stage
            # may spawn. Session-flag config, so the user's ~/.codex is untouched
            # and the arm is reproducible from the run's config alone. TOML
            # strings, spelled with JSON quoting, which TOML accepts for these.
            command = f"{loop['python']} {loop['hook']}"
            argv += [
                "-c",
                "hooks.Stop=[{hooks=[{type=\"command\",command="
                f"{json.dumps(command)},timeout=180}}]}}]",
                "-c",
                f"agents.{loop['role_name']}.description="
                + json.dumps("Attacks exactly one hypothesis and reports a verdict."),
                "-c",
                f"agents.{loop['role_name']}.config_file={json.dumps(str(loop['role_file']))}",
            ]
        argv += ["-m", model or self.param("model", "gpt-5.6-terra"), "--color", "never"]
        if schema is not None:
            schema_file = task_dir / "schema.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            argv += ["--output-schema", str(schema_file)]
        # `-` reads the prompt from stdin. Never pass it as an argv element:
        # a long prompt blows past the ~128KB argv limit.
        argv += ["--output-last-message", str(out), "-"]
        return argv

    def _shim_for(self, probes: Any, url: str) -> str:
        """The workspace shim for this engine's probe timeout."""
        return _shim_source(url, probes, float(self.param("probe_timeout", SHIM_TIMEOUT)))

    def _sandbox(self, ctx: RunContext) -> isolation.Sandbox | None:
        """The sandbox this task declares, if any.

        Codex brings its own shell and is offered no probes, so the benchmark's
        isolation cannot reach it through a probe the way it reaches every other
        method. The runner puts `Benchmark.sandbox(task)` on the run context for
        exactly this; a probe that carries a declaration (a run with
        `offer_tools`) is merged in. `engine_params.isolate` with nothing
        declared is an error -- nothing says what to show.
        """
        specs = []
        if getattr(ctx, "sandbox", None) is not None:
            specs.append(ctx.sandbox)
        probes = ctx.tools.probes.values() if ctx.tools else ()
        specs += [p.sandbox for p in probes if getattr(p, "sandbox", None) is not None]
        if not specs:
            if self.param("isolate", False):
                raise RuntimeError(
                    "engine_params.isolate is set but the task declares no sandbox; "
                    "set the benchmark's isolate option"
                )
            return None
        merged = specs[0]
        for spec in specs[1:]:
            merged = merged.merged(spec)
        return merged

    def _isolate(
        self, argv: list[str], sandbox: isolation.Sandbox, work: Path, task_dir: Path,
        workspace: Path | None = None,
    ) -> list[str]:
        """`argv` run inside `sandbox`, with what codex itself needs added.

        Codex's own sandbox writes only to its working directory and `/tmp`, so
        each declared writable path is also handed to it with `--add-dir`, or
        the report the instruction asks for can never be written. It gets a
        codex home of its own inside the working directory: the user's
        `~/.codex` holds earlier sessions, answers included, and session
        rollouts written there would be invisible to `_rollouts_for` anyway.

        The task's `workspace` -- the directory the run's shell opens in, where
        a stage is told to write its artifact and the only place the evaluator
        will read one from -- is mounted writable too. It is the shell probe's
        cwd, not part of the benchmark's declaration, so it was not here: a
        builder moved onto codex wrote its candidate in codex's own directory,
        found "the stated workspace directory absent", tested with its own
        python and submitted unrun.
        """
        at = argv.index("--cd") + 2
        writable = [*sandbox.writable, *([str(workspace)] if workspace else [])]
        added = [x for d in writable for x in ("--add-dir", d)]
        argv = [*argv[:at], *added, *argv[at:]]
        found = shutil.which(argv[0])
        extra_ro: list[str] = []
        if found:
            # The *resolved* path, not what `which` returned. `which` finds a
            # symlink on PATH, and following it inside the sandbox needs every
            # hop of the chain mounted -- here `~/.local/bin/codex` pointed at
            # `/nlp/scr/<user>/bin/codex`, itself a symlink into `/juice2/...`,
            # and only the two ends were mounted. bwrap reported the exec path
            # as "No such file or directory", which reads like a missing binary
            # rather than an unmounted middle hop. Exec the real file and the
            # chain does not have to exist in here at all.
            real = Path(found).resolve()
            argv = [str(real), *argv[1:]]
            if not str(real).startswith("/usr/"):
                extra_ro += [str(Path(found).parent), str(real.parent)]
        home = work / ".codex"
        home.mkdir(parents=True, exist_ok=True)
        env = {"HOME": str(work), "CODEX_HOME": str(home)}
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        return sandbox.wrap(
            argv, chdir=work, env=env,
            extra_read_only=extra_ro, extra_writable=[work, task_dir, workspace],
        )

    def _keep_transcript(self, ctx: RunContext, task_dir: Path, tag: str) -> None:
        """Save what codex actually did, beside the task's trace.

        Codex is the one engine whose steps we do not otherwise see: it decides
        for itself what to run, and `LLM._trace_call` never sees its calls
        because they do not go through our client. The transcript lived only in
        the `TemporaryDirectory`, so after a run there was no way to answer how
        it reached anything.
        """
        src = task_dir / "transcript.log"
        if not src.exists() or ctx.trace.path is None:
            return
        try:
            dest = ctx.trace.path.with_suffix(f".codex.{tag or 'run'}.log")
            dest.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        except OSError as e:  # noqa: BLE001 - a lost transcript is not a failed run
            ctx.trace.log("engine.transcript_failed", error=str(e))

    def _usd_cap(self, ctx: RunContext) -> float | None:
        """What this call may spend: the task budget's remainder, if watching."""
        if not self.param("usd_watchdog", False):
            return None
        limit = getattr(ctx.budget, "max_usd", None)
        if limit is None:
            return None
        return max(0.0, float(limit) - float(ctx.budget.spent.cost_usd))

    def _spend_so_far(self, work: Path, since: float) -> float:
        """Dollars the running session has used, read from its rollouts so far."""
        inp, out, _cached = _tokens_for_workspace(work, since)
        rate_in, rate_out = self.param("rates", (1.25, 10.0))
        return (inp * rate_in + out * rate_out) / 1_000_000

    async def _spawn(
        self,
        argv: list[str],
        prompt_file: Path,
        task_dir: Path,
        work: Path,
        usd_cap: float | None = None,
        since: float = 0.0,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)  # or Codex inherits this repo's venv
        env.update(extra_env or {})
        transcript = task_dir / "transcript.log"
        timeout = float(timeout or self.param("timeout", 1800))
        poll = float(self.param("usd_poll", 10.0))
        with open(prompt_file, "rb") as stdin, open(transcript, "wb") as log:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=stdin,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(work),
                env=env,
                start_new_session=True,
            )
            deadline = time.time() + timeout
            try:
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError
                    wait = remaining if usd_cap is None else min(poll, remaining)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=wait)
                        break
                    except TimeoutError:
                        if usd_cap is None:
                            raise
                        # The rollout is written as codex goes, so the spend
                        # is readable mid-flight; a poll is a sampler, not a
                        # limiter, and can overshoot by one poll's worth.
                        spent = self._spend_so_far(work, since)
                        if spent >= usd_cap:
                            _kill_group(proc)
                            await proc.wait()
                            return (
                                BUDGET_EXIT,
                                f"usd cap ${usd_cap:.2f} reached (${spent:.2f} observed)",
                            )
            except TimeoutError:
                _kill_group(proc)
                await proc.wait()
                return -1, f"timeout after {timeout}s"
        tail = transcript.read_text(encoding="utf-8", errors="replace")[-2000:]
        return proc.returncode or 0, tail

    # -- cost --------------------------------------------------------------

    def _recover_usage(self, work: Path, since: float, task_dir: Path) -> Usage:
        """Read real spend back out of Codex's own session rollout.

        Codex bills against its own credentials, so `LLM.complete` never sees
        these calls. Without this the cost column would silently read zero for
        a Codex run, which is worse than an approximate number.
        """
        usage = Usage(calls=1)
        # Every rollout whose cwd is this workspace, not only the newest: a
        # spawned falsifier is its own rollout in the same cwd, and taking the
        # latest file alone either dropped the parent or dropped the child --
        # both real spend, and under `codex-loop` the child can cost more.
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens = _tokens_for_workspace(
            work, since
        )
        if not (usage.input_tokens or usage.output_tokens):
            transcript = task_dir / "transcript.log"
            if transcript.exists():
                match = _TOKENS_RE.search(transcript.read_text(encoding="utf-8", errors="replace"))
                if match:
                    usage.output_tokens = int(match.group(1).replace(",", ""))
        rate_in, rate_out = self.param("rates", (1.25, 10.0))
        usage.cost_usd = (usage.input_tokens * rate_in + usage.output_tokens * rate_out) / 1_000_000
        return usage

    def check(self) -> list[str]:
        binary = self.param("binary", "codex")
        if shutil.which(binary) is None:
            return [f"{binary!r} not found on PATH"]
        return []


# -- helpers ---------------------------------------------------------------


#: Written into codex's workspace when `probe_transport` includes "python".
_SHIM = """\"\"\"The harness's probes, callable from this workspace.

Each function forwards to the running sheldon process and returns the probe's
result as text. They are the same probe objects every other method is offered —
the same per-task budgets, the same recorded retrievals — so a call here counts
exactly as it would anywhere else.

    python3 sheldon_probes.py <name> '<json arguments>'
\"\"\"

import json
import sys
import urllib.request

URL = {url!r}
TIMEOUT = {timeout!r}


def call(name, **arguments):
    \"\"\"Invoke a probe by name. Returns its result as a string.\"\"\"
    request = {{
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {{"name": name, "arguments": arguments}},
    }}
    body = json.dumps(request).encode()
    headers = {{"Content-Type": "application/json", "Accept": "application/json"}}
    with urllib.request.urlopen(
        urllib.request.Request(URL, data=body, headers=headers), timeout=TIMEOUT
    ) as response:
        payload = json.load(response)
    if "error" in payload:
        return "probe error: " + json.dumps(payload["error"])
    content = (payload.get("result") or {{}}).get("content") or []
    return "\\n".join(part.get("text", "") for part in content)


{functions}

if __name__ == "__main__":
    name = sys.argv[1]
    arguments = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {{}}
    print(call(name, **arguments))
"""


def _shim_source(url: str, probes: Any, timeout: float = SHIM_TIMEOUT) -> str:
    """The workspace module, with one wrapper per probe.

    `timeout` is how long a shim call waits on the bridge. It has to exceed the
    slowest probe or the agent is billed for work it never sees: a probe that
    outlasts the wait runs, is recorded, charges the budget and comes back to
    codex as `timed out` -- so a working artifact can be retried as a crash.
    """
    functions = []
    for name in sorted(probes.probes):
        probe = probes.probes[name]
        schema = probe.input_schema or {}
        fields = list((schema.get("properties") or {}).keys())
        params = ", ".join(f"{f}=None" for f in fields)
        passed = ", ".join(f"{f}={f}" for f in fields)
        doc = " ".join((probe.description or "").split())[:400]
        body = (
            f"def {name}({params}):\n"
            f'    """{doc}"""\n'
            f"    arguments = {{k: v for k, v in dict({passed}).items() if v is not None}}\n"
            f'    return call("{name}", **arguments)\n'
        )
        functions.append(body)
    return _SHIM.format(url=url, timeout=float(timeout), functions="\n".join(functions))


def _shim_prompt(probes: Any) -> str:
    """What the prompt has to say for the shim to be usable.

    Codex is not handed tool definitions the way a native tool loop is, so the
    signatures have to arrive as text or the file may as well not be there.
    """
    lines = [
        "## Tools available in this workspace",
        "",
        "`sheldon_probes.py` is already written in your working directory. It exposes the",
        "investigation tools for this task. Import it and call the functions -- this is",
        "the only way to reach the task's data, and answering without it means answering",
        "without evidence.",
        "",
        "```python",
        "import sheldon_probes",
        "print(sheldon_probes.NAME(arg=...))",
        "```",
        "",
        "Only calls made through `sheldon_probes` are on the record. A command you run in",
        "this shell is invisible to the investigation: an artifact you test only here is",
        "recorded as never run, and is sent back. Whatever the task must see you run --",
        "the artifact you submit, the check that measures it -- run it through these",
        "tools, by its file path where a tool takes one.",
        "",
    ]
    for name in sorted(probes.probes):
        probe = probes.probes[name]
        schema = probe.input_schema or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        args = ", ".join(f"{f}" + ("" if f in required else "=None") for f in properties)
        lines.append(f"- `{name}({args})` -- {' '.join((probe.description or '').split())[:220]}")
    return "\n".join(lines)


def _budget_prompt(cap: float) -> str:
    """Tell codex its spend limit, so pacing is a decision rather than a surprise."""
    return (
        "## Spend limit\n\n"
        f"About ${cap:.2f} of model usage remains for this task. The harness stops "
        "this session when it is spent; stopping earlier forfeits the remainder, so "
        "keep working until it does."
    )


def _provider_config() -> list[str]:
    """Point Codex at the same endpoint the rest of the harness uses.

    Without this, Codex would quietly run against whatever provider its own
    `~/.codex/config.toml` names — a different model, a different bill, and a
    comparison that no longer means what the run label says. Override wholesale
    with `engine_params.provider_config`.
    """
    base = os.environ.get("OPENAI_BASE_URL")
    if not base:
        return []
    from sheldon.core.llm import normalize_base_url  # noqa: PLC0415

    return [
        "model_provider=sheldon",
        'model_providers.sheldon.name="sheldon endpoint"',
        f'model_providers.sheldon.base_url="{normalize_base_url(base)}"',
        'model_providers.sheldon.env_key="OPENAI_API_KEY"',
        'model_providers.sheldon.wire_api="responses"',
    ]


def _terminal_schema(
    extra_tools: list[dict[str, Any]] | None, stop_on: set[str] | None
) -> tuple[dict | None, str | None]:
    """If the caller wants one specific tool called, use its schema as the
    output schema — the closest Codex can get to calling it."""
    if not stop_on or not extra_tools:
        return None, None
    for tool in extra_tools:
        if tool.get("name") in stop_on and tool.get("input_schema"):
            return _strict(tool["input_schema"]), tool["name"]
    return None, None


def _strict(schema: Any) -> Any:
    """Stamp `additionalProperties: false` on every object in the schema.

    `--output-schema` is enforced as a strict response format, and a strict
    provider rejects any object that does not say so — including the ones inside
    `$defs`, which is where a nested pydantic model ends up. Our schemas come
    from `model_json_schema()`, which never emits the key, so every structured
    Codex call failed the request outright:

        Invalid schema for response_format 'codex_output_schema':
        In context=(), 'additionalProperties' is required to be supplied and to
        be false.

    That is a 400 before any work happens, so the engine burned its whole
    timeout retrying and reported nothing. Copied rather than mutated: the tool
    definition is shared with the other engines, which do not want this.

    Strict mode has a second requirement with the same shape, and every `sheldon`
    stage tripped it: `required` must name *every* key in `properties`. Our
    schemas come from pydantic models whose optional fields carry defaults, so
    `model_json_schema()` leaves them out of `required` and the provider rejects
    the request:

        Invalid schema for response_format 'codex_output_schema':
        'required' is required to be supplied and to be an array including
        every key in properties. Missing 'rationale'.

    Assigned rather than defaulted, because the failing case is a `required`
    list that already exists and is merely incomplete. A field the model has
    nothing to say about comes back empty rather than absent, which is what the
    pydantic default would have produced anyway.
    """
    if isinstance(schema, list):
        return [_strict(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _strict(v) for k, v in schema.items()}
    if out.get("type") == "object" or "properties" in out:
        out.setdefault("additionalProperties", False)
        properties = out.get("properties")
        if isinstance(properties, dict) and properties:
            out["required"] = list(properties)
    return out


def _kill_group(proc: Any) -> None:
    import signal  # noqa: PLC0415

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _task_workspace(ctx: RunContext) -> Path | None:
    """The directory the run's shell opens in, read off the probes the way the
    method layer does: a probe that takes a path carries it as `workspace`, a
    shell as `cwd`. None where the run has no such directory."""
    tools = getattr(ctx, "tools", None)
    for probe in (tools.probes.values() if tools else ()):
        where = getattr(probe, "workspace", None) or getattr(probe, "cwd", None)
        if where:
            return Path(str(where))
    return None


def _session_cwd(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            event = json.loads(f.readline())
        if event.get("type") == "session_meta":
            return (event.get("payload") or {}).get("cwd")
    except (OSError, ValueError):
        return None
    return None


def _rollouts_for(work: Path, since: float) -> list[Path]:
    """Every Codex session since `since` whose cwd is our working directory.

    An isolated call has a codex home of its own inside the workspace, and its
    rollouts are there; otherwise they are in the user's `~/.codex`."""
    local = work / ".codex" / "sessions"
    root = local if local.is_dir() else Path(os.path.expanduser("~/.codex/sessions"))
    pattern = str(root / "**" / "rollout-*.jsonl")
    ours = [
        f
        for f in glob.glob(pattern, recursive=True)
        if os.path.getmtime(f) >= since and (cwd := _session_cwd(f)) and Path(cwd).resolve() == work
    ]
    return [Path(f) for f in sorted(ours, key=os.path.getmtime)]


def _latest_rollout(work: Path, since: float) -> Path | None:
    """The most recent Codex session whose cwd is our working directory."""
    ours = _rollouts_for(work, since)
    return ours[-1] if ours else None


def _tokens_for_workspace(work: Path, since: float) -> tuple[int, int, int]:
    """(uncached input, output, cached input) summed over the workspace's sessions."""
    inp = out = cached = 0
    for rollout in _rollouts_for(work, since):
        tokens = _tokens_from_rollout(rollout)
        if tokens:
            inp += tokens[0]
            out += tokens[1]
            cached += tokens[2]
    return inp, out, cached


def _tokens_from_rollout(path: Path) -> tuple[int, int, int] | None:
    """(uncached input, output, cached input). total_token_usage is cumulative,
    so the last one wins."""
    result = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"token_count"' not in line:
                    continue
                try:
                    info = (json.loads(line).get("payload") or {}).get("info") or {}
                except ValueError:
                    continue
                totals = info.get("total_token_usage")
                if isinstance(totals, dict):
                    cached = totals.get("cached_input_tokens", 0) or 0
                    inp = max(0, (totals.get("input_tokens", 0) or 0) - cached)
                    out = totals.get("output_tokens", 0) or 0
                    if inp + out > 0:
                        result = (inp, out, cached)
    except OSError:
        return None
    return result
