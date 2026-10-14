"""Recursive Language Models (github.com/alexzhang13/rlm) as an engine.

An RLM replaces one completion with a loop in which the model writes Python
into a REPL, can call itself over slices of its own context, and sets an
`answer` when it is done. That is a way of running one step, so it sits on the
engine axis beside `toolloop`, `react` and `codex`.

It belongs in codex's category rather than toolloop's, for two reasons that
both matter when reading a number off it:

- **It brings its own tools.** The REPL is RLM's, not `ctx.tools`, so a run on
  this engine is not comparable to one on `toolloop`, which is offered the
  benchmark's probes. Give it its own row.
- **Its model calls are its own.** They do not pass through `LLM._cached`, so
  they are not cached, not replayed, and not written into the task trace the
  way every other call is. `runs/` records what this engine reports and nothing
  finer, and a re-run costs full price.

The endpoint returns no price, so tokens are charged at `engine_params.rates`,
the way the codex engine charges its own.

Isolation is not optional where the benchmark declares it. The REPL executes
model-written code, and where a benchmark's answers sit in its data files an
unconfined REPL is one `open()` from the label. The call therefore runs in `worker.py` under the declared
sandbox, with the virtual environment mounted read-only so Python can start and
a per-call working directory mounted writable. The repository is not mounted.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from sheldon.core import isolation
from sheldon.core.io import load_dotenv
from sheldon.core.llm import normalize_base_url
from sheldon.core.registry import engines
from sheldon.core.types import RunContext, Usage
from sheldon.engines.base import Engine, EngineResult

WORKER = Path(__file__).resolve().parent / "worker.py"
DEFAULT_RATES = (2.50, 15.00)

#: What the root model is asked, while the task itself sits in the REPL.
#:
#: RLM splits a call in two: `prompt` becomes a `context` variable inside the
#: REPL, and only `root_prompt` reaches the conversation. Passing the task as
#: `prompt` alone therefore leaves the model holding a context and no question,
#: and it answers that it was never given one -- which is what the first runs
#: here did, on every task of both benchmarks, at full price.
#:
#: This stays benchmark-neutral on purpose. It names where the task is and what
#: shape the reply must take, and says nothing about what any particular task
#: contains; a directive carrying domain hints would hand this engine an
#: advantage the other engines are not offered.
ROOT_PROMPT = (
    "The complete task statement is in the REPL `context` variable. "
    "Read it there before answering, work through it in the REPL, and then give "
    "the full final answer in exactly the form the task asks for. "
    "Do not summarise it, describe it, or ask for it to be repeated."
)


@engines.register("rlm")
class RlmEngine(Engine):
    name = "rlm"
    blurb = "recursive language model: a REPL the model programs, and sub-calls on its own context"
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
        load_dotenv()  # the file beats a stale exported key, as everywhere else
        work = Path(self.param("cwd", "scratch/rlm-work")).resolve() / (
            f"{(ctx.task.id if getattr(ctx, 'task', None) else 'task')}--{tag or 'rlm'}"
            f"--{uuid.uuid4().hex[:6]}".replace("/", "_")
        )
        work.mkdir(parents=True, exist_ok=True)
        job_path, out_path = work / "job.json", work / "result.json"

        base_url = normalize_base_url(
            self.param("base_url", None) or os.environ.get("OPENAI_BASE_URL", "")
        )
        job = {
            "prompt": prompt,
            "system": system,
            # The stage's model, else this engine's own, else the run's. RLM
            # builds its client from the name and refuses an empty one, and
            # `RunContext` has no `model` of its own -- the run's is on `llm`.
            "model": (
                model
                or self.param("model", None)
                or getattr(getattr(ctx, "llm", None), "model", "")
                or ""
            ),
            "base_url": base_url,
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            # `max_steps` is the harness's step ceiling for this stage; RLM's
            # own iteration cap is the same idea under another name.
            "max_iterations": int(max_steps or self.param("max_iterations", 20)),
            "max_depth": int(self.param("max_depth", 1)),
            "max_timeout": self.param("max_timeout", None),
            "environment": self.param("environment", "local"),
            # The method's own system prompt rides here, ahead of the
            # directive, because RLM's system prompt slot cannot be borrowed
            # without destroying the REPL instructions it carries.
            "root_prompt": "\n\n".join(
                p for p in (system, self.param("root_prompt", ROOT_PROMPT)) if p
            ),
        }
        job_path.write_text(json.dumps(job), encoding="utf-8")

        argv = [sys.executable, str(WORKER), str(job_path), str(out_path)]
        sandbox = self._sandbox(ctx)
        if sandbox is not None:
            argv = sandbox.wrap(
                argv,
                chdir=work,
                # The virtual environment, and of the repository only the
                # directory this worker lives in -- never `data/`, which holds
                # the answers. `sys.prefix` is the venv root; resolving
                # `sys.executable` instead follows the symlink out to the base
                # interpreter and mounts nothing of the environment, which
                # fails as `execvp ... No such file or directory`.
                extra_read_only=[sys.prefix, sys.base_prefix, WORKER.parent],
                extra_writable=[work],
            )
            ctx.trace.log(
                "engine.isolated", engine=self.name, tag=tag,
                read_only=list(sandbox.read_only), writable=list(sandbox.writable),
            )

        started = time.time()
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        timeout = float(self.param("timeout", 1800))
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            ctx.trace.log("engine.error", engine=self.name, tag=tag,
                          error=f"rlm did not finish within {timeout:.0f}s")
            return EngineResult(text="", usage=Usage(), stopped_by="timeout", iterations=0)

        # Read the file before anything can remove the directory: an engine
        # that reads its output too late reports every paid call as empty.
        result: dict[str, Any] = {}
        if out_path.exists():
            try:
                result = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                result = {"error": f"unreadable result file: {exc}"}

        rate_in, rate_out = self.param("rates", DEFAULT_RATES)
        usage = Usage(
            calls=int(result.get("calls", 0)),
            input_tokens=int(result.get("input_tokens", 0)),
            output_tokens=int(result.get("output_tokens", 0)),
        )
        usage.cost_usd = (
            usage.input_tokens * float(rate_in) + usage.output_tokens * float(rate_out)
        ) / 1_000_000
        ctx.budget.charge(usage)

        text = str(result.get("text") or "")
        ctx.trace.log(
            "engine.rlm", tag=tag, iterations=int(result.get("iterations", 0)),
            calls=usage.calls, usd=round(usage.cost_usd, 4),
            elapsed_s=result.get("elapsed_s"), exit_code=proc.returncode,
            error=result.get("error"), chars=len(text),
        )
        if not text:
            # A paid call that returns nothing reads like a method declining to
            # answer; codex hid exactly this behind exit code 0 for a while.
            ctx.trace.log(
                "engine.error", engine=self.name, tag=tag, exit_code=proc.returncode,
                error=result.get("error") or (err.decode(errors="replace")[-600:] if err else ""),
                traceback=result.get("traceback"),
            )
        return EngineResult(
            text=text,
            usage=usage,
            stopped_by="rlm" if text else "rlm+empty",
            iterations=int(result.get("iterations", 0)),
        )

    def _sandbox(self, ctx: RunContext) -> isolation.Sandbox | None:
        """The sandbox this task declares, merged with any a probe carries.

        Same rule the codex engine follows: this engine is offered no probes,
        so the declaration reaches it off the run context. Asking for isolation
        where nothing declares it is an error, never a silent run on the host.
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
