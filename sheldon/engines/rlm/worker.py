"""The subprocess that actually runs a Recursive Language Model call.

`rlm.RLM.completion` is synchronous and its REPL executes model-written code in
this process. Both are reasons to keep it out of the harness: the runner is an
event loop with concurrent tasks, and an agent that can execute code can read
the benchmark's answers off the disk. So
the engine spawns this file instead, under the sandbox the benchmark declared,
and talks to it through two JSON files.

It deliberately imports nothing from `sheldon`. Under isolation only the virtual
environment is mounted, not the repository, so anything this file needed from
the harness would have to bring the repository -- and the answers -- with it.
The endpoint and key arrive in the job file, which lives in the one writable
directory the engine mounts.

    worker.py <job.json> <result.json>
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _usage(summary: Any) -> dict[str, int]:
    """Token totals out of RLM's `UsageSummary`, whatever shape it is in.

    The endpoint returns no price, so `total_cost` is None and the engine
    prices the tokens itself. Only the counts matter here.
    """
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    if summary is None:
        return totals
    per_model = getattr(summary, "model_usage_summaries", None)
    if isinstance(per_model, dict):
        for one in per_model.values():
            totals["calls"] += int(getattr(one, "total_calls", 0) or 0)
            totals["input_tokens"] += int(getattr(one, "total_input_tokens", 0) or 0)
            totals["output_tokens"] += int(getattr(one, "total_output_tokens", 0) or 0)
    return totals


def _unshadow() -> None:
    """Drop this file's own directory from the import path.

    Python puts a script's directory first on `sys.path`, and this one sits in
    the engine's own package. Nothing there may shadow the installed `rlm`
    package: a module of that name beside this file once made `import rlm` find
    the engine, which imports `sheldon`, which the sandbox does not mount.
    Removing the script directory makes the import reach the installed package.
    """
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != here]


def _finish(job: dict[str, Any], result: dict[str, Any]) -> None:
    """One final call, with no REPL, when the recursion ends holding nothing.

    RLM returns empty text when the model spends its iteration cap without
    committing an answer, and the cap is the harness's number, not the model's:
    a task can return a complete answer at six iterations and nothing at ten,
    so a larger allowance scores *worse*. An engine
    that reports that as an empty answer is scoring its own ceiling as a wrong
    answer, which is why both tool-using engines here already make a final
    tool-free call when they hit a step limit. This is that call.

    It is recorded in `finish_used` so a run can be read knowing which answers
    came from the recursion and which from the fallback.
    """
    from openai import OpenAI  # noqa: PLC0415 - inside the sandbox only

    client = OpenAI(base_url=job["base_url"], api_key=job["api_key"])
    reply = client.chat.completions.create(
        model=job["model"],
        messages=[
            {
                "role": "user",
                "content": (
                    f"{job.get('root_prompt') or ''}\n\n"
                    "Your working session ended before you gave an answer. Answer now, "
                    "directly and in full, with no further tool use.\n\n"
                    f"{job['prompt']}"
                ),
            }
        ],
    )
    result["text"] = str(reply.choices[0].message.content or "")
    result["finish_used"] = True
    usage = getattr(reply, "usage", None)
    if usage is not None:
        result["calls"] = int(result.get("calls", 0)) + 1
        result["input_tokens"] = int(result.get("input_tokens", 0)) + int(
            getattr(usage, "prompt_tokens", 0) or 0
        )
        result["output_tokens"] = int(result.get("output_tokens", 0)) + int(
            getattr(usage, "completion_tokens", 0) or 0
        )


def main() -> int:
    job_path, out_path = sys.argv[1], sys.argv[2]
    _unshadow()
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)

    result: dict[str, Any] = {
        "text": "",
        "error": None,
        "iterations": 0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    started = time.time()
    try:
        from rlm import RLM  # noqa: PLC0415 - imported inside the sandbox only

        backend_kwargs = {
            "model_name": job["model"],
            "base_url": job["base_url"],
            "api_key": job["api_key"],
        }
        kwargs: dict[str, Any] = {
            "backend": "openai",
            "backend_kwargs": backend_kwargs,
            # `local` is the REPL this whole file exists to confine. Under the
            # sandbox it can see the virtual environment and its own working
            # directory and nothing else.
            "environment": job.get("environment", "local"),
            "max_iterations": int(job["max_iterations"]),
            "max_depth": int(job.get("max_depth", 1)),
            "verbose": False,
        }
        if job.get("max_timeout"):
            kwargs["max_timeout"] = float(job["max_timeout"])
        # `custom_system_prompt` is deliberately never set. It *replaces* RLM's
        # own system prompt, which is the only thing telling the model that a
        # REPL exists, that the task is in `context`, and that code goes in
        # ```repl blocks. Feeding a method's system prompt in here cost two
        # full runs whose every task answered "I cannot access the REPL
        # `context` variable in this interface." The method's framing rides in
        # `root_prompt` instead.

        rlm = RLM(**kwargs)
        # Two arguments, not one: `prompt` lands in the REPL as `context` and
        # `root_prompt` is the only thing the root model sees in conversation.
        completion = rlm.completion(job["prompt"], root_prompt=job.get("root_prompt") or None)
        result["text"] = str(getattr(completion, "response", "") or "")
        result["error"] = getattr(completion, "error", None)
        result.update(_usage(getattr(completion, "usage_summary", None)))
        metadata = getattr(completion, "metadata", None) or {}
        if isinstance(metadata, dict):
            result["iterations"] = int(
                metadata.get("iterations") or metadata.get("num_iterations") or 0
            )
    except Exception as exc:  # noqa: BLE001 - the engine reports, never crashes the run
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=6)

    # Outside the `try`, so it covers both ways the recursion can end holding
    # nothing: returning empty text at the iteration cap, and *raising* at the
    # wall-clock one (`TimeoutExceededError`). Both are harness ceilings, and a
    # ceiling scored as a wrong answer is the bug this guards against; the
    # first version only caught the first case and a timeout still scored 0.
    if not result["text"].strip():
        try:
            _finish(job, result)
        except Exception as exc:  # noqa: BLE001 - report it, never crash the run
            result["finish_error"] = f"{type(exc).__name__}: {exc}"

    result["elapsed_s"] = round(time.time() - started, 2)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
