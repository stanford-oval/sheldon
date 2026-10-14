"""Run model-written Python in a separate process, with a timeout.

Used to execute a candidate program against a task's worked examples. This
executes generated code: it is subprocess-isolated and time-limited but *not* a
security sandbox, so it is off by default and only runs when a config sets `params.allow_code_exec: true`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _sandbox_env() -> dict[str, str]:
    """Environment for a sandbox subprocess.

    `PYTHONHASHSEED=0` is the load-bearing part. Without it the interpreter
    randomises string hashing per process, so model-written code that prints a
    `set` prints it in a different order each run. That output goes back into
    the conversation, so the next turn's request differs, so the cache misses,
    so a live call returns a different sample — and a run of an identical config
    stops reproducing: `{'5', '1'}` one time, `{'1', '5'}` the next.

    `LC_ALL` and `TZ` are pinned for the same reason, more cheaply.

    This is also why the subprocess runs under `-s` rather than `-I`: isolated
    mode implies `-E`, which makes the interpreter ignore every `PYTHON*`
    variable — `PYTHONHASHSEED` included, so the flag and the fix cannot both
    apply. `-s` still drops user site-packages, and passing an explicit env
    dict gives back the isolation `-I` was there for: nothing ambient reaches
    the child except PATH.
    """
    return {
        "PYTHONHASHSEED": "0",
        "PATH": os.environ.get("PATH", ""),
        "LC_ALL": "C",
        "TZ": "UTC",
    }

_HARNESS = r"""
import json, sys

src = json.loads(sys.stdin.read())
ns = {}
try:
    exec(src["program"], ns)
except Exception as e:
    print(json.dumps({"ok": False, "error": "compile: %s: %s" % (type(e).__name__, e)}))
    sys.exit(0)

fn = ns.get(src["entry"])
if not callable(fn):
    print(json.dumps({"ok": False, "error": "no callable named %r" % src["entry"]}))
    sys.exit(0)

out = []
for i, arg in enumerate(src["inputs"]):
    try:
        out.append({"i": i, "ok": True, "value": fn(arg)})
    except Exception as e:
        out.append({"i": i, "ok": False, "error": "%s: %s" % (type(e).__name__, e)})
print(json.dumps({"ok": True, "results": out}))
"""


def run_transform(
    program: str,
    inputs: list[Any],
    entry: str = "transform",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Apply `entry(x)` from `program` to each input.

    Returns {"ok": bool, "results": [{"i", "ok", "value"|"error"}], "error": str}.
    Never raises on bad model output — a broken program is a normal outcome.
    """
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "_run.py"
        harness.write_text(_HARNESS, encoding="utf-8")
        payload = json.dumps({"program": program, "inputs": inputs, "entry": entry})
        try:
            proc = subprocess.run(
                [sys.executable, "-s", str(harness)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=_sandbox_env(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout}s", "results": []}
        if proc.returncode != 0:
            return {"ok": False, "error": (proc.stderr or "nonzero exit")[:500], "results": []}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"unparseable output: {proc.stdout[:300]}", "results": []}


_SNIPPET_HARNESS = r"""
import io, json, sys, contextlib

src = json.loads(sys.stdin.read())
ns = dict(src.get("globals") or {})
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(src["code"], ns)
except Exception as e:
    print(json.dumps({
        "ok": False,
        "stdout": buf.getvalue()[-4000:],
        "error": "%s: %s" % (type(e).__name__, e),
    }))
    sys.exit(0)
print(json.dumps({"ok": True, "stdout": buf.getvalue()[-4000:], "error": None}))
"""


def run_snippet(
    code: str,
    globals_: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Execute a snippet and capture stdout.

    The general-purpose escape hatch behind `PythonProbe`: the falsifier writes a
    few lines that try to break a hypothesis, and gets back whatever it printed.
    Same caveat as `run_transform` — subprocess-isolated and time-limited, not a
    security sandbox.

    Returns {"ok": bool, "stdout": str, "error": str | None}.
    """
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "_snippet.py"
        harness.write_text(_SNIPPET_HARNESS, encoding="utf-8")
        payload = json.dumps({"code": code, "globals": globals_ or {}})
        try:
            proc = subprocess.run(
                [sys.executable, "-s", str(harness)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                env=_sandbox_env(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "error": f"timeout after {timeout}s"}
        if proc.returncode != 0:
            return {"ok": False, "stdout": "", "error": (proc.stderr or "nonzero exit")[:500]}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "stdout": proc.stdout[:2000], "error": "unparseable harness output"}
