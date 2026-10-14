"""Filesystem isolation for anything that runs an agent's commands.

An agent with a shell can read whatever the user running the harness can
read, and it goes looking: for the answer file of its own task, the task
manifest, the grading material, earlier runs' outputs -- none of which
`Task.blind()` can stop, because the leak is the filesystem, not the task
object. Copying the source elsewhere does not help while any path the agent is
shown names the repo or `/tmp` holds other sessions' copies.

A `Sandbox` is the answer to that: the command runs under bubblewrap in a
fresh mount namespace that holds the system directories read-only, a private
`/tmp`, and only the paths the benchmark declares -- read-only for what the
agent may inspect, writable for what it must produce. The environment is
cleared and rebuilt from what is declared, so no repo path rides in through
`PWD` or `VIRTUAL_ENV`. The network is shared on purpose: the agent has to
reach the telemetry services and the model endpoint, as upstream's agent does.

The benchmark declares the sandbox, on the probe that runs commands, so every
method is held to the same visibility; an engine that brings its own shell
(`codex`) reads the declaration off `ctx.tools` and wraps itself in it. Asking
for isolation where bubblewrap is missing is an error, never a silent fallback
to the host filesystem.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: The system, visible read-only to every sandbox. Nothing under a user's home.
SYSTEM_READ_ONLY = ("/usr", "/etc")
#: Mounted when present. On systemd hosts `/etc/resolv.conf` points into this
#: directory, and without it every hostname fails to resolve.
SYSTEM_READ_ONLY_TRY = ("/run/systemd/resolve",)
#: Merged-usr hosts make these symlinks into /usr; older hosts keep them as dirs.
SYSTEM_TOP_DIRS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"


def available() -> bool:
    """Whether bubblewrap is installed. User namespaces must also be allowed,
    which a failed wrap reports on its own."""
    return shutil.which("bwrap") is not None


def _norm(paths: Iterable[str | Path | None]) -> tuple[str, ...]:
    out: list[str] = []
    for p in paths:
        if not p:
            continue
        s = str(Path(p).expanduser().resolve())
        if s not in out:
            out.append(s)
    return tuple(out)


@dataclass(frozen=True)
class Sandbox:
    """What an agent's commands may see: paths read-only, paths writable, and
    the environment variables it is given. Everything else does not exist."""

    read_only: tuple[str, ...] = ()
    writable: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    #: Scheduling niceness for everything the agent runs, 0 for none. An
    #: agent's own runs of a program share the host with the evaluator, whose
    #: time limit is wall clock: on a loaded host, programs that finish in
    #: time alone are killed at the limit. A niced agent yields the core to
    #: the evaluation.
    nice: int = 0

    @classmethod
    def of(
        cls,
        read_only: Iterable[str | Path | None] = (),
        writable: Iterable[str | Path | None] = (),
        env: Mapping[str, str] | None = None,
        nice: int = 0,
    ) -> Sandbox:
        return cls(_norm(read_only), _norm(writable), tuple(sorted((env or {}).items())), int(nice))

    def merged(self, other: Sandbox) -> Sandbox:
        """Two declarations, as one: the union of paths; `other` wins on env."""
        return Sandbox(
            _norm([*self.read_only, *other.read_only]),
            _norm([*self.writable, *other.writable]),
            tuple(sorted({**dict(self.env), **dict(other.env)}.items())),
            max(self.nice, other.nice),
        )

    def wrap(
        self,
        argv: list[str],
        *,
        chdir: str | Path,
        env: Mapping[str, str] | None = None,
        extra_read_only: Iterable[str | Path | None] = (),
        extra_writable: Iterable[str | Path | None] = (),
    ) -> list[str]:
        """`argv` as a bubblewrap command line honouring this sandbox.

        `extra_*` are the caller's own paths -- an engine's working directory,
        a probe's retrieval index -- mounted beside the declared ones. Extra
        read-only paths are mounted only if they exist; declared ones must
        exist, and a missing one fails the command rather than hiding it.
        Writable paths are created first. Mount order matters: the private
        `/tmp` goes down before anything that lives under `/tmp` is bound.
        """
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError(
                "isolation was requested but bubblewrap (`bwrap`) is not installed; "
                "refusing to run an agent against the host filesystem"
            )
        out = [bwrap, "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--new-session"]
        for top in SYSTEM_TOP_DIRS:
            p = Path(top)
            if p.is_symlink():
                out += ["--symlink", os.readlink(p), top]
            elif p.is_dir():
                out += ["--ro-bind", top, top]
        for d in SYSTEM_READ_ONLY:
            out += ["--ro-bind", d, d]
        for d in SYSTEM_READ_ONLY_TRY:
            out += ["--ro-bind-try", d, d]
        out += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for p in self.read_only:
            out += ["--ro-bind", p, p]
        for p in _norm(extra_read_only):
            out += ["--ro-bind-try", p, p]
        for p in _norm([*self.writable, *extra_writable]):
            Path(p).mkdir(parents=True, exist_ok=True)
            out += ["--bind", p, p]
        out += ["--clearenv", "--setenv", "PATH", SAFE_PATH, "--setenv", "LANG", "C.UTF-8",
                "--setenv", "HOME", "/tmp"]
        for key, value in [*self.env, *(env or {}).items()]:
            out += ["--setenv", key, str(value)]
        out += ["--chdir", str(chdir), "--", *argv]
        if self.nice:
            # Outside bwrap, so the sandbox needs no `nice` of its own and
            # every process the command starts inherits the priority.
            out = [shutil.which("nice") or "nice", "-n", str(self.nice), *out]
        return out
