"""Probes: the attack surfaces a falsifier is allowed to use.

A `Verifier` is one fixed check. A `Probe` is a *named, described, schema'd*
capability the falsifier can choose between and call repeatedly — run this SQL,
search the web for this claim, grep the corpus for that passage, execute this
snippet. The benchmark supplies the probes that make sense for its domain; the
experiment config can add general-purpose ones.

Two axes vary independently, and conflating them is the mistake to avoid:

  who executes    client-side (we run it: SQL, python, corpus search) vs
                  server-side (Anthropic runs it: web search). Server probes
                  have no handler — results arrive in the response.

  how much it     `decisive` probes are oracles. If the worked examples
  counts          don't reproduce, the hypothesis is wrong, full stop. A web
                  search that turns up a contradicting sentence is an argument,
                  not a fact, so its refutations are capped by `max_strength`.

Both feed `refutation_strength`, which is what stops a confident-sounding search
result from killing a correct hypothesis the way a failing unit test should.

Every probe is bound by the same rule as `Verifier`: it must never read
`Task.answer`. Probes are constructed from a blinded task for exactly that
reason, and `tests/test_no_label_leak.py` checks it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from sheldon.core.isolation import Sandbox
from sheldon.core.sandbox import run_snippet

# Server-tool type strings. Bump these together with the model floor they need.
WEB_SEARCH_TOOL = "web_search_20260209"
WEB_FETCH_TOOL = "web_fetch_20260209"


@dataclass
class ProbeResult:
    """What one probe invocation found."""

    summary: str
    refutes: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = True
    strength: float | None = None  # None = use the probe's max_strength
    #: Override the render cap. A probe whose whole result is the point -- a
    #: shell command's output, an API response -- sets this, because the default
    #: is a summary line plus five rows and it is silent about the rest. An
    #: agent told "10000 matched" and shown five has been starved without
    #: knowing it.
    limit: int | None = None

    def render(self, limit: int = 4000) -> str:
        """What the model sees as the tool result."""
        parts = [self.summary]
        for item in self.evidence[:5]:
            parts.append(json.dumps(item, default=str)[:800])
        return "\n".join(parts)[: self.limit or limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary[:1200],
            "refutes": self.refutes,
            "evidence": self.evidence,
            "ok": self.ok,
            "strength": self.strength,
        }


class Probe(ABC):
    """One capability the falsifier can call.

    Subclasses implement `run(**kwargs)`. `name`, `description`, and
    `input_schema` become the tool definition the model sees, so the description
    should say what the probe *proves*, not just what it does.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    #: True if a refutation from this probe is ground truth, not an argument
    decisive: ClassVar[bool] = False
    #: ceiling on how far a refutation from this probe may move credence
    max_strength: ClassVar[float] = 0.7
    #: server-side probes are executed by Anthropic; we never call `run`
    server_side: ClassVar[bool] = False
    #: True for a probe that reads the run's own record rather than the world.
    #: It looks nothing up, so recording it would index the act of reading the
    #: index, and a stage that recalls twice would push real retrievals out.
    transient: ClassVar[bool] = False

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    @abstractmethod
    async def run(self, **kwargs: Any) -> ProbeResult:
        """Execute the probe. Server-side probes never reach this."""

    def tool_definition(self) -> dict[str, Any]:
        """The `tools` entry for this probe."""
        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": self.input_schema,
        }

    def strength_for(self, result: ProbeResult) -> float:
        if not result.refutes:
            return 0.0
        proposed = self.max_strength if result.strength is None else float(result.strength)
        return max(0.0, min(self.max_strength, proposed))


# -- server-side ----------------------------------------------------------


class ServerProbe(Probe):
    """A probe Anthropic executes. We contribute the definition, not a handler."""

    server_side = True

    @abstractmethod
    def tool_definition(self) -> dict[str, Any]: ...

    async def run(self, **kwargs: Any) -> ProbeResult:  # pragma: no cover
        raise RuntimeError(f"{self.name} is server-side; the API executes it")


class WebSearchProbe(ServerProbe):
    """Check a claim against the live web.

    Evidential, not decisive: a contradicting search result is a reason to lower
    credence, never a reason to declare a hypothesis dead.
    """

    name = "web_search"
    description = "Search the web for evidence that contradicts a claim."
    max_strength = 0.65

    def __init__(
        self,
        max_uses: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> None:
        if allowed_domains and blocked_domains:
            raise ValueError("web_search takes allowed_domains or blocked_domains, not both")
        self.max_uses = max_uses
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains

    def tool_definition(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": WEB_SEARCH_TOOL,
            "name": "web_search",
            "max_uses": self.max_uses,
        }
        if self.allowed_domains:
            d["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            d["blocked_domains"] = self.blocked_domains
        return d


class WebFetchProbe(ServerProbe):
    """Read a specific URL already mentioned in the conversation."""

    name = "web_fetch"
    description = "Fetch a URL that already appeared in the conversation and read it."
    max_strength = 0.75

    def __init__(self, max_uses: int = 5, max_content_tokens: int = 20000) -> None:
        self.max_uses = max_uses
        self.max_content_tokens = max_content_tokens

    def tool_definition(self) -> dict[str, Any]:
        return {
            "type": WEB_FETCH_TOOL,
            "name": "web_fetch",
            "max_uses": self.max_uses,
            "max_content_tokens": self.max_content_tokens,
            "citations": {"enabled": True},
        }


# -- client-side ----------------------------------------------------------


class VerifierProbe(Probe):
    """Adapts a `Benchmark.verifier` into a decisive probe.

    A benchmark's check against its worked examples becomes just one probe
    among several, and keeps the oracle strength that a hard reproduction
    failure deserves.
    """

    name = "check_against_examples"
    description = (
        "Run a candidate rule against every worked example given in the problem and report "
        "which ones it fails to reproduce. This is an exact check: a failure here means the "
        "rule is wrong."
    )
    decisive = True
    max_strength = 0.98

    def __init__(self, verifier: Any, candidate_hint: str = "program") -> None:
        self.verifier = verifier
        self.candidate_hint = candidate_hint

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "program": {
                    "type": "string",
                    "description": (
                        "Python source defining transform(grid) -> grid that implements the "
                        "rule exactly. Must run as written, with no placeholders."
                    ),
                }
            },
            "required": ["program"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        falsification = await self.verifier.check(kwargs)
        return ProbeResult(
            summary=falsification.detail or "checked",
            refutes=falsification.refuted,
            evidence=falsification.counterexamples,
            # A verifier that examined nothing did not clear the candidate.
            # Reporting ok=True here is how a disabled oracle ends up telling
            # the critic "an exact check already ran and it passed".
            ok=falsification.checked > 0 or falsification.refuted,
        )


class PythonProbe(Probe):
    """Run a short snippet and read back what it printed.

    The general escape hatch: when the falsifier can *compute* whether a claim
    holds, it should, rather than arguing about it.
    """

    name = "run_python"
    description = (
        "Execute a short Python snippet and return its stdout. Use this to actually test a "
        "claim — construct the edge case, apply the rule, print what you get. Print your "
        "findings; nothing else is returned."
    )
    max_strength = 0.9

    def __init__(self, timeout: float = 10.0, context: dict[str, Any] | None = None) -> None:
        self.timeout = timeout
        self.context = context or {}

    @property
    def input_schema(self) -> dict[str, Any]:
        keys = ", ".join(sorted(self.context)) or "none"
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": f"Python to execute. Predefined variables: {keys}.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        out = run_snippet(kwargs["code"], self.context, timeout=self.timeout)
        if not out["ok"]:
            return ProbeResult(
                summary=f"snippet failed: {out['error']}", refutes=False, ok=False,
                evidence=[{"stdout": out["stdout"], "error": out["error"]}],
            )
        stdout = out["stdout"].strip() or "(no output)"
        return ProbeResult(
            summary=stdout[:3000],
            refutes=False,
            evidence=[{"stdout": stdout}],
        )


class ShellProbe(Probe):
    """A shell, in a directory, with an environment. The agent's own hands.

    Some benchmarks do not hand an agent tools at all -- they hand it a machine:
    a source tree on disk and a service behind an environment variable, where
    working out the API is part of the task. Replacing that with tidy typed
    query functions measures a different, easier problem, and makes the harness
    the author of the interface.

    So this probe invents nothing. It runs what it is given and returns what came
    back, and the output cap is generous on purpose.

    Same caveat as `PythonProbe`, only broader -- a subprocess with a timeout is
    isolation, not a security sandbox, and this one runs arbitrary commands
    rather than arbitrary Python. It is opt-in per benchmark for that reason.
    """

    name = "run_shell"
    _BASE_DESCRIPTION = (
        "Run a shell command and return its output. This is your own shell: use it to "
        "query the APIs described in the task, read source files, and follow the evidence "
        "wherever it goes. Working out how to call those APIs is part of the job."
    )
    max_strength = 0.9

    def __init__(
        self,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        # Two minutes was not enough once the shell had a workspace to work in:
        # running a candidate over a task's data can take minutes, and a command
        # killed early throws that work away. Long enough for a benchmark's own job,
        # short enough that a hung command does not eat the round; override per
        # arm with `probes: [{run_shell: {timeout: 900}}]`.
        timeout: float = 600.0,
        max_output: int = 40000,
        record_dir: str | Path | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.cwd = str(cwd) if cwd else None
        #: What the commands may see, declared by the benchmark. None runs them
        #: on the host, which is only safe where nothing on disk is an answer.
        self.sandbox = sandbox
        self.env = dict(env or {})
        self.timeout = timeout
        self.max_output = max_output
        #: Where this run's own retrievals are written. Telling the agent about
        #: the directory is the whole mechanism -- with a shell there is no need
        #: for a `recall` tool, and `grep -l` across the directory beats reading
        #: an entry back by an id you had to know in advance.
        self.record_dir = str(record_dir) if record_dir else None
        if self.record_dir:
            self.env.setdefault("SHELDON_RETRIEVALS", self.record_dir)

    @property
    def description(self) -> str:  # type: ignore[override]
        if not self.record_dir:
            return self._BASE_DESCRIPTION
        return (
            f"{self._BASE_DESCRIPTION} Everything this run has already retrieved is "
            f"written as JSON under {self.record_dir} (also $SHELDON_RETRIEVALS), one "
            f"file per call with the arguments that produced it -- read or grep it "
            f"rather than repeating a query you have already made."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run, as you would type it.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    #: How many files under the working directory are watched for changes.
    #: A workspace holds a seed and a handful of artifacts; a command that
    #: unpacks a tree past this is not what the watch is for, and the walk
    #: stops rather than stalling every call after it.
    WATCH_LIMIT = 5000

    def _mtimes(self) -> dict[str, tuple[int, int]]:
        """`{relative path: (mtime_ns, size)}` for the files under `cwd`."""
        import os  # noqa: PLC0415

        if not self.cwd:
            return {}
        seen: dict[str, tuple[int, int]] = {}
        root = self.cwd
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                seen[os.path.relpath(full, root)] = (st.st_mtime_ns, st.st_size)
                if len(seen) >= self.WATCH_LIMIT:
                    return seen
        return seen

    @staticmethod
    def _changed(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
        return sorted(p for p, sig in after.items() if before.get(p) != sig)

    async def run(self, **kwargs: Any) -> ProbeResult:
        import asyncio  # noqa: PLC0415
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        command = str(kwargs.get("command") or "").strip()
        if not command:
            return ProbeResult(summary="no command given", ok=False)

        environ = {**os.environ, **self.env}
        argv = ["bash", "-lc", command]
        cwd = self.cwd
        if self.sandbox is not None:
            try:
                # The retrieval index is assigned by the runner after the probe
                # is built, so it joins the sandbox here, read-only.
                argv = self.sandbox.wrap(
                    argv,
                    chdir=self.cwd or "/tmp",
                    env=self.env,
                    extra_read_only=[self.record_dir],
                    # The directory the shell works in has to exist inside the
                    # sandbox and has to be writable, or `chdir` lands nowhere
                    # and every command fails. The benchmark declares what may
                    # be *seen*; the working directory is the harness's own and
                    # joins here, the way the retrieval index does.
                    extra_writable=[self.cwd],
                )
            except RuntimeError as e:
                return ProbeResult(summary=str(e), ok=False)
            cwd = None

        def run_it() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv, capture_output=True, text=True,
                timeout=self.timeout, cwd=cwd, env=environ,
            )

        before = self._mtimes()
        try:
            proc = await asyncio.to_thread(run_it)
        except subprocess.TimeoutExpired:
            return ProbeResult(summary=f"timed out after {self.timeout:.0f}s", ok=False)
        except Exception as e:
            return ProbeResult(summary=f"{type(e).__name__}: {e}", ok=False)

        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        out = out.strip()
        clipped = len(out) > self.max_output
        if clipped:
            # Say so, and say how much. A silent truncation is how an agent comes
            # to believe it has seen everything there was.
            out = (out[: self.max_output] +
                   f"\n[... truncated, {len(out) - self.max_output} more characters]")
        summary = out or f"(no output, exit {proc.returncode})"
        # Which files under the working directory this command wrote, so a
        # stage that collects an artifact by path can tell a call that ran the
        # file from one that edited it -- an edit after the last run means the
        # file on disk is not the one that ran, and nothing in the output says
        # so. Paths are relative to the working directory.
        changed = self._changed(before, self._mtimes())
        return ProbeResult(
            summary=summary,
            # The summary already carries the whole thing; a duplicate in
            # `evidence` would be re-rendered and clipped to 800 characters.
            evidence=[{"command": command, "exit": proc.returncode, "truncated": clipped,
                       "changed": changed}],
            ok=True,
            # The output *is* the result. Upstream's agent reads whatever the
            # API returned; ours must not be handed the first 4000 characters.
            limit=self.max_output + 2000,
        )


class SqlProbe(Probe):
    """Run read-only SQL against a table the task refers to.

    A query result is close to a fact, so this sits high on the strength scale —
    but only as high as the query is honest, which is why the schema is shown to
    the model up front rather than left to guesswork.
    """

    name = "run_sql"
    description = (
        "Run a read-only SQL query against the task's dataset and return the rows. Use this "
        "to look for records that contradict the hypothesis."
    )
    max_strength = 0.9

    _FORBIDDEN = re.compile(
        r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b", re.I
    )

    def __init__(self, db_path: str | Path, max_rows: int = 50, timeout: float = 10.0) -> None:
        self.db_path = Path(db_path)
        self.max_rows = max_rows
        self.timeout = timeout

    def schema_text(self) -> str:
        try:
            with self._connect() as con:
                rows = con.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
                ).fetchall()
            return "\n".join(r[0] for r in rows)[:4000] or "(no tables)"
        except sqlite3.Error as e:
            return f"(schema unavailable: {e})"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=self.timeout)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"A single read-only SELECT. Schema:\n{self.schema_text()}",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        query = str(kwargs.get("query", "")).strip().rstrip(";")
        if self._FORBIDDEN.search(query) or ";" in query:
            return ProbeResult(
                summary="rejected: only a single read-only SELECT is allowed",
                refutes=False,
                ok=False,
            )
        try:
            with self._connect() as con:
                con.row_factory = sqlite3.Row
                rows = con.execute(query).fetchmany(self.max_rows + 1)
        except sqlite3.Error as e:
            return ProbeResult(summary=f"SQL error: {e}", refutes=False, ok=False)

        truncated = len(rows) > self.max_rows
        records = [dict(r) for r in rows[: self.max_rows]]
        note = f" (truncated at {self.max_rows})" if truncated else ""
        return ProbeResult(
            summary=f"{len(records)} row(s){note}",
            refutes=False,
            evidence=records,
        )


class DocSearchProbe(Probe):
    """Search a local corpus — a paper, a filing, the task's own source material.

    Deliberately dependency-free scoring: token overlap over passages. Swap in
    BM25 or embeddings by subclassing and overriding `_score`; nothing else in
    the loop changes.
    """

    name = "search_documents"
    description = (
        "Search the source documents for passages relevant to a claim, and return them "
        "verbatim. Use this to check whether the material actually supports the hypothesis."
    )
    max_strength = 0.8

    def __init__(
        self,
        documents: dict[str, str] | None = None,
        root: str | Path | None = None,
        glob: str = "**/*.txt",
        window: int = 900,
        top_k: int = 4,
    ) -> None:
        self.window = window
        self.top_k = top_k
        self.documents: dict[str, str] = dict(documents or {})
        if root:
            for path in sorted(Path(root).glob(glob)):
                try:
                    self.documents[str(path.name)] = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

    @property
    def input_schema(self) -> dict[str, Any]:
        names = ", ".join(sorted(self.documents)[:20]) or "none loaded"
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Words to look for. Available documents: {names}.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    @staticmethod
    def _score(terms: set[str], passage: str) -> int:
        words = set(re.findall(r"[a-z0-9]+", passage.lower()))
        return len(terms & words)

    async def run(self, **kwargs: Any) -> ProbeResult:
        terms = set(re.findall(r"[a-z0-9]+", str(kwargs.get("query", "")).lower()))
        if not terms or not self.documents:
            return ProbeResult(summary="no query terms or no documents loaded", ok=False)

        scored: list[tuple[int, str, str]] = []
        for name, text in self.documents.items():
            for start in range(0, len(text), self.window // 2 or 1):
                passage = text[start : start + self.window]
                score = self._score(terms, passage)
                if score:
                    scored.append((score, name, passage))
        scored.sort(key=lambda x: -x[0])
        hits = scored[: self.top_k]
        if not hits:
            return ProbeResult(summary="no matching passages", refutes=False)
        return ProbeResult(
            summary=f"{len(hits)} passage(s) matched",
            refutes=False,
            evidence=[{"document": n, "passage": p.strip()} for _, n, p in hits],
        )


class CallableProbe(Probe):
    """Wrap any async function as a probe — the 'some other subroutine' case.

        probe = CallableProbe(
            name="check_units",
            description="Re-derive the quantity in SI units and report any mismatch.",
            input_schema={...},
            fn=my_async_checker,
            max_strength=0.85,
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Awaitable[ProbeResult | dict[str, Any] | str]],
        input_schema: dict[str, Any] | None = None,
        decisive: bool = False,
        max_strength: float = 0.7,
    ) -> None:
        self.name = name  # type: ignore[misc]
        self.description = description  # type: ignore[misc]
        self.decisive = decisive  # type: ignore[misc]
        self.max_strength = max_strength  # type: ignore[misc]
        self.fn = fn
        self._schema = input_schema or {
            "type": "object",
            "properties": {"argument": {"type": "string", "description": "Input to the check."}},
            "required": ["argument"],
            "additionalProperties": False,
        }

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._schema

    async def run(self, **kwargs: Any) -> ProbeResult:
        out = await self.fn(**kwargs)
        if isinstance(out, ProbeResult):
            return out
        if isinstance(out, dict):
            return ProbeResult(**out)
        return ProbeResult(summary=str(out))


# -- collections ----------------------------------------------------------


@dataclass
class Retrieval:
    """One thing the run looked up, kept so anything downstream can re-read it.

    The arguments matter as much as the result. Without them an index entry is
    prose a later stage has to reverse-engineer a call from -- and for probes
    that do not format their query into the summary, like `run_python`, there is
    nothing to reverse-engineer.
    """

    id: str
    probe: Probe
    arguments: dict[str, Any]
    result: ProbeResult

    def index_line(self) -> str:
        """The table-of-contents entry: enough to decide whether to read it."""
        args = json.dumps(self.arguments, default=str, sort_keys=True)
        return f"- [{self.id}] {self.probe.name} {args[:300]} -> {self.result.summary[:200]}"


class ProbeSet:
    """The probes available for one attack, plus dispatch."""

    def __init__(
        self,
        probes: list[Probe] | None = None,
        *,
        record: list[Retrieval] | None = None,
    ) -> None:
        self.probes: dict[str, Probe] = {}
        #: every (probe, result) this set has dispatched, in order. The engine
        #: owns the tool loop, so this is how a caller that needs the structured
        #: result gets at it without reaching into the loop.
        self.history: list[tuple[Probe, ProbeResult]] = []
        #: Everything the run has looked up, numbered, shared by reference with
        #: every view. `history` is *sliced* and so has to stay local to one
        #: stage call; this is only ever appended to and read whole, so sharing
        #: it is safe — and it is the only thing that survives a stage call,
        #: whose view is dropped when it returns. Without it the run's retrievals
        #: die with the sub-agents that made them.
        self.record: list[Retrieval] = [] if record is None else record
        #: Where retrievals are written as files, so an agent can read them back
        #: with the shell it already has. This replaces a `recall` tool: with a
        #: shell in hand there is no need for a bespoke way in, and `grep -l
        #: AD_FAILURE retrievals/` finds the entry that matters without knowing
        #: its number -- which is exactly the lookup that failed on the task
        #: where the loop retrieved the root cause and never cited it.
        self.record_dir: Path | None = None
        for probe in probes or []:
            self.add(probe)

    def add(self, probe: Probe) -> None:
        self.probes[probe.name] = probe

    def view(self, exclude: set[str] | None = None) -> ProbeSet:
        """A set with the same probes but its own history.

        `exclude` leaves named probes out -- the same probe objects otherwise,
        so a stage held away from the objective by config still shares every
        other tool, the record and the retrievals directory with the run.

        The falsifier reads `history[start:]` to find which probe backed a
        refutation. That slice is only meaningful if nothing else is appending
        to the same list — with concurrent attacks on one task it would pick up
        another attack's results, and since strength is the max over what backed
        it, a hypothesis could be killed at 0.98 because a *different* one failed
        a check. The probes themselves are shared deliberately: an attack must
        never be offered a different toolbox from the one the baselines got.
        """
        clone = ProbeSet(record=self.record)
        clone.probes = (
            {k: v for k, v in self.probes.items() if k not in exclude} if exclude else self.probes
        )
        # The record is shared, so ids stay unique and the files can share a
        # directory. Without this a view's retrievals were never written, while
        # the prompts told the agent to grep the directory for them.
        clone.record_dir = self.record_dir
        return clone

    def __len__(self) -> int:
        return len(self.probes)

    def __bool__(self) -> bool:
        return bool(self.probes)

    def __contains__(self, name: str) -> bool:
        return name in self.probes

    def get(self, name: str) -> Probe | None:
        return self.probes.get(name)

    def decisive(self) -> list[Probe]:
        return [p for p in self.probes.values() if p.decisive]

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Stable order: the tool list is part of the cache prefix, so it must
        not depend on dict insertion order across runs."""
        return [self.probes[name].tool_definition() for name in sorted(self.probes)]

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> ProbeResult:
        probe = self.probes.get(name)
        if probe is None:
            return ProbeResult(summary=f"no such probe: {name}", ok=False)
        if probe.server_side:  # pragma: no cover - never dispatched locally
            return ProbeResult(summary=f"{name} runs server-side", ok=False)
        try:
            result = await probe.run(**arguments)
        except Exception as e:
            result = ProbeResult(summary=f"{type(e).__name__}: {e}", ok=False)
        self.history.append((probe, result))
        if not probe.transient:
            entry = Retrieval(f"r{len(self.record) + 1}", probe, dict(arguments), result)
            self.record.append(entry)
            self._write(entry)
        return result

    def _write(self, entry: Retrieval) -> None:
        """Drop one retrieval into the run's directory, for the shell to find.

        Best effort on purpose: a full disk or a read-only path must not take
        down a paid run over a convenience. The file is what the agent reads, so
        it carries the query as well as the result -- a result whose query you
        cannot see is a citation of nothing.
        """
        if self.record_dir is None:
            return
        try:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / f"{entry.id}.json").write_text(
                json.dumps(
                    {
                        "id": entry.id,
                        "tool": entry.probe.name,
                        "arguments": entry.arguments,
                        "summary": entry.result.summary,
                        "ok": entry.result.ok,
                        "evidence": entry.result.evidence,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError:
            self.record_dir = None  # stop trying; one warning's worth of noise


class RecallProbe(Probe):
    """Read an entry out of the run's own record, in full, without re-running it.

    Every stage is shown an index of what the run has retrieved -- the query and
    what came back -- but only a fraction of the contents fit in a prompt. This
    is how a stage gets the rest.

    Re-running the query instead would be the obvious alternative and is worse
    in three ways: it spends a step, it reaches a live service that may answer
    differently, and it can fail. The content is already in memory. Recall is
    also `transient`, so reading the index never grows the index.

    It is offered to every method, like every other probe. For `direct` it is
    nearly useless -- a handful of calls all sit in its context already -- and
    that is the correct price for the invariant that makes the table comparable.
    """

    name = "recall"
    description = (
        "Read retrievals this run already made, in full, by their index ids "
        "(e.g. r12). Use it when the index shows a query whose contents you need "
        "and the summary is not enough. It re-reads what was found; it does not "
        "run anything."
    )
    transient = True

    def __init__(self, record: list[Retrieval], max_items: int = 5) -> None:
        # The live list, not a copy: entries are appended throughout the run and
        # a stage must be able to recall one made moments ago by another stage.
        self._record = record
        self.max_items = max_items

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Index ids to read, e.g. [\"r12\", \"r43\"].",
                }
            },
            "required": ["ids"],
        }

    async def run(self, ids: list[str] | str = (), **_: Any) -> ProbeResult:
        wanted = [ids] if isinstance(ids, str) else list(ids)
        if not wanted:
            return ProbeResult(summary="recall needs at least one id", ok=False)
        by_id = {r.id: r for r in self._record}
        found, missing = [], []
        for rid in wanted[: self.max_items]:
            hit = by_id.get(str(rid).strip())
            (found if hit else missing).append(hit or rid)
        if not found:
            return ProbeResult(
                summary=f"no such retrieval: {', '.join(str(m) for m in missing)}", ok=False
            )
        evidence: list[dict[str, Any]] = []
        for r in found:
            evidence.append({
                "kind": "recalled",
                "id": r.id,
                "probe": r.probe.name,
                "arguments": r.arguments,
                "summary": r.result.summary,
            })
            evidence.extend(r.result.evidence)
        note = f" ({len(missing)} unknown id(s) skipped)" if missing else ""
        return ProbeResult(
            summary=f"recalled {len(found)} retrieval(s){note}",
            evidence=evidence,
        )


# -- config-named probes --------------------------------------------------

#: Probes an experiment config can request by name, e.g. `params.probes: [web_search]`.
#: Benchmark-supplied probes are unioned with these.
PROBE_FACTORIES: dict[str, Callable[..., Probe]] = {
    "web_search": WebSearchProbe,
    "web_fetch": WebFetchProbe,
    "run_shell": ShellProbe,
    "run_python": PythonProbe,
    "run_sql": SqlProbe,
    "search_documents": DocSearchProbe,
}

#: What every run offers unless the config says otherwise: a way to look things
#: up, a way to read what it found, and a shell. Deliberately generic -- these
#: are capabilities an agent has anywhere, not an interface this harness invented
#: for one benchmark. Anything domain-shaped is the benchmark's to supply.
DEFAULT_PROBES: tuple[str, ...] = ("web_search", "web_fetch", "run_shell")


def build_probes(spec: Any) -> list[Probe]:
    """Turn a config `params.probes` value into probe instances.

    Accepts a list of names, or of {name: {kwargs}} objects:

        probes: [web_search, run_python]
        probes: [{web_search: {max_uses: 3, allowed_domains: [arxiv.org]}}]
    """
    out: list[Probe] = []
    for item in spec or []:
        if isinstance(item, str):
            name, kwargs = item, {}
        elif isinstance(item, dict) and len(item) == 1:
            name, kwargs = next(iter(item.items()))
            kwargs = kwargs or {}
        else:
            raise ValueError(f"bad probe spec: {item!r}")
        factory = PROBE_FACTORIES.get(name)
        if factory is None:
            raise ValueError(
                f"unknown probe {name!r}; available: {', '.join(sorted(PROBE_FACTORIES))}"
            )
        out.append(factory(**kwargs))
    return out
