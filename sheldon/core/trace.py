"""Structured per-task traces.

Every model call, every hypothesis, every falsification attempt, every belief
update lands in `runs/<run_id>/traces/<task_id>.jsonl`. When a method scores
badly the trace is how you find out whether the loop proposed the wrong thing,
or proposed the right thing and then talked itself out of it.
"""

from __future__ import annotations

import contextvars
import json
import time
from pathlib import Path
from typing import Any

# Set by the runner around each task. The LLM is shared across concurrent tasks
# and so cannot hold a trace of its own, but each task runs in its own asyncio
# Task and therefore its own context — so a call made anywhere below finds the
# right trace without every signature in between having to carry one.
_current: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "sheldon_trace", default=None
)


def current_trace() -> Trace | None:
    return _current.get()


def use_trace(trace: Trace | None) -> contextvars.Token:
    return _current.set(trace)


def reset_trace(token: contextvars.Token) -> None:
    _current.reset(token)


class Trace:
    """Append-only event log scoped to one task."""

    def __init__(
        self,
        task_id: str,
        path: str | Path | None = None,
        echo: bool = False,
        keep_through_round: int | None = None,
    ) -> None:
        self.task_id = task_id
        self.path = Path(path) if path else None
        self.echo = echo
        self.events: list[dict[str, Any]] = []
        self._t0 = time.monotonic()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            kept = self._kept(keep_through_round)
            self.path.write_text("".join(kept), encoding="utf-8")
            if kept:
                self.log("trace.resumed", kept_events=len(kept), through_round=keep_through_round)

    def _kept(self, through_round: int | None) -> list[str]:
        """The earlier rounds' events, when a task is resumed from a round.

        A task that picks up from `round_N.json` has its spaces, its budget and
        its retrievals back, and used to lose the account of how it got there:
        the file was truncated on every start, so the per-round view of a
        resumed run opened at round N+1 with hypotheses nobody was seen to
        propose. Kept
        up to and including the `round.done` of the checkpointed round; what
        the dead process wrote after that belongs to a round the resumed one
        will run again, and keeping it would show that round twice.
        """
        if through_round is None or self.path is None or not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            return []
        last = -1
        for i, line in enumerate(lines):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "round.done" and e.get("round") == through_round:
                last = i
        return lines[: last + 1] if last >= 0 else []

    def log(self, kind: str, **fields: Any) -> None:
        event = {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **fields}
        self.events.append(event)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if self.echo:
            print(f"[{self.task_id}] {kind}: {json.dumps(fields, default=str)}")

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]

    def __enter__(self) -> Trace:
        self._token = use_trace(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        reset_trace(self._token)


class NullTrace(Trace):
    """For tests and one-off calls where nothing should hit disk."""

    def __init__(self) -> None:
        super().__init__(task_id="-", path=None, echo=False)
