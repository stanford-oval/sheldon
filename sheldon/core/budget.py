"""Per-task spending limits.

The point is that a run can never quietly cost 40x what you expected because a
loop failed to converge. Every method checks `budget.ok()` at the top of each
round and the LLM wrapper charges the budget on every call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sheldon.core.types import Usage


class BudgetExceeded(RuntimeError):
    """Raised when a method keeps spending after the budget is gone."""


@dataclass
class Budget:
    """Limits for a single task. `None` means unlimited on that axis."""

    max_rounds: int | None = 4
    max_calls: int | None = 40
    max_output_tokens: int | None = None
    max_usd: float | None = 1.0
    #: Wall-clock seconds per task. A spend cap, not a kill switch: once it is
    #: reached no stage starts, and the required closing stage still runs --
    #: the same shape as upstream's agent timeout, where whatever report was
    #: written by the deadline is what gets graded.
    max_seconds: float | None = None
    spent: Usage = field(default_factory=Usage)
    rounds_used: int = 0
    started: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def charge(self, usage: Usage) -> None:
        self.spent += usage

    def start_round(self) -> None:
        self.rounds_used += 1

    def spend_reason(self) -> str | None:
        """Why we can't afford another *call*, or None. Rounds are not counted:
        `start_round()` fires before a round's work happens, so folding rounds
        in here would make every mid-round check report exhaustion during the
        last round and silently skip the falsifier's critic."""
        if self.max_calls is not None and self.spent.calls >= self.max_calls:
            return f"max_calls={self.max_calls}"
        if (
            self.max_output_tokens is not None
            and self.spent.output_tokens >= self.max_output_tokens
        ):
            return f"max_output_tokens={self.max_output_tokens}"
        if self.max_usd is not None and self.spent.cost_usd >= self.max_usd:
            return f"max_usd={self.max_usd}"
        if self.max_seconds is not None and self.elapsed() >= self.max_seconds:
            return f"max_seconds={self.max_seconds:g}"
        return None

    def round_reason(self) -> str | None:
        """Why we can't start another round, or None."""
        if self.max_rounds is not None and self.rounds_used >= self.max_rounds:
            return f"max_rounds={self.max_rounds}"
        return None

    def reason(self) -> str | None:
        """Why the loop should stop. Check this at the top of a round."""
        return self.round_reason() or self.spend_reason()

    def calls_left(self) -> int:
        """Headroom in calls. Unbounded budgets report a large finite number so
        a caller sizing a batch against it does not have to special-case None.

        Needed because k concurrent attacks all pass the same `ok()` check
        before any of them has spent anything, so the overshoot is bounded only
        by not opening more attacks than the budget can pay for.
        """
        if self.max_calls is None:
            return 1_000_000
        return max(0, self.max_calls - self.spent.calls)

    def usd_left(self) -> float | None:
        """Dollars before `max_usd` binds, or None where it is unset. Zero once
        it has bound; never negative, since a run can overshoot by one call."""
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.spent.cost_usd)

    def ok(self) -> bool:
        """Is there room for one more call? Check this mid-round."""
        return self.spend_reason() is None

    def require(self) -> None:
        why = self.spend_reason()
        if why is not None:
            raise BudgetExceeded(why)

    def fresh(self) -> Budget:
        """A copy with spending reset — one per task."""
        return Budget(
            max_rounds=self.max_rounds,
            max_calls=self.max_calls,
            max_output_tokens=self.max_output_tokens,
            max_usd=self.max_usd,
            max_seconds=self.max_seconds,
        )
