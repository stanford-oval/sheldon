"""Baselines. Every number in the headline table needs one of these next to it.

Both get the same `ctx.tools` the loop gets, so they can run code and query
the task exactly as `sheldon` can. Without that, the headline comparison
measures tool access, not method.

`direct` is one question, with tool calls allowed. `best-of-n` samples it n
times and takes the most common answer — the cost-matched control, because a
loop that spends 5 calls should be compared against 5 calls of sampling.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from sheldon.core.registry import methods
from sheldon.core.types import Prediction, RunContext, Task, Usage
from sheldon.methods.base import Method, answer_format_hint, answer_from_text, ask, timer

SYSTEM = (
    "You solve problems carefully and end with a clearly marked final answer. "
    "If you have tools, use them."
)

#: The instructed arm. Off by default, because the naive baseline is the
#: in-the-wild measurement. Switched on with `-p test_rule=true`
#: it becomes the ablation that prices the loop -- one round of "state a rule,
#: run it against the evidence" with no belief state and no repair -- and it
#: gives the baseline a trajectory in the same unit the loop has, since every
#: sandbox call then defines a `transform` that can be replayed afterwards.
#: It is a prompt, not a tool, so the offered toolbox stays identical.
TEST_RULE = (
    "Before you answer, state your rule as code and test it. In the sandbox, "
    "define `transform(grid)` implementing the rule you believe, run it on every "
    "demonstration input (`demos` is preloaded), and compare each result with "
    "the demonstration output. Say which pairs it reproduces. If any pair fails, "
    "revise the rule and test again. When it reproduces every pair, or you are "
    "out of steps, apply it to `test_input`. In your final message include the "
    "final `transform` in a ```python block, then the answer."
)


def build_prompt(task: Task, ctx: RunContext, test_rule: bool) -> str:
    prompt = f"{task.question}\n\n{answer_format_hint(ctx)}"
    if test_rule:
        prompt = f"{prompt}\n\n{TEST_RULE}"
    return prompt


@methods.register("direct")
class Direct(Method):
    name = "direct"
    blurb = "single model call (baseline)"

    async def run(self, task: Task, ctx: RunContext) -> Prediction:
        test_rule = bool(self.param("test_rule", False))
        with timer() as t:
            prompt = build_prompt(task, ctx, test_rule)
            # A rule-and-test loop needs several calls per rule; the engine's
            # default step ceiling would bind first and the arm would look worse
            # for a reason that has nothing to do with reasoning. Only the
            # instructed arm sets it, so the naive baseline's budget is untouched.
            calls = int(self.param("test_rule_calls", 12)) if test_rule else None
            text, usage = await ask(ctx, prompt, system=SYSTEM, tag="direct",
                                    max_tool_calls=calls)
            ctx.trace.log("direct.answer", chars=len(text), test_rule=test_rule)
            answer = answer_from_text(ctx, task, text)
        return Prediction(
            task_id=task.id,
            answer=answer,
            attempts=[answer],
            rounds=1,
            usage=usage,
            wall_s=t.elapsed,
            extra={"raw": text[-2000:]},
        )


@methods.register("best-of-n", "bon")
class BestOfN(Method):
    name = "best-of-n"
    blurb = "n independent samples, majority vote (cost-matched baseline)"

    async def run(self, task: Task, ctx: RunContext) -> Prediction:
        n = int(self.param("n", 5))
        test_rule = bool(self.param("test_rule", False))
        calls = int(self.param("test_rule_calls", 12)) if test_rule else None
        with timer() as t:
            prompt = build_prompt(task, ctx, test_rule)
            usage = Usage()

            async def sample(i: int) -> tuple[Any, Usage]:
                if not ctx.budget.ok():
                    ctx.trace.log("budget.stop", where="best-of-n", reason=ctx.budget.reason())
                    return None, Usage()
                # Vary the prompt so the cache doesn't return the same sample n times.
                text, u = await ask(
                    ctx,
                    f"{prompt}\n\n(attempt {i + 1} of {n}; reason independently)",
                    system=SYSTEM,
                    tag="bon",
                    max_tool_calls=calls,
                )
                return answer_from_text(ctx, task, text), u

            # The samples are independent by definition — that is the whole
            # method — so there was never a reason to wait for one before
            # starting the next. Order is restored below so the vote, and the
            # tie-break that falls back to insertion order, stay deterministic.
            results = await asyncio.gather(*(sample(i) for i in range(n)))
            answers: list[Any] = []
            for parsed, u in results:
                usage += u
                if parsed is not None:
                    answers.append(parsed)
            ranked = _rank_by_frequency(answers)
            ctx.trace.log("bon.done", sampled=len(answers), distinct=len(ranked))
        return Prediction(
            task_id=task.id,
            answer=ranked[0] if ranked else None,
            attempts=ranked[: max(1, getattr(ctx.benchmark, "n_attempts", 1))],
            rounds=len(answers),
            usage=usage,
            wall_s=t.elapsed,
            # Every sample, not just the ones that won the vote. The gap between
            # "the right answer was sampled" and "the right answer was chosen"
            # is the difference between needing better hypotheses and needing a
            # better way to pick among them, and only this tells them apart.
            extra={"samples": answers},
        )


def _rank_by_frequency(answers: list[Any]) -> list[Any]:
    """Most-agreed-upon first. Falls back to insertion order for unhashables."""
    try:
        counts = Counter(_freeze(a) for a in answers)
    except TypeError:
        return answers
    seen: dict[Any, Any] = {}
    for a in answers:
        seen.setdefault(_freeze(a), a)
    return [seen[k] for k, _ in counts.most_common()]


def _freeze(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    return obj
