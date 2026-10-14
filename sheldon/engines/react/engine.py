"""ReAct: Thought / Action / Observation as text, not as a tool protocol.

Why keep this alongside `toolloop`: the two fail differently. Native tool
calling is cleaner but is only as good as the endpoint's tool support — through
a proxy, tool schemas get rewritten, `strict` gets dropped, and some served
models have no tool support at all. ReAct needs nothing but text, so it is the
fallback that always works, and it is also the honest comparison point for
papers that report ReAct numbers.

The protocol, parsed leniently because models are inconsistent about it:

    Thought: ...
    Action: <tool name>
    Action Input: {"json": "arguments"}
    Observation: <filled in by us>
    ...
    Final Answer: ...
"""

from __future__ import annotations

import json
import re
from typing import Any

from sheldon.core.llm import ToolCall
from sheldon.core.registry import engines
from sheldon.core.types import RunContext, Usage
from sheldon.engines.base import Engine, EngineResult, tool_definitions
from sheldon.engines.toolloop.engine import FINISH

_ACTION = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.MULTILINE)
_ACTION_INPUT = re.compile(r"^\s*Action\s*Input\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|\Z)", re.MULTILINE | re.DOTALL)
_FINAL = re.compile(r"^\s*Final\s*Answer\s*:\s*(.*)\Z", re.MULTILINE | re.DOTALL)


@engines.register("react")
class ReActEngine(Engine):
    name = "react"
    blurb = "Thought/Action/Observation text protocol; needs no tool support"

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
        tools = tool_definitions(ctx, extra_tools)
        steps = max_steps if max_steps is not None else int(self.param("max_steps", 4))
        usage = Usage()
        calls: list[ToolCall] = []
        transcript = f"{prompt}\n\n{_protocol(tools)}\n"
        text, stopped_by = "", "max_steps"
        step = 0

        while step <= steps:
            step += 1
            if not ctx.budget.ok():
                stopped_by = "budget"
                break
            res = await ctx.llm.complete(transcript, system=system, model=model, tag=tag)
            usage += res.usage
            ctx.budget.charge(res.usage)
            reply = res.text
            transcript += reply

            final = _FINAL.search(reply)
            action = _ACTION.search(reply)
            if final or not action:
                text = final.group(1).strip() if final else reply.strip()
                stopped_by = "final_answer" if final else "no_action"
                break

            name = action.group(1).strip().strip("`\"'")
            raw = _ACTION_INPUT.search(reply)
            arguments = _parse_args(raw.group(1) if raw else "")

            if stop_on and name in stop_on:
                calls.append(ToolCall(name=name, input=arguments))
                text, stopped_by = reply.strip(), name
                break

            result = await ctx.tools.dispatch(name, arguments) if ctx.tools else None
            observation = result.render() if result else f"no such tool: {name}"
            if result is not None:
                calls.append(ToolCall(name=name, input=arguments, result=observation,
                                      is_error=not result.ok))
                ctx.trace.log("tool", probe=name, ok=result.ok, engine="react",
                              summary=result.summary)
            transcript += f"\nObservation: {observation[:4000]}\nThought:"
            text = reply.strip()

        if stopped_by == "max_steps" and ctx.budget.ok():
            # Same cutoff, same remedy as the tool loop: the step ceiling is
            # ours, so the text here is a Thought mid-reasoning, not an answer.
            # Ask once more, with the protocol withdrawn.
            transcript += f"\n\n{FINISH}\nFinal Answer:"
            res = await ctx.llm.complete(transcript, system=system, model=model, tag=tag)
            usage += res.usage
            ctx.budget.charge(res.usage)
            final = _FINAL.search(res.text)
            text = (final.group(1) if final else res.text).strip()
            stopped_by = "max_steps+finish"
            step += 1

        ctx.trace.log(
            "engine.done", engine=self.name, tag=tag,
            stopped_by=stopped_by, iterations=step,
            tool_calls=len(calls), chars=len(text or ""),
        )
        return EngineResult(
            text=text, usage=usage, tool_calls=calls, stopped_by=stopped_by, iterations=step
        )


def _protocol(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "Answer directly, ending with a line beginning `Final Answer:`."
    lines = ["You have these tools:"]
    for tool in tools:
        name = tool.get("name") or tool.get("type", "?")
        desc = tool.get("description", "")
        schema = tool.get("input_schema")
        lines.append(f"- {name}: {desc}")
        if schema:
            lines.append(f"  arguments: {json.dumps(schema.get('properties', {}))}")
    lines += [
        "",
        "Work in this format, one step at a time:",
        "",
        "Thought: what you are reasoning about",
        "Action: the tool name, exactly as written above",
        'Action Input: a JSON object of arguments',
        "",
        "Stop after `Action Input:` — the observation is filled in for you. When you are "
        "done, reply with `Final Answer:` and nothing after it but the answer.",
        "",
        "Thought:",
    ]
    return "\n".join(lines)


def _parse_args(raw: str) -> dict[str, Any]:
    """Models fence it, prefix it, and trail prose after it. Dig out the object."""
    raw = raw.strip()
    for candidate in (raw, *re.findall(r"\{.*\}", raw, re.DOTALL)):
        cleaned = candidate.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            parsed = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}
