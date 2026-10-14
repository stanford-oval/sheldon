"""Async model wrapper: caching, cost accounting, structured output, tools.

Deliberately thin. Methods should not import a provider SDK directly —
everything goes through `LLM.complete` / `LLM.converse` so that every call is
cached, budgeted, traced, and swappable for `FakeLLM` in tests.

Two backends sit behind that interface:

  anthropic   the Anthropic SDK, direct
  openai      any OpenAI-compatible endpoint, which is how a LiteLLM proxy is
              reached (`OPENAI_BASE_URL` + `OPENAI_API_KEY`)

Requests are built in Anthropic shape because that is what the rest of this repo
speaks; the OpenAI backend translates on the way out and back. `provider: auto`
picks the proxy when `OPENAI_BASE_URL` is set and the Anthropic SDK otherwise.

One capability does not survive the trip: `web_search` and `web_fetch` are
Anthropic *server* tools with no OpenAI-compatible equivalent. The OpenAI
backend drops them and says so, rather than letting a run quietly proceed with a
falsifier that cannot browse.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from sheldon.core.cache import DiskCache, cache_key
from sheldon.core.trace import current_trace
from sheldon.core.types import Usage

log = logging.getLogger(__name__)

#: How long to wait for the *connection*, and for the whole request.
#:
#: Both SDKs default to `connect=5s`, which is a coin flip on a machine that is
#: also doing work. With many CPU-bound subprocesses beside the event loop a TLS
#: handshake regularly takes longer than five seconds; all five attempts (one
#: plus `max_retries=4`) then fail inside ~30s and the task dies with
#: `APITimeoutError: Request timed out`, which reads like the endpoint is down
#: when nothing is wrong with it.
#:
#: Not removed, and not set to None: a genuinely hung request would then hang
#: the sweep with no upper bound, and a run that never ends is worse than a task
#: that fails loudly. The read budget is generous instead — a high-effort call
#: with a big `max_tokens` can legitimately take minutes.
#:
#: Environment variables rather than config, deliberately: a timeout changes how
#: a run behaves when the machine is busy, never what the model returns, so it
#: belongs with `SHELDON_CACHE` and `SHELDON_DATA` and out of `config_hash`.
CONNECT_TIMEOUT = float(os.environ.get("SHELDON_CONNECT_TIMEOUT", "30"))
REQUEST_TIMEOUT = float(os.environ.get("SHELDON_REQUEST_TIMEOUT", "900"))


def client_timeout() -> Any:
    """The timeout both backends are built with. `openai.Timeout` is what the
    installed `anthropic` re-exports, so one construction serves both."""
    from openai import Timeout  # noqa: PLC0415

    return Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)


# $ per million tokens, (input, output). Cache reads bill at ~0.1x input,
# cache writes at ~1.25x. Update alongside the model list.
PRICING: dict[str, tuple[float, float]] = {
    # Served through an OpenAI-compatible proxy.
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.5": (2.50, 15.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    # Direct Anthropic.
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

DEFAULT_MODEL = "gpt-5.6-terra"

# Models where thinking is adaptive and sampling params are rejected.
# Everything current falls here; kept explicit so an older model still works.
_ADAPTIVE_THINKING = {
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
}


def estimate_cost(model: str, usage: Usage) -> float:
    # An unknown model prices at the full-model rate rather than free: a cost
    # column that silently reads zero is worse than one that reads high.
    inp, out = PRICING.get(model, (2.50, 15.00))
    return (
        usage.input_tokens * inp
        + usage.cache_read_tokens * inp * 0.1
        + usage.cache_write_tokens * inp * 1.25
        + usage.output_tokens * out
    ) / 1_000_000


@dataclass
class LLMResult:
    text: str
    usage: Usage
    model: str
    parsed: Any | None = None
    stop_reason: str | None = None
    refusal: str | None = None
    cached: bool = False
    #: raw content blocks as plain dicts, so they can be echoed back into a
    #: multi-turn conversation and still be JSON-serializable for the cache
    content: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    result: Any = None
    is_error: bool = False


@dataclass
class ConverseResult:
    """The outcome of a tool-calling loop."""

    text: str
    messages: list[dict[str, Any]]
    usage: Usage
    model: str
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    stopped_by: str | None = None

    def calls_to(self, name: str) -> list[ToolCall]:
        return [c for c in self.tool_calls if c.name == name]


def normalize_base_url(url: str) -> str:
    """Turn a pasted endpoint into what an OpenAI client wants.

    People paste the full completions URL out of a proxy dashboard; the SDK
    wants the part before `/chat/completions` and appends the rest itself.
    """
    url = (url or "").rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


class Backend(ABC):
    """One provider. Requests arrive in Anthropic shape; translate as needed."""

    name: str = ""
    supports_server_tools: bool = True

    @abstractmethod
    async def call(self, req: dict[str, Any], schema: type[BaseModel] | None) -> LLMResult: ...

    async def aclose(self) -> None:
        """Release the HTTP pool. Must run *inside* the event loop that opened
        it — leaving it to interpreter teardown makes the SDK close its
        transport against a loop that is already gone, which surfaces as a
        `RuntimeError: Event loop is closed` traceback after a clean run."""
        client = getattr(self, "client", None)
        if client is not None:
            await client.close()


class AnthropicBackend(Backend):
    name = "anthropic"
    supports_server_tools = True

    def __init__(self, api_key: str | None = None, max_retries: int = 4) -> None:
        import anthropic  # noqa: PLC0415

        # Zero-arg construction picks up ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile. Don't require a key to exist here.
        kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": client_timeout()}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = anthropic.AsyncAnthropic(**kwargs)

    async def call(self, req: dict[str, Any], schema: type[BaseModel] | None) -> LLMResult:
        if schema is not None:
            resp = await self.client.messages.parse(output_format=schema, **req)
            parsed = getattr(resp, "parsed_output", None)
        else:
            if req["max_tokens"] >= 32000:
                # Large max_tokens needs streaming or the request can time out.
                async with self.client.messages.stream(**req) as stream:
                    resp = await stream.get_final_message()
            else:
                resp = await self.client.messages.create(**req)
            parsed = None

        content = [b.model_dump() for b in resp.content]
        text = "".join(b.get("text") or "" for b in content if b.get("type") == "text")
        u = resp.usage
        usage = Usage(
            calls=1,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        usage.cost_usd = estimate_cost(resp.model, usage)

        refusal = None
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            refusal = getattr(details, "category", None) or "refusal"

        return LLMResult(
            text=text,
            usage=usage,
            model=resp.model,
            parsed=parsed,
            stop_reason=resp.stop_reason,
            refusal=refusal,
            content=content,
        )


class OpenAIBackend(Backend):
    """Any OpenAI-compatible chat-completions endpoint — including a LiteLLM proxy.

    Translates the Anthropic-shaped request this repo builds into chat
    completions and back. Three things do not survive and are dropped on
    purpose rather than silently mangled:

      server tools     `web_search` / `web_fetch` have no equivalent; dropped
                       with a warning, so a browsing run fails loudly
      thinking/effort  no portable equivalent across proxied models
      cache_control    OpenAI-compatible endpoints cache automatically
    """

    name = "openai"
    supports_server_tools = False

    _STOP_REASONS = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None, max_retries: int = 4
    ) -> None:
        from openai import AsyncOpenAI  # noqa: PLC0415

        base = normalize_base_url(base_url or os.environ.get("OPENAI_BASE_URL", ""))
        self.client = AsyncOpenAI(
            base_url=base or None,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "unused",
            max_retries=max_retries,
            timeout=client_timeout(),
        )
        self._warned_server_tools: set[str] = set()

    # -- request translation ---------------------------------------------

    def _messages(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if req.get("system"):
            out.append({"role": "system", "content": req["system"]})
        for message in req["messages"]:
            role, content = message.get("role"), message.get("content")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                texts, calls = [], []
                for block in content or []:
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        call: dict[str, Any] = {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                        # Gemini's OpenAI-compatible endpoint signs each
                        # function call (`extra_content.google.thought_signature`)
                        # and refuses the next turn unless the signature comes
                        # back on the call it signed.
                        if block.get("extra_content"):
                            call["extra_content"] = block["extra_content"]
                        calls.append(call)
                entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue
            # A user turn carrying tool results becomes one `tool` message per
            # result, in order, followed by any free text.
            texts = []
            for block in content or []:
                if block.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": str(block.get("content", "")),
                        }
                    )
                elif block.get("type") == "text":
                    texts.append(block.get("text", ""))
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        return out

    def _tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        converted, dropped = [], []
        for tool in tools or []:
            if "input_schema" in tool:
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool["input_schema"],
                        },
                    }
                )
            else:
                dropped.append(tool.get("name") or tool.get("type") or "?")
        for name in dropped:
            if name not in self._warned_server_tools:
                self._warned_server_tools.add(name)
                log.warning(
                    "dropping server-side tool %r: it has no OpenAI-compatible equivalent. "
                    "Any probe relying on it will not run.",
                    name,
                )
        return converted

    def _params(self, req: dict[str, Any], schema: type[BaseModel] | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": req["model"],
            "messages": self._messages(req),
            "max_tokens": req["max_tokens"],
        }
        tools = self._tools(req.get("tools"))
        if tools:
            params["tools"] = tools
        if schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        return params

    # -- the call ---------------------------------------------------------

    async def call(self, req: dict[str, Any], schema: type[BaseModel] | None) -> LLMResult:
        params = self._params(req, schema)
        try:
            resp = await self.client.chat.completions.create(**params)
        except Exception as e:
            # Not every proxied model accepts json_schema. Retry in plain JSON
            # mode before giving up — the schema still gets validated below.
            if schema is None or not _is_response_format_error(e):
                raise
            log.info("json_schema rejected (%s); retrying in json_object mode", type(e).__name__)
            params["response_format"] = {"type": "json_object"}
            params["messages"] = [
                *params["messages"][:-1],
                _append_schema_hint(params["messages"][-1], schema),
            ]
            resp = await self.client.chat.completions.create(**params)

        choice = resp.choices[0]
        message = choice.message
        content: list[dict[str, Any]] = []
        text = message.content or ""
        if text:
            content.append({"type": "text", "text": text})
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            block: dict[str, Any] = {
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": arguments,
            }
            # Provider-specific fields the SDK does not model, kept so the
            # call can be echoed back whole (see `_messages`).
            extra = getattr(call, "extra_content", None) or (getattr(call, "model_extra", None) or {}).get("extra_content")
            if extra:
                block["extra_content"] = extra
            content.append(block)

        parsed = None
        if schema is not None and text:
            parsed = _parse_json_into(schema, text)

        return LLMResult(
            text=text,
            usage=self._usage(resp),
            model=getattr(resp, "model", req["model"]),
            parsed=parsed,
            stop_reason=self._STOP_REASONS.get(choice.finish_reason, choice.finish_reason),
            refusal="content_filter" if choice.finish_reason == "content_filter" else None,
            content=content,
        )

    @staticmethod
    def _usage(resp: Any) -> Usage:
        u = getattr(resp, "usage", None)
        prompt = getattr(u, "prompt_tokens", 0) or 0
        completion = getattr(u, "completion_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        # OpenAI counts cached tokens *inside* prompt_tokens; Anthropic reports
        # them separately. Subtract so the two providers' numbers mean the same
        # thing in a report.
        usage = Usage(
            calls=1,
            input_tokens=max(0, prompt - cached),
            output_tokens=completion,
            cache_read_tokens=cached,
        )
        hidden = getattr(resp, "_hidden_params", None) or {}
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
        usage.cost_usd = (
            float(cost) if cost is not None else estimate_cost(getattr(resp, "model", ""), usage)
        )
        return usage


def _is_response_format_error(e: Exception) -> bool:
    blob = str(e).lower()
    return "response_format" in blob or "json_schema" in blob


def _append_schema_hint(message: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    hint = (
        "\n\nReply with JSON only, matching this schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    return {**message, "content": f"{message.get('content', '')}{hint}"}


def _parse_json_into(schema: type[BaseModel], text: str) -> BaseModel | None:
    """Validate model text into `schema`, digging out the object if it is fenced."""
    for candidate in (text, *re.findall(r"\{.*\}", text, re.DOTALL)):
        try:
            return schema.model_validate_json(candidate)
        except Exception:
            continue
    return None


def make_backend(
    provider: str = "auto", base_url: str | None = None, api_key: str | None = None
) -> Backend:
    """`auto` prefers an OpenAI-compatible endpoint when one is configured."""
    if provider == "auto":
        provider = "openai" if (base_url or os.environ.get("OPENAI_BASE_URL")) else "anthropic"
    if provider == "openai":
        return OpenAIBackend(base_url, api_key)
    if provider == "anthropic":
        return AnthropicBackend(api_key)
    raise ValueError(f"unknown provider {provider!r}; expected auto, anthropic, or openai")



#: Faults that are about the moment rather than the request: the connection,
#: the gateway, the provider's own dependencies. Matched on status code and
#: class name so neither SDK's exception tree has to be imported here.
_TRANSIENT_NAMES = (
    "APIConnectionError", "APITimeoutError", "InternalServerError",
    "RateLimitError", "ServiceUnavailableError", "OverloadedError",
)


def _transient(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 408 or status == 429 or status >= 500
    return type(e).__name__ in _TRANSIENT_NAMES


class LLM:
    """One shared instance per run; safe to call from many asyncio tasks."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = "high",
        max_tokens: int = 16000,
        thinking: bool = True,
        cache: DiskCache | None = None,
        concurrency: int = 8,
        sem: asyncio.Semaphore | None = None,
        max_retries: int = 4,
        trace_prompts: bool = True,
        provider: str = "auto",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self._backend_impl: Backend | None = None
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.cache = cache if cache is not None else DiskCache()
        # Shared when several cells run at once, so `concurrency` stays a
        # statement about load on the endpoint rather than per-cell load.
        self._sem = sem or asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self.trace_prompts = trace_prompts
        #: what this run cost to produce, cache or no cache — the comparable number
        self.total = Usage()
        #: what was actually spent at the provider this time
        self.billed = Usage()

    def _backend(self) -> Backend:
        if self._backend_impl is None:
            self._backend_impl = make_backend(self.provider, self.base_url, self.api_key)
        return self._backend_impl

    @property
    def supports_server_tools(self) -> bool:
        """False on an OpenAI-compatible endpoint: `web_search` / `web_fetch`
        are Anthropic server tools with nowhere to run."""
        return self._backend().supports_server_tools

    def _request(
        self,
        prompt: str,
        system: str | None,
        model: str,
        effort: str,
        max_tokens: int,
        thinking: bool,
        cache_prompt: bool,
    ) -> dict[str, Any]:
        """The exact API payload. Also the cache key, so any edit is a miss."""
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            req["system"] = system
        if model in _ADAPTIVE_THINKING:
            # Sampling params (temperature/top_p/top_k) are rejected on these
            # models; thinking depth is controlled by effort instead.
            if thinking:
                req["thinking"] = {"type": "adaptive"}
            req["output_config"] = {"effort": effort}
        if cache_prompt:
            # Auto-caches the longest stable prefix. Put anything volatile
            # (the specific question) at the end of the prompt, not the start.
            req["cache_control"] = {"type": "ephemeral"}
        return req

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        schema: type[BaseModel] | None = None,
        cache_prompt: bool = True,
        tag: str = "",
    ) -> LLMResult:
        """One model call. Returns cached results transparently.

        `schema` (a pydantic model) switches to structured output and populates
        `LLMResult.parsed` with a validated instance.
        """
        req = self._request(
            prompt,
            system,
            model or self.model,
            effort or self.effort,
            max_tokens or self.max_tokens,
            self.thinking if thinking is None else thinking,
            cache_prompt,
        )
        return await self._cached(req, schema, tag)

    async def converse(
        self,
        prompt: str,
        *,
        tools: list[dict[str, Any]],
        execute: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        system: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        max_iterations: int = 6,
        finish: str | None = None,
        stop_on: set[str] | None = None,
        should_continue: Callable[[int], bool] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
        history: list[dict[str, Any]] | None = None,
        tag: str = "",
    ) -> ConverseResult:
        """Run a tool-calling loop and return everything that happened.

        `history` continues an earlier transcript (the `messages` of a previous
        result): `prompt` becomes the next user turn, joined onto a trailing
        user message -- the tool results of the call that ended it -- rather
        than sent as a second consecutive user turn. The returned `messages`
        hold the whole transcript, so a caller can keep going.

        A manual loop rather than the SDK tool runner, on purpose: every turn
        has to pass through the disk cache and the per-task budget, and has to
        be substitutable by `FakeLLM` in tests. The runner owns its own loop and
        would bypass all three.

        `execute(name, input)` runs client-side tools. Server-side tools (web
        search) need no handler — the API runs them and their results arrive in
        the response. `stop_on` names tools whose invocation ends the loop.

        `finish` is what to say when `max_iterations` runs out mid-tool-call.
        That ceiling is ours, not the model's: the turn that hits it is pure
        `tool_use`, so `text` is empty and the caller gets no answer at all —
        the harness scoring its own cutoff as a wrong answer. With `finish` set,
        one more tool-free call asks for the answer from what was gathered.
        `should_continue(iteration)` is the budget hook; returning False stops
        before spending another call. Pair it with `on_usage`, which fires after
        every turn — without that the caller only learns the cost once the loop
        is over, and `should_continue` has nothing to gate on.
        """
        model = model or self.model
        messages = _continue(history or [], prompt)
        usage = Usage()
        calls: list[ToolCall] = []
        stopped_by: str | None = None
        last: LLMResult | None = None
        iteration = 0

        while iteration < max_iterations:
            if should_continue is not None and not should_continue(iteration):
                stopped_by = "budget"
                break
            iteration += 1

            req = self._request(
                prompt,
                system,
                model,
                effort or self.effort,
                max_tokens or self.max_tokens,
                self.thinking if thinking is None else thinking,
                cache_prompt=True,
            )
            req["messages"] = messages
            req["tools"] = tools

            last = await self._cached(req, None, tag)
            usage += last.usage
            if on_usage is not None:
                on_usage(last.usage)
            messages = [*messages, {"role": "assistant", "content": last.content}]

            if last.stop_reason == "pause_turn":
                # A server-side tool hit its iteration limit; resend to continue.
                continue
            if last.stop_reason != "tool_use":
                stopped_by = last.stop_reason
                break

            uses = [b for b in last.content if b.get("type") == "tool_use"]
            if not uses:
                stopped_by = "no_tool_use_blocks"
                break

            results: list[dict[str, Any]] = []
            terminal = False
            for use in uses:
                name, args = use.get("name", ""), use.get("input") or {}
                if stop_on and name in stop_on:
                    calls.append(ToolCall(name=name, input=args, result=None))
                    stopped_by, terminal = name, True
                    results.append(
                        {"type": "tool_result", "tool_use_id": use.get("id"), "content": "recorded"}
                    )
                    continue
                if execute is None:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": use.get("id"),
                            "content": f"no handler for {name}",
                            "is_error": True,
                        }
                    )
                    continue
                try:
                    out = await execute(name, args)
                    is_error = False
                except Exception as e:
                    out, is_error = f"{type(e).__name__}: {e}", True
                calls.append(ToolCall(name=name, input=args, result=out, is_error=is_error))
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": use.get("id"),
                    "content": str(out)[:8000],
                }
                if is_error:
                    block["is_error"] = True
                results.append(block)

            # All results for one assistant turn go back in a single user
            # message; splitting them teaches the model to stop calling tools
            # in parallel.
            messages = [*messages, {"role": "user", "content": results}]
            if terminal:
                break
        else:
            stopped_by = "max_iterations"

        if (
            finish
            and stopped_by == "max_iterations"
            and not (last.text if last else "").strip()
            and (should_continue is None or should_continue(iteration))
        ):
            # Append to the trailing tool_result message rather than adding a
            # second user turn — consecutive same-role messages are not
            # portable across providers.
            tail = messages[-1] if messages else None
            if tail and tail.get("role") == "user" and isinstance(tail.get("content"), list):
                messages = [
                    *messages[:-1],
                    {**tail, "content": [*tail["content"], {"type": "text", "text": finish}]},
                ]
            else:
                messages = [*messages, {"role": "user", "content": finish}]

            req = self._request(
                prompt,
                system,
                model,
                effort or self.effort,
                max_tokens or self.max_tokens,
                self.thinking if thinking is None else thinking,
                cache_prompt=True,
            )
            req["messages"] = messages
            last = await self._cached(req, None, tag)
            usage += last.usage
            if on_usage is not None:
                on_usage(last.usage)
            messages = [*messages, {"role": "assistant", "content": last.content}]
            iteration += 1
            stopped_by = "max_iterations+finish"

        return ConverseResult(
            text=last.text if last else "",
            messages=messages,
            usage=usage,
            model=last.model if last else model,
            stop_reason=last.stop_reason if last else None,
            tool_calls=calls,
            iterations=iteration,
            stopped_by=stopped_by,
        )

    async def _cached(
        self, req: dict[str, Any], schema: type[BaseModel] | None, tag: str
    ) -> LLMResult:
        """Cache-wrapped API call. The key is the whole request, so a changed
        prompt, tool list, or tool result naturally forks the cache."""
        schema_name = f"{schema.__module__}.{schema.__qualname__}" if schema else None
        key = cache_key(
            {"req": req, "schema": schema_name, "provider": self._backend().name, "v": 3}
        )

        hit = self.cache.get(key)
        if hit is not None:
            # The tokens the call *cost*, replayed from the cache entry — not
            # zero. Two reasons, and the second is the serious one.
            #
            # Reporting: a replayed run otherwise looks free, and `$/task` stops
            # being comparable across runs the moment one of them reuses a
            # cache. What a method costs is a property of the method.
            #
            # Behaviour: `Budget` is charged from this, so with zero here a
            # replay never depletes its budget and can run further than the
            # live run did. A cached replay has to behave identically or it is
            # not a replay.
            #
            # What was actually *billed* this time is tracked separately, on
            # `LLM.billed`.
            result = LLMResult(
                text=hit["text"],
                usage=Usage(**(hit.get("usage") or {})),
                model=hit.get("model", req["model"]),
                stop_reason=hit.get("stop_reason"),
                refusal=hit.get("refusal"),
                cached=True,
                content=hit.get("content") or [],
            )
            if schema is not None and hit.get("parsed") is not None:
                result.parsed = schema.model_validate(hit["parsed"])
            self.total += result.usage
            self._trace_call(req, result, key, tag)
            return result

        async with self._sem:
            result = await self._api_call(req, schema)
        self._trace_call(req, result, key, tag)

        self.total += result.usage
        self.billed += result.usage
        self.cache.put(
            key,
            {
                "text": result.text,
                "parsed": result.parsed.model_dump() if result.parsed is not None else None,
                "model": result.model,
                "stop_reason": result.stop_reason,
                "refusal": result.refusal,
                "content": result.content,
                "usage": result.usage.to_dict(),
            },
            tag=tag,
        )
        return result

    def _trace_call(
        self, req: dict[str, Any], result: LLMResult, key: str, tag: str
    ) -> None:
        """Write the call itself into the task's trace: what went out, what came
        back, verbatim.

        The summaries a method logs — `chars=0`, `sampled=5 distinct=3` — say
        that a call happened, not what it said. That is enough to notice
        something is wrong and never enough to see what. `key` is the cache key,
        so even with `trace_prompts` off the whole exchange stays recoverable
        from `.cache/llm.db`.
        """
        trace = current_trace()
        if trace is None:
            return
        event: dict[str, Any] = {
            "tag": tag,
            "model": result.model,
            "cached": result.cached,
            "key": key,
            "stop_reason": result.stop_reason,
            "text": result.text,
            "usage": result.usage.to_dict(),
        }
        if result.refusal:
            event["refusal"] = result.refusal
        if result.parsed is not None:
            # A schema-backed call's real payload. The OpenAI path happens to
            # leave the JSON in `text` as well; the Anthropic one need not.
            dump = getattr(result.parsed, "model_dump", None)
            event["parsed"] = dump() if dump else result.parsed
        blocks = result.content or []
        tool_use = [b for b in blocks if b.get("type") == "tool_use"]
        if tool_use:
            event["tool_use"] = [
                {"name": b.get("name"), "input": b.get("input")} for b in tool_use
            ]
        thinking = [b.get("thinking") for b in blocks if b.get("type") == "thinking"]
        if thinking:
            event["thinking"] = thinking
        if self.trace_prompts:
            event["system"] = req.get("system")
            event["messages"] = req["messages"]
            if req.get("tools"):
                event["tools_offered"] = sorted(t.get("name", "") for t in req["tools"])
        trace.log("llm", **event)

    async def _api_call(self, req: dict[str, Any], schema: type[BaseModel] | None) -> LLMResult:
        """The one place that talks to a provider. `FakeLLM` overrides this.

        The SDK already retries 5xx and 429, but its whole ladder is over in a
        few seconds, which is the wrong timescale for a run that lives for
        hours: a transient 503 that outlasts the SDK's five attempts costs the
        run everything after it. The outer ladder waits minutes instead: a provider
        blip should cost a pause, never a run.

        Only transport and server-side faults wait. A 400, a refusal or a bad
        schema is the request's own fault and repeating it changes nothing.
        """
        delay = float(os.environ.get("SHELDON_RETRY_BASE", "20"))
        attempts = int(os.environ.get("SHELDON_RETRY_ATTEMPTS", "5"))
        for attempt in range(attempts):
            try:
                return await self._backend().call(req, schema)
            except Exception as e:  # noqa: BLE001 - the class is decided below
                if attempt == attempts - 1 or not _transient(e):
                    raise
                log.warning(
                    "provider error (%s), attempt %d/%d; waiting %.0fs: %s",
                    type(e).__name__, attempt + 1, attempts, delay, str(e)[:200],
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def aclose(self) -> None:
        """Close the provider connection, if one was ever opened."""
        if self._backend_impl is not None:
            await self._backend_impl.aclose()
            self._backend_impl = None

    def stats(self) -> dict[str, Any]:
        return {
            "usage": self.total.to_dict(),
            "billed": self.billed.to_dict(),
            "cache": self.cache.stats(),
            "provider": self._backend().name,
        }


class FakeLLM(LLM):
    """Deterministic stand-in so the whole pipeline runs with no API key.

    `make smoke` uses this. It is what makes a wiring bug a one-second failure
    instead of a five-minute, forty-dollar one.

    Two hooks drive the tool loop:

      `tool_responder(prompt, tools)` returns {"name", "input"} to emit a
      `tool_use` block, or None to end the turn. Stateless, so it is safe when
      many tasks run concurrently against one instance.

      `tool_script` is a fixed sequence of calls, popped one per turn. Precise,
      but order-dependent — use it only in single-task tests.
    """

    def __init__(
        self,
        responder: Any = None,
        tool_script: list[dict[str, Any]] | None = None,
        tool_responder: Any = None,
        **kw: Any,
    ) -> None:
        kw.setdefault("cache", DiskCache(enabled=False))
        super().__init__(**kw)
        self.responder = responder
        self.tool_script = list(tool_script or [])
        self.tool_responder = tool_responder
        self.prompts: list[str] = []
        #: the system turn of every call, beside `prompts`: a stage's standing
        #: instructions live there, and a test of them has to be able to read it
        self.systems: list[str] = []
        self.tool_turns = 0

    async def _api_call(self, req: dict[str, Any], schema: type[BaseModel] | None) -> LLMResult:
        prompt = _last_user_text(req["messages"])
        self.prompts.append(prompt)
        self.systems.append(str(req.get("system") or ""))

        step = None
        if req.get("tools"):
            if self.tool_script:
                step = self.tool_script.pop(0)
            elif self.tool_responder is not None:
                step = self.tool_responder(prompt, req["tools"])
        if step is not None:
            self.tool_turns += 1
            content = [
                {
                    "type": "tool_use",
                    "id": f"toolu_fake_{self.tool_turns}",
                    "name": step["name"],
                    "input": step.get("input", {}),
                }
            ]
            usage = Usage(calls=1, input_tokens=len(prompt) // 4, output_tokens=16)
            usage.cost_usd = estimate_cost(self.model, usage)
            return LLMResult(text="", usage=usage, model=self.model,
                             stop_reason="tool_use", content=content)

        text = self.responder(prompt, schema) if self.responder else "fake"
        parsed = None
        if schema is not None:
            if isinstance(text, schema):
                parsed, text = text, text.model_dump_json()
            elif isinstance(text, dict):
                parsed, text = schema.model_validate(text), "{}"
            else:
                parsed = _empty_instance(schema)
                text = parsed.model_dump_json()
        usage = Usage(calls=1, input_tokens=len(prompt) // 4, output_tokens=len(str(text)) // 4)
        usage.cost_usd = estimate_cost(self.model, usage)
        return LLMResult(
            text=str(text),
            usage=usage,
            model=self.model,
            parsed=parsed,
            stop_reason="end_turn",
            content=[{"type": "text", "text": str(text)}],
        )


def _continue(history: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """`history` with `prompt` as the next user turn, merged into a trailing
    user message since consecutive same-role turns are not portable."""
    tail = history[-1] if history else None
    if tail is None or tail.get("role") != "user":
        return [*history, {"role": "user", "content": prompt}]
    content = tail.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
    return [*history[:-1], {**tail, "content": [*blocks, {"type": "text", "text": prompt}]}]


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Flatten the most recent user turn to text, for FakeLLM and for logging."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        parts = []
        for block in content or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                parts.append(str(block.get("content", "")))
        return "\n".join(parts)
    return ""


def _empty_instance(schema: type[BaseModel]) -> BaseModel:
    """Best-effort default instance of a pydantic model, for FakeLLM."""
    values: dict[str, Any] = {}
    for name, f in schema.model_fields.items():
        if not f.is_required():
            continue
        ann = str(f.annotation)
        if "list" in ann or "List" in ann:
            values[name] = []
        elif "int" in ann:
            values[name] = 0
        elif "float" in ann:
            values[name] = 0.0
        elif "bool" in ann:
            values[name] = False
        elif "dict" in ann:
            values[name] = {}
        else:
            values[name] = ""
    return schema.model_validate(values)


def credentials_status() -> str:
    """What a live run would use. Not authoritative — an `ant auth login`
    profile works with no env var set."""
    if os.environ.get("OPENAI_BASE_URL"):
        host = os.environ["OPENAI_BASE_URL"].split("//")[-1].split("/")[0]
        keyed = "with key" if os.environ.get("OPENAI_API_KEY") else "NO OPENAI_API_KEY"
        return f"OpenAI-compatible endpoint at {host} ({keyed})"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "ANTHROPIC_API_KEY set"
    return "no env credentials (an `ant auth login` profile also works)"


def has_credentials() -> bool:
    return bool(
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
