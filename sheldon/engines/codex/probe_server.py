"""Serve `ctx.tools` to an external agent over MCP.

Codex runs as its own process with its own shell and file access, and cannot see
`ctx.tools`. That breaks the invariant the headline table rests on — every method
is offered the same probe set — so codex runs have always had to be reported in
their own row, measuring a different agent *and* a different toolbox at once.

This closes half of that. The probes are served over MCP streamable HTTP from
inside the running process, so codex calls the **same probe objects** every other
method holds: the same URLs, the same per-task budgets, the same
`ProbeSet.record`. What remains different is that codex also brings a shell, which
is a property of the agent rather than of what the harness offered it.

Serving the live objects rather than rebuilding them in the subprocess is the
whole point. A probe is a closure — it can bind a task's time window, or a
per-task budget that a concurrent fan-out is supposed to share — and a
reconstructed copy would silently hand out a second budget.

Only stdlib HTTP is used, matching the rest of the harness: a benchmark most runs
never touch should not add a dependency. The server binds an ephemeral port on
the loopback interface and is torn down with the call that started it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sheldon.core.probes import ProbeSet

#: The revision of MCP we answer as. Codex negotiates and accepts an older one.
PROTOCOL_VERSION = "2025-06-18"

#: How long one probe call may take before the bridge gives up on it, unless
#: the engine passes its own. Probes reach real services; a hung one must not
#: hold the agent's turn open forever. It has to be at least the shim's wait
#: (`engine_params.probe_timeout`): the bridge cancels the probe when it gives
#: up, so a shorter limit here threw away evaluations the shim was still
#: waiting for, and told the agent a slow program had "timed out".
CALL_TIMEOUT = 300.0


class _Handler(BaseHTTPRequestHandler):
    """One JSON-RPC request per POST. No SSE: every reply fits in a response."""

    server_version = "sheldon-probes/1"

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - silence stderr
        pass

    def _send(self, payload: dict[str, Any] | None, code: int = 200) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(code)
        if body:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _body(self) -> bytes:
        """The request body, however the client chose to frame it.

        Reading `Content-Length` alone is not enough: a chunked request carries
        no such header, so this read zero bytes, the empty body parsed as `{}`,
        and a `tools/call` -- having no `id` -- looked exactly like a
        notification. The server answered 202 and the client waited for a reply
        that was never coming, so every tool call hung until it timed out having
        never run, while `initialize` and `tools/list` went through untouched.
        That reads as a permissions problem and is a framing one.
        """
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()  # the trailing CRLF
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        try:
            raw = self._body()
            request = json.loads(raw) if raw.strip() else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._send({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": f"parse error: {e}"}}, 400)
            return

        # A notification has no id and takes no reply -- 202 with an empty body.
        if "id" not in request:
            self._send(None, 202)
            return

        result = self.server.bridge.handle(request)  # type: ignore[attr-defined]
        self._send(result)

    def do_GET(self) -> None:  # noqa: N802 - the SSE channel we do not open
        # 404, not 405. A client probes `/.well-known/oauth-authorization-server`
        # before it will use the server, and 405 says "GET works here, but not
        # this way" -- which reads as an auth endpoint that is misbehaving rather
        # than one that does not exist. This server is loopback-only and
        # unauthenticated by design.
        self.send_response(404)
        self.end_headers()

    def do_DELETE(self) -> None:  # noqa: N802 - session teardown
        self.send_response(200)
        self.end_headers()


class ProbeBridge:
    """An MCP server over one `ProbeSet`, live in this process."""

    def __init__(
        self, probes: ProbeSet, loop: asyncio.AbstractEventLoop, trace: Any = None,
        timeout: float = CALL_TIMEOUT,
    ) -> None:
        self.probes = probes
        self.timeout = timeout
        # The bridge dispatches outside the engine's tool loop, so nothing else
        # writes these to the trace. Without it a codex run's probe calls exist
        # only as a count, invisible,
        # while every other engine records each one. `sheldon show` is then blank
        # for the one engine whose steps we cannot otherwise see, since its model
        # calls never reach `LLM._trace_call` either.
        self.trace = trace
        # Probes are coroutines and the HTTP handler runs on its own thread, so
        # every dispatch is submitted back to the loop that owns them.
        self.loop = loop
        self.calls: list[str] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> str:
        """Bind an ephemeral loopback port and serve. Returns the MCP URL."""
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        httpd.bridge = self  # type: ignore[attr-defined]
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{httpd.server_address[1]}/mcp"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self) -> ProbeBridge:
        self.url = self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # -- protocol ----------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        rid, method = request.get("id"), request.get("method")
        try:
            result = self._dispatch(method, request.get("params") or {})
        except Exception as e:  # noqa: BLE001 - a bridge crash must not kill the run
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
        if result is None:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"no such method: {method}"}}
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _dispatch(self, method: str | None, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "sheldon-probes", "version": "1"},
            }
        if method in ("ping", "notifications/initialized"):
            return {}
        if method == "tools/list":
            return {"tools": [
                # MCP spells it `inputSchema`; our tool definitions are built in
                # Anthropic shape, which spells it `input_schema`.
                {"name": t["name"],
                 "description": t.get("description", ""),
                 "inputSchema": t.get("input_schema") or {"type": "object", "properties": {}}}
                for t in self.probes.tool_definitions()
            ]}
        if method == "tools/call":
            return self._call(params.get("name") or "", params.get("arguments") or {})
        return None

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        future = asyncio.run_coroutine_threadsafe(
            self.probes.dispatch(name, arguments), self.loop
        )
        try:
            result = future.result(timeout=self.timeout)
        except TimeoutError:
            future.cancel()
            return {"content": [{"type": "text", "text": f"{name} timed out after {self.timeout:.0f}s"}],
                    "isError": True}
        if self.trace is not None:
            self.trace.log(
                "tool", probe=name, ok=result.ok, summary=result.summary,
                via="mcp",
            )
        return {
            "content": [{"type": "text", "text": result.render()}],
            # A probe that could not run is not a probe that found nothing, and
            # the agent has to be able to tell those apart.
            "isError": not result.ok,
        }
