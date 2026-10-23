"""ASGI middleware: request IDs, access logging, body-size limits and load shedding.

Written as plain ASGI middleware rather than Starlette's ``BaseHTTPMiddleware``
because the latter buffers/wraps streaming responses and interacts badly with
client disconnects. Plain ASGI passes SSE through untouched.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from groundwire.core.context import new_request_id, set_request_id
from groundwire.core.errors import GroundwireError, OverloadedError
from groundwire.resilience.overload import AdmissionController

logger = logging.getLogger("groundwire.access")

REQUEST_ID_HEADER = b"x-request-id"


class RequestContextMiddleware:
    """Assign a request ID, echo it in the response, and log one access line."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Args:
            app: Downstream ASGI app.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(REQUEST_ID_HEADER)
        request_id = new_request_id(incoming.decode("latin-1") if incoming else None)
        set_request_id(request_id)
        started = time.perf_counter()
        status_holder: dict[str, int] = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            if path not in ("/healthz", "/metrics"):  # keep probes out of logs
                logger.info("request completed", extra={
                    "method": scope.get("method"), "path": path,
                    "status": status_holder["status"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                })


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` with HTTP 413.

    Checks ``Content-Length`` up front and also counts streamed chunks, so a
    client cannot bypass the limit with chunked transfer encoding.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        """Wrap ``app``.

        Args:
            app: Downstream ASGI app.
            max_bytes: Maximum accepted body size.
        """
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > self.max_bytes
            except ValueError:
                too_big = False  # let the server/framework reject a bad header
            if too_big:
                await _send_413(send, self.max_bytes)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if not response_started:
                await _send_413(send, self.max_bytes)


class _BodyTooLargeError(Exception):
    """Internal signal: streamed body exceeded the limit."""


async def _send_413(send: Send, max_bytes: int) -> None:
    body: dict[str, Any] = {"error": {
        "message": f"request body exceeds {max_bytes} bytes",
        "type": "request_too_large", "code": "request_too_large"}}
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})


# --------------------------------------------------------------------------- #
# Load shedding
# --------------------------------------------------------------------------- #

SHED_PATHS = frozenset({"/v1/chat/completions"})
SHED_LOG_INTERVAL_S = 10.0


class AdmissionMiddleware:
    """Refuse new chat requests with HTTP 503 when the gateway is overloaded.

    Only ``POST`` to :data:`SHED_PATHS` is subject to admission control, so
    health probes, metrics and the reviewer API keep working under overload
    (a reviewer must be able to approve a held call precisely when the system
    is busy). The admitted request counts as in flight until its response,
    including a whole SSE stream, has been sent.

    A refusal is deliberately cheap: no body is read, no handler runs, and
    at most one warning is logged every :data:`SHED_LOG_INTERVAL_S` seconds
    (refusals are counted in ``groundwire_load_shed_total``).
    """

    def __init__(self, app: ASGIApp, *,
                 controller: Callable[[], AdmissionController | None]) -> None:
        """Wrap ``app``.

        Args:
            app: Downstream ASGI app.
            controller: Returns the admission controller, or ``None`` before
                startup has built it (requests then pass through).
        """
        self.app = app
        self._controller = controller
        self._last_log = 0.0
        self._shed_since_log = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection.

        Args:
            scope: ASGI scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        controller = self._controller() if scope["type"] == "http" else None
        if (controller is None or scope.get("method") != "POST"
                or scope.get("path") not in SHED_PATHS):
            await self.app(scope, receive, send)
            return
        try:
            admission = controller.admit()
            admission.__enter__()
        except OverloadedError as exc:
            self._note_shed(exc)
            await _send_error(send, exc,
                              extra_headers=[(b"retry-after", str(exc.retry_after_s).encode())])
            return
        try:
            await self.app(scope, receive, send)
        finally:
            admission.__exit__(None, None, None)

    def _note_shed(self, exc: OverloadedError) -> None:
        self._shed_since_log += 1
        now = time.monotonic()
        if now - self._last_log >= SHED_LOG_INTERVAL_S:
            logger.warning("shedding load", extra={"refused": self._shed_since_log,
                                                    "reason": exc.details.get("reason"),
                                                    "detail": exc.message})
            self._last_log, self._shed_since_log = now, 0


async def _send_error(send: Send, exc: GroundwireError, *,
                      extra_headers: list[tuple[bytes, bytes]] | None = None) -> None:
    payload = json.dumps(exc.to_payload()).encode()
    await send({"type": "http.response.start", "status": exc.http_status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode()),
                            *(extra_headers or [])]})
    await send({"type": "http.response.body", "body": payload})
