"""HTTP middleware: correlation ids, access logs, security headers, error bodies.

* **Request ids.** A valid incoming ``X-Request-ID`` is kept (so a request can
  be traced from the browser, through this service, into Groundwire);
  otherwise one is generated. It is bound to the logging context and echoed
  in the response and in every error body.
* **Errors.** :class:`MoneyAgentError` becomes its public message and status;
  validation errors become 400 with field names; anything unexpected becomes
  a generic 500, with the traceback in the log only. No stack trace, SQL or
  upstream text is ever sent to a client.
* **Security headers** on every response: no MIME sniffing, no framing, no
  referrer leakage, and a strict Content-Security-Policy (scripts and styles
  only from this origin).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from money_agent.core.errors import InvalidInputError, MoneyAgentError
from money_agent.core.logging import get_request_id, reset_request_id, set_request_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; "
                                "img-src 'self' data:; connect-src 'self'; "
                                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
}


def install(app: FastAPI) -> None:
    """Register the middleware and error handlers on ``app``.

    Args:
        app: The application.
    """

    @app.middleware("http")
    async def context(request: Request,
                      call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = set_request_id(request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            logger.info("request", extra={
                "method": request.method, "path": getattr(route, "path", request.url.path),
                "status": status, "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
            reset_request_id(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.exception_handler(MoneyAgentError)
    async def expected(_: Request, exc: MoneyAgentError) -> JSONResponse:
        log = logger.error if exc.http_status >= 500 else logger.info
        log("request failed", extra={"error_code": exc.code, "detail": str(exc)})
        return JSONResponse(exc.to_payload(get_request_id()), status_code=exc.http_status)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in err["loc"] if p != "body")
                         for err in exc.errors()})
        error = InvalidInputError("request validation failed", details={"fields": fields})
        return JSONResponse(error.to_payload(get_request_id()), status_code=400)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse(MoneyAgentError("unhandled").to_payload(get_request_id()),
                            status_code=500)
