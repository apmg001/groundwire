"""Map exceptions to OpenAI-compatible JSON error responses.

Three layers, most specific first:

* :class:`GroundwireError` -> its own ``http_status`` and payload.
* Request validation errors -> 400 with field-level messages.
* Anything else -> 500 with a generic message. The traceback goes to the logs
  only; internals are never leaked to clients.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from groundwire.core.errors import GroundwireError, InvalidRequestError, OverloadedError

logger = logging.getLogger(__name__)


async def _groundwire_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GroundwireError)
    level = logging.ERROR if exc.http_status >= 500 else logging.INFO
    logger.log(level, "request failed", extra={"error_code": exc.code,
                                                "status": exc.http_status})
    headers = ({"Retry-After": str(exc.retry_after_s)} if isinstance(exc, OverloadedError)
               else None)
    return JSONResponse(status_code=exc.http_status, content=exc.to_payload(), headers=headers)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = [
        {"loc": ".".join(str(p) for p in err.get("loc", ())), "msg": err.get("msg", "")}
        for err in exc.errors()
    ]
    error = InvalidRequestError("request validation failed", details={"errors": problems})
    return JSONResponse(status_code=error.http_status, content=error.to_payload())


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(exc.detail), "type": "http_error",
                           "code": f"http_{exc.status_code}"}},
    )


async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "internal server error", "type": "internal_error",
                           "code": "internal_error"}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to ``app``.

    Args:
        app: FastAPI application.
    """
    app.add_exception_handler(GroundwireError, _groundwire_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)
