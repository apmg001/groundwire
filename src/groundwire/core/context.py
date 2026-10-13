"""Per-request context propagated through async code via ``contextvars``.

The request ID is set once by the HTTP middleware and is then visible to every
coroutine handling that request (router, providers, guardrails) without being
passed through every function signature. The logging filter reads it so every
log line can be correlated with a single request.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

_REQUEST_ID: ContextVar[str] = ContextVar("groundwire_request_id", default="-")

# Accept client-supplied IDs only if they are short and boring, so a caller
# cannot inject newlines or huge strings into our logs.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


def new_request_id(candidate: str | None = None) -> str:
    """Return a safe request ID, reusing the caller's if it is well-formed.

    Args:
        candidate: Value of an incoming ``X-Request-ID`` header, if any.

    Returns:
        ``candidate`` if it passes validation, otherwise a fresh ``gw-<hex>`` ID.
    """
    if candidate and _SAFE_ID.match(candidate):
        return candidate
    return f"gw-{uuid.uuid4().hex}"


def set_request_id(request_id: str) -> None:
    """Bind ``request_id`` to the current async context.

    Args:
        request_id: ID to bind.
    """
    _REQUEST_ID.set(request_id)


def get_request_id() -> str:
    """Return the request ID bound to the current context, or ``"-"``."""
    return _REQUEST_ID.get()
