"""Shared HTTP plumbing for HTTP-based provider adapters.

Centralises three things every HTTP adapter needs and that are easy to get
subtly wrong:

* building a pooled :class:`httpx.AsyncClient` with sane timeouts and limits,
* translating ``httpx`` exceptions and HTTP status codes into the
  :mod:`groundwire.core.errors` provider taxonomy,
* parsing server-sent events (SSE) line by line.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from groundwire.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)

_MAX_ERROR_BODY_CHARS = 500


# --------------------------------------------------------------------------- #
# Client construction
# --------------------------------------------------------------------------- #


def build_client(
    *,
    base_url: str,
    timeout_s: float,
    max_connections: int,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create a pooled async HTTP client for one provider.

    Args:
        base_url: Provider base URL.
        timeout_s: Read/write/pool timeout; connect timeout is capped at 10 s
            so a dead host fails fast and we fail over quickly.
        max_connections: Pool size.
        headers: Default headers (auth, versioning).
        transport: Optional custom transport (used by tests).

    Returns:
        A configured client. The caller owns it and must ``aclose`` it.
    """
    timeout = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s))
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max(1, max_connections // 2),
    )
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
        headers=headers or {},
        transport=transport,
    )


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #


@contextmanager
def translate_transport_errors(provider: str) -> Iterator[None]:
    """Convert ``httpx`` transport exceptions into provider errors.

    Args:
        provider: Provider name for error attribution.

    Yields:
        Nothing; wraps the block.

    Raises:
        ProviderTimeoutError: On any httpx timeout.
        ProviderConnectionError: On other transport failures.
    """
    try:
        yield
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(
            f"{provider}: request timed out ({type(exc).__name__})", provider=provider
        ) from exc
    except httpx.TransportError as exc:
        raise ProviderConnectionError(
            f"{provider}: transport error ({type(exc).__name__}: {exc})", provider=provider
        ) from exc


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP date).

    Args:
        value: Raw header value.

    Returns:
        Delay in seconds (never negative), or ``None`` if absent/unparseable.
    """
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(tz=UTC)).total_seconds())


def error_for_status(provider: str, status: int, body: str,
                     headers: httpx.Headers | None = None) -> ProviderError:
    """Map an upstream HTTP error status to a provider error.

    Args:
        provider: Provider name.
        status: Upstream HTTP status (>= 400).
        body: Response body, truncated before being stored.
        headers: Response headers, used for ``Retry-After``.

    Returns:
        The matching :class:`ProviderError` subclass instance (not raised).
    """
    snippet = body[:_MAX_ERROR_BODY_CHARS]
    message = f"{provider}: upstream HTTP {status}: {snippet}"
    kwargs: dict[str, Any] = {"provider": provider, "upstream_status": status}

    if status == 429:
        retry_after = parse_retry_after(headers.get("retry-after") if headers else None)
        return ProviderRateLimitError(message, retry_after_s=retry_after, **kwargs)
    if status in (401, 403):
        return ProviderAuthError(message, **kwargs)
    if status == 404:  # model/endpoint missing: the provider's problem, not the client's
        return ProviderNotFoundError(message, **kwargs)
    if status in (408, 409):  # timeout / conflict: transient by definition
        return ProviderServerError(message, **kwargs)
    if 400 <= status < 500:
        return ProviderBadRequestError(message, **kwargs)
    return ProviderServerError(message, **kwargs)  # 5xx, 529, anything odd


async def raise_for_status(provider: str, response: httpx.Response) -> None:
    """Raise a classified provider error if ``response`` is not 2xx.

    Works for both buffered and streamed responses (reads the body if needed).

    Args:
        provider: Provider name.
        response: Upstream response.

    Raises:
        ProviderError: Classified by status code.
    """
    if response.is_success:
        return
    try:
        body = (await response.aread()).decode("utf-8", errors="replace")
    except httpx.HTTPError:
        body = "<body unavailable>"
    raise error_for_status(provider, response.status_code, body, response.headers)


def decode_json(provider: str, response: httpx.Response) -> dict[str, Any]:
    """Decode a JSON object body.

    Args:
        provider: Provider name.
        response: Successful upstream response.

    Returns:
        The decoded object.

    Raises:
        ProviderResponseError: If the body is not a JSON object.
    """
    try:
        data = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderResponseError(f"{provider}: response is not valid JSON",
                                    provider=provider) from exc
    if not isinstance(data, dict):
        raise ProviderResponseError(f"{provider}: response JSON is not an object",
                                    provider=provider)
    return data


# --------------------------------------------------------------------------- #
# Server-sent events
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """One server-sent event.

    Attributes:
        event: Event name (``"message"`` if the server sent none).
        data: Joined ``data:`` payload.
    """

    event: str
    data: str


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    """Parse an SSE stream (per the WHATWG spec, minus ``id``/``retry``).

    Args:
        lines: Decoded lines from the response body.

    Yields:
        Events, dispatched on blank lines.
    """
    event_name = "message"
    data_lines: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue  # comment / keep-alive
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:  # stream ended without a trailing blank line
        yield SSEEvent(event=event_name, data="\n".join(data_lines))


def parse_event_json(provider: str, event: SSEEvent) -> dict[str, Any]:
    """Decode an SSE event's data as a JSON object.

    Args:
        provider: Provider name.
        event: Event to decode.

    Returns:
        The decoded object.

    Raises:
        ProviderResponseError: If the data is not a JSON object.
    """
    try:
        data = json.loads(event.data)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"{provider}: malformed SSE data", provider=provider) from exc
    if not isinstance(data, dict):
        raise ProviderResponseError(f"{provider}: SSE data is not an object", provider=provider)
    return data
