"""Structured logging with a per-request correlation id.

Logs are JSON in production (one object per line, easy to ship to any log
store) and plain text in development. Every record carries the current
``request_id`` (set by the API middleware) so all lines of one request can be
found together, including those from the agent loop and the gateway client.

Never log message text, statement rows or memory content: they are personal
financial data. Log ids, counts and codes.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "money_agent_request_id", default=None)

# Attributes every LogRecord has; anything else was passed via ``extra=``.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message", "asctime", "taskName"}


def set_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    """Bind a request id to the current context (task).

    Args:
        request_id: The id, or ``None`` to clear it.

    Returns:
        A token for :func:`reset_request_id`.
    """
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    """Restore the request id that was current before :func:`set_request_id`.

    Args:
        token: Token returned by :func:`set_request_id`.
    """
    _request_id.reset(token)


def get_request_id() -> str | None:
    """The current request id, if any."""
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    """Adds ``request_id`` to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message, request id, extras."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a record.

        Args:
            record: The record.

        Returns:
            A JSON line.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger once, at process start.

    Args:
        level: Level name (``DEBUG``, ``INFO``, ...).
        json_output: JSON lines (production) or human-readable text.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
