"""Structured logging setup.

Production logs are single-line JSON so they can be shipped to Loki, ELK or
CloudWatch and queried by field (``request_id``, ``provider``, ``route``...).
Local development can switch to readable text with ``GROUNDWIRE_LOG_JSON=false``.

Call sites use the standard library only::

    logger.info("provider call failed", extra={"provider": name, "attempt": 2})

Anything passed via ``extra`` becomes a top-level JSON field.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar

from groundwire.core.context import get_request_id

# Attributes every LogRecord has; anything else came from `extra=`.
_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys() | {"message", "asctime"}
)


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``record.request_id``.

        Args:
            record: Record being emitted.

        Returns:
            Always ``True`` (the record is never dropped).
        """
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to JSON.

        Args:
            record: Record to format.

        Returns:
            A single-line JSON string.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # default=str keeps logging from ever raising on odd values.
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    _FMT: ClassVar[str] = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        """Initialise with the development format string."""
        super().__init__(self._FMT)

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record`` and append any ``extra`` fields as ``k=v``.

        Args:
            record: Record to format.

        Returns:
            The formatted line.
        """
        base = super().format(record)
        extras = {
            k: v for k, v in vars(record).items()
            if k not in _RESERVED_ATTRS and k != "request_id"
        }
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure the root logger. Safe to call more than once.

    Args:
        level: Log level name (e.g. ``"INFO"``).
        json_output: Emit JSON when True, readable text otherwise.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Keep noisy third-party loggers at WARNING unless we are debugging.
    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
