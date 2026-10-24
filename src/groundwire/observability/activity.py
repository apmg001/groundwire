"""Recent-activity log: what the gateway just did, for the live dashboard.

Prometheus metrics answer "how many"; the dashboard also needs "what just
happened": this request failed over, that one had an email redacted, a refund
was held for approval. :class:`ActivityLog` keeps the last N requests as small
:class:`ActivityEvent` summaries, built from the ``groundwire`` metadata the
gateway already returns with every response.

Privacy by construction: an event holds **no prompt or answer text and no tool
arguments**, only counts, verdicts and names (PII *types* redacted, not the
values). The dashboard can be public without leaking what users asked.

Bounded: a fixed-size ring buffer, so memory does not grow with traffic.
Per-process like the metrics (each worker shows its own activity).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_CAPACITY = 200


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    """Summary of one finished (or failed) chat request.

    Attributes:
        seq: Monotonic sequence number (clients poll with ``after=seq``).
        at: Unix time the request finished.
        request_id: The request's ``X-Request-ID``.
        route: Route alias the client asked for.
        provider: Provider that answered (empty when none did).
        status: HTTP status returned.
        latency_ms: Time spent in the gateway, model call included.
        stream: Streamed response.
        error_code: Groundwire error code for failures.
        failed_over: Served by a provider other than the first one tried.
        attempts: Provider attempts made (retries and failover included).
        cache: Cache status (``hit_exact``, ``miss``, ``bypass``...).
        pii: Redacted PII types and counts (types only, never values).
        injection: Injection screening action (``none``, ``flag``, ``block``).
        grounding: Grounding verdict, when the request carried documents.
        answer_check: Answer-check verdict, when it ran.
        tools: Tool-call decisions: ``{"allowed": 1, "blocked": 0, ...}``.
        approvals: Approval ids created or used by this request.
    """

    seq: int
    at: float
    request_id: str
    route: str
    provider: str
    status: int
    latency_ms: float
    stream: bool = False
    error_code: str = ""
    failed_over: bool = False
    attempts: int = 0
    cache: str = ""
    pii: dict[str, int] = field(default_factory=dict)
    injection: str = ""
    grounding: str = ""
    answer_check: str = ""
    tools: dict[str, int] = field(default_factory=dict)
    approvals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def summarise_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the dashboard fields from a response's ``groundwire`` metadata.

    Tolerant by design: missing or oddly shaped sections are skipped, never
    raised, because a dashboard bug must not fail a request.

    Args:
        meta: The ``groundwire`` object of a chat completion response.

    Returns:
        Keyword arguments for :class:`ActivityEvent` (subset).
    """
    out: dict[str, Any] = {}
    attempts = meta.get("attempts")
    if isinstance(attempts, list):
        out["attempts"] = len(attempts)
        first = attempts[0] if attempts and isinstance(attempts[0], Mapping) else {}
        out["failed_over"] = bool(first) and first.get("provider") != meta.get("provider")
    safety = meta.get("safety")
    if isinstance(safety, Mapping):
        pii = safety.get("pii_redacted")
        if isinstance(pii, Mapping):
            out["pii"] = {str(k): int(v) for k, v in pii.items() if isinstance(v, int)}
        injection = safety.get("injection")
        if isinstance(injection, Mapping):
            out["injection"] = str(injection.get("action", ""))
    cache = meta.get("cache")
    if isinstance(cache, Mapping):
        out["cache"] = str(cache.get("status", ""))
    for key in ("grounding", "answer_check"):
        section = meta.get(key)
        if isinstance(section, Mapping) and section.get("verdict"):
            out[key] = str(section["verdict"])
    tools = meta.get("tools")
    if isinstance(tools, Mapping) and isinstance(tools.get("decisions"), list):
        counts: dict[str, int] = {}
        approvals: list[str] = []
        for decision in tools["decisions"]:
            if not isinstance(decision, Mapping):
                continue
            verdict = str(decision.get("decision", "unknown"))
            counts[verdict] = counts.get(verdict, 0) + 1
            if decision.get("approval_id"):
                approvals.append(str(decision["approval_id"]))
        out["tools"], out["approvals"] = counts, approvals
    return out


class ActivityLog:
    """Ring buffer of the most recent :class:`ActivityEvent` objects.

    Single event loop, no awaits inside methods: no lock needed.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """Create an empty log.

        Args:
            capacity: Events kept; older ones are dropped.

        Raises:
            ValueError: If ``capacity`` is not positive.
        """
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._events: deque[ActivityEvent] = deque(maxlen=capacity)
        self._seq = 0

    @property
    def last_seq(self) -> int:
        """Sequence number of the newest event (0 when empty)."""
        return self._seq

    def record(self, *, request_id: str, route: str, provider: str, status: int,
               latency_ms: float, stream: bool = False, error_code: str = "",
               metadata: Mapping[str, Any] | None = None,
               now: float | None = None) -> ActivityEvent:
        """Append one request's summary.

        Args:
            request_id: Request ID.
            route: Route alias requested.
            provider: Provider that answered ("" if none).
            status: HTTP status returned.
            latency_ms: Gateway time for the request.
            stream: Streamed response.
            error_code: Error code for failures.
            metadata: The response's ``groundwire`` metadata, if any.
            now: Clock override (tests).

        Returns:
            The stored event.
        """
        self._seq += 1
        event = ActivityEvent(seq=self._seq, at=now if now is not None else time.time(),
                              request_id=request_id, route=route, provider=provider,
                              status=status, latency_ms=round(latency_ms, 1), stream=stream,
                              error_code=error_code,
                              **summarise_metadata(metadata or {}))
        self._events.append(event)
        return event

    def recent(self, *, after: int = 0, limit: int = 50) -> list[ActivityEvent]:
        """Newest-first events with ``seq > after``.

        Args:
            after: Only events newer than this sequence number.
            limit: Maximum events returned.

        Returns:
            Up to ``limit`` events, newest first.
        """
        newer = [e for e in reversed(self._events) if e.seq > after]
        return newer[:max(0, limit)]

    def latency_percentiles(self, *quantiles: float) -> dict[str, float]:
        """Latency percentiles of the successful requests in the buffer.

        Args:
            quantiles: Values in (0, 1], e.g. ``0.5, 0.95``.

        Returns:
            ``{"p50": ms, ...}``; empty when no successful request is stored.
        """
        values = sorted(e.latency_ms for e in self._events if e.status < 400)
        if not values:
            return {}
        out: dict[str, float] = {}
        for q in quantiles:
            index = min(len(values) - 1, max(0, round(q * len(values)) - 1))
            out[f"p{round(q * 100)}"] = values[index]
        return out
