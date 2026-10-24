"""Dashboard read model: one JSON snapshot of what the gateway is doing.

The dashboard page polls two endpoints: a *summary* (totals, safety counts,
load, provider health) and the *activity* feed. This service builds both from
components the gateway already has (metrics registry, activity log, router,
admission controller), so the dashboard adds no new bookkeeping to the
request path apart from one activity record per request.

Read-only and content-free: nothing here exposes prompts, answers or tool
arguments. The one exception is deliberate and opt-in: with
``dashboard.public_demo`` the reviewer token is included so a demo visitor
can approve held calls (see :class:`~groundwire.core.config.DashboardConfig`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import SecretStr

from groundwire import __version__
from groundwire.core.config import GatewayConfig
from groundwire.observability.activity import ActivityLog
from groundwire.observability.metrics import GatewayMetrics
from groundwire.resilience.overload import AdmissionController
from groundwire.routing.router import Router

logger = logging.getLogger(__name__)

MAX_ACTIVITY_PAGE = 200

# Cache lookup labels that mean "answered from cache".
_HIT_PREFIX = "hit_"


class DashboardService:
    """Builds the dashboard's summary and activity views."""

    def __init__(self, config: GatewayConfig, *, metrics: GatewayMetrics, activity: ActivityLog,
                 router: Router, admission: AdmissionController | None = None,
                 reviewer_token: SecretStr | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        """Create the service.

        Args:
            config: Gateway configuration.
            metrics: Metrics registry (source of totals).
            activity: Recent-activity log.
            router: Router (routes and circuit states).
            admission: Admission controller (load figures), if any.
            reviewer_token: Reviewer token; only exposed in public-demo mode.
            clock: Wall clock (tests).
        """
        self._config = config
        self._metrics = metrics
        self._activity = activity
        self._router = router
        self._admission = admission
        self._reviewer_token = reviewer_token
        self._clock = clock
        self._started = clock()
        if config.dashboard.enabled and config.dashboard.public_demo:
            logger.warning("dashboard public_demo is on: every visitor can approve or reject "
                           "held tool calls. Use only where tool calls are not real.")

    @property
    def enabled(self) -> bool:
        """Whether the dashboard is served."""
        return self._config.dashboard.enabled

    @property
    def public_demo(self) -> bool:
        """Whether demo-only features (scenarios, shared reviewer token) are on."""
        return self._config.dashboard.public_demo

    def summary(self) -> dict[str, Any]:
        """Totals and current state, for the dashboard's tiles.

        Returns:
            JSON-ready snapshot.
        """
        m = self._metrics
        by_status = m.totals_by(m.requests, "status")
        total = sum(by_status.values())
        ok = by_status.get("success", 0.0)
        cache = m.totals_by(m.cache_lookups, "result")
        hits = sum(v for k, v in cache.items() if k.startswith(_HIT_PREFIX))
        lookups = hits + cache.get("miss", 0.0)
        injection = m.totals_by(m.injection_checks, "action")
        snapshot: dict[str, Any] = {
            "version": __version__,
            "uptime_s": round(self._clock() - self._started),
            "public_demo": self.public_demo,
            "requests": {"total": total, "ok": ok, "errors": total - ok,
                         "by_status": by_status},
            "latency_ms": self._activity.latency_percentiles(0.5, 0.95),
            "cache": {"hits": hits, "misses": cache.get("miss", 0.0),
                      "hit_rate": round(hits / lookups, 3) if lookups else 0.0},
            "failovers": sum(m.totals_by(m.failovers, "route").values()),
            "safety": {"pii_redacted": sum(m.totals_by(m.pii_redactions, "type").values()),
                       "pii_by_type": m.totals_by(m.pii_redactions, "type"),
                       # labels as recorded by SafetyGuard._apply_injection_policy
                       "injection_blocked": injection.get("blocked", 0.0),
                       "injection_flagged": injection.get("flagged", 0.0)},
            "grounding": m.totals_by(m.grounding, "verdict"),
            "answer_checks": m.totals_by(m.answer_checks, "verdict"),
            "tools": m.totals_by(m.tool_calls, "decision"),
            "approvals": {"pending": m.gauge_value(m.approvals_pending),
                          "events": m.totals_by(m.approvals, "event")},
            "load": self._load(),
            "providers": self._router.breaker_states(),
            "routes": self._router.routes,
        }
        if self.public_demo and self._reviewer_token is not None:
            snapshot["demo_reviewer_token"] = self._reviewer_token.get_secret_value()
        return snapshot

    def activity(self, *, after: int = 0, limit: int = 50) -> dict[str, Any]:
        """Recent requests, newest first.

        Args:
            after: Only events with a larger sequence number.
            limit: Maximum events (capped at :data:`MAX_ACTIVITY_PAGE`).

        Returns:
            ``{"last_seq": n, "events": [...]}``.
        """
        events = self._activity.recent(after=max(0, after),
                                       limit=min(max(1, limit), MAX_ACTIVITY_PAGE))
        return {"last_seq": self._activity.last_seq, "events": [e.to_dict() for e in events]}

    def _load(self) -> dict[str, Any]:
        m = self._metrics
        shed = sum(m.totals_by(m.load_shed, "reason").values())
        if self._admission is None:
            return {"in_flight": 0, "limit": None, "event_loop_lag_ms": 0.0, "shed": shed}
        return {"in_flight": self._admission.in_flight, "limit": self._admission.limit,
                "event_loop_lag_ms": round(self._admission.probe.lag_s * 1000, 2),
                "shed": shed}
