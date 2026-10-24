"""Tests for the dashboard's building blocks: activity log, metadata summary,
dashboard service, and optional providers (``skip_if_no_key``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import make_config
from groundwire.container import build_container
from groundwire.core.config import load_gateway_config
from groundwire.core.errors import ConfigurationError
from groundwire.observability.activity import ActivityLog, summarise_metadata
from groundwire.observability.metrics import GatewayMetrics
from groundwire.providers.factory import build_providers
from groundwire.routing.router import Router
from groundwire.services.dashboard import DashboardService

ROOT = Path(__file__).resolve().parents[2]

# Shape of real ``groundwire`` metadata (taken from a live response).
META: dict[str, Any] = {
    "route": "agent-demo-approval", "provider": "mock-stable",
    "attempts": [{"provider": "mock-flaky", "outcome": "error"},
                 {"provider": "mock-stable", "outcome": "success"}],
    "safety": {"injection": {"score": 0.0, "action": "none"},
               "pii_redacted": {"email": 1, "phone": 2}},
    "cache": {"status": "miss"},
    "grounding": {"verdict": "failed", "supported_ratio": 0.0},
    "answer_check": {"verdict": "mismatch", "issues": [{"kind": "number_mismatch"}]},
    "tools": {"decisions": [
        {"tool": "issue_refund", "decision": "pending_approval", "approval_id": "apr_1"},
        {"tool": "issue_refund", "decision": "blocked"},
        {"tool": "lookup", "decision": "allowed"}]},
}


# --------------------------------------------------------------------------- #
# summarise_metadata
# --------------------------------------------------------------------------- #


def test_summary_extracts_every_dashboard_field() -> None:
    assert summarise_metadata(META) == {
        "attempts": 2, "failed_over": True, "pii": {"email": 1, "phone": 2},
        "injection": "none", "cache": "miss", "grounding": "failed",
        "answer_check": "mismatch",
        "tools": {"pending_approval": 1, "blocked": 1, "allowed": 1}, "approvals": ["apr_1"]}


def test_summary_does_not_report_failover_when_the_primary_answered() -> None:
    meta = {"provider": "a", "attempts": [{"provider": "a", "outcome": "success"}]}
    assert summarise_metadata(meta)["failed_over"] is False


@pytest.mark.parametrize("garbage", [
    {}, {"attempts": "x", "safety": [], "tools": {"decisions": "nope"}},
    {"tools": {"decisions": [None, 3]}, "safety": {"pii_redacted": {"email": "many"}}}])
def test_summary_tolerates_unexpected_shapes(garbage: dict[str, Any]) -> None:
    summarise_metadata(garbage)  # must not raise: a dashboard bug can't fail a request


# --------------------------------------------------------------------------- #
# ActivityLog
# --------------------------------------------------------------------------- #


def _record(log: ActivityLog, status: int = 200, latency: float = 10.0) -> None:
    log.record(request_id="gw-1", route="default", provider="p", status=status,
               latency_ms=latency, metadata=META, now=1000.0)


def test_log_returns_newest_first_and_supports_incremental_polling() -> None:
    log = ActivityLog(capacity=10)
    for _ in range(3):
        _record(log)
    assert [e.seq for e in log.recent()] == [3, 2, 1]
    assert [e.seq for e in log.recent(after=2)] == [3]
    assert [e.seq for e in log.recent(limit=1)] == [3]
    assert log.last_seq == 3


def test_log_is_bounded() -> None:
    log = ActivityLog(capacity=10)
    for _ in range(25):
        _record(log)
    events = log.recent(limit=100)
    assert len(events) == 10
    assert events[-1].seq == 16  # the oldest 15 were dropped


def test_events_hold_no_prompt_answer_or_tool_arguments() -> None:
    log = ActivityLog()
    _record(log)
    event = log.recent()[0].to_dict()
    text = repr(event)
    assert "content" not in event and "arguments" not in text
    assert event["pii"] == {"email": 1, "phone": 2}  # types and counts only


def test_latency_percentiles_ignore_failed_requests() -> None:
    log = ActivityLog()
    for latency in range(1, 101):
        _record(log, latency=float(latency))
    _record(log, status=500, latency=99_999.0)
    assert log.latency_percentiles(0.5, 0.95) == {"p50": 50.0, "p95": 95.0}
    assert ActivityLog().latency_percentiles(0.5) == {}


def test_log_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        ActivityLog(0)


# --------------------------------------------------------------------------- #
# DashboardService
# --------------------------------------------------------------------------- #


def _service(public_demo: bool, token: str | None = "t" * 16) -> DashboardService:
    from pydantic import SecretStr

    config = make_config(dashboard={"public_demo": public_demo})
    metrics = GatewayMetrics()
    router = Router(config, build_providers(config), metrics=metrics)
    return DashboardService(config, metrics=metrics, activity=ActivityLog(), router=router,
                            reviewer_token=SecretStr(token) if token else None,
                            clock=lambda: 50.0)


def test_summary_counts_requests_and_safety_from_metrics() -> None:
    service = _service(public_demo=False)
    m = service._metrics
    m.requests.labels(route="default", provider="primary", status="success", stream="false").inc(3)
    m.requests.labels(route="default", provider="", status="prompt_injection_detected",
                      stream="false").inc()
    m.cache_lookups.labels(result="hit_exact").inc()
    m.cache_lookups.labels(result="miss").inc(3)
    m.pii_redactions.labels(type="email").inc(2)
    m.tool_calls.labels(decision="blocked", reason="argument_limit").inc()
    summary = service.summary()
    assert summary["requests"] == {"total": 4, "ok": 3, "errors": 1, "by_status": {
        "success": 3, "prompt_injection_detected": 1}}
    assert summary["cache"] == {"hits": 1, "misses": 3, "hit_rate": 0.25}
    assert summary["safety"]["pii_redacted"] == 2
    assert summary["tools"] == {"blocked": 1}
    assert set(summary["providers"]) == {"primary", "secondary"}
    assert summary["load"]["limit"] is None  # no admission controller given


def test_reviewer_token_is_exposed_only_in_public_demo_mode() -> None:
    assert "demo_reviewer_token" not in _service(public_demo=False).summary()
    assert _service(public_demo=True).summary()["demo_reviewer_token"] == "t" * 16
    assert "demo_reviewer_token" not in _service(public_demo=True, token=None).summary()


def test_public_demo_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    _service(public_demo=True)
    assert "every visitor can approve" in caplog.text


def test_activity_page_is_capped() -> None:
    service = _service(public_demo=False)
    for _ in range(5):
        _record(service._activity)
    page = service.activity(after=0, limit=10_000)
    assert page["last_seq"] == 5 and len(page["events"]) == 5


# --------------------------------------------------------------------------- #
# Optional providers (skip_if_no_key)
# --------------------------------------------------------------------------- #


def _cloud_config(skip: bool) -> Any:
    return make_config(
        providers=[{"name": "cloud", "type": "openai_compatible", "model": "m",
                    "base_url": "https://example.invalid/v1", "api_key_env": "GW_TEST_KEY",
                    "skip_if_no_key": skip},
                   {"name": "backup", "type": "mock", "model": "m"}],
        routes={"default": ["cloud", "backup"]})


def test_optional_provider_without_key_is_skipped(monkeypatch: pytest.MonkeyPatch,
                                                  caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("GW_TEST_KEY", raising=False)
    providers = build_providers(_cloud_config(skip=True))
    assert list(providers) == ["backup"]
    assert "API key not set" in caplog.text


def test_optional_provider_with_key_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW_TEST_KEY", "secret")
    assert list(build_providers(_cloud_config(skip=True))) == ["cloud", "backup"]


def test_required_provider_without_key_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GW_TEST_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="GW_TEST_KEY"):
        build_providers(_cloud_config(skip=False))


def test_demo_config_starts_without_a_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROUNDWIRE_APPROVER_TOKEN", "public-demo-reviewer-token")
    container = build_container(load_gateway_config(ROOT / "config" / "demo.yaml"))
    assert container.router.routes["default"] == ["fallback-mock"]
    assert container.dashboard is not None and container.dashboard.public_demo


def test_dashboard_page_never_uses_inner_html() -> None:
    """XSS guard: route names in the activity feed come from API clients."""
    page = (ROOT / "src" / "groundwire" / "api" / "static" / "dashboard.html").read_text()
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page
