"""Tests for admission control: lag probe, adaptive limit, controller, middleware."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from conftest import make_config
from groundwire.api.middleware import AdmissionMiddleware
from groundwire.core.config import AdmissionConfig
from groundwire.core.errors import OverloadedError
from groundwire.main import create_app
from groundwire.observability.metrics import GatewayMetrics
from groundwire.resilience.overload import (
    AdaptiveLimit,
    AdmissionController,
    LagProbe,
    ShedReason,
)

TARGET_S = 0.010


def _controller(**overrides: Any) -> tuple[AdmissionController, GatewayMetrics]:
    metrics = GatewayMetrics()
    settings: dict[str, Any] = {"initial_concurrency": 2, "min_concurrency": 1,
                                "max_concurrency": 4} | overrides
    return AdmissionController(AdmissionConfig(**settings), metrics=metrics), metrics


def _shed_count(metrics: GatewayMetrics) -> float:
    return float(metrics.load_shed.labels(reason=ShedReason.CONCURRENCY_LIMIT.value)._value.get())


# --------------------------------------------------------------------------- #
# LagProbe
# --------------------------------------------------------------------------- #


def test_probe_smooths_samples_and_ignores_negative_lateness() -> None:
    probe = LagProbe(0.05)
    assert probe.record(0.020) == pytest.approx(0.010)  # EWMA alpha 0.5 from 0
    assert probe.record(0.020) == pytest.approx(0.015)
    assert probe.record(-1.0) == pytest.approx(0.0075)  # early wake-up counts as 0


def test_probe_notifies_listeners_and_publishes_metric() -> None:
    metrics = GatewayMetrics()
    probe = LagProbe(0.05, metrics=metrics)
    seen: list[float] = []
    probe.subscribe(seen.append)
    probe.record(0.004)
    assert seen == [pytest.approx(0.002)]
    assert metrics.event_loop_lag._value.get() == pytest.approx(0.002)


def test_probe_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        LagProbe(0)


async def test_probe_measures_a_blocked_event_loop() -> None:
    probe = LagProbe(0.005)
    seen: list[float] = []
    probe.subscribe(seen.append)
    probe.start()
    probe.start()  # idempotent
    try:
        await asyncio.sleep(0.02)
        time.sleep(0.06)  # block the loop on purpose
        await asyncio.sleep(0.02)
        assert max(seen) > 0.02  # ~half of the 55 ms the probe overslept (EWMA)
    finally:
        await probe.stop()
    assert not probe.running
    await probe.stop()  # stopping twice is harmless


async def test_probe_survives_a_failing_listener(caplog: pytest.LogCaptureFixture) -> None:
    probe = LagProbe(0.005)

    def broken(_: float) -> None:
        raise RuntimeError("listener bug")

    probe.subscribe(broken)
    probe.start()
    try:
        await asyncio.sleep(0.03)
        assert probe.running
    finally:
        await probe.stop()
    assert "lag probe failed" in caplog.text


# --------------------------------------------------------------------------- #
# AdaptiveLimit
# --------------------------------------------------------------------------- #


def _limit(initial: int = 100) -> AdaptiveLimit:
    return AdaptiveLimit(initial=initial, minimum=8, maximum=200, target_lag_s=TARGET_S)


def test_limit_decreases_multiplicatively_when_lag_is_over_target() -> None:
    limit = _limit()
    assert limit.update(0.02, in_flight=100) == 90
    assert limit.update(0.02, in_flight=90) == 81


def test_limit_never_drops_below_minimum() -> None:
    limit = _limit(initial=10)
    for _ in range(50):
        limit.update(1.0, in_flight=10)
    assert limit.value == 8


def test_limit_grows_only_when_healthy_and_in_use() -> None:
    limit = _limit()
    assert limit.update(0.001, in_flight=10) == 100  # idle: an unused limit must not grow
    assert limit.update(0.001, in_flight=60) == 105  # +5%
    assert limit.update(0.007, in_flight=105) == 105  # dead band (between target/2 and target)


def test_limit_grows_at_least_one_and_respects_maximum() -> None:
    small = AdaptiveLimit(initial=8, minimum=8, maximum=9, target_lag_s=TARGET_S)
    assert small.update(0.0, in_flight=8) == 9  # 5% of 8 rounds below 1: still +1
    assert small.update(0.0, in_flight=9) == 9  # capped


@pytest.mark.parametrize(("initial", "minimum", "maximum", "target"), [
    (5, 8, 10, 0.01), (20, 8, 10, 0.01), (5, 0, 10, 0.01), (5, 1, 10, 0.0)])
def test_limit_rejects_inconsistent_settings(initial: int, minimum: int, maximum: int,
                                             target: float) -> None:
    with pytest.raises(ValueError):
        AdaptiveLimit(initial=initial, minimum=minimum, maximum=maximum, target_lag_s=target)


# --------------------------------------------------------------------------- #
# AdmissionController
# --------------------------------------------------------------------------- #


def test_controller_admits_up_to_the_limit_then_refuses() -> None:
    controller, metrics = _controller(retry_after_s=3)
    with controller.admit(), controller.admit():
        assert controller.in_flight == 2
        assert metrics.in_flight._value.get() == 2
        with pytest.raises(OverloadedError) as info, controller.admit():
            pass
    error = info.value
    assert error.http_status == 503
    assert error.code == "server_overloaded"
    assert error.retry_after_s == 3
    assert error.details == {"reason": "concurrency_limit"}
    assert controller.in_flight == 0  # the refused request was never counted
    assert _shed_count(metrics) == 1


def test_controller_releases_the_slot_when_the_request_fails() -> None:
    controller, _ = _controller()
    with pytest.raises(RuntimeError), controller.admit():
        raise RuntimeError("handler crashed")
    assert controller.in_flight == 0


def test_controller_lowers_its_limit_when_the_loop_lags() -> None:
    controller, metrics = _controller(initial_concurrency=4, max_concurrency=4)
    for _ in range(10):
        controller.probe.record(0.5)
    assert controller.limit == 1  # the configured minimum
    assert metrics.concurrency_limit._value.get() == 1
    with controller.admit(), pytest.raises(OverloadedError):
        controller.check()


def test_disabled_controller_never_refuses() -> None:
    controller, metrics = _controller(enabled=False, initial_concurrency=1,
                                      max_concurrency=1)
    with controller.admit(), controller.admit(), controller.admit():
        assert controller.in_flight == 3
    assert _shed_count(metrics) == 0


async def test_controller_start_is_a_no_op_when_disabled() -> None:
    controller, _ = _controller(enabled=False)
    controller.start()
    assert not controller.probe.running
    await controller.aclose()


async def test_controller_start_runs_the_probe_until_closed() -> None:
    controller, _ = _controller(probe_interval_ms=5)
    controller.start()
    assert controller.probe.running
    await controller.aclose()
    assert not controller.probe.running


@pytest.mark.parametrize("bounds", [
    {"min_concurrency": 10, "initial_concurrency": 5},
    {"initial_concurrency": 50, "max_concurrency": 10}])
def test_config_rejects_inconsistent_bounds(bounds: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="min_concurrency"):
        AdmissionConfig(**bounds)


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #


@pytest.fixture
def full_controller() -> Iterator[AdmissionController]:
    """A controller with its single slot already taken."""
    controller, _ = _controller(initial_concurrency=1, min_concurrency=1, max_concurrency=1,
                                retry_after_s=2)
    with controller.admit():
        yield controller


def _app(controller: AdmissionController | None) -> TestClient:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat() -> dict[str, str]:
        return {"ok": "chat"}

    @app.post("/v1/approvals/abc/approve")
    async def approve() -> dict[str, str]:
        return {"ok": "approve"}

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"ok": "health"}

    app.add_middleware(AdmissionMiddleware, controller=lambda: controller)
    return TestClient(app)


def test_middleware_refuses_chat_with_503_and_retry_after(
        full_controller: AdmissionController) -> None:
    response = _app(full_controller).post("/v1/chat/completions", json={})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json()["error"]["code"] == "server_overloaded"


def test_middleware_never_sheds_health_or_reviewer_routes(
        full_controller: AdmissionController) -> None:
    client = _app(full_controller)
    assert client.get("/healthz").status_code == 200
    assert client.post("/v1/approvals/abc/approve").status_code == 200


def test_middleware_counts_admitted_requests_and_releases_them() -> None:
    controller, _ = _controller()
    assert _app(controller).post("/v1/chat/completions", json={}).json() == {"ok": "chat"}
    assert controller.in_flight == 0


def test_middleware_passes_through_before_startup() -> None:
    assert _app(None).post("/v1/chat/completions", json={}).status_code == 200


def test_app_starts_and_stops_the_probe_with_its_lifespan() -> None:
    app = create_app(config=make_config())
    with TestClient(app):
        admission = app.state.container.admission
        assert admission is not None
        assert admission.probe.running
    assert not admission.probe.running
