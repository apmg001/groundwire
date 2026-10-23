"""Tests for the retry policy and circuit breaker."""

from __future__ import annotations

import random

import pytest

from groundwire.core.config import CircuitBreakerConfig, RetryConfig
from groundwire.core.errors import (
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from groundwire.resilience.circuit_breaker import CircuitBreaker, CircuitState
from groundwire.resilience.retry import RetryPolicy

# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _policy(max_attempts: int = 3, sleeps: _Sleeps | None = None) -> RetryPolicy:
    return RetryPolicy(RetryConfig(max_attempts=max_attempts, base_delay_s=0.1, max_delay_s=1.0),
                       sleep=sleeps or _Sleeps(), rng=random.Random(0))


async def test_retry_succeeds_after_transient_failures() -> None:
    sleeps = _Sleeps()
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderServerError("boom", provider="p")
        return "ok"

    assert await _policy(sleeps=sleeps).run(op, provider="p") == "ok"
    assert calls == 3
    assert len(sleeps.delays) == 2


async def test_retry_gives_up_after_max_attempts() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise ProviderServerError("boom", provider="p")

    with pytest.raises(ProviderServerError):
        await _policy(max_attempts=3).run(op, provider="p")
    assert calls == 3


@pytest.mark.parametrize(("retry_on_timeout", "expected_calls"), [(False, 1), (True, 3)])
async def test_timeouts_are_only_retried_in_place_when_enabled(
        retry_on_timeout: bool, expected_calls: int) -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise ProviderTimeoutError("slow", provider="p")

    policy = RetryPolicy(RetryConfig(max_attempts=3, retry_on_timeout=retry_on_timeout),
                         sleep=_Sleeps(), rng=random.Random(0))
    with pytest.raises(ProviderTimeoutError):
        await policy.run(op, provider="p")
    assert calls == expected_calls


async def test_non_retryable_error_is_not_retried() -> None:
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        raise ProviderBadRequestError("bad", provider="p")

    with pytest.raises(ProviderBadRequestError):
        await _policy().run(op, provider="p")
    assert calls == 1


def test_backoff_is_capped_and_jittered() -> None:
    policy = _policy()
    for attempt in range(1, 10):
        delay = policy.backoff_delay(attempt)
        assert 0 <= delay <= min(1.0, 0.1 * 2 ** (attempt - 1))


def test_retry_after_hint_is_honoured_but_capped() -> None:
    policy = _policy()
    assert policy.backoff_delay(1, retry_after_s=0.5) == 0.5
    assert policy.backoff_delay(1, retry_after_s=99) == 1.0


async def test_rate_limit_uses_retry_after() -> None:
    sleeps = _Sleeps()
    calls = 0

    async def op() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderRateLimitError("slow down", provider="p", retry_after_s=0.7)
        return "ok"

    await _policy(sleeps=sleeps).run(op, provider="p")
    assert sleeps.delays == [0.7]


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _breaker(clock: _Clock, threshold: int = 3, probes: int = 1) -> CircuitBreaker:
    cfg = CircuitBreakerConfig(failure_threshold=threshold, recovery_timeout_s=10,
                               half_open_max_calls=probes)
    return CircuitBreaker("p", cfg, clock=clock)


def test_breaker_opens_after_threshold() -> None:
    breaker = _breaker(_Clock())
    for _ in range(2):
        assert breaker.try_acquire()
        breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.try_acquire()


def test_success_resets_failure_count() -> None:
    breaker = _breaker(_Clock())
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_half_open_probe_success_closes() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock.now = 10
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.try_acquire()
    assert not breaker.try_acquire()  # only one probe allowed
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens_and_restarts_timer() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure()
    clock.now = 10
    assert breaker.try_acquire()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    clock.now = 15
    assert breaker.state is CircuitState.OPEN  # timer restarted at t=10
    clock.now = 20
    assert breaker.state is CircuitState.HALF_OPEN


def test_release_frees_probe_slot_without_verdict() -> None:
    clock = _Clock()
    breaker = _breaker(clock, threshold=1)
    breaker.record_failure()
    clock.now = 10
    assert breaker.try_acquire()
    breaker.release()
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.try_acquire()


def test_state_listener_errors_do_not_propagate() -> None:
    def broken_listener(name: str, state: CircuitState) -> None:
        raise RuntimeError("metrics down")

    cfg = CircuitBreakerConfig(failure_threshold=1)
    breaker = CircuitBreaker("p", cfg, clock=_Clock(), on_state_change=broken_listener)
    breaker.record_failure()  # must not raise
    assert breaker.state is CircuitState.OPEN
