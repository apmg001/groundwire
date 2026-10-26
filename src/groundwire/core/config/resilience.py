"""Resilience policy: retries, circuit breakers and load shedding.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    Field,
    model_validator,
)

from groundwire.core.config.base import StrictModel


class RetryConfig(StrictModel):
    """Retry policy applied per provider before failing over.

    Attributes:
        max_attempts: Total attempts per provider (1 disables retries).
        base_delay_s: Initial back-off delay.
        max_delay_s: Cap on any single back-off delay.
        retry_on_timeout: Retry the same provider after it timed out. Off by
            default: a provider that just used its whole timeout will most
            likely do it again, so the request fails over at once instead.
            In the chaos load test (primary hung for 30 s) retrying in place
            made the slowest requests wait two full timeouts (20 s) before
            failover.
    """

    max_attempts: Annotated[int, Field(ge=1, le=10)] = 2
    base_delay_s: Annotated[float, Field(ge=0, le=30)] = 0.2
    max_delay_s: Annotated[float, Field(ge=0, le=120)] = 2.0
    retry_on_timeout: bool = False


class CircuitBreakerConfig(StrictModel):
    """Circuit-breaker policy applied per provider.

    Attributes:
        failure_threshold: Consecutive failures that open the circuit.
        recovery_timeout_s: Time the circuit stays open before a probe.
        half_open_max_calls: Concurrent probe calls allowed while half-open.
    """

    failure_threshold: Annotated[int, Field(ge=1, le=1000)] = 5
    recovery_timeout_s: Annotated[float, Field(gt=0, le=3600)] = 30.0
    half_open_max_calls: Annotated[int, Field(ge=1, le=100)] = 1


class AdmissionConfig(StrictModel):
    """Load shedding: refuse new requests quickly when the gateway is overloaded.

    Event-loop lag drives an adaptive limit on requests in flight; requests
    over the limit get HTTP 503 with ``Retry-After`` at once. See
    :mod:`groundwire.resilience.overload` for the design and the measurements
    behind the defaults. Only ``POST /v1/chat/completions`` is limited; health
    checks, metrics and the reviewer API are always served.

    Attributes:
        enabled: Turn admission control on.
        max_event_loop_lag_ms: Lag the limit steers towards. Measured at 600
            users (3x capacity): a 10 ms target served 129 req/s at p99 1.2 s;
            20 ms served 102 req/s at p99 2.6 s (the loop is held at the
            target, and each request waits for it dozens of times). 100 users
            ran at 0.6 ms median lag, so 10 ms never sheds normal load.
        initial_concurrency: Limit at startup.
        min_concurrency: The limit never drops below this.
        max_concurrency: The limit never grows above this (bounds memory
            when upstreams are slow).
        probe_interval_ms: How often lag is sampled and the limit adjusted.
        retry_after_s: ``Retry-After`` value sent with refusals.
    """

    enabled: bool = True
    max_event_loop_lag_ms: Annotated[float, Field(gt=0, le=10_000)] = 10.0
    initial_concurrency: Annotated[int, Field(ge=1)] = 100
    min_concurrency: Annotated[int, Field(ge=1)] = 8
    max_concurrency: Annotated[int, Field(ge=1)] = 10_000
    probe_interval_ms: Annotated[float, Field(ge=5, le=1_000)] = 50.0
    retry_after_s: Annotated[int, Field(ge=0, le=300)] = 1

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if not self.min_concurrency <= self.initial_concurrency <= self.max_concurrency:
            raise ValueError("admission needs min_concurrency <= initial_concurrency "
                             "<= max_concurrency")
        return self
