"""Admission control: refuse new work quickly when the gateway is overloaded.

Why
---
Load tests (``python -m loadtest.run stress``) showed that one worker
saturates one CPU core at roughly 100-120 requests/s on a 2-CPU machine. Past
that point nothing fails; requests queue inside the event loop and latency
grows without bound (p99 19 s at 600 users). A client waiting 19 s has given
up, so the gateway burns CPU on answers nobody reads while every other request
gets slower. Refusing the excess *immediately* (HTTP 503 with ``Retry-After``)
keeps admitted requests fast and tells clients, load balancers and
autoscalers what is happening.

Signal: event-loop lag
----------------------
For an LLM gateway the number of open requests alone is a poor overload
signal: 10,000 requests waiting on a slow model cost almost no CPU, while 200
cheap cache hits can saturate a core. What hurts every request is the event
loop falling behind. :class:`LagProbe` sleeps for a fixed interval in a loop
and records how late it woke up: the time any ready work waits before it runs.

A request passes through the loop dozens of times (read body, call upstream,
read and forward each chunk), so its extra latency is roughly lag times that
count. Measured on the load-test stack: lag stayed under 14 ms while latency
was flat, and a median of 18 ms already more than doubled median latency.
Hence a target of 10 ms, not hundreds (see ``AdmissionConfig`` for the
10 vs 20 ms comparison).

Control: an adaptive concurrency limit (AIMD)
---------------------------------------------
The first version refused new requests with a probability that grew with lag.
It cut the median but not the tail (p95 still 4 s at 400 users), because lag
is a *late* signal: while lag is low a burst is admitted, and 300 ms later
all of those model replies arrive at once and need the CPU together.

So lag drives a limit on requests in flight instead, the way TCP congestion
control and Netflix's ``concurrency-limits`` do:

* lag above target: limit x 0.9 per probe tick (multiplicative decrease);
* lag below half the target and the limit actually in use: limit grows 5%
  per tick (at least 1);
* in between: hold (a dead band, so the limit does not oscillate).

New requests over the limit are refused at once. The limit converges to the
concurrency the CPU can serve at the target lag, whatever the model latency:
with a slow model each request costs little CPU per second, lag stays low,
and the limit grows. Requests already admitted are never cut off.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterator
from enum import StrEnum

from groundwire.core.config import AdmissionConfig
from groundwire.core.errors import OverloadedError
from groundwire.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)

EWMA_ALPHA = 0.5  # weight of the newest lag sample: reacts within 2-3 probe ticks
DECREASE_FACTOR = 0.9  # per over-target tick: halves the limit in about 7 ticks
INCREASE_FRACTION = 0.05  # per healthy tick: doubles the limit in about 14 ticks
IN_USE_FRACTION = 0.5  # only grow a limit that traffic is actually using


class ShedReason(StrEnum):
    """Why a request was refused (metric label)."""

    CONCURRENCY_LIMIT = "concurrency_limit"


# --------------------------------------------------------------------------- #
# Lag probe
# --------------------------------------------------------------------------- #


LagListener = Callable[[float], None]


class LagProbe:
    """Measures how far behind schedule the event loop runs.

    A background task sleeps ``interval_s`` repeatedly; each time it wakes it
    folds ``actual sleep - interval`` into an exponentially weighted moving
    average and notifies listeners with the new average.
    """

    def __init__(self, interval_s: float, *, metrics: GatewayMetrics | None = None) -> None:
        """Create a stopped probe.

        Args:
            interval_s: Sleep between samples.
            metrics: Where to publish the current lag (optional).

        Raises:
            ValueError: If ``interval_s`` is not positive.
        """
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval = interval_s
        self._metrics = metrics
        self._lag_s = 0.0
        self._listeners: list[LagListener] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def lag_s(self) -> float:
        """Smoothed event-loop lag in seconds (0 before the first sample)."""
        return self._lag_s

    @property
    def running(self) -> bool:
        """True while the background task is alive."""
        return self._task is not None and not self._task.done()

    def subscribe(self, listener: LagListener) -> None:
        """Call ``listener(lag_s)`` after every sample.

        Args:
            listener: Receives the smoothed lag in seconds.
        """
        self._listeners.append(listener)

    def record(self, sample_s: float) -> float:
        """Fold one lateness sample into the average and notify listeners.

        Args:
            sample_s: How late the probe woke up (negative values count as 0).

        Returns:
            The new smoothed lag.
        """
        self._lag_s = EWMA_ALPHA * max(sample_s, 0.0) + (1 - EWMA_ALPHA) * self._lag_s
        if self._metrics is not None:
            self._metrics.event_loop_lag.set(self._lag_s)
        for listener in self._listeners:
            listener(self._lag_s)
        return self._lag_s

    def start(self) -> None:
        """Start sampling on the running loop (idempotent).

        Raises:
            RuntimeError: If called outside a running event loop.
        """
        if self.running:
            return
        self._task = asyncio.get_running_loop().create_task(self._run(), name="lag-probe")

    async def stop(self) -> None:
        """Stop sampling and wait for the task to finish."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.sleep(self._interval)
            try:
                self.record(loop.time() - started - self._interval)
            except Exception:  # a listener bug must not stop overload protection
                logger.exception("lag probe failed to record a sample")


# --------------------------------------------------------------------------- #
# Adaptive limit
# --------------------------------------------------------------------------- #


class AdaptiveLimit:
    """AIMD-style concurrency limit driven by event-loop lag (pure; no I/O).

    Attributes:
        value: Current limit (requests in flight allowed).
    """

    def __init__(self, *, initial: int, minimum: int, maximum: int, target_lag_s: float) -> None:
        """Create the limit.

        Args:
            initial: Starting limit.
            minimum: The limit never drops below this (keeps the gateway useful).
            maximum: The limit never grows above this (memory bound).
            target_lag_s: Lag the controller steers towards.

        Raises:
            ValueError: If the bounds are inconsistent or the target is not positive.
        """
        if not 1 <= minimum <= initial <= maximum:
            raise ValueError("need 1 <= minimum <= initial <= maximum")
        if target_lag_s <= 0:
            raise ValueError("target_lag_s must be positive")
        self._limit = float(initial)
        self._min, self._max = minimum, maximum
        self._target = target_lag_s

    @property
    def value(self) -> int:
        """Current limit, rounded down."""
        return int(self._limit)

    def update(self, lag_s: float, in_flight: int) -> int:
        """Adjust the limit after a lag sample.

        Args:
            lag_s: Smoothed event-loop lag.
            in_flight: Requests in progress now.

        Returns:
            The new limit.
        """
        if lag_s > self._target:
            self._limit = max(self._min, self._limit * DECREASE_FACTOR)
        elif lag_s < self._target / 2 and in_flight >= IN_USE_FRACTION * self._limit:
            self._limit = min(self._max, self._limit + max(1.0, self._limit * INCREASE_FRACTION))
        return self.value


# --------------------------------------------------------------------------- #
# Admission controller
# --------------------------------------------------------------------------- #


class AdmissionController:
    """Decides whether a new request may start; tracks requests in flight.

    Usage (held for the whole response, including streaming)::

        with controller.admit():
            await handle_request()
    """

    def __init__(self, config: AdmissionConfig, *, metrics: GatewayMetrics | None = None,
                 probe: LagProbe | None = None) -> None:
        """Create the controller.

        Args:
            config: Admission settings.
            metrics: Metrics registry (optional).
            probe: Lag probe (tests inject one); built from config if omitted.
        """
        self._config = config
        self._metrics = metrics
        self._probe = probe or LagProbe(config.probe_interval_ms / 1000, metrics=metrics)
        self._limit = AdaptiveLimit(initial=config.initial_concurrency,
                                    minimum=config.min_concurrency,
                                    maximum=config.max_concurrency,
                                    target_lag_s=config.max_event_loop_lag_ms / 1000)
        self._in_flight = 0
        self._probe.subscribe(self._on_lag)
        self._publish()

    @property
    def in_flight(self) -> int:
        """Requests admitted and not yet finished."""
        return self._in_flight

    @property
    def limit(self) -> int:
        """Current concurrency limit."""
        return self._limit.value

    @property
    def probe(self) -> LagProbe:
        """The lag probe (for tests and diagnostics)."""
        return self._probe

    def check(self) -> None:
        """Admit or refuse one new request (does not count it; see :meth:`admit`).

        Raises:
            OverloadedError: The request should be refused.
        """
        if not self._config.enabled or self._in_flight < self._limit.value:
            return
        if self._metrics is not None:
            self._metrics.load_shed.labels(reason=ShedReason.CONCURRENCY_LIMIT.value).inc()
        raise OverloadedError(
            f"gateway is at capacity ({self._in_flight} requests in progress, event loop "
            f"{self._probe.lag_s * 1000:.0f} ms behind); retry shortly",
            retry_after_s=self._config.retry_after_s,
            details={"reason": ShedReason.CONCURRENCY_LIMIT.value})

    @contextlib.contextmanager
    def admit(self) -> Iterator[None]:
        """Admit one request and count it as in flight until the block exits.

        Raises:
            OverloadedError: The request was refused (nothing is counted).
        """
        self.check()
        self._in_flight += 1
        self._publish()
        try:
            yield
        finally:
            self._in_flight -= 1
            self._publish()

    def start(self) -> None:
        """Start the lag probe (no-op when admission control is disabled)."""
        if not self._config.enabled:
            return
        self._probe.start()
        logger.info("admission control on", extra={
            "target_lag_ms": self._config.max_event_loop_lag_ms,
            "concurrency": [self._config.min_concurrency, self._config.initial_concurrency,
                            self._config.max_concurrency]})

    async def aclose(self) -> None:
        """Stop the lag probe."""
        await self._probe.stop()

    def _on_lag(self, lag_s: float) -> None:
        before = self._limit.value
        after = self._limit.update(lag_s, self._in_flight)
        if after != before and self._metrics is not None:
            self._metrics.concurrency_limit.set(after)

    def _publish(self) -> None:
        if self._metrics is not None:
            self._metrics.in_flight.set(self._in_flight)
            self._metrics.concurrency_limit.set(self._limit.value)
