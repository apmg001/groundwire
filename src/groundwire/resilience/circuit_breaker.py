"""Per-provider circuit breaker.

State machine::

            failures >= threshold
    CLOSED ───────────────────────────► OPEN
      ▲                                  │
      │ probe succeeds                   │ recovery_timeout elapsed
      │                                  ▼
      └─────────────────────────────  HALF_OPEN
                 probe fails ──────────► OPEN (timer restarts)

* CLOSED: calls flow; consecutive failures are counted.
* OPEN: calls are rejected instantly, so a dead provider costs ~0 ms instead
  of a full timeout per request, and gets breathing room to recover.
* HALF_OPEN: up to ``half_open_max_calls`` probe calls are let through; the
  first result decides whether to close or re-open.

Concurrency: the gateway runs one asyncio event loop per worker. Every method
here is synchronous (no ``await``), so each state transition runs atomically
with respect to other coroutines and no lock is required. State is per worker
process; with several workers each keeps its own view, which is the usual
trade-off (a shared Redis-backed breaker is a later extension).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import StrEnum

from groundwire.core.config import CircuitBreakerConfig

logger = logging.getLogger(__name__)

Clock = Callable[[], float]


class CircuitState(StrEnum):
    """Breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


StateListener = Callable[[str, CircuitState], None]


class CircuitBreaker:
    """Consecutive-failure circuit breaker for one provider.

    Usage pattern (the router follows it)::

        if not breaker.try_acquire():
            ...skip provider...
        try:
            result = await call()
        except RetryableError:
            breaker.record_failure()
        else:
            breaker.record_success()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        *,
        clock: Clock = time.monotonic,
        on_state_change: StateListener | None = None,
    ) -> None:
        """Create a closed breaker.

        Args:
            name: Provider name (for logs/metrics).
            config: Thresholds and timings.
            clock: Monotonic clock (injectable for tests).
            on_state_change: Callback ``(name, new_state)`` used for metrics.
        """
        self.name = name
        self._config = config
        self._clock = clock
        self._on_state_change = on_state_change
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_in_flight = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CircuitState:
        """Current state, after applying any due OPEN -> HALF_OPEN transition."""
        self._maybe_half_open()
        return self._state

    @property
    def consecutive_failures(self) -> int:
        """Consecutive failures recorded while closed."""
        return self._consecutive_failures

    # ------------------------------------------------------------------ #
    # Protocol
    # ------------------------------------------------------------------ #

    def try_acquire(self) -> bool:
        """Ask permission to call the provider.

        Returns:
            True if the call may proceed. In HALF_OPEN this reserves a probe
            slot, which the caller releases via ``record_success`` /
            ``record_failure`` / ``release``.
        """
        self._maybe_half_open()
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.HALF_OPEN:
            if self._half_open_in_flight < self._config.half_open_max_calls:
                self._half_open_in_flight += 1
                return True
            return False
        return False

    def record_success(self) -> None:
        """Record a successful call; closes the circuit if it was probing."""
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            self._transition(CircuitState.CLOSED)
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed call that should count against the provider."""
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            self._open()
            return
        self._consecutive_failures += 1
        if (self._state is CircuitState.CLOSED
                and self._consecutive_failures >= self._config.failure_threshold):
            self._open()

    def release(self) -> None:
        """Release a probe slot without a verdict (e.g. the client cancelled,
        or the failure was the client's fault rather than the provider's)."""
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _maybe_half_open(self) -> None:
        if (self._state is CircuitState.OPEN
                and self._clock() - self._opened_at >= self._config.recovery_timeout_s):
            self._half_open_in_flight = 0
            self._transition(CircuitState.HALF_OPEN)

    def _open(self) -> None:
        self._opened_at = self._clock()
        self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        if new_state is self._state:
            return
        old = self._state
        self._state = new_state
        if new_state is CircuitState.CLOSED:
            self._consecutive_failures = 0
        log = logger.warning if new_state is CircuitState.OPEN else logger.info
        log("circuit state changed", extra={"provider": self.name, "from_state": old.value,
                                            "to_state": new_state.value})
        if self._on_state_change is not None:
            try:
                self._on_state_change(self.name, new_state)
            except Exception:
                logger.exception("circuit state listener failed", extra={"provider": self.name})
