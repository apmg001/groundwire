"""Retry policy with capped exponential backoff and full jitter.

Why full jitter: when a provider blips, hundreds of in-flight requests fail at
the same moment. Retrying them all after exactly 200 ms just recreates the
spike. Randomising each delay over ``[0, min(cap, base * 2**n)]`` spreads the
retries out (see the AWS Architecture Blog, "Exponential Backoff and Jitter").

Only errors marked ``retryable`` are retried, and timeouts only when
``retry_on_timeout`` is set: a provider that just used its whole timeout is
likely to again, so by default the router fails over at once instead of
making the client wait a second full timeout. A ``Retry-After`` hint from the
provider is honoured (capped at ``max_delay_s`` so one upstream cannot stall
the gateway).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from groundwire.core.config import RetryConfig
from groundwire.core.errors import ProviderError, ProviderTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")

SleepFn = Callable[[float], Awaitable[None]]


class RetryPolicy:
    """Retries an async operation on retryable :class:`ProviderError`.

    Attributes:
        config: Retry settings.
    """

    def __init__(
        self,
        config: RetryConfig,
        *,
        sleep: SleepFn = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        """Create the policy.

        Args:
            config: Retry settings.
            sleep: Sleep function (injectable for fast, deterministic tests).
            rng: Random source for jitter (injectable for tests).
        """
        self.config = config
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not crypto

    def backoff_delay(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Compute the delay before the next attempt.

        Args:
            attempt: 1-based number of the attempt that just failed.
            retry_after_s: Upstream ``Retry-After`` hint, if any.

        Returns:
            Delay in seconds, within ``[0, max_delay_s]``.
        """
        cap = self.config.max_delay_s
        if retry_after_s is not None:
            return min(cap, retry_after_s)
        ceiling = min(cap, self.config.base_delay_s * (2 ** (attempt - 1)))
        return self._rng.uniform(0, ceiling)

    def _should_retry(self, exc: ProviderError) -> bool:
        """Whether to retry ``exc`` on the same provider (failover is the router's job).

        Args:
            exc: The error from the attempt that just failed.

        Returns:
            True for retryable errors, except timeouts unless ``retry_on_timeout``.
        """
        if isinstance(exc, ProviderTimeoutError) and not self.config.retry_on_timeout:
            return False
        return exc.retryable

    async def run(self, operation: Callable[[], Awaitable[T]], *, provider: str) -> T:
        """Run ``operation``, retrying retryable provider errors.

        Args:
            operation: Zero-arg coroutine factory; called once per attempt.
            provider: Provider name, for logging.

        Returns:
            The operation's result.

        Raises:
            ProviderError: The last error, when it is not retryable or attempts
                are exhausted.
        """
        attempt = 1
        while True:
            try:
                return await operation()
            except ProviderError as exc:
                if not self._should_retry(exc) or attempt >= self.config.max_attempts:
                    raise
                delay = self.backoff_delay(attempt, exc.retry_after_s)
                logger.warning(
                    "retrying provider call",
                    extra={"provider": provider, "attempt": attempt,
                           "error_code": exc.code, "delay_s": round(delay, 3)},
                )
                await self._sleep(delay)
                attempt += 1
