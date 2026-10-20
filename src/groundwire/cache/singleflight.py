"""Request coalescing ("single flight").

When a popular question arrives many times at once (a launch, a retrying
client, a dashboard refreshed by a whole team), every copy would miss the
cache, because the first answer is not stored yet, and hit the provider. That is a
*cache stampede*. :class:`SingleFlight` lets the first request (the
*leader*) make the call while identical requests (*followers*) wait for its
result.

Failure semantics, chosen so a follower is never worse off than without
coalescing:

* the leader **fails**: followers receive the same exception (they would
  have hit the same providers);
* the leader is **cancelled** (its client disconnected): followers do *not*
  inherit the cancellation; the next one becomes leader and retries;
* a follower is cancelled: only that follower stops; the leader is unaffected.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _consume_outcome(future: asyncio.Future[Any]) -> None:
    """Mark a future's exception as retrieved (avoids 'never retrieved' warnings).

    Args:
        future: Completed future.
    """
    if not future.cancelled():
        future.exception()


class SingleFlight(Generic[T]):
    """Deduplicates concurrent calls that share a key."""

    def __init__(self) -> None:
        """Create an empty coalescer."""
        self._inflight: dict[str, asyncio.Future[T]] = {}

    def inflight(self) -> int:
        """Return the number of keys currently being computed."""
        return len(self._inflight)

    async def run(self, key: str, func: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Run ``func`` once per key at a time; share the result with followers.

        Args:
            key: Deduplication key (the exact cache key).
            func: Zero-argument coroutine factory that does the real work.

        Returns:
            ``(result, shared)``, where ``shared`` is True for followers that
            reused another request's result.

        Raises:
            Exception: Whatever ``func`` raises (for the leader and its followers).
            asyncio.CancelledError: If *this* caller is cancelled.
        """
        while True:
            existing = self._inflight.get(key)
            if existing is None:
                return await self._lead(key, func), False
            # asyncio.wait never raises the future's exception or cancellation,
            # so a cancelled leader can't take its followers down with it.
            await asyncio.wait({existing})
            if existing.cancelled():
                logger.debug("coalesced leader was cancelled; retrying as leader")
                continue
            return existing.result(), True

    async def _lead(self, key: str, func: Callable[[], Awaitable[T]]) -> T:
        """Execute ``func`` as the leader and publish the outcome.

        Args:
            key: Deduplication key.
            func: Work to run.

        Returns:
            The result.

        Raises:
            Exception: Whatever ``func`` raises.
            asyncio.CancelledError: If the leader is cancelled.
        """
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_outcome)
        self._inflight[key] = future
        try:
            result = await func()
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            self._inflight.pop(key, None)
