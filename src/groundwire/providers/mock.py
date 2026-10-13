"""Deterministic mock provider.

Used for three things:

* unit/integration tests without network,
* load testing the gateway itself (so you measure gateway overhead, not a
  laptop CPU generating tokens),
* chaos testing: ``failure_rate`` injects transient server errors so you can
  watch retries, circuit breakers and failover work under real traffic.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator

from groundwire.core.config import MockBehaviour
from groundwire.core.errors import ProviderServerError
from groundwire.domain.models import ChatCompletionRequest, Completion, StreamChunk, Usage
from groundwire.providers.base import Provider

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for mock usage accounting."""
    return max(1, len(text) // 4)


class MockProvider(Provider):
    """In-process fake provider with configurable latency and failures."""

    def __init__(self, *, name: str, model: str, behaviour: MockBehaviour) -> None:
        """Create the mock.

        Args:
            name: Provider name.
            model: Reported model id.
            behaviour: Response text, latency and failure injection settings.
        """
        super().__init__(name, model)
        self._behaviour = behaviour
        self._rng = random.Random(behaviour.seed)  # noqa: S311 - not crypto
        self.calls = 0  # exposed for tests

    async def _simulate(self) -> None:
        """Apply latency and maybe inject a failure.

        Raises:
            ProviderServerError: With probability ``failure_rate``.
        """
        self.calls += 1
        if self._behaviour.latency_s:
            await asyncio.sleep(self._behaviour.latency_s)
        if self._behaviour.failure_rate and self._rng.random() < self._behaviour.failure_rate:
            raise ProviderServerError(f"{self.name}: injected failure", provider=self.name,
                                      upstream_status=503)

    def _usage(self, request: ChatCompletionRequest) -> Usage:
        prompt = sum(_approx_tokens(m.content) for m in request.messages)
        return Usage.of(prompt, _approx_tokens(self._behaviour.response_text))

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        """Return the configured text. See :meth:`Provider.complete`."""
        await self._simulate()
        return Completion(text=self._behaviour.response_text, finish_reason="stop",
                          usage=self._usage(request), upstream_model=self.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream the configured text word by word. See :meth:`Provider.stream`."""
        await self._simulate()
        words = self._behaviour.response_text.split(" ")
        for i, word in enumerate(words):
            yield StreamChunk.text(word if i == 0 else " " + word, role=i == 0)
            await asyncio.sleep(0)  # yield control like a real network stream
        yield StreamChunk.finish("stop", self._usage(request))
