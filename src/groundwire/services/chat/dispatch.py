"""Get an answer from the providers: deadline, cascades, and request merging.

:class:`ProviderDispatcher` is the only part of the chat pipeline that talks
to the :class:`~groundwire.routing.router.Router`. It owns three policies:

* **the end-to-end deadline** (``request_timeout_s``) across retries and
  failover, turned into :class:`GatewayTimeoutError`;
* **cascade vs. plain route**: a cascade alias goes through
  :class:`CascadeRunner` (cheap model first, strong one when the answer fails
  the fact-check), anything else through the router;
* **single-flight merging**: identical in-flight requests (same exact cache
  key) share one provider call instead of each paying for it.
"""

from __future__ import annotations

import asyncio
import logging

from groundwire.cache.service import CacheLookup
from groundwire.cache.singleflight import SingleFlight
from groundwire.core.errors import GatewayTimeoutError
from groundwire.domain.models import ChatCompletionRequest
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.cascade import CascadeRunner
from groundwire.routing.router import RoutedStream, Router
from groundwire.safety.guard import PreparedRequest
from groundwire.services.chat.common import Answer, grounding_options, last_user_message
from groundwire.services.chat.stages import ChatStages

logger = logging.getLogger(__name__)


class ProviderDispatcher:
    """Calls the router or a cascade under the deadline."""

    def __init__(self, router: Router, cascade: CascadeRunner, stages: ChatStages,
                 metrics: GatewayMetrics, *, timeout_s: float) -> None:
        """Create the dispatcher.

        Args:
            router: Provider router.
            cascade: Cascade runner (only acts on routes configured as cascades).
            stages: Pipeline stages (PII restoration for cascade checks, the
                cache's coalescing key).
            metrics: Metrics sink.
            timeout_s: End-to-end deadline for an answer (non-streaming) or the
                first chunk (streaming).

        Raises:
            ValueError: If ``timeout_s`` is not positive.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._router = router
        self._cascade = cascade
        self._stages = stages
        self._metrics = metrics
        self._timeout_s = timeout_s
        self._flights: SingleFlight[Answer] = SingleFlight()

    def is_cascade(self, route: str) -> bool:
        """True if ``route`` is a cascade alias.

        Args:
            route: Resolved route alias.

        Returns:
            Whether answers on this route are produced by the cascade runner.
        """
        return self._router.cascade_config(route) is not None

    async def answer(self, route: str, prepared: PreparedRequest, lookup: CacheLookup | None,
                     original: ChatCompletionRequest) -> Answer:
        """Produce a full answer, merging identical in-flight requests.

        Args:
            route: Route or cascade alias.
            prepared: Safety-processed request.
            lookup: Cache lookup (its key drives merging); marked ``coalesced``
                in place when the answer is shared from another request.
            original: The client's request (unredacted question and documents
                for a cascade's fact-check).

        Returns:
            The provider's answer, in redacted form.

        Raises:
            GatewayTimeoutError: Deadline exceeded.
            AllProvidersFailedError: Every provider failed.
            ProviderBadRequestError: Upstream rejected the request.
        """
        key = self._stages.cache.coalescing_key(lookup)
        try:
            async with asyncio.timeout(self._timeout_s):
                if key is None:
                    return await self._call(route, prepared, original)
                answer, shared = await self._flights.run(
                    key, lambda: self._call(route, prepared, original))
        except TimeoutError as exc:
            raise GatewayTimeoutError(
                f"no provider answered within {self._timeout_s:.0f}s",
                details={"route": route},
            ) from exc
        if shared and lookup is not None:
            lookup.coalesced = True
            self._metrics.coalesced_requests.inc()
        return answer

    async def open_stream(self, route: str, prepared: PreparedRequest) -> RoutedStream:
        """Start a provider stream; only the *start* is bounded by the deadline.

        Args:
            route: Route alias (never a cascade: those are buffered).
            prepared: Safety-processed request.

        Returns:
            The started stream.

        Raises:
            GatewayTimeoutError: No provider started streaming in time.
            AllProvidersFailedError: Every provider failed to start a stream.
        """
        try:
            async with asyncio.timeout(self._timeout_s):
                return await self._router.stream(route, prepared.request)
        except TimeoutError as exc:
            raise GatewayTimeoutError(
                f"no provider started streaming within {self._timeout_s:.0f}s",
                details={"route": route},
            ) from exc

    async def _call(self, route: str, prepared: PreparedRequest,
                    original: ChatCompletionRequest) -> Answer:
        """One real provider call (no deadline or merging: the caller adds those).

        Args:
            route: Route or cascade alias.
            prepared: Safety-processed request.
            original: The client's request.

        Returns:
            The answer.

        Raises:
            AllProvidersFailedError: Every provider failed.
            ProviderBadRequestError: Upstream rejected the request.
        """
        if not self.is_cascade(route):
            routed = await self._router.complete(route, prepared.request)
            return Answer(completion=routed.completion, provider=routed.provider,
                          attempts=[a.as_dict() for a in routed.attempts])
        grounding = grounding_options(original)
        outcome = await self._cascade.complete(
            route, prepared.request,
            context=grounding.context if grounding else None,
            question=last_user_message(original),
            restore=lambda text: self._stages.safety.restore(text, prepared))
        return Answer(completion=outcome.routed.completion, provider=outcome.routed.provider,
                      attempts=outcome.attempts, cascade=outcome.decision,
                      report=outcome.report)
