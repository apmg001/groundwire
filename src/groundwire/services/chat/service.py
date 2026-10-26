"""The chat completion use case behind ``POST /v1/chat/completions``.

:class:`ChatService` is a thin facade. It decides *which path* a request
takes and records the outcome; the work itself is done by collaborators with
one job each:

=====================  ==================================================
:mod:`.stages`         safety, cache, tool gate, answer check, grounding,
                       request metrics (null objects when switched off)
:mod:`.dispatch`       providers: deadline, cascades, request merging
:mod:`.pipeline`       a full response, step by step
:mod:`.streaming`      a live stream, chunk by chunk
:mod:`.replay`         a finished answer (cache hit or buffered) as a stream
=====================  ==================================================

Paths::

    complete()            ─► CompletionPipeline
    stream(), cascade or
      answer may change   ─► CompletionPipeline ─► Replayer.response
    stream(), cache hit   ─► Replayer.cached
    stream(), otherwise   ─► ProviderDispatcher.open_stream ─► LiveStream

HTTP concerns (status codes, SSE framing, headers) stay in the API layer and
provider concerns in the router and adapters, so every piece here can be
tested with fakes on both sides.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from groundwire.agents.answer_check import AnswerChecker
from groundwire.agents.gate import ToolGate
from groundwire.cache.service import ResponseCache
from groundwire.core.errors import GroundwireError, InvalidRequestError
from groundwire.domain.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    GroundingMode,
)
from groundwire.guardrails.grounding import GroundingVerifier
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.cascade import CascadeRunner
from groundwire.routing.router import Router
from groundwire.safety.guard import SafetyGuard
from groundwire.services.chat.common import CACHE_PROVIDER, grounding_options
from groundwire.services.chat.dispatch import ProviderDispatcher
from groundwire.services.chat.pipeline import CompletionPipeline
from groundwire.services.chat.replay import Replayer
from groundwire.services.chat.stages import (
    AnswerCheckStage,
    CacheStage,
    ChatStages,
    GroundingStage,
    RequestRecorder,
    SafetyStage,
    ToolStage,
)
from groundwire.services.chat.streaming import LiveStream

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamSession:
    """A started stream, ready to be framed as SSE by the API layer.

    Attributes:
        provider: Provider serving the stream (``"cache"`` for cache hits).
        route: Route alias.
        events: JSON-serialisable payloads, one per SSE ``data:`` line
            (excluding the final ``[DONE]`` sentinel, which the API adds).
    """

    provider: str
    route: str
    events: AsyncIterator[dict[str, Any]] = field(repr=False)


class ChatService:
    """Orchestrates a chat completion end to end."""

    def __init__(
        self,
        router: Router,
        verifier: GroundingVerifier,
        metrics: GatewayMetrics,
        *,
        request_timeout_s: float,
        safety: SafetyGuard | None = None,
        cache: ResponseCache | None = None,
        cascade: CascadeRunner | None = None,
        tool_gate: ToolGate | None = None,
        answer_checker: AnswerChecker | None = None,
    ) -> None:
        """Create the service and wire its collaborators.

        Args:
            router: Provider router.
            verifier: Grounding verifier.
            metrics: Metrics sink.
            request_timeout_s: End-to-end deadline for producing a response
                (non-streaming) or the first chunk (streaming).
            safety: Prompt-injection and PII guard; ``None`` disables both.
            cache: Response cache; ``None`` disables caching and coalescing.
            cascade: Cascade runner (built from ``router`` and ``verifier`` if
                omitted; it only acts on routes configured as cascades).
            tool_gate: Guardrails (policy and human approval) for model tool calls;
                ``None`` disables them.
            answer_checker: Fact-check of final answers against tool results;
                ``None`` disables it.

        Raises:
            ValueError: If ``request_timeout_s`` is not positive.
        """
        self._router = router
        self._stages = ChatStages(
            safety=SafetyStage(safety),
            cache=CacheStage(cache),
            tools=ToolStage(tool_gate),
            answer_check=AnswerCheckStage(answer_checker, metrics),
            grounding=GroundingStage(verifier, metrics),
            recorder=RequestRecorder(metrics),
        )
        self._dispatcher = ProviderDispatcher(
            router, cascade or CascadeRunner(router, verifier, metrics=metrics),
            self._stages, metrics, timeout_s=request_timeout_s)
        self._pipeline = CompletionPipeline(self._stages, self._dispatcher)
        self._replayer = Replayer(self._stages)

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Serve a non-streaming chat completion.

        Args:
            request: Validated client request.

        Returns:
            The OpenAI-compatible response with ``groundwire`` metadata.

        Raises:
            ModelNotFoundError: Unknown route.
            GatewayTimeoutError: Deadline exceeded.
            AllProvidersFailedError: Every provider failed.
            ProviderBadRequestError: Upstream rejected the request.
            GroundingViolationError: Grounding failed in ``block`` mode.
            GuardrailUnavailableError: Grounding could not run (fail-closed).
            PromptInjectionError: Injection detected and the policy is ``block``.
            AnswerCheckFailedError: The answer contradicts its tool results and
                the answer check is in ``block`` mode.
        """
        route = self._router.resolve(request.model)
        started = time.perf_counter()
        provider = "none"
        status = "error"
        try:
            response = await self._pipeline.run(request, route)
            provider = str(response.groundwire.get("provider", "none"))
            status = "success"
            return response
        except GroundwireError as exc:
            status = exc.code
            raise
        finally:
            self._stages.recorder.record(route, provider, status, stream=False,
                                         started=started)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    async def stream(self, request: ChatCompletionRequest) -> StreamSession:
        """Start a streaming chat completion.

        Everything that can fail *before* the first byte (bad route, all
        providers down, deadline) raises here, so the API can still return a
        proper HTTP error. Failures after that are delivered in-band as an SSE
        ``error`` payload. Cache hits are replayed as a normal SSE stream.

        Cascade routes cannot stream tokens as they are generated: the answer
        must be complete before it can be fact-checked. They are produced in
        full, checked, then streamed out; the first token arrives later, but
        streaming clients keep working, and grounding ``block`` mode is
        supported because nothing is sent before the check. The same applies to
        agent answers when the answer check is in ``annotate`` or ``block``
        mode: the answer must be complete before it can be changed or refused.

        Args:
            request: Validated client request with ``stream=True``.

        Returns:
            The stream session.

        Raises:
            InvalidRequestError: Grounding ``block`` mode requested with streaming
                on a non-cascade route.
            GroundingViolationError: Grounding failed in ``block`` mode (cascades).
            ModelNotFoundError: Unknown route.
            GatewayTimeoutError: No stream started before the deadline.
            AllProvidersFailedError: Every provider failed to start a stream.
            PromptInjectionError: Injection detected and the policy is ``block``.
            AnswerCheckFailedError: The answer contradicts its tool results
                (answer check in ``block`` mode; raised before any byte is sent).
        """
        route = self._router.resolve(request.model)
        if self._dispatcher.is_cascade(route) or self._stages.answer_check.must_buffer(request):
            return await self._stream_buffered(request, route)
        grounding = grounding_options(request)
        if grounding is not None and grounding.mode is GroundingMode.BLOCK:
            raise InvalidRequestError(
                "grounding mode 'block' cannot be combined with stream=true: tokens "
                "are sent before the answer can be verified. Use mode 'flag' or disable streaming."
            )
        return await self._stream_live(request, route)

    async def _stream_live(self, request: ChatCompletionRequest, route: str) -> StreamSession:
        """Start a token-by-token stream (or replay a cache hit).

        Args:
            request: Client request with ``stream=True``.
            route: Resolved (non-cascade) route alias.

        Returns:
            The stream session.

        Raises:
            GroundwireError: Anything that fails before the first byte.
        """
        started = time.perf_counter()
        try:
            prepared = self._stages.safety.prepare(request)
            lookup = await self._stages.cache.lookup(prepared, route)
            if lookup is not None and lookup.hit:
                events = self._replayer.cached(request, route, lookup,
                                               grounding_options(request), started, prepared)
                return StreamSession(provider=CACHE_PROVIDER, route=route, events=events)
            routed = await self._dispatcher.open_stream(route, prepared)
        except GroundwireError as exc:
            self._stages.recorder.record(route, "none", exc.code, stream=True, started=started)
            raise
        live = LiveStream(self._stages, request=request, route=route, provider=routed.provider,
                          chunks=routed.chunks, prepared=prepared, lookup=lookup,
                          started=started)
        return StreamSession(provider=routed.provider, route=route, events=live.events())

    async def _stream_buffered(self, request: ChatCompletionRequest,
                               route: str) -> StreamSession:
        """Produce the full answer first, then stream it.

        Args:
            request: Client request with ``stream=True``.
            route: Resolved route or cascade alias.

        Returns:
            The stream session.

        Raises:
            GroundwireError: Anything :meth:`complete` raises, before any byte is sent.
        """
        started = time.perf_counter()
        try:
            response = await self._pipeline.run(request, route)
        except GroundwireError as exc:
            self._stages.recorder.record(route, "none", exc.code, stream=True, started=started)
            raise
        provider = str(response.groundwire.get("provider", "none"))
        return StreamSession(provider=provider, route=route,
                             events=self._replayer.response(request, route, response, started))
