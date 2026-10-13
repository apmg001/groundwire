"""Chat completion service: the use case behind ``POST /v1/chat/completions``.

Responsibilities (and nothing else):

* resolve the route from the request's ``model``,
* enforce the end-to-end deadline across retries and failover,
* run grounding verification when the client asks for it and apply the
  flag/block policy,
* record request-level metrics,
* shape the public response.

HTTP concerns (status codes, SSE framing, headers) stay in the API layer, and
provider concerns stay in the router/adapters, so this class is easy to test
with fakes on both sides.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from groundwire.core.errors import (
    GatewayTimeoutError,
    GroundingViolationError,
    GroundwireError,
    InvalidRequestError,
)
from groundwire.domain.models import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    GroundingMode,
    GroundingOptions,
    StreamChunk,
)
from groundwire.guardrails.grounding import GroundingReport, GroundingVerifier, Verdict
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.router import Router

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StreamSession:
    """A started stream, ready to be framed as SSE by the API layer.

    Attributes:
        provider: Provider serving the stream.
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
    ) -> None:
        """Create the service.

        Args:
            router: Provider router.
            verifier: Grounding verifier.
            metrics: Metrics sink.
            request_timeout_s: End-to-end deadline for producing a response
                (non-streaming) or the first chunk (streaming).
        """
        self._router = router
        self._verifier = verifier
        self._metrics = metrics
        self._timeout_s = request_timeout_s

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
        """
        route = self._router.resolve(request.model)
        started = time.perf_counter()
        provider = "none"
        status = "error"
        try:
            try:
                async with asyncio.timeout(self._timeout_s):
                    routed = await self._router.complete(route, request)
            except TimeoutError as exc:
                raise GatewayTimeoutError(
                    f"no provider answered within {self._timeout_s:.0f}s",
                    details={"route": route},
                ) from exc
            provider = routed.provider

            meta: dict[str, Any] = {
                "route": route,
                "provider": routed.provider,
                "upstream_model": routed.completion.upstream_model,
                "attempts": [a.as_dict() for a in routed.attempts],
            }
            grounding = request.groundwire.grounding if request.groundwire else None
            if grounding is not None:
                report = await self._check_grounding(routed.completion.text, grounding)
                meta["grounding"] = report.as_dict()

            status = "success"
            return ChatCompletionResponse(
                model=request.model,
                choices=[Choice(message=AssistantMessage(content=routed.completion.text),
                                finish_reason=routed.completion.finish_reason)],
                usage=routed.completion.usage,
                groundwire=meta,
            )
        except GroundwireError as exc:
            status = exc.code
            raise
        finally:
            self._record_request(route, provider, status, stream=False, started=started)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    async def stream(self, request: ChatCompletionRequest) -> StreamSession:
        """Start a streaming chat completion.

        Everything that can fail *before* the first byte (bad route, all
        providers down, deadline) raises here, so the API can still return a
        proper HTTP error. Failures after that are delivered in-band as an SSE
        ``error`` payload.

        Args:
            request: Validated client request with ``stream=True``.

        Returns:
            The stream session.

        Raises:
            InvalidRequestError: Grounding ``block`` mode requested with streaming.
            ModelNotFoundError: Unknown route.
            GatewayTimeoutError: No stream started before the deadline.
            AllProvidersFailedError: Every provider failed to start a stream.
        """
        grounding = request.groundwire.grounding if request.groundwire else None
        if grounding is not None and grounding.mode is GroundingMode.BLOCK:
            raise InvalidRequestError(
                "grounding mode 'block' cannot be combined with stream=true: tokens "
                "are sent before the answer can be verified. Use mode 'flag' or disable streaming."
            )

        route = self._router.resolve(request.model)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_s):
                routed = await self._router.stream(route, request)
        except TimeoutError as exc:
            self._record_request(route, "none", GatewayTimeoutError.code, stream=True,
                                 started=started)
            raise GatewayTimeoutError(
                f"no provider started streaming within {self._timeout_s:.0f}s",
                details={"route": route},
            ) from exc
        except GroundwireError as exc:
            self._record_request(route, "none", exc.code, stream=True, started=started)
            raise

        events = self._stream_events(request, route, routed.provider, routed.chunks,
                                     grounding, started)
        return StreamSession(provider=routed.provider, route=route, events=events)

    async def _stream_events(
        self,
        request: ChatCompletionRequest,
        route: str,
        provider: str,
        chunks: AsyncIterator[StreamChunk],
        grounding: GroundingOptions | None,
        started: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert router chunks into SSE payloads.

        Args:
            request: Client request.
            route: Route alias.
            provider: Serving provider.
            chunks: Router chunk iterator.
            grounding: Grounding options (``flag`` mode only).
            started: ``perf_counter`` at request start.

        Yields:
            JSON-serialisable payloads.
        """
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        parts: list[str] = []
        status = "error"
        try:
            async for chunk in chunks:
                chunk.id, chunk.created, chunk.model = stream_id, created, request.model
                if grounding is not None:
                    parts.append(chunk.content())
                yield chunk.model_dump(exclude_none=True)

            if grounding is not None:
                report = await self._check_grounding("".join(parts), grounding)
                yield {"id": stream_id, "object": "chat.completion.chunk", "created": created,
                       "model": request.model, "choices": [],
                       "groundwire": {"route": route, "provider": provider,
                                      "grounding": report.as_dict()}}
            status = "success"
        except GroundwireError as exc:
            status = exc.code
            logger.error("stream aborted", extra={"route": route, "provider": provider,
                                                  "error_code": exc.code})
            yield exc.to_payload()
        except (GeneratorExit, asyncio.CancelledError):
            status = "client_disconnected"
            logger.info("client disconnected mid-stream",
                        extra={"route": route, "provider": provider})
            raise
        finally:
            self._record_request(route, provider, status, stream=True, started=started)
            aclose = getattr(chunks, "aclose", None)
            if aclose is not None:
                await aclose()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _check_grounding(self, answer: str, options: GroundingOptions) -> GroundingReport:
        """Run grounding and apply the request's policy.

        Args:
            answer: Generated text.
            options: Grounding options.

        Returns:
            The report (when the answer passes, or fails in ``flag`` mode).

        Raises:
            GroundingViolationError: Failed in ``block`` mode.
            GuardrailUnavailableError: Scorer failed and policy is fail-closed.
        """
        report = await self._verifier.verify(answer, options.context)
        self._metrics.grounding.labels(verdict=report.verdict.value).inc()
        self._metrics.grounding_latency.observe(report.latency_ms / 1000)
        logger.info("grounding checked", extra={
            "verdict": report.verdict.value, "supported_ratio": round(report.supported_ratio, 3),
            "claims": len(report.claims), "latency_ms": round(report.latency_ms, 1)})

        if report.verdict is Verdict.FAILED and options.mode is GroundingMode.BLOCK:
            raise GroundingViolationError(
                "answer is not sufficiently supported by the provided context",
                details={"grounding": report.as_dict()},
            )
        return report

    def _record_request(self, route: str, provider: str, status: str, *, stream: bool,
                        started: float) -> None:
        stream_label = "true" if stream else "false"
        self._metrics.requests.labels(route=route, provider=provider, status=status,
                                      stream=stream_label).inc()
        self._metrics.request_latency.labels(route=route, stream=stream_label).observe(
            time.perf_counter() - started)
