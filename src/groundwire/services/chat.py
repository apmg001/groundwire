"""Chat completion service: the use case behind ``POST /v1/chat/completions``.

Responsibilities (and nothing else):

* resolve the route from the request's ``model``,
* apply the safety guard (injection screening, PII redaction/restoration),
* serve from the response cache when possible, and merge identical
  in-flight requests into one provider call,
* run cascade routes (cheap model first, strong model only when the answer
  fails the fact-check) through :class:`CascadeRunner`,
* enforce the tool gate on every tool call a model requests, after PII is
  restored and before the client (the agent) receives it: policy rules, and
  human approval for calls a rule holds,
* fact-check an agent's final answer against the tool results in the
  conversation (numbers and claimed actions), and flag/annotate/block it,
* enforce the end-to-end deadline across retries and failover,
* run grounding verification when the client asks for it and apply the
  flag/block policy,
* record request-level metrics,
* shape the public response.

Pipeline for one request::

    safety.prepare ─► cache.lookup ─┬─ hit ──────────────────────┐
                                    └─ miss ─► router (deadline) ─┤
                                                                  ▼
                         cache.store ◄─ grounding ◄─ safety.restore

The cache sits *after* redaction and *before* restoration, so it only ever
holds placeholder text, and answers that fail grounding are never stored.

HTTP concerns (status codes, SSE framing, headers) stay in the API layer, and
provider concerns stay in the router/adapters, so this class is easy to test
with fakes on both sides.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from groundwire.agents.answer_check import AnswerChecker, AnswerCheckReport
from groundwire.agents.gate import ToolGate
from groundwire.agents.policy import PolicyResult
from groundwire.cache.service import CacheLookup, CacheStatus, ResponseCache
from groundwire.cache.singleflight import SingleFlight
from groundwire.core.config import AnswerCheckAction
from groundwire.core.errors import (
    AnswerCheckFailedError,
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
    Completion,
    FinishReason,
    GroundingMode,
    GroundingOptions,
    StreamChunk,
    ToolCall,
    ToolCallAssembler,
    Usage,
)
from groundwire.guardrails.grounding import GroundingReport, GroundingVerifier, Verdict
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.cascade import CascadeDecision, CascadeRunner
from groundwire.routing.router import Router
from groundwire.safety.guard import PreparedRequest, SafetyGuard
from groundwire.safety.pii import StreamRestorer

logger = logging.getLogger(__name__)

CACHE_PROVIDER = "cache"

# Word-sized pieces (with their trailing whitespace) for replaying cached answers.
_STREAM_PIECE = re.compile(r"\s*\S+\s*|\s+")


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


@dataclass(slots=True)
class _Answer:
    """A completion in redacted form, from a provider or the cache.

    Attributes:
        completion: The completion (text still contains PII placeholders).
        provider: Who served it (a provider name or ``"cache"``).
        attempts: Provider attempts, as response metadata.
        cascade: The cascade decision, for cascade routes.
        report: Grounding report already computed on this answer (a cascade's
            cheap-tier check), reused instead of checking twice.
    """

    completion: Completion
    provider: str
    attempts: list[dict[str, Any]]
    cascade: CascadeDecision | None = None
    report: GroundingReport | None = None


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
        """Create the service.

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
        """
        self._router = router
        self._verifier = verifier
        self._metrics = metrics
        self._timeout_s = request_timeout_s
        self._safety = safety
        self._cache = cache
        self._cascade = cascade or CascadeRunner(router, verifier, metrics=metrics)
        self._tool_gate = tool_gate
        self._answer_checker = answer_checker
        self._flights: SingleFlight[_Answer] = SingleFlight()

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
            response = await self._respond(request, route)
            provider = str(response.groundwire.get("provider", "none"))
            status = "success"
            return response
        except GroundwireError as exc:
            status = exc.code
            raise
        finally:
            self._record_request(route, provider, status, stream=False, started=started)

    async def _respond(self, request: ChatCompletionRequest, route: str
                       ) -> ChatCompletionResponse:
        """Build a full (non-streamed) response: safety, cache, routing, grounding.

        Args:
            request: Client request.
            route: Resolved route or cascade alias.

        Returns:
            The response.

        Raises:
            GroundwireError: Any error :meth:`complete` documents.
        """
        prepared = self._prepare(request)
        lookup = await self._cache_lookup(prepared, route)
        if lookup is not None and lookup.hit and lookup.entry is not None:
            answer = _Answer(completion=_completion_from_lookup(lookup),
                             provider=CACHE_PROVIDER, attempts=[])
        else:
            answer = await self._call_provider(route, prepared, lookup, request)

        text = self._restore(answer.completion.text, prepared)
        proposed = self._restore_calls(answer.completion.tool_calls, prepared)
        calls, policy = await self._enforce_tools(proposed, request, prepared)
        finish_reason = answer.completion.finish_reason
        if policy is not None and proposed and not calls:
            # No call can run now (blocked or waiting for approval): answer in text,
            # so the agent loop ends cleanly.
            text = (text + "\n\n" if text.strip() else "") + policy.explanation()
            finish_reason = "stop"
        meta: dict[str, Any] = {
            "route": route,
            "provider": answer.provider,
            "upstream_model": answer.completion.upstream_model,
            "attempts": answer.attempts,
        }
        if answer.cascade is not None:
            meta["cascade"] = answer.cascade.as_dict()
        safety_meta = prepared.metadata()
        if safety_meta:
            meta["safety"] = safety_meta
        if lookup is not None:
            meta["cache"] = lookup.as_dict()
        if policy is not None:
            meta["tools"] = policy.as_dict()

        # Only a *final* answer is checked: when the model asked for tools (even
        # blocked ones), the text is a preamble or the gateway's own explanation.
        check: AnswerCheckReport | None = None
        if not proposed:
            check = self._check_answer(text, request)
            if check is not None:
                meta["answer_check"] = check.as_dict()

        report: GroundingReport | None = None
        grounding = request.groundwire.grounding if request.groundwire else None
        if grounding is not None and proposed and not answer.completion.text.strip():
            # A tool call is a request for action, not a factual claim: nothing
            # to check against documents (tool policies govern calls instead).
            meta["grounding"] = {"verdict": "skipped", "reason": "tool_calls"}
        elif grounding is not None:
            # A coalesced answer was checked with another request's PII values,
            # so only reuse a precomputed report for our own answer.
            reusable = None if lookup is not None and lookup.coalesced else answer.report
            # Ground the *restored* answer against the original (unredacted) context.
            report = await self._check_grounding(text, grounding, _last_user_message(request),
                                                 precomputed=reusable)
            meta["grounding"] = report.as_dict()

        if (lookup is not None and not lookup.hit and not lookup.coalesced
                and not (check is not None and check.failed)):
            await self._cache_store(lookup, answer.completion, answer.provider, report)

        if (check is not None and check.failed and text.strip()
                and self._answer_check_action() is AnswerCheckAction.ANNOTATE):
            # After grounding, so the warning itself is never fact-checked.
            text = f"{text}\n\n{check.warning()}"

        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=AssistantMessage(
                                content=text if text or not calls else None,
                                tool_calls=calls or None),
                            finish_reason=finish_reason)],
            usage=answer.completion.usage,
            groundwire=meta,
        )

    async def _call_provider(self, route: str, prepared: PreparedRequest,
                             lookup: CacheLookup | None,
                             original: ChatCompletionRequest) -> _Answer:
        """Call the router (or a cascade) under the deadline, coalescing identical requests.

        Args:
            route: Route or cascade alias.
            prepared: Safety-processed request.
            lookup: Cache lookup (its key drives coalescing); updated in place
                when the answer is shared from another in-flight request.
            original: The client's request (unredacted question and documents
                for a cascade's fact-check).

        Returns:
            The provider's answer.

        Raises:
            GatewayTimeoutError: Deadline exceeded.
            AllProvidersFailedError: Every provider failed.
            ProviderBadRequestError: Upstream rejected the request.
        """
        async def call() -> _Answer:
            if self._router.cascade_config(route) is not None:
                grounding = original.groundwire.grounding if original.groundwire else None
                outcome = await self._cascade.complete(
                    route, prepared.request,
                    context=grounding.context if grounding else None,
                    question=_last_user_message(original),
                    restore=lambda text: self._restore(text, prepared))
                return _Answer(completion=outcome.routed.completion,
                               provider=outcome.routed.provider, attempts=outcome.attempts,
                               cascade=outcome.decision, report=outcome.report)
            routed = await self._router.complete(route, prepared.request)
            return _Answer(completion=routed.completion, provider=routed.provider,
                           attempts=[a.as_dict() for a in routed.attempts])

        coalesce = (self._cache is not None and self._cache.coalesce and lookup is not None
                    and lookup.key is not None
                    and lookup.status in (CacheStatus.MISS, CacheStatus.ERROR))
        try:
            async with asyncio.timeout(self._timeout_s):
                if coalesce and lookup is not None and lookup.key is not None:
                    answer, shared = await self._flights.run(lookup.key.exact, call)
                    if shared:
                        lookup.coalesced = True
                        self._metrics.coalesced_requests.inc()
                else:
                    answer = await call()
        except TimeoutError as exc:
            raise GatewayTimeoutError(
                f"no provider answered within {self._timeout_s:.0f}s",
                details={"route": route},
            ) from exc
        return answer

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
        grounding = request.groundwire.grounding if request.groundwire else None
        route = self._router.resolve(request.model)
        if self._router.cascade_config(route) is not None or self._must_buffer(request):
            return await self._stream_buffered(request, route)
        if grounding is not None and grounding.mode is GroundingMode.BLOCK:
            raise InvalidRequestError(
                "grounding mode 'block' cannot be combined with stream=true: tokens "
                "are sent before the answer can be verified. Use mode 'flag' or disable streaming."
            )

        started = time.perf_counter()
        try:
            prepared = self._prepare(request)
            lookup = await self._cache_lookup(prepared, route)
            if lookup is not None and lookup.hit:
                events = self._replay_events(request, route, lookup, grounding, started,
                                             prepared)
                return StreamSession(provider=CACHE_PROVIDER, route=route, events=events)
            async with asyncio.timeout(self._timeout_s):
                routed = await self._router.stream(route, prepared.request)
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
                                     grounding, started, prepared, lookup)
        return StreamSession(provider=routed.provider, route=route, events=events)

    async def _stream_buffered(self, request: ChatCompletionRequest,
                               route: str) -> StreamSession:
        """Produce the full answer first (cascades), then stream it.

        Args:
            request: Client request with ``stream=True``.
            route: Resolved cascade alias.

        Returns:
            The stream session.

        Raises:
            GroundwireError: Anything :meth:`complete` raises, before any byte is sent.
        """
        started = time.perf_counter()
        try:
            response = await self._respond(request, route)
        except GroundwireError as exc:
            self._record_request(route, "none", exc.code, stream=True, started=started)
            raise
        provider = str(response.groundwire.get("provider", "none"))
        return StreamSession(provider=provider, route=route,
                             events=self._replay_response(request, route, response, started))

    async def _replay_response(self, request: ChatCompletionRequest, route: str,
                               response: ChatCompletionResponse,
                               started: float) -> AsyncIterator[dict[str, Any]]:
        """Stream a finished response as OpenAI-style SSE payloads.

        Args:
            request: Client request.
            route: Route alias.
            response: The finished response (text already PII-restored).
            started: ``perf_counter`` at request start.

        Yields:
            Content chunks, the finish chunk, then the ``groundwire`` metadata chunk.
        """
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        provider = str(response.groundwire.get("provider", "none"))
        choice = response.choices[0]
        status = "error"

        def stamp(chunk: StreamChunk) -> dict[str, Any]:
            chunk.id, chunk.created, chunk.model = stream_id, created, request.model
            return chunk.model_dump(exclude_none=True)

        try:
            pieces = _STREAM_PIECE.findall(choice.message.content or "")
            for index, piece in enumerate(pieces):
                yield stamp(StreamChunk.text(piece, role=index == 0))
            if choice.message.tool_calls:
                yield stamp(StreamChunk.tool_calls(choice.message.tool_calls, role=not pieces))
            yield stamp(StreamChunk.finish(choice.finish_reason, response.usage))
            yield {"id": stream_id, "object": "chat.completion.chunk", "created": created,
                   "model": request.model, "choices": [], "groundwire": response.groundwire}
            status = "success"
        except (GeneratorExit, asyncio.CancelledError):
            status = "client_disconnected"
            raise
        finally:
            self._record_request(route, provider, status, stream=True, started=started)

    async def _stream_events(
        self,
        request: ChatCompletionRequest,
        route: str,
        provider: str,
        chunks: AsyncIterator[StreamChunk],
        grounding: GroundingOptions | None,
        started: float,
        prepared: PreparedRequest,
        lookup: CacheLookup | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert router chunks into SSE payloads.

        Args:
            request: Client request.
            route: Route alias.
            provider: Serving provider.
            chunks: Router chunk iterator.
            grounding: Grounding options (``flag`` mode only).
            started: ``perf_counter`` at request start.
            prepared: Safety-processed request (for PII restoration).
            lookup: Cache lookup; a complete answer is stored when writable.

        Yields:
            JSON-serialisable payloads.
        """
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        parts: list[str] = []
        raw_parts: list[str] = []  # redacted text, for the cache
        finish_reason: FinishReason | None = None
        usage = Usage()
        upstream_model = provider
        status = "error"
        restorer: StreamRestorer | None = (self._safety.stream_restorer(prepared)
                                           if self._safety else None)
        # Text streams live; tool calls are assembled and sent once complete
        # (a partial call is useless to an agent and cannot be checked).
        assembler = ToolCallAssembler()
        calls: list[ToolCall] = []
        policy_result: PolicyResult | None = None
        released = False
        sent_text = False
        # Answer check in flag mode runs on the streamed text once it is complete
        # (other modes never get here: they use the buffered path).
        check_answer = self._answer_checker is not None and self._answer_checker.applies_to(
            request)
        collect_text = grounding is not None or check_answer

        async def release_calls() -> dict[str, Any] | None:
            nonlocal calls, policy_result, released
            if not assembler or released:
                return None
            released = True
            proposed = self._restore_calls(assembler.calls(), prepared)
            calls, policy_result = await self._enforce_tools(proposed, request, prepared)
            stamp = {"id": stream_id, "created": created, "model": request.model}
            if calls:
                return StreamChunk.tool_calls(calls, role=not sent_text).model_copy(
                    update=stamp).model_dump(exclude_none=True)
            if policy_result is not None:  # no call can run now: explain in text
                prefix = "\n\n" if sent_text else ""
                return StreamChunk.text(prefix + policy_result.explanation(),
                                        role=not sent_text).model_copy(
                    update=stamp).model_dump(exclude_none=True)
            return None

        try:
            async for chunk in chunks:
                if chunk.model:
                    upstream_model = chunk.model
                chunk.id, chunk.created, chunk.model = stream_id, created, request.model
                deltas = chunk.tool_call_deltas()
                if deltas:
                    assembler.feed(deltas)
                    for choice in chunk.choices:
                        choice.delta.tool_calls = None
                raw_parts.append(chunk.content())
                for choice in chunk.choices:
                    finish_reason = choice.finish_reason or finish_reason
                usage = chunk.usage or usage
                if restorer is not None:
                    _restore_chunk(chunk, restorer)
                    if chunk.choices and chunk.choices[0].finish_reason is not None:
                        # Release anything held back *before* the finish marker.
                        tail = restorer.flush()
                        if tail:
                            delta = chunk.choices[0].delta
                            delta.content = (delta.content or "") + tail
                if collect_text:
                    parts.append(chunk.content())
                finishing = any(c.finish_reason is not None for c in chunk.choices)
                if finishing:
                    ready = await release_calls()  # complete calls go out before the finish marker
                    if ready is not None:
                        yield ready
                    if policy_result is not None and not calls:
                        for choice in chunk.choices:  # no calls left: the turn ends in text
                            if choice.finish_reason == "tool_calls":
                                choice.finish_reason = "stop"
                if chunk.content() or chunk.usage or finishing:
                    sent_text = sent_text or bool(chunk.content())
                    yield chunk.model_dump(exclude_none=True)

            if restorer is not None:
                tail = restorer.flush()
                if tail:
                    parts.append(tail)
                    yield StreamChunk.text(tail).model_copy(update={
                        "id": stream_id, "created": created, "model": request.model,
                    }).model_dump(exclude_none=True)
            ready = await release_calls()  # stream ended without a finish marker
            if ready is not None:
                yield ready

            report: GroundingReport | None = None
            tool_only = bool(assembler) and not "".join(parts).strip()
            if grounding is not None and not tool_only:
                report = await self._check_grounding("".join(parts), grounding,
                                                     _last_user_message(request))
            if lookup is not None:
                await self._cache_store(
                    lookup, Completion(text="".join(raw_parts), finish_reason=finish_reason,
                                       usage=usage, upstream_model=upstream_model),
                    provider, report)
            extra = self._stream_metadata(route, provider, prepared, lookup, report)
            if grounding is not None and tool_only:
                extra = extra or {"route": route, "provider": provider}
                extra["grounding"] = {"verdict": "skipped", "reason": "tool_calls"}
            if policy_result is not None:
                extra = extra or {"route": route, "provider": provider}
                extra["tools"] = policy_result.as_dict()
            if check_answer and not assembler:
                check = self._check_answer("".join(parts), request)
                if check is not None:
                    extra = extra or {"route": route, "provider": provider}
                    extra["answer_check"] = check.as_dict()
            if extra is not None:
                yield {"id": stream_id, "object": "chat.completion.chunk", "created": created,
                       "model": request.model, "choices": [], "groundwire": extra}
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

    async def _replay_events(
        self,
        request: ChatCompletionRequest,
        route: str,
        lookup: CacheLookup,
        grounding: GroundingOptions | None,
        started: float,
        prepared: PreparedRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay a cached answer as an OpenAI-style SSE stream.

        Args:
            request: Client request.
            route: Route alias.
            lookup: The cache hit.
            grounding: Grounding options (``flag`` mode only).
            started: ``perf_counter`` at request start.
            prepared: Safety-processed request (for PII restoration).

        Yields:
            JSON-serialisable payloads.
        """
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        status = "error"
        completion = _completion_from_lookup(lookup)
        text = self._restore(completion.text, prepared)

        def stamp(chunk: StreamChunk) -> dict[str, Any]:
            chunk.id, chunk.created, chunk.model = stream_id, created, request.model
            return chunk.model_dump(exclude_none=True)

        try:
            for index, piece in enumerate(_STREAM_PIECE.findall(text)):
                yield stamp(StreamChunk.text(piece, role=index == 0))
            yield stamp(StreamChunk.finish(completion.finish_reason, completion.usage))

            report: GroundingReport | None = None
            if grounding is not None:
                report = await self._check_grounding(text, grounding,
                                                     _last_user_message(request))
            extra = self._stream_metadata(route, CACHE_PROVIDER, prepared, lookup, report)
            if extra is not None:
                yield {"id": stream_id, "object": "chat.completion.chunk", "created": created,
                       "model": request.model, "choices": [], "groundwire": extra}
            status = "success"
        except GroundwireError as exc:
            status = exc.code
            logger.error("cached stream aborted", extra={"route": route,
                                                         "error_code": exc.code})
            yield exc.to_payload()
        except (GeneratorExit, asyncio.CancelledError):
            status = "client_disconnected"
            raise
        finally:
            self._record_request(route, CACHE_PROVIDER, status, stream=True, started=started)

    @staticmethod
    def _stream_metadata(route: str, provider: str, prepared: PreparedRequest,
                         lookup: CacheLookup | None,
                         report: GroundingReport | None) -> dict[str, Any] | None:
        """Build the trailing ``groundwire`` metadata chunk, if there is anything to say.

        Args:
            route: Route alias.
            provider: Serving provider.
            prepared: Safety-processed request.
            lookup: Cache lookup, if caching is on.
            report: Grounding report, if grounding ran.

        Returns:
            The metadata, or ``None`` when no feature produced any.
        """
        extra: dict[str, Any] = {"route": route, "provider": provider}
        safety_meta = prepared.metadata()
        if safety_meta:
            extra["safety"] = safety_meta
        if lookup is not None:
            extra["cache"] = lookup.as_dict()
        if report is not None:
            extra["grounding"] = report.as_dict()
        return extra if len(extra) > 2 else None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _prepare(self, request: ChatCompletionRequest) -> PreparedRequest:
        """Apply the safety guard (if any) to a request.

        Args:
            request: Client request.

        Returns:
            The prepared request.

        Raises:
            PromptInjectionError: Injection detected and policy is ``block``.
            GuardrailUnavailableError: The injection detector failed.
        """
        if self._safety is None:
            return PreparedRequest(request=request)
        grounding = request.groundwire.grounding if request.groundwire else None
        return self._safety.prepare(request, grounding.context if grounding else None)

    async def _enforce_tools(self, calls: list[ToolCall], request: ChatCompletionRequest,
                             prepared: PreparedRequest
                             ) -> tuple[list[ToolCall], PolicyResult | None]:
        """Run the tool gate on the calls a model requested.

        Runs on PII-restored calls, so rules and approvals see real values (an
        email-domain rule must see the real address, not ``[EMAIL_1]``).

        Args:
            calls: Proposed calls.
            request: The client's request (offered tools, history, ``user``).
            prepared: Safety-processed request (injection verdict).

        Returns:
            ``(calls_to_release, result)``; ``result`` is ``None`` when the gate
            is off or there were no calls.

        Raises:
            ToolCallBlockedError: A call was blocked and ``on_violation`` is ``error``.
        """
        gate = self._tool_gate
        if gate is None:
            return calls, None
        return await gate.enforce(calls, request,
                                  injection_suspected=prepared.injection_action != "none")

    def _answer_check_action(self) -> AnswerCheckAction | None:
        """The configured answer-check action, or ``None`` when the check is off."""
        checker = self._answer_checker
        return checker.action if checker is not None and checker.enabled else None

    def _must_buffer(self, request: ChatCompletionRequest) -> bool:
        """True if this stream's answer may have to be changed or refused.

        ``annotate`` and ``block`` act on the complete answer, so agent
        conversations (those with tool results) are produced in full first.

        Args:
            request: Client request.

        Returns:
            Whether to use the buffered streaming path.
        """
        checker = self._answer_checker
        return (checker is not None and checker.applies_to(request)
                and checker.action is not AnswerCheckAction.FLAG)

    def _check_answer(self, answer: str,
                      request: ChatCompletionRequest) -> AnswerCheckReport | None:
        """Fact-check a final answer against the request's tool results.

        Args:
            answer: Final answer, PII-restored.
            request: Client request (unredacted tool results).

        Returns:
            The report, or ``None`` when the check is off or the conversation
            has no tool results.

        Raises:
            AnswerCheckFailedError: Issues found and the action is ``block``.
        """
        checker = self._answer_checker
        if checker is None or not checker.applies_to(request):
            return None
        report = checker.check(answer, request)
        self._metrics.answer_checks.labels(verdict=report.verdict.value).inc()
        for issue in report.issues:
            self._metrics.answer_check_issues.labels(kind=issue.kind.value).inc()
        if report.failed:
            # Tool names and issue kinds only: never the values (they may be personal data).
            logger.warning("agent answer contradicts tool results", extra={
                "issues": [issue.kind.value for issue in report.issues],
                "tools": sorted({issue.tool for issue in report.issues if issue.tool}),
                "action": checker.action.value})
            if checker.action is AnswerCheckAction.BLOCK:
                raise AnswerCheckFailedError(
                    "the answer does not match the results of the tools it used",
                    details={"answer_check": report.as_dict()})
        return report

    def _restore_calls(self, calls: list[ToolCall], prepared: PreparedRequest) -> list[ToolCall]:
        """Put real PII values back into tool-call arguments.

        Args:
            calls: Tool calls in redacted form.
            prepared: Safety-processed request holding the vault.

        Returns:
            The calls the client should see.
        """
        return self._safety.restore_tool_calls(calls, prepared) if self._safety else calls

    def _restore(self, text: str, prepared: PreparedRequest) -> str:
        """Put real PII values back into an answer.

        Args:
            text: Answer in redacted form.
            prepared: Safety-processed request holding the vault.

        Returns:
            The text the client should see.
        """
        return self._safety.restore(text, prepared) if self._safety else text

    async def _cache_lookup(self, prepared: PreparedRequest, route: str) -> CacheLookup | None:
        """Look the redacted request up in the cache.

        Args:
            prepared: Safety-processed request.
            route: Route alias.

        Returns:
            The lookup, or ``None`` when caching is disabled.
        """
        if self._cache is None:
            return None
        return await self._cache.lookup(prepared.request, route,
                                        injection_flagged=prepared.injection_action != "none")

    async def _cache_store(self, lookup: CacheLookup, completion: Completion, provider: str,
                           report: GroundingReport | None) -> None:
        """Store a provider answer unless it failed (or could not pass) grounding.

        Args:
            lookup: The request's cache lookup.
            completion: Answer in redacted form.
            provider: Provider that produced it.
            report: Grounding report, if grounding ran.
        """
        if self._cache is None:
            return
        if report is not None and report.verdict is not Verdict.PASSED:
            logger.info("answer not cached: grounding verdict %s", report.verdict.value)
            return
        await self._cache.store(lookup, completion, provider)

    async def _check_grounding(self, answer: str, options: GroundingOptions,
                               question: str | None = None, *,
                               precomputed: GroundingReport | None = None) -> GroundingReport:
        """Run grounding and apply the request's policy.

        Args:
            answer: Generated text.
            options: Grounding options.
            question: The user's question, for scorers that read the full prompt.
            precomputed: A report already computed on exactly this answer and
                context (a cascade's check); the policy is still applied.

        Returns:
            The report (when the answer passes, or fails in ``flag`` mode).

        Raises:
            GroundingViolationError: Failed in ``block`` mode.
            GuardrailUnavailableError: Scorer failed and policy is fail-closed.
        """
        report = precomputed or await self._verifier.verify(answer, options.context,
                                                            question=question)
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


def _completion_from_lookup(lookup: CacheLookup) -> Completion:
    """Turn a cache hit back into a provider-style completion.

    Args:
        lookup: A hit (``lookup.entry`` is set).

    Returns:
        The completion, still in redacted form.

    Raises:
        ValueError: If the lookup is not a hit.
    """
    entry = lookup.entry
    if entry is None:
        raise ValueError("lookup is not a cache hit")
    return Completion(text=entry.text, finish_reason=entry.finish_reason, usage=entry.usage,
                      upstream_model=entry.upstream_model)


def _restore_chunk(chunk: StreamChunk, restorer: StreamRestorer) -> None:
    """Restore PII placeholders in a streaming chunk in place.

    Args:
        chunk: Chunk to modify.
        restorer: Stateful restorer for this stream.
    """
    for choice in chunk.choices:
        if choice.delta.content:
            choice.delta.content = restorer.feed(choice.delta.content) or None


def _last_user_message(request: ChatCompletionRequest) -> str | None:
    """Return the most recent user message, used as the question for grounding.

    Args:
        request: Client request.

    Returns:
        The message text, or ``None`` if the conversation has no user turn.
    """
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    return None
