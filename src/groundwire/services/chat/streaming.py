"""Live streaming: provider chunks in, SSE payloads out, as they arrive.

A :class:`LiveStream` handles exactly one stream. Its phases::

    for each chunk:  absorb ─► restore PII ─► (on finish) release tool calls ─► emit
    after the last:  flush held-back text ─► release tool calls ─► trailer

* **Text streams live.** PII placeholders are restored on the fly; a
  placeholder split across chunks is held back until it is complete.
* **Tool calls do not.** A half-built call is useless to an agent and cannot
  be checked, so call deltas are assembled and released once complete,
  after the tool gate, and *before* the finish marker.
* **Checks run at the end.** Grounding (``flag`` mode only: ``block`` cannot
  stream) and the answer check (``flag`` mode only: other modes are buffered)
  run on the complete text; results go in the trailing ``groundwire`` payload.

Errors after the first byte cannot change the HTTP status any more, so they
are sent in-band as an ``error`` payload.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from groundwire.agents.policy import PolicyResult
from groundwire.cache.service import CacheLookup
from groundwire.core.errors import GroundwireError
from groundwire.domain.models import (
    ChatCompletionRequest,
    Completion,
    FinishReason,
    GroundingOptions,
    StreamChunk,
    ToolCall,
    ToolCallAssembler,
    Usage,
)
from groundwire.guardrails.grounding import GroundingReport
from groundwire.safety.guard import PreparedRequest
from groundwire.services.chat.common import (
    StreamIdentity,
    grounding_options,
    last_user_message,
    restore_chunk,
)
from groundwire.services.chat.pipeline import GROUNDING_SKIPPED_TOOL_CALLS
from groundwire.services.chat.stages import ChatStages

logger = logging.getLogger(__name__)


def stream_metadata(route: str, provider: str, prepared: PreparedRequest,
                    lookup: CacheLookup | None,
                    report: GroundingReport | None) -> dict[str, Any] | None:
    """Build the trailing ``groundwire`` metadata, if there is anything to say.

    Args:
        route: Route alias.
        provider: Serving provider.
        prepared: Safety-processed request.
        lookup: Cache lookup, if caching is on.
        report: Grounding report, if grounding ran.

    Returns:
        The metadata, or ``None`` when no feature produced any (route and
        provider alone are already in the response headers).
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


class LiveStream:
    """Converts one provider stream into SSE payloads."""

    def __init__(self, stages: ChatStages, *, request: ChatCompletionRequest, route: str,
                 provider: str, chunks: AsyncIterator[StreamChunk],
                 prepared: PreparedRequest, lookup: CacheLookup | None,
                 started: float) -> None:
        """Set up the stream's state.

        Args:
            stages: Pipeline stages.
            request: Client request.
            route: Route alias.
            provider: Serving provider.
            chunks: The router's chunk iterator (closed when the stream ends).
            prepared: Safety-processed request (for PII restoration).
            lookup: Cache lookup; a complete answer is stored when writable.
            started: ``perf_counter`` at request start.
        """
        self._stages = stages
        self._request = request
        self._route = route
        self._provider = provider
        self._chunks = chunks
        self._prepared = prepared
        self._lookup = lookup
        self._started = started
        self._identity = StreamIdentity.new(request.model)
        self._grounding: GroundingOptions | None = grounding_options(request)
        self._restorer = stages.safety.stream_restorer(prepared)
        # Answer check in flag mode runs on the streamed text once it is complete.
        self._check_answer = stages.answer_check.applies_to(request)
        self._collect_text = self._grounding is not None or self._check_answer
        # Accumulated as chunks arrive.
        self._parts: list[str] = []      # restored text, for the checks
        self._raw_parts: list[str] = []  # redacted text, for the cache
        self._finish_reason: FinishReason | None = None
        self._usage = Usage()
        self._upstream_model = provider
        self._assembler = ToolCallAssembler()
        self._calls: list[ToolCall] = []
        self._policy: PolicyResult | None = None
        self._released = False
        self._sent_text = False

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """The stream's SSE payloads (without the final ``[DONE]``).

        Yields:
            JSON-serialisable payloads.
        """
        status = "error"
        try:
            async for chunk in self._chunks:
                for payload in await self._on_chunk(chunk):
                    yield payload
            tail = self._flush_tail()
            if tail is not None:
                yield tail
            ready = await self._release_calls()  # stream ended without a finish marker
            if ready is not None:
                yield ready
            trailer = await self._finalise()
            if trailer is not None:
                yield trailer
            status = "success"
        except GroundwireError as exc:
            status = exc.code
            logger.error("stream aborted", extra={"route": self._route,
                                                  "provider": self._provider,
                                                  "error_code": exc.code})
            yield exc.to_payload()
        except (GeneratorExit, asyncio.CancelledError):
            status = "client_disconnected"
            logger.info("client disconnected mid-stream",
                        extra={"route": self._route, "provider": self._provider})
            raise
        finally:
            self._stages.recorder.record(self._route, self._provider, status, stream=True,
                                         started=self._started)
            aclose = getattr(self._chunks, "aclose", None)
            if aclose is not None:
                await aclose()

    # ------------------------------------------------------------------ #
    # Per chunk
    # ------------------------------------------------------------------ #

    async def _on_chunk(self, chunk: StreamChunk) -> list[dict[str, Any]]:
        """Process one provider chunk.

        Args:
            chunk: The chunk (modified in place).

        Returns:
            Payloads to send now, in order: released tool calls (on the finish
            chunk), then the chunk itself if it carries anything.
        """
        self._absorb(chunk)
        self._restore(chunk)
        if self._collect_text:
            self._parts.append(chunk.content())
        finishing = any(c.finish_reason is not None for c in chunk.choices)
        out = await self._on_finish(chunk) if finishing else []
        if chunk.content() or chunk.usage or finishing:
            self._sent_text = self._sent_text or bool(chunk.content())
            out.append(chunk.model_dump(exclude_none=True))
        return out

    async def _on_finish(self, chunk: StreamChunk) -> list[dict[str, Any]]:
        """Release complete tool calls before the finish marker goes out.

        Args:
            chunk: The chunk carrying a finish reason (modified in place).

        Returns:
            The released-calls payload, if any.
        """
        ready = await self._release_calls()
        if self._policy is not None and not self._calls:
            for choice in chunk.choices:  # no calls left: the turn ends in text
                if choice.finish_reason == "tool_calls":
                    choice.finish_reason = "stop"
        return [ready] if ready is not None else []

    def _absorb(self, chunk: StreamChunk) -> None:
        """Record what the chunk says, stamp it, and take its tool-call deltas out."""
        if chunk.model:
            self._upstream_model = chunk.model
        self._identity.stamp(chunk)
        deltas = chunk.tool_call_deltas()
        if deltas:
            self._assembler.feed(deltas)
            for choice in chunk.choices:
                choice.delta.tool_calls = None
        self._raw_parts.append(chunk.content())
        for choice in chunk.choices:
            self._finish_reason = choice.finish_reason or self._finish_reason
        self._usage = chunk.usage or self._usage

    def _restore(self, chunk: StreamChunk) -> None:
        """Restore PII in the chunk; release held-back text before a finish marker."""
        if self._restorer is None:
            return
        restore_chunk(chunk, self._restorer)
        if chunk.choices and chunk.choices[0].finish_reason is not None:
            tail = self._restorer.flush()
            if tail:
                delta = chunk.choices[0].delta
                delta.content = (delta.content or "") + tail

    async def _release_calls(self) -> dict[str, Any] | None:
        """Send the assembled tool calls through the gate, once.

        Returns:
            A payload with the released calls, or with the gate's explanation
            when no call can run now; ``None`` if there were no calls or they
            were already released.

        Raises:
            ToolCallBlockedError: A call was blocked and ``on_violation`` is ``error``.
        """
        if not self._assembler or self._released:
            return None
        self._released = True
        proposed = self._stages.safety.restore_calls(self._assembler.calls(), self._prepared)
        self._calls, self._policy = await self._stages.tools.enforce(
            proposed, self._request, self._prepared)
        first = not self._sent_text
        if self._calls:
            return self._identity.stamp(StreamChunk.tool_calls(self._calls, role=first))
        if self._policy is not None:  # no call can run now: explain in text
            prefix = "" if first else "\n\n"
            return self._identity.stamp(
                StreamChunk.text(prefix + self._policy.explanation(), role=first))
        return None

    # ------------------------------------------------------------------ #
    # After the last chunk
    # ------------------------------------------------------------------ #

    def _flush_tail(self) -> dict[str, Any] | None:
        """Text the PII restorer was still holding back when the stream ended."""
        if self._restorer is None:
            return None
        tail = self._restorer.flush()
        if not tail:
            return None
        self._parts.append(tail)
        return self._identity.stamp(StreamChunk.text(tail))

    async def _finalise(self) -> dict[str, Any] | None:
        """Run the end-of-stream checks, cache the answer, and build the trailer.

        Returns:
            The trailing ``groundwire`` payload, or ``None`` if there is nothing to say.

        Raises:
            GuardrailUnavailableError: Grounding could not run (fail-closed).
        """
        text = "".join(self._parts)
        tool_only = bool(self._assembler) and not text.strip()
        report: GroundingReport | None = None
        if self._grounding is not None and not tool_only:
            report = await self._stages.grounding.check(text, self._grounding,
                                                        last_user_message(self._request))
        completion = Completion(text="".join(self._raw_parts), finish_reason=self._finish_reason,
                                usage=self._usage, upstream_model=self._upstream_model)
        await self._stages.cache.store(self._lookup, completion, self._provider, report)
        extra = self._trailer_metadata(text, tool_only, report)
        return self._identity.trailer(extra) if extra is not None else None

    def _trailer_metadata(self, text: str, tool_only: bool,
                          report: GroundingReport | None) -> dict[str, Any] | None:
        """Metadata for the trailer: features' results, plus the flag-mode answer check.

        Args:
            text: The complete restored text.
            tool_only: The answer was only tool calls.
            report: Grounding report, if grounding ran.

        Returns:
            The metadata, or ``None`` when there is nothing to say.

        Raises:
            AnswerCheckFailedError: Never in practice (``block`` mode is buffered).
        """
        extra = stream_metadata(self._route, self._provider, self._prepared, self._lookup,
                                report)
        if self._grounding is not None and tool_only:
            extra = self._with(extra, "grounding", dict(GROUNDING_SKIPPED_TOOL_CALLS))
        if self._policy is not None:
            extra = self._with(extra, "tools", self._policy.as_dict())
        if self._check_answer and not self._assembler:
            check = self._stages.answer_check.check(text, self._request)
            if check is not None:
                extra = self._with(extra, "answer_check", check.as_dict())
        return extra

    def _with(self, extra: dict[str, Any] | None, key: str, value: Any) -> dict[str, Any]:
        """Add one entry to the trailer metadata, creating it if needed."""
        extra = extra or {"route": self._route, "provider": self._provider}
        extra[key] = value
        return extra
