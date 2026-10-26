"""Stream an answer that is already complete.

Two cases produce the whole answer before the first byte is sent:

* **cache hits**: the answer is stored; it is replayed word by word so a
  streaming client sees a normal SSE stream;
* **buffered streams**: cascades, and agent answers the answer check may
  change or refuse, must be complete before they can be checked. They go
  through :class:`~groundwire.services.chat.pipeline.CompletionPipeline`
  and the finished response is replayed here.

Either way, the payloads look exactly like a live stream's.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from groundwire.cache.service import CacheLookup
from groundwire.core.errors import GroundwireError
from groundwire.domain.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    GroundingOptions,
    StreamChunk,
)
from groundwire.guardrails.grounding import GroundingReport
from groundwire.safety.guard import PreparedRequest
from groundwire.services.chat.common import (
    CACHE_PROVIDER,
    StreamIdentity,
    completion_from_lookup,
    last_user_message,
    stream_pieces,
)
from groundwire.services.chat.stages import ChatStages
from groundwire.services.chat.streaming import stream_metadata

logger = logging.getLogger(__name__)


class Replayer:
    """Turns finished answers into SSE payloads."""

    def __init__(self, stages: ChatStages) -> None:
        """Create the replayer.

        Args:
            stages: Pipeline stages (PII restoration, grounding, metrics).
        """
        self._stages = stages

    async def cached(self, request: ChatCompletionRequest, route: str, lookup: CacheLookup,
                     grounding: GroundingOptions | None, started: float,
                     prepared: PreparedRequest) -> AsyncIterator[dict[str, Any]]:
        """Replay a cache hit as an OpenAI-style SSE stream.

        Args:
            request: Client request.
            route: Route alias.
            lookup: The cache hit.
            grounding: Grounding options (``flag`` mode only).
            started: ``perf_counter`` at request start.
            prepared: Safety-processed request (for PII restoration).

        Yields:
            Content chunks, the finish chunk, then (if any) the metadata trailer.
        """
        identity = StreamIdentity.new(request.model)
        status = "error"
        try:
            completion = completion_from_lookup(lookup)
            text = self._stages.safety.restore(completion.text, prepared)
            for index, piece in enumerate(stream_pieces(text)):
                yield identity.stamp(StreamChunk.text(piece, role=index == 0))
            yield identity.stamp(StreamChunk.finish(completion.finish_reason, completion.usage))

            report: GroundingReport | None = None
            if grounding is not None:
                report = await self._stages.grounding.check(text, grounding,
                                                            last_user_message(request))
            extra = stream_metadata(route, CACHE_PROVIDER, prepared, lookup, report)
            if extra is not None:
                yield identity.trailer(extra)
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
            self._stages.recorder.record(route, CACHE_PROVIDER, status, stream=True,
                                         started=started)

    async def response(self, request: ChatCompletionRequest, route: str,
                       response: ChatCompletionResponse,
                       started: float) -> AsyncIterator[dict[str, Any]]:
        """Replay a finished response as an OpenAI-style SSE stream.

        Args:
            request: Client request.
            route: Route alias.
            response: The finished response (text already PII-restored).
            started: ``perf_counter`` at request start.

        Yields:
            Content chunks, released tool calls, the finish chunk, then the
            ``groundwire`` metadata trailer.
        """
        identity = StreamIdentity.new(request.model)
        provider = str(response.groundwire.get("provider", "none"))
        choice = response.choices[0]
        status = "error"
        try:
            pieces = stream_pieces(choice.message.content or "")
            for index, piece in enumerate(pieces):
                yield identity.stamp(StreamChunk.text(piece, role=index == 0))
            if choice.message.tool_calls:
                yield identity.stamp(StreamChunk.tool_calls(choice.message.tool_calls,
                                                            role=not pieces))
            yield identity.stamp(StreamChunk.finish(choice.finish_reason, response.usage))
            yield identity.trailer(response.groundwire)
            status = "success"
        except (GeneratorExit, asyncio.CancelledError):
            status = "client_disconnected"
            raise
        finally:
            self._stages.recorder.record(route, provider, status, stream=True,
                                         started=started)
