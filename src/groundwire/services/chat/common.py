"""Small pieces shared by the chat pipeline's modules.

* :class:`Answer`: a completion in redacted form, from a provider or the cache.
* :class:`StreamIdentity`: the id, timestamp and model every SSE payload of
  one stream carries, and the two payload shapes (chunk and trailer).
* Request helpers: the grounding options and the last user message.

Nothing here does I/O or holds state across requests.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from groundwire.cache.service import CacheLookup
from groundwire.domain.models import (
    ChatCompletionRequest,
    Completion,
    GroundingOptions,
    StreamChunk,
)
from groundwire.guardrails.grounding import GroundingReport
from groundwire.routing.cascade import CascadeDecision
from groundwire.safety.pii import StreamRestorer

# Who served an answer that came from the response cache.
CACHE_PROVIDER = "cache"

# Word-sized pieces (with their trailing whitespace) for replaying finished answers.
_STREAM_PIECE = re.compile(r"\s*\S+\s*|\s+")


@dataclass(slots=True)
class Answer:
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

    @classmethod
    def from_cache(cls, lookup: CacheLookup) -> Answer:
        """Turn a cache hit back into an answer.

        Args:
            lookup: A hit (``lookup.entry`` is set).

        Returns:
            The answer, still in redacted form.

        Raises:
            ValueError: If the lookup is not a hit.
        """
        return cls(completion=completion_from_lookup(lookup), provider=CACHE_PROVIDER,
                   attempts=[])


@dataclass(frozen=True, slots=True)
class StreamIdentity:
    """What every SSE payload of one stream shares.

    Attributes:
        stream_id: OpenAI-style ``chatcmpl-...`` id.
        created: Unix timestamp of the stream's start.
        model: The ``model`` the client asked for (a route alias).
    """

    stream_id: str
    created: int
    model: str

    @classmethod
    def new(cls, model: str) -> StreamIdentity:
        """Start a new stream identity.

        Args:
            model: The client's ``model`` field.

        Returns:
            A fresh identity.
        """
        return cls(f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), model)

    def stamp(self, chunk: StreamChunk) -> dict[str, Any]:
        """Label a chunk with this stream's identity and serialise it.

        Args:
            chunk: Chunk to label (modified in place).

        Returns:
            The JSON-serialisable payload.
        """
        chunk.id, chunk.created, chunk.model = self.stream_id, self.created, self.model
        return chunk.model_dump(exclude_none=True)

    def trailer(self, meta: dict[str, Any]) -> dict[str, Any]:
        """The final payload carrying ``groundwire`` metadata and no choices.

        Args:
            meta: The metadata.

        Returns:
            The JSON-serialisable payload.
        """
        return {"id": self.stream_id, "object": "chat.completion.chunk",
                "created": self.created, "model": self.model, "choices": [],
                "groundwire": meta}


def stream_pieces(text: str) -> list[str]:
    """Split finished text into word-sized pieces for replay as a stream.

    Args:
        text: The text.

    Returns:
        Pieces that join back to ``text`` exactly.
    """
    return _STREAM_PIECE.findall(text)


def completion_from_lookup(lookup: CacheLookup) -> Completion:
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


def grounding_options(request: ChatCompletionRequest) -> GroundingOptions | None:
    """The grounding options the client sent, if any.

    Args:
        request: Client request.

    Returns:
        The options, or ``None`` when the client did not ask for grounding.
    """
    return request.groundwire.grounding if request.groundwire else None


def last_user_message(request: ChatCompletionRequest) -> str | None:
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


def restore_chunk(chunk: StreamChunk, restorer: StreamRestorer) -> None:
    """Restore PII placeholders in a streaming chunk in place.

    Args:
        chunk: Chunk to modify.
        restorer: Stateful restorer for this stream.
    """
    for choice in chunk.choices:
        if choice.delta.content:
            choice.delta.content = restorer.feed(choice.delta.content) or None
