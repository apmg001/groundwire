"""Domain models for chat completions.

The gateway speaks the OpenAI Chat Completions wire format at its edge, so any
OpenAI SDK or tool can use it by changing ``base_url``. Internally, every
provider adapter converts to and from these models, which keeps provider
quirks (Anthropic's separate ``system`` field, different stop reasons...)
inside the adapters.

Groundwire-specific options travel in an optional ``groundwire`` field on the
request. OpenAI SDKs send it via ``extra_body={"groundwire": {...}}``.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One chat message.

    OpenAI allows ``content`` to be a list of typed parts. The gateway accepts
    text parts and joins them; non-text parts are rejected explicitly rather
    than silently dropped.

    Attributes:
        role: Author role.
        content: Message text.
    """

    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def _normalise_content(cls, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, list):
            texts: list[str] = []
            for part in value:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError("only text content parts are supported")
                texts.append(str(part.get("text", "")))
            return "\n".join(texts)
        return value


# --------------------------------------------------------------------------- #
# Groundwire extension options
# --------------------------------------------------------------------------- #


class GroundingMode(StrEnum):
    """How a failed grounding check is handled."""

    FLAG = "flag"  # return the answer with a grounding report attached
    BLOCK = "block"  # refuse with HTTP 422 and the report


class GroundingOptions(BaseModel):
    """Per-request grounding options.

    Attributes:
        context: Source passages the answer must be supported by (e.g. the
            RAG chunks the caller put in the prompt).
        mode: Handling of a failed check.
    """

    model_config = ConfigDict(extra="forbid")

    context: Annotated[
        list[Annotated[str, Field(min_length=1)]], Field(min_length=1, max_length=200)
    ]
    mode: GroundingMode = GroundingMode.FLAG


class GroundwireOptions(BaseModel):
    """Container for gateway-specific request options.

    Attributes:
        grounding: Enable grounding verification for this request.
    """

    model_config = ConfigDict(extra="forbid")

    grounding: GroundingOptions | None = None


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Unknown OpenAI parameters (``n``, ``logprobs``...) are ignored so standard
    SDKs keep working; parameters the gateway does honour are validated.

    Attributes:
        model: Route alias (see ``routes`` in the gateway config).
        messages: Conversation, oldest first.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        max_tokens: Completion token cap.
        stop: Stop sequence(s).
        stream: Stream the answer as server-sent events.
        user: Opaque end-user identifier, forwarded for abuse tracking.
        groundwire: Gateway extension options.
    """

    model_config = ConfigDict(extra="ignore")

    model: Annotated[str, Field(min_length=1, max_length=128)] = "default"
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=1000)]
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1)] | None = None
    max_tokens: Annotated[int, Field(ge=1, le=200_000)] | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    user: Annotated[str, Field(max_length=256)] | None = None
    groundwire: GroundwireOptions | None = None

    @model_validator(mode="after")
    def _check_last_message(self) -> ChatCompletionRequest:
        if self.messages[-1].role == "assistant":
            # Assistant prefill is provider-specific; reject explicitly.
            raise ValueError("the last message must be from 'user' or 'system'")
        return self

    def stop_list(self) -> list[str] | None:
        """Return ``stop`` normalised to a list (or ``None``)."""
        if self.stop is None:
            return None
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #

FinishReason = Literal["stop", "length", "content_filter", "tool_calls"]


class Usage(BaseModel):
    """Token accounting.

    Attributes:
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens.
        total_tokens: Sum of both.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def of(cls, prompt: int, completion: int) -> Usage:
        """Build a usage record and compute the total.

        Args:
            prompt: Input tokens.
            completion: Output tokens.

        Returns:
            The usage record.
        """
        return cls(prompt_tokens=prompt, completion_tokens=completion,
                   total_tokens=prompt + completion)


class AssistantMessage(BaseModel):
    """Assistant reply inside a choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """A single completion choice (the gateway always returns one)."""

    index: int = 0
    message: AssistantMessage
    finish_reason: FinishReason | None = "stop"


class Completion(BaseModel):
    """Provider-agnostic result of a non-streaming call.

    This is what adapters return; the service turns it into the public
    :class:`ChatCompletionResponse`.

    Attributes:
        text: Generated text.
        finish_reason: Normalised finish reason.
        usage: Token usage reported by the provider.
        upstream_model: Model the provider says it used.
    """

    text: str
    finish_reason: FinishReason | None = "stop"
    usage: Usage = Usage()
    upstream_model: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response, plus gateway metadata.

    Attributes:
        id: Completion ID.
        object: Always ``"chat.completion"``.
        created: Unix timestamp.
        model: Route alias the client asked for.
        choices: Exactly one choice.
        usage: Token usage.
        groundwire: Gateway metadata (provider used, attempts, grounding).
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    groundwire: dict[str, Any] = Field(default_factory=dict)


class StreamDelta(BaseModel):
    """Incremental content in a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    """Choice inside a streaming chunk."""

    index: int = 0
    delta: StreamDelta
    finish_reason: FinishReason | None = None


class StreamChunk(BaseModel):
    """Provider-agnostic streaming chunk (OpenAI ``chat.completion.chunk``).

    Adapters yield these; the API layer serialises them as SSE ``data:`` lines.

    Attributes:
        id: Stream ID, shared by all chunks of one response.
        object: Always ``"chat.completion.chunk"``.
        created: Unix timestamp.
        model: Route alias (rewritten by the service).
        choices: Exactly one choice.
        usage: Present on the final chunk when the provider reports it.
    """

    id: str = ""
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice]
    usage: Usage | None = None

    @classmethod
    def text(cls, content: str, *, role: bool = False) -> StreamChunk:
        """Build a content chunk.

        Args:
            content: Text delta.
            role: Include ``role: assistant`` (first chunk convention).

        Returns:
            The chunk.
        """
        return cls(choices=[StreamChoice(delta=StreamDelta(
            role="assistant" if role else None, content=content))])

    @classmethod
    def finish(cls, reason: FinishReason | None, usage: Usage | None = None) -> StreamChunk:
        """Build the terminal chunk.

        Args:
            reason: Finish reason.
            usage: Final usage, if known.

        Returns:
            The chunk.
        """
        return cls(choices=[StreamChoice(delta=StreamDelta(), finish_reason=reason)], usage=usage)

    def content(self) -> str:
        """Return the text carried by this chunk (``""`` if none)."""
        return "".join(c.delta.content or "" for c in self.choices)
