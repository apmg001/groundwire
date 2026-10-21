"""Domain models for chat completions.

The gateway speaks the OpenAI Chat Completions wire format at its edge, so any
OpenAI SDK or tool can use it by changing ``base_url``. Internally, every
provider adapter converts to and from these models, which keeps provider
quirks (Anthropic's separate ``system`` field, different stop reasons...)
inside the adapters.

Groundwire-specific options travel in an optional ``groundwire`` field on the
request. OpenAI SDKs send it via ``extra_body={"groundwire": {...}}``.

Tool calling follows OpenAI's format too: ``tools`` on the request, assistant
``tool_calls`` (``arguments`` is a JSON *string*), and ``tool`` messages that
answer a call by ``tool_call_id``. The Anthropic adapter translates to and from
``tool_use`` / ``tool_result`` content blocks.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

# OpenAI's rule for function names; also accepted by Anthropic.
TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class FunctionDefinition(BaseModel):
    """A function the model may call.

    Attributes:
        name: Function name.
        description: What it does (the model reads this to decide when to call it).
        parameters: JSON Schema of the arguments object.
        strict: OpenAI structured-outputs flag, passed through.
    """

    model_config = ConfigDict(extra="ignore")

    name: Annotated[str, Field(pattern=TOOL_NAME_PATTERN)]
    description: Annotated[str, Field(max_length=8192)] | None = None
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class ToolDefinition(BaseModel):
    """A tool offered to the model (only ``function`` tools exist today).

    Attributes:
        type: Always ``"function"``.
        function: The function.
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """A function call requested by the model.

    Attributes:
        name: Function name.
        arguments: Arguments as a JSON **string**, exactly as the model wrote it
            (it may be invalid JSON: models make mistakes, and callers decide).
    """

    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    """One tool call in an assistant message.

    Attributes:
        id: Call id; the ``tool`` message answering it repeats this id.
        type: Always ``"function"``.
        function: The call.
    """

    id: Annotated[str, Field(min_length=1, max_length=256)]
    type: Literal["function"] = "function"
    function: FunctionCall

    def parsed_arguments(self) -> dict[str, Any] | None:
        """Decode ``arguments``.

        Returns:
            The arguments object, or ``None`` if it is not a JSON object.
        """
        try:
            value = json.loads(self.function.arguments or "{}")
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


class NamedToolChoice(BaseModel):
    """Force a specific function: ``{"type": "function", "function": {"name": ...}}``."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    function: dict[str, str]

    @property
    def name(self) -> str:
        """The forced function's name."""
        return self.function.get("name", "")


ToolChoice = Literal["auto", "none", "required"] | NamedToolChoice

# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """One chat message.

    OpenAI allows ``content`` to be a list of typed parts. The gateway accepts
    text parts and joins them; non-text parts are rejected explicitly rather
    than silently dropped. ``content`` is always a string internally (``None``
    becomes ``""``), so every safety check sees text.

    Attributes:
        role: Author role.
        content: Message text (for ``tool`` messages: the tool's result).
        tool_calls: Calls requested by an ``assistant`` message.
        tool_call_id: For ``tool`` messages: the call this result answers.
        name: Optional author name, passed through.
    """

    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _check_tool_fields(self) -> ChatMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a 'tool' message needs the 'tool_call_id' it answers")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only 'assistant' messages can contain tool_calls")
        return self

    def to_openai(self) -> dict[str, Any]:
        """Serialise in OpenAI's wire format (omitting unset fields).

        Returns:
            JSON-serialisable message.
        """
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [c.model_dump() for c in self.tool_calls]
            if not self.content:
                data["content"] = None  # OpenAI's convention for tool-call-only turns
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data

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


class CacheMode(StrEnum):
    """Per-request cache behaviour (like HTTP ``Cache-Control``)."""

    USE = "use"  # read from and write to the cache (default)
    BYPASS = "bypass"  # neither read nor write: always ask the provider
    REFRESH = "refresh"  # skip the read, ask the provider, overwrite the entry


class CacheOptions(BaseModel):
    """Per-request cache options.

    Attributes:
        mode: How this request uses the cache.
    """

    model_config = ConfigDict(extra="forbid")

    mode: CacheMode = CacheMode.USE


class GroundwireOptions(BaseModel):
    """Container for gateway-specific request options.

    Attributes:
        grounding: Enable grounding verification for this request.
        cache: Override cache behaviour for this request.
    """

    model_config = ConfigDict(extra="forbid")

    grounding: GroundingOptions | None = None
    cache: CacheOptions | None = None


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
        tools: Tools the model may call.
        tool_choice: ``auto``, ``none``, ``required``, or a named function.
        parallel_tool_calls: Allow several tool calls in one turn.
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
    tools: Annotated[list[ToolDefinition], Field(max_length=128)] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None
    groundwire: GroundwireOptions | None = None

    @model_validator(mode="after")
    def _check_last_message(self) -> ChatCompletionRequest:
        if self.messages[-1].role == "assistant":
            # Assistant prefill is provider-specific; reject explicitly.
            raise ValueError("the last message must be from 'user', 'system' or 'tool'")
        return self

    @model_validator(mode="after")
    def _check_tools(self) -> ChatCompletionRequest:
        names = [t.function.name for t in self.tools or []]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate tool names: {duplicates}")
        if isinstance(self.tool_choice, NamedToolChoice) and self.tool_choice.name not in names:
            raise ValueError(f"tool_choice names '{self.tool_choice.name}', which is not in tools")
        if self.tool_choice == "required" and not names:
            raise ValueError("tool_choice 'required' needs at least one tool")
        return self

    @property
    def uses_tools(self) -> bool:
        """True for agent traffic: tools offered, or tool calls/results in the history."""
        return bool(self.tools) or any(m.role == "tool" or m.tool_calls for m in self.messages)

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
    """Assistant reply inside a choice.

    Attributes:
        role: Always ``"assistant"``.
        content: Text, or ``None`` for a tool-call-only reply (OpenAI convention).
        tool_calls: Tool calls requested by the model.
    """

    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCall] | None = None

    @model_serializer(mode="wrap")
    def _keep_content(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # The API serialises with exclude_none, but OpenAI clients expect the
        # "content" key even when it is null (tool-call-only replies).
        data: dict[str, Any] = handler(self)
        data.setdefault("content", self.content)
        return data


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
        tool_calls: Tool calls requested by the model.
    """

    text: str
    finish_reason: FinishReason | None = "stop"
    usage: Usage = Usage()
    upstream_model: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


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


class FunctionCallDelta(BaseModel):
    """Partial function call in a stream (fields arrive in pieces)."""

    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """Partial tool call in a stream.

    OpenAI streams a call as several deltas sharing an ``index``: the first
    carries ``id`` and ``function.name``, later ones append ``arguments``.

    Attributes:
        index: Position of the call in this turn (groups the pieces).
        id: Call id (first piece only).
        type: ``"function"`` (first piece only).
        function: Name and/or an arguments fragment.
    """

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta = Field(default_factory=FunctionCallDelta)


class StreamDelta(BaseModel):
    """Incremental content in a streaming chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


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

    def tool_call_deltas(self) -> list[ToolCallDelta]:
        """Return the tool-call fragments carried by this chunk."""
        return [d for c in self.choices for d in (c.delta.tool_calls or [])]

    @classmethod
    def tool_calls(cls, calls: list[ToolCall], *, role: bool = False) -> StreamChunk:
        """Build one chunk carrying complete tool calls.

        Args:
            calls: Complete tool calls.
            role: Include ``role: assistant``.

        Returns:
            The chunk.
        """
        deltas = [ToolCallDelta(index=i, id=c.id, type="function",
                                function=FunctionCallDelta(name=c.function.name,
                                                           arguments=c.function.arguments))
                  for i, c in enumerate(calls)]
        return cls(choices=[StreamChoice(delta=StreamDelta(
            role="assistant" if role else None, tool_calls=deltas))])


# --------------------------------------------------------------------------- #
# Stream assembly
# --------------------------------------------------------------------------- #


class ToolCallAssembler:
    """Rebuilds complete tool calls from streamed fragments.

    Fragments with the same ``index`` belong to one call: the first usually
    carries ``id`` and ``name``, later ones append pieces of ``arguments``.
    The gateway assembles calls before forwarding them, because a partial
    call is useless to an agent and must be complete before it can be
    checked (PII restoration, tool policies).
    """

    def __init__(self) -> None:
        """Create an empty assembler."""
        self._ids: dict[int, str] = {}
        self._names: dict[int, str] = {}
        self._args: dict[int, list[str]] = {}

    def __bool__(self) -> bool:
        """True once any fragment has been seen."""
        return bool(self._names or self._args or self._ids)

    def feed(self, deltas: list[ToolCallDelta]) -> None:
        """Add fragments.

        Args:
            deltas: Fragments from one stream chunk.
        """
        for delta in deltas:
            if delta.id:
                self._ids[delta.index] = delta.id
            if delta.function.name:
                self._names[delta.index] = self._names.get(delta.index, "") + delta.function.name
            if delta.function.arguments:
                self._args.setdefault(delta.index, []).append(delta.function.arguments)

    def calls(self) -> list[ToolCall]:
        """Return the assembled calls, ordered by index.

        A call whose id never arrived gets a generated one, so the client can
        still answer it with a ``tool`` message.

        Returns:
            Complete tool calls.
        """
        indices = sorted({*self._ids, *self._names, *self._args})
        return [ToolCall(id=self._ids.get(i) or f"call_{uuid.uuid4().hex[:24]}",
                         function=FunctionCall(name=self._names.get(i, ""),
                                               arguments="".join(self._args.get(i, [])) or "{}"))
                for i in indices]
