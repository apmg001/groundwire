"""Adapter for the Anthropic Messages API.

Differences from the OpenAI format that this adapter hides:

* The system prompt is a top-level ``system`` field, not a message.
* ``max_tokens`` is required.
* Consecutive messages with the same role must be merged.
* Streaming uses named SSE events (``message_start``, ``content_block_delta``,
  ``message_delta``, ``message_stop``, ``ping``, ``error``).
* Stop reasons differ (``end_turn``, ``max_tokens``, ``stop_sequence``...).
* HTTP 529 means "overloaded" (treated as a retryable server error).
* Tools: ``{name, description, input_schema}``; ``tool_choice`` types are
  ``auto``/``any``/``tool``/``none``; a model's call is a ``tool_use`` content
  block whose ``input`` is a JSON *object* (OpenAI: a JSON *string*); a result
  goes back as a ``tool_result`` block inside a **user** message. Streaming
  sends ``content_block_start`` (id, name) then ``input_json_delta`` pieces.

``base_url`` is the API root, e.g. ``https://api.anthropic.com``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import SecretStr

from groundwire.core.errors import ProviderResponseError, ProviderServerError
from groundwire.domain.models import (
    ChatCompletionRequest,
    ChatMessage,
    Completion,
    FinishReason,
    FunctionCall,
    FunctionCallDelta,
    NamedToolChoice,
    StreamChoice,
    StreamChunk,
    StreamDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from groundwire.providers.base import Provider
from groundwire.providers.http import (
    build_client,
    decode_json,
    iter_sse,
    parse_event_json,
    raise_for_status,
    translate_transport_errors,
)

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _finish_reason(value: Any) -> FinishReason | None:
    """Map an Anthropic ``stop_reason`` to an OpenAI finish reason."""
    if value is None:
        return None
    return _STOP_REASON_MAP.get(str(value), "stop")


# --------------------------------------------------------------------------- #
# Format translation (pure functions, unit-tested directly)
# --------------------------------------------------------------------------- #

Block = dict[str, Any]


def _arguments_object(call: ToolCall) -> dict[str, Any]:
    """Anthropic needs ``input`` as an object; fall back to ``{}`` for bad JSON.

    Args:
        call: OpenAI-style call (arguments as a JSON string).

    Returns:
        The arguments object.
    """
    parsed = call.parsed_arguments()
    if parsed is None:
        logger.warning("tool call arguments are not a JSON object; sending {}",
                       extra={"tool": call.function.name})
        return {}
    return parsed


def message_blocks(message: ChatMessage) -> tuple[str, list[Block]]:
    """Translate one non-system message into an Anthropic role and content blocks.

    Args:
        message: Domain message (``user``, ``assistant`` or ``tool``).

    Returns:
        ``(role, blocks)``; ``tool`` results become ``user`` turns.
    """
    if message.role == "tool":
        return "user", [{"type": "tool_result", "tool_use_id": message.tool_call_id,
                         "content": message.content}]
    blocks: list[Block] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls or []:
        blocks.append({"type": "tool_use", "id": call.id, "name": call.function.name,
                       "input": _arguments_object(call)})
    return message.role, blocks


def merge_turns(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Build Anthropic's alternating user/assistant turns.

    Consecutive same-role messages are merged (Anthropic requires alternation),
    which also groups several ``tool_result`` blocks into one user turn, as the
    API expects after a parallel tool call.

    Args:
        messages: Non-system messages, oldest first.

    Returns:
        Anthropic ``messages``.
    """
    turns: list[dict[str, Any]] = []
    for message in messages:
        role, blocks = message_blocks(message)
        if not blocks:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    # Plain-text turns stay strings: smaller payloads, identical meaning.
    for turn in turns:
        if all(b["type"] == "text" for b in turn["content"]):
            turn["content"] = "\n\n".join(b["text"] for b in turn["content"])
    return turns


def tool_choice_payload(request: ChatCompletionRequest) -> dict[str, Any] | None:
    """Translate ``tool_choice`` / ``parallel_tool_calls``.

    Args:
        request: Client request.

    Returns:
        Anthropic ``tool_choice``, or ``None`` to use the default (auto).
    """
    choice = request.tool_choice
    if isinstance(choice, NamedToolChoice):
        payload: dict[str, Any] = {"type": "tool", "name": choice.name}
    elif choice == "required":
        payload = {"type": "any"}
    elif choice == "none":
        return {"type": "none"}
    elif choice == "auto" or request.parallel_tool_calls is False:
        payload = {"type": "auto"}
    else:
        return None
    if request.parallel_tool_calls is False:
        payload["disable_parallel_tool_use"] = True
    return payload


def parse_tool_use(blocks: list[Any]) -> list[ToolCall]:
    """Extract ``tool_use`` blocks as OpenAI-style calls.

    Args:
        blocks: Response content blocks.

    Returns:
        Tool calls, with ``input`` serialised to a JSON string.
    """
    return [ToolCall(id=str(b.get("id")), function=FunctionCall(
                name=str(b.get("name", "")),
                arguments=json.dumps(b.get("input") or {}, ensure_ascii=False)))
            for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]


class AnthropicProvider(Provider):
    """Provider for Anthropic's ``/v1/messages`` endpoint."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: SecretStr | None,
        timeout_s: float,
        max_connections: int,
        default_max_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            name: Provider name.
            model: Anthropic model id.
            base_url: API root.
            api_key: API key (required by the real API; optional for proxies).
            timeout_s: Per-call timeout.
            max_connections: Connection-pool size.
            default_max_tokens: Used when the client omits ``max_tokens``.
            transport: Optional custom transport (tests).
        """
        super().__init__(name, model)
        self._default_max_tokens = default_max_tokens
        headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if api_key is not None:
            headers["x-api-key"] = api_key.get_secret_value()
        self._client = build_client(
            base_url=base_url, timeout_s=timeout_s, max_connections=max_connections,
            headers=headers, transport=transport,
        )

    # ------------------------------------------------------------------ #
    # Request building
    # ------------------------------------------------------------------ #

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        """Translate a domain request into an Anthropic Messages body.

        Args:
            request: Client request.
            stream: Whether to stream.

        Returns:
            JSON-serialisable body.
        """
        system_parts = [m.content for m in request.messages if m.role == "system"]
        merged = merge_turns([m for m in request.messages if m.role != "system"])
        if not merged:
            # Only system messages were sent; Anthropic needs a user turn.
            merged.append({"role": "user", "content": "Continue."})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": merged,
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.tools:
            payload["tools"] = [{"name": t.function.name,
                                 "description": t.function.description or "",
                                 "input_schema": t.function.parameters}
                                for t in request.tools]
            choice = tool_choice_payload(request)
            if choice is not None:
                payload["tool_choice"] = choice
        optional = {
            "temperature": None if request.temperature is None else min(request.temperature, 1.0),
            "top_p": request.top_p,
            "stop_sequences": request.stop_list(),
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        return payload

    # ------------------------------------------------------------------ #
    # Provider API
    # ------------------------------------------------------------------ #

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        """Run a non-streaming completion. See :meth:`Provider.complete`."""
        with translate_transport_errors(self.name):
            response = await self._client.post(
                "/v1/messages", json=self._payload(request, stream=False))
        await raise_for_status(self.name, response)
        data = decode_json(self.name, response)

        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderResponseError(f"{self.name}: response missing content",
                                        provider=self.name)
        text = "".join(
            str(b.get("text", "")) for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        raw_usage = data.get("usage")
        usage_raw: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return Completion(
            text=text,
            finish_reason=_finish_reason(data.get("stop_reason")) or "stop",
            usage=Usage.of(int(usage_raw.get("input_tokens") or 0),
                           int(usage_raw.get("output_tokens") or 0)),
            upstream_model=str(data.get("model") or self.model),
            tool_calls=parse_tool_use(blocks),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion. See :meth:`Provider.stream`."""
        payload = self._payload(request, stream=True)
        input_tokens = 0
        output_tokens = 0
        finish: FinishReason | None = None
        first = True
        # Anthropic block index -> OpenAI tool-call index (0, 1, 2... per turn).
        tool_index: dict[int, int] = {}

        with translate_transport_errors(self.name):
            async with self._client.stream("POST", "/v1/messages", json=payload) as response:
                await raise_for_status(self.name, response)
                async for event in iter_sse(response.aiter_lines()):
                    data = parse_event_json(self.name, event)
                    kind = data.get("type", event.event)

                    if kind == "message_start":
                        usage = (data.get("message") or {}).get("usage") or {}
                        input_tokens = int(usage.get("input_tokens") or 0)
                    elif kind == "content_block_start":
                        block = data.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            index = tool_index.setdefault(int(data.get("index", 0)),
                                                          len(tool_index))
                            yield _tool_chunk(ToolCallDelta(
                                index=index, id=str(block.get("id")), type="function",
                                function=FunctionCallDelta(name=str(block.get("name", "")),
                                                           arguments="")), role=first)
                            first = False
                    elif kind == "content_block_delta":
                        delta = data.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield StreamChunk.text(str(delta["text"]), role=first)
                            first = False
                        elif delta.get("type") == "input_json_delta":
                            block_index = int(data.get("index", 0))
                            if block_index in tool_index and delta.get("partial_json"):
                                yield _tool_chunk(ToolCallDelta(
                                    index=tool_index[block_index],
                                    function=FunctionCallDelta(
                                        arguments=str(delta["partial_json"]))), role=False)
                    elif kind == "message_delta":
                        finish = _finish_reason((data.get("delta") or {}).get("stop_reason"))
                        output_tokens = int((data.get("usage") or {}).get("output_tokens") or 0)
                    elif kind == "message_stop":
                        break
                    elif kind == "error":
                        err = data.get("error") or {}
                        # Mid-stream errors are usually overload: transient.
                        raise ProviderServerError(
                            f"{self.name}: stream error {err.get('type')}: {err.get('message')}",
                            provider=self.name,
                        )
                    # ping, content_block_stop and unknown events are ignored.

        yield StreamChunk.finish(finish or "stop", Usage.of(input_tokens, output_tokens))

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()


def _tool_chunk(delta: ToolCallDelta, *, role: bool) -> StreamChunk:
    """Wrap one tool-call fragment in a stream chunk.

    Args:
        delta: Fragment.
        role: Include ``role: assistant`` (first chunk of the turn).

    Returns:
        The chunk.
    """
    return StreamChunk(choices=[StreamChoice(delta=StreamDelta(
        role="assistant" if role else None, tool_calls=[delta]))])
