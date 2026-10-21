"""Adapter for OpenAI-compatible Chat Completions APIs.

One adapter covers a large part of the ecosystem, because most servers copy
OpenAI's wire format: OpenAI itself, llama.cpp's ``llama-server``, Ollama's
``/v1`` endpoint, vLLM, LM Studio, Groq, Together, OpenRouter...

``base_url`` must include the version prefix, e.g. ``http://localhost:8080/v1``.

Tool calling is OpenAI's own format, so it passes through with validation:
``tools``/``tool_choice`` go upstream unchanged, ``message.tool_calls`` are
parsed into domain objects, and streamed ``delta.tool_calls`` fragments are
forwarded for the service to assemble.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from pydantic import SecretStr, ValidationError

from groundwire.core.errors import ProviderResponseError
from groundwire.domain.models import (
    ChatCompletionRequest,
    Completion,
    FinishReason,
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

_KNOWN_FINISH: frozenset[str] = frozenset({"stop", "length", "content_filter", "tool_calls"})


def _finish_reason(value: Any) -> FinishReason | None:
    """Normalise an upstream finish reason; unknown values map to ``"stop"``."""
    if value is None:
        return None
    return cast(FinishReason, value) if value in _KNOWN_FINISH else "stop"


def _tool_calls(name: str, raw: Any) -> list[ToolCall]:
    """Parse ``message.tool_calls`` from an upstream response.

    Args:
        name: Provider name, for errors.
        raw: The raw ``tool_calls`` value.

    Returns:
        Parsed calls (empty if absent).

    Raises:
        ProviderResponseError: If present but malformed.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ProviderResponseError(f"{name}: tool_calls is not a list", provider=name)
    try:
        return [ToolCall.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ProviderResponseError(f"{name}: malformed tool_calls: {exc.error_count()} errors",
                                    provider=name) from exc


def _tool_call_deltas(name: str, raw: Any) -> list[ToolCallDelta]:
    """Parse streamed ``delta.tool_calls`` fragments.

    Args:
        name: Provider name, for errors.
        raw: The raw fragments.

    Returns:
        Parsed fragments (empty if absent).

    Raises:
        ProviderResponseError: If present but malformed.
    """
    if not raw:
        return []
    try:
        return [ToolCallDelta.model_validate(item) for item in raw]
    except (ValidationError, TypeError) as exc:
        raise ProviderResponseError(f"{name}: malformed tool_call delta", provider=name) from exc


def _usage(raw: Any) -> Usage:
    """Parse an OpenAI ``usage`` object defensively (fields may be missing)."""
    if not isinstance(raw, dict):
        return Usage()
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    return Usage.of(prompt, completion)


class OpenAICompatibleProvider(Provider):
    """Provider for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: SecretStr | None,
        timeout_s: float,
        max_connections: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            name: Provider name.
            model: Upstream model id.
            base_url: API base including version prefix.
            api_key: Bearer token, or ``None`` for unauthenticated local servers.
            timeout_s: Per-call timeout.
            max_connections: Connection-pool size.
            transport: Optional custom transport (tests).
        """
        super().__init__(name, model)
        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        self._client = build_client(
            base_url=base_url, timeout_s=timeout_s, max_connections=max_connections,
            headers=headers, transport=transport,
        )

    # ------------------------------------------------------------------ #
    # Request building
    # ------------------------------------------------------------------ #

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        """Translate a domain request into the upstream JSON body.

        Args:
            request: Client request.
            stream: Whether to request SSE streaming.

        Returns:
            JSON-serialisable body.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in request.messages],
            "stream": stream,
        }
        if request.tools:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
        if request.tool_choice is not None:
            choice = request.tool_choice
            payload["tool_choice"] = (choice.model_dump() if isinstance(choice, NamedToolChoice)
                                      else choice)
        if request.parallel_tool_calls is not None and request.tools:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        optional = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop_list(),
            "user": request.user,
        }
        payload.update({k: v for k, v in optional.items() if v is not None})
        if stream:
            # Ask for a final usage chunk; servers that don't know it ignore it.
            payload["stream_options"] = {"include_usage": True}
        return payload

    # ------------------------------------------------------------------ #
    # Provider API
    # ------------------------------------------------------------------ #

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        """Run a non-streaming completion. See :meth:`Provider.complete`."""
        with translate_transport_errors(self.name):
            response = await self._client.post(
                "/chat/completions", json=self._payload(request, stream=False))
        await raise_for_status(self.name, response)
        data = decode_json(self.name, response)

        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            raw_calls = message.get("tool_calls")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderResponseError(
                f"{self.name}: response missing choices[0].message", provider=self.name
            ) from exc

        calls = _tool_calls(self.name, raw_calls)
        finish = _finish_reason(choice.get("finish_reason")) or "stop"
        if calls and finish == "stop":
            finish = "tool_calls"  # some servers (older llama.cpp, Ollama) report "stop"
        return Completion(
            text=text,
            finish_reason=finish,
            usage=_usage(data.get("usage")),
            upstream_model=str(data.get("model") or self.model),
            tool_calls=calls,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion. See :meth:`Provider.stream`."""
        payload = self._payload(request, stream=True)
        with translate_transport_errors(self.name):
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                await raise_for_status(self.name, response)
                first = True
                async for event in iter_sse(response.aiter_lines()):
                    if event.data.strip() == "[DONE]":
                        return
                    data = parse_event_json(self.name, event)
                    if "error" in data:
                        raise ProviderResponseError(
                            f"{self.name}: error in stream: {data['error']}", provider=self.name)
                    chunk = self._chunk_from(data, first=first)
                    if chunk is not None:
                        first = False
                        yield chunk

    def _chunk_from(self, data: dict[str, Any], *, first: bool) -> StreamChunk | None:
        """Convert one upstream SSE payload into a domain chunk.

        Args:
            data: Decoded SSE payload.
            first: Whether this would be the first emitted chunk.

        Returns:
            A chunk, or ``None`` if the payload carries nothing useful.
        """
        choices = data.get("choices") or []
        usage = _usage(data["usage"]) if isinstance(data.get("usage"), dict) else None
        if not choices:
            # Usage-only trailer (stream_options.include_usage).
            return StreamChunk.finish(None, usage) if usage else None
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        finish = _finish_reason(choice.get("finish_reason"))
        tool_deltas = _tool_call_deltas(self.name, delta.get("tool_calls"))
        if tool_deltas:
            return StreamChunk(usage=usage, choices=[StreamChoice(
                delta=StreamDelta(role="assistant" if first else None, content=content or None,
                                  tool_calls=tool_deltas),
                finish_reason=finish)])
        if finish is not None:
            if content:
                # Rare, but some servers put the last text in the finish chunk.
                chunk = StreamChunk.text(content, role=first)
                chunk.choices[0].finish_reason = finish
                chunk.usage = usage
                return chunk
            return StreamChunk.finish(finish, usage)
        if content:
            return StreamChunk.text(content, role=first)
        return None

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()
