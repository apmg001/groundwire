"""Adapter for the Anthropic Messages API.

Differences from the OpenAI format that this adapter hides:

* The system prompt is a top-level ``system`` field, not a message.
* ``max_tokens`` is required.
* Consecutive messages with the same role must be merged.
* Streaming uses named SSE events (``message_start``, ``content_block_delta``,
  ``message_delta``, ``message_stop``, ``ping``, ``error``).
* Stop reasons differ (``end_turn``, ``max_tokens``, ``stop_sequence``...).
* HTTP 529 means "overloaded" (treated as a retryable server error).

``base_url`` is the API root, e.g. ``https://api.anthropic.com``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import SecretStr

from groundwire.core.errors import ProviderResponseError, ProviderServerError
from groundwire.domain.models import (
    ChatCompletionRequest,
    Completion,
    FinishReason,
    StreamChunk,
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
        merged: list[dict[str, str]] = []
        for msg in request.messages:
            if msg.role == "system":
                continue
            if merged and merged[-1]["role"] == msg.role:
                merged[-1]["content"] += "\n\n" + msg.content
            else:
                merged.append({"role": msg.role, "content": msg.content})
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
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion. See :meth:`Provider.stream`."""
        payload = self._payload(request, stream=True)
        input_tokens = 0
        output_tokens = 0
        finish: FinishReason | None = None
        first = True

        with translate_transport_errors(self.name):
            async with self._client.stream("POST", "/v1/messages", json=payload) as response:
                await raise_for_status(self.name, response)
                async for event in iter_sse(response.aiter_lines()):
                    data = parse_event_json(self.name, event)
                    kind = data.get("type", event.event)

                    if kind == "message_start":
                        usage = (data.get("message") or {}).get("usage") or {}
                        input_tokens = int(usage.get("input_tokens") or 0)
                    elif kind == "content_block_delta":
                        delta = data.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield StreamChunk.text(str(delta["text"]), role=first)
                            first = False
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
                    # ping, content_block_start/stop and unknown events are ignored.

        yield StreamChunk.finish(finish or "stop", Usage.of(input_tokens, output_tokens))

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()
