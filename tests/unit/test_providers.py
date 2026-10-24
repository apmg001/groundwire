"""Tests for HTTP provider adapters and shared HTTP helpers (no real network)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from pydantic import SecretStr

from conftest import chat_request
from groundwire.core.config import MockBehaviour
from groundwire.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
)
from groundwire.domain.models import ChatCompletionRequest
from groundwire.providers.anthropic import AnthropicProvider
from groundwire.providers.http import iter_sse, parse_retry_after
from groundwire.providers.mock import MockProvider
from groundwire.providers.openai_compat import OpenAICompatibleProvider

OPENAI_URL = "https://llm.test/v1"
ANTHROPIC_URL = "https://anthropic.test"


def _openai() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(name="oa", model="gpt-x", base_url=OPENAI_URL,
                                    api_key=SecretStr("sk-test"), timeout_s=5,
                                    max_connections=4)


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(name="an", model="claude-x", base_url=ANTHROPIC_URL,
                             api_key=SecretStr("sk-ant"), timeout_s=5, max_connections=4,
                             default_max_tokens=256)


def _sse(*events: str) -> bytes:
    return "".join(f"{e}\n\n" for e in events).encode()


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def test_iter_sse_parses_events_comments_and_multiline() -> None:
    lines = [": keep-alive", "event: delta", "data: a", "data: b", "", "data: c", ""]
    events = [e async for e in iter_sse(_lines(*lines))]
    assert [(e.event, e.data) for e in events] == [("delta", "a\nb"), ("message", "c")]


async def test_iter_sse_flushes_trailing_event() -> None:
    events = [e async for e in iter_sse(_lines("data: last"))]
    assert [e.data for e in events] == ["last"]


@pytest.mark.parametrize(("value", "expected"), [
    (None, None), ("", None), ("3", 3.0), ("-5", 0.0), ("garbage", None),
    ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),  # date in the past clamps to 0
])
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    assert parse_retry_after(value) == expected


# --------------------------------------------------------------------------- #
# OpenAI-compatible
# --------------------------------------------------------------------------- #


@respx.mock
async def test_openai_complete_success_and_payload() -> None:
    route = respx.post(f"{OPENAI_URL}/chat/completions").respond(200, json={
        "model": "gpt-x-0613",
        "choices": [{"message": {"role": "assistant", "content": "Hi!"},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    })
    provider = _openai()
    result = await provider.complete(chat_request(temperature=0.2, max_tokens=10, stop="END"))
    await provider.aclose()

    assert result.text == "Hi!"
    assert result.finish_reason == "length"
    assert result.usage.total_tokens == 6
    assert result.upstream_model == "gpt-x-0613"

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "gpt-x"
    assert sent["temperature"] == 0.2
    assert sent["stop"] == ["END"]
    assert "top_p" not in sent  # unset params are not forwarded
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"


@pytest.mark.parametrize(("status", "error_type"), [
    (400, ProviderBadRequestError),
    (401, ProviderAuthError),
    (403, ProviderAuthError),
    (404, ProviderNotFoundError),  # e.g. a model the vendor retired
    (408, ProviderServerError),
    (422, ProviderBadRequestError),
    (500, ProviderServerError),
    (503, ProviderServerError),
])
@respx.mock
async def test_openai_status_classification(status: int, error_type: type[Exception]) -> None:
    respx.post(f"{OPENAI_URL}/chat/completions").respond(status, text="nope")
    with pytest.raises(error_type):
        await _openai().complete(chat_request())


@respx.mock
async def test_openai_rate_limit_parses_retry_after() -> None:
    respx.post(f"{OPENAI_URL}/chat/completions").respond(
        429, text="slow", headers={"Retry-After": "2"})
    with pytest.raises(ProviderRateLimitError) as info:
        await _openai().complete(chat_request())
    assert info.value.retry_after_s == 2.0
    assert info.value.retryable


@respx.mock
async def test_openai_timeout_and_connection_errors() -> None:
    route = respx.post(f"{OPENAI_URL}/chat/completions")
    route.side_effect = httpx.ReadTimeout("slow")
    with pytest.raises(ProviderTimeoutError):
        await _openai().complete(chat_request())
    route.side_effect = httpx.ConnectError("refused")
    with pytest.raises(ProviderConnectionError):
        await _openai().complete(chat_request())


@pytest.mark.parametrize("body", ["not json", "[]", '{"choices": []}', '{"choices": [{}]}'])
@respx.mock
async def test_openai_malformed_responses(body: str) -> None:
    respx.post(f"{OPENAI_URL}/chat/completions").respond(200, text=body)
    with pytest.raises(ProviderResponseError):
        await _openai().complete(chat_request())


@respx.mock
async def test_openai_stream() -> None:
    body = _sse(
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
        "data: [DONE]",
    )
    respx.post(f"{OPENAI_URL}/chat/completions").respond(
        200, content=body, headers={"content-type": "text/event-stream"})
    chunks = [c async for c in _openai().stream(chat_request(stream=True))]

    assert "".join(c.content() for c in chunks) == "Hello"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert any(c.choices[0].finish_reason == "stop" for c in chunks)
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5


@respx.mock
async def test_openai_stream_error_status_raises_before_first_chunk() -> None:
    respx.post(f"{OPENAI_URL}/chat/completions").respond(503, text="down")
    with pytest.raises(ProviderServerError):
        await anext(_openai().stream(chat_request(stream=True)))


@respx.mock
async def test_openai_stream_in_band_error() -> None:
    respx.post(f"{OPENAI_URL}/chat/completions").respond(
        200, content=_sse('data: {"error":{"message":"overloaded"}}'))
    with pytest.raises(ProviderResponseError):
        [c async for c in _openai().stream(chat_request(stream=True))]


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


def test_anthropic_payload_translation() -> None:
    request = ChatCompletionRequest(messages=[
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ], temperature=1.7, stop=["X"])
    payload = _anthropic()._payload(request, stream=False)

    assert payload["system"] == "Be brief."
    assert payload["messages"] == [{"role": "user", "content": "a\n\nb"}]  # merged
    assert payload["max_tokens"] == 256  # default applied
    assert payload["temperature"] == 1.0  # clamped to Anthropic's range
    assert payload["stop_sequences"] == ["X"]


@respx.mock
async def test_anthropic_complete() -> None:
    route = respx.post(f"{ANTHROPIC_URL}/v1/messages").respond(200, json={
        "model": "claude-x",
        "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " there"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    })
    result = await _anthropic().complete(chat_request())
    assert result.text == "Hello there"
    assert result.finish_reason == "length"
    assert result.usage.total_tokens == 10
    headers = route.calls.last.request.headers
    assert headers["x-api-key"] == "sk-ant"
    assert headers["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_anthropic_529_is_retryable_server_error() -> None:
    respx.post(f"{ANTHROPIC_URL}/v1/messages").respond(529, text="overloaded")
    with pytest.raises(ProviderServerError) as info:
        await _anthropic().complete(chat_request())
    assert info.value.retryable


@respx.mock
async def test_anthropic_stream() -> None:
    body = _sse(
        'event: message_start\ndata: {"type":"message_start",'
        '"message":{"usage":{"input_tokens":9}}}',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0}',
        "event: ping\ndata: {\"type\":\"ping\"}",
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"Hi"}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":" you"}}',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":2}}',
        'event: message_stop\ndata: {"type":"message_stop"}',
    )
    respx.post(f"{ANTHROPIC_URL}/v1/messages").respond(200, content=body)
    chunks = [c async for c in _anthropic().stream(chat_request(stream=True))]

    assert "".join(c.content() for c in chunks) == "Hi you"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 9
    assert chunks[-1].usage.completion_tokens == 2


@respx.mock
async def test_anthropic_stream_error_event() -> None:
    respx.post(f"{ANTHROPIC_URL}/v1/messages").respond(200, content=_sse(
        'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",'
        '"message":"Overloaded"}}'))
    with pytest.raises(ProviderServerError, match="overloaded_error"):
        [c async for c in _anthropic().stream(chat_request(stream=True))]


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


async def test_mock_failure_injection_is_reproducible() -> None:
    async def run() -> list[bool]:
        mock = MockProvider(name="m", model="m",
                            behaviour=MockBehaviour(failure_rate=0.5, seed=42))
        outcomes = []
        for _ in range(20):
            try:
                await mock.complete(chat_request())
                outcomes.append(True)
            except ProviderServerError:
                outcomes.append(False)
        return outcomes

    first, second = await run(), await run()
    assert first == second  # same seed, same failure pattern
    assert True in first and False in first


async def test_mock_stream_yields_text_and_usage() -> None:
    mock = MockProvider(name="m", model="m", behaviour=MockBehaviour(response_text="a b c"))
    chunks = [c async for c in mock.stream(chat_request(stream=True))]
    assert "".join(c.content() for c in chunks) == "a b c"
    assert chunks[-1].usage is not None
