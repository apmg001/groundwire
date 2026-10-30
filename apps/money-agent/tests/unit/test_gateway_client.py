"""The Groundwire HTTP client: parsing, retries, and error mapping (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from money_agent.adapters import groundwire as module
from money_agent.adapters.groundwire import GroundwireClient
from money_agent.core.errors import AssistantUnavailableError, MessageRefusedError
from money_agent.core.logging import reset_request_id, set_request_id

OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "get_spending_summary", "arguments": "{\"month\":\"2026-09\"}"}}]},
        "finish_reason": "tool_calls"}],
    "groundwire": {"provider": "groq", "answer_check": {"verdict": "skipped"}},
}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None
    monkeypatch.setattr(module.asyncio, "sleep", instant)


def _client(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
            ) -> tuple[GroundwireClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return GroundwireClient(http, base_url="http://gw:8000/", model="default",
                            api_key="k", **kwargs), seen


async def test_success_is_parsed_and_the_request_is_well_formed() -> None:
    client, seen = _client(lambda _: httpx.Response(200, json=OK_BODY))
    token = set_request_id("req-12345678")
    try:
        reply = await client.complete([{"role": "user", "content": "hi"}],
                                      tools=[{"type": "function"}], user="u1")
    finally:
        reset_request_id(token)
    assert reply.tool_calls[0].name == "get_spending_summary"
    assert reply.finish_reason == "tool_calls" and reply.gateway["provider"] == "groq"
    [request] = seen
    assert str(request.url) == "http://gw:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    assert request.headers["x-request-id"] == "req-12345678"
    body = json.loads(request.content)
    assert body["user"] == "u1" and body["tool_choice"] == "auto" and body["model"] == "default"


async def test_reply_length_is_capped_when_configured() -> None:
    client, seen = _client(lambda _: httpx.Response(200, json=OK_BODY), max_tokens=900)
    await client.complete([{"role": "user", "content": "hi"}], tools=[], user="u1")
    assert json.loads(seen[0].content)["max_tokens"] == 900


async def test_no_tools_means_no_tool_fields() -> None:
    client, seen = _client(lambda _: httpx.Response(200, json=OK_BODY))
    await client.complete([{"role": "user", "content": "hi"}], tools=[], user="u1")
    assert "tools" not in json.loads(seen[0].content)


async def test_transient_failures_are_retried() -> None:
    statuses = iter([503, 429, 200])

    def handler(_: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, json=OK_BODY if status == 200 else {},
                              headers={"Retry-After": "1"} if status == 429 else {})

    client, seen = _client(handler, max_retries=2)
    await client.complete([], tools=[], user="u")
    assert len(seen) == 3


async def test_retries_give_up_and_errors_are_mapped() -> None:
    client, seen = _client(lambda _: httpx.Response(
        503, json={"error": {"code": "all_providers_failed", "request_id": "gw-1"}}),
        max_retries=1)
    with pytest.raises(AssistantUnavailableError):
        await client.complete([], tools=[], user="u")
    assert len(seen) == 2


@pytest.mark.parametrize("code", ["prompt_injection_detected", "tool_call_blocked"])
async def test_refusals_are_not_retried(code: str) -> None:
    client, seen = _client(lambda _: httpx.Response(400, json={"error": {"code": code}}))
    with pytest.raises(MessageRefusedError) as caught:
        await client.complete([], tools=[], user="u")
    assert caught.value.details == {"reason": code} and len(seen) == 1


async def test_upstream_text_never_reaches_the_public_message() -> None:
    client, _ = _client(lambda _: httpx.Response(
        500, json={"error": {"code": "internal", "message": "stack trace at db.py:12"}}))
    with pytest.raises(AssistantUnavailableError) as caught:
        await client.complete([], tools=[], user="u")
    assert "stack trace" not in caught.value.public_message


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>proxy error</html>"),
    httpx.Response(200, json={"choices": []}),
])
async def test_malformed_responses(response: httpx.Response) -> None:
    client, _ = _client(lambda _: response)
    with pytest.raises(AssistantUnavailableError):
        await client.complete([], tools=[], user="u")


async def test_connection_errors_are_retried_then_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, seen = _client(handler, max_retries=2)
    with pytest.raises(AssistantUnavailableError):
        await client.complete([], tools=[], user="u")
    assert len(seen) == 3


def test_backoff_honours_retry_after_with_a_cap() -> None:
    assert module._backoff(0, "2") == 2.0
    assert module._backoff(0, "999") == module.MAX_RETRY_AFTER_S
    assert 0 <= module._backoff(3, "soon") <= module.BASE_BACKOFF_S * 8
