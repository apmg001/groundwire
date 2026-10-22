"""End-to-end tests of the agent answer check through the HTTP API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from conftest import ClientFactory, make_config
from groundwire.domain.models import (
    ChatCompletionRequest,
    Completion,
    FunctionCall,
    StreamChunk,
    ToolCall,
    Usage,
)
from groundwire.providers.base import Provider

pytestmark = pytest.mark.integration

LIE = "I have refunded $49.99 for order 7781."
TRUTH = "I have refunded $1 for order 7781."


class FixedModel(Provider):
    """Fake model with a fixed final answer, or a fixed tool call."""

    def __init__(self, name: str, text: str = "", call: str | None = None) -> None:
        super().__init__(name, f"{name}-model")
        self.text, self.call = text, call

    def _calls(self) -> list[ToolCall]:
        return [ToolCall(id="n1", function=FunctionCall(name=self.call, arguments="{}"))] \
            if self.call else []

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        return Completion(text=self.text, finish_reason="tool_calls" if self.call else "stop",
                          tool_calls=self._calls(), usage=Usage.of(5, 5),
                          upstream_model=self.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        for index, word in enumerate(self.text.split(" ")):
            yield StreamChunk.text(word if index == 0 else f" {word}", role=index == 0)
        if self.call:
            yield StreamChunk.tool_calls(self._calls(), role=not self.text)
        yield StreamChunk.finish("tool_calls" if self.call else "stop", Usage.of(5, 5))


def _client(client_factory: ClientFactory, text: str = LIE, *, call: str | None = None,
            **answer_check: Any) -> Any:
    config = make_config(answer_check=answer_check) if answer_check else make_config()
    return client_factory({"primary": FixedModel("primary", text, call),
                           "secondary": FixedModel("secondary", text, call)}, config)


def _tools(*names: str) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
            for n in names]


def _body(*, stream: bool = False, with_results: bool = True) -> dict[str, Any]:
    """The Qwen conversation: lookup returned total 49.99, refund returned amount 1."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Order 7781 arrived broken. Refund the full amount."}]
    if with_results:
        messages += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "lookup_order", "arguments": '{"order_id": 7781}'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "issue_refund",
                              "arguments": '{"order_id": 7781, "amount": 1}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": json.dumps({"order_id": 7781, "total": 49.99, "status": "delivered"})},
            {"role": "tool", "tool_call_id": "c2",
             "content": json.dumps({"refund_id": "R-7781", "amount": 1, "status": "refunded"})},
        ]
    return {"model": "default", "messages": messages, "stream": stream,
            "tools": _tools("lookup_order", "issue_refund")}


def _sse(text: str) -> list[Any]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def _streamed_text(payloads: list[Any]) -> str:
    return "".join(p["choices"][0]["delta"].get("content", "")
                   for p in payloads if p.get("choices"))


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #


def test_flag_mode_reports_the_lie(client_factory: ClientFactory) -> None:
    client = _client(client_factory)
    data = client.post("/v1/chat/completions", json=_body()).json()
    assert data["choices"][0]["message"]["content"] == LIE  # delivered unchanged
    check = data["groundwire"]["answer_check"]
    assert check["verdict"] == "mismatch"
    (issue,) = check["issues"]
    assert issue["kind"] == "number_mismatch" and issue["tool"] == "issue_refund"
    assert LIE[slice(*issue["span"])] == "$49.99"
    metrics = client.get("/metrics").text
    assert 'groundwire_answer_checks_total{verdict="mismatch"} 1.0' in metrics
    assert 'groundwire_answer_check_issues_total{kind="number_mismatch"} 1.0' in metrics


def test_true_answer_is_supported(client_factory: ClientFactory) -> None:
    client = _client(client_factory, TRUTH)
    check = client.post("/v1/chat/completions", json=_body()).json()["groundwire"][
        "answer_check"]
    assert check["verdict"] == "supported" and check["issues"] == []


def test_annotate_mode_appends_a_warning(client_factory: ClientFactory) -> None:
    client = _client(client_factory, action="annotate")
    content = client.post("/v1/chat/completions", json=_body()).json()[
        "choices"][0]["message"]["content"]
    assert content.startswith(LIE + "\n\n[Groundwire] Please verify this answer")
    assert "differs from what issue_refund returned" in content


def test_block_mode_refuses_the_answer(client_factory: ClientFactory) -> None:
    client = _client(client_factory, action="block")
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "answer_check_failed"
    assert error["details"]["answer_check"]["issues"][0]["kind"] == "number_mismatch"
    assert "49.99" not in response.text  # neither the answer nor the values leak


def test_block_mode_lets_true_answers_through(client_factory: ClientFactory) -> None:
    client = _client(client_factory, TRUTH, action="block")
    assert client.post("/v1/chat/completions", json=_body()).status_code == 200


def test_not_checked_without_tool_results(client_factory: ClientFactory) -> None:
    client = _client(client_factory)
    data = client.post("/v1/chat/completions", json=_body(with_results=False)).json()
    assert "answer_check" not in data["groundwire"]


def test_tool_call_turns_are_not_checked(client_factory: ClientFactory) -> None:
    client = _client(client_factory, "Refunding $49.99 now.", call="issue_refund",
                     action="block")
    data = client.post("/v1/chat/completions", json=_body()).json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert "answer_check" not in data["groundwire"]


def test_disabled_check(client_factory: ClientFactory) -> None:
    client = _client(client_factory, enabled=False)
    data = client.post("/v1/chat/completions", json=_body()).json()
    assert "answer_check" not in data["groundwire"]


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_streamed_flag_reports_in_the_trailing_metadata(client_factory: ClientFactory) -> None:
    client = _client(client_factory)
    payloads = _sse(client.post("/v1/chat/completions", json=_body(stream=True)).text)
    assert _streamed_text(payloads) == LIE
    assert payloads[-1]["groundwire"]["answer_check"]["verdict"] == "mismatch"


def test_streamed_annotate_is_buffered_and_warned(client_factory: ClientFactory) -> None:
    client = _client(client_factory, action="annotate")
    payloads = _sse(client.post("/v1/chat/completions", json=_body(stream=True)).text)
    text = _streamed_text(payloads)
    assert text.startswith(LIE) and "[Groundwire] Please verify this answer" in text
    finishes = [p["choices"][0]["finish_reason"] for p in payloads
                if p.get("choices") and p["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]


def test_streamed_block_fails_before_any_byte(client_factory: ClientFactory) -> None:
    client = _client(client_factory, action="block")
    response = client.post("/v1/chat/completions", json=_body(stream=True))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "answer_check_failed"


def test_streams_without_tool_results_stay_live(client_factory: ClientFactory) -> None:
    client = _client(client_factory, action="block")
    response = client.post("/v1/chat/completions", json=_body(stream=True, with_results=False))
    assert response.status_code == 200
    assert _streamed_text(_sse(response.text)) == LIE
