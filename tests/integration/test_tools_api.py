"""End-to-end tests of tool calling through the HTTP API."""

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
    FunctionCallDelta,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from groundwire.providers.base import Provider

pytestmark = pytest.mark.integration

REFUND_TOOL: dict[str, Any] = {"type": "function", "function": {
    "name": "issue_refund", "parameters": {"type": "object", "properties": {}}}}
EMAIL_TOOL: dict[str, Any] = {"type": "function", "function": {
    "name": "send_email", "parameters": {"type": "object", "properties": {}}}}


class AgentModel(Provider):
    """Fake model: calls ``tool`` with ``args`` first, then answers ``final`` after a result.

    Streams the tool call in small fragments, as real providers do.
    """

    def __init__(self, name: str, tool: str = "issue_refund",
                 args: dict[str, Any] | None = None, final: str = "Refund issued.",
                 text_before_call: str = "") -> None:
        super().__init__(name, f"{name}-model")
        self.tool, self.args, self.final = tool, args or {"order_id": 7781}, final
        self.text_before_call = text_before_call
        self.requests: list[ChatCompletionRequest] = []

    def _call(self) -> ToolCall:
        return ToolCall(id="call_1", function=FunctionCall(name=self.tool,
                                                           arguments=json.dumps(self.args)))

    def _wants_tool(self, request: ChatCompletionRequest) -> bool:
        return bool(request.tools) and request.messages[-1].role != "tool"

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        self.requests.append(request)
        if self._wants_tool(request):
            return Completion(text=self.text_before_call, finish_reason="tool_calls",
                              tool_calls=[self._call()], usage=Usage.of(5, 5),
                              upstream_model=self.model)
        return Completion(text=self.final, usage=Usage.of(5, 3), upstream_model=self.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        if not self._wants_tool(request):
            yield StreamChunk.text(self.final, role=True)
            yield StreamChunk.finish("stop", Usage.of(5, 3))
            return
        if self.text_before_call:
            yield StreamChunk.text(self.text_before_call, role=True)
        call = self._call()
        first = ToolCallDelta(index=0, id=call.id, type="function",
                              function=FunctionCallDelta(name=call.function.name, arguments=""))
        yield _delta_chunk(first)
        arguments = call.function.arguments
        for start in range(0, len(arguments), 4):  # 4-character fragments
            yield _delta_chunk(ToolCallDelta(index=0, function=FunctionCallDelta(
                arguments=arguments[start:start + 4])))
        yield StreamChunk.finish("tool_calls", Usage.of(5, 5))


def _delta_chunk(delta: ToolCallDelta) -> StreamChunk:
    chunk = StreamChunk.text("")
    chunk.choices[0].delta.content = None
    chunk.choices[0].delta.tool_calls = [delta]
    return chunk


def _providers(model: AgentModel) -> dict[str, Provider]:
    return {"primary": model, "secondary": AgentModel("secondary")}


def _body(*messages: dict[str, Any], tools: list[dict[str, Any]] | None = None,
          **extra: Any) -> dict[str, Any]:
    msgs = list(messages) or [{"role": "user", "content": "Refund order 7781"}]
    return {"model": "default", "messages": msgs, "tools": tools or [REFUND_TOOL], **extra}


def _sse(text: str) -> list[Any]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


# --------------------------------------------------------------------------- #
# Non-streaming agent loop
# --------------------------------------------------------------------------- #


def test_full_agent_loop(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(AgentModel("primary")))

    first = client.post("/v1/chat/completions", json=_body()).json()
    message = first["choices"][0]["message"]
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"] == [{"id": "call_1", "type": "function", "function": {
        "name": "issue_refund", "arguments": '{"order_id": 7781}'}}]

    # The agent runs the tool and sends the result back.
    second = client.post("/v1/chat/completions", json=_body(
        {"role": "user", "content": "Refund order 7781"}, message,
        {"role": "tool", "tool_call_id": "call_1", "content": "refund_id=R-1"})).json()
    assert second["choices"][0]["message"] == {"role": "assistant",
                                               "content": "Refund issued."}


def test_tool_arguments_carry_restored_pii(client_factory: ClientFactory) -> None:
    model = AgentModel("primary", tool="send_email", args={"to": "[EMAIL_1]"})
    config = make_config(safety={"pii": {"enabled": True}})
    client = client_factory(_providers(model), config)
    data = client.post("/v1/chat/completions", json=_body(
        {"role": "user", "content": "Email rahul@example.com the receipt"},
        tools=[EMAIL_TOOL])).json()
    arguments = data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"to": "rahul@example.com"}  # the real tool gets the real value
    assert "example.com" not in model.requests[0].model_dump_json()  # the model never saw it


def test_agent_traffic_bypasses_the_cache(client_factory: ClientFactory) -> None:
    model = AgentModel("primary")
    client = client_factory(_providers(model), make_config(cache={"enabled": True}))
    for _ in range(2):
        meta = client.post("/v1/chat/completions", json=_body()).json()["groundwire"]
        assert meta["cache"] == {"status": "bypass", "reason": "tool_use"}
    assert len(model.requests) == 2


def test_grounding_is_skipped_for_tool_calls(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(AgentModel("primary")))
    body = _body(groundwire={"grounding": {"context": ["Order 7781: 49.99 USD"],
                                           "mode": "block"}})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200  # not blocked: a call is not a claim
    assert response.json()["groundwire"]["grounding"] == {"verdict": "skipped",
                                                          "reason": "tool_calls"}


def test_invalid_tool_message_is_rejected(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(AgentModel("primary")))
    response = client.post("/v1/chat/completions", json=_body(
        {"role": "user", "content": "x"}, {"role": "tool", "content": "orphan result"}))
    assert response.status_code == 400  # OpenAI-style invalid_request_error


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_streamed_tool_call_arrives_complete_before_finish(client_factory: ClientFactory) -> None:
    model = AgentModel("primary", text_before_call="Let me do that. ")
    client = client_factory(_providers(model))
    payloads = _sse(client.post("/v1/chat/completions", json=_body(stream=True)).text)
    tool_chunks = [p for p in payloads if p.get("choices")
                   and p["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_chunks) == 1  # fragments were assembled into one chunk
    (call,) = tool_chunks[0]["choices"][0]["delta"]["tool_calls"]
    assert call["id"] == "call_1" and json.loads(call["function"]["arguments"]) == {
        "order_id": 7781}
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads
                   if p.get("choices"))
    assert text == "Let me do that. "
    finish_index = next(i for i, p in enumerate(payloads) if p.get("choices")
                        and p["choices"][0].get("finish_reason"))
    assert payloads.index(tool_chunks[0]) < finish_index
    assert payloads[finish_index]["choices"][0]["finish_reason"] == "tool_calls"


def test_streamed_tool_call_restores_pii(client_factory: ClientFactory) -> None:
    model = AgentModel("primary", tool="send_email", args={"to": "[EMAIL_1]"})
    client = client_factory(_providers(model), make_config(safety={"pii": {"enabled": True}}))
    payloads = _sse(client.post("/v1/chat/completions", json=_body(
        {"role": "user", "content": "Email rahul@example.com"}, tools=[EMAIL_TOOL],
        stream=True)).text)
    calls = [c for p in payloads if p.get("choices")
             for c in p["choices"][0]["delta"].get("tool_calls", [])]
    assert json.loads(calls[0]["function"]["arguments"]) == {"to": "rahul@example.com"}
    assert calls[0]["function"]["name"] == "send_email"
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"


def test_streaming_grounding_skipped_for_tool_calls(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(AgentModel("primary")))
    body = _body(stream=True, groundwire={"grounding": {"context": ["doc"], "mode": "flag"}})
    payloads = _sse(client.post("/v1/chat/completions", json=body).text)
    assert payloads[-1]["groundwire"]["grounding"]["verdict"] == "skipped"


def test_buffered_cascade_stream_replays_tool_calls(client_factory: ClientFactory) -> None:
    providers = [{"name": n, "type": "mock", "model": "m"}
                 for n in ("primary", "secondary", "cheap", "strong")]
    config = make_config(providers=providers,
                         cascades={"smart": {"cheap": ["cheap"], "strong": ["strong"]}})
    client = client_factory({**_providers(AgentModel("primary")), "cheap": AgentModel("cheap"),
                             "strong": AgentModel("strong")}, config)
    payloads = _sse(client.post("/v1/chat/completions",
                                json={**_body(stream=True), "model": "smart"}).text)
    calls = [c for p in payloads if p.get("choices")
             for c in p["choices"][0]["delta"].get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "issue_refund"
    assert payloads[-1]["groundwire"]["cascade"]["reason"] == "no_context"
