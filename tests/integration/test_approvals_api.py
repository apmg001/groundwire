"""End-to-end tests of human approval through the HTTP API."""

from __future__ import annotations

import json
import time
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

TOKEN = "test-reviewer-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
POLICY: dict[str, Any] = {"rules": [
    {"name": "refund-approval", "tools": ["issue_refund"], "action": "require_approval",
     "when": {"amount": {"min": 500}}, "arguments": {"amount": {"max": 5000}}},
    {"name": "refund-cap", "tools": ["issue_refund"], "arguments": {"amount": {"max": 5000}}},
]}


class RefundModel(Provider):
    """Fake model: refunds ``amount`` until it sees the tool result, then confirms."""

    def __init__(self, name: str, amount: float) -> None:
        super().__init__(name, f"{name}-model")
        self.amount = amount

    def _call(self) -> ToolCall:
        return ToolCall(id=f"call_{time.monotonic_ns()}", function=FunctionCall(
            name="issue_refund", arguments=json.dumps({"order_id": 4417, "amount": self.amount})))

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        if request.messages[-1].role == "tool":
            return Completion(text="Refund issued.", finish_reason="stop", usage=Usage.of(5, 5),
                              upstream_model=self.model)
        return Completion(text="", finish_reason="tool_calls", tool_calls=[self._call()],
                          usage=Usage.of(5, 5), upstream_model=self.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk.tool_calls([self._call()], role=True)
        yield StreamChunk.finish("tool_calls", Usage.of(5, 5))


def _client(client_factory: ClientFactory, monkeypatch: pytest.MonkeyPatch, *,
            amount: float = 1200, token: str | None = TOKEN, **config: Any) -> Any:
    if token is None:
        monkeypatch.delenv("GROUNDWIRE_APPROVER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GROUNDWIRE_APPROVER_TOKEN", token)
    gateway = make_config(tool_policy=config.pop("tool_policy", POLICY), **config)
    return client_factory({"primary": RefundModel("primary", amount),
                           "secondary": RefundModel("secondary", amount)}, gateway)


def _body(user: str = "alice", **extra: Any) -> dict[str, Any]:
    tool = {"type": "function", "function": {"name": "issue_refund", "parameters": {
        "type": "object", "properties": {"order_id": {"type": "integer"},
                                         "amount": {"type": "number"}}}}}
    return {"model": "default", "user": user, "tools": [tool],
            "messages": [{"role": "user", "content": "Refund order 4417"}], **extra}


def _decision(data: dict[str, Any]) -> dict[str, Any]:
    decision: dict[str, Any] = data["groundwire"]["tools"]["decisions"][0]
    return decision


def _hold(client: Any, user: str = "alice") -> str:
    data = client.post("/v1/chat/completions", json=_body(user)).json()
    assert _decision(data)["decision"] == "pending_approval"
    approval_id: str = _decision(data)["approval_id"]
    return approval_id


def _sse(text: str) -> list[Any]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


# --------------------------------------------------------------------------- #
# Agent side
# --------------------------------------------------------------------------- #


def test_held_call_ends_the_turn_with_an_explanation(
        client_factory: ClientFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    data = client.post("/v1/chat/completions", json=_body()).json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    decision = _decision(data)
    assert decision["decision"] == "pending_approval"
    assert decision["rule"] == "refund-approval"
    assert f"approval {decision['approval_id']}" in choice["message"]["content"]
    assert data["groundwire"]["tools"]["pending_approval"] == 1
    assert "1200" not in json.dumps(data)  # no argument values in text or metadata


def test_small_refunds_are_not_held(client_factory: ClientFactory,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch, amount=49.99)
    data = client.post("/v1/chat/completions", json=_body()).json()
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "issue_refund"


def test_status_endpoint_is_public_but_reveals_no_arguments(
        client_factory: ClientFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client)
    status = client.get(f"/v1/approvals/{approval_id}/status")
    assert status.status_code == 200
    assert set(status.json()) == {"id", "status", "tool", "expires_at"}
    assert status.json()["status"] == "pending"


# --------------------------------------------------------------------------- #
# Full flow
# --------------------------------------------------------------------------- #


def test_approve_then_the_same_call_is_released_once(
        client_factory: ClientFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client)

    pending = client.get("/v1/approvals", headers=AUTH).json()["data"]
    assert [r["id"] for r in pending] == [approval_id]
    assert pending[0]["arguments"] == {"order_id": 4417, "amount": 1200}  # reviewers see it

    decided = client.post(f"/v1/approvals/{approval_id}/approve", headers=AUTH,
                          json={"reviewer": "anita", "note": "verified"})
    assert decided.status_code == 200
    assert (decided.json()["status"], decided.json()["reviewer"]) == ("approved", "anita")

    released = client.post("/v1/chat/completions", json=_body()).json()
    assert released["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == (
        "issue_refund")
    assert _decision(released) == {"tool": "issue_refund",
                                   "id": _decision(released)["id"], "decision": "allowed",
                                   "detail": "approved by a reviewer", "rule": "refund-approval",
                                   "approval_id": approval_id}
    assert client.get(f"/v1/approvals/{approval_id}/status").json()["status"] == "consumed"

    replay = client.post("/v1/chat/completions", json=_body()).json()
    assert _decision(replay)["decision"] == "pending_approval"
    assert _decision(replay)["approval_id"] != approval_id

    metrics = client.get("/metrics").text
    for event in ("requested", "approved", "released"):
        assert f'groundwire_approvals_total{{event="{event}"}}' in metrics


def test_approval_is_scoped_to_the_user(client_factory: ClientFactory,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client, user="alice")
    client.post(f"/v1/approvals/{approval_id}/approve", headers=AUTH, json={"reviewer": "a"})
    other = client.post("/v1/chat/completions", json=_body(user="carol")).json()
    assert _decision(other)["decision"] == "pending_approval"


def test_rejected_call_is_blocked(client_factory: ClientFactory,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client)
    client.post(f"/v1/approvals/{approval_id}/reject", headers=AUTH,
                json={"reviewer": "anita", "note": "duplicate claim"})
    data = client.post("/v1/chat/completions", json=_body()).json()
    assert (_decision(data)["decision"], _decision(data)["reason"]) == (
        "blocked", "approval_rejected")
    assert "duplicate claim" not in json.dumps(data)  # reviewer notes stay with reviewers


def test_rejection_in_error_mode_is_a_422(client_factory: ClientFactory,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch,
                     tool_policy={**POLICY, "on_violation": "error"})
    approval_id = _hold(client)  # held calls are not violations
    client.post(f"/v1/approvals/{approval_id}/reject", headers=AUTH, json={"reviewer": "a"})
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "tool_call_blocked"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_streamed_hold_then_release(client_factory: ClientFactory,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    payloads = _sse(client.post("/v1/chat/completions", json=_body(stream=True)).text)
    text = "".join(p["choices"][0]["delta"].get("content", "")
                   for p in payloads if p.get("choices"))
    assert text.startswith("The requested action needs human approval")
    finishes = [p["choices"][0]["finish_reason"] for p in payloads
                if p.get("choices") and p["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]
    decision = payloads[-1]["groundwire"]["tools"]["decisions"][0]
    assert decision["decision"] == "pending_approval"

    client.post(f"/v1/approvals/{decision['approval_id']}/approve", headers=AUTH,
                json={"reviewer": "anita"})
    payloads = _sse(client.post("/v1/chat/completions", json=_body(stream=True)).text)
    calls = [c for p in payloads if p.get("choices")
             for c in p["choices"][0]["delta"].get("tool_calls", [])]
    assert [c["function"]["name"] for c in calls] == ["issue_refund"]


# --------------------------------------------------------------------------- #
# Reviewer API: security and errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-xxxxxxxxxx"},
                                     {"Authorization": f"Basic {TOKEN}"}])
def test_reviewer_endpoints_need_the_token(client_factory: ClientFactory,
                                           monkeypatch: pytest.MonkeyPatch,
                                           headers: dict[str, str]) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client)
    for method, path in [("GET", "/v1/approvals"), ("GET", f"/v1/approvals/{approval_id}"),
                         ("POST", f"/v1/approvals/{approval_id}/approve")]:
        response = client.request(method, path, headers=headers, json={"reviewer": "x"})
        assert response.status_code == 401, path
        assert response.json()["error"]["code"] == "unauthorized"
    assert client.get(f"/v1/approvals/{approval_id}/status").json()["status"] == "pending"


def test_reviewer_api_fails_closed_without_a_configured_token(
        client_factory: ClientFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch, token=None)
    approval_id = _hold(client)  # calls are still held: nothing runs unapproved
    response = client.post(f"/v1/approvals/{approval_id}/approve", headers=AUTH,
                           json={"reviewer": "anita"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "approvals_disabled"


def test_decision_errors(client_factory: ClientFactory,
                         monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client)
    ok = client.post(f"/v1/approvals/{approval_id}/approve", headers=AUTH, json={"reviewer": "a"})
    assert ok.status_code == 200
    twice = client.post(f"/v1/approvals/{approval_id}/reject", headers=AUTH, json={"reviewer": "b"})
    assert (twice.status_code, twice.json()["error"]["code"]) == (409, "approval_already_decided")

    unknown = client.get(f"/v1/approvals/apr_{'0' * 32}", headers=AUTH)
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (404, "approval_not_found")
    malformed = client.get("/v1/approvals/not-an-id/status")
    assert malformed.status_code == 400
    for body in ({}, {"reviewer": ""}, {"reviewer": "a", "extra": 1}):
        bad = client.post(f"/v1/approvals/{approval_id}/approve", headers=AUTH, json=body)
        assert bad.status_code == 400, body


def test_expired_request_cannot_be_approved(client_factory: ClientFactory,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch)
    approval_id = _hold(client)
    service = client.app.state.container.approvals
    monkeypatch.setattr(service, "_clock", lambda: time.time() + 10_000)  # past the 900s TTL
    response = client.post(f"/v1/approvals/{approval_id}/approve", headers=AUTH,
                           json={"reviewer": "anita"})
    assert (response.status_code, response.json()["error"]["code"]) == (410, "approval_expired")


def test_approvals_disabled(client_factory: ClientFactory,
                            monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(client_factory, monkeypatch, tool_policy={}, approvals={"enabled": False})
    response = client.get(f"/v1/approvals/apr_{'0' * 32}/status")
    assert (response.status_code, response.json()["error"]["code"]) == (503, "approvals_disabled")
