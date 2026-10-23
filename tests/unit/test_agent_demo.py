"""Tests for the agent demo's transcript labels (scripts/agent_demo.py).

The demo is a script, not a package module, so it is loaded from its path.
The gateway is faked with ``httpx.MockTransport``: no server or model needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agent_demo.py"


def _load_demo() -> ModuleType:
    """Import ``scripts/agent_demo.py`` by path.

    Returns:
        The loaded module.

    Raises:
        ImportError: If the script cannot be found or loaded.
    """
    spec = importlib.util.spec_from_file_location("agent_demo", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


def _reply(content: str | None, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    meta: dict[str, Any] = {"provider": "ollama-local"}
    if decisions:
        meta["tools"] = {"decisions": decisions}
    return {"choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}], "groundwire": meta}


def _client(*replies: dict[str, Any]) -> httpx.Client:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=queue.pop(0))

    return httpx.Client(base_url="http://gateway", transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# final_answer_label
# --------------------------------------------------------------------------- #


def test_plain_answer_is_labelled_as_the_model() -> None:
    assert demo.final_answer_label([], "ollama-local") == "model answered (via ollama-local)"


@pytest.mark.parametrize(("count", "noun"), [(1, "call"), (2, "calls")])
def test_blocked_answer_is_labelled_as_the_gateway(count: int, noun: str) -> None:
    blocked = [{"tool": "issue_refund", "decision": "blocked"}] * count
    label = demo.final_answer_label(blocked, "ollama-local")
    assert label.startswith(f"gateway blocked the tool {noun} and explained why")
    assert "model answered" not in label


def test_missing_provider_does_not_crash() -> None:
    assert "unknown provider" in demo.final_answer_label([], None)


# --------------------------------------------------------------------------- #
# run_agent end to end (fake gateway)
# --------------------------------------------------------------------------- #


def test_run_agent_marks_a_blocked_refund() -> None:
    blocked = {"tool": "issue_refund", "id": "c1", "decision": "blocked",
               "reason": "argument_limit", "detail": "'amount' is above the allowed maximum 5000"}
    text = "The requested action was not carried out: issue_refund (...)"
    with _client(_reply(text, [blocked])) as client:
        run = demo.run_agent(client, "default", "refund order 9001")
    assert run.answer == text
    assert run.steps[0].startswith("[1] gateway BLOCKED issue_refund")
    assert "gateway blocked the tool call and explained why" in run.steps[-1]


def test_run_agent_keeps_the_model_label_for_normal_answers() -> None:
    with _client(_reply("Refunded $49.99.", [])) as client:
        run = demo.run_agent(client, "default", "refund order 7781")
    assert run.steps == ["[1] model answered (via ollama-local)"]


def test_run_tool_reports_errors_to_the_model() -> None:
    result = json.loads(demo.run_tool("issue_refund", '{"order_id": 7781, "amount": 1000}'))
    assert "exceeds order total" in result["error"]


# --------------------------------------------------------------------------- #
# answer_check_lines
# --------------------------------------------------------------------------- #


def test_answer_check_mismatch_points_at_the_claim() -> None:
    answer = "I have refunded $49.99 for order 7781."
    check = {"verdict": "mismatch", "issues": [
        {"kind": "number_mismatch", "span": [16, 22], "tool": "issue_refund",
         "detail": "a number in the answer differs from what issue_refund returned"}]}
    lines = demo.answer_check_lines(check, answer)
    assert lines[0].startswith("gateway answer check: MISMATCH")
    assert lines[1] == ("  - '$49.99': a number in the answer differs from what "
                        "issue_refund returned")


def test_answer_check_supported_and_absent() -> None:
    assert demo.answer_check_lines({"verdict": "supported"}, "ok") == [
        "gateway answer check: the answer matches the tool results"]
    assert demo.answer_check_lines(None, "ok") == []


def test_answer_check_bad_span_does_not_crash() -> None:
    check = {"verdict": "mismatch", "issues": [{"kind": "number_mismatch", "span": [5, 99]}]}
    assert demo.answer_check_lines(check, "short")[1] == "  - '?': number_mismatch"


# --------------------------------------------------------------------------- #
# Human approval
# --------------------------------------------------------------------------- #


class _FakeGateway:
    """Holds the refund until it is approved, then releases it (like Groundwire)."""

    def __init__(self) -> None:
        self.status = "pending"
        self.approve_calls: list[dict[str, Any]] = []
        self.chat_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/approve"):
            self.approve_calls.append({"auth": request.headers.get("authorization"),
                                       "body": json.loads(request.content)})
            self.status = "approved"
            return httpx.Response(200, json={"status": "approved"})
        if path.endswith("/status"):
            return httpx.Response(200, json={"status": self.status})
        self.chat_calls += 1
        messages = json.loads(request.content)["messages"]
        if messages[-1]["role"] == "tool":
            return httpx.Response(200, json=_reply("Refunded $1200.", []))
        if self.status == "pending":
            held = {"tool": "issue_refund", "decision": "pending_approval",
                    "approval_id": "apr_" + "1" * 32}
            return httpx.Response(200, json=_reply("needs approval", [held]))
        call = {"id": "c1", "type": "function", "function": {
            "name": "issue_refund", "arguments": '{"order_id": 4417, "amount": 1200}'}}
        reply = _reply(None, [{"tool": "issue_refund", "decision": "allowed",
                               "approval_id": "apr_" + "1" * 32}])
        reply["choices"][0]["message"]["tool_calls"] = [call]
        return httpx.Response(200, json=reply)


def test_auto_approve_releases_the_call_and_finishes() -> None:
    gateway = _FakeGateway()
    waiter = demo.ApprovalWaiter(approve_with_token="t" * 20, sleep=lambda _: None)
    with httpx.Client(base_url="http://gw", transport=httpx.MockTransport(gateway.handler)) as c:
        run = demo.run_agent(c, "agent-demo-approval", demo.SCENARIOS["approval"],
                             waiter=waiter)
    assert run.answer == "Refunded $1200."
    assert any("HELD issue_refund" in s for s in run.steps)
    assert any("RELEASED issue_refund" in s for s in run.steps)
    assert gateway.approve_calls[0]["auth"] == "Bearer " + "t" * 20
    assert gateway.chat_calls == 3  # held, released, final answer


def test_waiter_times_out_while_pending() -> None:
    gateway = _FakeGateway()
    ticks = iter(range(0, 1000, 10))
    waiter = demo.ApprovalWaiter(timeout_s=25, sleep=lambda _: None, clock=lambda: next(ticks))
    with httpx.Client(base_url="http://gw", transport=httpx.MockTransport(gateway.handler)) as c:
        assert waiter.wait(c, ["apr_" + "1" * 32]) == {"apr_" + "1" * 32: "pending"}


def test_without_a_waiter_the_hold_is_the_answer() -> None:
    gateway = _FakeGateway()
    with httpx.Client(base_url="http://gw", transport=httpx.MockTransport(gateway.handler)) as c:
        run = demo.run_agent(c, "agent-demo-approval", demo.SCENARIOS["approval"])
    assert run.answer == "needs approval"
    assert "holding the call for human approval" in run.steps[-1]
