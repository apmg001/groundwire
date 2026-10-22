"""Unit tests for the agent answer check (groundwire.agents.answer_check).

The fixture conversation is the real failure seen with Qwen 2.5 3B: the
refund tool returned ``amount: 1`` and the model reported ``$49.99``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from groundwire.agents.answer_check import (
    AnswerChecker,
    CheckVerdict,
    IssueKind,
    collect_evidence,
)
from groundwire.core.config import AnswerCheckConfig
from groundwire.domain.models import ChatCompletionRequest

ORDER = {"order_id": 7781, "item": "wireless mouse", "total": 49.99, "currency": "USD",
         "status": "delivered"}


def _tool(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _request(refund_args: dict[str, Any] | None = None,
             refund_result: Any = None, *, user: str = "Order 7781 arrived broken.",
             tools: tuple[str, ...] = ("lookup_order", "issue_refund", "delete_user"),
             ) -> ChatCompletionRequest:
    """A conversation where the agent looked the order up and (maybe) refunded it."""
    calls = [{"id": "c1", "type": "function",
              "function": {"name": "lookup_order", "arguments": '{"order_id": 7781}'}}]
    results = [{"role": "tool", "tool_call_id": "c1", "content": json.dumps(ORDER)}]
    if refund_args is not None:
        calls.append({"id": "c2", "type": "function",
                      "function": {"name": "issue_refund", "arguments": json.dumps(refund_args)}})
        content = refund_result if isinstance(refund_result, str) else json.dumps(refund_result)
        results.append({"role": "tool", "tool_call_id": "c2", "content": content})
    return ChatCompletionRequest(model="default", tools=[_tool(t) for t in tools], messages=[
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": None, "tool_calls": calls}, *results])


REFUNDED_ONE = _request({"order_id": 7781, "amount": 1},
                        {"refund_id": "R-7781", "order_id": 7781, "amount": 1,
                         "status": "refunded"})
REFUND_FAILED = _request({"order_id": 7781, "amount": 500},
                         {"error": "refund 500 exceeds order total 49.99"})


@pytest.fixture
def checker() -> AnswerChecker:
    return AnswerChecker(AnswerCheckConfig())


def _kinds(checker: AnswerChecker, answer: str, request: ChatCompletionRequest) -> list[str]:
    return [issue.kind.value for issue in checker.check(answer, request).issues]


# --------------------------------------------------------------------------- #
# The real failure
# --------------------------------------------------------------------------- #


def test_catches_the_wrong_refund_amount(checker: AnswerChecker) -> None:
    answer = "I have refunded $49.99 for order 7781."
    report = checker.check(answer, REFUNDED_ONE)
    assert report.verdict is CheckVerdict.MISMATCH
    (issue,) = report.issues
    assert issue.kind is IssueKind.NUMBER_MISMATCH
    assert issue.tool == "issue_refund"
    assert answer[issue.span[0]:issue.span[1]] == "$49.99"
    assert report.tool_results == 2


def test_true_answer_passes(checker: AnswerChecker) -> None:
    report = checker.check("I have refunded $1 for order 7781.", REFUNDED_ONE)
    assert report.verdict is CheckVerdict.SUPPORTED
    assert report.numbers_checked == 2
    assert report.actions_checked == 1


@pytest.mark.parametrize("answer", [
    "Order 7781 total was $49.99; I refunded $1.",
    "Order 7781 (wireless mouse, total $49.99) was refunded: $1.",
    "Your order total is $49.99 and the refund amount was $1.",
    "Refunded $1.00 for order #7781.",
    "I\u2019ve refunded $1, it didn\u2019t take long.",
])
def test_numbers_bind_to_the_right_tool(checker: AnswerChecker, answer: str) -> None:
    assert _kinds(checker, answer, REFUNDED_ONE) == []


def test_rounding_to_the_answers_precision_is_allowed(checker: AnswerChecker) -> None:
    request = _request({"order_id": 7781, "amount": 1234.5678},
                       {"order_id": 7781, "amount": 1234.5678})
    assert _kinds(checker, "Refunded $1,234.57.", request) == []
    assert _kinds(checker, "Refunded $1,235.", request) == []
    assert _kinds(checker, "Refunded $1,240.", request) == ["number_mismatch"]


# --------------------------------------------------------------------------- #
# Unsupported numbers
# --------------------------------------------------------------------------- #


def test_invented_money_is_reported(checker: AnswerChecker) -> None:
    assert _kinds(checker, "Refunded $1. A $5.00 coupon was also applied.",
                  REFUNDED_ONE) == ["unsupported_number"]


def test_numbers_from_the_user_are_fine(checker: AnswerChecker) -> None:
    request = _request({"order_id": 7781, "amount": 1}, {"amount": 1},
                       user="Order 7781 broke; my ticket is 55512.")
    assert _kinds(checker, "Refunded $1. Ticket 55512 is now closed.", request) == []


def test_small_plain_counts_are_ignored(checker: AnswerChecker) -> None:
    assert _kinds(checker, "Refunded $1. It arrives in 5-7 business days.",
                  REFUNDED_ONE) == []


# --------------------------------------------------------------------------- #
# Action claims
# --------------------------------------------------------------------------- #


def test_claimed_action_that_never_ran(checker: AnswerChecker) -> None:
    report = checker.check("Your account has been deleted.", REFUNDED_ONE)
    (issue,) = report.issues
    assert (issue.kind, issue.tool) == (IssueKind.ACTION_NOT_TAKEN, "delete_user")


def test_claimed_success_of_a_failed_action(checker: AnswerChecker) -> None:
    assert _kinds(checker, "Done, your order has been refunded.",
                  REFUND_FAILED) == ["action_failed"]


@pytest.mark.parametrize("answer", [
    "Sorry, the order could not be refunded.",
    "The order was not refunded because the amount is too high.",
    "Nothing was refunded.",
    "Once approved, it will be refunded.",
    "I wasn't able to get it refunded.",
])
def test_negated_or_future_claims_are_not_claims(checker: AnswerChecker, answer: str) -> None:
    assert _kinds(checker, answer, REFUND_FAILED) == []


def test_read_only_tools_are_not_actions(checker: AnswerChecker) -> None:
    # "ordered" matches lookup_order's name, but a lookup is not an action.
    request = _request(tools=("lookup_order",))
    assert _kinds(checker, "The wireless mouse you ordered was delivered.", request) == []


# --------------------------------------------------------------------------- #
# Skips and robustness
# --------------------------------------------------------------------------- #


def test_skipped_without_tool_results(checker: AnswerChecker) -> None:
    request = ChatCompletionRequest(model="default",
                                    messages=[{"role": "user", "content": "hi"}])
    report = checker.check("Refunded $49.99.", request)
    assert (report.verdict, report.reason) == (CheckVerdict.SKIPPED, "no_tool_results")
    assert not checker.applies_to(request)


def test_skipped_for_empty_answer(checker: AnswerChecker) -> None:
    assert checker.check("   ", REFUNDED_ONE).reason == "empty_answer"


def test_disabled_checker_skips() -> None:
    off = AnswerChecker(AnswerCheckConfig(enabled=False))
    assert off.check("Refunded $49.99.", REFUNDED_ONE).reason == "disabled"
    assert not off.applies_to(REFUNDED_ONE)


def test_plain_text_tool_results_are_used() -> None:
    request = _request({"order_id": 7781, "amount": 1}, "Refund R-7781 done: 1.00 USD")
    checker = AnswerChecker(AnswerCheckConfig())
    assert _kinds(checker, "Refunded $1.", request) == []
    assert _kinds(checker, "Refunded $49.99.", request) == ["number_mismatch"]


def test_text_error_result_counts_as_failure() -> None:
    request = _request({"order_id": 7781, "amount": 1}, "Error: payment gateway timeout")
    evidence = collect_evidence(request)
    assert [e.failed for e in evidence] == [False, True]


def test_deeply_nested_results_do_not_crash(checker: AnswerChecker) -> None:
    # Sent as raw text, as a hostile tool would: too deep for json.loads to decode.
    hostile = '{"x":' * 5000 + "1" + "}" * 5000
    request = _request({"order_id": 7781, "amount": 1}, hostile)
    report = checker.check("Refunded $1.", request)
    assert report.verdict is CheckVerdict.SUPPORTED  # falls back to the raw text


def test_report_never_contains_values(checker: AnswerChecker) -> None:
    data = checker.check("I have refunded $49.99 for order 7781.", REFUNDED_ONE).as_dict()
    text = json.dumps(data)
    assert "49.99" not in text and "7781" not in text
    assert data["issues"][0]["span"] == [16, 22]


def test_warning_lists_each_detail_once(checker: AnswerChecker) -> None:
    report = checker.check("Refunded $49.99, I repeat, refunded $49.99.", REFUNDED_ONE)
    assert len(report.issues) == 2
    assert report.warning().count("differs from what issue_refund returned") == 1


def test_window_limits_binding() -> None:
    narrow = AnswerChecker(AnswerCheckConfig(window_words=1))
    # "$49.99" is 3 words from "refunded": unbound, and 49.99 is in the order data.
    assert _kinds(narrow, "Refunded, as you asked, $49.99.", REFUNDED_ONE) == []
