"""The tool framework, the toolkit, and the agent loop with self-correction."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from conftest import USER, FrozenClock, ScriptedLLM, answer, tool_call
from money_agent.adapters.inmemory import (
    InMemoryMemoryRepository,
    InMemoryStore,
    InMemoryTransactionRepository,
    InMemoryUserRepository,
)
from money_agent.agent.runner import (
    CORRECTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    WITHHELD_ANSWER,
    AgentRunner,
    VerificationStatus,
    verification_of,
)
from money_agent.agent.toolkit import default_tools
from money_agent.agent.tools import (
    MAX_RESULT_CHARS,
    Tool,
    ToolArgs,
    ToolContext,
    ToolRegistry,
    encode,
)
from money_agent.core.config import UnverifiedAnswers
from money_agent.core.errors import AgentError, NotFoundError
from money_agent.domain.ports import LLMReply, ToolCallRequest
from money_agent.memory.embeddings import HashingEmbedder
from money_agent.memory.service import MemoryService
from money_agent.services.finance import FinanceService
from money_agent.services.statements import StatementService
from money_agent.statements.categoriser import RuleCategoriser
from money_agent.statements.parser import StatementParser

CONTEXT = ToolContext(user_id=USER)


# --------------------------------------------------------------------------- #
# Framework
# --------------------------------------------------------------------------- #


class EchoArgs(ToolArgs):
    text: str
    times: int = 1


class EchoTool(Tool[EchoArgs]):
    def __init__(self, behaviour: str = "ok") -> None:
        super().__init__("echo", "Repeats text.", EchoArgs, timeout_s=0.05)
        self.behaviour = behaviour

    async def run(self, context: ToolContext, args: EchoArgs) -> dict[str, Any]:
        if self.behaviour == "slow":
            await asyncio.sleep(1)
        if self.behaviour == "crash":
            raise RuntimeError("secret internal detail")
        if self.behaviour == "expected":
            raise NotFoundError("internal", public_message="No such thing.")
        return {"echo": args.text * args.times}


def _call(arguments: str, name: str = "echo") -> ToolCallRequest:
    return ToolCallRequest("c1", name, arguments)


async def test_valid_call_runs() -> None:
    outcome = await EchoTool().invoke(CONTEXT, _call('{"text": "hi", "times": 2}'))
    assert outcome.ok and outcome.result == {"echo": "hihi"}
    assert outcome.message() == {"role": "tool", "tool_call_id": "c1",
                                 "content": '{"echo":"hihi"}'}


@pytest.mark.parametrize(("arguments", "problem"), [
    ("not json", "not valid JSON"), ('{"times": 2}', "text"),
    ('{"text": "a", "extra": 1}', "extra"), ('{"text": "a", "times": "x"}', "times")])
async def test_bad_arguments_come_back_as_fixable_errors(arguments: str, problem: str) -> None:
    outcome = await EchoTool().invoke(CONTEXT, _call(arguments))
    assert not outcome.ok and outcome.result["error"] == "invalid_arguments"
    assert problem in " ".join(outcome.result["problems"])


@pytest.mark.parametrize(("behaviour", "code"), [
    ("slow", "timeout"), ("crash", "tool_failed"), ("expected", "not_found")])
async def test_failures_are_results_without_internal_detail(behaviour: str, code: str) -> None:
    outcome = await EchoTool(behaviour).invoke(CONTEXT, _call('{"text": "a"}'))
    assert not outcome.ok and outcome.result["error"] == code
    assert "secret internal detail" not in json.dumps(outcome.result)


async def test_registry() -> None:
    registry = ToolRegistry.of([EchoTool()])
    unknown = await registry.invoke(CONTEXT, _call("{}", name="transfer_money"))
    assert not unknown.ok and unknown.result["error"] == "unknown_tool"
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry.of([EchoTool(), EchoTool()])
    [definition] = registry.definitions()
    assert definition["function"]["name"] == "echo"
    assert "title" not in json.dumps(definition)  # schema noise stripped
    for bad in ("Bad Name!", "UPPER", "x" * 65):
        with pytest.raises(ValueError, match="invalid tool name"):
            Tool.__init__(EchoTool(), bad, "x", EchoArgs)


def test_results_are_size_capped() -> None:
    assert json.loads(encode({"x": "y" * MAX_RESULT_CHARS}))["error"] == "result_too_large"


# --------------------------------------------------------------------------- #
# Toolkit on real services (in-memory)
# --------------------------------------------------------------------------- #


async def _registry(sample: bytes | None) -> ToolRegistry:
    store = InMemoryStore()
    transactions = InMemoryTransactionRepository(store)
    clock = FrozenClock()
    if sample is not None:
        await StatementService(InMemoryUserRepository(store), transactions, StatementParser(),
                               RuleCategoriser()).import_statement(USER, sample)
    finance = FinanceService(transactions, clock)
    memory = MemoryService(InMemoryMemoryRepository(store), HashingEmbedder(), clock)
    return ToolRegistry.of(default_tools(finance, memory))


async def _run(registry: ToolRegistry, name: str, **args: Any) -> dict[str, Any]:
    outcome = await registry.invoke(CONTEXT, ToolCallRequest("c", name, json.dumps(args)))
    return outcome.result


async def test_spending_summary_tool_returns_exact_strings(sample_bytes: bytes) -> None:
    registry = await _registry(sample_bytes)
    result = await _run(registry, "get_spending_summary")
    assert result["period"] == "2026-09"  # latest month with data, not "today's" October
    assert (result["income"], result["spending"], result["saved"]) == (
        "85212.35", "38185.00", "47027.35")
    assert result["savings_rate_percent"] == "55.2"
    assert result["spending_by_category"][0] == {
        "category": "rent", "label": "Rent", "total": "22000.00", "count": 1,
        "share_percent": "57.6"}


async def test_read_tools_on_the_sample(sample_bytes: bytes) -> None:
    registry = await _registry(sample_bytes)
    listed = await _run(registry, "list_transactions", category="food_dining", limit=3)
    assert listed["returned"] == 3 and listed["more_available"] is True
    assert {t["category"] for t in listed["transactions"]} == {"food_dining"}
    trend = await _run(registry, "get_monthly_trend", months=3)
    assert [m["month"] for m in trend["months"]] == ["2026-07", "2026-08", "2026-09"]
    recurring = await _run(registry, "find_recurring_payments")
    assert {"netflix", "zerodha coin sip"} <= {r["payee"] for r in recurring["recurring"]}
    plan = await _run(registry, "plan_savings_goal", target_amount=120000, months=6)
    assert plan["required_monthly"] == "20000.00" and plan["on_track"] is True
    bad = await _run(registry, "list_transactions", start_date="2026-09-10",
                     end_date="2026-09-01")
    assert bad["error"] == "invalid_arguments"


async def test_tools_report_missing_data_plainly() -> None:
    registry = await _registry(None)
    result = await _run(registry, "get_spending_summary")
    assert result["error"] == "no_transactions" and "Upload" in result["message"]


async def test_memory_tools_round_trip() -> None:
    registry = await _registry(None)
    created = await _run(registry, "remember", kind="goal",
                         content="Saving ₹1,20,000 for a bike by June 2027")
    assert created["status"] == "created"
    updated = await _run(registry, "remember", kind="goal", content="Bike goal moved to July",
                         replaces_memory_id=created["memory_id"])
    assert updated["status"] == "updated" and updated["memory_id"] == created["memory_id"]
    listed = await _run(registry, "list_memories")
    assert [m["content"] for m in listed["memories"]] == ["Bike goal moved to July"]
    assert (await _run(registry, "delete_memory",
                       memory_id=created["memory_id"]))["status"] == "deleted"
    again = await _run(registry, "delete_memory", memory_id=created["memory_id"])
    assert again["error"] == "not_found"
    bad = await _run(registry, "remember", kind="secret", content="xyz")
    assert bad["error"] == "invalid_arguments"
    assert (await _run(registry, "delete_memory", memory_id="nope"))["error"] == \
        "invalid_arguments"


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #


def _runner(llm: ScriptedLLM, **kwargs: Any) -> AgentRunner:
    return AgentRunner(llm, ToolRegistry.of([EchoTool()]), **kwargs)


START = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]


async def test_tool_then_verified_answer() -> None:
    llm = ScriptedLLM(tool_call("echo", {"text": "a"}, "c1"), answer("Echoed a."))
    run = await _runner(llm).run(CONTEXT, START)
    assert run.answer == "Echoed a." and run.model_calls == 2
    assert run.verification.status is VerificationStatus.VERIFIED
    assert [t.name for t in run.tools] == ["echo"]
    second = llm.calls[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": '{"echo":"a"}'}
    assert llm.calls[0]["user"] == USER


async def test_a_failed_check_is_corrected_before_the_user_sees_it() -> None:
    llm = ScriptedLLM(tool_call("echo", {"text": "a"}),
                      answer("You spent ₹999.", "mismatch", ["number_mismatch"]),
                      answer("You spent ₹100."))
    run = await _runner(llm).run(CONTEXT, START)
    assert run.answer == "You spent ₹100."
    assert run.verification.status is VerificationStatus.VERIFIED
    assert run.verification.corrected
    retry = llm.calls[2]["messages"]
    assert retry[-2] == {"role": "assistant", "content": "You spent ₹999."}
    assert retry[-1]["content"] == CORRECTION_PROMPT.format(problems="number_mismatch")


@pytest.mark.parametrize(("policy", "shown"), [
    (UnverifiedAnswers.WARN, "Still ₹999."), (UnverifiedAnswers.WITHHOLD, WITHHELD_ANSWER)])
async def test_still_unverified_answers_follow_the_policy(policy: UnverifiedAnswers,
                                                          shown: str) -> None:
    llm = ScriptedLLM(answer("₹999.", "mismatch", ["number_mismatch"]),
                      answer("Still ₹999.", "mismatch", ["number_mismatch", "action_failed"]))
    run = await _runner(llm, unverified=policy).run(CONTEXT, START)
    assert run.answer == shown
    assert run.verification.status is VerificationStatus.FAILED
    assert run.verification.problems == ("action_failed", "number_mismatch")
    assert not run.verification.corrected


async def test_step_budget_forces_a_final_answer_without_tools() -> None:
    llm = ScriptedLLM(tool_call("echo", {"text": "a"}), tool_call("echo", {"text": "b"}),
                      answer("Here is what I found."))
    run = await _runner(llm, max_steps=3).run(CONTEXT, START)
    assert run.answer == "Here is what I found." and run.model_calls == 3
    last = llm.calls[2]
    assert last["tools"] == [] and last["messages"][-1]["content"] == FINAL_ANSWER_PROMPT


async def test_no_answer_within_budget_is_an_agent_error() -> None:
    llm = ScriptedLLM(tool_call("echo", {"text": "a"}), tool_call("echo", {"text": "b"}))
    with pytest.raises(AgentError):
        await _runner(llm, max_steps=2).run(CONTEXT, START)
    with pytest.raises(AgentError):
        await _runner(ScriptedLLM(LLMReply(content="  "))).run(CONTEXT, START)
    with pytest.raises(ValueError):
        _runner(ScriptedLLM(), max_steps=0)


def test_verification_reading() -> None:
    assert verification_of(answer("x", None))[0] is VerificationStatus.UNAVAILABLE
    assert verification_of(answer("x", "skipped"))[0] is VerificationStatus.NOT_NEEDED
    assert verification_of(answer("x", "mismatch", ["number_mismatch", "number_mismatch"])) == (
        VerificationStatus.FAILED, ("number_mismatch",))
