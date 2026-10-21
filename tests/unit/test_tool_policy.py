"""Unit tests for agent guardrails: schema subset, argument limits, policy engine, config."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from conftest import make_config
from groundwire.agents.policy import BlockReason, ToolPolicy, _lookup, check_constraint
from groundwire.agents.schema import validate
from groundwire.core.config import ArgumentConstraint, ToolPolicyConfig
from groundwire.core.errors import ConfigurationError
from groundwire.domain.models import ChatCompletionRequest, FunctionCall, ToolCall

REFUND_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["order_id", "amount"], "additionalProperties": False,
    "properties": {"order_id": {"type": "integer"}, "amount": {"type": "number", "minimum": 0},
                   "reason": {"type": "string", "enum": ["damaged", "late"]}}}


def _tool(name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "parameters": schema or {"type": "object", "properties": {}}}}


TOOLS = [_tool("issue_refund", REFUND_SCHEMA), _tool("send_email"), _tool("delete_user"),
         _tool("lookup_order")]


def _call(name: str = "issue_refund", args: Any = None, call_id: str = "c1") -> ToolCall:
    arguments = args if isinstance(args, str) else json.dumps(
        args if args is not None else {"order_id": 7781, "amount": 49.99})
    return ToolCall(id=call_id, function=FunctionCall(name=name, arguments=arguments))


def _request(rounds: int = 0) -> ChatCompletionRequest:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "help"}]
    for n in range(rounds):
        messages += [{"role": "assistant", "content": None,
                      "tool_calls": [_call("lookup_order", {}, f"r{n}").model_dump()]},
                     {"role": "tool", "tool_call_id": f"r{n}", "content": "ok"}]
    return ChatCompletionRequest(messages=messages, tools=TOOLS)


def _policy(**config: Any) -> ToolPolicy:
    return ToolPolicy(ToolPolicyConfig.model_validate(config))


def _decide(policy: ToolPolicy, call: ToolCall, *, rounds: int = 0,
            injection: bool = False) -> Any:
    (decision,) = policy.evaluate([call], _request(rounds), injection_suspected=injection).decisions
    return decision


# --------------------------------------------------------------------------- #
# JSON Schema subset
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("value", "problem"), [
    ({"order_id": 7781, "amount": 49.99}, None),
    ({"order_id": 7781.0, "amount": 1}, None),  # 7781.0 is an integer in JSON Schema
    ({"order_id": "7781", "amount": 1}, "order_id should be integer"),
    ({"order_id": True, "amount": 1}, "order_id should be integer"),  # bool is not a number
    ({"order_id": 1}, "missing required argument 'amount'"),
    ({"order_id": 1, "amount": -5}, "amount is below the minimum 0"),
    ({"order_id": 1, "amount": 1, "reason": "bored"}, "reason is not one of the allowed values"),
    ({"order_id": 1, "amount": 1, "admin": True}, "unexpected argument 'admin'"),
    ([1, 2], "arguments should be object"),
])
def test_schema_validation(value: Any, problem: str | None) -> None:
    errors = validate(value, REFUND_SCHEMA)
    assert (errors[0] if errors else None) == problem


def test_schema_nested_arrays_and_lengths() -> None:
    schema = {"type": "object", "properties": {
        "to": {"type": "array", "minItems": 1, "maxItems": 2,
               "items": {"type": "string", "minLength": 3, "maxLength": 20}},
        "meta": {"type": "object", "required": ["id"], "properties": {"id": {"const": 1}}},
        "n": {"type": ["integer", "null"], "maximum": 10}}}
    assert validate({"to": ["abc"], "meta": {"id": 1}, "n": None}, schema) == []
    assert validate({"to": []}, schema) == ["to has fewer than 1 items"]
    assert validate({"to": ["a", "b", "c"]}, schema)[0] == "to has more than 2 items"
    assert validate({"to": ["ab"]}, schema) == ["to[0] is shorter than 3 characters"]
    assert validate({"to": ["x" * 21]}, schema) == ["to[0] is longer than 20 characters"]
    assert validate({"meta": {}}, schema) == ["missing required argument 'meta.id'"]
    assert validate({"meta": {"id": 2}}, schema) == ["meta.id is not the required constant"]
    assert validate({"n": 11}, schema) == ["n is above the maximum 10"]


def test_schema_unknown_keywords_and_non_dict_schemas_pass() -> None:
    assert validate({"x": "y"}, {"type": "object", "oneOf": [{}], "$ref": "#/x"}) == []
    assert validate("anything", None) == []
    assert validate(1, {"type": "made-up-type"}) == []


def test_schema_errors_are_capped_and_value_free() -> None:
    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    errors = validate({f"secret{i}": "rahul@example.com" for i in range(20)}, schema)
    assert len(errors) == 5 and not any("example.com" in e for e in errors)


# --------------------------------------------------------------------------- #
# Argument limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("value", "constraint", "problem"), [
    (100000, {"max": 5000}, "'amount' is above the allowed maximum 5000"),
    (-1, {"min": 0}, "'amount' is below the allowed minimum 0"),
    (5000, {"max": 5000}, None),
    ("100", {"max": 5000}, "'amount' must be a number"),
    (True, {"max": 5000}, "'amount' must be a number"),
    ("refund", {"enum": ["refund", "credit"]}, None),
    ("cash", {"enum": ["refund", "credit"]}, "'amount' is not one of the allowed values"),
    ("a@ourco.com", {"pattern": r"[^@]+@ourco\.com"}, None),
    ("a@ourco.com.evil.io", {"pattern": r"[^@]+@ourco\.com"},
     "'amount' does not match the allowed pattern"),  # fullmatch, not search
    (42, {"pattern": ".*"}, "'amount' must be text"),
    ("x" * 11, {"max_length": 10}, "'amount' is longer than 10 characters"),
])
def test_check_constraint(value: Any, constraint: dict[str, Any], problem: str | None) -> None:
    assert check_constraint("amount", value, ArgumentConstraint(**constraint)) == problem


def test_missing_arguments_and_dotted_paths() -> None:
    from groundwire.agents.policy import _MISSING

    assert check_constraint("x", _MISSING, ArgumentConstraint(max=1)) is None
    assert check_constraint("x", _MISSING, ArgumentConstraint(required=True)) == \
        "'x' is required"
    assert _lookup({"to": {"email": "a@b.c"}}, "to.email") == "a@b.c"
    assert _lookup({"to": "flat"}, "to.email") is _MISSING


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def test_allowed_call_passes() -> None:
    decision = _decide(_policy(), _call())
    assert decision.allowed and decision.reason is None


@pytest.mark.parametrize(("config", "kwargs", "reason"), [
    ({}, {"injection": True}, BlockReason.INJECTION_LOCKDOWN),
    ({"max_tool_rounds": 3}, {"rounds": 3}, BlockReason.TOO_MANY_ROUNDS),
    ({"rules": [{"tools": ["delete_*"], "action": "block"}]}, {}, None),
])
def test_global_checks(config: dict[str, Any], kwargs: dict[str, Any],
                       reason: BlockReason | None) -> None:
    decision = _decide(_policy(**config), _call(), **kwargs)
    assert decision.allowed is (reason is None)
    assert decision.reason == reason


def test_lockdown_and_loop_limit_can_be_relaxed() -> None:
    assert _decide(_policy(injection_lockdown=False), _call(), injection=True).allowed
    assert _decide(_policy(max_tool_rounds=3), _call(), rounds=2).allowed


def test_unknown_tool_is_blocked_unless_allowed() -> None:
    hallucinated = _call("transfer_funds", {})
    assert _decide(_policy(), hallucinated).reason is BlockReason.UNKNOWN_TOOL
    assert _decide(_policy(reject_unknown_tools=False), hallucinated).allowed


def test_first_matching_rule_wins() -> None:
    policy = _policy(rules=[
        {"tools": ["delete_user"], "action": "allow", "name": "admins-may-delete"},
        {"tools": ["delete_*"], "action": "block", "name": "no-deletes"},
    ])
    assert _decide(policy, _call("delete_user", {})).rule == "admins-may-delete"
    blocked = _decide(_policy(rules=[{"tools": ["delete_*"], "action": "block"}]),
                      _call("delete_user", {}))
    assert (blocked.reason, blocked.rule) == (BlockReason.RULE, "delete_*")


def test_default_action_block_makes_an_allowlist() -> None:
    policy = _policy(default_action="block", rules=[{"tools": ["lookup_*"]}])
    assert _decide(policy, _call("lookup_order", {})).allowed
    assert _decide(policy, _call("send_email", {})).reason is BlockReason.RULE


@pytest.mark.parametrize(("args", "detail"), [
    ("not json", "arguments are not a JSON object"),
    ('{"order_id": "abc", "amount": 1}', "order_id should be integer"),
])
def test_invalid_arguments(args: str, detail: str) -> None:
    decision = _decide(_policy(), _call(args=args))
    assert (decision.reason, decision.detail) == (BlockReason.INVALID_ARGUMENTS, detail)


def test_schema_check_can_be_disabled() -> None:
    assert _decide(_policy(validate_arguments=False), _call(args='{"order_id": "abc"}')).allowed


def test_argument_limits_from_the_matching_rule() -> None:
    policy = _policy(rules=[{"tools": ["issue_refund"], "name": "refund-cap",
                             "arguments": {"amount": {"max": 5000}}}])
    assert _decide(policy, _call(args={"order_id": 1, "amount": 4999})).allowed
    decision = _decide(policy, _call(args={"order_id": 1, "amount": 100000}))
    assert decision.reason is BlockReason.ARGUMENT_LIMIT and decision.rule == "refund-cap"
    assert decision.detail == "'amount' is above the allowed maximum 5000"


def test_checks_run_in_documented_order() -> None:
    policy = _policy(max_tool_rounds=1, rules=[{"tools": ["*"], "action": "block"}])
    unknown_and_injected = _call("nope", "not json")
    assert _decide(policy, unknown_and_injected, injection=True, rounds=5).reason is \
        BlockReason.INJECTION_LOCKDOWN
    assert _decide(policy, unknown_and_injected, rounds=5).reason is BlockReason.TOO_MANY_ROUNDS
    assert _decide(policy, unknown_and_injected).reason is BlockReason.UNKNOWN_TOOL


def test_parallel_calls_are_decided_independently() -> None:
    policy = _policy(rules=[{"tools": ["delete_*"], "action": "block"}])
    result = policy.evaluate([_call(), _call("delete_user", {}, "c2")], _request(),
                             injection_suspected=False)
    assert [c.id for c in result.allowed_calls] == ["c1"]
    assert [d.call.id for d in result.blocked] == ["c2"]
    assert result.as_dict()["blocked"] == 1


def test_decisions_and_logs_never_contain_argument_values(
        caplog: pytest.LogCaptureFixture) -> None:
    policy = _policy(rules=[{"tools": ["issue_refund"],
                             "arguments": {"amount": {"max": 5000}}}])
    caplog.set_level(logging.INFO)
    result = policy.evaluate([_call(args={"order_id": 1, "amount": 987654})], _request(),
                             injection_suspected=False)
    rendered = json.dumps(result.as_dict()) + result.explanation() + caplog.text
    assert "987654" not in rendered
    assert result.explanation().startswith("The requested action was not carried out")
    assert "issue_refund" in result.explanation()
    record = next(r for r in caplog.records if r.getMessage() == "tool call decision")
    assert record.decision == "blocked"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("policy", "message"), [
    ({"rules": [{"tools": ["x"], "arguments": {"to": {"pattern": "("}}}]}, "regular expression"),
    ({"rules": [{"tools": ["x"], "arguments": {"n": {"min": 5, "max": 1}}}]}, "greater"),
    ({"rules": [{"tools": []}]}, "tools"),
    ({"max_tool_rounds": 0}, "max_tool_rounds"),
    ({"on_violation": "ignore"}, "on_violation"),
])
def test_tool_policy_config_validation(policy: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        make_config(tool_policy=policy)


def test_shipped_configs_parse_tool_policy() -> None:
    from pathlib import Path

    from groundwire.core.config import load_gateway_config

    for name in ("config/groundwire.yaml", "config/groundwire.docker.yaml"):
        assert load_gateway_config(Path(name)).tool_policy.enabled
