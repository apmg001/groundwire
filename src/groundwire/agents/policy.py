"""Tool-call policy: decide, in code, whether each tool call a model requests may proceed.

The model only *proposes* actions; this module *decides*. Rules in a system
prompt are suggestions a model can forget or be talked out of; checks in the
gateway cannot. That is the core defence against OWASP LLM06 (Excessive
Agency): least privilege for what an agent may do, enforced outside the model.

Checks for every call, in order (the first failing check blocks the call):

1. **injection lockdown**: the request contained a suspected prompt injection
   (user message, document or tool result), so no tool may run at all;
2. **loop limit**: the conversation already has ``max_tool_rounds`` tool-call
   turns, so the agent is probably stuck (OWASP LLM10, unbounded consumption);
3. **unknown tool**: the model asked for a tool the request never offered
   (models do invent tool names);
4. **rules**: the first rule whose name pattern matches decides allow/block
   (firewall-style: read top to bottom, first match wins);
5. **arguments are valid**: a JSON object matching the tool's own schema;
6. **argument limits** from the matching rule (``amount <= 5000``...).

Decisions never contain argument values (they often hold personal data);
they say *which* tool, *what* was decided and *why*.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from groundwire.agents.schema import validate
from groundwire.core.config import (
    ArgumentConstraint,
    ToolAction,
    ToolPolicyConfig,
    ToolRule,
    ToolViolationHandling,
)
from groundwire.domain.models import ChatCompletionRequest, ToolCall

logger = logging.getLogger(__name__)

_MISSING = object()


class BlockReason(StrEnum):
    """Why a tool call was blocked (stable codes for clients and metrics)."""

    INJECTION_LOCKDOWN = "injection_lockdown"
    TOO_MANY_ROUNDS = "too_many_rounds"
    UNKNOWN_TOOL = "unknown_tool"
    RULE = "rule"
    INVALID_ARGUMENTS = "invalid_arguments"
    ARGUMENT_LIMIT = "argument_limit"


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolDecision:
    """The verdict on one tool call.

    Attributes:
        call: The call.
        allowed: Whether it may proceed.
        reason: Why it was blocked (``None`` when allowed).
        detail: Human-readable explanation, without argument values.
        rule: Name of the deciding rule, if any.
    """

    call: ToolCall
    allowed: bool
    reason: BlockReason | None = None
    detail: str | None = None
    rule: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (no argument values)."""
        data: dict[str, Any] = {"tool": self.call.function.name, "id": self.call.id,
                                "decision": "allowed" if self.allowed else "blocked"}
        if self.reason is not None:
            data["reason"] = self.reason.value
        if self.detail is not None:
            data["detail"] = self.detail
        if self.rule is not None:
            data["rule"] = self.rule
        return data


@dataclass(slots=True)
class PolicyResult:
    """Verdicts for all calls in one model turn.

    Attributes:
        decisions: One per call, in the model's order.
    """

    decisions: list[ToolDecision] = field(default_factory=list)

    @property
    def allowed_calls(self) -> list[ToolCall]:
        """Calls that may proceed."""
        return [d.call for d in self.decisions if d.allowed]

    @property
    def blocked(self) -> list[ToolDecision]:
        """Blocked verdicts."""
        return [d for d in self.decisions if not d.allowed]

    def as_dict(self) -> dict[str, Any]:
        """Response metadata."""
        return {"decisions": [d.as_dict() for d in self.decisions],
                "blocked": len(self.blocked)}

    def explanation(self) -> str:
        """Text for the client when every call was blocked (no argument values).

        Returns:
            A short sentence naming each blocked tool and why.
        """
        parts = [f"{d.call.function.name} ({d.detail or d.reason})" for d in self.blocked]
        return ("The requested action was not carried out because it was blocked by "
                "policy: " + "; ".join(parts) + ".")


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class ToolPolicy:
    """Evaluates tool calls against the configured policy."""

    def __init__(self, config: ToolPolicyConfig) -> None:
        """Create the policy.

        Args:
            config: Tool policy settings.
        """
        self._config = config

    @property
    def enabled(self) -> bool:
        """True when enforcement is on."""
        return self._config.enabled

    @property
    def on_violation(self) -> ToolViolationHandling:
        """How blocked calls are handled."""
        return self._config.on_violation

    def evaluate(self, calls: Sequence[ToolCall], request: ChatCompletionRequest, *,
                 injection_suspected: bool) -> PolicyResult:
        """Decide every call in one model turn.

        Args:
            calls: Tool calls the model requested (PII already restored, so
                rules see real values).
            request: The client's request (offered tools, conversation history).
            injection_suspected: The safety guard found a possible injection.

        Returns:
            The verdicts.
        """
        offered = {t.function.name: t.function.parameters for t in request.tools or []}
        rounds = sum(1 for m in request.messages if m.role == "assistant" and m.tool_calls)
        result = PolicyResult([self._decide(call, offered, rounds, injection_suspected)
                               for call in calls])
        for decision in result.decisions:
            logger.info("tool call decision", extra=decision.as_dict())
        return result

    def _decide(self, call: ToolCall, offered: dict[str, Any], rounds: int,
                injection_suspected: bool) -> ToolDecision:
        """Run every check on one call; the first failure blocks it."""
        name = call.function.name
        config = self._config
        if config.injection_lockdown and injection_suspected:
            return ToolDecision(call, False, BlockReason.INJECTION_LOCKDOWN,
                                "possible prompt injection in this request")
        if rounds >= config.max_tool_rounds:
            return ToolDecision(call, False, BlockReason.TOO_MANY_ROUNDS,
                                f"more than {config.max_tool_rounds} tool rounds")
        if config.reject_unknown_tools and name not in offered:
            return ToolDecision(call, False, BlockReason.UNKNOWN_TOOL,
                                "tool was not offered in this request")

        rule = self._match(name)
        rule_name = (rule.name or ",".join(rule.tools)) if rule else None
        action = rule.action if rule else config.default_action
        if action is ToolAction.BLOCK:
            return ToolDecision(call, False, BlockReason.RULE, "blocked by rule", rule_name)

        arguments = call.parsed_arguments()
        if arguments is None:
            return ToolDecision(call, False, BlockReason.INVALID_ARGUMENTS,
                                "arguments are not a JSON object", rule_name)
        if config.validate_arguments:
            problems = validate(arguments, offered.get(name))
            if problems:
                return ToolDecision(call, False, BlockReason.INVALID_ARGUMENTS,
                                    "; ".join(problems), rule_name)
        if rule is not None:
            for path, constraint in rule.arguments.items():
                problem = check_constraint(path, _lookup(arguments, path), constraint)
                if problem is not None:
                    return ToolDecision(call, False, BlockReason.ARGUMENT_LIMIT, problem,
                                        rule_name)
        return ToolDecision(call, True, rule=rule_name)

    def _match(self, name: str) -> ToolRule | None:
        """First rule whose pattern matches ``name`` (case-sensitive), or ``None``."""
        for rule in self._config.rules:
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in rule.tools):
                return rule
        return None


# --------------------------------------------------------------------------- #
# Argument limits
# --------------------------------------------------------------------------- #


def _lookup(arguments: dict[str, Any], path: str) -> Any:
    """Read a dotted path (``"recipient.email"``) from the arguments.

    Args:
        arguments: Decoded arguments.
        path: Dotted path.

    Returns:
        The value, or ``_MISSING``.
    """
    value: Any = arguments
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def check_constraint(path: str, value: Any, constraint: ArgumentConstraint) -> str | None:
    """Check one argument against its limits.

    Args:
        path: Argument path, for the message.
        value: The value, or the missing sentinel.
        constraint: Limits.

    Returns:
        A problem description without the value, or ``None`` if it holds.
    """
    if value is _MISSING:
        return f"'{path}' is required" if constraint.required else None
    numeric = isinstance(value, int | float) and not isinstance(value, bool)
    if constraint.min is not None or constraint.max is not None:
        if not numeric:
            return f"'{path}' must be a number"
        if constraint.min is not None and value < constraint.min:
            return f"'{path}' is below the allowed minimum {constraint.min:g}"
        if constraint.max is not None and value > constraint.max:
            return f"'{path}' is above the allowed maximum {constraint.max:g}"
    if constraint.enum is not None and value not in constraint.enum:
        return f"'{path}' is not one of the allowed values"
    if constraint.pattern is not None or constraint.max_length is not None:
        if not isinstance(value, str):
            return f"'{path}' must be text"
        if constraint.pattern is not None and re.fullmatch(constraint.pattern, value) is None:
            return f"'{path}' does not match the allowed pattern"
        if constraint.max_length is not None and len(value) > constraint.max_length:
            return f"'{path}' is longer than {constraint.max_length} characters"
    return None
