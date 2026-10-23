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
4. **rules**: the first rule that matches decides allow / block / require
   approval (firewall-style: read top to bottom, first match wins). A rule
   matches when the tool name matches and all its ``when`` conditions hold,
   which enables tiers such as "refunds of 500 or more need approval";
5. **arguments are valid**: a JSON object matching the tool's own schema;
6. **argument limits** from the matching rule (``amount <= 5000``...).

A call that passes every check under a ``require_approval`` rule gets the
verdict ``NEEDS_APPROVAL``. This module stays pure (no I/O): resolving that
verdict against the approval store is the job of
:class:`~groundwire.agents.gate.ToolGate`.

Decisions never contain argument values (they often hold personal data);
they say *which* tool, *what* was decided and *why*.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_UNAVAILABLE = "approval_unavailable"


class ToolVerdict(StrEnum):
    """The outcome for one call (stable values used in metadata and metrics)."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    NEEDS_APPROVAL = "needs_approval"  # policy says a human must approve (not yet resolved)
    PENDING_APPROVAL = "pending_approval"  # held: waiting for a reviewer


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolDecision:
    """The verdict on one call.

    Attributes:
        call: The call.
        verdict: What happens to it.
        reason: Why it was blocked (``None`` unless blocked).
        detail: Human-readable explanation, without argument values.
        rule: Name of the deciding rule, if any.
        approval_id: Approval record involved (held or released calls).
    """

    call: ToolCall
    verdict: ToolVerdict
    reason: BlockReason | None = None
    detail: str | None = None
    rule: str | None = None
    approval_id: str | None = None

    @property
    def allowed(self) -> bool:
        """True if the call may proceed now."""
        return self.verdict is ToolVerdict.ALLOWED

    @classmethod
    def allow(cls, call: ToolCall, rule: str | None = None) -> ToolDecision:
        """An allowed call."""
        return cls(call, ToolVerdict.ALLOWED, rule=rule)

    @classmethod
    def block(cls, call: ToolCall, reason: BlockReason, detail: str,
              rule: str | None = None) -> ToolDecision:
        """A blocked call."""
        return cls(call, ToolVerdict.BLOCKED, reason, detail, rule)

    def resolved(self, verdict: ToolVerdict, *, reason: BlockReason | None = None,
                 detail: str | None = None, approval_id: str | None = None) -> ToolDecision:
        """Return this decision with the approval outcome applied.

        Args:
            verdict: Final verdict.
            reason: Block reason, for blocked calls.
            detail: Explanation (keeps the current one if omitted).
            approval_id: Approval record involved.

        Returns:
            The updated decision.
        """
        return replace(self, verdict=verdict, reason=reason,
                       detail=detail if detail is not None else self.detail,
                       approval_id=approval_id)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (no argument values)."""
        data: dict[str, Any] = {"tool": self.call.function.name, "id": self.call.id,
                                "decision": self.verdict.value}
        if self.reason is not None:
            data["reason"] = self.reason.value
        if self.detail is not None:
            data["detail"] = self.detail
        if self.rule is not None:
            data["rule"] = self.rule
        if self.approval_id is not None:
            data["approval_id"] = self.approval_id
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
        return [d for d in self.decisions if d.verdict is ToolVerdict.BLOCKED]

    @property
    def held(self) -> list[ToolDecision]:
        """Calls waiting for a reviewer."""
        return [d for d in self.decisions if d.verdict is ToolVerdict.PENDING_APPROVAL]

    @property
    def needs_approval(self) -> list[ToolDecision]:
        """Calls the policy wants approved and that have not been resolved yet."""
        return [d for d in self.decisions if d.verdict is ToolVerdict.NEEDS_APPROVAL]

    def as_dict(self) -> dict[str, Any]:
        """Response metadata."""
        data: dict[str, Any] = {"decisions": [d.as_dict() for d in self.decisions],
                                "blocked": len(self.blocked)}
        if self.held:
            data["pending_approval"] = len(self.held)
        return data

    def explanation(self) -> str:
        """Text for the client when no call could proceed (no argument values).

        Returns:
            One sentence per outcome: held calls (with approval ids, so the
            agent can poll them) and blocked calls (with the reason).
        """
        sentences: list[str] = []
        if self.held:
            parts = [f"{d.call.function.name} (approval {d.approval_id})" for d in self.held]
            sentences.append("The requested action needs human approval before it can run: "
                             + "; ".join(parts) + ". It has not been carried out yet. Once it "
                             "is approved, ask again and it will go through.")
        unresolved = self.blocked + self.needs_approval
        if unresolved:
            parts = [f"{d.call.function.name} ({d.detail or d.reason})" for d in unresolved]
            sentences.append("The requested action was not carried out because it was "
                             "blocked by policy: " + "; ".join(parts) + ".")
        return " ".join(sentences)


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
            return ToolDecision.block(call, BlockReason.INJECTION_LOCKDOWN,
                                      "possible prompt injection in this request")
        if rounds >= config.max_tool_rounds:
            return ToolDecision.block(call, BlockReason.TOO_MANY_ROUNDS,
                                      f"more than {config.max_tool_rounds} tool rounds")
        if config.reject_unknown_tools and name not in offered:
            return ToolDecision.block(call, BlockReason.UNKNOWN_TOOL,
                                      "tool was not offered in this request")

        arguments = call.parsed_arguments()
        rule = self._match(name, arguments)
        rule_name = (rule.name or ",".join(rule.tools)) if rule else None
        action = rule.action if rule else config.default_action
        if action is ToolAction.BLOCK:
            return ToolDecision.block(call, BlockReason.RULE, "blocked by rule", rule_name)

        if arguments is None:
            return ToolDecision.block(call, BlockReason.INVALID_ARGUMENTS,
                                      "arguments are not a JSON object", rule_name)
        if config.validate_arguments:
            problems = validate(arguments, offered.get(name))
            if problems:
                return ToolDecision.block(call, BlockReason.INVALID_ARGUMENTS,
                                          "; ".join(problems), rule_name)
        if rule is not None:
            for path, constraint in rule.arguments.items():
                problem = check_constraint(path, _lookup(arguments, path), constraint)
                if problem is not None:
                    return ToolDecision.block(call, BlockReason.ARGUMENT_LIMIT, problem,
                                              rule_name)
        if action is ToolAction.REQUIRE_APPROVAL:
            return ToolDecision(call, ToolVerdict.NEEDS_APPROVAL,
                                detail="needs human approval", rule=rule_name)
        return ToolDecision.allow(call, rule_name)

    def _match(self, name: str, arguments: dict[str, Any] | None) -> ToolRule | None:
        """First rule whose pattern matches ``name`` and whose ``when`` holds, or ``None``.

        Args:
            name: Tool name (matched case-sensitively).
            arguments: Decoded arguments (``None`` if invalid: rules with
                ``when`` conditions then cannot match, and the call is blocked
                later as invalid anyway).

        Returns:
            The deciding rule.
        """
        for rule in self._config.rules:
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in rule.tools):
                continue
            if rule.when and (arguments is None or not _conditions_hold(arguments, rule.when)):
                continue
            return rule
        return None


def _conditions_hold(arguments: dict[str, Any],
                     conditions: dict[str, ArgumentConstraint]) -> bool:
    """True if every ``when`` condition holds (a missing argument never matches).

    Args:
        arguments: Decoded arguments.
        conditions: Path -> condition.

    Returns:
        Whether the rule applies.
    """
    for path, condition in conditions.items():
        value = _lookup(arguments, path)
        if value is _MISSING or check_constraint(path, value, condition) is not None:
            return False
    return True


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
