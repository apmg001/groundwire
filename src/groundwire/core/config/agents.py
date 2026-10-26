"""Agent guardrails: tool policy, human approval and the answer check.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from groundwire.core.config.base import StrictModel


class ToolAction(StrEnum):
    """What a tool rule does with a matching call."""

    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"  # hold the call until a human approves it


class ToolViolationHandling(StrEnum):
    """What the gateway does with a blocked tool call."""

    DROP = "drop"  # remove the call; other calls continue; text explains if none remain
    ERROR = "error"  # refuse the whole response (HTTP 422 tool_call_blocked)


class ArgumentConstraint(StrictModel):
    """Limits on one argument (all set limits must hold).

    Attributes:
        required: The argument must be present.
        min: Smallest allowed number.
        max: Largest allowed number.
        enum: Allowed values.
        pattern: Regular expression the whole string value must match.
        max_length: Longest allowed string.
    """

    required: bool = False
    min: float | None = None
    max: float | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    max_length: Annotated[int, Field(ge=0)] | None = None

    @field_validator("pattern")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regular expression {value!r}: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) is greater than max ({self.max})")
        return self


class ToolRule(StrictModel):
    """One ordered rule; the first rule that matches a call decides.

    A rule matches when the tool name matches one of ``tools`` *and* every
    ``when`` condition holds; otherwise evaluation falls through to the next
    rule (firewall-style). ``when`` makes amount-based tiers possible::

        - {tools: [issue_refund], action: require_approval, when: {amount: {min: 500}}}
        - {tools: [issue_refund], arguments: {amount: {max: 5000}}}

    Attributes:
        tools: Tool-name patterns (shell-style: ``*``, ``?``, ``[abc]``).
        action: Allow, block, or hold for human approval.
        when: Argument conditions that select this rule. A condition holds only
            when the argument is present and satisfies the limits.
        arguments: Argument name (dotted path for nested values) -> limits that
            must hold. Checked for allowed calls and for calls that need
            approval (a call over a hard limit is blocked, never sent to a
            reviewer).
        name: Label shown in audit logs, approvals and response metadata.
    """

    tools: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]
    action: ToolAction = ToolAction.ALLOW
    when: dict[str, ArgumentConstraint] = Field(default_factory=dict)
    arguments: dict[str, ArgumentConstraint] = Field(default_factory=dict)
    name: Annotated[str, Field(max_length=64)] | None = None


class ToolPolicyConfig(StrictModel):
    """Guardrails for tool calls requested by models (OWASP LLM06, LLM10).

    Checked in code on every tool call, before the agent receives it, so the
    model can neither forget nor be talked out of them.

    Attributes:
        enabled: Turn enforcement on.
        rules: Ordered rules; the first matching rule decides.
        default_action: Action when no rule matches.
        on_violation: Drop blocked calls (agent continues) or refuse the response.
        injection_lockdown: When the request contains a suspected prompt
            injection, block every tool call (text answers still go through).
        max_tool_rounds: Assistant tool-call turns allowed in one conversation
            before further calls are blocked (stops runaway agent loops).
        validate_arguments: Check arguments are a JSON object matching the
            tool's declared JSON Schema (types, required, enum, ranges).
        reject_unknown_tools: Block calls to tools the request did not offer.
    """

    enabled: bool = True
    rules: list[ToolRule] = Field(default_factory=list)
    default_action: ToolAction = ToolAction.ALLOW
    on_violation: ToolViolationHandling = ToolViolationHandling.DROP
    injection_lockdown: bool = True
    max_tool_rounds: Annotated[int, Field(ge=1, le=1000)] = 25
    validate_arguments: bool = True
    reject_unknown_tools: bool = True


class ApprovalStoreType(StrEnum):
    """Where pending approvals are kept."""

    MEMORY = "memory"  # single replica; lost on restart (agents simply ask again)


class ApprovalConfig(StrictModel):
    """Human-in-the-loop approval for tool calls (rules with ``action: require_approval``).

    An approval is permission for one *exact* call (same tool, same arguments,
    same ``user``), used at most once.

    Attributes:
        enabled: Allow rules to hold calls for approval. Rules that require
            approval are a configuration error when this is off.
        store: Storage backend.
        ttl_s: How long a request waits for a decision before it expires.
        grant_ttl_s: How long an approved call stays usable before it expires.
        retention_s: How long decided records stay visible for audit before
            they are purged from the store.
        max_records: Capacity of the store (all statuses).
        max_pending_per_scope: Pending requests one ``user`` may have open at
            once, so a looping agent cannot flood reviewers.
        reviewer_token_env: Environment variable holding the reviewer API token.
            When unset, the reviewer endpoints are disabled (fail closed).
    """

    enabled: bool = True
    store: ApprovalStoreType = ApprovalStoreType.MEMORY
    ttl_s: Annotated[float, Field(ge=10, le=7 * 24 * 3600)] = 900.0
    grant_ttl_s: Annotated[float, Field(ge=10, le=7 * 24 * 3600)] = 600.0
    retention_s: Annotated[float, Field(ge=60, le=90 * 24 * 3600)] = 86_400.0
    max_records: Annotated[int, Field(ge=10, le=1_000_000)] = 10_000
    max_pending_per_scope: Annotated[int, Field(ge=1, le=10_000)] = 20
    reviewer_token_env: Annotated[str, Field(pattern=r"^[A-Z_][A-Z0-9_]{0,127}$")] = (
        "GROUNDWIRE_APPROVER_TOKEN")  # noqa: S105 - name of an env var, not a secret


class AnswerCheckAction(StrEnum):
    """What the gateway does when an agent's answer contradicts its tool results."""

    FLAG = "flag"  # deliver unchanged; report issues in ``groundwire.answer_check``
    ANNOTATE = "annotate"  # append a visible warning to the answer
    BLOCK = "block"  # refuse the answer (HTTP 422 answer_check_failed)


class AnswerCheckConfig(StrictModel):
    """Fact-check an agent's final answer against the tool results it was given.

    Catches the agent that runs the right tools and then reports the wrong
    outcome ("refunded $49.99" when the refund tool returned ``amount: 1``),
    and claims of actions that never ran or failed. Deterministic: no model
    call, so it adds microseconds, not seconds.

    Attributes:
        enabled: Run the check on final answers in conversations with tool results.
        action: Flag, annotate or block an answer with issues.
        window_words: How far (in words) a number may be from a tool's words
            (its name or result field names) to be checked against that tool.
        min_unbound_value: Numbers not near any tool's words are only reported
            when they are at least this large, or have a currency sign,
            percent sign or decimal point (small counts like "2 items" are noise).
    """

    enabled: bool = True
    action: AnswerCheckAction = AnswerCheckAction.FLAG
    window_words: Annotated[int, Field(ge=1, le=50)] = 6
    min_unbound_value: Annotated[float, Field(ge=0)] = 100.0
