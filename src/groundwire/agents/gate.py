"""Tool gate: the single entry point the gateway uses to decide tool calls.

It composes two independent parts:

* :class:`~groundwire.agents.policy.ToolPolicy`, pure rules (no I/O), which
  returns allow / block / needs-approval for each call;
* :class:`~groundwire.approvals.service.ApprovalService`, which resolves
  needs-approval verdicts against the approval store: release an approved
  call (once), hold a pending one, block a rejected one, and block everything
  when approvals are unavailable (fail closed).

Keeping the policy pure and putting I/O here means each part can be tested,
replaced or extended on its own (a new rule type touches only the policy; a
new store touches only the approvals package; the chat service never changes).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from groundwire.agents.policy import (
    BlockReason,
    PolicyResult,
    ToolDecision,
    ToolPolicy,
    ToolVerdict,
)
from groundwire.approvals.service import ApprovalService, Resolution, scope_of
from groundwire.core.config import ToolViolationHandling
from groundwire.core.errors import ToolCallBlockedError
from groundwire.domain.models import ChatCompletionRequest, ToolCall
from groundwire.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)


class ToolGate:
    """Decides every tool call a model requests, including human approval."""

    def __init__(self, policy: ToolPolicy, *, approvals: ApprovalService | None = None,
                 metrics: GatewayMetrics | None = None) -> None:
        """Create the gate.

        Args:
            policy: Tool policy rules.
            approvals: Approval service (``None``: calls that need approval are
                blocked with ``approval_unavailable``).
            metrics: Metrics sink.
        """
        self._policy = policy
        self._approvals = approvals
        self._metrics = metrics

    @property
    def enabled(self) -> bool:
        """True when the policy is enforced."""
        return self._policy.enabled

    async def enforce(self, calls: Sequence[ToolCall], request: ChatCompletionRequest, *,
                      injection_suspected: bool) -> tuple[list[ToolCall], PolicyResult | None]:
        """Decide one model turn's calls.

        Args:
            calls: Proposed calls, PII-restored (rules and approvals see real values).
            request: The client's request (offered tools, history, ``user``).
            injection_suspected: The safety guard flagged a possible injection.

        Returns:
            ``(calls_to_release, result)``; ``result`` is ``None`` when the
            policy is off or there were no calls.

        Raises:
            ToolCallBlockedError: A call was blocked and ``on_violation`` is
                ``error``. Calls held for approval are not violations.
        """
        if not calls or not self._policy.enabled:
            return list(calls), None
        result = self._policy.evaluate(calls, request, injection_suspected=injection_suspected)
        if result.needs_approval:
            result = PolicyResult([await self._resolve(d, request)
                                   if d.verdict is ToolVerdict.NEEDS_APPROVAL else d
                                   for d in result.decisions])
        self._record(result)
        if result.blocked and self._policy.on_violation is ToolViolationHandling.ERROR:
            raise ToolCallBlockedError(
                "the model requested a tool call that the tool policy forbids",
                details={"tools": result.as_dict()})
        return result.allowed_calls, result

    async def _resolve(self, decision: ToolDecision,
                       request: ChatCompletionRequest) -> ToolDecision:
        """Turn a needs-approval verdict into allowed, held or blocked.

        Args:
            decision: A ``NEEDS_APPROVAL`` decision.
            request: The client's request (its ``user`` is the approval scope).

        Returns:
            The final decision.
        """
        if self._approvals is None:
            return decision.resolved(ToolVerdict.BLOCKED, reason=BlockReason.APPROVAL_UNAVAILABLE,
                                     detail="approvals are not enabled on this gateway")
        arguments = decision.call.parsed_arguments() or {}
        outcome = await self._approvals.resolve(
            tool=decision.call.function.name, arguments=arguments,
            scope=scope_of(request.user), rule=decision.rule)
        approval_id = outcome.approval.id if outcome.approval else None
        if outcome.resolution is Resolution.RELEASED:
            return decision.resolved(ToolVerdict.ALLOWED, detail="approved by a reviewer",
                                     approval_id=approval_id)
        if outcome.resolution is Resolution.PENDING:
            return decision.resolved(ToolVerdict.PENDING_APPROVAL, detail=outcome.detail,
                                     approval_id=approval_id)
        reason = (BlockReason.APPROVAL_REJECTED if outcome.resolution is Resolution.REJECTED
                  else BlockReason.APPROVAL_UNAVAILABLE)
        return decision.resolved(ToolVerdict.BLOCKED, reason=reason, detail=outcome.detail,
                                 approval_id=approval_id)

    def _record(self, result: PolicyResult) -> None:
        """Count final verdicts (labels are bounded: verdict and reason only)."""
        for decision in result.decisions:
            logger.info("tool call verdict", extra=decision.as_dict())
            if self._metrics is not None:
                self._metrics.tool_calls.labels(
                    decision=decision.verdict.value,
                    reason=decision.reason.value if decision.reason else "").inc()
