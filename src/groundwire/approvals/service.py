"""Approval use cases: request, decide, release once, expire.

:class:`ApprovalService` owns every rule of the approval lifecycle (see
:mod:`groundwire.approvals.models`); the store only persists records and
swaps them atomically. Two kinds of callers use it:

* **the gateway** (:class:`~groundwire.agents.gate.ToolGate`) calls
  :meth:`ApprovalService.resolve` for each tool call a policy rule holds:
  the first time a pending request is created; once a reviewer approves, the
  same exact call is released (once); a rejected call stays blocked;
* **reviewers** (the HTTP API) call :meth:`approve`, :meth:`reject`,
  :meth:`get` and :meth:`list`.

Design rules:

* **Fail closed.** If the store fails or is full, :meth:`resolve` reports
  ``UNAVAILABLE`` and the gateway blocks the call. A held action never runs
  because approval infrastructure was down.
* **Exactly-once release.** An approved record moves to ``CONSUMED`` with
  compare-and-set before the call is released, so two concurrent requests can
  never both execute one approval.
* **Lazy expiry.** Deadlines are enforced whenever a record is read; there is
  no background job to fail. Closed records are purged after ``retention_s``
  on writes.
* **Audit trail without personal data.** Every lifecycle event is logged on
  the ``groundwire.audit`` logger with ids, tool, rule, scope and reviewer,
  never the arguments (they are visible to reviewers via the API only).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from groundwire.approvals.models import (
    Approval,
    ApprovalStatus,
    InvalidTransitionError,
    call_fingerprint,
    new_approval_id,
)
from groundwire.approvals.store import ApprovalStore
from groundwire.core.config import ApprovalConfig
from groundwire.core.context import get_request_id
from groundwire.core.errors import (
    ApprovalConflictError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalStoreError,
)
from groundwire.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)
audit = logging.getLogger("groundwire.audit")

Clock = Callable[[], float]
IdFactory = Callable[[], str]

ANONYMOUS_SCOPE = "anonymous"
_MAX_CAS_RETRIES = 3


class Resolution(StrEnum):
    """What the gateway should do with a call that needs approval."""

    RELEASED = "released"  # approved: let it through (the approval is now used up)
    PENDING = "pending"  # waiting for a reviewer: hold it
    REJECTED = "rejected"  # a reviewer said no: block it
    UNAVAILABLE = "unavailable"  # store failed, full, or scope limit hit: block it


@dataclass(frozen=True, slots=True)
class ApprovalOutcome:
    """Result of :meth:`ApprovalService.resolve`.

    Attributes:
        resolution: What to do with the call.
        approval: The record involved (``None`` when unavailable).
        detail: Human-readable reason, without argument values.
    """

    resolution: Resolution
    approval: Approval | None = None
    detail: str | None = None


def scope_of(user: str | None) -> str:
    """Map a request's ``user`` field to an approval scope.

    Args:
        user: OpenAI ``user`` field (put an authenticated identity here in
            multi-tenant deployments).

    Returns:
        The scope (``"anonymous"`` when absent).
    """
    return user.strip() if user and user.strip() else ANONYMOUS_SCOPE


class ApprovalService:
    """Human-in-the-loop approvals for tool calls."""

    def __init__(self, store: ApprovalStore, config: ApprovalConfig, *,
                 metrics: GatewayMetrics | None = None, clock: Clock = time.time,
                 id_factory: IdFactory = new_approval_id) -> None:
        """Create the service.

        Args:
            store: Approval storage.
            config: Approval settings.
            metrics: Metrics sink (optional).
            clock: Wall clock in UNIX seconds (injected for tests).
            id_factory: Approval id generator (injected for tests).
        """
        self._store = store
        self._config = config
        self._metrics = metrics
        self._clock = clock
        self._new_id = id_factory

    # ------------------------------------------------------------------ #
    # Gateway side
    # ------------------------------------------------------------------ #

    async def resolve(self, *, tool: str, arguments: dict[str, Any], scope: str,
                      rule: str | None) -> ApprovalOutcome:
        """Decide what happens to a call that a policy rule holds for approval.

        Never raises: every failure becomes ``UNAVAILABLE`` (fail closed).

        Args:
            tool: Tool name.
            arguments: Decoded, PII-restored arguments.
            scope: Requester scope (see :func:`scope_of`).
            rule: Rule that required approval.

        Returns:
            The outcome.
        """
        try:
            fingerprint = call_fingerprint(tool, arguments)
            for _ in range(_MAX_CAS_RETRIES):
                outcome = await self._resolve_once(tool, arguments, scope, rule, fingerprint)
                if outcome is not None:
                    return outcome
            raise ApprovalStoreError("approval changed concurrently too many times")
        except (ApprovalStoreError, ValueError) as exc:
            logger.error("approval unavailable, call blocked", extra={
                "tool": tool, "rule": rule, "error": str(exc)})
            self._count("unavailable")
            return ApprovalOutcome(Resolution.UNAVAILABLE,
                                   detail="approval service unavailable")

    async def _resolve_once(self, tool: str, arguments: dict[str, Any], scope: str,
                            rule: str | None, fingerprint: str) -> ApprovalOutcome | None:
        """One optimistic attempt; ``None`` means a race was lost and it should retry."""
        now = self._clock()
        current = await self._store.find_latest(scope, fingerprint)
        if current is not None:
            current = await self._expire_if_due(current, now)
            if current is None:
                return None
            if current.status is ApprovalStatus.APPROVED:
                consumed = current.transition(ApprovalStatus.CONSUMED, now, consumed_at=now)
                if not await self._store.compare_and_set(current, consumed):
                    return None
                self._audit("released", consumed)
                self._count("released")
                return ApprovalOutcome(Resolution.RELEASED, consumed)
            if current.status is ApprovalStatus.PENDING:
                self._count("deduplicated")
                return ApprovalOutcome(Resolution.PENDING, current,
                                       detail="waiting for a reviewer")
            if current.status is ApprovalStatus.REJECTED:
                return ApprovalOutcome(Resolution.REJECTED, current,
                                       detail="a reviewer rejected this action")
            # EXPIRED or CONSUMED: this exact call needs a fresh approval.

        open_requests = await self._store.count(ApprovalStatus.PENDING, scope=scope)
        if open_requests >= self._config.max_pending_per_scope:
            self._count("unavailable")
            return ApprovalOutcome(Resolution.UNAVAILABLE,
                                   detail="too many approval requests are already waiting")
        await self._store.purge_closed(now - self._config.retention_s)
        approval = Approval(
            id=self._new_id(), scope=scope, fingerprint=fingerprint, tool=tool,
            arguments=arguments, rule=rule, status=ApprovalStatus.PENDING,
            created_at=now, expires_at=now + self._config.ttl_s, updated_at=now,
            request_id=get_request_id())
        await self._store.add(approval)
        self._audit("requested", approval)
        self._count("requested")
        await self._refresh_gauge()
        return ApprovalOutcome(Resolution.PENDING, approval, detail="waiting for a reviewer")

    # ------------------------------------------------------------------ #
    # Reviewer side
    # ------------------------------------------------------------------ #

    async def approve(self, approval_id: str, *, reviewer: str,
                      note: str | None = None) -> Approval:
        """Approve a pending request; the exact call may then run once.

        Args:
            approval_id: Approval id.
            reviewer: Who approved (recorded in the audit trail).
            note: Optional comment.

        Returns:
            The approved record (``expires_at`` is now the deadline to use it).

        Raises:
            ApprovalNotFoundError: Unknown id.
            ApprovalExpiredError: The request expired before the decision.
            ApprovalConflictError: Already decided, or decided concurrently.
            ApprovalStoreError: The store failed.
        """
        now = self._clock()
        current = await self._pending(approval_id, now)
        approved = current.transition(ApprovalStatus.APPROVED, now, decided_at=now,
                                      reviewer=reviewer, note=note,
                                      expires_at=now + self._config.grant_ttl_s)
        await self._swap_decision(current, approved)
        self._audit("approved", approved)
        self._count("approved")
        return approved

    async def reject(self, approval_id: str, *, reviewer: str,
                     note: str | None = None) -> Approval:
        """Reject a pending request; the exact call stays blocked.

        Args:
            approval_id: Approval id.
            reviewer: Who rejected.
            note: Optional reason (shown to reviewers, not to the agent).

        Returns:
            The rejected record.

        Raises:
            ApprovalNotFoundError: Unknown id.
            ApprovalExpiredError: The request expired before the decision.
            ApprovalConflictError: Already decided, or decided concurrently.
            ApprovalStoreError: The store failed.
        """
        now = self._clock()
        current = await self._pending(approval_id, now)
        rejected = current.transition(ApprovalStatus.REJECTED, now, decided_at=now,
                                      reviewer=reviewer, note=note)
        await self._swap_decision(current, rejected)
        self._audit("rejected", rejected)
        self._count("rejected")
        return rejected

    async def get(self, approval_id: str) -> Approval:
        """Return a record, with any overdue deadline applied.

        Args:
            approval_id: Approval id.

        Returns:
            The record.

        Raises:
            ApprovalNotFoundError: Unknown id.
            ApprovalStoreError: The store failed.
        """
        current = await self._store.get(approval_id)
        if current is None:
            raise ApprovalNotFoundError("no approval with this id")
        updated = await self._expire_if_due(current, self._clock())
        if updated is None:  # lost a race with another change: read the winner
            updated = await self._store.get(approval_id) or current
        return updated

    async def list(self, *, status: ApprovalStatus | None = None,
                   limit: int = 50) -> list[Approval]:
        """List records, newest first, with overdue deadlines applied.

        Args:
            status: Only this status (after expiry is applied).
            limit: Maximum records.

        Returns:
            The records.

        Raises:
            ApprovalStoreError: The store failed.
        """
        now = self._clock()
        records = await self._store.list(limit=self._config.max_records)
        fresh: list[Approval] = []
        for record in records:
            updated = await self._expire_if_due(record, now)
            fresh.append(updated or record)
        selected = [r for r in fresh if status is None or r.status is status]
        return selected[:max(0, limit)]

    async def aclose(self) -> None:
        """Release the store's resources."""
        await self._store.aclose()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _pending(self, approval_id: str, now: float) -> Approval:
        """Load a record that is still waiting for a decision.

        Raises:
            ApprovalNotFoundError: Unknown id.
            ApprovalExpiredError: Its decision deadline passed.
            ApprovalConflictError: It was already decided.
        """
        current = await self._store.get(approval_id)
        if current is None:
            raise ApprovalNotFoundError("no approval with this id")
        current = await self._expire_if_due(current, now) or current
        if current.status is ApprovalStatus.EXPIRED:
            raise ApprovalExpiredError("this approval request expired before a decision",
                                       details={"approval": current.summary()})
        if current.status is not ApprovalStatus.PENDING:
            raise ApprovalConflictError(f"this approval is already {current.status.value}",
                                        details={"approval": current.summary()})
        return current

    async def _swap_decision(self, current: Approval, updated: Approval) -> None:
        """Store a reviewer decision, or report that someone else decided first."""
        if not await self._store.compare_and_set(current, updated):
            raise ApprovalConflictError("this approval was decided by another request")
        await self._refresh_gauge()

    async def _expire_if_due(self, record: Approval, now: float) -> Approval | None:
        """Move an overdue open record to EXPIRED.

        Returns:
            The current record (expired if it was due), or ``None`` if a
            concurrent change won the race (callers re-read).
        """
        if not record.is_due(now):
            return record
        try:
            expired = record.transition(ApprovalStatus.EXPIRED, now)
        except InvalidTransitionError:  # pragma: no cover - is_due() implies an open status
            return record
        if not await self._store.compare_and_set(record, expired):
            return None
        self._audit("expired", expired)
        self._count("expired")
        await self._refresh_gauge()
        return expired

    async def _refresh_gauge(self) -> None:
        """Publish the number of pending requests."""
        if self._metrics is not None:
            self._metrics.approvals_pending.set(await self._store.count(ApprovalStatus.PENDING))

    def _count(self, event: str) -> None:
        if self._metrics is not None:
            self._metrics.approvals.labels(event=event).inc()

    @staticmethod
    def _audit(event: str, approval: Approval) -> None:
        """Record a lifecycle event (ids and names only, never argument values)."""
        audit.info("approval %s", event, extra={
            "event": f"approval.{event}", "approval_id": approval.id, "tool": approval.tool,
            "rule": approval.rule, "scope": approval.scope, "status": approval.status.value,
            "reviewer": approval.reviewer, "origin_request_id": approval.request_id})
