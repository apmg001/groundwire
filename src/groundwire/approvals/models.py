"""Approval domain model: the record, its status rules and the call fingerprint.

An approval is **permission for one exact tool call**: the same tool, the same
arguments and the same requester (``scope``), used at most once. Binding to
the exact arguments is the security property: a reviewer who approves a $600
refund has not approved a $60,000 one, even if the model tries again with a
different amount.

Lifecycle (every arrow is an atomic compare-and-set in the store)::

    PENDING ──approve──► APPROVED ──release──► CONSUMED
       │                    │
       ├──reject──► REJECTED │
       └──ttl──► EXPIRED ◄──┘ (grant not used in time)

This module is pure: no I/O, no clock, no logging. Times are passed in.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

ID_PREFIX = "apr_"


class ApprovalStatus(StrEnum):
    """Where an approval is in its lifecycle."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"

    @property
    def is_open(self) -> bool:
        """True while the record can still change because of a decision or a release."""
        return self in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED)


# Allowed transitions; anything else is a bug or a lost race.
_TRANSITIONS: dict[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.PENDING: frozenset({ApprovalStatus.APPROVED, ApprovalStatus.REJECTED,
                                       ApprovalStatus.EXPIRED}),
    ApprovalStatus.APPROVED: frozenset({ApprovalStatus.CONSUMED, ApprovalStatus.EXPIRED}),
    ApprovalStatus.REJECTED: frozenset(),
    ApprovalStatus.EXPIRED: frozenset(),
    ApprovalStatus.CONSUMED: frozenset(),
}


class InvalidTransitionError(ValueError):
    """A status change that the lifecycle does not allow."""


def new_approval_id() -> str:
    """Return a new unguessable approval id.

    128 random bits: knowing an id is what lets an agent poll its status, so
    ids must not be enumerable.

    Returns:
        An id like ``apr_3f9c...``.
    """
    return ID_PREFIX + secrets.token_hex(16)


def _canonical(value: Any) -> Any:
    """Normalise a JSON value so equal calls hash equally.

    ``1200`` and ``1200.0`` are the same amount; dict key order never matters.

    Args:
        value: Decoded JSON.

    Returns:
        The canonical form.

    Raises:
        ValueError: For NaN or infinity, which have no JSON representation.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("arguments contain a non-finite number")
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return str(value)


def call_fingerprint(tool: str, arguments: dict[str, Any]) -> str:
    """Hash a tool call so the same action always gets the same fingerprint.

    Args:
        tool: Tool name.
        arguments: Decoded arguments (PII restored: real values are what get approved).

    Returns:
        Hex SHA-256 of the canonical JSON of ``{tool, arguments}``.

    Raises:
        ValueError: If the arguments contain NaN or infinity.
    """
    payload = json.dumps({"tool": tool, "arguments": _canonical(arguments)},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iso(timestamp: float | None) -> str | None:
    """Format a UNIX timestamp as ISO 8601 UTC (``None`` passes through)."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Approval:
    """One approval request and everything decided about it.

    Immutable: every change produces a new record (:meth:`transition`), which
    the store swaps in with compare-and-set.

    Attributes:
        id: Unguessable id (``apr_...``).
        scope: Who asked (the request's ``user``); approvals never cross scopes.
        fingerprint: :func:`call_fingerprint` of the exact call.
        tool: Tool name.
        arguments: The exact arguments the reviewer is asked to approve.
        rule: Name of the policy rule that required approval.
        status: Lifecycle status.
        created_at: When it was requested (UNIX seconds).
        expires_at: When the current status lapses (pending: decision deadline;
            approved: deadline to use it).
        updated_at: Last status change (drives retention of closed records).
        request_id: Gateway request id that created it, for tracing.
        decided_at: When a reviewer decided.
        reviewer: Who decided.
        note: Reviewer's comment.
        consumed_at: When the approved call was released to the agent.
    """

    id: str
    scope: str
    fingerprint: str
    tool: str
    arguments: dict[str, Any]
    rule: str | None
    status: ApprovalStatus
    created_at: float
    expires_at: float
    updated_at: float
    request_id: str | None = None
    decided_at: float | None = None
    reviewer: str | None = None
    note: str | None = None
    consumed_at: float | None = None

    def is_due(self, now: float) -> bool:
        """True if an open record has passed its deadline."""
        return self.status.is_open and now >= self.expires_at

    def transition(self, status: ApprovalStatus, now: float, **changes: Any) -> Approval:
        """Return a copy in ``status`` with ``changes`` applied.

        Args:
            status: Target status.
            now: Time of the change (becomes ``updated_at``).
            **changes: Other fields to update.

        Returns:
            The new record.

        Raises:
            InvalidTransitionError: If the lifecycle forbids the move.
        """
        if status not in _TRANSITIONS[self.status]:
            raise InvalidTransitionError(f"{self.status.value} -> {status.value}")
        return replace(self, status=status, updated_at=now, **changes)

    def summary(self) -> dict[str, Any]:
        """Public view for the agent-facing status endpoint (no arguments, no scope)."""
        return {"id": self.id, "status": self.status.value, "tool": self.tool,
                "expires_at": _iso(self.expires_at)}

    def as_dict(self) -> dict[str, Any]:
        """Full view for reviewers (includes the arguments being approved)."""
        return {
            "id": self.id,
            "status": self.status.value,
            "tool": self.tool,
            "arguments": self.arguments,
            "rule": self.rule,
            "scope": self.scope,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "decided_at": _iso(self.decided_at),
            "reviewer": self.reviewer,
            "note": self.note,
            "consumed_at": _iso(self.consumed_at),
            "request_id": self.request_id,
        }
