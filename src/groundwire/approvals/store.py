"""Approval storage: the port and an in-memory adapter.

:class:`ApprovalStore` is the *port*. It is deliberately dumb: it stores
immutable :class:`Approval` records and offers one atomic primitive,
:meth:`~ApprovalStore.compare_and_set`. All lifecycle rules (expiry, limits,
single use) live in :class:`~groundwire.approvals.service.ApprovalService`, so
a shared adapter (Redis with ``WATCH``/Lua, Postgres with ``UPDATE ... WHERE
status = ...``) only has to implement storage and CAS, never business logic.

:class:`InMemoryApprovalStore` is the default adapter:

* **Atomic under asyncio**: no method awaits while mutating, so every call is
  atomic with respect to other requests on the event loop.
* **Bounded**: at most ``max_records``. When full, the oldest *closed* records
  (rejected, expired, consumed) are dropped first; open records are never
  evicted. If every record is open, :meth:`add` fails and callers fail closed.
* **Single replica**: state lives in the process and is lost on restart. That
  is safe (nothing is ever released without a live approval; agents simply ask
  again) but replicas do not share approvals. Use a shared adapter for that.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter

from groundwire.approvals.models import Approval, ApprovalStatus
from groundwire.core.errors import ApprovalStoreError

logger = logging.getLogger(__name__)


class ApprovalStore(ABC):
    """Port: persistence for approval records."""

    @abstractmethod
    async def add(self, approval: Approval) -> None:
        """Insert a new record and make it the latest for its (scope, fingerprint).

        Args:
            approval: New record (its id must be new).

        Raises:
            ApprovalStoreError: If the store is full or unavailable.
        """

    @abstractmethod
    async def get(self, approval_id: str) -> Approval | None:
        """Return a record by id, or ``None``."""

    @abstractmethod
    async def find_latest(self, scope: str, fingerprint: str) -> Approval | None:
        """Return the most recent record for this exact call in this scope, or ``None``."""

    @abstractmethod
    async def compare_and_set(self, expected: Approval, updated: Approval) -> bool:
        """Replace ``expected`` with ``updated`` only if nobody changed it meanwhile.

        Args:
            expected: The record as the caller last read it.
            updated: The new version (same id).

        Returns:
            True if swapped; False if the stored record's status no longer
            matches ``expected`` (another request won the race) or it is gone.
        """

    @abstractmethod
    async def list(self, *, status: ApprovalStatus | None = None,
                   limit: int = 50) -> list[Approval]:
        """Return records, newest first, optionally filtered by status."""

    @abstractmethod
    async def count(self, status: ApprovalStatus, *, scope: str | None = None) -> int:
        """Count records with ``status`` (in one scope, or overall)."""

    @abstractmethod
    async def purge_closed(self, before: float) -> int:
        """Delete closed records last changed before ``before``. Returns how many."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources (connections). The in-memory store has none."""


class InMemoryApprovalStore(ApprovalStore):
    """Process-local, bounded approval store."""

    def __init__(self, max_records: int = 10_000) -> None:
        """Create the store.

        Args:
            max_records: Capacity across all statuses.

        Raises:
            ValueError: If ``max_records`` is not positive.
        """
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self._max = max_records
        self._records: dict[str, Approval] = {}
        self._latest: dict[tuple[str, str], str] = {}
        self._counts: Counter[tuple[ApprovalStatus, str]] = Counter()

    # ------------------------------------------------------------------ #
    # Port
    # ------------------------------------------------------------------ #

    async def add(self, approval: Approval) -> None:
        """See :meth:`ApprovalStore.add`."""
        if approval.id in self._records:
            raise ApprovalStoreError(f"duplicate approval id {approval.id}")
        if len(self._records) >= self._max and not self._evict_closed():
            raise ApprovalStoreError("approval store is full of open requests",
                                     details={"max_records": self._max})
        self._records[approval.id] = approval
        self._latest[(approval.scope, approval.fingerprint)] = approval.id
        self._counts[(approval.status, approval.scope)] += 1

    async def get(self, approval_id: str) -> Approval | None:
        """See :meth:`ApprovalStore.get`."""
        return self._records.get(approval_id)

    async def find_latest(self, scope: str, fingerprint: str) -> Approval | None:
        """See :meth:`ApprovalStore.find_latest`."""
        approval_id = self._latest.get((scope, fingerprint))
        return self._records.get(approval_id) if approval_id else None

    async def compare_and_set(self, expected: Approval, updated: Approval) -> bool:
        """See :meth:`ApprovalStore.compare_and_set`."""
        if expected.id != updated.id:
            raise ValueError("compare_and_set must keep the same id")
        current = self._records.get(expected.id)
        if current is None or current.status is not expected.status:
            return False
        self._records[updated.id] = updated
        self._counts[(current.status, current.scope)] -= 1
        self._counts[(updated.status, updated.scope)] += 1
        return True

    async def list(self, *, status: ApprovalStatus | None = None,
                   limit: int = 50) -> list[Approval]:
        """See :meth:`ApprovalStore.list`."""
        records = [r for r in self._records.values() if status is None or r.status is status]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:max(0, limit)]

    async def count(self, status: ApprovalStatus, *, scope: str | None = None) -> int:
        """See :meth:`ApprovalStore.count`."""
        if scope is not None:
            return self._counts[(status, scope)]
        return sum(n for (s, _), n in self._counts.items() if s is status)

    async def purge_closed(self, before: float) -> int:
        """See :meth:`ApprovalStore.purge_closed`."""
        stale = [r for r in self._records.values()
                 if not r.status.is_open and r.updated_at < before]
        for record in stale:
            self._remove(record)
        return len(stale)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _evict_closed(self) -> bool:
        """Drop the oldest closed record to make room. False if none exists."""
        closed = [r for r in self._records.values() if not r.status.is_open]
        if not closed:
            return False
        self._remove(min(closed, key=lambda r: r.updated_at))
        logger.warning("approval store full: evicted the oldest closed record",
                       extra={"max_records": self._max})
        return True

    def _remove(self, record: Approval) -> None:
        """Delete a record and its index entries."""
        del self._records[record.id]
        key = (record.scope, record.fingerprint)
        if self._latest.get(key) == record.id:
            del self._latest[key]
        self._counts[(record.status, record.scope)] -= 1
        if self._counts[(record.status, record.scope)] <= 0:
            del self._counts[(record.status, record.scope)]
