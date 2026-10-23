"""Human-in-the-loop approval for tool calls.

* :mod:`~groundwire.approvals.models`: the approval record, its lifecycle and
  the exact-call fingerprint (pure domain, no I/O);
* :mod:`~groundwire.approvals.store`: the storage port and the in-memory adapter;
* :mod:`~groundwire.approvals.service`: the use cases (request, decide,
  release once, expire), used by the gateway and the reviewer API;
* :mod:`~groundwire.approvals.factory`: builds the service from config.
"""

from groundwire.approvals.factory import build_approval_service
from groundwire.approvals.models import Approval, ApprovalStatus, call_fingerprint
from groundwire.approvals.service import (
    ApprovalOutcome,
    ApprovalService,
    Resolution,
    scope_of,
)
from groundwire.approvals.store import ApprovalStore, InMemoryApprovalStore

__all__ = [
    "Approval",
    "ApprovalOutcome",
    "ApprovalService",
    "ApprovalStatus",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "Resolution",
    "build_approval_service",
    "call_fingerprint",
    "scope_of",
]
