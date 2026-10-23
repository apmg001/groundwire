"""HTTP routes for human approval of held tool calls.

Agent-facing (no reviewer token; the unguessable id is the capability):

* ``GET  /v1/approvals/{id}/status``: status, tool and deadline only.

Reviewer-facing (``Authorization: Bearer <token>``, token from the env var
named by ``approvals.reviewer_token_env``; fail closed when it is unset):

* ``GET  /v1/approvals?status=pending&limit=50``: list, newest first.
* ``GET  /v1/approvals/{id}``: full record, including the arguments to approve.
* ``POST /v1/approvals/{id}/approve``: ``{"reviewer": "...", "note": "..."}``.
* ``POST /v1/approvals/{id}/reject``: same body.

Handlers stay thin: authenticate, call :class:`ApprovalService`, serialise.
Domain errors map to HTTP via :mod:`groundwire.api.errors` (404 unknown id,
409 already decided, 410 expired, 401 bad token, 503 disabled).
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query
from pydantic import BaseModel, ConfigDict, Field

from groundwire.api.routes import get_container
from groundwire.approvals.models import ApprovalStatus
from groundwire.approvals.service import ApprovalService
from groundwire.container import Container
from groundwire.core.errors import ApprovalsDisabledError, ReviewerAuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/approvals", tags=["approvals"])

_ID = Annotated[str, Field(pattern=r"^apr_[0-9a-f]{32}$")]


class DecisionBody(BaseModel):
    """Body of approve/reject requests.

    Attributes:
        reviewer: Who is deciding (recorded in the audit trail).
        note: Optional comment or reason.
    """

    model_config = ConfigDict(extra="forbid")

    reviewer: Annotated[str, Field(min_length=1, max_length=128)]
    note: Annotated[str, Field(max_length=1000)] | None = None


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def get_approvals(container: Container = Depends(get_container)) -> ApprovalService:  # noqa: B008
    """Return the approval service.

    Args:
        container: Injected container.

    Returns:
        The service.

    Raises:
        ApprovalsDisabledError: Approvals are switched off in config.
    """
    if container.approvals is None:
        raise ApprovalsDisabledError("approvals are not enabled on this gateway")
    return container.approvals


def require_reviewer(
    authorization: Annotated[str | None, Header()] = None,
    container: Container = Depends(get_container),  # noqa: B008
) -> None:
    """Check the reviewer bearer token (constant-time comparison).

    Args:
        authorization: ``Authorization`` header.
        container: Injected container.

    Raises:
        ApprovalsDisabledError: No reviewer token is configured (fail closed).
        ReviewerAuthError: Missing or wrong token.
    """
    expected = container.reviewer_token
    if expected is None:
        raise ApprovalsDisabledError(
            "the reviewer API is disabled: set "
            f"{container.config.approvals.reviewer_token_env} on the gateway")
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.strip().encode(), expected.get_secret_value().encode()):
        logger.warning("reviewer authentication failed")
        raise ReviewerAuthError("missing or invalid reviewer token")


Service = Annotated[ApprovalService, Depends(get_approvals)]
Reviewer = Depends(require_reviewer)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get("/{approval_id}/status")
async def approval_status(approval_id: _ID, service: Service) -> dict[str, Any]:
    """Agent-facing status of one approval (no arguments, no reviewer details).

    Args:
        approval_id: Approval id.
        service: Injected service.

    Returns:
        ``{id, status, tool, expires_at}``.
    """
    return (await service.get(approval_id)).summary()


@router.get("", dependencies=[Reviewer])
async def list_approvals(
    service: Service,
    status: ApprovalStatus | None = Query(default=ApprovalStatus.PENDING),  # noqa: B008
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """List approvals, newest first.

    Args:
        service: Injected service.
        status: Filter (default: pending).
        limit: Maximum records.

    Returns:
        ``{"object": "list", "data": [...]}``.
    """
    records = await service.list(status=status, limit=limit)
    return {"object": "list", "data": [r.as_dict() for r in records]}


@router.get("/{approval_id}", dependencies=[Reviewer])
async def get_approval(approval_id: _ID, service: Service) -> dict[str, Any]:
    """Full record for a reviewer, including the exact arguments.

    Args:
        approval_id: Approval id.
        service: Injected service.

    Returns:
        The record.
    """
    return (await service.get(approval_id)).as_dict()


@router.post("/{approval_id}/approve", dependencies=[Reviewer])
async def approve(approval_id: _ID, body: DecisionBody, service: Service) -> dict[str, Any]:
    """Approve a pending call; the agent's next identical call will run once.

    Args:
        approval_id: Approval id.
        body: Reviewer and note.
        service: Injected service.

    Returns:
        The approved record.
    """
    record = await service.approve(approval_id, reviewer=body.reviewer, note=body.note)
    return record.as_dict()


@router.post("/{approval_id}/reject", dependencies=[Reviewer])
async def reject(approval_id: _ID, body: DecisionBody, service: Service) -> dict[str, Any]:
    """Reject a pending call; the agent's identical call stays blocked.

    Args:
        approval_id: Approval id.
        body: Reviewer and note.
        service: Injected service.

    Returns:
        The rejected record.
    """
    record = await service.reject(approval_id, reviewer=body.reviewer, note=body.note)
    return record.as_dict()
