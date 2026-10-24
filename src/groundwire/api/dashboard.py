"""Dashboard routes: the page and its two read-only JSON endpoints.

* ``GET /``: redirects to the dashboard (when it is enabled).
* ``GET /dashboard``: a single self-contained HTML page (no build step, no CDN).
* ``GET /v1/dashboard/summary``: totals and current state.
* ``GET /v1/dashboard/activity?after=<seq>&limit=<n>``: recent requests.

All return 404 when ``dashboard.enabled`` is false. Approve/reject actions on
the page use the normal reviewer API (``/v1/approvals``) with its token.
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from groundwire.api.routes import get_container
from groundwire.container import Container
from groundwire.services.dashboard import MAX_ACTIVITY_PAGE, DashboardService

router = APIRouter()

_PAGE = "dashboard.html"


def _dashboard(container: Container = Depends(get_container)) -> DashboardService:  # noqa: B008
    """FastAPI dependency: the dashboard service, or 404 when it is off.

    Raises:
        HTTPException: 404 if the dashboard is disabled.
    """
    if container.dashboard is None or not container.dashboard.enabled:
        raise HTTPException(status_code=404, detail="dashboard is disabled")
    return container.dashboard


@functools.cache
def _page() -> str:
    """The dashboard HTML, read from the installed package once."""
    return (resources.files("groundwire.api") / "static" / _PAGE).read_text(encoding="utf-8")


@router.get("/", include_in_schema=False)
async def root(_: DashboardService = Depends(_dashboard)) -> RedirectResponse:  # noqa: B008
    """Send browsers to the dashboard."""
    return RedirectResponse("/dashboard", status_code=307)


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def page(_: DashboardService = Depends(_dashboard)) -> HTMLResponse:  # noqa: B008
    """Serve the dashboard page."""
    return HTMLResponse(_page(), headers={"Cache-Control": "no-cache"})


@router.get("/v1/dashboard/summary")
async def summary(dashboard: DashboardService = Depends(_dashboard)) -> dict[str, Any]:  # noqa: B008
    """Totals, safety counts, load and provider health.

    Args:
        dashboard: Injected dashboard service.

    Returns:
        JSON snapshot.
    """
    return dashboard.summary()


@router.get("/v1/dashboard/activity")
async def activity(
    after: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=MAX_ACTIVITY_PAGE),
    dashboard: DashboardService = Depends(_dashboard),  # noqa: B008
) -> dict[str, Any]:
    """Recent requests, newest first (no prompt or answer text).

    Args:
        after: Only events with a larger sequence number.
        limit: Maximum events.
        dashboard: Injected dashboard service.

    Returns:
        ``{"last_seq": n, "events": [...]}``.
    """
    return dashboard.activity(after=after, limit=limit)
