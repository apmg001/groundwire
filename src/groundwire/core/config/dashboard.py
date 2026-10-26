"""Live dashboard settings.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    Field,
)

from groundwire.core.config.base import StrictModel


class DashboardConfig(StrictModel):
    """Live dashboard served by the gateway at ``/dashboard``.

    The dashboard shows counts, verdicts and recent request summaries, never
    prompt or answer text, so it is safe to expose read-only.

    Attributes:
        enabled: Serve ``/dashboard`` and its JSON API.
        activity_capacity: Recent requests kept for the activity feed.
        public_demo: Demo deployments only. Shows scenario buttons and hands
            the reviewer token to the page so any visitor can approve or
            reject held calls. Never enable this where tool calls do real
            things: it makes every visitor a reviewer.
    """

    enabled: bool = True
    activity_capacity: Annotated[int, Field(ge=10, le=10_000)] = 200
    public_demo: bool = False
