"""Shared base for configuration models.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
)


class StrictModel(BaseModel):
    """Base for config models: unknown keys are errors (catches typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
