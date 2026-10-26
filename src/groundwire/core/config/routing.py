"""Cascade routing: a cheap model first, a strong one only when needed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    Field,
)

from groundwire.core.config.base import StrictModel


class CascadeTier(StrEnum):
    """The two tiers of a cascade."""

    CHEAP = "cheap"
    STRONG = "strong"


class CascadeCheckFailure(StrEnum):
    """What a cascade does when the fact-check cannot score the cheap answer."""

    ESCALATE = "escalate"  # correctness first: ask the strong tier
    SERVE_CHEAP = "serve_cheap"  # cost first: deliver the unchecked cheap answer


class CascadeConfig(StrictModel):
    """Cheap-first routing that escalates when the fact-check fails.

    Measured on RAGTruth (``make bench-cascade-token``): GPT-3.5 -> GPT-4 matched
    GPT-4's hallucination rate at ~15% of its cost, but Llama-2-7B -> GPT-4 only
    halved the cost while doubling errors. Savings are capped by how often the
    cheap tier is right, so enable a cascade only where the cheap model is decent,
    and watch ``groundwire_cascade_requests_total``.

    Attributes:
        cheap: Provider chain tried first (normal retries and failover apply).
        strong: Provider chain used when the cheap answer fails the check.
        escalate_below: Escalate when the verifier's answer score is below this.
            Its meaning follows ``grounding.aggregation`` (with LettuceDetect and
            ``min``: 0.5 = escalate if any token is more likely made up than not).
        on_check_error: Behaviour when the check cannot run.
        without_context: Tier for requests that carry no grounding documents
            (nothing to check against).
    """

    cheap: Annotated[list[str], Field(min_length=1)]
    strong: Annotated[list[str], Field(min_length=1)]
    escalate_below: Annotated[float, Field(ge=0, le=1)] = 0.5
    on_check_error: CascadeCheckFailure = CascadeCheckFailure.ESCALATE
    without_context: CascadeTier = CascadeTier.STRONG
