"""Cascade routing: cheap model first, strong model only when the answer fails a fact-check.

::

    request ─► cheap tier ─► fact-check ──score >= threshold──► deliver cheap answer
                   │              │
                   │              └─score < threshold─► strong tier ─► deliver
                   └─ tier down ─────────────────────► strong tier ─► deliver

The fact-check is the gateway's own :class:`GroundingVerifier` (LettuceDetect
in production), run on the cheap answer against the documents the client
supplied in ``groundwire.grounding.context``.

Why this exists, and when it does not pay off, was measured before it was
built (``make bench-cascade-token`` on RAGTruth): GPT-3.5 -> GPT-4 matched
GPT-4's hallucination rate at ~15% of its cost, while Llama-2-7B -> GPT-4 only
halved the cost and doubled errors. Savings are capped by how often the cheap
tier is right.

Fault tolerance: a cascade must never be *less* available than its tiers.

* cheap tier fully down       -> serve from the strong tier,
* strong tier down on escalation -> serve the cheap answer (and say so),
* the check itself fails      -> ``on_check_error`` (default: escalate),
* no documents to check       -> ``without_context`` tier (default: strong),
  falling back to the other tier if that one is down.

Client faults (upstream 400) are never retried on the other tier.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from groundwire.core.config import CascadeCheckFailure, CascadeConfig, CascadeTier
from groundwire.core.errors import (
    AllProvidersFailedError,
    AttemptRecord,
    GuardrailUnavailableError,
    ModelNotFoundError,
)
from groundwire.domain.models import ChatCompletionRequest
from groundwire.guardrails.grounding import GroundingReport, GroundingVerifier, Verdict
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.router import RoutedCompletion, Router, tier_route

logger = logging.getLogger(__name__)

Attempts = list[tuple[CascadeTier, AttemptRecord]]


class CascadeReason(StrEnum):
    """Why a tier served the answer."""

    PASSED = "passed"  # cheap answer passed the check
    LOW_SUPPORT = "low_support"  # cheap answer failed the check -> escalated
    CHECK_FAILED = "check_failed"  # the check could not run
    NO_CONTEXT = "no_context"  # no documents to check against
    CHEAP_UNAVAILABLE = "cheap_unavailable"  # every cheap provider failed
    STRONG_UNAVAILABLE = "strong_unavailable"  # escalation failed; cheap answer served


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CascadeDecision:
    """What the cascade did, for response metadata and metrics.

    Attributes:
        cascade: Cascade alias.
        tier: Tier whose answer was delivered.
        reason: Why.
        threshold: Configured ``escalate_below``.
        score: Answer score of the cheap answer, if it was checked.
        check_ms: Time spent on the check.
    """

    cascade: str
    tier: CascadeTier
    reason: CascadeReason
    threshold: float
    score: float | None = None
    check_ms: float | None = None

    @property
    def escalated(self) -> bool:
        """True when the cheap answer was replaced by the strong tier's."""
        return self.reason in (CascadeReason.LOW_SUPPORT, CascadeReason.CHECK_FAILED) and \
            self.tier is CascadeTier.STRONG

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (no answer text)."""
        data: dict[str, Any] = {"cascade": self.cascade, "tier": self.tier.value,
                                "reason": self.reason.value, "escalated": self.escalated,
                                "threshold": self.threshold}
        if self.score is not None:
            data["cheap_score"] = round(self.score, 4)
        if self.check_ms is not None:
            data["check_ms"] = round(self.check_ms, 1)
        return data


@dataclass(slots=True)
class CascadeOutcome:
    """The delivered answer plus how it was chosen.

    Attributes:
        routed: The delivered completion (redacted form) and its provider.
        attempts: Every provider attempt across both tiers, tagged with the tier.
        decision: The cascade decision.
        report: Grounding report of the delivered text when it is the checked
            cheap answer (lets the caller skip a second identical check).
    """

    routed: RoutedCompletion
    attempts: list[dict[str, Any]] = field(default_factory=list)
    decision: CascadeDecision | None = None
    report: GroundingReport | None = None


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class CascadeRunner:
    """Executes cascade routes on top of the router and the grounding verifier."""

    def __init__(self, router: Router, verifier: GroundingVerifier, *,
                 metrics: GatewayMetrics | None = None) -> None:
        """Create the runner.

        Args:
            router: Router that owns the cascade tiers.
            verifier: Fact-checker used to judge cheap answers.
            metrics: Metrics sink.
        """
        self._router = router
        self._verifier = verifier
        self._metrics = metrics

    async def complete(self, name: str, request: ChatCompletionRequest, *,
                       context: list[str] | None, question: str | None,
                       restore: Callable[[str], str] = lambda text: text) -> CascadeOutcome:
        """Serve one request through a cascade.

        Args:
            name: Cascade alias.
            request: Request to send to providers (already PII-redacted).
            context: Grounding documents (unredacted), or ``None``.
            question: The user's question, for the token detector's prompt.
            restore: Puts PII back into an answer, so the check compares real
                values with the documents (placeholders would never match).

        Returns:
            The outcome.

        Raises:
            ModelNotFoundError: If ``name`` is not a cascade.
            AllProvidersFailedError: If both tiers are down.
            ProviderBadRequestError: If a provider rejected the request.
        """
        config = self._config(name)
        if not context:
            outcome = await self._without_context(name, config, request)
        else:
            outcome = await self._checked(name, config, request, context, question, restore)
        decision = outcome.decision
        if decision is not None:
            self._record(decision)
            logger.info("cascade decided", extra=decision.as_dict())
        return outcome

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    async def _checked(self, name: str, config: CascadeConfig,
                       request: ChatCompletionRequest, context: list[str],
                       question: str | None, restore: Callable[[str], str]) -> CascadeOutcome:
        """Cheap tier, fact-check, escalate if needed."""
        attempts: Attempts = []
        try:
            cheap = await self._call(name, CascadeTier.CHEAP, request, attempts)
        except AllProvidersFailedError:
            strong = await self._last_resort(name, CascadeTier.STRONG, request, attempts)
            return _outcome(strong, attempts, CascadeDecision(
                name, CascadeTier.STRONG, CascadeReason.CHEAP_UNAVAILABLE,
                config.escalate_below))

        started = time.perf_counter()
        report: GroundingReport | None
        try:
            report = await self._verifier.verify(restore(cheap.completion.text), context,
                                                 question=question)
        except GuardrailUnavailableError as exc:
            logger.warning("cascade check failed", extra={"cascade": name,
                                                          "error": exc.message})
            report = None
        check_ms = (time.perf_counter() - started) * 1000
        if self._metrics is not None:
            self._metrics.cascade_check_latency.labels(cascade=name).observe(check_ms / 1000)

        if report is None or report.verdict is Verdict.UNVERIFIED:
            if config.on_check_error is CascadeCheckFailure.SERVE_CHEAP:
                return _outcome(cheap, attempts, CascadeDecision(
                    name, CascadeTier.CHEAP, CascadeReason.CHECK_FAILED, config.escalate_below,
                    check_ms=check_ms))
            return await self._escalate(name, config, request, cheap, attempts,
                                        CascadeReason.CHECK_FAILED, None, check_ms)

        if report.answer_score >= config.escalate_below:
            return _outcome(cheap, attempts, CascadeDecision(
                name, CascadeTier.CHEAP, CascadeReason.PASSED, config.escalate_below,
                report.answer_score, check_ms), report=report)
        return await self._escalate(name, config, request, cheap, attempts,
                                    CascadeReason.LOW_SUPPORT, report.answer_score, check_ms)

    async def _escalate(self, name: str, config: CascadeConfig,
                        request: ChatCompletionRequest, cheap: RoutedCompletion,
                        attempts: Attempts, reason: CascadeReason,
                        score: float | None, check_ms: float) -> CascadeOutcome:
        """Ask the strong tier; keep the cheap answer if the strong tier is down."""
        try:
            strong = await self._call(name, CascadeTier.STRONG, request, attempts)
        except AllProvidersFailedError:
            logger.warning("cascade escalation failed; serving the cheap answer",
                           extra={"cascade": name, "reason": reason.value})
            return _outcome(cheap, attempts, CascadeDecision(
                name, CascadeTier.CHEAP, CascadeReason.STRONG_UNAVAILABLE,
                config.escalate_below, score, check_ms))
        return _outcome(strong, attempts, CascadeDecision(
            name, CascadeTier.STRONG, reason, config.escalate_below, score, check_ms))

    async def _without_context(self, name: str, config: CascadeConfig,
                               request: ChatCompletionRequest) -> CascadeOutcome:
        """No documents: use the configured tier, falling back to the other one."""
        primary = config.without_context
        other = CascadeTier.CHEAP if primary is CascadeTier.STRONG else CascadeTier.STRONG
        attempts: Attempts = []
        try:
            routed = await self._call(name, primary, request, attempts)
            reason, tier = CascadeReason.NO_CONTEXT, primary
        except AllProvidersFailedError:
            routed = await self._last_resort(name, other, request, attempts)
            tier = other
            reason = (CascadeReason.CHEAP_UNAVAILABLE if primary is CascadeTier.CHEAP
                      else CascadeReason.STRONG_UNAVAILABLE)
        return _outcome(routed, attempts, CascadeDecision(
            name, tier, reason, config.escalate_below))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _config(self, name: str) -> CascadeConfig:
        """Look up a cascade's policy.

        Args:
            name: Cascade alias.

        Returns:
            The policy.

        Raises:
            ModelNotFoundError: If ``name`` is not a cascade.
        """
        config = self._router.cascade_config(name)
        if config is None:
            raise ModelNotFoundError(f"'{name}' is not a configured cascade")
        return config

    async def _last_resort(self, name: str, tier: CascadeTier, request: ChatCompletionRequest,
                           attempts: Attempts) -> RoutedCompletion:
        """Call the remaining tier after the other one is down.

        Args:
            name: Cascade alias.
            tier: Tier to call.
            request: Request.
            attempts: Accumulator, extended in place.

        Returns:
            The tier's completion.

        Raises:
            AllProvidersFailedError: Both tiers are down; carries every
                attempt from both, under the cascade's name.
        """
        try:
            return await self._call(name, tier, request, attempts)
        except AllProvidersFailedError as exc:
            raise AllProvidersFailedError(name, [record for _, record in attempts]) from exc

    async def _call(self, name: str, tier: CascadeTier, request: ChatCompletionRequest,
                    attempts: Attempts) -> RoutedCompletion:
        """Call one tier, recording its attempts (also when it fails).

        Args:
            name: Cascade alias.
            tier: Tier to call.
            request: Request.
            attempts: Accumulator, extended in place.

        Returns:
            The tier's completion.

        Raises:
            AllProvidersFailedError: If every provider in the tier failed.
            ProviderBadRequestError: If a provider rejected the request.
        """
        try:
            routed = await self._router.complete(tier_route(name, tier), request)
        except AllProvidersFailedError as exc:
            attempts.extend((tier, record) for record in exc.attempts)
            raise
        attempts.extend((tier, record) for record in routed.attempts)
        return routed

    def _record(self, decision: CascadeDecision) -> None:
        if self._metrics is not None:
            self._metrics.cascade_requests.labels(
                cascade=decision.cascade, tier=decision.tier.value,
                reason=decision.reason.value).inc()


def _outcome(routed: RoutedCompletion, attempts: Attempts, decision: CascadeDecision, *,
             report: GroundingReport | None = None) -> CascadeOutcome:
    """Package a result, tagging each attempt with its tier.

    Args:
        routed: Delivered completion.
        attempts: Tier-tagged attempt records.
        decision: The decision.
        report: Report on the delivered text, if it was checked.

    Returns:
        The outcome.
    """
    return CascadeOutcome(routed=routed,
                          attempts=[{**record.as_dict(), "tier": tier.value}
                                    for tier, record in attempts],
                          decision=decision, report=report)
