"""Grounding verification: is the answer supported by the supplied context?

Algorithm:

1. Split the answer into claims (sentences / bullets).
2. Drop non-claims: sentences with fewer than ``min_claim_tokens`` content
   words ("Sure!", "Here you go:").
3. Split the context into passages small enough for the scorer.
4. Score every (passage, claim) pair; a claim's support is its best passage.
5. A claim is *supported* if support >= ``support_threshold``.
6. The answer *passes* if supported_claims / claims >= ``min_supported_ratio``.

An answer with no checkable claims passes trivially (nothing to contradict).

The verifier never raises for a "bad" answer; it returns a report. Policy
(flag vs block) is applied by the caller. If the scorer itself fails, the
configured failure policy decides between "unverified" (fail-open) and
:class:`GuardrailUnavailableError` (fail-closed).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from groundwire.core.config import GroundingConfig, GroundingFailurePolicy, GroundingScorerType
from groundwire.core.errors import GuardrailUnavailableError
from groundwire.guardrails.scorers import LexicalScorer, NLIScorer, SupportScorer
from groundwire.guardrails.text import chunk_passages, content_tokens, split_claims

logger = logging.getLogger(__name__)

# Bound work per request so one huge answer/context cannot monopolise the CPU.
MAX_CLAIMS = 64
MAX_PASSAGES = 256


class Verdict(StrEnum):
    """Outcome of a grounding check."""

    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"  # scorer failed and policy is fail-open


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Grounding result for one claim.

    Attributes:
        claim: Claim text.
        score: Best support score across passages.
        supported: Whether ``score`` met the threshold.
        best_passage: Index of the best-supporting passage.
    """

    claim: str
    score: float
    supported: bool
    best_passage: int


@dataclass(slots=True)
class GroundingReport:
    """Full grounding report returned to clients.

    Attributes:
        verdict: Overall verdict.
        scorer: Scorer used.
        supported_ratio: Supported claims / checked claims.
        claims: Per-claim results.
        latency_ms: Time spent verifying.
        truncated: True if claims/passages were capped.
        error: Scorer error message when ``verdict`` is ``unverified``.
    """

    verdict: Verdict
    scorer: str
    supported_ratio: float = 1.0
    claims: list[ClaimResult] = field(default_factory=list)
    latency_ms: float = 0.0
    truncated: bool = False
    error: str | None = None

    @property
    def unsupported(self) -> list[ClaimResult]:
        """Claims that failed the threshold."""
        return [c for c in self.claims if not c.supported]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["claims"] = [
            {**asdict(c), "score": round(c.score, 4)} for c in self.claims
        ]
        data["supported_ratio"] = round(self.supported_ratio, 4)
        data["latency_ms"] = round(self.latency_ms, 2)
        return data


class GroundingVerifier:
    """Checks answers against context using a pluggable scorer."""

    def __init__(self, config: GroundingConfig, scorer: SupportScorer, *,
                 max_concurrency: int = 2) -> None:
        """Create the verifier.

        Args:
            config: Thresholds and failure policy.
            scorer: Support scorer.
            max_concurrency: Max concurrent scoring jobs. Scorers are CPU-bound,
                so running dozens at once only thrashes the CPU; excess requests
                queue here instead.
        """
        self._config = config
        self._scorer = scorer
        self._limit = asyncio.Semaphore(max_concurrency)

    @property
    def scorer_name(self) -> str:
        """Name of the active scorer."""
        return self._scorer.name

    def extract_claims(self, answer: str) -> list[str]:
        """Return the checkable claims in ``answer``.

        Args:
            answer: Model answer.

        Returns:
            Claims with at least ``min_claim_tokens`` content tokens.
        """
        return [c for c in split_claims(answer)
                if len(content_tokens(c)) >= self._config.min_claim_tokens]

    async def verify(self, answer: str, context: list[str]) -> GroundingReport:
        """Verify ``answer`` against ``context``.

        Args:
            answer: Model answer.
            context: Source passages supplied by the client.

        Returns:
            The grounding report (``failed`` is a normal outcome, not an error).

        Raises:
            GuardrailUnavailableError: If scoring fails and policy is fail-closed.
        """
        started = time.perf_counter()
        claims = self.extract_claims(answer)
        passages = chunk_passages(context)
        truncated = len(claims) > MAX_CLAIMS or len(passages) > MAX_PASSAGES
        claims, passages = claims[:MAX_CLAIMS], passages[:MAX_PASSAGES]

        if not claims:
            return GroundingReport(verdict=Verdict.PASSED, scorer=self._scorer.name,
                                   latency_ms=_ms_since(started), truncated=truncated)
        if not passages:
            # Context contained no text: nothing can be supported.
            unsupported = [ClaimResult(c, 0.0, False, -1) for c in claims]
            return self._report(unsupported, started, truncated)

        pairs = [(p, c) for c in claims for p in passages]
        try:
            async with self._limit:
                scores = await asyncio.to_thread(self._scorer.score_batch, pairs)
        except GuardrailUnavailableError as exc:
            return self._on_scorer_failure(exc, started)
        except Exception as exc:  # noqa: BLE001 - any scorer crash is a guardrail outage
            return self._on_scorer_failure(GuardrailUnavailableError(str(exc)), started)

        if len(scores) != len(pairs):
            return self._on_scorer_failure(
                GuardrailUnavailableError("scorer returned wrong number of scores"), started)

        n = len(passages)
        results: list[ClaimResult] = []
        for i, claim in enumerate(claims):
            row = scores[i * n:(i + 1) * n]
            best = max(range(n), key=row.__getitem__)
            results.append(ClaimResult(claim, row[best],
                                       row[best] >= self._config.support_threshold, best))
        return self._report(results, started, truncated)

    def _report(self, results: list[ClaimResult], started: float,
                truncated: bool) -> GroundingReport:
        ratio = sum(r.supported for r in results) / len(results)
        verdict = (Verdict.PASSED if ratio >= self._config.min_supported_ratio
                   else Verdict.FAILED)
        return GroundingReport(verdict=verdict, scorer=self._scorer.name,
                               supported_ratio=ratio, claims=results,
                               latency_ms=_ms_since(started), truncated=truncated)

    def _on_scorer_failure(self, exc: GuardrailUnavailableError,
                           started: float) -> GroundingReport:
        logger.error("grounding scorer failed",
                     extra={"scorer": self._scorer.name, "error": exc.message,
                            "policy": self._config.on_error.value})
        if self._config.on_error is GroundingFailurePolicy.FAIL_CLOSED:
            raise exc
        return GroundingReport(verdict=Verdict.UNVERIFIED, scorer=self._scorer.name,
                               supported_ratio=0.0, latency_ms=_ms_since(started),
                               error=exc.message)


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def build_verifier(config: GroundingConfig) -> GroundingVerifier:
    """Build a verifier with the configured scorer.

    Args:
        config: Grounding configuration.

    Returns:
        The verifier.

    Raises:
        ConfigurationError: If the NLI scorer is selected but cannot load.
    """
    scorer: SupportScorer
    if config.scorer is GroundingScorerType.NLI:
        scorer = NLIScorer(config.nli_model)
    else:
        scorer = LexicalScorer()
    return GroundingVerifier(config, scorer)
