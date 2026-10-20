"""Grounding verification: is the answer supported by the supplied context?

Algorithm:

1. Split the answer into claims (sentences / bullets).
2. Drop non-claims: sentences with fewer than ``min_claim_tokens`` content
   words ("Sure!", "Here you go:").
3. Split the context into passages small enough for the scorer.
4. Score each claim's support:

   * ``evidence_passages = 0``: score every (passage, claim) pair; the claim's
     support is its best passage.
   * ``evidence_passages = k``: pick the ``k`` passages that share the most
     words with the claim, join them, and score the claim once against them.

5. A claim is *supported* if support >= ``support_threshold``.
6. Combine claims into a verdict (``aggregation``):

   * ``ratio``: pass if supported_claims / claims >= ``min_supported_ratio``,
   * ``min`` / ``lowest_two`` / ``mean``: aggregate the claim scores into one
     answer score and pass if it is >= ``support_threshold``.

   ``lowest_two`` was selected on the RAGTruth *train* split: judging only
   the single weakest claim over-penalises long answers, while a plain mean
   lets one invented fact hide among many correct ones.

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

from groundwire.core.config import (
    AnswerAggregation,
    GroundingConfig,
    GroundingFailurePolicy,
    GroundingScorerType,
)
from groundwire.core.errors import GuardrailUnavailableError
from groundwire.guardrails.scorers import (
    HybridScorer,
    LexicalScorer,
    NLIScorer,
    SupportScorer,
)
from groundwire.guardrails.text import chunk_passages, content_tokens, split_claims
from groundwire.guardrails.token_detector import (
    TokenDetector,
    TransformersTokenDetector,
    build_prompt,
    max_prob_in_range,
)

Scorer = SupportScorer | TokenDetector

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
        answer_score: Claim scores combined by the configured aggregation
            (for ``ratio`` mode this equals ``supported_ratio``).
        aggregation: Aggregation used.
        claims: Per-claim results.
        latency_ms: Time spent verifying.
        truncated: True if claims/passages were capped.
        error: Scorer error message when ``verdict`` is ``unverified``.
    """

    verdict: Verdict
    scorer: str
    supported_ratio: float = 1.0
    answer_score: float = 1.0
    aggregation: str = AnswerAggregation.RATIO.value
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
        data["answer_score"] = round(self.answer_score, 4)
        data["latency_ms"] = round(self.latency_ms, 2)
        return data


class GroundingVerifier:
    """Checks answers against context using a pluggable scorer."""

    def __init__(self, config: GroundingConfig, scorer: Scorer, *,
                 max_concurrency: int = 2) -> None:
        """Create the verifier.

        Args:
            config: Thresholds and failure policy.
            scorer: Pairwise support scorer, or a token detector that reads the
                whole prompt at once.
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

    async def verify(self, answer: str, context: list[str], *, question: str | None = None,
                     prompt: str | None = None) -> GroundingReport:
        """Verify ``answer`` against ``context``.

        Args:
            answer: Model answer.
            context: Source passages supplied by the client.
            question: The user's question (used by token detectors to build
                their prompt; ignored by pairwise scorers).
            prompt: Exact prompt the answer was generated from. Token detectors
                use it verbatim instead of building one (benchmarks pass the
                dataset's original prompt).

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
                                   aggregation=self._config.aggregation.value,
                                   latency_ms=_ms_since(started), truncated=truncated)
        if isinstance(self._scorer, TokenDetector):
            return await self._verify_with_tokens(
                self._scorer, answer, claims,
                prompt if prompt is not None else build_prompt(context, question),
                started, truncated)
        if not passages:
            # Context contained no text: nothing can be supported.
            unsupported = [ClaimResult(c, 0.0, False, -1) for c in claims]
            return self._report(unsupported, started, truncated)

        pairs, best_index = self._build_pairs(claims, passages)
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

        threshold = self._config.support_threshold
        results: list[ClaimResult] = []
        if best_index is not None:  # evidence-first: one pair per claim
            for claim, score, best in zip(claims, scores, best_index, strict=True):
                results.append(ClaimResult(claim, score, score >= threshold, best))
        else:  # all pairs: best passage wins
            n = len(passages)
            for i, claim in enumerate(claims):
                row = scores[i * n:(i + 1) * n]
                best = max(range(n), key=row.__getitem__)
                results.append(ClaimResult(claim, row[best], row[best] >= threshold, best))
        return self._report(results, started, truncated)

    async def _verify_with_tokens(self, detector: TokenDetector, answer: str,
                                  claims: list[str], prompt: str, started: float,
                                  truncated: bool) -> GroundingReport:
        """Score claims from per-token hallucination probabilities.

        A claim's support is ``1 - max(token probability)`` over the answer
        tokens inside it, so one confidently hallucinated word marks the claim.

        Args:
            detector: Token detector.
            answer: Model answer.
            claims: Claims extracted from the answer.
            prompt: Prompt given to the detector.
            started: ``perf_counter`` at the start of verification.
            truncated: Whether claims/passages were capped.

        Returns:
            The report.

        Raises:
            GuardrailUnavailableError: If detection fails and policy is fail-closed.
        """
        try:
            async with self._limit:
                tokens = await asyncio.to_thread(detector.detect, prompt, answer)
        except GuardrailUnavailableError as exc:
            return self._on_scorer_failure(exc, started)
        except Exception as exc:  # noqa: BLE001 - any detector crash is a guardrail outage
            return self._on_scorer_failure(GuardrailUnavailableError(str(exc)), started)

        threshold = self._config.support_threshold
        results: list[ClaimResult] = []
        cursor = 0
        for claim in claims:
            start = answer.find(claim, cursor)
            if start < 0:  # should not happen; fall back to the whole answer (conservative)
                start, end = 0, len(answer)
            else:
                end = start + len(claim)
                cursor = end
            score = 1.0 - max_prob_in_range(tokens, start, end)
            results.append(ClaimResult(claim, score, score >= threshold, -1))
        return self._report(results, started, truncated)

    def _build_pairs(self, claims: list[str], passages: list[str]
                     ) -> tuple[list[tuple[str, str]], list[int] | None]:
        """Build the (premise, claim) pairs to score.

        Args:
            claims: Claims to check.
            passages: Context passages.

        Returns:
            ``(pairs, best_index)``. In all-pairs mode ``best_index`` is
            ``None``; in evidence-first mode it holds, per claim, the index of
            the passage that shared the most words with it.
        """
        k = self._config.evidence_passages
        if k == 0:
            return [(p, c) for c in claims for p in passages], None

        passage_tokens = [content_tokens(p) for p in passages]
        pairs: list[tuple[str, str]] = []
        best_index: list[int] = []
        for claim in claims:
            claim_tokens = content_tokens(claim)
            ranked = sorted(range(len(passages)),
                            key=lambda i: (-len(claim_tokens & passage_tokens[i]), i))[:k]
            # Keep document order so the joined evidence still reads naturally.
            premise = " ".join(passages[i] for i in sorted(ranked))
            pairs.append((premise, claim))
            best_index.append(ranked[0])
        return pairs, best_index

    def _report(self, results: list[ClaimResult], started: float,
                truncated: bool) -> GroundingReport:
        ratio = sum(r.supported for r in results) / len(results)
        aggregation = self._config.aggregation
        if aggregation is AnswerAggregation.RATIO:
            answer_score = ratio
            passed = ratio >= self._config.min_supported_ratio
        else:
            answer_score = aggregate([r.score for r in results], aggregation)
            passed = answer_score >= self._config.support_threshold
        return GroundingReport(verdict=Verdict.PASSED if passed else Verdict.FAILED,
                               scorer=self._scorer.name, supported_ratio=ratio,
                               answer_score=answer_score, aggregation=aggregation.value,
                               claims=results, latency_ms=_ms_since(started),
                               truncated=truncated)

    def _on_scorer_failure(self, exc: GuardrailUnavailableError,
                           started: float) -> GroundingReport:
        logger.error("grounding scorer failed",
                     extra={"scorer": self._scorer.name, "error": exc.message,
                            "policy": self._config.on_error.value})
        if self._config.on_error is GroundingFailurePolicy.FAIL_CLOSED:
            raise exc
        return GroundingReport(verdict=Verdict.UNVERIFIED, scorer=self._scorer.name,
                               supported_ratio=0.0, answer_score=0.0,
                               aggregation=self._config.aggregation.value,
                               latency_ms=_ms_since(started), error=exc.message)


def aggregate(scores: list[float], aggregation: AnswerAggregation) -> float:
    """Combine claim scores into one answer score.

    Args:
        scores: Per-claim support scores (non-empty).
        aggregation: A score-based aggregation (not ``ratio``).

    Returns:
        The answer score in [0, 1].

    Raises:
        ValueError: If ``scores`` is empty or ``aggregation`` is ``ratio``.
    """
    if not scores:
        raise ValueError("cannot aggregate an empty list of scores")
    if aggregation is AnswerAggregation.MIN:
        return min(scores)
    if aggregation is AnswerAggregation.LOWEST_TWO:
        weakest = sorted(scores)[:2]
        return sum(weakest) / len(weakest)
    if aggregation is AnswerAggregation.MEAN:
        return sum(scores) / len(scores)
    raise ValueError("'ratio' is threshold-based and has no single answer score")


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def build_scorer(config: GroundingConfig) -> Scorer:
    """Build the configured scorer.

    Args:
        config: Grounding configuration.

    Returns:
        A pairwise support scorer or a token detector.

    Raises:
        ConfigurationError: If a model-based scorer is selected but the model
            or the ``nli`` extra is unavailable.
    """
    if config.scorer is GroundingScorerType.LEXICAL:
        return LexicalScorer()
    if config.scorer is GroundingScorerType.TOKEN:
        return TransformersTokenDetector.load(config.token_model,
                                              max_length=config.token_max_length)
    nli = NLIScorer.load(config.nli_model)
    if config.scorer is GroundingScorerType.HYBRID:
        return HybridScorer(nli)
    return nli


def build_verifier(config: GroundingConfig) -> GroundingVerifier:
    """Build a verifier with the configured scorer.

    Args:
        config: Grounding configuration.

    Returns:
        The verifier.

    Raises:
        ConfigurationError: If an NLI-based scorer is selected but cannot load.
    """
    return GroundingVerifier(config, build_scorer(config))
