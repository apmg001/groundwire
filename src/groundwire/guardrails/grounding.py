"""Grounding verification: is the answer supported by the supplied context?

Algorithm:

1. Split the answer into claims (sentences / bullets).
2. Drop non-claims: sentences with fewer than ``min_claim_tokens`` content
   words ("Sure!", "Here you go:").
3. Chunk the context (sentence-aligned, overlapping, with section headings;
   see :mod:`groundwire.guardrails.evidence`).
4. Score each claim's support:

   * Exhaustive: score every (chunk, claim) pair and keep the best chunk.
     Used when ``evidence_passages = 0`` and the context is short
     (``evidence.all_pairs_max_chunks``).
   * Retrieved: search all chunks with the claim itself as the query
     (BM25, optionally re-ranked by meaning). For long contexts each of the
     top ``k`` hits is scored as its own window (the hit plus neighbours) and
     the best window wins, so a fact on page 60 is found instead of cut off,
     and a number from an unrelated hit cannot "support" a wrong figure.
     ``evidence_passages = k`` keeps its original meaning: the ``k`` hits
     joined and scored once (lets NLI combine facts across them).
   * Token detectors read one prompt: a context that fits their window is
     passed whole (exactly as the model was benchmarked); a longer one is
     reduced to every claim's evidence, fairly shared within the window.

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
    EvidenceRetrieverType,
    GroundingConfig,
    GroundingFailurePolicy,
    GroundingScorerType,
)
from groundwire.core.errors import ConfigurationError, GuardrailUnavailableError
from groundwire.guardrails.evidence import (
    BM25Retriever,
    Chunk,
    ChunkedContext,
    Chunker,
    EvidenceSelector,
    HybridRetriever,
    Retriever,
    SentenceChunker,
    TextEncoder,
)
from groundwire.guardrails.scorers import (
    HybridScorer,
    LexicalScorer,
    NLIScorer,
    SupportScorer,
)
from groundwire.guardrails.text import content_tokens, split_claims
from groundwire.guardrails.token_detector import (
    TokenDetector,
    TransformersTokenDetector,
    build_prompt,
    max_prob_in_range,
)

Scorer = SupportScorer | TokenDetector

logger = logging.getLogger(__name__)

# Bound work per request so one huge answer cannot monopolise the CPU. Context
# size is bounded by ``evidence.max_chunks`` (retrieval keeps the cost per claim small).
MAX_CLAIMS = 64

# Conservative characters per token for budgeting a token detector's window
# (English averages ~4; lower = safer, since overflowing text is cut).
CHARS_PER_TOKEN = 3.0
PROMPT_OVERHEAD_CHARS = 600  # template text, passage labels, question


class EvidenceMode(StrEnum):
    """How a report's evidence was chosen."""

    NONE = "none"  # nothing to check (no claims or empty context)
    ALL_CHUNKS = "all_chunks"  # every claim scored against every chunk
    RETRIEVED = "retrieved"  # each claim scored against its retrieved evidence
    FULL_CONTEXT = "full_context"  # token detector read the whole context
    RETRIEVED_FOR_WINDOW = "retrieved_for_window"  # token detector read claims' evidence
    CLIENT_PROMPT = "client_prompt"  # token detector read a prompt supplied verbatim


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
        score: Support score (best chunk, or the claim's joined evidence).
        supported: Whether ``score`` met the threshold.
        best_passage: Index of the best-matching chunk (-1 if unknown).
        document: Context document containing that chunk (-1 if unknown).
            Clients that send one context entry per page get the page here.
    """

    claim: str
    score: float
    supported: bool
    best_passage: int
    document: int = -1


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
        truncated: True if claims or context chunks were capped.
        error: Scorer error message when ``verdict`` is ``unverified``.
        context_chunks: Chunks the context was split into.
        evidence_mode: How evidence was chosen (see :class:`EvidenceMode`).
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
    context_chunks: int = 0
    evidence_mode: str = EvidenceMode.NONE.value

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


@dataclass(frozen=True, slots=True)
class _Plan:
    """Pairs to score for a pairwise scorer, and what each pair refers to.

    Every mode is the same shape: each pair belongs to one claim (``owners``)
    and points at one chunk (``targets``); a claim's support is its best pair.
    """

    pairs: list[tuple[str, str]]
    owners: list[int]
    targets: list[Chunk | None]
    mode: EvidenceMode
    context: ChunkedContext


class GroundingVerifier:
    """Checks answers against context using a pluggable scorer.

    Collaborators are injected (defaults built from config): a
    :class:`~groundwire.guardrails.evidence.Chunker`, a
    :class:`~groundwire.guardrails.evidence.Retriever` and an
    :class:`~groundwire.guardrails.evidence.EvidenceSelector`. All per-request
    state (chunks, index) lives in local variables and is discarded.
    """

    def __init__(self, config: GroundingConfig, scorer: Scorer, *,
                 chunker: Chunker | None = None, retriever: Retriever | None = None,
                 selector: EvidenceSelector | None = None, max_concurrency: int = 2) -> None:
        """Create the verifier.

        Args:
            config: Thresholds, evidence settings and failure policy.
            scorer: Pairwise support scorer, or a token detector that reads the
                whole prompt at once.
            chunker: Splits context into chunks (default: from ``config.evidence``).
            retriever: Finds each claim's evidence (default: BM25).
            selector: Builds evidence windows (default: from ``config``).
            max_concurrency: Max concurrent verifications. Scoring is CPU-bound,
                so running dozens at once only thrashes the CPU; excess requests
                queue here instead.
        """
        evidence = config.evidence
        self._config = config
        self._scorer = scorer
        self._chunker = chunker or SentenceChunker(
            max_chars=evidence.chunk_chars, overlap_sentences=evidence.overlap_sentences,
            attach_headings=evidence.attach_headings, max_chunks=evidence.max_chunks)
        self._retriever = retriever or BM25Retriever()
        self._selector = selector or EvidenceSelector(
            top_k=config.evidence_passages or evidence.top_k, neighbours=evidence.neighbours,
            max_chars=evidence.claim_evidence_chars)
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
            context: Source documents supplied by the client (any length; one
                entry per page gives page-level evidence references).
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
        claims_truncated = len(claims) > MAX_CLAIMS
        claims = claims[:MAX_CLAIMS]
        if not claims:
            return GroundingReport(verdict=Verdict.PASSED, scorer=self._scorer.name,
                                   aggregation=self._config.aggregation.value,
                                   latency_ms=_ms_since(started), truncated=claims_truncated)
        try:
            async with self._limit:
                if isinstance(self._scorer, TokenDetector):
                    return await asyncio.to_thread(
                        self._verify_with_tokens, self._scorer, answer, claims, context,
                        question, prompt, started, claims_truncated)
                return await asyncio.to_thread(self._verify_pairwise, self._scorer, claims,
                                               context, started, claims_truncated)
        except GuardrailUnavailableError as exc:
            return self._on_scorer_failure(exc, started)
        except Exception as exc:  # noqa: BLE001 - any scorer crash is a guardrail outage
            return self._on_scorer_failure(GuardrailUnavailableError(str(exc)), started)

    # ------------------------------------------------------------------ #
    # Pairwise scorers (lexical, NLI, hybrid)
    # ------------------------------------------------------------------ #

    def _verify_pairwise(self, scorer: SupportScorer, claims: list[str], context: list[str],
                         started: float, claims_truncated: bool) -> GroundingReport:
        """Plan pairs, score them, and build the report (runs in a worker thread)."""
        plan = self._plan(claims, context)
        truncated = claims_truncated or plan.context.truncated
        if not plan.context.chunks:
            # Context contained no text: nothing can be supported.
            return self._report([ClaimResult(c, 0.0, False, -1) for c in claims], started,
                                truncated, plan.context, EvidenceMode.NONE)
        scores = scorer.score_batch(plan.pairs)
        if len(scores) != len(plan.pairs):
            raise GuardrailUnavailableError("scorer returned wrong number of scores")

        threshold = self._config.support_threshold
        best: list[tuple[float, Chunk | None]] = [(0.0, None)] * len(claims)
        for owner, score, target in zip(plan.owners, scores, plan.targets, strict=True):
            if best[owner][1] is None or score > best[owner][0]:
                best[owner] = (score, target)
        results = [_claim_result(claim, score, threshold, target)
                   for claim, (score, target) in zip(claims, best, strict=True)]
        return self._report(results, started, truncated, plan.context, plan.mode)

    def _plan(self, claims: list[str], context: list[str]) -> _Plan:
        """Chunk the context and decide which (evidence, claim) pairs to score.

        Args:
            claims: Claims to check.
            context: Source documents.

        Returns:
            Pairs to score plus what they refer to.
        """
        chunked = self._chunker.chunk(context)
        chunks = chunked.chunks
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        targets: list[Chunk | None] = []
        joined = self._config.evidence_passages > 0
        exhaustive = (not joined
                      and len(chunks) <= self._config.evidence.all_pairs_max_chunks)
        if exhaustive or not chunks:
            for owner, claim in enumerate(claims):
                for chunk in chunks:
                    pairs.append((chunk.scoring_text, claim))
                    owners.append(owner)
                    targets.append(chunk)
            return _Plan(pairs, owners, targets, EvidenceMode.ALL_CHUNKS, chunked)
        index = self._retriever.build(chunked)
        for owner, claim in enumerate(claims):
            windows = ([self._selector.for_claim(index, claim)] if joined
                       else self._selector.windows(index, claim))
            for evidence in windows or [self._selector.for_claim(index, claim)]:
                pairs.append((evidence.text, claim))
                owners.append(owner)
                targets.append(evidence.best)
        return _Plan(pairs, owners, targets, EvidenceMode.RETRIEVED, chunked)

    # ------------------------------------------------------------------ #
    # Token detectors (LettuceDetect)
    # ------------------------------------------------------------------ #

    def _verify_with_tokens(self, detector: TokenDetector, answer: str, claims: list[str],
                            context: list[str], question: str | None, prompt: str | None,
                            started: float, claims_truncated: bool) -> GroundingReport:
        """Score claims from per-token hallucination probabilities (worker thread).

        A claim's support is ``1 - max(token probability)`` over the answer
        tokens inside it, so one confidently hallucinated word marks the claim.
        """
        chunked = ChunkedContext((), len(context))
        if prompt is not None:
            mode = EvidenceMode.CLIENT_PROMPT
        else:
            budget = self._token_budget(answer, question)
            if sum(len(doc) for doc in context) <= budget:
                mode, prompt = EvidenceMode.FULL_CONTEXT, build_prompt(context, question)
            else:
                chunked = self._chunker.chunk(context)
                index = self._retriever.build(chunked)
                evidence = self._selector.for_answer(index, claims, budget)
                prompt = build_prompt([c.scoring_text for c in evidence], question)
                mode = EvidenceMode.RETRIEVED_FOR_WINDOW
        tokens = detector.detect(prompt, answer)

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
        return self._report(results, started, claims_truncated or chunked.truncated,
                            chunked, mode)

    def _token_budget(self, answer: str, question: str | None) -> int:
        """Characters of context that fit the detector's window with the answer."""
        configured = self._config.evidence.token_context_chars
        if configured is not None:
            return configured
        window = int(self._config.token_max_length * CHARS_PER_TOKEN)
        return max(500, window - len(answer) - len(question or "") - PROMPT_OVERHEAD_CHARS)

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def _report(self, results: list[ClaimResult], started: float, truncated: bool,
                context: ChunkedContext, mode: EvidenceMode) -> GroundingReport:
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
                               truncated=truncated, context_chunks=len(context.chunks),
                               evidence_mode=mode.value)

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


def _claim_result(claim: str, score: float, threshold: float, best: Chunk | None) -> ClaimResult:
    return ClaimResult(claim, score, score >= threshold,
                       best.index if best else -1, best.document if best else -1)


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


def build_retriever(config: GroundingConfig, encoder: TextEncoder | None = None) -> Retriever:
    """Build the configured evidence retriever.

    Args:
        config: Grounding configuration.
        encoder: Sentence encoder for ``hybrid`` (the composition root supplies it).

    Returns:
        The retriever.

    Raises:
        ConfigurationError: If ``hybrid`` is configured without an encoder.
    """
    evidence = config.evidence
    if evidence.retriever is EvidenceRetrieverType.HYBRID:
        if encoder is None:
            raise ConfigurationError("grounding.evidence.retriever 'hybrid' needs a sentence "
                                     "encoder (install the nli extra: make install-nli)")
        return HybridRetriever(encoder, candidates=evidence.hybrid_candidates)
    return BM25Retriever()


def build_verifier(config: GroundingConfig, *, encoder: TextEncoder | None = None
                   ) -> GroundingVerifier:
    """Build a verifier with the configured scorer and evidence retriever.

    Args:
        config: Grounding configuration.
        encoder: Sentence encoder, required when ``evidence.retriever`` is ``hybrid``.

    Returns:
        The verifier.

    Raises:
        ConfigurationError: If a model-based scorer cannot load, or ``hybrid``
            retrieval has no encoder.
    """
    return GroundingVerifier(config, build_scorer(config),
                             retriever=build_retriever(config, encoder))
