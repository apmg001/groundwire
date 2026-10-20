"""Score question pairs the way the live semantic cache would.

For every QQP pair we pretend ``question1`` is already in the cache and
``question2`` arrives. The cache **hits** (serves question1's answer) when:

* the normalised questions are identical (exact match), or
* cosine similarity >= threshold, **and** (when guards are on)
  :func:`~groundwire.cache.guards.semantic_mismatch` finds no conflict.

The ground truth says whether serving that answer would have been right:

==================  =======================  ===========================
                    duplicate (same intent)  not duplicate
==================  =======================  ===========================
cache hits          correct reuse (TP)       **wrong answer served** (FP)
cache misses        missed saving (FN)       correct miss (TN)
==================  =======================  ===========================

So the two numbers that matter are the **hit rate on duplicates** (savings)
and the **false-hit rate on non-duplicates** (wrong answers). Both are
independent of how often duplicates occur, unlike precision.

This module reuses the gateway's own normalisation, embedder and guards, so
the results describe the code that runs in production, not a re-implementation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

from groundwire.cache.embeddings import Embedder, Vector, cosine
from groundwire.cache.guards import semantic_mismatch
from groundwire.cache.keys import normalise_query
from groundwire.evals.metrics import BinaryMetrics, confusion
from groundwire.evals.qqp import QuestionPair

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Per-pair scores
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PairScore:
    """How the cache sees one pair.

    Attributes:
        pair: The labelled pair.
        similarity: Cosine similarity of the normalised questions' embeddings.
        exact: The normalised questions are identical (an exact-match hit).
        guard: Guard veto reason, or ``None`` if the guards allow a match.
    """

    pair: QuestionPair
    similarity: float
    exact: bool
    guard: str | None

    def hits(self, threshold: float, *, use_guard: bool = True) -> bool:
        """Would the cache serve question1's answer for question2?

        Args:
            threshold: Similarity threshold.
            use_guard: Apply the number/placeholder guards.

        Returns:
            True on a cache hit.
        """
        if self.exact:
            return True
        if use_guard and self.guard is not None:
            return False
        return self.similarity >= threshold


@dataclass(frozen=True, slots=True)
class EmbeddingStats:
    """Embedding cost observed during scoring.

    Attributes:
        unique_questions: Distinct normalised questions embedded.
        seconds: Wall time spent embedding them (batched).
    """

    unique_questions: int
    seconds: float

    @property
    def per_second(self) -> float:
        """Batched throughput in questions per second."""
        return self.unique_questions / self.seconds if self.seconds > 0 else 0.0


def score_pairs(embedder: Embedder, pairs: Sequence[QuestionPair], *,
                chunk_size: int = 1024) -> tuple[list[PairScore], EmbeddingStats]:
    """Embed every distinct question once and score all pairs.

    Args:
        embedder: Sentence embedder (the same class the gateway uses).
        pairs: Pairs to score.
        chunk_size: Questions per ``embed`` call (progress is logged per chunk).

    Returns:
        ``(scores, stats)``, scores in the order of ``pairs``.

    Raises:
        CacheUnavailableError: If the embedder fails.
        ValueError: If ``chunk_size`` is not positive or the embedder returns
            the wrong number of vectors.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    normalised = [(normalise_query(p.question1), normalise_query(p.question2)) for p in pairs]
    # Similar lengths in one batch = less padding = noticeably faster on CPU.
    unique = sorted({q for pair in normalised for q in pair}, key=lambda q: (len(q), q))
    vectors: dict[str, Vector] = {}
    started = time.perf_counter()
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start:start + chunk_size]
        embedded = embedder.embed(chunk)
        if len(embedded) != len(chunk):
            raise ValueError(f"embedder returned {len(embedded)} vectors for {len(chunk)} texts")
        vectors.update(zip(chunk, embedded, strict=True))
        logger.info("embedded %d / %d questions", min(start + chunk_size, len(unique)),
                    len(unique))
    stats = EmbeddingStats(unique_questions=len(unique), seconds=time.perf_counter() - started)

    scores = [
        PairScore(pair=pair, similarity=cosine(vectors[q1], vectors[q2]), exact=q1 == q2,
                  guard=semantic_mismatch(q2, q1))
        for pair, (q1, q2) in zip(pairs, normalised, strict=True)
    ]
    return scores, stats


def measure_lookup_latency(embedder: Embedder, questions: Sequence[str],
                           *, samples: int = 100) -> list[float]:
    """Time single-question embeddings, as a live cache lookup does them.

    Args:
        embedder: Sentence embedder.
        questions: Pool of questions to time.
        samples: Number of questions to time.

    Returns:
        Latencies in milliseconds.
    """
    timings: list[float] = []
    for question in list(questions)[:samples]:
        started = time.perf_counter()
        embedder.embed([normalise_query(question)])
        timings.append((time.perf_counter() - started) * 1000)
    return timings


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def outcome(scores: Sequence[PairScore], threshold: float, *,
            use_guard: bool = True) -> BinaryMetrics:
    """Confusion matrix of cache decisions (positive = duplicate, predicted = hit).

    ``recall`` is the hit rate on duplicates and ``false_block_rate`` is the
    false-hit rate on non-duplicates (the field name comes from the shared
    metrics type).

    Args:
        scores: Pair scores.
        threshold: Similarity threshold.
        use_guard: Apply the guards.

    Returns:
        The metrics.
    """
    return confusion([s.hits(threshold, use_guard=use_guard) for s in scores],
                     [s.pair.is_duplicate for s in scores])


@dataclass(frozen=True, slots=True)
class TunedThreshold:
    """Threshold chosen for a false-hit budget.

    Attributes:
        budget: Maximum allowed false-hit rate.
        threshold: Lowest threshold that stays within the budget.
        hit_rate: Hit rate on duplicates at that threshold (tuning data).
        false_hit_rate: False-hit rate at that threshold (tuning data).
    """

    budget: float
    threshold: float
    hit_rate: float
    false_hit_rate: float


def threshold_for_false_hit_budget(scores: Sequence[PairScore], budget: float, *,
                                   use_guard: bool = True) -> TunedThreshold:
    """Lowest similarity threshold whose false-hit rate stays within ``budget``.

    Lowering the threshold can only add hits, so the lowest threshold within
    budget also has the highest hit rate. Implemented as one sorted sweep
    (``O(n log n)``), with tied similarities decided together.

    Args:
        scores: Pair scores (tuning split).
        budget: Maximum false-hit rate in [0, 1].
        use_guard: Apply the guards.

    Returns:
        The tuned threshold. If even the strictest semantic threshold exceeds
        the budget (exact matches alone are wrong too often), ``threshold`` is
        ``inf``, i.e. semantic matching off.

    Raises:
        ValueError: If ``scores`` is empty, single-class, or the budget is invalid.
    """
    if not 0 <= budget <= 1:
        raise ValueError("budget must be in [0, 1]")
    positives = sum(s.pair.is_duplicate for s in scores)
    negatives = len(scores) - positives
    if not positives or not negatives:
        raise ValueError("need both duplicate and non-duplicate pairs to tune")

    always = [s for s in scores if s.exact]
    tp = sum(s.pair.is_duplicate for s in always)
    fp = len(always) - tp
    best = TunedThreshold(budget, float("inf"), tp / positives, fp / negatives)
    if fp / negatives > budget:
        return best

    candidates = sorted((s for s in scores if not s.exact
                         and not (use_guard and s.guard is not None)),
                        key=lambda s: s.similarity, reverse=True)
    for similarity, group in groupby(candidates, key=lambda s: s.similarity):
        members = list(group)
        group_tp = sum(s.pair.is_duplicate for s in members)
        if (fp + len(members) - group_tp) / negatives > budget:
            break
        tp += group_tp
        fp += len(members) - group_tp
        best = TunedThreshold(budget, similarity, tp / positives, fp / negatives)
    return best
