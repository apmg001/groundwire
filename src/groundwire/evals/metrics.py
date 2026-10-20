"""Classification metrics for hallucination detection.

Positive class = *hallucinated*. A detector that flags everything gets perfect
recall and useless precision, so precision, recall and F1 are reported
together, plus ROC-AUC, which measures how well the score ranks hallucinated
responses above grounded ones regardless of any threshold.

Score convention: each response gets a *support score* in [0, 1] (the weakest
claim's support). Lower means more suspicious, so a response is predicted
hallucinated when ``score < threshold``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Confusion counts and derived metrics.

    Attributes:
        tp: Hallucinated responses correctly flagged.
        fp: Grounded responses wrongly flagged (blocked good answers).
        tn: Grounded responses correctly passed.
        fn: Hallucinated responses missed.
    """

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        """Number of examples."""
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        """Of the responses flagged, the fraction truly hallucinated."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Of the hallucinated responses, the fraction flagged."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        """Fraction of correct predictions."""
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def false_block_rate(self) -> float:
        """Fraction of grounded answers wrongly blocked (false positive rate)."""
        negatives = self.fp + self.tn
        return self.fp / negatives if negatives else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return counts and metrics rounded for reporting."""
        return {
            "n": self.n, "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "accuracy": round(self.accuracy, 4),
            "false_block_rate": round(self.false_block_rate, 4),
        }


def confusion(predicted: Sequence[bool], actual: Sequence[bool]) -> BinaryMetrics:
    """Build a confusion matrix.

    Args:
        predicted: Predicted hallucination flags.
        actual: Ground-truth flags.

    Returns:
        The metrics.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    tp = sum(p and a for p, a in zip(predicted, actual, strict=True))
    fp = sum(p and not a for p, a in zip(predicted, actual, strict=True))
    fn = sum(a and not p for p, a in zip(predicted, actual, strict=True))
    return BinaryMetrics(tp=tp, fp=fp, tn=len(actual) - tp - fp - fn, fn=fn)


def at_threshold(scores: Sequence[float], actual: Sequence[bool],
                 threshold: float) -> BinaryMetrics:
    """Metrics when flagging every response with ``score < threshold``.

    Args:
        scores: Support scores (lower = more suspicious).
        actual: Ground-truth flags.
        threshold: Decision threshold.

    Returns:
        The metrics.
    """
    return confusion([s < threshold for s in scores], actual)


# --------------------------------------------------------------------------- #
# Threshold selection and ranking quality
# --------------------------------------------------------------------------- #


def best_f1_threshold(scores: Sequence[float], actual: Sequence[bool]) -> tuple[float, float]:
    """Find the threshold that maximises F1 on the given data.

    Candidates are midpoints between consecutive distinct scores (plus the
    extremes), so every distinct decision boundary is tried.

    Args:
        scores: Support scores.
        actual: Ground-truth flags.

    Returns:
        ``(threshold, f1)``. Use it on a *different* split than you report on,
        otherwise the result is optimistic.

    Raises:
        ValueError: If inputs are empty or of different lengths.
    """
    if not scores or len(scores) != len(actual):
        raise ValueError("scores must be non-empty and match actual in length")
    distinct = sorted(set(scores))
    candidates = [distinct[0]] + [
        (a + b) / 2 for a, b in pairwise(distinct)
    ] + [distinct[-1] + 1e-9]
    best_threshold, best_f1 = candidates[0], -1.0
    for threshold in candidates:
        f1 = at_threshold(scores, actual, threshold).f1
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold, best_f1


def roc_auc(scores: Sequence[float], actual: Sequence[bool]) -> float | None:
    """Probability a random hallucinated response scores lower than a grounded one.

    Computed with the Mann-Whitney U statistic (ties count half). 0.5 is
    random guessing, 1.0 is a perfect ranking.

    Args:
        scores: Support scores (lower = more suspicious).
        actual: Ground-truth flags.

    Returns:
        ROC-AUC, or ``None`` if only one class is present.
    """
    positives = [s for s, a in zip(scores, actual, strict=True) if a]
    negatives = [s for s, a in zip(scores, actual, strict=True) if not a]
    if not positives or not negatives:
        return None
    # Rank all scores ascending with average ranks for ties.
    ordered = sorted((s, a) for s, a in zip(scores, actual, strict=True))
    ranks: list[float] = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    rank_sum_neg = sum(r for r, (_, a) in zip(ranks, ordered, strict=True) if not a)
    n_pos, n_neg = len(positives), len(negatives)
    # U for "negative scores above positive scores" = AUC of the lower-is-positive rule.
    u = rank_sum_neg - n_neg * (n_neg + 1) / 2
    return u / (n_pos * n_neg)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Args:
        values: Values (any order).
        fraction: Percentile in [0, 1].

    Returns:
        The percentile, or 0.0 for empty input.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))]
