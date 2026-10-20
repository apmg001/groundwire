"""Summaries and Markdown reports for the semantic-cache benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from groundwire.evals.cachebench.scoring import (
    EmbeddingStats,
    PairScore,
    TunedThreshold,
    outcome,
)
from groundwire.evals.metrics import BinaryMetrics, percentile, roc_auc

DEFAULT_SWEEP: tuple[float, ...] = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _cache_metrics(metrics: BinaryMetrics) -> dict[str, float | int]:
    """Rename shared metrics into cache vocabulary.

    Args:
        metrics: Confusion metrics (positive = duplicate, predicted = hit).

    Returns:
        JSON-serialisable dict.
    """
    return {
        "hit_rate": round(metrics.recall, 4),
        "false_hit_rate": round(metrics.false_block_rate, 4),
        "precision": round(metrics.precision, 4),
        "correct_hits": metrics.tp, "wrong_hits": metrics.fp,
        "missed": metrics.fn, "correct_misses": metrics.tn,
    }


def _example(score: PairScore) -> dict[str, Any]:
    """One pair for error analysis.

    Args:
        score: Pair score.

    Returns:
        JSON-serialisable dict.
    """
    return {"id": score.pair.id, "cached": score.pair.question1,
            "incoming": score.pair.question2, "similarity": round(score.similarity, 4),
            "duplicate": score.pair.is_duplicate, "guard": score.guard}


def build_summary(scores: Sequence[PairScore], *, tuned: Sequence[TunedThreshold],
                  configured_threshold: float, stats: EmbeddingStats,
                  latency_ms: Sequence[float], run_info: dict[str, Any],
                  sweep: Sequence[float] = DEFAULT_SWEEP, examples: int = 10) -> dict[str, Any]:
    """Aggregate test-split results.

    Args:
        scores: Test pair scores.
        tuned: Thresholds tuned on the train split.
        configured_threshold: The gateway's shipped ``similarity_threshold``.
        stats: Embedding throughput on the test split.
        latency_ms: Single-question embedding latencies.
        run_info: Run settings to record.
        sweep: Fixed thresholds to tabulate.
        examples: Error-analysis examples to keep per category.

    Returns:
        JSON-serialisable summary.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("no scores to summarise")
    duplicates = sum(s.pair.is_duplicate for s in scores)
    auc = roc_auc([-s.similarity for s in scores], [s.pair.is_duplicate for s in scores])

    allowed = [s for s in scores if not s.exact and s.guard is None and not s.pair.is_duplicate]
    vetoed = [s for s in scores if not s.exact and s.guard is not None]
    return {
        "run": run_info,
        "pairs": len(scores),
        "duplicate_rate": round(duplicates / len(scores), 4),
        "exact_matches": sum(s.exact for s in scores),
        "roc_auc": None if auc is None else round(auc, 4),
        "configured": {
            "threshold": configured_threshold,
            "with_guard": _cache_metrics(outcome(scores, configured_threshold)),
            "without_guard": _cache_metrics(outcome(scores, configured_threshold,
                                                    use_guard=False)),
        },
        "tuned": [{
            "budget": t.budget,
            "threshold": None if t.threshold == float("inf") else round(t.threshold, 4),
            "train_hit_rate": round(t.hit_rate, 4),
            "train_false_hit_rate": round(t.false_hit_rate, 4),
            "test": _cache_metrics(outcome(scores, t.threshold)),
        } for t in tuned],
        "sweep": [{
            "threshold": threshold,
            "with_guard": _cache_metrics(outcome(scores, threshold)),
            "without_guard": _cache_metrics(outcome(scores, threshold, use_guard=False)),
        } for threshold in sweep],
        "guard": {
            "vetoed_pairs": len(vetoed),
            "vetoed_non_duplicates": sum(not s.pair.is_duplicate for s in vetoed),
            "vetoed_duplicates": sum(s.pair.is_duplicate for s in vetoed),
            "by_reason": {reason: sum(s.guard == reason for s in vetoed)
                          for reason in sorted({s.guard for s in vetoed if s.guard})},
        },
        "embedding": {
            "unique_questions": stats.unique_questions,
            "batched_questions_per_s": round(stats.per_second, 1),
            "lookup_ms_p50": round(percentile(latency_ms, 0.5), 2),
            "lookup_ms_p95": round(percentile(latency_ms, 0.95), 2),
        },
        "hardest_wrong_matches": [_example(s) for s in
                                  sorted(allowed, key=lambda s: -s.similarity)[:examples]],
        "guard_saves": [_example(s) for s in
                        sorted((s for s in vetoed if not s.pair.is_duplicate),
                               key=lambda s: -s.similarity)[:examples]],
    }


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _row(label: str, m: dict[str, Any]) -> str:
    return (f"| {label} | {_pct(m['hit_rate'])} | {_pct(m['false_hit_rate'])} | "
            f"{_pct(m['precision'])} | {m['correct_hits']} | {m['wrong_hits']} |")


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as Markdown.

    Args:
        summary: Output of :func:`build_summary`.

    Returns:
        Markdown text.
    """
    run = summary["run"]
    header = ("| Threshold | Hit rate (duplicates reused) | False-hit rate (wrong answers) "
              "| Precision of hits | Correct hits | Wrong hits |")
    rule = "|---|---|---|---|---|---|"
    auc = summary["roc_auc"]
    lines = [
        "# Groundwire semantic-cache benchmark: Quora Question Pairs",
        "",
        f"- **Embedding model:** `{run['model']}`",
        f"- **Test set:** GLUE QQP validation, {summary['pairs']} pairs "
        f"({_pct(summary['duplicate_rate'])} duplicates), seed {run['seed']}",
        f"- **Tuning set:** {run['tune_pairs'] or 'none'} pairs sampled from QQP train",
        f"- **Similarity ROC-AUC:** {'n/a' if auc is None else f'{auc:.3f}'} "
        "(how well similarity separates duplicates from non-duplicates)",
        f"- **Exact matches after normalisation:** {summary['exact_matches']}",
        f"- **Run date:** {run['date']}",
        "",
    ]
    if summary["tuned"]:
        lines += [
            "## Thresholds tuned on train, measured on validation",
            "",
            "| False-hit budget | Threshold (from train) | Hit rate | False-hit rate "
            "| Precision of hits | Correct hits | Wrong hits |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in summary["tuned"]:
            m = t["test"]
            threshold = "off" if t["threshold"] is None else f"{t['threshold']:.4f}"
            lines.append(f"| ≤ {_pct(t['budget'])} | {threshold} | {_pct(m['hit_rate'])} | "
                         f"{_pct(m['false_hit_rate'])} | {_pct(m['precision'])} | "
                         f"{m['correct_hits']} | {m['wrong_hits']} |")
        lines.append("")

    configured = summary["configured"]
    lines += [
        f"## Shipped default (`similarity_threshold: {configured['threshold']}`)",
        "",
        header, rule,
        _row("with guards", configured["with_guard"]),
        _row("similarity only", configured["without_guard"]),
        "",
        "## Threshold sweep (with guards)",
        "",
        header, rule,
        *(_row(f"{s['threshold']:.2f}", s["with_guard"]) for s in summary["sweep"]),
        "",
    ]
    guard = summary["guard"]
    emb = summary["embedding"]
    lines += [
        "## Guards and cost",
        "",
        f"- Guards vetoed **{guard['vetoed_pairs']}** pairs: "
        f"{guard['vetoed_non_duplicates']} non-duplicates (wrong answers prevented, if they "
        f"were above the threshold) and {guard['vetoed_duplicates']} duplicates (savings "
        f"lost). By reason: {guard['by_reason'] or 'none'}.",
        f"- Embedding cost: **{emb['lookup_ms_p50']:.1f} ms p50 / {emb['lookup_ms_p95']:.1f} "
        f"ms p95** per lookup (one question, CPU); {emb['batched_questions_per_s']:.0f} "
        "questions/s batched.",
        "",
        "## Hardest wrong matches (non-duplicates the guards allow, by similarity)",
        "",
        "| Similarity | Cached question | Incoming question |",
        "|---|---|---|",
        *(f"| {e['similarity']:.3f} | {_cell(e['cached'])} | {_cell(e['incoming'])} |"
          for e in summary["hardest_wrong_matches"]),
        "",
        "**How to read this**",
        "",
        "- *Hit rate*: of question pairs that humans judged to be the same question, how "
        "often the cache reused the answer (money saved).",
        "- *False-hit rate*: of pairs that are **different** questions, how often the cache "
        "served the wrong answer anyway. This is the number to keep small.",
        "- *Precision of hits*: of all answers served from cache, how many were right. It "
        "depends on how often users really repeat themselves, so on your traffic it will "
        "differ from QQP's.",
        "- Thresholds in the first table were chosen on the **train** split and applied "
        "unchanged to **validation**.",
        "- QQP labels are human judgements with known noise; some \"wrong matches\" above "
        "are arguably duplicates. Treat the numbers as a careful estimate, not ground truth.",
    ]
    return "\n".join(lines) + "\n"


def _cell(text: str, limit: int = 90) -> str:
    """Escape and shorten text for a Markdown table cell.

    Args:
        text: Raw text.
        limit: Maximum characters.

    Returns:
        Safe cell text.
    """
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"
