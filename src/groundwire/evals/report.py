"""Aggregate benchmark results into JSON and Markdown reports."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from groundwire.evals.metrics import at_threshold, percentile, roc_auc
from groundwire.evals.runner import ExampleResult


def summarise(results: Sequence[ExampleResult], threshold: float) -> dict[str, Any]:
    """Compute metrics for one group of results.

    Args:
        results: Results to summarise (non-empty).
        threshold: Decision threshold on the support score.

    Returns:
        Metrics, ROC-AUC, base rate, latency and error counts.
    """
    scores = [r.support_score for r in results]
    actual = [r.is_hallucinated for r in results]
    latencies = [r.latency_ms for r in results]
    auc = roc_auc(scores, actual)
    rate = sum(actual) / len(actual) if actual else 0.0
    return {
        **at_threshold(scores, actual, threshold).as_dict(),
        "roc_auc": None if auc is None else round(auc, 4),
        "hallucination_rate": round(rate, 4),
        # F1 of a useless detector that flags every response: the bar to beat.
        "f1_flag_everything": round(2 * rate / (1 + rate), 4) if rate else 0.0,
        "latency_ms_p50": round(percentile(latencies, 0.5), 2),
        "latency_ms_p95": round(percentile(latencies, 0.95), 2),
        "avg_claims": round(statistics.fmean(r.claims for r in results), 2) if results else 0,
        "scorer_errors": sum(r.error is not None for r in results),
    }


def build_summary(results: Sequence[ExampleResult], threshold: float,
                  run_info: dict[str, Any]) -> dict[str, Any]:
    """Build the full summary: overall plus one entry per task.

    Args:
        results: All test results.
        threshold: Decision threshold.
        run_info: Settings to record alongside the numbers (scorer, seed...).

    Returns:
        JSON-serialisable summary.
    """
    tasks = sorted({r.task for r in results})
    return {
        "run": run_info,
        "threshold": round(threshold, 4),
        "overall": summarise(results, threshold),
        "by_task": {t: summarise([r for r in results if r.task == t], threshold) for t in tasks},
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as a Markdown report.

    Args:
        summary: Output of :func:`build_summary`.

    Returns:
        Markdown text.
    """
    run = summary["run"]
    rows = [("**All tasks**", summary["overall"]), *summary["by_task"].items()]
    lines = [
        f"# Groundwire hallucination benchmark: {run['dataset']}",
        "",
        f"- **Scorer:** `{run['scorer']}`" + (f" (`{run['nli_model']}`)"
                                                if run.get("nli_model") else ""),
        f"- **Test set:** {run['split']} split, {summary['overall']['n']} responses "
        f"({run['per_task'] or 'all'} per task, seed {run['seed']})",
        f"- **Label policy:** `{run['label_policy']}`",
        f"- **Threshold:** {summary['threshold']} ({run['threshold_source']})",
        f"- **Run date:** {run['date']}",
        "",
        "| Task | Responses | Hallucinated | Precision | Recall | F1 | F1 if flagging "
        "everything | ROC-AUC | Good answers blocked | Latency p50 / p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        lines.append(
            f"| {name} | {m['n']} | {_pct(m['hallucination_rate'])} | {_pct(m['precision'])} "
            f"| {_pct(m['recall'])} | {_pct(m['f1'])} | {_pct(m['f1_flag_everything'])} | "
            f"{auc} "
            f"| {_pct(m['false_block_rate'])} | {m['latency_ms_p50']:.0f} / "
            f"{m['latency_ms_p95']:.0f} ms |"
        )
    lines += [
        "",
        "**How to read this**",
        "",
        "- *Precision*: of the answers Groundwire flagged, how many really were hallucinated.",
        "- *Recall*: of the hallucinated answers, how many Groundwire caught.",
        "- *Good answers blocked*: grounded answers wrongly flagged (false positive rate).",
        "- *F1 if flagging everything*: a detector that blocks every answer. "
        "A real detector must beat this, especially on tasks where most answers "
        "are hallucinated.",
        "- *ROC-AUC*: how well the score separates hallucinated from grounded answers, "
        "independent of the threshold (0.5 = random, 1.0 = perfect).",
        "- The threshold was chosen on the **train** split and applied unchanged to the "
        "**test** split, so these numbers are not tuned on the data they report."
        if run["threshold_source"].startswith("tuned") else
        "- The threshold was set manually, not tuned.",
    ]
    errors = summary["overall"]["scorer_errors"]
    if errors:
        lines += ["", f"> ⚠️ {errors} responses could not be scored and were counted as "
                      "not flagged."]
    return "\n".join(lines) + "\n"
