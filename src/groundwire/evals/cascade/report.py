"""Summaries and Markdown reports for the cascade simulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from groundwire.evals.cascade.simulate import (
    CascadeCase,
    PolicyOutcome,
    cascade,
    cheap_only,
    detector_quality,
    oracle,
    strong_only,
)


def _policy(outcome: PolicyOutcome, ratios: Sequence[float]) -> dict[str, Any]:
    return {
        "policy": outcome.policy, "prompts": outcome.prompts,
        "hallucination_rate": round(outcome.hallucination_rate, 4),
        "escalation_rate": round(outcome.escalation_rate, 4),
        "hallucinated": outcome.hallucinated, "escalated": outcome.escalated,
        "relative_cost": {str(r): round(outcome.relative_cost(r), 4) for r in ratios},
    }


def build_summary(cases: Sequence[CascadeCase], *, threshold: float, ratios: Sequence[float],
                  sweep: Sequence[float], run_info: dict[str, Any]) -> dict[str, Any]:
    """Compare policies overall, per task and across thresholds.

    Args:
        cases: Cascade cases.
        threshold: The cascade's escalation threshold.
        ratios: Strong/cheap price ratios to cost.
        sweep: Thresholds for the trade-off table.
        run_info: Run settings to record.

    Returns:
        JSON-serialisable summary.

    Raises:
        ValueError: If there are no cases or no price ratios.
    """
    if not cases or not ratios:
        raise ValueError("need at least one case and one price ratio")
    tasks = sorted({c.task for c in cases})
    quality = detector_quality(cases, threshold)
    return {
        "run": run_info,
        "threshold": threshold,
        "price_ratios": list(ratios),
        "policies": [_policy(p, ratios) for p in (
            cheap_only(cases), cascade(cases, threshold), oracle(cases), strong_only(cases))],
        "detector": {"caught": quality.caught, "missed": quality.missed,
                     "wasted": quality.wasted, "kept": quality.kept},
        "by_task": {task: [_policy(p, ratios) for p in (
            cheap_only(subset), cascade(subset, threshold), strong_only(subset))]
                    for task in tasks
                    for subset in [[c for c in cases if c.task == task]]},
        "sweep": [_policy(cascade(cases, t), ratios) | {"threshold": t} for t in sweep],
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _cost(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as Markdown.

    Args:
        summary: Output of :func:`build_summary`.

    Returns:
        Markdown text.
    """
    run = summary["run"]
    ratios = [str(r) for r in summary["price_ratios"]]
    cost_head = " | ".join(f"Cost vs strong-only (R={float(r):g})" for r in ratios)
    cost_rule = "|---" * len(ratios)
    names = {"cheap_only": "Cheap model only", "cascade": "**Cascade (Groundwire)**",
             "oracle": "Perfect detector (ceiling)", "strong_only": "Strong model only"}

    def row(p: dict[str, Any]) -> str:
        costs = " | ".join(_cost(p["relative_cost"][r]) for r in ratios)
        return (f"| {names[p['policy']]} | {_pct(p['hallucination_rate'])} | "
                f"{_pct(p['escalation_rate'])} | {costs} |")

    pol = {p["policy"]: p for p in summary["policies"]}
    mid = ratios[len(ratios) // 2]
    casc, strong, cheap = pol["cascade"], pol["strong_only"], pol["cheap_only"]
    det = summary["detector"]
    lines = [
        "# Groundwire cascade simulation: RAGTruth",
        "",
        f"- **Cheap model:** `{run['cheap']}` → **strong model:** `{run['strong']}`",
        f"- **Escalation check:** `{run['scorer']}` scorer, escalate when the answer score "
        f"< {summary['threshold']}",
        f"- **Prompts:** {cheap['prompts']} RAGTruth test prompts, answered by both models, "
        f"labelled by humans",
        f"- **Run date:** {run['date']}",
        "",
        f"**Result (R={float(mid):g}):** the cascade delivers "
        f"{_pct(casc['hallucination_rate'])} hallucinated answers (cheap only: "
        f"{_pct(cheap['hallucination_rate'])}, strong only: {_pct(strong['hallucination_rate'])}) "
        f"at {_cost(casc['relative_cost'][mid])} of the strong-only cost, escalating "
        f"{_pct(casc['escalation_rate'])} of requests.",
        "",
        f"| Policy | Hallucinated answers delivered | Paid for both models | {cost_head} |",
        f"|---|---|---{cost_rule}|",
        *(row(p) for p in summary["policies"]),
        "",
        "## Escalation decisions",
        "",
        f"- **Caught:** {det['caught']} hallucinated cheap answers replaced by the strong model",
        f"- **Missed:** {det['missed']} hallucinated cheap answers delivered anyway",
        f"- **Wasted:** {det['wasted']} good cheap answers escalated (paid twice for nothing)",
        f"- **Kept:** {det['kept']} good cheap answers delivered at cheap cost",
        "",
        "## By task",
        "",
        f"| Task | Policy | Hallucinated | Paid for both | Cost (R={float(mid):g}) |",
        "|---|---|---|---|---|",
    ]
    for task, policies in summary["by_task"].items():
        for p in policies:
            lines.append(f"| {task} | {names[p['policy']]} | {_pct(p['hallucination_rate'])} | "
                         f"{_pct(p['escalation_rate'])} | {_cost(p['relative_cost'][mid])} |")
    lines += [
        "",
        "## Threshold trade-off",
        "",
        f"| Threshold | Hallucinated | Paid for both | Cost (R={float(mid):g}) |",
        "|---|---|---|---|",
        *(f"| {s['threshold']:g} | {_pct(s['hallucination_rate'])} | "
          f"{_pct(s['escalation_rate'])} | {_cost(s['relative_cost'][mid])} |"
          for s in summary["sweep"]),
        "",
        "**How to read this**",
        "",
        "- *R* is how many times more a strong-model call costs than a cheap one. Cost is "
        "relative to sending everything to the strong model (100%).",
        "- A cascade request always pays for the cheap call, plus the strong call when it "
        "escalates; escalated requests are also slower (two generations plus the check).",
        "- *Perfect detector* escalates exactly the answers humans marked as hallucinated: the "
        "best any fact-checker could do with these two models.",
        "- Thresholds come from earlier tuning on the RAGTruth **train** split; these numbers "
        "are on the **test** split.",
        "- RAGTruth's models are from 2023. Newer small models hallucinate less, which would "
        "mean fewer escalations; re-run with your own models' answers before deciding.",
    ]
    return "\n".join(lines) + "\n"
