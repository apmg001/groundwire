"""Render gate results for terminals, CI step summaries and calibration."""

from __future__ import annotations

from collections.abc import Sequence

import yaml

from groundwire.evals.gate.checks import GateResult, suggest_bound


def _fmt(value: float | None) -> str:
    if value is None:
        return "not measured"
    return f"{value:.0f}" if value.is_integer() and value > 1 else f"{value:.4f}"


def render_markdown(suite: str, results: Sequence[GateResult]) -> str:
    """Markdown report: one table per gate, overall verdict first.

    Args:
        suite: Suite name.
        results: Gate results.

    Returns:
        Markdown text (also suitable for ``$GITHUB_STEP_SUMMARY``).
    """
    passed = all(r.passed for r in results)
    verdict = "✅ PASSED" if passed else "❌ FAILED"
    lines = [f"## Quality gate `{suite}`: {verdict}", ""]
    for r in results:
        icon = "✅" if r.passed else "❌"
        n = r.metrics.get("n")
        size = f", {n:.0f} examples" if n else ""
        lines += [f"### {icon} `{r.gate.name}` ({r.gate.benchmark}{size}, {r.seconds:.1f}s)", ""]
        if r.error is not None:
            lines += [f"> Could not run: {r.error}", "",
                      "A gate that cannot measure counts as failed.", ""]
            continue
        lines += ["| Metric | Measured | Required | Result |", "|---|---|---|---|"]
        for c in r.checks:
            lines.append(f"| {c.metric} | {_fmt(c.value)} | {c.bound.describe()} | "
                         f"{'pass' if c.passed else '**FAIL**'} |")
        lines.append("")
    if not passed:
        lines += ["If a drop is intended (e.g. a deliberate speed/accuracy trade-off), lower the "
                  "bound in `config/quality_gates.yaml` in the same PR, so reviewers see it."]
    return "\n".join(lines).rstrip() + "\n"


def render_calibration(results: Sequence[GateResult], tolerance: float) -> str:
    """YAML ``checks:`` blocks suggested from measured values.

    Args:
        results: Gate results from a calibration run.
        tolerance: Slack applied to each measured value.

    Returns:
        Text to review and paste into the gate file.
    """
    blocks: list[str] = [f"# Suggested bounds (measured value -/+ {tolerance:g}). Review before "
                         "committing: bounds should only move on purpose."]
    for r in results:
        if r.error is not None:
            blocks.append(f"# {r.gate.name}: could not run ({r.error})")
            continue
        measured = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(r.metrics.items()))
        checks = {}
        for metric in r.gate.checks:
            value = r.metrics.get(metric)
            if value is None:
                continue
            bound = suggest_bound(r.gate, metric, value, tolerance)
            checks[metric] = {k: v for k, v in (("min", bound.min), ("max", bound.max))
                              if v is not None}
        blocks.append(f"# {r.gate.name}: measured {measured}\n"
                      + yaml.safe_dump({r.gate.name: {"checks": checks}}, sort_keys=False))
    return "\n".join(blocks)
