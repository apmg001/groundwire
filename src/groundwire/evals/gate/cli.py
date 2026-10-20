"""Command-line entry point for quality gates.

Examples::

    # What CI runs on every push (no ML dependencies, seconds)
    python -m groundwire.evals.gate --suite fast

    # Model-based gates (needs make install-nli and the QQP data)
    python -m groundwire.evals.gate --suite full

    # Measure and print suggested bounds instead of enforcing them
    python -m groundwire.evals.gate --suite full --calibrate

The Markdown report goes to stdout and, inside GitHub Actions, is appended to
``$GITHUB_STEP_SUMMARY`` so it shows on the run page. A JSON copy is written
to ``results/gates/<suite>.json`` for artifacts.

Exit codes: 0 all gates passed, 1 a gate failed (or could not run),
2 invalid gate file or arguments, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from groundwire.core.errors import ConfigurationError
from groundwire.core.logging import configure_logging
from groundwire.evals.gate.checks import DataDirs, GateRunner, default_runner, run_gates
from groundwire.evals.gate.report import render_calibration, render_markdown
from groundwire.evals.gate.spec import load_spec

logger = logging.getLogger("groundwire.evals.gate")

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="python -m groundwire.evals.gate",
                                     description="Run benchmark quality gates.")
    parser.add_argument("--suite", default="fast", help="suite to run (default: fast)")
    parser.add_argument("--spec", type=Path, default=Path("config/quality_gates.yaml"),
                        help="gate file (default: %(default)s)")
    parser.add_argument("--ragtruth-dir", type=Path, default=Path("data/ragtruth"))
    parser.add_argument("--qqp-dir", type=Path, default=Path("data/qqp"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/gates"),
                        help="where the JSON results go (default: %(default)s)")
    parser.add_argument("--calibrate", action="store_true",
                        help="print suggested bounds from measured values; do not enforce")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="slack for --calibrate suggestions (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    if not 0 <= args.tolerance <= 0.5:
        parser.error("--tolerance must be between 0 and 0.5")
    return args


def _write_step_summary(markdown: str) -> None:
    """Append to the GitHub Actions job summary when running in CI.

    Args:
        markdown: Report text.
    """
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    try:
        with Path(target).open("a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    except OSError as exc:  # a summary is a nicety; never fail the gate over it
        logger.warning("could not write the step summary: %s", exc)


def main(argv: list[str] | None = None, *, runner: GateRunner | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).
        runner: Benchmark runner override (tests).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO", json_output=False)
    try:
        gates = load_spec(args.spec).suite(args.suite)
    except ConfigurationError as exc:
        logger.error("%s", exc.message)
        return EXIT_USAGE

    run = runner or default_runner(DataDirs(ragtruth=args.ragtruth_dir, qqp=args.qqp_dir))
    try:
        results = asyncio.run(run_gates(gates, run))
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_INTERRUPTED

    if args.calibrate:
        sys.stdout.write("\n" + render_calibration(results, args.tolerance) + "\n")
        return EXIT_OK if all(r.error is None for r in results) else EXIT_FAILED

    markdown = render_markdown(args.suite, results)
    sys.stdout.write("\n" + markdown)
    _write_step_summary(markdown)
    try:
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{args.suite}.json").write_text(
            json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write JSON results: %s", exc)
    return EXIT_OK if all(r.passed for r in results) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
