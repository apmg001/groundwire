"""Command-line entry point for the cascade simulation.

Examples::

    # Local Llama-2-7B first, GPT-4 when the lexical check fails (no ML deps)
    python -m groundwire.evals.cascade

    # LettuceDetect as the escalation check (needs make install-nli)
    python -m groundwire.evals.cascade --scorer token

    # Cheap cloud -> strong cloud
    python -m groundwire.evals.cascade --cheap gpt-3.5-turbo-0613 --scorer token

Outputs go to ``results/cascade/<timestamp>-<cheap>-<scorer>/``: ``summary.json``,
``report.md`` and ``cases.jsonl`` (one line per prompt).

Exit codes: 0 success, 1 benchmark error (bad data/model), 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from groundwire import __version__
from groundwire.core.config import (
    AnswerAggregation,
    GroundingConfig,
    GroundingFailurePolicy,
    GroundingScorerType,
)
from groundwire.core.errors import ConfigurationError, DatasetError
from groundwire.core.logging import configure_logging
from groundwire.evals.cascade.report import build_summary, render_markdown
from groundwire.evals.cascade.simulate import CascadeCase, build_cases, pair_by_source
from groundwire.evals.cli import DEFAULT_NLI_MODEL, build_scorer
from groundwire.evals.datasets import EvalExample, load_ragtruth
from groundwire.evals.runner import evaluate
from groundwire.guardrails.grounding import GroundingVerifier, Scorer
from groundwire.guardrails.token_detector import DEFAULT_TOKEN_MODEL

logger = logging.getLogger("groundwire.evals.cascade")

RAGTRUTH_MODELS = ("gpt-4-0613", "gpt-3.5-turbo-0613", "llama-2-70b-chat",
                   "llama-2-13b-chat", "llama-2-7b-chat", "mistral-7B-instruct")


@dataclass(frozen=True, slots=True)
class ScorerDefaults:
    """Settings each scorer was tuned with on the RAGTruth train split.

    Attributes:
        threshold: Escalation threshold.
        aggregation: Claim-score aggregation.
        min_claim_tokens: Shortest checked claim.
        sweep: Thresholds for the trade-off table.
    """

    threshold: float
    aggregation: AnswerAggregation
    min_claim_tokens: int
    sweep: tuple[float, ...]


SCORER_DEFAULTS: dict[GroundingScorerType, ScorerDefaults] = {
    GroundingScorerType.LEXICAL: ScorerDefaults(0.3014, AnswerAggregation.LOWEST_TWO, 3,
                                                (0.1, 0.2, 0.3014, 0.4, 0.5, 0.6)),
    GroundingScorerType.TOKEN: ScorerDefaults(0.5, AnswerAggregation.MIN, 1,
                                              (0.1, 0.3, 0.5, 0.7, 0.9)),
    GroundingScorerType.NLI: ScorerDefaults(0.5, AnswerAggregation.LOWEST_TWO, 3,
                                            (0.1, 0.3, 0.5, 0.7)),
    GroundingScorerType.HYBRID: ScorerDefaults(0.5, AnswerAggregation.LOWEST_TWO, 3,
                                               (0.1, 0.3, 0.5, 0.7)),
}


def _positive(value: str) -> float:
    """argparse type for a positive number.

    Args:
        value: Raw argument.

    Returns:
        The number.

    Raises:
        argparse.ArgumentTypeError: If not a positive number.
    """
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m groundwire.evals.cascade",
        description="Simulate cheap-first routing with fact-check escalation on RAGTruth.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ragtruth"))
    parser.add_argument("--cheap", choices=RAGTRUTH_MODELS, default="llama-2-7b-chat",
                        help="cheap tier (default: %(default)s)")
    parser.add_argument("--strong", choices=RAGTRUTH_MODELS, default="gpt-4-0613",
                        help="strong tier (default: %(default)s)")
    parser.add_argument("--scorer", choices=[s.value for s in GroundingScorerType
                                             if s in SCORER_DEFAULTS],
                        default=GroundingScorerType.LEXICAL.value)
    parser.add_argument("--model", default=None, help="model id for nli/hybrid/token scorers")
    parser.add_argument("--threshold", type=float, default=None,
                        help="escalation threshold (default: the scorer's tuned value)")
    parser.add_argument("--price-ratios", type=_positive, nargs="+", default=[5.0, 15.0, 30.0],
                        help="strong/cheap price ratios to cost (default: 5 15 30)")
    parser.add_argument("--per-task", type=int, default=0,
                        help="prompts per task; 0 = all 150 per task (default)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=Path("results/cascade"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.cheap == args.strong:
        parser.error("--cheap and --strong must differ")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.per_task < 0:
        parser.error("--per-task must be >= 0")
    return args


def sample_pairs(pairs: list[tuple[EvalExample, EvalExample]], per_task: int,
                 seed: int) -> list[tuple[EvalExample, EvalExample]]:
    """Seeded sample of prompts per task (all if ``per_task`` is 0).

    Args:
        pairs: All pairs.
        per_task: Prompts per task.
        seed: Sampling seed.

    Returns:
        The sample, in the original order.
    """
    if not per_task:
        return pairs
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, (cheap, _) in enumerate(pairs):
        by_task[cheap.task].append(index)
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling
    keep = {i for indices in by_task.values()
            for i in (rng.sample(indices, per_task) if per_task < len(indices) else indices)}
    return [pair for index, pair in enumerate(pairs) if index in keep]


async def score_cases(args: argparse.Namespace, *, scorer: Scorer | None = None
                      ) -> list[CascadeCase]:
    """Load data, score the cheap answers and build cascade cases.

    Args:
        args: Parsed arguments.
        scorer: Scorer override (tests).

    Returns:
        Cascade cases.

    Raises:
        DatasetError: On missing or malformed data.
        ConfigurationError: If the scorer cannot be built.
        ValueError: If no prompt has answers from both models.
    """
    examples = load_ragtruth(args.data_dir, split="test")
    pairs = sample_pairs(pair_by_source(examples, args.cheap, args.strong),
                         args.per_task, args.seed)
    kind = GroundingScorerType(args.scorer)
    defaults = SCORER_DEFAULTS[kind]
    built = scorer or build_scorer(kind, args.model or DEFAULT_NLI_MODEL, None,
                                   args.model or DEFAULT_TOKEN_MODEL)
    config = GroundingConfig(scorer=kind, aggregation=defaults.aggregation,
                             min_claim_tokens=defaults.min_claim_tokens,
                             on_error=GroundingFailurePolicy.FAIL_OPEN)
    verifier = GroundingVerifier(config, built, max_concurrency=1)
    logger.info("scoring %d %s answers with the %s scorer", len(pairs), args.cheap, kind.value)
    results = await evaluate(verifier, [cheap for cheap, _ in pairs])
    return build_cases(pairs, results)


def run(args: argparse.Namespace, *, scorer: Scorer | None = None) -> Path:
    """Execute the simulation and write outputs.

    Args:
        args: Parsed arguments.
        scorer: Scorer override (tests).

    Returns:
        Output directory.

    Raises:
        DatasetError: On missing or malformed data.
        ConfigurationError: If the scorer cannot be built.
        ValueError: If no prompt has answers from both models.
        OSError: If outputs cannot be written.
    """
    cases = asyncio.run(score_cases(args, scorer=scorer))
    kind = GroundingScorerType(args.scorer)
    defaults = SCORER_DEFAULTS[kind]
    threshold = args.threshold if args.threshold is not None else defaults.threshold
    now = datetime.now(tz=UTC)
    run_info: dict[str, Any] = {
        "cheap": args.cheap, "strong": args.strong, "scorer": kind.value,
        "per_task": args.per_task or "all", "seed": args.seed,
        "groundwire_version": __version__, "date": now.strftime("%Y-%m-%d"),
    }
    summary = build_summary(cases, threshold=threshold, ratios=args.price_ratios,
                            sweep=sorted({*defaults.sweep, threshold}), run_info=run_info)
    base_dir: Path = args.out_dir
    out_dir = base_dir / f"{now.strftime('%Y%m%d-%H%M%S')}-{args.cheap}-{kind.value}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    with (out_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case)) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "INFO", json_output=False)
    try:
        out_dir = run(args)
    except (DatasetError, ConfigurationError) as exc:
        logger.error("simulation failed: %s", exc.message)
        return 1
    except ValueError as exc:
        logger.error("simulation failed: %s", exc)
        return 1
    except OSError as exc:
        logger.error("could not write results: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    sys.stdout.write("\n" + (out_dir / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
