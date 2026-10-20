"""Command-line entry point for the hallucination benchmark.

Examples::

    # Fast, dependency-free run of the default (lexical) scorer
    python -m groundwire.evals --data-dir data/ragtruth

    # NLI and hybrid scorers (pip install -e '.[nli]')
    python -m groundwire.evals --data-dir data/ragtruth --scorer hybrid --per-task 100

Outputs go to ``results/<timestamp>-<scorer>/``:

* ``summary.json``: all metrics and run settings (machine-readable),
* ``report.md``: a Markdown table you can paste into the README,
* ``predictions.jsonl``: one line per test example, for error analysis.

Exit codes: 0 success, 1 benchmark error (bad data/config), 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from groundwire import __version__
from groundwire.core.config import (
    GroundingConfig,
    GroundingFailurePolicy,
    GroundingScorerType,
)
from groundwire.core.errors import ConfigurationError, DatasetError
from groundwire.core.logging import configure_logging
from groundwire.evals.datasets import RAGTRUTH_TASKS, LabelPolicy, load_ragtruth
from groundwire.evals.metrics import best_f1_threshold
from groundwire.evals.report import build_summary, render_markdown
from groundwire.evals.runner import ExampleResult, evaluate
from groundwire.guardrails.grounding import GroundingVerifier
from groundwire.guardrails.scorers import HybridScorer, LexicalScorer, NLIScorer, SupportScorer

logger = logging.getLogger("groundwire.evals")

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_THRESHOLD = 0.6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m groundwire.evals",
        description="Benchmark Groundwire's grounding verifier on RAGTruth.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/ragtruth"),
                        help="RAGTruth clone (default: data/ragtruth)")
    parser.add_argument("--scorer", choices=[s.value for s in GroundingScorerType],
                        default=GroundingScorerType.LEXICAL.value)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL,
                        help=f"Hugging Face NLI model (default: {DEFAULT_NLI_MODEL})")
    parser.add_argument("--per-task", type=int, default=100,
                        help="test responses per task; 0 = all (default: 100)")
    parser.add_argument("--dev-per-task", type=int, default=50,
                        help="train responses per task used to tune the threshold; "
                             "0 = no tuning, use --threshold (default: 50)")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"fixed decision threshold (default when not tuning: "
                             f"{DEFAULT_THRESHOLD})")
    parser.add_argument("--tasks", nargs="+", choices=RAGTRUTH_TASKS, default=list(RAGTRUTH_TASKS))
    parser.add_argument("--label-policy", choices=[p.value for p in LabelPolicy],
                        default=LabelPolicy.STRICT.value)
    parser.add_argument("--min-claim-tokens", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads for the NLI model (default: torch decides)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.per_task < 0 or args.dev_per_task < 0:
        parser.error("--per-task and --dev-per-task must be >= 0")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.threshold is not None and args.dev_per_task:
        parser.error("use either --threshold or --dev-per-task tuning, not both "
                     "(pass --dev-per-task 0 with --threshold)")
    return args


def build_scorer(kind: GroundingScorerType, nli_model: str,
                 threads: int | None) -> SupportScorer:
    """Instantiate the requested scorer.

    Args:
        kind: Scorer type.
        nli_model: NLI model id (ignored for lexical).
        threads: CPU threads for NLI.

    Returns:
        The scorer.

    Raises:
        ConfigurationError: If NLI is requested but unavailable.
    """
    if kind is GroundingScorerType.LEXICAL:
        return LexicalScorer()
    nli = NLIScorer.load(nli_model, num_threads=threads)
    return HybridScorer(nli) if kind is GroundingScorerType.HYBRID else nli


def write_outputs(out_dir: Path, summary: dict[str, Any],
                  results: list[ExampleResult]) -> None:
    """Write summary, report and per-example predictions.

    Args:
        out_dir: Directory to create and write into.
        summary: Benchmark summary.
        results: Per-example results.

    Raises:
        OSError: If the files cannot be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")


async def run(args: argparse.Namespace) -> Path:
    """Execute the benchmark.

    Args:
        args: Parsed arguments.

    Returns:
        Output directory.

    Raises:
        DatasetError: On missing or malformed data.
        ConfigurationError: If the scorer cannot be built.
        OSError: If outputs cannot be written.
    """
    kind = GroundingScorerType(args.scorer)
    policy = LabelPolicy(args.label_policy)
    scorer = build_scorer(kind, args.nli_model, args.threads)
    config = GroundingConfig(scorer=kind, min_claim_tokens=args.min_claim_tokens,
                             min_supported_ratio=1.0,
                             on_error=GroundingFailurePolicy.FAIL_OPEN)
    verifier = GroundingVerifier(config, scorer, max_concurrency=1)

    if args.dev_per_task:
        logger.info("tuning threshold on %d train responses per task", args.dev_per_task)
        dev = load_ragtruth(args.data_dir, split="train", tasks=args.tasks,
                            per_task=args.dev_per_task, seed=args.seed, label_policy=policy)
        dev_results = await evaluate(verifier, dev)
        threshold, dev_f1 = best_f1_threshold([r.support_score for r in dev_results],
                                              [r.is_hallucinated for r in dev_results])
        threshold_source = (f"tuned for best F1 on {len(dev)} train responses; "
                            f"train F1 {dev_f1:.3f}")
    else:
        threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
        threshold_source = "fixed"
    logger.info("decision threshold %.4f (%s)", threshold, threshold_source)

    test = load_ragtruth(args.data_dir, split="test", tasks=args.tasks,
                         per_task=args.per_task or None, seed=args.seed, label_policy=policy)
    results = await evaluate(verifier, test)

    now = datetime.now(tz=UTC)
    run_info = {
        "dataset": "RAGTruth", "split": "test", "scorer": kind.value,
        "nli_model": args.nli_model if kind is not GroundingScorerType.LEXICAL else None,
        "per_task": args.per_task, "dev_per_task": args.dev_per_task, "seed": args.seed,
        "label_policy": policy.value, "min_claim_tokens": args.min_claim_tokens,
        "threshold_source": threshold_source, "groundwire_version": __version__,
        "date": now.strftime("%Y-%m-%d"),
    }
    summary = build_summary(results, threshold, run_info)
    base_dir: Path = args.out_dir
    out_dir = base_dir / f"{now.strftime('%Y%m%d-%H%M%S')}-{kind.value}"
    write_outputs(out_dir, summary, results)

    overall = summary["overall"]
    logger.info(
        "RESULT %s: precision=%.3f recall=%.3f f1=%.3f auc=%s blocked_good=%.3f -> %s",
        kind.value, overall["precision"], overall["recall"], overall["f1"],
        overall["roc_auc"], overall["false_block_rate"], out_dir / "report.md",
    )
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
        out_dir = asyncio.run(run(args))
    except (DatasetError, ConfigurationError) as exc:
        logger.error("benchmark failed: %s", exc.message)
        return 1
    except OSError as exc:
        logger.error("could not write results: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    sys.stdout.write("\n" + render_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
