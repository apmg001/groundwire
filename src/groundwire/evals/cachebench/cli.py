"""Command-line entry point for the semantic-cache benchmark.

Examples::

    # Download QQP (GLUE, from Hugging Face) and run with defaults
    make bench-cache

    # Directly, with a smaller tuning sample and custom budgets
    python -m groundwire.evals.cachebench --tune-pairs 5000 --budgets 0.02 0.005

Outputs go to ``results/cache/<timestamp>-qqp/``:

* ``summary.json``: every metric and run setting (machine-readable),
* ``report.md``: Markdown tables you can paste into the README,
* ``scores.jsonl``: one line per validation pair (similarity, label, guard).

Exit codes: 0 success, 1 benchmark error (bad data/model), 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from groundwire import __version__
from groundwire.cache.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, TransformersEmbedder
from groundwire.core.config import CacheConfig
from groundwire.core.errors import CacheUnavailableError, ConfigurationError, DatasetError
from groundwire.core.logging import configure_logging
from groundwire.evals.cachebench.report import DEFAULT_SWEEP, build_summary, render_markdown
from groundwire.evals.cachebench.scoring import (
    PairScore,
    TunedThreshold,
    measure_lookup_latency,
    score_pairs,
    threshold_for_false_hit_budget,
)
from groundwire.evals.qqp import load_qqp, qqp_path

logger = logging.getLogger("groundwire.evals.cachebench")


def _budget(value: str) -> float:
    """argparse type for a false-hit budget in [0, 1].

    Args:
        value: Raw argument.

    Returns:
        The budget.

    Raises:
        argparse.ArgumentTypeError: If out of range or not a number.
    """
    try:
        budget = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value}") from exc
    if not 0 <= budget <= 1:
        raise argparse.ArgumentTypeError("budgets are fractions in [0, 1], e.g. 0.01 for 1%")
    return budget


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m groundwire.evals.cachebench",
        description="Measure semantic-cache hit and false-hit rates on Quora Question Pairs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/qqp"),
                        help="directory with train/validation .parquet or .tsv files")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL,
                        help="sentence-embedding model id (default: %(default)s)")
    parser.add_argument("--tune-pairs", type=int, default=20_000,
                        help="train pairs sampled to tune thresholds; 0 skips tuning")
    parser.add_argument("--test-pairs", type=int, default=0,
                        help="validation pairs to evaluate; 0 = all (~40K)")
    parser.add_argument("--budgets", type=_budget, nargs="+", default=[0.05, 0.01, 0.001],
                        help="false-hit budgets to tune for (default: 0.05 0.01 0.001)")
    parser.add_argument("--seed", type=int, default=13, help="sampling seed")
    parser.add_argument("--batch-size", type=int, default=64, help="texts per forward pass")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for torch")
    parser.add_argument("--out-dir", type=Path, default=Path("results/cache"),
                        help="where to write results")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    if args.tune_pairs < 0 or args.test_pairs < 0:
        parser.error("--tune-pairs and --test-pairs must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def write_outputs(out_dir: Path, summary: dict[str, Any], scores: list[PairScore]) -> None:
    """Write summary, report and per-pair scores.

    Args:
        out_dir: Directory to create and write into.
        summary: Benchmark summary.
        scores: Per-pair validation scores.

    Raises:
        OSError: If the files cannot be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for s in scores:
            handle.write(json.dumps({
                "id": s.pair.id, "similarity": round(s.similarity, 5), "exact": s.exact,
                "guard": s.guard, "duplicate": s.pair.is_duplicate,
            }) + "\n")


def run(args: argparse.Namespace, *, embedder: Embedder | None = None) -> Path:
    """Execute the benchmark.

    Args:
        args: Parsed arguments.
        embedder: Embedder override (tests); loaded from ``args.model`` if omitted.

    Returns:
        Output directory.

    Raises:
        DatasetError: On missing or malformed data.
        ConfigurationError: If the embedding model cannot be loaded.
        CacheUnavailableError: If embedding fails mid-run.
        OSError: If outputs cannot be written.
    """
    # Load data before the model: a missing file should fail in milliseconds,
    # not after a model download.
    train = (load_qqp(qqp_path(args.data_dir, "train"), limit=args.tune_pairs, seed=args.seed)
             if args.tune_pairs else [])
    test = load_qqp(qqp_path(args.data_dir, "validation"), limit=args.test_pairs or None,
                    seed=args.seed)
    model = embedder or TransformersEmbedder.load(args.model, batch_size=args.batch_size,
                                                  num_threads=args.threads)

    tuned: list[TunedThreshold] = []
    if train:
        logger.info("scoring %d train pairs for threshold tuning", len(train))
        train_scores, _ = score_pairs(model, train)
        tuned = [threshold_for_false_hit_budget(train_scores, budget)
                 for budget in sorted(args.budgets, reverse=True)]
        for t in tuned:
            logger.info("budget %.3f%% -> threshold %.4f (train hit rate %.3f)",
                        t.budget * 100, t.threshold, t.hit_rate)

    logger.info("scoring %d validation pairs", len(test))
    scores, stats = score_pairs(model, test)
    latency = measure_lookup_latency(model, [p.question2 for p in test])

    now = datetime.now(tz=UTC)
    run_info = {
        "dataset": "QQP (GLUE)", "model": getattr(model, "name", args.model),
        "tune_pairs": args.tune_pairs, "test_pairs": args.test_pairs or "all",
        "seed": args.seed, "groundwire_version": __version__,
        "date": now.strftime("%Y-%m-%d"),
    }
    summary = build_summary(scores, tuned=tuned,
                            configured_threshold=CacheConfig().similarity_threshold,
                            stats=stats, latency_ms=latency, run_info=run_info,
                            sweep=DEFAULT_SWEEP)
    base_dir: Path = args.out_dir
    out_dir = base_dir / f"{now.strftime('%Y%m%d-%H%M%S')}-qqp"
    write_outputs(out_dir, summary, scores)

    shipped = summary["configured"]["with_guard"]
    logger.info("RESULT threshold %.2f: hit_rate=%.3f false_hit_rate=%.4f auc=%s -> %s",
                summary["configured"]["threshold"], shipped["hit_rate"],
                shipped["false_hit_rate"], summary["roc_auc"], out_dir / "report.md")
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
    except (DatasetError, ConfigurationError, CacheUnavailableError) as exc:
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
