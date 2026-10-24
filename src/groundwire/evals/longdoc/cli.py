"""Long-document grounding benchmark: does verification still work on long inputs?

Compares three ways of verifying the same RAGTruth answers:

* ``short``: the original short context (reference: what the verifier can
  do when the context is small).
* ``legacy``: the long haystack with the previous algorithm (first 256
  passages of ~600 characters, every claim against every passage; the token
  scorer reads the whole prompt and its model cuts it at the window).
* ``retrieval``: the long haystack with evidence retrieval (current default).

Each strategy's decision threshold is tuned for best F1 on *train* haystacks
and applied unchanged to *test* haystacks. Reported per strategy: F1,
precision, recall, ROC-AUC, good answers blocked, latency, and **evidence
from the true source**: for grounded answers, the share of claims whose best
evidence chunk came from the needle (the document the answer was written
from) rather than a distractor.

Run::

    python -m groundwire.evals.longdoc --context-chars 60000
    make bench-longdoc
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from groundwire.core.config import (
    AnswerAggregation,
    GroundingConfig,
    GroundingFailurePolicy,
    GroundingScorerType,
)
from groundwire.core.errors import ConfigurationError, DatasetError
from groundwire.evals.datasets import RAGTRUTH_TASKS, load_ragtruth
from groundwire.evals.longdoc.haystack import LongExample, NeedlePosition, build_haystack
from groundwire.evals.metrics import at_threshold, best_f1_threshold, percentile, roc_auc
from groundwire.guardrails.evidence import SentenceChunker
from groundwire.guardrails.grounding import GroundingVerifier, Verdict, build_scorer

logger = logging.getLogger("groundwire.evals.longdoc")

LEGACY_MAX_PASSAGES = 256  # the previous verifier's hard cap
LEGACY_PASSAGE_CHARS = 600


class Strategy(StrEnum):
    """How the context is verified."""

    SHORT = "short"
    LEGACY = "legacy"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True, slots=True)
class Outcome:
    """One verification result.

    Attributes:
        score: Answer support score (lower = more suspicious).
        hallucinated: Ground-truth label.
        latency_ms: Verification time.
        position: Needle position (long strategies) or ``None``.
        from_needle: Per claim, whether its best evidence came from the needle.
        truncated: Whether the verifier capped the context.
    """

    score: float
    hallucinated: bool
    latency_ms: float
    position: NeedlePosition | None
    from_needle: tuple[bool, ...]
    truncated: bool


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


def base_config(scorer: GroundingScorerType) -> GroundingConfig:
    """The configuration the main RAGTruth benchmark uses for ``scorer``."""
    if scorer is GroundingScorerType.TOKEN:
        return GroundingConfig(scorer=scorer, aggregation=AnswerAggregation.MIN,
                               min_claim_tokens=1, on_error=GroundingFailurePolicy.FAIL_OPEN)
    return GroundingConfig(scorer=scorer, aggregation=AnswerAggregation.LOWEST_TWO,
                           on_error=GroundingFailurePolicy.FAIL_OPEN)


def make_verifier(strategy: Strategy, config: GroundingConfig,
                  scorer: Any) -> GroundingVerifier:
    """Build the verifier for a strategy (sharing one loaded scorer).

    ``legacy`` reproduces the previous algorithm exactly: the same splitter
    (verified identical on all RAGTruth contexts), capped at 256 passages,
    every claim scored against every passage; for token scorers, the whole
    context in one prompt.

    Args:
        strategy: Strategy.
        config: Base configuration.
        scorer: Loaded scorer or token detector.

    Returns:
        The verifier.
    """
    if strategy is not Strategy.LEGACY:
        return GroundingVerifier(config, scorer, max_concurrency=1)
    legacy = config.model_copy(update={"evidence": config.evidence.model_copy(update={
        "all_pairs_max_chunks": 4_096, "token_context_chars": 1_000_000})})
    chunker = SentenceChunker(max_chars=LEGACY_PASSAGE_CHARS, overlap_sentences=0,
                              attach_headings=False, max_chunks=LEGACY_MAX_PASSAGES)
    return GroundingVerifier(legacy, scorer, chunker=chunker, max_concurrency=1)


async def run_strategy(verifier: GroundingVerifier, examples: Sequence[LongExample],
                       strategy: Strategy) -> list[Outcome]:
    """Verify every example with one strategy.

    Args:
        verifier: Strategy's verifier.
        examples: Long examples (``short`` uses their original context).
        strategy: Strategy being run.

    Returns:
        One outcome per example, in order.
    """
    outcomes: list[Outcome] = []
    for long in examples:
        short = strategy is Strategy.SHORT
        context = list(long.example.context if short else long.context)
        needle = frozenset(range(len(context))) if short else long.needle_documents
        report = await verifier.verify(long.example.response, context,
                                       question=long.example.meta.get("question"))
        if report.verdict is Verdict.UNVERIFIED or not report.claims:
            score = 1.0
        else:
            score = report.answer_score
        outcomes.append(Outcome(
            score=score, hallucinated=long.example.is_hallucinated,
            latency_ms=report.latency_ms, position=None if short else long.position,
            from_needle=tuple(c.document in needle for c in report.claims if c.document >= 0),
            truncated=report.truncated))
    return outcomes


# --------------------------------------------------------------------------- #
# Metrics and report
# --------------------------------------------------------------------------- #


def summarise(test: Sequence[Outcome], threshold: float) -> dict[str, Any]:
    """Metrics for one strategy on the test haystacks.

    Args:
        test: Test outcomes.
        threshold: Decision threshold tuned on train.

    Returns:
        JSON-ready metrics.
    """
    scores = [o.score for o in test]
    labels = [o.hallucinated for o in test]
    metrics = at_threshold(scores, labels, threshold)
    grounded_claims = [hit for o in test if not o.hallucinated for hit in o.from_needle]
    latencies = [o.latency_ms for o in test]
    by_position: dict[str, Any] = {}
    for position in NeedlePosition:
        subset = [o for o in test if o.position is position]
        if subset:
            sub = at_threshold([o.score for o in subset], [o.hallucinated for o in subset],
                               threshold)
            auc = roc_auc([o.score for o in subset], [o.hallucinated for o in subset])
            by_position[position.value] = {"n": len(subset), "f1": round(sub.f1, 4),
                                           "roc_auc": round(auc, 4) if auc is not None else None}
    auc = roc_auc(scores, labels)
    return {
        "threshold": round(threshold, 4), **metrics.as_dict(),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "evidence_from_source": (round(sum(grounded_claims) / len(grounded_claims), 4)
                                 if grounded_claims else None),
        "truncated": round(sum(o.truncated for o in test) / len(test), 4) if test else 0.0,
        "latency_p50_ms": round(percentile(latencies, 0.5), 1) if latencies else 0.0,
        "latency_p95_ms": round(percentile(latencies, 0.95), 1) if latencies else 0.0,
        "by_position": by_position,
    }


def render(summary: dict[str, Any]) -> str:
    """Markdown report."""
    run = summary["run"]
    lines = ["# Long-document grounding benchmark (RAGTruth haystacks)", "",
             f"- **Scorer:** `{run['scorer']}`",
             f"- **Context:** each answer's real source hidden among other RAGTruth documents "
             f"to ≥ {run['context_chars']:,} characters (mean {run['mean_chars']:,}); "
             "source at start / middle / end in turn",
             f"- **Test:** {run['test']} answers; thresholds tuned on {run['dev']} train answers "
             "per strategy", f"- **Run date:** {run['date']}", "",
             "| Strategy | F1 | Precision | Recall | ROC-AUC | Good answers blocked | "
             "Evidence from true source | Context truncated | Latency p50 / p95 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in summary["strategies"].items():
        evidence = "-" if m["evidence_from_source"] is None else f"{m['evidence_from_source']:.1%}"
        auc = "-" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        lines.append(f"| `{name}` | {m['f1']:.1%} | {m['precision']:.1%} | {m['recall']:.1%} | "
                     f"{auc} | {m['false_block_rate']:.1%} | {evidence} | {m['truncated']:.0%} | "
                     f"{m['latency_p50_ms']:.0f} / {m['latency_p95_ms']:.0f} ms |")
    lines += ["", "**By position of the true source (ROC-AUC):**", "",
              "| Strategy | start | middle | end |", "|---|---|---|---|"]
    for name, m in summary["strategies"].items():
        cells = [m["by_position"].get(p.value, {}).get("roc_auc") for p in NeedlePosition]
        if any(c is not None for c in cells):
            lines.append(f"| `{name}` | " + " | ".join("-" if c is None else f"{c:.3f}"
                                                       for c in cells) + " |")
    lines += ["", "`short` uses the original short context (the ceiling). *Evidence from true "
              "source*: for grounded answers, share of claims whose best evidence came from the "
              "document the answer was written from."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="python -m groundwire.evals.longdoc",
                                     description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ragtruth"))
    parser.add_argument("--scorer", choices=[s.value for s in GroundingScorerType],
                        default=GroundingScorerType.LEXICAL.value)
    parser.add_argument("--context-chars", type=int, default=60_000)
    parser.add_argument("--per-task", type=int, default=150, help="test answers per task")
    parser.add_argument("--dev-per-task", type=int, default=100,
                        help="train answers per task for threshold tuning")
    parser.add_argument("--tasks", nargs="+", default=list(RAGTRUTH_TASKS))
    parser.add_argument("--strategies", nargs="+", default=[s.value for s in Strategy],
                        choices=[s.value for s in Strategy])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("results/longdoc"))
    args = parser.parse_args(argv)
    if args.context_chars < 1_000 or args.per_task < 1 or args.dev_per_task < 1:
        parser.error("--context-chars >= 1000, --per-task >= 1 and --dev-per-task >= 1")
    return args


async def run(args: argparse.Namespace) -> Path:
    """Run the benchmark and write ``report.md`` and ``summary.json``.

    Args:
        args: Parsed arguments.

    Returns:
        Output directory.

    Raises:
        DatasetError: On missing or malformed data.
        ConfigurationError: If the scorer cannot be built.
    """
    kind = GroundingScorerType(args.scorer)
    config = base_config(kind)
    scorer = build_scorer(config)
    splits = {}
    for split, per_task in (("train", args.dev_per_task), ("test", args.per_task)):
        examples = load_ragtruth(args.data_dir, split=split, tasks=args.tasks,
                                 per_task=per_task, seed=args.seed)
        pool = load_ragtruth(args.data_dir, split=split, tasks=RAGTRUTH_TASKS, seed=args.seed)
        splits[split] = build_haystack(examples, pool, target_chars=args.context_chars,
                                       seed=args.seed)
    results: dict[str, Any] = {}
    for name in args.strategies:
        strategy = Strategy(name)
        verifier = make_verifier(strategy, config, scorer)
        started = time.perf_counter()
        dev = await run_strategy(verifier, splits["train"], strategy)
        threshold, _ = best_f1_threshold([o.score for o in dev], [o.hallucinated for o in dev])
        test = await run_strategy(verifier, splits["test"], strategy)
        results[name] = summarise(test, threshold)
        logger.info("%s done in %.0fs", name, time.perf_counter() - started)

    test_examples = splits["test"]
    summary = {"run": {"scorer": kind.value, "context_chars": args.context_chars,
                       "mean_chars": round(sum(e.chars for e in test_examples)
                                           / len(test_examples)),
                       "test": len(test_examples), "dev": len(splits["train"]),
                       "seed": args.seed, "date": datetime.now(UTC).date().isoformat()},
               "strategies": results}
    out: Path = Path(args.out) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render(summary), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Exit code (0 ok, 2 bad data or configuration).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        out = asyncio.run(run(parse_args(argv)))
    except (DatasetError, ConfigurationError) as exc:
        logger.error("benchmark could not run: %s", exc)
        return 2
    sys.stdout.write((out / "report.md").read_text(encoding="utf-8"))
    return 0
