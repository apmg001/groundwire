"""Run one gate's benchmark and return its metrics.

The runners call the same code the full benchmarks use (``load_ragtruth`` +
``GroundingVerifier``, ``load_qqp`` + cache scoring), so a gate checks the
production code path on a smaller, fixed sample.
"""

from __future__ import annotations

import logging
from pathlib import Path

from groundwire.cache.embeddings import Embedder, TransformersEmbedder
from groundwire.core.config import GroundingConfig, GroundingFailurePolicy
from groundwire.evals.cachebench.scoring import outcome, score_pairs
from groundwire.evals.cli import DEFAULT_NLI_MODEL, build_scorer
from groundwire.evals.datasets import LabelPolicy, load_ragtruth
from groundwire.evals.gate.spec import QQPGate, RagTruthGate
from groundwire.evals.metrics import roc_auc
from groundwire.evals.qqp import load_qqp, qqp_path
from groundwire.evals.report import summarise
from groundwire.evals.runner import evaluate
from groundwire.guardrails.grounding import GroundingVerifier, Scorer
from groundwire.guardrails.token_detector import DEFAULT_TOKEN_MODEL

logger = logging.getLogger(__name__)

Metrics = dict[str, float]


async def run_ragtruth_gate(gate: RagTruthGate, data_dir: Path, *,
                            scorer: Scorer | None = None) -> Metrics:
    """Evaluate the grounding verifier on the gate's RAGTruth sample.

    Args:
        gate: Gate definition.
        data_dir: RAGTruth clone.
        scorer: Scorer override (tests); built from the gate if omitted.

    Returns:
        Metric name -> value, plus ``n`` (responses evaluated).

    Raises:
        DatasetError: If the data is missing or malformed.
        ConfigurationError: If the scorer's model cannot be loaded.
    """
    # Load data first: a missing dataset fails in milliseconds, not after a model load.
    examples = load_ragtruth(data_dir, split="test", per_task=gate.per_task, seed=gate.seed,
                             label_policy=LabelPolicy.STRICT)
    built = scorer or build_scorer(gate.scorer, gate.model or DEFAULT_NLI_MODEL, None,
                                   gate.model or DEFAULT_TOKEN_MODEL)
    config = GroundingConfig(scorer=gate.scorer, min_claim_tokens=gate.min_claim_tokens,
                             aggregation=gate.aggregation,
                             evidence_passages=gate.evidence_passages,
                             on_error=GroundingFailurePolicy.FAIL_OPEN)
    verifier = GroundingVerifier(config, built, max_concurrency=1)
    results = await evaluate(verifier, examples)
    summary = summarise(results, gate.threshold)
    metrics: Metrics = {name: float(summary[name]) for name in (
        "precision", "recall", "f1", "accuracy", "false_block_rate", "scorer_errors")}
    metrics["roc_auc"] = float(summary["roc_auc"]) if summary["roc_auc"] is not None else 0.0
    metrics["n"] = float(len(results))
    return metrics


def run_qqp_gate(gate: QQPGate, data_dir: Path, *, embedder: Embedder | None = None) -> Metrics:
    """Evaluate semantic-cache decisions on the gate's QQP sample.

    Args:
        gate: Gate definition.
        data_dir: Directory with the QQP files.
        embedder: Embedder override (tests); loaded from the gate if omitted.

    Returns:
        Metric name -> value, plus ``n`` (pairs evaluated).

    Raises:
        DatasetError: If the data is missing or malformed.
        ConfigurationError: If the embedding model cannot be loaded.
        CacheUnavailableError: If embedding fails.
    """
    pairs = load_qqp(qqp_path(data_dir, "validation"), limit=gate.pairs, seed=gate.seed)
    model = embedder or TransformersEmbedder.load(gate.model, batch_size=64)
    scores, _ = score_pairs(model, pairs)
    result = outcome(scores, gate.threshold, use_guard=gate.use_guard)
    auc = roc_auc([-s.similarity for s in scores], [s.pair.is_duplicate for s in scores])
    return {
        "hit_rate": result.recall,
        "false_hit_rate": result.false_block_rate,
        "precision": result.precision,
        "roc_auc": auc if auc is not None else 0.0,
        "n": float(len(scores)),
    }
