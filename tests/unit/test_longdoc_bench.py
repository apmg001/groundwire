"""Tests for the long-document benchmark (its numbers go in the README)."""

from __future__ import annotations

import pytest

from groundwire.core.config import GroundingScorerType
from groundwire.core.errors import DatasetError
from groundwire.evals.datasets import EvalExample
from groundwire.evals.longdoc.cli import (
    LEGACY_MAX_PASSAGES,
    Outcome,
    Strategy,
    base_config,
    make_verifier,
    render,
    run_strategy,
    summarise,
)
from groundwire.evals.longdoc.haystack import NeedlePosition, build_haystack
from groundwire.guardrails.scorers import LexicalScorer


def _example(i: int, source: str, *, hallucinated: bool = False) -> EvalExample:
    return EvalExample(id=f"e{i}", task="QA", context=(f"Source {source} says the fee is {i}0 "
                                                       "rupees for members.",),
                       response=f"The fee is {i}0 rupees for members.",
                       is_hallucinated=hallucinated, meta={"source_id": source})


POOL = [_example(i, f"s{i}") for i in range(200)]


def test_haystack_hides_the_needle_at_cycling_positions() -> None:
    examples = POOL[:3]
    longs = build_haystack(examples, POOL, target_chars=2_000)
    assert [x.position for x in longs] == list(NeedlePosition)
    for long, example in zip(longs, examples, strict=True):
        assert long.chars >= 2_000
        [needle] = long.needle_documents
        assert long.context[needle] == example.context[0]
    assert min(longs[0].needle_documents) == 0
    assert max(longs[2].needle_documents) == len(longs[2].context) - 1


def test_haystack_never_uses_the_examples_own_source_as_a_distractor() -> None:
    twin = EvalExample(id="twin", task="QA", context=("Twin text shared by two answers.",),
                       response="x", is_hallucinated=False, meta={"source_id": "s0"})
    longs = build_haystack([POOL[0]], [*POOL, twin], target_chars=1_500)
    distractors = [d for i, d in enumerate(longs[0].context)
                   if i not in longs[0].needle_documents]
    assert twin.context[0] not in distractors


def test_haystack_is_reproducible_and_validates_inputs() -> None:
    first = build_haystack(POOL[:2], POOL, target_chars=1_000, seed=5)
    assert first == build_haystack(POOL[:2], POOL, target_chars=1_000, seed=5)
    with pytest.raises(DatasetError, match="too small"):
        build_haystack(POOL[:1], POOL[:3], target_chars=100_000)
    with pytest.raises(DatasetError):
        build_haystack(POOL[:1], POOL, target_chars=0)


def test_legacy_strategy_reproduces_the_old_passage_cap() -> None:
    verifier = make_verifier(Strategy.LEGACY, base_config(GroundingScorerType.LEXICAL),
                             LexicalScorer())
    chunked = verifier._chunker.chunk(["One sentence here."] * (LEGACY_MAX_PASSAGES + 50))
    assert len(chunked.chunks) == LEGACY_MAX_PASSAGES and chunked.truncated


async def test_retrieval_finds_the_needle_that_legacy_cannot_reach() -> None:
    pool = [_example(i, f"s{i}") for i in range(600)]
    # Three examples so positions cycle start, middle, end; keep the one at the end,
    # past the legacy cap of 256 passages (each pool document is one short passage).
    [at_end] = [x for x in build_haystack(pool[5:8], pool, target_chars=20_000)
                if x.position is NeedlePosition.END]
    assert min(at_end.needle_documents) > LEGACY_MAX_PASSAGES
    config = base_config(GroundingScorerType.LEXICAL)
    for strategy, expect_found in ((Strategy.RETRIEVAL, True), (Strategy.LEGACY, False)):
        [outcome] = await run_strategy(make_verifier(strategy, config, LexicalScorer()),
                                       [at_end], strategy)
        assert outcome.from_needle and all(outcome.from_needle) is expect_found, strategy
        assert outcome.truncated is (strategy is Strategy.LEGACY)


def test_summary_metrics_and_report() -> None:
    outcomes = [Outcome(0.9, False, 10.0, NeedlePosition.START, (True, True), False),
                Outcome(0.1, True, 20.0, NeedlePosition.MIDDLE, (False,), False),
                Outcome(0.8, False, 30.0, NeedlePosition.END, (True, False), True)]
    summary = summarise(outcomes, threshold=0.5)
    assert summary["f1"] == 1.0 and summary["roc_auc"] == 1.0
    assert summary["evidence_from_source"] == 0.75  # grounded answers' claims only
    assert summary["truncated"] == round(1 / 3, 4)
    text = render({"run": {"scorer": "lexical", "context_chars": 60_000, "mean_chars": 61_000,
                           "test": 3, "dev": 3, "date": "2026-10-24"},
                   "strategies": {"retrieval": summary}})
    assert "| `retrieval` | 100.0% |" in text and "75.0%" in text
