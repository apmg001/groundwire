"""Tests for the NLI and hybrid scorers, using a fake model (no torch needed)."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar

import pytest

from groundwire.core.config import GroundingConfig, GroundingScorerType
from groundwire.core.errors import ConfigurationError, GuardrailUnavailableError
from groundwire.guardrails.grounding import build_scorer
from groundwire.guardrails.scorers import (
    HybridScorer,
    LexicalScorer,
    NLIScorer,
    Pair,
    SupportScorer,
    numbers_supported,
)

# --------------------------------------------------------------------------- #
# Minimal fakes for tokenizer / model / torch
# --------------------------------------------------------------------------- #


class _Rows:
    """List-of-rows stand-in for a 2-D tensor."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def __getitem__(self, key: tuple[slice, int]) -> _Rows:
        _, column = key
        return _Rows([[row[column]] for row in self.rows])

    def tolist(self) -> list[float]:
        return [row[0] for row in self.rows]


class _FakeTorch:
    @staticmethod
    @contextmanager
    def inference_mode() -> Iterator[None]:
        yield

    @staticmethod
    def softmax(logits: _Rows, dim: int) -> _Rows:
        return logits  # the fake model already returns probabilities


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, premises: list[str], hypotheses: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"premises": premises, "hypotheses": hypotheses}


class _Config:
    id2label: ClassVar[dict[int, str]] = {0: "contradiction", 1: "entailment", 2: "neutral"}


class _Output:
    def __init__(self, logits: _Rows) -> None:
        self.logits = logits


class _FakeModel:
    """'Entails' when every hypothesis word appears in the premise."""

    config = _Config()

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.batches = 0

    def __call__(self, premises: list[str], hypotheses: list[str]) -> _Output:
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.batches += 1
        rows = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            entailed = all(w in premise.lower() for w in hypothesis.lower().split())
            rows.append([0.05, 0.9, 0.05] if entailed else [0.6, 0.1, 0.3])
        return _Output(_Rows(rows))


def _nli(batch_size: int = 2, fail: bool = False) -> tuple[NLIScorer, _FakeModel, _FakeTokenizer]:
    model, tokenizer = _FakeModel(fail=fail), _FakeTokenizer()
    scorer = NLIScorer(tokenizer=tokenizer, model=model, torch_module=_FakeTorch(),
                       batch_size=batch_size)
    return scorer, model, tokenizer


# --------------------------------------------------------------------------- #
# NLI
# --------------------------------------------------------------------------- #


def test_nli_scores_entailment_probability_in_batches() -> None:
    scorer, model, tokenizer = _nli(batch_size=2)
    pairs = [("the sky is blue", "sky blue"), ("the sky is blue", "grass red"),
             ("cats purr", "cats purr")]
    assert scorer.score_batch(pairs) == [0.9, 0.1, 0.9]
    assert model.batches == 2  # 3 pairs, batch size 2
    assert tokenizer.calls[0]["truncation"] == "only_first"  # never truncate the claim


def test_nli_wraps_runtime_errors() -> None:
    scorer, _, _ = _nli(fail=True)
    with pytest.raises(GuardrailUnavailableError, match="out of memory"):
        scorer.score_batch([("a", "b")])


def test_nli_rejects_bad_settings() -> None:
    with pytest.raises(ConfigurationError):
        NLIScorer(tokenizer=_FakeTokenizer(), model=_FakeModel(), torch_module=_FakeTorch(),
                  batch_size=0)


@pytest.mark.parametrize(("labels", "expected"), [
    ({0: "contradiction", 1: "entailment", 2: "neutral"}, 1),
    ({0: "NOT_ENTAILMENT", 1: "ENTAILMENT"}, 1),
    ({0: "ENTAILMENT", 1: "NOT_ENTAILMENT"}, 0),
])
def test_entailment_index_lookup(labels: dict[int, str], expected: int) -> None:
    assert NLIScorer._find_entailment_index(labels) == expected


def test_entailment_index_missing() -> None:
    with pytest.raises(ConfigurationError):
        NLIScorer._find_entailment_index({0: "positive", 1: "negative"})


def test_load_without_nli_extra_raises_configuration_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    with pytest.raises(ConfigurationError, match="nli"):
        NLIScorer.load("any/model")


def test_build_scorer_lexical_needs_no_model() -> None:
    assert isinstance(build_scorer(GroundingConfig()), LexicalScorer)


@pytest.mark.parametrize("kind", [GroundingScorerType.NLI, GroundingScorerType.HYBRID])
def test_build_scorer_nli_kinds(monkeypatch: pytest.MonkeyPatch,
                                kind: GroundingScorerType) -> None:
    fake, _, _ = _nli()
    monkeypatch.setattr(NLIScorer, "load", classmethod(lambda cls, name, **kw: fake))
    scorer = build_scorer(GroundingConfig(scorer=kind))
    assert isinstance(scorer, HybridScorer if kind is GroundingScorerType.HYBRID else NLIScorer)


# --------------------------------------------------------------------------- #
# Hybrid
# --------------------------------------------------------------------------- #


class _RecordingScorer(SupportScorer):
    name = "recording"

    def __init__(self, value: float = 0.9, wrong_length: bool = False) -> None:
        self.seen: list[Pair] = []
        self.value = value
        self.wrong_length = wrong_length

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        self.seen.extend(pairs)
        return [] if self.wrong_length else [self.value] * len(pairs)


def test_numbers_supported() -> None:
    assert numbers_supported("Total 1,200 USD on 15 March", "It was 1200 USD")
    assert not numbers_supported("Total 1,200 USD", "It was 1500 USD")
    assert numbers_supported("no digits here", "no digits either")


def test_hybrid_gates_numbers_before_semantic_scorer() -> None:
    semantic = _RecordingScorer(0.9)
    pairs = [("Refund is 49.99 USD", "Refund is 89.99 USD"),
             ("Refund is 49.99 USD", "Refund is 49.99 USD"),
             ("The store opens daily", "The store is open every day")]
    scores = HybridScorer(semantic).score_batch(pairs)
    assert scores == [0.0, 0.9, 0.9]
    assert pairs[0] not in semantic.seen  # the model never ran on the bad number
    assert len(semantic.seen) == 2


def test_hybrid_skips_model_when_everything_fails_the_gate() -> None:
    semantic = _RecordingScorer()
    assert HybridScorer(semantic).score_batch([("1", "2")]) == [0.0]
    assert semantic.seen == []


def test_hybrid_detects_wrong_length_from_semantic_scorer() -> None:
    with pytest.raises(GuardrailUnavailableError):
        HybridScorer(_RecordingScorer(wrong_length=True)).score_batch([("a b", "a b")])
