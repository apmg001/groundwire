"""Tests for the token-level hallucination detector and its verifier integration."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from conftest import ScriptedProvider, make_config
from groundwire.container import build_container
from groundwire.core.config import AnswerAggregation, GroundingConfig, GroundingFailurePolicy
from groundwire.core.errors import ConfigurationError, GuardrailUnavailableError
from groundwire.guardrails.grounding import GroundingVerifier, Verdict
from groundwire.guardrails.token_detector import (
    TokenDetector,
    TokenProb,
    TransformersTokenDetector,
    build_prompt,
    max_prob_in_range,
)
from groundwire.main import create_app

# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #


def test_build_prompt_qa_matches_training_format() -> None:
    prompt = build_prompt(["Refund is 49.99 USD.", "Issued on 2 May."], "How much?")
    assert prompt.startswith("Briefly answer the following question:\nHow much?\n")
    assert "following 2 passages:" in prompt
    assert "passage 1: Refund is 49.99 USD.\npassage 2: Issued on 2 May." in prompt
    assert prompt.endswith("output:")


@pytest.mark.parametrize("question", [None, "   "])
def test_build_prompt_without_question_uses_summary_format(question: str | None) -> None:
    prompt = build_prompt(["Some article."], question)
    assert prompt == "Summarize the following text:\npassage 1: Some article.\noutput:"


def test_max_prob_in_range() -> None:
    tokens = [TokenProb(0, 3, 0.1), TokenProb(4, 9, 0.9), TokenProb(10, 12, 0.2)]
    assert max_prob_in_range(tokens, 0, 4) == 0.1
    assert max_prob_in_range(tokens, 0, 12) == 0.9
    assert max_prob_in_range(tokens, 20, 30) == 0.0


# --------------------------------------------------------------------------- #
# Transformers detector with fakes
# --------------------------------------------------------------------------- #


class _Tensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, key: Any) -> _Tensor:
        if isinstance(key, tuple):  # [0, :, label]
            batch, _, label = key
            return _Tensor([row[label] for row in self.data[batch]])
        return _Tensor(self.data[key])

    def tolist(self) -> Any:
        return self.data


class _Encoding(dict[str, Any]):
    def __init__(self, seq_ids: list[int | None], offsets: list[tuple[int, int]]) -> None:
        super().__init__(input_ids=_Tensor([[0] * len(seq_ids)]),
                         attention_mask=_Tensor([[1] * len(seq_ids)]),
                         offset_mapping=_Tensor([offsets]))
        self._seq_ids = seq_ids

    def sequence_ids(self, index: int) -> list[int | None]:
        return self._seq_ids


class _FakeTokenizer:
    """Whitespace tokenizer that mimics a fast tokenizer's pair encoding."""

    is_fast = True

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, prompt: str, answer: str, **kwargs: Any) -> _Encoding:
        self.kwargs = kwargs
        seq_ids: list[int | None] = [None]
        offsets: list[tuple[int, int]] = [(0, 0)]
        for _ in prompt.split():
            seq_ids.append(0)
            offsets.append((0, 1))
        seq_ids.append(None)
        offsets.append((0, 0))
        position = 0
        for word in answer.split():
            start = answer.index(word, position)
            seq_ids.append(1)
            offsets.append((start, start + len(word)))
            position = start + len(word)
        seq_ids.append(None)
        offsets.append((0, 0))
        return _Encoding(seq_ids, offsets)


class _Output:
    def __init__(self, logits: _Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Marks the word '89.99' as hallucinated."""

    def __init__(self, tokenizer: _FakeTokenizer, fail: bool = False) -> None:
        self.tokenizer = tokenizer
        self.fail = fail
        self.last_answer = ""

    def __call__(self, input_ids: _Tensor, attention_mask: _Tensor) -> _Output:
        if self.fail:
            raise RuntimeError("out of memory")
        n = len(input_ids.data[0])
        rows = [[0.9, 0.1] for _ in range(n)]
        rows[-2] = [0.05, 0.95]  # last answer token
        return _Output(_Tensor([rows]))


class _FakeTorch:
    @staticmethod
    @contextmanager
    def inference_mode() -> Iterator[None]:
        yield

    @staticmethod
    def softmax(logits: _Tensor, dim: int) -> _Tensor:
        return logits


def _detector(fail: bool = False) -> TransformersTokenDetector:
    tokenizer = _FakeTokenizer()
    return TransformersTokenDetector(tokenizer=tokenizer, model=_FakeModel(tokenizer, fail),
                                     torch_module=_FakeTorch(), max_length=64)


def test_detector_returns_answer_tokens_with_answer_offsets() -> None:
    detector = _detector()
    tokens = detector.detect("context words here", "Refund is 89.99")
    assert [(t.start, t.end) for t in tokens] == [(0, 6), (7, 9), (10, 15)]
    assert tokens[-1].prob == 0.95 and tokens[0].prob == 0.1
    assert detector._tokenizer.kwargs["truncation"] == "only_first"


def test_detector_empty_answer_and_failures() -> None:
    assert _detector().detect("ctx", "   ") == []
    with pytest.raises(GuardrailUnavailableError, match="out of memory"):
        _detector(fail=True).detect("ctx", "Refund is 89.99")


def test_detector_rejects_tiny_max_length() -> None:
    with pytest.raises(ConfigurationError):
        TransformersTokenDetector(tokenizer=_FakeTokenizer(), model=None,
                                  torch_module=_FakeTorch(), max_length=8)


def test_load_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ConfigurationError, match="nli"):
        TransformersTokenDetector.load("any/model")


# --------------------------------------------------------------------------- #
# Verifier integration
# --------------------------------------------------------------------------- #


class _StubDetector(TokenDetector):
    name = "stub"

    def __init__(self, flagged_word: str | None = None, fail: bool = False) -> None:
        self.flagged_word = flagged_word
        self.fail = fail
        self.prompts: list[str] = []

    def detect(self, prompt: str, answer: str) -> list[TokenProb]:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("boom")
        tokens = []
        position = 0
        for word in answer.split():
            start = answer.index(word, position)
            position = start + len(word)
            prob = 0.97 if self.flagged_word and self.flagged_word in word else 0.02
            tokens.append(TokenProb(start, position, prob))
        return tokens


CONTEXT = ["Refund for order 7781 is 49.99 USD.", "It was issued on 2 May."]
ANSWER = "The refund for order 7781 is 89.99 USD. It was issued on 2 May."


def _verifier(detector: TokenDetector, **cfg: Any) -> GroundingVerifier:
    config = GroundingConfig(aggregation=AnswerAggregation.MIN, support_threshold=0.5,
                             min_claim_tokens=1, **cfg)
    return GroundingVerifier(config, detector)


async def test_token_scores_map_onto_the_right_claim() -> None:
    report = await _verifier(_StubDetector("89.99")).verify(ANSWER, CONTEXT, question="How much?")
    assert report.scorer == "stub"
    assert [round(c.score, 2) for c in report.claims] == [0.03, 0.98]
    assert report.claims[0].supported is False and report.claims[1].supported is True
    assert report.verdict is Verdict.FAILED


async def test_clean_answer_passes() -> None:
    report = await _verifier(_StubDetector()).verify(ANSWER, CONTEXT)
    assert report.verdict is Verdict.PASSED


async def test_prompt_is_built_from_question_or_used_verbatim() -> None:
    detector = _StubDetector()
    verifier = _verifier(detector)
    await verifier.verify(ANSWER, CONTEXT, question="How much was refunded?")
    await verifier.verify(ANSWER, CONTEXT, prompt="ORIGINAL PROMPT")
    assert "How much was refunded?" in detector.prompts[0]
    assert detector.prompts[1] == "ORIGINAL PROMPT"


async def test_detector_failure_follows_policy() -> None:
    report = await _verifier(_StubDetector(fail=True)).verify(ANSWER, CONTEXT)
    assert report.verdict is Verdict.UNVERIFIED
    with pytest.raises(GuardrailUnavailableError):
        await _verifier(_StubDetector(fail=True),
                        on_error=GroundingFailurePolicy.FAIL_CLOSED).verify(ANSWER, CONTEXT)


async def test_gateway_passes_user_question_to_detector() -> None:
    from fastapi.testclient import TestClient

    detector = _StubDetector("89.99")
    config = make_config()
    verifier = _verifier(detector)
    providers = {"primary": ScriptedProvider("primary", ANSWER),
                 "secondary": ScriptedProvider("secondary")}
    app = create_app(config=config, container_factory=lambda: build_container(
        config, providers=providers, verifier=verifier))
    with TestClient(app) as client:
        body = {"messages": [{"role": "user", "content": "How much was my refund?"}],
                "groundwire": {"grounding": {"context": CONTEXT, "mode": "block"}}}
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 422
    assert "How much was my refund?" in detector.prompts[0]
