"""The verifier on long contexts: evidence is found wherever it is, and token
detectors only see a reduced prompt when the full one would not fit."""

from __future__ import annotations

from typing import Any

import pytest

from groundwire.core.config import (
    AnswerAggregation,
    EvidenceRetrieverType,
    GroundingConfig,
)
from groundwire.core.errors import ConfigurationError
from groundwire.guardrails.evidence import BM25Retriever, SentenceChunker
from groundwire.guardrails.grounding import (
    EvidenceMode,
    GroundingVerifier,
    Verdict,
    build_retriever,
)
from groundwire.guardrails.scorers import LexicalScorer
from groundwire.guardrails.token_detector import TokenDetector, TokenProb, build_prompt

# 80 pages of unrelated text, then the page that matters. The old verifier only
# read the first 256 passages (~150k characters) and would never reach page 80.
FILLER_PAGE = " ".join(f"Clause {i} describes general account rules for members."
                       for i in range(40))
FEES_PAGE = ("Late Payment Fee\nA late-payment fee of 500 rupees applies to overdue invoices. "
             "The fee is charged after 30 days.")
PAGES = [FILLER_PAGE] * 80 + [FEES_PAGE] + [FILLER_PAGE] * 5


def _verifier(**cfg: Any) -> GroundingVerifier:
    return GroundingVerifier(GroundingConfig(**cfg), LexicalScorer())


async def test_evidence_far_into_a_long_document_is_found() -> None:
    answer = "A late-payment fee of 500 rupees applies to overdue invoices."
    report = await _verifier().verify(answer, PAGES)
    assert report.verdict is Verdict.PASSED
    assert report.evidence_mode == EvidenceMode.RETRIEVED.value
    assert report.claims[0].document == 80  # the page that supports it
    assert not report.truncated


async def test_a_wrong_number_on_page_80_is_caught() -> None:
    answer = "The late-payment fee of 500 rupees is charged after 15 days."
    report = await _verifier().verify(answer, PAGES)
    assert report.verdict is Verdict.FAILED
    assert report.claims[0].document == 80  # it looked in the right place


async def test_short_contexts_still_score_every_chunk() -> None:
    report = await _verifier().verify("A late-payment fee of 500 rupees applies.", [FEES_PAGE])
    assert report.evidence_mode == EvidenceMode.ALL_CHUNKS.value
    assert report.verdict is Verdict.PASSED


async def test_evidence_passages_forces_retrieval_even_for_short_contexts() -> None:
    report = await _verifier(evidence_passages=2).verify(
        "A late-payment fee of 500 rupees applies.", [FEES_PAGE])
    assert report.evidence_mode == EvidenceMode.RETRIEVED.value


async def test_context_beyond_max_chunks_is_reported_as_truncated() -> None:
    report = await _verifier(evidence={"max_chunks": 10}).verify(
        "A late-payment fee of 500 rupees applies.", PAGES)
    assert report.truncated
    assert report.verdict is Verdict.FAILED  # page 80 was never chunked
    assert report.context_chunks == 10


async def test_an_empty_context_supports_nothing() -> None:
    report = await _verifier().verify("A late-payment fee of 500 rupees applies.", ["", "  "])
    assert report.verdict is Verdict.FAILED and report.evidence_mode == EvidenceMode.NONE.value


async def test_report_exposes_chunks_and_mode() -> None:
    report = await _verifier().verify("A late-payment fee of 500 rupees applies.", PAGES)
    data = report.as_dict()
    assert data["context_chunks"] > 100 and data["evidence_mode"] == "retrieved"
    assert data["claims"][0]["document"] == 80


async def test_injected_collaborators_are_used() -> None:
    class CountingRetriever(BM25Retriever):
        builds = 0

        def build(self, context: Any) -> Any:
            CountingRetriever.builds += 1
            return super().build(context)

    verifier = GroundingVerifier(GroundingConfig(), LexicalScorer(),
                                 chunker=SentenceChunker(max_chars=300),
                                 retriever=CountingRetriever())
    await verifier.verify("A late-payment fee of 500 rupees applies.", PAGES)
    assert CountingRetriever.builds == 1  # one index per request, shared by all claims


# --------------------------------------------------------------------------- #
# Token detectors
# --------------------------------------------------------------------------- #


class _RecordingDetector(TokenDetector):
    name = "recording"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def detect(self, prompt: str, answer: str) -> list[TokenProb]:
        self.prompts.append(prompt)
        return [TokenProb(0, len(answer), 0.01)]


def _token_verifier(detector: TokenDetector, **cfg: Any) -> GroundingVerifier:
    config = GroundingConfig(aggregation=AnswerAggregation.MIN, min_claim_tokens=1, **cfg)
    return GroundingVerifier(config, detector)


async def test_token_detector_gets_the_whole_context_when_it_fits() -> None:
    detector = _RecordingDetector()
    report = await _token_verifier(detector).verify("The fee is 500 rupees.", [FEES_PAGE],
                                                    question="What is the fee?")
    assert detector.prompts == [build_prompt([FEES_PAGE], "What is the fee?")]  # unchanged
    assert report.evidence_mode == EvidenceMode.FULL_CONTEXT.value


async def test_token_detector_gets_each_claims_evidence_when_the_context_is_too_long() -> None:
    detector = _RecordingDetector()
    report = await _token_verifier(detector, token_max_length=1024).verify(
        "A late-payment fee of 500 rupees applies.", PAGES, question="What is the fee?")
    [prompt] = detector.prompts
    assert "late-payment fee of 500 rupees" in prompt
    assert len(prompt) < 1024 * 3 + 1_000  # fits the window budget
    assert report.evidence_mode == EvidenceMode.RETRIEVED_FOR_WINDOW.value


async def test_a_client_supplied_prompt_is_used_verbatim() -> None:
    detector = _RecordingDetector()
    report = await _token_verifier(detector).verify("The fee is 500.", PAGES,
                                                    prompt="exact prompt")
    assert detector.prompts == ["exact prompt"]
    assert report.evidence_mode == EvidenceMode.CLIENT_PROMPT.value


async def test_configured_token_budget_overrides_the_derived_one() -> None:
    detector = _RecordingDetector()
    await _token_verifier(detector, evidence={"token_context_chars": 600}).verify(
        "A late-payment fee of 500 rupees applies.", [FEES_PAGE] * 3, question="Fee?")
    assert len(detector.prompts[0]) < 600 + 800


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_hybrid_retrieval_needs_an_encoder() -> None:
    config = GroundingConfig(evidence={"retriever": EvidenceRetrieverType.HYBRID})
    with pytest.raises(ConfigurationError, match="encoder"):
        build_retriever(config)

    class Encoder:
        def embed(self, texts: Any) -> list[list[float]]:
            return [[1.0] for _ in texts]

    assert build_retriever(config, Encoder()).name == "hybrid"
    assert build_retriever(GroundingConfig()).name == "bm25"
