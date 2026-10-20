"""Tests for grounding text utilities, scorers and the verifier."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from groundwire.core.config import AnswerAggregation, GroundingConfig, GroundingFailurePolicy
from groundwire.core.errors import GuardrailUnavailableError
from groundwire.guardrails.grounding import GroundingVerifier, Verdict, aggregate
from groundwire.guardrails.scorers import LexicalScorer, NLIScorer, Pair, SupportScorer
from groundwire.guardrails.text import chunk_passages, content_tokens, numbers, split_claims

CONTEXT = [
    "Invoice INV-1042 was issued to Acme Corp on 1 March 2026. "
    "The total amount due is 1,200 USD. Payment terms are net 30 days."
]

# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #


def test_split_claims_handles_sentences_bullets_and_abbreviations() -> None:
    text = "The total is 3.5 million. It is due soon, e.g. next week!\n- First item\n2) Second"
    assert split_claims(text) == [
        "The total is 3.5 million.",
        "It is due soon, e.g. next week!",
        "First item",
        "Second",
    ]


def test_numbers_are_normalised() -> None:
    assert numbers("Pay $1,200.00 or 15% by 2026") == {"1200", "15%", "2026"}


def test_content_tokens_drop_stopwords_and_stem() -> None:
    assert content_tokens("The invoices are for the companies") == {"invoice", "company"}


def test_chunk_passages_respects_soft_limit() -> None:
    doc = " ".join(f"Sentence number {i} is here." for i in range(50))
    passages = chunk_passages([doc], max_chars=100)
    assert len(passages) > 1
    assert all(len(p) <= 100 for p in passages)


# --------------------------------------------------------------------------- #
# Lexical scorer
# --------------------------------------------------------------------------- #


def test_lexical_scorer_supports_matching_claim() -> None:
    assert LexicalScorer().score(CONTEXT[0], "The total amount due is 1,200 USD.") == 1.0


def test_lexical_scorer_rejects_invented_number() -> None:
    # Same words, wrong amount: the classic billing hallucination.
    assert LexicalScorer().score(CONTEXT[0], "The total amount due is 1,500 USD.") == 0.0


def test_lexical_scorer_partial_overlap() -> None:
    score = LexicalScorer().score(CONTEXT[0], "Acme Corp received a refund voucher")
    assert 0 < score < 0.6


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


def _verifier(scorer: SupportScorer | None = None, **cfg: object) -> GroundingVerifier:
    return GroundingVerifier(GroundingConfig(**cfg), scorer or LexicalScorer())  # type: ignore[arg-type]


async def test_grounded_answer_passes() -> None:
    answer = "Invoice INV-1042 was issued to Acme Corp. The total amount due is 1,200 USD."
    report = await _verifier().verify(answer, CONTEXT)
    assert report.verdict is Verdict.PASSED
    assert report.supported_ratio == 1.0
    assert len(report.claims) == 2


async def test_hallucinated_answer_fails_and_names_the_claim() -> None:
    answer = ("Invoice INV-1042 was issued to Acme Corp. The total amount due is 4,800 USD. "
              "Acme Corp also qualifies for a loyalty discount programme.")
    report = await _verifier().verify(answer, CONTEXT)
    assert report.verdict is Verdict.FAILED
    unsupported = [c.claim for c in report.unsupported]
    assert any("4,800" in c for c in unsupported)
    assert any("loyalty" in c for c in unsupported)


async def test_answer_without_claims_passes_trivially() -> None:
    report = await _verifier().verify("Sure! Here you go.", CONTEXT)
    assert report.verdict is Verdict.PASSED
    assert report.claims == []


async def test_report_is_json_serialisable() -> None:
    import json

    report = await _verifier().verify("The total amount due is 1,200 USD.", CONTEXT)
    data = report.as_dict()
    json.dumps(data)
    assert data["verdict"] == "passed"


class _BrokenScorer(SupportScorer):
    name = "broken"

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        raise RuntimeError("model crashed")


class _WrongLengthScorer(SupportScorer):
    name = "wrong"

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        return [1.0]


async def test_scorer_failure_fail_open_returns_unverified() -> None:
    report = await _verifier(_BrokenScorer()).verify("The total amount due is 1,200 USD.",
                                                    CONTEXT)
    assert report.verdict is Verdict.UNVERIFIED
    assert report.error is not None


async def test_scorer_failure_fail_closed_raises() -> None:
    verifier = _verifier(_BrokenScorer(), on_error=GroundingFailurePolicy.FAIL_CLOSED)
    with pytest.raises(GuardrailUnavailableError):
        await verifier.verify("The total amount due is 1,200 USD.", CONTEXT)


async def test_scorer_returning_wrong_shape_is_treated_as_failure() -> None:
    report = await _verifier(_WrongLengthScorer()).verify(
        "The total amount due is 1,200 USD. Payment terms are net 30 days.", CONTEXT)
    assert report.verdict is Verdict.UNVERIFIED


def test_nli_entailment_index_lookup() -> None:
    assert NLIScorer._find_entailment_index(
        {0: "contradiction", 1: "entailment", 2: "neutral"}) == 1
    assert NLIScorer._find_entailment_index({0: "ENTAILMENT", 1: "NOT_ENTAILMENT"}) == 0


@pytest.mark.parametrize(("text", "expected"), [
    ("Open 11 am to 5:30 PM", {"t11:00", "t17:30"}),
    ("hours: 11:0-17:30", {"t11:00", "t17:30"}),
    ("from 9:00 a.m. until 2 p.m.", {"t09:00", "t14:00"}),
    ("12 pm and 12 am", {"t12:00", "t00:00"}),
    ("score 25:99", {"25", "99"}),  # not a valid time: kept as plain numbers
])
def test_numbers_normalise_clock_times(text: str, expected: set[str]) -> None:
    assert numbers(text) == expected


def test_lexical_accepts_same_time_in_different_formats() -> None:
    passage = "hours.Monday: 11:0-17:0."
    # Same times, different format: passes the number gate (scored on word overlap).
    assert LexicalScorer().score(passage, "It is open from 11 am to 5 pm on Monday.") > 0.0
    # A genuinely different opening time is still rejected outright.
    assert LexicalScorer().score(passage, "It is open from 9 am to 5 pm on Monday.") == 0.0


# --------------------------------------------------------------------------- #
# Aggregation and evidence-first pairing
# --------------------------------------------------------------------------- #



@pytest.mark.parametrize(("aggregation", "expected"), [
    (AnswerAggregation.MIN, 0.1),
    (AnswerAggregation.LOWEST_TWO, 0.3),
    (AnswerAggregation.MEAN, 0.6),
])
def test_aggregate(aggregation: AnswerAggregation, expected: float) -> None:
    assert aggregate([0.9, 0.1, 0.5, 0.9], aggregation) == pytest.approx(expected)


def test_aggregate_single_claim_and_errors() -> None:
    assert aggregate([0.4], AnswerAggregation.LOWEST_TWO) == 0.4
    with pytest.raises(ValueError):
        aggregate([], AnswerAggregation.MIN)
    with pytest.raises(ValueError):
        aggregate([0.5], AnswerAggregation.RATIO)


async def test_score_based_verdict_uses_support_threshold() -> None:
    answer = ("Invoice INV-1042 was issued to Acme Corp. The total amount due is 1,200 USD. "
              "Acme Corp also qualifies for a loyalty discount programme.")
    lenient = _verifier(aggregation=AnswerAggregation.MEAN, support_threshold=0.5)
    strict = _verifier(aggregation=AnswerAggregation.MIN, support_threshold=0.5)
    mean_report = await lenient.verify(answer, CONTEXT)
    min_report = await strict.verify(answer, CONTEXT)
    assert mean_report.aggregation == "mean"
    assert mean_report.verdict is Verdict.PASSED  # one weak claim averaged away
    assert min_report.verdict is Verdict.FAILED  # weakest claim decides
    assert min_report.answer_score == min(c.score for c in min_report.claims)
    assert min_report.as_dict()["answer_score"] == round(min_report.answer_score, 4)


class _CountingScorer(LexicalScorer):
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        self.pairs.extend(pairs)
        return super().score_batch(pairs)


async def test_evidence_first_scores_each_claim_once_against_joined_passages() -> None:
    context = ["The museum opened in 1960.", "Unrelated text about weather in Spain.",
               "It holds 4,000 paintings by local artists."]
    answer = "The museum opened in 1960 and holds 4,000 paintings."
    all_pairs, evidence = _CountingScorer(), _CountingScorer()
    await _verifier(all_pairs).verify(answer, context)
    report = await _verifier(evidence, evidence_passages=2).verify(answer, context)

    assert len(evidence.pairs) == 1  # one premise per claim instead of one per passage
    premise, _ = evidence.pairs[0]
    assert "1960" in premise and "4,000" in premise and "weather" not in premise
    # Facts spread over two passages: only the joined evidence supports the claim.
    assert report.claims[0].score == 1.0
    assert max(LexicalScorer().score(p, answer) for p, _ in all_pairs.pairs) == 0.0
