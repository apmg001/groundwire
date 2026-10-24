"""Tests for evidence chunking, retrieval and selection."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import ClassVar

import pytest

from groundwire.guardrails.evidence import (
    BM25Retriever,
    Chunk,
    EvidenceSelector,
    HybridRetriever,
    SentenceChunker,
    heading_of,
)
from groundwire.guardrails.text import chunk_passages, retrieval_terms

FILLER = " ".join(f"Clause {i} covers general account rules and usage terms." for i in range(60))
FEES = "Late Payment Fee\nA late-payment fee of 500 rupees applies. It is charged after 30 days."


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def _texts(chunks: Sequence[Chunk]) -> list[str]:
    return [c.text for c in chunks]


def test_without_overlap_or_headings_the_legacy_splitter_is_reproduced() -> None:
    docs = [FILLER, FEES, "Short one.\nTwo lines here."]
    legacy = SentenceChunker(max_chars=200, overlap_sentences=0, attach_headings=False)
    assert _texts(legacy.chunk(docs).chunks) == chunk_passages(docs, max_chars=200)


def test_overlap_repeats_the_last_sentence_at_the_start_of_the_next_chunk() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(6)]
    chunks = SentenceChunker(max_chars=60, overlap_sentences=1,
                             attach_headings=False).chunk([" ".join(sentences)]).chunks
    for previous, current in pairwise(chunks):
        assert current.text.startswith(previous.text.split(". ")[-1].rstrip("."))


def test_overlap_is_skipped_when_it_would_fill_half_the_next_chunk() -> None:
    long_sentence = "This sentence is deliberately long enough to exceed half the limit."
    chunks = SentenceChunker(max_chars=80, overlap_sentences=1, attach_headings=False).chunk(
        [f"{long_sentence} {long_sentence.replace('This', 'That')}"]).chunks
    assert len(chunks) == 2 and not chunks[1].text.startswith("This")


def test_headings_are_attached_and_sections_never_mix() -> None:
    doc = "# Refunds\nRefunds take 5 days.\n## Late Payment Fee\nIt is charged after 30 days."
    chunks = SentenceChunker(max_chars=1_000).chunk([doc]).chunks
    assert [(c.heading, c.text) for c in chunks] == [
        ("Refunds", "Refunds take 5 days."),
        ("Late Payment Fee", "It is charged after 30 days.")]
    assert chunks[1].scoring_text == "Late Payment Fee. It is charged after 30 days."


@pytest.mark.parametrize(("line", "expected"), [
    ("## Fees", "Fees"),
    ("4.2 Late payment fee", "Late payment fee"),
    ("Section 3 Refunds and Returns", "Refunds and Returns"),
    ("REFUND POLICY", "REFUND POLICY"),
    ("Late Payment Fee", "Late Payment Fee"),
    ("name: The Eagle", None),          # data records are not headings
    ("Name: The Eagle", None),          # ... even when capitalised
    ("The fee is 500.", None),          # sentences are not headings
    ("We accept cards", None),          # lower-case words: a sentence fragment
    ("", None),
    ("A" * 150, None),
])
def test_heading_detection_is_conservative(line: str, expected: str | None) -> None:
    assert heading_of(line) == expected


def test_chunks_record_document_and_position() -> None:
    chunks = SentenceChunker(max_chars=40, attach_headings=False).chunk(
        ["One sentence here. Another sentence here.", "Page two text."]).chunks
    assert [(c.document, c.position, c.index) for c in chunks] == [(0, 0, 0), (0, 1, 1),
                                                                    (1, 0, 2)]


def test_max_chunks_truncates_and_says_so() -> None:
    context = SentenceChunker(max_chars=50, max_chunks=3).chunk([FILLER])
    assert len(context.chunks) == 3 and context.truncated


@pytest.mark.parametrize("bad", [{"max_chars": 0}, {"overlap_sentences": -1}, {"max_chunks": 0}])
def test_chunker_rejects_bad_limits(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SentenceChunker(**bad)


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


def _index(docs: list[str], **kwargs: float) -> object:
    return BM25Retriever(**kwargs).build(SentenceChunker(max_chars=200).chunk(docs))


def test_retrieval_terms_keep_repeats_and_whole_numbers() -> None:
    assert retrieval_terms("Fees: fees of 1,200 at 5 pm") == ["fee", "fee", "#1200", "#t17:00"]


def test_bm25_finds_the_chunk_with_the_claims_numbers() -> None:
    index = BM25Retriever().build(SentenceChunker().chunk([FILLER, FEES, FILLER]))
    [hit] = index.search("The late fee is 500 rupees", 1)
    assert hit.chunk.document == 1 and "500" in hit.chunk.text and hit.rank == 1


def test_bm25_weights_numbers_above_common_words() -> None:
    docs = ["The account fee is 999 and the account fee is due.", "Payment of 750 is due."]
    unweighted = _index(docs, number_weight=0.01).search("account fee 750", 1)  # type: ignore[attr-defined]
    weighted = _index(docs, number_weight=5.0).search("account fee 750", 1)  # type: ignore[attr-defined]
    assert "999" in unweighted[0].chunk.text and "750" in weighted[0].chunk.text


def test_bm25_returns_nothing_for_unrelated_or_empty_queries() -> None:
    index = _index([FEES])
    assert index.search("zebra quantum", 3) == []  # type: ignore[attr-defined]
    assert index.search("late fee", 0) == []  # type: ignore[attr-defined]
    assert BM25Retriever().build(SentenceChunker().chunk([])).search("late fee", 3) == []


@pytest.mark.parametrize("bad", [{"k1": -1.0}, {"b": 1.5}, {"number_weight": 0.0}])
def test_bm25_rejects_bad_parameters(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(**bad)


# --------------------------------------------------------------------------- #
# Hybrid
# --------------------------------------------------------------------------- #


class _ConceptEncoder:
    """Maps words to concepts so a paraphrase shares a vector with its source."""

    CONCEPTS: ClassVar[dict[str, int]] = {
        "refund": 0, "money": 0, "back": 0, "rainfall": 1, "monsoon": 1}

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        vectors = []
        for text in texts:
            vector = [0.0, 0.0, 0.1]
            for word in text.lower().replace(".", " ").split():
                if word in self.CONCEPTS:
                    vector[self.CONCEPTS[word]] += 1.0
            norm = sum(v * v for v in vector) ** 0.5
            vectors.append([v / norm for v in vector])
        return vectors


def test_hybrid_finds_a_paraphrase_that_bm25_misses() -> None:
    docs = ["Monsoon rainfall was heavy this year.", "A refund is issued within five days."]
    context = SentenceChunker().chunk(docs)
    query = "You get your money back"  # no word in common with the refund sentence
    assert BM25Retriever().build(context).search(query, 1) == []
    [hit] = HybridRetriever(_ConceptEncoder()).build(context).search(query, 1)
    assert "refund" in hit.chunk.text


def test_hybrid_embeds_each_chunk_only_once_per_request() -> None:
    encoder = _ConceptEncoder()
    index = HybridRetriever(encoder).build(SentenceChunker().chunk(["A refund is issued."] * 3))
    index.search("refund", 2)
    index.search("money back", 2)
    chunk_embeddings = [t for t in encoder.embedded if t not in ("refund", "money back")]
    assert len(chunk_embeddings) == 3


def test_hybrid_rejects_an_encoder_returning_the_wrong_count() -> None:
    class Broken:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0]]

    index = HybridRetriever(Broken()).build(SentenceChunker().chunk(["One.", "Two."]))
    with pytest.raises(ValueError, match="wrong number"):
        index.search("one", 1)


def test_hybrid_rejects_zero_candidates() -> None:
    with pytest.raises(ValueError):
        HybridRetriever(_ConceptEncoder(), candidates=0)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _numbered_context(n: int = 8) -> object:
    docs = [" ".join(f"Item {d}-{i} costs {d * 100 + i} rupees." for i in range(5))
            for d in range(n)]
    return BM25Retriever().build(SentenceChunker(max_chars=60, overlap_sentences=0,
                                                 attach_headings=False).chunk(docs))


def test_claim_evidence_adds_same_document_neighbours_in_order() -> None:
    index = _numbered_context()
    evidence = EvidenceSelector(top_k=1, neighbours=1).for_claim(index, "costs 302 rupees")  # type: ignore[arg-type]
    assert evidence.best is not None and "302" in evidence.best.text
    assert [c.index for c in evidence.chunks] == sorted(c.index for c in evidence.chunks)
    assert {c.document for c in evidence.chunks} == {3}
    assert len(evidence.chunks) > 1


def test_neighbours_never_cross_into_another_document() -> None:
    docs = ["Page one is about rainfall.", "Page two says the fee is 500.",
            "Page three is about trees."]
    index = BM25Retriever().build(SentenceChunker().chunk(docs))
    [window] = EvidenceSelector(top_k=1, neighbours=1).windows(index, "the fee is 500")
    assert [c.document for c in window.chunks] == [1]


def test_claim_evidence_respects_its_budget() -> None:
    index = _numbered_context()
    evidence = EvidenceSelector(top_k=3, neighbours=1, max_chars=50).for_claim(
        index, "costs 302 rupees")  # type: ignore[arg-type]
    assert len(evidence.chunks) == 1  # the best hit is always kept


def test_answer_evidence_gives_every_claim_its_best_chunk_first() -> None:
    index = _numbered_context()
    claims = ["costs 101 rupees", "costs 504 rupees", "costs 702 rupees"]
    budget = sum(len(index.search(c, 1)[0].chunk.scoring_text) + 1 for c in claims)  # type: ignore[attr-defined]
    chosen = EvidenceSelector(top_k=3, neighbours=1).for_answer(index, claims, budget)  # type: ignore[arg-type]
    assert {c.document for c in chosen} == {1, 5, 7}


def test_answer_evidence_falls_back_to_the_start_when_nothing_matches() -> None:
    index = _numbered_context()
    chosen = EvidenceSelector().for_answer(index, ["zebra quantum"], 200)  # type: ignore[arg-type]
    assert chosen and chosen[0].index == 0


@pytest.mark.parametrize("bad", [{"top_k": 0}, {"neighbours": -1}, {"max_chars": 0}])
def test_selector_rejects_bad_settings(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        EvidenceSelector(**bad)
