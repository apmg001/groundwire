"""Grounding verification and long-document evidence retrieval settings.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import (
    Field,
)

from groundwire.core.config.base import StrictModel


class GroundingScorerType(StrEnum):
    """Available grounding scorers."""

    LEXICAL = "lexical"  # fast, dependency-free; checks content words and numbers
    NLI = "nli"  # transformer NLI model; needs the `nli` extra
    HYBRID = "hybrid"  # numeric gate + NLI; needs the `nli` extra
    TOKEN = "token"  # noqa: S105 - fine-tuned token classifier (LettuceDetect), `nli` extra


class GroundingFailurePolicy(StrEnum):
    """What to do when the grounding check itself cannot run."""

    FAIL_OPEN = "fail_open"  # return the answer, marked as unverified
    FAIL_CLOSED = "fail_closed"  # refuse the request with 503


class AnswerAggregation(StrEnum):
    """How per-claim support scores combine into one verdict for the answer."""

    RATIO = "ratio"  # pass if >= min_supported_ratio of claims meet support_threshold
    MIN = "min"  # score = weakest claim
    LOWEST_TWO = "lowest_two"  # score = mean of the two weakest claims
    MEAN = "mean"  # score = mean of all claims


class EvidenceRetrieverType(StrEnum):
    """How the verifier finds each claim's evidence in long contexts."""

    BM25 = "bm25"  # keyword + exact-number matching; no model, fast
    HYBRID = "hybrid"  # BM25 candidates re-ranked by a sentence encoder (needs `nli` extra)


class EvidenceConfig(StrictModel):
    """How context is chunked and how each claim's evidence is found.

    Applies per request, in memory: nothing is stored. See
    :mod:`groundwire.guardrails.evidence` for the reasoning behind each knob.

    Attributes:
        retriever: ``bm25`` (default) or ``hybrid`` (adds semantic re-ranking).
        embedding_model: Sentence encoder for ``hybrid``.
        hybrid_candidates: BM25 results re-ranked per claim by ``hybrid``.
        chunk_chars: Soft chunk size; sentences are never cut.
        overlap_sentences: Sentences repeated between consecutive chunks, so
            a fact on a boundary appears whole in one chunk.
        attach_headings: Prefix chunks with their section heading.
        top_k: Chunks retrieved per claim.
        neighbours: Chunks added on each side of a hit (same document).
        claim_evidence_chars: Evidence budget per claim (pairwise scorers).
        all_pairs_max_chunks: With ``evidence_passages: 0``, contexts up to
            this many chunks are scored claim-by-chunk exhaustively; larger
            ones switch to retrieval (exhaustive scoring grows as
            claims x chunks).
        max_chunks: Hard cap per request; text beyond it is not checked and
            the report says ``truncated``.
        token_context_chars: Context budget for the token scorer's window;
            ``None`` derives it from ``token_max_length``. Contexts that fit
            are passed whole (as the model was benchmarked); longer ones are
            reduced to each claim's evidence.
    """

    retriever: EvidenceRetrieverType = EvidenceRetrieverType.BM25
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_candidates: Annotated[int, Field(ge=1, le=1_000)] = 50
    chunk_chars: Annotated[int, Field(ge=100, le=8_000)] = 600
    overlap_sentences: Annotated[int, Field(ge=0, le=5)] = 1
    attach_headings: bool = True
    top_k: Annotated[int, Field(ge=1, le=20)] = 3
    neighbours: Annotated[int, Field(ge=0, le=3)] = 1
    claim_evidence_chars: Annotated[int, Field(ge=200, le=50_000)] = 4_000
    all_pairs_max_chunks: Annotated[int, Field(ge=0, le=4_096)] = 64
    max_chunks: Annotated[int, Field(ge=1, le=500_000)] = 20_000
    token_context_chars: Annotated[int | None, Field(ge=500, le=1_000_000)] = None


class GroundingConfig(StrictModel):
    """Grounding-verification settings.

    Attributes:
        scorer: Which scorer to use.
        nli_model: Hugging Face model id for the NLI scorer.
        token_model: Hugging Face model id for the token scorer.
        token_max_length: Maximum tokens (prompt + answer) for the token scorer.
        support_threshold: Per-claim score needed to count as supported.
        min_supported_ratio: Fraction of claims that must be supported for
            the answer to pass.
        min_claim_tokens: Sentences with fewer content tokens are not
            treated as factual claims (e.g. "Sure!").
        aggregation: How claim scores become an answer verdict. ``ratio`` uses
            ``min_supported_ratio``; the score-based modes compare the
            aggregated score with ``support_threshold``.
        evidence_passages: 0 scores each claim against every chunk separately
            (best one wins) while the context has at most
            ``evidence.all_pairs_max_chunks`` chunks, and retrieves evidence
            beyond that. ``k > 0`` always retrieves the ``k`` best chunks per
            claim (plus neighbours) and scores the claim once against them
            joined: far fewer model calls, and claims that combine facts from
            several chunks can be supported.
        evidence: Chunking and evidence retrieval.
        on_error: Policy when the scorer fails.
    """

    scorer: GroundingScorerType = GroundingScorerType.LEXICAL
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    token_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"  # noqa: S105 - model id
    token_max_length: Annotated[int, Field(ge=32, le=32_768)] = 4096
    support_threshold: Annotated[float, Field(ge=0, le=1)] = 0.6
    min_supported_ratio: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_claim_tokens: Annotated[int, Field(ge=1, le=50)] = 3
    aggregation: AnswerAggregation = AnswerAggregation.RATIO
    evidence_passages: Annotated[int, Field(ge=0, le=10)] = 0
    evidence: EvidenceConfig = EvidenceConfig()
    on_error: GroundingFailurePolicy = GroundingFailurePolicy.FAIL_OPEN
