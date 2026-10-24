"""Find the chunks most likely to support (or contradict) a claim.

At verification time the search query is the *claim*, not the user's
question: the answer itself says what to look for. A claim that no chunk
supports, even its best match, is unsupported.

Two retrievers behind one port (:class:`Retriever` builds an
:class:`EvidenceIndex` per request; the index answers ``search(query, k)``):

* :class:`BM25Retriever` (default): Okapi BM25 over stemmed content words,
  plus every number as its own term with extra weight, because invented or
  changed figures are the hallucination that matters most and an exact
  ``#1200`` match is the strongest evidence signal there is. Pure Python,
  no model, cheap enough to index tens of thousands of chunks per request.
* :class:`HybridRetriever`: BM25 picks candidates, a sentence encoder
  re-ranks them by meaning, and the two rankings are merged with reciprocal
  rank fusion (RRF). Catches paraphrases BM25 misses ("refund" vs "money
  back"). The encoder is any object with ``embed(texts)`` returning unit
  vectors (:class:`TextEncoder`), so this module does not depend on where
  the model comes from; the composition root injects one.

Indexes are built per request and discarded with it: nothing is stored.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from groundwire.guardrails.evidence.chunking import Chunk, ChunkedContext
from groundwire.guardrails.text import NUMBER_TERM_PREFIX, retrieval_terms

RRF_K = 60  # standard reciprocal-rank-fusion constant (Cormack et al., 2009)


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result.

    Attributes:
        chunk: The matching chunk.
        score: Retriever-specific relevance (higher is better; not comparable
            across retrievers).
        rank: 1-based rank in the result list.
    """

    chunk: Chunk
    score: float
    rank: int


class EvidenceIndex(ABC):
    """A searchable view of one request's chunks."""

    @property
    @abstractmethod
    def chunks(self) -> tuple[Chunk, ...]:
        """All indexed chunks, in document order."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[Hit]:
        """Return up to ``k`` chunks most relevant to ``query``, best first.

        Args:
            query: Search text (a claim).
            k: Maximum results.

        Returns:
            Hits ranked best first; empty when nothing is relevant.
        """


class Retriever(ABC):
    """Builds an :class:`EvidenceIndex` for one request's chunks."""

    name: str = "retriever"

    @abstractmethod
    def build(self, context: ChunkedContext) -> EvidenceIndex:
        """Index ``context``.

        Args:
            context: Chunked request context.

        Returns:
            A searchable index (valid for this request only).
        """


class TextEncoder(Protocol):
    """Anything that turns texts into unit-length vectors."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed ``texts`` (one unit vector each, in order)."""
        ...


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


class BM25Index(EvidenceIndex):
    """Okapi BM25 over one request's chunks (inverted index, built eagerly)."""

    def __init__(self, chunks: tuple[Chunk, ...], *, k1: float, b: float,
                 number_weight: float) -> None:
        """Index ``chunks``.

        Args:
            chunks: Chunks to index.
            k1: Term-frequency saturation.
            b: Length normalisation (0 = none, 1 = full).
            number_weight: Multiplier for number terms (``#1200``).
        """
        self._chunks = chunks
        self._k1, self._b, self._number_weight = k1, b, number_weight
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for position, chunk in enumerate(chunks):
            counts = Counter(retrieval_terms(chunk.scoring_text))
            self._lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((position, tf))
        total = sum(self._lengths)
        self._average_length = total / len(chunks) if chunks and total else 1.0

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """All indexed chunks, in document order."""
        return self._chunks

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency (always positive).

        Args:
            term: Index term.

        Returns:
            ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
        """
        df = len(self._postings.get(term, ()))
        n = len(self._chunks)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Hit]:
        """Rank chunks by BM25 against the query's terms.

        Args:
            query: Search text.
            k: Maximum results.

        Returns:
            Up to ``k`` hits with a positive score, best first (ties: earlier chunk).
        """
        if k < 1 or not self._chunks:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(retrieval_terms(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            weight = self.idf(term) * (self._number_weight
                                       if term.startswith(NUMBER_TERM_PREFIX) else 1.0)
            for position, tf in postings:
                norm = 1 - self._b + self._b * self._lengths[position] / self._average_length
                scores[position] += weight * tf * (self._k1 + 1) / (tf + self._k1 * norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [Hit(self._chunks[pos], score, rank)
                for rank, (pos, score) in enumerate(ranked, start=1)]


class BM25Retriever(Retriever):
    """Keyword retrieval with exact number matching (no model)."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, number_weight: float = 2.0) -> None:
        """Configure BM25.

        Args:
            k1: Term-frequency saturation (standard 1.2-2.0).
            b: Length normalisation (standard 0.75).
            number_weight: Extra weight for number terms.

        Raises:
            ValueError: On out-of-range parameters.
        """
        if k1 < 0 or not 0 <= b <= 1 or number_weight <= 0:
            raise ValueError("need k1 >= 0, 0 <= b <= 1, number_weight > 0")
        self._k1, self._b, self._number_weight = k1, b, number_weight

    def build(self, context: ChunkedContext) -> BM25Index:
        """Index ``context`` with BM25.

        Args:
            context: Chunked request context.

        Returns:
            The index.
        """
        return BM25Index(context.chunks, k1=self._k1, b=self._b,
                         number_weight=self._number_weight)


# --------------------------------------------------------------------------- #
# Hybrid: BM25 candidates, dense re-ranking, reciprocal rank fusion
# --------------------------------------------------------------------------- #


class HybridIndex(EvidenceIndex):
    """BM25 first stage, embedding re-rank, RRF merge. Embeds lazily, once per chunk."""

    def __init__(self, sparse: BM25Index, encoder: TextEncoder, *, candidates: int) -> None:
        """Wrap a BM25 index.

        Args:
            sparse: BM25 index over the same chunks.
            encoder: Sentence encoder returning unit vectors.
            candidates: Chunks re-ranked per query (bounds embedding cost).
        """
        self._sparse = sparse
        self._encoder = encoder
        self._candidates = candidates
        self._vectors: dict[int, Sequence[float]] = {}

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """All indexed chunks, in document order."""
        return self._sparse.chunks

    def search(self, query: str, k: int) -> list[Hit]:
        """Merge BM25 and embedding rankings over the candidate set.

        Candidates are BM25's top results, topped up with chunks in document
        order when BM25 finds fewer (so a pure paraphrase with no shared word
        can still be found in small documents).

        Args:
            query: Search text.
            k: Maximum results.

        Returns:
            Up to ``k`` hits ranked by fused score.
        """
        if k < 1 or not self.chunks:
            return []
        sparse_hits = self._sparse.search(query, self._candidates)
        pool = [h.chunk for h in sparse_hits]
        seen = {c.index for c in pool}
        for chunk in self.chunks:
            if len(pool) >= self._candidates:
                break
            if chunk.index not in seen:
                pool.append(chunk)
                seen.add(chunk.index)
        self._embed_missing(pool)
        query_vector = self._encoder.embed([query])[0]
        dense_ranked = sorted(pool, key=lambda c: (-_dot(query_vector, self._vectors[c.index]),
                                                   c.index))
        fused: dict[int, float] = defaultdict(float)
        for hit in sparse_hits:
            fused[hit.chunk.index] += 1.0 / (RRF_K + hit.rank)
        for rank, chunk in enumerate(dense_ranked, start=1):
            fused[chunk.index] += 1.0 / (RRF_K + rank)
        by_index = {c.index: c for c in pool}
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [Hit(by_index[i], score, rank) for rank, (i, score) in enumerate(ranked, start=1)]

    def _embed_missing(self, chunks: list[Chunk]) -> None:
        missing = [c for c in chunks if c.index not in self._vectors]
        if not missing:
            return
        vectors = self._encoder.embed([c.scoring_text for c in missing])
        if len(vectors) != len(missing):
            raise ValueError("encoder returned the wrong number of vectors")
        for chunk, vector in zip(missing, vectors, strict=True):
            self._vectors[chunk.index] = vector


class HybridRetriever(Retriever):
    """BM25 plus semantic re-ranking (needs a sentence encoder)."""

    name = "hybrid"

    def __init__(self, encoder: TextEncoder, *, sparse: BM25Retriever | None = None,
                 candidates: int = 50) -> None:
        """Configure the hybrid retriever.

        Args:
            encoder: Sentence encoder returning unit vectors.
            sparse: BM25 first stage (defaults to :class:`BM25Retriever`).
            candidates: Chunks re-ranked per query.

        Raises:
            ValueError: If ``candidates`` < 1.
        """
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._encoder = encoder
        self._sparse = sparse or BM25Retriever()
        self._candidates = candidates

    def build(self, context: ChunkedContext) -> HybridIndex:
        """Index ``context`` (BM25 now, embeddings lazily per query).

        Args:
            context: Chunked request context.

        Returns:
            The index.
        """
        return HybridIndex(self._sparse.build(context), self._encoder,
                           candidates=self._candidates)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product (= cosine similarity for unit vectors)."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b, strict=True))
