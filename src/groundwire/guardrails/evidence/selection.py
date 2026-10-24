"""Turn search hits into the evidence a scorer reads.

Two consumers, two shapes:

* **Pairwise scorers** (lexical, NLI) check one claim at a time:
  :meth:`EvidenceSelector.windows` returns one *window* per hit (the hit and
  its neighbours: contiguous text, so a fact split across a boundary reads
  whole), scored separately so the best window wins. Windows are not merged:
  a number in an unrelated hit ("Clause 15") must not make a claim's wrong
  figure ("after 15 days") look supported. :meth:`EvidenceSelector.for_claim`
  joins all hits instead, for scorers that should combine facts across them.
* **Token detectors** (LettuceDetect) read one prompt with every claim's
  evidence together and have a fixed window: :meth:`EvidenceSelector.for_answer`
  fills a character budget round-robin, so *every* claim gets its best chunk
  before any claim gets its second, then neighbours, then lower ranks.

Neighbours (the chunk before and after a hit, same document) are cheap and
rescue facts that straddle a boundary despite the overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from groundwire.guardrails.evidence.chunking import Chunk
from groundwire.guardrails.evidence.retrieval import EvidenceIndex, Hit


@dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence chosen for one claim.

    Attributes:
        chunks: Chunks in document order.
        best: The top-ranked chunk (``None`` when nothing matched).
    """

    chunks: tuple[Chunk, ...]
    best: Chunk | None

    @property
    def text(self) -> str:
        """Evidence text for a scorer (chunks with headings, document order)."""
        return " ".join(c.scoring_text for c in self.chunks)


class EvidenceSelector:
    """Selects evidence windows from an :class:`EvidenceIndex`."""

    def __init__(self, *, top_k: int = 3, neighbours: int = 1, max_chars: int = 4_000) -> None:
        """Configure selection.

        Args:
            top_k: Search hits taken per claim.
            neighbours: Chunks added on each side of a hit (same document).
            max_chars: Character budget per claim's evidence.

        Raises:
            ValueError: On out-of-range values.
        """
        if top_k < 1 or neighbours < 0 or max_chars < 1:
            raise ValueError("need top_k >= 1, neighbours >= 0, max_chars >= 1")
        self.top_k = top_k
        self._neighbours = neighbours
        self._max_chars = max_chars

    def windows(self, index: EvidenceIndex, claim: str) -> list[Evidence]:
        """One evidence window per hit: the hit and its neighbours, within budget.

        Args:
            index: Request index.
            claim: Claim text.

        Returns:
            Windows in rank order (empty when no chunk matches the claim).
        """
        windows: list[Evidence] = []
        for hit in index.search(claim, self.top_k):
            chosen = _fill([[hit.chunk], self._around(index, hit)], self._max_chars)
            windows.append(Evidence(_in_document_order(chosen), hit.chunk))
        return windows

    def for_claim(self, index: EvidenceIndex, claim: str) -> Evidence:
        """Evidence for one claim: its hits first, then their neighbours, within budget.

        Args:
            index: Request index.
            claim: Claim text.

        Returns:
            The evidence (empty when no chunk matches the claim at all).
        """
        hits = index.search(claim, self.top_k)
        chosen = _fill([[h.chunk] for h in hits] + [self._around(index, h) for h in hits],
                       self._max_chars)
        return Evidence(_in_document_order(chosen), hits[0].chunk if hits else None)

    def for_answer(self, index: EvidenceIndex, claims: Sequence[str],
                   budget_chars: int) -> tuple[Chunk, ...]:
        """Evidence for a whole answer within one window, fair across claims.

        Order of preference: every claim's rank-1 hit, then every claim's
        rank-2 hit, ..., then neighbours of rank-1 hits. If no claim matches
        anything, the start of the context fills the budget (a detector
        still needs something to read).

        Args:
            index: Request index.
            claims: Claims in the answer.
            budget_chars: Character budget for all evidence together.

        Returns:
            Chosen chunks in document order.
        """
        per_claim: list[list[Hit]] = [index.search(claim, self.top_k) for claim in claims]
        groups: list[list[Chunk]] = []
        for rank in range(self.top_k):
            groups += [[hits[rank].chunk] for hits in per_claim if len(hits) > rank]
        groups += [self._around(index, hits[0]) for hits in per_claim if hits]
        chosen = _fill(groups, budget_chars)
        if not chosen:
            chosen = _fill([[c] for c in index.chunks], budget_chars)
        return _in_document_order(chosen)

    def _around(self, index: EvidenceIndex, hit: Hit) -> list[Chunk]:
        """Neighbours of a hit within the same document (excluding the hit)."""
        if not self._neighbours:
            return []
        chunks = index.chunks
        here = hit.chunk.index
        lo, hi = max(0, here - self._neighbours), min(len(chunks), here + self._neighbours + 1)
        return [c for c in chunks[lo:hi]
                if c.index != here and c.document == hit.chunk.document]


def _fill(groups: list[list[Chunk]], budget: int) -> list[Chunk]:
    """Take chunks group by group, skipping duplicates and anything over budget."""
    chosen: list[Chunk] = []
    seen: set[int] = set()
    used = 0
    for group in groups:
        for chunk in group:
            if chunk.index in seen:
                continue
            size = len(chunk.scoring_text) + 1
            if used + size > budget and chosen:
                continue
            chosen.append(chunk)
            seen.add(chunk.index)
            used += size
    return chosen


def _in_document_order(chunks: list[Chunk]) -> tuple[Chunk, ...]:
    return tuple(sorted(chunks, key=lambda c: c.index))
