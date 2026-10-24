"""Split context documents into chunks that keep their meaning.

The verifier compares each claim in an answer with *chunks* of the
documents the client supplied. How the documents are cut decides what the
verifier can see, so three common failure modes are handled here:

* **A fact split across a boundary** ("The late fee is ₹500." | "It applies
  after 30 days."): chunks *overlap* by ``overlap_sentences``, so a fact on a
  boundary appears whole in at least one chunk.
* **A chunk that lost its subject** ("It applies after 30 days." without the
  "Late payment fee" heading three chunks earlier): every chunk carries the
  nearest section *heading*, and sections are never mixed in one chunk.
* **Lost position**: every chunk records which document it came from and
  where, so a report can say *which page* supports or contradicts a claim
  (clients that send one context entry per page get page-level references).

Chunks are sentence-aligned (a sentence is never cut) and packed greedily up
to ``max_chars``; a single sentence longer than that becomes its own chunk.
With ``overlap_sentences=0`` and ``attach_headings=False`` the output is
exactly :func:`groundwire.guardrails.text.chunk_passages`, which is how the
long-document benchmark reproduces the previous behaviour.

Chunking is pure, deterministic and dependency-free (regex only).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from groundwire.guardrails.text import split_claims

# Headings: markdown ("## Fees"), numbered ("4.2 Late payment", "Section 3 Refunds"),
# or a short title-like line ("Late Payment Fee", "REFUND POLICY") on its own line.
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>\S.{0,120}?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:(?:section|chapter|part|article|clause)\s+)?"
    r"(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+(?P<title>[A-Za-z][^\n]{0,100})$",
    re.IGNORECASE)
_TERMINAL_PUNCTUATION = ".!?:;,"
MAX_HEADING_CHARS = 100
MAX_HEADING_WORDS = 12


@dataclass(frozen=True, slots=True)
class Chunk:
    """One piece of one context document.

    Attributes:
        text: The chunk's sentences, joined with spaces.
        document: Index of the source document in the request's context list.
        index: Position of the chunk across all documents (0-based, stable).
        position: Position of the chunk within its document (0-based).
        heading: Nearest section heading above the chunk ("" if none).
    """

    text: str
    document: int
    index: int
    position: int
    heading: str = ""

    @property
    def scoring_text(self) -> str:
        """Text given to scorers: the heading (if any) followed by the chunk."""
        return f"{self.heading}. {self.text}" if self.heading else self.text


@dataclass(frozen=True, slots=True)
class ChunkedContext:
    """Chunks of all documents in a request.

    Attributes:
        chunks: Chunks in document order.
        documents: Number of documents supplied.
        truncated: True if ``max_chunks`` was reached and later text was not chunked.
    """

    chunks: tuple[Chunk, ...]
    documents: int
    truncated: bool = False

    @property
    def total_chars(self) -> int:
        """Characters across all chunk texts (overlap counted once per chunk)."""
        return sum(len(c.text) for c in self.chunks)


class Chunker(ABC):
    """Splits context documents into chunks."""

    @abstractmethod
    def chunk(self, documents: Sequence[str]) -> ChunkedContext:
        """Chunk ``documents``.

        Args:
            documents: Context documents, in the order the client sent them.

        Returns:
            The chunks.
        """


class SentenceChunker(Chunker):
    """Sentence-aligned chunks with overlap and section headings."""

    def __init__(self, *, max_chars: int = 600, overlap_sentences: int = 1,
                 attach_headings: bool = True, max_chunks: int = 20_000) -> None:
        """Configure the chunker.

        Args:
            max_chars: Soft size limit per chunk (sentences are never cut).
            overlap_sentences: Sentences repeated at the start of the next chunk.
            attach_headings: Detect section headings and attach them to chunks.
            max_chunks: Hard cap per request (bounds CPU and memory).

        Raises:
            ValueError: If a limit is out of range.
        """
        if max_chars < 1 or overlap_sentences < 0 or max_chunks < 1:
            raise ValueError("need max_chars >= 1, overlap_sentences >= 0, max_chunks >= 1")
        self._max_chars = max_chars
        self._overlap = overlap_sentences
        self._headings = attach_headings
        self._max_chunks = max_chunks

    def chunk(self, documents: Sequence[str]) -> ChunkedContext:
        """Chunk ``documents`` (see the module docstring for the rules).

        Args:
            documents: Context documents.

        Returns:
            The chunks, with ``truncated`` set if ``max_chunks`` was hit.
        """
        chunks: list[Chunk] = []
        for document, text in enumerate(documents):
            position = 0
            for heading, sentences in self._sections(text):
                for piece in self._pack(sentences):
                    if len(chunks) >= self._max_chunks:
                        return ChunkedContext(tuple(chunks), len(documents), truncated=True)
                    chunks.append(Chunk(piece, document, len(chunks), position, heading))
                    position += 1
        return ChunkedContext(tuple(chunks), len(documents))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _sections(self, text: str) -> list[tuple[str, list[str]]]:
        """Group a document's sentences under their headings.

        Without heading detection the whole document is one section, which
        reproduces the legacy splitter exactly.
        """
        if not self._headings:
            return [("", split_claims(text))]
        sections: list[tuple[str, list[str]]] = []
        heading: str = ""
        sentences: list[str] = []
        for line in text.splitlines():
            title = heading_of(line)
            if title is not None:
                if sentences:
                    sections.append((heading, sentences))
                heading, sentences = title, []
                continue
            sentences.extend(split_claims(line))
        if sentences:
            sections.append((heading, sentences))
        return sections

    def _pack(self, sentences: list[str]) -> list[str]:
        """Greedy packing up to ``max_chars``, carrying ``overlap`` sentences over."""
        pieces: list[str] = []
        current: list[str] = []
        length = 0
        for sentence in sentences:
            if current and length + 1 + len(sentence) > self._max_chars:
                pieces.append(" ".join(current))
                current = self._carry_over(current)
                length = len(" ".join(current))
            current.append(sentence)
            length = length + 1 + len(sentence) if length else len(sentence)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _carry_over(self, previous: list[str]) -> list[str]:
        """Sentences repeated at the start of the next chunk.

        Overlap is skipped when it would fill most of the next chunk, so long
        sentences don't create near-duplicate chunks.
        """
        if not self._overlap:
            return []
        tail = previous[-self._overlap:]
        return tail if len(" ".join(tail)) <= self._max_chars // 2 else []


def heading_of(line: str) -> str | None:
    """Return the heading title if ``line`` looks like a section heading.

    Deliberately conservative: a false heading would be glued onto unrelated
    chunks. Lines with a colon ("name: The Eagle", common in data records)
    or ending in sentence punctuation are never headings.

    Args:
        line: One line of a document.

    Returns:
        The heading text without its numbering, or ``None``.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return None
    markdown = _MARKDOWN_HEADING.match(stripped)
    if markdown:
        return markdown.group("title").strip()
    if ":" in stripped or stripped[-1] in _TERMINAL_PUNCTUATION:
        return None
    words = stripped.split()
    if len(words) > MAX_HEADING_WORDS:
        return None
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        return numbered.group("title").strip()
    letters = [w for w in words if w[:1].isalpha()]
    if not letters or not stripped[0].isupper():
        return None
    if stripped.isupper() and len(stripped) > 3:
        return stripped
    capitalised = sum(1 for w in letters if w[0].isupper() or len(w) <= 3)
    return stripped if len(words) >= 2 and capitalised == len(letters) else None
