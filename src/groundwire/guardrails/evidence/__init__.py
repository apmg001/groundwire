"""Evidence for grounding checks: chunk the context, find each claim's support.

Per request and in memory only: chunk the documents the client sent
(:mod:`.chunking`), index them (:mod:`.retrieval`), pick each claim's
evidence (:mod:`.selection`), then discard everything. This lets the
verifier check answers against documents far longer than a scorer's input
window, without storing any customer document.
"""

from groundwire.guardrails.evidence.chunking import (
    Chunk,
    ChunkedContext,
    Chunker,
    SentenceChunker,
    heading_of,
)
from groundwire.guardrails.evidence.retrieval import (
    BM25Index,
    BM25Retriever,
    EvidenceIndex,
    Hit,
    HybridIndex,
    HybridRetriever,
    Retriever,
    TextEncoder,
)
from groundwire.guardrails.evidence.selection import Evidence, EvidenceSelector

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "Chunk",
    "ChunkedContext",
    "Chunker",
    "Evidence",
    "EvidenceIndex",
    "EvidenceSelector",
    "Hit",
    "HybridIndex",
    "HybridRetriever",
    "Retriever",
    "SentenceChunker",
    "TextEncoder",
    "heading_of",
]
