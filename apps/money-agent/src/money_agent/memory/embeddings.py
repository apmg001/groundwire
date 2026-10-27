"""Embedders: text to unit-length vectors, for memory search.

Two implementations of the :class:`~money_agent.domain.ports.Embedder` port:

* :class:`HashingEmbedder` (default): the "hashing trick" over words and word
  pairs. Deterministic, dependency-free, microseconds per text, fits a 512 MB
  free-tier server. It measures *word overlap*, not meaning: "bike" and
  "motorcycle" are unrelated to it. Good enough for a few hundred short,
  user-written memories, where the same words come back ("my bike goal").
* :class:`HttpEmbedder`: any OpenAI-compatible ``/embeddings`` endpoint
  (OpenAI, Ollama, vLLM, TEI). Real semantic similarity; costs a network call.

Switching is a setting (``MONEY_EMBEDDER``). Stored vectors must be rebuilt
when switching, since the two vector spaces are unrelated.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence

import httpx

from money_agent.core.errors import AssistantUnavailableError

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your", "is", "am", "are", "was",
    "be", "to", "of", "and", "or", "in", "on", "at", "for", "by", "with", "it", "this",
    "that", "do", "does", "have", "has", "want", "would", "like", "please", "so", "as",
    "about", "what", "how", "much", "can", "will", "should", "from", "up",
})


def _stem(word: str) -> str:
    """Very light stemming so "saving"/"savings", "bikes"/"bike" and "EMIs"/"EMI" meet.

    Deliberately minimal (plural and -ing only): an aggressive stemmer would
    merge words that should stay apart, and this embedder has no other way
    to tell them apart.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"           # subsidies -> subsidy
    for suffix in ("ings", "ing"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]  # savings, saving -> sav
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                 # bikes -> bike, emis -> emi (not "loss")
    return word


def tokens(text: str) -> list[str]:
    """Meaningful, stemmed words of ``text``.

    Args:
        text: Any text.

    Returns:
        Words without stop words.
    """
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashingEmbedder:
    """Signed feature hashing of words and adjacent word pairs."""

    def __init__(self, dimension: int = 384) -> None:
        """Create the embedder.

        Args:
            dimension: Vector size.

        Raises:
            ValueError: If ``dimension`` is below 8.
        """
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector size."""
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts.

        A text with no meaningful words embeds to the zero vector (callers
        treat it as "nothing to search for").

        Args:
            texts: Texts.

        Returns:
            Unit-length vectors (or zero vectors).
        """
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        words = tokens(text)
        features = Counter(words)
        features.update(f"{a} {b}" for a, b in itertools.pairwise(words))
        vector = [0.0] * self._dimension
        for feature, count in features.items():
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            weight = 1.0 + math.log(count)
            if " " in feature:
                weight *= 0.5  # pairs refine, words decide
            vector[(value >> 1) % self._dimension] += sign * weight
        return _normalise(vector)


class HttpEmbedder:
    """OpenAI-compatible ``POST {base_url}/embeddings``."""

    def __init__(self, client: httpx.AsyncClient, *, base_url: str, model: str,
                 dimension: int, api_key: str | None = None, max_retries: int = 2) -> None:
        """Create the embedder.

        Args:
            client: Shared HTTP client (connection pooling; closed by the owner).
            base_url: API base, e.g. ``http://localhost:11434/v1``.
            model: Embedding model name.
            dimension: Expected vector size (checked on every response).
            api_key: Bearer token, if the API needs one.
            max_retries: Retries for transient failures.
        """
        self._client = client
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._model = model
        self._dimension = dimension
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._max_retries = max_retries

    @property
    def dimension(self) -> int:
        """Vector size."""
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with the remote model.

        Args:
            texts: Texts.

        Returns:
            Unit-length vectors, in input order.

        Raises:
            AssistantUnavailableError: The API failed, or returned vectors of
                the wrong size or count.
        """
        if not texts:
            return []
        payload = {"model": self._model, "input": list(texts)}
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self._url, json=payload,
                                                   headers=self._headers)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise AssistantUnavailableError(f"embeddings unreachable: {exc}") from exc
            else:
                if response.status_code < 500 and response.status_code != 429:
                    return self._parse(response, len(texts))
                if attempt == self._max_retries:
                    raise AssistantUnavailableError(
                        f"embeddings HTTP {response.status_code}")
            await asyncio.sleep(0.2 * 2 ** attempt)
        raise AssistantUnavailableError("embeddings failed")  # pragma: no cover - loop exits above

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        if response.status_code != 200:
            raise AssistantUnavailableError(f"embeddings HTTP {response.status_code}")
        try:
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [[float(v) for v in item["embedding"]] for item in data]
        except (ValueError, KeyError, TypeError) as exc:
            raise AssistantUnavailableError("malformed embeddings response") from exc
        if len(vectors) != expected or any(len(v) != self._dimension for v in vectors):
            raise AssistantUnavailableError(
                f"embeddings response has the wrong shape (want {expected} x {self._dimension})")
        return [_normalise(v) for v in vectors]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0 if either is zero).

    Args:
        a: A vector.
        b: A vector of the same size.

    Returns:
        Similarity in -1..1.

    Raises:
        ValueError: If the sizes differ.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same size")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
