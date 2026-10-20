"""Sentence embeddings for semantic cache matching.

An :class:`Embedder` turns text into a unit-length vector, so cosine
similarity is just a dot product. Questions that mean the same thing ("how do
I reset my password" / "I forgot my password, how can I change it") land close
together even with few words in common.

The default model is ``sentence-transformers/all-MiniLM-L6-v2`` (Apache-2.0):
22M parameters, 384 dimensions, a few milliseconds per question on a laptop
CPU. It is loaded with plain ``transformers`` and mean pooling, so no extra
dependency beyond the existing ``nli`` extra is needed.

Embedding is CPU-bound; callers run :meth:`Embedder.embed` in a worker thread
(``asyncio.to_thread``) so the event loop keeps serving other requests.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from groundwire.core.errors import CacheUnavailableError, ConfigurationError

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Vector = tuple[float, ...]

# math.sumprod (Python 3.12+) is a fast, accurate C dot product; fall back on 3.11.
_SUMPROD: Callable[[Iterable[float], Iterable[float]], float] | None = getattr(
    math, "sumprod", None)

# --------------------------------------------------------------------------- #
# Vector maths
# --------------------------------------------------------------------------- #


def normalise(values: Sequence[float]) -> Vector:
    """Scale a vector to unit length.

    Args:
        values: Raw vector.

    Returns:
        The unit vector.

    Raises:
        ValueError: If the vector is empty, all zeros, or not finite.
    """
    if not values:
        raise ValueError("cannot normalise an empty vector")
    norm = math.sqrt(math.fsum(v * v for v in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalise a zero or non-finite vector")
    return tuple(v / norm for v in values)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two *unit* vectors (a plain dot product).

    Args:
        a: First unit vector.
        b: Second unit vector, same length.

    Returns:
        Similarity in [-1, 1].

    Raises:
        ValueError: If the lengths differ.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    if _SUMPROD is not None:
        return _SUMPROD(a, b)
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class Embedder(ABC):
    """Maps texts to unit-length vectors."""

    name: str = "embedder"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed.

        Returns:
            One unit vector per text, in order.

        Raises:
            CacheUnavailableError: If the model cannot run.
        """


# --------------------------------------------------------------------------- #
# Hugging Face implementation
# --------------------------------------------------------------------------- #


class TransformersEmbedder(Embedder):
    """Mean-pooled sentence embeddings from a ``transformers`` encoder.

    Construct with :meth:`load` in production; the constructor accepts
    pre-built objects so tests can inject fakes.
    """

    def __init__(self, *, tokenizer: Any, model: Any, torch_module: Any,
                 max_length: int = 256, batch_size: int = 32,
                 model_name: str = "custom") -> None:
        """Wrap a loaded tokenizer and encoder.

        Args:
            tokenizer: Tokenizer for the model.
            model: Encoder returning ``last_hidden_state``.
            torch_module: The ``torch`` module.
            max_length: Token cap per text.
            batch_size: Texts per forward pass.
            model_name: Label for logs.

        Raises:
            ConfigurationError: On invalid sizes.
        """
        if max_length < 1 or batch_size < 1:
            raise ConfigurationError("embedder max_length and batch_size must be >= 1")
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._max_length = max_length
        self._batch_size = batch_size
        self.name = model_name

    @classmethod
    def load(cls, model_name: str = DEFAULT_EMBEDDING_MODEL, *, max_length: int = 256,
             num_threads: int | None = None) -> TransformersEmbedder:
        """Download (or read from cache) and load an embedding model.

        Args:
            model_name: Hugging Face model id or local directory.
            max_length: Token cap per text.
            num_threads: CPU threads for torch; ``None`` keeps torch's default.

        Returns:
            A ready embedder.

        Raises:
            ConfigurationError: If the ``nli`` extra is missing or the model
                cannot be loaded.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "semantic caching needs the 'nli' extra (make install-nli), "
                "or set cache.semantic: false for exact matching only"
            ) from exc
        if num_threads is not None:
            torch.set_num_threads(max(1, num_threads))
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"cannot load embedding model '{model_name}': {exc}") from exc
        model.eval()
        logger.info("cache embedder loaded", extra={"model": model_name})
        return cls(tokenizer=tokenizer, model=model, torch_module=torch,
                   max_length=max_length, model_name=model_name)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts. See :meth:`Embedder.embed`."""
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(list(texts[start:start + self._batch_size])))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[Vector]:
        """Run one forward pass with attention-masked mean pooling.

        Args:
            batch: Texts (at most ``batch_size``).

        Returns:
            Unit vectors.

        Raises:
            CacheUnavailableError: If tokenisation or inference fails.
        """
        try:
            encoded = self._tokenizer(batch, padding=True, truncation=True,
                                      max_length=self._max_length, return_tensors="pt")
            with self._torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            rows: list[list[float]] = pooled.tolist()
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise CacheUnavailableError(f"embedding failed: {exc}") from exc
        try:
            return [normalise(row) for row in rows]
        except ValueError as exc:
            raise CacheUnavailableError(f"embedding produced an invalid vector: {exc}") from exc
