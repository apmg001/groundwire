"""Support scorers: how strongly does a passage support a claim?

Two implementations behind one interface (strategy pattern):

* :class:`LexicalScorer`: dependency-free and sub-millisecond. Scores the
  fraction of the claim's content words found in the passage, and hard-fails
  any claim containing a number absent from the passage. Catches the most
  damaging hallucination in business settings (invented amounts, dates, IDs,
  quantities) but cannot detect paraphrased contradictions ("did" vs "did not").

* :class:`NLIScorer`: a cross-encoder natural-language-inference model
  (DeBERTa by default) returning P(entailment). Much stronger semantically;
  needs the ``nli`` extra and a few hundred MB of RAM. Runs on CPU.

Scorers are synchronous and CPU-bound. The verifier runs them in a worker
thread so the event loop keeps serving other requests.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from groundwire.core.errors import ConfigurationError, GuardrailUnavailableError
from groundwire.guardrails.text import content_tokens, numbers

logger = logging.getLogger(__name__)

Pair = tuple[str, str]  # (premise/passage, hypothesis/claim)


class SupportScorer(ABC):
    """Scores (passage, claim) pairs in [0, 1]; higher = better supported."""

    name: str = "base"

    @abstractmethod
    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        """Score many pairs at once.

        Args:
            pairs: ``(passage, claim)`` tuples.

        Returns:
            One score in [0, 1] per pair, same order.

        Raises:
            GuardrailUnavailableError: If the scorer cannot run.
        """


# --------------------------------------------------------------------------- #
# Lexical
# --------------------------------------------------------------------------- #


class LexicalScorer(SupportScorer):
    """Content-word overlap with a strict numeric check."""

    name = "lexical"

    def score(self, passage: str, claim: str) -> float:
        """Score one pair.

        Args:
            passage: Source passage.
            claim: Claim to check.

        Returns:
            0.0 if the claim cites a number the passage lacks; otherwise the
            fraction of claim content tokens present in the passage.
        """
        claim_numbers = numbers(claim)
        if claim_numbers and not claim_numbers <= numbers(passage):
            return 0.0
        claim_tokens = content_tokens(claim)
        if not claim_tokens:
            return 1.0  # nothing checkable
        return len(claim_tokens & content_tokens(passage)) / len(claim_tokens)

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        """Score many pairs. See :meth:`SupportScorer.score_batch`."""
        return [self.score(passage, claim) for passage, claim in pairs]


# --------------------------------------------------------------------------- #
# NLI
# --------------------------------------------------------------------------- #


class NLIScorer(SupportScorer):
    """Cross-encoder NLI scorer returning P(entailment)."""

    name = "nli"

    def __init__(self, model_name: str, *, batch_size: int = 16, max_length: int = 512) -> None:
        """Load the tokenizer and model (eagerly, at startup).

        Args:
            model_name: Hugging Face model id or local path.
            batch_size: Pairs per forward pass.
            max_length: Token truncation limit per pair.

        Raises:
            ConfigurationError: If the ``nli`` extra is not installed, the model
                cannot be loaded, or it has no entailment label.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "grounding.scorer=nli needs the 'nli' extra: pip install 'groundwire[nli]'"
            ) from exc

        try:
            self._tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self._model: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot load NLI model '{model_name}': {exc}") from exc

        self._model.eval()
        self._torch: Any = torch
        self._batch_size = batch_size
        self._max_length = max_length
        self._entail_idx = self._find_entailment_index(self._model.config.id2label)
        logger.info("NLI scorer loaded", extra={"model": model_name,
                                                "entailment_index": self._entail_idx})

    @staticmethod
    def _find_entailment_index(id2label: dict[int, str]) -> int:
        """Locate the entailment class, whatever order the model uses.

        Args:
            id2label: Model's index -> label mapping.

        Returns:
            Index of the entailment label.

        Raises:
            ConfigurationError: If no label looks like "entailment".
        """
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                return int(idx)
        raise ConfigurationError(f"NLI model has no entailment label: {id2label}")

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        """Score many pairs. See :meth:`SupportScorer.score_batch`."""
        scores: list[float] = []
        try:
            for start in range(0, len(pairs), self._batch_size):
                batch = pairs[start:start + self._batch_size]
                encoded = self._tokenizer(
                    [p for p, _ in batch], [c for _, c in batch],
                    truncation=True, padding=True, max_length=self._max_length,
                    return_tensors="pt",
                )
                with self._torch.inference_mode():
                    logits = self._model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1)[:, self._entail_idx]
                scores.extend(float(p) for p in probs.tolist())
        except (RuntimeError, ValueError) as exc:  # torch raises RuntimeError on OOM etc.
            raise GuardrailUnavailableError(f"NLI scoring failed: {exc}") from exc
        return scores
