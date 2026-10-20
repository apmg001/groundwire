"""Support scorers: how strongly does a passage support a claim?

Three implementations behind one interface (strategy pattern):

* :class:`LexicalScorer`: dependency-free and sub-millisecond. Scores the
  fraction of the claim's content words found in the passage, and hard-fails
  any claim containing a number absent from the passage. Catches the most
  damaging hallucination in business settings (invented amounts, dates, IDs,
  quantities) but cannot detect paraphrased contradictions ("did" vs "did not").

* :class:`NLIScorer`: a cross-encoder natural-language-inference model
  (DeBERTa by default) returning P(entailment). Much stronger semantically;
  needs the ``nli`` extra and a few hundred MB of RAM. Runs on CPU.

* :class:`HybridScorer`: the numeric gate from the lexical scorer in front of
  any semantic scorer (normally NLI). Pairs whose numbers don't match are
  scored 0 *without* running the model, which is both more accurate (NLI
  models are known to be weak at exact numbers) and cheaper.

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


def numbers_supported(passage: str, claim: str) -> bool:
    """Return True if every number in ``claim`` also appears in ``passage``.

    Args:
        passage: Source passage.
        claim: Claim to check.

    Returns:
        True when the claim has no numbers, or all of them occur in the passage.
    """
    claim_numbers = numbers(claim)
    return not claim_numbers or claim_numbers <= numbers(passage)


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
        if not numbers_supported(passage, claim):
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
    """Cross-encoder NLI scorer returning P(entailment).

    Construct with :meth:`load` in production. The plain constructor takes an
    already-built tokenizer and model so tests can inject lightweight fakes.
    """

    name = "nli"

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        batch_size: int = 16,
        max_length: int = 512,
        model_name: str = "custom",
    ) -> None:
        """Wrap a loaded tokenizer and sequence-classification model.

        Args:
            tokenizer: Hugging Face tokenizer (or compatible fake).
            model: Sequence-classification model with ``config.id2label``.
            torch_module: The ``torch`` module (injected to keep it optional).
            batch_size: Pairs per forward pass.
            max_length: Token truncation limit per pair.
            model_name: Label used in logs and reports.

        Raises:
            ConfigurationError: If the model has no entailment label, or the
                batch settings are invalid.
        """
        if batch_size < 1 or max_length < 8:
            raise ConfigurationError("NLI batch_size must be >= 1 and max_length >= 8")
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._batch_size = batch_size
        self._max_length = max_length
        self.model_name = model_name
        self._entail_idx = self._find_entailment_index(model.config.id2label)

    @classmethod
    def load(
        cls,
        model_name: str,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        num_threads: int | None = None,
    ) -> NLIScorer:
        """Download (or read from cache) and load an NLI cross-encoder.

        Args:
            model_name: Hugging Face model id or local directory.
            batch_size: Pairs per forward pass.
            max_length: Token truncation limit per pair.
            num_threads: CPU threads for torch; ``None`` keeps torch's default.

        Returns:
            A ready scorer.

        Raises:
            ConfigurationError: If the ``nli`` extra is missing or the model
                cannot be loaded.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "the NLI scorer needs the 'nli' extra: pip install -e '.[nli]'"
            ) from exc

        if num_threads is not None:
            torch.set_num_threads(max(1, num_threads))
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot load NLI model '{model_name}': {exc}") from exc

        model.eval()
        scorer = cls(tokenizer=tokenizer, model=model, torch_module=torch,
                     batch_size=batch_size, max_length=max_length, model_name=model_name)
        logger.info("NLI scorer loaded", extra={"model": model_name,
                                                "entailment_index": scorer._entail_idx,
                                                "threads": torch.get_num_threads()})
        return scorer

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
            if "entail" in str(label).lower() and "not" not in str(label).lower():
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
                    # Truncate only the passage, so the claim is never cut off.
                    truncation="only_first", padding=True, max_length=self._max_length,
                    return_tensors="pt",
                )
                with self._torch.inference_mode():
                    logits = self._model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1)[:, self._entail_idx]
                scores.extend(float(p) for p in probs.tolist())
        except (RuntimeError, ValueError) as exc:  # torch raises RuntimeError on OOM etc.
            raise GuardrailUnavailableError(f"NLI scoring failed: {exc}") from exc
        return scores


# --------------------------------------------------------------------------- #
# Hybrid
# --------------------------------------------------------------------------- #


class HybridScorer(SupportScorer):
    """Numeric gate first, semantic scorer second.

    A pair whose claim cites a number missing from the passage scores 0.0 and
    is never sent to the semantic scorer. Remaining pairs get the semantic
    score unchanged.
    """

    name = "hybrid"

    def __init__(self, semantic: SupportScorer) -> None:
        """Create the scorer.

        Args:
            semantic: Scorer used for pairs that pass the numeric gate.
        """
        self._semantic = semantic

    def score_batch(self, pairs: Sequence[Pair]) -> list[float]:
        """Score many pairs. See :meth:`SupportScorer.score_batch`."""
        scores = [0.0] * len(pairs)
        passing = [i for i, (passage, claim) in enumerate(pairs)
                   if numbers_supported(passage, claim)]
        if passing:
            semantic_scores = self._semantic.score_batch([pairs[i] for i in passing])
            if len(semantic_scores) != len(passing):
                raise GuardrailUnavailableError("semantic scorer returned wrong number of scores")
            for i, score in zip(passing, semantic_scores, strict=True):
                scores[i] = score
        return scores
