"""Token-level hallucination detection with a fine-tuned token classifier.

Unlike the pairwise scorers in :mod:`groundwire.guardrails.scorers` (which
judge one claim against one passage), a token detector reads the *whole*
prompt (instructions + question + every passage) together with the answer,
and returns, for each answer token, the probability that it is unsupported.
Seeing all the evidence at once is what pairwise scorers lack.

The default model is **LettuceDetect** (KR Labs, MIT licence): ModernBERT
fine-tuned on the RAGTruth *train* split for exactly this task, with a 4K
token context. Inputs follow its training format: ``tokenizer(prompt, answer)``
with only the prompt truncated, and label 1 = hallucinated.

Prompt templates below are reproduced from LettuceDetect
(https://github.com/KRLabsOrg/LettuceDetect, MIT licence) so live gateway
traffic is formatted the way the model was trained.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from groundwire.core.errors import ConfigurationError, GuardrailUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"  # noqa: S105 - model id
HALLUCINATED_LABEL = 1

# --------------------------------------------------------------------------- #
# Prompt formatting (LettuceDetect training format)
# --------------------------------------------------------------------------- #

_QA_TEMPLATE = (
    "Briefly answer the following question:\n{question}\n"
    "Bear in mind that your response should be strictly based on the following "
    "{num_passages} passages:\n{context}\n"
    "In case the passages do not contain the necessary information to answer the "
    'question, please reply with: "Unable to answer based on given passages."\noutput:'
)
_SUMMARY_TEMPLATE = "Summarize the following text:\n{text}\noutput:"


def build_prompt(context: Sequence[str], question: str | None) -> str:
    """Format context (and optional question) the way the detector was trained.

    Args:
        context: Source passages.
        question: The user's question; ``None`` for summarisation-style tasks.

    Returns:
        The prompt string.
    """
    block = "\n".join(f"passage {i}: {passage}" for i, passage in enumerate(context, start=1))
    if question is None or not question.strip():
        return _SUMMARY_TEMPLATE.format(text=block)
    return _QA_TEMPLATE.format(question=question.strip(), num_passages=len(context),
                               context=block)


# --------------------------------------------------------------------------- #
# Detector interface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TokenProb:
    """Hallucination probability for one answer token.

    Attributes:
        start: Start character offset in the answer.
        end: End character offset in the answer (exclusive).
        prob: Probability the token is unsupported, in [0, 1].
    """

    start: int
    end: int
    prob: float


class TokenDetector(ABC):
    """Returns per-token hallucination probabilities for an answer."""

    name: str = "token"

    @abstractmethod
    def detect(self, prompt: str, answer: str) -> list[TokenProb]:
        """Score every answer token.

        Args:
            prompt: Full prompt the answer was generated from.
            answer: Answer to check.

        Returns:
            One entry per non-empty answer token, in order.

        Raises:
            GuardrailUnavailableError: If the model cannot run.
        """


def max_prob_in_range(tokens: Sequence[TokenProb], start: int, end: int) -> float:
    """Highest hallucination probability among tokens overlapping ``[start, end)``.

    Args:
        tokens: Token probabilities.
        start: Range start (inclusive).
        end: Range end (exclusive).

    Returns:
        The maximum, or 0.0 if no token overlaps the range.
    """
    return max((t.prob for t in tokens if t.start < end and t.end > start), default=0.0)


# --------------------------------------------------------------------------- #
# Hugging Face implementation
# --------------------------------------------------------------------------- #


class TransformersTokenDetector(TokenDetector):
    """Token classifier loaded with ``transformers`` (e.g. LettuceDetect).

    Construct with :meth:`load` in production; the constructor accepts
    pre-built objects so tests can inject fakes.
    """

    name = "token"

    def __init__(self, *, tokenizer: Any, model: Any, torch_module: Any,
                 max_length: int = 4096, model_name: str = "custom") -> None:
        """Wrap a loaded tokenizer and token-classification model.

        Args:
            tokenizer: Fast tokenizer supporting ``return_offsets_mapping``.
            model: Token-classification model (2 labels, 1 = hallucinated).
            torch_module: The ``torch`` module.
            max_length: Maximum tokens per (prompt, answer) input.
            model_name: Label for logs and reports.

        Raises:
            ConfigurationError: If ``max_length`` is too small.
        """
        if max_length < 32:
            raise ConfigurationError("token detector max_length must be >= 32")
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._max_length = max_length
        self.model_name = model_name

    @classmethod
    def load(cls, model_name: str = DEFAULT_TOKEN_MODEL, *, max_length: int = 4096,
             num_threads: int | None = None) -> TransformersTokenDetector:
        """Download (or read from cache) and load a token classifier.

        Args:
            model_name: Hugging Face model id or local directory.
            max_length: Maximum tokens per input.
            num_threads: CPU threads for torch; ``None`` keeps torch's default.

        Returns:
            A ready detector.

        Raises:
            ConfigurationError: If the ``nli`` extra is missing or the model
                cannot be loaded.
        """
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "the token detector needs the 'nli' extra: make install-nli"
            ) from exc
        if num_threads is not None:
            torch.set_num_threads(max(1, num_threads))
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot load token model '{model_name}': {exc}") from exc
        if not getattr(tokenizer, "is_fast", False):
            raise ConfigurationError(f"'{model_name}' needs a fast tokenizer (offset mapping)")
        model.eval()
        logger.info("token detector loaded", extra={"model": model_name,
                                                    "threads": torch.get_num_threads()})
        return cls(tokenizer=tokenizer, model=model, torch_module=torch,
                   max_length=max_length, model_name=model_name)

    def detect(self, prompt: str, answer: str) -> list[TokenProb]:
        """Score answer tokens. See :meth:`TokenDetector.detect`."""
        if not answer.strip():
            return []
        try:
            encoding = self._tokenizer(
                prompt, answer,
                truncation="only_first",  # never cut the answer
                max_length=self._max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            sequence_ids = encoding.sequence_ids(0)
            offsets = encoding.pop("offset_mapping")[0].tolist()
            if len(sequence_ids) >= self._max_length:
                logger.warning("token detector input truncated to %d tokens; "
                               "some context was not seen", self._max_length)
            inputs = {k: v for k, v in encoding.items() if k in ("input_ids", "attention_mask")}
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits
            probs = self._torch.softmax(logits, dim=-1)[0, :, HALLUCINATED_LABEL].tolist()
        except (RuntimeError, ValueError, KeyError, IndexError) as exc:
            raise GuardrailUnavailableError(f"token detector failed: {exc}") from exc

        tokens: list[TokenProb] = []
        for seq_id, (start, end), prob in zip(sequence_ids, offsets, probs, strict=True):
            if seq_id == 1 and end > start:
                tokens.append(TokenProb(int(start), int(end), float(prob)))
        return tokens
