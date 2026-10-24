"""Turn RAGTruth examples into long-document examples ("needle in a haystack").

RAGTruth contexts are short (1-3k characters), but real clients send long
documents. To measure verification on long inputs without inventing labels,
each example's real source (the *needle*, which its human hallucination label
refers to) is hidden among *distractor* documents taken from other RAGTruth
sources until the context reaches ``target_chars``. The label stays valid:
the answer was written from the needle, and distractors come from unrelated
sources, so they can neither support a hallucinated claim on purpose nor
change what the answer was based on.

Where the needle sits matters (the old verifier only looked at the first ~150k
characters; a token detector only at its first window), so positions cycle
through start, middle and end.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from groundwire.core.errors import DatasetError
from groundwire.evals.datasets import EvalExample


class NeedlePosition(StrEnum):
    """Where the real source is placed in the long context."""

    START = "start"
    MIDDLE = "middle"
    END = "end"


@dataclass(frozen=True, slots=True)
class LongExample:
    """An example whose context is the needle hidden among distractors.

    Attributes:
        example: The original RAGTruth example (label, answer, short context).
        context: Long context: distractor documents with the needle inserted.
        needle_documents: Indices in ``context`` of the original documents.
        position: Where the needle was placed.
    """

    example: EvalExample
    context: tuple[str, ...]
    needle_documents: frozenset[int]
    position: NeedlePosition

    @property
    def chars(self) -> int:
        """Total context length in characters."""
        return sum(len(doc) for doc in self.context)


def build_haystack(examples: Sequence[EvalExample], pool: Sequence[EvalExample], *,
                   target_chars: int, seed: int = 13) -> list[LongExample]:
    """Build one long-context example per example.

    Args:
        examples: Examples to extend.
        pool: Examples whose documents may serve as distractors (an example's
            own source is never used as its distractor).
        target_chars: Minimum total context length.
        seed: Random seed (same seed, same haystacks).

    Returns:
        Long examples, in input order; positions cycle start, middle, end.

    Raises:
        DatasetError: If the pool cannot supply enough distractor text.
    """
    if target_chars < 1:
        raise DatasetError("target_chars must be positive")
    rng = random.Random(seed)  # noqa: S311 - benchmark sampling, not security
    documents = [(e.meta.get("source_id"), doc) for e in pool for doc in e.context if doc.strip()]
    unique = list(dict.fromkeys(documents))
    positions = list(NeedlePosition)
    built: list[LongExample] = []
    for number, example in enumerate(examples):
        source = example.meta.get("source_id")
        candidates = [doc for sid, doc in unique if sid != source]
        rng.shuffle(candidates)
        needle = list(example.context)
        distractors: list[str] = []
        length = sum(len(doc) for doc in needle)
        for doc in candidates:
            if length >= target_chars:
                break
            distractors.append(doc)
            length += len(doc)
        if length < target_chars:
            raise DatasetError(f"distractor pool too small for {target_chars} characters")
        position = positions[number % len(positions)]
        at = {NeedlePosition.START: 0, NeedlePosition.MIDDLE: len(distractors) // 2,
              NeedlePosition.END: len(distractors)}[position]
        context = distractors[:at] + needle + distractors[at:]
        built.append(LongExample(example, tuple(context),
                                 frozenset(range(at, at + len(needle))), position))
    return built
