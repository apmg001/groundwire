"""Guards that veto semantic cache matches embeddings get wrong.

Sentence embeddings capture what a question is *about*, not its exact values.
Measured on a laptop with ``all-MiniLM-L6-v2``:

* "What is the total of invoice INV-1043?" vs the cached "...INV-1042?" scored
  **0.964**, *higher* than a genuine paraphrase of the same question (0.921).

A similarity threshold alone would serve one customer's invoice to another.
These guards compare the facts that must match exactly. The same function is
used by the live cache and by the benchmark, so measured numbers describe
the code that actually runs.
"""

from __future__ import annotations

import re

from groundwire.guardrails.text import numbers

_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")

NUMBERS_DIFFER = "numbers_differ"
PLACEHOLDERS_DIFFER = "placeholders_differ"


def semantic_mismatch(query: str, cached_query: str) -> str | None:
    """Return why two similar questions must not share an answer, if they must not.

    Args:
        query: Incoming question (normalised or raw).
        cached_query: Question of the cached entry.

    Returns:
        ``"numbers_differ"``, ``"placeholders_differ"``, or ``None`` when the
        match is acceptable.
    """
    if numbers(query) != numbers(cached_query):
        return NUMBERS_DIFFER
    if set(_PLACEHOLDER.findall(query.upper())) != set(_PLACEHOLDER.findall(cached_query.upper())):
        return PLACEHOLDERS_DIFFER
    return None
