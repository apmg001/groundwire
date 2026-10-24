"""Text utilities for grounding: claim splitting, tokenisation, number extraction.

Deliberately dependency-free (regex only) so the default grounding path adds
no model downloads and negligible latency.
"""

from __future__ import annotations

import re

# Split after ., ! or ? followed by whitespace and an uppercase letter/digit/quote,
# or on newlines. Avoids splitting "e.g. this" and "3.5 million".
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
# Clock times: "9 am", "11:30 PM", "5p.m." (12-hour) or "16:30", "21:0" (24-hour).
_TIME = re.compile(
    r"(?<![\w.:])(\d{1,2})(?::(\d{1,2}))?\s*([ap])\.?\s*m\b\.?"
    r"|(?<![\w.:])(\d{1,2}):(\d{1,2})(?!\d)",
    re.IGNORECASE,
)

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "there", "here", "which", "who",
    "whom", "whose", "what", "when", "where", "why", "how", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their", "mine",
    "yours", "do", "does", "did", "done", "doing", "have", "has", "had", "having", "can",
    "could", "should", "would", "may", "might", "must", "shall", "will", "just", "also",
    "very", "so", "such", "than", "too", "into", "over", "under", "about", "above", "below",
    "up", "down", "out", "off", "again", "further", "once", "only", "own", "same", "other",
    "some", "any", "each", "few", "more", "most", "all", "both", "no", "nor", "not", "s",
    "t", "don", "now", "sure", "yes", "okay", "ok", "well",
})


def split_claims(text: str) -> list[str]:
    """Split an answer into candidate claims (sentences / bullet items).

    Args:
        text: Model answer.

    Returns:
        Non-empty, stripped sentences with list markers removed.
    """
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    claims: list[str] = []
    for part in parts:
        cleaned = _BULLET_PREFIX.sub("", part).strip()
        if cleaned:
            claims.append(cleaned)
    return claims


def _stem(token: str) -> str:
    """Very light stemming so "invoices" matches "invoice"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Return lower-cased, lightly stemmed content words (no stopwords).

    Args:
        text: Input text.

    Returns:
        Set of content tokens.
    """
    return {
        _stem(tok) for tok in _WORD.findall(text.lower())
        if tok not in STOPWORDS and (len(tok) > 2 or tok.isdigit())
    }


NUMBER_TERM_PREFIX = "#"


def retrieval_terms(text: str) -> list[str]:
    """Terms for keyword retrieval (BM25): content words plus whole numbers.

    Unlike :func:`content_tokens` this keeps repeats (term frequency matters
    to BM25) and adds every normalised number or clock time as one term
    prefixed with :data:`NUMBER_TERM_PREFIX` ("1,200" -> ``"#1200"``), so a
    claim's figures can be matched exactly and weighted on their own.

    Args:
        text: Input text.

    Returns:
        Terms in order of appearance (with repeats).
    """
    words = [_stem(tok) for tok in _WORD.findall(text.lower())
             if tok not in STOPWORDS and len(tok) > 2 and not tok.isdigit()]
    return words + [NUMBER_TERM_PREFIX + n for n in sorted(numbers(text))]


def _time_token(match: re.Match[str]) -> str | None:
    """Convert a clock-time match to a canonical ``"tHH:MM"`` token.

    Args:
        match: A :data:`_TIME` match.

    Returns:
        The token, or ``None`` if the values are not a valid time.
    """
    if match.group(3):  # 12-hour form
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if match.group(3).lower() == "p" else 0)
    else:  # 24-hour form
        hour, minute = int(match.group(4)), int(match.group(5))
        if hour > 24:
            return None
        hour %= 24
    if minute > 59:
        return None
    return f"t{hour:02d}:{minute:02d}"


def numbers(text: str) -> set[str]:
    """Extract normalised numbers and clock times.

    Plain numbers: ``"1,200.00" -> "1200"``, ``"15%" -> "15%"``.
    Clock times in any common format map to one 24-hour token, so
    ``"5 pm"``, ``"5:00 PM"`` and ``"17:0"`` all become ``"t17:00"``.
    Without this, an answer that correctly says "open 11 am to 5 pm" would
    look invented against source data that stores ``"11:0-17:0"``.

    Args:
        text: Input text.

    Returns:
        Set of normalised number and time strings.
    """
    found: set[str] = set()

    def replace_time(match: re.Match[str]) -> str:
        token = _time_token(match)
        if token is None:
            return match.group(0)
        found.add(token)
        return " "

    remainder = _TIME.sub(replace_time, text)
    for match in _NUMBER.findall(remainder):
        raw = match.replace(",", "").lstrip("+")
        percent = raw.endswith("%")
        raw = raw.rstrip("%")
        if "." in raw:
            raw = raw.rstrip("0").rstrip(".")
        found.add(raw + ("%" if percent else ""))
    return found


def chunk_passages(context: list[str], *, max_chars: int = 600) -> list[str]:
    """Split context documents into passages small enough for an NLI premise.

    Sentences are packed greedily up to ``max_chars``; a single oversized
    sentence becomes its own passage (the scorer truncates it).

    Args:
        context: Source documents.
        max_chars: Soft passage length limit.

    Returns:
        Passages, in document order.
    """
    passages: list[str] = []
    for doc in context:
        current = ""
        for sentence in split_claims(doc):
            if current and len(current) + 1 + len(sentence) > max_chars:
                passages.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            passages.append(current)
    return passages
