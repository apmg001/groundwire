"""PII detection, redaction and restoration.

Personal data in a prompt is replaced with stable placeholders *before* the
request reaches any model provider, and the placeholders in the model's answer
are swapped back *before* the client sees it::

    "Card 4111 1111 1111 1111, mail rahul@example.com"
        -> provider sees "Card [CARD_1], mail [EMAIL_1]"
        -> model answers "Refund sent to [EMAIL_1]."
        -> client sees  "Refund sent to rahul@example.com."

Design points:

* **Checksums cut false positives.** Card numbers must pass the Luhn check and
  Aadhaar numbers the Verhoeff check, so order IDs and amounts are left alone.
* **Consistent placeholders.** The same value always maps to the same
  placeholder within a request, so the model can still reason ("both emails
  are the same customer").
* **The vault never leaves the process.** Raw values are held only in memory
  for the life of one request and are never logged.

Limitations (documented, not hidden): regex detection covers structured
identifiers. It does **not** reliably find personal names, street addresses or
free-text secrets; plug a NER-based :class:`PIIDetector` in for those.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Entity types and matches
# --------------------------------------------------------------------------- #


class PIIType(StrEnum):
    """Supported PII categories."""

    EMAIL = "email"
    PHONE = "phone"
    CARD = "card"
    AADHAAR = "aadhaar"
    PAN = "pan"
    IP_ADDRESS = "ip_address"


@dataclass(frozen=True, slots=True)
class PIIMatch:
    """One detected PII span.

    Attributes:
        type: Category.
        start: Start offset in the text.
        end: End offset (exclusive).
        value: Matched text.
    """

    type: PIIType
    start: int
    end: int
    value: str


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def luhn_valid(digits: str) -> bool:
    """Luhn checksum used by payment card numbers.

    Args:
        digits: Digits only.

    Returns:
        True if the checksum is valid.
    """
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """Verhoeff checksum used by Aadhaar numbers.

    Args:
        digits: Digits only.

    Returns:
        True if the checksum is valid.
    """
    if not digits.isdigit():
        return False
    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][int(char)]]
    return check == 0


def _valid_ipv4(text: str) -> bool:
    parts = text.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 and str(int(p)) == p
                                   for p in parts)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


class PIIDetector(ABC):
    """Finds PII spans in text."""

    @abstractmethod
    def detect(self, text: str) -> list[PIIMatch]:
        """Find PII in ``text``.

        Args:
            text: Input text.

        Returns:
            Non-overlapping matches sorted by start offset.
        """


@dataclass(frozen=True, slots=True)
class _Rule:
    type: PIIType
    pattern: re.Pattern[str]
    priority: int  # higher wins on overlap


_RULES: tuple[_Rule, ...] = (
    _Rule(PIIType.EMAIL, re.compile(
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), 60),
    _Rule(PIIType.CARD, re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), 50),
    _Rule(PIIType.AADHAAR, re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"), 45),
    _Rule(PIIType.PAN, re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), 40),
    _Rule(PIIType.PHONE, re.compile(
        r"(?<![\w+])(?:"
        r"\+\d{1,3}[ -]?(?:\(\d{1,4}\)[ -]?)?\d{2,5}(?:[ -]?\d{2,5}){1,3}"  # international
        r"|(?:0)?[6-9]\d{4}[ -]?\d{5}"  # Indian mobile
        r"|\(\d{3}\)[ -]?\d{3}-\d{4}|\d{3}-\d{3}-\d{4}"  # North American
        r")(?!\d)"), 30),
    _Rule(PIIType.IP_ADDRESS, re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), 20),
)


def _validate(rule_type: PIIType, value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if rule_type is PIIType.CARD:
        return 13 <= len(digits) <= 19 and luhn_valid(digits)
    if rule_type is PIIType.AADHAAR:
        return len(digits) == 12 and verhoeff_valid(digits)
    if rule_type is PIIType.IP_ADDRESS:
        return _valid_ipv4(value)
    if rule_type is PIIType.PHONE:
        return 10 <= len(digits) <= 15
    return True


class RegexPIIDetector(PIIDetector):
    """Pattern + checksum detector for structured identifiers."""

    def __init__(self, types: Iterable[PIIType] | None = None) -> None:
        """Create the detector.

        Args:
            types: Categories to detect; all supported types if omitted.
        """
        enabled = set(types) if types is not None else set(PIIType)
        self._rules = [r for r in _RULES if r.type in enabled]

    def detect(self, text: str) -> list[PIIMatch]:
        """Find PII. See :meth:`PIIDetector.detect`."""
        candidates: list[tuple[int, PIIMatch]] = []
        for rule in self._rules:
            for match in rule.pattern.finditer(text):
                value = match.group(0)
                if _validate(rule.type, value):
                    candidates.append(
                        (rule.priority, PIIMatch(rule.type, match.start(), match.end(), value)))
        # Resolve overlaps: higher priority first, then longer span.
        candidates.sort(key=lambda c: (-c[0], -(c[1].end - c[1].start), c[1].start))
        chosen: list[PIIMatch] = []
        for _, candidate in candidates:
            if all(candidate.end <= m.start or candidate.start >= m.end for m in chosen):
                chosen.append(candidate)
        return sorted(chosen, key=lambda m: m.start)


# --------------------------------------------------------------------------- #
# Redaction and restoration
# --------------------------------------------------------------------------- #

PLACEHOLDER_PATTERN = re.compile(r"\[(" + "|".join(t.name for t in PIIType) + r")_(\d{1,4})\]")
MAX_PLACEHOLDER_LENGTH = max(len(t.name) for t in PIIType) + 7  # "[" + NAME + "_" + 4 digits + "]"


@dataclass(slots=True)
class PIIVault:
    """Per-request mapping between placeholders and original values.

    Attributes:
        placeholders: Placeholder -> original value.
    """

    placeholders: dict[str, str] = field(default_factory=dict)
    _by_value: dict[tuple[PIIType, str], str] = field(default_factory=dict, repr=False)
    _counters: Counter[PIIType] = field(default_factory=Counter, repr=False)

    def placeholder_for(self, match: PIIMatch) -> str:
        """Return the placeholder for a value, creating it on first sight.

        Args:
            match: Detected PII.

        Returns:
            A placeholder such as ``"[EMAIL_1]"``.
        """
        key = (match.type, match.value)
        existing = self._by_value.get(key)
        if existing is not None:
            return existing
        self._counters[match.type] += 1
        placeholder = f"[{match.type.name}_{self._counters[match.type]}]"
        self._by_value[key] = placeholder
        self.placeholders[placeholder] = match.value
        return placeholder

    def counts(self) -> dict[str, int]:
        """Number of distinct values redacted per type (safe to log)."""
        return {t.value: n for t, n in self._counters.items()}

    def __bool__(self) -> bool:
        """True if anything was redacted."""
        return bool(self.placeholders)

    def __repr__(self) -> str:
        """Never print raw values."""
        return f"PIIVault(counts={self.counts()})"


class PIIRedactor:
    """Replaces PII with placeholders and restores them afterwards."""

    def __init__(self, detector: PIIDetector) -> None:
        """Create the redactor.

        Args:
            detector: PII detector.
        """
        self._detector = detector

    def redact(self, text: str, vault: PIIVault) -> str:
        """Replace every detected PII span with its placeholder.

        Args:
            text: Input text.
            vault: Request vault (updated in place).

        Returns:
            Redacted text.
        """
        matches = self._detector.detect(text)
        if not matches:
            return text
        pieces: list[str] = []
        cursor = 0
        for match in matches:
            pieces.append(text[cursor:match.start])
            pieces.append(vault.placeholder_for(match))
            cursor = match.end
        pieces.append(text[cursor:])
        return "".join(pieces)

    @staticmethod
    def restore(text: str, vault: PIIVault) -> str:
        """Swap known placeholders back to their original values.

        Unknown placeholder-looking strings are left untouched.

        Args:
            text: Model output.
            vault: Request vault.

        Returns:
            Restored text.
        """
        if not vault:
            return text
        return PLACEHOLDER_PATTERN.sub(
            lambda m: vault.placeholders.get(m.group(0), m.group(0)), text)


class StreamRestorer:
    """Restores placeholders in a token stream where one may span chunks.

    Text is released as soon as it cannot be the start of a placeholder;
    anything after an unmatched ``"["`` is held back (at most
    :data:`MAX_PLACEHOLDER_LENGTH` characters) until it resolves.
    """

    def __init__(self, vault: PIIVault) -> None:
        """Create the restorer.

        Args:
            vault: Request vault.
        """
        self._vault = vault
        self._buffer = ""

    def feed(self, text: str) -> str:
        """Accept a chunk and return the text that is safe to emit now.

        Args:
            text: Next chunk of model output.

        Returns:
            Restored text (possibly empty while a placeholder is pending).
        """
        if not self._vault:
            return text
        self._buffer += text
        cut = self._buffer.rfind("[")
        if cut == -1 or "]" in self._buffer[cut:] or \
                len(self._buffer) - cut > MAX_PLACEHOLDER_LENGTH:
            ready, self._buffer = self._buffer, ""
        else:
            ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return PIIRedactor.restore(ready, self._vault)

    def flush(self) -> str:
        """Return whatever is still buffered (end of stream).

        Returns:
            Restored remaining text.
        """
        ready, self._buffer = self._buffer, ""
        return PIIRedactor.restore(ready, self._vault)


def redact_all(redactor: PIIRedactor, texts: Sequence[str], vault: PIIVault) -> list[str]:
    """Redact several texts with one shared vault.

    Args:
        redactor: Redactor.
        texts: Texts to redact.
        vault: Shared vault.

    Returns:
        Redacted texts, same order.
    """
    return [redactor.redact(text, vault) for text in texts]
