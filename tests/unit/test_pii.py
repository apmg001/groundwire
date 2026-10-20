"""Tests for PII detection, redaction and restoration."""

from __future__ import annotations

from itertools import pairwise

import pytest

from groundwire.safety.pii import (
    MAX_PLACEHOLDER_LENGTH,
    PIIRedactor,
    PIIType,
    PIIVault,
    RegexPIIDetector,
    StreamRestorer,
    luhn_valid,
    redact_all,
    verhoeff_valid,
)


def _valid_aadhaar(prefix: str = "23456789012") -> str:
    """Append the Verhoeff check digit to an 11-digit prefix."""
    for digit in "0123456789":
        if verhoeff_valid(prefix + digit):
            return prefix + digit
    raise AssertionError("no check digit found")


AADHAAR = _valid_aadhaar()
DETECTOR = RegexPIIDetector()


def _types(text: str) -> list[tuple[PIIType, str]]:
    return [(m.type, m.value) for m in DETECTOR.detect(text)]


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def test_luhn() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("41a1")


def test_verhoeff() -> None:
    assert verhoeff_valid(AADHAAR)
    wrong = AADHAAR[:-1] + str((int(AADHAAR[-1]) + 1) % 10)
    assert not verhoeff_valid(wrong)
    assert not verhoeff_valid("12ab")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("text", "expected"), [
    ("mail rahul.sharma+test@example.co.in now",
     [(PIIType.EMAIL, "rahul.sharma+test@example.co.in")]),
    ("card 4111 1111 1111 1111.", [(PIIType.CARD, "4111 1111 1111 1111")]),
    ("card 4111-1111-1111-1111", [(PIIType.CARD, "4111-1111-1111-1111")]),
    (f"aadhaar {AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}",
     [(PIIType.AADHAAR, f"{AADHAAR[:4]} {AADHAAR[4:8]} {AADHAAR[8:]}")]),
    ("PAN ABCDE1234F filed", [(PIIType.PAN, "ABCDE1234F")]),
    ("call 98765 43210", [(PIIType.PHONE, "98765 43210")]),
    ("call +91 98765 43210", [(PIIType.PHONE, "+91 98765 43210")]),
    ("call (510) 889-8690", [(PIIType.PHONE, "(510) 889-8690")]),
    ("call +44 20 7946 0958", [(PIIType.PHONE, "+44 20 7946 0958")]),
    ("from 192.168.1.20 at", [(PIIType.IP_ADDRESS, "192.168.1.20")]),
])
def test_detects_each_type(text: str, expected: list[tuple[PIIType, str]]) -> None:
    assert _types(text) == expected


@pytest.mark.parametrize("text", [
    "Order 7781 totals 1,200.00 USD, due 2026-10-20.",
    "Invoice INV-1042 for 49.99 USD.",
    "card 4111 1111 1111 1112",  # fails Luhn
    "id 1234 5678 9012",  # 12 digits but fails Verhoeff
    "version 999.300.1.2",  # not a valid IPv4
    "abcde1234f lower-case is not a PAN",
    "The meeting is at 10:30 in room 204.",
])
def test_ignores_non_pii(text: str) -> None:
    assert _types(text) == []


def test_types_can_be_restricted() -> None:
    detector = RegexPIIDetector([PIIType.EMAIL])
    assert [m.type for m in detector.detect("a@b.com 98765 43210")] == [PIIType.EMAIL]


def test_overlaps_resolve_to_one_match() -> None:
    matches = DETECTOR.detect("ping +91 98765 43210 or a@b.io")
    assert [m.type for m in matches] == [PIIType.PHONE, PIIType.EMAIL]
    assert all(a.end <= b.start for a, b in pairwise(matches))


# --------------------------------------------------------------------------- #
# Redaction and restoration
# --------------------------------------------------------------------------- #


def test_redact_and_restore_round_trip() -> None:
    redactor = PIIRedactor(DETECTOR)
    vault = PIIVault()
    text = "Email a@b.com and again a@b.com, card 4111 1111 1111 1111, other c@d.org"
    redacted = redactor.redact(text, vault)
    assert redacted == ("Email [EMAIL_1] and again [EMAIL_1], card [CARD_1], "
                        "other [EMAIL_2]")
    assert vault.counts() == {"email": 2, "card": 1}
    assert redactor.restore("Sent to [EMAIL_1] and [EMAIL_2].", vault) == \
        "Sent to a@b.com and c@d.org."


def test_restore_leaves_unknown_placeholders_and_empty_vault_alone() -> None:
    vault = PIIVault()
    assert PIIRedactor.restore("[EMAIL_9] stays", vault) == "[EMAIL_9] stays"
    PIIRedactor(DETECTOR).redact("a@b.com", vault)
    assert PIIRedactor.restore("[EMAIL_9] and [EMAIL_1]", vault) == "[EMAIL_9] and a@b.com"


def test_text_without_pii_is_unchanged() -> None:
    vault = PIIVault()
    assert PIIRedactor(DETECTOR).redact("nothing here", vault) == "nothing here"
    assert not vault


def test_vault_repr_never_shows_raw_values() -> None:
    vault = PIIVault()
    PIIRedactor(DETECTOR).redact("secret@corp.com", vault)
    assert "secret@corp.com" not in repr(vault)


def test_redact_all_shares_one_vault() -> None:
    vault = PIIVault()
    out = redact_all(PIIRedactor(DETECTOR), ["a@b.com", "again a@b.com"], vault)
    assert out == ["[EMAIL_1]", "again [EMAIL_1]"]


# --------------------------------------------------------------------------- #
# Streaming restoration
# --------------------------------------------------------------------------- #


def _vault_with_email() -> PIIVault:
    vault = PIIVault()
    PIIRedactor(DETECTOR).redact("rahul@example.com", vault)
    return vault


@pytest.mark.parametrize("chunks", [
    ["Sent to [EMAIL_1] today."],
    ["Sent to [EM", "AIL_1] today."],
    ["Sent to [", "EMAIL", "_1", "]", " today."],
    ["Sent to [EMAIL_1", "] today."],
])
def test_stream_restorer_handles_placeholders_split_across_chunks(chunks: list[str]) -> None:
    restorer = StreamRestorer(_vault_with_email())
    out = "".join(restorer.feed(c) for c in chunks) + restorer.flush()
    assert out == "Sent to rahul@example.com today."


def test_stream_restorer_does_not_hold_ordinary_brackets_forever() -> None:
    restorer = StreamRestorer(_vault_with_email())
    first = restorer.feed("see [note")
    assert first == "see "  # might be a placeholder, so held back
    released = restorer.feed(" " + "x" * MAX_PLACEHOLDER_LENGTH)
    assert released.startswith("[note")  # too long to be a placeholder: released


def test_stream_restorer_passthrough_without_vault() -> None:
    restorer = StreamRestorer(PIIVault())
    assert restorer.feed("[EM") == "[EM"
    assert restorer.flush() == ""
