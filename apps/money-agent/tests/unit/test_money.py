"""Money: exact paise arithmetic and the formats Indian statements use."""

from __future__ import annotations

from decimal import Decimal

import pytest

from money_agent.domain.money import InvalidAmountError, Money


@pytest.mark.parametrize(("text", "paise"), [
    ("1,23,456.78", 12345678),      # lakh grouping
    ("123,456.78", 12345678),       # western grouping
    ("₹ 500", 50000),
    ("Rs. 1,200", 120000),
    ("INR 99.5", 9950),
    ("(500.00)", -50000),           # accounting negative
    ("500.00 Dr", -50000),
    ("500.00 Cr", 50000),
    ("-42.10", -4210),
    ("+7", 700),
    ("0.005", 1),                   # half-up to the paisa
    ("1\u00a0000", 100000),        # no-break space inside a number
])
def test_parse_accepts_statement_formats(text: str, paise: int) -> None:
    assert Money.parse(text).paise == paise


@pytest.mark.parametrize("text", ["", "abc", "12.3.4", "1,2", "--", "₹", "1e5", "nan"])
def test_parse_rejects_non_amounts(text: str) -> None:
    with pytest.raises(InvalidAmountError):
        Money.parse(text)


def test_of_rejects_non_finite_values() -> None:
    with pytest.raises(InvalidAmountError):
        Money.of(Decimal("Infinity"))
    with pytest.raises(InvalidAmountError):
        Money.of("not a number")


@pytest.mark.parametrize(("paise", "text"), [
    (0, "₹0.00"), (5, "₹0.05"), (99900, "₹999.00"), (100000, "₹1,000.00"),
    (12345678, "₹1,23,456.78"), (1234567890, "₹1,23,45,678.90"), (-50000, "-₹500.00"),
])
def test_format_inr_uses_indian_grouping(paise: int, text: str) -> None:
    assert Money(paise).format_inr() == text


def test_arithmetic_is_exact() -> None:
    total = sum((Money.of("0.10") for _ in range(10)), Money.zero())
    assert total == Money.of("1.00")  # 0.1 * 10 is exactly 1 rupee, unlike floats
    assert Money.of(10) - Money.of(15) == Money.of(-5)
    assert abs(Money.of(-3)) == Money.of(3) and -Money.of(3) == Money.of(-3)
    assert Money.of(100).divided_by(3) == Money.of("33.33")
    assert Money.of(100).times(Decimal("0.075")) == Money.of("7.50")
    assert str(Money.of("12450.5")) == "12450.50"
    assert not Money.zero() and Money.of(1)


def test_ratio_and_division_guards() -> None:
    assert Money.of(50).ratio_to(Money.of(200)) == Decimal("0.25")
    assert Money.of(50).ratio_to(Money.zero()) is None
    with pytest.raises(ValueError, match="positive"):
        Money.of(1).divided_by(0)
