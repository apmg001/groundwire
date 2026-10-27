"""Money as whole paise.

Floating point cannot represent most rupee amounts exactly (``0.1 + 0.2 !=
0.3``), and errors add up across thousands of transactions. Every amount in
this application is therefore an integer number of paise, wrapped in
:class:`Money` so it cannot be mixed up with a count or a percentage.

Parsing accepts what Indian bank exports actually contain: ``₹1,23,456.78``
(lakh grouping), ``Rs. 500``, ``INR 500.00``, ``(500.00)`` for negatives,
and ``500.00 Dr`` / ``500.00 Cr`` suffixes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

PAISE_PER_RUPEE = 100
_CURRENCY = re.compile(r"(?i)(₹|\binr\b|\brs\.?)")
_SUFFIX = re.compile(r"(?i)\s*\b(dr|cr)\.?\s*$")
_NUMBER = re.compile(r"^[+-]?(\d{1,3}(,\d{2,3})*|\d+)(\.\d+)?$")


class InvalidAmountError(ValueError):
    """Text is not a recognisable amount of money."""


@dataclass(frozen=True, slots=True, order=True)
class Money:
    """An amount of Indian rupees, held as whole paise.

    Attributes:
        paise: The amount in paise (may be negative for net figures).
    """

    paise: int

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def zero(cls) -> Money:
        """₹0."""
        return cls(0)

    @classmethod
    def of(cls, rupees: Decimal | int | str) -> Money:
        """From a rupee value, rounded half-up to the paisa.

        Args:
            rupees: Rupees (``Decimal("12.345")`` becomes 1235 paise).

        Returns:
            The amount.

        Raises:
            InvalidAmountError: If ``rupees`` is not a finite number.
        """
        try:
            value = Decimal(rupees) if not isinstance(rupees, Decimal) else rupees
        except InvalidOperation as exc:
            raise InvalidAmountError(f"not a number: {rupees!r}") from exc
        if not value.is_finite():
            raise InvalidAmountError(f"not a finite number: {rupees!r}")
        paise = (value * PAISE_PER_RUPEE).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        return cls(int(paise))

    @classmethod
    def parse(cls, text: str) -> Money:
        """Parse an amount as written in a bank statement.

        ``"(500.00)"`` and ``"500.00 Dr"`` are negative; ``"500 Cr"`` is positive.

        Args:
            text: The text.

        Returns:
            The (signed) amount.

        Raises:
            InvalidAmountError: If the text is empty or not an amount.
        """
        raw = text.strip()
        negative = False
        suffix = _SUFFIX.search(raw)
        if suffix:
            negative = suffix.group(1).lower() == "dr"
            raw = raw[:suffix.start()]
        raw = _CURRENCY.sub("", raw).replace(" ", "").replace("\u00a0", "")
        if raw.startswith("(") and raw.endswith(")"):
            negative, raw = True, raw[1:-1]
        if raw.startswith("-"):
            negative, raw = not negative, raw[1:]
        elif raw.startswith("+"):
            raw = raw[1:]
        if not raw or not _NUMBER.match(raw):
            raise InvalidAmountError(f"not an amount: {text!r}")
        amount = cls.of(raw.replace(",", ""))
        return -amount if negative else amount

    # ------------------------------------------------------------------ #
    # Arithmetic
    # ------------------------------------------------------------------ #

    def __add__(self, other: Money) -> Money:
        return Money(self.paise + other.paise)

    def __sub__(self, other: Money) -> Money:
        return Money(self.paise - other.paise)

    def __neg__(self) -> Money:
        return Money(-self.paise)

    def __abs__(self) -> Money:
        return Money(abs(self.paise))

    def __bool__(self) -> bool:
        return self.paise != 0

    def times(self, factor: Decimal | int) -> Money:
        """Multiply, rounding half-up to the paisa.

        Args:
            factor: Multiplier.

        Returns:
            The product.
        """
        return Money.of(self.rupees * Decimal(factor))

    def divided_by(self, divisor: int) -> Money:
        """Divide into ``divisor`` parts, rounding half-up to the paisa.

        Args:
            divisor: A positive whole number.

        Returns:
            One part.

        Raises:
            ValueError: If ``divisor`` is not positive.
        """
        if divisor <= 0:
            raise ValueError("divisor must be positive")
        return Money.of(self.rupees / divisor)

    def ratio_to(self, other: Money) -> Decimal | None:
        """``self / other`` as a decimal, or ``None`` when ``other`` is zero."""
        return None if not other.paise else Decimal(self.paise) / Decimal(other.paise)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    @property
    def rupees(self) -> Decimal:
        """The amount in rupees, with exactly two decimal places."""
        return (Decimal(self.paise) / PAISE_PER_RUPEE).quantize(Decimal("0.01"))

    def __str__(self) -> str:
        """Plain rupees with two decimals (``"12450.50"``), for machine-readable output."""
        return str(self.rupees)

    def format_inr(self) -> str:
        """Human format with Indian digit grouping: ``₹1,23,456.78``."""
        rupees, paise = divmod(abs(self.paise), PAISE_PER_RUPEE)
        digits = str(rupees)
        if len(digits) > 3:
            head, tail = digits[:-3], digits[-3:]
            groups = [head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1]
            digits = ",".join([*groups, tail])
        sign = "-" if self.paise < 0 else ""
        return f"{sign}₹{digits}.{paise:02d}"
