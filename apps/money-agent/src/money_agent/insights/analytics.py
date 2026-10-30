"""Money maths: summaries, trends, recurring payments and goal plans.

The agent never does arithmetic itself. Language models are unreliable at
sums over hundreds of rows, so every figure the user sees is computed here,
exactly (integer paise), returned by a tool, and then checked by Groundwire's
answer check against what the tool returned.

Definitions (stated once, used everywhere, shown to users on request):

* **income**: money in, except refunds;
* **spending**: money out, except investments (moving money to an investment
  is saving, not spending);
* **invested**: money out to investments (SIPs, brokers, NPS, PPF);
* **saved**: ``income + refunds - spending`` (includes what was invested);
* **savings rate**: ``saved / income``, as a percentage (none without income).

All functions are pure: transactions in, results out.
"""

from __future__ import annotations

import calendar
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from money_agent.domain.models import Category, Direction, Transaction
from money_agent.domain.money import Money

PERCENT = Decimal("0.1")
# Words in narrations that say how money moved, not who received it.
_CHANNEL_WORDS = frozenset({
    "upi", "pos", "neft", "imps", "rtgs", "ach", "nach", "ecs", "dr", "cr", "txn", "ref",
    "to", "by", "from", "payment", "paid", "pay", "bil", "onl", "inb", "mb", "ib", "debit",
    "credit", "card", "purchase", "autopay", "si", "mandate", "via", "the", "at", "on",
    "for", "and", "pvt", "ltd", "india", "in", "com", "www", "co", "bill", "p", "s",
})
_NON_WORD = re.compile(r"[^a-z]+")
_UPI_HANDLE = re.compile(r"@[a-z0-9.\-]+")  # "netflix@hdfc": the bank is not the payee


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Period:
    """An inclusive date range.

    Attributes:
        start: First day.
        end: Last day.
    """

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("period start must not be after its end")

    @classmethod
    def month(cls, year: int, month: int) -> Period:
        """A calendar month.

        Raises:
            ValueError: If the month is invalid.
        """
        last = calendar.monthrange(year, month)[1]
        return cls(date(year, month, 1), date(year, month, last))

    @classmethod
    def last_full_months(cls, today: date, count: int) -> Period:
        """The ``count`` complete calendar months before ``today``'s month.

        Raises:
            ValueError: If ``count`` is not positive.
        """
        if count < 1:
            raise ValueError("count must be positive")
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        for _ in range(count - 1):
            start = (start - timedelta(days=1)).replace(day=1)
        return cls(start, end)

    def contains(self, day: date) -> bool:
        """True if ``day`` is in the range."""
        return self.start <= day <= self.end

    @property
    def label(self) -> str:
        """``2026-09`` for a whole month, else ``2026-09-01..2026-10-15``."""
        is_month = (self.start.day == 1 and self.start.year == self.end.year
                    and self.start.month == self.end.month
                    and self.end.day == calendar.monthrange(self.end.year, self.end.month)[1])
        return (f"{self.start:%Y-%m}" if is_month
                else f"{self.start.isoformat()}..{self.end.isoformat()}")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CategoryTotal:
    """Spending (or income) in one category.

    Attributes:
        category: The category.
        total: Sum of its transactions.
        count: Number of transactions.
        share_percent: Share of the period's spending (or income), 0.1% precision.
    """

    category: Category
    total: Money
    count: int
    share_percent: Decimal


@dataclass(frozen=True, slots=True)
class Totals:
    """The headline figures for a set of transactions (see module definitions)."""

    income: Money
    refunds: Money
    spending: Money
    invested: Money

    @property
    def saved(self) -> Money:
        """Income plus refunds, minus spending."""
        return self.income + self.refunds - self.spending

    @property
    def savings_rate_percent(self) -> Decimal | None:
        """Saved as a share of income, or ``None`` without income."""
        ratio = self.saved.ratio_to(self.income)
        return None if ratio is None else _percent(ratio)


@dataclass(frozen=True, slots=True)
class PeriodSummary:
    """Everything about one period.

    Attributes:
        period: The period.
        totals: Headline figures.
        transactions: Number of transactions.
        spending_by_category: Spending categories, largest first.
        income_by_category: Income categories, largest first.
    """

    period: Period
    totals: Totals
    transactions: int
    spending_by_category: tuple[CategoryTotal, ...]
    income_by_category: tuple[CategoryTotal, ...]


@dataclass(frozen=True, slots=True)
class MonthTotals:
    """Headline figures for one calendar month.

    Attributes:
        month: ``YYYY-MM``.
        totals: Its figures.
    """

    month: str
    totals: Totals


@dataclass(frozen=True, slots=True)
class RecurringPayment:
    """A payment that repeats about monthly (rent, SIPs, subscriptions, EMIs).

    Attributes:
        payee: Normalised payee name.
        category: Category of its latest payment.
        typical_amount: Median amount.
        months_seen: Distinct months it was paid in.
        last_paid_on: Date of the latest payment.
    """

    payee: str
    category: Category
    typical_amount: Money
    months_seen: int
    last_paid_on: date


@dataclass(frozen=True, slots=True)
class GoalPlan:
    """Whether a savings target is reachable in time at the current pace.

    Attributes:
        target: Amount to reach.
        already_saved: Amount already put aside for it.
        months: Months available.
        required_monthly: Saving needed each month.
        average_monthly_saving: Average ``saved`` over the reference months
            (``None`` when there is no history).
        reference_months: Months the average is based on.
        on_track: Whether the average covers the requirement.
        monthly_gap: Extra saving needed each month (zero when on track).
        monthly_surplus: Average saving beyond the requirement (zero when short).
        months_needed_at_pace: Months to reach the target at the average pace
            (``None`` without history or when the average is not positive).
    """

    target: Money
    already_saved: Money
    months: int
    required_monthly: Money
    average_monthly_saving: Money | None
    reference_months: int
    on_track: bool | None
    monthly_gap: Money | None
    monthly_surplus: Money | None = None
    months_needed_at_pace: int | None = None


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #


def totals(transactions: Iterable[Transaction]) -> Totals:
    """Headline figures (see module definitions).

    Args:
        transactions: Any transactions.

    Returns:
        The totals.
    """
    income = refunds = spending = invested = Money.zero()
    for txn in transactions:
        if txn.direction is Direction.CREDIT:
            if txn.category is Category.REFUND:
                refunds += txn.amount
            else:
                income += txn.amount
        elif txn.category is Category.INVESTMENTS:
            invested += txn.amount
        else:
            spending += txn.amount
    return Totals(income=income, refunds=refunds, spending=spending, invested=invested)


def summarise(transactions: Sequence[Transaction], period: Period) -> PeriodSummary:
    """Summarise the transactions that fall in ``period``.

    Args:
        transactions: Transactions (others outside the period are ignored).
        period: The period.

    Returns:
        The summary.
    """
    inside = [t for t in transactions if period.contains(t.posted_on)]
    figures = totals(inside)
    spent = [t for t in inside if t.direction is Direction.DEBIT
             and t.category is not Category.INVESTMENTS]
    earned = [t for t in inside if t.direction is Direction.CREDIT
              and t.category is not Category.REFUND]
    return PeriodSummary(
        period=period, totals=figures, transactions=len(inside),
        spending_by_category=_by_category(spent, figures.spending),
        income_by_category=_by_category(earned, figures.income))


def monthly_trend(transactions: Sequence[Transaction]) -> list[MonthTotals]:
    """Headline figures per calendar month, oldest first.

    Args:
        transactions: Transactions.

    Returns:
        One entry per month that has transactions.
    """
    months: dict[str, list[Transaction]] = defaultdict(list)
    for txn in transactions:
        months[f"{txn.posted_on:%Y-%m}"].append(txn)
    return [MonthTotals(month, totals(items)) for month, items in sorted(months.items())]


def payee_of(description: str) -> str:
    """A stable payee name from a narration.

    ``"UPI/DR/412345678901/NETFLIX COM/HDFC/netflix@hdfc"`` -> ``"netflix"``.

    Args:
        description: Bank narration.

    Returns:
        Up to three meaningful words, or ``""`` if none.
    """
    text = _UPI_HANDLE.sub(" ", description.lower())
    words = [w for w in _NON_WORD.split(text)
             if len(w) > 1 and w not in _CHANNEL_WORDS]
    unique = list(dict.fromkeys(words))
    return " ".join(unique[:3])


def recurring_payments(transactions: Sequence[Transaction], *, tolerance: Decimal = Decimal(
        "0.15"), min_months: int = 3) -> list[RecurringPayment]:
    """Payments to the same payee, of about the same amount, about once a month.

    A payee counts when it was paid in at least ``min_months`` distinct months
    (or in every month of the data, when the data covers fewer), at most about
    once a month on average (daily coffee is not a subscription), and with at
    least 80% of payments within ``tolerance`` of the median amount.

    Args:
        transactions: Transactions (credits are ignored).
        tolerance: Allowed relative difference from the median amount.
        min_months: Distinct months required.

    Returns:
        Recurring payments, largest typical amount first.
    """
    covered = {f"{t.posted_on:%Y-%m}" for t in transactions}
    needed = max(2, min(min_months, len(covered)))
    groups: dict[str, list[Transaction]] = defaultdict(list)
    for txn in transactions:
        if txn.direction is Direction.DEBIT:
            payee = payee_of(txn.description)
            if payee:
                groups[payee].append(txn)
    found: list[RecurringPayment] = []
    for payee, items in groups.items():
        months = {f"{t.posted_on:%Y-%m}" for t in items}
        if len(months) < needed or len(items) > len(months) * 1.5:
            continue
        median = Decimal(statistics.median(t.amount.paise for t in items))
        close = [t for t in items if abs(t.amount.paise - median) <= median * tolerance]
        if len(close) < 0.8 * len(items):
            continue
        latest = max(items, key=lambda t: t.posted_on)
        found.append(RecurringPayment(
            payee=payee, category=latest.category,
            typical_amount=Money(int(median.quantize(Decimal(1), rounding=ROUND_HALF_UP))),
            months_seen=len(months), last_paid_on=latest.posted_on))
    return sorted(found, key=lambda r: (-r.typical_amount.paise, r.payee))


def plan_goal(transactions: Sequence[Transaction], *, target: Money, months: int,
              already_saved: Money, today: date, reference_months: int = 3) -> GoalPlan:
    """How much must be saved per month to reach ``target`` in ``months``.

    The current pace is the average ``saved`` over the last ``reference_months``
    complete months that have data.

    Args:
        transactions: The user's transactions.
        target: Goal amount.
        months: Months until the deadline.
        already_saved: Amount already set aside.
        today: Today's date.
        reference_months: Complete months used for the current pace.

    Returns:
        The plan.

    Raises:
        ValueError: If ``target`` or ``months`` is not positive, or
            ``already_saved`` is negative.
    """
    if target.paise <= 0 or months < 1 or already_saved.paise < 0:
        raise ValueError("target and months must be positive; already_saved not negative")
    remaining = target - already_saved if target > already_saved else Money.zero()
    required = remaining.divided_by(months)
    window = Period.last_full_months(today, reference_months)
    history = monthly_trend([t for t in transactions if window.contains(t.posted_on)])
    if not history:
        return GoalPlan(target, already_saved, months, required, None, 0, None, None)
    average = sum((m.totals.saved for m in history), Money.zero()).divided_by(len(history))
    gap = required - average if required > average else Money.zero()
    surplus = average - required if average > required else Money.zero()
    needed = (None if average.paise <= 0
              else -(-remaining.paise // average.paise))  # ceiling division, exact
    return GoalPlan(target, already_saved, months, required, average, len(history),
                    on_track=not gap, monthly_gap=gap, monthly_surplus=surplus,
                    months_needed_at_pace=needed)


def _by_category(items: Sequence[Transaction], whole: Money) -> tuple[CategoryTotal, ...]:
    sums: dict[Category, Money] = defaultdict(Money.zero)
    counts: dict[Category, int] = defaultdict(int)
    for txn in items:
        sums[txn.category] += txn.amount
        counts[txn.category] += 1
    rows = [CategoryTotal(category, total, counts[category],
                          _percent(total.ratio_to(whole) or Decimal(0)))
            for category, total in sums.items()]
    return tuple(sorted(rows, key=lambda r: (-r.total.paise, r.category.value)))


def _percent(ratio: Decimal) -> Decimal:
    return (ratio * 100).quantize(PERCENT, rounding=ROUND_HALF_UP)
