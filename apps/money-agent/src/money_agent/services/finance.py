"""Use case: answer questions about a user's money.

The read side shared by the API (dashboards) and the agent's tools, so both
always show the same numbers for the same question.

**Periods.** When no period is given, the default is the *latest month that
has data*, not the calendar month: a user who uploaded September's statement
in late October asking "how am I doing?" means September, and an empty
October would answer nothing useful.

**Scale.** Analytics load the period's rows and compute in Python (exact
integer paise, one code path for every backend). A user's year is a few
thousand rows, which is milliseconds. If that ever stops being true, the
place to push aggregation into SQL is this class, behind the same methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from money_agent.core.errors import InvalidInputError, NotFoundError
from money_agent.domain.models import TransactionPage, TransactionQuery
from money_agent.domain.money import Money
from money_agent.domain.ports import Clock, TransactionRepository
from money_agent.insights import analytics
from money_agent.insights.analytics import (
    GoalPlan,
    MonthTotals,
    Period,
    PeriodSummary,
    RecurringPayment,
)

MAX_PERIOD_DAYS = 366 * 2
RECURRING_LOOKBACK_DAYS = 190


class NoDataError(NotFoundError):
    """The user has no transactions yet: there is nothing to analyse (HTTP 404)."""

    code = "no_transactions"
    default_public_message = "Upload a bank statement first, so there is something to analyse."


@dataclass(frozen=True, slots=True)
class DataRange:
    """The dates the user's data covers.

    Attributes:
        first: Earliest transaction date.
        last: Latest transaction date.
    """

    first: date
    last: date


class FinanceService:
    """Read-only questions over a user's transactions."""

    def __init__(self, transactions: TransactionRepository, clock: Clock) -> None:
        """Create the service.

        Args:
            transactions: Transaction storage.
            clock: Time source.
        """
        self._transactions = transactions
        self._clock = clock

    async def data_range(self, user_id: str) -> DataRange:
        """Dates covered by the user's data.

        Raises:
            NoDataError: No transactions.
            StorageUnavailableError: The store failed.
        """
        bounds = await self._transactions.date_bounds(user_id)
        if bounds is None:
            raise NoDataError(f"user {user_id} has no transactions")
        return DataRange(*bounds)

    async def overview(self, user_id: str) -> DataRange | None:
        """The dates covered, or ``None`` before the first upload (a normal state).

        Raises:
            StorageUnavailableError: The store failed.
        """
        bounds = await self._transactions.date_bounds(user_id)
        return None if bounds is None else DataRange(*bounds)

    async def resolve_period(self, user_id: str, *, month: str | None = None,
                             start: date | None = None, end: date | None = None) -> Period:
        """Turn optional inputs into a concrete period.

        Args:
            user_id: Owner.
            month: ``YYYY-MM`` (wins over start/end).
            start: First day (defaults to the start of the data).
            end: Last day (defaults to the end of the data).

        Returns:
            The period; the latest month with data when nothing is given.

        Raises:
            InvalidInputError: Malformed month, reversed range, or over two years.
            NoDataError: No transactions.
            StorageUnavailableError: The store failed.
        """
        if month is not None:
            return _parse_month(month)
        data = await self.data_range(user_id)
        if start is None and end is None:
            return Period.month(data.last.year, data.last.month)
        period_start, period_end = start or data.first, end or data.last
        if period_start > period_end:
            raise InvalidInputError("start after end",
                                    public_message="The start date is after the end date.")
        if (period_end - period_start).days > MAX_PERIOD_DAYS:
            raise InvalidInputError("period too long",
                                    public_message="Please choose a period of up to two years.")
        return Period(period_start, period_end)

    async def summary(self, user_id: str, period: Period) -> PeriodSummary:
        """Income, spending and categories for a period.

        Raises:
            StorageUnavailableError: The store failed.
        """
        rows = await self._transactions.between(user_id, period.start, period.end)
        return analytics.summarise(rows, period)

    async def monthly_trend(self, user_id: str, months: int) -> list[MonthTotals]:
        """The last ``months`` months that have data, oldest first.

        Raises:
            InvalidInputError: ``months`` outside 1..24.
            NoDataError: No transactions.
            StorageUnavailableError: The store failed.
        """
        if not 1 <= months <= 24:
            raise InvalidInputError("months out of range",
                                    public_message="Choose between 1 and 24 months.")
        data = await self.data_range(user_id)
        start = data.last.replace(day=1)
        for _ in range(months - 1):
            start = (start - timedelta(days=1)).replace(day=1)
        rows = await self._transactions.between(user_id, start, data.last)
        return analytics.monthly_trend(rows)

    async def recurring(self, user_id: str) -> list[RecurringPayment]:
        """Monthly-ish payments over the last six months of data.

        Raises:
            NoDataError: No transactions.
            StorageUnavailableError: The store failed.
        """
        data = await self.data_range(user_id)
        rows = await self._transactions.between(
            user_id, data.last - timedelta(days=RECURRING_LOOKBACK_DAYS), data.last)
        return analytics.recurring_payments(rows)

    async def plan_goal(self, user_id: str, *, target: Money, months: int,
                        already_saved: Money) -> GoalPlan:
        """Required monthly saving for a target, against the current pace.

        The pace is measured on the three complete months before the latest
        data (not before today), for the same reason as default periods.

        Raises:
            InvalidInputError: Non-positive target or months, or negative savings.
            NoDataError: No transactions.
            StorageUnavailableError: The store failed.
        """
        if target.paise <= 0 or months < 1 or already_saved.paise < 0:
            raise InvalidInputError("bad goal inputs", public_message=(
                "The target and the number of months must be positive."))
        data = await self.data_range(user_id)
        reference_today = data.last + timedelta(days=1)
        rows = await self._transactions.between(user_id, data.last - timedelta(days=120),
                                                data.last)
        return analytics.plan_goal(rows, target=target, months=months,
                                   already_saved=already_saved, today=reference_today)

    async def transactions(self, user_id: str, query: TransactionQuery) -> TransactionPage:
        """One page of transactions.

        Raises:
            StorageUnavailableError: The store failed.
        """
        return await self._transactions.page(user_id, query)

    def today(self) -> date:
        """Today in the users' time zone."""
        return self._clock.today()


def _parse_month(month: str) -> Period:
    try:
        year, number = (int(part) for part in month.split("-"))
        return Period.month(year, number)
    except ValueError as exc:
        raise InvalidInputError(f"bad month {month!r}", public_message=(
            "Months are written as YYYY-MM, for example 2026-09.")) from exc
