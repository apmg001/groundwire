"""Analytics (pure) and the finance service (periods, defaults, validation)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from conftest import USER, FrozenClock, make_txn
from money_agent.adapters.inmemory import InMemoryStore, InMemoryTransactionRepository
from money_agent.core.errors import InvalidInputError
from money_agent.domain.models import Category, Direction
from money_agent.domain.money import Money
from money_agent.insights import analytics
from money_agent.insights.analytics import Period
from money_agent.services.finance import FinanceService, NoDataError

CREDIT = Direction.CREDIT
SEPT = Period.month(2026, 9)


def _month(year: int, month: int) -> list:  # type: ignore[type-arg]
    """One typical month: salary, rent, SIP, food, a refund."""
    return [make_txn(date(year, month, 1), "SALARY", "80000", CREDIT, Category.SALARY),
            make_txn(date(year, month, 3), "RENT", "20000", category=Category.RENT),
            make_txn(date(year, month, 5), "SIP", "10000", category=Category.INVESTMENTS),
            make_txn(date(year, month, 9), "SWIGGY", "1500", category=Category.FOOD_DINING),
            make_txn(date(year, month, 20), "SWIGGY", "500", category=Category.FOOD_DINING),
            make_txn(date(year, month, 21), "REFUND", "300", CREDIT, Category.REFUND)]


def test_totals_follow_the_documented_definitions() -> None:
    figures = analytics.totals(_month(2026, 9))
    assert figures.income == Money.of(80000)     # refunds are not income
    assert figures.refunds == Money.of(300)
    assert figures.spending == Money.of(22000)   # investments are not spending
    assert figures.invested == Money.of(10000)
    assert figures.saved == Money.of(58300)      # income + refunds - spending
    assert figures.savings_rate_percent == Decimal("72.9")
    assert analytics.totals([]).savings_rate_percent is None


def test_summary_only_counts_the_period_and_ranks_categories() -> None:
    rows = _month(2026, 8) + _month(2026, 9)
    summary = analytics.summarise(rows, SEPT)
    assert summary.transactions == 6
    assert [(c.category, c.total, c.count, c.share_percent)
            for c in summary.spending_by_category] == [
        (Category.RENT, Money.of(20000), 1, Decimal("90.9")),
        (Category.FOOD_DINING, Money.of(2000), 2, Decimal("9.1"))]
    assert summary.income_by_category[0].category is Category.SALARY


def test_monthly_trend_is_oldest_first() -> None:
    trend = analytics.monthly_trend(_month(2026, 9) + _month(2026, 7))
    assert [m.month for m in trend] == ["2026-07", "2026-09"]


def test_recurring_finds_fixed_payments_but_not_frequent_or_variable_ones() -> None:
    rows = []
    for month in (7, 8, 9):
        rows.append(make_txn(date(2026, month, 7), "UPI/DR/123/NETFLIX COM/netflix@hdfc",
                             "649", category=Category.SUBSCRIPTIONS))
        rows.append(make_txn(date(2026, month, 10), "BESCOM ELECTRICITY",
                             ["1800", "2000", "1900"][month - 7], category=Category.UTILITIES))
        rows.append(make_txn(date(2026, month, 12), "GIFT SHOP",
                             ["100", "5000", "900"][month - 7]))  # varies wildly
        for day in range(1, 25, 2):  # coffee every other day: frequent, not a subscription
            rows.append(make_txn(date(2026, month, day), "BLUE TOKAI COFFEE", "220"))
    found = {r.payee: r for r in analytics.recurring_payments(rows)}
    assert set(found) == {"netflix", "bescom electricity"}
    assert found["netflix"].typical_amount == Money.of(649)
    assert found["netflix"].months_seen == 3
    assert found["bescom electricity"].typical_amount == Money.of(1900)  # median


def test_payee_names_drop_channels_ids_and_upi_handles() -> None:
    assert analytics.payee_of("UPI/DR/412345678901/NETFLIX COM/netflix@hdfc") == "netflix"
    assert analytics.payee_of("ACH D- ZERODHA COIN SIP") == "zerodha coin sip"
    assert analytics.payee_of("123 456") == ""


def test_goal_plan_compares_required_and_current_pace() -> None:
    rows = _month(2026, 7) + _month(2026, 8) + _month(2026, 9)
    on_track = analytics.plan_goal(rows, target=Money.of(120000), months=6,
                                   already_saved=Money.of(20000), today=date(2026, 10, 15))
    assert on_track.required_monthly == Money.of("16666.67")
    assert on_track.average_monthly_saving == Money.of(58300)
    assert (on_track.reference_months, on_track.on_track) == (3, True)
    assert on_track.monthly_gap == Money.zero()
    stretch = analytics.plan_goal(rows, target=Money.of(1000000), months=6,
                                  already_saved=Money.zero(), today=date(2026, 10, 15))
    assert stretch.on_track is False
    assert stretch.monthly_gap == Money.of("166666.67") - Money.of(58300)
    no_history = analytics.plan_goal([], target=Money.of(100), months=1,
                                     already_saved=Money.zero(), today=date(2026, 10, 15))
    assert no_history.on_track is None and no_history.average_monthly_saving is None
    done = analytics.plan_goal(rows, target=Money.of(100), months=2,
                               already_saved=Money.of(500), today=date(2026, 10, 15))
    assert done.required_monthly == Money.zero()
    with pytest.raises(ValueError):
        analytics.plan_goal(rows, target=Money.zero(), months=1, already_saved=Money.zero(),
                            today=date(2026, 10, 15))


def test_periods() -> None:
    assert Period.month(2026, 2).end == date(2026, 2, 28)
    assert Period.last_full_months(date(2026, 10, 15), 3) == Period(date(2026, 7, 1),
                                                                    date(2026, 9, 30))
    assert Period.month(2026, 9).label == "2026-09"
    assert Period(date(2026, 9, 1), date(2026, 9, 15)).label == "2026-09-01..2026-09-15"
    with pytest.raises(ValueError):
        Period(date(2026, 9, 2), date(2026, 9, 1))
    with pytest.raises(ValueError):
        Period.last_full_months(date(2026, 9, 1), 0)


# --------------------------------------------------------------------------- #
# Finance service
# --------------------------------------------------------------------------- #


async def _finance(rows: list) -> FinanceService:  # type: ignore[type-arg]
    repo = InMemoryTransactionRepository(InMemoryStore())
    await repo.add_many(USER, rows)
    return FinanceService(repo, FrozenClock())


async def test_default_period_is_the_latest_month_with_data() -> None:
    finance = await _finance(_month(2026, 8) + _month(2026, 9))  # "today" is 28 Oct
    assert await finance.resolve_period(USER) == SEPT
    assert await finance.resolve_period(USER, month="2026-08") == Period.month(2026, 8)
    assert await finance.resolve_period(USER, start=date(2026, 9, 2)) == Period(
        date(2026, 9, 2), date(2026, 9, 21))


@pytest.mark.parametrize("kwargs", [
    {"month": "2026-13"}, {"month": "september"},
    {"start": date(2026, 9, 10), "end": date(2026, 9, 1)},
    {"start": date(2020, 1, 1), "end": date(2026, 1, 1)},
])
async def test_bad_periods_are_refused_with_a_public_message(kwargs: dict) -> None:  # type: ignore[type-arg]
    finance = await _finance(_month(2026, 9))
    with pytest.raises(InvalidInputError) as caught:
        await finance.resolve_period(USER, **kwargs)
    assert caught.value.public_message


async def test_no_data_is_its_own_error() -> None:
    finance = await _finance([])
    with pytest.raises(NoDataError) as caught:
        await finance.resolve_period(USER)
    assert caught.value.code == "no_transactions"
    with pytest.raises(NoDataError):
        await finance.recurring(USER)


async def test_trend_and_goal_through_the_service() -> None:
    finance = await _finance(_month(2026, 7) + _month(2026, 8) + _month(2026, 9))
    assert [m.month for m in await finance.monthly_trend(USER, 2)] == ["2026-08", "2026-09"]
    with pytest.raises(InvalidInputError):
        await finance.monthly_trend(USER, 0)
    plan = await finance.plan_goal(USER, target=Money.of(120000), months=6,
                                   already_saved=Money.zero())
    # The data ends on 21 September: September is partial, so the pace comes from the
    # complete months before it that have data (July and August).
    assert plan.reference_months == 2 and plan.on_track
    with pytest.raises(InvalidInputError):
        await finance.plan_goal(USER, target=Money.of(-1), months=6, already_saved=Money.zero())
