"""The agent's tools: read the user's money, remember what matters to them.

Read tools (``get_*``, ``list_*``, ``find_*``, ``plan_*``) only compute; the
two write tools (``remember``, ``delete_memory``) only touch the user's own
memories. No tool can move money or reach outside the user's data.

**Output conventions** (they matter to the answer check, which compares every
number in the final answer with these results):

* amounts are plain rupee strings with two decimals (``"12450.50"``); the
  model writes them as ``₹12,450.50``, which the check reads as the same number;
* percentages are strings with one decimal (``"55.2"``);
* every result says which period it covers, so the model never has to guess.

Names are chosen with the check in mind too: the memory tool is ``remember``,
not ``save_memory``, because "you saved ₹47,027" is a statement about money,
and must not read as a claim that a save tool ran.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from pydantic import Field, field_validator

from money_agent.agent.tools import Tool, ToolArgs, ToolContext, error_result
from money_agent.domain.models import (
    Category,
    Direction,
    MemoryKind,
    Transaction,
    TransactionQuery,
)
from money_agent.domain.money import Money
from money_agent.insights.analytics import CategoryTotal, Totals
from money_agent.memory.service import MemoryService
from money_agent.services.finance import FinanceService

MAX_LISTED = 20
MAX_TARGET_RUPEES = Decimal("100000000")  # ₹10 crore


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


def _totals(figures: Totals) -> dict[str, Any]:
    rate = figures.savings_rate_percent
    return {"income": str(figures.income), "spending": str(figures.spending),
            "invested": str(figures.invested), "refunds": str(figures.refunds),
            "saved": str(figures.saved),
            "savings_rate_percent": None if rate is None else str(rate)}


def _category_rows(rows: tuple[CategoryTotal, ...]) -> list[dict[str, Any]]:
    return [{"category": r.category.value, "label": r.category.label, "total": str(r.total),
             "count": r.count, "share_percent": str(r.share_percent)} for r in rows]


def _transaction(txn: Transaction) -> dict[str, Any]:
    return {"date": txn.posted_on.isoformat(), "description": txn.description,
            "amount": str(txn.amount), "direction": txn.direction.value,
            "category": txn.category.value}


# --------------------------------------------------------------------------- #
# Read tools
# --------------------------------------------------------------------------- #


class PeriodArgs(ToolArgs):
    """A period: a month, or a date range. Omit both for the latest month with data."""

    month: Annotated[str | None, Field(pattern=r"^\d{4}-\d{2}$",
                                       description="Calendar month, YYYY-MM.")] = None
    start_date: Annotated[date | None, Field(description="First day, YYYY-MM-DD.")] = None
    end_date: Annotated[date | None, Field(description="Last day, YYYY-MM-DD.")] = None


class SpendingSummaryTool(Tool[PeriodArgs]):
    """Income, spending, savings and categories for a period."""

    def __init__(self, finance: FinanceService) -> None:
        """Create the tool.

        Args:
            finance: Finance queries.
        """
        super().__init__(
            "get_spending_summary",
            "Income, spending, investments, savings, savings rate and spending by category "
            "for a month or date range. Defaults to the latest month with data. Use it for "
            "any question about how much was earned, spent or saved.", PeriodArgs)
        self._finance = finance

    async def run(self, context: ToolContext, args: PeriodArgs) -> dict[str, Any]:
        """Summarise the period (see class docstring)."""
        period = await self._finance.resolve_period(
            context.user_id, month=args.month, start=args.start_date, end=args.end_date)
        summary = await self._finance.summary(context.user_id, period)
        return {"period": period.label, "start": period.start.isoformat(),
                "end": period.end.isoformat(), "transactions": summary.transactions,
                **_totals(summary.totals),
                "spending_by_category": _category_rows(summary.spending_by_category),
                "income_by_category": _category_rows(summary.income_by_category)}


class ListTransactionsArgs(ToolArgs):
    """Filters for listing transactions."""

    category: Category | None = None
    search: Annotated[str | None, Field(max_length=60,
                                        description="Text in the bank narration, e.g. a "
                                                    "merchant name.")] = None
    direction: Direction | None = None
    start_date: date | None = None
    end_date: date | None = None
    limit: Annotated[int, Field(ge=1, le=MAX_LISTED)] = 10


class ListTransactionsTool(Tool[ListTransactionsArgs]):
    """Individual transactions, newest first."""

    def __init__(self, finance: FinanceService) -> None:
        """Create the tool.

        Args:
            finance: Finance queries.
        """
        super().__init__(
            "list_transactions",
            "Individual transactions, newest first, filtered by category, merchant text, "
            f"direction or dates (at most {MAX_LISTED}). Use it to show or explain specific "
            "payments, not to add up totals (use get_spending_summary for totals).",
            ListTransactionsArgs)
        self._finance = finance

    async def run(self, context: ToolContext, args: ListTransactionsArgs) -> dict[str, Any]:
        """List matching transactions (see class docstring)."""
        if args.start_date and args.end_date and args.start_date > args.end_date:
            return error_result("invalid_arguments", "start_date is after end_date.")
        page = await self._finance.transactions(context.user_id, TransactionQuery(
            start=args.start_date, end=args.end_date, category=args.category,
            direction=args.direction, search=args.search, limit=args.limit))
        return {"transactions": [_transaction(t) for t in page.items],
                "returned": len(page.items), "more_available": page.next_cursor is not None}


class TrendArgs(ToolArgs):
    """How many months to show."""

    months: Annotated[int, Field(ge=1, le=12)] = 6


class MonthlyTrendTool(Tool[TrendArgs]):
    """Month-by-month income, spending and savings."""

    def __init__(self, finance: FinanceService) -> None:
        """Create the tool.

        Args:
            finance: Finance queries.
        """
        super().__init__(
            "get_monthly_trend",
            "Income, spending, investments and savings for each recent month that has data, "
            "oldest first. Use it to compare months or spot trends.", TrendArgs)
        self._finance = finance

    async def run(self, context: ToolContext, args: TrendArgs) -> dict[str, Any]:
        """Monthly figures (see class docstring)."""
        months = await self._finance.monthly_trend(context.user_id, args.months)
        return {"months": [{"month": m.month, **_totals(m.totals)} for m in months]}


class NoArgs(ToolArgs):
    """This tool takes no arguments."""


class RecurringPaymentsTool(Tool[NoArgs]):
    """Subscriptions, rent, EMIs and SIPs: payments that repeat monthly."""

    def __init__(self, finance: FinanceService) -> None:
        """Create the tool.

        Args:
            finance: Finance queries.
        """
        super().__init__(
            "find_recurring_payments",
            "Payments that repeat about monthly (rent, EMIs, SIPs, subscriptions, bills), "
            "with their typical amount. Use it for questions about fixed costs or "
            "subscriptions.", NoArgs)
        self._finance = finance

    async def run(self, context: ToolContext, args: NoArgs) -> dict[str, Any]:
        """Recurring payments (see class docstring)."""
        found = await self._finance.recurring(context.user_id)
        total = sum((r.typical_amount for r in found), Money.zero())
        return {"recurring": [{"payee": r.payee, "category": r.category.value,
                               "typical_amount": str(r.typical_amount),
                               "months_seen": r.months_seen,
                               "last_paid_on": r.last_paid_on.isoformat()} for r in found],
                "monthly_total": str(total)}


class GoalPlanArgs(ToolArgs):
    """A savings target."""

    target_amount: Annotated[Decimal, Field(gt=0, le=MAX_TARGET_RUPEES,
                                            description="Target in rupees.")]
    months: Annotated[int, Field(ge=1, le=120, description="Months until the deadline.")]
    already_saved: Annotated[Decimal, Field(ge=0, le=MAX_TARGET_RUPEES,
                                            description="Rupees already put aside.")] = \
        Decimal(0)


class PlanGoalTool(Tool[GoalPlanArgs]):
    """Monthly saving needed for a target, against the current pace."""

    def __init__(self, finance: FinanceService) -> None:
        """Create the tool.

        Args:
            finance: Finance queries.
        """
        super().__init__(
            "plan_savings_goal",
            "How much must be saved each month to reach a target in a number of months, "
            "compared with the user's average monthly saving over recent complete months.",
            GoalPlanArgs)
        self._finance = finance

    async def run(self, context: ToolContext, args: GoalPlanArgs) -> dict[str, Any]:
        """Plan the goal (see class docstring)."""
        plan = await self._finance.plan_goal(
            context.user_id, target=Money.of(args.target_amount), months=args.months,
            already_saved=Money.of(args.already_saved))
        return {"target": str(plan.target), "already_saved": str(plan.already_saved),
                "months": plan.months, "required_monthly": str(plan.required_monthly),
                "average_monthly_saving": None if plan.average_monthly_saving is None
                else str(plan.average_monthly_saving),
                "based_on_months": plan.reference_months, "on_track": plan.on_track,
                "monthly_gap": None if plan.monthly_gap is None else str(plan.monthly_gap)}


# --------------------------------------------------------------------------- #
# Memory tools
# --------------------------------------------------------------------------- #


class RememberArgs(ToolArgs):
    """Something to remember about the user."""

    kind: Annotated[MemoryKind, Field(description="goal, fact or preference")]
    content: Annotated[str, Field(min_length=3, max_length=300,
                                  description="One short sentence, in the third person, e.g. "
                                              "'Saving for a bike: ₹1,20,000 by March 2027'.")]
    replaces_memory_id: Annotated[str | None, Field(
        description="Id of a memory this one updates (from 'What you know').")] = None

    @field_validator("replaces_memory_id")
    @classmethod
    def _uuid(cls, value: str | None) -> str | None:
        if value is not None:
            uuid.UUID(value)
        return value


class RememberTool(Tool[RememberArgs]):
    """Stores a goal, fact or preference for future conversations."""

    def __init__(self, memory: MemoryService) -> None:
        """Create the tool.

        Args:
            memory: Memory service.
        """
        super().__init__(
            "remember",
            "Remember a goal, fact or preference the user stated, for future conversations. "
            "Use it when the user tells you something lasting (a goal, their salary date, "
            "how they like answers), not for one-off questions. To change something you "
            "already know, pass its id as replaces_memory_id.", RememberArgs)
        self._memory = memory

    async def run(self, context: ToolContext, args: RememberArgs) -> dict[str, Any]:
        """Store the memory (see class docstring)."""
        result = await self._memory.remember(
            context.user_id, args.kind, args.content,
            replaces=uuid.UUID(args.replaces_memory_id) if args.replaces_memory_id else None)
        return {"status": "created" if result.created else "updated",
                "memory_id": str(result.memory.id), "kind": result.memory.kind.value,
                "content": result.memory.content}


class DeleteMemoryArgs(ToolArgs):
    """Which memory to delete."""

    memory_id: Annotated[str, Field(description="Id from 'What you know'.")]

    @field_validator("memory_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class DeleteMemoryTool(Tool[DeleteMemoryArgs]):
    """Deletes a memory the user no longer wants kept."""

    def __init__(self, memory: MemoryService) -> None:
        """Create the tool.

        Args:
            memory: Memory service.
        """
        super().__init__(
            "delete_memory",
            "Delete something you remembered, when the user asks you to forget it or it is "
            "no longer true.", DeleteMemoryArgs)
        self._memory = memory

    async def run(self, context: ToolContext, args: DeleteMemoryArgs) -> dict[str, Any]:
        """Delete the memory (see class docstring)."""
        await self._memory.forget(context.user_id, uuid.UUID(args.memory_id))
        return {"status": "deleted", "memory_id": args.memory_id}


class ListMemoriesTool(Tool[NoArgs]):
    """Everything remembered about the user."""

    def __init__(self, memory: MemoryService) -> None:
        """Create the tool.

        Args:
            memory: Memory service.
        """
        super().__init__(
            "list_memories",
            "Everything you remember about the user (goals, facts, preferences), with ids. "
            "Use it when the user asks what you know about them.", NoArgs)
        self._memory = memory

    async def run(self, context: ToolContext, args: NoArgs) -> dict[str, Any]:
        """List memories (see class docstring)."""
        memories = await self._memory.all(context.user_id)
        return {"memories": [{"id": str(m.id), "kind": m.kind.value, "content": m.content,
                              "updated_on": m.updated_at.date().isoformat()}
                             for m in memories]}


def default_tools(finance: FinanceService, memory: MemoryService) -> list[Tool[Any]]:
    """Every tool the money agent offers.

    Args:
        finance: Finance queries.
        memory: Memory service.

    Returns:
        The tools.
    """
    return [SpendingSummaryTool(finance), ListTransactionsTool(finance),
            MonthlyTrendTool(finance), RecurringPaymentsTool(finance), PlanGoalTool(finance),
            RememberTool(memory), DeleteMemoryTool(memory), ListMemoriesTool(memory)]
