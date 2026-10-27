"""Request and response models: the public API contract.

Kept separate from domain objects on purpose: the API can evolve (rename a
field, add a version) without touching the domain, and the domain never
leaks a field by accident. Amounts are strings with two decimals, so JSON
clients never round money through floating point.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from money_agent.agent.conversation import MAX_MESSAGE_CHARS, ChatReply
from money_agent.agent.runner import ToolTrace, Verification
from money_agent.domain.models import ChatMessage, Memory, Transaction
from money_agent.insights.analytics import CategoryTotal, MonthTotals, PeriodSummary, Totals
from money_agent.services.statements import ImportReport


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class ChatRequest(_Model):
    """A chat message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]
    conversation_id: uuid.UUID | None = None


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class RowProblemOut(_Model):
    """An unreadable statement row."""

    line: int
    reason: str


class ImportOut(_Model):
    """Result of a statement upload."""

    rows_read: int
    imported: int
    duplicates: int
    problem_count: int
    problems: list[RowProblemOut]
    first_date: str
    last_date: str
    columns: dict[str, str]

    @classmethod
    def of(cls, report: ImportReport) -> ImportOut:
        """From the service's report."""
        return cls(rows_read=report.rows_read, imported=report.imported,
                   duplicates=report.duplicates, problem_count=report.problem_count,
                   problems=[RowProblemOut(line=p.line, reason=p.reason)
                             for p in report.problems],
                   first_date=report.first_date, last_date=report.last_date,
                   columns=report.columns)


class OverviewOut(_Model):
    """What data the user has (an empty account is a normal state, not an error)."""

    has_data: bool
    first_date: str | None
    last_date: str | None


class TransactionOut(_Model):
    """One transaction."""

    id: uuid.UUID
    date: str
    description: str
    amount: str
    direction: str
    category: str
    category_label: str

    @classmethod
    def of(cls, txn: Transaction) -> TransactionOut:
        """From a domain transaction."""
        return cls(id=txn.id, date=txn.posted_on.isoformat(), description=txn.description,
                   amount=str(txn.amount), direction=txn.direction.value,
                   category=txn.category.value, category_label=txn.category.label)


class TransactionPageOut(_Model):
    """A page of transactions."""

    items: list[TransactionOut]
    next_cursor: str | None


class TotalsOut(_Model):
    """Headline figures."""

    income: str
    spending: str
    invested: str
    refunds: str
    saved: str
    savings_rate_percent: str | None

    @classmethod
    def of(cls, totals: Totals) -> TotalsOut:
        """From domain totals."""
        rate = totals.savings_rate_percent
        return cls(income=str(totals.income), spending=str(totals.spending),
                   invested=str(totals.invested), refunds=str(totals.refunds),
                   saved=str(totals.saved),
                   savings_rate_percent=None if rate is None else str(rate))


class CategoryOut(_Model):
    """One category's total."""

    category: str
    label: str
    total: str
    count: int
    share_percent: str

    @classmethod
    def of(cls, row: CategoryTotal) -> CategoryOut:
        """From a domain category total."""
        return cls(category=row.category.value, label=row.category.label,
                   total=str(row.total), count=row.count, share_percent=str(row.share_percent))


class SummaryOut(_Model):
    """A period summary."""

    period: str
    start: str
    end: str
    transactions: int
    totals: TotalsOut
    spending_by_category: list[CategoryOut]
    income_by_category: list[CategoryOut]

    @classmethod
    def of(cls, summary: PeriodSummary) -> SummaryOut:
        """From a domain summary."""
        return cls(period=summary.period.label, start=summary.period.start.isoformat(),
                   end=summary.period.end.isoformat(), transactions=summary.transactions,
                   totals=TotalsOut.of(summary.totals),
                   spending_by_category=[CategoryOut.of(r)
                                         for r in summary.spending_by_category],
                   income_by_category=[CategoryOut.of(r) for r in summary.income_by_category])


class MonthOut(_Model):
    """One month's figures."""

    month: str
    totals: TotalsOut

    @classmethod
    def of(cls, month: MonthTotals) -> MonthOut:
        """From domain month totals."""
        return cls(month=month.month, totals=TotalsOut.of(month.totals))


class MemoryOut(_Model):
    """A memory."""

    id: uuid.UUID
    kind: str
    content: str
    updated_at: str

    @classmethod
    def of(cls, memory: Memory) -> MemoryOut:
        """From a domain memory."""
        return cls(id=memory.id, kind=memory.kind.value, content=memory.content,
                   updated_at=memory.updated_at.isoformat())


class VerificationOut(_Model):
    """The answer check's outcome."""

    status: str
    problems: list[str]
    corrected: bool

    @classmethod
    def of(cls, verification: Verification) -> VerificationOut:
        """From the runner's verification."""
        return cls(status=verification.status.value, problems=list(verification.problems),
                   corrected=verification.corrected)


class ToolUseOut(_Model):
    """A tool the agent used."""

    name: str
    ok: bool
    error: str | None

    @classmethod
    def of(cls, trace: ToolTrace) -> ToolUseOut:
        """From the runner's trace."""
        return cls(name=trace.name, ok=trace.ok, error=trace.error)


class ChatOut(_Model):
    """The agent's reply."""

    conversation_id: uuid.UUID
    answer: str
    verification: VerificationOut
    tools: list[ToolUseOut]
    memories_used: int

    @classmethod
    def of(cls, reply: ChatReply) -> ChatOut:
        """From the chat service's reply."""
        return cls(conversation_id=reply.conversation_id, answer=reply.answer,
                   verification=VerificationOut.of(reply.verification),
                   tools=[ToolUseOut.of(t) for t in reply.tools],
                   memories_used=reply.memories_used)


class MessageOut(_Model):
    """A stored chat message."""

    role: str
    content: str
    created_at: str
    meta: dict[str, Any]

    @classmethod
    def of(cls, message: ChatMessage) -> MessageOut:
        """From a stored message."""
        return cls(role=message.role.value, content=message.content,
                   created_at=message.created_at.isoformat(), meta=message.meta)
