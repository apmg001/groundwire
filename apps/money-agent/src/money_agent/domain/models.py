"""Domain entities and value objects.

Plain, immutable dataclasses with no framework or database types in them, so
every layer (parser, analytics, agent tools, repositories, API) speaks the
same language and can be tested without I/O.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from money_agent.domain.money import Money

MAX_MEMORY_CHARS = 500
MAX_PAGE_SIZE = 200


class Direction(StrEnum):
    """Which way money moved."""

    DEBIT = "debit"    # money out
    CREDIT = "credit"  # money in


class Category(StrEnum):
    """Spending and income categories."""

    SALARY = "salary"
    OTHER_INCOME = "other_income"
    REFUND = "refund"
    RENT = "rent"
    GROCERIES = "groceries"
    FOOD_DINING = "food_dining"
    TRANSPORT = "transport"
    FUEL = "fuel"
    SHOPPING = "shopping"
    UTILITIES = "utilities"
    SUBSCRIPTIONS = "subscriptions"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    EDUCATION = "education"
    TRAVEL = "travel"
    LOAN_EMI = "loan_emi"
    INVESTMENTS = "investments"
    INSURANCE = "insurance"
    CASH = "cash_withdrawal"
    TRANSFERS = "transfers"
    FEES = "fees_charges"
    OTHER = "other"

    @property
    def label(self) -> str:
        """Human-readable name (``food_dining`` -> ``Food & dining``)."""
        return _LABELS.get(self, self.value.replace("_", " ").capitalize())

    @property
    def is_income(self) -> bool:
        """True for categories that represent money coming in."""
        return self in (Category.SALARY, Category.OTHER_INCOME, Category.REFUND)


_LABELS = {
    Category.FOOD_DINING: "Food & dining",
    Category.LOAN_EMI: "Loan EMIs",
    Category.CASH: "Cash withdrawals",
    Category.FEES: "Fees & charges",
}


@dataclass(frozen=True, slots=True)
class Transaction:
    """One line of a bank statement.

    Attributes:
        id: Stable identifier.
        user_id: Owner.
        posted_on: Date the bank posted it.
        description: The bank's narration, as written.
        amount: Size of the movement (never negative; see ``direction``).
        direction: Money out or in.
        category: Assigned category.
        fingerprint: Identity across uploads, so re-uploading never double-counts.
        balance: Running balance after this line, if the statement had one.
    """

    id: uuid.UUID
    user_id: str
    posted_on: date
    description: str
    amount: Money
    direction: Direction
    category: Category
    fingerprint: str
    balance: Money | None = None

    def __post_init__(self) -> None:
        if self.amount.paise < 0:
            raise ValueError("transaction amount must not be negative; use direction")

    @property
    def signed(self) -> Money:
        """Positive for money in, negative for money out."""
        return self.amount if self.direction is Direction.CREDIT else -self.amount


class MemoryKind(StrEnum):
    """What a memory is about."""

    GOAL = "goal"              # "saving for a bike by March"
    FACT = "fact"              # "my salary comes on the 1st", "I live in Pune"
    PREFERENCE = "preference"  # "keep answers short", "don't suggest credit cards"


@dataclass(frozen=True, slots=True)
class Memory:
    """Something the user told the agent that should outlive the conversation.

    Attributes:
        id: Stable identifier.
        user_id: Owner.
        kind: Goal, fact or preference.
        content: The memory, in one short sentence.
        created_at: When it was first stored.
        updated_at: When it was last replaced by a newer version.
    """

    id: uuid.UUID
    user_id: str
    kind: MemoryKind
    content: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.content.strip() or len(self.content) > MAX_MEMORY_CHARS:
            raise ValueError(f"memory content must be 1-{MAX_MEMORY_CHARS} characters")


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """A memory with its similarity to a query (cosine, -1..1)."""

    memory: Memory
    similarity: float


class MessageRole(StrEnum):
    """Who wrote a stored chat message."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One stored message of a conversation (tool traffic is not stored).

    Attributes:
        role: User or assistant.
        content: Text.
        created_at: When it was stored.
        meta: Assistant metadata (verification result, tools used).
    """

    role: MessageRole
    content: str
    created_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Cursor:
    """Position in a newest-first transaction list (keyset pagination).

    Keyset cursors stay fast on any page (no ``OFFSET`` scan) and never skip or
    repeat rows when new transactions arrive between pages.

    Attributes:
        posted_on: Date of the last row on the previous page.
        id: Id of that row (tie-breaker within a date).
    """

    posted_on: date
    id: uuid.UUID

    def encode(self) -> str:
        """Opaque string for API clients."""
        raw = f"{self.posted_on.isoformat()}|{self.id}".encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> Cursor:
        """Parse a cursor from :meth:`encode`.

        Args:
            token: The opaque string.

        Returns:
            The cursor.

        Raises:
            ValueError: If the token is malformed.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            day, ident = base64.urlsafe_b64decode(padded.encode()).decode().split("|")
            return cls(date.fromisoformat(day), uuid.UUID(ident))
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("malformed cursor") from exc


@dataclass(frozen=True, slots=True)
class TransactionQuery:
    """Filters for listing a user's transactions (all optional, combined with AND).

    Attributes:
        start: First date included.
        end: Last date included.
        category: Only this category.
        direction: Only money in or out.
        search: Case-insensitive substring of the description.
        limit: Page size (1..MAX_PAGE_SIZE).
        after: Continue after this cursor.
    """

    start: date | None = None
    end: date | None = None
    category: Category | None = None
    direction: Direction | None = None
    search: str | None = None
    limit: int = 50
    after: Cursor | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")


@dataclass(frozen=True, slots=True)
class TransactionPage:
    """One page of transactions, newest first.

    Attributes:
        items: The transactions.
        next_cursor: Cursor for the next page, or ``None`` on the last page.
    """

    items: tuple[Transaction, ...]
    next_cursor: Cursor | None
