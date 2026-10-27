"""In-memory implementations of the storage ports.

For tests and quick local runs (``MONEY_STORAGE=memory``): same behaviour as
the Postgres adapters (the shared contract tests run against both), but data
lives in one process and is lost on restart. Never use it with more than one
replica.

Each method finishes its read-modify-write without awaiting, so concurrent
requests on the event loop cannot interleave inside one.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from money_agent.core.errors import NotFoundError
from money_agent.domain.models import (
    ChatMessage,
    Cursor,
    Memory,
    MemoryKind,
    ScoredMemory,
    Transaction,
    TransactionPage,
    TransactionQuery,
)
from money_agent.memory.embeddings import cosine


@dataclass
class InMemoryStore:
    """All data, shared by the in-memory repositories of one container."""

    users: set[str] = field(default_factory=set)
    transactions: dict[str, dict[str, Transaction]] = field(
        default_factory=lambda: defaultdict(dict))  # user -> fingerprint -> txn
    memories: dict[str, dict[uuid.UUID, tuple[Memory, list[float]]]] = field(
        default_factory=lambda: defaultdict(dict))
    conversations: dict[uuid.UUID, str] = field(default_factory=dict)  # id -> user
    messages: dict[uuid.UUID, list[ChatMessage]] = field(
        default_factory=lambda: defaultdict(list))


class InMemoryUserRepository:
    """Users."""

    def __init__(self, store: InMemoryStore) -> None:
        """Create the repository.

        Args:
            store: Shared store.
        """
        self._store = store

    async def ensure(self, user_id: str) -> None:
        """Create the user if missing (see port)."""
        self._store.users.add(user_id)


class InMemoryTransactionRepository:
    """Transactions."""

    def __init__(self, store: InMemoryStore) -> None:
        """Create the repository.

        Args:
            store: Shared store.
        """
        self._store = store

    async def add_many(self, user_id: str, transactions: Sequence[Transaction]) -> int:
        """Insert new transactions (see port).

        Raises:
            ValueError: If a transaction belongs to another user.
        """
        if any(t.user_id != user_id for t in transactions):
            raise ValueError("every transaction must belong to user_id")
        rows = self._store.transactions[user_id]
        new = 0
        for txn in transactions:
            if txn.fingerprint not in rows:
                rows[txn.fingerprint] = txn
                new += 1
        return new

    async def page(self, user_id: str, query: TransactionQuery) -> TransactionPage:
        """Keyset-paginated listing, newest first (see port)."""
        needle = query.search.lower() if query.search else None
        rows = sorted(self._store.transactions[user_id].values(),
                      key=lambda t: (t.posted_on, t.id), reverse=True)
        matching = [
            t for t in rows
            if (query.start is None or t.posted_on >= query.start)
            and (query.end is None or t.posted_on <= query.end)
            and (query.category is None or t.category is query.category)
            and (query.direction is None or t.direction is query.direction)
            and (needle is None or needle in t.description.lower())
            and (query.after is None
                 or (t.posted_on, t.id) < (query.after.posted_on, query.after.id))
        ]
        items = tuple(matching[:query.limit])
        more = len(matching) > query.limit
        return TransactionPage(items, Cursor(items[-1].posted_on, items[-1].id) if more else None)

    async def between(self, user_id: str, start: date, end: date) -> list[Transaction]:
        """All transactions in a range, oldest first (see port)."""
        return sorted((t for t in self._store.transactions[user_id].values()
                       if start <= t.posted_on <= end), key=lambda t: (t.posted_on, t.id))

    async def date_bounds(self, user_id: str) -> tuple[date, date] | None:
        """Earliest and latest dates (see port)."""
        dates = [t.posted_on for t in self._store.transactions[user_id].values()]
        return (min(dates), max(dates)) if dates else None


class InMemoryMemoryRepository:
    """Memories with exact cosine search."""

    def __init__(self, store: InMemoryStore) -> None:
        """Create the repository.

        Args:
            store: Shared store.
        """
        self._store = store

    async def add(self, memory: Memory, vector: Sequence[float]) -> None:
        """Insert a memory (see port)."""
        self._store.memories[memory.user_id][memory.id] = (memory, list(vector))

    async def replace(self, memory: Memory, vector: Sequence[float]) -> None:
        """Overwrite a memory (see port)."""
        rows = self._store.memories[memory.user_id]
        if memory.id not in rows:
            raise NotFoundError(f"memory {memory.id} not found")
        rows[memory.id] = (memory, list(vector))

    async def search(self, user_id: str, vector: Sequence[float], *, limit: int,
                     kind: MemoryKind | None = None) -> list[ScoredMemory]:
        """Most similar memories (see port)."""
        scored = [ScoredMemory(m, cosine(vector, v))
                  for m, v in self._store.memories[user_id].values()
                  if kind is None or m.kind is kind]
        scored.sort(key=lambda s: s.similarity, reverse=True)
        return scored[:max(limit, 0)]

    async def all(self, user_id: str, kind: MemoryKind | None = None) -> list[Memory]:
        """Every memory, newest first (see port)."""
        rows = [m for m, _ in self._store.memories[user_id].values()
                if kind is None or m.kind is kind]
        return sorted(rows, key=lambda m: (m.updated_at, str(m.id)), reverse=True)

    async def get(self, user_id: str, memory_id: uuid.UUID) -> Memory | None:
        """One memory (see port)."""
        row = self._store.memories[user_id].get(memory_id)
        return row[0] if row else None

    async def count(self, user_id: str) -> int:
        """Memory count (see port)."""
        return len(self._store.memories[user_id])

    async def delete(self, user_id: str, memory_id: uuid.UUID) -> bool:
        """Delete one memory (see port)."""
        return self._store.memories[user_id].pop(memory_id, None) is not None


class InMemoryConversationRepository:
    """Conversations and messages."""

    def __init__(self, store: InMemoryStore) -> None:
        """Create the repository.

        Args:
            store: Shared store.
        """
        self._store = store

    async def create(self, user_id: str) -> uuid.UUID:
        """Start a conversation (see port)."""
        conversation_id = uuid.uuid4()
        self._store.conversations[conversation_id] = user_id
        return conversation_id

    async def exists(self, user_id: str, conversation_id: uuid.UUID) -> bool:
        """Ownership check (see port)."""
        return self._store.conversations.get(conversation_id) == user_id

    async def append(self, user_id: str, conversation_id: uuid.UUID,
                     messages: Sequence[ChatMessage]) -> None:
        """Add messages (see port)."""
        if self._store.conversations.get(conversation_id) != user_id:
            raise NotFoundError(f"conversation {conversation_id} not found")
        self._store.messages[conversation_id].extend(messages)

    async def recent(self, user_id: str, conversation_id: uuid.UUID,
                     limit: int) -> list[ChatMessage]:
        """Last ``limit`` messages, oldest first (see port)."""
        if limit <= 0 or self._store.conversations.get(conversation_id) != user_id:
            return []
        return list(self._store.messages[conversation_id][-limit:])


class InMemoryHealth:
    """Always healthy (there is nothing to reach)."""

    async def ping(self) -> None:
        """Nothing to check."""
