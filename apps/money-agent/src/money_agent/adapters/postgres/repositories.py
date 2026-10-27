"""Postgres implementations of the storage ports.

Rules followed by every query here:

* parameters are always bound (``%s``), never formatted into SQL;
* every statement filters on ``user_id``, so one user can never read or
  change another's rows, even with a guessed id;
* bulk writes are one statement per batch (``unnest`` of arrays), not one
  round trip per row;
* vectors are sent as pgvector text literals (``'[0.1,0.2]'::vector``) and
  never read back, so no driver extension is needed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import date
from typing import Any

from psycopg import sql
from psycopg.rows import TupleRow

from money_agent.adapters.postgres.database import Database
from money_agent.core.errors import NotFoundError
from money_agent.domain.models import (
    Category,
    ChatMessage,
    Cursor,
    Direction,
    Memory,
    MemoryKind,
    MessageRole,
    ScoredMemory,
    Transaction,
    TransactionPage,
    TransactionQuery,
)
from money_agent.domain.money import Money

INSERT_BATCH = 1_000


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector's text form: ``[0.1,0.2,...]``.

    Args:
        vector: The vector.

    Returns:
        The literal (``repr`` keeps full float precision).
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _category(value: str) -> Category:
    try:
        return Category(value)
    except ValueError:
        return Category.OTHER  # a category removed in a later version


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #


class PostgresUserRepository:
    """Users table."""

    def __init__(self, db: Database) -> None:
        """Create the repository.

        Args:
            db: Database.
        """
        self._db = db

    async def ensure(self, user_id: str) -> None:
        """Create the user if missing (see port)."""
        async with self._db.transaction("ensure user") as conn:
            await conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                               (user_id,))


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #

_TXN_COLUMNS = sql.SQL("id, user_id, posted_on, description, amount_paise, direction, "
                       "category, fingerprint, balance_paise")


def _transaction(row: TupleRow) -> Transaction:
    return Transaction(id=row[0], user_id=row[1], posted_on=row[2], description=row[3],
                       amount=Money(row[4]), direction=Direction(row[5]),
                       category=_category(row[6]), fingerprint=row[7],
                       balance=None if row[8] is None else Money(row[8]))


class PostgresTransactionRepository:
    """Transactions table."""

    def __init__(self, db: Database) -> None:
        """Create the repository.

        Args:
            db: Database.
        """
        self._db = db

    async def add_many(self, user_id: str, transactions: Sequence[Transaction]) -> int:
        """Insert new transactions; existing fingerprints are skipped (see port).

        Raises:
            ValueError: If a transaction belongs to another user.
        """
        if any(t.user_id != user_id for t in transactions):
            raise ValueError("every transaction must belong to user_id")
        inserted = 0
        async with self._db.transaction("add transactions") as conn:
            for start in range(0, len(transactions), INSERT_BATCH):
                batch = transactions[start:start + INSERT_BATCH]
                cursor = await conn.execute(
                    sql.SQL("INSERT INTO transactions ({cols}) "
                            "SELECT * FROM unnest(%s::uuid[], %s::text[], %s::date[], "
                            "%s::text[], %s::bigint[], %s::text[], %s::text[], %s::text[], "
                            "%s::bigint[]) "
                            "ON CONFLICT (user_id, fingerprint) DO NOTHING RETURNING 1"
                            ).format(cols=_TXN_COLUMNS),
                    ([t.id for t in batch], [t.user_id for t in batch],
                     [t.posted_on for t in batch], [t.description for t in batch],
                     [t.amount.paise for t in batch], [t.direction.value for t in batch],
                     [t.category.value for t in batch], [t.fingerprint for t in batch],
                     [None if t.balance is None else t.balance.paise for t in batch]))
                inserted += len(await cursor.fetchall())
        return inserted

    async def page(self, user_id: str, query: TransactionQuery) -> TransactionPage:
        """Keyset-paginated listing, newest first (see port)."""
        where: list[sql.Composable] = [sql.SQL("user_id = %s")]
        params: list[Any] = [user_id]
        if query.start:
            where.append(sql.SQL("posted_on >= %s"))
            params.append(query.start)
        if query.end:
            where.append(sql.SQL("posted_on <= %s"))
            params.append(query.end)
        if query.category:
            where.append(sql.SQL("category = %s"))
            params.append(query.category.value)
        if query.direction:
            where.append(sql.SQL("direction = %s"))
            params.append(query.direction.value)
        if query.search:
            where.append(sql.SQL("description ILIKE %s"))
            params.append("%" + _escape_like(query.search) + "%")
        if query.after:
            where.append(sql.SQL("(posted_on, id) < (%s, %s)"))
            params.extend([query.after.posted_on, query.after.id])
        params.append(query.limit + 1)  # one extra row tells us whether a next page exists
        query_sql = sql.SQL("SELECT {cols} FROM transactions WHERE {where} "
                            "ORDER BY posted_on DESC, id DESC LIMIT %s").format(
            cols=_TXN_COLUMNS, where=sql.SQL(" AND ").join(where))
        async with self._db.transaction("list transactions") as conn:
            cursor = await conn.execute(query_sql, params)
            rows = await cursor.fetchall()
        items = tuple(_transaction(row) for row in rows[:query.limit])
        more = len(rows) > query.limit
        return TransactionPage(items=items, next_cursor=Cursor(items[-1].posted_on, items[-1].id)
                               if more else None)

    async def between(self, user_id: str, start: date, end: date) -> list[Transaction]:
        """All transactions in a date range, oldest first (see port)."""
        async with self._db.transaction("transactions between") as conn:
            cursor = await conn.execute(
                sql.SQL("SELECT {cols} FROM transactions WHERE user_id = %s "
                        "AND posted_on BETWEEN %s AND %s ORDER BY posted_on, id"
                        ).format(cols=_TXN_COLUMNS), (user_id, start, end))
            return [_transaction(row) for row in await cursor.fetchall()]

    async def date_bounds(self, user_id: str) -> tuple[date, date] | None:
        """Earliest and latest dates (see port)."""
        async with self._db.transaction("transaction dates") as conn:
            cursor = await conn.execute(
                "SELECT min(posted_on), max(posted_on) FROM transactions WHERE user_id = %s",
                (user_id,))
            row = await cursor.fetchone()
        return None if row is None or row[0] is None else (row[0], row[1])


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# --------------------------------------------------------------------------- #
# Memories
# --------------------------------------------------------------------------- #

_MEMORY_COLUMNS = sql.SQL("id, user_id, kind, content, created_at, updated_at")


def _memory(row: TupleRow) -> Memory:
    return Memory(id=row[0], user_id=row[1], kind=MemoryKind(row[2]), content=row[3],
                  created_at=row[4], updated_at=row[5])


class PostgresMemoryRepository:
    """Memories table with pgvector embeddings."""

    def __init__(self, db: Database) -> None:
        """Create the repository.

        Args:
            db: Database.
        """
        self._db = db

    async def add(self, memory: Memory, vector: Sequence[float]) -> None:
        """Insert a memory (see port)."""
        async with self._db.transaction("add memory") as conn:
            await conn.execute(
                sql.SQL("INSERT INTO memories ({cols}, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s::vector)"
                        ).format(cols=_MEMORY_COLUMNS),
                (memory.id, memory.user_id, memory.kind.value, memory.content,
                 memory.created_at, memory.updated_at, vector_literal(vector)))

    async def replace(self, memory: Memory, vector: Sequence[float]) -> None:
        """Overwrite a memory (see port)."""
        async with self._db.transaction("replace memory") as conn:
            cursor = await conn.execute(
                "UPDATE memories SET kind = %s, content = %s, updated_at = %s, "
                "embedding = %s::vector WHERE id = %s AND user_id = %s",
                (memory.kind.value, memory.content, memory.updated_at, vector_literal(vector),
                 memory.id, memory.user_id))
            if cursor.rowcount == 0:
                raise NotFoundError(f"memory {memory.id} not found")

    async def search(self, user_id: str, vector: Sequence[float], *, limit: int,
                     kind: MemoryKind | None = None) -> list[ScoredMemory]:
        """Exact cosine search over the user's memories (see port)."""
        if limit <= 0:
            return []
        params: list[Any] = [vector_literal(vector), user_id]
        kind_filter = sql.SQL("")
        if kind is not None:
            kind_filter = sql.SQL(" AND kind = %s")
            params.append(kind.value)
        params.append(limit)
        async with self._db.transaction("search memories") as conn:
            cursor = await conn.execute(
                sql.SQL("SELECT {cols}, 1 - (embedding <=> q.v) AS similarity "
                        "FROM memories, (SELECT %s::vector AS v) AS q WHERE user_id = %s"
                        "{kind} ORDER BY embedding <=> q.v LIMIT %s").format(
                    cols=_MEMORY_COLUMNS, kind=kind_filter), params)
            rows = await cursor.fetchall()
        return [ScoredMemory(_memory(row), float(row[6])) for row in rows]

    async def all(self, user_id: str, kind: MemoryKind | None = None) -> list[Memory]:
        """Every memory, newest first (see port)."""
        params: list[Any] = [user_id]
        kind_filter = sql.SQL("")
        if kind is not None:
            kind_filter = sql.SQL(" AND kind = %s")
            params.append(kind.value)
        async with self._db.transaction("list memories") as conn:
            cursor = await conn.execute(
                sql.SQL("SELECT {cols} FROM memories WHERE user_id = %s{kind} "
                        "ORDER BY updated_at DESC, id").format(cols=_MEMORY_COLUMNS,
                                                               kind=kind_filter), params)
            return [_memory(row) for row in await cursor.fetchall()]

    async def get(self, user_id: str, memory_id: uuid.UUID) -> Memory | None:
        """One memory (see port)."""
        async with self._db.transaction("get memory") as conn:
            cursor = await conn.execute(
                sql.SQL("SELECT {cols} FROM memories WHERE user_id = %s AND id = %s"
                        ).format(cols=_MEMORY_COLUMNS), (user_id, memory_id))
            row = await cursor.fetchone()
        return None if row is None else _memory(row)

    async def count(self, user_id: str) -> int:
        """Memory count (see port)."""
        async with self._db.transaction("count memories") as conn:
            cursor = await conn.execute("SELECT count(*) FROM memories WHERE user_id = %s",
                                        (user_id,))
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def delete(self, user_id: str, memory_id: uuid.UUID) -> bool:
        """Delete one memory (see port)."""
        async with self._db.transaction("delete memory") as conn:
            cursor = await conn.execute("DELETE FROM memories WHERE user_id = %s AND id = %s",
                                        (user_id, memory_id))
            return cursor.rowcount > 0


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #


class PostgresConversationRepository:
    """Conversations and messages tables."""

    def __init__(self, db: Database) -> None:
        """Create the repository.

        Args:
            db: Database.
        """
        self._db = db

    async def create(self, user_id: str) -> uuid.UUID:
        """Start a conversation (see port)."""
        conversation_id = uuid.uuid4()
        async with self._db.transaction("create conversation") as conn:
            await conn.execute("INSERT INTO conversations (id, user_id) VALUES (%s, %s)",
                               (conversation_id, user_id))
        return conversation_id

    async def exists(self, user_id: str, conversation_id: uuid.UUID) -> bool:
        """Ownership check (see port)."""
        async with self._db.transaction("conversation exists") as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                (conversation_id, user_id))
            return await cursor.fetchone() is not None

    async def append(self, user_id: str, conversation_id: uuid.UUID,
                     messages: Sequence[ChatMessage]) -> None:
        """Add messages atomically, after an ownership check (see port)."""
        async with self._db.transaction("append messages") as conn:
            cursor = await conn.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s AND user_id = %s",
                (conversation_id, user_id))
            if cursor.rowcount == 0:
                raise NotFoundError(f"conversation {conversation_id} not found")
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s)",
                    [(conversation_id, m.role.value, m.content, json.dumps(m.meta),
                      m.created_at) for m in messages])

    async def recent(self, user_id: str, conversation_id: uuid.UUID,
                     limit: int) -> list[ChatMessage]:
        """Last ``limit`` messages, oldest first (see port)."""
        if limit <= 0:
            return []
        async with self._db.transaction("recent messages") as conn:
            cursor = await conn.execute(
                "SELECT m.role, m.content, m.created_at, m.meta FROM messages m "
                "JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.conversation_id = %s AND c.user_id = %s "
                "ORDER BY m.id DESC LIMIT %s", (conversation_id, user_id, limit))
            rows = await cursor.fetchall()
        return [ChatMessage(role=MessageRole(r[0]), content=r[1], created_at=r[2], meta=r[3])
                for r in reversed(rows)]
