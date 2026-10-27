"""One storage contract, two implementations.

Every test runs against the in-memory adapters and, when
``MONEY_TEST_DATABASE_URL`` points at a Postgres with pgvector, against the
Postgres adapters too. Passing both proves the test double used everywhere
else behaves like production storage.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta

import pytest

from conftest import OTHER_USER, PG_URL, USER, make_txn
from money_agent.adapters.inmemory import (
    InMemoryConversationRepository,
    InMemoryHealth,
    InMemoryMemoryRepository,
    InMemoryStore,
    InMemoryTransactionRepository,
    InMemoryUserRepository,
)
from money_agent.adapters.postgres.database import Database
from money_agent.adapters.postgres.repositories import (
    PostgresConversationRepository,
    PostgresMemoryRepository,
    PostgresTransactionRepository,
    PostgresUserRepository,
)
from money_agent.core.errors import NotFoundError
from money_agent.domain.models import (
    Category,
    ChatMessage,
    Direction,
    Memory,
    MemoryKind,
    MessageRole,
    TransactionQuery,
)
from money_agent.domain.ports import (
    ConversationRepository,
    HealthCheck,
    MemoryRepository,
    TransactionRepository,
    UserRepository,
)
from money_agent.memory.embeddings import HashingEmbedder

NOW = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
TABLES = "messages, conversations, memories, transactions, users"


@dataclass
class Repos:
    users: UserRepository
    transactions: TransactionRepository
    memories: MemoryRepository
    conversations: ConversationRepository
    health: HealthCheck


BACKENDS = ["memory", pytest.param("postgres", marks=[
    pytest.mark.postgres,
    pytest.mark.skipif(PG_URL is None, reason="MONEY_TEST_DATABASE_URL not set")])]


@pytest.fixture(params=BACKENDS)
async def repos(request: pytest.FixtureRequest) -> AsyncIterator[Repos]:
    if request.param == "memory":
        store = InMemoryStore()
        yield Repos(InMemoryUserRepository(store), InMemoryTransactionRepository(store),
                    InMemoryMemoryRepository(store), InMemoryConversationRepository(store),
                    InMemoryHealth())
        return
    assert PG_URL is not None
    db = Database(PG_URL, min_size=1, max_size=4)
    await db.open(migrate=True, vector_dimension=384)
    async with db.transaction("reset") as conn:
        await conn.execute(f"TRUNCATE {TABLES} CASCADE")
    try:
        yield Repos(PostgresUserRepository(db), PostgresTransactionRepository(db),
                    PostgresMemoryRepository(db), PostgresConversationRepository(db), db)
    finally:
        await db.close()


async def _users(repos: Repos) -> None:
    await repos.users.ensure(USER)
    await repos.users.ensure(USER)  # idempotent
    await repos.users.ensure(OTHER_USER)


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #


async def test_add_many_is_idempotent_per_user(repos: Repos) -> None:
    await _users(repos)
    rows = [make_txn(date(2026, 9, d), f"SHOP {d}", "100") for d in range(1, 6)]
    assert await repos.transactions.add_many(USER, rows) == 5
    assert await repos.transactions.add_many(USER, rows) == 0
    theirs = [replace(t, id=uuid.uuid4(), user_id=OTHER_USER) for t in rows]
    assert await repos.transactions.add_many(OTHER_USER, theirs) == 5  # same fingerprints
    with pytest.raises(ValueError):
        await repos.transactions.add_many(USER, theirs)
    assert await repos.transactions.add_many(USER, []) == 0


async def test_pages_are_newest_first_and_never_skip_or_repeat(repos: Repos) -> None:
    await _users(repos)
    rows = [make_txn(date(2026, 9, 1 + i // 3), f"ROW {i}", str(10 + i)) for i in range(10)]
    await repos.transactions.add_many(USER, rows)
    seen, cursor = [], None
    while True:
        page = await repos.transactions.page(USER, TransactionQuery(limit=3, after=cursor))
        seen.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert len(seen) == 10 and len({t.id for t in seen}) == 10
    keys = [(t.posted_on, t.id) for t in seen]
    assert keys == sorted(keys, reverse=True)


async def test_filters_combine(repos: Repos) -> None:
    await _users(repos)
    await repos.transactions.add_many(USER, [
        make_txn(date(2026, 9, 1), "SWIGGY ORDER", "300", category=Category.FOOD_DINING),
        make_txn(date(2026, 9, 2), "Swiggy Instamart", "800", category=Category.GROCERIES),
        make_txn(date(2026, 9, 3), "SALARY", "50000", Direction.CREDIT, Category.SALARY),
        make_txn(date(2026, 8, 30), "SWIGGY ORDER", "250", category=Category.FOOD_DINING),
        make_txn(date(2026, 9, 4), "100%_CASHBACK", "5", Direction.CREDIT)])

    async def descriptions(**kwargs: object) -> list[str]:
        page = await repos.transactions.page(USER, TransactionQuery(**kwargs))  # type: ignore[arg-type]
        return [t.description for t in page.items]

    assert await descriptions(search="swiggy") == ["Swiggy Instamart", "SWIGGY ORDER",
                                                   "SWIGGY ORDER"]
    assert await descriptions(search="swiggy", start=date(2026, 9, 1),
                              category=Category.FOOD_DINING) == ["SWIGGY ORDER"]
    assert await descriptions(direction=Direction.CREDIT, end=date(2026, 9, 3)) == ["SALARY"]
    assert await descriptions(search="100%_") == ["100%_CASHBACK"]  # LIKE wildcards are literal
    assert await descriptions(search="%") == ["100%_CASHBACK"]
    assert await descriptions(search="_") == ["100%_CASHBACK"]


async def test_range_queries_and_bounds(repos: Repos) -> None:
    await _users(repos)
    assert await repos.transactions.date_bounds(USER) is None
    rows = [make_txn(date(2026, 9, d), "X", "1") for d in (3, 1, 2)]
    await repos.transactions.add_many(USER, rows)
    between = await repos.transactions.between(USER, date(2026, 9, 1), date(2026, 9, 2))
    assert [t.posted_on.day for t in between] == [1, 2]  # oldest first, end inclusive
    assert await repos.transactions.date_bounds(USER) == (date(2026, 9, 1), date(2026, 9, 3))
    assert await repos.transactions.between(OTHER_USER, date(2026, 1, 1),
                                            date(2026, 12, 31)) == []
    [stored] = await repos.transactions.between(USER, date(2026, 9, 3), date(2026, 9, 3))
    assert stored == rows[0]  # every field round-trips


# --------------------------------------------------------------------------- #
# Memories
# --------------------------------------------------------------------------- #


async def _memory(repos: Repos, user: str, kind: MemoryKind, text: str,
                  at: datetime = NOW) -> Memory:
    memory = Memory(uuid.uuid4(), user, kind, text, at, at)
    [vector] = await HashingEmbedder().embed([text])
    await repos.memories.add(memory, vector)
    return memory


async def test_memory_search_is_ranked_filtered_and_isolated(repos: Repos) -> None:
    await _users(repos)
    salary = await _memory(repos, USER, MemoryKind.FACT, "Salary is credited on the 1st")
    await _memory(repos, USER, MemoryKind.FACT, "Has two cats")
    await _memory(repos, USER, MemoryKind.GOAL, "Salary hike goal next year")
    await _memory(repos, OTHER_USER, MemoryKind.FACT, "Salary is credited on the 1st")
    [query] = await HashingEmbedder().embed(["when is my salary credited"])
    hits = await repos.memories.search(USER, query, limit=2, kind=MemoryKind.FACT)
    assert hits[0].memory.id == salary.id
    assert hits[0].similarity > hits[1].similarity
    assert all(h.memory.user_id == USER for h in await repos.memories.search(USER, query,
                                                                             limit=10))
    assert len(await repos.memories.search(USER, query, limit=10)) == 3
    assert await repos.memories.search(USER, query, limit=0) == []


async def test_memory_crud(repos: Repos) -> None:
    await _users(repos)
    old = await _memory(repos, USER, MemoryKind.GOAL, "Bike by March")
    newer = await _memory(repos, USER, MemoryKind.FACT, "Lives in Pune", NOW + timedelta(1))
    assert [m.id for m in await repos.memories.all(USER)] == [newer.id, old.id]
    assert [m.id for m in await repos.memories.all(USER, MemoryKind.GOAL)] == [old.id]
    assert await repos.memories.get(USER, old.id) == old
    assert await repos.memories.get(OTHER_USER, old.id) is None
    assert await repos.memories.count(USER) == 2
    moved = replace(old, content="Bike by June", updated_at=NOW + timedelta(2))
    [vector] = await HashingEmbedder().embed(["Bike by June"])
    await repos.memories.replace(moved, vector)
    assert (await repos.memories.all(USER))[0].content == "Bike by June"
    with pytest.raises(NotFoundError):
        await repos.memories.replace(replace(moved, user_id=OTHER_USER), vector)
    assert not await repos.memories.delete(OTHER_USER, old.id)  # not theirs
    assert await repos.memories.delete(USER, old.id)
    assert not await repos.memories.delete(USER, old.id)
    assert await repos.memories.count(USER) == 1


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #


async def test_conversations_keep_order_and_owner(repos: Repos) -> None:
    await _users(repos)
    conversation = await repos.conversations.create(USER)
    assert await repos.conversations.exists(USER, conversation)
    assert not await repos.conversations.exists(OTHER_USER, conversation)
    messages = [ChatMessage(MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                            f"message {i}", NOW + timedelta(seconds=i),
                            {"n": i} if i % 2 else {}) for i in range(5)]
    await repos.conversations.append(USER, conversation, messages[:3])
    await repos.conversations.append(USER, conversation, messages[3:])
    recent = await repos.conversations.recent(USER, conversation, 3)
    assert [m.content for m in recent] == ["message 2", "message 3", "message 4"]
    assert recent[1].meta == {"n": 3}
    assert await repos.conversations.recent(OTHER_USER, conversation, 10) == []
    assert await repos.conversations.recent(USER, conversation, 0) == []
    with pytest.raises(NotFoundError):
        await repos.conversations.append(OTHER_USER, conversation, messages)
    with pytest.raises(NotFoundError):
        await repos.conversations.append(USER, uuid.uuid4(), messages)


async def test_health(repos: Repos) -> None:
    await repos.health.ping()
