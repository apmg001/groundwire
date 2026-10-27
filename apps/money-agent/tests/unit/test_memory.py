"""Long-term memory: embedder properties and remember/recall/forget policy."""

from __future__ import annotations

import math
import uuid
from datetime import timedelta

import pytest

from conftest import USER, FrozenClock
from money_agent.adapters.inmemory import InMemoryMemoryRepository, InMemoryStore
from money_agent.core.errors import InvalidInputError, LimitReachedError, NotFoundError
from money_agent.domain.models import MemoryKind
from money_agent.memory.embeddings import HashingEmbedder, cosine, tokens
from money_agent.memory.service import MAX_ALWAYS_RECALLED, MemoryService

GOAL, FACT, PREF = MemoryKind.GOAL, MemoryKind.FACT, MemoryKind.PREFERENCE


async def test_hashing_embedder_is_deterministic_unit_length_and_word_based() -> None:
    embedder = HashingEmbedder(64)
    [a, b, c, empty] = await embedder.embed(["Saving for a bike", "saving for bikes",
                                             "Monthly rent split", "the and of"])
    assert a == b  # stemming: saving/bikes meet saving/bike
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0)
    assert cosine(a, c) < 0.3
    assert not any(empty)  # stop words only: nothing to search for
    assert embedder.dimension == 64
    assert tokens("My salary comes on the 1st") == ["salary", "come", "1st"]
    assert tokens("EMIs subsidies loss savings") == ["emi", "subsidy", "loss", "sav"]
    with pytest.raises(ValueError):
        HashingEmbedder(4)
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])


def _service(clock: FrozenClock, **kwargs: int | float) -> MemoryService:
    return MemoryService(InMemoryMemoryRepository(InMemoryStore()), HashingEmbedder(), clock,
                         **kwargs)  # type: ignore[arg-type]


async def test_remember_creates_then_replaces_a_near_duplicate(clock: FrozenClock) -> None:
    memory = _service(clock)
    first = await memory.remember(USER, GOAL, "Saving for a bike, ₹1,20,000 by March 2027")
    clock.current += timedelta(days=3)
    second = await memory.remember(USER, GOAL, "Saving for a bike, ₹1,20,000 by April 2027")
    assert first.created and not second.created
    assert second.memory.id == first.memory.id
    assert second.replaced == "Saving for a bike, ₹1,20,000 by March 2027"
    assert second.memory.updated_at > second.memory.created_at
    assert len(await memory.all(USER)) == 1


async def test_different_goals_are_kept_apart(clock: FrozenClock) -> None:
    memory = _service(clock)
    await memory.remember(USER, GOAL, "Saving for a bike by March")
    await memory.remember(USER, GOAL, "Saving for a laptop by March")
    await memory.remember(USER, FACT, "Saving for a bike by March")  # other kind: separate
    assert len(await memory.all(USER)) == 3


async def test_explicit_replacement_by_id(clock: FrozenClock) -> None:
    memory = _service(clock)
    old = await memory.remember(USER, GOAL, "Saving for a bike by March")
    new = await memory.remember(USER, GOAL, "Bike goal moved: ₹1.5 lakh by June",
                                replaces=old.memory.id)
    assert not new.created and new.memory.id == old.memory.id
    with pytest.raises(NotFoundError):
        await memory.remember(USER, GOAL, "x y z", replaces=uuid.uuid4())


async def test_limits_and_validation(clock: FrozenClock) -> None:
    memory = _service(clock, max_per_user=2)
    await memory.remember(USER, FACT, "Lives in Pune")
    await memory.remember(USER, FACT, "Works in software")
    with pytest.raises(LimitReachedError):
        await memory.remember(USER, FACT, "Has a cat")
    for bad in ("", "   ", "x" * 501):
        with pytest.raises(InvalidInputError):
            await memory.remember(USER, FACT, bad)
    with pytest.raises(ValueError):
        _service(clock, duplicate_similarity=1.5)


async def test_recall_always_brings_goals_and_preferences_plus_relevant_facts(
        clock: FrozenClock) -> None:
    memory = _service(clock)
    await memory.remember(USER, GOAL, "Saving for a bike by March")
    await memory.remember(USER, PREF, "Prefers short answers")
    await memory.remember(USER, FACT, "Salary is credited on the 1st")
    await memory.remember(USER, FACT, "Has two cats")
    recalled = [m.content for m in await memory.recall(USER, "when is my salary credited?")]
    assert "Saving for a bike by March" in recalled and "Prefers short answers" in recalled
    assert "Salary is credited on the 1st" in recalled
    assert "Has two cats" not in recalled
    standing = await memory.recall(USER, "the and of")  # nothing to search for
    assert {m.kind for m in standing} == {GOAL, PREF}
    assert len(await memory.recall(USER, "salary", limit=0)) == 2


async def test_recall_caps_standing_memories(clock: FrozenClock) -> None:
    memory = _service(clock)
    for i in range(MAX_ALWAYS_RECALLED + 3):
        clock.current += timedelta(minutes=1)
        await memory.remember(USER, GOAL, f"Goal number {i} for item{i}")
    recalled = await memory.recall(USER, "anything", limit=0)
    assert len(recalled) == MAX_ALWAYS_RECALLED
    assert recalled[0].content.startswith(f"Goal number {MAX_ALWAYS_RECALLED + 2}")  # newest


async def test_forget_and_isolation(clock: FrozenClock) -> None:
    memory = _service(clock)
    stored = await memory.remember(USER, FACT, "Lives in Pune")
    with pytest.raises(NotFoundError):
        await memory.forget("another-user-1234567890", stored.memory.id)  # not theirs
    await memory.forget(USER, stored.memory.id)
    with pytest.raises(NotFoundError):
        await memory.forget(USER, stored.memory.id)
    assert await memory.all(USER) == []
