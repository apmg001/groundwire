"""Shared fixtures: a frozen clock, a scripted model, sample data, wired apps."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from money_agent.api.app import create_app
from money_agent.container import Container, build_container
from money_agent.core.config import Settings, load_settings
from money_agent.domain.models import Category, Direction, Transaction
from money_agent.domain.money import Money
from money_agent.domain.ports import LLMReply, ToolCallRequest

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_statement.csv"
USER = "test-user-0123456789abcdef"
OTHER_USER = "other-user-0123456789abcdef"
PG_URL = os.environ.get("MONEY_TEST_DATABASE_URL")


class FrozenClock:
    """A clock stuck at a fixed instant (advance it by hand)."""

    def __init__(self, now: datetime = datetime(2026, 10, 28, 9, 30, tzinfo=UTC)) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current

    def today(self) -> date:
        return self.current.date()


def tool_call(name: str, arguments: dict[str, Any] | None = None,
              call_id: str | None = None) -> LLMReply:
    """A reply asking for one tool call."""
    return LLMReply(content=None, tool_calls=(ToolCallRequest(
        call_id or f"call_{uuid.uuid4().hex[:8]}", name, json.dumps(arguments or {})),),
        finish_reason="tool_calls")


def answer(text: str, verdict: str | None = "supported",
           issues: Sequence[str] = ()) -> LLMReply:
    """A final reply, with the gateway's answer-check verdict."""
    gateway: dict[str, Any] = {}
    if verdict is not None:
        gateway["answer_check"] = {"verdict": verdict,
                                   "issues": [{"kind": kind} for kind in issues]}
    return LLMReply(content=text, finish_reason="stop", gateway=gateway)


class ScriptedLLM:
    """Returns queued replies in order and records what it was sent."""

    def __init__(self, *replies: LLMReply) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def queue(self, *replies: LLMReply) -> None:
        self.replies.extend(replies)

    async def complete(self, messages: Sequence[dict[str, Any]], *,
                       tools: Sequence[dict[str, Any]], user: str) -> LLMReply:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": list(tools),
                           "user": user})
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)


def make_txn(day: date, description: str, rupees: str, direction: Direction = Direction.DEBIT,
             category: Category = Category.OTHER, user: str = USER) -> Transaction:
    """A transaction with a unique fingerprint."""
    return Transaction(id=uuid.uuid4(), user_id=user, posted_on=day, description=description,
                       amount=Money.of(rupees), direction=direction, category=category,
                       fingerprint=uuid.uuid4().hex)


@pytest.fixture
def sample_bytes() -> bytes:
    return SAMPLE.read_bytes()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def settings() -> Settings:
    return load_settings(storage="memory", log_json=False)


@pytest.fixture
async def container(settings: Settings, llm: ScriptedLLM,
                    clock: FrozenClock) -> AsyncIterator[Container]:
    built = build_container(settings, llm=llm, clock=clock)
    await built.start()
    yield built
    await built.close()


@pytest.fixture
def client(settings: Settings, llm: ScriptedLLM, clock: FrozenClock) -> Iterator[TestClient]:
    app = create_app(build_container(settings, llm=llm, clock=clock))
    with TestClient(app, headers={"X-User-Id": USER}) as test_client:
        yield test_client
