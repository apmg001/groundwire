"""Ports: the interfaces the application needs from the outside world.

The core (statements, insights, memory, agent, services) depends only on these
protocols. Adapters implement them: Postgres and in-memory storage, the
Groundwire HTTP client, embedders. :mod:`money_agent.container` picks the
implementations, so swapping Postgres for another store, or Groundwire for
another gateway, touches one adapter and the composition root, nothing else.

Every storage method takes ``user_id`` explicitly: there is no call that can
read across users, which makes tenant isolation easy to review.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from money_agent.domain.models import (
    ChatMessage,
    Memory,
    MemoryKind,
    ScoredMemory,
    Transaction,
    TransactionPage,
    TransactionQuery,
)

# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


class UserRepository(Protocol):
    """Users are created on first use; the id comes from the identity provider."""

    async def ensure(self, user_id: str) -> None:
        """Create the user if it does not exist (idempotent).

        Args:
            user_id: The user's id.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...


class TransactionRepository(Protocol):
    """A user's transactions."""

    async def add_many(self, user_id: str, transactions: Sequence[Transaction]) -> int:
        """Store transactions, skipping ones already stored (same fingerprint).

        Args:
            user_id: Owner (every transaction must belong to them).
            transactions: Transactions to store.

        Returns:
            How many were new.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def page(self, user_id: str, query: TransactionQuery) -> TransactionPage:
        """One page of matching transactions, newest first.

        Args:
            user_id: Owner.
            query: Filters and paging.

        Returns:
            The page.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def between(self, user_id: str, start: date, end: date) -> list[Transaction]:
        """All transactions in a date range (inclusive), oldest first, for analytics.

        Args:
            user_id: Owner.
            start: First date.
            end: Last date.

        Returns:
            The transactions.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def date_bounds(self, user_id: str) -> tuple[date, date] | None:
        """Earliest and latest transaction dates, or ``None`` if there are none.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...


class MemoryRepository(Protocol):
    """A user's memories and their vectors."""

    async def add(self, memory: Memory, vector: Sequence[float]) -> None:
        """Store a new memory.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def replace(self, memory: Memory, vector: Sequence[float]) -> None:
        """Overwrite an existing memory's content, vector and ``updated_at``.

        Raises:
            NotFoundError: No such memory for this user.
            StorageUnavailableError: The store failed.
        """
        ...

    async def search(self, user_id: str, vector: Sequence[float], *, limit: int,
                     kind: MemoryKind | None = None) -> list[ScoredMemory]:
        """The user's memories most similar to ``vector`` (cosine), best first.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def all(self, user_id: str, kind: MemoryKind | None = None) -> list[Memory]:
        """All of the user's memories, newest first.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def get(self, user_id: str, memory_id: uuid.UUID) -> Memory | None:
        """One of the user's memories, or ``None``.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def count(self, user_id: str) -> int:
        """How many memories the user has.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def delete(self, user_id: str, memory_id: uuid.UUID) -> bool:
        """Delete one memory.

        Returns:
            True if it existed.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...


class ConversationRepository(Protocol):
    """Chat history (user and assistant text only; tool traffic is not kept)."""

    async def create(self, user_id: str) -> uuid.UUID:
        """Start a conversation.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def exists(self, user_id: str, conversation_id: uuid.UUID) -> bool:
        """True if the conversation exists and belongs to the user.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...

    async def append(self, user_id: str, conversation_id: uuid.UUID,
                     messages: Sequence[ChatMessage]) -> None:
        """Add messages in order.

        Raises:
            NotFoundError: No such conversation for this user.
            StorageUnavailableError: The store failed.
        """
        ...

    async def recent(self, user_id: str, conversation_id: uuid.UUID,
                     limit: int) -> list[ChatMessage]:
        """The last ``limit`` messages, oldest first.

        Raises:
            StorageUnavailableError: The store failed.
        """
        ...


@runtime_checkable
class HealthCheck(Protocol):
    """Something the readiness probe can ping."""

    async def ping(self) -> None:
        """Raise if the dependency is not usable.

        Raises:
            StorageUnavailableError: The dependency is down.
        """
        ...


# --------------------------------------------------------------------------- #
# AI
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    """Turns text into vectors for memory search."""

    @property
    def dimension(self) -> int:
        """Vector size."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts (unit-length vectors, one per text, same order).

        Raises:
            AssistantUnavailableError: A remote embedder failed.
        """
        ...


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """A tool call the model asked for.

    Attributes:
        id: Call id (echoed back with the result).
        name: Tool name.
        arguments: Raw JSON arguments, as the model wrote them.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class LLMReply:
    """One model turn, as returned through the gateway.

    Attributes:
        content: Text (``None`` when the turn is only tool calls).
        tool_calls: Calls the model wants run.
        finish_reason: ``stop``, ``tool_calls``, ``length``...
        gateway: The gateway's ``groundwire`` metadata (answer check, tools, safety).
    """

    content: str | None
    tool_calls: tuple[ToolCallRequest, ...] = ()
    finish_reason: str | None = None
    gateway: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Chat completions with tools, OpenAI-style."""

    async def complete(self, messages: Sequence[dict[str, Any]], *,
                       tools: Sequence[dict[str, Any]], user: str) -> LLMReply:
        """Ask the model for its next turn.

        Args:
            messages: OpenAI-format conversation.
            tools: OpenAI-format tool definitions.
            user: Opaque end-user id (lets the gateway isolate users' cache entries).

        Returns:
            The reply.

        Raises:
            AssistantUnavailableError: The gateway or every model failed.
            MessageRefusedError: The gateway refused the message.
        """
        ...


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


class IdentityProvider(Protocol):
    """Resolves the caller of a request to a user id.

    Framework-free on purpose (it receives headers, not a web framework's
    request object), so the composition root and the services never depend on
    the HTTP layer. Production: verify a signed token (OIDC/JWT) and return its
    subject.
    """

    async def identify(self, headers: Mapping[str, str]) -> str:
        """Return the caller's user id.

        Args:
            headers: Request headers (case-insensitive mapping).

        Returns:
            A stable user id.

        Raises:
            AuthenticationError: The caller could not be identified.
        """
        ...


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


class Clock(Protocol):
    """The current time (injected, so tests can freeze it)."""

    def now(self) -> datetime:
        """Timezone-aware current time."""
        ...

    def today(self) -> date:
        """Today's date in the users' time zone."""
        ...
