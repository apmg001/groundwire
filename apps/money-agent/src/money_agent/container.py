"""Composition root: the one place that knows the concrete classes.

Everything else depends on ports (protocols). This module reads
:class:`Settings`, picks adapters (Postgres or in-memory, hashing or HTTP
embedder), wires the services, and owns the lifecycle of shared resources
(the database pool, the HTTP client): :meth:`Container.start` and
:meth:`Container.close` are called once by the app's lifespan.

Tests build a container with in-memory storage and a fake LLM client through
:func:`build_container`'s overrides, with no network and no database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from money_agent.adapters.clock import SystemClock
from money_agent.adapters.groundwire import GroundwireClient
from money_agent.adapters.identity import DemoIdentity
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
from money_agent.agent.conversation import ChatService
from money_agent.agent.runner import AgentRunner
from money_agent.agent.toolkit import default_tools
from money_agent.agent.tools import ToolRegistry
from money_agent.core.config import EmbedderType, Settings, StorageBackend
from money_agent.domain.ports import (
    Clock,
    ConversationRepository,
    Embedder,
    HealthCheck,
    IdentityProvider,
    LLMClient,
    MemoryRepository,
    TransactionRepository,
    UserRepository,
)
from money_agent.memory.embeddings import HashingEmbedder, HttpEmbedder
from money_agent.memory.service import MemoryService
from money_agent.services.finance import FinanceService
from money_agent.services.statements import StatementService
from money_agent.statements.categoriser import RuleCategoriser
from money_agent.statements.parser import StatementParser

logger = logging.getLogger(__name__)


@dataclass
class Repositories:
    """The storage adapters, chosen together (one backend for all)."""

    users: UserRepository
    transactions: TransactionRepository
    memories: MemoryRepository
    conversations: ConversationRepository
    health: HealthCheck


@dataclass
class Container:
    """Wired application services and the resources they share."""

    settings: Settings
    statements: StatementService
    finance: FinanceService
    memory: MemoryService
    chat: ChatService
    identity: IdentityProvider
    health: HealthCheck
    embedder: Embedder
    http: httpx.AsyncClient
    database: Database | None = None

    async def start(self) -> None:
        """Open shared resources (migrations run here, before traffic).

        Raises:
            StorageUnavailableError: The database is unreachable.
            ConfigurationError: Schema and settings disagree.
        """
        if self.database is not None:
            await self.database.open(migrate=self.settings.run_migrations,
                                     vector_dimension=self.embedder.dimension)
        logger.info("money agent started", extra={
            "storage": self.settings.storage.value, "embedder": self.settings.embedder.value,
            "gateway": self.settings.gateway_url})

    async def close(self) -> None:
        """Release shared resources (idempotent)."""
        await self.http.aclose()
        if self.database is not None:
            await self.database.close()


def _repositories(settings: Settings) -> tuple[Repositories, Database | None]:
    if settings.storage is StorageBackend.POSTGRES:
        assert settings.database_url is not None  # guaranteed by Settings validation
        db = Database(settings.database_url.get_secret_value(), min_size=settings.db_pool_min,
                      max_size=settings.db_pool_max)
        return Repositories(PostgresUserRepository(db), PostgresTransactionRepository(db),
                            PostgresMemoryRepository(db), PostgresConversationRepository(db),
                            db), db
    store = InMemoryStore()
    return Repositories(InMemoryUserRepository(store), InMemoryTransactionRepository(store),
                        InMemoryMemoryRepository(store), InMemoryConversationRepository(store),
                        InMemoryHealth()), None


def _embedder(settings: Settings, http: httpx.AsyncClient) -> Embedder:
    if settings.embedder is EmbedderType.HTTP:
        assert settings.embedding_url and settings.embedding_model  # validated by Settings
        key = settings.embedding_api_key.get_secret_value() if settings.embedding_api_key \
            else None
        return HttpEmbedder(http, base_url=settings.embedding_url,
                            model=settings.embedding_model, dimension=settings.embedding_dim,
                            api_key=key)
    return HashingEmbedder(settings.embedding_dim)


def build_container(settings: Settings, *, llm: LLMClient | None = None,
                    clock: Clock | None = None, identity: IdentityProvider | None = None,
                    http: httpx.AsyncClient | None = None) -> Container:
    """Wire the application.

    Args:
        settings: Validated settings.
        llm: Model client override (tests); Groundwire over HTTP otherwise.
        clock: Clock override (tests).
        identity: Identity provider override; the demo header otherwise.
        http: HTTP client override (tests, to route to an in-process gateway).

    Returns:
        The container (call :meth:`Container.start` before serving).
    """
    http_client = http or httpx.AsyncClient(
        timeout=httpx.Timeout(settings.gateway_timeout_s, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))
    time_source = clock or SystemClock(settings.zone)
    repos, database = _repositories(settings)
    embedder = _embedder(settings, http_client)

    finance = FinanceService(repos.transactions, time_source)
    memory = MemoryService(repos.memories, embedder, time_source,
                           max_per_user=settings.max_memories_per_user)
    statements = StatementService(
        repos.users, repos.transactions,
        StatementParser(max_bytes=settings.max_statement_bytes,
                        max_rows=settings.max_statement_rows), RuleCategoriser())
    model = llm or GroundwireClient(
        http_client, base_url=settings.gateway_url, model=settings.gateway_model,
        api_key=settings.gateway_api_key.get_secret_value() if settings.gateway_api_key
        else None, timeout_s=settings.gateway_timeout_s,
        max_retries=settings.gateway_max_retries, max_tokens=settings.max_answer_tokens)
    runner = AgentRunner(model, ToolRegistry.of(default_tools(finance, memory)),
                         max_steps=settings.agent_max_steps,
                         unverified=settings.unverified_answers)
    chat = ChatService(repos.users, repos.conversations, memory, finance, runner, time_source,
                       history_limit=settings.history_messages,
                       recall_limit=settings.memory_recall_limit)
    return Container(settings=settings, statements=statements, finance=finance, memory=memory,
                     chat=chat, identity=identity or DemoIdentity(), health=repos.health,
                     embedder=embedder, http=http_client, database=database)
