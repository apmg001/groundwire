"""Postgres connection pool, migrations and health.

* **Pool**: one :class:`psycopg_pool.AsyncConnectionPool` per process, sized
  by settings. Requests borrow a connection for one unit of work and return it,
  so a process holds at most ``db_pool_max`` connections however many requests
  are in flight (excess requests wait, bounded by ``POOL_TIMEOUT_S``).
* **Migrations**: plain SQL files shipped inside the package, applied in name
  order, each in its own transaction, recorded in ``schema_migrations``. A
  Postgres *advisory lock* serialises them, so ten replicas starting at once
  apply each migration exactly once.
* **Errors**: every driver error is logged and re-raised as
  :class:`StorageUnavailableError`, so no SQL detail ever reaches a user.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from importlib import resources

import psycopg
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from money_agent.core.errors import ConfigurationError, StorageUnavailableError

logger = logging.getLogger(__name__)

MIGRATIONS_PACKAGE = "money_agent.adapters.postgres.migrations"
# Arbitrary constant shared by every replica: "the migration lock".
MIGRATION_LOCK_KEY = 7_406_198_231
POOL_TIMEOUT_S = 10.0
# Sessions run in UTC whatever the server's default is; the app converts to
# the users' zone itself (Clock), so stored timestamps never depend on it.
SESSION_OPTIONS = "-c TimeZone=UTC"


@contextmanager
def storage_errors(operation: str) -> Iterator[None]:
    """Turn driver errors into :class:`StorageUnavailableError`.

    Args:
        operation: What was being done, for the log.

    Raises:
        StorageUnavailableError: On any database error.
    """
    try:
        yield
    except PoolTimeout as exc:
        logger.error("database pool exhausted", extra={"operation": operation})
        raise StorageUnavailableError(f"{operation}: pool timeout") from exc
    except psycopg.Error as exc:
        logger.error("database error", extra={"operation": operation,
                                              "sqlstate": getattr(exc, "sqlstate", None)})
        raise StorageUnavailableError(f"{operation}: {type(exc).__name__}") from exc


class Database:
    """Owns the pool; created and closed by the composition root."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 10) -> None:
        """Configure (but do not open) the pool.

        Args:
            dsn: Postgres connection string.
            min_size: Connections kept open.
            max_size: Most connections at once.
        """
        self._dsn = dsn
        self._pool = AsyncConnectionPool(dsn, min_size=min_size, max_size=max_size,
                                         timeout=POOL_TIMEOUT_S, open=False,
                                         kwargs={"autocommit": False,
                                                 "options": SESSION_OPTIONS})

    async def open(self, *, migrate: bool = True, vector_dimension: int | None = None) -> None:
        """Apply migrations (optionally), open the pool and check the schema.

        Args:
            migrate: Apply pending migrations first.
            vector_dimension: Expected size of ``memories.embedding``.

        Raises:
            StorageUnavailableError: The database is unreachable.
            ConfigurationError: The stored vector size differs from the embedder's.
        """
        if migrate:
            await self.migrate()
        with storage_errors("open pool"):
            await self._pool.open(wait=True, timeout=POOL_TIMEOUT_S)
        if vector_dimension is not None:
            await self._check_vector_dimension(vector_dimension)
        logger.info("database ready")

    async def close(self) -> None:
        """Close every connection."""
        await self._pool.close()

    @asynccontextmanager
    async def transaction(self, operation: str) -> AsyncIterator[psycopg.AsyncConnection]:
        """Borrow a connection inside a transaction (committed on success).

        Args:
            operation: Name for logs.

        Yields:
            The connection.

        Raises:
            StorageUnavailableError: On any database error.
        """
        with storage_errors(operation):
            async with self._pool.connection() as conn, conn.transaction():
                yield conn

    async def ping(self) -> None:
        """Run ``SELECT 1``.

        Raises:
            StorageUnavailableError: The database is not usable.
        """
        async with self.transaction("ping") as conn:
            await conn.execute("SELECT 1")

    # ------------------------------------------------------------------ #
    # Migrations
    # ------------------------------------------------------------------ #

    async def migrate(self) -> list[str]:
        """Apply pending migrations under the advisory lock.

        Returns:
            Names of the migrations applied now.

        Raises:
            StorageUnavailableError: The database is unreachable or a migration failed.
        """
        files = sorted((f for f in resources.files(MIGRATIONS_PACKAGE).iterdir()
                        if f.name.endswith(".sql")), key=lambda f: f.name)
        applied_now: list[str] = []
        with storage_errors("migrate"):
            async with await psycopg.AsyncConnection.connect(
                    self._dsn, autocommit=True, options=SESSION_OPTIONS) as conn:
                await conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
                try:
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS schema_migrations ("
                        " version TEXT PRIMARY KEY,"
                        " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
                    cursor = await conn.execute("SELECT version FROM schema_migrations")
                    done = {row[0] for row in await cursor.fetchall()}
                    for migration in files:
                        if migration.name in done:
                            continue
                        async with conn.transaction():
                            await conn.execute(migration.read_text(encoding="utf-8"))
                            await conn.execute(
                                "INSERT INTO schema_migrations (version) VALUES (%s)",
                                (migration.name,))
                        applied_now.append(migration.name)
                        logger.info("migration applied", extra={"migration": migration.name})
                finally:
                    await conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
        return applied_now

    async def _check_vector_dimension(self, expected: int) -> None:
        async with self.transaction("check schema") as conn:
            cursor = await conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'")
            row = await cursor.fetchone()
        if row is None or row[0] != expected:
            raise ConfigurationError(
                f"memories.embedding has size {row[0] if row else None}, but the embedder "
                f"produces {expected}: set MONEY_EMBEDDING_DIM to match, or add a migration")
