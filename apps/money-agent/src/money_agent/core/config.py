"""Settings, read from ``MONEY_*`` environment variables.

Everything that changes per deployment lives here; nothing is read from the
environment anywhere else. Secrets are :class:`SecretStr`, so they never show
up in logs or reprs. Settings are validated once at startup: a bad value stops
the process instead of failing the first request.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from money_agent.core.errors import ConfigurationError


class StorageBackend(StrEnum):
    """Where user data is kept."""

    POSTGRES = "postgres"  # production: durable, shared by every replica
    MEMORY = "memory"      # tests and quick local runs: lost on restart, one process only


class EmbedderType(StrEnum):
    """How memory text is turned into vectors."""

    HASHING = "hashing"  # dependency-free, deterministic, word-overlap similarity
    HTTP = "http"        # any OpenAI-compatible /embeddings endpoint (real semantics)


class UnverifiedAnswers(StrEnum):
    """What to do with an answer that still fails the answer check after a retry."""

    WARN = "warn"          # show it, clearly flagged as unverified
    WITHHOLD = "withhold"  # show a safe message instead


class Settings(BaseSettings):
    """All runtime settings.

    Attributes:
        storage: Storage backend.
        database_url: Postgres DSN (required for ``postgres``).
        db_pool_min: Minimum pooled connections per process.
        db_pool_max: Maximum pooled connections per process.
        run_migrations: Apply pending migrations at startup.
        gateway_url: Base URL of the Groundwire gateway.
        gateway_model: Route alias to ask Groundwire for.
        gateway_api_key: Bearer token for the gateway, if it requires one.
        gateway_timeout_s: Timeout for one gateway call.
        gateway_max_retries: Retries for transient gateway failures.
        max_answer_tokens: Cap on each model reply (reasoning included). Free tiers
            count it against tokens-per-minute, so it is kept modest.
        embedder: Embedder type.
        embedding_dim: Vector size (must match the migration's ``vector(N)``).
        embedding_url: Base URL of an OpenAI-compatible embeddings API (``http``).
        embedding_model: Embedding model name (``http``).
        embedding_api_key: Key for the embeddings API (``http``).
        agent_max_steps: Model calls allowed per user message.
        unverified_answers: Policy for answers that fail the answer check.
        history_messages: Earlier messages sent along with a new one.
        memory_recall_limit: Memories added to the prompt per message.
        max_memories_per_user: Hard cap on stored memories per user.
        max_statement_bytes: Upload size limit.
        max_statement_rows: Row limit per statement.
        timezone: Users' time zone, for "this month" and "today".
        log_level: Log level name.
        log_json: JSON logs (production) or text.
        host: Interface to bind.
        port: Port to bind.
    """

    model_config = SettingsConfigDict(env_prefix="MONEY_", extra="ignore")

    storage: StorageBackend = StorageBackend.MEMORY
    database_url: SecretStr | None = None
    db_pool_min: Annotated[int, Field(ge=1, le=100)] = 1
    db_pool_max: Annotated[int, Field(ge=1, le=200)] = 10
    run_migrations: bool = True

    gateway_url: str = "http://127.0.0.1:8000"
    gateway_model: Annotated[str, Field(min_length=1, max_length=128)] = "default"
    gateway_api_key: SecretStr | None = None
    gateway_timeout_s: Annotated[float, Field(gt=0, le=300)] = 45.0
    gateway_max_retries: Annotated[int, Field(ge=0, le=5)] = 2
    max_answer_tokens: Annotated[int, Field(ge=64, le=32_000)] = 1_200

    embedder: EmbedderType = EmbedderType.HASHING
    embedding_dim: Annotated[int, Field(ge=8, le=4096)] = 384
    embedding_url: str | None = None
    embedding_model: str | None = None
    embedding_api_key: SecretStr | None = None

    agent_max_steps: Annotated[int, Field(ge=1, le=20)] = 6
    unverified_answers: UnverifiedAnswers = UnverifiedAnswers.WARN
    history_messages: Annotated[int, Field(ge=0, le=200)] = 6
    memory_recall_limit: Annotated[int, Field(ge=0, le=50)] = 6
    max_memories_per_user: Annotated[int, Field(ge=1, le=10_000)] = 500
    max_statement_bytes: Annotated[int, Field(ge=1_024, le=50_000_000)] = 5_000_000
    max_statement_rows: Annotated[int, Field(ge=1, le=500_000)] = 50_000

    timezone: str = "Asia/Kolkata"
    log_level: str = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8100

    @field_validator("gateway_url", "embedding_url")
    @classmethod
    def _strip_slash(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/") if value else value

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown time zone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _check_combinations(self) -> Self:
        if self.storage is StorageBackend.POSTGRES and self.database_url is None:
            raise ValueError("MONEY_DATABASE_URL is required when MONEY_STORAGE=postgres")
        if self.db_pool_min > self.db_pool_max:
            raise ValueError("MONEY_DB_POOL_MIN cannot exceed MONEY_DB_POOL_MAX")
        if self.embedder is EmbedderType.HTTP and not (self.embedding_url
                                                       and self.embedding_model):
            raise ValueError("MONEY_EMBEDDING_URL and MONEY_EMBEDDING_MODEL are required "
                             "when MONEY_EMBEDDER=http")
        return self

    @property
    def zone(self) -> ZoneInfo:
        """The configured time zone."""
        return ZoneInfo(self.timezone)


def load_settings(**overrides: object) -> Settings:
    """Read and validate settings.

    Args:
        **overrides: Values that take precedence over the environment (tests).

    Returns:
        The settings.

    Raises:
        ConfigurationError: If any value is invalid.
    """
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors())
        raise ConfigurationError(f"invalid settings: {problems}") from exc
