"""Configuration loading and validation.

Configuration is split in two on purpose:

* :class:`Settings` comes from environment variables (``GROUNDWIRE_*``).
  It holds process-level knobs: bind address, log level, and the path to the
  gateway config file. This is what you change per deployment/container.

* :class:`GatewayConfig` comes from a YAML file. It describes *behaviour*:
  providers, routes (fallback chains), resilience policy and guardrails. It is
  versioned with the code and reviewed like code.

Secrets never live in YAML. A provider declares ``api_key_env: OPENAI_API_KEY``
and the key is read from the environment at startup. Missing keys for enabled
providers fail the boot instead of failing the first request.

All validation happens at load time so a bad config never reaches traffic.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from groundwire.core.errors import ConfigurationError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Process settings (environment)
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Process-level settings read from ``GROUNDWIRE_*`` environment variables.

    Attributes:
        config_path: Path to the gateway YAML config.
        host: Interface to bind.
        port: Port to bind.
        log_level: Root log level name.
        log_json: Emit JSON logs (production) instead of human-readable text.
        workers: Uvicorn worker processes.
    """

    model_config = SettingsConfigDict(env_prefix="GROUNDWIRE_", extra="ignore")

    config_path: Path = Path("config/groundwire.yaml")
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    log_level: str = "INFO"
    log_json: bool = True
    workers: Annotated[int, Field(ge=1, le=64)] = 1

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in logging.getLevelNamesMapping():
            raise ValueError(f"unknown log level '{value}'")
        return level


# --------------------------------------------------------------------------- #
# Gateway config (YAML)
# --------------------------------------------------------------------------- #


class _StrictModel(BaseModel):
    """Base for config models: unknown keys are errors (catches typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderType(StrEnum):
    """Supported provider adapter types."""

    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI, llama.cpp server, Ollama, vLLM, Groq...
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # deterministic fake for tests, load and chaos testing


class MockBehaviour(_StrictModel):
    """Behaviour of a ``mock`` provider.

    Attributes:
        response_text: Text the mock returns.
        latency_s: Artificial latency per call.
        failure_rate: Probability in [0, 1] that a call fails with a server error.
        seed: Seed for the failure RNG, for reproducible chaos tests.
    """

    response_text: str = "This is a mock response from Groundwire."
    latency_s: Annotated[float, Field(ge=0, le=60)] = 0.0
    failure_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    seed: int | None = None


class ProviderConfig(_StrictModel):
    """One upstream provider.

    Attributes:
        name: Unique provider name used in routes, logs and metrics.
        type: Adapter type.
        model: Upstream model identifier sent to the provider.
        base_url: Provider base URL (required for HTTP providers).
        api_key_env: Name of the env var holding the API key, if any.
        timeout_s: Per-call timeout (connect + read), in seconds.
        max_connections: Connection-pool size for this provider.
        enabled: Disabled providers are skipped and need no API key.
        default_max_tokens: Used when the client sends no ``max_tokens``
            (Anthropic requires the field).
        mock: Behaviour for ``mock`` providers.
    """

    name: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.\-]+$")]
    type: ProviderType
    model: Annotated[str, Field(min_length=1)]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: Annotated[float, Field(gt=0, le=600)] = 60.0
    max_connections: Annotated[int, Field(ge=1, le=1000)] = 50
    enabled: bool = True
    default_max_tokens: Annotated[int, Field(ge=1, le=200_000)] = 1024
    mock: MockBehaviour = MockBehaviour()

    @model_validator(mode="after")
    def _check_base_url(self) -> Self:
        if self.type is not ProviderType.MOCK and not self.base_url:
            raise ValueError(f"provider '{self.name}': base_url is required for type '{self.type}'")
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"provider '{self.name}': base_url must start with http:// or https://")
        return self


class RetryConfig(_StrictModel):
    """Retry policy applied per provider before failing over.

    Attributes:
        max_attempts: Total attempts per provider (1 disables retries).
        base_delay_s: Initial back-off delay.
        max_delay_s: Cap on any single back-off delay.
    """

    max_attempts: Annotated[int, Field(ge=1, le=10)] = 2
    base_delay_s: Annotated[float, Field(ge=0, le=30)] = 0.2
    max_delay_s: Annotated[float, Field(ge=0, le=120)] = 2.0


class CircuitBreakerConfig(_StrictModel):
    """Circuit-breaker policy applied per provider.

    Attributes:
        failure_threshold: Consecutive failures that open the circuit.
        recovery_timeout_s: Time the circuit stays open before a probe.
        half_open_max_calls: Concurrent probe calls allowed while half-open.
    """

    failure_threshold: Annotated[int, Field(ge=1, le=1000)] = 5
    recovery_timeout_s: Annotated[float, Field(gt=0, le=3600)] = 30.0
    half_open_max_calls: Annotated[int, Field(ge=1, le=100)] = 1


class GroundingScorerType(StrEnum):
    """Available grounding scorers."""

    LEXICAL = "lexical"  # fast, dependency-free; checks content words and numbers
    NLI = "nli"  # transformer NLI model; needs the `nli` extra
    HYBRID = "hybrid"  # numeric gate + NLI; needs the `nli` extra


class GroundingFailurePolicy(StrEnum):
    """What to do when the grounding check itself cannot run."""

    FAIL_OPEN = "fail_open"  # return the answer, marked as unverified
    FAIL_CLOSED = "fail_closed"  # refuse the request with 503


class AnswerAggregation(StrEnum):
    """How per-claim support scores combine into one verdict for the answer."""

    RATIO = "ratio"  # pass if >= min_supported_ratio of claims meet support_threshold
    MIN = "min"  # score = weakest claim
    LOWEST_TWO = "lowest_two"  # score = mean of the two weakest claims
    MEAN = "mean"  # score = mean of all claims


class GroundingConfig(_StrictModel):
    """Grounding-verification settings.

    Attributes:
        scorer: Which scorer to use.
        nli_model: Hugging Face model id for the NLI scorer.
        support_threshold: Per-claim score needed to count as supported.
        min_supported_ratio: Fraction of claims that must be supported for
            the answer to pass.
        min_claim_tokens: Sentences with fewer content tokens are not
            treated as factual claims (e.g. "Sure!").
        aggregation: How claim scores become an answer verdict. ``ratio`` uses
            ``min_supported_ratio``; the score-based modes compare the
            aggregated score with ``support_threshold``.
        evidence_passages: 0 scores each claim against every passage
            separately (best one wins). ``k > 0`` first picks the ``k`` passages
            sharing the most words with the claim and scores the claim once
            against them joined together: far fewer model calls, and claims
            that combine facts from several passages can be supported.
        on_error: Policy when the scorer fails.
    """

    scorer: GroundingScorerType = GroundingScorerType.LEXICAL
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    support_threshold: Annotated[float, Field(ge=0, le=1)] = 0.6
    min_supported_ratio: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_claim_tokens: Annotated[int, Field(ge=1, le=50)] = 3
    aggregation: AnswerAggregation = AnswerAggregation.RATIO
    evidence_passages: Annotated[int, Field(ge=0, le=10)] = 0
    on_error: GroundingFailurePolicy = GroundingFailurePolicy.FAIL_OPEN


class GatewayConfig(_StrictModel):
    """Root of the YAML gateway configuration.

    Attributes:
        providers: All upstream providers.
        routes: Route alias -> ordered fallback chain of provider names.
            Clients select a route via the OpenAI ``model`` field.
        default_route: Route used when a client sends ``model: "default"``
            or the alias is omitted. Must exist in ``routes``.
        request_timeout_s: End-to-end deadline across retries and failover.
        retry: Retry policy.
        circuit_breaker: Circuit-breaker policy.
        grounding: Grounding-verification settings.
    """

    providers: Annotated[list[ProviderConfig], Field(min_length=1)]
    routes: Annotated[dict[str, list[str]], Field(min_length=1)]
    default_route: str = "default"
    request_timeout_s: Annotated[float, Field(gt=0, le=1800)] = 120.0
    retry: RetryConfig = RetryConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    grounding: GroundingConfig = GroundingConfig()

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        names = [p.name for p in self.providers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate provider names: {duplicates}")

        known = set(names)
        enabled = {p.name for p in self.providers if p.enabled}
        for route, chain in self.routes.items():
            if not chain:
                raise ValueError(f"route '{route}' has an empty provider chain")
            unknown = [p for p in chain if p not in known]
            if unknown:
                raise ValueError(f"route '{route}' references unknown providers: {unknown}")
            if len(set(chain)) != len(chain):
                raise ValueError(f"route '{route}' lists a provider more than once")
            if not any(p in enabled for p in chain):
                raise ValueError(f"route '{route}' has no enabled providers")

        if self.default_route not in self.routes:
            raise ValueError(f"default_route '{self.default_route}' is not defined in routes")
        return self

    def enabled_providers(self) -> list[ProviderConfig]:
        """Return only the providers marked ``enabled``."""
        return [p for p in self.providers if p.enabled]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_gateway_config(path: Path) -> GatewayConfig:
    """Load and validate the gateway YAML configuration.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If the file is missing, unreadable, not valid YAML,
            or fails schema/cross-reference validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc}") from exc

    try:
        raw: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"config root in {path} must be a mapping")

    return parse_gateway_config(raw, source=str(path))


def parse_gateway_config(raw: dict[str, Any], *, source: str = "<dict>") -> GatewayConfig:
    """Validate an already-parsed config mapping.

    Args:
        raw: Parsed configuration mapping.
        source: Label used in error messages.

    Returns:
        The validated configuration.

    Raises:
        ConfigurationError: If validation fails.
    """
    try:
        config = GatewayConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(f"invalid gateway config ({source}): {problems}") from exc

    logger.info(
        "gateway config loaded",
        extra={
            "source": source,
            "providers": [p.name for p in config.enabled_providers()],
            "routes": list(config.routes),
        },
    )
    return config


def resolve_api_key(provider: ProviderConfig) -> SecretStr | None:
    """Read a provider's API key from the environment.

    Args:
        provider: Provider whose ``api_key_env`` should be resolved.

    Returns:
        The key wrapped in :class:`SecretStr` (so it never appears in logs or
        reprs), or ``None`` when the provider needs no key.

    Raises:
        ConfigurationError: If ``api_key_env`` is set but the variable is
            missing or empty.
    """
    if not provider.api_key_env:
        return None
    value = os.environ.get(provider.api_key_env, "").strip()
    if not value:
        raise ConfigurationError(
            f"provider '{provider.name}' needs env var '{provider.api_key_env}', "
            "which is not set (or set enabled: false to skip this provider)"
        )
    return SecretStr(value)
