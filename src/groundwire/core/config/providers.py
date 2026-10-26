"""Upstream provider configuration and API-key resolution.

Secrets never live in YAML: a provider names an environment variable
(``api_key_env``) and :func:`resolve_api_key` reads it at startup.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    Field,
    SecretStr,
    model_validator,
)

from groundwire.core.config.base import StrictModel
from groundwire.core.errors import ConfigurationError


class ProviderType(StrEnum):
    """Supported provider adapter types."""

    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI, llama.cpp server, Ollama, vLLM, Groq...
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # deterministic fake for tests, load and chaos testing


class MockToolCall(StrictModel):
    """A tool call a ``mock`` provider makes when the request offers that tool.

    Attributes:
        name: Tool (function) name.
        arguments: Arguments object.
    """

    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    arguments: dict[str, Any] = Field(default_factory=dict)


class MockBehaviour(StrictModel):
    """Behaviour of a ``mock`` provider.

    Attributes:
        response_text: Text the mock returns.
        latency_s: Artificial latency per call.
        failure_rate: Probability in [0, 1] that a call fails with a server error.
        seed: Seed for the failure RNG, for reproducible chaos tests.
        tool_call: Call this tool when it is offered and the last message is not
            already a tool result (so an agent loop ends with ``response_text``).
    """

    response_text: str = "This is a mock response from Groundwire."
    latency_s: Annotated[float, Field(ge=0, le=60)] = 0.0
    failure_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    seed: int | None = None
    tool_call: MockToolCall | None = None


class ProviderConfig(StrictModel):
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
        skip_if_no_key: Start without this provider (warning logged) when its
            ``api_key_env`` variable is unset, instead of refusing to start.
            For optional providers in front of a fallback (a public demo
            whose cloud key may be absent); routes must keep another provider.
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
    skip_if_no_key: bool = False
    default_max_tokens: Annotated[int, Field(ge=1, le=200_000)] = 1024
    mock: MockBehaviour = MockBehaviour()

    @model_validator(mode="after")
    def _check_base_url(self) -> Self:
        if self.type is not ProviderType.MOCK and not self.base_url:
            raise ValueError(f"provider '{self.name}': base_url is required for type '{self.type}'")
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"provider '{self.name}': base_url must start with http:// or https://")
        return self


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
