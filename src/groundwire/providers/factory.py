"""Build provider adapters from configuration (factory pattern).

This is the only module that knows the mapping from ``ProviderType`` to an
adapter class, so adding a vendor touches exactly one ``match`` arm here.
"""

from __future__ import annotations

import logging
import os
from typing import assert_never

from groundwire.core.config import GatewayConfig, ProviderConfig, ProviderType, resolve_api_key
from groundwire.core.errors import ConfigurationError
from groundwire.providers.anthropic import AnthropicProvider
from groundwire.providers.base import Provider
from groundwire.providers.mock import MockProvider
from groundwire.providers.openai_compat import OpenAICompatibleProvider

logger = logging.getLogger(__name__)


def build_provider(config: ProviderConfig) -> Provider:
    """Instantiate one provider adapter.

    Args:
        config: Provider configuration.

    Returns:
        The adapter.

    Raises:
        ConfigurationError: If a required API key is missing or the config is
            inconsistent with the adapter type.
    """
    api_key = resolve_api_key(config)
    match config.type:
        case ProviderType.MOCK:
            return MockProvider(name=config.name, model=config.model, behaviour=config.mock)
        case ProviderType.OPENAI_COMPATIBLE | ProviderType.ANTHROPIC:
            if config.base_url is None:  # guarded by config validation; defensive
                raise ConfigurationError(f"provider '{config.name}' has no base_url")
            if config.type is ProviderType.ANTHROPIC:
                return AnthropicProvider(
                    name=config.name, model=config.model, base_url=config.base_url,
                    api_key=api_key, timeout_s=config.timeout_s,
                    max_connections=config.max_connections,
                    default_max_tokens=config.default_max_tokens,
                )
            return OpenAICompatibleProvider(
                name=config.name, model=config.model, base_url=config.base_url,
                api_key=api_key, timeout_s=config.timeout_s,
                max_connections=config.max_connections,
            )
        case _:
            assert_never(config.type)


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    """Instantiate all enabled providers.

    Args:
        config: Gateway configuration.

    Returns:
        Mapping of provider name to adapter.

    Raises:
        ConfigurationError: If any provider fails to build. HTTP clients are
            created lazily-cheap (no sockets are opened until first use), so a
            partial build holds no network resources worth unwinding.
    """
    providers: dict[str, Provider] = {}
    for provider_config in config.enabled_providers():
        if _key_missing(provider_config) and provider_config.skip_if_no_key:
            logger.warning("provider skipped: API key not set", extra={
                "provider": provider_config.name, "env": provider_config.api_key_env})
            continue
        providers[provider_config.name] = build_provider(provider_config)
        logger.info("provider ready", extra={"provider": provider_config.name,
                                             "type": provider_config.type.value,
                                             "model": provider_config.model})
    return providers


def _key_missing(config: ProviderConfig) -> bool:
    """True if the provider names an API-key variable that is unset or empty."""
    return bool(config.api_key_env) and not os.environ.get(config.api_key_env or "", "").strip()


async def close_providers(providers: dict[str, Provider]) -> None:
    """Close every provider, logging (not raising) individual failures.

    Args:
        providers: Providers to close.
    """
    for name, provider in providers.items():
        try:
            await provider.aclose()
        except Exception:
            logger.exception("failed to close provider", extra={"provider": name})
