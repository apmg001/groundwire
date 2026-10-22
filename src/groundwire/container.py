"""Composition root: builds and owns the gateway's object graph.

All wiring happens here, once, at startup. Nothing else in the codebase
constructs its own dependencies, which keeps modules decoupled and lets tests
swap any piece (e.g. inject fake providers) through :func:`build_container`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from groundwire.agents.answer_check import AnswerChecker
from groundwire.agents.policy import ToolPolicy
from groundwire.cache import ResponseCache, build_cache
from groundwire.core.config import GatewayConfig
from groundwire.guardrails.grounding import GroundingVerifier, build_verifier
from groundwire.observability.metrics import GatewayMetrics
from groundwire.providers.base import Provider
from groundwire.providers.factory import build_providers, close_providers
from groundwire.routing.cascade import CascadeRunner
from groundwire.routing.router import Router
from groundwire.safety.guard import SafetyGuard
from groundwire.services.chat import ChatService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    """The gateway's long-lived components.

    Attributes:
        config: Gateway configuration.
        providers: Enabled providers by name.
        metrics: Metrics registry.
        router: Provider router.
        verifier: Grounding verifier.
        safety: Prompt-injection and PII guard.
        cache: Response cache (``None`` when disabled).
        chat: Chat completion service.
    """

    config: GatewayConfig
    providers: dict[str, Provider]
    metrics: GatewayMetrics
    router: Router
    verifier: GroundingVerifier
    safety: SafetyGuard
    cache: ResponseCache | None
    chat: ChatService

    async def aclose(self) -> None:
        """Release provider connection pools."""
        await close_providers(self.providers)
        logger.info("container closed")


def build_container(
    config: GatewayConfig,
    *,
    providers: dict[str, Provider] | None = None,
    verifier: GroundingVerifier | None = None,
    safety: SafetyGuard | None = None,
    cache: ResponseCache | None = None,
) -> Container:
    """Wire up every component from configuration.

    Args:
        config: Validated gateway configuration.
        providers: Pre-built providers (tests); built from config if omitted.
        verifier: Pre-built verifier (tests); built from config if omitted.
        safety: Pre-built safety guard (tests); built from config if omitted.
        cache: Pre-built response cache (tests); built from config if omitted.

    Returns:
        The container.

    Raises:
        ConfigurationError: On missing API keys, unloadable scorer or embedding
            models, or routes without enabled providers.
    """
    built_providers = providers if providers is not None else build_providers(config)
    # No cleanup is needed if wiring below fails: httpx clients open no sockets
    # until their first request, so half-built providers hold no resources.
    metrics = GatewayMetrics()
    router = Router(config, built_providers, metrics=metrics)
    grounding = verifier or build_verifier(config.grounding)
    guard = safety or SafetyGuard(config.safety, metrics=metrics)
    response_cache = cache or build_cache(config.cache, metrics=metrics)
    cascade = CascadeRunner(router, grounding, metrics=metrics)
    chat = ChatService(router, grounding, metrics, request_timeout_s=config.request_timeout_s,
                       safety=guard, cache=response_cache, cascade=cascade,
                       tool_policy=ToolPolicy(config.tool_policy),
                       answer_checker=AnswerChecker(config.answer_check))
    return Container(config=config, providers=built_providers, metrics=metrics,
                     router=router, verifier=grounding, safety=guard, cache=response_cache,
                     chat=chat)
