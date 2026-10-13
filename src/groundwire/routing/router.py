"""Router: serves a request from a route's ordered provider fallback chain.

For each provider in the chain, in order:

1. Ask its circuit breaker for permission; skip it instantly if open.
2. Call it through the retry policy (retryable errors are retried in place).
3. On success, return. On a failover-eligible error, record it and try the
   next provider. On a client-fault error (upstream 400), stop and surface it.

If the chain is exhausted, :class:`AllProvidersFailedError` carries a record
of every attempt so operators (and clients) can see exactly what happened.

Streaming caveat: failover is only possible *before the first chunk reaches
the client*. The router therefore opens the stream and pulls the first chunk
inside the retry/failover loop; once a chunk has been handed out, a later
failure is surfaced as a stream error rather than silently switching models
mid-sentence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial
from typing import TypeVar

from groundwire.core.config import GatewayConfig
from groundwire.core.errors import (
    AllProvidersFailedError,
    AttemptRecord,
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    ProviderResponseError,
)
from groundwire.domain.models import ChatCompletionRequest, Completion, StreamChunk
from groundwire.observability.metrics import GatewayMetrics
from groundwire.providers.base import Provider
from groundwire.resilience.circuit_breaker import CircuitBreaker, CircuitState
from groundwire.resilience.retry import RetryPolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RoutedCompletion:
    """A completion plus routing metadata.

    Attributes:
        completion: The provider's completion.
        provider: Name of the provider that served it.
        attempts: Every attempt made, in order (last one is the success).
    """

    completion: Completion
    provider: str
    attempts: list[AttemptRecord]


@dataclass(slots=True)
class RoutedStream:
    """A live stream plus routing metadata.

    Attributes:
        provider: Name of the provider serving the stream.
        attempts: Attempts made before the stream started.
        chunks: The chunk iterator (first chunk included). Consume it fully or
            close it (``aclose``) to release the upstream connection.
    """

    provider: str
    attempts: list[AttemptRecord]
    chunks: AsyncIterator[StreamChunk] = field(repr=False)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


class Router:
    """Routes requests across providers with retries, breakers and failover."""

    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        *,
        retry_policy: RetryPolicy | None = None,
        metrics: GatewayMetrics | None = None,
        breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        """Create the router.

        Args:
            config: Gateway configuration (routes, policies).
            providers: Enabled providers by name.
            retry_policy: Retry policy; built from config if omitted.
            metrics: Metrics sink; optional so the router is usable standalone.
            breakers: Pre-built breakers (tests); built from config if omitted.

        Raises:
            ConfigurationError: If a route ends up with no usable provider.
        """
        self._metrics = metrics
        self._providers = providers
        self._retry = retry_policy or RetryPolicy(config.retry)
        self._default_route = config.default_route

        on_change = metrics.set_circuit_state if metrics else None
        self._breakers = breakers or {
            name: CircuitBreaker(name, config.circuit_breaker, on_state_change=on_change)
            for name in providers
        }
        if metrics:
            for name in providers:
                metrics.set_circuit_state(name, CircuitState.CLOSED)

        # Disabled providers are dropped from chains here, once.
        self._routes: dict[str, list[str]] = {}
        for route, chain in config.routes.items():
            usable = [p for p in chain if p in providers]
            if not usable:
                raise ConfigurationError(f"route '{route}' has no enabled providers")
            self._routes[route] = usable

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def routes(self) -> dict[str, list[str]]:
        """Route alias -> usable provider chain (copy)."""
        return {k: list(v) for k, v in self._routes.items()}

    def breaker_states(self) -> dict[str, str]:
        """Provider name -> current circuit state."""
        return {name: b.state.value for name, b in self._breakers.items()}

    def resolve(self, model: str) -> str:
        """Map a client ``model`` value to a route alias.

        Args:
            model: Value of the request's ``model`` field.

        Returns:
            The route alias.

        Raises:
            ModelNotFoundError: If no such route exists.
        """
        if model == "default":
            return self._default_route
        if model in self._routes:
            return model
        raise ModelNotFoundError(
            f"model '{model}' is not a configured route",
            details={"available": sorted(self._routes)},
        )

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #

    async def complete(self, route: str, request: ChatCompletionRequest) -> RoutedCompletion:
        """Serve a non-streaming request from ``route``'s chain.

        Args:
            route: Resolved route alias.
            request: Client request.

        Returns:
            The completion with routing metadata.

        Raises:
            ProviderBadRequestError: If a provider rejected the request itself.
            AllProvidersFailedError: If every provider failed or was skipped.
        """
        attempts: list[AttemptRecord] = []
        for provider in self._chain(route):
            started = time.perf_counter()
            result = await self._attempt(provider, attempts, partial(provider.complete, request))
            if result is None:
                continue
            self._on_success(route, provider.name, attempts, time.perf_counter() - started)
            usage = result.usage
            if self._metrics:
                self._metrics.record_tokens(provider.name, usage.prompt_tokens,
                                            usage.completion_tokens)
            return RoutedCompletion(completion=result, provider=provider.name, attempts=attempts)
        raise AllProvidersFailedError(route, attempts)

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    async def stream(self, route: str, request: ChatCompletionRequest) -> RoutedStream:
        """Open a stream from ``route``'s chain, failing over until one starts.

        Args:
            route: Resolved route alias.
            request: Client request.

        Returns:
            The live stream. The first chunk has already been received.

        Raises:
            ProviderBadRequestError: If a provider rejected the request itself.
            AllProvidersFailedError: If no provider could start a stream.
        """
        attempts: list[AttemptRecord] = []
        for provider in self._chain(route):
            started = time.perf_counter()
            opened = await self._attempt(
                provider, attempts, partial(self._open_stream, provider, request))
            if opened is None:
                continue
            first, rest = opened
            self._on_success(route, provider.name, attempts, time.perf_counter() - started)
            return RoutedStream(provider=provider.name, attempts=attempts,
                                chunks=self._continue_stream(provider, first, rest))
        raise AllProvidersFailedError(route, attempts)

    @staticmethod
    async def _open_stream(
        provider: Provider, request: ChatCompletionRequest
    ) -> tuple[StreamChunk, AsyncIterator[StreamChunk]]:
        """Start a provider stream and pull its first chunk.

        Args:
            provider: Provider to stream from.
            request: Client request.

        Returns:
            ``(first_chunk, iterator_positioned_after_it)``.

        Raises:
            ProviderError: If the stream fails or ends before any chunk.
        """
        iterator = provider.stream(request)
        try:
            first = await anext(iterator)
        except StopAsyncIteration as exc:
            raise ProviderResponseError(f"{provider.name}: stream ended with no data",
                                        provider=provider.name) from exc
        except BaseException:
            await _safe_aclose(iterator)
            raise
        return first, iterator

    async def _continue_stream(
        self, provider: Provider, first: StreamChunk, rest: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """Yield the first chunk, then the rest, tracking mid-stream failures.

        Args:
            provider: Provider serving the stream.
            first: Already-received first chunk.
            rest: Remaining chunks.

        Yields:
            Stream chunks.

        Raises:
            ProviderError: If the provider fails mid-stream (no failover).
        """
        breaker = self._breakers[provider.name]
        prompt = completion = 0
        async with aclosing(rest):  # type: ignore[type-var]
            try:
                yield first
                async for chunk in rest:
                    if chunk.usage:
                        prompt, completion = (chunk.usage.prompt_tokens,
                                              chunk.usage.completion_tokens)
                    yield chunk
            except ProviderError as exc:
                if exc.failover:
                    breaker.record_failure()
                logger.error("provider failed mid-stream",
                             extra={"provider": provider.name, "error_code": exc.code})
                if self._metrics:
                    self._metrics.record_attempt(provider.name, "error", exc.code)
                raise
        if self._metrics:
            self._metrics.record_tokens(provider.name, prompt, completion)

    # ------------------------------------------------------------------ #
    # Shared attempt logic
    # ------------------------------------------------------------------ #

    def _chain(self, route: str) -> list[Provider]:
        try:
            return [self._providers[name] for name in self._routes[route]]
        except KeyError as exc:  # resolve() should have caught this
            raise ModelNotFoundError(f"route '{route}' is not configured") from exc

    async def _attempt(
        self,
        provider: Provider,
        attempts: list[AttemptRecord],
        call: Callable[[], Awaitable[T]],
    ) -> T | None:
        """Run one provider attempt (with retries) under its circuit breaker.

        Args:
            provider: Provider to call.
            attempts: Attempt log to append to.
            call: Coroutine factory performing the provider call.

        Returns:
            The call's result, or ``None`` if the router should fail over.

        Raises:
            ProviderError: For errors that must not fail over (client faults).
            asyncio.CancelledError: Propagated after releasing the breaker.
        """
        breaker = self._breakers[provider.name]
        if not breaker.try_acquire():
            attempts.append(AttemptRecord(provider.name, "skipped_circuit_open"))
            if self._metrics:
                self._metrics.record_attempt(provider.name, "skipped_circuit_open")
            logger.info("provider skipped: circuit open", extra={"provider": provider.name})
            return None

        try:
            result = await self._retry.run(call, provider=provider.name)
        except asyncio.CancelledError:
            breaker.release()  # client went away / deadline: no verdict on provider
            raise
        except ProviderError as exc:
            attempts.append(AttemptRecord(provider.name, "error", exc.code, exc.message))
            if self._metrics:
                self._metrics.record_attempt(provider.name, "error", exc.code)
            if not exc.failover:
                breaker.release()
                raise
            breaker.record_failure()
            logger.warning("provider failed, failing over",
                           extra={"provider": provider.name, "error_code": exc.code})
            return None

        breaker.record_success()
        attempts.append(AttemptRecord(provider.name, "success"))
        return result

    def _on_success(self, route: str, provider: str, attempts: list[AttemptRecord],
                    elapsed_s: float) -> None:
        if self._metrics:
            self._metrics.record_attempt(provider, "success")
            self._metrics.provider_latency.labels(provider=provider).observe(elapsed_s)
            if len(attempts) > 1:
                self._metrics.failovers.labels(route=route).inc()
        if len(attempts) > 1:
            logger.info("request served after failover",
                        extra={"route": route, "provider": provider, "attempts": len(attempts)})


async def _safe_aclose(iterator: AsyncIterator[StreamChunk]) -> None:
    """Close an async generator, ignoring errors raised while closing."""
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        logger.debug("error while closing provider stream", exc_info=True)
