"""Tests for routing, failover and circuit-breaker integration."""

from __future__ import annotations

import pytest

from conftest import MidStreamFailure, ScriptedProvider, chat_request, make_config
from groundwire.core.errors import (
    AllProvidersFailedError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderNotFoundError,
    ProviderServerError,
    ProviderTimeoutError,
)
from groundwire.observability.metrics import GatewayMetrics
from groundwire.routing.router import Router


def _err(provider: str) -> ProviderServerError:
    return ProviderServerError("boom", provider=provider, upstream_status=503)


def _router(primary: ScriptedProvider, secondary: ScriptedProvider,
            metrics: GatewayMetrics | None = None) -> Router:
    return Router(make_config(), {"primary": primary, "secondary": secondary}, metrics=metrics)


async def test_primary_success() -> None:
    primary, secondary = ScriptedProvider("primary", "hi"), ScriptedProvider("secondary")
    result = await _router(primary, secondary).complete("default", chat_request())
    assert result.provider == "primary"
    assert result.completion.text == "hi"
    assert secondary.calls == 0


async def test_retry_then_success_on_same_provider() -> None:
    primary = ScriptedProvider("primary", _err("primary"), "recovered")
    result = await _router(primary, ScriptedProvider("secondary")).complete(
        "default", chat_request())
    assert result.provider == "primary"
    assert primary.calls == 2


async def test_failover_after_retries_exhausted() -> None:
    metrics = GatewayMetrics()
    primary = ScriptedProvider("primary", ProviderServerError("502", provider="primary"))
    secondary = ScriptedProvider("secondary", "backup answer")
    result = await _router(primary, secondary, metrics).complete("default", chat_request())

    assert result.provider == "secondary"
    assert primary.calls == 2  # max_attempts
    assert [a.outcome for a in result.attempts] == ["error", "success"]
    assert metrics.failovers.labels(route="default")._value.get() == 1


async def test_timeout_fails_over_without_retrying_the_slow_provider() -> None:
    primary = ScriptedProvider("primary", ProviderTimeoutError("slow", provider="primary"))
    result = await _router(primary, ScriptedProvider("secondary", "ok")).complete(
        "default", chat_request())
    assert result.provider == "secondary"
    assert primary.calls == 1  # one full timeout, not two


async def test_auth_error_fails_over_without_retry() -> None:
    primary = ScriptedProvider("primary", ProviderAuthError("401", provider="primary"))
    result = await _router(primary, ScriptedProvider("secondary", "ok")).complete(
        "default", chat_request())
    assert result.provider == "secondary"
    assert primary.calls == 1


async def test_missing_model_fails_over_and_counts_against_the_breaker() -> None:
    """A 404 means the provider is misconfigured (retired model), not that the client erred."""
    primary = ScriptedProvider("primary", ProviderNotFoundError(
        "model does not exist", provider="primary", upstream_status=404))
    router = _router(primary, ScriptedProvider("secondary", "backup answer"))
    result = await router.complete("default", chat_request())
    assert result.provider == "secondary"
    assert primary.calls == 1  # not retried: it will not fix itself
    assert result.attempts[0].error_code == "provider_not_found"
    await router.complete("default", chat_request())  # second failure (threshold is 2)
    assert router.breaker_states()["primary"] == "open"  # now skipped without a call
    await router.complete("default", chat_request())
    assert primary.calls == 2


async def test_bad_request_is_surfaced_without_failover() -> None:
    primary = ScriptedProvider("primary", ProviderBadRequestError("400", provider="primary"))
    secondary = ScriptedProvider("secondary", "ok")
    with pytest.raises(ProviderBadRequestError):
        await _router(primary, secondary).complete("default", chat_request())
    assert secondary.calls == 0


async def test_all_providers_failed_reports_attempts() -> None:
    router = _router(ScriptedProvider("primary", _err("primary")),
                     ScriptedProvider("secondary", _err("secondary")))
    with pytest.raises(AllProvidersFailedError) as info:
        await router.complete("default", chat_request())
    assert [a.provider for a in info.value.attempts] == ["primary", "secondary"]
    assert info.value.http_status == 503


async def test_open_circuit_skips_provider_instantly() -> None:
    primary = ScriptedProvider("primary", _err("primary"))
    secondary = ScriptedProvider("secondary", "ok")
    router = _router(primary, secondary)

    for _ in range(2):  # failure_threshold = 2
        await router.complete("default", chat_request())
    assert router.breaker_states()["primary"] == "open"

    calls_before = primary.calls
    result = await router.complete("default", chat_request())
    assert result.provider == "secondary"
    assert primary.calls == calls_before  # not called at all
    assert result.attempts[0].outcome == "skipped_circuit_open"


def test_resolve_routes() -> None:
    router = _router(ScriptedProvider("primary"), ScriptedProvider("secondary"))
    assert router.resolve("default") == "default"
    with pytest.raises(ModelNotFoundError):
        router.resolve("gpt-9")


def test_disabled_providers_are_dropped_from_chains() -> None:
    config = make_config(providers=[
        {"name": "primary", "type": "mock", "model": "m"},
        {"name": "secondary", "type": "mock", "model": "m", "enabled": False},
    ])
    router = Router(config, {"primary": ScriptedProvider("primary")})
    assert router.routes == {"default": ["primary"]}


def test_route_without_built_providers_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Router(make_config(), {})


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


async def _collect(router: Router) -> tuple[str, str]:
    routed = await router.stream("default", chat_request(stream=True))
    text = "".join([c.content() async for c in routed.chunks])
    return routed.provider, text


async def test_stream_happy_path() -> None:
    router = _router(ScriptedProvider("primary", "hello there world"),
                     ScriptedProvider("secondary"))
    assert await _collect(router) == ("primary", "hello there world")


async def test_stream_fails_over_before_first_chunk() -> None:
    router = _router(ScriptedProvider("primary", _err("primary")),
                     ScriptedProvider("secondary", "from backup"))
    assert await _collect(router) == ("secondary", "from backup")


async def test_stream_mid_stream_failure_is_not_failed_over() -> None:
    primary = ScriptedProvider("primary", MidStreamFailure("partial answer", _err("primary")))
    secondary = ScriptedProvider("secondary", "unused")
    router = _router(primary, secondary)
    routed = await router.stream("default", chat_request(stream=True))
    received: list[str] = []
    with pytest.raises(ProviderServerError):
        async for chunk in routed.chunks:
            received.append(chunk.content())
    assert "".join(received) == "partial answer"
    assert secondary.calls == 0


async def test_stream_that_ends_empty_fails_over() -> None:
    class EmptyStream(ScriptedProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def,override]
            self.calls += 1
            return
            yield  # pragma: no cover - makes this an async generator

    router = _router(EmptyStream("primary"), ScriptedProvider("secondary", "backup"))
    assert await _collect(router) == ("secondary", "backup")
