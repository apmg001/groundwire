"""Shared test fixtures and fakes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from groundwire.container import build_container
from groundwire.core.config import GatewayConfig, parse_gateway_config
from groundwire.domain.models import ChatCompletionRequest, Completion, StreamChunk, Usage
from groundwire.main import create_app
from groundwire.providers.base import Provider

# --------------------------------------------------------------------------- #
# Scripted fake provider
# --------------------------------------------------------------------------- #


@dataclass
class MidStreamFailure:
    """Outcome: emit ``text`` as chunks, then raise ``error``."""

    text: str
    error: Exception


Outcome = str | Exception | MidStreamFailure


class ScriptedProvider(Provider):
    """Provider whose successive calls follow a script of outcomes.

    A ``str`` outcome succeeds with that text, an ``Exception`` is raised, and
    a :class:`MidStreamFailure` streams some text before failing. The last
    outcome repeats once the script is exhausted.
    """

    def __init__(self, name: str, *outcomes: Outcome, delay_s: float = 0.0) -> None:
        super().__init__(name, f"{name}-model")
        self._outcomes = list(outcomes) or ["ok"]
        self._delay_s = delay_s
        self.calls = 0
        self.closed = False
        self.requests: list[ChatCompletionRequest] = []

    def _next(self, request: ChatCompletionRequest | None = None) -> Outcome:
        if request is not None:
            self.requests.append(request)
        index = min(self.calls, len(self._outcomes) - 1)
        self.calls += 1
        return self._outcomes[index]

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        outcome = self._next(request)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if isinstance(outcome, Exception):
            raise outcome
        text = outcome.text if isinstance(outcome, MidStreamFailure) else outcome
        return Completion(text=text, usage=Usage.of(3, 5), upstream_model=self.model)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        outcome = self._next(request)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if isinstance(outcome, Exception):
            raise outcome
        text = outcome.text if isinstance(outcome, MidStreamFailure) else outcome
        for i, word in enumerate(text.split(" ")):
            yield StreamChunk.text(word if i == 0 else " " + word, role=i == 0)
        if isinstance(outcome, MidStreamFailure):
            raise outcome.error
        yield StreamChunk.finish("stop", Usage.of(3, 5))

    async def aclose(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #


def make_config(**overrides: Any) -> GatewayConfig:
    """Build a small valid config: route ``default`` = [primary, secondary]."""
    raw: dict[str, Any] = {
        "providers": [
            {"name": "primary", "type": "mock", "model": "m"},
            {"name": "secondary", "type": "mock", "model": "m"},
        ],
        "routes": {"default": ["primary", "secondary"]},
        "retry": {"max_attempts": 2, "base_delay_s": 0.0, "max_delay_s": 0.0},
        "circuit_breaker": {"failure_threshold": 2, "recovery_timeout_s": 30},
        "request_timeout_s": 5,
    }
    raw.update(overrides)
    return parse_gateway_config(raw)


def chat_request(text: str = "hello", **kwargs: Any) -> ChatCompletionRequest:
    """Build a one-message request."""
    return ChatCompletionRequest(messages=[{"role": "user", "content": text}], **kwargs)


@pytest.fixture
def config() -> GatewayConfig:
    return make_config()


# --------------------------------------------------------------------------- #
# App client
# --------------------------------------------------------------------------- #

ClientFactory = Callable[..., TestClient]


@pytest.fixture
def client_factory() -> Iterator[ClientFactory]:
    """Return a factory that builds a TestClient around given providers."""
    clients: list[TestClient] = []

    def factory(providers: dict[str, Provider], config: GatewayConfig | None = None) -> TestClient:
        gateway_config = config or make_config()
        app = create_app(
            config=gateway_config,
            container_factory=lambda: build_container(gateway_config, providers=providers),
        )
        client = TestClient(app)
        client.__enter__()  # run lifespan
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)
