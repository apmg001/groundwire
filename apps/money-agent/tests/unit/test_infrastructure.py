"""Settings validation, logging, the HTTP embedder, the clock and container wiring."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from money_agent.adapters.clock import SystemClock
from money_agent.container import build_container
from money_agent.core.config import EmbedderType, StorageBackend, load_settings
from money_agent.core.errors import AssistantUnavailableError, ConfigurationError
from money_agent.core.logging import (
    JsonFormatter,
    configure_logging,
    get_request_id,
    reset_request_id,
    set_request_id,
)
from money_agent.memory import embeddings
from money_agent.memory.embeddings import HttpEmbedder

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("overrides", "problem"), [
    ({"storage": "postgres"}, "MONEY_DATABASE_URL"),
    ({"db_pool_min": 5, "db_pool_max": 2}, "POOL_MIN"),
    ({"embedder": "http"}, "EMBEDDING_URL"),
    ({"gateway_url": "groundwire:8000"}, "http://"),
    ({"timezone": "Mars/Olympus"}, "time zone"),
    ({"agent_max_steps": 0}, "agent_max_steps"),
])
def test_bad_settings_stop_startup_with_a_clear_message(overrides: dict, problem: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(ConfigurationError, match=problem):
        load_settings(**overrides)


def test_settings_normalise_and_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONEY_GATEWAY_URL", "http://gw:8000/")
    monkeypatch.setenv("MONEY_DATABASE_URL", "postgresql://u:secret@db/money")
    settings = load_settings()
    assert settings.gateway_url == "http://gw:8000"
    assert "secret" not in repr(settings)  # secrets never appear in reprs or logs
    assert settings.zone == ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_json_logs_carry_the_request_id_and_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_output=True)
    token = set_request_id("req-abcdef12")
    try:
        assert get_request_id() == "req-abcdef12"
        logging.getLogger("money_agent.test").info("hello %s", "world", extra={"rows": 3})
    finally:
        reset_request_id(token)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["message"] == "hello world" and line["rows"] == 3
    assert line["request_id"] == "req-abcdef12" and line["level"] == "INFO"
    assert get_request_id() is None


def test_exceptions_are_serialised() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None,
                                   exc_info=__import__("sys").exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exception"]


def test_text_logs_for_development(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", json_output=False)
    logging.getLogger("money_agent.test").debug("plain")
    assert "plain" in capsys.readouterr().out
    configure_logging("WARNING")


# --------------------------------------------------------------------------- #
# HTTP embedder
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None
    monkeypatch.setattr(embeddings.asyncio, "sleep", instant)


def _embedder(handler: object, dimension: int = 2) -> HttpEmbedder:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return HttpEmbedder(http, base_url="http://emb/v1/", model="m", dimension=dimension,
                        api_key="k", max_retries=1)


async def test_http_embedder_orders_and_normalises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0, 2]},
                                                  {"index": 0, "embedding": [3, 4]}]})

    vectors = await _embedder(handler).embed(["a", "b"])
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]
    assert await _embedder(handler).embed([]) == []
    assert _embedder(handler).dimension == 2


@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2, 3]}]}),  # wrong size
    httpx.Response(200, json={"oops": True}),
    httpx.Response(400, json={}),
    httpx.Response(503, json={}),
])
async def test_http_embedder_failures(response: httpx.Response) -> None:
    with pytest.raises(AssistantUnavailableError):
        await _embedder(lambda _: response).embed(["a"])


async def test_http_embedder_retries_connection_errors() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(AssistantUnavailableError):
        await _embedder(handler).embed(["a"])
    assert len(attempts) == 2


# --------------------------------------------------------------------------- #
# Clock and wiring
# --------------------------------------------------------------------------- #


def test_system_clock_uses_the_configured_zone() -> None:
    clock = SystemClock(ZoneInfo("Asia/Kolkata"))
    assert clock.now().utcoffset() is not None
    assert clock.today() == clock.now().date()
    assert abs((SystemClock().now() - datetime.now(UTC)).total_seconds()) < 5


async def test_container_wires_postgres_and_the_http_embedder() -> None:
    settings = load_settings(storage=StorageBackend.POSTGRES,
                             database_url="postgresql://u:p@127.0.0.1:1/x",
                             embedder=EmbedderType.HTTP, embedding_url="http://emb/v1",
                             embedding_model="nomic", embedding_api_key="key")
    container = build_container(settings)
    try:
        assert container.database is not None
        assert isinstance(container.embedder, HttpEmbedder)
    finally:
        await container.close()
