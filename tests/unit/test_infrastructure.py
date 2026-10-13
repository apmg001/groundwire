"""Tests for logging, request context and the provider factory."""

from __future__ import annotations

import json
import logging

import pytest

from conftest import make_config
from groundwire.core.config import ProviderConfig
from groundwire.core.context import get_request_id, new_request_id, set_request_id
from groundwire.core.errors import ConfigurationError
from groundwire.core.logging import JsonFormatter, RequestIdFilter, TextFormatter, configure_logging
from groundwire.providers.anthropic import AnthropicProvider
from groundwire.providers.factory import build_provider, build_providers, close_providers
from groundwire.providers.mock import MockProvider
from groundwire.providers.openai_compat import OpenAICompatibleProvider

# --------------------------------------------------------------------------- #
# Request context
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("candidate", [None, "", "bad id with spaces", "x" * 200, "a\nb"])
def test_unsafe_request_ids_are_replaced(candidate: str | None) -> None:
    assert new_request_id(candidate).startswith("gw-")


def test_safe_request_id_is_kept_and_bound() -> None:
    assert new_request_id("trace-1:abc") == "trace-1:abc"
    set_request_id("rid-42")
    assert get_request_id() == "rid-42"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("gw.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    RequestIdFilter().filter(record)
    return record


def test_json_formatter_includes_extras_and_request_id() -> None:
    set_request_id("rid-json")
    line = JsonFormatter().format(_record(provider="openai", attempt=2))
    data = json.loads(line)
    assert data["msg"] == "hello world"
    assert data["request_id"] == "rid-json"
    assert data["provider"] == "openai"
    assert data["attempt"] == 2


def test_json_formatter_never_fails_on_odd_values() -> None:
    data = json.loads(JsonFormatter().format(_record(obj=object())))
    assert "object" in data["obj"]


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("kaboom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
    assert "kaboom" in json.loads(JsonFormatter().format(record))["exc"]


def test_text_formatter_appends_extras() -> None:
    line = TextFormatter().format(_record(route="default"))
    assert "hello world" in line
    assert "route=default" in line


def test_configure_logging_is_idempotent() -> None:
    configure_logging("DEBUG", json_output=False)
    configure_logging("INFO", json_output=True)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #


def test_factory_builds_each_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW_KEY", "secret")
    oa = build_provider(ProviderConfig(name="a", type="openai_compatible", model="m",
                                       base_url="http://x/v1"))
    an = build_provider(ProviderConfig(name="b", type="anthropic", model="m",
                                       base_url="https://x", api_key_env="GW_KEY"))
    mk = build_provider(ProviderConfig(name="c", type="mock", model="m"))
    assert isinstance(oa, OpenAICompatibleProvider)
    assert isinstance(an, AnthropicProvider)
    assert isinstance(mk, MockProvider)
    assert "secret" not in repr(an)


def test_factory_fails_fast_on_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GW_MISSING", raising=False)
    with pytest.raises(ConfigurationError, match="GW_MISSING"):
        build_provider(ProviderConfig(name="a", type="openai_compatible", model="m",
                                      base_url="http://x/v1", api_key_env="GW_MISSING"))


async def test_build_and_close_providers() -> None:
    providers = build_providers(make_config())
    assert set(providers) == {"primary", "secondary"}
    await close_providers(providers)


async def test_close_providers_continues_after_failure() -> None:
    class Exploding(MockProvider):
        async def aclose(self) -> None:
            raise RuntimeError("cannot close")

    from groundwire.core.config import MockBehaviour

    closed: list[str] = []

    class Tracking(MockProvider):
        async def aclose(self) -> None:
            closed.append(self.name)

    await close_providers({
        "bad": Exploding(name="bad", model="m", behaviour=MockBehaviour()),
        "good": Tracking(name="good", model="m", behaviour=MockBehaviour()),
    })
    assert closed == ["good"]
