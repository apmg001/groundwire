"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from groundwire.core.config import (
    ProviderConfig,
    Settings,
    load_gateway_config,
    parse_gateway_config,
    resolve_api_key,
)
from groundwire.core.errors import ConfigurationError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "providers": [{"name": "a", "type": "mock", "model": "m"}],
        "routes": {"default": ["a"]},
    }
    raw.update(overrides)
    return raw


@pytest.mark.parametrize("name", ["groundwire.yaml", "groundwire.docker.yaml"])
def test_shipped_configs_are_valid(name: str) -> None:
    config = load_gateway_config(REPO_ROOT / "config" / name)
    assert "default" in config.routes
    assert "demo" in config.routes


def test_minimal_config_gets_defaults() -> None:
    config = parse_gateway_config(_raw())
    assert config.retry.max_attempts == 2
    assert config.circuit_breaker.failure_threshold == 5
    assert config.default_route == "default"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"routes": {"default": ["missing"]}}, "unknown providers"),
        ({"routes": {"default": []}}, "empty provider chain"),
        ({"routes": {"default": ["a", "a"]}}, "more than once"),
        ({"default_route": "nope"}, "default_route"),
        ({"unknown_key": 1}, "Extra inputs"),
        ({"providers": [{"name": "a", "type": "mock", "model": "m"},
                        {"name": "a", "type": "mock", "model": "m"}]}, "duplicate"),
        ({"providers": [{"name": "a", "type": "openai_compatible", "model": "m"}]},
         "base_url is required"),
        ({"providers": [{"name": "a", "type": "mock", "model": "m", "enabled": False}]},
         "no enabled providers"),
    ],
)
def test_invalid_configs_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        parse_gateway_config(_raw(**overrides))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_gateway_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("providers: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_gateway_config(path)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_gateway_config(path)


def test_resolve_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProviderConfig(name="p", type="openai_compatible", model="m",
                              base_url="https://x", api_key_env="GW_TEST_KEY")
    monkeypatch.delenv("GW_TEST_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="GW_TEST_KEY"):
        resolve_api_key(provider)

    monkeypatch.setenv("GW_TEST_KEY", "sk-secret")
    key = resolve_api_key(provider)
    assert key is not None
    assert key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(key)  # never leaks into logs


def test_settings_reject_bad_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWIRE_LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError, match="unknown log level"):
        Settings()
