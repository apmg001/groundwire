"""Tests for deployment checks (worker count vs. per-process state)."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import make_config
from groundwire import main as main_module
from groundwire.core.config import Settings
from groundwire.core.deployment import check_deployment, effective_workers
from groundwire.core.errors import ConfigurationError
from groundwire.main import create_app

APPROVAL_RULE: dict[str, Any] = {"rules": [
    {"name": "refund-approval", "tools": ["issue_refund"], "action": "require_approval"}]}


# --------------------------------------------------------------------------- #
# effective_workers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("configured", "environ", "expected"), [
    (1, {}, 1),
    (4, {}, 4),
    (1, {"WEB_CONCURRENCY": "3"}, 3),       # uvicorn's default for --workers
    (2, {"WEB_CONCURRENCY": "3"}, 3),
    (1, {"WEB_CONCURRENCY": " "}, 1),
    (1, {"WEB_CONCURRENCY": "lots"}, 1),    # uvicorn itself rejects this
])
def test_effective_workers(configured: int, environ: dict[str, str], expected: int) -> None:
    assert effective_workers(configured, environ) == expected


# --------------------------------------------------------------------------- #
# check_deployment
# --------------------------------------------------------------------------- #


def test_single_worker_is_always_fine() -> None:
    assert check_deployment(make_config(tool_policy=APPROVAL_RULE), workers=1) == []


def test_in_memory_approvals_refuse_several_workers() -> None:
    with pytest.raises(ConfigurationError) as info:
        check_deployment(make_config(tool_policy=APPROVAL_RULE), workers=4)
    message = info.value.message
    assert "4 worker processes" in message and "refund-approval" in message
    assert "GROUNDWIRE_WORKERS=1" in message and "one replica" in message


@pytest.mark.parametrize("overrides", [
    {},                                                          # no rule holds calls
    {"tool_policy": {**APPROVAL_RULE, "enabled": False}},        # policy off
])
def test_approvals_unused_means_several_workers_are_allowed(overrides: dict[str, Any]) -> None:
    warnings = check_deployment(make_config(**overrides), workers=4)
    assert any("circuit breakers" in w for w in warnings)


def test_default_action_counts_as_an_approval_rule() -> None:
    with pytest.raises(ConfigurationError):
        check_deployment(make_config(tool_policy={"default_action": "require_approval"}),
                         workers=2)


def test_in_memory_cache_is_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        warnings = check_deployment(make_config(cache={"enabled": True}), workers=2)
    assert any("response cache is in memory" in w for w in warnings)
    assert "response cache is in memory" in caplog.text


# --------------------------------------------------------------------------- #
# Startup paths
# --------------------------------------------------------------------------- #


def test_app_startup_fails_under_web_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """``uvicorn --factory`` with $WEB_CONCURRENCY never reaches our entrypoint."""
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    app = create_app(settings=Settings(), config=make_config(tool_policy=APPROVAL_RULE))
    with pytest.raises(ConfigurationError, match="worker processes"), TestClient(app):
        pass


def test_app_starts_with_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    app = create_app(settings=Settings(), config=make_config(tool_policy=APPROVAL_RULE))
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200


def test_entrypoint_exits_before_spawning_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.setenv("GROUNDWIRE_WORKERS", "4")
    monkeypatch.setattr(main_module, "load_gateway_config",
                        lambda _path: make_config(tool_policy=APPROVAL_RULE))
    started: list[bool] = []
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **k: started.append(True))
    with pytest.raises(SystemExit) as info:
        main_module.run()
    assert info.value.code == 2
    assert started == []  # uvicorn never started
