"""Tests for the reviewer CLI (scripts/approvals.py), with a fake gateway."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "approvals.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("approvals_cli", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load()
RECORD = {"id": "apr_" + "a" * 32, "status": "pending", "tool": "issue_refund",
          "arguments": {"amount": 1200}, "rule": "refund-approval", "scope": "alice",
          "created_at": "2026-10-23T10:00:00+00:00", "reviewer": None}


def _client(handler: object) -> httpx.Client:
    return httpx.Client(base_url="http://gw", transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_list_prints_one_line_per_approval_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"object": "list", "data": [RECORD]})

    cli.configure_output()
    args = cli.build_parser().parse_args(["list"])
    with _client(handler) as client:
        assert cli.run(args, client) == 0
    out = capsys.readouterr().out
    assert RECORD["id"] in out and '"amount":1200' in out
    assert seen == ["http://gw/v1/approvals?status=pending&limit=50"]


def test_approve_sends_reviewer_and_note() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={**RECORD, "status": "approved", "reviewer": "anita"})

    args = cli.build_parser().parse_args(["approve", RECORD["id"], "--reviewer", "anita",
                                          "--note", "verified"])
    with _client(handler) as client:
        assert cli.run(args, client) == 0
    assert bodies == [{"reviewer": "anita", "note": "verified"}]


def test_gateway_errors_become_api_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": {"message": "this approval is already approved"}})

    args = cli.build_parser().parse_args(["reject", RECORD["id"]])
    with _client(handler) as client, pytest.raises(cli.ApiError, match="409"):
        cli.run(args, client)


def test_missing_token_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli.TOKEN_ENV, raising=False)
    assert cli.main(["list"]) == 2
