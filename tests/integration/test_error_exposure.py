"""Clients never see raw upstream error text; operators still get it in the logs.

Provider error messages embed the upstream response body ("The model
`llama-3.1-8b-instant` does not exist or you do not have access to it", account
hints, vendor ids). Users of an app built on Groundwire should see a clean,
fixed message instead; the full text belongs in the logs, found by request ID.
"""

from __future__ import annotations

import json
import logging

import pytest

from conftest import ClientFactory, MidStreamFailure, ScriptedProvider
from groundwire.core.errors import (
    ProviderBadRequestError,
    ProviderNotFoundError,
    ProviderServerError,
)

pytestmark = pytest.mark.integration

# Stands in for an upstream response body.
SECRET = "model `vendor-internal-7b` does not exist for org_9f8e7d"


def _chat(client, **extra):  # type: ignore[no-untyped-def]
    return client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "hi"}], **extra})


def _upstream(cls, provider: str):  # type: ignore[no-untyped-def]
    return cls(f"{provider}: upstream HTTP 4xx: {SECRET}", provider=provider)


def test_rejected_request_hides_the_upstream_body(client_factory: ClientFactory,
                                                  caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="groundwire")
    rejected = _upstream(ProviderBadRequestError, "primary")
    client = client_factory({"primary": ScriptedProvider("primary", rejected),
                             "secondary": ScriptedProvider("secondary")})
    response = _chat(client)
    assert response.status_code == 400
    error = response.json()["error"]
    assert SECRET not in response.text
    assert error["code"] == "upstream_rejected_request"
    assert error["message"].startswith("The AI provider could not process this request")
    assert error["request_id"] == response.headers["x-request-id"]
    # Still available to operators, in the structured log record.
    assert any(SECRET in str(getattr(r, "detail", "")) for r in caplog.records)


def test_all_providers_failed_lists_attempts_without_raw_text(
        client_factory: ClientFactory) -> None:
    client = client_factory({
        "primary": ScriptedProvider("primary", _upstream(ProviderNotFoundError, "primary")),
        "secondary": ScriptedProvider("secondary", _upstream(ProviderServerError, "secondary"))})
    response = _chat(client)
    assert response.status_code == 503
    assert SECRET not in response.text
    attempts = response.json()["error"]["details"]["attempts"]
    codes = [a["error_code"] for a in attempts]
    assert codes[:2] == ["provider_not_found", "provider_server_error"]
    assert all("message" not in a for a in attempts)


def test_failover_metadata_on_success_has_no_raw_text(client_factory: ClientFactory) -> None:
    client = client_factory({
        "primary": ScriptedProvider("primary", _upstream(ProviderNotFoundError, "primary")),
        "secondary": ScriptedProvider("secondary", "backup answer")})
    response = _chat(client)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "backup answer"
    assert SECRET not in response.text
    assert response.json()["groundwire"]["attempts"][0] == {
        "provider": "primary", "outcome": "error", "error_code": "provider_not_found"}


def test_in_band_stream_errors_are_sanitised_too(client_factory: ClientFactory) -> None:
    failure = MidStreamFailure("partial answer", _upstream(ProviderServerError, "primary"))
    client = client_factory({"primary": ScriptedProvider("primary", failure),
                             "secondary": ScriptedProvider("secondary")})
    with client.stream("POST", "/v1/chat/completions", json={
            "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as response:
        body = response.read().decode()
    assert SECRET not in body
    errors = [json.loads(line[6:]) for line in body.splitlines()
              if line.startswith("data: {") and '"error"' in line]
    assert errors and errors[0]["error"]["message"] == "The AI provider had an internal error."
