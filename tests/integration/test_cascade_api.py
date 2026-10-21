"""End-to-end tests of cascade routes through the HTTP API."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import ClientFactory, ScriptedProvider, make_config
from groundwire.core.config import GatewayConfig

pytestmark = pytest.mark.integration

DOCS = ["The refund for order 7781 is 49.99 USD and was issued on 2 May."]
GOOD = "The refund for order 7781 is 49.99 USD."
BAD = "The refund for order 7781 is 89.99 USD."


def _config(**extra: Any) -> GatewayConfig:
    providers = [{"name": n, "type": "mock", "model": "m"}
                 for n in ("primary", "secondary", "cheap", "strong")]
    return make_config(providers=providers,
                       cascades={"smart": {"cheap": ["cheap"], "strong": ["strong"]}}, **extra)


def _providers(cheap: str = BAD, strong: str = GOOD) -> dict[str, ScriptedProvider]:
    return {"primary": ScriptedProvider("primary", "p"),
            "secondary": ScriptedProvider("secondary", "s"),
            "cheap": ScriptedProvider("cheap", cheap), "strong": ScriptedProvider("strong", strong)}


def _body(*, docs: list[str] | None = DOCS, mode: str = "flag", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "smart",
                            "messages": [{"role": "user", "content": "How much is the refund?"}],
                            **extra}
    if docs is not None:
        body["groundwire"] = {"grounding": {"context": docs, "mode": mode}}
    return body


def _sse(text: str) -> list[Any]:
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_escalation_is_visible_in_the_response(client_factory: ClientFactory) -> None:
    providers = _providers()
    client = client_factory(providers, _config())
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 200
    data = response.json()
    meta = data["groundwire"]
    assert data["choices"][0]["message"]["content"] == GOOD
    assert response.headers["x-groundwire-provider"] == "strong"
    assert meta["cascade"]["tier"] == "strong" and meta["cascade"]["reason"] == "low_support"
    assert [a["tier"] for a in meta["attempts"]] == ["cheap", "strong"]
    assert meta["grounding"]["verdict"] == "passed"  # the strong answer was checked too


def test_passing_cheap_answer_is_checked_once(client_factory: ClientFactory) -> None:
    providers = _providers(cheap=GOOD, strong="unused")
    client = client_factory(providers, _config())
    meta = client.post("/v1/chat/completions", json=_body()).json()["groundwire"]
    assert meta["cascade"]["tier"] == "cheap" and providers["strong"].calls == 0
    # One cascade check, reused for the client's grounding report (not run twice).
    text = client.get("/metrics").text
    assert 'groundwire_grounding_checks_total{verdict="passed"} 1.0' in text
    assert "groundwire_cascade_check_duration_seconds_count" in text


def test_block_mode_still_applies_after_escalation(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(cheap=BAD, strong=BAD), _config())
    response = client.post("/v1/chat/completions", json=_body(mode="block"))
    assert response.status_code == 422


def test_no_documents_goes_to_the_strong_tier(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(), _config())
    meta = client.post("/v1/chat/completions", json=_body(docs=None)).json()["groundwire"]
    assert meta["cascade"]["reason"] == "no_context" and meta["provider"] == "strong"


def test_cascade_answers_are_cached(client_factory: ClientFactory) -> None:
    providers = _providers()
    client = client_factory(providers, _config(cache={"enabled": True}))
    client.post("/v1/chat/completions", json=_body())
    hit = client.post("/v1/chat/completions", json=_body()).json()["groundwire"]
    assert hit["cache"]["status"] == "hit" and hit["provider"] == "cache"
    assert providers["cheap"].calls == 1 and providers["strong"].calls == 1


def test_streaming_cascade_is_buffered_then_replayed(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(), _config())
    response = client.post("/v1/chat/completions", json=_body(stream=True, mode="block"))
    assert response.status_code == 200
    assert response.headers["x-groundwire-provider"] == "strong"
    chunks = _sse(response.text)
    content = "".join(c["choices"][0]["delta"].get("content", "")
                      for c in chunks if c.get("choices"))
    assert content == GOOD
    assert chunks[-1]["groundwire"]["cascade"]["reason"] == "low_support"
    assert response.text.rstrip().endswith("data: [DONE]")


def test_streaming_block_mode_failure_is_an_http_error(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(cheap=BAD, strong=BAD), _config())
    response = client.post("/v1/chat/completions", json=_body(stream=True, mode="block"))
    assert response.status_code == 422  # nothing was streamed before the check


def test_block_mode_with_streaming_is_still_rejected_on_plain_routes(
        client_factory: ClientFactory) -> None:
    client = client_factory(_providers(), _config())
    body = {**_body(stream=True, mode="block"), "model": "default"}
    assert client.post("/v1/chat/completions", json=body).status_code == 400


def test_models_and_readiness_include_cascades(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(), _config())
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["smart"]["cascade"] == {"cheap": ["cheap"], "strong": ["strong"]}
    assert client.get("/readyz").status_code == 200


def test_cascade_metrics(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(), _config())
    client.post("/v1/chat/completions", json=_body())
    text = client.get("/metrics").text
    assert ('groundwire_cascade_requests_total{cascade="smart",reason="low_support",'
            'tier="strong"} 1.0') in text
    assert 'groundwire_requests_total{provider="strong",route="smart",status="success"' in text
