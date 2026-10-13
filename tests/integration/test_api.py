"""End-to-end tests of the HTTP API with fake providers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import ClientFactory, MidStreamFailure, ScriptedProvider, make_config
from groundwire.core.errors import ProviderBadRequestError, ProviderServerError

pytestmark = pytest.mark.integration

CONTEXT = ["The refund for order 7781 is 49.99 USD and was issued on 2 May."]


def _providers(primary: Any = "Hello from primary", secondary: Any = "Hello from secondary"
               ) -> dict[str, ScriptedProvider]:
    p = primary if isinstance(primary, tuple) else (primary,)
    s = secondary if isinstance(secondary, tuple) else (secondary,)
    return {"primary": ScriptedProvider("primary", *p),
            "secondary": ScriptedProvider("secondary", *s)}


def _body(**extra: Any) -> dict[str, Any]:
    return {"model": "default", "messages": [{"role": "user", "content": "hi"}], **extra}


def _sse_payloads(text: str) -> list[Any]:
    payloads = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #


def test_chat_completion_openai_shape(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"] == {"role": "assistant",
                                             "content": "Hello from primary"}
    assert data["usage"]["total_tokens"] == 8
    assert data["groundwire"]["provider"] == "primary"
    assert response.headers["x-groundwire-provider"] == "primary"
    assert response.headers["x-request-id"].startswith("gw-")


def test_request_id_is_propagated(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    response = client.post("/v1/chat/completions", json=_body(),
                           headers={"X-Request-ID": "trace-123"})
    assert response.headers["x-request-id"] == "trace-123"


def test_failover_is_visible_in_metadata(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary=ProviderServerError("down", provider="primary")))
    data = client.post("/v1/chat/completions", json=_body()).json()
    assert data["groundwire"]["provider"] == "secondary"
    assert [a["outcome"] for a in data["groundwire"]["attempts"]] == ["error", "success"]


def test_all_providers_down_returns_503(client_factory: ClientFactory) -> None:
    err = ProviderServerError("down", provider="x")
    client = client_factory(_providers(primary=err, secondary=err))
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "all_providers_failed"


def test_upstream_bad_request_returns_400(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary=ProviderBadRequestError("ctx", provider="p")))
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_rejected_request"


def test_deadline_returns_504(client_factory: ClientFactory) -> None:
    slow = {"primary": ScriptedProvider("primary", "late", delay_s=2),
            "secondary": ScriptedProvider("secondary", "late", delay_s=2)}
    client = client_factory(slow, make_config(request_timeout_s=0.2))
    response = client.post("/v1/chat/completions", json=_body())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "gateway_timeout"


@pytest.mark.parametrize(("body", "status", "code"), [
    ({"messages": []}, 400, "invalid_request_error"),
    ({"messages": [{"role": "robot", "content": "x"}]}, 400, "invalid_request_error"),
    ({"messages": [{"role": "user", "content": "x"}], "temperature": 5}, 400,
     "invalid_request_error"),
    ({"model": "gpt-9", "messages": [{"role": "user", "content": "x"}]}, 404, "model_not_found"),
])
def test_validation_errors(client_factory: ClientFactory, body: dict[str, Any], status: int,
                           code: str) -> None:
    client = client_factory(_providers())
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_oversized_body_returns_413(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    huge = "x" * (5 * 1024 * 1024)
    response = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": huge}]})
    assert response.status_code == 413


def test_unknown_openai_params_are_ignored(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    response = client.post("/v1/chat/completions",
                           json=_body(n=1, presence_penalty=0.1, logprobs=False))
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


def test_grounding_flag_mode_attaches_report(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary="The refund for order 7781 is 89.99 USD."))
    body = _body(groundwire={"grounding": {"context": CONTEXT, "mode": "flag"}})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    grounding = response.json()["groundwire"]["grounding"]
    assert grounding["verdict"] == "failed"
    assert grounding["claims"][0]["supported"] is False


def test_grounding_block_mode_returns_422(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary="The refund for order 7781 is 89.99 USD."))
    body = _body(groundwire={"grounding": {"context": CONTEXT, "mode": "block"}})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "grounding_violation"
    assert error["details"]["grounding"]["verdict"] == "failed"


def test_grounded_answer_passes_in_block_mode(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary="The refund for order 7781 is 49.99 USD."))
    body = _body(groundwire={"grounding": {"context": CONTEXT, "mode": "block"}})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["groundwire"]["grounding"]["verdict"] == "passed"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_streaming_sse_format(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary="one two three"))
    with client.stream("POST", "/v1/chat/completions", json=_body(stream=True)) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-groundwire-provider"] == "primary"
        payloads = _sse_payloads(response.read().decode())

    assert payloads[-1] == "[DONE]"
    chunks = payloads[:-1]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "one two three"
    assert len({c["id"] for c in chunks}) == 1  # one stream id
    assert all(c["model"] == "default" for c in chunks)


def test_streaming_fails_over_before_first_byte(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary=ProviderServerError("down", provider="p"),
                                       secondary="backup stream"))
    with client.stream("POST", "/v1/chat/completions", json=_body(stream=True)) as response:
        assert response.headers["x-groundwire-provider"] == "secondary"
        response.read()


def test_streaming_mid_stream_error_is_sent_in_band(client_factory: ClientFactory) -> None:
    failing = MidStreamFailure("partial text", ProviderServerError("died", provider="primary"))
    client = client_factory(_providers(primary=failing))
    with client.stream("POST", "/v1/chat/completions", json=_body(stream=True)) as response:
        payloads = _sse_payloads(response.read().decode())
    assert any(isinstance(p, dict) and "error" in p for p in payloads)
    assert payloads[-1] == "[DONE]"


def test_streaming_all_down_returns_http_error(client_factory: ClientFactory) -> None:
    err = ProviderServerError("down", provider="x")
    client = client_factory(_providers(primary=err, secondary=err))
    response = client.post("/v1/chat/completions", json=_body(stream=True))
    assert response.status_code == 503


def test_streaming_grounding_flag_appends_report(client_factory: ClientFactory) -> None:
    client = client_factory(_providers(primary="The refund for order 7781 is 49.99 USD."))
    body = _body(stream=True, groundwire={"grounding": {"context": CONTEXT}})
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        payloads = _sse_payloads(response.read().decode())
    report_chunk = payloads[-2]
    assert report_chunk["choices"] == []
    assert report_chunk["groundwire"]["grounding"]["verdict"] == "passed"


def test_streaming_with_block_mode_is_rejected(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    body = _body(stream=True, groundwire={"grounding": {"context": CONTEXT, "mode": "block"}})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Operational endpoints
# --------------------------------------------------------------------------- #


def test_models_lists_routes(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    data = client.get("/v1/models").json()
    assert [m["id"] for m in data["data"]] == ["default"]


def test_health_and_readiness(client_factory: ClientFactory) -> None:
    err = ProviderServerError("down", provider="x")
    client = client_factory(_providers(primary=err, secondary=err))
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200

    # failure_threshold=2 in the test config: two failed requests open both circuits.
    for _ in range(2):
        client.post("/v1/chat/completions", json=_body())
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["unavailable_routes"] == ["default"]
    assert client.get("/healthz").status_code == 200  # liveness unaffected


def test_metrics_endpoint(client_factory: ClientFactory) -> None:
    client = client_factory(_providers())
    client.post("/v1/chat/completions", json=_body())
    text = client.get("/metrics").text
    assert "groundwire_requests_total" in text
    assert 'provider="primary"' in text
    assert "groundwire_circuit_state" in text


def test_providers_are_closed_on_shutdown(client_factory: ClientFactory) -> None:
    providers = _providers()
    client = client_factory(providers)
    client.__exit__(None, None, None)
    assert all(p.closed for p in providers.values())
