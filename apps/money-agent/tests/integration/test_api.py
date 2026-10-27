"""The HTTP API, end to end with in-memory storage and a scripted model."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import OTHER_USER, USER, ScriptedLLM, answer, tool_call
from money_agent.domain.ports import LLMReply


def _upload(client: TestClient, data: bytes, name: str = "statement.csv") -> dict:  # type: ignore[type-arg]
    response = client.post("/v1/me/statements", files={"file": (name, data, "text/csv")})
    assert response.status_code == 201, response.text
    return response.json()  # type: ignore[no-any-return]


def test_health_and_page(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    page = client.get("/")
    assert page.status_code == 200 and "Groundwire Money" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert client.get("/static/app.js").status_code == 200


def test_requests_need_an_identity(client: TestClient) -> None:
    for headers in ({"X-User-Id": ""}, {"X-User-Id": "short"}, {"X-User-Id": "bad id!" * 4}):
        response = client.get("/v1/me/memories", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthenticated"


def test_upload_summary_and_pagination(client: TestClient, sample_bytes: bytes) -> None:
    report = _upload(client, sample_bytes)
    assert (report["imported"], report["duplicates"], report["rows_read"]) == (53, 0, 53)
    assert _upload(client, sample_bytes)["duplicates"] == 53

    summary = client.get("/v1/me/summary").json()
    assert summary["period"] == "2026-09"
    assert summary["totals"]["saved"] == "47027.35"
    assert client.get("/v1/me/summary?month=2026-07").json()["period"] == "2026-07"

    items, cursor = [], None
    while True:
        params = {"limit": 20, **({"cursor": cursor} if cursor else {})}
        page = client.get("/v1/me/transactions", params=params).json()
        items += page["items"]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(items) == 53 and len({i["id"] for i in items}) == 53
    food = client.get("/v1/me/transactions?category=food_dining&limit=200").json()["items"]
    assert len(food) == 18 and food[0]["category_label"] == "Food & dining"
    trend = client.get("/v1/me/trend?months=3").json()
    assert [m["month"] for m in trend] == ["2026-07", "2026-08", "2026-09"]


def test_users_never_see_each_other(client: TestClient, sample_bytes: bytes) -> None:
    _upload(client, sample_bytes)
    other = {"X-User-Id": OTHER_USER}
    assert client.get("/v1/me/transactions", headers=other).json()["items"] == []
    response = client.get("/v1/me/summary", headers=other)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_transactions"
    assert client.get("/v1/me/overview", headers=other).json() == {
        "has_data": False, "first_date": None, "last_date": None}
    assert client.get("/v1/me/overview").json() == {
        "has_data": True, "first_date": "2026-07-01", "last_date": "2026-09-30"}


@pytest.mark.parametrize(("path", "code"), [
    ("/v1/me/transactions?cursor=garbage", "invalid_input"),
    ("/v1/me/transactions?limit=0", "invalid_input"),
    ("/v1/me/transactions?start=2026-09-10&end=2026-09-01", "invalid_input"),
    ("/v1/me/summary?month=2026-9", "invalid_input"),
    ("/v1/me/trend?months=99", "invalid_input"),
])
def test_bad_inputs_get_clean_errors(client: TestClient, path: str, code: str) -> None:
    response = client.get(path)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == code and error["request_id"] == response.headers["x-request-id"]


def test_bad_statements_are_explained(client: TestClient) -> None:
    response = client.post("/v1/me/statements",
                           files={"file": ("x.csv", b"hello,world\n1,2\n", "text/csv")})
    assert response.status_code == 422
    assert "CSV export" in response.json()["error"]["message"]


def test_oversized_uploads_stop_early(llm: ScriptedLLM) -> None:
    from conftest import FrozenClock
    from money_agent.api.app import create_app
    from money_agent.container import build_container
    from money_agent.core.config import load_settings

    small = load_settings(storage="memory", max_statement_bytes=2_000)
    app = create_app(build_container(small, llm=llm, clock=FrozenClock()))
    with TestClient(app, headers={"X-User-Id": USER}) as client:
        response = client.post("/v1/me/statements",
                               files={"file": ("big.csv", b"x" * 10_000, "text/csv")})
    assert response.status_code == 413
    assert response.json()["error"]["details"] == {"max_bytes": 2_000}


def test_chat_with_tools_memory_and_history(client: TestClient, llm: ScriptedLLM,
                                            sample_bytes: bytes) -> None:
    _upload(client, sample_bytes)
    llm.queue(tool_call("get_spending_summary"),
              tool_call("remember", {"kind": "goal",
                                     "content": "Saving ₹1,20,000 for a bike by June 2027"}),
              answer("In September you saved ₹47,027.35. I've remembered your bike goal."))
    first = client.post("/v1/me/chat", json={"message": "How did I do? Also I'm saving for "
                                                        "a bike, ₹1.2 lakh by June 2027."})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["verification"] == {"status": "verified", "problems": [], "corrected": False}
    assert [t["name"] for t in body["tools"]] == ["get_spending_summary", "remember"]

    # The tool result the model saw holds the exact figures.
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["saved"] == "47027.35"

    memories = client.get("/v1/me/memories").json()
    assert [m["content"] for m in memories] == ["Saving ₹1,20,000 for a bike by June 2027"]

    llm.queue(answer("Your bike goal is ₹1,20,000 by June 2027.", "skipped"))
    second = client.post("/v1/me/chat", json={"message": "What's my goal again?",
                                              "conversation_id": body["conversation_id"]})
    assert second.json()["memories_used"] == 1
    sent = llm.calls[-1]["messages"]
    assert "bike by June 2027" in sent[0]["content"]          # memory in the system prompt
    assert sent[1]["content"].startswith("How did I do?")     # history replayed
    assert sent[2]["role"] == "assistant"

    history = client.get(f"/v1/me/conversations/{body['conversation_id']}/messages").json()
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[1]["meta"]["verification"]["status"] == "verified"

    memory_id = memories[0]["id"]
    assert client.delete(f"/v1/me/memories/{memory_id}").status_code == 204
    assert client.delete(f"/v1/me/memories/{memory_id}").status_code == 404


def test_chat_errors_are_user_safe(client: TestClient, llm: ScriptedLLM) -> None:
    other = client.post("/v1/me/chat", json={
        "message": "hi", "conversation_id": "00000000-0000-0000-0000-000000000000"})
    assert other.status_code == 404
    assert client.post("/v1/me/chat", json={"message": ""}).status_code == 400
    assert client.post("/v1/me/chat", json={"message": "x" * 4001}).status_code == 400
    assert client.post("/v1/me/chat", json={"message": "hi", "user_id": "x"}).status_code == 400

    llm.queue(LLMReply(content=""))
    failed = client.post("/v1/me/chat", json={"message": "hi"})
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "agent_error"


def test_unexpected_errors_hide_internals(client: TestClient, llm: ScriptedLLM) -> None:
    class Exploding(ScriptedLLM):
        async def complete(self, *args: object, **kwargs: object) -> LLMReply:
            raise RuntimeError("password=hunter2 at db.py:12")

    client.app.state.container.chat._runner._llm = Exploding()  # type: ignore[attr-defined]
    response = TestClient(client.app, raise_server_exceptions=False,
                          headers={"X-User-Id": USER}).post("/v1/me/chat",
                                                            json={"message": "hi"})
    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert response.json()["error"]["code"] == "internal_error"


def test_request_ids_are_kept_or_made(client: TestClient) -> None:
    kept = client.get("/healthz", headers={"X-Request-ID": "trace-abc12345"})
    assert kept.headers["x-request-id"] == "trace-abc12345"
    made = client.get("/healthz", headers={"X-Request-ID": "bad id <script>"})
    assert len(made.headers["x-request-id"]) == 32
