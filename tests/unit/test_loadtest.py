"""Tests for the load-test harness: profiles, fake upstream and report maths.

The harness produces the numbers quoted in the README, so its arithmetic is
tested like product code. ``run.py`` needs the ``load`` extra (psutil); its
tests are skipped without it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loadtest.fake_upstream import Behaviour, create_app  # noqa: E402
from loadtest.profiles import FIXED, stages  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "issue_refund", "parameters": {}}}]


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def test_plateau_profiles_follow_duration_and_users() -> None:
    assert stages("load", duration_s=60, users=100) == [(60, 100, 50.0)]
    assert stages("soak", duration_s=600, users=80) == [(600, 80, 20.0)]


def test_fixed_profiles_ignore_duration_and_increase_monotonically() -> None:
    assert stages("stress", duration_s=1, users=1) == FIXED["stress"]
    ends = [end for end, _, _ in FIXED["stress"]]
    users = [u for _, u, _ in FIXED["stress"]]
    assert ends == sorted(ends) and users == sorted(users)


@pytest.mark.parametrize(("profile", "duration", "users"), [
    ("nope", 60, 10), ("load", 0, 10), ("chaos", 60, 0)])
def test_profiles_reject_bad_input(profile: str, duration: float, users: int) -> None:
    with pytest.raises(ValueError):
        stages(profile, duration_s=duration, users=users)


# --------------------------------------------------------------------------- #
# Fake upstream
# --------------------------------------------------------------------------- #


@pytest.fixture
def upstream() -> TestClient:
    return TestClient(create_app(Behaviour(latency_ms=0, tokens=3, token_interval_ms=0)))


def _chat(**extra: Any) -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}


def test_fake_upstream_answers_like_openai(upstream: TestClient) -> None:
    body = upstream.post("/v1/chat/completions", json=_chat()).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "the order was"
    assert body["usage"]["completion_tokens"] == 3


def test_fake_upstream_calls_the_offered_tool_once(upstream: TestClient) -> None:
    first = upstream.post("/v1/chat/completions", json=_chat(tools=TOOLS)).json()
    call = first["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "issue_refund"
    assert json.loads(call["function"]["arguments"]) == {"order_id": 7781, "amount": 49.99}
    after_result = _chat(tools=TOOLS)
    after_result["messages"].append({"role": "tool", "content": "{}", "tool_call_id": "x"})
    second = upstream.post("/v1/chat/completions", json=after_result).json()
    assert second["choices"][0]["message"]["content"]  # answers once the tool has run


def test_fake_upstream_streams_sse_to_done(upstream: TestClient) -> None:
    text = upstream.post("/v1/chat/completions", json=_chat(stream=True)).text
    frames = [line for line in text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "data: [DONE]"
    words = [json.loads(f[6:])["choices"][0]["delta"].get("content", "") for f in frames[:-1]]
    assert "".join(words) == "the order was"


def test_fake_upstream_chaos_injects_errors_and_counts_them(upstream: TestClient) -> None:
    assert upstream.post("/admin/chaos", json={"error_rate": 1.0}).json()["error_rate"] == 1.0
    assert upstream.post("/v1/chat/completions", json=_chat()).status_code == 503
    upstream.post("/admin/chaos", json={"error_rate": 0.0})
    assert upstream.post("/v1/chat/completions", json=_chat()).status_code == 200
    stats = upstream.get("/admin/stats").json()
    assert (stats["requests"], stats["errors_injected"]) == (2, 1)


@pytest.mark.parametrize("bad", [{"error_rate": 2}, {"latency_ms": -1}, {"unknown": 1}])
def test_fake_upstream_rejects_invalid_chaos(upstream: TestClient, bad: dict[str, Any]) -> None:
    assert upstream.post("/admin/chaos", json=bad).status_code == 422


# --------------------------------------------------------------------------- #
# Report maths (run.py)
# --------------------------------------------------------------------------- #


@pytest.fixture
def run() -> Any:
    pytest.importorskip("psutil")
    from loadtest import run as module
    return module


def _row(requests: float, failures: float = 0, **extra: float) -> dict[str, float]:
    return {"requests": requests, "failures": failures, "rps": 0, "p50": 0, "p95": 0,
            "p99": 0, "max": 0, "mean": 0} | extra


def test_served_excludes_refusals_and_the_aggregate_row(run: Any) -> None:
    totals = {"chat": _row(90, 1), "agent tool call": _row(10), run.SHED: _row(100),
              "Aggregated": _row(200, 1)}
    assert run.served(totals) == {"requests": 100, "failures": 1, "shed": 100,
                                  "shed_rate": 0.5}


def test_by_stage_skips_ramp_up_and_reports_chat_latency(run: Any) -> None:
    history = []
    for t in range(20):  # one 20 s stage; the first 5 s are ramp-up
        slow = t < 5
        history += [
            {"t": float(t), "name": "chat", "users": 10, "rps": 5, "fails": 0,
             "p50": 900 if slow else 300, "p95": 2000 if slow else 400,
             "p99": 3000 if slow else 500},
            {"t": float(t), "name": run.SHED, "users": 10, "rps": 2, "fails": 0,
             "p50": 5, "p95": 5, "p99": 5},
            {"t": float(t), "name": "Aggregated", "users": 10, "rps": 7, "fails": 1,
             "p50": 0, "p95": 0, "p99": 0}]
    [stage] = run.by_stage(history, [(20, 10, 10)])
    assert stage == {"users": 10, "served_rps": 5.0, "shed_rps": 2.0, "fail_per_s": 1.0,
                     "p50_ms": 300, "p95_ms": 400, "p99_ms": 500}


def _report(run: Any, profile: str, **fields: Any) -> Any:
    return run.Report(profile=profile, machine="test", users=10, duration_s=60,
                      latency_ms=300, **fields)


def test_load_checks_use_mean_overhead_and_shed_rate(run: Any) -> None:
    report = _report(run, "load", totals={"chat": _row(1000, mean=314.0)},
                     baseline={"chat": _row(1000, mean=302.0)})
    run.evaluate(report, [])
    checks = {c.name: c for c in report.checks}
    assert checks["gateway overhead, mean (ms)"].measured == pytest.approx(12.0)
    assert report.passed


def test_load_fails_when_requests_are_shed_at_normal_load(run: Any) -> None:
    report = _report(run, "load", totals={"chat": _row(990), run.SHED: _row(10)})
    run.evaluate(report, [])
    assert not report.passed


def test_stress_fails_when_a_stage_tail_is_too_slow(run: Any) -> None:
    report = _report(run, "stress", totals={"chat": _row(1000)},
                     stages=[{"p99_ms": 400.0}, {"p99_ms": 3000.0}])
    run.evaluate(report, [])
    assert [c.passed for c in report.checks] == [True, False]


def test_render_marks_failed_checks(run: Any) -> None:
    report = _report(run, "stress", totals={"chat": _row(1000)}, stages=[{
        "users": 600, "served_rps": 120.0, "shed_rps": 390.0, "fail_per_s": 0.0,
        "p50_ms": 390.0, "p95_ms": 900.0, "p99_ms": 3000.0}])
    run.evaluate(report, [])
    text = run.render(report)
    assert text.startswith("# Load test `stress`: FAILED")
    assert "**FAIL**" in text
    assert "| 600 | 120.0 | 390.0 |" in text


def test_leak_check_ignores_a_cache_filling_up_but_catches_steady_growth(run: Any) -> None:
    def samples(rss: list[float]) -> list[Any]:
        return [run.Sample(t=float(i), rss_mb=v, cpu_percent=50.0) for i, v in enumerate(rss)]

    filling = [70 + min(i, 50) * 0.3 for i in range(200)]  # grows, then plateaus at 85 MB
    leaking = [70 + i * 0.1 for i in range(200)]  # grows the whole run
    assert run.summarise_resources(samples(filling))["rss_growth_pct"] == 0.0
    assert run.summarise_resources(samples(leaking))["rss_growth_pct"] > 5
