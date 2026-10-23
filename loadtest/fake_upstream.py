"""Fake OpenAI-compatible model server for load and chaos testing.

Load tests must measure the *gateway*, not a real model: a real provider adds
seconds of variable latency, costs money and rate-limits the test. This server
answers ``POST /v1/chat/completions`` like OpenAI does (JSON or SSE stream),
with a configurable, deterministic delay, so any time above that delay is
gateway overhead.

Behaviour (all configurable at runtime, so chaos can be injected mid-test):

* ``latency_ms``: time to first token;
* ``tokens`` / ``token_interval_ms``: answer length and streaming pace;
* ``error_rate``: share of requests answered with HTTP 503;
* ``extra_latency_ms``: added delay, to simulate a slow provider.

Admin endpoints (used by :mod:`loadtest.run`):

* ``POST /admin/chaos`` with any of the fields above;
* ``GET /admin/stats``: requests served and errors injected;
* ``GET /healthz``.

Run::

    python -m loadtest.fake_upstream --port 9101 --latency-ms 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("fake_upstream")

WORDS = ("the order was refunded and the customer was told the amount and the date "  # noqa: SIM905
         "it will arrive on their card within five business days").split()


@dataclass
class Behaviour:
    """Current behaviour of the fake model (mutable at runtime via /admin/chaos).

    Attributes:
        latency_ms: Time to first token.
        tokens: Words in each answer.
        token_interval_ms: Delay between streamed words.
        error_rate: Share of requests answered with HTTP 503.
        extra_latency_ms: Extra delay added to every request (slow provider).
    """

    latency_ms: float = 300.0
    tokens: int = 40
    token_interval_ms: float = 5.0
    error_rate: float = 0.0
    extra_latency_ms: float = 0.0


class ChaosUpdate(BaseModel):
    """Body of ``POST /admin/chaos`` (only the fields sent are changed)."""

    model_config = ConfigDict(extra="forbid")

    latency_ms: float | None = Field(default=None, ge=0, le=120_000)
    tokens: int | None = Field(default=None, ge=1, le=4_000)
    token_interval_ms: float | None = Field(default=None, ge=0, le=10_000)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    extra_latency_ms: float | None = Field(default=None, ge=0, le=120_000)


@dataclass
class Stats:
    """Counters for ``GET /admin/stats``."""

    requests: int = 0
    errors_injected: int = 0
    streams: int = 0
    tool_calls: int = 0


def _wants_tool_call(body: dict[str, Any]) -> bool:
    """Answer with a tool call when tools are offered and no tool result came back yet."""
    messages = body.get("messages") or []
    return bool(body.get("tools")) and bool(messages) and messages[-1].get("role") == "user"


def _tool_call() -> dict[str, Any]:
    """A small, policy-allowed refund call."""
    return {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": "issue_refund",
                         "arguments": json.dumps({"order_id": 7781, "amount": 49.99})}}


def _text(tokens: int) -> list[str]:
    """Deterministic answer words."""
    return [WORDS[i % len(WORDS)] for i in range(tokens)]


def create_app(behaviour: Behaviour | None = None, *, seed: int = 7) -> FastAPI:
    """Build the fake upstream app.

    Args:
        behaviour: Initial behaviour.
        seed: Seed for error injection (same seed, same error pattern).

    Returns:
        The app.
    """
    state = behaviour or Behaviour()
    stats = Stats()
    rng = random.Random(seed)  # noqa: S311 - test traffic, not crypto
    app = FastAPI(title="fake-upstream")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/stats")
    async def get_stats() -> dict[str, Any]:
        return {**asdict(stats), "behaviour": asdict(state)}

    @app.post("/admin/chaos")
    async def chaos(update: ChaosUpdate) -> dict[str, Any]:
        for name, value in update.model_dump(exclude_none=True).items():
            setattr(state, name, value)
        logger.info("behaviour changed: %s", asdict(state))
        return asdict(state)

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        stats.requests += 1
        if state.error_rate and rng.random() < state.error_rate:
            stats.errors_injected += 1
            await asyncio.sleep(0.005)
            return JSONResponse(status_code=503, content={"error": {
                "message": "injected failure", "type": "server_error"}})
        await asyncio.sleep((state.latency_ms + state.extra_latency_ms) / 1000)
        model = str(body.get("model", "fake"))
        tool = _wants_tool_call(body)
        stats.tool_calls += int(tool)
        if body.get("stream"):
            stats.streams += 1
            return StreamingResponse(_stream(model, tool), media_type="text/event-stream")
        words = _text(state.tokens)
        message: dict[str, Any] = {"role": "assistant",
                                   "content": None if tool else " ".join(words)}
        if tool:
            message["tool_calls"] = [_tool_call()]
        prompt = sum(len(str(m.get("content") or "")) for m in body.get("messages", [])) // 4
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": len(words),
                      "total_tokens": prompt + len(words)}})

    async def _stream(model: str, tool: bool) -> AsyncIterator[bytes]:
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

        def frame(delta: dict[str, Any], finish: str | None = None) -> bytes:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        if tool:
            call = _tool_call()
            yield frame({"role": "assistant", "tool_calls": [{"index": 0, **call}]})
            yield frame({}, "tool_calls")
        else:
            for i, word in enumerate(_text(state.tokens)):
                yield frame({"role": "assistant", "content": word} if i == 0
                            else {"content": " " + word})
                if state.token_interval_ms:
                    await asyncio.sleep(state.token_interval_ms / 1000)
            yield frame({}, "stop")
        yield b"data: [DONE]\n\n"

    return app


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Fake OpenAI-compatible upstream for load tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--latency-ms", type=float, default=300.0)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--token-interval-ms", type=float, default=5.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s fake_upstream %(message)s")
    app = create_app(Behaviour(latency_ms=args.latency_ms, tokens=args.tokens,
                               token_interval_ms=args.token_interval_ms,
                               error_rate=args.error_rate))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
