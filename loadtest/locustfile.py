# Locust ships incomplete type hints (e.g. failure(msg) is typed as taking no argument).
# mypy: disable-error-code="no-untyped-call,call-arg"
"""Locust traffic for Groundwire load, stress, spike, chaos and soak tests.

Two user classes send the *same* request mix:

* ``GatewayUser``: through Groundwire (the system under test);
* ``DirectUser``: straight to the fake upstream (the baseline). The difference
  between the two runs is the gateway's own overhead.

Request mix (weights), chosen to look like a real app:

=========================  ======  ==============================================
Task                       Weight  What it exercises
=========================  ======  ==============================================
chat                       50      unique question (cache miss), PII redaction,
                                   grounding check on half of them
chat (cached)              10      one of 20 repeated questions (cache hits)
chat (stream)              25      SSE streaming, PII restore across chunks
agent tool call            15      tool calling + tool policy + JSON PII handling
=========================  ======  ==============================================

Refusals by the gateway's load shedding (HTTP 503 ``server_overloaded``) are
reported under their own name, ``shed (503)``, and count as correct
behaviour: refusing excess load at once is the design. Real errors stay
failures, and the latency rows contain only requests that were served.

For streamed requests Locust's response time is the time to the first byte
(headers), which is the latency a streaming user feels; the stream is still
read to the end and checked.

The load shape comes from ``LOAD_PROFILE`` (``load``, ``stress``, ``spike``,
``chaos``, ``soak``); durations scale with ``LOAD_DURATION`` (seconds) where a
profile is a single plateau.
"""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from typing import Any

from locust import LoadTestShape, between, task
from locust.contrib.fasthttp import FastHttpUser

from loadtest.profiles import stages  # needs the repo root on PYTHONPATH (run.py sets it)

logger = logging.getLogger("loadtest")

SHED_NAME = "shed (503)"

PROFILE = os.environ.get("LOAD_PROFILE", "load")
DURATION = float(os.environ.get("LOAD_DURATION", "60"))
USERS = int(os.environ.get("LOAD_USERS", "100"))

CONTEXT = ["Order 7781 (wireless mouse, 49.99 USD) was delivered on 12 October. "
           "Refunds are issued to the original card within five business days."]
REPEATED = [f"What is the refund policy for item category {i}?" for i in range(20)]
TOOLS = [{"type": "function", "function": {
    "name": "issue_refund", "description": "Refund an order.",
    "parameters": {"type": "object", "required": ["order_id", "amount"],
                   "properties": {"order_id": {"type": "integer"},
                                  "amount": {"type": "number"}}}}}]


def chat_body(question: str, *, grounded: bool = False, stream: bool = False,
              tools: bool = False, gateway: bool = True) -> dict[str, Any]:
    """Build an OpenAI-format request body.

    Args:
        question: User message.
        grounded: Ask the gateway to check the answer against ``CONTEXT``.
        stream: Request SSE streaming.
        tools: Offer the refund tool (the fake model then calls it).
        gateway: Include Groundwire-only fields (the direct baseline omits them).

    Returns:
        The JSON body.
    """
    body: dict[str, Any] = {"model": "default" if gateway else "fake-model",
                            "messages": [{"role": "user", "content": question}],
                            "temperature": 0, "stream": stream}
    if tools:
        body["tools"] = TOOLS
    if grounded and gateway:
        body["groundwire"] = {"grounding": {"context": CONTEXT, "mode": "flag"}}
    return body


class _Traffic(FastHttpUser):
    """Shared request mix; subclasses set the target and path."""

    abstract = True
    wait_time = between(0.5, 1.5)  # think time between a user's requests
    path = "/v1/chat/completions"
    gateway = True

    @staticmethod
    def _shed(resp: Any) -> bool:
        """Report a load-shedding refusal under :data:`SHED_NAME` as a success.

        Args:
            resp: Locust response context.

        Returns:
            True if the response was a refusal (and has been reported).
        """
        if resp.status_code == 503 and "server_overloaded" in (resp.text or ""):
            resp.request_meta["name"] = SHED_NAME
            resp.success()
            return True
        return False

    def _post(self, name: str, body: dict[str, Any], *, expect_tool: bool = False) -> None:
        with self.client.post(self.path, json=body, name=name, catch_response=True) as resp:
            if self._shed(resp):
                return
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                message = resp.json()["choices"][0]["message"]
            except (ValueError, KeyError, IndexError, TypeError):
                resp.failure("malformed body")
                return
            if expect_tool and not message.get("tool_calls"):
                resp.failure("expected a tool call")
            elif not expect_tool and not message.get("content"):
                resp.failure("empty answer")
            else:
                resp.success()

    @task(50)
    def chat(self) -> None:
        """Unique question with an email in it (PII path), grounded half the time.

        The unique part sits outside the email on purpose: PII redaction turns
        every email into ``[EMAIL_1]``, so questions that differ only in the
        email are identical after redaction and are (correctly) cache hits.
        """
        question = (f"Customer rahul@example.com asks about ticket {uuid.uuid4().hex[:8]}: "
                    "where is order 7781?")
        self._post("chat", chat_body(question, grounded=random.random() < 0.5,  # noqa: S311
                                     gateway=self.gateway))

    @task(10)
    def chat_cached(self) -> None:
        """A popular question (cache hit after the first time)."""
        self._post("chat (cached)", chat_body(random.choice(REPEATED),  # noqa: S311
                                              gateway=self.gateway))

    @task(25)
    def chat_stream(self) -> None:
        """Streamed answer; the whole stream must arrive and end with [DONE]."""
        body = chat_body(f"Explain the refund for ticket {uuid.uuid4().hex[:8]}",
                         stream=True, gateway=self.gateway)
        with self.client.post(self.path, json=body, name="chat (stream)", stream=True,
                              catch_response=True) as resp:
            if self._shed(resp):
                return
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            text = resp.text or ""
            if '"error"' in text:
                resp.failure("in-band stream error")
            elif "data: [DONE]" not in text:
                resp.failure("stream ended early")
            else:
                resp.success()

    @task(15)
    def agent(self) -> None:
        """Agent turn: the model requests a refund tool call that the policy allows."""
        self._post("agent tool call", chat_body("Refund order 7781 in full", tools=True,
                                                gateway=self.gateway), expect_tool=True)


class GatewayUser(_Traffic):
    """Traffic through Groundwire (host: the gateway)."""


class DirectUser(_Traffic):
    """The same traffic straight to the fake upstream (host: the upstream): the baseline."""

    gateway = False


# --------------------------------------------------------------------------- #
# Load shapes
# --------------------------------------------------------------------------- #

STAGES = stages(PROFILE, duration_s=DURATION, users=USERS)


class ProfileShape(LoadTestShape):
    """Runs the stages of ``LOAD_PROFILE``."""

    def tick(self) -> tuple[int, float] | None:
        """Return ``(users, spawn_rate)`` for now, or ``None`` to stop."""
        elapsed: float = self.get_run_time()
        for end, users, rate in STAGES:
            if elapsed < end:
                return users, rate
        return None


def describe() -> str:
    """Human-readable summary of the active profile (for reports)."""
    return json.dumps({"profile": PROFILE, "stages": STAGES})
