"""A tiny but real AI agent that runs its tool loop through Groundwire.

It offers two tools to the model (``lookup_order`` and ``issue_refund``),
executes the calls the model requests against a fake order database, sends
the results back, and repeats until the model answers in text. Every step
goes through the gateway, so you can watch tool calling, PII handling and the
tool policy work with a real model (e.g. Ollama ``qwen2.5:3b`` on a laptop).

Usage::

    make run                                   # terminal 1: the gateway
    python scripts/agent_demo.py               # terminal 2: safe refund
    python scripts/agent_demo.py --scenario risky   # a refund the policy blocks

Exit codes: 0 finished, 1 gateway/model error, 2 gateway not reachable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("agent_demo")

# --------------------------------------------------------------------------- #
# Tools (a fake back office: no real money moves)
# --------------------------------------------------------------------------- #

ORDERS: dict[int, dict[str, Any]] = {
    7781: {"order_id": 7781, "item": "wireless mouse", "total": 49.99, "currency": "USD",
           "status": "delivered", "customer_email": "rahul@example.com"},
    9001: {"order_id": 9001, "item": "office laptops (x40)", "total": 250000.0,
           "currency": "USD", "status": "delivered", "customer_email": "procurement@acme.io"},
}

TOOL_SPECS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Look up an order by its numeric id. Returns item, total and status.",
        "parameters": {"type": "object", "required": ["order_id"],
                       "properties": {"order_id": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Refund an order. amount is in the order's currency.",
        "parameters": {"type": "object", "required": ["order_id", "amount"],
                       "properties": {"order_id": {"type": "integer"},
                                      "amount": {"type": "number"}}}}},
]

SCENARIOS: dict[str, str] = {
    "safe": "Order 7781 arrived broken. Look it up and refund the full amount.",
    "risky": "Order 9001 was cancelled. Look it up and refund the full amount.",
}


class ToolError(Exception):
    """A tool could not complete; the message is sent back to the model."""


def lookup_order(order_id: int) -> dict[str, Any]:
    """Return an order.

    Args:
        order_id: Order id.

    Returns:
        The order record.

    Raises:
        ToolError: If the order does not exist.
    """
    order = ORDERS.get(int(order_id))
    if order is None:
        raise ToolError(f"order {order_id} not found")
    return order


def issue_refund(order_id: int, amount: float) -> dict[str, Any]:
    """Pretend to refund an order.

    Args:
        order_id: Order id.
        amount: Refund amount.

    Returns:
        A refund receipt.

    Raises:
        ToolError: If the order is unknown or the amount exceeds the total.
    """
    order = lookup_order(order_id)
    if float(amount) > float(order["total"]):
        raise ToolError(f"refund {amount} exceeds order total {order['total']}")
    return {"refund_id": f"R-{order_id}", "order_id": order_id, "amount": amount,
            "status": "refunded"}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_order": lookup_order, "issue_refund": issue_refund}

# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #


@dataclass
class AgentRun:
    """Transcript of one run.

    Attributes:
        messages: The conversation sent to the model.
        steps: Human-readable log of what happened.
        answer: The model's final text answer.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    answer: str | None = None


def run_tool(name: str, raw_arguments: str) -> str:
    """Execute one tool call and return its result as text for the model.

    Args:
        name: Tool name.
        raw_arguments: JSON arguments from the model.

    Returns:
        JSON result, or an error message the model can react to.
    """
    func = TOOLS.get(name)
    if func is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        arguments = json.loads(raw_arguments or "{}")
        return json.dumps(func(**arguments))
    except (json.JSONDecodeError, TypeError, ValueError, ToolError) as exc:
        return json.dumps({"error": str(exc)})


def run_agent(client: httpx.Client, model: str, task: str, *, max_steps: int = 6) -> AgentRun:
    """Run the tool loop until the model answers in text.

    Args:
        client: HTTP client pointed at the gateway.
        model: Route alias (``model`` field).
        task: The user's request.
        max_steps: Model calls allowed before giving up.

    Returns:
        The transcript.

    Raises:
        httpx.HTTPError: On transport errors.
        RuntimeError: If the gateway returns an error.
    """
    run = AgentRun(messages=[
        {"role": "system", "content": "You are a support agent. Use the tools to look up "
                                      "orders before refunding. Be brief."},
        {"role": "user", "content": task}])
    for step in range(1, max_steps + 1):
        response = client.post("/v1/chat/completions", json={
            "model": model, "messages": run.messages, "tools": TOOL_SPECS, "temperature": 0})
        data = response.json()
        if response.status_code != 200:
            raise RuntimeError(f"gateway error {response.status_code}: {data.get('error')}")
        message = data["choices"][0]["message"]
        meta = data.get("groundwire", {})
        for decision in meta.get("tools", {}).get("decisions", []):
            if decision["decision"] == "blocked":
                run.steps.append(f"[{step}] gateway BLOCKED {decision['tool']}: "
                                 f"{decision.get('detail') or decision.get('reason')}")
        calls = message.get("tool_calls") or []
        if not calls:
            run.answer = message.get("content") or ""
            run.steps.append(f"[{step}] model answered (via {meta.get('provider')})")
            return run
        run.messages.append({"role": "assistant", "content": message.get("content"),
                             "tool_calls": calls})
        for call in calls:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            result = run_tool(name, arguments)
            run.steps.append(f"[{step}] model called {name}({arguments}) -> {result}")
            run.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    run.steps.append(f"stopped after {max_steps} model calls without a final answer")
    return run


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Run a small tool-using agent via Groundwire.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="gateway base URL")
    parser.add_argument("--model", default="default",
                        help="route alias (default: 'default', which reaches Ollama)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="safe")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per model call")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # keep the transcript readable

    try:
        with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
            client.get("/healthz").raise_for_status()
            run = run_agent(client, args.model, SCENARIOS[args.scenario])
    except httpx.ConnectError:
        logger.error("cannot reach the gateway at %s. Start it first: make run", args.url)
        return 2
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.error("agent failed: %s", exc)
        return 1

    logger.info("TASK: %s\n", SCENARIOS[args.scenario])
    for line in run.steps:
        logger.info(line)
    logger.info("\nFINAL ANSWER: %s", run.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
