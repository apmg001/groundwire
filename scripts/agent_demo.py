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
    python scripts/agent_demo.py --sequential       # one tool call per step
    python scripts/agent_demo.py --model agent-demo-liar  # no model: a misreported refund
    python scripts/agent_demo.py --model agent-demo-approval --scenario approval --auto-approve

``--sequential`` asks for ``parallel_tool_calls: false`` so the model must see
one tool's result before choosing the next call (look up the order, *then*
refund). Not every server honours that flag, so the agent also enforces it:
if several calls come back anyway, it runs only the first and reports it.

After the final answer, the transcript shows the gateway's answer check:
whether what the model *said* matches what the tools *returned*.

Human approval: when the gateway holds a call for approval, ``--wait-approval``
makes the agent wait (approve it from another terminal with ``make approve
ID=...``), and ``--auto-approve`` plays the manager using the reviewer token in
``GROUNDWIRE_APPROVER_TOKEN``. Once approved, the agent asks again with the
same conversation; the gateway releases the exact approved call, once.

Exit codes: 0 finished, 1 gateway/model error, 2 gateway not reachable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
    4417: {"order_id": 4417, "item": "standing desk", "total": 1200.0, "currency": "USD",
           "status": "delivered", "customer_email": "meera@example.com"},
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
    "approval": "Order 4417 arrived damaged. Look it up and refund the full amount.",
}

TOKEN_ENV = "GROUNDWIRE_APPROVER_TOKEN"  # noqa: S105 - env var name, not a secret


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


def final_answer_label(blocked: list[dict[str, Any]], provider: str | None,
                       held: list[dict[str, Any]] | None = None) -> str:
    """Say who wrote the final text, so the transcript does not mislead.

    When the gateway drops or holds every requested tool call, the reply the
    agent receives is the gateway's explanation (after any text the model
    wrote first), not a normal model answer.

    Args:
        blocked: Blocked tool decisions from this step's ``groundwire`` metadata.
        provider: Provider that served the model call, if reported.
        held: Decisions waiting for human approval.

    Returns:
        A one-line description for the transcript.
    """
    via = f"via {provider or 'unknown provider'}"
    if held:
        ids = ", ".join(str(d.get("approval_id")) for d in held)
        return (f"gateway is holding the call for human approval ({ids}); "
                f"nothing was carried out (model {via})")
    if blocked:
        noun = "call" if len(blocked) == 1 else "calls"
        return (f"gateway blocked the tool {noun} and explained why "
                f"(model {via}; no action was taken)")
    return f"model answered ({via})"


def answer_check_lines(check: dict[str, Any] | None, answer: str) -> list[str]:
    """Describe the gateway's answer check for the transcript.

    Args:
        check: The ``groundwire.answer_check`` metadata, if the gateway sent it.
        answer: The final answer (the issue spans point into it).

    Returns:
        Lines to print (empty when the gateway did not check the answer).
    """
    if not check:
        return []
    verdict = check.get("verdict")
    if verdict == "supported":
        return ["gateway answer check: the answer matches the tool results"]
    if verdict != "mismatch":
        return [f"gateway answer check: skipped ({check.get('reason')})"]
    lines = ["gateway answer check: MISMATCH, the answer does not match the tool results"]
    for issue in check.get("issues", []):
        start, end = (issue.get("span") or [0, 0])[:2]
        claim = answer[start:end] if 0 <= start < end <= len(answer) else "?"
        lines.append(f"  - '{claim}': {issue.get('detail') or issue.get('kind')}")
    return lines


class ApprovalError(RuntimeError):
    """The approval API refused a request the demo needed."""


@dataclass
class ApprovalWaiter:
    """Waits for held calls to be decided (optionally playing the manager).

    Attributes:
        poll_interval_s: Seconds between status checks.
        timeout_s: Give up after this long (statuses are returned as they are).
        approve_with_token: Reviewer token; when set, the waiter approves the
            requests itself, as a manager would (demo only).
        reviewer: Name recorded as the approver.
        sleep: Sleep function (injected in tests).
        clock: Monotonic clock (injected in tests).
    """

    poll_interval_s: float = 1.0
    timeout_s: float = 300.0
    approve_with_token: str | None = None
    reviewer: str = "demo-manager"
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def wait(self, client: httpx.Client, approval_ids: list[str]) -> dict[str, str]:
        """Return each approval's status once none is pending (or on timeout).

        Args:
            client: HTTP client pointed at the gateway.
            approval_ids: Held approvals to wait for.

        Returns:
            Approval id -> final status (``pending`` if the timeout hit first).

        Raises:
            ApprovalError: If approving or reading a status fails.
            httpx.HTTPError: On transport errors.
        """
        if self.approve_with_token:
            for approval_id in approval_ids:
                self._approve(client, approval_id)
        else:
            for approval_id in approval_ids:
                logger.info("waiting for a reviewer: make approve ID=%s  (or reject)", approval_id)
        deadline = self.clock() + self.timeout_s
        while True:
            statuses = {i: self._status(client, i) for i in approval_ids}
            if "pending" not in statuses.values() or self.clock() >= deadline:
                return statuses
            self.sleep(self.poll_interval_s)

    def _approve(self, client: httpx.Client, approval_id: str) -> None:
        """Approve one request with the reviewer token (already-decided is fine)."""
        response = client.post(
            f"/v1/approvals/{approval_id}/approve",
            headers={"Authorization": f"Bearer {self.approve_with_token}"},
            json={"reviewer": self.reviewer, "note": "approved by the agent demo"})
        if response.status_code not in (200, 409):
            raise ApprovalError(f"approve failed ({response.status_code}): "
                                f"{response.json().get('error', {}).get('message')}")

    @staticmethod
    def _status(client: httpx.Client, approval_id: str) -> str:
        """Read one approval's status from the agent-facing endpoint."""
        response = client.get(f"/v1/approvals/{approval_id}/status")
        if response.status_code != 200:
            raise ApprovalError(f"status check failed ({response.status_code})")
        return str(response.json().get("status"))


def run_agent(client: httpx.Client, model: str, task: str, *, max_steps: int = 6,
              sequential: bool = False, waiter: ApprovalWaiter | None = None) -> AgentRun:
    """Run the tool loop until the model answers in text.

    Args:
        client: HTTP client pointed at the gateway.
        model: Route alias (``model`` field).
        task: The user's request.
        max_steps: Model calls allowed before giving up.
        sequential: Ask for one tool call per step, and enforce it if the
            server ignores the request.
        waiter: When set, held calls are waited for; once approved, the agent
            asks again with the same conversation and the call is released.

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
        body: dict[str, Any] = {"model": model, "messages": run.messages, "tools": TOOL_SPECS,
                                "temperature": 0}
        if sequential:
            body["parallel_tool_calls"] = False
        response = client.post("/v1/chat/completions", json=body)
        data = response.json()
        if response.status_code != 200:
            raise RuntimeError(f"gateway error {response.status_code}: {data.get('error')}")
        message = data["choices"][0]["message"]
        meta = data.get("groundwire", {})
        decisions = meta.get("tools", {}).get("decisions", [])
        blocked = [d for d in decisions if d.get("decision") == "blocked"]
        held = [d for d in decisions if d.get("decision") == "pending_approval"]
        for decision in blocked:
            run.steps.append(f"[{step}] gateway BLOCKED {decision.get('tool')}: "
                             f"{decision.get('detail') or decision.get('reason')}")
        for decision in decisions:
            if decision.get("decision") == "allowed" and decision.get("approval_id"):
                run.steps.append(f"[{step}] gateway RELEASED {decision.get('tool')}: "
                                 f"approved by a reviewer ({decision['approval_id']})")
        calls = message.get("tool_calls") or []
        if not calls and held and waiter is not None:
            ids = [str(d.get("approval_id")) for d in held]
            for decision in held:
                run.steps.append(f"[{step}] gateway HELD {decision.get('tool')} for human "
                                 f"approval ({decision.get('approval_id')})")
            statuses = waiter.wait(client, ids)
            run.steps.append(f"[{step}] reviewer decision: "
                             + ", ".join(f"{i} {s}" for i, s in statuses.items()))
            if all(status == "approved" for status in statuses.values()):
                continue  # ask again with the same conversation: the gateway releases the call
        if not calls:
            run.answer = message.get("content") or ""
            run.steps.append(
                f"[{step}] {final_answer_label(blocked, meta.get('provider'), held)}")
            run.steps.extend(f"[{step}] {line}"
                             for line in answer_check_lines(meta.get("answer_check"), run.answer))
            return run
        if sequential and len(calls) > 1:
            names = ", ".join(c["function"]["name"] for c in calls)
            run.steps.append(f"[{step}] server ignored parallel_tool_calls=false and returned "
                             f"{len(calls)} calls ({names}); running only the first")
            calls = calls[:1]  # the model will ask again once it has seen this result
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
    parser.add_argument("--sequential", action="store_true",
                        help="one tool call per step (parallel_tool_calls=false)")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--wait-approval", action="store_true",
                          help="wait for a reviewer to decide held calls")
    approval.add_argument("--auto-approve", action="store_true",
                          help=f"approve held calls yourself using ${TOKEN_ENV} (demo only)")
    parser.add_argument("--approval-timeout", type=float, default=300.0,
                        help="seconds to wait for a reviewer")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # keep the transcript readable

    waiter: ApprovalWaiter | None = None
    if args.auto_approve:
        token = os.environ.get(TOKEN_ENV, "").strip()
        if not token:
            logger.error("--auto-approve needs the reviewer token in $%s", TOKEN_ENV)
            return 2
        waiter = ApprovalWaiter(timeout_s=args.approval_timeout, approve_with_token=token)
    elif args.wait_approval:
        waiter = ApprovalWaiter(timeout_s=args.approval_timeout)

    try:
        with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
            client.get("/healthz").raise_for_status()
            run = run_agent(client, args.model, SCENARIOS[args.scenario],
                            sequential=args.sequential, waiter=waiter)
    except httpx.ConnectError:
        logger.error("cannot reach the gateway at %s. Start it first: make run", args.url)
        return 2
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.error("agent failed: %s", exc)
        return 1

    mode = "sequential" if args.sequential else "parallel allowed"
    logger.info("TASK: %s  [%s]\n", SCENARIOS[args.scenario], mode)
    for line in run.steps:
        logger.info(line)
    logger.info("\nFINAL ANSWER: %s", run.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
