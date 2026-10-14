"""Chaos demo: fire concurrent requests at the ``demo`` route and report outcomes.

The ``demo`` route chains ``mock-flaky`` (50% injected failures) in front of
``mock-stable``. With the gateway running, this script shows retries, circuit
breaking and failover keeping the success rate at 100%::

    make run      # terminal 1: start the gateway
    make chaos    # terminal 2: run this script

Or do both in one command (starts a temporary gateway, runs, stops it)::

    make demo

Uses only the standard library plus ``httpx`` (already a gateway dependency).

Exit codes:
    0  every request succeeded
    1  some requests failed
    2  the gateway could not be reached (nothing was sent)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("chaos_demo")

EXIT_OK = 0
EXIT_REQUEST_FAILURES = 1
EXIT_GATEWAY_UNREACHABLE = 2

TRANSPORT_ERROR = 0  # pseudo status code for requests that never got a response


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class GatewayUnreachableError(Exception):
    """The gateway did not answer the pre-flight health check."""


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RequestResult:
    """Outcome of one demo request.

    Attributes:
        status: HTTP status, or ``TRANSPORT_ERROR`` if no response arrived.
        provider: Provider that served it (``X-Groundwire-Provider``), if any.
        latency_s: Wall-clock latency.
        error: Transport error type name, when ``status`` is ``TRANSPORT_ERROR``.
    """

    status: int
    provider: str
    latency_s: float
    error: str | None = None


# --------------------------------------------------------------------------- #
# Gateway interaction
# --------------------------------------------------------------------------- #


async def preflight(client: httpx.AsyncClient, base_url: str, route: str) -> None:
    """Check the gateway is up and the route exists before sending load.

    Args:
        client: Shared HTTP client.
        base_url: Gateway base URL (for error messages).
        route: Route alias that will be targeted.

    Raises:
        GatewayUnreachableError: If ``/healthz`` cannot be reached, or the
            route is not configured on the gateway.
    """
    try:
        health = await client.get("/healthz", timeout=3)
        health.raise_for_status()
    except httpx.ConnectError as exc:
        raise GatewayUnreachableError(
            f"no gateway is listening at {base_url}.\n"
            "  Start it first in another terminal:   make run\n"
            "  Or run both in one command:           make demo"
        ) from exc
    except httpx.HTTPError as exc:
        raise GatewayUnreachableError(
            f"gateway at {base_url} failed its health check: {exc}") from exc

    try:
        models = (await client.get("/v1/models", timeout=3)).json()
        available = [m["id"] for m in models.get("data", [])]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise GatewayUnreachableError(f"could not list routes: {exc}") from exc
    if route not in available:
        raise GatewayUnreachableError(
            f"route '{route}' is not configured on the gateway (available: {available}).\n"
            "  Is it running with config/groundwire.yaml?")


async def send_one(client: httpx.AsyncClient, route: str,
                   sem: asyncio.Semaphore) -> RequestResult:
    """Send one chat request.

    Args:
        client: Shared HTTP client.
        route: Route alias to target.
        sem: Concurrency limiter.

    Returns:
        The request's outcome. Transport errors are recorded, not raised,
        so one failure never aborts the whole run.
    """
    body = {"model": route, "messages": [{"role": "user", "content": "ping"}]}
    async with sem:
        started = time.perf_counter()
        try:
            response = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return RequestResult(TRANSPORT_ERROR, "-", time.perf_counter() - started,
                                 error=type(exc).__name__)
        return RequestResult(response.status_code,
                             response.headers.get("x-groundwire-provider", "-"),
                             time.perf_counter() - started)


async def fetch_circuits(client: httpx.AsyncClient) -> dict[str, Any]:
    """Read circuit states from ``/readyz``; never raises.

    Args:
        client: Shared HTTP client.

    Returns:
        Provider -> state, or ``{}`` if the gateway could not be reached.
    """
    try:
        response = await client.get("/readyz", timeout=3)
        circuits: dict[str, Any] = response.json().get("circuits", {})
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("could not read circuit states: %s", exc)
        return {}
    return circuits


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Args:
        sorted_values: Values in ascending order (non-empty).
        fraction: Percentile in [0, 1].

    Returns:
        The percentile value.
    """
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def report(results: list[RequestResult], elapsed_s: float, circuits: dict[str, Any]) -> None:
    """Log a summary of the run.

    Args:
        results: Every request outcome.
        elapsed_s: Total wall-clock time.
        circuits: Circuit states after the run.
    """
    statuses = Counter(r.status for r in results)
    providers = Counter(r.provider for r in results if r.status == 200)
    errors = Counter(r.error for r in results if r.error)
    latencies = sorted(r.latency_s for r in results)

    logger.info("requests:       %d in %.2fs (%.0f req/s)",
                len(results), elapsed_s, len(results) / max(elapsed_s, 1e-9))
    logger.info("status codes:   %s", dict(statuses))
    logger.info("served by:      %s", dict(providers))
    if errors:
        logger.warning("transport errors: %s", dict(errors))
    logger.info("latency:        p50=%.1fms  p95=%.1fms  p99=%.1fms",
                statistics.median(latencies) * 1000,
                percentile(latencies, 0.95) * 1000, percentile(latencies, 0.99) * 1000)
    logger.info("circuit states: %s", circuits)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def main(base_url: str, route: str, total: int, concurrency: int) -> int:
    """Run the demo.

    Args:
        base_url: Gateway base URL.
        route: Route alias.
        total: Number of requests.
        concurrency: Max in-flight requests.

    Returns:
        Process exit code (see module docstring).
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        try:
            await preflight(client, base_url, route)
        except GatewayUnreachableError as exc:
            logger.error("error: %s", exc)
            return EXIT_GATEWAY_UNREACHABLE

        logger.info("sending %d requests to route '%s' (%d concurrent)...",
                    total, route, concurrency)
        sem = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        results = await asyncio.gather(*(send_one(client, route, sem) for _ in range(total)))
        elapsed = time.perf_counter() - started
        circuits = await fetch_circuits(client)

    report(list(results), elapsed, circuits)
    succeeded = sum(r.status == 200 for r in results)
    return EXIT_OK if succeeded == total else EXIT_REQUEST_FAILURES


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--route", default="demo")
    parser.add_argument("--requests", type=_positive_int, default=200)
    parser.add_argument("--concurrency", type=_positive_int, default=20)
    args = parser.parse_args()
    try:
        sys.exit(asyncio.run(main(args.url, args.route, args.requests, args.concurrency)))
    except KeyboardInterrupt:
        logger.info("interrupted")
        sys.exit(130)
