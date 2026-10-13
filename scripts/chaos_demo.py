"""Chaos demo: fire concurrent requests at the ``demo`` route and report outcomes.

The ``demo`` route chains ``mock-flaky`` (50% injected failures) in front of
``mock-stable``. Run the gateway, then this script, and watch retries, circuit
breaking and failover keep the success rate at 100%::

    groundwire                                   # terminal 1
    python scripts/chaos_demo.py --requests 500  # terminal 2

Uses only the standard library plus ``httpx`` (already a gateway dependency).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import time
from collections import Counter

import httpx

logger = logging.getLogger("chaos_demo")


async def _one(client: httpx.AsyncClient, route: str, sem: asyncio.Semaphore
               ) -> tuple[int, str, float]:
    """Send one request.

    Args:
        client: Shared HTTP client.
        route: Route alias to target.
        sem: Concurrency limiter.

    Returns:
        ``(status_code, serving_provider, latency_s)``; status 0 on transport error.
    """
    body = {"model": route, "messages": [{"role": "user", "content": "ping"}]}
    async with sem:
        started = time.perf_counter()
        try:
            response = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            logger.warning("transport error: %s", exc)
            return 0, "transport_error", time.perf_counter() - started
        provider = response.headers.get("x-groundwire-provider", "-")
        return response.status_code, provider, time.perf_counter() - started


async def main(base_url: str, route: str, total: int, concurrency: int) -> int:
    """Run the demo and print a summary.

    Args:
        base_url: Gateway base URL.
        route: Route alias.
        total: Number of requests.
        concurrency: Max in-flight requests.

    Returns:
        Process exit code (0 if every request succeeded).
    """
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        started = time.perf_counter()
        results = await asyncio.gather(*(_one(client, route, sem) for _ in range(total)))
        elapsed = time.perf_counter() - started

        circuits = (await client.get("/readyz")).json().get("circuits", {})

    statuses = Counter(status for status, _, _ in results)
    providers = Counter(provider for status, provider, _ in results if status == 200)
    latencies = sorted(lat for _, _, lat in results)
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]

    logger.info("requests: %d in %.2fs (%.0f req/s)", total, elapsed, total / elapsed)
    logger.info("status codes: %s", dict(statuses))
    logger.info("served by: %s", dict(providers))
    logger.info("latency p50=%.1fms p95=%.1fms", p50 * 1000, p95 * 1000)
    logger.info("circuit states: %s", circuits)
    return 0 if statuses.get(200, 0) == total else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--route", default="demo")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.url, args.route, args.requests, args.concurrency)))
