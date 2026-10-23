"""Deployment checks: does this config work with this many processes?

Some components keep state inside the process (the in-memory approval store,
the in-memory response cache, circuit breakers). With several worker
processes, each worker has its own copy, and requests are spread across
workers at random. What that means depends on the component:

* **approvals (memory store)**: *incorrect*. A call held on worker 1 and
  approved there is retried on worker 3, which has never seen the approval,
  so it stays pending forever, and reviewers only see one worker's queue.
  Safe (nothing runs unapproved) but broken, so startup is refused.
* **response cache (memory)**: correct but less effective (each worker
  warms its own cache). Warned.
* **circuit breakers / metrics**: correct; each worker learns provider
  health on its own and exposes its own metrics. Warned.

Worker count is read from where the process manager reads it: the
``GROUNDWIRE_WORKERS`` setting (the ``groundwire`` entrypoint) and
``WEB_CONCURRENCY`` (uvicorn's default for ``--workers``). Separate replicas
(several containers) cannot be detected from inside one process; the same
limits apply to them, and the README says so.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from groundwire.core.config import ApprovalStoreType, GatewayConfig, ToolAction
from groundwire.core.errors import ConfigurationError

logger = logging.getLogger(__name__)

WEB_CONCURRENCY_ENV = "WEB_CONCURRENCY"


def effective_workers(configured: int, environ: Mapping[str, str]) -> int:
    """Return the number of worker processes this deployment will run.

    Args:
        configured: ``Settings.workers`` (``GROUNDWIRE_WORKERS``).
        environ: Process environment (for ``WEB_CONCURRENCY``).

    Returns:
        The larger of the two (an unparseable ``WEB_CONCURRENCY`` is ignored
        here; uvicorn itself rejects it).
    """
    raw = environ.get(WEB_CONCURRENCY_ENV, "").strip()
    try:
        from_env = int(raw) if raw else 1
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", WEB_CONCURRENCY_ENV, raw)
        from_env = 1
    return max(configured, from_env, 1)


def _approval_rules(config: GatewayConfig) -> list[str]:
    """Names of the tool rules that hold calls for approval."""
    policy = config.tool_policy
    names = [rule.name or ",".join(rule.tools) for rule in policy.rules
             if rule.action is ToolAction.REQUIRE_APPROVAL]
    if policy.default_action is ToolAction.REQUIRE_APPROVAL:
        names.append("default_action")
    return names if policy.enabled else []


def check_deployment(config: GatewayConfig, *, workers: int) -> list[str]:
    """Validate that ``config`` is correct when run with ``workers`` processes.

    Args:
        config: Gateway configuration.
        workers: Worker processes (see :func:`effective_workers`).

    Returns:
        Warnings for components that work but lose effectiveness (also logged).

    Raises:
        ConfigurationError: A component would behave incorrectly.
    """
    if workers <= 1:
        return []
    approvals = config.approvals
    held_by = _approval_rules(config)
    if approvals.enabled and approvals.store is ApprovalStoreType.MEMORY and held_by:
        raise ConfigurationError(
            f"approvals use the in-memory store, which is per process, but {workers} worker "
            f"processes are configured (rules {held_by} hold calls for approval): an approval "
            "made on one worker would never release the call on another. Run one worker "
            "(GROUNDWIRE_WORKERS=1, unset WEB_CONCURRENCY) and one replica, or use a shared "
            "approval store; separate containers do not share the in-memory store either.")

    warnings: list[str] = []
    if config.cache.enabled:
        warnings.append(f"response cache is in memory: each of the {workers} workers keeps "
                        "its own copy, so hit rates are lower than with one process")
    warnings.append(f"circuit breakers and metrics are per process: each of the {workers} "
                    "workers tracks provider health and exposes metrics separately")
    for warning in warnings:
        logger.warning("deployment: %s", warning)
    return warnings
