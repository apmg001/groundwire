"""Build the approval service from configuration (the composition step)."""

from __future__ import annotations

import logging

from groundwire.approvals.service import ApprovalService
from groundwire.approvals.store import ApprovalStore, InMemoryApprovalStore
from groundwire.core.config import ApprovalConfig, ApprovalStoreType
from groundwire.core.errors import ConfigurationError
from groundwire.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)


def build_store(config: ApprovalConfig) -> ApprovalStore:
    """Create the configured store adapter.

    Args:
        config: Approval settings.

    Returns:
        The store.

    Raises:
        ConfigurationError: For a store type this build does not support.
    """
    if config.store is ApprovalStoreType.MEMORY:
        return InMemoryApprovalStore(max_records=config.max_records)
    raise ConfigurationError(f"unsupported approval store: {config.store}")  # pragma: no cover


def build_approval_service(config: ApprovalConfig, *,
                           metrics: GatewayMetrics | None = None) -> ApprovalService | None:
    """Create the approval service, or ``None`` when approvals are disabled.

    Args:
        config: Approval settings.
        metrics: Metrics sink.

    Returns:
        The service, or ``None``.

    Raises:
        ConfigurationError: For an unsupported store.
    """
    if not config.enabled:
        logger.info("approvals disabled")
        return None
    service = ApprovalService(build_store(config), config, metrics=metrics)
    logger.info("approvals enabled", extra={"store": config.store.value,
                                           "ttl_s": config.ttl_s,
                                           "grant_ttl_s": config.grant_ttl_s})
    return service
