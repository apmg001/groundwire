"""Build the response cache from configuration (used by the composition root)."""

from __future__ import annotations

import logging

from groundwire.cache.embeddings import Embedder, TransformersEmbedder
from groundwire.cache.service import ResponseCache
from groundwire.cache.store import CacheStore, InMemoryCacheStore
from groundwire.core.config import CacheConfig
from groundwire.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)


def build_cache(config: CacheConfig, *, metrics: GatewayMetrics | None = None,
                store: CacheStore | None = None,
                embedder: Embedder | None = None) -> ResponseCache | None:
    """Create the response cache, or ``None`` when caching is disabled.

    Args:
        config: Cache settings.
        metrics: Metrics sink.
        store: Storage override (tests, shared stores); in-memory if omitted.
        embedder: Embedder override (tests); loaded from config when
            ``semantic`` is on and none is given.

    Returns:
        The cache, or ``None``.

    Raises:
        ConfigurationError: If semantic matching is on but the embedding model
            cannot be loaded (fail fast at boot, not on the first request).
    """
    if not config.enabled:
        return None
    if config.semantic and embedder is None:
        embedder = TransformersEmbedder.load(config.embedding_model,
                                             max_length=config.embedding_max_length)
    cache = ResponseCache(config, store or InMemoryCacheStore(config.max_entries),
                          embedder=embedder if config.semantic else None, metrics=metrics)
    logger.info("response cache enabled", extra={
        "semantic": cache.semantic, "ttl_s": config.ttl_s, "max_entries": config.max_entries,
        "similarity_threshold": config.similarity_threshold})
    return cache
