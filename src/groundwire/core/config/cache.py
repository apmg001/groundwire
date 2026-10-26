"""Response cache settings.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    Field,
)

from groundwire.core.config.base import StrictModel


class CacheConfig(StrictModel):
    """Response cache (exact + optional semantic matching).

    Attributes:
        enabled: Turn the cache on.
        semantic: Also match *reworded* questions using sentence embeddings
            (needs the ``nli`` extra). Off = exact matches only, no extra deps.
        embedding_model: Hugging Face sentence-embedding model id.
        embedding_max_length: Token cap for the embedded question.
        similarity_threshold: Minimum cosine similarity for a semantic hit.
            Higher = fewer wrong answers served, fewer hits. The default 0.96
            is the threshold tuned on QQP train for a 1% false-hit budget with
            MiniLM; on QQP validation it served wrong answers to 0.9% of
            different questions and reused 15% of rewordings (see README).
            Re-run ``make bench-cache`` if you change the embedding model.
        ttl_s: Seconds an entry stays valid.
        max_entries: Entries kept in memory; least recently used are evicted.
        max_temperature: Requests sampled hotter than this are not cached
            (the caller asked for variety). ``None`` caches any temperature.
        lookup_timeout_s: Budget for a lookup; slower lookups are skipped and
            the request goes to the provider (the cache never adds big latency).
        coalesce: Merge identical requests that arrive while one is in flight
            into a single provider call (protects providers from stampedes).
        partition_by_user: Keep a separate cache per OpenAI ``user`` field
            (multi-tenant isolation) instead of sharing answers across users.
        namespace: Free-form version tag mixed into every key. Change it to
            invalidate the whole cache, e.g. after changing a system prompt.
    """

    enabled: bool = False
    semantic: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_max_length: Annotated[int, Field(ge=16, le=8192)] = 256
    similarity_threshold: Annotated[float, Field(gt=0, le=1)] = 0.96
    ttl_s: Annotated[float, Field(gt=0, le=30 * 24 * 3600)] = 3600.0
    max_entries: Annotated[int, Field(ge=1, le=1_000_000)] = 10_000
    max_temperature: Annotated[float, Field(ge=0, le=2)] | None = 1.0
    lookup_timeout_s: Annotated[float, Field(gt=0, le=10)] = 0.5
    coalesce: bool = True
    partition_by_user: bool = False
    namespace: Annotated[str, Field(min_length=1, max_length=64)] = "v1"
