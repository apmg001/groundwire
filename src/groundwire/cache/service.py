"""ResponseCache: the policy layer between the chat service and the store.

Lookup order for an eligible request:

1. **Exact**: hash of (namespace, normalised question). Cheap and always on.
2. **Semantic** (optional): embed the question, take the nearest entry in the
   same namespace, and accept it only if

   * cosine similarity >= ``similarity_threshold``, **and**
   * the two questions mention the same **numbers** (order 7781 vs 7782 embed
     almost identically but need different answers), **and**
   * they contain the same **PII placeholders** ([EMAIL_1] vs no email).

What is never cached:

* requests the client marks ``bypass`` (``refresh`` skips only the read),
* agent traffic (tools offered, or tool calls/results in the history),
* requests sampled hotter than ``max_temperature`` (the caller wants variety),
* requests whose prompt was **flagged as a possible injection** (stops an
  attacker from planting an answer that other users would then receive),
* answers that were cut off (``finish_reason != "stop"``) or are empty,
* answers that **failed grounding** (the chat service decides that).

Fault tolerance: the cache is an optimisation, so it **fails open**. Any store
or embedder error, or a lookup slower than ``lookup_timeout_s``, is logged and
counted, and the request simply goes to the provider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from groundwire.cache.embeddings import Embedder, Vector
from groundwire.cache.guards import semantic_mismatch
from groundwire.cache.keys import CacheKey, build_cache_key
from groundwire.cache.store import CacheEntry, CacheStore
from groundwire.core.config import CacheConfig
from groundwire.core.errors import CacheUnavailableError
from groundwire.domain.models import CacheMode, ChatCompletionRequest, Completion
from groundwire.observability.metrics import GatewayMetrics

logger = logging.getLogger(__name__)

T = TypeVar("T")

# --------------------------------------------------------------------------- #
# Lookup result
# --------------------------------------------------------------------------- #


class CacheStatus(StrEnum):
    """Outcome of a cache lookup."""

    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"  # policy or client chose not to read the cache
    ERROR = "error"  # the cache failed; the request went to the provider


class MatchKind(StrEnum):
    """How a hit was found."""

    EXACT = "exact"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class CacheLookup:
    """Everything the chat service needs after a lookup.

    Attributes:
        status: Lookup outcome.
        key: Request key (``None`` when caching does not apply at all).
        writable: Whether the provider's answer may be stored afterwards.
        entry: The cached entry on a hit.
        match: How the hit was found.
        similarity: Cosine similarity for semantic hits (and near misses).
        vector: Query embedding, reused when storing.
        reason: Why the lookup was bypassed or a near miss was rejected.
        age_s: Age of the served entry in seconds.
        coalesced: Set by the chat service when the answer came from an
            identical in-flight request.
    """

    status: CacheStatus
    key: CacheKey | None = None
    writable: bool = False
    entry: CacheEntry | None = None
    match: MatchKind | None = None
    similarity: float | None = None
    vector: Vector | None = None
    reason: str | None = None
    age_s: float | None = None
    coalesced: bool = False

    @property
    def hit(self) -> bool:
        """True when an answer can be served from the cache."""
        return self.status is CacheStatus.HIT and self.entry is not None

    def as_dict(self) -> dict[str, Any]:
        """Render response metadata (no question text, no answer text).

        Returns:
            JSON-serialisable dict.
        """
        data: dict[str, Any] = {"status": self.status.value}
        if self.match is not None:
            data["match"] = self.match.value
        if self.similarity is not None:
            data["similarity"] = round(self.similarity, 4)
        if self.age_s is not None:
            data["age_s"] = round(self.age_s, 1)
        if self.hit and self.entry is not None:
            data["tokens_saved"] = self.entry.usage.total_tokens
        if self.reason is not None:
            data["reason"] = self.reason
        if self.coalesced:
            data["coalesced"] = True
        return data


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class ResponseCache:
    """Exact + semantic response cache with safety-aware policy."""

    def __init__(self, config: CacheConfig, store: CacheStore, *,
                 embedder: Embedder | None = None,
                 metrics: GatewayMetrics | None = None) -> None:
        """Create the cache.

        Args:
            config: Cache settings.
            store: Storage adapter.
            embedder: Sentence embedder; ``None`` = exact matching only.
            metrics: Metrics sink.
        """
        self._config = config
        self._store = store
        self._embedder = embedder
        self._metrics = metrics

    @property
    def semantic(self) -> bool:
        """True when semantic matching is active."""
        return self._embedder is not None

    @property
    def coalesce(self) -> bool:
        """True when identical in-flight requests should be merged."""
        return self._config.coalesce

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    async def lookup(self, request: ChatCompletionRequest, route: str, *,
                     injection_flagged: bool = False) -> CacheLookup:
        """Look a (redacted) request up. Never raises.

        Args:
            request: Request exactly as it would be sent to providers.
            route: Resolved route alias.
            injection_flagged: The safety guard flagged this prompt.

        Returns:
            The lookup result.
        """
        started = time.perf_counter()
        result = await self._lookup(request, route, injection_flagged=injection_flagged)
        self._record_lookup(result, time.perf_counter() - started)
        return result

    async def _lookup(self, request: ChatCompletionRequest, route: str, *,
                      injection_flagged: bool) -> CacheLookup:
        """Lookup logic without metrics; see :meth:`lookup`."""
        reason = self._bypass_reason(request, injection_flagged)
        if reason is not None:
            return CacheLookup(status=CacheStatus.BYPASS, reason=reason)

        key = build_cache_key(request, route, namespace_tag=self._config.namespace,
                              partition_by_user=self._config.partition_by_user)
        mode = _cache_mode(request)
        if mode is CacheMode.REFRESH:
            return CacheLookup(status=CacheStatus.BYPASS, key=key, writable=True,
                               reason="client_refresh")
        try:
            async with asyncio.timeout(self._config.lookup_timeout_s):
                return await self._find(key)
        except TimeoutError:
            logger.warning("cache lookup timed out; serving from provider",
                           extra={"timeout_s": self._config.lookup_timeout_s})
            return CacheLookup(status=CacheStatus.ERROR, key=key, writable=True,
                               reason="lookup_timeout")
        except CacheUnavailableError as exc:
            logger.warning("cache lookup failed; serving from provider",
                           extra={"error": exc.message})
            return CacheLookup(status=CacheStatus.ERROR, key=key, writable=True,
                               reason="lookup_failed")

    async def _find(self, key: CacheKey) -> CacheLookup:
        """Exact, then semantic, search.

        Args:
            key: Request key.

        Returns:
            A hit or a miss.

        Raises:
            CacheUnavailableError: If the store or embedder fails.
        """
        entry = await self._call_store(self._store.get(key.exact))
        if entry is not None:
            return await self._serve(key, entry, MatchKind.EXACT, similarity=None, vector=None)

        if self._embedder is None:
            return CacheLookup(status=CacheStatus.MISS, key=key, writable=True)

        vector = await self._embed(key.query)
        neighbour = await self._call_store(self._store.nearest(key.namespace, vector))
        if neighbour is None:
            return CacheLookup(status=CacheStatus.MISS, key=key, writable=True, vector=vector)
        if neighbour.similarity < self._config.similarity_threshold:
            return CacheLookup(status=CacheStatus.MISS, key=key, writable=True, vector=vector,
                               similarity=neighbour.similarity)
        rejection = semantic_mismatch(key.query, neighbour.entry.query)
        if rejection is not None:
            logger.info("semantic near-miss rejected",
                        extra={"reason": rejection,
                               "similarity": round(neighbour.similarity, 4)})
            return CacheLookup(status=CacheStatus.MISS, key=key, writable=True, vector=vector,
                               similarity=neighbour.similarity, reason=rejection)
        return await self._serve(key, neighbour.entry, MatchKind.SEMANTIC,
                                 similarity=neighbour.similarity, vector=vector)

    async def _serve(self, key: CacheKey, entry: CacheEntry, match: MatchKind, *,
                     similarity: float | None, vector: Vector | None) -> CacheLookup:
        """Build a hit and record the use.

        Args:
            key: Request key.
            entry: Entry being served.
            match: Match kind.
            similarity: Similarity for semantic hits.
            vector: Query vector, if computed.

        Returns:
            The hit.

        Raises:
            CacheUnavailableError: If the store fails.
        """
        await self._call_store(self._store.touch(entry))
        return CacheLookup(status=CacheStatus.HIT, key=key, writable=False, entry=entry,
                           match=match, similarity=similarity, vector=vector,
                           age_s=max(0.0, self._store.now() - entry.created_at))

    # ------------------------------------------------------------------ #
    # Store
    # ------------------------------------------------------------------ #

    async def store(self, lookup: CacheLookup, completion: Completion, provider: str) -> bool:
        """Store a provider answer if policy allows. Never raises.

        Args:
            lookup: The lookup made for this request.
            completion: Provider answer (redacted form, *before* PII restore).
            provider: Provider that served it.

        Returns:
            True if the answer was stored.
        """
        if not lookup.writable or lookup.key is None:
            return False
        if completion.finish_reason != "stop" or not completion.text.strip():
            self._count_store("skipped")
            return False
        key = lookup.key
        try:
            async with asyncio.timeout(self._config.lookup_timeout_s):
                vector = lookup.vector
                if vector is None and self._embedder is not None:
                    vector = await self._embed(key.query)
                entry = CacheEntry(
                    key=key.exact, namespace=key.namespace, query=key.query, vector=vector,
                    text=completion.text, finish_reason=completion.finish_reason,
                    usage=completion.usage, upstream_model=completion.upstream_model,
                    provider=provider)
                await self._call_store(self._store.put(entry, self._config.ttl_s))
        except (TimeoutError, CacheUnavailableError) as exc:
            logger.warning("cache store failed; answer not cached",
                           extra={"error": str(exc) or type(exc).__name__})
            self._count_store("error")
            return False
        self._count_store("stored")
        if self._metrics is not None:
            self._metrics.cache_entries.set(self._store.size())
        return True

    async def clear(self) -> None:
        """Remove every cached entry.

        Raises:
            CacheUnavailableError: If the store fails.
        """
        await self._call_store(self._store.clear())
        if self._metrics is not None:
            self._metrics.cache_entries.set(0)
        logger.info("cache cleared")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _bypass_reason(self, request: ChatCompletionRequest,
                       injection_flagged: bool) -> str | None:
        """Return why this request must not use the cache, or ``None``.

        Args:
            request: Client request.
            injection_flagged: Safety guard verdict.

        Returns:
            A short reason code, or ``None`` if the request is eligible.
        """
        if _cache_mode(request) is CacheMode.BYPASS:
            return "client_bypass"
        if request.uses_tools:
            # Agent turns depend on live tool results and can trigger actions;
            # replaying a cached "call issue_refund" would be dangerous.
            return "tool_use"
        if injection_flagged:
            return "injection_flagged"
        limit = self._config.max_temperature
        if limit is not None and request.temperature is not None and request.temperature > limit:
            return "temperature"
        return None

    async def _embed(self, text: str) -> Vector:
        """Embed one question in a worker thread.

        Args:
            text: Normalised question.

        Returns:
            Unit vector.

        Raises:
            CacheUnavailableError: If the embedder fails or misbehaves.
        """
        if self._embedder is None:  # pragma: no cover - guarded by callers
            raise CacheUnavailableError("semantic matching is not enabled")
        vectors = await asyncio.to_thread(self._embedder.embed, [text])
        if len(vectors) != 1:
            raise CacheUnavailableError(f"embedder returned {len(vectors)} vectors for 1 text")
        return vectors[0]

    @staticmethod
    async def _call_store(operation: Awaitable[T]) -> T:
        """Await a store operation, normalising adapter failures.

        A shared store (Redis, a vector DB) fails with its own exception
        types; they all become :class:`CacheUnavailableError` here, so the
        fail-open handling above has exactly one exception to catch.

        Args:
            operation: Awaitable returned by a store method.

        Returns:
            The operation's result.

        Raises:
            CacheUnavailableError: If the store raised.
        """
        try:
            return await operation
        except CacheUnavailableError:
            raise
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise CacheUnavailableError(f"cache store failed: {exc}") from exc

    def _record_lookup(self, result: CacheLookup, seconds: float) -> None:
        """Count a lookup outcome.

        Args:
            result: Lookup result.
            seconds: Lookup duration.
        """
        if self._metrics is None:
            return
        label = (f"hit_{result.match.value}" if result.hit and result.match is not None
                 else result.status.value)
        self._metrics.cache_lookups.labels(result=label).inc()
        self._metrics.cache_lookup_latency.observe(seconds)
        if result.hit and result.entry is not None:
            self._metrics.cache_tokens_saved.inc(result.entry.usage.total_tokens)

    def _count_store(self, outcome: str) -> None:
        """Count a store outcome.

        Args:
            outcome: ``stored``, ``skipped`` or ``error``.
        """
        if self._metrics is not None:
            self._metrics.cache_stores.labels(outcome=outcome).inc()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _cache_mode(request: ChatCompletionRequest) -> CacheMode:
    """Return the request's cache mode (default ``use``).

    Args:
        request: Client request.

    Returns:
        The mode.
    """
    options = request.groundwire.cache if request.groundwire else None
    return options.mode if options is not None else CacheMode.USE
