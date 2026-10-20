"""Cache storage: the port and an in-memory adapter.

:class:`CacheStore` is the *port*. The service only talks to this interface,
so the in-memory store below can be swapped for a shared one (Redis with
vector search, pgvector, Qdrant) when several gateway replicas should share a
cache. The methods are ``async`` for that reason, even though the in-memory
implementation never waits.

:class:`InMemoryCacheStore` is the default adapter:

* **Bounded**: at most ``max_entries``; the least recently *used* entry is
  evicted first (LRU), so memory use is predictable.
* **Expiring**: entries have a TTL and are dropped lazily when touched, plus a
  sweep on every write, so stale answers are never served.
* **Partitioned**: semantic search scans only the entries of one namespace
  (same route, parameters and conversation), never the whole cache.
* **Safe under asyncio**: no method awaits while mutating, so each call is
  atomic with respect to other requests on the event loop.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from groundwire.cache.embeddings import Vector, cosine
from groundwire.domain.models import FinishReason, Usage

logger = logging.getLogger(__name__)

Clock = Callable[[], float]

# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CacheEntry:
    """One cached answer.

    The text is stored exactly as the provider returned it, i.e. still
    containing PII placeholders; each request restores its own values.

    Attributes:
        key: Exact-match key.
        namespace: Namespace the entry belongs to.
        query: Normalised question it answers.
        vector: Embedding of ``query`` (``None`` when semantic matching is off).
        text: Answer text (redacted form).
        finish_reason: Provider finish reason.
        usage: Token usage of the original call (what a hit saves).
        upstream_model: Model that produced the answer.
        provider: Provider that produced the answer.
        created_at: Store clock time of insertion.
        expires_at: Store clock time after which the entry is invalid.
        hits: Times the entry has been served.
    """

    key: str
    namespace: str
    query: str
    vector: Vector | None
    text: str
    finish_reason: FinishReason | None
    usage: Usage
    upstream_model: str
    provider: str
    created_at: float = 0.0
    expires_at: float = 0.0
    hits: int = 0


@dataclass(frozen=True, slots=True)
class Neighbour:
    """Result of a nearest-neighbour search.

    Attributes:
        entry: Closest entry.
        similarity: Cosine similarity to the query vector.
    """

    entry: CacheEntry
    similarity: float


# --------------------------------------------------------------------------- #
# Port
# --------------------------------------------------------------------------- #


class CacheStore(ABC):
    """Storage interface for cached answers."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Return a live entry by exact key (call :meth:`touch` once it is served).

        Args:
            key: Exact-match key.

        Returns:
            The entry, or ``None`` if absent or expired.
        """

    @abstractmethod
    async def nearest(self, namespace: str, vector: Vector) -> Neighbour | None:
        """Find the most similar live entry within a namespace.

        Args:
            namespace: Namespace to search.
            vector: Unit query vector.

        Returns:
            The closest entry, or ``None`` if the namespace has no vectors.
        """

    @abstractmethod
    async def put(self, entry: CacheEntry, ttl_s: float) -> None:
        """Insert or replace an entry.

        Args:
            entry: Entry to store (timestamps are set by the store).
            ttl_s: Time to live in seconds.
        """

    @abstractmethod
    async def touch(self, entry: CacheEntry) -> None:
        """Record that an entry was served (LRU position and hit count).

        Args:
            entry: Entry that was served.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Remove every entry."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of stored entries (live or not yet swept)."""

    @abstractmethod
    def now(self) -> float:
        """Return the store's clock time, the basis for ``created_at``."""


# --------------------------------------------------------------------------- #
# In-memory adapter
# --------------------------------------------------------------------------- #


class InMemoryCacheStore(CacheStore):
    """Bounded, expiring, namespace-partitioned in-process store."""

    def __init__(self, max_entries: int = 10_000, *, clock: Clock = time.monotonic) -> None:
        """Create the store.

        Args:
            max_entries: Capacity; least recently used entries are evicted beyond it.
            clock: Monotonic clock (injectable for tests).

        Raises:
            ValueError: If ``max_entries`` is not positive.
        """
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        # namespace -> keys (dict used as an insertion-ordered set)
        self._namespaces: dict[str, dict[str, None]] = {}

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    async def get(self, key: str) -> CacheEntry | None:
        """See :meth:`CacheStore.get`."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            self._remove(key)
            return None
        return entry

    async def nearest(self, namespace: str, vector: Vector) -> Neighbour | None:
        """See :meth:`CacheStore.nearest`.

        A linear scan over one namespace. Namespaces are small (same route,
        parameters and conversation prefix), and ``cosine`` runs in C on
        Python 3.12+, so this stays well under a millisecond for thousands of
        entries. A shared store would use an ANN index instead.
        """
        keys = self._namespaces.get(namespace)
        if not keys:
            return None
        now = self._clock()
        best: Neighbour | None = None
        expired: list[str] = []
        for key in keys:
            entry = self._entries[key]
            if entry.expires_at <= now:
                expired.append(key)
                continue
            if entry.vector is None or len(entry.vector) != len(vector):
                continue
            similarity = cosine(vector, entry.vector)
            if best is None or similarity > best.similarity:
                best = Neighbour(entry, similarity)
        for key in expired:
            self._remove(key)
        return best

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    async def put(self, entry: CacheEntry, ttl_s: float) -> None:
        """See :meth:`CacheStore.put`."""
        now = self._clock()
        entry.created_at = now
        entry.expires_at = now + ttl_s
        if entry.key in self._entries:
            self._remove(entry.key)
        self._entries[entry.key] = entry
        self._namespaces.setdefault(entry.namespace, {})[entry.key] = None
        self._evict(now)

    async def touch(self, entry: CacheEntry) -> None:
        """See :meth:`CacheStore.touch`."""
        if entry.key in self._entries:
            self._entries.move_to_end(entry.key)
            entry.hits += 1

    async def clear(self) -> None:
        """See :meth:`CacheStore.clear`."""
        self._entries.clear()
        self._namespaces.clear()

    def size(self) -> int:
        """See :meth:`CacheStore.size`."""
        return len(self._entries)

    def now(self) -> float:
        """See :meth:`CacheStore.now`."""
        return self._clock()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _evict(self, now: float) -> None:
        """Drop expired entries from the LRU end, then enforce capacity.

        Args:
            now: Current clock time.
        """
        while self._entries:
            oldest_key, oldest = next(iter(self._entries.items()))
            if oldest.expires_at > now and len(self._entries) <= self._max_entries:
                break
            self._remove(oldest_key)

    def _remove(self, key: str) -> None:
        """Delete an entry and its namespace index slot.

        Args:
            key: Entry key.
        """
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        keys = self._namespaces.get(entry.namespace)
        if keys is not None:
            keys.pop(key, None)
            if not keys:
                del self._namespaces[entry.namespace]
