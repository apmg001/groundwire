"""Response caching: exact and semantic reuse of earlier answers.

* :mod:`~groundwire.cache.keys`: which requests may share an answer.
* :mod:`~groundwire.cache.embeddings`: sentence embeddings for semantic matching.
* :mod:`~groundwire.cache.store`: storage port and the in-memory LRU/TTL adapter.
* :mod:`~groundwire.cache.singleflight`: merges identical in-flight requests.
* :mod:`~groundwire.cache.service`: policy (eligibility, guards, fail-open).
"""

from groundwire.cache.factory import build_cache
from groundwire.cache.service import CacheLookup, CacheStatus, MatchKind, ResponseCache

__all__ = ["CacheLookup", "CacheStatus", "MatchKind", "ResponseCache", "build_cache"]
