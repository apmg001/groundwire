"""Unit tests for the response cache: keys, embeddings, store, coalescing, policy."""

from __future__ import annotations

import asyncio
import math
import operator
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import reduce
from typing import Any

import pytest

from conftest import chat_request
from groundwire.cache.embeddings import Embedder, TransformersEmbedder, Vector, cosine, normalise
from groundwire.cache.factory import build_cache
from groundwire.cache.keys import build_cache_key, normalise_query
from groundwire.cache.service import CacheLookup, CacheStatus, MatchKind, ResponseCache
from groundwire.cache.singleflight import SingleFlight
from groundwire.cache.store import CacheEntry, CacheStore, InMemoryCacheStore, Neighbour
from groundwire.core.config import CacheConfig
from groundwire.core.errors import CacheUnavailableError, ConfigurationError
from groundwire.domain.models import ChatCompletionRequest, Completion, Usage
from groundwire.observability.metrics import GatewayMetrics

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TableEmbedder(Embedder):
    """Returns a fixed vector per text (unknown texts get a unique axis)."""

    name = "table"

    def __init__(self, table: dict[str, Sequence[float]] | None = None, *,
                 fail: bool = False, delay_s: float = 0.0) -> None:
        self.table = {k: normalise(v) for k, v in (table or {}).items()}
        self.fail = fail
        self.delay_s = delay_s
        self.calls: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self.fail:
            raise CacheUnavailableError("model crashed")
        if self.delay_s:
            import time
            time.sleep(self.delay_s)
        self.calls.extend(texts)
        out = []
        for text in texts:
            if text not in self.table:
                axis = [0.0] * 16
                axis[(len(self.table) + 8) % 16] = 1.0
                self.table[text] = normalise(axis)
            out.append(self.table[text])
        return out


class BrokenStore(InMemoryCacheStore):
    """Store whose operations fail like a dropped Redis connection."""

    async def get(self, key: str) -> CacheEntry | None:
        raise ConnectionError("redis: connection reset")

    async def put(self, entry: CacheEntry, ttl_s: float) -> None:
        raise ConnectionError("redis: connection reset")


def _completion(text: str = "Paris.", finish: Any = "stop") -> Completion:
    return Completion(text=text, finish_reason=finish, usage=Usage.of(10, 5),
                      upstream_model="m1")


def _cache(*, embedder: Embedder | None = None, store: CacheStore | None = None,
           metrics: GatewayMetrics | None = None, **overrides: Any) -> ResponseCache:
    config = CacheConfig(enabled=True, semantic=embedder is not None, **overrides)
    return ResponseCache(config, store or InMemoryCacheStore(100), embedder=embedder,
                         metrics=metrics)


async def _prime(cache: ResponseCache, request: ChatCompletionRequest, text: str = "Paris."
                 ) -> CacheLookup:
    lookup = await cache.lookup(request, "default")
    assert await cache.store(lookup, _completion(text), "primary")
    return lookup


def _counter(metrics: GatewayMetrics, name: str, **labels: str) -> float:
    value = metrics.registry.get_sample_value(name, labels or None)
    return value or 0.0


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def test_normalise_query_ignores_case_spacing_and_trailing_punctuation() -> None:
    assert normalise_query("  What is   the CAPITAL of France?! ") == \
        "what is the capital of france"
    assert normalise_query("ﬁle") == "file"  # NFKC ligature


def test_equivalent_requests_share_a_key() -> None:
    a = build_cache_key(chat_request("What is X?"), "default", namespace_tag="v1")
    b = build_cache_key(chat_request("what is  x"), "default", namespace_tag="v1")
    assert a == b


@pytest.mark.parametrize("change", [
    lambda: build_cache_key(chat_request("q"), "other", namespace_tag="v1"),
    lambda: build_cache_key(chat_request("q"), "default", namespace_tag="v2"),
    lambda: build_cache_key(chat_request("q", temperature=0.5), "default", namespace_tag="v1"),
    lambda: build_cache_key(chat_request("q", max_tokens=5), "default", namespace_tag="v1"),
    lambda: build_cache_key(ChatCompletionRequest(messages=[
        {"role": "system", "content": "be terse"}, {"role": "user", "content": "q"}]),
        "default", namespace_tag="v1"),
    lambda: build_cache_key(chat_request("q", groundwire={"grounding": {"context": ["doc"]}}),
                            "default", namespace_tag="v1"),
])
def test_namespace_changes_with_anything_the_model_sees(change: Callable[[], Any]) -> None:
    base = build_cache_key(chat_request("q"), "default", namespace_tag="v1")
    other = change()
    assert other.namespace != base.namespace
    assert other.exact != base.exact


def test_user_partitioning_is_opt_in() -> None:
    a, b = chat_request("q", user="alice"), chat_request("q", user="bob")
    shared = [build_cache_key(r, "d", namespace_tag="v1") for r in (a, b)]
    assert shared[0] == shared[1]
    split = [build_cache_key(r, "d", namespace_tag="v1", partition_by_user=True) for r in (a, b)]
    assert split[0].namespace != split[1].namespace


# --------------------------------------------------------------------------- #
# Vector maths
# --------------------------------------------------------------------------- #


def test_normalise_and_cosine() -> None:
    v = normalise([3.0, 4.0])
    assert v == pytest.approx((0.6, 0.8))
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(normalise([1, 0]), normalise([0, 1])) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [[], [0.0, 0.0], [math.inf, 1.0]])
def test_normalise_rejects_degenerate_vectors(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        normalise(bad)


def test_cosine_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        cosine((1.0,), (1.0, 0.0))


# --------------------------------------------------------------------------- #
# TransformersEmbedder with a tiny fake tensor library
# --------------------------------------------------------------------------- #


def _bcast(a: Any, b: Any, op: Callable[[float, float], float]) -> Any:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == len(b):
            return [_bcast(x, y, op) for x, y in zip(a, b, strict=True)]
        if len(b) == 1:
            return [_bcast(x, b[0], op) for x in a]
        return [_bcast(a[0], y, op) for y in b]
    if isinstance(a, list):
        return [_bcast(x, b, op) for x in a]
    if isinstance(b, list):
        return [_bcast(a, y, op) for y in b]
    return op(a, b)


def _leaves(data: Any, fn: Callable[[float], Any]) -> Any:
    return [_leaves(x, fn) for x in data] if isinstance(data, list) else fn(data)


class _T:
    """Just enough tensor behaviour for mean pooling."""

    dtype = "float32"

    def __init__(self, data: Any) -> None:
        self.data = data

    def unsqueeze(self, dim: int) -> _T:
        return _T(_leaves(self.data, lambda x: [x]))

    def to(self, dtype: Any) -> _T:
        return self

    def __mul__(self, other: _T) -> _T:
        return _T(_bcast(self.data, other.data, operator.mul))

    def __truediv__(self, other: _T) -> _T:
        return _T(_bcast(self.data, other.data, operator.truediv))

    def sum(self, dim: int) -> _T:
        return _T([reduce(lambda x, y: _bcast(x, y, operator.add), row) for row in self.data])

    def clamp(self, min: float) -> _T:  # noqa: A002 - torch's signature
        return _T(_leaves(self.data, lambda x: max(x, min)))

    def tolist(self) -> Any:
        return self.data


class _FakeTorch:
    @staticmethod
    @contextmanager
    def inference_mode() -> Iterator[None]:
        yield


class _FakeTokenizer:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        width = max(len(t.split()) for t in texts)
        mask = [[1] * len(t.split()) + [0] * (width - len(t.split())) for t in texts]
        return {"input_ids": [t.split() for t in texts], "attention_mask": _T(mask)}


class _Out:
    def __init__(self, hidden: _T) -> None:
        self.last_hidden_state = hidden


class _FakeEncoder:
    """Token 'a' -> [1, 0], anything else -> [0, 1]; padding -> [100, 100]."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def __call__(self, input_ids: list[list[str]], attention_mask: _T) -> _Out:
        if self.fail:
            raise RuntimeError("out of memory")
        width = len(attention_mask.data[0])
        rows = []
        for tokens in input_ids:
            row = [[1.0, 0.0] if t == "a" else [0.0, 1.0] for t in tokens]
            rows.append(row + [[100.0, 100.0]] * (width - len(tokens)))
        return _Out(_T(rows))


def _embedder(fail: bool = False, batch_size: int = 32) -> TransformersEmbedder:
    return TransformersEmbedder(tokenizer=_FakeTokenizer(), model=_FakeEncoder(fail),
                                torch_module=_FakeTorch(), batch_size=batch_size)


def test_embedder_mean_pools_ignoring_padding_and_normalises() -> None:
    vectors = _embedder().embed(["a a", "a b b b"])
    assert vectors[0] == pytest.approx((1.0, 0.0))  # padding ([100,100]) ignored
    assert vectors[1] == pytest.approx(normalise([0.25, 0.75]))


def test_embedder_batches() -> None:
    embedder = _embedder(batch_size=1)
    assert len(embedder.embed(["a", "b", "a b"])) == 3


def test_embedder_wraps_failures() -> None:
    with pytest.raises(CacheUnavailableError, match="out of memory"):
        _embedder(fail=True).embed(["a"])


def test_embedder_rejects_bad_sizes() -> None:
    with pytest.raises(ConfigurationError):
        TransformersEmbedder(tokenizer=None, model=None, torch_module=None, batch_size=0)


def test_embedder_load_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ConfigurationError, match="semantic: false"):
        TransformersEmbedder.load()


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def _entry(key: str, namespace: str = "ns", vector: Sequence[float] | None = None,
           query: str = "q") -> CacheEntry:
    return CacheEntry(key=key, namespace=namespace, query=query,
                      vector=normalise(vector) if vector else None, text=f"answer {key}",
                      finish_reason="stop", usage=Usage.of(1, 1), upstream_model="m",
                      provider="p")


async def test_store_get_put_and_ttl() -> None:
    clock = FakeClock()
    store = InMemoryCacheStore(10, clock=clock)
    await store.put(_entry("k"), ttl_s=60)
    assert (await store.get("k")) is not None
    clock.now += 61
    assert await store.get("k") is None
    assert store.size() == 0


async def test_store_evicts_least_recently_used() -> None:
    store = InMemoryCacheStore(2, clock=FakeClock())
    for key in ("a", "b"):
        await store.put(_entry(key), ttl_s=60)
    a = await store.get("a")
    assert a is not None
    await store.touch(a)  # "b" is now the least recently used
    await store.put(_entry("c"), ttl_s=60)
    assert await store.get("b") is None
    assert await store.get("a") is not None and await store.get("c") is not None
    assert a.hits == 1


async def test_store_replacing_a_key_keeps_one_entry() -> None:
    store = InMemoryCacheStore(10, clock=FakeClock())
    await store.put(_entry("k"), ttl_s=60)
    await store.put(_entry("k"), ttl_s=60)
    assert store.size() == 1


async def test_nearest_searches_only_its_namespace_and_skips_expired() -> None:
    clock = FakeClock()
    store = InMemoryCacheStore(10, clock=clock)
    await store.put(_entry("close-other-ns", namespace="other", vector=[1, 0]), ttl_s=60)
    await store.put(_entry("far", vector=[0, 1]), ttl_s=600)
    await store.put(_entry("close", vector=[1, 0.1]), ttl_s=60)
    await store.put(_entry("no-vector"), ttl_s=600)

    found = await store.nearest("ns", normalise([1, 0]))
    assert isinstance(found, Neighbour) and found.entry.key == "close"

    clock.now += 61  # "close" expires
    found = await store.nearest("ns", normalise([1, 0]))
    assert found is not None and found.entry.key == "far"
    assert await store.nearest("missing", normalise([1, 0])) is None


async def test_store_clear_and_validation() -> None:
    store = InMemoryCacheStore(10)
    await store.put(_entry("k"), ttl_s=60)
    await store.clear()
    assert store.size() == 0
    with pytest.raises(ValueError):
        InMemoryCacheStore(0)


# --------------------------------------------------------------------------- #
# SingleFlight
# --------------------------------------------------------------------------- #


async def test_singleflight_shares_one_call() -> None:
    flights: SingleFlight[str] = SingleFlight()
    calls = 0

    async def work() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "result"

    results = await asyncio.gather(*(flights.run("k", work) for _ in range(5)))
    assert calls == 1
    assert [r for r, _ in results] == ["result"] * 5
    assert sum(shared for _, shared in results) == 4
    assert flights.inflight() == 0


async def test_singleflight_propagates_errors_to_followers() -> None:
    flights: SingleFlight[str] = SingleFlight()

    async def work() -> str:
        await asyncio.sleep(0.01)
        raise ValueError("upstream down")

    results = await asyncio.gather(*(flights.run("k", work) for _ in range(3)),
                                   return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in results)


async def test_singleflight_cancelled_leader_does_not_cancel_followers() -> None:
    flights: SingleFlight[str] = SingleFlight()
    calls = 0

    async def work() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return f"call {calls}"

    leader = asyncio.create_task(flights.run("k", work))
    await asyncio.sleep(0)  # leader registers
    follower = asyncio.create_task(flights.run("k", work))
    await asyncio.sleep(0.01)
    leader.cancel()
    result, shared = await follower
    assert (result, shared) == ("call 2", False)  # follower took over as leader
    with pytest.raises(asyncio.CancelledError):
        await leader


async def test_singleflight_distinct_keys_run_independently() -> None:
    flights: SingleFlight[str] = SingleFlight()

    async def work() -> str:
        await asyncio.sleep(0.01)
        return "x"

    results = await asyncio.gather(flights.run("a", work), flights.run("b", work))
    assert [shared for _, shared in results] == [False, False]


# --------------------------------------------------------------------------- #
# ResponseCache policy
# --------------------------------------------------------------------------- #


async def test_exact_hit_after_store() -> None:
    metrics = GatewayMetrics()
    cache = _cache(metrics=metrics)
    first = await cache.lookup(chat_request("What is the capital of France?"), "default")
    assert first.status is CacheStatus.MISS and first.writable
    assert await cache.store(first, _completion(), "primary")

    hit = await cache.lookup(chat_request("what is the capital of france"), "default")
    assert hit.hit and hit.match is MatchKind.EXACT
    assert hit.entry is not None and hit.entry.text == "Paris."
    assert hit.as_dict()["tokens_saved"] == 15
    assert _counter(metrics, "groundwire_cache_lookups_total", result="hit_exact") == 1
    assert _counter(metrics, "groundwire_cache_tokens_saved_total") == 15
    assert _counter(metrics, "groundwire_cache_entries") == 1


async def test_semantic_hit_for_a_reworded_question() -> None:
    embedder = TableEmbedder({
        "what is the capital of france": [1, 0, 0],
        "which city is france's capital": [0.97, 0.2, 0],
    })
    cache = _cache(embedder=embedder)
    await _prime(cache, chat_request("What is the capital of France?"))
    hit = await cache.lookup(chat_request("Which city is France's capital?"), "default")
    assert hit.hit and hit.match is MatchKind.SEMANTIC
    assert hit.similarity is not None and hit.similarity > 0.96
    assert hit.as_dict()["match"] == "semantic"


async def test_semantic_below_threshold_is_a_miss() -> None:
    embedder = TableEmbedder({"capital of france": [1, 0], "capital of germany": [0.8, 0.6]})
    cache = _cache(embedder=embedder)
    await _prime(cache, chat_request("capital of France"))
    miss = await cache.lookup(chat_request("capital of Germany"), "default")
    assert miss.status is CacheStatus.MISS
    assert miss.similarity == pytest.approx(0.8)
    assert miss.vector is not None  # reused when this answer gets stored


@pytest.mark.parametrize(("cached", "incoming", "reason"), [
    ("refund status for order 7781", "refund status for order 7782", "numbers_differ"),
    ("send the invoice to [EMAIL_1]", "send the invoice to me", "placeholders_differ"),
])
async def test_semantic_guard_rejects_lookalike_questions(cached: str, incoming: str,
                                                          reason: str) -> None:
    embedder = TableEmbedder({normalise_query(cached): [1, 0],
                              normalise_query(incoming): [0.999, 0.04]})
    cache = _cache(embedder=embedder)
    await _prime(cache, chat_request(cached))
    miss = await cache.lookup(chat_request(incoming), "default")
    assert miss.status is CacheStatus.MISS and miss.reason == reason


async def test_answers_are_not_shared_across_namespaces() -> None:
    cache = _cache()
    await _prime(cache, chat_request("q"))
    other = await cache.lookup(chat_request("q", temperature=0.2), "default")
    assert other.status is CacheStatus.MISS


@pytest.mark.parametrize(("request_", "flagged", "reason"), [
    (chat_request("q", groundwire={"cache": {"mode": "bypass"}}), False, "client_bypass"),
    (chat_request("q", temperature=1.5), False, "temperature"),
    (chat_request("q"), True, "injection_flagged"),
])
async def test_bypass_rules(request_: ChatCompletionRequest, flagged: bool, reason: str) -> None:
    cache = _cache()
    await _prime(cache, chat_request("q"))
    lookup = await cache.lookup(request_, "default", injection_flagged=flagged)
    assert lookup.status is CacheStatus.BYPASS and lookup.reason == reason
    assert not lookup.writable
    assert not await cache.store(lookup, _completion(), "primary")


async def test_any_temperature_is_cached_when_limit_is_none() -> None:
    cache = _cache(max_temperature=None)
    lookup = await cache.lookup(chat_request("q", temperature=2.0), "default")
    assert lookup.status is CacheStatus.MISS


async def test_refresh_skips_the_read_but_overwrites() -> None:
    cache = _cache()
    await _prime(cache, chat_request("q"), text="old")
    refresh = await cache.lookup(chat_request("q", groundwire={"cache": {"mode": "refresh"}}),
                                 "default")
    assert refresh.status is CacheStatus.BYPASS and refresh.writable
    assert await cache.store(refresh, _completion("new"), "primary")
    hit = await cache.lookup(chat_request("q"), "default")
    assert hit.entry is not None and hit.entry.text == "new"


async def test_refresh_computes_the_embedding_at_store_time() -> None:
    embedder = TableEmbedder({"q": [1, 0]})
    cache = _cache(embedder=embedder)
    refresh = await cache.lookup(chat_request("q", groundwire={"cache": {"mode": "refresh"}}),
                                 "default")
    await cache.store(refresh, _completion(), "primary")
    assert embedder.calls == ["q"]


@pytest.mark.parametrize("completion", [_completion(finish="length"), _completion(text="  ")])
async def test_truncated_or_empty_answers_are_not_stored(completion: Completion) -> None:
    metrics = GatewayMetrics()
    cache = _cache(metrics=metrics)
    lookup = await cache.lookup(chat_request("q"), "default")
    assert not await cache.store(lookup, completion, "primary")
    assert _counter(metrics, "groundwire_cache_stores_total", outcome="skipped") == 1


async def test_store_failures_fail_open() -> None:
    metrics = GatewayMetrics()
    cache = _cache(store=BrokenStore(10), metrics=metrics)
    lookup = await cache.lookup(chat_request("q"), "default")
    assert lookup.status is CacheStatus.ERROR and lookup.reason == "lookup_failed"
    assert lookup.writable
    assert not await cache.store(lookup, _completion(), "primary")
    assert _counter(metrics, "groundwire_cache_lookups_total", result="error") == 1
    assert _counter(metrics, "groundwire_cache_stores_total", outcome="error") == 1


async def test_embedder_failure_fails_open() -> None:
    cache = _cache(embedder=TableEmbedder(fail=True))
    lookup = await cache.lookup(chat_request("q"), "default")
    assert lookup.status is CacheStatus.ERROR
    assert not await cache.store(lookup, _completion(), "primary")


async def test_slow_lookup_is_skipped() -> None:
    cache = _cache(embedder=TableEmbedder(delay_s=0.2), lookup_timeout_s=0.05)
    lookup = await cache.lookup(chat_request("q"), "default")
    assert lookup.status is CacheStatus.ERROR and lookup.reason == "lookup_timeout"


async def test_clear_empties_the_cache() -> None:
    metrics = GatewayMetrics()
    cache = _cache(metrics=metrics)
    await _prime(cache, chat_request("q"))
    await cache.clear()
    assert (await cache.lookup(chat_request("q"), "default")).status is CacheStatus.MISS
    assert _counter(metrics, "groundwire_cache_entries") == 0


def test_metadata_never_contains_text() -> None:
    entry = _entry("k")
    lookup = CacheLookup(status=CacheStatus.HIT, entry=entry, match=MatchKind.SEMANTIC,
                         similarity=0.951234, age_s=3.14, coalesced=True)
    assert lookup.as_dict() == {"status": "hit", "match": "semantic", "similarity": 0.9512,
                                "age_s": 3.1, "tokens_saved": 2, "coalesced": True}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def test_build_cache_disabled_returns_none() -> None:
    assert build_cache(CacheConfig()) is None


def test_build_cache_exact_only_needs_no_model() -> None:
    cache = build_cache(CacheConfig(enabled=True))
    assert cache is not None and not cache.semantic


def test_build_cache_semantic_loads_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_load(cls: type, name: str, **kwargs: Any) -> Embedder:
        seen.update(name=name, **kwargs)
        return TableEmbedder()

    monkeypatch.setattr(TransformersEmbedder, "load", classmethod(fake_load))
    cache = build_cache(CacheConfig(enabled=True, semantic=True, embedding_max_length=128))
    assert cache is not None and cache.semantic
    assert seen == {"name": "sentence-transformers/all-MiniLM-L6-v2", "max_length": 128}
