"""Prometheus metrics for the gateway.

Metrics live on an explicit :class:`CollectorRegistry` owned by each app
instance rather than the global default registry. That keeps tests isolated
(no "duplicated timeseries" errors) and lets you run several apps in one
process.

Label cardinality is kept bounded on purpose: labels are route aliases,
provider names and error codes, all of which come from config or code. Never
put user IDs, prompts or request IDs in labels.

Multi-worker note: with ``workers > 1`` each process has its own registry;
use ``prometheus_client``'s multiprocess mode or scrape per worker.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from groundwire.resilience.circuit_breaker import CircuitState

# Latency buckets sized for LLM calls: tens of ms (cache/mock) up to minutes.
_LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160)

_STATE_VALUE = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}


class GatewayMetrics:
    """All gateway metrics, bound to one registry.

    Attributes:
        registry: The registry metrics are registered on.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Create and register every metric.

        Args:
            registry: Registry to use; a fresh one is created if omitted.
        """
        self.registry = registry or CollectorRegistry()
        r = self.registry

        self.requests = Counter(
            "groundwire_requests_total", "Chat completion requests by outcome.",
            ["route", "provider", "status", "stream"], registry=r)
        self.request_latency = Histogram(
            "groundwire_request_duration_seconds", "End-to-end request latency.",
            ["route", "stream"], buckets=_LATENCY_BUCKETS, registry=r)
        self.provider_attempts = Counter(
            "groundwire_provider_attempts_total", "Provider attempts by outcome.",
            ["provider", "outcome", "error_code"], registry=r)
        self.provider_latency = Histogram(
            "groundwire_provider_duration_seconds", "Successful provider call latency.",
            ["provider"], buckets=_LATENCY_BUCKETS, registry=r)
        self.failovers = Counter(
            "groundwire_failovers_total", "Requests served by a non-primary provider.",
            ["route"], registry=r)
        self.circuit_state = Gauge(
            "groundwire_circuit_state", "Circuit state (0=closed, 1=half_open, 2=open).",
            ["provider"], registry=r)
        self.tokens = Counter(
            "groundwire_tokens_total", "Tokens processed.",
            ["provider", "kind"], registry=r)
        self.grounding = Counter(
            "groundwire_grounding_checks_total", "Grounding verdicts.",
            ["verdict"], registry=r)
        self.injection_checks = Counter(
            "groundwire_injection_checks_total", "Prompt-injection screening outcomes.",
            ["action"], registry=r)
        self.pii_redactions = Counter(
            "groundwire_pii_redactions_total", "Distinct PII values redacted, by type.",
            ["type"], registry=r)
        self.grounding_latency = Histogram(
            "groundwire_grounding_duration_seconds", "Grounding check latency.",
            buckets=_LATENCY_BUCKETS, registry=r)
        self.cache_lookups = Counter(
            "groundwire_cache_lookups_total",
            "Cache lookups (hit_exact, hit_semantic, miss, bypass, error).",
            ["result"], registry=r)
        self.cache_stores = Counter(
            "groundwire_cache_stores_total", "Cache writes (stored, skipped, error).",
            ["outcome"], registry=r)
        self.cache_lookup_latency = Histogram(
            "groundwire_cache_lookup_duration_seconds", "Cache lookup latency.",
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
            registry=r)
        self.cache_entries = Gauge(
            "groundwire_cache_entries", "Entries currently held by the cache.", registry=r)
        self.cache_tokens_saved = Counter(
            "groundwire_cache_tokens_saved_total",
            "Provider tokens not spent because an answer was served from cache.", registry=r)
        self.cascade_requests = Counter(
            "groundwire_cascade_requests_total",
            "Cascade outcomes: which tier served the answer, and why.",
            ["cascade", "tier", "reason"], registry=r)
        self.cascade_check_latency = Histogram(
            "groundwire_cascade_check_duration_seconds",
            "Time to fact-check the cheap tier's answer.",
            ["cascade"], buckets=_LATENCY_BUCKETS, registry=r)
        # Labelled by decision and reason only: tool names come from clients, so
        # a per-tool label would have unbounded cardinality.
        self.tool_calls = Counter(
            "groundwire_tool_calls_total", "Tool calls requested by models, by policy verdict.",
            ["decision", "reason"], registry=r)
        self.approvals = Counter(
            "groundwire_approvals_total",
            "Approval lifecycle events (requested, deduplicated, approved, rejected, "
            "expired, released, unavailable).",
            ["event"], registry=r)
        self.approvals_pending = Gauge(
            "groundwire_approvals_pending", "Approval requests waiting for a reviewer.",
            registry=r)
        self.answer_checks = Counter(
            "groundwire_answer_checks_total",
            "Agent answers fact-checked against tool results, by verdict.",
            ["verdict"], registry=r)
        self.answer_check_issues = Counter(
            "groundwire_answer_check_issues_total",
            "Problems found in agent answers, by kind.",
            ["kind"], registry=r)
        self.coalesced_requests = Counter(
            "groundwire_coalesced_requests_total",
            "Requests answered by an identical in-flight request.", registry=r)
        self.load_shed = Counter(
            "groundwire_load_shed_total",
            "New requests refused with 503 because the gateway was overloaded, by reason.",
            ["reason"], registry=r)
        self.event_loop_lag = Gauge(
            "groundwire_event_loop_lag_seconds",
            "Smoothed event-loop lag (how long ready work waits to run).", registry=r)
        self.in_flight = Gauge(
            "groundwire_requests_in_flight",
            "Chat requests admitted and not yet finished (streams included).", registry=r)
        self.concurrency_limit = Gauge(
            "groundwire_concurrency_limit",
            "Adaptive limit on chat requests in flight (lowered when the event loop lags).",
            registry=r)

    # ------------------------------------------------------------------ #
    # Recording helpers
    # ------------------------------------------------------------------ #

    def set_circuit_state(self, provider: str, state: CircuitState) -> None:
        """Record a provider's circuit state.

        Args:
            provider: Provider name.
            state: New state.
        """
        self.circuit_state.labels(provider=provider).set(_STATE_VALUE[state])

    def record_attempt(self, provider: str, outcome: str, error_code: str = "") -> None:
        """Count one provider attempt.

        Args:
            provider: Provider name.
            outcome: ``success``, ``error`` or ``skipped_circuit_open``.
            error_code: Error code for failed attempts.
        """
        self.provider_attempts.labels(provider=provider, outcome=outcome,
                                      error_code=error_code).inc()

    def record_tokens(self, provider: str, prompt: int, completion: int) -> None:
        """Count tokens for a provider.

        Args:
            provider: Provider name.
            prompt: Prompt tokens.
            completion: Completion tokens.
        """
        if prompt:
            self.tokens.labels(provider=provider, kind="prompt").inc(prompt)
        if completion:
            self.tokens.labels(provider=provider, kind="completion").inc(completion)

    def render(self) -> bytes:
        """Return metrics in the Prometheus text exposition format."""
        return generate_latest(self.registry)
