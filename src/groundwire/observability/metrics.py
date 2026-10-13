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
        self.grounding_latency = Histogram(
            "groundwire_grounding_duration_seconds", "Grounding check latency.",
            buckets=_LATENCY_BUCKETS, registry=r)

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
