"""Groundwire: a self-hosted, OpenAI-compatible LLM gateway.

Groundwire sits between applications and LLM providers (local or cloud) and
adds the reliability and trust layer most teams end up hand-rolling:

* provider routing with ordered fallback chains,
* retries with jittered exponential backoff and per-provider circuit breakers,
* end-to-end request deadlines,
* grounding verification of answers against caller-supplied context,
* structured logging and Prometheus metrics.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
