"""Exception hierarchy for Groundwire.

Every error raised inside the gateway derives from :class:`GroundwireError`.
Each class carries two pieces of metadata the rest of the system relies on:

* ``http_status`` / ``code``: how the API layer renders the error to clients,
  in an OpenAI-compatible ``{"error": {...}}`` envelope.
* ``retryable`` (provider errors only): whether the resilience layer may retry
  the same provider, and whether the failure should count against that
  provider's circuit breaker.

Keeping this policy on the exception types (rather than scattered ``if``
statements) means adapters only have to classify failures once, and the
router, retry policy and API layer all agree on what a failure means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class GroundwireError(Exception):
    """Base class for all Groundwire errors.

    Attributes:
        http_status: HTTP status the API layer should return.
        code: Stable, machine-readable error code for clients.
    """

    http_status: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Create the error.

        Args:
            message: Human-readable description, safe to show to API clients.
            details: Optional structured context included in the error body.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_payload(self) -> dict[str, Any]:
        """Render the error in an OpenAI-compatible envelope.

        Returns:
            A JSON-serialisable dict of the form ``{"error": {...}}``.
        """
        body: dict[str, Any] = {"message": self.message, "type": self.code, "code": self.code}
        if self.details:
            body["details"] = self.details
        return {"error": body}


# --------------------------------------------------------------------------- #
# Configuration and request errors
# --------------------------------------------------------------------------- #


class ConfigurationError(GroundwireError):
    """The gateway configuration is invalid. Raised at startup (fail fast)."""

    code = "configuration_error"


class InvalidRequestError(GroundwireError):
    """The client request is malformed or unsupported."""

    http_status = 400
    code = "invalid_request_error"


class ModelNotFoundError(GroundwireError):
    """The requested model/route alias is not configured."""

    http_status = 404
    code = "model_not_found"


# --------------------------------------------------------------------------- #
# Provider errors
# --------------------------------------------------------------------------- #


class ProviderError(GroundwireError):
    """A single upstream provider call failed.

    Attributes:
        provider: Name of the provider that failed.
        retryable: True if retrying (or failing over) may succeed; such
            failures also count against the provider's circuit breaker.
        failover: True if the router should try the next provider in the chain.
        retry_after_s: Upstream back-off hint in seconds, if one was given.
    """

    http_status = 502
    code = "provider_error"
    retryable: bool = False
    failover: bool = True

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        upstream_status: int | None = None,
        retry_after_s: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: Description of the failure.
            provider: Name of the provider that failed.
            upstream_status: HTTP status returned by the provider, if any.
            retry_after_s: Parsed ``Retry-After`` hint, if any.
            details: Optional structured context.
        """
        super().__init__(message, details=details)
        self.provider = provider
        self.upstream_status = upstream_status
        self.retry_after_s = retry_after_s


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within its timeout."""

    code = "provider_timeout"
    retryable = True


class ProviderConnectionError(ProviderError):
    """The provider could not be reached (DNS, refused, reset, TLS...)."""

    code = "provider_unreachable"
    retryable = True


class ProviderRateLimitError(ProviderError):
    """The provider returned HTTP 429."""

    code = "provider_rate_limited"
    retryable = True


class ProviderServerError(ProviderError):
    """The provider returned a 5xx (or Anthropic's 529 overloaded)."""

    code = "provider_server_error"
    retryable = True


class ProviderAuthError(ProviderError):
    """The provider rejected our credentials (401/403).

    Not retryable (it will not fix itself), but we fail over: one provider's
    bad key is a gateway misconfiguration, not the client's fault.
    """

    code = "provider_auth_error"
    retryable = False
    failover = True


class ProviderBadRequestError(ProviderError):
    """The provider rejected the request itself (400/404/422).

    Not retryable and no failover: the same request would most likely be
    rejected everywhere, so we surface it to the client as a 400.
    """

    http_status = 400
    code = "upstream_rejected_request"
    retryable = False
    failover = False


class ProviderResponseError(ProviderError):
    """The provider answered, but the payload was malformed or empty."""

    code = "provider_bad_response"
    retryable = True


# --------------------------------------------------------------------------- #
# Routing / resilience errors
# --------------------------------------------------------------------------- #


class CircuitOpenError(GroundwireError):
    """A provider's circuit breaker is open; the call was short-circuited."""

    http_status = 503
    code = "circuit_open"

    def __init__(self, provider: str) -> None:
        """Create the error.

        Args:
            provider: Name of the provider whose circuit is open.
        """
        super().__init__(f"circuit open for provider '{provider}'")
        self.provider = provider


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Outcome of one provider attempt within a routed request.

    Attributes:
        provider: Provider name.
        outcome: ``"success"``, ``"error"`` or ``"skipped_circuit_open"``.
        error_code: Error code when the attempt failed.
        message: Error message when the attempt failed.
    """

    provider: str
    outcome: str
    error_code: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return {k: v for k, v in {
            "provider": self.provider,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "message": self.message,
        }.items() if v is not None}


class AllProvidersFailedError(GroundwireError):
    """Every provider in the route's fallback chain failed or was skipped."""

    http_status = 503
    code = "all_providers_failed"

    def __init__(self, route: str, attempts: list[AttemptRecord]) -> None:
        """Create the error.

        Args:
            route: Route alias that was being served.
            attempts: Per-provider outcomes, in the order they were tried.
        """
        super().__init__(
            f"all providers failed for route '{route}'",
            details={"route": route, "attempts": [a.as_dict() for a in attempts]},
        )
        self.route = route
        self.attempts = attempts


class GatewayTimeoutError(GroundwireError):
    """The end-to-end request deadline expired before any provider succeeded."""

    http_status = 504
    code = "gateway_timeout"


# --------------------------------------------------------------------------- #
# Guardrail errors
# --------------------------------------------------------------------------- #


class GroundingViolationError(GroundwireError):
    """The answer failed grounding verification and the policy is ``block``."""

    http_status = 422
    code = "grounding_violation"


class GuardrailUnavailableError(GroundwireError):
    """A guardrail could not run (model missing, crashed...) and policy is fail-closed."""

    http_status = 503
    code = "guardrail_unavailable"


# --------------------------------------------------------------------------- #
# Evaluation errors
# --------------------------------------------------------------------------- #


class DatasetError(GroundwireError):
    """A benchmark dataset is missing, unreadable or malformed."""

    code = "dataset_error"
