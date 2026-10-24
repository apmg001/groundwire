"""Exception hierarchy for Groundwire.

Every error raised inside the gateway derives from :class:`GroundwireError`.
Each class carries two pieces of metadata the rest of the system relies on:

* ``http_status`` / ``code``: how the API layer renders the error to clients,
  in an OpenAI-compatible ``{"error": {...}}`` envelope.
* ``public_message``: what clients see. For errors raised by Groundwire it is
  the message itself; for provider errors it is a fixed, safe sentence per
  error type, because the raw message embeds the upstream response body
  (vendor model names, account hints, internal ids). The raw text stays on
  ``message`` for logs only. Every error body also carries the ``request_id``,
  so a user can quote it and an operator can find the full detail in the logs.
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

from groundwire.core.context import get_request_id

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
            message: Full description, for logs (and for clients, unless a
                subclass overrides :attr:`public_message`).
            details: Optional structured context included in the error body.
                Must never carry upstream response text.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    @property
    def public_message(self) -> str:
        """The message clients see (the full message, unless overridden)."""
        return self.message

    def to_payload(self) -> dict[str, Any]:
        """Render the error in an OpenAI-compatible envelope, safe for clients.

        Returns:
            ``{"error": {"message", "type", "code", "request_id", ["details"]}}``.
        """
        body: dict[str, Any] = {"message": self.public_message, "type": self.code,
                                "code": self.code, "request_id": get_request_id()}
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
    public_text: str = "The AI provider returned an error."

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

    @property
    def public_message(self) -> str:
        """A fixed sentence per error type; never the upstream response text."""
        return self.public_text


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within its timeout."""

    code = "provider_timeout"
    retryable = True
    public_text = "The AI provider took too long to respond."


class ProviderConnectionError(ProviderError):
    """The provider could not be reached (DNS, refused, reset, TLS...)."""

    code = "provider_unreachable"
    retryable = True
    public_text = "The AI provider could not be reached."


class ProviderRateLimitError(ProviderError):
    """The provider returned HTTP 429."""

    code = "provider_rate_limited"
    retryable = True
    public_text = "The AI provider is limiting requests right now. Please retry shortly."


class ProviderServerError(ProviderError):
    """The provider returned a 5xx (or Anthropic's 529 overloaded)."""

    code = "provider_server_error"
    retryable = True
    public_text = "The AI provider had an internal error."


class ProviderAuthError(ProviderError):
    """The provider rejected our credentials (401/403).

    Not retryable (it will not fix itself), but we fail over: one provider's
    bad key is a gateway misconfiguration, not the client's fault.
    """

    code = "provider_auth_error"
    retryable = False
    failover = True
    public_text = "The gateway could not authenticate with the AI provider."


class ProviderNotFoundError(ProviderError):
    """The provider answered 404: its model or endpoint does not exist.

    The client asked for a route alias, not an upstream model, so a 404 means
    this provider is misconfigured (or the vendor retired the model, as Groq
    did with ``llama-3.1-8b-instant``). Like an auth error: not retryable,
    but fail over, and count against the breaker so a broken provider is
    skipped instead of failing every request.
    """

    code = "provider_not_found"
    retryable = False
    failover = True
    public_text = "The AI model configured for this route is not available."


class ProviderBadRequestError(ProviderError):
    """The provider rejected the request itself (400/422 and other 4xx).

    Not retryable and no failover: the same request would most likely be
    rejected everywhere, so we surface it to the client as a 400.
    """

    http_status = 400
    code = "upstream_rejected_request"
    retryable = False
    failover = False
    public_text = ("The AI provider could not process this request (for example, it may be "
                   "too long or use an option the model does not support).")


class ProviderResponseError(ProviderError):
    """The provider answered, but the payload was malformed or empty."""

    code = "provider_bad_response"
    retryable = True
    public_text = "The AI provider sent an invalid response."


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
        message: Full error text when the attempt failed. Internal: logged,
            never serialised for clients (it may embed upstream responses).
    """

    provider: str
    outcome: str
    error_code: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the client-safe view of the record (no raw error text)."""
        return {k: v for k, v in {
            "provider": self.provider,
            "outcome": self.outcome,
            "error_code": self.error_code,
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


class OverloadedError(GroundwireError):
    """The gateway refused a new request because it is overloaded (load shedding).

    Attributes:
        retry_after_s: Seconds the client should wait before retrying
            (sent as the ``Retry-After`` header).
    """

    http_status = 503
    code = "server_overloaded"

    def __init__(self, message: str, *, retry_after_s: int,
                 details: dict[str, Any] | None = None) -> None:
        """Create the error.

        Args:
            message: Human-readable description.
            retry_after_s: Suggested wait before retrying.
            details: Optional structured context.
        """
        super().__init__(message, details=details)
        self.retry_after_s = retry_after_s


# --------------------------------------------------------------------------- #
# Guardrail errors
# --------------------------------------------------------------------------- #


class GroundingViolationError(GroundwireError):
    """The answer failed grounding verification and the policy is ``block``."""

    http_status = 422
    code = "grounding_violation"


class ToolCallBlockedError(GroundwireError):
    """The model requested a tool call that the tool policy forbids (``on_violation: error``)."""

    http_status = 422
    code = "tool_call_blocked"


class ApprovalNotFoundError(GroundwireError):
    """No approval with this id (never existed, or purged after retention)."""

    http_status = 404
    code = "approval_not_found"


class ApprovalConflictError(GroundwireError):
    """The approval was already decided (or changed concurrently)."""

    http_status = 409
    code = "approval_already_decided"


class ApprovalExpiredError(GroundwireError):
    """The approval expired before a decision was made."""

    http_status = 410
    code = "approval_expired"


class ApprovalsDisabledError(GroundwireError):
    """Approvals, or the reviewer API, are not enabled on this gateway."""

    http_status = 503
    code = "approvals_disabled"


class ReviewerAuthError(GroundwireError):
    """Missing or wrong reviewer token."""

    http_status = 401
    code = "unauthorized"


class ApprovalStoreError(GroundwireError):
    """The approval store failed or is full (callers fail closed)."""

    http_status = 503
    code = "approval_store_unavailable"


class AnswerCheckFailedError(GroundwireError):
    """An agent's answer contradicts its tool results and the policy is ``block``."""

    http_status = 422
    code = "answer_check_failed"


class PromptInjectionError(GroundwireError):
    """The request looks like a prompt-injection attempt and policy is ``block``."""

    http_status = 400
    code = "prompt_injection_detected"


class GuardrailUnavailableError(GroundwireError):
    """A guardrail could not run (model missing, crashed...) and policy is fail-closed."""

    http_status = 503
    code = "guardrail_unavailable"


class CacheUnavailableError(GroundwireError):
    """The response cache could not serve a lookup or store an entry.

    Internal only: the chat service catches it and carries on without the
    cache (fail-open), so clients never see this error.
    """

    http_status = 503
    code = "cache_unavailable"


# --------------------------------------------------------------------------- #
# Evaluation errors
# --------------------------------------------------------------------------- #


class DatasetError(GroundwireError):
    """A benchmark dataset is missing, unreadable or malformed."""

    code = "dataset_error"
