"""Error hierarchy for the money agent.

Every error the application raises on purpose is a :class:`MoneyAgentError`
carrying three things:

* ``code``: a stable, machine-readable identifier (clients branch on it);
* ``http_status``: what the API layer answers with;
* ``public_message``: text that is safe to show an end user.

The exception's own message (``str(exc)``) is for logs only: it may contain
internal detail (an upstream error body, a SQL state), which must never reach
a user. The API layer sends ``public_message`` and logs the rest.
"""

from __future__ import annotations

from typing import Any, ClassVar


class MoneyAgentError(Exception):
    """Base for all expected application errors."""

    code: ClassVar[str] = "internal_error"
    http_status: ClassVar[int] = 500
    default_public_message: ClassVar[str] = "Something went wrong on our side. Please try again."

    def __init__(self, message: str, *, public_message: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        """Create the error.

        Args:
            message: Internal description, for logs.
            public_message: User-safe text; the class default when omitted.
            details: Extra machine-readable context (must be user-safe).
        """
        super().__init__(message)
        self.public_message = public_message or self.default_public_message
        self.details = details or {}

    def to_payload(self, request_id: str | None = None) -> dict[str, Any]:
        """The public error body.

        Args:
            request_id: Correlation id, so support can find the logs.

        Returns:
            ``{"error": {...}}`` without any internal detail.
        """
        error: dict[str, Any] = {"code": self.code, "message": self.public_message}
        if self.details:
            error["details"] = self.details
        if request_id:
            error["request_id"] = request_id
        return {"error": error}


class ConfigurationError(MoneyAgentError):
    """Settings are invalid; raised at startup, never during a request."""

    code = "configuration_error"


class AuthenticationError(MoneyAgentError):
    """The caller could not be identified."""

    code = "unauthenticated"
    http_status = 401
    default_public_message = "Please sign in again."


class InvalidInputError(MoneyAgentError):
    """A request is well-formed JSON but its values are not acceptable."""

    code = "invalid_input"
    http_status = 400
    default_public_message = "Some of the values you sent are not valid."


class NotFoundError(MoneyAgentError):
    """A referenced resource does not exist (or belongs to someone else)."""

    code = "not_found"
    http_status = 404
    default_public_message = "We couldn't find that."


class LimitReachedError(MoneyAgentError):
    """A per-user quota (statement size, memory count, ...) was reached."""

    code = "limit_reached"
    http_status = 409
    default_public_message = "You've reached the limit for this."


class StatementTooLargeError(MoneyAgentError):
    """An uploaded statement exceeds the size or row limit."""

    code = "statement_too_large"
    http_status = 413
    default_public_message = "This statement is too large. Please upload a shorter period."


class StatementFormatError(MoneyAgentError):
    """An uploaded statement cannot be read as a transaction table."""

    code = "statement_unreadable"
    http_status = 422
    default_public_message = (
        "We couldn't read this statement. Please upload the CSV export from your bank, "
        "with date, description and amount columns.")


class StorageUnavailableError(MoneyAgentError):
    """The database could not be reached or failed unexpectedly."""

    code = "storage_unavailable"
    http_status = 503
    default_public_message = "We're having trouble saving your data. Please try again shortly."


class AssistantUnavailableError(MoneyAgentError):
    """The AI gateway (or every model behind it) failed or timed out."""

    code = "assistant_unavailable"
    http_status = 503
    default_public_message = "The assistant is busy right now. Please try again in a moment."


class MessageRefusedError(MoneyAgentError):
    """The gateway refused the message (for example, a prompt-injection attempt)."""

    code = "message_refused"
    http_status = 422
    default_public_message = (
        "This message was blocked by our safety checks. Please rephrase it and try again.")


class AgentError(MoneyAgentError):
    """The agent loop could not reach an answer (for example, too many steps)."""

    code = "agent_error"
    http_status = 502
    default_public_message = "The assistant couldn't finish that. Please try asking differently."
