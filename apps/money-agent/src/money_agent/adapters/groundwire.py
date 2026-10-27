"""The :class:`~money_agent.domain.ports.LLMClient` port, over Groundwire's HTTP API.

The money agent never talks to a model provider directly. Every model call
goes through Groundwire, which (configured in ``deploy/groundwire.yaml``):

* redacts personal data (phone numbers, emails, card numbers, PAN, Aadhaar)
  before it reaches the model, and restores it in the answer;
* screens messages for prompt injection;
* fails over between models and sheds load when overloaded;
* fact-checks the final answer against the tool results (``answer_check``).

This module only speaks HTTP. Its policies:

* **Retries** only where a retry can help: connection errors, timeouts, 429,
  502, 503, 504, with exponential backoff plus jitter, honouring
  ``Retry-After`` (capped). A 4xx is the request's fault and is not retried.
* **Errors** become :class:`AssistantUnavailableError` (gateway or models
  down) or :class:`MessageRefusedError` (the gateway refused the content).
  The gateway's own error text is logged with its request id, never shown.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence
from typing import Any

import httpx

from money_agent.core.errors import AssistantUnavailableError, MessageRefusedError
from money_agent.core.logging import get_request_id
from money_agent.domain.ports import LLMReply, ToolCallRequest

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
REFUSAL_CODES = frozenset({"prompt_injection_detected", "tool_call_blocked",
                           "answer_check_failed", "grounding_violation"})
MAX_RETRY_AFTER_S = 5.0
BASE_BACKOFF_S = 0.3


class GroundwireClient:
    """Chat completions through a Groundwire gateway."""

    def __init__(self, client: httpx.AsyncClient, *, base_url: str, model: str,
                 api_key: str | None = None, timeout_s: float = 45.0,
                 max_retries: int = 2) -> None:
        """Create the client.

        Args:
            client: Shared HTTP client (pooling; owned and closed by the container).
            base_url: Gateway base URL, e.g. ``http://groundwire:8000``.
            model: Route alias to request.
            api_key: Bearer token, if the gateway requires one.
            timeout_s: Timeout per attempt.
            max_retries: Retries for transient failures.
        """
        self._client = client
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout_s
        self._max_retries = max_retries

    async def complete(self, messages: Sequence[dict[str, Any]], *,
                       tools: Sequence[dict[str, Any]], user: str) -> LLMReply:
        """Ask for the next turn (see the port).

        Args:
            messages: OpenAI-format conversation.
            tools: OpenAI-format tool definitions (empty: no tools this turn).
            user: Opaque end-user id.

        Returns:
            The reply, with the gateway's ``groundwire`` metadata.

        Raises:
            AssistantUnavailableError: The gateway or every model behind it failed.
            MessageRefusedError: The gateway refused the content.
        """
        payload: dict[str, Any] = {"model": self._model, "messages": list(messages),
                                   "user": user, "temperature": 0.2}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        headers = dict(self._headers)
        request_id = get_request_id()
        if request_id:
            headers["X-Request-ID"] = request_id  # one id across both services' logs
        response = await self._post_with_retries(payload, headers)
        return self._parse(response)

    async def _post_with_retries(self, payload: dict[str, Any],
                                 headers: dict[str, str]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            final = attempt == self._max_retries
            try:
                response = await self._client.post(self._url, json=payload, headers=headers,
                                                   timeout=self._timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                logger.warning("gateway unreachable", extra={"attempt": attempt + 1,
                                                             "error": type(exc).__name__})
                if final:
                    raise AssistantUnavailableError(f"gateway unreachable: {exc!r}") from exc
                await asyncio.sleep(_backoff(attempt))
                continue
            if response.status_code not in RETRYABLE_STATUS or final:
                return response
            logger.warning("gateway busy; retrying", extra={
                "attempt": attempt + 1, "status": response.status_code})
            await asyncio.sleep(_backoff(attempt, response.headers.get("retry-after")))
        raise AssistantUnavailableError("unreachable")  # pragma: no cover - loop returns

    @staticmethod
    def _parse(response: httpx.Response) -> LLMReply:
        try:
            body: Any = response.json()
        except ValueError as exc:
            raise AssistantUnavailableError(
                f"gateway returned non-JSON (HTTP {response.status_code})") from exc
        if response.status_code != 200:
            error = body.get("error", {}) if isinstance(body, dict) else {}
            code = str(error.get("code", "unknown"))
            logger.warning("gateway error", extra={
                "status": response.status_code, "gateway_code": code,
                "gateway_request_id": error.get("request_id")})
            if code in REFUSAL_CODES:
                raise MessageRefusedError(f"gateway refused: {code}", details={"reason": code})
            raise AssistantUnavailableError(f"gateway HTTP {response.status_code}: {code}")
        try:
            choice = body["choices"][0]
            message = choice["message"]
            calls = tuple(ToolCallRequest(id=str(c["id"]), name=str(c["function"]["name"]),
                                          arguments=str(c["function"].get("arguments") or "{}"))
                          for c in message.get("tool_calls") or ())
        except (KeyError, IndexError, TypeError) as exc:
            raise AssistantUnavailableError("malformed gateway response") from exc
        gateway = body.get("groundwire")
        return LLMReply(content=message.get("content"), tool_calls=calls,
                        finish_reason=choice.get("finish_reason"),
                        gateway=gateway if isinstance(gateway, dict) else {})


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    """Exponential backoff with full jitter, or the server's ``Retry-After`` (capped)."""
    if retry_after:
        try:
            return min(float(retry_after), MAX_RETRY_AFTER_S)
        except ValueError:
            pass
    return random.uniform(0, BASE_BACKOFF_S * 2 ** attempt)  # noqa: S311 - jitter, not crypto
