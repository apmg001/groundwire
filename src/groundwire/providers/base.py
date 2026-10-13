"""Provider port: the interface every LLM adapter implements.

The router and services depend only on :class:`Provider`. Adding a new vendor
means writing one adapter class; nothing else in the gateway changes.

Contract for implementations:

* Never raise raw transport/library exceptions. Translate every failure into a
  :class:`~groundwire.core.errors.ProviderError` subclass so the resilience
  layer can decide whether to retry, fail over, or surface the error.
* ``stream`` must raise *before yielding* if the call cannot start, so the
  router can still fail over to another provider.
* Release network resources in :meth:`aclose`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from groundwire.domain.models import ChatCompletionRequest, Completion, StreamChunk


class Provider(ABC):
    """Abstract LLM provider.

    Attributes:
        name: Unique provider name (from config).
        model: Upstream model identifier.
    """

    def __init__(self, name: str, model: str) -> None:
        """Initialise common provider state.

        Args:
            name: Unique provider name.
            model: Upstream model identifier.
        """
        self.name = name
        self.model = model

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> Completion:
        """Run a non-streaming completion.

        Args:
            request: Validated client request.

        Returns:
            The completion.

        Raises:
            ProviderError: On any upstream failure (classified by subclass).
        """

    @abstractmethod
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion as chunks.

        Args:
            request: Validated client request.

        Returns:
            An async iterator of chunks; the last one carries ``finish_reason``.

        Raises:
            ProviderError: On any upstream failure (classified by subclass).
        """

    async def aclose(self) -> None:  # noqa: B027 - optional hook, default no-op
        """Release resources (HTTP pools...). Default: nothing to release."""

    def __repr__(self) -> str:
        """Return a debug representation without secrets."""
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"
