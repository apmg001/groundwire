"""Chat completions: the use case behind ``POST /v1/chat/completions``.

Responsibilities (and nothing else):

* resolve the route from the request's ``model``,
* apply the safety guard (injection screening, PII redaction/restoration),
* serve from the response cache when possible, and merge identical
  in-flight requests into one provider call,
* run cascade routes (cheap model first, strong model only when the answer
  fails the fact-check),
* enforce the tool gate on every tool call a model requests, after PII is
  restored and before the client (the agent) receives it,
* fact-check an agent's final answer against the tool results in the
  conversation, and flag/annotate/block it,
* enforce the end-to-end deadline across retries and failover,
* run grounding verification when the client asks for it and apply the
  flag/block policy,
* record request-level metrics, and shape the public response.

The cache sits *after* redaction and *before* restoration, so it only ever
holds placeholder text, and answers that fail grounding are never stored.

See :mod:`groundwire.services.chat.service` for how the modules fit together.
"""

from __future__ import annotations

from groundwire.services.chat.common import CACHE_PROVIDER
from groundwire.services.chat.service import ChatService, StreamSession

__all__ = ["CACHE_PROVIDER", "ChatService", "StreamSession"]
