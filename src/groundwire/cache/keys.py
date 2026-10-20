"""Cache keys: decide which requests are allowed to share an answer.

A cached answer may only be reused when the model would have seen the *same
input*. A key therefore has two parts:

* **namespace**: everything that must match *exactly*: the route, sampling
  parameters, the whole conversation except the latest message, the grounding
  documents, the configured version tag, and (optionally) the end user. Two
  requests in different namespaces never share answers, however similar.
* **query**: the latest message, normalised. Exact matching compares its hash;
  semantic matching compares its embedding, but only *within* the namespace.

Keys are built from the request **after PII redaction**, so a key never
contains raw personal data, and two customers asking "refund [EMAIL_1]" share
the answer *with the placeholder*, which each request then restores with its
own values. That is safe because the model only ever saw the placeholder.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from groundwire.domain.models import ChatCompletionRequest

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = " \t\n.?!"


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Identity of a request for caching purposes.

    Attributes:
        namespace: SHA-256 of everything that must match exactly.
        query: Normalised latest message (the part matched semantically).
        exact: SHA-256 of ``namespace`` + ``query`` (the exact-match key).
    """

    namespace: str
    query: str
    exact: str


def normalise_query(text: str) -> str:
    """Normalise a message so trivial differences still hit exactly.

    Unicode is NFKC-normalised, case-folded and whitespace-collapsed, and
    trailing punctuation is dropped ("What is X?" == "what is x").

    Args:
        text: Raw message text.

    Returns:
        The normalised text.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", folded).strip().rstrip(_TRAILING_PUNCTUATION).strip()


def _digest(payload: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable value.

    Args:
        payload: Value to hash (dict keys are sorted, so order never matters).

    Returns:
        Hex digest.
    """
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_cache_key(request: ChatCompletionRequest, route: str, *, namespace_tag: str,
                    partition_by_user: bool = False) -> CacheKey:
    """Build the cache key for a (redacted) request.

    Args:
        request: The request exactly as it will be sent to providers.
        route: Resolved route alias.
        namespace_tag: Configured version tag (bump to invalidate everything).
        partition_by_user: Include ``request.user`` in the namespace.

    Returns:
        The key.
    """
    *history, latest = request.messages
    grounding = request.groundwire.grounding if request.groundwire else None
    namespace = _digest({
        "tag": namespace_tag,
        "route": route,
        "params": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stop": request.stop_list(),
        },
        "history": [[m.role, m.content] for m in history],
        "latest_role": latest.role,
        "context": grounding.context if grounding else None,
        "user": request.user if partition_by_user else None,
    })
    query = normalise_query(latest.content)
    return CacheKey(namespace=namespace, query=query, exact=_digest([namespace, query]))
