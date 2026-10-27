"""Demo identity: a random, client-generated id in ``X-User-Id``.

Implements :class:`~money_agent.domain.ports.IdentityProvider`. The browser
generates 128 random bits once and sends them on every request. That cannot be
guessed, which is enough for a demo on synthetic data, and **not** enough for
real bank data: in production, replace this class with one that verifies a
signed token from your login provider (OIDC/JWT). Nothing else changes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from money_agent.core.errors import AuthenticationError

USER_ID_HEADER = "x-user-id"
_USER_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class DemoIdentity:
    """Trusts a well-formed ``X-User-Id`` header (demo only; see module docstring)."""

    async def identify(self, headers: Mapping[str, str]) -> str:
        """Read and validate the header.

        Args:
            headers: Request headers.

        Returns:
            The user id.

        Raises:
            AuthenticationError: Missing or malformed id.
        """
        user_id = headers.get(USER_ID_HEADER, "")
        if not _USER_ID.match(user_id):
            raise AuthenticationError("missing or malformed X-User-Id",
                                      public_message="Reload the page to start a session.")
        return user_id
