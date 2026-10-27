"""The real clock (the :class:`~money_agent.domain.ports.Clock` port)."""

from __future__ import annotations

from datetime import UTC, date, datetime, tzinfo


class SystemClock:
    """Real time; "today" is computed in the users' time zone."""

    def __init__(self, zone: tzinfo = UTC) -> None:
        """Create the clock.

        Args:
            zone: Time zone for :meth:`today` and :meth:`now`.
        """
        self._zone = zone

    def now(self) -> datetime:
        """Current time (timezone-aware)."""
        return datetime.now(self._zone)

    def today(self) -> date:
        """Today's date in the configured zone (it can differ from UTC's)."""
        return self.now().date()
