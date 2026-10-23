"""Load-test profiles: the user counts over time for each scenario.

Shared by the Locust shape (``locustfile.py``) and the report
(``run.py``), so the stage windows in a report always match what ran.
"""

from __future__ import annotations

Stage = tuple[float, int, float]  # (end time s, users, spawn rate users/s)

PROFILES = ("load", "stress", "spike", "chaos", "soak")

# Profiles whose stages are fixed (they ignore --duration and --users).
FIXED: dict[str, list[Stage]] = {
    # Steps until latency or errors break: where is the capacity limit?
    "stress": [(30, 25, 50), (60, 50, 50), (90, 100, 50), (120, 200, 50),
               (150, 400, 100), (180, 600, 100)],
    # 15x the users within one second, then back: survival and recovery.
    "spike": [(30, 20, 20), (75, 300, 300), (120, 20, 300)],
}


def stages(profile: str, *, duration_s: float, users: int) -> list[Stage]:
    """Return the stages for ``profile``.

    Args:
        profile: One of :data:`PROFILES`.
        duration_s: Plateau length for ``load``, ``chaos`` and ``soak``.
        users: Plateau users for those profiles.

    Returns:
        ``(end_s, users, spawn_rate)`` stages; the test stops after the last.

    Raises:
        ValueError: Unknown profile, or a non-positive duration or user count.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
    if profile in FIXED:
        return FIXED[profile]
    if duration_s <= 0 or users <= 0:
        raise ValueError("duration_s and users must be positive")
    spawn = 20.0 if profile == "soak" else 50.0
    return [(duration_s, users, spawn)]
