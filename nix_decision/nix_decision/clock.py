from __future__ import annotations

from datetime import datetime, timedelta, timezone


class Clock:
    def now(self) -> datetime:
        raise NotImplementedError


class SystemClock(Clock):
    """
    Production clock.

    Nix Decision always uses the real system clock unless
    a different clock is explicitly injected.
    """

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SimulatedClock(Clock):
    """
    Testing clock.

    Allows the exact same cognitive engine to be tested against
    historical/future scenarios without changing the engine itself.
    """

    def __init__(self, start: datetime):
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def set(self, timestamp: datetime) -> None:
        self._now = timestamp
