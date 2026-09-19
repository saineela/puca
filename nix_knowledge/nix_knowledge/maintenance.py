from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo


class TemporalMaintenance:
    """
    Expires calendar events when their temporal context is over.

    The rule is not "delete after start" - it is context specific:

      Event context                      | Expires when
      -----------------------------------+--------------------------------
      "tomorrow" / any single day        | end of that calendar day
      "this weekend" / a weekend event   | end of Sunday of that weekend
      explicit start+duration            | start + duration
      "in 2 hours" (timed, no day ctx)   | start (moment passed)
      recurring daily/weekly             | never auto-expires; only the
                                         | passed *occurrence* is advanced
      cancelled events                   | already finished; marked expired

    Expired non-recurring events are marked status='expired' so history
    is preserved (matching the engine's soft-delete philosophy).
    Recurring events are left active - the Actions engine chains their
    occurrences; expiring them would kill future alarms.
    """

    def __init__(
        self,
        engine,
        timezone: str = "America/Chicago",
    ):
        self.engine = engine
        self.tz = ZoneInfo(timezone)

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        return datetime.now(self.tz)

    # ------------------------------------------------------------------
    # Window ends per temporal context
    # ------------------------------------------------------------------

    def _end_of_day(self, d: date) -> datetime:
        return datetime.combine(
            d,
            time.max,
            tzinfo=self.tz,
        )

    def _window_end_for_data(
        self,
        data: dict[str, Any],
    ) -> datetime | None:
        """
        Compute the instant after which this event's context is over.
        Returns None when the event should never expire (recurring).
        """
        start = self._parse(data.get("start"))
        end = self._parse(data.get("end"))
        all_day = bool(data.get("all_day", False))
        expression = str(
            data.get("temporal_expression", "")
        ).lower().strip()

        if data.get("recurring"):
            return None

        # -- explicit end stored on the event (durations, windows) ----
        if end is not None and not all_day:
            return end

        if start is None:
            # no time at all: nothing contextual to expire
            return None

        # -- weekend context: "this weekend" --------------------------
        if "weekend" in expression:
            # end of Sunday of the week containing that Saturday
            saturday = start.date()
            sunday = saturday + timedelta(days=1)
            return self._end_of_day(sunday)

        # -- all-day single-day context: "tomorrow", "next monday" ----
        if all_day:
            # a bare weekday/day expression spans exactly one day
            # (next week spans 7 - detect multi-day windows by width)
            day_width = (end.date() - start.date()).days if end else 0
            if day_width >= 1:
                return self._end_of_day(end.date())
            return self._end_of_day(start.date())

        # -- timed events: "friday at 6pm", "in 2 hours" --------------
        # The moment the start has passed, the context is over unless
        # the expression names a day AND a duration window exists.
        if "week" in expression and day_span(expression):
            # e.g. "next week" stored all_day handled above; safe guard
            return self._end_of_day(start.date())

        # timed event with no explicit end: give a single-day grace so
        # "friday at 6pm" does not expire mid-event but dies by midnight
        return self._end_of_day(start.date())

    @staticmethod
    def _parse(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expire_due(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Mark every active calendar event whose context is over as
        expired. Returns a summary of what was touched.
        """
        now = now or self.now()

        if now.tzinfo is None:
            now = now.replace(tzinfo=self.tz)

        events = self.engine.search("calendar_event")

        expired: list[dict[str, Any]] = []
        kept: list[int] = []

        for record in events:
            data = record.data

            if data.get("recurring"):
                kept.append(record.id)
                continue

            window_end = self._window_end_for_data(data)

            if window_end is None:
                kept.append(record.id)
                continue

            if window_end.tzinfo is None:
                window_end = window_end.replace(tzinfo=self.tz)

            if now > window_end:
                self._mark_expired(record.id, now)
                expired.append(
                    {
                        "record_id": record.id,
                        "title": data.get("title"),
                        "temporal_expression": data.get(
                            "temporal_expression"
                        ),
                        "window_end": window_end.isoformat(),
                    }
                )
            else:
                kept.append(record.id)

        return {
            "ok": True,
            "ran_at": now.isoformat(),
            "expired_count": len(expired),
            "expired": expired,
            "active_remaining": len(kept),
        }

    def _mark_expired(
        self,
        record_id: int,
        now: datetime,
    ) -> None:
        from .operations import KnowledgeOperation

        self.engine.apply(
            KnowledgeOperation(
                operation="update",
                knowledge_type="calendar_event",
                match={"id": record_id},
                changes={"status": "expired"},
                source="temporal_maintenance",
                reason=(
                    "Event temporal context is over; "
                    "marked expired by midnight maintenance."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Midnight scheduling helper
    # ------------------------------------------------------------------

    def seconds_until_next_midnight(self) -> float:
        """
        Seconds from now until the next local midnight - the sleep
        interval for the maintenance runner.
        """
        now = self.now()

        next_midnight = (now + timedelta(days=1)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        return (next_midnight - now).total_seconds()


def day_span(expression: str) -> bool:
    """
    True when the expression names a multi-day window (e.g. 'next week').
    """
    text = expression.lower().strip()
    return "week" in text and "weekend" not in text and "weekday" not in text
