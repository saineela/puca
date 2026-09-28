from __future__ import annotations

"""
TemporalContext: the time-management brain shared by every Knowledge
function.

The user rarely states absolute dates. Every request carries an
implicit temporal frame - "what do I have today", "show me next
week", "when am I free" - and every knowledge record has a temporal
footprint. This module is the single authority that connects the two:

  - window(name)         absolute [start, end] for a named window
  - parse_window(expr)   window for an arbitrary temporal expression
  - occurrence()         one concrete occurrence of a recurring event
  - occurrence_window()  every occurrence inside a window
  - relevancy()          time distance from now, for ranking
  - relative_label()     human labels ("today", "tomorrow", "in 3 days")

Python is authoritative for time; the selector model never handles
absolute dates.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime


class TemporalContext:
    def __init__(self, timezone: str = "America/Chicago"):
        self.timezone = ZoneInfo(timezone)

        self._resolver: object | None = None
        self._resolver_tz: str | None = None

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def _day_start(self, d: date) -> datetime:
        return datetime.combine(
            d,
            time.min,
            tzinfo=self.timezone,
        )

    def _day_end(self, d: date) -> datetime:
        return datetime.combine(
            d,
            time.max,
            tzinfo=self.timezone,
        )

    # ------------------------------------------------------------------
    # Resolver access (lazy to avoid an import cycle)
    # ------------------------------------------------------------------

    def _get_resolver(self):
        if self._resolver is None or self._resolver_tz != self.timezone:
            from .temporal import TemporalResolver

            self._resolver = TemporalResolver(
                timezone=str(self.timezone)
            )
            self._resolver_tz = str(self.timezone)

        return self._resolver

    def resolve(self, expression: str, now=None):
        return self._get_resolver().resolve(expression, now=now)

    # ------------------------------------------------------------------
    # Named windows
    # ------------------------------------------------------------------

    def window(self, name: str) -> Window:
        name = name.lower().strip()

        now = self.now()
        today = now.date()

        if name in {"today", "now"}:
            return Window(
                self._day_start(today),
                self._day_end(today),
            )

        if name == "tomorrow":
            d = today + timedelta(days=1)
            return Window(self._day_start(d), self._day_end(d))

        if name == "yesterday":
            d = today - timedelta(days=1)
            return Window(self._day_start(d), self._day_end(d))

        if name == "this_week":
            monday = today - timedelta(days=today.weekday())
            sunday = monday + timedelta(days=6)
            return Window(
                self._day_start(monday),
                self._day_end(sunday),
            )

        if name == "next_week":
            monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
            sunday = monday + timedelta(days=6)
            return Window(
                self._day_start(monday),
                self._day_end(sunday),
            )

        if name == "last_week":
            monday = (
                today
                - timedelta(days=today.weekday())
                - timedelta(days=7)
            )
            sunday = monday + timedelta(days=6)
            return Window(
                self._day_start(monday),
                self._day_end(sunday),
            )

        if name == "this_month":
            start = today.replace(day=1)
            if today.month == 12:
                nxt = today.replace(
                    year=today.year + 1, month=1, day=1
                )
            else:
                nxt = today.replace(month=today.month + 1, day=1)
            return Window(
                self._day_start(start),
                self._day_end(nxt - timedelta(days=1)),
            )

        if name == "this_weekend":
            weekday = today.weekday()

            if weekday in (5, 6):
                # said during the weekend: the current one
                saturday = today - timedelta(days=weekday - 5)
            else:
                saturday = today + timedelta(days=(5 - weekday) % 7)

            sunday = saturday + timedelta(days=1)
            return Window(
                self._day_start(saturday),
                self._day_end(sunday),
            )

        if name == "upcoming":
            return Window(now, self._day_end(today + timedelta(days=365)))

        if name == "upcoming_week":
            return Window(now, self._day_end(today + timedelta(days=7)))

        if name == "recent":
            return Window(
                self._day_start(today - timedelta(days=7)),
                now,
            )

        raise ValueError(f"Unknown window: {name}")

    # ------------------------------------------------------------------
    # Arbitrary expression -> window
    # ------------------------------------------------------------------

    def parse_window(self, expression: str) -> Window | None:
        """
        Window for an arbitrary temporal expression. Falls back to the
        canonical named windows when the free-form resolver fails.
        """
        text = expression.lower().strip()

        aliases = {
            "today": "today",
            "now": "today",
            "tomorrow": "tomorrow",
            "tmr": "tomorrow",
            "tmrw": "tomorrow",
            "yesterday": "yesterday",
            "this week": "this_week",
            "next week": "next_week",
            "last week": "last_week",
            "this month": "this_month",
            "this weekend": "this_weekend",
            "weekend": "this_weekend",
            "upcoming": "upcoming",
            "coming up": "upcoming",
            "soon": "upcoming_week",
        }

        if text in aliases:
            return self.window(aliases[text])

        resolved = self.resolve(text)

        if resolved is None:
            return None

        end = resolved.end

        if end is None:
            end = resolved.start

        return Window(resolved.start, end)

    # ------------------------------------------------------------------
    # Recurring occurrences
    # ------------------------------------------------------------------

    def occurrence(
        self,
        *,
        start: datetime,
        recurrence: str,
        target_date: date,
    ) -> Window | None:
        """
        One concrete occurrence of a recurring event whose series
        anchor is `start`, materialized on `target_date`.

        The anchor hour/minute carries over; the calendar date comes
        from the target day. Returns None when the occurrence would
        start before the series anchor (never in the series' past).
        """
        start = self._ensure_aware(start)

        candidate = datetime.combine(
            target_date,
            start.timetz(),
        )

        if candidate < start:
            return None

        if recurrence == "daily":
            if (candidate.date() - start.date()).days % 1 != 0:
                return None
            return Window(candidate, candidate)

        if recurrence == "weekly":
            if candidate.weekday() != start.weekday():
                return None
            return Window(candidate, candidate)

        return None

    def occurrence_window(
        self,
        *,
        start: datetime,
        end: datetime | None,
        all_day: bool,
        recurrence: str,
        window: Window,
    ) -> list[Window]:
        """
        Every occurrence of a recurring event overlapping `window`.

        All-day recurring series get full-day occurrence windows; the
        anchor day itself is included, matching the "today counts"
        semantics of the resolver.
        """
        start = self._ensure_aware(start)
        anchor_date = start.date()

        # the anchor day bounds the series' past
        if all_day:
            series_start = self._day_start(anchor_date)
            series_end = self._day_end(anchor_date)
        else:
            series_start = start
            series_end = start

        occurrences: list[Window] = []

        if recurrence == "daily":
            step = timedelta(days=1)
        elif recurrence == "weekly":
            step = timedelta(weeks=1)
        elif isinstance(recurrence, str) and recurrence.startswith("interval_") and recurrence.endswith("_days"):
            try:
                step = timedelta(days=int(recurrence[len("interval_"):-len("_days")]))
            except (TypeError, ValueError):
                return []
        else:
            return []

        # walk day-by-day from the window start (bounded scan); the
        # first candidate day is at or after the series anchor
        day = max(window.start.date(), anchor_date)

        window_end_date = window.end.date()

        while day <= window_end_date:
            if all_day:
                occ_start = self._day_start(day)
                occ_end = self._day_end(day)
            else:
                occ = datetime.combine(day, start.timetz())
                occ_start = self._ensure_aware(occ)
                occ_end = occ_start

            if occ_start >= series_start:
                candidate = Window(occ_start, occ_end)

                if self._overlaps(candidate, window):
                    occurrences.append(candidate)

            day += step

        return occurrences

    # ------------------------------------------------------------------
    # Relevancy and labels
    # ------------------------------------------------------------------

    def relevancy(self, moment: datetime) -> float:
        """
        Time distance from now as a ranking weight:

          now          -> 1.0
          tomorrow     -> ~0.9973
          a week away  -> ~0.98
          a year away  -> ~0.75
          yesterday    -> ~0.9973 (past events remain close)

        Magnitude is ~exp(-abs(delta) / 30 days); sign is ignored -
        "just happened" is as relevant as "about to happen".
        """
        moment = self._ensure_aware(moment)
        delta = abs((moment - self.now()).total_seconds())

        return pow(
            0.5,
            delta / (30 * 24 * 3600),
        )

    def relative_label(self, moment: datetime) -> str:
        """
        Human label for an absolute moment relative to now:
        "today", "tomorrow", "yesterday", "in 3 days",
        "2 days ago", "saturday", "next saturday".
        """
        moment = self._ensure_aware(moment)
        now = self.now()

        target = moment.date()
        today = now.date()

        days = (target - today).days

        same_time_of_day = (
            abs((moment - now).total_seconds()) < 60
        )

        if days == 0:
            if same_time_of_day:
                return "now"

            if moment > now:
                return "later today"
            return "earlier today"

        if days == 1:
            return "tomorrow"

        if days == -1:
            return "yesterday"

        if days == 2:
            return "in 2 days"

        if days == -2:
            return "2 days ago"

        weekday_names = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        if 0 < days <= 6:
            return weekday_names[target.weekday()]

        if -6 <= days < 0:
            label = weekday_names[target.weekday()]

            return f"last {label}"

        if days > 6:
            if (target - today).days % 7 == 0 and days <= 35:
                weeks = days // 7

                if weeks == 1:
                    return "next week"

                return f"in {weeks} weeks"

            if target.year != today.year:
                return target.strftime("%b %d, %Y")

            return target.strftime("%b %d")

        weeks = abs(days) // 7

        if abs(days) % 7 == 0 and weeks <= 4:
            return f"{weeks} weeks ago"

        if target.year != today.year:
            return target.strftime("%b %d, %Y")

        return target.strftime("%b %d")

    def describe(
        self,
        *,
        start: datetime,
        end: datetime | None = None,
        all_day: bool = False,
    ) -> str:
        """
        One-line human description of when an event happens.
        """
        start = self._ensure_aware(start)
        label = self.relative_label(start)

        if all_day:
            return label

        def _clock(moment: datetime) -> str:
            if moment.minute == 0:
                return moment.strftime("%I%p").lstrip("0").lower()
            return moment.strftime("%I:%M%p").lstrip("0").lower()

        clock = _clock(start)

        if end is not None:
            end = self._ensure_aware(end)

            if end.date() != start.date():
                end_label = self.relative_label(end)
                return f"{label} {clock} -> {end_label} {_clock(end)}"

            end_clock = _clock(end)

            if end_clock != clock:
                return f"{label} {clock}-{end_clock}"

        return f"{label} {clock}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _overlaps(a: Window, b: Window) -> bool:
        return a.start <= b.end and b.start <= a.end

    def _ensure_aware(self, moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=self.timezone)

        return moment
