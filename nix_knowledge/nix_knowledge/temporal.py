from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class TemporalResult:
    expression: str
    start: datetime
    end: datetime | None
    all_day: bool
    recurring: bool = False
    recurrence: str | None = None
    confidence: float = 1.0


class TemporalResolver:
    """
    Converts natural-language temporal expressions into
    timezone-aware absolute datetimes.

    Week-relative semantics (Monday-based weeks):

      "thursday"             -> next occurrence of Thursday (today counts)
      "this week thursday"   -> Thursday of the current week, literally
      "next week thursday"   -> Thursday of the following week
      "next thursday"        -> Thursday of the following week
      "thursday next week"   -> Thursday of the following week

    Calendar dates:

      "september 12th"       -> this year (rolls to next year once past)
      "september 12 2027"    -> that exact date
      "12 september 2027"    -> that exact date
      "9/12" / "9/12/2027"   -> US month/day
      "2027-09-12"           -> ISO date

    All of the above accept an optional "at <time>" suffix and an
    optional leading "on".

    Time ranges:

      "from 7pm to 8pm"      -> today 19:00-20:00 (rolls to tomorrow
                                once the whole window has passed)
      "today from 7pm to 8pm" -> that day's window

    Recurring expressions without a clock time ("every monday",
    "every day") are all-day events; today counts as the first
    occurrence.

    This component only interprets time.
    It never modifies the Knowledge Base.
    """

    WEEKDAYS = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    MONTHS = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sept": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    def __init__(self, timezone: str = "America/Chicago"):
        self.timezone = ZoneInfo(timezone)

        self._weekday_alt = (
            "monday|tuesday|wednesday|thursday|friday|"
            "saturday|sunday"
        )

        self._month_alt = "|".join(
            sorted(self.MONTHS, key=len, reverse=True)
        )

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def _day_bounds(self, d: date):
        start = datetime.combine(
            d,
            time.min,
            tzinfo=self.timezone,
        )

        end = datetime.combine(
            d,
            time.max,
            tzinfo=self.timezone,
        )

        return start, end

    def _parse_clock(self, value: str) -> time | None:
        value = value.strip().lower().replace(".", "")

        if value == "noon":
            return time(12, 0)

        if value == "midnight":
            return time(0, 0)

        match = re.fullmatch(
            r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            value,
        )

        if not match:
            return None

        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)

        if minute > 59:
            return None

        if meridiem:
            if hour < 1 or hour > 12:
                return None

            if meridiem == "am":
                if hour == 12:
                    hour = 0
            else:
                if hour != 12:
                    hour += 12
        elif hour > 23:
            return None

        return time(hour, minute)

    # ------------------------------------------------------------------
    # Weekday targeting
    # ------------------------------------------------------------------

    def _weekday_target(
        self,
        qualifier: str | None,
        weekday: int,
        now: datetime,
    ) -> date:
        """
        Resolve a weekday to a calendar date.

          None / "on"   next occurrence, today counts
          "this"        this week's weekday, literally (Monday-based)
          "next"        following week's weekday
        """
        if qualifier in {"this", "this week"}:
            monday = now.date() - timedelta(
                days=now.weekday()
            )
            target = monday + timedelta(days=weekday)
            # Bare "this Friday" means the upcoming Friday when the
            # current week's Friday has already passed; explicit "this
            # week Friday" retains its documented literal semantics.
            if qualifier == "this" and target < now.date():
                target += timedelta(days=7)
            return target

        if qualifier in {"next", "next week"}:
            next_monday = now.date() + timedelta(
                days=7 - now.weekday()
            )
            return next_monday + timedelta(days=weekday)

        days_ahead = (weekday - now.weekday()) % 7

        return now.date() + timedelta(days=days_ahead)

    # ------------------------------------------------------------------
    # Calendar date resolution
    # ------------------------------------------------------------------

    def _month_day_result(
        self,
        *,
        month: int,
        day: int,
        year: int | None,
        clock: time | None,
        now: datetime,
        original: str,
    ) -> TemporalResult | None:
        """
        Build a result for an explicit calendar date.

        When no year is given the current year is used and the date
        rolls to next year once it has passed (so "january 5th" said
        in September means next January).
        """
        if year is None:
            year = now.year

            try:
                target = date(year, month, day)
            except ValueError:
                return None

            if target < now.date():
                year += 1

        try:
            target = date(year, month, day)
        except ValueError:
            return None

        if clock is None:
            start, end = self._day_bounds(target)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        start = datetime.combine(
            target,
            clock,
            tzinfo=self.timezone,
        )

        return TemporalResult(
            expression=original,
            start=start,
            end=start,
            all_day=False,
        )

    # ------------------------------------------------------------------
    # Main resolver
    # ------------------------------------------------------------------

    def resolve(
        self,
        expression: str,
        now: datetime | None = None,
    ) -> TemporalResult | None:
        original = expression.strip()
        text = original.lower().strip()

        if not text:
            return None

        now = now or self.now()

        # Normalize natural day-part wrappers before parsing. These are
        # common in speech and must not be left in the event title:
        # "during the morning time" -> "morning".
        text = re.sub(
            r"\b(?:during|in)\s+(?:the\s+)?"
            r"(morning|afternoon|evening|tonight)"
            r"(?:\s+time)?\b",
            r"\1",
            text,
        )
        text = re.sub(r"\b(morning|afternoon|evening|tonight)\s+time\b", r"\1", text)
        # Speech commonly inserts "more" in relative offsets and uses
        # "around" before a clock. They do not change the intended date or
        # time, so normalize them before the strict symbolic grammar runs.
        text = re.sub(
            r"\bin\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"more\s+(minutes?|hours?|days?|weeks?)\b",
            r"in \1 \2",
            text,
        )
        text = re.sub(r"\baround\s+(?=\d|noon|midnight)", "at ", text)
        # Normalize spoken day offsets before the strict grammar and before
        # candidate-span extraction, so a title cannot swallow "day after".
        text = re.sub(
            r"\b(?:the\s+)?day\s+after\s+tomorrow\b",
            "in 2 days",
            text,
        )
        # An explicit clock range makes a preceding day-part redundant:
        # "next Monday morning from 9am to 11am" is the same concrete
        # window as "next Monday from 9am to 11am".
        text = re.sub(
            rf"((?:today|tomorrow|tmr|yesterday|next\s+(?:{self._weekday_alt})|"
            rf"(?:this|next|on)\s+(?:{self._weekday_alt}))\s+)"
            r"(?:morning|afternoon|evening|tonight)\s+(?=(?:from|at)\b)",
            r"\1",
            text,
        )

        wd = self._weekday_alt
        months = self._month_alt

        # --------------------------------------------------------
        # Compound relative dates and day parts
        #
        # Voice requests often combine an offset, a reference day, a
        # day-part, and an explicit range:
        #   "5 days after today during the morning time"
        #   "in five days from 9am to 11am"
        #   "5 days after tomorrow morning"
        # The whole expression is resolved here before the generic
        # suffix/range parsers, so Core never stores the offset as part
        # of the event title or silently falls back to today's date.
        # --------------------------------------------------------

        amount_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        amount_pattern = (
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        )
        compound = re.fullmatch(
            rf"(?:(?P<amount>{amount_pattern})\s+"
            rf"(?P<unit>days?|weeks?)\s+"
            rf"(?P<relation>after|before|from)\s+"
            rf"(?P<base>today|tomorrow|tmr|"
            rf"yesterday|now)|"
            rf"in\s+(?P<in_amount>{amount_pattern})\s+"
            rf"(?P<in_unit>days?|weeks?))"
            rf"(?:\s+(?P<tail>.*))?",
            text,
        )

        if compound:
            raw_amount = compound.group("amount") or compound.group("in_amount")
            amount = (
                int(raw_amount)
                if raw_amount.isdigit()
                else amount_words[raw_amount]
            )
            if (
                (compound.group("unit") or "").startswith("week")
                or (compound.group("in_unit") or "").startswith("week")
            ):
                amount *= 7
            base = compound.group("base") or "today"
            base_offsets = {
                "today": 0,
                "tomorrow": 1,
                "tmr": 1,
                "yesterday": -1,
                "now": 0,
            }
            relation = compound.group("relation") or "from"
            signed_amount = -amount if relation == "before" else amount
            target = now.date() + timedelta(
                days=base_offsets[base] + signed_amount
            )
            tail = (compound.group("tail") or "").strip()

            period_hours = {
                "morning": (6, 12),
                "afternoon": (12, 17),
                "evening": (17, 22),
                "tonight": (18, 23),
            }
            period = tail.strip()
            start_clock = end_clock = None

            # An explicit clock range is more precise than a broad
            # day-part qualifier: "morning from 9am to 11am" means
            # 09:00-11:00, while plain "morning" means the full
            # morning window.
            range_match = re.fullmatch(
                r"(?:(?:morning|afternoon|evening|tonight)\s+)?"
                r"(?:from\s+)?(.+?)\s+(?:to|until|till)\s+(.+)",
                period,
            )
            if range_match:
                start_clock = self._parse_clock(range_match.group(1))
                end_clock = self._parse_clock(range_match.group(2))
                if start_clock is None or end_clock is None:
                    return None
            elif period.startswith("at "):
                start_clock = self._parse_clock(period[3:])
                if start_clock is None:
                    return None
                end_clock = start_clock
            elif period in period_hours:
                start_hour, end_hour = period_hours[period]
                start_clock = time(start_hour, 0)
                end_clock = time(end_hour, 0)
            elif period:
                # A compound expression with an unrecognized tail is
                # unsafe to execute; do not guess or discard it.
                return None

            if start_clock is None:
                start, end = self._day_bounds(target)
                return TemporalResult(
                    expression=original,
                    start=start,
                    end=end,
                    all_day=True,
                )

            start = datetime.combine(target, start_clock, tzinfo=self.timezone)
            end = datetime.combine(target, end_clock, tzinfo=self.timezone)
            if end_clock == start_clock and not range_match:
                end = start
            elif end <= start:
                end += timedelta(days=1)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=False,
            )

        # --------------------------------------------------------
        # Relative duration
        # --------------------------------------------------------

        match = re.fullmatch(
            r"in\s+(?P<amount>\d+|a\s+couple(?:\s+of)?|couple(?:\s+of)?|"
            r"a\s+few|few|several)\s+"
            r"(?P<unit>minute|minutes|min|mins|"
            r"hour|hours|hr|hrs|"
            r"day|days|"
            r"week|weeks)",
            text,
        )

        if match:
            raw_amount = re.sub(
                r"\s+",
                " ",
                match.group("amount"),
            )

            if raw_amount in {"couple", "a couple", "couple of", "a couple of"}:
                amount = 2
            elif raw_amount in {"few", "a few"}:
                amount = 3
            elif raw_amount == "several":
                amount = 4
            else:
                amount = int(raw_amount)

            unit = match.group("unit")

            if unit.startswith("min"):
                delta = timedelta(minutes=amount)
            elif unit.startswith("hour") or unit in {"hr", "hrs"}:
                delta = timedelta(hours=amount)
            elif unit.startswith("day"):
                delta = timedelta(days=amount)
            else:
                delta = timedelta(weeks=amount)

            start = now + delta

            return TemporalResult(
                expression=original,
                start=start,
                end=start,
                all_day=False,
            )

        # --------------------------------------------------------
        # Day + time range
        #
        # today from 7pm to 8pm
        # tomorrow from 9am until 5pm
        # friday from 7pm to 9pm
        # next monday from 10am to 11:30am
        # --------------------------------------------------------

        match = re.fullmatch(
            rf"(today|tomorrow|tmr|yesterday|"
            rf"next\s+(?:{wd})|"
            rf"(?:(on|this(?:\s+week)?|next(?:\s+week)?)\s+)?(?:{wd}))\s+"
            r"(?:from\s+)?(.+?)\s+(?:to|until|till)\s+(.+)",
            text,
        )

        if match:
            day_expression = match.group(1)
            qualifier = match.group(2)

            day_result = self.resolve(
                day_expression,
                now,
            )

            if day_result is None:
                return None

            start_clock = self._parse_clock(match.group(3))
            end_clock = self._parse_clock(match.group(4))

            if start_clock is None or end_clock is None:
                return None

            start = day_result.start.replace(
                hour=start_clock.hour,
                minute=start_clock.minute,
                second=0,
                microsecond=0,
            )

            end = day_result.start.replace(
                hour=end_clock.hour,
                minute=end_clock.minute,
                second=0,
                microsecond=0,
            )

            if end <= start:
                end += timedelta(days=1)

            # bare weekday whose window already passed today rolls
            # to next week, matching the single-time semantics
            if (
                qualifier in (None, "on")
                and re.fullmatch(
                    rf"(?:on\s+)?(?:{wd})",
                    day_expression,
                )
                and start <= now
            ):
                start += timedelta(days=7)
                end += timedelta(days=7)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=False,
            )

        # --------------------------------------------------------
        # Time ranges
        #
        # from 7pm to 8pm
        # 7pm to 8:30pm
        # from 7pm until 9pm
        #
        # A range without a day applies to today; if the whole
        # window has already passed, it rolls to tomorrow.
        # --------------------------------------------------------

        match = re.fullmatch(
            r"(?:from\s+)?(.+?)\s+(?:to|until|till)\s+(.+)",
            text,
        )

        if match:
            start_clock = self._parse_clock(match.group(1))
            end_clock = self._parse_clock(match.group(2))

            if start_clock is not None and end_clock is not None:
                start = now.replace(
                    hour=start_clock.hour,
                    minute=start_clock.minute,
                    second=0,
                    microsecond=0,
                )

                end = now.replace(
                    hour=end_clock.hour,
                    minute=end_clock.minute,
                    second=0,
                    microsecond=0,
                )

                if end <= start:
                    end += timedelta(days=1)

                if end <= now:
                    start += timedelta(days=1)
                    end += timedelta(days=1)

                return TemporalResult(
                    expression=original,
                    start=start,
                    end=end,
                    all_day=False,
                )

        # --------------------------------------------------------
        # Month-name dates
        #
        # september 12th
        # september 12 2027
        # september 12th at 3pm
        # on september 12th
        # 12 september 2027
        # the 12th of september
        # --------------------------------------------------------

        match = re.fullmatch(
            rf"(?:on\s+)?(?:the\s+)?"
            rf"(?:({months})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?"
            rf"(?:\s*,?\s*(\d{{4}}))?"
            rf"|(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({months})\.?"
            rf"(?:\s*,?\s*(\d{{4}}))?)"
            rf"(?:\s+at\s+(.+))?",
            text,
        )

        if match:
            if match.group(1):
                month = self.MONTHS[match.group(1)]
                day = int(match.group(2))
                year = (
                    int(match.group(3))
                    if match.group(3)
                    else None
                )
            else:
                month = self.MONTHS[match.group(5)]
                day = int(match.group(4))
                year = (
                    int(match.group(6))
                    if match.group(6)
                    else None
                )

            clock = (
                self._parse_clock(match.group(7))
                if match.group(7)
                else None
            )

            if match.group(7) and clock is None:
                return None

            return self._month_day_result(
                month=month,
                day=day,
                year=year,
                clock=clock,
                now=now,
                original=original,
            )

        # --------------------------------------------------------
        # Numeric dates
        #
        # 9/12
        # 9/12/2027
        # on 9/12 at 2pm
        # --------------------------------------------------------

        match = re.fullmatch(
            r"(?:on\s+)?"
            r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?"
            r"(?:\s+at\s+(.+))?",
            text,
        )

        if match:
            month = int(match.group(1))
            day = int(match.group(2))

            year = (
                int(match.group(3))
                if match.group(3)
                else None
            )

            clock = (
                self._parse_clock(match.group(4))
                if match.group(4)
                else None
            )

            if match.group(4) and clock is None:
                return None

            if not 1 <= month <= 12:
                return None

            return self._month_day_result(
                month=month,
                day=day,
                year=year,
                clock=clock,
                now=now,
                original=original,
            )

        # --------------------------------------------------------
        # ISO dates
        #
        # 2027-09-12
        # 2027-09-12 at 3pm
        # --------------------------------------------------------

        match = re.fullmatch(
            r"(?:on\s+)?"
            r"(\d{4})-(\d{1,2})-(\d{1,2})"
            r"(?:\s+at\s+(.+))?",
            text,
        )

        if match:
            clock = (
                self._parse_clock(match.group(4))
                if match.group(4)
                else None
            )

            if match.group(4) and clock is None:
                return None

            return self._month_day_result(
                month=int(match.group(2)),
                day=int(match.group(3)),
                year=int(match.group(1)),
                clock=clock,
                now=now,
                original=original,
            )

        # --------------------------------------------------------
        # Month-only dates
        #
        # next june / last december / in may / this september
        # september (upcoming occurrence)
        #
        # A whole-month window (first day through last day).
        # Bare "may" is not accepted alone: it is a modal verb.
        # --------------------------------------------------------

        match = re.fullmatch(
            rf"(?:(in|this|next|last)\s+)?({months})\.?",
            text,
        )

        if match:
            qualifier = match.group(1)
            month_name = match.group(2)
            month = self.MONTHS[month_name]

            if qualifier is None and month_name == "may":
                return None

            if qualifier == "this":
                year = now.year
            elif qualifier == "last":
                year = now.year if month < now.month else now.year - 1
            elif qualifier in {"next", "in"}:
                year = (
                    now.year
                    if month > now.month
                    else now.year + 1
                )
            else:
                year = (
                    now.year
                    if month >= now.month
                    else now.year + 1
                )

            try:
                first = date(year, month, 1)

                if month == 12:
                    after = date(year + 1, 1, 1)
                else:
                    after = date(year, month + 1, 1)
            except ValueError:
                return None

            last_day = after - timedelta(days=1)

            start, _ = self._day_bounds(first)
            _, end = self._day_bounds(last_day)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        # --------------------------------------------------------
        # Today / tomorrow / yesterday
        # --------------------------------------------------------

        day_offsets = {
            "today": 0,
            "tomorrow": 1,
            "tmr": 1,
            "yesterday": -1,
        }

        if text in day_offsets:
            target = now.date() + timedelta(
                days=day_offsets[text]
            )

            start, end = self._day_bounds(target)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        # --------------------------------------------------------
        # Day + explicit time
        #
        # tomorrow at 6pm
        # tmr at 18:00
        # next monday at 7:30am
        # --------------------------------------------------------

        match = re.fullmatch(
            rf"(today|tomorrow|tmr|yesterday|"
            rf"next\s+(?:{wd})|"
            rf"(?:on\s+)?(?:{wd}))\s+at\s+(.+)",
            text,
        )

        if match:
            day_expression = match.group(1)
            clock_expression = match.group(2)

            day_result = self.resolve(
                day_expression,
                now,
            )

            if day_result is None:
                return None

            clock = self._parse_clock(clock_expression)

            if clock is None:
                return None

            start = day_result.start.replace(
                hour=clock.hour,
                minute=clock.minute,
                second=0,
                microsecond=0,
            )

            return TemporalResult(
                expression=original,
                start=start,
                end=start,
                all_day=False,
            )

        # --------------------------------------------------------
        # Weekday + explicit time (with week qualifiers)
        #
        # friday at 6pm
        # on friday at 6pm
        # next week friday at 6pm
        # this week friday at 6pm
        # --------------------------------------------------------

        match = re.fullmatch(
            rf"(?:(on|this(?:\s+week)?|next(?:\s+week)?)\s+)?"
            rf"({wd})\s+at\s+(.+)",
            text,
        )

        if match:
            qualifier = match.group(1)
            weekday = self.WEEKDAYS[match.group(2)]

            clock = self._parse_clock(match.group(3))

            if clock is None:
                return None

            target = self._weekday_target(
                qualifier if qualifier != "on" else None,
                weekday,
                now,
            )

            start = datetime.combine(
                target,
                clock,
                tzinfo=self.timezone,
            )

            # if the chosen time already passed this week and the
            # qualifier was implicit, roll to the next occurrence
            if (
                qualifier in (None, "on")
                and start <= now
            ):
                start += timedelta(days=7)

            return TemporalResult(
                expression=original,
                start=start,
                end=start,
                all_day=False,
            )

        # --------------------------------------------------------
        # Parts of the day
        # --------------------------------------------------------

        periods = {
            "morning": (6, 12),
            "this morning": (6, 12),
            "afternoon": (12, 17),
            "this afternoon": (12, 17),
            "evening": (17, 22),
            "this evening": (17, 22),
            "tonight": (18, 23),
        }

        if text in periods:
            start_hour, end_hour = periods[text]

            start = now.replace(
                hour=start_hour,
                minute=0,
                second=0,
                microsecond=0,
            )

            end = now.replace(
                hour=end_hour,
                minute=59,
                second=59,
                microsecond=999999,
            )

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=False,
            )

        # --------------------------------------------------------
        # tomorrow morning, tomorrow evening, etc.
        # --------------------------------------------------------

        match = re.fullmatch(
            r"(today|tomorrow|tmr|yesterday)\s+"
            r"(morning|afternoon|evening|tonight)",
            text,
        )

        if match:
            day_expression = match.group(1)
            period = match.group(2)

            day_result = self.resolve(
                day_expression,
                now,
            )

            if day_result is None:
                return None

            period_hours = {
                "morning": (6, 12),
                "afternoon": (12, 17),
                "evening": (17, 22),
                "tonight": (18, 23),
            }

            start_hour, end_hour = period_hours[period]

            start = day_result.start.replace(
                hour=start_hour,
                minute=0,
                second=0,
                microsecond=0,
            )

            end = day_result.start.replace(
                hour=end_hour,
                minute=59,
                second=59,
                microsecond=999999,
            )

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=False,
            )

        # --------------------------------------------------------
        # Weekdays
        #
        # thursday / on thursday        -> next occurrence (today counts)
        # this week thursday            -> literal this week
        # next thursday                 -> following week
        # next week thursday            -> following week
        # thursday next week            -> following week
        # --------------------------------------------------------

        match = re.fullmatch(
            rf"(?:(on|this(?:\s+week)?|next(?:\s+week)?)\s+)?({wd})",
            text,
        )

        if match:
            qualifier = match.group(1)
            weekday = self.WEEKDAYS[match.group(2)]

            target = self._weekday_target(
                qualifier if qualifier != "on" else None,
                weekday,
                now,
            )

            start, end = self._day_bounds(target)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        match = re.fullmatch(
            rf"({wd})\s+(this|next)\s+week",
            text,
        )

        if match:
            weekday = self.WEEKDAYS[match.group(1)]

            target = self._weekday_target(
                match.group(2),
                weekday,
                now,
            )

            start, end = self._day_bounds(target)

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        # --------------------------------------------------------
        # Next week
        # --------------------------------------------------------

        if text == "next week":
            days_until_monday = (
                7 - now.weekday()
            )

            monday = now.date() + timedelta(
                days=days_until_monday
            )

            sunday = monday + timedelta(days=6)

            start = datetime.combine(
                monday,
                time.min,
                tzinfo=self.timezone,
            )

            end = datetime.combine(
                sunday,
                time.max,
                tzinfo=self.timezone,
            )

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        # --------------------------------------------------------
        # This week / next week (Monday-based, matching
        # TemporalContext.week_start)
        # --------------------------------------------------------

        if text in ("this week", "next week"):
            if text == "this week":
                monday = now.date() - timedelta(days=now.weekday())
            else:
                monday = (
                    now.date()
                    - timedelta(days=now.weekday())
                    + timedelta(weeks=1)
                )

            sunday = monday + timedelta(days=6)

            start = datetime.combine(
                monday,
                time.min,
                tzinfo=self.timezone,
            )

            end = datetime.combine(
                sunday,
                time.max,
                tzinfo=self.timezone,
            )

            # A forward range said while it is already running ("this
            # week" on a Thursday) must not begin in the past: the
            # remaining window starts NOW. Otherwise reminders are
            # scheduled for days that already went by.
            if text == "this week" and start < now:
                start = now

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=True,
            )

        # --------------------------------------------------------
        # This weekend
        # --------------------------------------------------------

        weekend_match = re.fullmatch(
            r"this\s+weekend(?:\s+at\s+(.+))?",
            text,
        )
        if weekend_match:
            weekend_clock = (
                self._parse_clock(weekend_match.group(1))
                if weekend_match.group(1)
                else None
            )
            if weekend_match.group(1) and weekend_clock is None:
                return None
            weekday = now.weekday()

            if weekday in (5, 6):
                # said during the weekend: the current one
                saturday = now.date() - timedelta(
                    days=weekday - 5
                )
            else:
                saturday = now.date() + timedelta(
                    days=(5 - weekday) % 7
                )

            sunday = saturday + timedelta(days=1)

            if weekend_clock is not None:
                # A timed phrase names one occurrence, not a two-day range:
                # choose the next Saturday/Sunday occurrence in this weekend,
                # or the next weekend if both have already passed.
                saturday_moment = datetime.combine(
                    saturday, weekend_clock, tzinfo=self.timezone
                )
                sunday_moment = datetime.combine(
                    sunday, weekend_clock, tzinfo=self.timezone
                )
                if saturday_moment >= now:
                    start = saturday_moment
                elif sunday_moment >= now:
                    start = sunday_moment
                else:
                    start = saturday_moment + timedelta(days=7)
                end = start
                all_day = False
            else:
                start = datetime.combine(
                    saturday, time.min, tzinfo=self.timezone
                )
                end = datetime.combine(
                    sunday, time.max, tzinfo=self.timezone
                )
                # "this weekend" said on a Sunday: the window begins now.
                if start < now:
                    start = now
                all_day = True

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=all_day,
            )

        # --------------------------------------------------------
        # Recurring: every day / daily, every <weekday>
        #
        # every day at 7am
        # every sunday at 7am
        # every sunday
        # --------------------------------------------------------

        interval_match = re.fullmatch(
            r"every\s+(\d+)\s+days(?:\s+at\s+(.+))?",
            text,
        )
        if interval_match:
            interval = int(interval_match.group(1))
            if interval < 1 or interval > 365:
                return None
            clock_expression = interval_match.group(2)
            clock = self._parse_clock(clock_expression) if clock_expression else None
            if clock_expression and clock is None:
                return None
            all_day = clock is None
            days_ahead = 0
            if clock is not None:
                today_at_clock = now.replace(
                    hour=clock.hour, minute=clock.minute, second=0, microsecond=0
                )
                if today_at_clock <= now:
                    days_ahead = interval
            candidate_date = now.date() + timedelta(days=days_ahead)
            if clock is None:
                start, end = self._day_bounds(candidate_date)
            else:
                start = datetime.combine(candidate_date, clock, tzinfo=self.timezone)
                end = start
            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=all_day,
                recurring=True,
                recurrence=f"interval_{interval}_days",
            )

        match = re.fullmatch(
            rf"every\s+"
            rf"(day|daily|{wd})"
            rf"(?:\s+at\s+(.+))?",
            text,
        )

        if match:
            target = match.group(1)
            clock_expression = match.group(2)

            clock = (
                self._parse_clock(clock_expression)
                if clock_expression
                else None
            )

            if clock_expression and clock is None:
                return None

            # "every monday" with no clock time is an all-day
            # recurring event; today counts as the first occurrence
            all_day = clock is None

            if target in {"day", "daily"}:
                recurrence = "daily"
                days_ahead = 0

                if clock is not None:
                    today_at_clock = now.replace(
                        hour=clock.hour,
                        minute=clock.minute,
                        second=0,
                        microsecond=0,
                    )

                    if today_at_clock <= now:
                        days_ahead = 1
            else:
                recurrence = "weekly"
                weekday = self.WEEKDAYS[target]
                days_ahead = (
                    weekday - now.weekday()
                ) % 7

                # with a clock time that already passed today,
                # the first occurrence rolls to next week
                if (
                    days_ahead == 0
                    and clock is not None
                    and now.hour >= clock.hour
                ):
                    days_ahead = 7

            candidate_date = now.date() + timedelta(
                days=days_ahead
            )

            if clock is None:
                start, end = self._day_bounds(
                    candidate_date
                )
            else:
                start = datetime.combine(
                    candidate_date,
                    clock,
                    tzinfo=self.timezone,
                )

                end = start

            return TemporalResult(
                expression=original,
                start=start,
                end=end,
                all_day=all_day,
                recurring=True,
                recurrence=recurrence,
            )

        # --------------------------------------------------------
        # Explicit clock time
        # --------------------------------------------------------

        match = re.fullmatch(
            r"at\s+(.+)",
            text,
        )

        if match:
            clock = self._parse_clock(
                match.group(1)
            )

            if clock is None:
                return None

            start = now.replace(
                hour=clock.hour,
                minute=clock.minute,
                second=0,
                microsecond=0,
            )

            if start <= now:
                start += timedelta(days=1)

            return TemporalResult(
                expression=original,
                start=start,
                end=start,
                all_day=False,
            )

        return None
