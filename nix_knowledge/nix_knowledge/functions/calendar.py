from __future__ import annotations

from datetime import datetime
from typing import Any

from ..context import TemporalContext, Window
from ..engine import KnowledgeEngine
from ..operations import KnowledgeOperation


# Upper bound on expanded occurrences per record so a daily series
# queried over a year cannot flood the result set.
_MAX_OCCURRENCES_PER_RECORD = 60


def _display_moment(moment: datetime) -> str:
    """Render an aware moment with an unambiguous local date and time."""
    date_part = moment.strftime("%A, %B %d, %Y").replace(" 0", " ")
    time_part = moment.strftime("%I:%M:%S %p").lstrip("0")
    return f"{date_part} at {time_part}"


def _parse_moment(value: Any, context: TemporalContext) -> datetime | None:
    """
    Parse an ISO-8601 moment (or a natural-language expression)
    into an aware datetime in the context timezone.
    """
    if not value:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        resolved = context.resolve(value)

        if resolved is None:
            return None

        moment = resolved.start

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=context.timezone)

    return moment


def create_calendar_event(
    engine: KnowledgeEngine,
    title: str,
    start: str,
    end: str | None,
    all_day: bool,
    temporal_expression: str,
    confidence: float = 1.0,
    source: str = "knowledge_needle",
    reason: str | None = None,
    recurring: bool = False,
    recurrence: str | None = None,
):
    data = {
        "title": title,
        "start": start,
        "end": end,
        "all_day": all_day,
        "temporal_expression": temporal_expression,
        "recurring": recurring,
        "recurrence": recurrence,
        "action_policy": "once",
    }

    record = engine.create(
        "calendar_event",
        data,
        confidence=confidence,
        source=source,
        reason=reason,
    )

    return {
        "ok": True,
        "record_id": record.id,
        "data": record.data,
    }


def _entry(
    record,
    data: dict[str, Any],
    *,
    start: datetime,
    end: datetime,
    all_day: bool,
    context: TemporalContext,
    occurrence: bool,
) -> dict[str, Any]:
    relative = context.describe(
        start=start,
        end=end if end > start else None,
        all_day=all_day,
    )

    # Keep both the user's relative wording and absolute local timestamps.
    # Core/Casper must use these fields instead of guessing what "tmr" or
    # "tomorrow" means from its own clock.
    return {
        "record_id": record.id,
        "title": data.get("title"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "start_date": start.date().isoformat(),
        "start_time": start.strftime("%H:%M:%S"),
        "end_date": end.date().isoformat(),
        "end_time": end.strftime("%H:%M:%S"),
        "start_local": _display_moment(start),
        "end_local": _display_moment(end),
        "timezone": str(context.timezone),
        "all_day": all_day,
        "status": data.get("status", "scheduled"),
        "temporal_expression": data.get("temporal_expression"),
        "recurring": bool(data.get("recurring")),
        "recurrence": data.get("recurrence"),
        "occurrence": occurrence,
        "when": relative,
        "temporal_grounding": {
            "original_expression": data.get("temporal_expression"),
            "relative_label": relative,
            "timezone": str(context.timezone),
            "start": {
                "iso": start.isoformat(),
                "date": start.date().isoformat(),
                "time": start.strftime("%H:%M:%S"),
                "local": _display_moment(start),
            },
            "end": {
                "iso": end.isoformat(),
                "date": end.date().isoformat(),
                "time": end.strftime("%H:%M:%S"),
                "local": _display_moment(end),
            },
        },
    }


def _entries_for_record(
    record,
    data: dict[str, Any],
    query: Window | None,
    context: TemporalContext,
) -> list[dict[str, Any]]:
    """
    Result entries for one record against the query window.

    Non-recurring events match by overlap; recurring events are
    expanded into concrete occurrences inside the window. With no
    query window, the series/event anchor is returned as-is.
    """
    start = _parse_moment(data.get("start"), context)
    end = _parse_moment(data.get("end"), context)

    all_day = bool(data.get("all_day", False))
    recurring = bool(data.get("recurring"))
    recurrence = data.get("recurrence")

    if start is None:
        return []

    if end is None:
        end = start

    if recurring and (
        recurrence in {"daily", "weekly"}
        or (isinstance(recurrence, str) and recurrence.startswith("interval_") and recurrence.endswith("_days"))
    ) and query is not None:
        occurrences = context.occurrence_window(
            start=start,
            end=end,
            all_day=all_day,
            recurrence=recurrence,
            window=query,
        )

        return [
            _entry(
                record,
                data,
                start=occurrence.start,
                end=occurrence.end,
                all_day=all_day,
                context=context,
                occurrence=True,
            )
            for occurrence in occurrences[
                :_MAX_OCCURRENCES_PER_RECORD
            ]
        ]

    if query is not None:
        footprint = Window(start, end)

        if not context._overlaps(footprint, query):
            return []

    return [
        _entry(
            record,
            data,
            start=start,
            end=end,
            all_day=all_day,
            context=context,
            occurrence=False,
        )
    ]


def find_calendar_events(
    engine: KnowledgeEngine,
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
    window: str | None = None,
    include_cancelled: bool | None = None,
    context: TemporalContext | None = None,
):
    """
    Find calendar events with full time sense.

    Matching is overlap-based against the query window, which can be
    a canonical name ("today", "next_week", "upcoming") or any
    natural-language expression ("this weekend", "on friday",
    "september 12th"). Recurring events are expanded into concrete
    occurrences inside the window. Results are ranked by temporal
    relevance (closeness to now, past and future alike).

    Cancelled events are excluded from window browsing but included
    in title lookups, where their status is itself the answer.

    This is a read-only operation. It never modifies the database.
    """
    context = context or TemporalContext()

    title_key = title.lower().strip() if title else None

    # A title lookup includes cancelled records: "when is my tsa
    # meeting" deserves "it was cancelled", not "not found".
    if include_cancelled is None:
        include_cancelled = bool(title_key)

    query: Window | None = None
    window_info: dict[str, Any] | None = None

    if window:
        query = context.parse_window(window)

        if query is None:
            return {
                "ok": False,
                "error": (
                    "Could not interpret the time window: "
                    f"'{window}'."
                ),
            }

        window_info = {
            "expression": window,
            "start": query.start.isoformat(),
            "end": query.end.isoformat(),
        }

    if start or end:
        start_moment = _parse_moment(start, context)
        end_moment = _parse_moment(end, context)

        if start and start_moment is None:
            return {
                "ok": False,
                "error": f"Could not interpret start boundary: '{start}'.",
            }

        if end and end_moment is None:
            return {
                "ok": False,
                "error": f"Could not interpret end boundary: '{end}'.",
            }

        lo = start_moment or datetime.min.replace(
            tzinfo=context.timezone
        )
        hi = end_moment or datetime.max.replace(
            tzinfo=context.timezone
        )

        query = Window(lo, hi)

        if window_info is None:
            window_info = {
                "expression": None,
                "start": lo.isoformat(),
                "end": hi.isoformat(),
            }

    records = engine.search("calendar_event")

    exact: list[dict] = []
    substring: list[dict] = []

    for record in records:
        data = record.data

        status = data.get("status", "scheduled")

        if status == "cancelled" and not include_cancelled:
            continue

        tier = None

        if title_key:
            event_title = str(data.get("title", ""))

            if title_key == event_title.lower().strip():
                tier = exact
            elif title_key in event_title.lower():
                tier = substring
            else:
                continue

        entries = _entries_for_record(
            record,
            data,
            query,
            context,
        )

        if tier is exact:
            exact.extend(entries)
        elif tier is substring:
            substring.extend(entries)
        else:
            # no title filter: single result list
            exact.extend(entries)

    # When the user gave a title and exact matches exist, the exact
    # matches are the answer - keeping substring matches too would make
    # "cancel robotics" ambiguous against "robotics meetup".
    if title_key and exact:
        results = exact
    else:
        results = exact + substring

    results.sort(
        key=lambda entry: -context.relevancy(
            datetime.fromisoformat(entry["start"])
        )
    )

    response: dict[str, Any] = {
        "ok": True,
        "count": len(results),
        "events": results,
    }

    if window_info is not None:
        response["window"] = window_info

    return response


def update_calendar_event(
    engine: KnowledgeEngine,
    record_id: int,
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
    all_day: bool | None = None,
    temporal_expression: str | None = None,
    status: str | None = None,
    reason: str | None = None,
):
    """
    Update an existing calendar event.

    Only supplied fields are changed.
    """

    changes: dict[str, Any] = {}

    if title is not None:
        changes["title"] = title

    if start is not None:
        changes["start"] = start

    if end is not None:
        changes["end"] = end

    if all_day is not None:
        changes["all_day"] = all_day

    if temporal_expression is not None:
        changes["temporal_expression"] = temporal_expression

    if status is not None:
        changes["status"] = status

    if not changes:
        return {
            "ok": False,
            "error": "No changes supplied.",
        }

    record = engine.apply(
        KnowledgeOperation(
            operation="update",
            knowledge_type="calendar_event",
            match={"id": record_id},
            changes=changes,
            source="knowledge_needle",
            reason=reason or "Calendar event updated from natural-language request.",
        )
    )

    return {
        "ok": True,
        "record_id": record.id,
        "data": record.data,
    }


def cancel_calendar_event(
    engine: KnowledgeEngine,
    record_id: int,
    reason: str | None = None,
):
    """
    Cancel an existing calendar event.

    Cancellation is represented as a status change rather than deletion,
    preserving the event's history.
    """

    return update_calendar_event(
        engine,
        record_id=record_id,
        status="cancelled",
        reason=reason or "Calendar event cancelled from natural-language request.",
    )
