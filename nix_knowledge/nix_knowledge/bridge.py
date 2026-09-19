from __future__ import annotations

from datetime import datetime
from typing import Any

from nix_actions.engine import ActionsEngine


ACTION_TYPE_BY_INTENT = {
    # calendar wake-up style intents -> physical alarm
    "alarm": "alarm",
    "reminder": "reminder",
}


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def schedule_event_actions(
    actions_engine: ActionsEngine,
    *,
    record_id: int | None,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Translate a stored calendar_event into deterministic actions.

    Contract between Knowledge and Actions:

      Knowledge event data
      --------------------
      title               - event label
      start / end         - ISO-8601 local datetimes
      all_day             - true for date-level events
      recurring           - true when the event repeats
      recurrence          - "daily" | "weekly" | null
      action_policy       - "once" | "every_occurrence"
      action_type         - "alarm" | "reminder" | "notify" | null
      action_message      - override payload message (optional)

    Actions side effects
    --------------------
      non-recurring       - one pending action at the event start
      recurring           - one pending action at the next occurrence,
                            chained forward by the Actions engine
                            after each fire

    The engine stores what was scheduled in the response so the
    Knowledge record and the Actions queue stay traceable via
    source_record_id == knowledge record id.
    """
    title = str(data.get("title", "event"))
    start = _parse(data.get("start"))

    if start is None:
        return {
            "ok": False,
            "reason": "event has no parseable start time",
        }

    recurring = bool(data.get("recurring"))
    recurrence = data.get("recurrence") if recurring else None

    if recurring and recurrence not in {"daily", "weekly"}:
        return {
            "ok": False,
            "reason": (
                f"unsupported recurrence '{recurrence}'"
            ),
        }

    action_type = data.get("action_type") or "reminder"

    payload = {
        "title": title,
        "message": data.get("action_message")
        or f"{title}",
        "all_day": bool(data.get("all_day", False)),
    }

    action = actions_engine.schedule(
        action_type=action_type,
        scheduled_for=start,
        payload=payload,
        recurrence=recurrence if recurring else None,
        source="knowledge",
        source_record_id=record_id,
        knowledge_type="calendar_event",
        metadata={
            "temporal_expression": data.get(
                "temporal_expression"
            ),
        },
    )

    return {
        "ok": True,
        "scheduled_action_ids": [action.id],
        "action_type": action.action_type,
        "first_fire_at": action.scheduled_for.isoformat(),
        "recurrence": recurrence,
    }


def cancel_event_actions(
    actions_engine: ActionsEngine,
    *,
    record_id: int,
    reason: str | None = None,
) -> dict[str, Any]:
    """
    Cancel every pending action spawned by a knowledge record,
    e.g. when the event itself is cancelled.
    """
    cancelled = actions_engine.cancel(
        source_record_id=record_id,
        reason=reason or "source event cancelled",
    )

    return {
        "ok": True,
        "cancelled_actions": cancelled,
    }
