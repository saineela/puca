from __future__ import annotations

"""
Moments: read-time rendering of superseded person-state history.

When a state is superseded ("my sister is sick" -> "she is cured"),
the old record stays in the knowledge base with status='superseded'
and valid_until stamped. Instead of deleting that history (the user
asks "how was my sister doing last time"), this module COMBINES the
superseded records with the current state into one rendered line at
READ time:

    maanvi was sick Sep 11-12 (2 days); she is cured now (since Sep 12)

Design decisions (owner-approved 2026-09-12):
  - Compute at read time: no new records, no duplicate truth, the
    exact DB timestamps remain the single source.
  - Explicit dates ALWAYS: no "recently"/"yesterday" relative words -
    relative time is how past-memory hallucinations start. The year is
    included whenever it is not the current year.
  - Nothing here writes to the database.

All deterministic: no model calls anywhere in this module.
"""

import json
import re
from datetime import datetime
from typing import Any

# "2026-09-12T16:05:05.575607+00:00" -> date parts
_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")

_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    match = _TS_RE.match(str(raw))
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
    except ValueError:
        return None


def _fmt_date(moment: datetime, *, current_year: int) -> str:
    """'Sep 12' - year added only when it differs from the current one."""
    if moment.year != current_year:
        return f"{_MONTHS[moment.month - 1]} {moment.day}, {moment.year}"
    return f"{_MONTHS[moment.month - 1]} {moment.day}"


def _fmt_span(start: datetime, end: datetime, *, current_year: int) -> str:
    """'Sep 11' when same day, 'Sep 11-12' when a real span."""
    if start == end:
        return _fmt_date(start, current_year=current_year)
    return (
        f"{_fmt_date(start, current_year=current_year)}-"
        f"{_fmt_date(end, current_year=current_year)}"
    )


def _subject_label(data: dict[str, Any]) -> str:
    return data.get("name") or data.get("subject") or "someone close"


def _state_phrase(state: str, valence: str) -> str:
    """'was sick' / 'was doing well' for recovery states."""
    if state in ("better", "recovered", "cured", "healthy"):
        return "was doing well"
    return f"was {state}" if valence != "good" else f"was {state}"


def _current_phrase(state: str) -> str:
    if state in ("better", "recovered", "cured", "healthy"):
        return "doing well now"
    return f"{state} now"


def _days_between(start: datetime, end: datetime) -> int:
    return max((end - start).days, 0)


def collect_moments(
    engine,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Group superseded person records by subject and render one moment
    line per subject that HAS history, optionally filtered by a person
    mention in the query (same token matching as find_states).
    """
    try:
        rows = engine.database.execute(
            """
            SELECT id, data, created_at, valid_until FROM knowledge
            WHERE status = 'superseded' AND knowledge_type = 'person'
              AND valid_until IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
    except Exception:  # noqa: BLE001 - history must never break reads
        return []

    current_year = datetime.now().year
    q = (query or "").lower()

    # subject -> list of past-state dicts (ordered by record id)
    history: dict[str, list[dict[str, Any]]] = {}
    for row_id, raw, created_at, valid_until in rows:
        try:
            data = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:  # noqa: BLE001
            continue
        if data.get("statement_type") != "current_state":
            continue

        subject = _subject_label(data)

        # same filter semantics as find_states: any query token >2 chars
        # must appear in the record
        if q:
            haystack = json.dumps(data, default=str).lower()
            tokens = [
                t for t in re.findall(r"[a-z]+", q) if len(t) > 2
            ]
            if tokens and not any(t in haystack for t in tokens):
                continue

        start = _parse_ts(created_at) or _parse_ts(valid_until)
        end = _parse_ts(valid_until)
        if start is None or end is None:
            continue

        history.setdefault(subject, []).append(
            {
                "record_id": int(row_id),
                "state": data.get("state", ""),
                "valence": data.get("valence", "neutral"),
                "start": start,
                "end": end,
            }
        )

    moments: list[dict[str, Any]] = []
    for subject, past_states in history.items():
        # merge consecutive same-state entries (user repeated the same
        # news several times) - keep each state's earliest start and
        # latest end
        merged: list[dict[str, Any]] = []
        for entry in past_states:
            if (
                merged
                and merged[-1]["state"] == entry["state"]
                and entry["start"] <= merged[-1]["end"]
            ):
                merged[-1]["end"] = max(merged[-1]["end"], entry["end"])
            else:
                merged.append(dict(entry))

        current = _current_state(engine, subject)
        moments.append(
            {
                "subject": subject,
                "text": _render(subject, merged, current, current_year),
                "past_states": [
                    {
                        "state": e["state"],
                        "from": e["start"].isoformat(),
                        "until": e["end"].isoformat(),
                    }
                    for e in merged
                ],
                "current": current,
            }
        )
    return moments


def _current_state(
    engine, subject: str
) -> dict[str, Any] | None:
    """The active person record for this subject, if any."""
    try:
        rows = engine.database.execute(
            """
            SELECT data, created_at FROM knowledge
            WHERE status = 'active' AND knowledge_type = 'person'
            ORDER BY id
            """
        ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    for raw, created_at in rows:
        try:
            data = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:  # noqa: BLE001
            continue
        if _subject_label(data) == subject:
            return {
                "state": data.get("state", ""),
                "since": created_at,
            }
    return None


def _render(
    subject: str,
    past_states: list[dict[str, Any]],
    current: dict[str, Any] | None,
    current_year: int,
) -> str:
    """One explicit-date line per subject."""
    parts: list[str] = []
    for entry in past_states:
        span = _fmt_span(entry["start"], entry["end"], current_year=current_year)
        days = _days_between(entry["start"], entry["end"])
        duration = f" ({days} day{'s' if days != 1 else ''})" if days else ""
        parts.append(
            f"{subject} {_state_phrase(entry['state'], entry['valence'])} "
            f"{span}{duration}"
        )

    if current:
        since = _parse_ts(current.get("since"))
        if since:
            parts.append(
                f"{subject} is {_current_phrase(current['state'])} "
                f"(since {_fmt_date(since, current_year=current_year)})"
            )
        else:
            parts.append(
                f"{subject} is {_current_phrase(current['state'])} now"
            )
    elif parts:
        parts.append(f"no current state for {subject} since then")

    return "; ".join(parts)
