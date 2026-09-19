from __future__ import annotations

import select
import shutil
import sys
import termios
import time
import tty
from datetime import datetime
from typing import Any

from .engine import ActionsEngine

# ANSI escape helpers (stdlib only, no external TUI dependency).
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

FG_CYAN = "\033[36m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_RED = "\033[31m"
FG_MAGENTA = "\033[35m"
FG_BLUE = "\033[34m"
FG_WHITE = "\033[97m"
FG_GREY = "\033[90m"

STATUS_COLORS = {
    "pending": FG_YELLOW,
    "fired": FG_GREEN,
    "failed": FG_RED,
    "cancelled": FG_GREY,
}

EVENT_COLORS = {
    "captured": FG_CYAN,
    "fired": FG_GREEN,
    "failed": FG_RED,
    "cancelled": FG_GREY,
    "rescheduled": FG_YELLOW,
    "purged": FG_MAGENTA,
    "knowledge_updated": FG_BLUE,
}

TYPE_COLORS = {
    "alarm": FG_MAGENTA,
    "reminder": FG_CYAN,
    "notify": FG_BLUE,
    "webhook": FG_YELLOW,
    "maintenance": FG_GREY,
}

VIEWS = ("actions", "events", "sessions", "handlers")

VIEW_TITLES = {
    "actions": "CAPTURED ACTIONS",
    "events": "ACTIVITY (audit trail)",
    "sessions": "SESSIONS (day / week buckets)",
    "handlers": "REGISTERED HANDLERS",
}

REFRESH_SECONDS = 2.0


def paint(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def truncate(text: Any, width: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def short_time(dt: datetime | None) -> str:
    return dt.strftime("%a %H:%M") if dt else "-"


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")


def hide_cursor(hide: bool) -> None:
    sys.stdout.write("\033[?25l" if hide else "\033[?25h")


# ----------------------------------------------------------------------
# Rendering pieces
# ----------------------------------------------------------------------


def render_header(engine: ActionsEngine) -> None:
    stats = engine.stats()
    tz = engine.timezone

    now = datetime.now(tz)
    week_tag = engine.session_tag_for(now, bucket="week")
    day_tag = engine.session_tag_for(now, bucket="day")

    width = shutil.get_terminal_size().columns

    title = " NIX ACTIONS "
    subtitle = "capture service · deterministic scheduler · no AI"
    left = f"{paint(title, FG_WHITE + BOLD)}{paint(subtitle, FG_GREY)}"

    right_bits = [
        paint(f" {stats['pending']} pending ", FG_YELLOW),
        paint(f" {stats['due_now']} due ", FG_RED),
        paint(f" {stats['fired']} fired ", FG_GREEN),
        paint(f" {stats['failed']} failed ", FG_RED),
    ]
    right = " ".join(right_bits)

    gap = max(1, width - len(title) - len(subtitle) - len(right) - 12)
    print(f"{left}{' ' * gap}{right}")
    print(paint("═" * width, FG_GREY))
    print(
        paint("now ", FG_GREY)
        + f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        + paint(f"   session(week) {week_tag}   session(day) {day_tag}", FG_GREY)
    )


def render_footer(view: str, paused: bool) -> None:
    width = shutil.get_terminal_size().columns

    tabs = "  ".join(
        (
            f"[{i}] {name.upper()}"
            if name == view
            else f" {i}  {name} "
        )
        for i, name in enumerate(VIEWS, start=1)
    )

    keys = "r run-due(mock)  k cancel  q quit"
    if paused:
        keys = paint("⏸ paused", FG_MAGENTA) + "  " + keys

    gap = max(1, width - len(tabs) - 40)
    print(paint("─" * width, FG_GREY))
    print(f"{tabs}{' ' * gap}{paint(keys, FG_GREY)}")


def render_actions(
    engine: ActionsEngine,
    rows_limit: int,
    *,
    status_filter: str | None = "pending",
    session_filter: str | None,
) -> int:
    rows = engine.list_actions(
        status=None if status_filter == "all" else status_filter,
        session_tag=session_filter,
    )

    if rows_limit < len(rows):
        rows = rows[-rows_limit:]

    header = (
        f"{'ID':>4}  {'STATUS':<9}  {'TYPE':<11}  "
        f"{'WHEN':<12}  {'SOURCE':<14}  {'SESSION':<17}  "
        f"{'REC':<6}  DETAIL"
    )
    print(paint(header, FG_WHITE))

    if not rows:
        print(paint("(no actions captured yet)", FG_GREY))
        return len(rows)

    for action in rows:
        status_color = STATUS_COLORS.get(action.status, "")
        type_color = TYPE_COLORS.get(action.action_type, "")

        detail = action.payload.get("message") or action.payload.get("title") or ""
        device = action.payload.get("device")
        if device and device != "default":
            detail = f"{device} · {detail}"

        session_label = action.session_tag or "-"

        line = (
            f"{action.id:>4}  "
            f"{paint(f'{action.status:<9}', status_color)}  "
            f"{paint(f'{action.action_type:<11}', type_color)}  "
            f"{short_time(action.scheduled_for):<12}  "
            f"{paint(f'{action.source:<14}', FG_CYAN)}  "
            f"{paint(f'{session_label:<17}', FG_GREY)}  "
            f"{(action.recurrence or '-'):<6}  "
            f"{paint(truncate(detail, 40), FG_WHITE)}"
        )

        if action.status == "failed" and action.last_error:
            line += paint(f"  ⚠ {truncate(action.last_error, 40)}", FG_RED)

        print(line)

    print(paint(f"  {len(rows)} row(s) shown", FG_GREY))
    return len(rows)


def render_events(engine: ActionsEngine, rows_limit: int) -> int:
    events = engine.list_events(limit=rows_limit)

    header = (
        f"{'#':>4}  {'WHEN':<12}  {'KIND':<17}  "
        f"{'ACT':>4}  {'SESSION':<17}  DETAIL"
    )
    print(paint(header, FG_WHITE))

    if not events:
        print(paint("(no activity recorded yet)", FG_GREY))
        return 0

    for event in reversed(events):
        color = EVENT_COLORS.get(event.kind, "")
        session_label = event.session_tag or "-"
        action_label = event.action_id if event.action_id is not None else "-"
        line = (
            f"{event.id:>4}  "
            f"{short_time(event.created_at):<12}  "
            f"{paint(f'{event.kind:<17}', color)}  "
            f"{action_label:>4}  "
            f"{paint(f'{session_label:<17}', FG_GREY)}  "
            f"{paint(truncate(event.detail, 60), FG_WHITE)}"
        )
        print(line)

    return len(events)


def render_sessions(engine: ActionsEngine, rows_limit: int) -> int:
    sessions = engine.list_sessions()
    sessions = sessions[-rows_limit:]

    header = (
        f"{'SESSION':<17}  {'BUCKET':<6}  {'RANGE':<23}  "
        f"{'TOTAL':>5}  {'PEND':>4}  {'FIRD':>4}  {'FAIL':>4}  {'CANC':>4}"
    )
    print(paint(header, FG_WHITE))

    if not sessions:
        print(paint("(no sessions yet)", FG_GREY))
        return 0

    for session in sessions:
        span = (
            f"{session.start.isoformat()} → {session.end.isoformat()}"
            if session.end
            else session.start.isoformat()
        )
        print(
            f"{paint(f'{session.session_tag:<17}', FG_CYAN)}  "
            f"{session.bucket:<6}  {span:<23}  "
            f"{session.total:>5}  "
            f"{paint(f'{session.pending:>4}', FG_YELLOW)}  "
            f"{paint(f'{session.fired:>4}', FG_GREEN)}  "
            f"{paint(f'{session.failed:>4}', FG_RED)}  "
            f"{paint(f'{session.cancelled:>4}', FG_GREY)}"
        )

    print(
        paint(
            "  sessions clear on their own schedule; "
            "the knowledge base never clears",
            FG_GREY,
        )
    )
    return len(sessions)


def render_handlers() -> int:
    from .handlers import get_handler, list_handlers

    names = list_handlers()

    header = f"{'ACTION TYPE':<12}  {'HANDLER':<24}  BOUND"
    print(paint(header, FG_WHITE))

    if not names:
        print(paint("(no handlers registered)", FG_GREY))
        return 0

    for name in names:
        handler = get_handler(name)
        bound = (
            getattr(handler, "__module__", "?")
            + "."
            + getattr(handler, "__name__", "?")
        )
        mock = "· stub (mock until real devices wired)"
        print(
            f"{paint(f'{name:<12}', TYPE_COLORS.get(name, ''))}  "
            f"{bound:<24}  {paint(mock, FG_GREY)}"
        )

    return len(names)


def render_view(
    engine: ActionsEngine,
    view: str,
    rows_limit: int,
    *,
    status_filter: str | None,
    session_filter: str | None,
) -> None:
    if view == "actions":
        render_actions(
            engine,
            rows_limit,
            status_filter=status_filter,
            session_filter=session_filter,
        )
    elif view == "events":
        render_events(engine, rows_limit)
    elif view == "sessions":
        render_sessions(engine, rows_limit)
    else:
        render_handlers()


# ----------------------------------------------------------------------
# Keyboard
# ----------------------------------------------------------------------


class _RawKeys:
    """Minimal stdin key reader; restores the terminal on exit."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved: list[Any] | None = None
        self.enabled = False

    def __enter__(self) -> "_RawKeys":
        try:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except termios.error:
            self.enabled = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self, timeout: float) -> str | None:
        if not self.enabled:
            time.sleep(timeout)
            return None

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None

        chunk = sys.stdin.read(1)
        if chunk != "\033":
            return chunk
        follow = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.05)[0] else ""
        if follow in ("[A", "[B", "[C", "[D"):
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}[follow]
        return chunk


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


def run_dashboard(
    engine: ActionsEngine,
    *,
    refresh_seconds: float = REFRESH_SECONDS,
    once: bool = False,
) -> None:
    view = "actions"
    status_filter: str | None = "pending"
    session_filter: str | None = None
    paused = False
    message = "loading…"

    try:
        rows_limit = max(6, shutil.get_terminal_size().lines - 10)
    except Exception:
        rows_limit = 20

    with _RawKeys() as keys:
        hide_cursor(hide=True)
        try:
            while True:
                clear_screen()

                render_header(engine)
                print()

                render_view(
                    engine,
                    view,
                    rows_limit,
                    status_filter=status_filter,
                    session_filter=session_filter,
                )

                print()
                render_footer(view, paused)

                hint = f"view={view}"
                if session_filter:
                    hint += f" session={session_filter}"
                print(paint(f"  {hint}  ·  {message}", FG_GREY))

                if once:
                    break

                wait = 3600.0 if paused else refresh_seconds
                key = keys.poll(wait)

                message = ""

                if key is None:
                    continue

                if key == "q":
                    break
                if key in ("1", "2", "3", "4"):
                    view = VIEWS[int(key) - 1]
                    message = f"→ {VIEW_TITLES[view]}"
                elif key in ("up", "down"):
                    if view == "actions":
                        order = ["pending", "all", "fired", "failed", "cancelled"]
                        index = order.index(status_filter or "all")
                        index = (index + (1 if key == "down" else -1)) % len(order)
                        status_filter = order[index]
                        message = f"status filter: {status_filter}"
                elif key == "p":
                    paused = not paused
                    message = "paused" if paused else "resumed"
                elif key == "r":
                    fired = engine.run_due(mock=True)
                    message = f"ran due (mock): {len(fired)} fired"
                elif key == "k":
                    pending = engine.list_actions(status="pending")
                    if session_filter:
                        pending = [
                            a
                            for a in pending
                            if a.session_tag == session_filter
                        ]
                    if not pending:
                        message = "nothing pending to cancel"
                    else:
                        count = sum(
                            engine.cancel(
                                action_id=a.id,
                                reason="cancelled via dashboard",
                            )
                            for a in pending
                        )
                        message = f"cancelled {count} action(s)"
                else:
                    message = f"unbound key: {key!r}"
        finally:
            hide_cursor(hide=False)
            print(RESET)
