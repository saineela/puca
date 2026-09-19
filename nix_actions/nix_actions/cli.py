from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .engine import ActionsEngine

DEFAULT_DB = "actions.db"
DEFAULT_TZ = "America/Chicago"


def _open_engine(args) -> ActionsEngine:
    return ActionsEngine(
        args.database,
        timezone=args.timezone,
    )


def _print_actions(actions) -> None:
    if not actions:
        print("(none)")
        return

    for action in actions:
        print(
            json.dumps(
                {
                    "id": action.id,
                    "type": action.action_type,
                    "scheduled_for": action.scheduled_for.isoformat(),
                    "recurrence": action.recurrence,
                    "status": action.status,
                    "source_record_id": action.source_record_id,
                    "payload": action.payload,
                    "last_error": action.last_error,
                },
                default=str,
            )
        )


def cmd_run(args) -> int:
    """
    Run due actions once and exit (suitable for cron), or loop with
    --loop for a persistent runner. --mock flips statuses and records
    events without executing any handler code (capture mode).
    """
    engine = _open_engine(args)

    try:
        if not args.loop:
            fired = engine.run_due(mock=args.mock)
            mode = "mock" if args.mock else "live"
            print(f"fired {len(fired)} action(s) ({mode})")
            return 0

        print(
            f"nix_actions runner looping "
            f"(db={args.database}, tz={args.timezone}, "
            f"mode={'mock' if args.mock else 'live'}); Ctrl-C to stop"
        )

        while True:
            engine.run_due(mock=args.mock)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print()
        return 0

    finally:
        engine.close()


def cmd_list(args) -> int:
    engine = _open_engine(args)

    try:
        _print_actions(
            engine.list_actions(
                status=None if args.status == "all" else args.status,
                session_tag=args.session,
                source=args.source,
            )
        )
    finally:
        engine.close()

    return 0


def cmd_schedule(args) -> int:
    """
    Manually schedule an action; useful for testing handlers.
    """
    engine = _open_engine(args)

    try:
        tz = ZoneInfo(args.timezone)
        when = datetime.fromisoformat(args.at)

        if when.tzinfo is None:
            when = when.replace(tzinfo=tz)

        action = engine.schedule(
            action_type=args.type,
            scheduled_for=when,
            payload=json.loads(args.payload) if args.payload else {},
            recurrence=args.recurrence,
            source="cli",
        )

        print(f"scheduled action #{action.id}")
        _print_actions([action])

    finally:
        engine.close()

    return 0


def cmd_cancel(args) -> int:
    engine = _open_engine(args)

    try:
        count = engine.cancel(
            action_id=args.action_id,
            reason="cancelled via CLI",
        )
        print(f"cancelled {count} action(s)")

    finally:
        engine.close()

    return 0


def cmd_capture(args) -> int:
    """
    Intake command standing in for nix_core / nix_knowledge pushes.
    """
    engine = _open_engine(args)

    try:
        tz = ZoneInfo(args.timezone)
        when = datetime.fromisoformat(args.at)

        if when.tzinfo is None:
            when = when.replace(tzinfo=tz)

        action = engine.capture(
            action_type=args.type,
            scheduled_for=when,
            payload=json.loads(args.payload) if args.payload else {},
            recurrence=args.recurrence,
            source=args.source,
            session_bucket=args.session_bucket,
            metadata={"via": "cli"},
        )

        print(f"captured action #{action.id} [{action.session_tag}]")
        _print_actions([action])

    finally:
        engine.close()

    return 0


def cmd_events(args) -> int:
    engine = _open_engine(args)

    try:
        for event in engine.list_events(
            kind=args.kind,
            limit=args.limit,
        ):
            print(
                json.dumps(
                    {
                        "id": event.id,
                        "kind": event.kind,
                        "action_id": event.action_id,
                        "session_tag": event.session_tag,
                        "detail": event.detail,
                        "created_at": event.created_at.isoformat(),
                    },
                    default=str,
                )
            )
    finally:
        engine.close()

    return 0


def cmd_dashboard(args) -> int:
    """
    Full-screen terminal dashboard: live stats, actions, activity,
    sessions and handlers. Keys: 1-4 views, up/down filter, r run
    due (mock), k cancel pending, p pause, q quit.
    """
    from .dashboard import run_dashboard

    engine = _open_engine(args)

    try:
        run_dashboard(engine, refresh_seconds=args.refresh)
    finally:
        engine.close()

    return 0


def cmd_demo(args) -> int:
    """
    Schedule a few actions in the near past/future and fire them.
    """
    engine = _open_engine(args)

    try:
        now = datetime.now(ZoneInfo(args.timezone))

        past = engine.schedule(
            action_type="alarm",
            scheduled_for=now - timedelta(seconds=1),
            payload={
                "title": "wake up",
                "message": "You asked to be woken up",
            },
            source="demo",
        )

        future = engine.schedule(
            action_type="reminder",
            scheduled_for=now + timedelta(minutes=5),
            payload={"message": "future reminder"},
            source="demo",
        )

        weekly = engine.schedule(
            action_type="notify",
            scheduled_for=now - timedelta(seconds=1),
            payload={"message": "weekly robot check"},
            recurrence="weekly",
            source="demo",
        )

        print("--- firing due actions ---")
        fired = engine.run_due()
        for action in fired:
            print(
                f"  #{action.id} {action.action_type} "
                f"-> {action.status}"
                + (
                    f" ({action.last_error})"
                    if action.last_error
                    else ""
                )
            )

        print("--- still pending ---")
        _print_actions(engine.list_actions(status="pending"))

        # demonstrate weekly materialization: fire the next occurrence too
        next_week = now + timedelta(days=7, minutes=1)
        engine.run_due(now=next_week)

        print(f"--- after simulated {next_week.date()} ---")
        _print_actions(engine.list_actions(status="pending"))

    finally:
        engine.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nix-actions",
        description=(
            "Nix Actions Engine - deterministic action scheduling "
            "and hardware triggers"
        ),
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DB,
        help=f"Actions database path (default {DEFAULT_DB})",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TZ,
        help=f"Timezone name (default {DEFAULT_TZ})",
    )

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser(
        "run",
        help="fire due actions once, or loop with --loop",
    )
    run.add_argument(
        "--loop",
        action="store_true",
        help="keep running and ticking forever",
    )
    run.add_argument(
        "--interval",
        type=float,
        default=20.0,
        help="seconds between ticks when looping (default 20)",
    )
    run.add_argument(
        "--mock",
        action="store_true",
        help="capture mode: flip statuses without running handlers",
    )
    run.set_defaults(func=cmd_run)

    listing = sub.add_parser("list", help="list actions")
    listing.add_argument(
        "--status",
        default="pending",
        help="filter by status, or 'all'",
    )
    listing.add_argument(
        "--session",
        default=None,
        help="filter by session tag, e.g. week-2026-08-31",
    )
    listing.add_argument(
        "--source",
        default=None,
        help="filter by source, e.g. nix_core or nix_knowledge",
    )
    listing.set_defaults(func=cmd_list)

    capture = sub.add_parser(
        "capture",
        help="capture an action (stand-in for nix_core/knowledge push)",
    )
    capture.add_argument("--type", required=True)
    capture.add_argument("--at", required=True)
    capture.add_argument("--payload", default="{}")
    capture.add_argument("--recurrence", default=None)
    capture.add_argument(
        "--source",
        default="nix_core",
        help="nix_core, nix_knowledge, user, ... (default nix_core)",
    )
    capture.add_argument(
        "--session-bucket",
        default="week",
        choices=["day", "week"],
        help="session bucket for tagging (default week)",
    )
    capture.set_defaults(func=cmd_capture)

    events = sub.add_parser(
        "events",
        help="show the audit trail",
    )
    events.add_argument(
        "--kind",
        default=None,
        help="filter by kind: captured, fired, failed, cancelled, ...",
    )
    events.add_argument(
        "--limit",
        type=int,
        default=50,
        help="max rows (default 50)",
    )
    events.set_defaults(func=cmd_events)

    dashboard = sub.add_parser(
        "dashboard",
        help="open the live terminal dashboard",
    )
    dashboard.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="refresh seconds (default 2)",
    )
    dashboard.set_defaults(func=cmd_dashboard)

    schedule = sub.add_parser(
        "schedule",
        help="schedule a test action",
    )
    schedule.add_argument("--type", required=True)
    schedule.add_argument("--at", required=True)
    schedule.add_argument("--payload", default="{}")
    schedule.add_argument("--recurrence", default=None)
    schedule.set_defaults(func=cmd_schedule)

    cancel = sub.add_parser(
        "cancel",
        help="cancel a pending action by id",
    )
    cancel.add_argument("--action-id", type=int, required=True)
    cancel.set_defaults(func=cmd_cancel)

    demo = sub.add_parser(
        "demo",
        help="schedule and fire demo actions",
    )
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
