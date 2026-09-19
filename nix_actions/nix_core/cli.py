from __future__ import annotations

import argparse
import json
import sys

from .core import NixCore

DEFAULT_DB = "nix_core.db"
DEFAULT_ACTIONS_DB = "actions.db"
DEFAULT_TZ = "America/Chicago"


def _open_core(args) -> NixCore:
    return NixCore(
        database_path=args.database,
        actions_database_path=args.actions_database,
        timezone=args.timezone,
        session_bucket=args.session_bucket,
        event_log_path=args.event_log,
    )


def _print_turns(turns) -> None:
    if not turns:
        print("(no turns)")
        return

    for turn in turns:
        marker = " [PRUNED]" if turn.pruned else ""
        print(
            json.dumps(
                {
                    "id": turn.id,
                    "session": turn.session_tag,
                    "role": turn.role,
                    "content": turn.content,
                    "refs": turn.refs,
                    "pruned": turn.pruned,
                    "pruned_reason": turn.pruned_reason,
                    "created_at": turn.created_at.isoformat(),
                },
                default=str,
            )
        )


def cmd_ask(args) -> int:
    core = _open_core(args)

    try:
        core.maintain()

        response = core.handle_request(text=args.text)

        print(response.reply)
        print(
            json.dumps(
                {
                    "session": response.session_tag,
                    "turn_id": response.turn_id,
                    "knowledge_used": [
                        f"{r.knowledge_type}#{r.id} v{r.version}"
                        for r in response.knowledge_used
                    ],
                    "actions_captured": response.actions_captured,
                    "pruned_turns": response.pruned_turns,
                },
                default=str,
            )
        )
    finally:
        core.close()

    return 0


def cmd_history(args) -> int:
    core = _open_core(args)

    try:
        _print_turns(
            core.sessions.list_turns(
                session_tag=args.session,
                include_pruned=args.all,
                limit=args.limit,
            )
        )
    finally:
        core.close()

    return 0


def cmd_context(args) -> int:
    """What nix_core would actually feed to a brain right now."""
    core = _open_core(args)

    try:
        turns = core.sessions.context_window(
            session_bucket=args.session_bucket,
            limit=args.limit,
        )
        _print_turns(turns)
    finally:
        core.close()

    return 0


def cmd_sessions(args) -> int:
    core = _open_core(args)

    try:
        for stat in core.sessions.session_stats():
            print(
                json.dumps(
                    stat,
                    default=str,
                )
            )
    finally:
        core.close()

    return 0


def cmd_prune(args) -> int:
    """
    Simulate nix_knowledge reporting a record update: cancels linked
    actions in nix_actions and prunes the turns that referenced it.
    """
    core = _open_core(args)

    try:
        result = core.on_knowledge_updated(
            knowledge_type=args.knowledge_type,
            source_record_id=args.record_id,
            reason=args.reason or "updated upstream",
        )
        print(
            json.dumps(
                {
                    "record": (
                        f"{args.knowledge_type or 'record'}"
                        f"#{args.record_id}"
                    ),
                    **result,
                }
            )
        )
    finally:
        core.close()

    return 0


def cmd_maintain(args) -> int:
    core = _open_core(args)

    try:
        result = core.maintain()
        print(json.dumps(result))
    finally:
        core.close()

    return 0


def cmd_demo(args) -> int:
    """
    End-to-end: ask with knowledge, update the knowledge record, watch
    the stale turns get pruned and the linked actions get cancelled.
    """
    from nix_core.knowledge import KnowledgeRecord, StaticKnowledgeProvider

    provider = StaticKnowledgeProvider(
        records=[
            KnowledgeRecord(
                id=7,
                knowledge_type="task",
                content="Robotics class Tuesday 18:00, bring the kit",
                version=1,
            )
        ]
    )

    core = NixCore(
        knowledge=provider,
        database_path=args.database,
        actions_database_path=args.actions_database,
        timezone=args.timezone,
        session_bucket=args.session_bucket,
        event_log_path=args.event_log,
    )

    try:
        print("--- ask: referencing task#7 v1 ---")
        first = core.handle_request(
            text="when is remind me about the robotics class"
        )
        print(first.reply)

        print("--- upstream updates task#7 to v2 ---")
        provider.put(
            KnowledgeRecord(
                id=7,
                knowledge_type="task",
                content="Robotics class MOVED to Thursday 20:00",
                version=2,
            )
        )
        result = core.on_knowledge_updated(
            knowledge_type="task",
            source_record_id=7,
            reason="class moved",
        )
        print(json.dumps(result))

        print("--- ask again: sees only fresh knowledge ---")
        second = core.handle_request(text="robotics class time?")
        print(second.reply)

        print("--- context window (pruned turns excluded) ---")
        _print_turns(
            core.sessions.context_window(
                session_bucket=core.session_bucket,
            )
        )

        print("--- captured actions in nix_actions ---")
        for action in core.actions.list_actions(status=None):
            print(
                json.dumps(
                    {
                        "id": action.id,
                        "type": action.action_type,
                        "status": action.status,
                        "source_record_id": action.source_record_id,
                        "session": action.session_tag,
                    },
                    default=str,
                )
            )
    finally:
        core.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nix-core",
        description=(
            "Nix Core - session-brain: conversations in day/week "
            "sessions, knowledge pulled from providers, outcomes "
            "captured into nix_actions"
        ),
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DB,
        help=f"Sessions database path (default {DEFAULT_DB})",
    )
    parser.add_argument(
        "--actions-database",
        default=DEFAULT_ACTIONS_DB,
        help=(
            "nix_actions database path "
            f"(default {DEFAULT_ACTIONS_DB})"
        ),
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TZ,
        help=f"Timezone name (default {DEFAULT_TZ})",
    )
    parser.add_argument(
        "--session-bucket",
        default="week",
        choices=["day", "week"],
        help="Session granularity (default week)",
    )
    parser.add_argument(
        "--event-log",
        default=None,
        help="Optional JSONL file for nix_core decision logs",
    )

    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser(
        "ask",
        help="handle one user request",
    )
    ask.add_argument("text")
    ask.set_defaults(func=cmd_ask)

    history = sub.add_parser(
        "history",
        help="show logged turns",
    )
    history.add_argument(
        "--session",
        default=None,
        help="filter by session tag, e.g. week-2026-08-31",
    )
    history.add_argument(
        "--all",
        action="store_true",
        help="include pruned turns",
    )
    history.add_argument(
        "--limit",
        type=int,
        default=50,
    )
    history.set_defaults(func=cmd_history)

    context = sub.add_parser(
        "context",
        help="show the live context window (un-pruned, current session)",
    )
    context.add_argument(
        "--session-bucket",
        default=None,
        choices=["day", "week"],
    )
    context.add_argument(
        "--limit",
        type=int,
        default=50,
    )
    context.set_defaults(func=cmd_context)

    sessions = sub.add_parser(
        "sessions",
        help="per-session statistics",
    )
    sessions.set_defaults(func=cmd_sessions)

    prune = sub.add_parser(
        "prune",
        help="propagate a knowledge-record update (prune + cancel)",
    )
    prune.add_argument("--record-id", type=int, required=True)
    prune.add_argument("--knowledge-type", default=None)
    prune.add_argument("--reason", default=None)
    prune.set_defaults(func=cmd_prune)

    maintain = sub.add_parser(
        "maintain",
        help="clear stale sessions and purge old finished actions",
    )
    maintain.set_defaults(func=cmd_maintain)

    demo = sub.add_parser(
        "demo",
        help="run the knowledge-update propagation demo",
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
