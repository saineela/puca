"""
Nix Actions HTTP API.

Exposes the deterministic action engine and the nix_core session
store over HTTP so nix_core (lightweight, no torch) can log
conversational turns and manage scheduled actions.

  GET  /health     -> engine status
  POST /log        -> {"role", "content", "refs"} append a session turn
  GET  /context    -> ?limit=N un-pruned turns of the current session
  POST /propagate  -> {"knowledge_type", "source_record_id", "reason"}
                      upstream knowledge change: cancels linked actions
                      and prunes session turns that referenced it
  POST /reschedule -> {"source_record_id", "scheduled_for"} move linked
                      actions to a new time
  POST /run        -> {"mock": bool} fire due actions now
  GET  /stats      -> engine counters

Run from the nix_knowledge venv (it has nix_actions installed):

    ~/nix_knowledge/.venv/bin/python \
        ~/nix_actions/scripts/actions_api.py

Environment:
    NIX_ACTIONS_API_HOST (default 127.0.0.1)
    NIX_ACTIONS_API_PORT (default 8200)
    NIX_ACTIONS_DB       (default <repo>/data/actions.db)
    NIX_CORE_DB          (default <repo>/data/nix_core.db)
    NIX_TZ               (default America/Chicago)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "nix_actions"))
sys.path.insert(0, str(REPO_ROOT / "nix_knowledge"))
DATA_DIR = Path(os.environ.get("NIX_DATA_DIR", REPO_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

from nix_actions.engine import ActionsEngine  # noqa: E402
from nix_core.core import NixCore  # noqa: E402
from nix_core.knowledge import KnowledgeProvider, KnowledgeRecord  # noqa: E402

TIMEZONE = os.environ.get("NIX_TZ", "America/Chicago")
ACTIONS_DB = os.environ.get(
    "NIX_ACTIONS_DB", str(DATA_DIR / "actions.db")
)
CORE_DB = os.environ.get(
    "NIX_CORE_DB", str(DATA_DIR / "nix_core.db")
)


class _RemoteKnowledge(KnowledgeProvider):
    """
    nix_core's KnowledgeProvider, backed by the nix_knowledge API.

    Durable facts live in nix_knowledge; the actions API host itself
    holds none. When the knowledge API is down, lookups return
    nothing instead of failing the turn.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def lookup(self, *, query, knowledge_type=None, limit=5):
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/digest",
                json={"query": query, "limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        records: list[KnowledgeRecord] = []

        for index, line in enumerate(
            (payload.get("digest") or "").splitlines()
        ):
            if line.startswith("- "):
                records.append(
                    KnowledgeRecord(
                        id=10_000 + index,
                        knowledge_type="digest_line",
                        content=line[2:],
                        version=1,
                    )
                )

        return records

    def version_of(self, *, knowledge_type, record_id):
        # Digest lines are ephemeral views, not durable records.
        return 1 if knowledge_type == "digest_line" else -1

    def is_current(self, *, knowledge_type, record_id, version):
        return self.version_of(
            knowledge_type=knowledge_type, record_id=record_id
        ) == version


_core: NixCore | None = None
_actions: ActionsEngine | None = None


def _reset_store() -> dict[str, int]:
    """Clear scheduled actions and conversation turns for a fresh user state."""
    core = _get_core()
    actions = _get_actions()
    turns = core.sessions.connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    scheduled = actions.connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
    captures = actions.connection.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0]
    with core.sessions.connection:
        core.sessions.connection.execute("DELETE FROM turns")
    with actions.connection:
        actions.connection.execute("DELETE FROM actions")
        actions.connection.execute("DELETE FROM capture_events")
    return {
        "conversation_turns": int(turns),
        "scheduled_actions": int(scheduled),
        "capture_events": int(captures),
    }


def _get_core() -> NixCore:
    global _core
    if _core is None:
        _core = NixCore(
            knowledge=_RemoteKnowledge(
                os.environ.get(
                    "NIX_KNOWLEDGE_API_URL",
                    "http://127.0.0.1:8100",
                )
            ),
            database_path=CORE_DB,
            actions_database_path=ACTIONS_DB,
            timezone=TIMEZONE,
            session_bucket="week",
        )
    return _core


def _get_actions() -> ActionsEngine:
    global _actions
    if _actions is None:
        _actions = ActionsEngine(ACTIONS_DB, timezone=TIMEZONE)
    return _actions


class Handler(BaseHTTPRequestHandler):
    server_version = "NixActionsAPI/1.0"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, default=str
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def log_message(self, fmt, *args):
        if os.environ.get("NIX_API_VERBOSE"):
            super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            actions = _get_actions()
            core = _get_core()
            self._json(
                {
                    "ok": True,
                    "service": "nix_actions",
                    "timezone": TIMEZONE,
                    "stats": actions.stats(),
                    "sessions": core.sessions.session_stats()[-3:],
                }
            )
            return

        if parsed.path == "/context":
            limit = int(
                parse_qs(parsed.query).get("limit", ["20"])[0]
            )
            core = _get_core()
            core.maintain()
            filters = parse_qs(parsed.query)
            location = filters.get("location", [""])[0].strip()
            conversation_id = filters.get("conversation_id", [""])[0].strip()
            turns = core.sessions.context_window(
                limit=limit,
                location=location if "location" in filters else None,
                conversation_id=(
                    conversation_id if "conversation_id" in filters else None
                ),
            )
            if "location" in filters or "conversation_id" in filters:
                # A partial scope must never fall back to the shared daily or
                # weekly session; that would reintroduce cross-chat leakage.
                if not location or not conversation_id:
                    turns = []
            self._json(
                {
                    "turns": [
                        {
                            "role": turn.role,
                            "content": turn.content,
                            "session": turn.session_tag,
                            "created_at": turn.created_at.isoformat(),
                            "refs": turn.refs,
                        }
                        for turn in turns
                    ]
                }
            )
            return

        if parsed.path == "/sessions":
            core = _get_core()
            core.maintain()
            sessions = []
            for stat in core.sessions.session_stats():
                turns = core.sessions.list_turns(
                    session_tag=stat["session_tag"],
                    include_pruned=True,
                    limit=500,
                )
                sessions.append({
                    **stat,
                    "session_tag": stat["session_tag"],
                    "turn_count": len(turns),
                    "first_turn": stat.get("first_turn"),
                    "last_turn": stat.get("last_turn"),
                    "turns": [
                        {
                            "id": turn.id,
                            "session_tag": turn.session_tag,
                            "role": turn.role,
                            "content": turn.content,
                            "created_at": turn.created_at.isoformat(),
                            "pruned": turn.pruned,
                            "refs": turn.refs,
                        }
                        for turn in turns
                    ],
                })
            self._json({"ok": True, "sessions": sessions})
            return

        if parsed.path == "/stats":
            self._json(_get_actions().stats())
            return

        self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        payload = self._read_json()

        if path == "/reset":
            if payload.get("confirm") != "RESET_ALL":
                self._json({"ok": False, "error": "confirmation required"}, 400)
                return
            try:
                self._json({"ok": True, **_reset_store()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            return

        if path == "/log":
            role = str(payload.get("role") or "user")
            content = str(payload.get("content") or "")
            refs = payload.get("refs") or {}

            if not content:
                self._json(
                    {"ok": False, "error": "missing content"}, 400
                )
                return

            core = _get_core()
            turn = core.sessions.add_turn(
                role=role,
                content=content,
                refs=refs,
                session_bucket=core.session_bucket,
            )
            self._json(
                {
                    "ok": True,
                    "turn_id": turn.id,
                    "session_tag": turn.session_tag,
                }
            )
            return

        if path == "/propagate":
            record_id = payload.get("source_record_id")
            if record_id is None:
                self._json(
                    {"ok": False, "error": "missing source_record_id"},
                    400,
                )
                return

            result = _get_core().on_knowledge_updated(
                knowledge_type=payload.get("knowledge_type"),
                source_record_id=int(record_id),
                reason=str(
                    payload.get("reason") or "knowledge updated"
                ),
            )
            self._json({"ok": True, **result})
            return

        if path == "/reschedule":
            record_id = payload.get("source_record_id")
            when = payload.get("scheduled_for")

            if record_id is None or not when:
                self._json(
                    {
                        "ok": False,
                        "error": (
                            "missing source_record_id or "
                            "scheduled_for"
                        ),
                    },
                    400,
                )
                return

            count = _get_actions().reschedule(
                source_record_id=int(record_id),
                scheduled_for=datetime.fromisoformat(str(when)),
                reason=str(payload.get("reason") or "moved upstream"),
            )
            self._json({"ok": True, "rescheduled": count})
            return

        if path == "/run":
            fired = _get_actions().run_due(
                mock=bool(payload.get("mock", False))
            )
            self._json(
                {
                    "ok": True,
                    "fired": [
                        {
                            "id": action.id,
                            "type": action.action_type,
                            "status": action.status,
                        }
                        for action in fired
                    ],
                }
            )
            return

        self._json({"ok": False, "error": "not found"}, 404)


def main() -> int:
    host = os.environ.get("NIX_ACTIONS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("NIX_ACTIONS_API_PORT", "8200"))

    server = ThreadingHTTPServer((host, port), Handler)

    print(f"nix_actions API on http://{host}:{port}")
    print(f"  actions.db:  {ACTIONS_DB}")
    print(f"  nix_core.db: {CORE_DB}")
    print(f"  timezone:    {TIMEZONE}")

    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
