"""
Nix Test Console.

One command, one browser tab: send requests to the Nix brain exactly
like the Orange Pi gateway would, and watch everything that happens in
the background - routing decisions, knowledge tool execution, action
scheduling, bridge propagation, session logging, and the chat model.

    ~/nix_knowledge/.venv/bin/python ~/nix_core/console.py

- Binds to a random free port (override: NIX_CONSOLE_PORT / HOST) and
  prints the URL. The browser only ever talks to this one port.
- Uses the real nix_knowledge / nix_actions APIs when they are already
  running; otherwise runs them in-process through an internal bridge, so
  starting the console does not open any extra listening ports.
- Every downstream call is traced in memory and shown live in the UI.

HTTP surface (all JSON except "/"):

  GET  /                -> the console UI
  GET  /api/health      -> knowledge / actions / ollama / mode status
  GET  /api/feed        -> trace events + turns + actions + knowledge
  GET  /api/state       -> same DB views without the trace
  POST /api/send        -> {"text", "location"}  full Brain.handle
  POST /api/classify    -> {"text"} dry run: routing only, no effects
  POST /api/run         -> {"mock": true} fire due actions now
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlencode, urlparse

# ----------------------------------------------------------------------
# Paths so the sibling packages import cleanly from anywhere.
# ----------------------------------------------------------------------

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CORE_DIR)
DASHBOARD_PATH = os.path.join(CORE_DIR, "dashboard.html")
MODELDEV_PATH = os.path.join(REPO_ROOT, "NIX-Modeldev", "index.html")
DASHBOARD_PAGES = frozenset({
    "home", "models", "conversations", "memories", "people", "events",
    "api", "docs", "settings", "skills", "release-notes",
})
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_knowledge"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_knowledge", "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_actions"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_actions", "scripts"))

import requests  # noqa: E402

from brain import Brain, KnowledgeClient, ActionsClient  # noqa: E402
from user_profile import get_user_name, save_user_name  # noqa: E402
from config import (  # noqa: E402
    KNOWLEDGE_API_URL,
    ACTIONS_API_URL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TIMEZONE,
    CASPER_BACKEND,
    TABBY_API_URL,
    TABBY_MODEL,
    USE_KNOWLEDGE_MODEL_GATE,
    OPENAI_API_KEY,
    OPENAI_API_ALLOW_ORIGIN,
)

DATA_DIR = os.environ.get(
    "NIX_DATA_DIR", os.path.join(REPO_ROOT, "data")
)
os.makedirs(DATA_DIR, exist_ok=True)
KNOWLEDGE_DB = os.environ.get(
    "NIX_KNOWLEDGE_DB", os.path.join(DATA_DIR, "knowledge.db")
)
ACTIONS_DB = os.environ.get(
    "NIX_ACTIONS_DB", os.path.join(DATA_DIR, "actions.db")
)
CORE_DB = os.environ.get(
    "NIX_CORE_DB", os.path.join(DATA_DIR, "nix_core.db")
)
SKILLS_STATE_PATH = os.environ.get(
    "NIX_SKILLS_STATE_PATH", os.path.join(DATA_DIR, "skills.json")
)
SKILLS_INSTALL_DIR = os.environ.get(
    "NIX_SKILLS_INSTALL_DIR", os.path.join(DATA_DIR, "skills", "installed")
)
SKILLS_REPOSITORY_MANIFEST = "nix-skills.json"
SKILLS_MAX_REPOSITORY_SKILLS = 24
SKILLS_MAX_PACKAGE_FILES = 25
SKILLS_MAX_MANIFEST_BYTES = 64 * 1024
SKILLS_MAX_FILE_BYTES = 256 * 1024
SKILLS_MAX_PACKAGE_BYTES = 1024 * 1024
SKILL_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+){0,7}\Z")
SKILL_REPOSITORY_PART_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
SKILL_FILE_PART_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SKILL_CATALOG = {
    "web-search": {
        "id": "web-search",
        "name": "Web Search",
        "version": "0.1.0",
        "description": (
            "A starter skill entry for web search. Installation/status is "
            "available; search execution is intentionally not implemented yet."
        ),
        "category": "Research",
        "implementation_status": "logic_pending",
        "runnable": False,
    },
}
_skills_state_lock = threading.Lock()

# ----------------------------------------------------------------------
# In-memory trace ring: every background call lands here.
# ----------------------------------------------------------------------

TRACE: deque[dict] = deque(maxlen=600)
_trace_lock = threading.Lock()


def _trace(stage: str, detail: str, *, ok: bool = True, ms: float | None = None) -> None:
    with _trace_lock:
        TRACE.append(
            {
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "stage": stage,
                "detail": detail[:300],
                "ok": ok,
                "ms": round(ms, 1) if ms is not None else None,
            }
        )


class _Timed:
    """Context helper: trace a downstream call with its latency."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        self.t0 = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        ms = (time.perf_counter() - self.t0) * 1000
        _trace(
            self.stage,
            self.detail,
            ok=exc_type is None,
            ms=ms,
        )


# ----------------------------------------------------------------------
# Instrumented clients: identical behavior, plus tracing.
# ----------------------------------------------------------------------


class TracedKnowledge(KnowledgeClient):
    def process(self, text: str):
        with _Timed("knowledge /process", text):
            return super().process(text)

    def classify(self, text: str) -> str:
        with _Timed("knowledge /classify (model gate)", text):
            return super().classify(text)

    def digest(self) -> str:
        with _Timed("knowledge /digest", "verified knowledge block"):
            return super().digest()

    def memory_block(self, current_text=None) -> str:
        with _Timed(
            "knowledge /memory_block",
            f"grounded context: {str(current_text or '')[:100]}",
        ):
            return super().memory_block(current_text)


class TracedActions(ActionsClient):
    def log_turn(self, **kwargs):
        with _Timed(
            "actions /log",
            f"{kwargs.get('role')}: {str(kwargs.get('content'))[:80]}",
        ):
            return super().log_turn(**kwargs)

    def context(
        self,
        limit: int = 20,
        *,
        location: str | None = None,
        conversation_id: str | None = None,
    ):
        scope = (
            f"{location}/{conversation_id}"
            if location or conversation_id
            else "current session"
        )
        with _Timed("actions /context", f"{scope} ({limit})"):
            return super().context(
                limit,
                location=location,
                conversation_id=conversation_id,
            )


# ----------------------------------------------------------------------
# Service resolution: reuse running APIs, otherwise embed them.
# ----------------------------------------------------------------------

# Sibling APIs answer through an in-process HTTP bridge instead of their
# own sockets, so the console listens on exactly one port.
_local_services: dict[str, object] = {}
_bridge_installed = False


def _reachable(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=1.5).ok
    except Exception:
        return False


class _EmbeddedServer:
    """Stand-in for ThreadingHTTPServer when driving a Handler directly."""


def _dechunk(payload: bytes) -> bytes:
    out = bytearray()
    while payload:
        head, sep, rest = payload.partition(b"\r\n")
        if not sep:
            break
        try:
            size = int(head.split(b";")[0].strip() or b"0", 16)
        except ValueError:
            break
        if size <= 0:
            break
        out += rest[:size]
        payload = rest[size + 2 :]
    return bytes(out)


def _local_call(module, method: str, url: str, kwargs: dict) -> requests.Response:
    """Answer one requests call through a sibling API Handler.

    The Handler runs over a socketpair - no listening port is opened and
    nothing outside this process can reach it.
    """
    parsed = urlparse(str(url))
    target = parsed.path or "/"
    query = parsed.query
    params = kwargs.get("params")
    if params:
        extra = urlencode(params, doseq=True)
        query = f"{query}&{extra}" if query else extra
    if query:
        target = f"{target}?{query}"

    headers = {str(k): str(v) for k, v in dict(kwargs.get("headers") or {}).items()}
    body = b""
    if kwargs.get("json") is not None:
        body = json.dumps(kwargs["json"], ensure_ascii=False, default=str).encode()
        headers.setdefault("Content-Type", "application/json")
    elif kwargs.get("data") is not None:
        data = kwargs["data"]
        if isinstance(data, bytes):
            body = data
        elif isinstance(data, str):
            body = data.encode()
        else:
            body = json.dumps(data, ensure_ascii=False, default=str).encode()
    headers.setdefault("Host", "nix-embedded")
    headers.setdefault("Connection", "close")
    headers["Content-Length"] = str(len(body))

    try:
        timeout = float(kwargs.get("timeout") or 120)
    except (TypeError, ValueError):
        timeout = 120.0

    raw_request = (
        f"{method.upper()} {target} HTTP/1.1\r\n"
        + "".join(f"{key}: {value}\r\n" for key, value in headers.items())
        + "\r\n"
    ).encode("latin-1") + body

    client_sock, server_sock = socket.socketpair()
    client_sock.settimeout(timeout)
    failure: list[BaseException] = []

    def _serve() -> None:
        try:
            module.Handler(server_sock, ("127.0.0.1", 0), _EmbeddedServer())
        except BaseException as exc:  # surface the failure to the caller
            failure.append(exc)
            try:
                payload = json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                ).encode()
                server_sock.sendall(
                    b"HTTP/1.1 500 Internal Server Error\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(payload)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                    + payload
                )
            except OSError:
                pass
        finally:
            try:
                server_sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            try:
                server_sock.close()
            except OSError:
                pass

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    chunks: list[bytes] = []
    try:
        client_sock.sendall(raw_request)
        while True:
            chunk = client_sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except socket.timeout as exc:
        raise requests.exceptions.Timeout(
            f"{url} timed out after {timeout:g}s"
        ) from exc
    except OSError as exc:
        raise requests.exceptions.ConnectionError(str(exc)) from exc
    finally:
        try:
            client_sock.close()
        except OSError:
            pass
    thread.join(timeout=5)

    raw_response = b"".join(chunks)
    head, separator, content = raw_response.partition(b"\r\n\r\n")
    if not separator:
        detail = failure[0] if failure else f"no response ({len(raw_response)} bytes)"
        raise requests.exceptions.ConnectionError(f"{url}: {detail}")

    lines = head.split(b"\r\n")
    status_parts = lines[0].decode("latin-1").split(" ", 2)
    response_headers: dict[str, str] = {}
    for line in lines[1:]:
        key, sep, value = line.partition(b":")
        if sep:
            response_headers[key.decode("latin-1").strip()] = value.decode(
                "latin-1"
            ).strip()
    if response_headers.get("Transfer-Encoding", "").lower() == "chunked":
        content = _dechunk(content)

    response = requests.models.Response()
    try:
        response.status_code = int(status_parts[1])
    except (IndexError, ValueError):
        response.status_code = 0
    response.reason = status_parts[2] if len(status_parts) > 2 else ""
    response.headers.update(response_headers)
    response._content = content
    response.url = url
    response.encoding = "utf-8"
    return response


def _install_bridge() -> None:
    """Route nix-*.local URLs straight to the in-process Handlers."""
    global _bridge_installed
    if _bridge_installed:
        return
    original_request = requests.Session.request

    def _request(self, method, url, *args, **kwargs):
        module = _local_services.get(urlparse(str(url)).netloc)
        if module is not None:
            return _local_call(module, str(method), str(url), kwargs)
        return original_request(self, method, url, *args, **kwargs)

    requests.Session.request = _request
    _bridge_installed = True


def _embed(module_name: str) -> str:
    """Run a sibling API in-process; opens no listening port."""
    module = __import__(module_name)
    host = f"nix-{module_name}.local"
    _local_services[host] = module
    _install_bridge()
    return f"http://{host}"


def resolve_services() -> tuple[str, str, list[str]]:
    """Reuse local sibling APIs, or run isolated APIs in-process."""
    global KNOWLEDGE_DB, ACTIONS_DB, CORE_DB
    notes: list[str] = []
    isolated_data = os.environ.get("NIX_ISOLATE_DASHBOARD_DATA") == "1"

    if isolated_data:
        os.makedirs(DATA_DIR, exist_ok=True)
        # A sandbox dashboard must not inherit personal/default DB paths or
        # reuse already-running APIs, even if their health endpoints respond.
        KNOWLEDGE_DB = os.path.join(DATA_DIR, "knowledge.db")
        ACTIONS_DB = os.path.join(DATA_DIR, "actions.db")
        CORE_DB = os.path.join(DATA_DIR, "nix_core.db")
        os.environ["NIX_DATA_DIR"] = DATA_DIR
        os.environ["NIX_KNOWLEDGE_DB"] = KNOWLEDGE_DB
        os.environ["NIX_ACTIONS_DB"] = ACTIONS_DB
        os.environ["NIX_CORE_DB"] = CORE_DB
        os.environ["NIX_REQUEST_LOG"] = "0"

    knowledge_url = KNOWLEDGE_API_URL
    if not isolated_data and _reachable(knowledge_url):
        notes.append(f"knowledge API found running at {knowledge_url}")
    else:
        knowledge_url = _embed("knowledge_api")
        notes.append(f"knowledge API served in-process at {knowledge_url} (no extra port)")

    # The actions API enriches sessions through the knowledge API;
    # keep it pointed at whichever knowledge URL we settled on.
    os.environ["NIX_KNOWLEDGE_API_URL"] = knowledge_url

    actions_url = ACTIONS_API_URL
    if not isolated_data and _reachable(actions_url):
        notes.append(f"actions API found running at {actions_url}")
    else:
        actions_url = _embed("actions_api")
        notes.append(f"actions API served in-process at {actions_url} (no extra port)")

    return knowledge_url, actions_url, notes


# ----------------------------------------------------------------------
# Read-only DB views for the activity panels.
# ----------------------------------------------------------------------


def _rows(db: str, sql: str, params: tuple = ()) -> list[dict]:
    try:
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in connection.execute(sql, params)]
        finally:
            connection.close()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _view_actions() -> list[dict]:
    rows = _rows(
        ACTIONS_DB,
        "SELECT * FROM actions ORDER BY id DESC LIMIT 25",
    )
    for row in rows:
        if "payload" in row and isinstance(row["payload"], str):
            row["payload"] = row["payload"][:160]
    return rows


def _view_knowledge() -> list[dict]:
    rows = _rows(
        KNOWLEDGE_DB,
        "SELECT id, knowledge_type, status, data, created_at "
        "FROM knowledge ORDER BY id DESC LIMIT 25",
    )
    for row in rows:
        if isinstance(row.get("data"), str):
            row["data"] = row["data"][:160]
    return rows


def _view_turns() -> list[dict]:
    return _rows(
        CORE_DB,
        "SELECT id, session_tag, role, content, created_at "
        "FROM turns ORDER BY id DESC LIMIT 100",
    )


def _group_conversation_sessions(sessions: list[dict]) -> list[dict]:
    """Split day/week buckets into searchable, source-aware conversations."""
    grouped: dict[str, dict] = {}
    for session in sessions:
        session_tag = str(session.get("session_tag") or "unbucketed")
        for turn in session.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            refs = turn.get("refs") or {}
            if isinstance(refs, str):
                try:
                    refs = json.loads(refs)
                except json.JSONDecodeError:
                    refs = {}
            if not isinstance(refs, dict):
                refs = {}

            conversation_id = str(refs.get("conversation_id") or "").strip()
            location = str(refs.get("location") or "").strip()
            source = {
                "openai-api": "API",
                "dashboard": "Dashboard",
                "console": "Dashboard",
            }.get(location, location.title() if location else "Conversation")
            if conversation_id:
                group_id = f"conversation:{source.lower()}:{conversation_id}"
                key = group_id
            else:
                group_id = f"session:{session_tag}"
                key = group_id
                source = "Conversation history"

            conversation = grouped.setdefault(
                key,
                {
                    "id": group_id,
                    "conversation_id": conversation_id or None,
                    "session_tag": session_tag,
                    "source": source,
                    "turns": [],
                },
            )
            conversation["turns"].append(turn)

    conversations = list(grouped.values())
    for conversation in conversations:
        turns = conversation["turns"]
        turns.sort(
            key=lambda turn: (
                str(turn.get("created_at") or ""),
                int(turn.get("id") or 0),
            )
        )
        user_turn = next(
            (turn for turn in turns if turn.get("role") == "user"),
            None,
        )
        first_message = str((user_turn or turns[0]).get("content") or "").strip() if turns else ""
        conversation["title"] = (
            first_message[:72] + ("…" if len(first_message) > 72 else "")
            if first_message
            else "Untitled conversation"
        )
        last_turn = turns[-1] if turns else {}
        preview = str(last_turn.get("content") or "").strip()
        conversation["preview"] = preview[:180] + ("…" if len(preview) > 180 else "")
        conversation["turn_count"] = len(turns)
        conversation["first_turn"] = str(turns[0].get("created_at") or "") if turns else ""
        conversation["last_turn"] = str(last_turn.get("created_at") or "")
        conversation["pruned"] = sum(bool(turn.get("pruned")) for turn in turns)

    return sorted(
        conversations,
        key=lambda item: item.get("last_turn") or "",
        reverse=True,
    )


def _view_sessions() -> list[dict]:

    """Return live chat sessions from the Actions/Core session service.

    The dashboard process has its own environment and historically read a
    second, unused ``CORE_DB`` path, making the sessions panel appear empty.
    The Actions API owns the live SessionStore, so query it first and retain a
    read-only DB fallback for older deployments.
    """
    try:
        base_url = (
            getattr(_brain.actions, "base_url", None)
            if _brain is not None
            else ACTIONS_API_URL
        ) or ACTIONS_API_URL
        response = requests.get(f"{base_url.rstrip('/')}/sessions", timeout=5)
        response.raise_for_status()
        payload = response.json()
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            return _group_conversation_sessions(sessions)
    except Exception:
        pass

    turns = _rows(
        CORE_DB,
        "SELECT id, session_tag, role, content, refs, pruned, created_at "
        "FROM turns ORDER BY id DESC LIMIT 500",
    )
    grouped: dict[str, dict] = {}
    for turn in turns:
        tag = str(turn.get("session_tag") or "unbucketed")
        session = grouped.setdefault(
            tag,
            {
                "session_tag": tag,
                "turn_count": 0,
                "first_turn": turn.get("created_at"),
                "last_turn": turn.get("created_at"),
                "turns": [],
            },
        )
        session["turn_count"] += 1
        session["first_turn"] = min(
            session["first_turn"] or turn.get("created_at") or "",
            turn.get("created_at") or "",
        )
        session["last_turn"] = max(
            session["last_turn"] or turn.get("created_at") or "",
            turn.get("created_at") or "",
        )
        session["turns"].append(turn)
    return _group_conversation_sessions(list(grouped.values()))


def _past_cutoff(now_iso: str) -> str:
    """Events starting within the last minute still count as upcoming."""
    try:
        from datetime import timedelta

        return (
            datetime.fromisoformat(now_iso) - timedelta(seconds=60)
        ).isoformat()
    except Exception:
        return now_iso


def _schedule_view() -> dict:
    """Upcoming events with every stored parameter, plus the reminder
    alarm/reminder actions attached to each event via source_record_id."""
    events = _rows(
        KNOWLEDGE_DB,
        "SELECT id, knowledge_type, status, data, created_at, updated_at "
        "FROM knowledge WHERE knowledge_type = 'calendar_event' "
        "ORDER BY id DESC LIMIT 200",
    )
    actions = _rows(
        ACTIONS_DB,
        "SELECT id, action_type, scheduled_for, payload, recurrence, "
        "recurrence_end, source, source_record_id, knowledge_type, status, "
        "created_at, fired_at, last_error, metadata "
        "FROM actions ORDER BY id DESC LIMIT 200",
    )

    by_record: dict = {}
    for a in actions:
        if a.get("source_record_id") is not None:
            by_record.setdefault(a["source_record_id"], []).append(a)

    parsed = []
    now = datetime.now().isoformat()
    for e in events:
        raw = e.get("data")
        try:
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            data = {"raw": raw}
        attached = by_record.get(e["id"], [])
        start = str(data.get("start") or "")
        parsed.append(
            {
                "id": e["id"],
                "status": e.get("status"),
                "title": data.get("title") or "(untitled)",
                "start": start,
                "end": data.get("end"),
                "all_day": bool(data.get("all_day")),
                "temporal_expression": data.get("temporal_expression"),
                "recurrence": data.get("recurrence"),
                "extra": {
                    k: v
                    for k, v in data.items()
                    if k
                    not in (
                        "title", "start", "end", "all_day",
                        "temporal_expression", "recurrence",
                    )
                    and v not in (None, "", [], {})
                },
                "created_at": e.get("created_at"),
                # 60s grace: events clamped to "now" at creation must
                # not instantly display as past
                "is_past": bool(start and start < _past_cutoff(now)),
                "actions": [
                    {
                        "id": a.get("id"),
                        "type": a.get("action_type"),
                        "status": a.get("status"),
                        "fires": a.get("scheduled_for"),
                        "fired_at": a.get("fired_at"),
                        "recurrence": a.get("recurrence"),
                        "recurrence_end": a.get("recurrence_end"),
                        "source": a.get("source"),
                        "last_error": a.get("last_error"),
                    }
                    for a in attached
                ],
            }
        )
    parsed.sort(key=lambda x: x["start"] or "")
    upcoming = [e for e in parsed if not e["is_past"] and e["status"] == "active"]
    past = [e for e in parsed if e["is_past"] or e["status"] != "active"]
    return {
        "upcoming": upcoming,
        "past": past,
        "standalone_actions": [
            {
                "id": a.get("id"),
                "type": a.get("action_type"),
                "status": a.get("status"),
                "fires": a.get("scheduled_for"),
                "fired_at": a.get("fired_at"),
                "source": a.get("source"),
                "payload": a.get("payload"),
            }
            for a in actions
            if a.get("source_record_id") is None
        ],
    }


def _kb_view() -> list[dict]:
    """Full knowledge base with parsed data for the KB page."""
    rows = _rows(
        KNOWLEDGE_DB,
        "SELECT id, knowledge_type, status, data, created_at, updated_at "
        "FROM knowledge ORDER BY id",
    )
    parsed: list[dict] = []
    for row in rows:
        raw = row.get("data")
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = {"raw": raw}
        else:
            data = raw or {}
        parsed.append(
            {
                "id": row.get("id"),
                "type": row.get("knowledge_type"),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "data": data,
            }
        )
    return parsed


def _people_view() -> list[dict]:
    """People close to the user with their CURRENT state each."""
    rows = _rows(
        KNOWLEDGE_DB,
        "SELECT id, data, created_at, updated_at FROM knowledge "
        "WHERE knowledge_type = 'person' AND status = 'active' "
        "ORDER BY updated_at DESC",
    )
    people: list[dict] = []
    for row in rows:
        raw = row.get("data")
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                continue
        else:
            data = raw or {}
        if data.get("statement_type") != "current_state":
            continue
        people.append(
            {
                "id": row.get("id"),
                "name": data.get("name"),
                "subject": data.get("subject", ""),
                "role": data.get("role"),
                "state": data.get("state", ""),
                "valence": data.get("valence", "neutral"),
                "value": data.get("value", ""),
                "updated_at": row.get("updated_at"),
            }
        )
    return people


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------


def _assistant_models_status() -> dict:
    """Expose selected/default model identity separately from local readiness."""
    if CASPER_BACKEND != "transformers":
        model = TABBY_MODEL if CASPER_BACKEND == "tabby" else OLLAMA_MODEL
        return {
            "active": model,
            "default_model": model,
            "assistant_name": "Casper",
            "backend": CASPER_BACKEND,
            "active_status_available": True,
            "models": [{
                "id": model,
                "label": f"Casper · {CASPER_BACKEND} backend",
                "name": "Casper",
                "backend": CASPER_BACKEND,
                "present": True,
                "active": True,
                "official": True,
                "experimental": False,
                "default": True,
                "production_default": True,
            }],
            "exclusive": True,
            "selectable": False,
        }

    # Official Luna is the source-configured Transformers default even if status
    # discovery fails. Keep that fact distinct from a live selection/load.
    from luna_runtime import LUNA_V6_MODEL_ID

    try:
        from assistant_model import (
            DEFAULT_MODEL_ID,
            assistant_name_for_model,
            available_models,
            selected_model,
        )
        default_model = DEFAULT_MODEL_ID
        active = selected_model()
        assistant_name = assistant_name_for_model(active)
    except Exception as exc:
        return {
            "active": None,
            "default_model": LUNA_V6_MODEL_ID,
            "assistant_name": "Luna",
            "backend": "transformers",
            "active_status_available": False,
            "models": [],
            "exclusive": True,
            "selectable": False,
            "status_error": type(exc).__name__,
        }

    try:
        models = available_models()
    except Exception as exc:
        return {
            "active": active,
            "default_model": default_model,
            "assistant_name": assistant_name,
            "backend": "transformers",
            "active_status_available": True,
            "active_model_present": False,
            "models": [],
            "exclusive": True,
            "selectable": False,
            "status_error": type(exc).__name__,
        }

    return {
        "active": active,
        "default_model": default_model,
        "assistant_name": assistant_name,
        "backend": "transformers",
        "active_status_available": True,
        "models": models,
        "active_model_present": any(
            model["id"] == active and model.get("present")
            for model in models
        ),
        "exclusive": True,
        "selectable": True,
    }


def _casper_models_status() -> dict:
    """Backward-compatible alias for older health clients."""
    return _assistant_models_status()


def _luna_models_status() -> dict:
    """Expose official V6 runtime and gated V7 plan without loading."""
    try:
        from luna_model import available_models, runtime_status
        return {**runtime_status(), "models": available_models()}
    except Exception as exc:
        return {
            "active": None,
            "loaded": False,
            "models": [],
            "release_status": "local_luna_runtime_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _tabby_health() -> dict:
    """Health for the optional TabbyAPI/ExLlama backend."""
    try:
        from tabby_client import TabbyClient

        return {
            "backend": "tabby",
            "endpoint": TABBY_API_URL,
            **TabbyClient().health(),
        }
    except Exception as exc:
        return {
            "backend": "tabby",
            "endpoint": TABBY_API_URL,
            "model": TABBY_MODEL,
            "ok": False,
            "model_loaded": False,
            "error": type(exc).__name__,
        }


def _ollama_health() -> dict:
    host = os.environ.get("NIX_OLLAMA_HOST", OLLAMA_HOST)
    model = os.environ.get("NIX_OLLAMA_MODEL", OLLAMA_MODEL)
    try:
        response = requests.get(
            f"http://{host}:11434/api/tags", timeout=2
        )
        names = [
            entry.get("name", "")
            for entry in response.json().get("models", [])
        ]
        return {
            "ok": response.ok,
            "host": f"{host}:11434",
            "model": model,
            "backend": CASPER_BACKEND,
            "model_loaded": model in names,
            "models": names,
        }
    except Exception as exc:
        return {
            "ok": False,
            "host": f"{host}:11434",
            "model": model,
            "backend": CASPER_BACKEND,
            "error": f"{type(exc).__name__}",
            "models": [],
        }


# ----------------------------------------------------------------------
# The brain (built once, lazily, after services resolve)
# ----------------------------------------------------------------------

_brain: Brain | None = None
_brain_lock = threading.Lock()


def get_brain() -> Brain:
    global _brain
    with _brain_lock:
        if _brain is None:
            knowledge_url, actions_url, _ = resolve_services()
            _brain = Brain(
                knowledge=TracedKnowledge(knowledge_url),
                actions=TracedActions(actions_url),
            )
            _trace(
                "console ready",
                f"knowledge={knowledge_url} actions={actions_url} "
                f"chat={getattr(_brain.ollama, 'model', OLLAMA_MODEL)} "
                f"backend={CASPER_BACKEND}",
            )
            try:
                _warmup_status = _brain.warmup()
            except Exception as exc:
                _warmup_status = {"error": f"{type(exc).__name__}: {exc}"}
            # Expose startup state without making health re-run warm-up.
            Handler.warmup_status = _warmup_status
        return _brain


# ----------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------


def _openai_content_text(content: object) -> str:
    """Extract text from OpenAI string or multipart content."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return str(text) if text is not None else ""
    if isinstance(content, list):
        return " ".join(
            _openai_content_text(part)
            for part in content
            if isinstance(part, (str, dict))
        )
    return "" if content is None else str(content)


def _openai_api_token_guard(text: str) -> str | None:
    """Identify OpenWebUI metadata-task tokens even without embedded history.

    The Task + JSON-output-key signature identifies automatic follow-up,
    title, and tag jobs. This narrow check is only used by the
    OpenAI-compatible endpoint; it is not a general prompt-injection filter.
    """
    normalized = " ".join(str(text or "").casefold().split())[:100_000]
    words = set(re.sub(r"[^a-z0-9]+", " ", normalized).split())
    has_task_header = bool(re.search(
        r"(?:^|\s)(?:#{1,6}\s*)?\btask\s*:", normalized
    ))
    has_json_contract = "json" in words and any(phrase in normalized for phrase in (
        "json format", "json object", "raw json", "json key",
    ))
    has_output_key = any(key in normalized for key in (
        '"follow_ups"', "'follow_ups'", '"follow-ups"',
        '"title"', "'title'", '"tags"', "'tags'",
    ))
    # OpenWebUI can submit metadata jobs with no transcript at all. Treat
    # the explicit Task + JSON-schema shape as the API-token signature; do
    # not depend on a chat-history block or any weekly session contents.
    if not (has_task_header and has_json_contract and has_output_key):
        return None

    generates = any(word in words for word in (
        "suggest", "generate", "create", "write", "produce",
        "categorize", "categorise", "summarize", "summarise",
    ))
    if not generates:
        return None

    if (
        re.search(r"\bfollow[\s_-]*ups?\b", normalized)
        and re.search(r"\b(?:questions?|prompts?)\b", normalized)
        and ("user" in words or "users" in words or "point of view" in normalized)
        and any(key in normalized for key in (
            "follow_ups", "follow-ups", "follow ups",
        ))
    ):
        return "follow_ups"
    if (
        "title" in words
        and any(phrase in normalized for phrase in (
            "summariz", "summaris", "main theme", "3-5 word", "3 to 5 word",
        ))
    ):
        return "title"
    if (
        ("tag" in words or "tags" in words)
        and any(phrase in normalized for phrase in (
            "categor", "main theme", "subtopic",
        ))
    ):
        return "tags"
    return None


def _openai_conversation_id(handler: BaseHTTPRequestHandler, payload: dict) -> str:
    """Namespace client thread IDs so API history cannot collide with UI IDs."""
    conversation_id = str(handler.headers.get("X-Conversation-ID") or "").strip()
    metadata = payload.get("metadata")
    if not conversation_id and isinstance(metadata, dict):
        for key in ("conversation_id", "chat_id", "thread_id", "session_id"):
            conversation_id = str(metadata.get(key) or "").strip()
            if conversation_id:
                break
    if not conversation_id:
        for key in (
            "conversation_id", "chat_id", "thread_id", "session_id",
            "chat", "conversation",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                value = value.get("id")
            conversation_id = str(value or "").strip()
            if conversation_id:
                break
    # OpenAI's `user` is a user identity, not a thread ID. Reusing it would
    # merge unrelated OpenWebUI chats into one pending context.
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    conversation_id = conversation_id[:256]
    if conversation_id.startswith("openai-api:"):
        return conversation_id
    return f"openai-api:{conversation_id}"


class SkillRepositoryError(ValueError):
    """Invalid, unsupported, or incomplete public Nix skill repository."""


def _read_skills_state() -> dict:
    try:
        with open(SKILLS_STATE_PATH, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    installed = saved.get("installed", [])
    repositories = saved.get("repositories", [])
    return {
        "installed": [value for value in installed if isinstance(value, str)]
        if isinstance(installed, list) else [],
        "repositories": [value for value in repositories if isinstance(value, dict)]
        if isinstance(repositories, list) else [],
    }


def _write_skills_state(saved: dict) -> None:
    parent = os.path.dirname(os.path.abspath(SKILLS_STATE_PATH))
    os.makedirs(parent, exist_ok=True)
    temporary_path = f"{SKILLS_STATE_PATH}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(saved, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, SKILLS_STATE_PATH)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _github_repository_url(value: str) -> tuple[str, str, str]:
    """Accept only a canonical public GitHub repository URL, not a file URL."""
    if not isinstance(value, str) or len(value) > 500:
        raise SkillRepositoryError("Enter a public GitHub repository URL.")
    try:
        parsed = urlparse(value.strip())
    except ValueError as exc:
        raise SkillRepositoryError("Enter a valid public GitHub repository URL.") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
    ):
        raise SkillRepositoryError(
            "Use an HTTPS repository URL in the form https://github.com/owner/repo."
        )
    match = re.fullmatch(
        r"/([A-Za-z0-9_.-]{1,100})/([A-Za-z0-9_.-]{1,100})(?:/)?",
        parsed.path,
    )
    if not match:
        raise SkillRepositoryError(
            "Use a repository root URL (https://github.com/owner/repo), not a file or branch URL."
        )
    owner, repository = match.groups()
    if owner in {".", ".."} or repository in {".", ".."}:
        raise SkillRepositoryError("Invalid GitHub owner or repository name.")
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not repository:
        raise SkillRepositoryError("Invalid GitHub repository name.")
    return owner, repository, f"https://github.com/{owner}/{repository}"


def _validate_github_ref(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200:
        raise SkillRepositoryError("GitHub repository has an invalid default branch.")
    parts = value.split("/")
    if any(
        not part
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", part)
        or ".." in part
        or part.endswith(".")
        or part.endswith(".lock")
        for part in parts
    ):
        raise SkillRepositoryError("GitHub repository has an unsupported default branch name.")
    return value


def _github_raw_url(owner: str, repository: str, branch: str, path: str) -> str:
    owner = str(owner)
    repository = str(repository)
    if (
        owner in {".", ".."}
        or repository in {".", ".."}
        or not SKILL_REPOSITORY_PART_PATTERN.fullmatch(owner)
        or not SKILL_REPOSITORY_PART_PATTERN.fullmatch(repository)
    ):
        raise SkillRepositoryError("Invalid GitHub repository metadata.")
    if not isinstance(path, str) or any(
        not SKILL_FILE_PART_PATTERN.fullmatch(part) or part in {".", ".."}
        for part in path.split("/")
    ):
        raise SkillRepositoryError("Refusing an unsafe GitHub file path.")
    safe_branch = _validate_github_ref(branch)
    encoded_branch = "/".join(quote(part, safe="") for part in safe_branch.split("/"))
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
    return (
        f"https://raw.githubusercontent.com/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}/{encoded_branch}/{encoded_path}"
    )


def _fetch_github_bytes(url: str, max_bytes: int, expected_host: str) -> bytes:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise SkillRepositoryError("Refusing an unexpected GitHub resource URL.") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != expected_host
        or parsed.query
        or parsed.fragment
    ):
        raise SkillRepositoryError("Refusing an unexpected GitHub resource URL.")
    response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json, application/json, text/plain",
            "User-Agent": "Nix-PUCA-Skills/1.0",
        },
        timeout=(3.05, 6),
        allow_redirects=False,
        stream=True,
    )
    try:
        if 300 <= response.status_code < 400:
            raise SkillRepositoryError("GitHub redirected this resource; use the canonical repository URL.")
        if response.status_code != 200:
            raise SkillRepositoryError(
                f"GitHub resource is unavailable (HTTP {response.status_code}). Check that the repository is public and contains the Nix skills manifest."
            )
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                reported_length = int(content_length)
            except ValueError:
                reported_length = None
            if reported_length is not None and reported_length > max_bytes:
                raise SkillRepositoryError("A repository file exceeds the Nix skill size limit.")
        content = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            if len(content) + len(chunk) > max_bytes:
                raise SkillRepositoryError("A repository file exceeds the Nix skill size limit.")
            content.extend(chunk)
        return bytes(content)
    finally:
        response.close()


def _fetch_github_json(url: str, max_bytes: int, expected_host: str) -> dict:
    body = _fetch_github_bytes(url, max_bytes, expected_host)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkillRepositoryError("A Nix skills manifest must be valid UTF-8 JSON.") from exc
    if not isinstance(parsed, dict):
        raise SkillRepositoryError("A Nix skills manifest must contain a JSON object.")
    return parsed


def _manifest_text(value, field: str, *, required: bool = True, limit: int = 240) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise SkillRepositoryError(f"Manifest field '{field}' must be text.")
    value = re.sub(r"[\x00-\x20\x7f]+", " ", value).strip()
    if required and not value:
        raise SkillRepositoryError(f"Manifest field '{field}' is required.")
    if len(value) > limit:
        raise SkillRepositoryError(f"Manifest field '{field}' is too long (max {limit} characters).")
    return value


def _validate_skill_file_path(path: str) -> str:
    if not isinstance(path, str) or len(path) > 180:
        raise SkillRepositoryError("Skill file paths must be short relative paths.")
    parts = path.split("/")
    if (
        not parts
        or len(parts) > 8
        or any(
            part in {".", ".."} or not SKILL_FILE_PART_PATTERN.fullmatch(part)
            for part in parts
        )
        or path == "skill.json"
    ):
        raise SkillRepositoryError(f"Unsafe or unsupported skill file path: {path!r}.")
    return path


def _manifest_string_list(value, field: str, *, max_items: int, max_length: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > max_items:
        raise SkillRepositoryError(f"Manifest field '{field}' must be a list of at most {max_items} strings.")
    result = []
    for item in value:
        text = _manifest_text(item, field, limit=max_length)
        if not text:
            raise SkillRepositoryError(f"Manifest field '{field}' contains an empty value.")
        result.append(text)
    if len(set(result)) != len(result):
        raise SkillRepositoryError(f"Manifest field '{field}' contains duplicates.")
    return result


def _parse_skill_manifest(manifest: dict, manifest_path: str, repository_publisher: str) -> dict:
    if manifest.get("kind") != "nix-skill" or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise SkillRepositoryError("Skill manifest must use kind 'nix-skill' and schema_version 1.")
    path_match = re.fullmatch(r"skills/([a-z0-9]+(?:-[a-z0-9]+){0,7})/skill\.json", manifest_path)
    if not path_match:
        raise SkillRepositoryError("Skill manifests must live at skills/<skill-id>/skill.json.")
    skill_slug = _manifest_text(manifest.get("id"), "id", limit=64)
    if not SKILL_ID_PATTERN.fullmatch(skill_slug) or path_match.group(1) != skill_slug:
        raise SkillRepositoryError("Skill id must be a lowercase slug matching its skills/<id> folder.")
    files = _manifest_string_list(
        manifest.get("files", []), "files", max_items=SKILLS_MAX_PACKAGE_FILES, max_length=180
    )
    files = [_validate_skill_file_path(path) for path in files]
    for index, path in enumerate(files):
        if any(other.startswith(path + "/") or path.startswith(other + "/") for other in files[index + 1:]):
            raise SkillRepositoryError("Skill file paths cannot contain file/directory conflicts.")
    entrypoint = _manifest_text(
        manifest.get("entrypoint"), "entrypoint", required=False, limit=180
    )
    if entrypoint:
        _validate_skill_file_path(entrypoint)
        if entrypoint not in files:
            raise SkillRepositoryError("A skill entrypoint must also appear in the files list.")
    icon = _manifest_text(manifest.get("icon", "✦"), "icon", limit=8)
    if (
        not icon
        or not icon.isprintable()
        or re.search(r"https?://|data:|[/\\]", icon, flags=re.IGNORECASE)
    ):
        raise SkillRepositoryError("Skill icon must be a short printable text glyph, not a remote image URL.")
    return {
        "slug": skill_slug,
        "name": _manifest_text(manifest.get("name"), "name", limit=80),
        "version": _manifest_text(manifest.get("version"), "version", limit=40),
        "description": _manifest_text(manifest.get("description"), "description", limit=500),
        "category": _manifest_text(manifest.get("category", "Tools"), "category", limit=32),
        "publisher": _manifest_text(
            manifest.get("publisher", repository_publisher), "publisher", limit=80
        ),
        "license": _manifest_text(manifest.get("license"), "license", limit=40),
        "icon": icon,
        "tags": _manifest_string_list(manifest.get("tags", []), "tags", max_items=8, max_length=24),
        "permissions": _manifest_string_list(
            manifest.get("permissions", []), "permissions", max_items=12, max_length=200
        ),
        "files": files,
        "entrypoint": entrypoint,
        "manifest_path": manifest_path,
    }


def _import_skill_repository(repository_url: str) -> dict:
    owner, repository, _ = _github_repository_url(repository_url)
    api_url = (
        f"https://api.github.com/repos/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}"
    )
    repo_data = _fetch_github_json(api_url, 128 * 1024, "api.github.com")
    if repo_data.get("private") is not False:
        raise SkillRepositoryError("Only public GitHub repositories can be added.")
    full_name = _manifest_text(repo_data.get("full_name"), "full_name", limit=205)
    name_parts = full_name.split("/")
    if (
        len(name_parts) != 2
        or any(part in {".", ".."} for part in name_parts)
        or any(not SKILL_REPOSITORY_PART_PATTERN.fullmatch(part) for part in name_parts)
        or name_parts[0].lower() != owner.lower()
        or name_parts[1].removesuffix(".git").lower() != repository.lower()
    ):
        raise SkillRepositoryError("GitHub returned invalid or mismatched repository metadata.")
    owner, repository = name_parts
    branch = _validate_github_ref(repo_data.get("default_branch"))
    canonical_url = f"https://github.com/{owner}/{repository}"
    raw_base = _github_raw_url(owner, repository, branch, SKILLS_REPOSITORY_MANIFEST)
    catalog = _fetch_github_json(raw_base, SKILLS_MAX_MANIFEST_BYTES, "raw.githubusercontent.com")
    if catalog.get("kind") != "nix-skills-repository" or type(catalog.get("schema_version")) is not int or catalog["schema_version"] != 1:
        raise SkillRepositoryError("Repository manifest must use kind 'nix-skills-repository' and schema_version 1.")
    repository_name = _manifest_text(catalog.get("name"), "repository name", limit=100)
    repository_description = _manifest_text(
        catalog.get("description", ""), "repository description", required=False, limit=500
    )
    publisher = _manifest_text(catalog.get("publisher"), "publisher", limit=80)
    entries = catalog.get("skills")
    if not isinstance(entries, list) or not entries or len(entries) > SKILLS_MAX_REPOSITORY_SKILLS:
        raise SkillRepositoryError(
            f"Repository must list between 1 and {SKILLS_MAX_REPOSITORY_SKILLS} skills."
        )
    skills = []
    paths = set()
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SkillRepositoryError("Each repository skill entry must be a manifest reference.")
        manifest_path = entry.get("manifest")
        path_match = re.fullmatch(
            r"skills/([a-z0-9]+(?:-[a-z0-9]+){0,7})/skill\.json",
            manifest_path if isinstance(manifest_path, str) else "",
        )
        if not path_match or manifest_path in paths:
            raise SkillRepositoryError("Skill references must uniquely use skills/<skill-id>/skill.json.")
        paths.add(manifest_path)
        manifest_url = _github_raw_url(owner, repository, branch, manifest_path)
        manifest = _fetch_github_json(
            manifest_url, SKILLS_MAX_MANIFEST_BYTES, "raw.githubusercontent.com"
        )
        skill = _parse_skill_manifest(manifest, manifest_path, publisher)
        remote_id = f"github:{owner.lower()}/{repository.lower()}:{skill['slug']}"
        if remote_id in ids:
            raise SkillRepositoryError("Repository contains duplicate skill ids.")
        ids.add(remote_id)
        skill["id"] = remote_id
        skills.append(skill)
    repository_record = {
        "key": f"{owner.lower()}/{repository.lower()}",
        "url": canonical_url,
        "owner": owner,
        "repository": repository,
        "default_branch": branch,
        "name": repository_name,
        "description": repository_description,
        "publisher": publisher,
        "skills": skills,
    }
    with _skills_state_lock:
        saved = _read_skills_state()
        existing = next(
            (
                item for item in saved["repositories"]
                if item.get("key") == repository_record["key"]
            ),
            None,
        )
        old_skill_ids = {
            skill.get("id")
            for skill in existing.get("skills", [])
            if isinstance(skill, dict) and isinstance(skill.get("id"), str)
        } if existing else set()
        new_skill_ids = {
            skill.get("id") for skill in repository_record["skills"]
        }
        installed_ids = set(saved["installed"])
        installed_ids.difference_update(old_skill_ids - new_skill_ids)
        saved["installed"] = sorted(installed_ids)
        saved["repositories"] = [
            item for item in saved["repositories"]
            if item.get("key") != repository_record["key"]
        ] + [repository_record]
        _write_skills_state(saved)
        for removed_id in old_skill_ids - new_skill_ids:
            _remove_skill_package(removed_id)
    return _skills_payload()


def _remote_skill_refs(saved: dict):
    for repository in saved.get("repositories", []):
        if not isinstance(repository, dict):
            continue
        for skill in repository.get("skills", []):
            if isinstance(skill, dict):
                yield repository, skill


def _find_remote_skill(saved: dict, skill_id: str):
    for repository, skill in _remote_skill_refs(saved):
        if skill.get("id") == skill_id:
            return repository, skill
    return None


def _public_remote_skill(repository: dict, skill: dict, installed: set[str]) -> dict:
    return {
        "id": skill.get("id"),
        "name": skill.get("name"),
        "version": skill.get("version"),
        "description": skill.get("description"),
        "category": skill.get("category"),
        "publisher": skill.get("publisher"),
        "license": skill.get("license"),
        "icon": skill.get("icon", "✦"),
        "tags": skill.get("tags", []),
        "permissions": skill.get("permissions", []),
        "files": skill.get("files", []),
        "community": True,
        "repository_name": repository.get("name", "GitHub community"),
        "repository_url": repository.get("url", ""),
        "implementation_status": "execution_disabled",
        "runnable": False,
        "installed": skill.get("id") in installed,
    }


def _skills_payload() -> dict:
    """Return starter catalog and community packages with persisted install state."""
    with _skills_state_lock:
        saved = _read_skills_state()
    installed = set(saved["installed"])
    skills = [
        {**skill, "installed": skill_id in installed, "community": False}
        for skill_id, skill in SKILL_CATALOG.items()
    ]
    skills.extend(
        _public_remote_skill(repository, skill, installed)
        for repository, skill in _remote_skill_refs(saved)
    )
    repositories = [
        {
            "name": repository.get("name", "GitHub community"),
            "url": repository.get("url", ""),
            "skill_count": len(repository.get("skills", [])),
        }
        for repository in saved["repositories"]
    ]
    return {"ok": True, "skills": skills, "repositories": repositories}


def _skill_package_path(skill_id: str) -> str:
    root = os.path.realpath(os.path.abspath(SKILLS_INSTALL_DIR))
    package_path = os.path.join(root, hashlib.sha256(skill_id.encode("utf-8")).hexdigest())
    if os.path.commonpath((root, package_path)) != root:
        raise SkillRepositoryError("Invalid installed skill package path.")
    return package_path


def _install_remote_skill_package(repository: dict, skill: dict) -> None:
    owner = repository.get("owner")
    repo_name = repository.get("repository")
    branch = _validate_github_ref(repository.get("default_branch"))
    files = skill.get("files", [])
    if not isinstance(files, list) or len(files) > SKILLS_MAX_PACKAGE_FILES:
        raise SkillRepositoryError("Installed skill has an invalid files list.")
    contents = []
    total_bytes = 0
    checked_paths = []
    for path in files:
        safe_path = _validate_skill_file_path(path)
        if safe_path in checked_paths or any(
            existing.startswith(safe_path + "/") or safe_path.startswith(existing + "/")
            for existing in checked_paths
        ):
            raise SkillRepositoryError("Installed skill contains conflicting file paths.")
        checked_paths.append(safe_path)
        source_path = f"skills/{skill.get('slug')}/{safe_path}"
        url = _github_raw_url(owner, repo_name, branch, source_path)
        body = _fetch_github_bytes(url, SKILLS_MAX_FILE_BYTES, "raw.githubusercontent.com")
        total_bytes += len(body)
        if total_bytes > SKILLS_MAX_PACKAGE_BYTES:
            raise SkillRepositoryError("Installed skill package exceeds the 1 MiB total size limit.")
        contents.append((safe_path, body))
    root = os.path.realpath(os.path.abspath(SKILLS_INSTALL_DIR))
    os.makedirs(root, mode=0o700, exist_ok=True)
    package_path = _skill_package_path(str(skill.get("id", "")))
    temp_path = tempfile.mkdtemp(prefix=".nix-skill-", dir=root)
    try:
        manifest = {
            "schema_version": 1,
            "kind": "nix-skill",
            "id": skill.get("slug"),
            "name": skill.get("name"),
            "version": skill.get("version"),
            "description": skill.get("description"),
            "category": skill.get("category"),
            "publisher": skill.get("publisher"),
            "license": skill.get("license"),
            "icon": skill.get("icon", "✦"),
            "tags": skill.get("tags", []),
            "permissions": skill.get("permissions", []),
            "files": checked_paths,
            "entrypoint": skill.get("entrypoint", ""),
        }
        with open(os.path.join(temp_path, "skill.json"), "xb", buffering=0) as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        for relative_path, body in contents:
            target = os.path.join(temp_path, *relative_path.split("/"))
            parent = os.path.dirname(target)
            os.makedirs(parent, exist_ok=True)
            if os.path.commonpath((temp_path, os.path.realpath(parent))) != temp_path:
                raise SkillRepositoryError("Skill package file escaped its temporary install directory.")
            with open(target, "xb", buffering=0) as handle:
                handle.write(body)
        backup_path = None
        if os.path.lexists(package_path):
            backup_path = tempfile.mkdtemp(prefix=".nix-skill-backup-", dir=root)
            os.rmdir(backup_path)
            os.replace(package_path, backup_path)
        try:
            os.replace(temp_path, package_path)
        except OSError:
            if backup_path and os.path.lexists(backup_path):
                os.replace(backup_path, package_path)
                backup_path = None
            raise
        if backup_path:
            if os.path.islink(backup_path) or not os.path.isdir(backup_path):
                os.unlink(backup_path)
            else:
                shutil.rmtree(backup_path)
    finally:
        if os.path.lexists(temp_path):
            if os.path.islink(temp_path) or not os.path.isdir(temp_path):
                os.unlink(temp_path)
            else:
                shutil.rmtree(temp_path)


def _remove_skill_package(skill_id: str) -> None:
    package_path = _skill_package_path(skill_id)
    if os.path.islink(package_path) or (os.path.lexists(package_path) and not os.path.isdir(package_path)):
        os.unlink(package_path)
    elif os.path.isdir(package_path):
        shutil.rmtree(package_path)


def _set_skill_installed(skill_id: str, installed: bool) -> dict:
    """Install static package files/metadata without ever enabling code execution."""
    if not isinstance(skill_id, str) or not skill_id:
        raise KeyError(skill_id)
    with _skills_state_lock:
        saved = _read_skills_state()
        is_builtin = skill_id in SKILL_CATALOG
        remote_ref = _find_remote_skill(saved, skill_id)
        if not is_builtin and remote_ref is None:
            raise KeyError(skill_id)
        if not installed:
            installed_ids = set(saved["installed"])
            installed_ids.discard(skill_id)
            if remote_ref is not None:
                _remove_skill_package(skill_id)
            saved["installed"] = sorted(installed_ids)
            _write_skills_state(saved)
            remote_ref = None
    if installed and remote_ref is not None:
        repository, skill = remote_ref
        _install_remote_skill_package(repository, skill)
    if installed:
        with _skills_state_lock:
            saved = _read_skills_state()
            if skill_id not in SKILL_CATALOG and _find_remote_skill(saved, skill_id) is None:
                raise SkillRepositoryError("Skill repository changed before installation completed; preview it again.")
            installed_ids = set(saved["installed"])
            installed_ids.add(skill_id)
            saved["installed"] = sorted(installed_ids)
            _write_skills_state(saved)
    return _skills_payload()


class Handler(BaseHTTPRequestHandler):
    server_version = "NixConsole/1.0"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", OPENAI_API_ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Conversation-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, markup: str) -> None:
        body = markup.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def log_message(self, fmt, *args):
        pass  # the trace feed is the log

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", OPENAI_API_ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Conversation-ID")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _openai_authorized(self) -> bool:
        """Authenticate external OpenAI-compatible requests when configured."""
        if not OPENAI_API_KEY:
            return True
        supplied = self.headers.get("Authorization", "")
        return supplied == f"Bearer {OPENAI_API_KEY}"

    def _stream_openai_result(
        self,
        *,
        text: str,
        history: list[dict[str, str]],
        conversation_id: str,
        model: str,
    ) -> None:
        """Run the complete Core pipeline and emit OpenAI SSE chunks.

        Casper's current local Transformers wrapper exposes a complete-response
        interface, not native token callbacks. We still provide a real SSE
        transport: comments keep long requests alive, then the final grounded
        Casper response is emitted in small chunks for Open WebUI's streaming
        renderer. Knowledge, Actions, session logging, and verification all
        run in the worker before any assistant content is emitted.
        """
        response_id = f"chatcmpl_nix_{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Access-Control-Allow-Origin", OPENAI_API_ALLOW_ORIGIN)
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(payload: dict) -> None:
            body = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            self.wfile.write(body)
            self.wfile.flush()

        emit({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                get_brain().handle,
                text=text,
                location="openai-api",
                conversation_id=conversation_id,
                session_context=history,
            )
            while not future.done():
                # SSE comments are invisible to Open WebUI but prevent a
                # proxy/browser from deciding that a long model call stalled.
                self.wfile.write(b": nix pipeline active\n\n")
                self.wfile.flush()
                time.sleep(0.35)
            try:
                result = future.result()
            except Exception as exc:
                emit({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": f"Core request failed: {type(exc).__name}: {exc}"}, "finish_reason": "stop"}],
                })
                emit({"error": {"message": f"Core request failed: {type(exc).__name__}: {exc}", "type": "server_error"}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

        reply = str(result.get("reply") or "")
        # Chunk the already-grounded final answer so Open WebUI renders it
        # progressively. Never stream raw Knowledge/tool output to the user.
        for start in range(0, len(reply), 32):
            emit({
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": reply[start:start + 32]}, "finish_reason": None}],
            })
            time.sleep(0.008)
        emit({
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "nix": {"route": result.get("route"), "rule": result.get("rule"), "details": result.get("details", {})},
        })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- GET ------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"

        if path.rstrip("/") == "/NIX-Modeldev":
            try:
                with open(MODELDEV_PATH, "r", encoding="utf-8") as handle:
                    self._html(handle.read())
            except OSError:
                self._json({"ok": False, "error": "model research archive unavailable"}, 500)
            return

        path_segments = [segment for segment in path.split("/") if segment]
        page = path_segments[-1] if path_segments else "home"
        is_dashboard_path = (
            page in DASHBOARD_PAGES
            and not any(segment in {"api", "v1"} for segment in path_segments[:-1])
        )
        if is_dashboard_path:
            self._html(_dashboard_markup())
            return

        path = _application_route_path(path)

        if path == "/v1/models":
            if not self._openai_authorized():
                self._json({"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}, 401)
                return
            status = _assistant_models_status()
            active_model = status.get("active") or "unknown"
            data = [
                {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "nix-puca"}
                for model_id in (
                    [active_model]
                    if status.get("backend") != "transformers"
                    else [
                        model["id"] for model in status.get("models", [])
                        if model.get("present")
                    ]
                )
            ]
            self._json({"object": "list", "data": data})
            return

        if path in {"/api/models", "/api/model"}:
            status = _assistant_models_status()
            self._json({"ok": "error" not in status, **status})
            return

        if path == "/api/skills":
            self._json(_skills_payload())
            return

        if path == "/api/profile":
            self._json({"ok": True, "user_name": get_user_name()})
            return

        if path == "/api/luna/model":
            self._json({
                "ok": False,
                "error": "Luna Pro v1 is retired and cannot be loaded or selected.",
                "code": "model_retired",
            }, 410)
            return

        if path == "/api/health":
            brain = get_brain()
            knowledge = brain.knowledge.health() or {"ok": False}
            actions = brain.actions.health() or {"ok": False}
            ollama = _ollama_health()
            tabby = _tabby_health() if CASPER_BACKEND == "tabby" else {
                "backend": "tabby", "ok": False, "model_loaded": False,
                "model": TABBY_MODEL, "not_selected": True,
            }
            try:
                import torch

                gpu = {
                    "name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None,
                    "allocated_gib": round(
                        torch.cuda.memory_allocated() / 2**30, 3
                    ) if torch.cuda.is_available() else 0.0,
                    "reserved_gib": round(
                        torch.cuda.memory_reserved() / 2**30, 3
                    ) if torch.cuda.is_available() else 0.0,
                }
            except Exception:
                gpu = {"name": None, "allocated_gib": None, "reserved_gib": None}
            self._json(
                {
                    "knowledge": knowledge,
                    "actions": actions,
                    "ollama": ollama,
                    "tabby": tabby,
                    "assistant": {
                        **_assistant_models_status(),
                        "active": getattr(brain.ollama, "model", OLLAMA_MODEL),
                        "warmup": getattr(Handler, "warmup_status", None),
                    },
                    "assistant_models": _assistant_models_status(),
                    # Compatibility aliases for older dashboards/clients.
                    "casper": {
                        "backend": CASPER_BACKEND,
                        "model": getattr(brain.ollama, "model", OLLAMA_MODEL),
                        "active": True,
                        "warmup": getattr(Handler, "warmup_status", None),
                    },
                    "casper_models": _assistant_models_status(),
                    "luna": _luna_models_status(),
                    "gpu": gpu,
                    "warmup": getattr(Handler, "warmup_status", None),
                    "embedded_services": len(_local_services),
                    "timezone": TIMEZONE,
                }
            )
            return

        if path == "/api/feed":
            with _trace_lock:
                trace = list(TRACE)
            trace.reverse()  # newest first
            self._json(
                {
                    "trace": trace,
                    "turns": _view_turns(),
                    "actions": _view_actions(),
                    "knowledge": _view_knowledge(),
                }
            )
            return

        if path == "/api/memory":
            try:
                brain = get_brain()
                self._json({"ok": True, "memory": brain.knowledge.memory_block()})
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 502)
            return

        if path == "/api/sessions":
            self._json({"ok": True, "sessions": _view_sessions()})
            return

        if path == "/api/state":
            self._json(
                {
                    "turns": _view_turns(),
                    "actions": _view_actions(),
                    "knowledge": _view_knowledge(),
                }
            )
            return

        if path == "/api/tests/info":
            batches = _load_corpora()
            self._json(
                {
                    "ok": True,
                    "batches": [
                        {"id": b["id"], "label": b["label"],
                         "size": len(b["prompts"])}
                        for b in batches
                    ],
                }
            )
            return

        if path == "/api/kb":
            self._json({"ok": True, "records": _kb_view()})
            return

        if path == "/api/people":
            self._json({"ok": True, "people": _people_view()})
            return

        if path == "/api/intent":
            body = None
            try:
                brain = get_brain()
                length = int(self.headers.get("Content-Length") or 0)
                raw = json.loads(self.rfile.read(length) or b"{}")
                text = str(raw.get("text") or "").strip()
                body = brain.knowledge.intent(text) if text else {"ok": False}
            except Exception:
                body = {"ok": False}
            self._json(body)
            return

        if path == "/api/schedule":
            self._json({"ok": True, **_schedule_view()})
            return

        self._json({"ok": False, "error": "not found"}, 404)

    # -- POST -----------------------------------------------------------

    def do_POST(self):
        path = _application_route_path(urlparse(self.path).path.rstrip("/") or "/")
        payload = self._read_json()

        if path == "/api/luna/model":
            self._json({
                "ok": False,
                "error": "Luna Pro v1 is retired and cannot be loaded or selected.",
                "code": "model_retired",
            }, 410)
            return

        if path == "/api/luna/chat":
            self._json({
                "ok": False,
                "error": "The direct Luna research chat is retired. Use the official NIX chat pipeline.",
                "code": "model_retired",
            }, 410)
            return

        if path == "/api/skills":
            action = str(payload.get("action") or "").strip()
            if action == "add_repository":
                try:
                    result = _import_skill_repository(payload.get("repository_url"))
                    _trace("skills marketplace", "public GitHub repository added")
                    self._json(result)
                except SkillRepositoryError as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                except requests.RequestException as exc:
                    self._json({"ok": False, "error": f"GitHub request failed: {exc}"}, 502)
                except OSError as exc:
                    self._json({"ok": False, "error": f"Could not save skill repository: {exc}"}, 500)
                return
            if action in {"install", "uninstall"}:
                skill_id = payload.get("skill_id")
                try:
                    result = _set_skill_installed(skill_id, action == "install")
                    _trace(
                        "skills marketplace",
                        f"{'installed' if action == 'install' else 'uninstalled'} {str(skill_id)[:100]}",
                    )
                    self._json(result)
                except KeyError:
                    self._json({"ok": False, "error": "skill was not found in the catalog"}, 404)
                except SkillRepositoryError as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                except requests.RequestException as exc:
                    self._json({"ok": False, "error": f"GitHub package download failed: {exc}"}, 502)
                except OSError as exc:
                    self._json({"ok": False, "error": f"Could not install skill package: {exc}"}, 500)
                return
            self._json({"ok": False, "error": "action must be add_repository, install, or uninstall"}, 400)
            return

        if path == "/api/profile":
            if "user_name" not in payload:
                self._json({"ok": False, "error": "missing user_name"}, 400)
                return
            try:
                user_name = save_user_name(payload["user_name"])
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
                return
            _trace(
                "instance profile updated",
                "preferred user name set" if user_name else "preferred user name cleared",
            )
            self._json({"ok": True, "user_name": user_name})
            return

        if path == "/api/model":
            requested = str(payload.get("model") or "").strip()
            if not requested:
                self._json({"ok": False, "error": "missing model"}, 400)
                return
            from luna_runtime import RETIRED_LUNA_MODEL_IDS
            if requested in RETIRED_LUNA_MODEL_IDS:
                self._json({
                    "ok": False,
                    "error": "Luna Pro v1 is retired and cannot be selected.",
                    "code": "model_retired",
                }, 410)
                return
            if CASPER_BACKEND != "transformers":
                self._json({
                    "ok": False,
                    "error": "Official adapter selection requires NIX_CASPER_BACKEND=transformers.",
                    "code": "model_selection_backend_unsupported",
                }, 409)
                return
            try:
                from assistant_model import select_model
                active = select_model(requested)
                _trace("official assistant model switch", f"exclusive model active={active}")
                self._json({
                    "ok": True,
                    "active": active,
                    "assistant_name": _assistant_models_status().get("assistant_name"),
                    "exclusive": True,
                })
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except FileNotFoundError as exc:
                self._json({"ok": False, "error": str(exc), "code": "model_unavailable"}, 409)
            except Exception as exc:
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            return

        if path == "/v1/chat/completions":
            if not self._openai_authorized():
                self._json({"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}, 401)
                return
            messages = payload.get("messages") or []
            if not isinstance(messages, list):
                self._json({"error": {"message": "messages must be an array", "type": "invalid_request_error", "param": "messages"}}, 400)
                return
            user_messages = [
                (index, message)
                for index, message in enumerate(messages)
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if not user_messages:
                self._json({"error": {"message": "at least one user message is required", "type": "invalid_request_error", "param": "messages"}}, 400)
                return
            last_user_index, last_user_message = user_messages[-1]
            text = _openai_content_text(last_user_message.get("content", "")).strip()
            if not text:
                self._json({"error": {"message": "last user message is empty", "type": "invalid_request_error", "param": "messages"}}, 400)
                return

            metadata_prompt = _openai_api_token_guard(text)
            if metadata_prompt is None:
                metadata_prompt = next((
                    kind
                    for message in messages
                    if isinstance(message, dict)
                    and message.get("role") in {"system", "developer"}
                    for kind in [_openai_api_token_guard(
                        _openai_content_text(message.get("content", ""))
                    )]
                    if kind
                ), None)
            if metadata_prompt:
                self._json({"error": {
                    "message": (
                        "API Token Guard rejected OpenWebUI's automatic "
                        "follow-up, title, or tag generation request."
                    ),
                    "type": "invalid_request_error",
                    "code": "api_token_guard_rejected",
                    "param": "messages",
                }}, 400)
                return

            # A prior metadata task may still be included in a client's full
            # transcript after its HTTP error. Drop it and subsequent replies
            # until a real user turn starts a fresh conversational segment.
            history = []
            skip_auxiliary_segment = False
            for message in messages[:last_user_index]:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")
                content = _openai_content_text(message.get("content", "")).strip()
                if role == "user":
                    skip_auxiliary_segment = bool(
                        _openai_api_token_guard(content)
                    )
                    if skip_auxiliary_segment:
                        continue
                elif role == "assistant":
                    if skip_auxiliary_segment:
                        continue
                else:
                    continue
                if content:
                    history.append({"role": role, "content": content})

            conversation_id = _openai_conversation_id(self, payload)
            requested_model = str(payload.get("model") or "").strip()
            from luna_runtime import RETIRED_LUNA_MODEL_IDS
            if requested_model in RETIRED_LUNA_MODEL_IDS:
                self._json({"error": {
                    "message": "Luna Pro v1 is retired and cannot be selected.",
                    "type": "invalid_request_error",
                    "code": "model_retired",
                }}, 410)
                return
            status = _assistant_models_status()
            active_model = status.get("active") or "unknown"
            if requested_model and requested_model != active_model:
                self._json({"error": {"message": f"Model {requested_model} is not active. Select an official model from the dashboard first.", "type": "invalid_request_error", "code": "model_not_active"}}, 409)
                return
            model = active_model
            if payload.get("stream"):
                self._stream_openai_result(
                    text=text,
                    history=history,
                    conversation_id=conversation_id,
                    model=model,
                )
                return
            try:
                result = get_brain().handle(text=text, location="openai-api", conversation_id=conversation_id, session_context=history)
            except Exception as exc:
                self._json({"error": {"message": f"Core request failed: {type(exc).__name__}: {exc}", "type": "server_error"}}, 500)
                return
            created = int(time.time())
            reply = str(result.get("reply") or "")
            self._json({
                "id": f"chatcmpl_nix_{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "nix": {"route": result.get("route"), "rule": result.get("rule"), "details": result.get("details", {})},
            })
            return

        if path == "/api/send":
            text = str(payload.get("text") or "").strip()
            location = str(payload.get("location") or "console")
            if not text:
                self._json({"ok": False, "error": "missing text"}, 400)
                return

            requested_model = str(payload.get("model") or "").strip()
            from luna_runtime import RETIRED_LUNA_MODEL_IDS
            if requested_model in RETIRED_LUNA_MODEL_IDS:
                self._json({
                    "ok": False,
                    "error": "Luna Pro v1 is retired and cannot be selected.",
                    "code": "model_retired",
                }, 410)
                return
            if requested_model:
                active_model = _assistant_models_status().get("active")
                if requested_model != active_model:
                    self._json({
                        "ok": False,
                        "error": f"Model {requested_model} is not active. Select it in Settings first.",
                        "code": "model_not_active",
                    }, 409)
                    return

            brain = get_brain()
            t0 = time.perf_counter()
            with _trace_lock:
                trace_before = len(TRACE)
            try:
                result = brain.handle(
                    text=text,
                    location=location,
                    conversation_id=str(payload.get("conversation_id") or "") or None,
                )
            except Exception as exc:
                _trace("brain error", f"{type(exc).__name__}: {exc}", ok=False)
                self._json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500
                )
                return

            elapsed = (time.perf_counter() - t0) * 1000
            with _trace_lock:
                request_trace = list(TRACE)[trace_before:]
            trace_by_stage = {}
            for event in request_trace:
                stage = event.get("stage")
                if stage and event.get("ms") is not None:
                    trace_by_stage[stage] = event["ms"]
            # Only use the routing engine's own measurement. Never estimate
            # routing by subtracting downstream stages: that remainder is
            # usually assistant generation and was the source of the misleading
            # 4049ms "Core routing" display.
            routing_details = result.get("details", {}).get("routing_engine", {})
            route_ms = routing_details.get("latency_ms")
            route_reported = route_ms is not None
            route_ms = round(float(route_ms), 3) if route_reported else 0.0
            pipeline = [
                {"id": "received", "label": "Received user query", "status": "complete", "ms": 0.0},
                {"id": "route", "label": "Core routing", "status": "complete" if route_reported else "not_reported", "ms": route_ms},
            ]
            predictor_ms = trace_by_stage.get("knowledge /classify (model gate)")
            if predictor_ms is not None:
                pipeline.append({"id": "predictor", "label": "Nix_predictor · Qwen 2.5 0.5B", "status": "complete", "ms": predictor_ms})
            else:
                pipeline.append({"id": "predictor", "label": "Nix_predictor · Qwen 2.5 0.5B", "status": "skipped", "ms": 0.0})
            knowledge_ms = trace_by_stage.get("knowledge /process")
            if knowledge_ms is not None:
                pipeline.append({"id": "knowledge", "label": "Knowledge retrieval and validation", "status": "complete", "ms": knowledge_ms})
            else:
                pipeline.append({"id": "knowledge", "label": "Knowledge retrieval and validation", "status": "skipped", "ms": 0.0})
            formatter_ms = result.get("details", {}).get("formatter_ms")
            if formatter_ms is not None:
                formatter_attempted = result.get("details", {}).get("core_formatter", {}).get("attempted", True)
                pipeline.append({"id": "formatter", "label": f"{result.get('details', {}).get('assistant_name', 'Assistant')} Knowledge formatter", "status": "complete" if formatter_attempted else "skipped", "ms": round(float(formatter_ms), 1)})
            pipeline.append({"id": "final", "label": f"{result.get('details', {}).get('assistant_name', 'Assistant')} · {result.get('details', {}).get('official_model', 'model unreported')} final response", "status": "complete", "ms": round(elapsed, 1)})
            result["pipeline"] = pipeline
            _trace(
                "brain reply",
                f"route={result.get('route')} rule={result.get('rule')} "
                f"reply={str(result.get('reply'))[:120]}",
                ms=elapsed,
            )
            self._json({"ok": True, "latency_ms": round(elapsed, 1), **result})
            return

        if path == "/api/classify":
            text = str(payload.get("text") or "").strip()
            if not text:
                self._json({"ok": False, "error": "missing text"}, 400)
                return

            from brain import creative_chat_request, is_assistant_identity_request, split_clauses
            from router import classify as rule_classify

            clauses = split_clauses(text)
            if len(clauses) == 1 and creative_chat_request(text):
                route, features = "chat", {"route": "chat", "rule": "creative_request_guard"}
            elif is_assistant_identity_request(text):
                route, features = "chat", {"route": "chat", "rule": "assistant_identity"}
            else:
                route, features = rule_classify(text)
            outcome: dict = {
                "deterministic": {"route": route, "features": features}
            }

            if route == "unknown" and USE_KNOWLEDGE_MODEL_GATE:
                brain = get_brain()
                gate = brain.knowledge.classify(text)
                outcome["model_gate"] = {"route": gate}
            elif route == "unknown":
                outcome["model_gate"] = {
                    "route": "chat",
                    "disabled": True,
                    "reason": "core_hybrid_default",
                }

            final = route if route != "unknown" else outcome.get(
                "model_gate", {}
            ).get("route")
            _trace(
                "classify (dry run)",
                f"{text[:80]} -> {final} "
                f"({outcome['deterministic']['features'].get('rule') or 'rules abstained'})",
            )
            self._json({"ok": True, "final": final, **outcome})
            return

        if path == "/api/run":
            brain = get_brain()
            try:
                response = requests.post(
                    f"{brain.actions.base_url}/run",
                    json={"mock": bool(payload.get("mock", True))},
                    timeout=15,
                )
                data = response.json()
            except Exception as exc:
                self._json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 502
                )
                return

            fired = data.get("fired", [])
            _trace(
                "actions /run",
                f"{len(fired)} action(s) fired "
                f"({', '.join(str(a.get('type')) for a in fired) or 'none'})",
            )
            self._json({"ok": True, "fired": fired})
            return

        if path == "/api/tests/run":
            self._run_tests(payload)
            return

        if path == "/api/kb/delete":
            record_id = int(payload.get("id") or 0)
            if not record_id:
                self._json({"ok": False, "error": "missing id"}, 400)
                return
            try:
                brain = get_brain()
                response = requests.post(
                    f"{brain.knowledge.base_url}/delete_record",
                    json={"id": record_id},
                    timeout=15,
                )
                data = response.json()
                if not response.ok:
                    self._json(data, response.status_code)
                    return
                deleted = int(data.get("deleted") or 0)
            except Exception as exc:
                self._json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 502
                )
                return
            _trace("kb delete", f"record #{record_id} removed ({deleted})")
            self._json({"ok": True, "deleted": deleted})
            return

        if path == "/api/reset":
            if payload.get("confirm") != "RESET_ALL":
                self._json({"ok": False, "error": "confirmation required"}, 400)
                return
            try:
                brain = get_brain()
                knowledge_response = requests.post(
                    f"{brain.knowledge.base_url}/reset",
                    json={"confirm": "RESET_ALL"},
                    timeout=30,
                )
                actions_response = requests.post(
                    f"{brain.actions.base_url}/reset",
                    json={"confirm": "RESET_ALL"},
                    timeout=30,
                )
                knowledge = knowledge_response.json()
                actions = actions_response.json()
                if not knowledge_response.ok or not actions_response.ok:
                    self._json({"ok": False, "knowledge": knowledge, "actions": actions}, 502)
                    return
                with _trace_lock:
                    TRACE.clear()
                _trace("complete reset", "Knowledge, indexes, actions, and sessions cleared")
                self._json({"ok": True, "knowledge": knowledge, "actions": actions})
            except Exception as exc:
                self._json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 502
                )
            return

        self._json({"ok": False, "error": "not found"}, 404)

    # -- test runner -----------------------------------------------------

    def _run_tests(self, payload: dict) -> None:
        """Run a labeled corpus through the REAL pipeline in a sandbox:
        real subprocess APIs on copied databases, model gate, chat
        model, bridge scheduling - every prompt answered exactly as
        production would, without touching live data."""
        batch = str(payload.get("batch") or "main")
        limit = int(payload.get("limit") or 0)
        seed = int(payload.get("seed") or 42)

        batches = {b["id"]: b for b in _load_corpora()}
        if batch not in batches:
            self._json({"ok": False, "error": f"unknown batch {batch}"}, 400)
            return

        prompts = batches[batch]["prompts"]
        if limit and limit < len(prompts):
            rng = random.Random(seed)
            prompts = rng.sample(prompts, limit)

        from router import classify as rule_classify

        sandbox = _get_sandbox()
        brain = Brain(
            knowledge=KnowledgeClient(sandbox.knowledge_url),
            actions=ActionsClient(sandbox.actions_url),
            log_requests=False,  # sandbox runs stay out of official logs
        )

        results = []
        t_all = time.perf_counter()
        for text, expected in prompts:
            t0 = time.perf_counter()
            rule_route, feats = rule_classify(text)
            rule_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            try:
                outcome = brain.handle(text=text, location="test-runner")
                error = None
            except Exception as exc:
                outcome = {}
                error = f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t1) * 1000
            got = outcome.get("route") or "error"
            reply = outcome.get("reply")
            if error:
                reply = error
            passed = _route_match(got, expected)
            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "got": got,
                    "pass": passed,
                    "rule": outcome.get("rule") or feats.get("rule"),
                    "ms": round(ms, 1),
                    "rule_ms": round(rule_ms, 3),
                    "reply": (reply or "")[:200],
                }
            )
        total_ms = (time.perf_counter() - t_all) * 1000

        passed = sum(1 for r in results if r["pass"])
        by_route: dict[str, list[float]] = {}
        by_rule: dict[str, tuple[int, float]] = {}
        for r in results:
            by_route.setdefault(r["got"], []).append(r["ms"])
            key = r["rule"] or "(none)"
            n, tot = by_rule.get(key, (0, 0.0))
            by_rule[key] = (n + 1, tot + r["ms"])

        def _avg(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 1) if xs else 0.0

        self._json(
            {
                "ok": True,
                "mode": "realtime",
                "batch": batch,
                "ran": len(results),
                "passed": passed,
                "accuracy": round(100 * passed / len(results), 2)
                if results else 0.0,
                # abstained prompts that the model layer resolved into
                # the correct route still count as passes above;
                # misroutes = confident wrong answers only.
                "misroutes": sum(
                    1 for r in results
                    if not r["pass"] and r["got"] not in ("unknown", "error")
                ),
                "errors": sum(
                    1 for r in results if r["got"] == "error"
                ),
                "total_ms": round(total_ms, 1),
                "avg_ms": _avg([r["ms"] for r in results]),
                "max_ms": round(max((r["ms"] for r in results), default=0.0), 1),
                "by_route": {
                    k: {"count": len(v), "avg_ms": _avg(v)}
                    for k, v in by_route.items()
                },
                "by_rule": sorted(
                    (
                        {
                            "rule": k,
                            "count": n,
                            "avg_ms": round(tot / n, 1) if n else 0.0,
                        }
                        for k, (n, tot) in by_rule.items()
                    ),
                    key=lambda x: -x["count"],
                ),
                "failures": [
                    r for r in results if not r["pass"]
                ][:80],
                "results": results,
            }
        )
        _trace(
            "test run (realtime)",
            f"batch={batch} ran={len(results)} "
            f"accuracy={100 * passed / len(results):.1f}% "
            f"avg={_avg([r['ms'] for r in results])}ms",
        )


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Realtime test runner: sandboxed full pipeline
# ----------------------------------------------------------------------


class RealtimeSandbox:
    """Real OS subprocess pair (knowledge API + actions API) wired to
    COPY of the live databases: identical starting data, zero effect on
    production. Prompts run through the complete Brain pipeline - model
    gate, chat model, bridge scheduling - and every write lands in the
    sandbox only."""

    def __init__(self) -> None:
        self.workdir: str | None = None
        self.procs: list = []
        self.knowledge_url: str | None = None
        self.actions_url: str | None = None

    def start(self) -> None:
        venv_python = sys.executable
        self.workdir = tempfile.mkdtemp(prefix="nix_test_sandbox_")

        # Copy live DBs so tests see real data. Missing files start
        # empty (fresh engines create their schema).
        for src, name in (
            (KNOWLEDGE_DB, "knowledge.db"),
            (ACTIONS_DB, "actions.db"),
            (CORE_DB, "nix_core.db"),
        ):
            dst = os.path.join(self.workdir, name)
            if src and os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                open(dst, "wb").close()

        env = os.environ.copy()
        env.update(
            {
                "NIX_KNOWLEDGE_DB": os.path.join(self.workdir, "knowledge.db"),
                "NIX_ACTIONS_DB": os.path.join(self.workdir, "actions.db"),
                "NIX_CORE_DB": os.path.join(self.workdir, "nix_core.db"),
            }
        )

        repo_root = os.path.dirname(_SYS)  # ~/nix_core -> ~
        knowledge_script = os.path.join(
            repo_root, "nix_knowledge", "scripts", "knowledge_api.py"
        )
        actions_script = os.path.join(
            repo_root, "nix_actions", "scripts", "actions_api.py"
        )

        def _free_port() -> int:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            return port

        kport, aport = _free_port(), _free_port()
        env["NIX_KNOWLEDGE_API_PORT"] = str(kport)
        env["NIX_KNOWLEDGE_API_URL"] = f"http://127.0.0.1:{kport}"

        kproc = subprocess.Popen(
            [venv_python, knowledge_script],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # actions API pointed at the sandbox knowledge API
        aenv = env.copy()
        aenv["NIX_ACTIONS_API_PORT"] = str(aport)
        aenv["NIX_KNOWLEDGE_API_URL"] = f"http://127.0.0.1:{kport}"
        aproc = subprocess.Popen(
            [venv_python, actions_script],
            env=aenv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.procs = [kproc, aproc]
        self.knowledge_url = f"http://127.0.0.1:{kport}"
        self.actions_url = f"http://127.0.0.1:{aport}"

        deadline = time.time() + 30
        while time.time() < deadline:
            if _reachable(self.knowledge_url) and _reachable(self.actions_url):
                return
            if kproc.poll() is not None or aproc.poll() is not None:
                break
            time.sleep(0.3)
        raise RuntimeError("sandbox APIs failed to start")

    def stop(self) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self.procs = []
        if self.workdir and os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir = None


_SANDBOX: RealtimeSandbox | None = None
_SANDBOX_LOCK = threading.Lock()


def _route_match(got: str, expected: str) -> bool:
    """Route comparison tolerant of multi-clause orderings: the brain
    reports compound routes as 'a+b' in clause order, so expected
    'knowledge+chat' matches got 'chat+knowledge' (same clause set)."""
    if got == expected:
        return True
    if "+" in str(expected) and "+" in str(got):
        return set(str(got).split("+")) == set(str(expected).split("+"))
    return False


def _get_sandbox() -> RealtimeSandbox:
    global _SANDBOX
    with _SANDBOX_LOCK:
        if _SANDBOX is None:
            sandbox = RealtimeSandbox()
            sandbox.start()
            _SANDBOX = sandbox
        return _SANDBOX


# ----------------------------------------------------------------------
# Test runner: corpus batches for the Testing page
# ----------------------------------------------------------------------

_SYS = os.path.dirname(os.path.abspath(__file__))
if _SYS not in sys.path:
    sys.path.insert(0, _SYS)


def _load_corpora() -> list[dict]:
    """Load the labeled corpora. Hard-corpus intent labels differ from
    routes (knowledge/chat), so map directly."""
    from router_corpus import CORPUS

    batches: list[dict] = [
        {"id": "main", "label": "main corpus (generated realistic prompts)",
         "prompts": [(t, e) for t, e in CORPUS]},
    ]
    try:
        from hard_corpus import HARD_PROMPTS

        batches.append({
            "id": "hard",
            "label": "hard corpus (unusual patterns, model decides)",
            "prompts": [(p, i) for p, i in HARD_PROMPTS],
        })
    except Exception:
        pass
    try:
        from adversarial_corpus import ADVERSARIAL_CORPUS

        batches.append({
            "id": "adversarial",
            "label": "adversarial corpus (1000 confusing/grounded prompts)",
            "prompts": [(t, e) for t, e, _cat, _n in ADVERSARIAL_CORPUS],
        })
    except Exception:
        pass
    return batches


# ----------------------------------------------------------------------
# UI (single page, no external assets)
# ----------------------------------------------------------------------

_PAGE_V1_UNUSED = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nix Test Console</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #2d333b;
    --text: #e6edf3; --dim: #8b949e;
    --green: #3fb950; --red: #f85149; --amber: #d29922;
    --blue: #58a6ff; --purple: #bc8cff; --cyan: #39c5cf;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 { font-size: 15px; margin: 0; letter-spacing: .4px; }
  .chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border: 1px solid var(--border);
    border-radius: 999px; font-size: 12px; color: var(--dim);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); }
  .dot.ok { background: var(--green); } .dot.bad { background: var(--red); }
  main {
    display: grid; grid-template-columns: minmax(380px, 1fr) 460px;
    gap: 16px; padding: 16px 20px; max-width: 1400px; margin: 0 auto;
  }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .panel h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: .8px;
    color: var(--dim); margin: 0 0 12px;
  }
  textarea {
    width: 100%; min-height: 64px; resize: vertical;
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; font: inherit;
  }
  textarea:focus { outline: 1px solid var(--blue); }
  .row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  button {
    background: #21262d; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; font: inherit; cursor: pointer;
  }
  button:hover { border-color: var(--blue); }
  button.primary { background: #1f6feb; border-color: #1f6feb; }
  button:disabled { opacity: .5; cursor: wait; }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
  .presets button { font-size: 12px; padding: 5px 10px; color: var(--dim); }
  .response { margin-top: 16px; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    font-size: 11px; font-weight: 600; letter-spacing: .5px;
  }
  .badge.knowledge { background: #12351c; color: var(--green); }
  .badge.chat { background: #0c2d6b; color: var(--blue); }
  .badge.person { background: #3a1d52; color: #d9b3ff; }
  .badge.unknown { background: #3d2e00; color: var(--amber); }
  .meta { color: var(--dim); font-size: 12px; margin-left: 8px; }
  .reply {
    margin-top: 10px; padding: 12px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 8px; white-space: pre-wrap;
  }
  details { margin-top: 10px; }
  summary { cursor: pointer; color: var(--dim); font-size: 12px; }
  pre.details {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; overflow: auto;
    max-height: 260px; font-size: 12px; color: var(--cyan);
  }
  /* feed */
  .tabs { display: flex; gap: 4px; margin-bottom: 10px; flex-wrap: wrap; }
  .tabs button {
    font-size: 12px; padding: 5px 12px; border-radius: 999px;
  }
  .tabs button.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  .feed { max-height: 640px; overflow-y: auto; }
  .evt {
    padding: 7px 10px; border-left: 3px solid var(--border);
    margin-bottom: 6px; background: var(--bg); border-radius: 0 6px 6px 0;
    font-size: 12.5px;
  }
  .evt.ok { border-left-color: var(--green); }
  .evt.bad { border-left-color: var(--red); }
  .evt .when { color: var(--dim); font-size: 11px; }
  .evt .stage { color: var(--purple); font-weight: 600; }
  .evt .ms { color: var(--amber); font-size: 11px; float: right; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--border); }
  th { color: var(--dim); font-weight: 500; }
  td.snip { max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .foot { color: var(--dim); font-size: 11px; padding: 8px 20px 20px; text-align: center; }
  .spinner { color: var(--dim); font-size: 12px; }
  #mic { }
  #mic.listening { background: #b62324; border-color: #f85149; color: #fff; animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: .55; } }
  #test-batch { max-width: 340px; }
  .sumgrid { display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0; font-size: 13px; }
  .sumgrid b { font-size: 16px; display: block; }
  .sumgrid .good { color: #3fb950; }
  .sumgrid .poor { color: #f85149; }
  .rpass { color: #3fb950; font-weight: 600; }
  .rfail { color: #f85149; font-weight: 600; }
  .runk { color: #d29922; font-weight: 600; }
</style>
</head>
<body>
<div class="splash" id="splash"><div class="splash-inner"><div class="splash-mark">◎</div><h1>Nix PUCA</h1><p>Casper V5 · personal user companion architecture</p><div class="splash-line"></div><span class="meta">initializing private workspace</span></div></div>
<header>
  <h1>&#9678; NIX TEST CONSOLE</h1>
  <span class="chip" id="chip-knowledge"><span class="dot"></span>knowledge</span>
  <span class="chip" id="chip-actions"><span class="dot"></span>actions</span>
  <span class="chip" id="chip-ollama"><span class="dot"></span><span id="chip-model">chat backend</span></span>
  <span class="chip" id="chip-mode"><span class="dot ok"></span><span id="mode-text">mode</span></span>
  <span class="chip" id="chip-online"><span class="dot"></span><span id="online-text">checking system</span></span>
  <button class="act theme-toggle" id="theme-toggle" title="Toggle light and dark mode"><span class="theme-glyph">☾</span><span id="theme-label">dark</span></button>
</header>

<main>
  <!-- left: request -->
  <section class="panel">
    <h2>Request</h2>
    <textarea id="input" placeholder="Type a request exactly as you'd say it to Nix... (Enter to send)"></textarea>
    <div class="row">
      <button class="primary" id="send">Send &#9654;</button>
      <button id="dry">Classify only (dry run)</button>
      <button id="run-actions" title="Fire due actions in mock mode">Fire due actions</button>
      <button id="mic" title="Speak your request (Google speech-to-text via Chrome/Edge)">&#127908; Speak</button>
      <label class="meta" style="display:inline-flex;align-items:center;gap:4px">
        <input type="checkbox" id="auto-send"> auto-send
      </label>
      <span class="spinner" id="busy"></span>
      <span class="meta" id="speech-status"></span>
    </div>        <div class="presets" id="presets"></div>
        <div class="panel" style="margin-top:16px;padding:12px" id="pipeline-panel">
          <h2>Request path <span class="meta" id="pipeline-total"></span></h2>
          <div class="status-flow" id="status-flow"><div class="meta">Send a request to watch Casper's path through the system.</div></div>
        </div>
        <div class="response" id="response" hidden>

      <span class="badge" id="r-route"></span>
      <span class="meta" id="r-meta"></span>
      <div class="reply" id="r-reply"></div>
      <details>
        <summary>raw pipeline details</summary>
        <pre class="details" id="r-details"></pre>
      </details>
    </div>
  </section>

  <!-- right: background activity -->
  <section class="panel">
    <h2>Background activity <span class="meta" id="feed-updated"></span></h2>
    <div class="tabs" id="tabs">
      <button data-tab="trace" class="active">trace</button>
      <button data-tab="actions">actions</button>
      <button data-tab="knowledge">knowledge</button>
      <button data-tab="turns">sessions</button>
    </div>
    <div class="feed" id="feed"></div>
  </section>
</main>

<!-- testing -->
<section class="panel" id="test-panel">
  <h2>Testing <span class="meta" id="test-status"></span></h2>
  <div class="row">
    <select id="test-batch"></select>
    <input id="test-limit" type="number" min="0" placeholder="limit (0 = all)" style="width:130px">
    <button class="primary" id="test-run">Run tests &#9654;</button>
    <button id="test-filter">Failures only: off</button>
  </div>
  <div class="meta" style="margin:4px 0">realtime mode: every prompt runs through the full pipeline (model gate, chat model, bridge) against sandboxed copies of your databases - nothing touches live data. ~1-3s per prompt; use a limit for a quick sample.</div>
  <div id="test-summary"></div>
  <div class="feed" id="test-feed"><div class='evt'>pick a batch and run</div></div>
</section>

<div class="foot" id="foot"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const THEME_KEY = "nix-puca-theme";
function applyTheme(theme) {
  const light = theme === "light";
  document.body.classList.toggle("light", light);
  const glyph = $("theme-toggle")?.querySelector(".theme-glyph");
  if (glyph) glyph.textContent = light ? "☀" : "☾";
  if ($("theme-label")) $("theme-label").textContent = light ? "light" : "dark";
  localStorage.setItem(THEME_KEY, light ? "light" : "dark");
}
applyTheme(localStorage.getItem(THEME_KEY) || "dark");
$("theme-toggle")?.addEventListener("click", () => {
  applyTheme(document.body.classList.contains("light") ? "dark" : "light");
});
const CHAT_SESSION_ID = (() => {
  const key = "casper-dashboard-session";
  let value = window.localStorage.getItem(key);
  if (!value) {
    value = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : String(Date.now()) + "-" + Math.random().toString(16).slice(2);
    window.localStorage.setItem(key, value);
  }
  return value;
})();
let tab = "trace";
let autoRefresh = true;

const PRESETS = [
  ["chat / world", "tell me one fun fact about octopuses, keep it short"],
  ["internet", "what's the capital of australia?"],
  ["store fact", "remember that my garage door code is 4821"],
  ["recall fact", "what is my garage door code?"],
  ["schedule", "remind me to stretch tomorrow at 9am"],
  ["browse", "what's on my schedule this week?"],
  ["cancel", "cancel my stretch reminder"],
];

function renderPresets() {
  const holder = $("presets");
  holder.innerHTML = "";
  for (const [label, text] of PRESETS) {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = text;
    b.onclick = () => { $("input").value = text; send(false); };
    holder.appendChild(b);
  }
}

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const raw = await r.text();
  let data;
  try {
    data = JSON.parse(raw);
  } catch (_) {
    throw new Error(`API ${r.status}: expected JSON but received ${raw.slice(0, 120)}`);
  }
  if (!r.ok) throw new Error(data.error || `API request failed (${r.status})`);
  return data;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

// ---- mood sounds (Web Audio, no files) ----
let MOOD_AUDIO = null;
function playMood(sound) {
  try {
    MOOD_AUDIO = MOOD_AUDIO || new (window.AudioContext || window.webkitAudioContext)();
    const ctx = MOOD_AUDIO;
    if (ctx.state === "suspended") ctx.resume();
    const now = ctx.currentTime;
    const gain = ctx.createGain();
    gain.connect(ctx.destination);
    const note = (freq, t0, dur, type, peak) => {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = type; osc.frequency.value = freq;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(peak, t0 + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(g); g.connect(gain);
      osc.start(t0); osc.stop(t0 + dur + 0.05);
    };
    if (sound === "trumpet") {
      // rising celebratory fanfare: C5 E5 G5 C6
      gain.gain.value = 0.5;
      [523.25, 659.25, 783.99, 1046.5].forEach((f, i) =>
        note(f, now + i * 0.14, 0.22, "sawtooth", 0.16));
      note(1046.5, now + 0.56, 0.5, "sawtooth", 0.2);
    } else if (sound === "sympathy") {
      // soft descending two-tone: A4 -> F4
      gain.gain.value = 0.4;
      note(440.0, now, 0.35, "sine", 0.12);
      note(349.23, now + 0.3, 0.5, "sine", 0.12);
    } else {
      // subtle acknowledgment blip
      gain.gain.value = 0.3;
      note(660.0, now, 0.12, "sine", 0.08);
    }
  } catch (e) { /* audio is best-effort */ }
}

// ---- neural filler while phi composes (never "hmm") ----
const FILLERS = {
  good: ["Let me check on that! \u263A", "One sec - this sounds good! \u266A", "Working on it, good news incoming..."],
  bad: ["Give me a moment...", "Checking on that for you.", "One moment - looking into it."],
  neutral: ["Looking that up...", "One sec.", "Checking now..."]
};
let fillerTimer = null;
function startFiller(text) {
  const box = $("r-reply");
  fetch("/api/intent", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})})
    .then((r) => r.json())
    .then((intent) => {
      const valence = intent && intent.valence && FILLERS[intent.valence] ? intent.valence : "neutral";
      const lines = FILLERS[valence];
      const pick = lines[Math.floor(Math.random() * lines.length)];
      box.hidden = false;
      box.textContent = pick + "";
      const dots = document.createElement("span");
      dots.className = "meta"; dots.textContent = " ...";
      box.appendChild(dots);
    })
    .catch(() => {});
}

async function send(dry) {
  const text = $("input").value.trim();
  if (!text) return;
  $("busy").textContent = "working...";
  $("send").disabled = true;
  if (!dry) startFiller(text);
  try {
    const result = dry ? await post("/api/classify", {text})
                       : await post("/api/send", {
                           text,
                           location: "console",
                           conversation_id: CHAT_SESSION_ID,
                         });
    show(result, dry);
    if (result && result.mood && result.mood.valence) {
      playMood(result.mood.valence === "good" ? "trumpet" : result.mood.valence === "bad" ? "sympathy" : "blip");
    }
  } catch (e) {
    show({ok: false, error: String(e)}, dry);
  } finally {
    $("busy").textContent = "";
    $("send").disabled = false;
    refreshFeed();
  }
}

function show(result, dry) {
  const box = $("response");
  box.hidden = false;
  const badge = $("r-route");
  const final = dry ? result.final : result.route;
  badge.textContent = dry ? (final || "?").toUpperCase() + " (dry run)"
                          : (final || "?").toUpperCase();
  badge.className = "badge " + (final === "knowledge" ? "knowledge"
                       : final === "chat" ? "chat" : "unknown");
  const rule = dry
    ? (result.deterministic?.features?.rule || "rules abstained")
    : (result.rule || "-");
  const ms = result.latency_ms != null ? result.latency_ms + " ms" : "";
  $("r-meta").textContent = `rule: ${rule}  ${ms}`;
  $("r-reply").textContent = result.error
    ? "ERROR: " + result.error
    : (result.reply || result.final || "(no reply)");
  $("r-details").textContent = JSON.stringify(result, null, 2);
}

function dot(ok) { return ok ? "ok" : "bad"; }

async function refreshHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    setChip("chip-knowledge", h.knowledge.ok, "knowledge");
    setChip("chip-actions", h.actions.ok, "actions");
    const o = h.casper?.backend === "tabby" ? h.tabby : h.ollama;
    setChip("chip-ollama", o.ok && o.model_loaded,
      `${o.model}${o.model_loaded ? "" : " (missing)"}`);
    $("mode-text").textContent =
      h.embedded_services > 0
        ? `embedded APIs (${h.embedded_services})`
        : "live APIs";
    const endpoint = o.host || o.endpoint || "local";
    $("foot").textContent =
      `timezone ${h.timezone} | knowledge ${h.knowledge.knowledge_records ?? "?"} records` +
      ` | ${h.casper?.backend || "chat"} ${o.model} on ${endpoint}` +
      ` | advertised models: ${(o.models || []).join(", ") || "none"}`;
  } catch (e) { /* health chip stays stale */ }
}

function setChip(id, ok, label) {
  const chip = $(id);
  chip.querySelector(".dot").className = "dot " + dot(ok);
}

function evtHtml(e) {
  const cls = e.ok ? "ok" : "bad";
  const ms = e.ms != null ? `<span class="ms">${e.ms} ms</span>` : "";
  return `<div class="evt ${cls}">${ms}
    <span class="when">${esc(e.ts.slice(11, 19))}</span>
    <span class="stage"> ${esc(e.stage)}</span><br>
    <span style="color:var(--dim)">${esc(e.detail)}</span></div>`;
}

function tableHtml(rows, cols) {
  if (!rows.length) return "<div class='evt'>nothing yet</div>";
  if (rows.length && rows[0].error)
    return `<div class="evt bad">${esc(rows[0].error)}</div>`;
  const head = cols.map(c => `<th>${c[1]}</th>`).join("");
  const body = rows.map(r => "<tr>" + cols.map(([key]) => {
    let v = r[key];
    if (v != null && typeof v === "object") v = JSON.stringify(v);
    return `<td class="snip" title="${esc(v)}">${esc(v)}</td>`;
  }).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

const COLS = {
  actions: [["id", "id"], ["action_type", "type"], ["status", "status"],
            ["scheduled_for", "fires"], ["source_record_id", "src"]],
  knowledge: [["id", "id"], ["knowledge_type", "type"], ["status", "status"],
              ["data", "data"], ["created_at", "created"]],
  turns: [["id", "id"], ["role", "role"], ["content", "content"],
          ["session_tag", "session"], ["created_at", "at"]],
};

function renderFeed(data) {
  const feed = $("feed");
  if (tab === "trace") {
    feed.innerHTML = data.trace.length
      ? data.trace.map(evtHtml).join("")
      : "<div class='evt'>no activity yet - send a request</div>";
  } else {
    feed.innerHTML = tableHtml(data[tab] || [], COLS[tab]);
  }
  $("feed-updated").textContent = "updated " + new Date().toLocaleTimeString();
}

async function refreshFeed() {
  try { renderFeed(await (await fetch("/api/feed")).json()); }
  catch (e) { /* transient */ }
}

$("send").onclick = () => send(false);
$("dry").onclick = () => send(true);
$("run-actions").onclick = async () => {
  $("busy").textContent = "firing...";
  try { await post("/api/run", {mock: true}); }
  finally { $("busy").textContent = ""; refreshFeed(); }
};
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(false); }
});
$("tabs").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  tab = b.dataset.tab;
  document.querySelectorAll("#tabs button").forEach(
    (x) => x.classList.toggle("active", x === b));
  refreshFeed();
};
document.addEventListener("visibilitychange", () => {
  autoRefresh = !document.hidden;
});

// ---------------- speech to text (Google via Web Speech API) -------
let recog = null, listening = false, finalHold = "";

function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $("mic").title = "not supported in this browser - use Chrome/Edge"; return; }
  recog = new SR();
  recog.lang = "en-US";          // Google speech-to-text backend
  recog.continuous = true;
  recog.interimResults = true;
  recog.onstart = () => {
    listening = true;
    $("mic").classList.add("listening");
    $("speech-status").textContent = "listening...";
  };
  recog.onend = () => {
    listening = false;
    $("mic").classList.remove("listening");
    $("speech-status").textContent = "";
  };
  recog.onerror = (e) => {
    $("speech-status").textContent =
      e.error === "not-allowed" ? "mic blocked - allow access" : "speech error: " + e.error;
  };
  recog.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalHold += t;
      else interim += t;
    }
    $("input").value = (finalHold + interim).trim();
    if (finalHold.trim() && !interim.trim()) {
      if ($("auto-send").checked) send(false);
      finalHold = "";
    }
  };
}

$("mic").onclick = () => {
  if (!recog) { initSpeech(); }
  if (!recog) { $("speech-status").textContent = "speech recognition unavailable (use Chrome/Edge)"; return; }
  if (listening) { recog.stop(); return; }
  finalHold = "";
  try { recog.start(); } catch (e) { /* already started */ }
};

initSpeech();

// ---------------- testing page --------------------------------------
let testsShown = true, failuresOnly = false, lastResults = null;

async function loadTestBatches() {
  try {
    const d = await (await fetch("/api/tests/info")).json();
    const sel = $("test-batch");
    sel.innerHTML = "";
    for (const b of d.batches || []) {
      const o = document.createElement("option");
      o.value = b.id;
      o.textContent = `${b.label} - ${b.size} prompts`;
      sel.appendChild(o);
    }
  } catch (e) { /* panel stays empty */ }
}

async function runTests() {
  const batch = $("test-batch").value;
  if (!batch) return;
  const limit = parseInt($("test-limit").value || "0", 10) || 0;
  $("test-status").textContent = "running...";
  $("test-run").disabled = true;
  const t0 = performance.now();
  try {
    const d = await post("/api/tests/run", {batch, limit});
    lastResults = d;
    const wall = Math.round(performance.now() - t0);
    renderSummary(d, wall);
    renderResults();
    $("test-status").textContent =
      `done in ${wall} ms wall - ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    $("test-status").textContent = "error: " + e;
  } finally {
    $("test-run").disabled = false;
  }
}

function renderSummary(d, wallMs) {
  const cls = d.accuracy >= 99 ? "good" : d.accuracy >= 90 ? "" : "poor";
  $("test-summary").innerHTML = `
    <div class="sumgrid">
      <div><b class="${cls}">${d.accuracy}%</b>accuracy<br><span class="meta">${d.passed}/${d.ran} passed</span></div>
      <div><b class="${d.misroutes === 0 ? "good" : "poor"}">${d.misroutes}</b>misroutes<br><span class="meta">wrong confident routes</span></div>
      <div><b class="${d.errors ? "poor" : "good"}">${d.errors}</b>pipeline errors<br><span class="meta">exceptions in sandbox</span></div>
      <div><b>${d.avg_ms} ms</b>avg per prompt<br><span class="meta">max ${d.max_ms} ms - ${d.total_ms} ms total</span></div>
      ${Object.entries(d.by_route || {}).map(([r, v]) =>
        `<div><b>${v.count}</b>${r}<br><span class="meta">avg ${v.avg_ms} ms</span></div>`).join("")}
    </div>`;
}

function renderResults() {
  if (!lastResults) return;
  const rows = failuresOnly
    ? (lastResults.results || []).filter(r => !r.pass)
    : (lastResults.results || []);
  const feed = $("test-feed");
  if (!rows.length) {
    feed.innerHTML = failuresOnly
      ? "<div class='evt rpass'>zero failures</div>"
      : "<div class='evt'>no rows</div>";
    return;
  }
  const lim = 300;
  const shown = rows.slice(0, lim);
  feed.innerHTML = `<table><thead><tr><th>#</th><th>result</th><th>expected</th><th>got</th><th>rule</th><th>time</th><th>prompt</th><th>nix replied</th></tr></thead><tbody>` +
    shown.map((r, i) => {
      const mark = r.pass ? "<span class='rpass'>PASS</span>"
        : r.got === "error" ? "<span class='rfail'>ERR</span>"
        : r.got === "unknown" ? "<span class='runk'>MODEL</span>"
        : "<span class='rfail'>FAIL</span>";
      return `<tr><td>${i + 1}</td><td>${mark}</td><td>${esc(r.expected)}</td><td>${esc(r.got)}</td><td>${esc(r.rule || "-")}</td><td>${r.ms} ms</td><td class="snip" title="${esc(r.text)}">${esc(r.text)}</td><td class="snip" title="${esc(r.reply)}">${esc(r.reply)}</td></tr>`;
    }).join("") + `</tbody></table>` +
    (rows.length > lim ? `<div class="evt">... ${rows.length - lim} more rows</div>` : "");
}

$("test-run").onclick = runTests;
$("test-filter").onclick = () => {
  failuresOnly = !failuresOnly;
  $("test-filter").textContent = "Failures only: " + (failuresOnly ? "on" : "off");
  renderResults();
};

renderPresets();
loadTestBatches();
refreshHealth();
refreshFeed();
setInterval(() => { if (autoRefresh) refreshFeed(); }, 2000);
setInterval(() => { if (autoRefresh) refreshHealth(); }, 10000);
</script>
</body>
</html>
"""


# ======================================================================
# UI v2: three pages (Console / Knowledge Base / Testing), dark theme
# ======================================================================

PAGE_V2 = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nix PUCA V3 · Casper V5</title>
<style>
  /* Nix PUCA V3 visual system: quiet, editorial, warm-tech; no chatbot gradients. */
  :root { --bg:#090b10; --panel:#10141b; --panel2:#151a22; --border:#27303b; --text:#edf1f5; --dim:#8995a5; --accent:#8bb8a0; --accent2:#c9a86a; --green:#72d09a; --red:#ec837d; --purple:#b8a4d8; }
  body { background: radial-gradient(circle at 82% -10%, rgba(139,184,160,.10), transparent 34%), var(--bg); letter-spacing:.005em; }
  header { min-height:64px; padding:12px 24px; background:rgba(10,13,18,.92); border-color:#26302e; backdrop-filter:blur(18px); }
  header h1 { color:var(--accent); font-size:16px; letter-spacing:.18em; }
  header h1::after { content:"  PUCA / V3"; color:var(--dim); font-size:10px; letter-spacing:.12em; }
  nav { gap:2px; }
  nav button { padding:8px 12px; }
  .theme-toggle { margin-left:4px; }
  nav button.active { color:var(--accent); background:rgba(139,184,160,.10); border-color:rgba(139,184,160,.3); }
  main { max-width:1560px; padding:28px 24px 48px; }
  .panel { border-radius:16px; box-shadow:0 12px 35px rgba(0,0,0,.14); }
  .panel h2 { color:var(--accent); letter-spacing:.14em; }
  textarea { min-height:112px; border-radius:14px; background:#0d1117; border-color:#334047; }
  textarea:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(139,184,160,.10); }
  button.act.primary { background:var(--accent); border-color:var(--accent); color:#0b1110; }
  button.act:hover { border-color:var(--accent); transform:translateY(-1px); }
  .puca-hero { display:flex; justify-content:space-between; gap:24px; align-items:flex-end; padding:8px 0 24px; }
  .eyebrow { color:var(--accent2); text-transform:uppercase; letter-spacing:.2em; font-size:11px; }
  .hero-title { font:500 clamp(28px,4vw,52px)/1.05 Georgia,serif; margin:8px 0; letter-spacing:-.03em; }
  .hero-copy { color:var(--dim); max-width:650px; margin:0; }
  .status-flow { display:grid; gap:8px; margin-top:14px; }
  .flow-stage { display:flex; align-items:center; gap:10px; padding:10px 12px; border:1px solid var(--border); background:#0d1117; border-radius:11px; transition:.2s; }
  .flow-stage.active { border-color:var(--accent2); background:rgba(201,168,106,.08); }
  .flow-stage.complete { border-color:rgba(114,208,154,.38); }
  .flow-stage.skipped { opacity:.48; }
  .flow-icon { width:19px; height:19px; border:1px solid var(--dim); border-radius:50%; display:grid; place-items:center; font-size:11px; color:var(--dim); }
  .complete .flow-icon { background:var(--green); border-color:var(--green); color:#07100b; }
  .active .flow-icon { border-color:var(--accent2); box-shadow:0 0 0 4px rgba(201,168,106,.10); }
  .flow-name { flex:1; font-size:13px; }
  .flow-time { color:var(--accent2); font-variant-numeric:tabular-nums; font-size:12px; }
  .brand-card { border-left:3px solid var(--accent); padding:14px 16px; background:linear-gradient(90deg,rgba(139,184,160,.08),transparent); }
  .brand-card h3 { font:500 24px Georgia,serif; margin:0 0 5px; }
  .brand-card p { color:var(--dim); margin:0; }
  .architecture { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  .arch-node { padding:14px; min-height:105px; border:1px solid var(--border); border-radius:12px; background:var(--panel2); }
  .arch-node b { display:block; color:var(--accent); margin-bottom:7px; }
  .arch-node small { color:var(--dim); line-height:1.45; }
  .splash { position:fixed; inset:0; z-index:20; display:grid; place-items:center; background:#090b10; transition:opacity .5s, visibility .5s; }
  .splash.hide { opacity:0; visibility:hidden; pointer-events:none; }
  .splash-inner { text-align:center; animation:rise .7s ease both; }
  .splash-mark { color:var(--accent); font-size:48px; letter-spacing:.2em; }
  .splash-inner h1 { font:500 34px Georgia,serif; margin:14px 0 6px; }
  .splash-inner p { color:var(--dim); letter-spacing:.14em; text-transform:uppercase; font-size:10px; }
  .splash-line { width:160px; height:1px; margin:24px auto; background:linear-gradient(90deg,transparent,var(--accent),transparent); }
  @keyframes rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
  .legal-copy { max-width:820px; color:#c2c9d2; line-height:1.75; }
  .theme-toggle { display:inline-flex; align-items:center; gap:7px; white-space:nowrap; }
  .theme-toggle .theme-glyph { font-size:14px; }
  body.light { --bg:#f5f7f8; --panel:#ffffff; --panel2:#f0f3f4; --border:#dce3e5; --text:#18211f; --dim:#63716f; --accent:#28765c; --accent2:#9b6a1e; --green:#22834d; --red:#c94843; --purple:#7358a8; background:linear-gradient(135deg,#f7faf9,#eef3f1); }
  body.light header { background:rgba(255,255,255,.88); border-color:#dce5e1; box-shadow:0 1px 12px rgba(34,59,49,.06); }
  body.light textarea, body.light .flow-stage { background:#fbfcfc; }
  body.light .response, body.light pre.details { background:#f5f8f7; }
  body.light .panel { box-shadow:0 10px 28px rgba(45,67,59,.07); }
  body.light .splash { background:#f7faf9; }
  body.light .hero-copy, body.light .legal-copy { color:#53625f; }
  body.light .brand-card { background:linear-gradient(90deg,rgba(40,118,92,.08),transparent); }
  body.light .badge.active { color:#63716f; }
  /* V3 motion layer: depth and movement without noisy neon effects. */
  body::before, body::after { content:""; position:fixed; pointer-events:none; z-index:-1; border-radius:50%; filter:blur(1px); opacity:.5; }
  body::before { width:42vw; height:42vw; right:-18vw; top:9vh; background:radial-gradient(circle,rgba(139,184,160,.12),transparent 68%); animation:drift-one 16s ease-in-out infinite alternate; }
  body::after { width:28vw; height:28vw; left:-14vw; bottom:4vh; background:radial-gradient(circle,rgba(201,168,106,.08),transparent 68%); animation:drift-two 19s ease-in-out infinite alternate; }
  @keyframes drift-one { from { transform:translate3d(0,0,0) scale(1); } to { transform:translate3d(-5vw,3vh,0) scale(1.12); } }
  @keyframes drift-two { from { transform:translate3d(0,0,0); } to { transform:translate3d(6vw,-4vh,0) scale(1.16); } }
  header { box-shadow:0 1px 0 rgba(255,255,255,.025),0 14px 40px rgba(0,0,0,.18); }
  nav button { position:relative; transition:color .2s,background .2s,transform .2s; }
  nav button::after { content:""; position:absolute; left:14px; right:14px; bottom:2px; height:2px; border-radius:3px; background:var(--accent); transform:scaleX(0); transition:transform .25s ease; }
  nav button.active::after { transform:scaleX(1); }
  nav button:hover { transform:translateY(-2px); }
  .chip { transition:transform .2s, border-color .2s, background .2s; }
  .chip:hover { transform:translateY(-2px); border-color:rgba(139,184,160,.45); }
  .panel { background:linear-gradient(145deg,rgba(255,255,255,.035),transparent 45%),var(--panel); backdrop-filter:blur(16px); animation:panel-in .5s ease both; }
  .page.active { animation:page-in .42s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes page-in { from { opacity:0; transform:translateY(10px) scale(.99); } to { opacity:1; transform:none; } }
  @keyframes panel-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
  .puca-hero { position:relative; overflow:hidden; padding:26px 4px 30px; }
  .puca-hero::after { content:""; position:absolute; width:180px; height:180px; right:8%; top:0; border:1px solid rgba(139,184,160,.18); border-radius:50%; box-shadow:0 0 0 18px rgba(139,184,160,.025),0 0 0 38px rgba(139,184,160,.018); animation:orbreathe 5s ease-in-out infinite; }
  @keyframes orbreathe { 50% { transform:scale(1.08); opacity:.65; } }
  .hero-title { animation:headline-in .7s .08s ease both; }
  .hero-copy { animation:headline-in .7s .16s ease both; }
  @keyframes headline-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
  .brand-card { position:relative; z-index:1; transition:transform .3s,box-shadow .3s; }
  .brand-card:hover { transform:translateY(-5px) rotate(.4deg); box-shadow:0 16px 35px rgba(0,0,0,.22); }
  .flow-stage { position:relative; overflow:hidden; }
  .flow-stage::before { content:""; position:absolute; inset:0; background:linear-gradient(90deg,transparent,rgba(139,184,160,.08),transparent); transform:translateX(-100%); }
  .flow-stage.active::before { animation:flow-sweep 1.4s ease-in-out infinite; }
  @keyframes flow-sweep { to { transform:translateX(100%); } }
  .flow-icon { transition:all .25s; }
  .complete .flow-icon { animation:check-pop .35s ease both; }
  @keyframes check-pop { from { transform:scale(.65); } 70% { transform:scale(1.15); } to { transform:scale(1); } }
  button.act { transition:transform .2s,box-shadow .2s,border-color .2s; }
  button.act:hover { box-shadow:0 7px 18px rgba(0,0,0,.16); }
  .presets button { transition:transform .2s,border-color .2s,color .2s,background .2s; }
  .presets button:hover { transform:translateY(-2px); background:rgba(139,184,160,.08); }
  .splash { background:radial-gradient(circle at 50% 42%,rgba(139,184,160,.10),transparent 23%),#090b10; }
  .splash-mark { animation:mark-pulse 2.3s ease-in-out infinite; text-shadow:0 0 28px rgba(139,184,160,.5); }
  .splash-line { animation:line-grow 1.1s .25s ease both; }
  @keyframes mark-pulse { 50% { transform:scale(1.08); opacity:.72; } }
  @keyframes line-grow { from { width:0; opacity:0; } to { width:160px; opacity:1; } }
  /* Neumorphic surfaces: soft paired shadows, with inset controls for depth. */
  .panel, .brand-card, .kbcard, .arch-node { box-shadow:-7px -7px 16px rgba(255,255,255,.025), 9px 10px 22px rgba(0,0,0,.24); }
  .panel h2 { text-shadow:0 1px 1px rgba(0,0,0,.25); }
  textarea, input, select, pre.details { box-shadow:inset 4px 4px 10px rgba(0,0,0,.22), inset -3px -3px 8px rgba(255,255,255,.025); }
  button.act, .chip, .presets button { box-shadow:-3px -3px 7px rgba(255,255,255,.025), 4px 5px 10px rgba(0,0,0,.22); }
  button.act:active, .presets button:active { transform:translateY(1px); box-shadow:inset 3px 3px 8px rgba(0,0,0,.2), inset -2px -2px 6px rgba(255,255,255,.02); }
  .flow-stage { box-shadow:inset 3px 3px 8px rgba(0,0,0,.18), inset -2px -2px 6px rgba(255,255,255,.018); }
  .flow-stage.complete { box-shadow:inset 3px 3px 8px rgba(0,0,0,.14), 0 0 16px rgba(114,208,154,.05); }
  body.light .panel, body.light .brand-card, body.light .kbcard, body.light .arch-node { box-shadow:-8px -8px 18px rgba(255,255,255,.92), 10px 12px 24px rgba(45,67,59,.12); }
  body.light textarea, body.light input, body.light select, body.light pre.details { box-shadow:inset 4px 4px 9px rgba(51,73,66,.08), inset -4px -4px 9px rgba(255,255,255,.95); }
  body.light button.act, body.light .chip, body.light .presets button { box-shadow:-4px -4px 9px rgba(255,255,255,.95), 5px 6px 12px rgba(45,67,59,.12); }
  body.light button.act:active, body.light .presets button:active { box-shadow:inset 3px 3px 8px rgba(45,67,59,.13), inset -2px -2px 6px rgba(255,255,255,.9); }
  body.light .flow-stage { box-shadow:inset 3px 3px 8px rgba(45,67,59,.06), inset -3px -3px 8px rgba(255,255,255,.9); }
  @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; } }
  .legal-copy h3 { color:var(--accent); font-weight:500; margin-top:26px; }
  @media (max-width:800px) { .architecture { grid-template-columns:1fr 1fr; } .puca-hero { display:block; } }
  :root {
    --bg: #0b0e14;
    --panel: #11151c;
    --panel2: #161b24;
    --border: #232a36;
    --text: #e6e9ef;
    --dim: #8a93a6;
    --accent: #4c8dff;
    --green: #3fb950;
    --red: #f85149;
    --amber: #d29922;
    --purple: #a371f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 10px; padding: 10px 18px;
    border-bottom: 1px solid var(--border); background: var(--panel);
    position: sticky; top: 0; z-index: 5; flex-wrap: wrap;
  }
  header h1 { font-size: 15px; margin: 0 10px 0 0; letter-spacing: .04em; }
  nav { display: flex; gap: 4px; margin-right: auto; }
  nav button {
    background: none; border: 1px solid transparent; color: var(--dim);
    padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: 13px;
  }
  nav button.active { background: var(--panel2); color: var(--text); border-color: var(--border); }
  nav button:hover { color: var(--text); }
  .chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
    border: 1px solid var(--border); border-radius: 999px; font-size: 12px;
    color: var(--dim); background: var(--panel2);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); }
  .dot.ok { background: var(--green); }
  .dot.bad { background: var(--red); }
  main { padding: 18px; max-width: 1500px; margin: 0 auto; }
  .page { display: none; }
  .page.active { display: block; }
  .grid { display: grid; grid-template-columns: minmax(380px, 5fr) minmax(420px, 6fr); gap: 16px; }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; margin-bottom: 16px;
  }
  .panel h2 {
    margin: 0 0 12px; font-size: 12px; text-transform: uppercase;
    letter-spacing: .08em; color: var(--dim);
  }
  textarea {
    width: 100%; min-height: 84px; resize: vertical; border-radius: 10px;
    border: 1px solid var(--border); background: var(--panel2); color: var(--text);
    padding: 12px; font: inherit;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
  button.act {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    padding: 8px 14px; border-radius: 9px; cursor: pointer; font-size: 13px;
  }
  button.act:hover { border-color: var(--accent); }
  button.act.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button.act.primary:disabled { opacity: .5; cursor: wait; }
  button.act.listening { background: var(--red); border-color: var(--red); color: #fff; animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: .55; } }
  .presets { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  .presets button {
    background: none; border: 1px dashed var(--border); color: var(--dim);
    padding: 4px 10px; border-radius: 999px; cursor: pointer; font-size: 12px;
  }
  .presets button:hover { color: var(--text); border-color: var(--accent); }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: .05em;
  }
  .badge.knowledge { background: rgba(76,141,255,.15); color: #79a9ff; border: 1px solid rgba(76,141,255,.4); }
  .badge.chat { background: rgba(163,113,247,.15); color: #c39bff; border: 1px solid rgba(163,113,247,.4); }
  .badge.person { background: rgba(217,179,255,.15); color: #d9b3ff; border: 1px solid rgba(217,179,255,.4); }
  .badge.unknown, .badge.error { background: rgba(248,81,73,.12); color: #ff8a84; border: 1px solid rgba(248,81,73,.4); }
  .response { margin-top: 14px; border: 1px solid var(--border); border-radius: 10px; padding: 12px; background: var(--panel2); }
  .reply { margin-top: 8px; font-size: 15px; line-height: 1.5; white-space: pre-wrap; }
  details > summary { cursor: pointer; color: var(--dim); font-size: 12px; margin-top: 8px; }
  pre.details {
    font-size: 11px; line-height: 1.5; overflow: auto; max-height: 340px;
    background: #0d1117; padding: 10px; border-radius: 8px; border: 1px solid var(--border);
  }
  .meta { color: var(--dim); font-size: 12px; }
  .ms { color: var(--accent); font-size: 11px; margin-left: 6px; }
  .feed, .kbfeed { max-height: 560px; overflow: auto; }
  .evt { padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 12px; }
  .evt.ok { border-left: 3px solid var(--green); }
  .evt.bad { border-left: 3px solid var(--red); }
  .evt .when { color: var(--dim); font-size: 11px; }
  .evt .stage { font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--dim); font-weight: 600; padding: 6px 8px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--panel); }
  td { padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  tr:hover td { background: rgba(76,141,255,.04); }
  .snip { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .rpass { color: var(--green); font-weight: 700; }
  .rfail { color: var(--red); font-weight: 700; }
  .runk { color: var(--amber); font-weight: 700; }
  .sumgrid { display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0; }
  .sumgrid > div { min-width: 90px; }
  .sumgrid b { font-size: 22px; display: block; font-weight: 700; }
  .sumgrid .good { color: var(--green); }
  .sumgrid .poor { color: var(--red); }
  .sumgrid .warn { color: var(--amber); }
  /* knowledge base page */
  .kbwrap { display: grid; grid-template-columns: 250px 1fr; gap: 16px; }
  @media (max-width: 800px) { .kbwrap { grid-template-columns: 1fr; } }
  .kbcat { display: flex; flex-direction: column; gap: 4px; }
  .kbcat button { text-align: left; }
  .kbcat button .cnt { float: right; color: var(--dim); }
  .kbcard {
    border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
    margin-bottom: 10px; background: var(--panel2);
  }
  .kbcard .kbhead { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
  .kbcard .kbid { color: var(--dim); font-size: 11px; }
  .kbcard .del { margin-left: auto; background: none; border: 1px solid var(--border); color: var(--dim); border-radius: 6px; cursor: pointer; font-size: 11px; padding: 2px 8px; }
  .kbcard .del:hover { color: var(--red); border-color: var(--red); }
  .kbline { display: flex; gap: 8px; padding: 2px 0; font-size: 13px; }
  .kbline .k { color: var(--dim); min-width: 110px; }
  .kbline .v { white-space: pre-wrap; word-break: break-word; }
  .badge.fact { background: rgba(63,185,80,.12); color: #6fdd8b; border: 1px solid rgba(63,185,80,.35); }
  .badge.calendar_event { background: rgba(76,141,255,.15); color: #79a9ff; border: 1px solid rgba(76,141,255,.4); }
  .badge.preference { background: rgba(210,153,34,.12); color: #e3b341; border: 1px solid rgba(210,153,34,.4); }
  .badge.key { background: rgba(255,122,214,.12); color: #ff9ad8; border: 1px solid rgba(255,122,214,.35); }
  .badge.active { background: transparent; color: var(--dim); border: 1px solid var(--border); }
  .badge.cancelled { background: rgba(248,81,73,.1); color: #ff8a84; border: 1px solid rgba(248,81,73,.3); }
  .foot { padding: 14px 18px; color: var(--dim); font-size: 12px; border-top: 1px solid var(--border); }
  .spinner { color: var(--accent); font-size: 12px; }
  select, input[type=number] {
    background: var(--panel2); border: 1px solid var(--border); color: var(--text);
    padding: 7px 10px; border-radius: 8px; font: inherit;
  }
  input[type=checkbox] { accent-color: var(--accent); }
</style>
</head>
<body>
<div class="splash" id="splash"><div class="splash-inner"><div class="splash-mark">◎</div><h1>Nix PUCA</h1><p>Casper V5 · personal user companion architecture</p><div class="splash-line"></div><span class="meta">initializing private workspace</span></div></div>
<header>
  <h1>&#9678; NIX</h1>
  <nav id="nav">
    <button data-page="console" class="active">Home</button>
    <button data-page="kb">Knowledge</button>
    <button data-page="memory">Memory</button>
    <button data-page="people">People</button>
    <button data-page="schedule">Events</button>
    <button data-page="architecture">Architecture</button>
    <button data-page="tests">Testing</button>
    <button data-page="privacy">Privacy</button>
    <button data-page="terms">Terms</button>
  </nav>
  <span class="chip" id="chip-knowledge"><span class="dot"></span>knowledge</span>
  <span class="chip" id="chip-actions"><span class="dot"></span>actions</span>
  <span class="chip" id="chip-ollama"><span class="dot"></span><span id="chip-model">chat backend</span></span>
  <span class="chip" id="chip-mode"><span class="dot ok"></span><span id="mode-text">mode</span></span>
  <span class="chip" id="chip-online"><span class="dot"></span><span id="online-text">checking system</span></span>
  <button class="act theme-toggle" id="theme-toggle" title="Toggle light and dark mode"><span class="theme-glyph">☾</span><span id="theme-label">dark</span></button>
</header>

<main>
  <!-- ============ PAGE: console ============ -->
  <section class="page active" id="page-console">
    <div class="puca-hero"><div><div class="eyebrow">Nix PUCA system · version 3</div><h2 class="hero-title">A quieter kind of intelligence.</h2><p class="hero-copy">Casper V5 is a private personal companion: grounded in your memory, attentive to context, and deliberate about what it stores.</p></div><div class="brand-card"><h3>Casper V5</h3><p>PUCA / local / warm / grounded</p></div></div>
    <div class="grid">
      <div class="panel">
        <h2>Request</h2>
        <textarea id="input" placeholder="Type or speak a request exactly as you'd say it to Nix... (Enter to send)"></textarea>
        <div class="row">
          <button class="act primary" id="send">Send &#9654;</button>
          <button class="act" id="dry">Classify only</button>
          <button class="act" id="run-actions" title="Fire due actions in mock mode">Fire due actions</button>
          <button class="act" id="mic" title="Speak (Google speech-to-text, Chrome/Edge)">&#127908; Speak</button>
          <label class="meta" style="display:inline-flex;align-items:center;gap:4px">
            <input type="checkbox" id="auto-send"> auto-send
          </label>
          <span class="spinner" id="busy"></span>
          <span class="meta" id="speech-status"></span>
        </div>
        <div class="presets" id="presets"></div>
        <div class="panel" style="margin-top:16px;padding:12px" id="pipeline-panel">
          <h2>Request path <span class="meta" id="pipeline-total"></span></h2>
          <div class="status-flow" id="status-flow"><div class="meta">Send a request to watch Casper's path through the system.</div></div>
        </div>
        <div class="response" id="response" hidden>
          <span class="badge" id="r-route"></span>
          <span class="meta" id="r-meta"></span>
          <div class="reply" id="r-reply"></div>
          <details>
            <summary>raw pipeline details</summary>
            <pre class="details" id="r-details"></pre>
          </details>
        </div>
      </div>
      <div class="panel">
        <h2>Background activity <span class="meta" id="feed-updated"></span></h2>
        <div class="row" style="margin:0 0 8px">
          <div class="tabs" id="tabs" style="display:flex;gap:4px;flex-wrap:wrap">
            <button class="act" data-tab="trace" style="padding:4px 10px">trace</button>
            <button class="act" data-tab="actions" style="padding:4px 10px">actions</button>
            <button class="act" data-tab="knowledge" style="padding:4px 10px">knowledge</button>
            <button class="act" data-tab="turns" style="padding:4px 10px">sessions</button>
          </div>
        </div>
        <div class="feed" id="feed"></div>
      </div>
    </div>
  </section>

  <!-- ============ PAGE: knowledge base ============ -->
  <section class="page" id="page-kb">
    <div class="kbwrap">
      <div class="panel" style="align-self:start">
        <h2>Categories</h2>
        <div class="kbcat" id="kb-cats"></div>
        <hr style="border:none;border-top:1px solid var(--border);margin:12px 0">
        <h2>People</h2>
        <div class="kbcat" id="kb-people" style="margin-top:6px"></div>
        <hr style="border:none;border-top:1px solid var(--border);margin:12px 0">
        <div class="meta" id="kb-stats"></div>
        <hr style="border:none;border-top:1px solid var(--border);margin:12px 0">
        <h2>Danger zone</h2>
        <button class="act" id="complete-reset" style="width:100%;color:var(--red);border-color:rgba(248,81,73,.45)">Complete reset</button>
        <div class="meta" style="margin-top:6px">Clears Knowledge, indexes, actions, and chat sessions. Models are kept.</div>
      </div>
      <div class="panel">
        <h2>Records <input id="kb-search" placeholder="filter..." style="float:right;width:180px;background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:4px 10px;border-radius:8px;font:inherit"></h2>
        <div class="kbfeed" id="kb-list"></div>
      </div>
      <div class="panel" style="grid-column:1/-1" id="kb-session-panel">
        <h2>Conversation sessions</h2>
        <div class="feed" id="kb-sessions" style="max-height:320px"></div>
      </div>
      <div class="panel" style="grid-column:1/-1">
        <h2>Actions ledger</h2>
        <div class="feed" id="kb-actions" style="max-height:320px"></div>
      </div>
    </div>
  </section>

  <!-- ============ PAGE: schedule ============ -->
  <section class="page" id="page-schedule">
    <div class="panel">
      <h2>Upcoming events &amp; reminders <span class="meta" id="sched-updated"></span></h2>
      <div class="feed" id="sched-upcoming"><div class='evt'>loading...</div></div>
    </div>
    <div class="panel">
      <h2>Standalone actions (no linked event)</h2>
      <div class="feed" id="sched-standalone" style="max-height:260px"></div>
    </div>
    <div class="panel">
      <h2>Past / cancelled</h2>
      <div class="feed" id="sched-past" style="max-height:260px"></div>
    </div>
  </section>

  <!-- ============ PAGE: testing ============ -->
  <section class="page" id="page-tests">
    <div class="panel">
      <h2>Realtime pipeline tests <span class="meta" id="test-status"></span></h2>
      <div class="row">
        <select id="test-batch"></select>
        <input id="test-limit" type="number" min="0" placeholder="limit (0 = all)" style="width:130px">
        <button class="act primary" id="test-run">Run tests &#9654;</button>
        <button class="act" id="test-filter">Failures only: off</button>
      </div>
      <div class="meta" style="margin:6px 0">Every prompt runs through the full pipeline (model gate, chat model, bridge scheduling) against sandboxed copies of your databases. Nothing touches live data. Expect ~1-3s per prompt.</div>
      <div id="test-summary"></div>
      <div class="feed" id="test-feed"><div class='evt'>pick a batch and run</div></div>
    </div>
  </section>

  <!-- ============ PAGE: memory ============ -->
  <section class="page" id="page-memory">
    <div class="puca-hero"><div><div class="eyebrow">Grounded context</div><h2 class="hero-title">Memory, rendered carefully.</h2><p class="hero-copy">This is the exact bounded memory block supplied to Casper when personal context is relevant.</p></div></div>
    <div class="panel"><h2>Current memory block</h2><pre class="details" id="memory-block" style="min-height:260px;max-height:none">loading...</pre></div>
  </section>

  <!-- ============ PAGE: people ============ -->
  <section class="page" id="page-people">
    <div class="puca-hero"><div><div class="eyebrow">Personal circle</div><h2 class="hero-title">People, as they are now.</h2><p class="hero-copy">Current states are temporal. Older states remain history, not the present.</p></div></div>
    <div class="panel"><h2>People and current states</h2><div id="people-page-list" class="kbfeed"><div class="evt">loading...</div></div></div>
  </section>

  <!-- ============ PAGE: architecture ============ -->
  <section class="page" id="page-architecture">
    <div class="puca-hero"><div><div class="eyebrow">Casper V5 · system map</div><h2 class="hero-title">Grounded before conversational.</h2><p class="hero-copy">Every request moves through an explicit path. Memory can inform Casper, but Knowledge never becomes the final speaker.</p></div></div>
    <div class="panel"><h2>Architecture and flow</h2><div class="architecture">
      <div class="arch-node"><b>01 · User query</b><small>Text or voice arrives at Core with session and location context.</small></div>
      <div class="arch-node"><b>02 · Core</b><small>Routes, protects identity and creative requests, and carries context.</small></div>
      <div class="arch-node"><b>03 · Nix_predictor</b><small>Qwen 2.5 0.5B handles ambiguous and multi-intent Knowledge planning.</small></div>
      <div class="arch-node"><b>04 · Knowledge</b><small>Temporal parser, symbolic validation, people states, facts, and events.</small></div>
      <div class="arch-node"><b>05 · Actions</b><small>Deterministic schedules, reminders, propagation, and session storage.</small></div>
      <div class="arch-node"><b>06 · Casper V5</b><small>Local QLoRA conversational model creates the final warm response.</small></div>
      <div class="arch-node"><b>07 · Privacy boundary</b><small>Personal records and models stay on the configured local machine.</small></div>
      <div class="arch-node"><b>08 · PUCA identity</b><small>Created and built by Sai Neela, living in Nix's PUCA system.</small></div>
    </div></div>
    <div class="panel brand-card"><h3>What makes V3 different</h3><p>It separates durable memory, temporal truth, deterministic actions, and humanlike conversation instead of asking one model to improvise everything.</p><p style="margin-top:8px"><a href="https://github.com/saineela/nix-puca" style="color:var(--accent)">Project source on GitHub ↗</a></p></div>
  </section>

  <!-- ============ PAGE: privacy ============ -->
  <section class="page" id="page-privacy"><div class="panel legal-copy"><div class="eyebrow">Nix PUCA V3</div><h2 class="hero-title">Privacy policy</h2><p>This development system is designed for local, personal use. Conversations, Knowledge records, people states, events, action logs, and model files are stored on the configured machine unless you deliberately connect an external service.</p><h3>What is stored</h3><p>Casper may store user-provided personal facts, temporal states, events, preferences, and conversation-session turns so it can provide continuity. The Knowledge and Actions services are separate so records can be inspected and cleared.</p><h3>Control</h3><p>You can inspect or delete records from the dashboard. The Complete reset control clears local Knowledge, derived indexes, actions, and session turns. It does not delete source code or model files.</p><h3>External services</h3><p>If web search, speech recognition, or another connector is enabled, that connector may process the specific request it receives. Review its configuration before enabling it.</p></div></section>

  <!-- ============ PAGE: terms ============ -->
  <section class="page" id="page-terms"><div class="panel legal-copy"><div class="eyebrow">Nix PUCA V3</div><h2 class="hero-title">Terms of use</h2><p>Casper is experimental personal software, not a medical, legal, financial, emergency, or safety-critical service. Verify important dates, reminders, health information, and decisions independently.</p><h3>Personal responsibility</h3><p>You are responsible for the accuracy of stored information, connected actions, credentials, and any reminders or schedules created through the system.</p><h3>Local development software</h3><p>This project is provided for development and personal experimentation. Availability, model behavior, generated responses, and connector behavior are not guaranteed.</p><h3>Respect and safety</h3><p>Do not use the system to violate another person’s privacy, automate harmful actions, or expose credentials and sensitive data through an untrusted network.</p></div></section>
</main>

<div class="foot" id="foot"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const THEME_KEY = "nix-puca-theme";
function applyTheme(theme) {
  const light = theme === "light";
  document.body.classList.toggle("light", light);
  const glyph = $("theme-toggle")?.querySelector(".theme-glyph");
  if (glyph) glyph.textContent = light ? "☀" : "☾";
  if ($("theme-label")) $("theme-label").textContent = light ? "light" : "dark";
  localStorage.setItem(THEME_KEY, light ? "light" : "dark");
}
applyTheme(localStorage.getItem(THEME_KEY) || "dark");
$("theme-toggle")?.addEventListener("click", () => {
  applyTheme(document.body.classList.contains("light") ? "dark" : "light");
});
const CHAT_SESSION_ID = (() => {
  const key = "casper-dashboard-session";
  let value = window.localStorage.getItem(key);
  if (!value) {
    value = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : String(Date.now()) + "-" + Math.random().toString(16).slice(2);
    window.localStorage.setItem(key, value);
  }
  return value;
})();

function esc(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const raw = await r.text();
  let data;
  try {
    data = JSON.parse(raw);
  } catch (_) {
    throw new Error(`API ${r.status}: expected JSON but received ${raw.slice(0, 120)}`);
  }
  if (!r.ok) throw new Error(data.error || `API request failed (${r.status})`);
  return data;
}

// ---------------- navigation ----------------
let page = "console";
$("nav").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  page = b.dataset.page;
  document.querySelectorAll("nav button").forEach((x) =>
    x.classList.toggle("active", x === b));
  document.querySelectorAll(".page").forEach((s) =>
    s.classList.toggle("active", s.id === "page-" + page));
  if (page === "kb") refreshKb();
  if (page === "memory") refreshMemory();
  if (page === "people") refreshPeoplePage();
  if (page === "schedule") refreshSchedule();
};

setTimeout(() => $("splash")?.classList.add("hide"), 1100);

// ---------------- health ----------------
function dot(ok) { return ok ? "ok" : "bad"; }
function setChip(id, ok) {
  $(id).querySelector(".dot").className = "dot " + dot(ok);
}
let LAST_HEALTH = null;
async function refreshHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    LAST_HEALTH = h;
    setChip("chip-knowledge", h.knowledge.ok);
    setChip("chip-actions", h.actions.ok);
    const o = h.casper?.backend === "tabby" ? h.tabby : h.ollama;
    setChip("chip-ollama", o.ok && o.model_loaded);
    $("chip-model").textContent = o.model + (o.model_loaded ? "" : " (missing)");
    $("mode-text").textContent = h.embedded_services > 0
      ? `embedded APIs (${h.embedded_services})` : "live APIs";
    const online = Boolean(h.knowledge?.ok && h.actions?.ok && h.casper?.active);
    setChip("chip-online", online);
    $("online-text").textContent = online ? "system online" : "system degraded";
    const endpoint = o.host || o.endpoint || "local";
    $("foot").textContent =
      `timezone ${h.timezone} | ${h.casper?.backend || "chat"} ${endpoint} | models: ${(o.models || []).join(", ") || "none"} | github.com/saineela/nix-puca`;
  } catch (e) { /* stale */ }
}

// ================= CONSOLE PAGE =================
const PRESETS = [
  ["chat / world", "tell me one fun fact about octopuses, keep it short"],
  ["internet", "what's the capital of australia?"],
  ["store fact", "remember that my garage door code is 4821"],
  ["recall fact", "what is my garage door code?"],
  ["schedule", "remind me to stretch tomorrow at 9am"],
  ["set alarm", "set an alarm for 7am"],
  ["browse", "what's on my schedule this week?"],
  ["cancel", "cancel my stretch reminder"],
];

function renderPresets() {
  const holder = $("presets");
  holder.innerHTML = "";
  for (const [label, text] of PRESETS) {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = text;
    b.onclick = () => { $("input").value = text; send(false); };
    holder.appendChild(b);
  }
}

let pipelineTimer = null;
const PIPELINE_DEFAULTS = [
  ["received", "Received user query"],
  ["route", "Core routing"],
  ["predictor", "Nix_predictor · Qwen 2.5 0.5B"],
  ["knowledge", "Knowledge retrieval and validation"],
  ["final", "Luna V6 final response"],
];
function renderPipeline(stages, live = false) {
  const holder = $("status-flow");
  if (!holder) return;
  holder.innerHTML = stages.map((s, i) => {
    const status = s.status || "pending";
    const icon = status === "complete" ? "✓" : status === "active" ? "·" : "";
    const time = s.ms ? `${Number(s.ms).toFixed(1)} ms` : status === "active" ? "running" : status;
    return `<div class="flow-stage ${status}"><span class="flow-icon">${icon}</span><span class="flow-name">${esc(s.label)}</span><span class="flow-time">${esc(time)}</span></div>`;
  }).join("");
  const done = stages.filter(s => s.status === "complete").reduce((n, s) => n + Number(s.ms || 0), 0);
  $("pipeline-total").textContent = live ? "processing" : `${done.toFixed(1)} ms total`;
}
function beginPipeline() {
  if (pipelineTimer) clearInterval(pipelineTimer);
  let current = 0;
  const started = PIPELINE_DEFAULTS.map(([id, label]) => ({id, label, status:"pending", ms:0}));
  started[0].status = "complete";
  renderPipeline(started, true);
  pipelineTimer = setInterval(() => {
    if (current < started.length - 1) {
      current += 1;
      started.forEach((s, i) => s.status = i < current ? "complete" : i === current ? "active" : "pending");
      renderPipeline(started, true);
    }
  }, 420);
}
function finishPipeline(result) {
  if (pipelineTimer) clearInterval(pipelineTimer);
  renderPipeline(result.pipeline || PIPELINE_DEFAULTS.map(([id,label]) => ({id,label,status:"complete",ms:0})), false);
}

async function send(dry) {
  const text = $("input").value.trim();
  if (!text) return;
  if (!dry) beginPipeline();
  $("busy").textContent = "working...";
  $("send").disabled = true;
  try {
    const result = dry ? await post("/api/classify", {text})
                       : await post("/api/send", {
                           text,
                           location: "console",
                           conversation_id: CHAT_SESSION_ID,
                         });
    show(result, dry);
    if (!dry) finishPipeline(result);
  } catch (e) {
    const failed = {ok: false, error: String(e)};
    show(failed, dry);
    if (!dry) finishPipeline(failed);
  } finally {
    $("busy").textContent = "";
    $("send").disabled = false;
    refreshFeed();
    if (page === "kb") refreshKb();
  }
}

function show(result, dry) {
  const box = $("response");
  box.hidden = false;
  const badge = $("r-route");
  const final = dry ? result.final : result.route;
  badge.textContent = dry ? (final || "?").toUpperCase() + " (dry run)"
                          : (final || "?").toUpperCase();
  badge.className = "badge " + (final === "knowledge" ? "knowledge"
                       : final === "chat" ? "chat" : "unknown");
  const rule = dry
    ? (result.deterministic?.features?.rule || "rules abstained")
    : (result.rule || "-");
  const ms = result.latency_ms != null ? result.latency_ms + " ms" : "";
  $("r-meta").textContent = `rule: ${rule}  ${ms}`;
  $("r-reply").textContent = result.error
    ? "ERROR: " + result.error
    : (result.reply || result.final || "(no reply)");
  $("r-details").textContent = JSON.stringify(result, null, 2);
}

// speech to text (Google via Web Speech API)
let recog = null, listening = false, finalHold = "";
function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  recog = new SR();
  recog.lang = "en-US";
  recog.continuous = true;
  recog.interimResults = true;
  recog.onstart = () => {
    listening = true;
    $("mic").classList.add("listening");
    $("speech-status").textContent = "listening...";
  };
  recog.onend = () => {
    listening = false;
    $("mic").classList.remove("listening");
    $("speech-status").textContent = "";
  };
  recog.onerror = (e) => {
    $("speech-status").textContent =
      e.error === "not-allowed" ? "mic blocked - allow access" : "speech error: " + e.error;
  };
  recog.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalHold += t;
      else interim += t;
    }
    $("input").value = (finalHold + interim).trim();
    if (finalHold.trim() && !interim.trim()) {
      if ($("auto-send").checked) send(false);
      finalHold = "";
    }
  };
}
$("mic").onclick = () => {
  if (!recog) initSpeech();
  if (!recog) { $("speech-status").textContent = "speech recognition unavailable (use Chrome/Edge)"; return; }
  if (listening) { recog.stop(); return; }
  finalHold = "";
  try { recog.start(); } catch (e) { /* already started */ }
};
initSpeech();

// background feed
let feedTab = "trace";
const COLS = {
  actions: [["id", "id"], ["action_type", "type"], ["status", "status"],
            ["scheduled_for", "fires"], ["source_record_id", "src"]],
  knowledge: [["id", "id"], ["knowledge_type", "type"], ["status", "status"],
              ["data", "data"], ["created_at", "created"]],
  turns: [["id", "id"], ["role", "role"], ["content", "content"],
          ["session_tag", "session"], ["created_at", "at"]],
};
function tableHtml(rows, cols) {
  if (!rows.length) return "<div class='evt'>nothing yet</div>";
  if (rows.length && rows[0].error)
    return `<div class="evt bad">${esc(rows[0].error)}</div>`;
  const head = cols.map(c => `<th>${c[1]}</th>`).join("");
  const body = rows.map(r => "<tr>" + cols.map(([key]) => {
    let v = r[key];
    if (v != null && typeof v === "object") v = JSON.stringify(v);
    return `<td class="snip" title="${esc(v)}">${esc(v)}</td>`;
  }).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
function evtHtml(e) {
  const cls = e.ok ? "ok" : "bad";
  const ms = e.ms != null ? `<span class="ms">${e.ms} ms</span>` : "";
  return `<div class="evt ${cls}">${ms}
    <span class="when">${esc(e.ts.slice(11, 19))}</span>
    <span class="stage"> ${esc(e.stage)}</span><br>
    <span class="meta">${esc(e.detail)}</span></div>`;
}
function renderFeed(data) {
  const feed = $("feed");
  if (feedTab === "trace") {
    feed.innerHTML = data.trace.length
      ? data.trace.map(evtHtml).join("")
      : "<div class='evt'>no activity yet - send a request</div>";
  } else {
    feed.innerHTML = tableHtml(data[feedTab] || [], COLS[feedTab]);
  }
  $("feed-updated").textContent = "updated " + new Date().toLocaleTimeString();
}
async function refreshFeed() {
  try { renderFeed(await (await fetch("/api/feed")).json()); }
  catch (e) { /* transient */ }
}
$("tabs").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  feedTab = b.dataset.tab;
  document.querySelectorAll("#tabs button").forEach(
    (x) => x.classList.toggle("active", x === b));
  refreshFeed();
};

$("send").onclick = () => send(false);
$("dry").onclick = () => send(true);
$("run-actions").onclick = async () => {
  $("busy").textContent = "firing...";
  try { await post("/api/run", {mock: true}); }
  finally { $("busy").textContent = ""; refreshFeed(); }
};
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(false); }
});

// ================= KNOWLEDGE BASE PAGE =================
let KB = [];
let kbFilter = "all";
let kbQuery = "";

// ---- People sub-tab: members + their current states ----
let PEOPLE = [];

function personCard(p) {
  const who = p.name || p.subject || "someone";
  const mood = p.valence === "good" ? "\u{1F60A}" : p.valence === "bad" ? "\u{1F915}" : "\u{1F610}";
  return `<div class="kbcard" data-id="${p.id}">
    <div class="kbhead">
      <span class="badge person">person</span>
      <span class="kbid">${esc(who)} ${mood}</span>
      ${p.role ? `<span class="meta">${esc(p.role)}</span>` : ""}
    </div>
    <div class="kbline"><span class="k">current state</span><span class="v">${esc(p.state || "?")} (${esc(p.valence || "neutral")})</span></div>
    <div class="kbline"><span class="k">learned</span><span class="v">${esc(String(p.value || ""))}</span></div>
    <div class="kbid" style="margin-top:6px">updated ${esc(String(p.updated_at || "").replace("T", " ").slice(0, 16))}</div>
  </div>`;
}

function renderPeople() {
  const holder = $("kb-people");
  if (!holder) return;
  if (!PEOPLE.length) {
    holder.innerHTML = "<div class='meta'>no people yet</div>";
    return;
  }
  const view = (kbFilter === "all" || kbFilter === "person") ? PEOPLE : [];
  holder.innerHTML = view.length
    ? view.map(personCard).join("")
    : "<div class='meta'>no people</div>";
}

async function refreshPeople() {
  try {
    const d = await (await fetch("/api/people")).json();
    PEOPLE = d.people || [];
    renderPeople();
  } catch (e) { /* keep last */ }
}
async function refreshPeoplePage() {
  await refreshPeople();
  const holder = $("people-page-list");
  if (holder) holder.innerHTML = PEOPLE.length ? PEOPLE.map(personCard).join("") : "<div class='evt'>No current people states stored.</div>";
}
async function refreshMemory() {
  try {
    const d = await (await fetch("/api/memory")).json();
    $("memory-block").textContent = d.memory || "No stored memory context.";
  } catch (e) { $("memory-block").textContent = "Memory service unavailable."; }
}

function kbPretty(rec) {
  const d = rec.data || {};
  const lines = [];
  if (rec.type === "calendar_event") {
    if (d.title) lines.push(["title", d.title]);
    if (d.start) lines.push(["starts", String(d.start).replace("T", " ").slice(0, 16)]);
    if (d.end && d.end !== d.start) lines.push(["ends", String(d.end).replace("T", " ").slice(0, 16)]);
    if (d.all_day) lines.push(["all day", "yes"]);
    if (d.temporal_expression) lines.push(["from", d.temporal_expression]);
    if (d.recurrence) lines.push(["repeats", d.recurrence]);
    if (d.location) lines.push(["where", d.location]);
    if (d.notes) lines.push(["notes", d.notes]);
    for (const [k, v] of Object.entries(d)) {
      if (!["title","start","end","all_day","temporal_expression","recurrence","location","notes"].includes(k) && v != null && v !== "")
        lines.push([k.replaceAll("_", " "), String(v)]);
    }
  } else if (rec.type === "preference") {
    if (d.subject) lines.push(["about", d.subject]);
    if (d.stance) lines.push(["stance", d.stance]);
    if (d.strength) lines.push(["strength", d.strength]);
    if (d.value) lines.push(["detail", d.value]);
    for (const [k, v] of Object.entries(d)) {
      if (!["subject","stance","strength","value"].includes(k) && v != null && v !== "")
        lines.push([k.replaceAll("_", " "), String(v)]);
    }
  } else if (rec.type === "person") {
    if (d.value) lines.push(["state", d.value]);
    if (d.name || d.subject) lines.push(["who", d.name || d.subject]);
    if (d.role) lines.push(["relation", d.role]);
    if (d.state) lines.push(["current state", `${d.state} (${d.valence || "neutral"})`]);
  } else if (rec.type === "key") {
    if (d.value) lines.push(["key", d.value]);
    if (d.subject) lines.push(["about", d.subject]);
    if (d.predicate) lines.push(["relation", d.predicate.replaceAll("_", " ")]);
    const src = (d.context || {}).source;
    if (src) lines.push(["learned from", `"${src}"`]);
  } else {
    if (d.value) lines.push(["fact", d.value]);
    for (const [k, v] of Object.entries(d)) {
      if (k !== "value" && v != null && v !== "")
        lines.push([k.replaceAll("_", " "), String(v)]);
    }
  }
  if (!lines.length) lines.push(["data", JSON.stringify(d)]);
  return lines;
}

function kbCard(rec) {
  const lines = kbPretty(rec);
  return `<div class="kbcard" data-id="${rec.id}">
    <div class="kbhead">
      <span class="badge ${esc(rec.type)}">${esc(rec.type)}</span>
      ${rec.data && rec.data.sensitive ? '<span class="badge cancelled">sensitive</span>' : ""}
      <span class="badge ${esc(rec.status)}">${esc(rec.status)}</span>
      <span class="kbid">#${rec.id}</span>
      <button class="del" title="delete record">delete</button>
    </div>
    ${lines.map(([k, v]) => `<div class="kbline"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("")}
    <div class="kbid" style="margin-top:6px">stored ${esc(String(rec.created_at || "").replace("T", " ").slice(0, 16))}</div>
  </div>`;
}

function renderKb() {
  const cats = {};
  for (const r of KB) cats[r.type] = (cats[r.type] || 0) + 1;
  cats["person"] = Math.max(cats["person"] || 0, PEOPLE.length);
  const order = ["person", "key", "calendar_event", "fact", "preference"];
  const names = {person: "People (current states)", key: "Keys (auto-learned)", calendar_event: "Calendar events", fact: "Facts", preference: "Preferences"};
  const catBtns = ["all", ...order.filter((t) => cats[t])].map((t) => {
    const n = t === "all" ? KB.length : cats[t];
    const label = t === "all" ? "All records" : (names[t] || t);
    return `<button class="act ${kbFilter === t ? "primary" : ""}" data-cat="${t}" style="width:100%">${label}<span class="cnt">${n}</span></button>`;
  });
  $("kb-cats").innerHTML = catBtns.join("");
  const statusCounts = {};
  for (const r of KB) statusCounts[r.status] = (statusCounts[r.status] || 0) + 1;
  $("kb-stats").innerHTML =
    Object.entries(statusCounts).map(([s, n]) => `${n} ${s}`).join(" &middot; ")
    + `<br>last updated ${new Date().toLocaleTimeString()}`;

  const q = kbQuery.toLowerCase();
  const rows = KB.filter((r) =>
    (kbFilter === "all" || r.type === kbFilter)
    && (!q || JSON.stringify(r.data).toLowerCase().includes(q)));
  $("kb-list").innerHTML = rows.length
    ? rows.map(kbCard).join("")
    : "<div class='evt'>no records match</div>";
}

function renderSessionGroups(sessions) {
  if (!sessions.length) return "<div class='evt'>no chat sessions yet</div>";
  return sessions.map((session) => {
    const turns = (session.turns || []).slice().reverse();
    const preview = turns.slice(0, 12).map((turn) =>
      `<div class="evt"><b>${esc(turn.role)}</b> ${esc(turn.content)}<br><span class="meta">${esc(turn.created_at || "")}</span></div>`
    ).join("");
    return `<details class="kbcard" open>
      <summary><b>${esc(session.session_tag)}</b> · ${session.turn_count} turns · ${esc(session.first_turn || "")} → ${esc(session.last_turn || "")}</summary>
      <div class="feed" style="max-height:260px;margin-top:8px">${preview}</div>
    </details>`;
  }).join("");
}

async function refreshKb() {
  try {
    const d = await (await fetch("/api/kb")).json();
    KB = d.records || [];
    renderKb();
    refreshPeople();
    $("kb-sessions").innerHTML = "<div class='evt'>loading...</div>";
    const sessions = await (await fetch("/api/sessions")).json();
    $("kb-sessions").innerHTML = renderSessionGroups(sessions.sessions || []);
    const feed = await (await fetch("/api/feed")).json();
    $("kb-actions").innerHTML = tableHtml(feed.actions || [], COLS.actions);
  } catch (e) {
    $("kb-list").innerHTML = `<div class="evt bad">${esc(String(e))}</div>`;
  }
}

$("kb-cats").onclick = (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  kbFilter = b.dataset.cat;
  renderKb();
};
$("kb-search").addEventListener("input", () => {
  kbQuery = $("kb-search").value.trim();
  renderKb();
});
$("complete-reset").onclick = async () => {
  const warning = "This permanently clears all stored Knowledge, semantic indexes, scheduled actions, and conversation sessions. Model files and code are not affected. Continue?";
  if (!confirm(warning)) return;
  if (prompt("Type RESET ALL to confirm:") !== "RESET ALL") return;
  const button = $("complete-reset");
  button.disabled = true;
  button.textContent = "Resetting...";
  try {
    const d = await post("/api/reset", {confirm: "RESET_ALL"});
    if (!d.ok) throw new Error(d.error || "reset failed");
    alert("Complete reset finished. Casper will now use an empty Knowledge base.");
    await refreshKb();
    await refreshFeed();
    await refreshSchedule();
  } catch (e) {
    alert("Reset failed: " + e);
  } finally {
    button.disabled = false;
    button.textContent = "Complete reset";
  }
};

$("kb-list").onclick = async (e) => {
  const btn = e.target.closest("button.del");
  if (!btn) return;
  const card = btn.closest(".kbcard");
  const id = card.dataset.id;
  if (!confirm("Delete record #" + id + " from the knowledge base?")) return;
  const d = await post("/api/kb/delete", {id: parseInt(id, 10)});
  if (d.ok) { card.remove(); refreshKb(); }
};

// ================= SCHEDULE PAGE =================
let SCHED = null;

function fmtWhen(iso) {
  if (!iso) return "?";
  const s = String(iso).replace("T", " ");
  return s.slice(0, 16) + (s.length > 16 ? "" : "");
}

function actionLine(a) {
  const st = a.status || "?";
  const cls = st === "fired" ? "rpass" : st === "cancelled" ? "rfail" : "runk";
  let line = `<div class="kbline"><span class="k">${esc(a.type || "action")} <span class="${cls}">${esc(st)}</span></span><span class="v">fires ${esc(fmtWhen(a.fires))}`;
  if (a.fired_at) line += ` | fired ${esc(fmtWhen(a.fired_at))}`;
  if (a.recurrence && a.recurrence !== "none") line += ` | repeats ${esc(a.recurrence)}`;
  if (a.recurrence_end) line += ` | until ${esc(fmtWhen(a.recurrence_end))}`;
  if (a.last_error) line += ` | <span class="rfail">error: ${esc(a.last_error)}</span>`;
  line += `</span></div>`;
  return line;
}

function eventCard(e) {
  const lines = [
    ["starts", fmtWhen(e.start) + (e.all_day ? " (all day)" : "")],
  ];
  if (e.end && e.end !== e.start) lines.push(["ends", fmtWhen(e.end)]);
  if (e.temporal_expression) lines.push(["from phrase", `"${e.temporal_expression}"`]);
  if (e.recurrence) lines.push(["repeats", e.recurrence]);
  for (const [k, v] of Object.entries(e.extra || {}))
    lines.push([k.replaceAll("_", " "), String(v)]);
  lines.push(["record", `#${e.id} (${e.status})`]);
  const acts = (e.actions || []).map(actionLine).join("");
  return `<div class="kbcard">
    <div class="kbhead">
      <span class="badge calendar_event">event</span>
      <b>${esc(e.title)}</b>
      ${e.is_past ? '<span class="badge cancelled">past</span>' : ""}
    </div>
    ${lines.map(([k, v]) => `<div class="kbline"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("")}
    ${acts ? `<div style="margin-top:6px" class="meta">linked actions:</div>${acts}` : '<div class="meta" style="margin-top:6px">no linked actions</div>'}
  </div>`;
}

function renderSchedule() {
  if (!SCHED) return;
  const up = SCHED.upcoming || [];
  $("sched-upcoming").innerHTML = up.length
    ? up.map(eventCard).join("")
    : "<div class='evt'>nothing scheduled - create something from the console</div>";
  const sa = SCHED.standalone_actions || [];
  $("sched-standalone").innerHTML = sa.length
    ? tableHtml(sa, [["id", "id"], ["type", "type"], ["status", "status"],
                     ["fires", "fires"], ["source", "source"], ["payload", "payload"]])
    : "<div class='evt'>none</div>";
  const past = SCHED.past || [];
  $("sched-past").innerHTML = past.length
    ? past.map(eventCard).join("")
    : "<div class='evt'>none</div>";
  $("sched-updated").textContent = "updated " + new Date().toLocaleTimeString();
}

async function refreshSchedule() {
  try {
    SCHED = await (await fetch("/api/schedule")).json();
    renderSchedule();
  } catch (e) {
    $("sched-upcoming").innerHTML = `<div class="evt bad">${esc(String(e))}</div>`;
  }
}

// ================= TESTING PAGE =================
let lastResults = null, failuresOnly = false, running = false;

async function loadTestBatches() {
  try {
    const d = await (await fetch("/api/tests/info")).json();
    const sel = $("test-batch");
    sel.innerHTML = "";
    for (const b of d.batches || []) {
      const o = document.createElement("option");
      o.value = b.id;
      o.textContent = `${b.label} - ${b.size} prompts`;
      sel.appendChild(o);
    }
  } catch (e) { /* empty */ }
}

async function runTests() {
  if (running) return;
  const batch = $("test-batch").value;
  if (!batch) return;
  const limit = parseInt($("test-limit").value || "0", 10) || 0;
  running = true;
  $("test-run").disabled = true;
  $("test-status").textContent = "running full pipeline...";
  $("test-feed").innerHTML = "<div class='evt'>running... (prompts execute for real; this takes a moment)</div>";
  const t0 = performance.now();
  try {
    const d = await post("/api/tests/run", {batch, limit});
    lastResults = d;
    renderSummary(d, Math.round(performance.now() - t0));
    renderResults();
    $("test-status").textContent =
      `done in ${Math.round(performance.now() - t0) / 1000}s - ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    $("test-status").textContent = "error: " + e;
  } finally {
    running = false;
    $("test-run").disabled = false;
  }
}

function renderSummary(d) {
  const cls = d.accuracy >= 99 ? "good" : d.accuracy >= 90 ? "warn" : "poor";
  $("test-summary").innerHTML = `
    <div class="sumgrid">
      <div><b class="${cls}">${d.accuracy}%</b><span class="meta">accuracy - ${d.passed}/${d.ran} passed</span></div>
      <div><b class="${d.misroutes === 0 ? "good" : "poor"}">${d.misroutes}</b><span class="meta">misroutes</span></div>
      <div><b class="${d.errors ? "poor" : "good"}">${d.errors}</b><span class="meta">pipeline errors</span></div>
      <div><b>${d.avg_ms}</b><span class="meta">avg ms / prompt (max ${d.max_ms})</span></div>
      <div><b>${(d.total_ms / 1000).toFixed(1)}s</b><span class="meta">total run time</span></div>
      ${Object.entries(d.by_route || {}).map(([r, v]) =>
        `<div><b>${v.count}</b><span class="meta">${r} (avg ${v.avg_ms} ms)</span></div>`).join("")}
    </div>`;
}

function renderResults() {
  if (!lastResults) return;
  const rows = failuresOnly
    ? (lastResults.results || []).filter((r) => !r.pass)
    : (lastResults.results || []);
  const feed = $("test-feed");
  if (!rows.length) {
    feed.innerHTML = failuresOnly
      ? "<div class='evt rpass'>zero failures</div>"
      : "<div class='evt'>no rows</div>";
    return;
  }
  const lim = 300;
  const shown = rows.slice(0, lim);
  feed.innerHTML = `<table><thead><tr><th>#</th><th>result</th><th>expected</th><th>got</th><th>rule</th><th>time</th><th>prompt</th><th>nix replied</th></tr></thead><tbody>` +
    shown.map((r, i) => {
      const mark = r.pass ? "<span class='rpass'>PASS</span>"
        : r.got === "error" ? "<span class='rfail'>ERR</span>"
        : r.got === "unknown" ? "<span class='runk'>MODEL</span>"
        : "<span class='rfail'>FAIL</span>";
      return `<tr><td>${i + 1}</td><td>${mark}</td><td>${esc(r.expected)}</td><td>${esc(r.got)}</td><td>${esc(r.rule || "-")}</td><td>${r.ms} ms</td><td class="snip" title="${esc(r.text)}">${esc(r.text)}</td><td class="snip" title="${esc(r.reply)}">${esc(r.reply)}</td></tr>`;
    }).join("") + `</tbody></table>` +
    (rows.length > lim ? `<div class="evt">... ${rows.length - lim} more rows</div>` : "");
}

$("test-run").onclick = runTests;
$("test-filter").onclick = () => {
  failuresOnly = !failuresOnly;
  $("test-filter").textContent = "Failures only: " + (failuresOnly ? "on" : "off");
  renderResults();
};

// ---------------- boot ----------------
renderPresets();
loadTestBatches();
refreshHealth();
refreshFeed();
refreshSchedule();
setInterval(() => refreshFeed(), 2000);
setInterval(() => refreshHealth(), 10000);
setInterval(() => { if (page === "schedule") refreshSchedule(); }, 5000);
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------
# Dashboard asset
# ----------------------------------------------------------------------


def _application_route_path(path: str) -> str:
    """Strip an optional hosting prefix before API-route dispatch."""
    segments = [segment for segment in path.split("/") if segment]
    for index, segment in enumerate(segments):
        if index > 0 and segment in {"api", "v1"}:
            return "/" + "/".join(segments[index:])
    return path


def _dashboard_markup() -> str:
    """Serve the maintainable connected dashboard asset."""
    try:
        with open(DASHBOARD_PATH, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        # Keep the legacy inline dashboard as an emergency fallback if an
        # incomplete deployment omitted the asset.
        return PAGE_V2


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def build_app(host: str = "127.0.0.1", port: int | None = None):
    """Create the console server (and resolve/embed sibling services)."""
    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]

    get_brain()  # resolve/embed services before serving

    server = ThreadingHTTPServer((host, port), Handler)
    return server, int(server.server_address[1])


def _lan_ip() -> str:
    """Best-effort address of this machine on the local network."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def main() -> int:
    host = os.environ.get("NIX_CONSOLE_HOST", "0.0.0.0")
    port_env = os.environ.get("NIX_CONSOLE_PORT")
    port = int(port_env) if port_env else None

    server, port = build_app(host=host, port=port)

    lan = host if host not in ("0.0.0.0", "::") else _lan_ip()
    print()
    print("  NIX TEST CONSOLE")
    print(f"  local : http://127.0.0.1:{port}")
    print(f"  LAN   : http://{lan}:{port}   (from any device on your network)")
    print(f"  ports : {port} only - Knowledge/Actions run in-process")
    print(f"  chat backend : {CASPER_BACKEND} ({OLLAMA_MODEL} / {TABBY_MODEL})")
    print(f"  timezone   : {TIMEZONE}")
    print("  Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nconsole stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
