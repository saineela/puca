"""
Nix Test Console.

One command, one browser tab: send requests to the Nix brain exactly
like the Orange Pi gateway would, and watch everything that happens in
the background - routing decisions, knowledge tool execution, action
scheduling, bridge propagation, session logging, and the chat model.

    ~/nix_knowledge/.venv/bin/python ~/nix_core/console.py

- Binds to a random free port (override: NIX_CONSOLE_PORT / HOST) and
  prints the URL.
- Uses the real nix_knowledge / nix_actions APIs when they are already
  running; otherwise boots them in-thread so this single process is a
  complete test rig.
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

import json
import os
import random
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# ----------------------------------------------------------------------
# Paths so the sibling packages import cleanly from anywhere.
# ----------------------------------------------------------------------

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CORE_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_knowledge"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_knowledge", "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_actions"))
sys.path.insert(0, os.path.join(REPO_ROOT, "nix_actions", "scripts"))

import requests  # noqa: E402

from brain import Brain, KnowledgeClient, ActionsClient  # noqa: E402
from config import (  # noqa: E402
    KNOWLEDGE_API_URL,
    ACTIONS_API_URL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TIMEZONE,
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


class TracedActions(ActionsClient):
    def log_turn(self, **kwargs):
        with _Timed(
            "actions /log",
            f"{kwargs.get('role')}: {str(kwargs.get('content'))[:80]}",
        ):
            return super().log_turn(**kwargs)

    def context(self, limit: int = 20):
        with _Timed("actions /context", f"session window ({limit})"):
            return super().context(limit)


# ----------------------------------------------------------------------
# Service resolution: reuse running APIs, otherwise embed them.
# ----------------------------------------------------------------------

_embedded: list[ThreadingHTTPServer] = []


def _reachable(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=1.5).ok
    except Exception:
        return False


def _embed(module_name: str) -> str:
    """Boot one of the sibling API modules in-thread on a free port."""
    module = __import__(module_name)
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    threading.Thread(
        target=server.serve_forever, daemon=True
    ).start()
    _embedded.append(server)
    return f"http://127.0.0.1:{server.server_address[1]}"


def resolve_services() -> tuple[str, str, list[str]]:
    """
    Returns (knowledge_url, actions_url, notes).
    """
    notes: list[str] = []

    knowledge_url = KNOWLEDGE_API_URL
    if _reachable(knowledge_url):
        notes.append(f"knowledge API found running at {knowledge_url}")
    else:
        knowledge_url = _embed("knowledge_api")
        notes.append(f"knowledge API embedded at {knowledge_url}")

    # The actions API enriches sessions through the knowledge API;
    # keep it pointed at whichever knowledge URL we settled on.
    os.environ["NIX_KNOWLEDGE_API_URL"] = knowledge_url

    actions_url = ACTIONS_API_URL
    if _reachable(actions_url):
        notes.append(f"actions API found running at {actions_url}")
    else:
        actions_url = _embed("actions_api")
        notes.append(f"actions API embedded at {actions_url}")

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
        "FROM turns ORDER BY id DESC LIMIT 25",
    )


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
            "model_loaded": model in names,
            "models": names,
        }
    except Exception as exc:
        return {
            "ok": False,
            "host": f"{host}:11434",
            "model": model,
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
                f"chat={OLLAMA_MODEL}",
            )
        return _brain


# ----------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "NixConsole/1.0"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
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
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def log_message(self, fmt, *args):
        pass  # the trace feed is the log

    # -- GET ------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._html(PAGE_V2)
            return

        if path == "/api/health":
            brain = get_brain()
            knowledge = brain.knowledge.health() or {"ok": False}
            actions = brain.actions.health() or {"ok": False}
            self._json(
                {
                    "knowledge": knowledge,
                    "actions": actions,
                    "ollama": _ollama_health(),
                    "embedded_services": len(_embedded),
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
        path = urlparse(self.path).path
        payload = self._read_json()

        if path == "/api/send":
            text = str(payload.get("text") or "").strip()
            location = str(payload.get("location") or "console")
            if not text:
                self._json({"ok": False, "error": "missing text"}, 400)
                return

            brain = get_brain()
            t0 = time.perf_counter()
            try:
                result = brain.handle(text=text, location=location)
            except Exception as exc:
                _trace("brain error", f"{type(exc).__name__}: {exc}", ok=False)
                self._json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500
                )
                return

            elapsed = (time.perf_counter() - t0) * 1000
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

            from router import classify as rule_classify

            route, features = rule_classify(text)
            outcome: dict = {
                "deterministic": {"route": route, "features": features}
            }

            if route == "unknown":
                brain = get_brain()
                gate = brain.knowledge.classify(text)
                outcome["model_gate"] = {"route": gate}

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
                connection = sqlite3.connect(KNOWLEDGE_DB, timeout=5)
                cursor = connection.execute(
                    "DELETE FROM knowledge WHERE id = ?", (record_id,)
                )
                connection.commit()
                deleted = cursor.rowcount
                connection.close()
            except Exception as exc:
                self._json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500
                )
                return
            _trace("kb delete", f"record #{record_id} removed ({deleted})")
            self._json({"ok": True, "deleted": deleted})
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
<header>
  <h1>&#9678; NIX TEST CONSOLE</h1>
  <span class="chip" id="chip-knowledge"><span class="dot"></span>knowledge</span>
  <span class="chip" id="chip-actions"><span class="dot"></span>actions</span>
  <span class="chip" id="chip-ollama"><span class="dot"></span><span id="chip-model">chat model</span></span>
  <span class="chip" id="chip-mode"><span class="dot ok"></span><span id="mode-text">mode</span></span>
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
    </div>
    <div class="presets" id="presets"></div>

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
  return r.json();
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
                       : await post("/api/send", {text, location: "console"});
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
    const o = h.ollama;
    setChip("chip-ollama", o.ok && o.model_loaded,
      `${o.model}${o.model_loaded ? "" : " (missing)"}`);
    $("mode-text").textContent =
      h.embedded_services > 0
        ? `embedded APIs (${h.embedded_services})`
        : "live APIs";
    $("foot").textContent =
      `timezone ${h.timezone} | knowledge ${h.knowledge.knowledge_records ?? "?"} records` +
      ` | chat model ${o.model} on ${o.host}` +
      ` | ollama models: ${(o.models || []).join(", ") || "none"}`;
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
<title>Nix Console</title>
<style>
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
<header>
  <h1>&#9678; NIX</h1>
  <nav id="nav">
    <button data-page="console" class="active">Console</button>
    <button data-page="kb">Knowledge Base</button>
    <button data-page="schedule">Schedule</button>
    <button data-page="tests">Testing</button>
  </nav>
  <span class="chip" id="chip-knowledge"><span class="dot"></span>knowledge</span>
  <span class="chip" id="chip-actions"><span class="dot"></span>actions</span>
  <span class="chip" id="chip-ollama"><span class="dot"></span><span id="chip-model">chat model</span></span>
  <span class="chip" id="chip-mode"><span class="dot ok"></span><span id="mode-text">mode</span></span>
</header>

<main>
  <!-- ============ PAGE: console ============ -->
  <section class="page active" id="page-console">
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
</main>

<div class="foot" id="foot"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);

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
  return r.json();
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
  if (page === "schedule") refreshSchedule();
};

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
    const o = h.ollama;
    setChip("chip-ollama", o.ok && o.model_loaded);
    $("chip-model").textContent = o.model + (o.model_loaded ? "" : " (missing)");
    $("mode-text").textContent = h.embedded_services > 0
      ? `embedded APIs (${h.embedded_services})` : "live APIs";
    $("foot").textContent =
      `timezone ${h.timezone} | ollama ${o.host} | models: ${(o.models || []).join(", ") || "none"}`;
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

async function send(dry) {
  const text = $("input").value.trim();
  if (!text) return;
  $("busy").textContent = "working...";
  $("send").disabled = true;
  try {
    const result = dry ? await post("/api/classify", {text})
                       : await post("/api/send", {text, location: "console"});
    show(result, dry);
  } catch (e) {
    show({ok: false, error: String(e)}, dry);
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

async function refreshKb() {
  try {
    const d = await (await fetch("/api/kb")).json();
    KB = d.records || [];
    renderKb();
    refreshPeople();
    const st = LAST_HEALTH;
    $("kb-sessions").innerHTML = "<div class='evt'>loading...</div>";
    const feed = await (await fetch("/api/feed")).json();
    $("kb-sessions").innerHTML = tableHtml(feed.turns || [], COLS.turns);
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
# Entry point
# ----------------------------------------------------------------------


def build_app(host: str = "0.0.0.0", port: int | None = None):
    """Create the console server (and resolve/embed sibling services)."""
    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]

    get_brain()  # resolve/embed services before serving

    server = ThreadingHTTPServer((host, port), Handler)
    return server, port


def main() -> int:
    host = os.environ.get("NIX_CONSOLE_HOST", "0.0.0.0")
    port_env = os.environ.get("NIX_CONSOLE_PORT")
    port = int(port_env) if port_env else None

    server, port = build_app(host=host, port=port)

    print()
    print("  NIX TEST CONSOLE")
    print(f"  local : http://127.0.0.1:{port}")
    print(f"  LAN   : http://{host}:{port}   (from any device on your network)")
    print(f"  chat model : {OLLAMA_MODEL} on {OLLAMA_HOST}:11434")
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
