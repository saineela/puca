"""
Nix Knowledge HTTP API.

Lets nix_core (and anything else on the LAN) use the knowledge engine
over HTTP:

  GET  /health    -> engine + model status
  POST /process   -> {"text"}  full KnowledgeNeedle pipeline (creates /
                     finds / updates / cancels calendar events and
                     facts; schedules actions through the bridge)
  POST /classify  -> {"text"}  knowledge-vs-chat decision. Deterministic
                     rules first (nix_core already applied its own
                     rules); the Qwen model gate decides the rest.
  POST /digest    -> compact verified-knowledge block for the chat model

Run from the nix_knowledge venv:

    ~/nix_knowledge/.venv/bin/python scripts/knowledge_api.py

Environment:
    NIX_KNOWLEDGE_API_HOST (default 127.0.0.1)
    NIX_KNOWLEDGE_API_PORT (default 8100)
    NIX_KNOWLEDGE_DB       (default <repo>/data/knowledge.db)
    NIX_ACTIONS_DB         (default <repo>/data/actions.db)
    NIX_TZ                 (default America/Chicago)
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Make `import nix_knowledge` work when launched as a plain script.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)

from nix_actions.engine import ActionsEngine  # noqa: E402
from nix_knowledge.engine import KnowledgeEngine  # noqa: E402

try:
    from nix_knowledge.semantic.classifier import SemanticClassifier
    from nix_knowledge.semantic.embedder import Embedder
except Exception:  # noqa: BLE001 - neural layer optional
    SemanticClassifier = None
    Embedder = None

TIMEZONE = os.environ.get("NIX_TZ", "America/Chicago")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("NIX_DATA_DIR", REPO_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DB = os.environ.get(
    "NIX_KNOWLEDGE_DB",
    str(DATA_DIR / "knowledge.db"),
)
ACTIONS_DB = os.environ.get(
    "NIX_ACTIONS_DB",
    str(DATA_DIR / "actions.db"),
)

# Nix_predictor is the local Qwen2.5 0.5B constrained function selector.
# Deterministic rules remain the safety/fast path, while the predictor handles
# indirect and multi-intent requests. Set to 0 only for diagnostics.
MODEL_GATE_ENV = "NIX_KNOWLEDGE_MODEL_GATE"  # "1" force, "0" disable

_state_lock = threading.Lock()
_needle = None          # KnowledgeNeedle, lazy-loaded on first use
_actions_engine = None  # shared with the needle bridge


def _get_actions_engine() -> ActionsEngine:
    global _actions_engine
    if _actions_engine is None:
        _actions_engine = ActionsEngine(
            ACTIONS_DB, timezone=TIMEZONE
        )
    return _actions_engine


def _get_needle():
    """
    Lazy KnowledgeNeedle: the Qwen model only loads when a request
    actually needs it, so /process on deterministic rules and /digest
    stay fast even on a cold start.
    """
    global _needle
    if _needle is None:
        from nix_knowledge.needle import KnowledgeNeedle

        with _state_lock:
            if _needle is None:
                _needle = KnowledgeNeedle(
                    KnowledgeEngine(KNOWLEDGE_DB),
                    timezone=TIMEZONE,
                    actions_engine=_get_actions_engine(),
                )
    return _needle


# ----------------------------------------------------------------------
# Classifier: knowledge vs chat
# ----------------------------------------------------------------------


def classify(text: str) -> dict:
    """
    Decide knowledge vs chat for one request.

    Layer 1: the deterministic rules from nix_knowledge (the same
    lexical router the needle pipeline uses internally). These are
    high-precision knowledge-side signals.

    Layer 2 (model gate): when the rules abstain, the Qwen model
    decides with a constrained prompt. The model only ever answers
    knowledge or chat.
    """
    from nix_knowledge.rules import route

    try:
        from nix_knowledge.temporal import TemporalResolver

        routed = route(text, TemporalResolver(timezone=TIMEZONE))
    except Exception:
        routed = None

    if routed is not None:
        return {
            "route": "knowledge",
            "reason": "deterministic_rules",
            "function": routed[0],
        }

    gate = os.environ.get(MODEL_GATE_ENV, "1")
    if gate == "0":
        return {"route": "chat", "reason": "rules_abstained_gate_off"}

    try:
        needle = _get_needle()
    except Exception as exc:  # model unavailable: degrade, don't fail
        return {
            "route": "chat",
            "reason": f"model_unavailable: {type(exc).__name__}",
        }

    verdict = _model_gate(needle, text)
    return verdict


# Gate policy: any tool call counts as a knowledge decision. The
# dangerous world-question shapes ("how do i make french fries",
# "what is the capital of X") are already intercepted upstream by
# nix_core's deterministic rules, so by the time the gate sees a
# request it is ambiguous personal-vs-world - exactly what the model
# should decide, mutations included ("just so you know, my spare key
# is under the mat" -> create_fact).


def _model_gate(needle, text: str) -> dict:
    """
    Constrained model classification: the model must emit a tool call
    to pick knowledge. Anything else - no call, parse failure, crash -
    is chat. Keeps false-knowledge routes rare while letting the model
    resolve indirect storage and scheduling intent.
    """
    with _state_lock:
        try:
            tool_call = needle._generate_tool_call(text)
        except Exception as exc:
            return {
                "route": "chat",
                "reason": f"model_error: {type(exc).__name__}",
            }

    if tool_call is None:
        return {"route": "chat", "reason": "model_abstained"}

    return {
        "route": "knowledge",
        "reason": "model_selected_function",
        "function": tool_call.get("name"),
    }


# ----------------------------------------------------------------------
# Digest: verified personal knowledge for the chat model
# ----------------------------------------------------------------------


def build_digest() -> str:
    """
    Compact block of durable knowledge: upcoming calendar events and
    stored facts/preferences. This is what lets the zero-knowledge
    chat model still answer personal questions mid-conversation.
    """
    engine = KnowledgeEngine(KNOWLEDGE_DB)

    try:
        lines: list[str] = []

        from nix_knowledge.context import TemporalContext

        context = TemporalContext(timezone=TIMEZONE)

        events = engine.search("calendar_event")
        upcoming: list[tuple[str, str]] = []

        for record in events:
            data = record.data or {}
            if data.get("status") == "cancelled":
                continue

            start = data.get("start")
            if not start:
                continue

            try:
                moment = datetime.fromisoformat(str(start))
            except ValueError:
                continue

            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=context.timezone)

            if moment < datetime.now(context.timezone) - timedelta(
                hours=12
            ):
                continue

            upcoming.append(
                (
                    moment.isoformat(),
                    f"{data.get('title', 'event')} - "
                    f"{context.describe(start=moment)}",
                )
            )

        upcoming.sort(key=lambda pair: pair[0])

        if upcoming:
            lines.append("Upcoming events:")
            lines.extend(f"- {text}" for _, text in upcoming[:8])

        facts = [
            record.data.get("value", "")
            for record in engine.search("fact")
            if record.data.get("value")
        ]

        if facts:
            lines.append("Stored facts and preferences:")
            lines.extend(f"- {fact}" for fact in facts[:10])

        keys = [
            record.data.get("value", "")
            for record in engine.search("key")
            if record.data.get("value")
        ]

        if keys:
            lines.append("People and things in your life:")
            lines.extend(f"- {key}" for key in keys[:12])

        states = [
            record.data.get("value", "")
            for record in engine.search("person")
            if record.data.get("statement_type") == "current_state"
            and record.data.get("value")
        ]
        if states:
            lines.append("How people close to the user are doing right now:")
            lines.extend(f"- {state}" for state in states[:8])

        return "\n".join(lines[:32])

    finally:
        engine.close()


def build_memory_block(current_text: str | None = None) -> str:
    """
    Full MEMORY block for NixLM: facts + people + current states +
    dated moments + the user's current emotional state + pending
    (escalated) context. Read-time rendering only - nothing is stored,
    nothing is summarized away. This is the "connected" half of the
    separate-but-connected contract (see nix_knowledge/memory_block.py).
    """
    from nix_knowledge.memory_block import render_memory_block

    engine = KnowledgeEngine(KNOWLEDGE_DB)
    try:
        return render_memory_block(engine, current_text=current_text)
    finally:
        engine.close()


# ----------------------------------------------------------------------
# Neural intent: emotion + category for the waiting filler
# ----------------------------------------------------------------------

_intent_classifier = None
_intent_lock = threading.Lock()


def analyze_intent(text: str) -> dict:
    """
    Fast neural read of the utterance (frozen-embedder heads, ~ms on
    GPU, no LLM): dominant emotion, positive/negative valence, topic
    category. Core uses this to show a mood-matched filler line while
    the chat model composes the real reply.
    """
    global _intent_classifier
    if SemanticClassifier is None or Embedder is None:
        return {"ok": False, "reason": "semantic_layer_unavailable"}

    with _intent_lock:
        if _intent_classifier is None:
            try:
                _intent_classifier = SemanticClassifier(Embedder())
                if not _intent_classifier.load():
                    return {
                        "ok": False,
                        "reason": "classifier_not_trained",
                    }
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "reason": f"load_failed: {exc}"}
        try:
            result = _intent_classifier.classify(text)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"classify_failed: {exc}"}

    emotions = result.get("emotion") or []
    top = emotions[0] if emotions else ("neutral", 0.0)
    category = result.get("category") or ("general", 0.0)
    tone = result.get("tone") or ("neutral", 0.0)

    positive_emotions = {
        "joy", "amusement", "excitement", "love", "optimism",
        "gratitude", "pride", "approval", "admiration", "caring",
        "desire", "relief",
    }
    negative_emotions = {
        "sadness", "anger", "fear", "disgust", "disappointment",
        "grief", "remorse", "embarrassment", "annoyance",
        "disapproval", "nervousness", "worry",
    }
    valence = "neutral"
    for label, _prob in emotions[:5]:
        if label in positive_emotions:
            valence = "good"
            break
        if label in negative_emotions:
            valence = "bad"
            break

    return {
        "ok": True,
        "emotion": top[0],
        "emotion_confidence": round(float(top[1]), 3),
        "valence": valence,
        "category": category[0],
        "tone": tone[0],
    }


# ----------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "NixKnowledgeAPI/1.0"

    # -- helpers -------------------------------------------------------

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

    def log_message(self, fmt, *args):  # quiet default access log
        if os.environ.get("NIX_API_VERBOSE"):
            super().log_message(fmt, *args)

    # -- routes --------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            engine = KnowledgeEngine(KNOWLEDGE_DB)
            try:
                records = len(engine.search())
            finally:
                engine.close()

            self._json(
                {
                    "ok": True,
                    "service": "nix_knowledge",
                    "timezone": TIMEZONE,
                    "knowledge_records": records,
                    "model_loaded": _needle is not None,
                    "actions_db": ACTIONS_DB,
                }
            )
            return

        self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        payload = self._read_json()
        text = str(payload.get("text") or "").strip()

        if path == "/warmup":
            try:
                _get_needle()
                self._json({"ok": True, "model_loaded": True})
            except Exception as exc:
                self._json({
                    "ok": False,
                    "model_loaded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }, 503)
            return

        if path == "/process":
            if not text:
                self._json(
                    {"ok": False, "error": "missing text"}, 400
                )
                return

            try:
                needle = _get_needle()
            except Exception as exc:
                self._json(
                    {
                        "ok": False,
                        "error": (
                            f"model load failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    },
                    503,
                )
                return

            with _state_lock:
                result = needle.process(text)

            self._json(result)
            return

        if path == "/classify":
            if not text:
                self._json(
                    {"ok": False, "error": "missing text"}, 400
                )
                return

            self._json(classify(text))
            return

        if path == "/digest":
            self._json({"digest": build_digest()})
            return

        if path == "/memory_block":
            self._json(
                {
                    "memory_block": build_memory_block(
                        current_text=text or None
                    )
                }
            )
            return

        if path == "/intent":
            if not text:
                self._json(
                    {"ok": False, "error": "missing text"}, 400
                )
                return
            self._json(analyze_intent(text))
            return

        if path == "/keys":
            """
            Two modes:

            {"text": ...}   - Key Finding for non-knowledge routes:
              chat-routed utterances still teach durable personal
              facts ("btw my brother Alex loves hiking").

            {"lookup": ...} - does the KB know this person? Used by
              the brain to send "who is Maanvi" to recall while
              letting "who is einstein" fall to world chat.
            """
            lookup = str(payload.get("lookup") or "").strip()
            if lookup:
                matches: list[str] = []
                try:
                    engine = KnowledgeEngine(KNOWLEDGE_DB)
                    try:
                        needle_term = lookup.lower()
                        for record in engine.search("key"):
                            data = record.data or {}
                            value = str(data.get("value", ""))
                            subject = str(data.get("subject", ""))
                            if (
                                needle_term in value.lower()
                                or needle_term in subject.lower()
                            ):
                                matches.append(value)
                    finally:
                        engine.close()
                except Exception:
                    matches = []
                self._json({"ok": True, "matches": matches})
                return

            found: list[str] = []
            if text:
                try:
                    from zoneinfo import ZoneInfo

                    from nix_knowledge.keys import (
                        extract_keys,
                        key_text,
                        store_keys,
                    )

                    engine = KnowledgeEngine(KNOWLEDGE_DB)
                    try:
                        stored = store_keys(
                            engine,
                            extract_keys(text),
                            actions_engine=_get_actions_engine(),
                            timezone=ZoneInfo(TIMEZONE),
                        )
                        found = [key_text(k) for k in stored]
                    finally:
                        engine.close()
                except Exception:
                    found = []
            self._json({"ok": True, "keys_found": found})
            return

        self._json({"ok": False, "error": "not found"}, 404)


def main() -> int:
    host = os.environ.get("NIX_KNOWLEDGE_API_HOST", "127.0.0.1")
    port = int(os.environ.get("NIX_KNOWLEDGE_API_PORT", "8100"))

    server = ThreadingHTTPServer((host, port), Handler)

    print(f"nix_knowledge API on http://{host}:{port}")
    print(f"  knowledge.db: {KNOWLEDGE_DB}")
    print(f"  actions.db:   {ACTIONS_DB}")
    print(f"  timezone:     {TIMEZONE}")
    print("  model loads lazily on first /process or model classify")

    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
