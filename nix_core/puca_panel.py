"""PUCA Panel: the screen that shows what NIX is playing.

The Music page is a console tab; this is the same music on a screen of its own,
on a port of its own, so a wall-mounted display can be pointed at it once and
left there. Two things make that worth a separate listener rather than another
tab:

* it is read from across a room — one large now-playing card, large library
  tiles, a large volume control, and the lyrics of whatever is playing;
* it is spoken to — the page runs the browser's own speech recognition, so the
  person standing in front of it can say what to play.

The page itself ships inside the installed music provider's package
(``PUCA-Panel/``), so the display is versioned with the skill whose music it
shows, while the API stays this console's: the panel listener serves the same
``/api/music`` handlers the Music page and the mobile app use, so there is one
implementation of the music API instead of a second, drifting copy. The panel
origin is also the API origin, so the page needs no cross-origin setup.

One more thing lives here: when Nix cannot tell which of several matches a
spoken request meant, the options are parked as a *question* and the panel asks
the user. Recording it in Core (not in the page) means the answer still works
after the display is reloaded, and every surface — panel, Music page, reply —
can read the same question.

The port is fixed and public: ``NIX_PANEL_PORT`` (default ``49170``), separate
from the console's own fixed ``49117``, so settings, bookmarks and a kiosk
startup file survive console restarts.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from zoneinfo import ZoneInfo

import console
import music_api

PANEL_NAME = "PUCA Panel"
PANEL_HOST = os.environ.get("NIX_PANEL_HOST", "0.0.0.0")
PANEL_PORT = int(os.environ.get("NIX_PANEL_PORT", "49170"))

# The page's folder inside the music provider's package. The name is fixed by
# convention rather than by a manifest field: a manifest may not invent new keys,
# and this keeps one source of truth for "the skill whose music the panel shows".
PANEL_DIR = "PUCA-Panel"
PANEL_PAGE = f"{PANEL_DIR}/panel.html"
PANEL_ENTRYPOINTS = (PANEL_PAGE, f"{PANEL_DIR}/index.html")
# What a wall display needs to paint itself. Anything else in the package stays
# unservable from here, however it got declared.
PANEL_EXTENSIONS = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".webmanifest": "application/manifest+json",
}

# A question the user never answers should not sit on the wall display all night.
QUESTION_TTL_SECONDS = 15 * 60
MAX_OPTIONS = 6
MAX_TEXT = 300
MAX_PROMPT = 200

# The display is dark through the night and light through the day. 22:00–06:00
# is the quiet window: dark from 10PM, light again at 6AM.
DARK_FROM_HOUR = 22
DARK_UNTIL_HOUR = 6


class PanelError(Exception):
    """A bounded, user-facing failure with an HTTP status and a stable code."""

    def __init__(self, message: str, status: int = 400, code: str = "panel_error"):
        super().__init__(message)
        self.status = status
        self.code = code


# ---------------------------------------------------------------------------
# The provider skill that carries the panel
# ---------------------------------------------------------------------------


def _installed_skill_packages() -> list[tuple[str, dict]]:
    """Every installed skill as (skill_id, manifest), in install order."""
    saved = music_api._saved_state()
    installed = [item for item in (saved.get("installed") or []) if isinstance(item, str)]
    skills: list[tuple[str, dict]] = []
    for skill_id in installed:
        try:
            manifest = music_api._manifest_for(skill_id)
        except music_api.MusicApiError:
            continue
        skills.append((skill_id, manifest))
    return skills


def panel_skill() -> tuple[str, dict] | None:
    """The installed skill that both provides music and carries the panel page."""
    for skill_id, manifest in _installed_skill_packages():
        device_ui = manifest.get("device_ui")
        media = device_ui.get("media") if isinstance(device_ui, dict) else None
        if not isinstance(media, dict) or not media.get("provider"):
            continue
        if panel_page_path(skill_id, manifest) is not None:
            return skill_id, manifest
    return None


def panel_assets(skill_id: str, manifest: dict) -> list[str]:
    """Declared, servable panel files of one skill package."""
    declared = [path for path in (manifest.get("files") or []) if isinstance(path, str)]
    return [
        path for path in declared
        if path.startswith(PANEL_DIR + "/")
        and os.path.splitext(path)[1].lower() in PANEL_EXTENSIONS
    ]


def panel_page_path(skill_id: str, manifest: dict | None = None) -> str | None:
    """Absolute path of the panel's entry page, when the package declares one."""
    if manifest is None:
        try:
            manifest = music_api._manifest_for(skill_id)
        except music_api.MusicApiError:
            return None
    declared = set(panel_assets(skill_id, manifest))
    for entry in PANEL_ENTRYPOINTS:
        if entry not in declared:
            continue
        package = os.path.realpath(os.path.abspath(console._skill_package_path(skill_id)))
        target = os.path.realpath(os.path.join(package, *entry.split("/")))
        if os.path.commonpath((package, target)) == package and os.path.isfile(target):
            return target
    return None


def panel_info() -> dict:
    """What the console can say about the panel without starting it."""
    found = panel_skill()
    entry = panel_page_path(found[0], found[1]) if found else None
    return {
        "name": PANEL_NAME,
        "host": PANEL_HOST,
        "port": PANEL_PORT,
        "url": f"http://{_lan_ip()}:{PANEL_PORT}/",
        "skill_id": found[0] if found else None,
        "skill_name": found[1].get("name", found[0]) if found else None,
        "page": PANEL_PAGE,
        "available": bool(entry),
        "reason": "" if entry else "No installed music provider ships a PUCA-Panel page.",
    }


# ---------------------------------------------------------------------------
# The question the panel asks
# ---------------------------------------------------------------------------


_question_lock = threading.RLock()
_question: dict | None = None


def _bounded_text(value: Any, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _option(entry: Any) -> dict | None:
    """One choice, reduced to what the display paints and the answer plays."""
    if not isinstance(entry, dict):
        return None
    video_id = _bounded_text(entry.get("video_id"), 64)
    title = _bounded_text(entry.get("title"))
    if not video_id or not title:
        return None
    duration = entry.get("duration_sec")
    return {
        "video_id": video_id,
        "title": title,
        "artist": _bounded_text(entry.get("artist"), 200),
        "album": _bounded_text(entry.get("album"), 200),
        "duration_sec": int(duration) if isinstance(duration, (int, float)) and duration > 0 else 0,
        "artwork_url": _bounded_text(entry.get("artwork_url"), 1000),
    }


def ask_choice(prompt: str, options: list[Any], *, source: str = "search", query: str = "") -> dict:
    """Park a choice for the user and return the question as the panel sees it.

    The first option is not played here: the whole point is that Nix does not
    know which one was meant, so playing the top match would be the guess this
    exists to avoid.
    """
    choices = [choice for choice in (_option(entry) for entry in options or []) if choice]
    if not choices:
        raise PanelError("A question needs at least one playable option.", 400, "no_options")
    asked_at = time.time()
    question = {
        "id": f"{int(asked_at)}-{os.getpid()}",
        "prompt": _bounded_text(prompt, MAX_PROMPT) or "Which one did you mean?",
        "query": _bounded_text(query, MAX_PROMPT),
        "source": _bounded_text(source, 32) or "search",
        "options": choices[:MAX_OPTIONS],
        "asked_at": asked_at,
        "expires_at": asked_at + QUESTION_TTL_SECONDS,
    }
    global _question
    with _question_lock:
        _question = question
    return dict(question)


def pending_question() -> dict | None:
    """The question waiting for an answer, or None once it has gone stale."""
    global _question
    with _question_lock:
        question = _question
        if question is None:
            return None
        if float(question.get("expires_at") or 0) <= time.time():
            _question = None
            return None
        return dict(question)


def clear_question() -> bool:
    """Drop the pending question; return whether one was waiting."""
    global _question
    with _question_lock:
        had = _question is not None
        _question = None
        return had


def answer_question(index: Any) -> dict:
    """Play the chosen option and clear the question.

    The question is cleared before the provider is called, so a double tap on a
    wall display can never start two tracks.
    """
    question = pending_question()
    if question is None:
        raise PanelError("Nothing is waiting for an answer.", 409, "no_question")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(question["options"]):
        raise PanelError("Choose one of the offered options.", 400, "bad_option")
    choice = question["options"][index]
    clear_question()
    try:
        skill_id, result = music_api._provider_call("play", {"video_id": choice["video_id"]})
    except music_api.MusicApiError as exc:
        raise PanelError(str(exc), exc.status, exc.code) from exc
    music_api._start_playback(result, skill_id)
    return {"ok": True, "skill_id": skill_id, "chosen": choice, "result": result}


# ---------------------------------------------------------------------------
# Day and night
# ---------------------------------------------------------------------------


def _timezone() -> ZoneInfo:
    name = getattr(console, "TIMEZONE", None) or os.environ.get("NIX_TZ", "UTC")
    try:
        return ZoneInfo(str(name))
    except Exception:  # an unknown zone falls back rather than breaking the display
        return ZoneInfo("UTC")


def daypart(now: float | None = None) -> str:
    """``dark`` inside the quiet window, ``light`` outside it.

    The hour comes from the console's own configured timezone, so the display
    follows the house clock even when the screen's own clock is wrong or the
    browser reports UTC.
    """
    moment = datetime.fromtimestamp(now if now is not None else time.time(), tz=_timezone())
    return "dark" if moment.hour >= DARK_FROM_HOUR or moment.hour < DARK_UNTIL_HOUR else "light"


def theme() -> dict:
    moment = datetime.fromtimestamp(time.time(), tz=_timezone())
    return {
        "mode": daypart(),
        "hour": moment.hour,
        "minute": moment.minute,
        "timezone": str(getattr(console, "TIMEZONE", "UTC")),
        "dark_from_hour": DARK_FROM_HOUR,
        "dark_until_hour": DARK_UNTIL_HOUR,
        "local_time": moment.strftime("%H:%M"),
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def panel_state() -> dict:
    """Everything the display needs for one paint, in one round trip.

    A wall display polls on a timer, so the cheap snapshot, the queue and the
    question travel together instead of costing three requests. Lyrics are left
    out on purpose: they are per-track and the page caches them by track id.
    """
    payload: dict = {
        "ok": True,
        "panel": panel_info(),
        "theme": theme(),
        "question": pending_question(),
        "server_time": datetime.now(tz=_timezone()).isoformat(timespec="seconds"),
    }
    try:
        payload["now_playing"] = music_api._now_playing_payload()
    except Exception as exc:  # a device hiccup must not blank the display
        payload["now_playing"] = None
        payload["now_playing_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    try:
        skill_id, result = music_api._provider_call("queue", {"op": "list"})
        payload["queue"] = music_api._queue_payload(skill_id, result)
    except Exception as exc:
        payload["queue"] = None
        payload["queue_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return payload


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


# The music endpoints the display uses, mapped to the same handlers the console
# registers. Keeping the table here means the panel can never serve a route the
# console does not, and a new console route is added in one place.
_MUSIC_ROUTES: dict[str, str] = {
    "GET /api/music": "music_index",
    "GET /api/music/now-playing": "music_now_playing",
    "POST /api/music/control": "music_control",
    "GET /api/music/search": "music_search",
    "POST /api/music/play": "music_play",
    "GET /api/music/queue": "music_queue",
    "POST /api/music/queue": "music_queue",
    "GET /api/music/library": "music_library",
    "POST /api/music/save": "music_save",
    "GET /api/music/lyrics": "music_lyrics",
    "GET /api/music/artwork": "music_artwork",
}


def _lan_ip() -> str:
    """Best-effort address of this machine on the local network."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


class PanelHandler(BaseHTTPRequestHandler):
    """Serve the panel page, its assets, the music API and the question routes."""

    server_version = "NIXPUCApanel/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib name
        # The console prints its own request log; the panel stays quiet except
        # for failures, which the handlers below report through the response.
        return

    # -- helpers ----------------------------------------------------------
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, exc: PanelError) -> None:
        self._json({"ok": False, "code": exc.code, "error": str(exc)}, exc.status)

    def _body(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or 0)
        except (TypeError, ValueError):
            raise PanelError("Content-Length must be a number.", 400, "bad_length")
        if length <= 0:
            return {}
        if length > 64 * 1024:
            raise PanelError("Request body is too large.", 413, "body_too_large")
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8") or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise PanelError("Send a JSON object.", 400, "bad_json")
        if not isinstance(parsed, dict):
            raise PanelError("Send a JSON object.", 400, "bad_json")
        return parsed

    def _send_file(self, target: str, content_type: str) -> None:
        try:
            with open(target, "rb") as handle:
                body = handle.read()
        except OSError:
            raise PanelError("Panel file not found.", 404, "not_found")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        # The page is versioned with the skill package it came from, so a
        # reload must never paint yesterday's display from the browser cache.
        self.send_header("Cache-Control", "no-cache, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_panel_asset(self, asset: str) -> None:
        found = panel_skill()
        if found is None:
            raise PanelError(
                "No installed music provider ships a PUCA-Panel page.", 404, "no_panel"
            )
        skill_id, manifest = found
        if asset not in set(panel_assets(skill_id, manifest)):
            raise PanelError("That panel asset is not declared by the skill.", 404, "not_found")
        package = os.path.realpath(os.path.abspath(console._skill_package_path(skill_id)))
        target = os.path.realpath(os.path.join(package, *asset.split("/")))
        if os.path.commonpath((package, target)) != package or not os.path.isfile(target):
            raise PanelError("Panel file not found.", 404, "not_found")
        self._send_file(target, PANEL_EXTENSIONS[os.path.splitext(asset)[1].lower()])

    def _panel_page(self) -> None:
        found = panel_skill()
        entry = panel_page_path(found[0], found[1]) if found else None
        if entry is None:
            raise PanelError(
                "No installed music provider ships a PUCA-Panel page.", 404, "no_panel"
            )
        self._send_file(entry, PANEL_EXTENSIONS[".html"])

    def _dispatch_music(self, method: str, path: str) -> bool:
        name = _MUSIC_ROUTES.get(f"{method} {path}")
        if name is None:
            return False
        handler = getattr(music_api, name)
        try:
            handled = handler(self, None)
        except music_api.MusicApiError as exc:
            handled = music_api._fail(self, exc)
        if not handled:
            self._json({"ok": False, "error": "That music route did not answer."}, 500)
        return True

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        path = urllib.parse.urlparse(self.path).path
        asset = urllib.parse.unquote(path.lstrip("/"))
        try:
            if asset in ("", "panel", "index.html"):
                return self._panel_page()
            if asset.startswith(PANEL_DIR + "/"):
                return self._serve_panel_asset(asset)
            if asset == "api/panel/state":
                return self._json(panel_state())
            if asset == "api/panel":
                return self._json({"ok": True, **panel_info(), "question": pending_question()})
            if asset.startswith("api/music") and self._dispatch_music("GET", path):
                return None
            raise PanelError("No such panel route.", 404, "not_found")
        except PanelError as exc:
            self._fail(exc)
        return None

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        path = urllib.parse.urlparse(self.path).path
        asset = urllib.parse.unquote(path.lstrip("/"))
        try:
            if asset.startswith("api/music") and self._dispatch_music("POST", path):
                return None
            if asset == "api/panel/answer":
                body = self._body()
                # Answer first, then report: the reply must not carry the
                # question it has just closed.
                outcome = answer_question(body.get("index"))
                return self._json({**panel_state(), **outcome})
            if asset == "api/panel/question":
                body = self._body()
                question = ask_choice(
                    str(body.get("prompt") or ""),
                    body.get("options") if isinstance(body.get("options"), list) else [],
                    source=str(body.get("source") or "search"),
                    query=str(body.get("query") or ""),
                )
                return self._json({"ok": True, "question": question})
            if asset == "api/panel/dismiss":
                return self._json({"ok": True, "dismissed": clear_question()})
            raise PanelError("No such panel route.", 404, "not_found")
        except PanelError as exc:
            self._fail(exc)
        return None


def create_server(host: str = PANEL_HOST, port: int = PANEL_PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), PanelHandler)
    server.daemon_threads = True
    return server


def start_in_thread(host: str = PANEL_HOST, port: int = PANEL_PORT) -> tuple[ThreadingHTTPServer, int]:
    """Start the panel listener on the console's behalf and return it.

    A display is not worth taking the console down for: a port already in use is
    reported and the console keeps running without the panel.
    """
    server = create_server(host, port)
    resolved = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, name="puca-panel", daemon=True)
    thread.start()
    return server, resolved


def main() -> int:
    console.get_brain()
    server, port = start_in_thread()
    host = PANEL_HOST if PANEL_HOST not in ("0.0.0.0", "::") else _lan_ip()
    info = panel_info()
    print()
    print(f"  {PANEL_NAME.upper()}")
    print(f"  local : http://127.0.0.1:{port}/")
    print(f"  LAN   : http://{host}:{port}/   (point the wall display here)")
    print(f"  page  : {info['page']} from {info['skill_id'] or 'no installed music provider'}")
    print("  Ctrl+C to stop")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\npanel stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
