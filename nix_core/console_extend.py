"""Extension entrypoint for the NIX console.

Keep ``console.py`` as the base implementation. Add new HTTP endpoints here
with :func:`register_route`; every request not claimed by an extension falls
through to the original console handler unchanged.

Example::

    def get_extension_status(handler, match):
        handler._json({"ok": True, "extension": "ready"})
        return True

    register_route("GET", r"/api/extensions/status", get_extension_status)

Run with ``python nix_core/console_extend.py`` instead of editing console.py.
"""
from __future__ import annotations

import os
import re
import socket
import ssl
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Callable
from urllib.parse import urlparse

import console as base_console
import mobile_app_api
import music_api
import puca_panel
import system_model_api
import voice
import voice_api


ExtensionCallback = Callable[[BaseHTTPRequestHandler, re.Match[str]], bool]


@dataclass(frozen=True)
class ExtensionRoute:
    method: str
    expression: str
    pattern: re.Pattern[str]
    callback: ExtensionCallback


_routes: list[ExtensionRoute] = []
_routes_lock = threading.RLock()


def register_route(method: str, expression: str, callback: ExtensionCallback) -> None:
    """Register a full-match path regex; return True from its callback if handled.

    The callback receives the request handler and regex match, and is
    responsible for writing the response. Return False to delegate to the
    base console. Extensions are local trusted Python code, like console.py.
    """
    normalized_method = str(method).upper()
    if normalized_method not in {"GET", "POST", "OPTIONS"}:
        raise ValueError("Extension routes support GET, POST, and OPTIONS only.")
    if not isinstance(expression, str) or not expression.startswith("/") or len(expression) > 200:
        raise ValueError("Extension route expressions must be short path regexes beginning with '/'.")
    if not callable(callback):
        raise TypeError("Extension route callback must be callable.")
    route = ExtensionRoute(normalized_method, expression, re.compile(expression), callback)
    with _routes_lock:
        if any(item.method == route.method and item.expression == route.expression for item in _routes):
            raise ValueError("That extension route is already registered.")
        _routes.append(route)


register_route("GET", r"/api/mobile/v1", mobile_app_api.api_index)
register_route("GET", r"/api/mobile/v1/home-copy", mobile_app_api.home_copy)

# Official device registration for the NIX Home mobile app.
register_route("GET", r"/api/mobile/v1/devices", mobile_app_api.devices_index)
register_route("POST", r"/api/mobile/v1/devices/report", mobile_app_api.devices_report)
register_route("POST", r"/api/mobile/v1/devices/discover", mobile_app_api.devices_discover)
register_route("POST", r"/api/mobile/v1/devices/register", mobile_app_api.devices_register)
register_route("POST", r"/api/mobile/v1/devices/unregister", mobile_app_api.devices_unregister)
register_route("GET", r"/api/mobile/v1/devices/list", mobile_app_api.devices_list)
register_route("GET", r"/api/mobile/v1/devices/([0-9A-Fa-f:]{12,17})", mobile_app_api.devices_get)
register_route("POST", r"/api/mobile/v1/devices/mock/seed", mobile_app_api.devices_mock_seed)

# Music page + control API. The page is filled by an installed skill that
# declares device_ui.page; the same endpoints back the NIX Home mobile app.
register_route("GET", r"/api/music", music_api.music_index)
register_route("GET", r"/api/music/now-playing", music_api.music_now_playing)
register_route("POST", r"/api/music/control", music_api.music_control)
register_route("GET", r"/api/music/search", music_api.music_search)
register_route("POST", r"/api/music/play", music_api.music_play)
register_route("GET", r"/api/music/library", music_api.music_library)
register_route("GET", r"/api/music/queue", music_api.music_queue)
register_route("POST", r"/api/music/queue", music_api.music_queue)
# The independent queue service calls this back, one track at a time, for a
# track it was handed as a bare id. It resolves and nothing else: no queue is
# touched and nothing starts playing.
register_route("POST", r"/api/music/queue/resolve", music_api.music_queue_resolve)
# …and this is the player callback: the queue service has chosen the track, and
# this end presses the speaker's button for it. Internal by name and by purpose.
register_route("POST", r"/internal/music/queue/play", music_api.music_queue_play)
register_route("POST", r"/api/music/save", music_api.music_save)
register_route("GET", r"/api/music/lyrics", music_api.music_lyrics)
# Covers come from the console's own cache, so the page paints them without a
# CDN round trip; the source image URL travels in the query string.
register_route("GET", r"/api/music/artwork", music_api.music_artwork)
register_route("GET", r"/api/skills/([^/]+)/page/([A-Za-z0-9][^?]*)", music_api.music_page_asset)

# Speech to text on this machine (nix_core/voice.py). The pages that record audio
# — the PUCA Panel, and the Music page through the same helper — post the clip to
# their own origin, so the panel origin serves these too (see puca_panel._ROUTES).
register_route("GET", r"/api/voice", voice_api.voice_index)
register_route("POST", r"/api/voice/transcribe", voice_api.voice_transcribe)

# Mobile aliases: the NIX Home app mounts the same music page and calls the
# same operations under the versioned mobile prefix.
register_route("GET", r"/api/mobile/v1/music", music_api.music_index)
register_route("GET", r"/api/mobile/v1/music/now-playing", music_api.music_now_playing)
register_route("POST", r"/api/mobile/v1/music/control", music_api.music_control)
register_route("GET", r"/api/mobile/v1/music/search", music_api.music_search)
register_route("POST", r"/api/mobile/v1/music/play", music_api.music_play)
register_route("GET", r"/api/mobile/v1/music/library", music_api.music_library)
register_route("GET", r"/api/mobile/v1/music/queue", music_api.music_queue)
register_route("POST", r"/api/mobile/v1/music/queue", music_api.music_queue)
register_route("POST", r"/api/mobile/v1/music/save", music_api.music_save)
register_route("GET", r"/api/mobile/v1/music/lyrics", music_api.music_lyrics)
register_route("GET", r"/api/mobile/v1/music/artwork", music_api.music_artwork)

# The PUCA Panel is served on its own fixed port (puca_panel.PANEL_PORT), but
# the console still answers "/api/panel" so the dashboard, the docs and anyone
# on the LAN can find where that display lives without guessing.
def panel_index(handler, _match=None) -> bool:
    return music_api._respond(handler, {"ok": True, **puca_panel.panel_info(), "question": puca_panel.pending_question()})


register_route("GET", r"/api/panel", panel_index)

# PUCA system model v3 measurement surface (read-only, no personal prompts).
register_route("GET", r"/api/system-model", system_model_api.system_model_index)
register_route("GET", r"/api/system-model/inference", system_model_api.inference_metrics_view)
register_route("GET", r"/api/system-model/privacy", system_model_api.privacy_view)
register_route("GET", r"/api/system-model/proactivity", system_model_api.proactivity_view)
register_route("GET", r"/api/system-model/context", system_model_api.context_view)
register_route("GET", r"/api/system-model/turn-taking", system_model_api.turn_taking_view)


def extensions_status(handler, _match=None) -> bool:
    """Every route an extension has registered, for the docs and for support."""
    with _routes_lock:
        routes = [
            {"method": route.method, "path": route.expression}
            for route in _routes
        ]
    return music_api._respond(handler, {"ok": True, "count": len(routes), "routes": routes})


register_route("GET", r"/api/extensions/status", extensions_status)


def unregister_route(method: str, expression: str) -> bool:
    """Remove a previously registered route; return whether one was found."""
    normalized_method = str(method).upper()
    with _routes_lock:
        for index, route in enumerate(_routes):
            if route.method == normalized_method and route.expression == expression:
                del _routes[index]
                return True
    return False


def _dispatch_extension(handler: BaseHTTPRequestHandler, method: str) -> bool:
    raw_path = urlparse(handler.path).path.rstrip("/") or "/"
    normalized_path = base_console._application_route_path(raw_path)
    paths = (raw_path,) if normalized_path == raw_path else (raw_path, normalized_path)
    with _routes_lock:
        routes = tuple(_routes)
    for path in paths:
        for route in routes:
            if route.method != method:
                continue
            match = route.pattern.fullmatch(path)
            if match is None:
                continue
            try:
                handled = route.callback(handler, match)
                if not isinstance(handled, bool):
                    raise TypeError("Extension route callback must return True or False.")
                if handled:
                    return True
            except Exception as exc:
                handler._json({"ok": False, "error": f"extension route failed: {type(exc).__name__}"}, 500)
                return True
    return False


_BASE_HANDLER = base_console.Handler


class ExtendedHandler(_BASE_HANDLER):
    """Dispatch registered additions, then preserve the full base behavior."""

    def do_GET(self):
        if not _dispatch_extension(self, "GET"):
            super().do_GET()

    def do_POST(self):
        if not _dispatch_extension(self, "POST"):
            super().do_POST()

    def do_OPTIONS(self):
        if not _dispatch_extension(self, "OPTIONS"):
            super().do_OPTIONS()


def build_app(host: str = "127.0.0.1", port: int | None = None):
    """Build the extended server using the base console's existing services."""
    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]
    base_console.get_brain()
    server = base_console.ThreadingHTTPServer((host, port), ExtendedHandler)
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
    server, port = build_app(host=host, port=49117)
    lan = host if host not in ("0.0.0.0", "::") else _lan_ip()
    # The PUCA Panel is a second, deliberate listener: a wall display is pointed
    # at its fixed port once and left there, and it serves the same music API, so
    # the display never needs the console tab. A port already taken is reported
    # and the console still starts without it.
    panel_host = os.environ.get("NIX_PANEL_HOST", puca_panel.PANEL_HOST)
    panel_server = None
    panel_note = "disabled (port unavailable)"
    try:
        panel_server, panel_port = puca_panel.start_in_thread(
            host=panel_host,
            port=int(os.environ.get("NIX_PANEL_PORT", puca_panel.PANEL_PORT)),
        )
        panel_skill = puca_panel.panel_skill()
        panel_note = (
            f"http://{lan}:{panel_port}/ ({panel_skill[0]})"
            if panel_skill
            else f"http://{lan}:{panel_port}/ (no installed music provider ships the page)"
        )
    except OSError as exc:
        panel_note = f"disabled ({type(exc).__name__}: {exc})"
    # The same display over TLS. A browser only allows a microphone on a secure
    # origin, so the http URL above can show everything and hear nothing; this one
    # records. It is self-signed, so the browser asks to be trusted once.
    panel_tls_server = None
    panel_tls_note = "disabled (port unavailable)"
    try:
        panel_tls_server, panel_tls_port = puca_panel.start_tls_in_thread(
            host=panel_host,
            port=int(os.environ.get("NIX_PANEL_TLS_PORT", puca_panel.PANEL_TLS_PORT)),
        )
        panel_tls_note = f"https://{lan}:{panel_tls_port}/ (same display, microphone allowed)"
    except (OSError, ssl.SSLError) as exc:
        panel_tls_note = f"disabled ({type(exc).__name__}: {exc})"
    try:
        print()
        print("  NIX TEST CONSOLE (extended)")
        print(f"  local : http://127.0.0.1:{port}")
        print(f"  LAN   : http://{lan}:{port}   (from any device on your network)")
        print(f"  ports : {port} console (Knowledge/Actions run in-process)")
        print(f"          {panel_note} - PUCA Panel")
        print(f"          {panel_tls_note}")
        print(f"  hearing: {voice.available()['model']} on {voice.device_name()} (local, no cloud)")
        print(f"  chat backend : {base_console.CASPER_BACKEND} ({base_console.OLLAMA_MODEL} / {base_console.TABBY_MODEL})")
        print(f"  timezone   : {base_console.TIMEZONE}")
        print("  Ctrl+C to stop")
        print()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nconsole stopped.")
    finally:
        for running in (panel_server, panel_tls_server):
            if running is not None:
                running.shutdown()
                running.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
