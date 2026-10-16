"""The music queue as a service of its own.

``nix_core/music_queue.py`` is the queue; this is the process that owns it.

    python3 nix_core/music_queue_service.py --port 49119 --db /root/data/music-queue.db

Why a separate process rather than a module inside the console:

  * the queue outlives the console. A restart of the dashboard, a skill update,
    or an OpenDeck reinstall no longer touches the line-up;
  * resolving a long playlist is a background job, and a background job belongs
    in its own process, not in the request path of the UI that is polling it;
  * the console stops being the queue's memory. It proxies reads and writes,
    and the provider is asked once per track that actually needs asking.

The provider is reached through a *resolver* — in production a call back into
the console, which is where the installed skill lives:

    service = QueueService(port, db, limits, resolver=console_resolver)

so this file never holds provider credentials, and a test can pass a plain
function instead. The rate limit, the cache and the "only resolve what is in
front of the cursor" rule live here, which is what keeps a long playlist from
turning into a burst of provider calls.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import music_queue
from music_queue import MusicQueue, QueueLimits

DEFAULT_PORT = int(os.environ.get("NIX_MUSIC_QUEUE_PORT", "49119"))
TOKEN = os.environ.get("NIX_MUSIC_QUEUE_TOKEN", "").strip()
# How often the background looks for tracks to warm. One second is plenty: the
# work itself is rate limited, this is only how quickly new tracks are noticed.
TICK_SECONDS = float(os.environ.get("NIX_MUSIC_QUEUE_TICK", "1.0"))
# How long the background leaves the provider alone after the provider fails to
# answer at all (as opposed to answering that it does not know a track).
PROVIDER_COOLDOWN_SECONDS = float(os.environ.get("NIX_MUSIC_QUEUE_PROVIDER_COOLDOWN", "30"))
MAX_BODY = 2 * 1024 * 1024


class ProviderUnavailable(RuntimeError):
    """The provider could not be reached at all, so it refused nothing.

    The difference matters to the misses rule: a track the provider refuses is
    asked about again more slowly and finally not at all, but a provider that is
    not answering has said nothing about the track and nothing is counted
    against it. The background waits out a short cooldown instead of asking about
    the next track, and asks again once the provider is back.
    """


class QueueService:
    """The queue, plus the background that keeps it warm."""

    def __init__(
        self,
        queue: MusicQueue,
        resolver: Callable[[str], dict | None] | None = None,
        console_url: str = "",
        cooldown: float = PROVIDER_COOLDOWN_SECONDS,
    ) -> None:
        self.queue = queue
        self.resolver = resolver
        self.console_url = console_url.rstrip("/")
        self.cooldown = max(0.0, float(cooldown))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._inflight = threading.BoundedSemaphore(queue.limits.resolve_concurrency)
        self.resolved_calls = 0
        self.provider_errors = 0
        self._paused_until = 0.0

    # ── limits ────────────────────────────────────────────────────────────
    def set_limits(self, limits: dict | None) -> QueueLimits:
        """Adopt the limits the OpenDeck instance declares.

        The console pushes these whenever it reads the installed manifest, so
        the queue follows the provider it is actually queued against instead of
        a number baked in here.
        """
        provider = {"limits": limits} if isinstance(limits, dict) else None
        adopted = QueueLimits.from_provider(provider)
        self.queue.limits = adopted
        self.queue.bucket.retune(adopted.resolves_per_minute)
        return adopted

    # ── queue operations ──────────────────────────────────────────────────
    def queue_op(self, body: dict) -> dict:
        """Run one queue operation, in the provider's own vocabulary.

        The names are the ones the provider skill declares (``remove``,
        ``move``, ``front``, ``next_up``, ``clear``, ``shuffle``, ``repeat``,
        ``play``, ``next``), so a caller that already speaks them — the page,
        the wall panel, the assistant's tool — gains no new language by talking
        to the queue's own service.
        """
        queue = self.queue
        op = str(body.get("op") or body.get("action") or "").strip().lower()
        index = body.get("index")
        index = int(index) if isinstance(index, (int, float)) else None
        if op == "list":
            return queue.state(offset=0, limit=2000).as_dict()
        if op == "remove":
            if index is None:
                raise ValueError("remove needs index=<queue position>")
            return {**queue.remove(index), "message": "Removed from the queue."}
        if op == "move":
            to = body.get("to")
            to = int(to) if isinstance(to, (int, float)) else None
            if index is None or to is None:
                raise ValueError("move needs index=<from> and to=<to>")
            return {**queue.move(index, to), "message": "Queue reordered."}
        if op in ("front", "next_up"):
            if index is None:
                raise ValueError(f"{op} needs index=<queue position>")
            result = queue.place(index, op)
            return {**result, "message": "Moved to the front of the queue." if op == "front"
                    else "Playing next."}
        if op == "clear":
            return {**queue.clear(), "message": "Queue cleared."}
        if op == "shuffle":
            wanted = body.get("on")
            on = (not queue.state().shuffle) if wanted is None else bool(wanted)
            return {**queue.shuffle_on(on), "message": "Shuffling the queue." if on
                    else "Playing the queue in order."}
        if op == "repeat":
            mode = body.get("mode")
            if mode is None:
                mode = {"off": "all", "all": "one", "one": "off"}[queue.state().repeat]
            if mode not in ("off", "all", "one"):
                raise ValueError("repeat mode must be off, all, or one")
            return {**queue.set_preference("repeat", mode), "message": f"Repeat: {mode}."}
        if op in ("set_index", "play"):
            if index is None:
                raise ValueError(f"{op} needs index=<queue position>")
            return {**queue.set_index(index), "message": "Queue position selected."}
        if op == "next":
            return queue.start_next(auto=bool(body.get("auto")))
        raise ValueError(f"unknown queue op: {op or '(none)'}")

    # ── the background ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="queue-prefetch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.warm()
            except Exception:  # a warm loop must never take the queue down
                pass
            self._stop.wait(TICK_SECONDS)

    def pending(self) -> list[dict[str, Any]]:
        return self.queue.pending()

    def warm(self, limit: int | None = None) -> int:
        """Resolve the next few tracks, within the provider's rate limit.

        Each track costs one token from the bucket and one resolver call. When
        the bucket is empty the work simply waits for the next tick, which is
        the whole point: a playlist of five hundred songs resolves over minutes
        at the provider's pace rather than as one storm of requests.

        A resolver that raises is a provider that did not answer, which counts
        against nothing but the clock: see ``ProviderUnavailable``. The pause
        and the cooldown are read here rather than in the loop so a caller
        (``POST /warm``) gets the same restraint the background has.
        """
        if self.resolver is None:
            return 0
        if time.time() < self._paused_until:
            return 0
        wanted = self.pending() if limit is None else self.queue.pending(limit)
        batches: list[dict[str, Any]] = []
        for entry in wanted:
            if not self.queue.bucket.take():
                break
            if not self._inflight.acquire(blocking=False):
                self.queue.bucket.tokens = min(self.queue.bucket.capacity, self.queue.bucket.tokens + 1)
                break
            try:
                resolved = self.resolver(entry["video_id"])
            except Exception:
                # The provider did not answer, so it refused nothing: the track
                # keeps its place and its clean record, the token is handed
                # back, and the pass stops here rather than putting the same
                # question about every other track to a provider that is down.
                self.provider_errors += 1
                self._paused_until = time.time() + self.cooldown
                self.queue.bucket.tokens = min(self.queue.bucket.capacity,
                                               self.queue.bucket.tokens + 1)
                break
            finally:
                self._inflight.release()
            self.resolved_calls += 1
            self.queue.counters["provider_calls"] += 1
            if resolved:
                batches.append({**resolved, "video_id": entry["video_id"]})
            else:
                self.queue.note_miss(entry["video_id"])
        if batches:
            # One write for the whole poll: the page reading the queue in the
            # meantime sees either the old window or the warm one, never half a
            # transaction's worth of rows.
            self.queue.apply_batch(batches)
        return len(batches)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def console_resolver(console_url: str, token: str = "", timeout: float = 20.0):
    """A resolver that asks the console, which asks the installed provider skill.

    The console is where the provider lives, so the queue never needs the
    skill's own address or credentials — one plain call, one track.
    """
    url = console_url.rstrip("/") + "/api/music/queue/resolve"

    def resolver(video_id: str) -> dict | None:
        body = json.dumps({"video_id": video_id}).encode()
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            **({"X-Nix-Token": token} if token else {}),
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            # A 404 is the provider answering — it does not know this track, and
            # that is a miss. Anything else is the console or the skill in
            # trouble, which says nothing about the track, so it is raised as
            # the provider being unavailable rather than counted against it.
            if exc.code == 404:
                return None
            raise ProviderUnavailable(f"the console answered {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ProviderUnavailable(str(exc)) from exc
        track = payload.get("track") if isinstance(payload, dict) else None
        if isinstance(track, dict) and track.get("title"):
            return track
        return None

    return resolver


def _handler_for(service: QueueService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NixMusicQueue/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the default stderr line
            pass

        # ── plumbing ─────────────────────────────────────────────────────
        def _send(self, payload: dict, status: int = 200) -> None:
            body = music_queue.dump_json(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # A queue read is a live read; the revision is what makes caching
            # safe, so HTTP caching is off and the caller's revision is asked.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            if not TOKEN:
                return True
            return self.headers.get("X-Nix-Token", "") == TOKEN

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return {}
            if length <= 0 or length > MAX_BODY:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return {}

        def _query(self) -> dict[str, str]:
            from urllib.parse import parse_qs, urlparse
            return {key: values[0] for key, values in parse_qs(urlparse(self.path).query).items()}

        def _number(self, raw: Any, fallback: int, low: int, high: int) -> int:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return fallback
            return max(low, min(high, value))

        # ── routes ───────────────────────────────────────────────────────
        def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
            if not self._authorised():
                return self._send({"ok": False, "error": "unauthorised"}, 401)
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/"):
                return self._send({"ok": True, "service": "music-queue",
                                   "revision": service.queue.revision(),
                                   "limits": service.queue.limits.as_dict()})
            if path == "/state":
                query = self._query()
                revision = query.get("revision")
                return self._send(service.queue.state(
                    offset=self._number(query.get("from"), 0, 0, 1_000_000),
                    limit=self._number(query.get("limit"), service.queue.limits.window, 1, 2000),
                    revision=None if revision in (None, "") else self._number(revision, -1, 0, 2**31),
                ).as_dict())
            if path == "/all":
                return self._send({"ok": True, "queue": service.queue.snapshot(),
                                   "total": service.queue.total(),
                                   "index": service.queue.state().index,
                                   "revision": service.queue.revision()})
            if path == "/pending":
                limit = self._query().get("limit")
                return self._send({"ok": True,
                                   "pending": service.pending() if limit is None
                                   else service.queue.pending(self._number(limit, 0, 0, 500))})
            if path == "/stats":
                return self._send({"ok": True, **service.queue.stats(),
                                   "resolver_calls": service.resolved_calls,
                                   "provider_errors": service.provider_errors})
            return self._send({"ok": False, "error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorised():
                return self._send({"ok": False, "error": "unauthorised"}, 401)
            path = self.path.split("?", 1)[0]
            body = self._body()
            try:
                if path == "/add":
                    tracks = body.get("tracks")
                    if not isinstance(tracks, list):
                        raw_ids = body.get("video_ids")
                        tracks = [{"video_id": item} for item in raw_ids] if isinstance(raw_ids, list) else []
                    mode = str(body.get("mode") or "append")
                    if mode not in ("append", "play_next", "replace"):
                        return self._send({"ok": False, "error": "mode must be append, play_next or replace"}, 400)
                    if not tracks:
                        return self._send({"ok": False, "error": "no tracks to queue"}, 400)
                    if len(tracks) > service.queue.limits.max_batch * 20:
                        # Even a very long playlist is read in a few calls, which
                        # is what the ceiling is for: one request cannot be an
                        # unbounded allocation.
                        return self._send({"ok": False, "error": "too many tracks in one write"}, 413)
                    return self._send(service.queue.add_many(
                        tracks, playlist_id=str(body.get("playlist_id") or ""), mode=mode))
                if path == "/limits":
                    limits = body.get("limits")
                    adopted = service.set_limits(limits if isinstance(limits, dict) else None)
                    return self._send({"ok": True, "limits": adopted.as_dict()})
                if path == "/resolutions":
                    tracks = body.get("tracks")
                    count = service.queue.apply_batch(tracks if isinstance(tracks, list) else [])
                    return self._send({"ok": True, "applied": count,
                                       "revision": service.queue.revision()})
                if path == "/warm":
                    limit = body.get("limit")
                    return self._send({"ok": True, "warmed": service.warm(
                        None if limit is None else self._number(limit, 0, 0, 500))})
                if path == "/ops":
                    return self._send(service.queue_op(body))
            except IndexError as exc:
                return self._send({"ok": False, "error": str(exc)}, 404)
            except ValueError as exc:
                return self._send({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - defensive
                return self._send({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            return self._send({"ok": False, "error": "not found"}, 404)

    return Handler


def serve(
    port: int = DEFAULT_PORT,
    db: str | None = None,
    console_url: str = "",
    limits: dict | None = None,
    token: str = TOKEN,
) -> ThreadingHTTPServer:
    """Build (but do not start) the queue server, with its background warmed."""
    queue = MusicQueue(db or music_queue.default_db(),
                       QueueLimits.from_provider({"limits": limits} if limits else None))
    resolver = console_resolver(console_url, token) if console_url else None
    service = QueueService(queue, resolver=resolver, console_url=console_url)
    service.start()
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler_for(service))
    server.queue_service = service  # type: ignore[attr-defined]
    # Publish the address other processes find the queue by: the console reads
    # its own environment, but a skill worker is started with almost none.
    music_queue.publish_service(f"http://127.0.0.1:{port}", token, queue.published_file)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NIX music queue service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--db", default=music_queue.default_db())
    parser.add_argument("--console", default=os.environ.get("NIX_CONSOLE_URL", ""))
    parser.add_argument("--no-prefetch", action="store_true")
    args = parser.parse_args(argv)
    # An empty --db (a systemd unit whose variable is unset reaches here as an
    # empty argument) must not quietly become an in-memory queue: the line-up
    # has to survive the service restarting.
    if not args.db:
        args.db = music_queue.default_db()
    server = serve(
        port=args.port,
        db=args.db,
        console_url="" if args.no_prefetch else args.console,
    )
    print(f"music queue on http://0.0.0.0:{args.port} · db={args.db} · "
          f"console={args.console or 'off'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        # A withdrawn address is what stops a stopped service from being
        # discovered as if it were still there.
        music_queue.withdraw_service(f"http://127.0.0.1:{args.port}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
