"""The play queue, owned by a service of its own.

The queue used to live inside the music skill: the console asked the provider
worker for it on every read, and the worker held a hundred tracks in one JSON
file. That is fine for "play this next" and wrong for a playlist: adding a
three-hour playlist meant one provider round trip per song, and the hundredth
song was refused outright.

This module is the queue itself, with no HTTP and no provider in it:

  * the whole line-up is stored in SQLite, so a playlist of any length is a
    handful of rows rather than a request per song;
  * metadata that the caller already knows (a playlist listing hands over its
    own titles) is stored as given, and only tracks that arrive as a bare id
    are left for the provider to fill in — in the background, at the rate the
    OpenDeck instance says it can take;
  * a *revision* is bumped on every change, so a page that polls can send the
    revision it holds and be told "unchanged" without the queue being sent
    again at all.

The service that serves it (``nix_core/music_queue_service.py``) is a thin
HTTP wrapper, and ``nix_core/music_api.py`` proxies the console's own queue
routes here when the service is running.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

# A track is stored as the fields the queue has always carried, so every
# existing reader (the page, the wall panel, the assistant) is unchanged.
TRACK_FIELDS = ("video_id", "title", "artist", "album", "duration_sec", "artwork_url")

# A track the provider does not know (a deleted video, an id that was typed)
# would otherwise be asked about on every tick, forever, spending the provider's
# rate limit on a song that is never going to arrive. It is asked a few times,
# each wait twice as long as the last, and then left alone: the entry stays in
# the queue saying what it is, and asking again is what re-adding it means.
MISS_LIMIT = 5
MISS_BACKOFF_BASE = 15.0
MISS_BACKOFF_MAX = 3600.0


def miss_backoff(misses: int) -> float:
    """How long to wait before asking about a track that failed this many times."""
    if misses <= 0:
        return 0.0
    return min(MISS_BACKOFF_BASE * (2 ** (misses - 1)), MISS_BACKOFF_MAX)


# Where the queue keeps its two files: the line-up, and the small note that says
# where the service is listening. Both sit in the console's data directory
# (``NIX_DATA_DIR``, which every other store here already follows and which the
# console exports for the processes it starts), so a box that was told to keep
# its data somewhere else finds its own queue there rather than in this
# checkout's. Both are read when they are used rather than at import, which is
# what lets a test point them at a directory of its own.
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def data_dir() -> str:
    """The directory the queue's database and published address live in."""
    return os.environ.get("NIX_DATA_DIR") or _DATA_DIR


def default_db() -> str:
    """Where a queue keeps its line-up unless it was told otherwise."""
    return os.environ.get("NIX_MUSIC_QUEUE_DB") or os.path.join(data_dir(), "music-queue.db")


def service_file() -> str:
    """Where the queue says where it is.

    The console reads its own environment, but a skill worker is started with
    almost none on purpose (see skill_runtime), so the service publishes its
    address in the data directory instead: one small file, readable only by its
    owner, that whoever needs to find the queue reads.
    """
    return os.environ.get("NIX_MUSIC_QUEUE_FILE") or os.path.join(data_dir(), "music-queue.json")


def publish_service(url: str, token: str = "", path: str | None = None) -> str | None:
    """Publish the service's own address, for callers that cannot read its env."""
    target = path or service_file()
    if not target or not str(url or "").startswith("http"):
        return None
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"url": str(url).rstrip("/"), "token": token, "pid": os.getpid(), "at": time.time()}
    try:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.chmod(target, 0o600)
    except OSError:
        # A service that cannot publish is still a service; this is only how
        # other processes find it.
        return None
    return target


def withdraw_service(url: str = "", path: str | None = None) -> None:
    """Remove the published address, if it is still this service's own."""
    target = path or service_file()
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return
    published = str(payload.get("url") or "").rstrip("/")
    if url and published != str(url).rstrip("/"):
        return
    try:
        os.remove(target)
    except OSError:
        pass


def discover_service(path: str | None = None, max_age: float | None = None) -> dict | None:
    """The queue service's address, as it published it, or None.

    ``max_age`` is deliberately opt-in: a stale file means a service that is not
    running, and the caller that finds it can decide whether that is a failure
    to report or a reason to fall back.
    """
    target = path or service_file()
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    url = str(payload.get("url") or "").strip().rstrip("/")
    if not url.startswith("http"):
        return None
    if max_age is not None:
        try:
            age = time.time() - float(payload.get("at") or 0)
        except (TypeError, ValueError):
            return None
        if age > max_age:
            return None
    return {"url": url, "token": str(payload.get("token") or "")}


@dataclass(frozen=True)
class QueueLimits:
    """What the queue is allowed to do, as the OpenDeck instance declares it.

    These are the provider's limits, not ours: how long a queue it will accept,
    how many tracks one write may carry, how many it is willing to resolve in a
    minute, and how far ahead of the playing track it wants the queue warm.
    The defaults are the conservative set the console falls back to when the
    installed skill declares nothing.
    """

    max_tracks: int = 5000          # the queue's own ceiling (was 100)
    max_batch: int = 200            # tracks one add may carry
    window: int = 50                # tracks one read hands the page
    prefetch: int = 25              # tracks to resolve ahead of the cursor
    prefetch_keep: int = 200        # resolved tracks kept in the metadata cache
    resolves_per_minute: int = 60   # provider lookups the background may make
    resolve_concurrency: int = 2    # ...and how many at once
    cache_seconds: int = 600        # how long a resolved track stays warm

    @staticmethod
    def _bounded(value: Any, low: int, high: int, fallback: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return fallback
        return max(low, min(high, number))

    @classmethod
    def from_provider(cls, provider: dict | None) -> "QueueLimits":
        """Read the limits an OpenDeck provider declares for this queue.

        The skill's manifest carries them under the media provider block, next
        to the actions the console already reads:

            "queue": {
              "tool": "opendeck_music",
              "arguments": {"action": "queue", "op": "list"},
              "limits": {"max_tracks": 5000, "prefetch": 25, ...}
            }

        Anything missing is left at its default, so a skill that declares
        nothing still gets a queue that works.
        """
        base = cls()
        if not isinstance(provider, dict):
            return base
        # The declared queue block is where they belong (next to the action that
        # reaches the queue); a flat "limits" is accepted too, so a skill that
        # declares them beside the provider's other actions still works.
        declared: Any = provider.get("limits")
        queue_block = provider.get("queue")
        if isinstance(queue_block, dict) and isinstance(queue_block.get("limits"), dict):
            declared = queue_block["limits"]
        if not isinstance(declared, dict):
            declared = {}
        return replace(
            base,
            max_tracks=cls._bounded(declared.get("max_tracks"), 1, 100000, base.max_tracks),
            max_batch=cls._bounded(declared.get("max_batch"), 1, 5000, base.max_batch),
            window=cls._bounded(declared.get("window"), 1, 1000, base.window),
            prefetch=cls._bounded(declared.get("prefetch"), 0, 1000, base.prefetch),
            prefetch_keep=cls._bounded(declared.get("prefetch_keep"), 0, 100000, base.prefetch_keep),
            resolves_per_minute=cls._bounded(
                declared.get("resolves_per_minute"), 0, 6000, base.resolves_per_minute
            ),
            resolve_concurrency=cls._bounded(
                declared.get("resolve_concurrency"), 1, 8, base.resolve_concurrency
            ),
            cache_seconds=cls._bounded(declared.get("cache_seconds"), 0, 86400, base.cache_seconds),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "max_tracks": self.max_tracks,
            "max_batch": self.max_batch,
            "window": self.window,
            "prefetch": self.prefetch,
            "prefetch_keep": self.prefetch_keep,
            "resolves_per_minute": self.resolves_per_minute,
            "resolve_concurrency": self.resolve_concurrency,
            "cache_seconds": self.cache_seconds,
        }


class Bucket:
    """A token bucket: how the queue keeps its provider calls to a rate.

    One token per resolved track, refilled at ``per_minute``. A bucket of zero
    means "no background resolving at all", which is what a provider that asks
    for no prefetch gets.
    """

    def __init__(self, per_minute: int, burst: int | None = None) -> None:
        self.rate = max(0.0, float(per_minute) / 60.0)
        self.capacity = float(burst if burst is not None else max(1, per_minute))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def take(self, count: int = 1) -> bool:
        if self.rate <= 0:
            return False
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= count:
                self.tokens -= count
                return True
            return False

    def retune(self, per_minute: int) -> None:
        with self._lock:
            self.rate = max(0.0, float(per_minute) / 60.0)
            self.capacity = float(max(1, per_minute))
            self.tokens = min(self.tokens, self.capacity)


def _text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return text.strip()[:limit]


def _duration(value: Any) -> int:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100000, seconds))


def _video_id(value: Any) -> str:
    return _text(value, 64)


def clean_track(raw: Any) -> dict[str, Any] | None:
    """One queue entry, reduced to exactly the fields the queue stores.

    A track with no id is not queueable — there would be nothing to play — and
    is dropped rather than stored as a hole.
    """
    if not isinstance(raw, dict):
        return None
    video_id = _video_id(raw.get("video_id") or raw.get("id"))
    if not video_id:
        return None
    return {
        "video_id": video_id,
        "title": _text(raw.get("title"), 300),
        "artist": _text(raw.get("artist"), 200),
        "album": _text(raw.get("album"), 300),
        "duration_sec": _duration(raw.get("duration_sec")),
        "artwork_url": _text(raw.get("artwork_url"), 500),
    }


def known(track: dict[str, Any]) -> bool:
    """Is this track's metadata already good enough to show without asking?

    A title is the page's minimum: a row with an id and nothing else would
    render as a blank line, so those are the ones the background resolves.
    """
    return bool(track.get("title"))


@dataclass
class QueueState:
    """What a reader is handed: a window of the queue, and its revision."""

    items: list[dict[str, Any]] = field(default_factory=list)
    offset: int = 0
    total: int = 0
    index: int = -1
    shuffle: bool = False
    repeat: str = "off"
    revision: int = 0
    unchanged: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "ok": True,
            "queue": self.items,
            "offset": self.offset,
            "count": len(self.items),
            "total": self.total,
            "index": self.index,
            "shuffle": self.shuffle,
            "repeat": self.repeat,
            "revision": self.revision,
            "unchanged": self.unchanged,
        }
        return payload


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    uid          INTEGER PRIMARY KEY AUTOINCREMENT,
    pos          INTEGER NOT NULL,
    video_id     TEXT    NOT NULL,
    title        TEXT    NOT NULL DEFAULT '',
    artist       TEXT    NOT NULL DEFAULT '',
    album        TEXT    NOT NULL DEFAULT '',
    duration_sec INTEGER NOT NULL DEFAULT 0,
    artwork_url  TEXT    NOT NULL DEFAULT '',
    playlist_id  TEXT    NOT NULL DEFAULT '',
    prepared     INTEGER NOT NULL DEFAULT 0,
    added_at     REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS items_pos ON items (pos);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tracks (
    video_id     TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    artist       TEXT NOT NULL DEFAULT '',
    album        TEXT NOT NULL DEFAULT '',
    duration_sec INTEGER NOT NULL DEFAULT 0,
    artwork_url  TEXT NOT NULL DEFAULT '',
    resolved_at  REAL NOT NULL DEFAULT 0,
    misses       INTEGER NOT NULL DEFAULT 0
);
"""


class MusicQueue:
    """The queue: a line-up, a cursor, and a warm metadata cache.

    Every read is a window plus a revision. A caller that already holds the
    revision it saw last time gets ``unchanged`` back and nothing else, which
    is what keeps a page polling this from asking the provider for anything at
    all.
    """

    def __init__(self, path: str | None = None, limits: QueueLimits | None = None) -> None:
        self.path = path if path is not None else default_db()
        self.path = self.path or ":memory:"
        self.limits = limits or QueueLimits()
        self._lock = threading.RLock()
        directory = os.path.dirname(self.path)
        if directory and self.path != ":memory:":
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self.bucket = Bucket(self.limits.resolves_per_minute)
        self.counters = {
            "adds": 0, "tracks_added": 0, "resolved": 0, "resolve_misses": 0,
            "reads": 0, "unchanged_reads": 0, "provider_calls": 0,
        }

    @property
    def published_file(self) -> str:
        """Where this queue's own address is published: beside its database.

        One file per queue, so two of them (a test's, a second box's) cannot
        overwrite each other's address; a queue with no file of its own falls
        back to the standard one.
        """
        if not self.path or self.path == ":memory:":
            return service_file()
        base, _extension = os.path.splitext(self.path)
        return f"{base}.json" if base else service_file()

    # ── storage helpers ───────────────────────────────────────────────────
    def _meta_get(self, key: str, fallback: str = "") -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else fallback

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _bump(self) -> int:
        revision = int(self._meta_get("revision", "0") or 0) + 1
        self._meta_set("revision", str(revision))
        return revision

    def revision(self) -> int:
        return int(self._meta_get("revision", "0") or 0)

    def _current_index(self) -> int:
        return int(self._meta_get("index", "-1") or -1)

    def _rows(self, offset: int = 0, limit: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM items ORDER BY pos"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, max(0, offset)]
        return list(self._conn.execute(sql, params))

    @staticmethod
    def _row_track(row: sqlite3.Row) -> dict[str, Any]:
        track = {"video_id": row["video_id"], "title": row["title"], "artist": row["artist"],
                 "album": row["album"], "duration_sec": row["duration_sec"],
                 "artwork_url": row["artwork_url"]}
        return track

    def total(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()
        return int(row["n"] if row else 0)

    # ── reads ─────────────────────────────────────────────────────────────
    def state(
        self,
        offset: int = 0,
        limit: int | None = None,
        revision: int | None = None,
        *,
        full: bool = False,
    ) -> QueueState:
        """A window of the queue, or "nothing changed" for a known revision.

        ``revision`` is what the caller last saw. When it still matches, the
        reply carries no tracks at all: the caller's copy is already the queue,
        so sending it again would be the very traffic this service exists to
        avoid.
        """
        with self._lock:
            self.counters["reads"] += 1
            current = self.revision()
            if revision is not None and revision == current and not full:
                self.counters["unchanged_reads"] += 1
                return QueueState(
                    offset=offset, total=self.total(), index=self._current_index(),
                    shuffle=self._meta_get("shuffle", "0") == "1",
                    repeat=self._meta_get("repeat", "off") or "off",
                    revision=current, unchanged=True,
                )
            window = limit if limit is not None else self.limits.window
            window = max(1, min(2000, window))
            offset = max(0, offset)
            rows = self._rows(offset, window)
            return QueueState(
                items=[self._row_track(row) for row in rows],
                offset=offset,
                total=self.total(),
                index=self._current_index(),
                shuffle=self._meta_get("shuffle", "0") == "1",
                repeat=self._meta_get("repeat", "off") or "off",
                revision=current,
            )

    def snapshot(self) -> list[dict[str, Any]]:
        """The whole line-up, for the console's own reply to an existing page."""
        with self._lock:
            return [self._row_track(row) for row in self._rows()]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            prepared = self._conn.execute(
                "SELECT COUNT(*) AS n FROM items WHERE prepared = 1").fetchone()["n"]
            # An entry the provider refused often enough that the queue stopped
            # asking: it is still in the line-up saying what it is, it is just
            # not costing the provider anything any more. Waiting for the
            # provider and given up on are therefore two different numbers, and
            # "pending" means the first one — what the background would ask about
            # if it looked right now.
            given_up = ("EXISTS (SELECT 1 FROM tracks WHERE tracks.video_id = items.video_id "
                        "AND tracks.misses >= ? AND tracks.resolved_at >= items.added_at)")
            pending = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM items WHERE prepared = 0 AND title = '' "
                f"AND NOT {given_up}", (MISS_LIMIT,)).fetchone()["n"]
            refused = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM items WHERE prepared = 0 AND title = '' "
                f"AND {given_up}", (MISS_LIMIT,)).fetchone()["n"]
            cached = self._conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]
            return {
                **self.counters,
                "total": self.total(),
                "prepared": int(prepared),
                "pending": int(pending),
                "unresolvable": int(refused),
                "cached_tracks": int(cached),
                "revision": self.revision(),
                "limits": self.limits.as_dict(),
                "db": self.path,
            }

    # ── writes ────────────────────────────────────────────────────────────
    def add_many(
        self,
        tracks: Iterable[Any],
        playlist_id: str = "",
        mode: str = "append",
    ) -> dict[str, Any]:
        """Queue a whole list at once — a playlist, or a whole library page.

        ``mode`` says where it lands: ``append`` (behind everything waiting),
        ``play_next`` (right after the playing track) or ``replace`` (the
        queue becomes this list). The return value says exactly what happened
        — how many were accepted, how many the ceiling refused, and how many
        were not queueable at all — because a caller that sent four hundred
        songs is owed a number rather than a shrug.
        """
        wanted = list(tracks or [])
        cleaned = [track for track in (clean_track(raw) for raw in wanted) if track]
        with self._lock:
            if mode == "replace":
                self._conn.execute("DELETE FROM items")
                self._meta_set("index", "-1")
                self._meta_set("shuffle_base", "[]")
                self._meta_set("shuffle", "0")
            room = self.limits.max_tracks - self.total()
            accepted = cleaned[: max(0, room)] if room > 0 else []
            # The playing track keeps its place: a list added "next" is spliced
            # in behind the cursor, and an appended list lands at the end, so
            # the song that is on stays the song that is on.
            start = self._insert_point(mode)
            # Room is made for a list that lands in the middle: everything from
            # the insert point down shifts by the length of the list, so a
            # "play next" splices in rather than overwriting what waited there.
            if accepted and start < self.total():
                self._conn.execute(
                    "UPDATE items SET pos = pos + ? WHERE pos >= ?", (len(accepted), start))
            now = time.time()
            rows: list[tuple] = []
            for position, track in enumerate(accepted, start=start):
                warmed = self._warm(track)
                rows.append((position, warmed["video_id"], warmed["title"], warmed["artist"],
                             warmed["album"], warmed["duration_sec"], warmed["artwork_url"],
                             _text(playlist_id, 100), 1 if known(warmed) else 0, now))
            self._conn.executemany(
                "INSERT INTO items (pos, video_id, title, artist, album, duration_sec, "
                "artwork_url, playlist_id, prepared, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            if mode == "replace":
                # A queue that is replaced starts at its first song; one that is
                # merely added to is still on whatever it was on.
                self._meta_set("index", "0" if accepted else "-1")
            elif mode == "play_next" and self._current_index() < 0 and accepted:
                self._meta_set("index", "-1")
            self._renumber()
            self._bump()
            self.counters["adds"] += 1
            self.counters["tracks_added"] += len(accepted)
            self._conn.commit()
            return {
                "ok": True,
                "accepted": len(accepted),
                "refused": len(cleaned) - len(accepted),
                "unqueueable": len(wanted) - len(cleaned),
                "revision": self.revision(),
                "total": self.total(),
                "index": self._current_index(),
                "mode": mode,
            }

    def _warm(self, track: dict[str, Any]) -> dict[str, Any]:
        """Fill gaps from the metadata cache, so a re-added song costs nothing."""
        if known(track):
            self._remember(track)
            return track
        row = self._conn.execute(
            "SELECT * FROM tracks WHERE video_id = ?", (track["video_id"],)).fetchone()
        if not row:
            return track
        if self.limits.cache_seconds and time.time() - row["resolved_at"] > self.limits.cache_seconds:
            return track
        merged = dict(track)
        for key in ("title", "artist", "album", "duration_sec", "artwork_url"):
            if not merged.get(key):
                merged[key] = row[key]
        return merged

    def _insert_point(self, mode: str) -> int:
        total = self.total()
        if mode != "play_next":
            return 0 if mode == "replace" else total
        index = self._current_index()
        if index < 0:
            return total
        return min(total, index + 1)

    def _renumber(self) -> None:
        rows = self._rows()
        self._conn.executemany(
            "UPDATE items SET pos = ? WHERE uid = ?",
            [(pos, row["uid"]) for pos, row in enumerate(rows)],
        )

    def remove(self, index: int) -> dict[str, Any]:
        with self._lock:
            rows = self._rows()
            if index < 0 or index >= len(rows):
                raise IndexError("That queue position does not exist.")
            self._conn.execute("DELETE FROM items WHERE uid = ?", (rows[index]["uid"],))
            current = self._current_index()
            if current == index:
                current = -1
            elif current > index:
                current -= 1
            self._meta_set("index", str(current))
            self._renumber()
            self._bump()
            self._conn.commit()
            return {"ok": True, "revision": self.revision(), "total": self.total(),
                    "index": current}

    def clear(self) -> dict[str, Any]:
        with self._lock:
            removed = self.total()
            self._conn.execute("DELETE FROM items")
            self._meta_set("index", "-1")
            self._meta_set("shuffle_base", "[]")
            self._bump()
            self._conn.commit()
            return {"ok": True, "removed": removed, "revision": self.revision(), "total": 0,
                    "index": -1}

    def move(self, index: int, to: int) -> dict[str, Any]:
        """Drag a row to a new position; the cursor follows the song it pointed at."""
        with self._lock:
            rows = self._rows()
            count = len(rows)
            if not (0 <= index < count and 0 <= to < count):
                raise IndexError("That queue position does not exist.")
            if index != to:
                order = [row["uid"] for row in rows]
                current = self._current_index()
                playing_uid = order[current] if 0 <= current < count else None
                moved = order.pop(index)
                order.insert(to, moved)
                self._conn.executemany(
                    "UPDATE items SET pos = ? WHERE uid = ?",
                    [(pos, uid) for pos, uid in enumerate(order)],
                )
                if playing_uid is not None:
                    current = order.index(playing_uid)
                self._meta_set("index", str(current))
            self._bump()
            self._conn.commit()
            return {"ok": True, "revision": self.revision(), "total": count,
                    "index": self._current_index()}

    def place(self, index: int, where: str) -> dict[str, Any]:
        """``front`` (first) or ``next_up`` (right behind the playing track)."""
        with self._lock:
            rows = self._rows()
            count = len(rows)
            if not 0 <= index < count:
                raise IndexError("That queue position does not exist.")
            order = [row["uid"] for row in rows]
            current = self._current_index()
            playing_uid = order[current] if 0 <= current < count else None
            moved = order.pop(index)
            target = 0 if where == "front" else (
                min(count - 1, order.index(playing_uid) + 1) if playing_uid in order else 0
            )
            order.insert(target, moved)
            self._conn.executemany(
                "UPDATE items SET pos = ? WHERE uid = ?",
                [(pos, uid) for pos, uid in enumerate(order)],
            )
            if playing_uid is not None:
                current = order.index(playing_uid)
            self._meta_set("index", str(current))
            self._bump()
            self._conn.commit()
            return {"ok": True, "revision": self.revision(), "total": count,
                    "index": self._current_index(), "position": target}

    def set_preference(self, key: str, value: str) -> dict[str, Any]:
        with self._lock:
            self._meta_set(key, value)
            self._bump()
            self._conn.commit()
            return {"ok": True, "revision": self.revision(), key: value}

    def set_index(self, index: int) -> dict[str, Any]:
        with self._lock:
            total = self.total()
            self._meta_set("index", str(index if -1 <= index < total else -1))
            self._bump()
            self._conn.commit()
            return {"ok": True, "revision": self.revision(), "index": self._current_index(),
                    "total": total}

    def start_next(self, *, auto: bool = False, drop_finished: bool = True) -> dict[str, Any]:
        """Move the cursor to the next queued track and hand it back.

        A track that has had its turn leaves the line-up — the queue is what is
        left to play, not a record of what has already played — unless repeat
        is on, which is the same rule the provider worker has always applied.
        """
        with self._lock:
            total = self.total()
            current = self._current_index()
            repeat = self._meta_get("repeat", "off") or "off"
            if total == 0:
                return {"ok": False, "code": "queue_empty", "revision": self.revision(),
                        "total": 0, "index": -1, "message": "The queue is empty."}
            if repeat == "one" and auto and 0 <= current < total:
                next_index = current
            else:
                next_index = current + 1 if current >= 0 else 0
                if next_index >= total and repeat == "all":
                    next_index = 0
            if next_index >= total:
                return {"ok": False, "code": "queue_end", "revision": self.revision(),
                        "total": total, "index": current,
                        "message": "The queue has no next track to play."}
            if drop_finished and repeat == "off" and 0 <= current < total and next_index > current:
                rows = self._rows()
                self._conn.execute("DELETE FROM items WHERE uid = ?", (rows[current]["uid"],))
                self._renumber()
                next_index -= 1
            self._meta_set("index", str(next_index))
            self._bump()
            self._conn.commit()
            rows = self._rows()
            track = self._row_track(rows[next_index]) if next_index < len(rows) else {}
            return {"ok": True, "revision": self.revision(), "total": self.total(),
                    "index": next_index, "track": track, "repeat": repeat,
                    "shuffle": self._meta_get("shuffle", "0") == "1"}

    def shuffle_on(self, on: bool) -> dict[str, Any]:
        with self._lock:
            self._meta_set("shuffle", "1" if on else "0")
            self._bump()
            self._conn.commit()
            return {"ok": True, "revision": self.revision(), "shuffle": bool(on)}

    # ── the metadata cache and what still needs resolving ─────────────────
    def _remember(self, track: dict[str, Any]) -> None:
        # A resolution clears the failures behind it: the provider knows this
        # track now, so nothing it said before about not knowing it applies.
        self._conn.execute(
            "INSERT INTO tracks (video_id, title, artist, album, duration_sec, artwork_url, "
            "resolved_at, misses) VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
            "ON CONFLICT(video_id) DO UPDATE SET title = excluded.title, artist = excluded.artist, "
            "album = excluded.album, duration_sec = excluded.duration_sec, "
            "artwork_url = excluded.artwork_url, resolved_at = excluded.resolved_at, misses = 0",
            (track["video_id"], track.get("title", ""), track.get("artist", ""),
             track.get("album", ""), track.get("duration_sec", 0), track.get("artwork_url", ""),
             time.time()),
        )

    def pending(self, limit: int | None = None) -> list[dict[str, Any]]:
        """The tracks the background should resolve next: the ones after the cursor.

        Only tracks that are still in front of the playing position are worth a
        provider call: a song already behind the cursor will never be reached,
        and resolving it would spend the provider's rate limit for nothing.

        A track the provider has refused is asked again more slowly each time and
        finally not at all (``MISS_LIMIT``), so one dead id cannot spend the
        provider's rate limit once a tick for the life of the queue. Queueing it
        again is a new request for it, and starts the counting over.
        """
        with self._lock:
            window = self.limits.prefetch if limit is None else limit
            if window <= 0:
                return []
            index = self._current_index()
            rows = self._rows()
            start = index + 1 if index >= 0 else 0
            out: list[dict[str, Any]] = []
            now = time.time()
            for row in rows[start:]:
                if known({"title": row["title"]}):
                    continue
                cached = self._conn.execute(
                    "SELECT * FROM tracks WHERE video_id = ?", (row["video_id"],)).fetchone()
                if cached and cached["title"] and (
                    not self.limits.cache_seconds
                    or now - cached["resolved_at"] <= self.limits.cache_seconds
                ):
                    continue
                misses = int(cached["misses"] or 0) if cached else 0
                # A track added again after it failed is a new request for it,
                # so the failures before that do not count against it.
                if cached and float(row["added_at"] or 0) > float(cached["resolved_at"] or 0):
                    misses = 0
                if misses >= MISS_LIMIT:
                    continue
                if misses and now - float(cached["resolved_at"] or 0) < miss_backoff(misses):
                    continue
                out.append({"video_id": row["video_id"], "uid": row["uid"],
                            "title": row["title"], "artist": row["artist"]})
                if len(out) >= window:
                    break
            return out

    def apply_resolution(self, video_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Store what the provider said about a track, and warm every row holding it."""
        track = clean_track({**fields, "video_id": video_id})
        if not track:
            raise ValueError("a resolved track needs a video_id")
        with self._lock:
            self._remember(track)
            self._conn.execute(
                "UPDATE items SET title = ?, artist = ?, album = ?, duration_sec = ?, "
                "artwork_url = ?, prepared = 1 WHERE video_id = ?",
                (track["title"], track["artist"], track["album"], track["duration_sec"],
                 track["artwork_url"], track["video_id"]),
            )
            self._prune_cache()
            self._bump()
            self._conn.commit()
            self.counters["resolved"] += 1
            return {"ok": True, "revision": self.revision(), "video_id": track["video_id"]}

    def apply_batch(self, tracks: Iterable[Any]) -> int:
        """Store a batch of resolutions in one transaction.

        A poll that resolves twenty tracks should cost one write, not twenty.
        """
        cleaned = [track for track in (clean_track(raw) for raw in tracks or []) if track]
        if not cleaned:
            return 0
        with self._lock:
            for track in cleaned:
                self._remember(track)
                self._conn.execute(
                    "UPDATE items SET title = ?, artist = ?, album = ?, duration_sec = ?, "
                    "artwork_url = ?, prepared = 1 WHERE video_id = ?",
                    (track["title"], track["artist"], track["album"], track["duration_sec"],
                     track["artwork_url"], track["video_id"]),
                )
            self._prune_cache()
            self._bump()
            self._conn.commit()
            self.counters["resolved"] += len(cleaned)
            return len(cleaned)

    def note_miss(self, video_id: str) -> None:
        """The provider did not know this track: remember that, so we stop asking."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO tracks (video_id, resolved_at, misses) VALUES (?, ?, 1) "
                "ON CONFLICT(video_id) DO UPDATE SET misses = misses + 1, resolved_at = excluded.resolved_at",
                (_video_id(video_id), time.time()),
            )
            self._conn.commit()
            self.counters["resolve_misses"] += 1

    def _prune_cache(self) -> None:
        keep = self.limits.prefetch_keep
        if keep <= 0:
            self._conn.execute("DELETE FROM tracks")
            return
        self._conn.execute(
            "DELETE FROM tracks WHERE video_id IN ("
            "  SELECT video_id FROM tracks ORDER BY resolved_at DESC LIMIT -1 OFFSET ?"
            ")",
            (keep,),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()


def limits_from_manifest(manifest: dict | None) -> QueueLimits:
    """The limits an installed skill's manifest declares for this queue."""
    if not isinstance(manifest, dict):
        return QueueLimits()
    device_ui = manifest.get("device_ui")
    if not isinstance(device_ui, dict):
        return QueueLimits()
    media = device_ui.get("media")
    if not isinstance(media, dict):
        return QueueLimits()
    provider = media.get("provider")
    return QueueLimits.from_provider(provider if isinstance(provider, dict) else None)


def dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)
