"""Local cache for music catalog reads and artwork covers.

Everything the console serves for music used to be a round trip: the page asked
``/api/music/library``, the console called the provider skill, the skill asked
OpenDeck, and only then could the page draw anything. The same request a second
later did that work again. This module keeps what the provider already answered:

* a catalog entry (playlists, a playlist's tracks, likes, history, search
  results) is replayed from disk immediately, and a background refresh asks the
  provider again — its reply replaces the cached copy when it arrives, so the
  next reader sees the new data without anyone having waited for it;
* an artwork cover is fetched once and served from disk afterwards, so the page
  paints covers from the console (and from the browser cache after the first
  paint) instead of a CDN round trip.

Both stores live under ``<NIX_DATA_DIR>/music-cache`` and are bounded, so a
console that has run for months holds a fixed amount of data. A refresh that
fails never destroys the copy already cached: the old copy keeps being served
and another attempt is made after ``RETRY_AFTER``, so an offline device is not
hammered by every page poll. Set ``NIX_MUSIC_CACHE=0`` to bypass all of it.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

import console

CACHE_DIR_NAME = "music-cache"
CATALOG_NAME = "catalog.json"
ARTWORK_DIR_NAME = "artwork"
ARTWORK_INDEX_NAME = "index.json"

# A cached catalog answer is replayed as-is for this long; once it is older, a
# reader still gets the cached copy but the provider is asked again right away in
# the background. Playlists and search results do not change second by second,
# and a write through the API invalidates the entries it affects.
DEFAULT_CATALOG_TTL = 300.0
# How long a failed refresh waits before being tried again.
RETRY_AFTER = 30.0

# Bounds. The catalog is small JSON; the artwork store is capped by count and by
# total size, and the oldest covers are dropped first.
MAX_CATALOG_ENTRIES = 96
MAX_ARTWORK_FILES = 256
MAX_ARTWORK_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ARTWORK_BYTES = 2 * 1024 * 1024
ARTWORK_TIMEOUT = 6.0
# Covers are keyed by their source URL, so the same key always means the same
# image and a browser may keep it for a month.
ARTWORK_BROWSER_SECONDS = 30 * 24 * 60 * 60
ALLOWED_ARTWORK_TYPES = (
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/avif",
    "image/bmp",
)
ARTWORK_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "image/bmp": ".bmp",
}
ARTWORK_USER_AGENT = "nix-puca-console/1.0 (music artwork cache)"

_lock = threading.RLock()
_inflight: dict[str, threading.Thread] = {}


class ArtworkError(Exception):
    """An artwork request that cannot be served, with a stable code and status."""

    def __init__(self, message: str, status: int = 502, code: str = "artwork_unavailable"):
        super().__init__(message)
        self.status = status
        self.code = code


# ---------------------------------------------------------------------------
# Configuration and paths (read at call time so a console can be told to use a
# different data directory, and so tests never touch the real one)
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """Whether reads and covers go through the cache; on unless switched off."""
    return str(os.environ.get("NIX_MUSIC_CACHE", "1")).strip().lower() not in {"0", "false", "no", "off"}


def catalog_ttl() -> float:
    try:
        ttl = float(os.environ.get("NIX_MUSIC_CACHE_TTL", DEFAULT_CATALOG_TTL))
    except (TypeError, ValueError):
        return DEFAULT_CATALOG_TTL
    return ttl if ttl > 0 else DEFAULT_CATALOG_TTL


def cache_dir() -> str:
    return os.path.join(console.DATA_DIR, CACHE_DIR_NAME)


def catalog_path() -> str:
    return os.path.join(cache_dir(), CATALOG_NAME)


def artwork_dir() -> str:
    return os.path.join(cache_dir(), ARTWORK_DIR_NAME)


def artwork_index_path() -> str:
    return os.path.join(artwork_dir(), ARTWORK_INDEX_NAME)


# ---------------------------------------------------------------------------
# Small JSON helpers: the cache is an optimisation, so an unreadable or
# unwritable file degrades to "no cached copy" instead of failing a request
# ---------------------------------------------------------------------------


def _read_json(path: str, default: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _write_json(path: str, payload: dict) -> bool:
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
        return True
    except OSError:
        return False


def _entry_key(namespace: str, key: str) -> str:
    return f"{namespace}|{key}"


# ---------------------------------------------------------------------------
# Catalog entries
# ---------------------------------------------------------------------------


def _catalog() -> dict:
    catalog = _read_json(catalog_path(), {"version": 1, "entries": {}})
    entries = catalog.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": 1, "entries": entries}


def lookup(namespace: str, key: str) -> dict | None:
    """The stored entry for one read, or None when nothing is cached."""
    with _lock:
        entry = _catalog()["entries"].get(_entry_key(namespace, key))
    return entry if isinstance(entry, dict) else None


def remember(namespace: str, key: str, payload: Any, saved_at: float | None = None) -> bool:
    """Store one provider answer, replacing any older copy of the same read."""
    if payload is None:
        return False
    with _lock:
        catalog = _catalog()
        catalog["entries"][_entry_key(namespace, key)] = {
            "saved_at": float(saved_at if saved_at is not None else time.time()),
            "payload": payload,
            "attempts": 0,
            "last_error": "",
            "last_attempt": 0.0,
        }
        _prune_catalog(catalog)
        return _write_json(catalog_path(), catalog)


def _prune_catalog(catalog: dict) -> None:
    """Keep the most recently stored entries; drop the oldest beyond the cap."""
    entries = catalog["entries"]
    if len(entries) <= MAX_CATALOG_ENTRIES:
        return
    ordered = sorted(entries.items(), key=lambda item: float((item[1] or {}).get("saved_at") or 0.0), reverse=True)
    catalog["entries"] = dict(ordered[:MAX_CATALOG_ENTRIES])


def _record_failure(namespace: str, key: str, error: Exception) -> None:
    """Note a failed refresh and when it happened, so the next one backs off."""
    with _lock:
        catalog = _catalog()
        entry = catalog["entries"].get(_entry_key(namespace, key))
        if not isinstance(entry, dict):
            return
        entry["attempts"] = int(entry.get("attempts") or 0) + 1
        entry["last_attempt"] = time.time()
        entry["last_error"] = f"{type(error).__name__}: {str(error)[:160]}"
        _write_json(catalog_path(), catalog)


def backing_off(entry: dict) -> bool:
    """Whether a copy whose refresh just failed should be left alone for now."""
    if not entry.get("last_error"):
        return False
    last_attempt = entry.get("last_attempt")
    if not isinstance(last_attempt, (int, float)):
        return False
    return (time.time() - float(last_attempt)) < RETRY_AFTER


def invalidate(namespace: str | None = None) -> int:
    """Drop cached reads whose data just changed; all of them by default."""
    with _lock:
        catalog = _catalog()
        entries = catalog["entries"]
        if namespace is None:
            dropped = len(entries)
            catalog["entries"] = {}
        else:
            prefix = f"{namespace}|"
            keep = {key: value for key, value in entries.items() if not key.startswith(prefix)}
            dropped = len(entries) - len(keep)
            catalog["entries"] = keep
        if dropped:
            _write_json(catalog_path(), catalog)
        return dropped


def _age(entry: dict) -> float:
    saved_at = entry.get("saved_at")
    if not isinstance(saved_at, (int, float)):
        return float("inf")
    return max(0.0, time.time() - float(saved_at))


def _cacheable(payload: Any) -> bool:
    """Only a real answer is worth remembering; an error reply is not."""
    return isinstance(payload, dict) and bool(payload) and payload.get("ok") is not False


def _refresh_later(namespace: str, key: str, fetch: Callable[[], Any]) -> None:
    """Ask the provider again in the background, once per read at a time."""
    token = _entry_key(namespace, key)
    with _lock:
        if token in _inflight:
            return
        thread = threading.Thread(
            target=_refresh_worker,
            args=(token, namespace, key, fetch),
            name=f"music-cache-{namespace}",
            daemon=True,
        )
        _inflight[token] = thread
    thread.start()


def _refresh_worker(token: str, namespace: str, key: str, fetch: Callable[[], Any]) -> None:
    try:
        payload = fetch()
    except Exception as exc:  # a refresh must never take the caller down with it
        _record_failure(namespace, key, exc)
    else:
        if _cacheable(payload):
            remember(namespace, key, payload)
    finally:
        with _lock:
            _inflight.pop(token, None)


def wait_idle(timeout: float = 5.0) -> bool:
    """Wait for background refreshes to finish; ``False`` if one is still running.

    Used by tests and by shutdown paths that want the store settled.
    """
    deadline = time.time() + max(0.0, timeout)
    while True:
        with _lock:
            running = [thread for thread in _inflight.values() if thread.is_alive()]
        if not running:
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        for thread in running:
            thread.join(min(0.25, remaining))


def read(
    namespace: str,
    key: str,
    fetch: Callable[[], Any],
    ttl: float | None = None,
    refresh: bool = False,
) -> tuple[Any, str, float | None]:
    """Read one provider result through the cache.

    Returns ``(payload, status, age)`` where status is one of:

    ``hit``      the cached copy is younger than the TTL and was replayed;
    ``stale``    an older copy was replayed and a background refresh started;
    ``miss``     nothing was cached, so the provider answered synchronously;
    ``refresh``  the caller asked for fresh data, which was stored and returned;
    ``disabled`` the cache is switched off, so the provider answered directly.

    ``age`` is the age in seconds of the copy that was replayed, or None when the
    answer came straight from the provider. A provider failure is raised as-is
    when there is nothing cached to fall back on; otherwise the cached copy is
    returned and the failure is recorded for the next refresh to back off from.
    """
    if not enabled():
        return fetch(), "disabled", None

    ttl = catalog_ttl() if ttl is None else float(ttl)
    entry = lookup(namespace, key)
    payload = entry.get("payload") if isinstance(entry, dict) else None
    age = _age(entry) if isinstance(entry, dict) else None

    if entry is None or payload is None or refresh:
        try:
            fresh = fetch()
        except Exception as exc:
            if entry is None or payload is None:
                raise
            _record_failure(namespace, key, exc)
            return payload, "stale", age
        if _cacheable(fresh):
            remember(namespace, key, fresh)
        return fresh, ("refresh" if entry is not None else "miss"), None

    if age is not None and age > ttl:
        if not backing_off(entry):
            _refresh_later(namespace, key, fetch)
        return payload, "stale", age

    return payload, "hit", age


# ---------------------------------------------------------------------------
# Artwork covers
# ---------------------------------------------------------------------------


def artwork_key(url: str) -> str:
    """The stable cache key of one cover URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def _etag(key: str, saved_at: float) -> str:
    return f'"{key}-{int(saved_at)}"'


def _artwork_index() -> dict:
    index = _read_json(artwork_index_path(), {"version": 1, "entries": {}})
    entries = index.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": 1, "entries": entries}


def _artwork_entry(key: str) -> dict | None:
    entry = _artwork_index()["entries"].get(key)
    return entry if isinstance(entry, dict) else None


def _artwork_file(entry: dict) -> str:
    name = entry.get("file")
    return os.path.join(artwork_dir(), os.path.basename(str(name))) if name else ""


def _read_artwork_file(entry: dict) -> bytes | None:
    path = _artwork_file(entry)
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _prune_artwork(index: dict) -> None:
    """Keep the newest covers inside both caps, deleting the files that go."""
    entries = index["entries"]
    ordered = sorted(entries.items(), key=lambda item: float((item[1] or {}).get("saved_at") or 0.0), reverse=True)
    keep: dict = {}
    total = 0
    for key, entry in ordered:
        size = int((entry or {}).get("bytes") or 0)
        if len(keep) >= MAX_ARTWORK_FILES or (keep and total + size > MAX_ARTWORK_TOTAL_BYTES):
            path = _artwork_file(entry or {})
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            continue
        keep[key] = entry
        total += size
    index["entries"] = keep


def store_artwork(url: str, body: bytes, content_type: str, saved_at: float | None = None) -> str:
    """Write one cover to disk and index it; return its ETag."""
    key = artwork_key(url)
    stamp = float(saved_at if saved_at is not None else time.time())
    etag = _etag(key, stamp)
    with _lock:
        directory = artwork_dir()
        try:
            os.makedirs(directory, exist_ok=True)
            name = key + ARTWORK_EXTENSIONS.get(content_type, ".img")
            path = os.path.join(directory, name)
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "wb") as handle:
                handle.write(body)
            os.replace(temporary, path)
        except OSError:
            return etag
        index = _artwork_index()
        index["entries"][key] = {
            "url": url,
            "file": os.path.basename(path),
            "content_type": content_type,
            "bytes": len(body),
            "saved_at": stamp,
            "etag": etag,
        }
        _prune_artwork(index)
        _write_json(artwork_index_path(), index)
    return etag


def _require_public_host(host: str) -> None:
    """Refuse to fetch from the console's own networks.

    The artwork route takes a URL from a provider (and, on a console that is
    reachable from a browser, from whoever calls it). Without this check the
    console would fetch any address it can reach — a router page, a metadata
    endpoint, another service on the LAN — and hand the reply back, so only
    addresses that resolve to the public internet are allowed.
    """
    candidates = {host}
    try:
        for info in socket.getaddrinfo(host, None):
            candidates.add(str(info[4][0]))
    except OSError as exc:
        raise ArtworkError(f"The artwork host could not be resolved ({host}).") from exc
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate.split("%")[0])
        except ValueError:
            continue
        if not address.is_global:
            raise ArtworkError("Artwork must be served from a public address.", 400, "artwork_unavailable")


def download_artwork(url: str, timeout: float = ARTWORK_TIMEOUT) -> tuple[bytes, str]:
    """Fetch one cover from a public address, bounded in time and in size."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ArtworkError("Artwork must be an http(s) URL.", 400, "validation_error")
    _require_public_host(parsed.hostname)
    request = urllib.request.Request(url, headers={"User-Agent": ARTWORK_USER_AGENT, "Accept": "image/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final = urllib.parse.urlsplit(response.geturl() or url)
            if final.hostname and final.hostname != parsed.hostname:
                # A redirect must not be a way around the public-address rule.
                _require_public_host(final.hostname)
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_ARTWORK_TYPES:
                raise ArtworkError(
                    f"The artwork URL did not answer with an image ({content_type or 'no content type'})."
                )
            body = response.read(MAX_ARTWORK_BYTES + 1)
    except ArtworkError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ArtworkError(f"The artwork could not be fetched: {exc}") from exc
    return body, content_type


def artwork(url: str, fetch: Callable[[str], tuple[bytes, str]] | None = None) -> tuple[bytes, str, str, str]:
    """Serve one cover: ``(body, content_type, etag, status)``.

    A cover already on disk is returned without touching the network; anything
    else is fetched once, validated and stored. ``status`` is ``hit`` or
    ``miss``, which the route reports to the page.
    """
    if not isinstance(url, str) or not url:
        raise ArtworkError("Provide src=<image url>.", 400, "validation_error")
    key = artwork_key(url)
    entry = _artwork_entry(key)
    if enabled() and entry is not None:
        body = _read_artwork_file(entry)
        if body:
            saved_at = float(entry.get("saved_at") or 0.0)
            content_type = str(entry.get("content_type") or "image/jpeg")
            return body, content_type, str(entry.get("etag") or _etag(key, saved_at)), "hit"
    body, content_type = (fetch or download_artwork)(url)
    if not body:
        raise ArtworkError("The artwork response was empty.")
    if len(body) > MAX_ARTWORK_BYTES:
        raise ArtworkError("The artwork image is too large to cache.", 502, "artwork_too_large")
    content_type = str(content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_ARTWORK_TYPES:
        raise ArtworkError(f"The artwork URL did not answer with an image ({content_type or 'no content type'}).")
    etag = _etag(key, time.time())
    if enabled():
        etag = store_artwork(url, bytes(body), content_type)
    return bytes(body), content_type, etag, "miss"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def stats() -> dict:
    """What the cache currently holds, for the API index and for diagnostics."""
    catalog = _catalog()
    index = _artwork_index()
    total = sum(int((entry or {}).get("bytes") or 0) for entry in index["entries"].values())
    return {
        "enabled": enabled(),
        "dir": cache_dir(),
        "ttl_sec": catalog_ttl(),
        "catalog_entries": len(catalog["entries"]),
        "artwork_files": len(index["entries"]),
        "artwork_bytes": total,
    }
