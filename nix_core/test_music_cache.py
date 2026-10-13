"""Tests for the console-side music cache.

The cache is why a second look at the Music page does not wait on the device: a
catalog read is replayed from disk, a cover is served from disk, and a background
refresh replaces what it can. These tests pin the parts that are easy to get
wrong — serving the stale copy while refreshing, never losing a good copy to a
failed refresh, and refusing to fetch anything that is not a public image — plus
the bounds that keep the store from growing forever.
"""
import json
import threading
import time
from pathlib import Path

import pytest

import console
import music_cache


@pytest.fixture
def cache_env(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("NIX_MUSIC_CACHE", raising=False)
    monkeypatch.delenv("NIX_MUSIC_CACHE_TTL", raising=False)
    return tmp_path


def counting_fetch(payloads):
    """A provider stub that answers with the next payload and counts its calls."""
    calls = []

    def fetch():
        calls.append(len(calls) + 1)
        return payloads[min(len(calls) - 1, len(payloads) - 1)]

    fetch.calls = calls
    return fetch


# ── catalog entries ──────────────────────────────────────────────────────


def test_a_second_read_comes_from_the_cache_without_asking_again(cache_env):
    fetch = counting_fetch([{"ok": True, "count": 1}])
    payload, status, age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "miss" and payload["count"] == 1 and age is None

    payload, status, age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "hit" and payload["count"] == 1
    assert age is not None and age < 5
    assert len(fetch.calls) == 1


def test_a_stale_copy_answers_immediately_and_is_refreshed_behind_it(cache_env):
    music_cache.remember("library", "kind=playlists", {"ok": True, "count": 1}, saved_at=time.time() - 3600)
    fetch = counting_fetch([{"ok": True, "count": 2}])

    payload, status, age = music_cache.read("library", "kind=playlists", fetch, ttl=60)
    # The reader gets what is on disk, not a wait for the provider.
    assert status == "stale" and payload["count"] == 1
    assert age is not None and age >= 3600

    assert music_cache.wait_idle(5.0) is True
    assert len(fetch.calls) == 1
    stored = music_cache.lookup("library", "kind=playlists")
    assert stored["payload"]["count"] == 2
    assert stored["saved_at"] > time.time() - 30

    payload, status, _age = music_cache.read("library", "kind=playlists", fetch, ttl=60)
    assert status == "hit" and payload["count"] == 2
    assert len(fetch.calls) == 1


def test_a_failed_refresh_keeps_the_cached_copy_and_backs_off(cache_env):
    music_cache.remember("library", "kind=likes", {"ok": True, "count": 3}, saved_at=time.time() - 3600)
    calls = []

    def fetch():
        calls.append(1)
        raise RuntimeError("device unreachable")

    payload, status, _age = music_cache.read("library", "kind=likes", fetch, ttl=60)
    assert status == "stale" and payload["count"] == 3
    assert music_cache.wait_idle(5.0) is True

    entry = music_cache.lookup("library", "kind=likes")
    assert entry["payload"]["count"] == 3  # a failed refresh never clobbers a good copy
    assert "device unreachable" in entry["last_error"]
    assert entry["attempts"] == 1

    # A reader inside the back-off window is served the copy without another try.
    payload, status, _age = music_cache.read("library", "kind=likes", fetch, ttl=60)
    assert status == "stale" and payload["count"] == 3
    assert len(calls) == 1


def test_a_stale_copy_starts_only_one_refresh_at_a_time(cache_env):
    music_cache.remember("library", "kind=likes", {"ok": True, "count": 1}, saved_at=time.time() - 3600)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fetch():
        calls.append(1)
        started.set()
        release.wait(5)
        return {"ok": True, "count": 2}

    music_cache.read("library", "kind=likes", fetch, ttl=60)
    assert started.wait(5) is True
    music_cache.read("library", "kind=likes", fetch, ttl=60)
    music_cache.read("library", "kind=likes", fetch, ttl=60)
    release.set()
    assert music_cache.wait_idle(5.0) is True
    assert len(calls) == 1


def test_refresh_forces_a_synchronous_reload(cache_env):
    music_cache.remember("library", "kind=playlists", {"ok": True, "count": 1})
    payload, status, age = music_cache.read("library", "kind=playlists", counting_fetch([{"ok": True, "count": 2}]), refresh=True)
    assert status == "refresh" and payload["count"] == 2 and age is None
    assert music_cache.lookup("library", "kind=playlists")["payload"]["count"] == 2


def test_a_refresh_that_fails_still_serves_the_cached_copy(cache_env):
    music_cache.remember("library", "kind=playlists", {"ok": True, "count": 1})

    def fetch():
        raise RuntimeError("provider down")

    payload, status, age = music_cache.read("library", "kind=playlists", fetch, refresh=True)
    assert status == "stale" and payload["count"] == 1 and age is not None


def test_a_miss_that_fails_reports_the_provider_error(cache_env):
    def fetch():
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        music_cache.read("library", "kind=playlists", fetch)
    assert music_cache.lookup("library", "kind=playlists") is None


def test_an_error_reply_is_not_remembered(cache_env):
    fetch = counting_fetch([{"ok": False, "error": "provider busy"}])
    payload, status, _age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "miss" and payload["ok"] is False

    payload, status, _age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "miss" and len(fetch.calls) == 2
    assert music_cache.lookup("library", "kind=playlists") is None


def test_writes_invalidate_the_cached_catalog(cache_env):
    music_cache.remember("library", "kind=likes", {"ok": True})
    music_cache.remember("search", "query=Miles", {"ok": True})

    assert music_cache.invalidate("library") == 1
    assert music_cache.lookup("library", "kind=likes") is None
    assert music_cache.lookup("search", "query=Miles") is not None

    assert music_cache.invalidate() == 1
    assert music_cache.lookup("search", "query=Miles") is None


def test_the_catalog_store_is_bounded(cache_env):
    for index in range(music_cache.MAX_CATALOG_ENTRIES + 8):
        music_cache.remember("library", f"kind=likes|limit={index}", {"ok": True, "index": index}, saved_at=float(index))

    entries = json.loads(Path(music_cache.catalog_path()).read_text(encoding="utf-8"))["entries"]
    assert len(entries) == music_cache.MAX_CATALOG_ENTRIES
    newest = music_cache.MAX_CATALOG_ENTRIES + 7
    assert music_cache.lookup("library", f"kind=likes|limit={newest}") is not None
    assert music_cache.lookup("library", "kind=likes|limit=0") is None


def test_a_corrupt_catalog_file_is_ignored(cache_env):
    Path(music_cache.cache_dir()).mkdir(parents=True, exist_ok=True)
    Path(music_cache.catalog_path()).write_text("{not json", encoding="utf-8")
    fetch = counting_fetch([{"ok": True, "count": 1}])
    payload, status, _age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "miss" and payload["count"] == 1


def test_the_cache_can_be_switched_off(cache_env, monkeypatch):
    monkeypatch.setenv("NIX_MUSIC_CACHE", "0")
    fetch = counting_fetch([{"ok": True, "count": 1}])
    payload, status, age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "disabled" and age is None

    payload, status, _age = music_cache.read("library", "kind=playlists", fetch)
    assert status == "disabled" and len(fetch.calls) == 2
    assert music_cache.lookup("library", "kind=playlists") is None

    body, _content_type, _etag, status = music_cache.artwork(
        "https://img.example/cover.png", fetch=lambda _url: (b"cover", "image/png")
    )
    assert status == "miss" and body == b"cover"
    assert music_cache._artwork_entry(music_cache.artwork_key("https://img.example/cover.png")) is None


# ── artwork covers ───────────────────────────────────────────────────────


def test_a_cover_is_fetched_once_and_then_served_from_disk(cache_env):
    cover = b"\x89PNG\r\n\x1a\n cover bytes"
    calls = []

    def fetch(url):
        calls.append(url)
        return cover, "image/png"

    body, content_type, etag, status = music_cache.artwork("https://img.example/cover.png", fetch=fetch)
    assert status == "miss" and content_type == "image/png" and body == cover
    assert calls == ["https://img.example/cover.png"]

    body, content_type, same_etag, status = music_cache.artwork(
        "https://img.example/cover.png", fetch=lambda _url: pytest.fail("a cached cover must not be fetched again")
    )
    assert status == "hit" and body == cover and content_type == "image/png"
    assert same_etag == etag


def test_artwork_refuses_a_non_image_answer(cache_env):
    with pytest.raises(music_cache.ArtworkError) as refused:
        music_cache.artwork("https://img.example/page.html", fetch=lambda _url: (b"<html>", "text/html"))
    assert "image" in str(refused.value)
    assert music_cache._artwork_entry(music_cache.artwork_key("https://img.example/page.html")) is None


def test_artwork_refuses_an_oversized_cover(cache_env):
    oversized = b"x" * (music_cache.MAX_ARTWORK_BYTES + 1)
    with pytest.raises(music_cache.ArtworkError) as refused:
        music_cache.artwork("https://img.example/big.png", fetch=lambda _url: (oversized, "image/png"))
    assert refused.value.code == "artwork_too_large"


def test_artwork_refuses_a_non_http_source(cache_env):
    with pytest.raises(music_cache.ArtworkError) as refused:
        music_cache.download_artwork("file:///etc/passwd")
    assert refused.value.status == 400 and refused.value.code == "validation_error"


@pytest.mark.parametrize("host", ["127.0.0.1", "10.1.2.3", "192.168.0.10", "169.254.10.10", "::1"])
def test_artwork_refuses_the_console_own_networks(cache_env, host):
    """The route takes a caller-supplied URL, so it must not reach the LAN."""
    url = f"http://[{host}]/cover.png" if ":" in host else f"http://{host}/cover.png"
    with pytest.raises(music_cache.ArtworkError, match="public address"):
        music_cache.download_artwork(url)


def test_a_public_literal_address_passes_the_guard(cache_env):
    music_cache._require_public_host("93.184.216.34")


def test_the_artwork_store_is_bounded(cache_env):
    for index in range(music_cache.MAX_ARTWORK_FILES + 4):
        music_cache.store_artwork(
            f"https://img.example/{index}.png", b"cover", "image/png", saved_at=float(index)
        )
    index_file = music_cache._artwork_index()
    assert len(index_file["entries"]) == music_cache.MAX_ARTWORK_FILES
    assert music_cache._artwork_entry(music_cache.artwork_key(f"https://img.example/{music_cache.MAX_ARTWORK_FILES + 3}.png")) is not None
    assert music_cache._artwork_entry(music_cache.artwork_key("https://img.example/0.png")) is None


def test_stats_report_what_is_cached(cache_env):
    music_cache.remember("library", "kind=playlists", {"ok": True})
    music_cache.artwork("https://img.example/cover.png", fetch=lambda _url: (b"cover", "image/png"))

    stats = music_cache.stats()
    assert stats["enabled"] is True
    assert stats["catalog_entries"] == 1
    assert stats["artwork_files"] == 1
    assert stats["artwork_bytes"] == len(b"cover")
    assert stats["ttl_sec"] == music_cache.DEFAULT_CATALOG_TTL
    assert stats["dir"].endswith(music_cache.CACHE_DIR_NAME)
