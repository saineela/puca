"""The independent music queue: engine and its own service.

What is being proved here is the reason the queue left the skill:

  * a playlist longer than the old hundred-track ceiling is queued in one
    write, and the whole of it is readable in windows;
  * reading with a revision the reader already holds costs nothing — that is
    what stops a polling page from being traffic;
  * tracks that arrive as bare ids are warmed in the background at the rate
    the OpenDeck instance declares, and each one is asked of the provider
    once, from the cache after that;
  * the transport rules the queue has always had (a finished track leaves,
    repeat-one repeats, repeat-all wraps) still hold.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import music_queue  # noqa: E402
import music_queue_service  # noqa: E402
from music_queue import Bucket, MusicQueue, QueueLimits  # noqa: E402


@pytest.fixture()
def queue():
    with tempfile.TemporaryDirectory() as folder:
        yield MusicQueue(os.path.join(folder, "queue.db"))


def tracks(count: int, prefix: str = "Song") -> list[dict]:
    return [
        {"video_id": f"v{index:04d}", "title": f"{prefix} {index}", "artist": "Artist",
         "duration_sec": 180 + index, "artwork_url": f"http://art/{index}.jpg"}
        for index in range(count)
    ]


# ── a playlist is one write, not one write per song ──────────────────────


def test_a_playlist_of_more_than_a_hundred_songs_is_queued_in_one_write(queue):
    result = queue.add_many(tracks(320), playlist_id="pl-liked")
    assert result["accepted"] == 320
    assert result["refused"] == 0
    assert result["total"] == 320
    # The old queue refused everything past a hundred; this one keeps the lot.
    assert queue.total() == 320


def test_the_whole_queue_is_readable_in_windows_and_says_how_big_it_is(queue):
    queue.add_many(tracks(240))
    first = queue.state(offset=0, limit=50)
    assert len(first.items) == 50
    assert first.total == 240
    assert first.items[0]["title"] == "Song 0"
    last = queue.state(offset=200, limit=50)
    assert len(last.items) == 40
    assert last.items[0]["title"] == "Song 200"
    assert last.offset == 200


def test_a_write_past_the_ceiling_is_reported_rather_than_silently_dropped(queue):
    queue.limits = QueueLimits(max_tracks=10)
    result = queue.add_many(tracks(14))
    assert result["accepted"] == 10
    assert result["refused"] == 4
    assert queue.total() == 10


def test_an_entry_with_no_id_is_not_queueable(queue):
    result = queue.add_many([
        {"title": "Nowhere"},
        {"video_id": "keepme", "title": "Keep me"},
    ])
    assert result["accepted"] == 1
    assert result["unqueueable"] == 1
    assert queue.snapshot()[0]["video_id"] == "keepme"


# ── reads with a revision cost nothing ───────────────────────────────────


def test_a_reader_that_holds_the_current_revision_gets_no_queue_back(queue):
    queue.add_many(tracks(20))
    revision = queue.state().revision
    again = queue.state(revision=revision)
    assert again.unchanged is True
    assert again.items == []
    assert again.total == 20
    assert queue.stats()["unchanged_reads"] == 1


def test_a_change_moves_the_revision_so_the_reader_is_told_again(queue):
    queue.add_many(tracks(4))
    revision = queue.state().revision
    queue.add_many([{"video_id": "extra", "title": "Extra"}])
    assert queue.state(revision=revision).unchanged is False
    assert queue.state(revision=revision).total == 5


# ── the background warms the tracks in front of the cursor ───────────────


def test_only_tracks_after_the_cursor_are_worth_resolving(queue):
    queue.add_many(tracks(6))
    queue.set_index(0)
    queue.add_many([{"video_id": "v0001"}], mode="play_next")  # a bare id, already known
    pending = queue.pending()
    assert pending == []  # every track already carries its title
    queue.add_many([{"video_id": "v0011"}, {"video_id": "v0012"}, {"video_id": "v0013"}])
    queue.set_index(6)
    assert [entry["video_id"] for entry in queue.pending()] == ["v0011", "v0012", "v0013"]
    # A track behind the cursor is never asked about: it will not be reached.
    queue.set_index(9)
    assert queue.pending() == []


def test_the_prefetch_window_is_the_provider_s_own_limit(queue):
    queue.limits = QueueLimits(prefetch=2)
    queue.add_many([{"video_id": f"bare{i}"} for i in range(9)])
    assert len(queue.pending()) == 2
    assert len(queue.pending(limit=5)) == 5


def test_a_resolved_track_stops_being_pending_and_fills_every_row(queue):
    queue.add_many([{"video_id": "same"}, {"video_id": "same"}])
    queue.apply_resolution("same", {"title": "Same Song", "artist": "A", "duration_sec": 200})
    state = queue.state()
    assert [item["title"] for item in state.items] == ["Same Song", "Same Song"]
    assert queue.pending() == []
    assert queue.stats()["resolved"] == 1


def test_a_resolved_track_is_queued_with_its_metadata_next_time(queue):
    queue.apply_resolution("cached", {"title": "From cache", "artist": "A"})
    queue.add_many([{"video_id": "cached"}])
    assert queue.snapshot()[0]["title"] == "From cache"


def test_the_cache_keeps_only_the_tracks_it_was_told_to_keep(queue):
    queue.limits = QueueLimits(prefetch_keep=3)
    for index in range(6):
        queue.apply_resolution(f"c{index}", {"title": f"Cached {index}"})
    assert queue.stats()["cached_tracks"] == 3


def test_a_provider_that_does_not_know_a_track_is_remembered_as_a_miss(queue):
    queue.add_many([{"video_id": "ghost"}])
    queue.note_miss("ghost")
    assert queue.stats()["resolve_misses"] == 1


# ── a dead id cannot spend the whole rate limit ──────────────────────────
#
# The other half of "the rate limit keeps a long playlist off the provider":
# without it, the one id the provider will never know is asked about once a
# tick for the life of the queue, and the songs waiting behind it are the ones
# that go unresolved.


def test_a_refused_track_is_not_asked_again_on_the_very_next_pass(queue):
    queue.add_many([{"video_id": "ghost"}])
    assert [entry["video_id"] for entry in queue.pending()] == ["ghost"]
    queue.note_miss("ghost")
    # The first refusal buys a wait, so the next tick does not spend another
    # provider call on the same dead id.
    assert queue.pending() == []
    assert queue.stats()["pending"] == 1, "still waiting on the provider, not given up on"


def test_the_wait_between_attempts_doubles_and_then_stops_growing():
    assert music_queue.miss_backoff(0) == 0.0
    assert music_queue.miss_backoff(1) == music_queue.MISS_BACKOFF_BASE
    assert music_queue.miss_backoff(2) == 2 * music_queue.miss_backoff(1)
    assert music_queue.miss_backoff(music_queue.MISS_LIMIT) == 16 * music_queue.MISS_BACKOFF_BASE
    assert music_queue.miss_backoff(99) == music_queue.MISS_BACKOFF_MAX


def test_a_track_refused_often_enough_is_left_alone_but_stays_in_the_queue(queue):
    queue.add_many([{"video_id": "ghost"}, {"video_id": "keeper", "title": "Keeper"}])
    for _ in range(music_queue.MISS_LIMIT):
        queue.note_miss("ghost")
    assert queue.pending() == []
    stats = queue.stats()
    assert stats["unresolvable"] == 1
    assert stats["pending"] == 0
    # Given up on is not the same as removed: the row is still there, saying
    # which track it was, and the queue is still the length the listener saw.
    assert queue.state().total == 2
    assert queue.state().items[0]["video_id"] == "ghost"


def test_a_dead_id_does_not_stop_the_tracks_behind_it_from_being_warmed(queue):
    queue.add_many([{"video_id": "ghost"}, {"video_id": "real"}])
    for _ in range(music_queue.MISS_LIMIT):
        queue.note_miss("ghost")
    assert [entry["video_id"] for entry in queue.pending()] == ["real"]


def test_queueing_a_track_again_asks_the_provider_again(queue):
    queue.add_many([{"video_id": "ghost"}])
    for _ in range(music_queue.MISS_LIMIT):
        queue.note_miss("ghost")
    assert queue.pending() == []
    queue.add_many([{"video_id": "ghost"}])
    assert [entry["video_id"] for entry in queue.pending()] == ["ghost"]


def test_a_track_that_arrives_after_being_refused_stops_being_unresolvable(queue):
    queue.add_many([{"video_id": "ghost"}])
    for _ in range(music_queue.MISS_LIMIT):
        queue.note_miss("ghost")
    assert queue.stats()["unresolvable"] == 1
    queue.apply_resolution("ghost", {"title": "Ghost", "artist": "A"})
    assert queue.stats()["unresolvable"] == 0
    assert queue.state().items[0]["title"] == "Ghost"


# ── the transport rules are unchanged ────────────────────────────────────


def test_a_track_that_finished_leaves_the_queue(queue):
    queue.add_many(tracks(3))
    queue.set_index(0)
    result = queue.start_next(auto=True)
    assert result["track"]["title"] == "Song 1"
    assert result["total"] == 2
    assert [item["title"] for item in queue.snapshot()] == ["Song 1", "Song 2"]


def test_repeat_one_plays_the_same_track_again(queue):
    queue.add_many(tracks(3))
    queue.set_preference("repeat", "one")
    queue.set_index(1)
    result = queue.start_next(auto=True)
    assert result["track"]["title"] == "Song 1"
    assert result["total"] == 3


def test_repeat_all_wraps_at_the_end(queue):
    queue.add_many(tracks(3))
    queue.set_preference("repeat", "all")
    queue.set_index(2)
    result = queue.start_next()
    assert result["track"]["title"] == "Song 0"
    assert result["index"] == 0


def test_an_exhausted_queue_says_so_without_a_track(queue):
    queue.add_many(tracks(1))
    queue.set_index(0)
    result = queue.start_next()
    assert result["ok"] is False
    assert result["code"] == "queue_end"


def test_moving_a_row_carries_the_cursor_with_the_song_it_pointed_at(queue):
    queue.add_many(tracks(4))
    queue.set_index(1)
    queue.move(0, 3)
    state = queue.state()
    assert [item["title"] for item in state.items] == ["Song 1", "Song 2", "Song 3", "Song 0"]
    assert state.items[state.index]["title"] == "Song 1"


def test_play_next_lands_behind_the_playing_track(queue):
    queue.add_many(tracks(4))
    queue.set_index(0)
    queue.add_many([{"video_id": "new", "title": "New"}], mode="play_next")
    assert [item["title"] for item in queue.snapshot()] == \
        ["Song 0", "New", "Song 1", "Song 2", "Song 3"]
    assert queue.state().index == 0


def test_replace_makes_the_queue_the_new_list(queue):
    queue.add_many(tracks(5))
    result = queue.add_many([{"video_id": "only", "title": "Only"}], mode="replace")
    assert result["total"] == 1
    assert queue.state().index == 0


def test_removing_the_playing_track_leaves_the_queue_without_a_cursor(queue):
    queue.add_many(tracks(4))
    queue.set_index(2)
    queue.remove(2)
    assert queue.state().index == -1
    assert queue.total() == 3


# ── limits come from the OpenDeck instance ───────────────────────────────


def test_limits_are_read_from_the_manifest_the_provider_declares():
    manifest = {
        "device_ui": {
            "media": {
                "provider": {
                    "queue": {
                        "tool": "opendeck_music",
                        "arguments": {"action": "queue", "op": "list"},
                        "limits": {"max_tracks": 8000, "prefetch": 40,
                                   "resolves_per_minute": 30, "window": 80},
                    }
                }
            }
        }
    }
    limits = music_queue.limits_from_manifest(manifest)
    assert limits.max_tracks == 8000
    assert limits.prefetch == 40
    assert limits.resolves_per_minute == 30
    assert limits.window == 80
    # Anything the skill does not declare keeps a working default.
    assert limits.max_batch == QueueLimits().max_batch


def test_a_manifest_with_nothing_declared_is_still_a_working_queue():
    assert music_queue.limits_from_manifest({}) == QueueLimits()
    assert music_queue.limits_from_manifest(None) == QueueLimits()


def test_a_nonsense_limit_cannot_break_the_queue():
    limits = QueueLimits.from_provider({"limits": {"max_tracks": "lots", "prefetch": -4}})
    assert limits.max_tracks == QueueLimits().max_tracks
    assert limits.prefetch == 0


# ── the rate limit is what keeps a long playlist off the provider ────────


def test_the_bucket_hands_out_tokens_at_the_declared_rate():
    bucket = Bucket(per_minute=3, burst=1)
    assert bucket.take() is True
    assert bucket.take() is False  # one burst token, and nothing refilled yet


def test_a_provider_that_asks_for_no_resolving_is_not_asked():
    service = music_queue_service.QueueService(
        MusicQueue(":memory:", QueueLimits(resolves_per_minute=0, prefetch=10)),
        resolver=lambda video_id: {"title": "never"},
    )
    service.queue.add_many([{"video_id": "a"}, {"video_id": "b"}])
    assert service.warm() == 0
    assert service.resolved_calls == 0


def test_warming_resolves_the_pending_tracks_and_stops(queue_service_factory):
    service, calls = queue_service_factory(prefetch=3, per_minute=600)
    service.queue.add_many([{"video_id": "a"}, {"video_id": "b"}])
    assert service.warm() == 2
    assert calls == ["a", "b"]
    # Second pass: both are warm, so the provider is not asked again.
    assert service.warm() == 0
    assert calls == ["a", "b"]
    assert [item["title"] for item in service.queue.snapshot()] == ["A", "B"]


def test_a_track_the_provider_does_not_know_does_not_stall_the_queue(queue_service_factory):
    service, _calls = queue_service_factory(prefetch=3, per_minute=600,
                                            resolver=lambda video_id: None)
    service.queue.add_many([{"video_id": "a"}])
    assert service.warm() == 0
    assert service.queue.stats()["resolve_misses"] == 1


def test_the_warm_loop_spends_one_call_on_a_dead_id_not_one_per_tick(queue_service_factory):
    asked: list[str] = []

    def dead_resolver(video_id: str):
        asked.append(video_id)
        return None if video_id == "dead" else {"title": video_id.upper()}

    service, _calls = queue_service_factory(prefetch=3, per_minute=600, resolver=dead_resolver)
    service.queue.add_many([{"video_id": "dead"}, {"video_id": "alive"}])
    service.warm()
    assert asked == ["dead", "alive"], "both are asked the first time round"
    # Twenty more ticks: the dead id is inside its wait, and the queue does not
    # spend the provider's rate limit asking about it again.
    for _ in range(20):
        service.warm()
    assert asked == ["dead", "alive"]
    assert service.queue.stats()["resolve_misses"] == 1


def test_a_provider_that_is_not_answering_refuses_nothing(queue_service_factory):
    def broken(video_id: str):
        raise music_queue_service.ProviderUnavailable("the skill is not answering")

    service, _calls = queue_service_factory(prefetch=3, per_minute=600, resolver=broken)
    service.queue.add_many([{"video_id": "a"}, {"video_id": "b"}])
    assert service.warm() == 0
    # Nothing was answered about either track, so neither is marked as refused:
    # a provider that is down must not cost a song its place in the queue.
    assert service.queue.stats()["resolve_misses"] == 0
    assert service.provider_errors == 1
    assert [entry["video_id"] for entry in service.pending()] == ["a", "b"]
    assert service.queue.stats()["pending"] == 2


def test_a_provider_that_is_down_is_asked_once_and_then_left_alone(queue_service_factory):
    asked: list[str] = []

    def broken(video_id: str):
        asked.append(video_id)
        raise music_queue_service.ProviderUnavailable("down")

    service, _calls = queue_service_factory(prefetch=3, per_minute=600, resolver=broken,
                                            cooldown=30)
    service.queue.add_many([{"video_id": "a"}, {"video_id": "b"}])
    for _ in range(10):
        service.warm()
    # One failed call tells us the provider is down; the rest of the queue is
    # not put to it, and the cooldown is what keeps the next ticks quiet too.
    assert asked == ["a"]
    assert service.provider_errors == 1
    assert service.queue.counters["provider_calls"] == 0


def test_the_console_s_answer_says_whether_a_track_was_refused_or_the_provider_is_down():
    """The queue's callback has to tell the two apart, or it gives up wrongly."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import socket

    class Console(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quieter than the default line
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            asked = json.loads(self.rfile.read(length) or b"{}").get("video_id")
            if asked == "known":
                return self._answer(200, {"ok": True, "track": {"video_id": asked, "title": "Known"}})
            if asked == "unknown":
                return self._answer(404, {"ok": False, "error": "not known to the provider"})
            return self._answer(503, {"ok": False, "error": "the skill is not answering"})

        def _answer(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    console = ThreadingHTTPServer(("127.0.0.1", port), Console)
    threading.Thread(target=console.serve_forever, daemon=True).start()
    try:
        resolve = music_queue_service.console_resolver(f"http://127.0.0.1:{port}")
        assert resolve("known")["title"] == "Known"
        assert resolve("unknown") is None, "the provider answered: it does not know this track"
        with pytest.raises(music_queue_service.ProviderUnavailable):
            resolve("offline")
    finally:
        console.shutdown()
        console.server_close()


def test_the_service_adopts_the_limits_the_console_pushes(queue_service_factory):
    service, _calls = queue_service_factory()
    adopted = service.set_limits({"max_tracks": 900, "prefetch": 5, "resolves_per_minute": 7})
    assert adopted.max_tracks == 900
    assert service.queue.limits.prefetch == 5
    assert service.queue.bucket.rate == pytest.approx(7 / 60)


# ── the service over HTTP ────────────────────────────────────────────────


@pytest.fixture()
def queue_service_factory():
    made = []

    def factory(prefetch: int = 5, per_minute: int = 600, resolver=None, cooldown: float = 0.0):
        calls: list[str] = []

        def default_resolver(video_id: str):
            calls.append(video_id)
            return {"title": video_id.upper(), "artist": "A", "duration_sec": 100}

        queue = MusicQueue(":memory:", QueueLimits(prefetch=prefetch, resolves_per_minute=per_minute))
        service = music_queue_service.QueueService(queue, resolver=resolver or default_resolver,
                                                   cooldown=cooldown)
        made.append(service)
        return service, calls

    yield factory
    for service in made:
        service.stop()


@pytest.fixture()
def live_service(queue_service_factory):
    service, calls = queue_service_factory()
    server = music_queue_service._handler_for(service)

    class Server(music_queue_service.ThreadingHTTPServer):
        pass

    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = Server(("127.0.0.1", port), server)
    httpd.queue_service = service
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", service, calls
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def _post(url: str, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_a_long_playlist_goes_over_the_wire_in_one_call(live_service):
    base, service, _calls = live_service
    result = _post(base + "/add", {"tracks": tracks(400), "playlist_id": "pl"})
    assert result["accepted"] == 400
    first = _get(base + "/state?from=0&limit=50")
    assert first["total"] == 400
    assert len(first["queue"]) == 50
    later = _get(base + "/state?from=350&limit=50")
    assert len(later["queue"]) == 50
    assert later["queue"][0]["title"] == "Song 350"


def test_a_reader_holding_the_revision_is_told_nothing_changed(live_service):
    base, service, _calls = live_service
    _post(base + "/add", {"tracks": tracks(30)})
    revision = _get(base + "/state?limit=10")["revision"]
    again = _get(base + f"/state?limit=10&revision={revision}")
    assert again["unchanged"] is True
    assert again["queue"] == []
    assert service.queue.stats()["unchanged_reads"] == 1


def test_the_console_can_push_limits_and_bulk_resolutions(live_service):
    base, service, _calls = live_service
    adopted = _post(base + "/limits", {"limits": {"max_tracks": 321, "prefetch": 4}})
    assert adopted["limits"]["max_tracks"] == 321
    applied = _post(base + "/resolutions", {"tracks": [
        {"video_id": "x", "title": "X"}, {"video_id": "y", "title": "Y"}]})
    assert applied["applied"] == 2


def test_queue_operations_run_over_http_in_the_provider_s_vocabulary(live_service):
    base, service, _calls = live_service
    _post(base + "/add", {"tracks": tracks(4)})
    moved = _post(base + "/ops", {"op": "move", "index": 0, "to": 3})
    assert moved["ok"] is True
    state = _get(base + "/state")
    assert [item["title"] for item in state["queue"]] == ["Song 1", "Song 2", "Song 3", "Song 0"]
    assert _post(base + "/ops", {"op": "next_up", "index": 3})["ok"] is True
    assert _post(base + "/ops", {"op": "remove", "index": 0})["ok"] is True
    assert _post(base + "/ops", {"op": "clear"})["removed"] == 3
    assert _get(base + "/state")["total"] == 0


def test_an_unknown_operation_is_a_bad_request_not_a_crash(live_service):
    base, _service, _calls = live_service
    with pytest.raises(urllib.error.HTTPError) as failure:
        _post(base + "/ops", {"op": "teleport"})
    assert failure.value.code == 400


# ── how other processes find the service ─────────────────────────────────
#
# The console reads its own environment, but NIX starts a skill worker with
# almost none on purpose, so the service publishes its address in the data
# directory: one small file that whoever needs to find the queue reads.


def test_the_service_publishes_its_address_for_processes_with_no_environment(tmp_path):
    published = str(tmp_path / "music-queue.json")
    music_queue.publish_service("http://127.0.0.1:49119/", path=published)
    found = music_queue.discover_service(path=published)
    assert found == {"url": "http://127.0.0.1:49119", "token": ""}
    assert oct(os.stat(published).st_mode & 0o777) == "0o600", "readable only by its owner"


def test_a_withdrawn_address_is_no_longer_discovered(tmp_path):
    published = str(tmp_path / "music-queue.json")
    music_queue.publish_service("http://127.0.0.1:49119", path=published)
    music_queue.withdraw_service("http://127.0.0.1:49119", path=published)
    assert music_queue.discover_service(path=published) is None


def test_one_service_does_not_withdraw_another_one_s_address(tmp_path):
    published = str(tmp_path / "music-queue.json")
    music_queue.publish_service("http://127.0.0.1:49119", path=published)
    music_queue.withdraw_service("http://127.0.0.1:50000", path=published)
    assert music_queue.discover_service(path=published) is not None


def test_an_unreadable_or_nonsense_address_is_not_a_queue(tmp_path):
    published = tmp_path / "music-queue.json"
    assert music_queue.discover_service(path=str(published)) is None
    published.write_text("not json", encoding="utf-8")
    assert music_queue.discover_service(path=str(published)) is None
    published.write_text(json.dumps({"url": "file:///etc/passwd"}), encoding="utf-8")
    assert music_queue.discover_service(path=str(published)) is None
    published.write_text(json.dumps({"at": 1, "url": "http://127.0.0.1:49119"}), encoding="utf-8")
    assert music_queue.discover_service(path=str(published), max_age=0.001) is None


def test_each_queue_publishes_beside_its_own_database(tmp_path):
    """One file per queue, so two of them cannot overwrite each other's address."""
    queue = MusicQueue(str(tmp_path / "music-queue.db"))
    assert queue.published_file == str(tmp_path / "music-queue.json")
    # A queue with no file of its own has nowhere to publish beside, so it uses
    # the standard address; an in-memory queue is not a service at all.
    assert MusicQueue(":memory:").published_file == music_queue.service_file()
    monkeypatch_file = tmp_path / "given.json"
    music_queue.publish_service("http://127.0.0.1:49119", path=str(monkeypatch_file))
    assert music_queue.discover_service(path=str(monkeypatch_file))["url"].endswith("49119")


def test_the_queues_own_files_follow_the_data_directory(monkeypatch, tmp_path):
    """A box that keeps its data elsewhere finds its own queue there.

    The database and the published address sit in the console's data directory,
    which is read when they are used: that is what lets a test's console and a
    queue service running on the same machine stay out of each other's way.
    """
    monkeypatch.setenv("NIX_DATA_DIR", str(tmp_path))
    assert music_queue.default_db() == str(tmp_path / "music-queue.db")
    assert music_queue.service_file() == str(tmp_path / "music-queue.json")
    assert MusicQueue().path == str(tmp_path / "music-queue.db")
    music_queue.publish_service("http://127.0.0.1:49119")
    assert music_queue.discover_service()["url"].endswith("49119")
    music_queue.withdraw_service()
    assert music_queue.discover_service() is None

    # An address and a database someone named by hand still win.
    monkeypatch.setenv("NIX_MUSIC_QUEUE_FILE", str(tmp_path / "given.json"))
    monkeypatch.setenv("NIX_MUSIC_QUEUE_DB", str(tmp_path / "given.db"))
    assert music_queue.service_file() == str(tmp_path / "given.json")
    assert music_queue.default_db() == str(tmp_path / "given.db")


# ── one speaker, one queue: the service decides and hands off ────────────
#
# Three devices are open on the same speaker — the wall panel, the phone and a
# laptop — and each of them can see the speaker's state. Before this, each one
# also decided what came next, so one finished track could be advanced three
# times, and a song played from one device was cut off by another's queue. The
# rule these tests hold the service to: the queue decides, and one finished
# track moves the line-up exactly once, however many devices are watching.


def _playing_service(count: int = 3, *, with_player: bool = True):
    played: list[dict] = []

    def player(request: dict) -> dict:
        played.append(request)
        return {"ok": True}

    service = music_queue_service.QueueService(
        MusicQueue(":memory:", QueueLimits(prefetch=0, resolves_per_minute=0)),
        player=player if with_player else None,
    )
    service.queue.add_many(tracks(count))
    return service, played


def _finished(video_id: str, *, position: float = 180.0, duration: float = 180.0,
              queue_active: bool = True) -> dict:
    return {"state": "idle", "video_id": video_id, "position": position,
            "duration": duration, "queue_active": queue_active}


def _started(video_id: str, *, position: float = 3.0, duration: float = 180.0) -> dict:
    return {"state": "playing", "video_id": video_id, "position": position, "duration": duration}


def test_selecting_a_queue_row_is_played_by_the_service():
    service, played = _playing_service()
    reply = service.queue_op({"op": "play", "index": 1})
    assert reply["executed"] is True
    assert [(call["video_id"], call["reason"]) for call in played] == [("v0001", "select")]


def test_a_skip_is_one_hand_off_and_names_the_next_track():
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    reply = service.queue_op({"op": "next"})
    assert reply["ok"] is True
    assert reply["executed"] is True
    assert played[-1]["video_id"] == "v0001"
    assert played[-1]["reason"] == "skip"
    # The service is the one executing, so the caller has nothing to hand over.
    assert len(played) == 2


def test_three_devices_reporting_one_finished_track_advance_it_once():
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    played.clear()

    service.report_playback(_started("v0000"))
    first = service.report_playback(_finished("v0000"))
    assert first["advanced"] is True
    assert first["track"]["video_id"] == "v0001"

    # The phone and the laptop poll the same finished track a moment later.
    for _ in range(3):
        again = service.report_playback(_finished("v0000"))
        assert again["advanced"] is False

    assert [call["video_id"] for call in played] == ["v0001"]
    assert service.advanced == 2  # the select, then the one end-of-track


def test_a_track_stopped_before_its_end_does_not_move_the_line_up():
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    played.clear()
    service.report_playback(_started("v0000", position=10))
    stopped = service.report_playback(_finished("v0000", position=20, duration=180))
    assert stopped["advanced"] is False
    assert played == []


def test_a_one_off_play_ending_does_not_move_the_queues_line_up():
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    played.clear()
    service.report_playback(_started("searched"))
    reply = service.report_playback(_finished("searched", queue_active=False))
    assert reply["advanced"] is False
    assert played == []


def test_a_finished_track_that_is_not_the_queued_one_is_ignored():
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    played.clear()
    service.report_playback(_started("v0002"))
    reply = service.report_playback(_finished("v0002"))
    assert reply["advanced"] is False
    assert played == []


def test_without_a_player_the_service_still_decides_and_says_so():
    """A service started without a player still owns the choice.

    The console then performs the hand-off from the reply — once, because the
    reply also carries ``executed: false`` and the same one-advance rule.
    """
    service, _played = _playing_service(with_player=False)
    service.queue_op({"op": "play", "index": 0})
    service.report_playback(_started("v0000"))
    reply = service.report_playback(_finished("v0000"))
    assert reply["advanced"] is True
    assert reply["executed"] is False
    assert reply["track"]["video_id"] == "v0001"
    again = service.report_playback(_finished("v0000"))
    assert again["advanced"] is False


def test_a_paused_track_is_not_a_finished_track():
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    played.clear()
    service.report_playback(_started("v0000"))
    paused = service.report_playback({"state": "paused", "video_id": "v0000",
                                      "position": 40, "duration": 180})
    assert paused["advanced"] is False
    resumed = service.report_playback(_started("v0000", position=41))
    assert resumed["advanced"] is False
    assert played == []


def test_the_service_shares_what_it_is_playing_with_every_reader(live_service):
    base, service, _calls = live_service
    _post(base + "/add", {"tracks": tracks(3)})
    _post(base + "/ops", {"op": "play", "index": 0})
    _post(base + "/playback", _started("v0000", position=12))
    seen = _get(base + "/playback")
    assert seen["video_id"] == "v0000"
    assert seen["position"] == 12.0
    # The queue's own read carries the same speaker state, so the page, the
    # phone and the wall panel mirror one snapshot instead of three guesses.
    assert _get(base + "/state?limit=10")["playback"]["video_id"] == "v0000"


def test_the_service_reports_the_hand_off_it_performed(live_service):
    base, service, _calls = live_service
    _post(base + "/add", {"tracks": tracks(3)})
    reply = _post(base + "/ops", {"op": "play", "index": 0})
    # ``live_service`` builds the service without a player, so the decision is
    # the service's and the press is the console's: exactly the split the reply
    # has to describe.
    assert reply["executed"] is False
    assert reply["track"]["video_id"] == "v0000"


def test_a_service_started_for_the_console_is_its_own_player(tmp_path, monkeypatch):
    """The production wiring: the queue chooses, the console presses.

    ``serve`` is what the unit file runs. Given a console address it must come
    up able to resolve (asks the console, which owns the provider skill) and
    able to play (posts the chosen track back to the console) — otherwise a box
    would run a queue that can never start anything, and the old arrangement,
    where the console decided, would quietly be the one still in force.
    """
    monkeypatch.setenv("NIX_MUSIC_QUEUE_FILE", str(tmp_path / "music-queue.json"))
    server = music_queue_service.serve(
        port=0, db=str(tmp_path / "queue.db"), console_url="http://127.0.0.1:49117")
    try:
        service = server.queue_service
        assert service.resolver is not None
        assert service.player is not None
    finally:
        server.queue_service.stop()
        server.server_close()
        music_queue.withdraw_service(path=str(tmp_path / "music-queue.json"))


def test_simultaneous_reports_from_three_devices_advance_once():
    """The wall, the phone and the laptop can poll in the same instant.

    Reading the last report and writing the next one is one step here, so only
    the first of three simultaneous "the track ended" reports advances the
    line-up — the others are told it is already done. Without that, the same
    finish would be seen three times and the speaker would skip two songs.
    """
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    service.report_playback(_started("v0000"))
    played.clear()

    answers: list[dict] = []

    def poll() -> None:
        answers.append(service.report_playback(_finished("v0000")))

    threads = [threading.Thread(target=poll) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(1 for answer in answers if answer["advanced"]) == 1
    assert [call["video_id"] for call in played] == ["v0001"]


def test_a_speaker_that_never_reports_playing_still_ends_a_track():
    """This box's speaker reports idle for the whole track.

    The hand-off resolves a stream the speaker fetches, and its own state never
    turns to "playing" — so the console's own clock is the only progress signal.
    The queue must still follow that track to its end (it did before, and a
    queue that stops at the last song is worse than useless), must advance on
    the end exactly once, and must not read the two reports in between as two
    endings.
    """
    service, played = _playing_service()
    service.queue_op({"op": "play", "index": 0})
    played.clear()

    early = service.report_playback(_finished("v0000", position=20, duration=180))
    assert early["advanced"] is False
    middle = service.report_playback(_finished("v0000", position=120, duration=180))
    assert middle["advanced"] is False

    ended = service.report_playback(_finished("v0000", position=181, duration=180))
    assert ended["advanced"] is True
    assert [call["video_id"] for call in played] == ["v0001"]

    # The phone reports the same end a moment later: still one advance.
    assert service.report_playback(_finished("v0000", position=182, duration=180))["advanced"] is False
    assert [call["video_id"] for call in played] == ["v0001"]


def test_repeat_one_advances_at_the_end_of_every_pass():
    """A repeated track has one ending per pass, not one ending ever.

    The mark that stops a second reader of an ending is cleared when the
    position falls back to the start of the track: that is a new pass, and its
    own ending is a new advance. Without the restart rule, repeat-one would
    advance once and then sit on the song forever.
    """
    service, played = _playing_service()
    service.queue_op({"op": "repeat", "mode": "one"})
    service.queue_op({"op": "play", "index": 0})
    played.clear()

    service.report_playback(_finished("v0000", position=100, duration=180))
    first = service.report_playback(_finished("v0000", position=181, duration=180))
    assert first["advanced"] is True and first["track"]["video_id"] == "v0000"
    assert service.report_playback(_finished("v0000", position=190, duration=180))["advanced"] is False

    # The next pass: the position is back at the start, then it ends again.
    service.report_playback(_finished("v0000", position=4, duration=180))
    second = service.report_playback(_finished("v0000", position=181, duration=180))
    assert second["advanced"] is True
    assert [call["video_id"] for call in played] == ["v0000", "v0000"]
