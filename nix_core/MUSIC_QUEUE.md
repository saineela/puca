# The music queue service

**Status:** implemented in `music_queue.py` (the queue) and
`music_queue_service.py` (the process that owns it), served on port 49119 by
`nix-music-queue.service`. The console and the OpenDeck worker both use it when
`NIX_MUSIC_QUEUE_URL` is set, when they find the address the service publishes,
and fall back to the queue inside the skill when there is neither.

## Why it is its own service

The queue used to live inside the music skill: one JSON file, at most a hundred
tracks, and every change was a round trip to the provider worker. That is the
right shape for "play this next" and the wrong one for a playlist:

- **A playlist is one write, not one per song.** Adding a three-hour playlist
  used to be one provider call per track — and the hundred-and-first track was
  refused outright. The queue now takes a whole list in a single request and
  stores it in SQLite.
- **Reads are windows with a revision.** A reader sends the revision it already
  holds and is told `unchanged` with no queue in the reply at all. A page polling
  every few seconds therefore costs one small request, not the whole line-up, and
  the provider is not involved at all.
- **The provider is asked once per track, at the rate it declares.** Tracks that
  arrive as bare ids (a stored playlist, a shared link) are resolved by a
  background loop, limited by the provider's own `resolves_per_minute`, and each
  answer is cached, so a song is asked about once and never again.
- **A track the provider does not know is asked about a few times, then left
  alone.** An id the provider refuses (a deleted video, an id typed by hand) is
  retried with a doubling wait and after `MISS_LIMIT` refusals not at all, so one
  dead id cannot spend the whole rate limit once a tick while the songs behind it
  go unresolved. It is not removed: the row stays in the line-up saying which
  track it was, `/stats` counts it under `unresolvable` instead of `pending`, and
  queueing it again is a new request that starts the counting over.
- **A provider that is not answering is not a refusal.** A resolver that raises
  (the console or the skill is down) has said nothing about the track, so nothing
  is counted against it: the background hands back its token, stops the pass
  rather than asking about every other track, and waits
  `NIX_MUSIC_QUEUE_PROVIDER_COOLDOWN` (30s) before trying again. The console's
  answer is what draws the line: `404` means "I do not know this track", anything
  else means the provider is in trouble.
- **It outlives the console.** Restarting the dashboard, updating the skill, or
  reinstalling OpenDeck no longer touches the line-up.

## Running it

```bash
python3 nix_core/music_queue_service.py --port 49119 --db /root/data/music-queue.db \
    --console http://127.0.0.1:49117
```

| Flag | Meaning |
| --- | --- |
| `--port` | Where it listens (env `NIX_MUSIC_QUEUE_PORT`, default 49119). |
| `--db` | Its SQLite file (env `NIX_MUSIC_QUEUE_DB`, default `<data directory>/music-queue.db`). |
| `--console` | The console, which is where the installed provider skill lives; this is what the background resolver calls back into. `--no-prefetch` turns the background off entirely. |

Its two files — the database and the note that says where it is listening — sit
in the console's data directory: `NIX_DATA_DIR` when that is set, this checkout's
`data/` otherwise. They are resolved when they are used rather than when the
module is loaded, so a box that keeps its data elsewhere finds its own queue
there, and so do two queues on one machine that were given a data directory each.
`NIX_MUSIC_QUEUE_DB` and `NIX_MUSIC_QUEUE_FILE` name either one by hand and win
over the data directory.

| Env | Meaning |
| --- | --- |
| `NIX_MUSIC_QUEUE_URL` | How the console and the skill reach it. Set on the console service, it reaches both (the skill worker is started by the console). |
| `NIX_MUSIC_QUEUE_DB` | The queue's SQLite file (default `<data directory>/music-queue.db`). |
| `NIX_MUSIC_QUEUE_FILE` | Where the service publishes its address (default `<data directory>/music-queue.json`). |
| `NIX_MUSIC_QUEUE_TOKEN` | Sent as `X-Nix-Token` and required on every request when set. The service binds `0.0.0.0`, so this is what keeps a queue off the LAN. |
| `NIX_MUSIC_QUEUE_TICK` | How often the background looks for tracks to warm (default 1.0s). |
| `NIX_MUSIC_QUEUE_PROVIDER_COOLDOWN` | How long the background leaves a provider alone after a failed call (default 30s). |

## Its HTTP surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, the current revision, and the limits in force. |
| `GET` | `/state?from=&limit=&revision=` | A window of the queue plus `total`, `index`, `shuffle`, `repeat` and `revision`. With `revision=<the one the caller holds>`, an unchanged queue answers `{"unchanged":true,"queue":[]}`. |
| `GET` | `/all` | The whole line-up, for a caller that genuinely needs it. |
| `GET` | `/pending?limit=` | The tracks the background still has to resolve: refused ids inside their wait are still here, the ones it has given up on are not. |
| `GET` | `/stats` | Counters: adds, tracks added, resolved, misses, reads, unchanged reads, provider calls, and the limits — plus `pending` (still waiting on the provider), `unresolvable` (refused often enough that the queue stopped asking) and `provider_errors` (calls the provider did not answer at all). |
| `POST` | `/add` | `{"tracks":[{video_id,title,…},…],"playlist_id":…,"mode":"append"\|"play_next"\|"replace"}` — the bulk write. |
| `POST` | `/ops` | The provider's own vocabulary: `remove`, `move`, `front`, `next_up`, `clear`, `shuffle`, `repeat`, `set_index`/`play`, `next`. |
| `POST` | `/limits` | The queue limits the OpenDeck instance declares (see below); the console pushes them whenever it reads the installed manifest. |
| `POST` | `/resolutions` | A bulk fill of resolved metadata, for a caller that resolved tracks itself. |
| `POST` | `/warm` | Resolve the next few tracks now, within the same rate limit. |

## The limits come from the OpenDeck instance

A music provider declares them next to the action that reaches its queue, and
the console validates them (`console._validate_skill_queue_limits`):

```json
"queue": {
  "tool": "opendeck_music",
  "arguments": {"action": "queue", "op": "list"},
  "limits": {
    "max_tracks": 5000, "max_batch": 200, "window": 50, "prefetch": 25,
    "resolves_per_minute": 60, "resolve_concurrency": 2, "cache_seconds": 600
  }
}
```

| Limit | What it controls |
| --- | --- |
| `max_tracks` | The queue's ceiling. The old fixed hundred is gone; this is the provider's number. |
| `max_batch` | How many tracks one write may carry. A longer list is written in this many per request. |
| `window` | How many tracks one read hands the page by default. |
| `prefetch` | How far ahead of the cursor the background warms. |
| `resolves_per_minute` | The provider's rate limit, enforced with a token bucket. `0` means "never call me for this". |
| `resolve_concurrency` | How many resolutions may be in flight at once. |
| `cache_seconds` / `prefetch_keep` | How long a resolved track stays warm, and how many are kept. |

Anything a skill does not declare keeps a working default, so a provider that
says nothing still gets a queue.

## How the console and the skill use it

- Nothing has to be configured: the service publishes its address in the data
directory (`music-queue.json`, owner-readable only), and `music_api` and the
OpenDeck worker read that file when `NIX_MUSIC_QUEUE_URL` is not in their
environment — which matters for the worker, because a skill is started with
almost no environment on purpose. An address someone wrote down is authoritative
and its failures are reported; a published address that has gone is treated as a
service that stopped, and the queue goes back to the provider skill.
- `GET /api/music/queue` proxies `/state`, adding `total`, `offset`, `revision`
  and `unchanged` to the payload every existing reader already knows. `?from=`,
  `?limit=` and `?revision=` are passed through.
- `POST /api/music/queue` proxies the operations, and adds `add_many` (a whole
  playlist in one call, written in `max_batch` pieces) and `resolve` (read-only:
  resolve one track's metadata, used as the background's callback).
- `POST /api/music/queue/resolve` is what the service calls back for, one track
  at a time.
- Playing a queued track is unchanged from the outside: the service decides
  *which* track, and the provider skill still resolves the stream URL and
  receives the hand-off, so a queued song plays exactly like a searched one.
- The assistant's own queue tool forwards to the service too (the worker reads
  `NIX_MUSIC_QUEUE_URL`), so the page and the assistant are reading one line-up
  rather than two.
- A queue read while the service is unreachable is reported as
  `queue_service_unavailable` rather than answered from somewhere else: a
  line-up the speaker is not following is worse than saying so.
