# nix_core

The central routing brain of the Nix ecosystem. Every incoming request
(voice text from the Orange Pi gateway or anything else) is classified
into one of three destinations and answered:

```
                        ┌─────────────────────────────────────┐
   Orange Pi / client   │            nix_core :9000           │
   ──── websocket ────▶ │  ws_server.py → brain.py → router   │
                        └───────┬──────────────┬──────────────┘
              knowledge route   │              │   chat route
              (personal data)   │              │   (world/chat)
                                ▼              ▼
              ┌────────────────────────┐   ┌──────────────────┐
              │ nix_knowledge :8100    │   │ Ollama :11434    │
              │  /process /classify    │   │ phi-4 + SearXNG  │
              │  /digest  /health      │   │ web search,      │
              │                        │   │ zero user knowledge
              │  bridge (auto)         │   └──────────────────┘
              └──────────┬─────────────┘
                         │ schedules actions
                         ▼
              ┌────────────────────────┐
              │ nix_actions :8200      │
              │  /log /context         │
              │  /propagate /run       │
              │  (session store too)   │
              └────────────────────────┘
```

## Routes

| Route      | Destination        | Examples                                                        |
|------------|--------------------|-----------------------------------------------------------------|
| knowledge  | nix_knowledge API  | "remember my wifi password is X", "when is my tsa meeting", "i have a dentist appointment tomorrow", "set an alarm for 6am", "cancel my dentist appointment" |
| chat       | Ollama (phi-4)     | "what's the weather", "who won the game", "tell me a joke", "hi" |
| unknown    | model gate @ nix_knowledge | anything the rules abstain on ("i like metal music")   |

### Classification accuracy

`router.py` is a deterministic high-precision lexical classifier —
same design as `nix_knowledge/rules.py`. `eval_router.py` reports:

```
Corpus size: 75   Accuracy: 100.0%
  chat       35/35 (100%)   knowledge 40/40 (100%)
```

Requests the rules cannot resolve escalate to the **model gate**
(`knowledge_api.py /classify`): the Qwen2.5-0.5B selector must emit a
tool call to pick knowledge; anything else (no call, parse failure,
model down) degrades to chat. This makes false-knowledge routing rare
while chatty wording wrapping a knowledge request ("hey so um what
time is my meeting tomorrow") still lands correctly.

### The zero-knowledge chat model, with knowledge

The Ollama model is never told durable facts directly. On every chat
turn nix_core:

1. replays the current session turns (`nix_actions API /context`),
2. attaches a compact digest of verified knowledge (`/digest`: upcoming
   events + stored facts) to the system prompt,

so "what time is my meeting tomorrow" mid-conversation is answered
from real knowledge even though the chat model itself knows nothing.

## Key Finding Algorithm

Every utterance is scanned (deterministically, no model calls) for
micro-facts about the user's life - **Keys** - before the request is
routed:

    "my sister, named Maanvi is very naughty"
        -> user has a sister
        -> user's sister is named Maanvi
        -> sister Maanvi is naughty

Keys are extracted FIRST on every request (before routing) and stored
unconditionally - even when the routed operation fails or finds
nothing. Compound self-introductions are split into segments so each
clause yields its own keys:

    "So, My name is Sai Neela, people call me sai, I am born on
    February 25 2009, and love programmign and hardware and wish to
    pursue Computer Engineering in college"
        -> user's name is Sai Neela
        -> user goes by Sai
        -> user's birthday is February 25, 2009   (+reminder Feb 22)
        -> user loves programmign and hardware
        -> user wants to pursue Computer Engineering

and the reply acknowledges them: "Got it: your name is Sai Neela, you
go by Sai, ...". Self-introductions route deterministically to
`store_profile_keys` (the model gate never guesses on them) and are
never decomposed into calendar subtasks. Keys ride along in every
knowledge response under `result.keys_found`, appear in the chat
model's `/digest` so phi4-mini knows your people mid-conversation,
and surface in recall ("who is Maanvi" -> their keys). Chat-routed
utterances also teach keys via a `/keys` endpoint, so even a passing
"btw my brother Alex loves hiking" builds the profile. A bare "who is
<name>" question is sent to recall when the name is a known key
person, and to world chat otherwise. When a personal statement's
routed operation finds nothing, the reply acknowledges what was
learned: "Got it: you have a sister, your sister is named Maanvi...".

Attribute families extracted (all deterministic):

| Family | Example | Notes |
|---|---|---|
| identity | "my name is Alex" | supersedes old name |
| birthday | "my birthday is June 3" | auto-reminder 3 days before |
| favorites | "my favorite band is X" | supersedes per favorite |
| allergies/health | "i'm allergic to peanuts" | sensitive flag |
| relation attrs | "my sister works at NASA", "my son is 5" | works_at/lives_in/studies_at/age |
| work/school | "i work as an electrician" | supersedes |
| location/age | "i live in Dallas" | supersedes |
| goals | "i want to learn Spanish" | |
| routines | "i wake up at 6am", "gym every morning" | supersedes per routine |
| devices | "my phone is an iPhone 15" | supersedes per device |
| diet | "i'm vegetarian", "i never eat cilantro" | |
| sizes | "i wear size 10 shoes" | |
| contact | "my email is ..." | sensitive flag |
| nix prefs | "use metric", "speak spanish" | supersedes |

Supersession: a new value for a supersedable subject+predicate closes
the old record (status="superseded") instead of coexisting. Sensitive
keys carry `sensitive: true` and render a badge on the KB page.

Module: `nix_knowledge/keys.py`; tests: `tests/test_keys.py`.

## Scheduling flow (knowledge → actions)

Requests like "i have a dentist appointment tomorrow at 3pm" go to
`nix_knowledge /process`, which:

1. resolves the time (`TemporalResolver`),
2. creates a durable `calendar_event` record,
3. via `bridge.py` schedules the alarm/reminder in `nix_actions`,
   linked by `source_record_id`.

Cancelling or rescheduling through knowledge propagates automatically:
linked pending actions are cancelled/moved, and nix_core session turns
referencing the record are pruned (`/propagate`), so stale context can
never be mistaken for current context.

## Running

All three processes use the **nix_knowledge venv** (torch lives there;
nix_core itself needs only websockets + requests):

```bash
# 1. knowledge API (8100) - loads the Qwen model lazily
~/nix_knowledge/.venv/bin/python ~/nix_knowledge/scripts/knowledge_api.py

# 2. actions API (8200) - session store + scheduler
~/nix_knowledge/.venv/bin/python ~/nix_actions/scripts/actions_api.py

# 3. websocket gateway (9000)
~/nix_knowledge/.venv/bin/python ~/nix_core/ws_server.py
```

Configuration lives in `nix_core/config.py`, all overridable via env
(`NIX_KNOWLEDGE_API_URL`, `NIX_ACTIONS_API_URL`, `NIX_OLLAMA_API_URL`,
`NIX_OLLAMA_MODEL`, `NIX_AUTH_TOKEN`, `NIX_WS_PORT`, `NIX_TZ`, ...).

Point the chat model at your phi-4 + SearXNG Ollama server:

```bash
export NIX_OLLAMA_HOST=192.168.0.154
export NIX_OLLAMA_MODEL=phi4-mini:latest   # default; the model on .154
```

Websocket protocol (unchanged from the original gateway contract):

```
client -> {"token": "$NIX_AUTH_TOKEN"}                     first message
client -> {"text_prompt": "...", "location": "desk_area"}
server -> {"type": "status", "msg": "thinking"}
server -> {"type": "reply", "msg": "...", "route": "knowledge|chat", "rule": "..."}
```

## Files

| File              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `ws_server.py`    | websocket gateway, auth, per-connection loop              |
| `brain.py`        | pipeline: classify → route → service call → format reply  |
| `router.py`       | deterministic 3-way classifier (rules)                    |
| `router_corpus.py`| 1076 labeled requests (75 hand-written + generated realistic families: fillers, typos, ASR) |
| `hard_corpus.py`  | 60 unusual-pattern prompts; rules must never be wrong, abstains go to the model |
| `eval_router.py`  | accuracy report over the corpus                           |
| `test_router.py`  | pytest: corpus + edge cases                               |
| `console.py`      | self-hosted console: Console / Knowledge Base / Testing pages, live trace, speech input, realtime sandboxed pipeline tests |
| `request_log.py`  | JSONL request logging (one line per request, daily files)   |
| `analyze_logs.py` | log analysis: summaries, problem buckets, export            |
| `config.py`       | endpoints, token, model name, timeouts                    |

## Testing

```bash
cd ~/nix_core
~/nix_knowledge/.venv/bin/python -m pytest test_router.py hard_corpus.py -q
~/nix_knowledge/.venv/bin/python eval_router.py
```

Current state: **1076/1076 (100%)** on the main corpus, 0 misroutes on
the hard corpus (rules resolve ~57%, the rest abstain to the model by
design), avg routing time 0.1 ms per prompt.

## Test console

```bash
~/nix_knowledge/.venv/bin/python ~/nix_core/console.py
# or pin the port:
NIX_CONSOLE_PORT=35567 ~/nix_knowledge/.venv/bin/python ~/nix_core/console.py
```

Four pages, no external assets:

- **Console** — request box with **Speak** mic (Google speech-to-text
  via the Web Speech API; Chrome/Edge, optional auto-send), presets,
  dry-run classify, fire-due-actions, and the live background trace
  (knowledge /process, model gate, bridge, actions /log, digest).
- **Schedule** — upcoming events and their linked reminder/alarm
  actions with every parameter (start, end, all-day, source phrase,
  recurrence, fire times, status, errors), plus standalone actions and
  past/cancelled events.
- **Knowledge Base** — the actual store, parsed for humans: category
  sidebar (events / facts / preferences with counts), search, human
  field rendering (title, starts, repeats, stance...), per-record
  delete, plus session turns and the actions ledger.
- **Testing** — realtime pipeline runs: every prompt executes through
  the full Brain (model gate, chat model, bridge scheduling) against
  **sandboxed copies** of the live databases — real behavior, zero
  pollution. Per-prompt PASS/FAIL plus Nix's actual reply, timing
  averages, misroute and error counts, failures-only filter.

The sibling packages' own suites (151 tests) cover the knowledge
engine, bridge propagation, and the action scheduler.

## Request logs

Every request through the Brain is appended to
`~/nix_core/logs/requests-YYYY-MM-DD.jsonl`: timestamp, request,
route, rule, reply, latency, clause splits, and bounded pipeline
details. Sandbox test runs are excluded. Disable with
`NIX_REQUEST_LOG=0`, relocate with `NIX_REQUEST_LOG_DIR`.

After a testing session, review what to fix:

```bash
~/nix_knowledge/.venv/bin/python ~/nix_core/analyze_logs.py          # summary + problem buckets
~/nix_knowledge/.venv/bin/python ~/nix_core/analyze_logs.py --problems   # detailed problem list
~/nix_knowledge/.venv/bin/python ~/nix_core/analyze_logs.py --tail 20    # last 20 requests
~/nix_knowledge/.venv/bin/python ~/nix_core/analyze_logs.py --export bad.jsonl
```

Problem buckets: exceptions, unknown/error routes, engine errors,
API-unreachable replies, model-decided routes, slow (>8s) requests,
empty replies.
