# nix_actions

Deterministic (no-AI) action scheduler, capture service, and hardware
trigger layer for the Nix ecosystem. **Knowledge decides *what* should
happen; Actions decides *when it fires* and *executes it*.**

This package contains no intelligence: it never interprets natural
language and never decides policy. It captures, organizes, tracks, and
executes.

## Roles in the Nix ecosystem

| Component      | Role                                                                 |
|----------------|----------------------------------------------------------------------|
| `nix_core`     | Decides what to do with a user request; holds session history only   |
| `nix_knowledge`| Durable knowledge base; **never clears**; pushes actions here         |
| `nix_actions`  | Capture service + scheduler + executors (this package)               |

## nix_core (session brain)

`nix_core` holds **conversations only** — day/week-bucketed sessions
that clear when their bucket rolls over. It never stores durable
facts; it pulls knowledge from a `KnowledgeProvider` (the interface
nix_knowledge implements; a `StaticKnowledgeProvider` ships for
standalone use) and schedules outcomes by capturing actions into
`nix_actions`.

```python
from nix_core import NixCore, StaticKnowledgeProvider, KnowledgeRecord

core = NixCore(
    knowledge=StaticKnowledgeProvider(records=[...]),
    database_path="nix_core.db",
    actions_database_path="actions.db",
)

response = core.handle_request(text="remind me about the robotics class")
response.reply, response.session_tag, response.actions_captured
```

Staleness rule: every turn stores which knowledge records (and
versions) it referenced. When a record changes upstream, turns
referencing it are **pruned from the context window** (kept on disk
for audit only) and the linked actions are cancelled — so a stale log
entry can never be mistaken for current context.

```bash
nix-core ask "remind me about the robotics class"
nix-core context            # exactly what the brain would see now
nix-core sessions           # per-session stats (active vs pruned)
nix-core prune --record-id 7 --knowledge-type task
nix-core maintain           # clear rolled-over sessions, purge actions
nix-core demo               # full propagation walkthrough
```

Session tags (`week-2026-08-31`, `day-2026-09-06`) are identical to
nix_actions', so both dashboards/CLI tools speak the same session
language.

## Capture service model

Upstream components call `ActionsEngine.capture()` instead of touching
the database directly:

```python
from nix_actions import ActionsEngine

engine = ActionsEngine("actions.db", timezone="America/Chicago")

action = engine.capture(
    action_type="reminder",          # alarm | reminder | notify | webhook | maintenance
    scheduled_for=when,              # timezone-aware datetime
    payload={"message": "..."},
    source="nix_core",               # nix_core | nix_knowledge | knowledge | user | cli | demo
    source_record_id=42,             # upstream record this came from (optional)
    knowledge_type="task",           # upstream record type (optional)
    session_bucket="week",           # day | week
)
```

`capture()` validates the source, tags the action with a session, and
writes a `captured` event to the audit trail.

## Sessions

- `session_bucket="day"` → tags like `day-2026-09-06`
- `session_bucket="week"` → Monday-based tags like `week-2026-08-31`

Sessions are *labels*, so the dashboard can group and count them. The
clearing policy lives upstream: nix_core clears each day/week session
when it rolls over and prunes entries whose knowledge was updated, while
nix_knowledge never clears.

## Mock mode (current stage)

Handlers are still stubs. `run_due(mock=True)` (or `nix-actions run
--mock`) flips statuses and records events **without executing any
handler code**, so the dashboard is a true capture service today.
Later, real device APIs, music services, etc. replace the stub bodies
in `handlers.py` — and `--mock` is simply dropped.

## Upstream lifecycle propagation

Actions remember the upstream record they came from
(`source` + `source_record_id`), and lifecycle changes **propagate to
nix_actions automatically** — an event scheduled by nix_knowledge that
is cancelled upstream is cancelled here too.

```python
# upstream event was cancelled -> linked actions get cancelled
engine.cancel(source_record_id=42, reason="event cancelled upstream")

# upstream event moved to a new time -> linked actions move too,
# session tag is re-derived from the new time
engine.reschedule(
    source_record_id=42,
    scheduled_for=new_time,
    reason="postponed",
)

# upstream record content was edited -> old actions cancelled and
# marked for nix_core to prune out of its session history
engine.on_knowledge_updated(
    source_record_id=42,
    knowledge_type="task",
    reason="task edited",
)
```

Record-based cancellation/reschedule matches actions from **any
upstream source** (`knowledge`, `nix_core`, `nix_knowledge`), so it
works no matter which component captured the action. Narrow the match
with `sources=("nix_knowledge",)` or `knowledge_type="task"` if
needed. Actions from local sources (`user`, `cli`, `demo`) are never
touched by upstream record changes.

Every propagation lands in the audit trail as `cancelled`,
`rescheduled`, or `knowledge_updated` events — visible live in the
dashboard's EVENTS tab.

## Dashboard

```bash
nix-actions dashboard                # uses ./actions.db
nix-actions --database nix.db dashboard --refresh 1
```

Live full-screen view with four tabs:

| Key | View       | Shows                                                       |
|-----|------------|-------------------------------------------------------------|
| `1` | ACTIONS    | Captured actions: status, type, when, source, session, rec  |
| `2` | EVENTS     | Audit trail: captured / fired / failed / cancelled / purged |
| `3` | SESSIONS   | Day/week buckets with per-status counts                     |
| `4` | HANDLERS   | Registered handlers and their binding status                |

Other keys: `↑`/`↓` cycle the status filter (actions tab), `r` run due
(mock), `k` cancel pending, `p` pause refresh, `q` quit. Stdlib only —
no external TUI dependency.

## CLI

```bash
nix-actions capture --type alarm --at "2026-09-07T07:00" \
    --payload '{"message":"standup"}' --source nix_core
nix-actions run --mock           # fire due actions (capture mode)
nix-actions run --loop --mock    # persistent mock runner
nix-actions list --status all --session week-2026-08-31 --source nix_core
nix-actions events --kind captured --limit 20
nix-actions cancel --action-id 6
```

## API surface

- `ActionsEngine` — capture / schedule / list / stats / sessions /
  events / run_due / cancel / on_knowledge_updated / purge_finished
- `Action`, `CaptureEvent`, `SessionInfo` — frozen dataclasses
- `handlers.get_handler`, `handlers.register_handler` — bind real
  device APIs per action type

## Tests

```bash
python -m pytest tests/ -q
```
