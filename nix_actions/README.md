# nix_actions

`nix_actions` is NIX's deterministic action and scheduling service. Upstream components decide **what** should happen; Actions records **when** it should happen, tracks its lifecycle, and eventually executes the appropriate handler.

Actions does not interpret natural language and does not invent policy.

## Responsibilities

- Capture reminders, alarms, notifications, webhooks, and maintenance actions.
- Validate timezone-aware schedules and payloads.
- Track pending, fired, failed, cancelled, and rescheduled states.
- Maintain day/week session labels for dashboard and audit views.
- Propagate Knowledge event changes to linked actions.
- Provide a mock runner for safe development and testing.
- Expose HTTP and CLI interfaces.

## Relationship to other packages

| Package | Responsibility |
|---|---|
| `nix_core` | Conversational routing and final response generation. |
| `nix_knowledge` | Durable memory and natural-language event interpretation. |
| `nix_actions` | This package: deterministic capture, scheduling, and execution. |
| `nix_decision` | Future explicit proactive trigger evaluation. |

## Action lifecycle

```text
Knowledge/Core capture
        |
        v
validated action + source record link
        |
        v
pending -> fired / failed / cancelled
```

Actions retain `source` and `source_record_id` so a cancelled or rescheduled Knowledge event can update its reminders. Changes are also recorded in the audit trail.

## Python API

```python
from datetime import datetime
from nix_actions import ActionsEngine

engine = ActionsEngine("actions.db", timezone="America/Chicago")

engine.capture(
    action_type="reminder",
    scheduled_for=datetime.now().astimezone(),
    payload={"message": "Check robotics schedule"},
    source="nix_knowledge",
    source_record_id=42,
    knowledge_type="calendar_event",
    session_bucket="week",
)
```

The main engine supports capture, listing, statistics, sessions, audit events, cancellation, rescheduling, propagation, due-action execution, and cleanup.

## HTTP service

Start the API from the repository root:

```bash
python nix_actions/scripts/actions_api.py
```

The service commonly runs on port `8200`. Important endpoints include:

- `GET /health` — service and scheduler status.
- `GET /sessions` — grouped chat/action session information.
- `POST /log` — record a conversation turn.
- `GET /context` — current active session context.
- `POST /run` — run due actions, preferably with mock mode during development.
- `POST /propagate` — apply Knowledge lifecycle changes.

## CLI

When installed through the package entry point:

```bash
nix-actions capture --type reminder --at "2026-09-21T09:00" \
  --payload '{"message":"robotics practice"}' --source nix_knowledge
nix-actions run --mock
nix-actions list --status all
nix-actions events --kind captured --limit 20
nix-actions cancel --action-id 6
```

The dashboard is optional and uses a standard-library terminal interface:

```bash
nix-actions dashboard
```

## Sessions

Actions use labels rather than opaque conversation ownership:

- `day-YYYY-MM-DD`
- `week-YYYY-MM-DD` (Monday start)

Conversation/session storage is used by Core for active context. Knowledge records themselves are never cleared by session rollover.

## Safety

- Use `run --mock` while developing handlers.
- Do not execute tests against the live personal database.
- Use temporary SQLite paths for integration tests.
- Keep credentials and device APIs outside tracked source files.

## Tests

From the repository root:

```bash
(cd nix_actions && PYTHONPATH=../nix_knowledge:. python -m pytest tests -q)
```

The suite covers capture validation, scheduling, propagation, session behavior, handler boundaries, and maintenance.
