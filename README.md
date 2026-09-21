# NIX / Casper

NIX is a modular personal-user companion platform. It combines deterministic personal knowledge, temporal memory, scheduled actions, conversational Core orchestration, and a future trigger-driven decision layer.

The conversational identity is **Casper**, a Personal User Companion Agent (PUCA), created and built by Sai Neela and living in NIX's PUCA system.

## Architecture

```text
Client / voice gateway
        |
        v
nix_core  --->  nix_knowledge  --->  nix_actions
   |                 |                   |
 Casper chat     durable memory       schedules/actions
   |
 nix_decision (future proactive trigger layer)
```

| Package | Responsibility |
|---|---|
| [`nix_core`](nix_core/) | Request routing, conversation state, Casper responses, websocket gateway, and dashboard console. |
| [`nix_knowledge`](nix_knowledge/) | Durable facts, people, temporal states, calendar events, retrieval, and symbolic/neural parsing. |
| [`nix_actions`](nix_actions/) | Deterministic action capture, reminders, scheduling, session storage, and propagation. |
| [`nix_decision`](nix_decision/) | Initial foundation for future explicit triggers that may ask Core to check in with a user. It currently has no calling rules. |
| [`testing-echo-connect`](testing-echo-connect/) | Optional Echo/Home Assistant integration prototype; not required by the main stack. |

## Current principles

- **Core owns speech and planning.** Knowledge returns grounded context; Casper creates the final user-facing response. Core uses deterministic safety rules plus the small constrained Nix_predictor for indirect and multi-intent planning.
- **Knowledge is durable.** Facts, people, temporal states, and events are stored separately and temporal updates supersede prior states.
- **Actions are deterministic.** Natural-language interpretation belongs upstream; Actions schedules and executes captured operations.
- **Decision is explicit and gated.** Future proactive triggers must be supplied explicitly and can only reach Core when the user is known to be home.
- **Relative time becomes absolute time.** Requests such as `tmr`, `in 2 more days`, and `next Monday` are resolved in the configured timezone before storage or response formatting.
- **Private runtime data stays local.** Databases, logs, model weights, adapters, credentials, environments, and caches are ignored by Git.

## Requirements

- Python 3.10+
- Per-package dependencies installed in the appropriate environment
- Optional local model files for Casper and the Knowledge model gate
- NVIDIA CUDA is supported for the local Transformers/PEFT Casper runtime

Configuration is documented in [`.env.example`](.env.example). Copy it to `.env`, set a unique `NIX_AUTH_TOKEN`, and never commit the resulting file.

## Run the stack

From the repository root, start the services in separate terminals:

```bash
python nix_knowledge/scripts/knowledge_api.py
python nix_actions/scripts/actions_api.py
python nix_core/console.py       # browser dashboard, optional
python nix_core/ws_server.py     # websocket gateway, when needed
```

The console normally serves on the configured `NIX_CONSOLE_PORT` (the current development setup uses port `35567`). The websocket gateway normally uses `NIX_WS_PORT`.

For a complete model/runtime setup, see [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md).

## Test

Run package suites from the repository root:

```bash
(cd nix_knowledge && python -m pytest tests -q)
(cd nix_actions && PYTHONPATH=../nix_knowledge:. python -m pytest tests -q)
(cd nix_core && python -m pytest -q)
PYTHONPATH=. python -m pytest -q nix_decision/nix_decision/test_engine.py
```

Some Core tests require local services or model backends. Prefer the hermetic/package tests for ordinary changes, and use temporary databases for integration experiments.

## Documentation map

- [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) — detailed model lineage, runtime paths, temporal fixes, service operations, known caveats, and future-agent handoff.
- [`nix_core/README.md`](nix_core/README.md) — request flow, dashboard, websocket protocol, and Casper integration.
- [`nix_knowledge/README.md`](nix_knowledge/README.md) — memory model, temporal parser, APIs, and isolated testing.
- [`nix_actions/README.md`](nix_actions/README.md) — action lifecycle, scheduler, sessions, and propagation.
- [`nix_decision/README.md`](nix_decision/README.md) — future proactive decision foundation and safety gates.

## Repository hygiene

Do not commit:

- `.env` files or credentials
- SQLite databases and request logs
- Model weights, QLoRA adapters, datasets, or generated checkpoints
- Virtual environments and Python caches
- Runtime PID/temp files

Use `.gitignore` and keep production/personal data outside tracked source files.
