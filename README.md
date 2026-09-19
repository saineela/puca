# Nix

Nix is a modular personal-assistant stack with deterministic routing, durable knowledge, scheduled actions, and a websocket gateway.

## Repository layout

- `nix_core/` — request routing, conversation orchestration, websocket gateway, console, and routing evaluations.
- `nix_knowledge/` — durable facts/events, temporal resolution, semantic retrieval, and HTTP API.
- `nix_actions/` — deterministic action capture, scheduling, session storage, and HTTP API.
- `nix_decision/` — standalone decision-engine prototype with its own tests.
- `testing-echo-connect/` — optional Home Assistant/Echo integration prototype; it is not required by the Nix services.

Runtime databases, models, logs, virtual environments, caches, and credentials are intentionally excluded from version control.

## Requirements

- Python 3.10+
- A virtual environment with the dependencies declared by the package you are running
- `requests` and `websockets` for the core gateway
- Optional local model dependencies and model files for semantic/model-gate features

The services can run independently for development. The knowledge and actions APIs use SQLite files under `data/` by default; set `NIX_DATA_DIR` or the individual database variables to place them elsewhere.

## Configuration

Copy `.env.example` to `.env` and provide a unique `NIX_AUTH_TOKEN` before starting the websocket gateway. Never commit `.env`, database files, model weights, logs, or API credentials.

Important variables include:

- `NIX_AUTH_TOKEN` — required websocket authentication secret.
- `NIX_KNOWLEDGE_API_URL` / `NIX_ACTIONS_API_URL` — sibling service URLs.
- `NIX_KNOWLEDGE_DB`, `NIX_ACTIONS_DB`, `NIX_CORE_DB` — SQLite paths.
- `NIX_DATA_DIR` — default directory for runtime databases.
- `NIX_OLLAMA_API_URL` / `NIX_OLLAMA_MODEL` — optional chat backend.
- `NIX_WS_HOST` / `NIX_WS_PORT` — websocket bind configuration.

## Running locally

Run each service from the repository root with the same Python environment:

```bash
python nix_knowledge/scripts/knowledge_api.py
python nix_actions/scripts/actions_api.py
python nix_core/ws_server.py
```

The console is optional:

```bash
python nix_core/console.py
```

## Testing

Run the deterministic and hermetic suites first:

```bash
(cd nix_knowledge && python -m pytest tests -q)
(cd nix_actions && PYTHONPATH=../nix_knowledge:. python -m pytest tests -q)
(cd nix_core && python -m pytest test_router.py hard_corpus.py -q)
python -m nix_decision.test_engine
python -m nix_decision.test_knowledge
```

Evaluation and benchmark scripts are under `nix_core/` and `nix_knowledge/scripts/`. The subprocess E2E suite uses temporary databases and skips the live chat assertion when its optional backend is unavailable.

## Security and release notes

- Supply secrets through environment variables; no development secret is embedded in the source.
- Do not run tests against live databases. Use temporary paths for experiments.
- Model weights and downloaded datasets are local deployment artifacts, not source-distribution files.
- The Echo/Home Assistant prototype is isolated from the core service path and requires its own credentials.
