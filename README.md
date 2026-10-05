<p align="center">
  <img src="https://res.cloudinary.com/dh5uxc6ql/image/upload/v1790917615/93d18a47-5f3a-46e1-ad71-705b2680442f_anp8f1.png" alt="NIX PUCA" width="360">
</p>

<p align="center">
  <strong>A local-first Personal User Companion Agent foundation.</strong><br>
  Structured memory. Deterministic actions. A configurable conversational model.
</p>

<p align="center">
  <a href="#what-is-a-puca">Overview</a> ·
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#benchmarks-and-model-evaluation">Benchmarks</a> ·
  <a href="ROADMAP.md">Roadmap</a>
</p>

<p align="center">
  <a href="https://github.com/saineela/puca">Repository</a> ·
  <a href="nix_core/README.md">Core docs</a> ·
  <a href="nix_knowledge/README.md">Knowledge docs</a> ·
  <a href="nix_actions/README.md">Actions docs</a> ·
  <a href="nix_decision/README.md">Decision docs</a>
</p>

NIX PUCA is a modular software foundation for a **Personal User Companion Agent (PUCA)**: a self-hosted personal assistant built around explicit boundaries between conversation, durable memory, temporal interpretation, and scheduled actions. It is a Python project with local SQLite persistence and optional local or separately operated model backends.

The design is intentionally more structured than a single prompt wrapped around an LLM. NIX Core routes and composes requests; NIX Knowledge stores and retrieves personal records; NIX Actions manages scheduled work. A configured conversational model produces natural-language responses, while application code validates and manages persistence and scheduling.

**Project identity:** the user-facing assistant identity is **Casper**, created and built by Sai Neela in NIX. Casper is separate from the model adapters: the local Transformers selector currently defaults to **Luna V6**, and any supported backend can be selected through configuration.

> **Project status:** this repository is a development foundation, not a certified or hosted product. Model quality, device compatibility, latency, VRAM use, and privacy behavior depend on local configuration and require deployment-specific validation. No license file has been added yet — see [License and contributions](#license-and-contributions).

## What is a PUCA?

**PUCA** means **Personal User Companion Agent**. In NIX it describes a personal-assistant architecture where conversational language generation is only one part of the system. Personal facts, event times, reminders, presence signals, device access, and external integrations are handled by separately defined services and policies — never assumed to be correct just because a model generated text. PUCA is a project term for this architecture, not a standardized product category.

## Why NIX

A personal assistant needs to do more than answer one prompt. It needs to distinguish a question from a request to store something, resolve “tomorrow” in the configured timezone, keep a reminder linked to the event that created it, ask when a person is ambiguous, and never claim an action succeeded before the scheduler confirms it.

NIX makes these responsibilities visible in code:

- **Core routes and composes.** A deterministic route engine runs before model/service work; Core remains the final response boundary.
- **Knowledge stores and grounds.** Facts, relationships, changing person states, and events are persisted locally and returned as structured records.
- **Actions schedules captured operations.** It does not independently interpret natural language or invent policy.
- **Decision is gated.** The current package is a foundation only; it has no registered proactive calling rules.
- **Models are replaceable components.** Transformers/PEFT, Ollama, and TabbyAPI paths are optional and have different runtime requirements.

## Features

### Available in this repository

- **Deterministic Core routing:** classifies common social, world-chat, personal, temporal, and action requests before downstream work. Ambiguous requests may abstain rather than silently execute a model-selected action.
- **Durable local Knowledge:** SQLite records for facts, preferences, people, relationships, temporal states, and calendar events.
- **Temporal interpretation:** converts supported relative dates and times to timezone-aware timestamps before storage or presentation.
- **Deterministic scheduling:** Actions tracks pending, fired, failed, cancelled, and rescheduled operations and can link them to source Knowledge records.
- **Conversation continuity:** bounded session history and clarification continuation scoped to a conversation.
- **Device skills:** targeted device requests (for example ring-light power, color, and brightness) run through a validated plan pipeline — optional local prompt cleansing, per-step planning, Core schema validation, device execution, and readback — before the assistant confirms anything. See [Device skill pipeline](#device-skill-pipeline).
- **Core console:** browser dashboard, traces, conversations, event and memory views, model settings, and local administration.
- **OpenAI-compatible API:** `GET /v1/models` and `POST /v1/chat/completions` for compatible clients. API Token Guard rejects known Open WebUI follow-up/title/tag metadata jobs before Core/model invocation so they do not pollute conversations.
- **WebSocket text gateway:** token-authenticated text transport for client/voice-gateway integrations. This is not, by itself, a complete speech-recognition or text-to-speech system.
- **Optional semantic retrieval:** local Sentence Transformers/BGE embeddings when dependencies and model files are supplied.
- **Optional model backends:** local Transformers/PEFT adapter selection, Ollama, and a TabbyAPI HTTP client. NIX does not include or launch those separately operated services automatically.
- **Isolated integration prototype:** `testing-echo-connect/` explores a future Echo/Home Assistant boundary and is not on the primary request path.

### Planned, not yet implemented

The [**roadmap**](ROADMAP.md) details proposed Nix-Skills, Android access, real-time voice, open-source Web Search, ESP32/OPNsense presence, opt-in location/timeline integrations, and Echo Dot research. The planned NIX Home mobile client has an [API and screen-flow blueprint](nix_core/MOBILE_APP_API.md); no mobile app or authenticated mobile API suite is implemented. These are future plans, not currently available features. In particular, the community Skills marketplace saves bounded static files and does **not** execute community code; the catalog's Web Search entry is not a working search connector. (Built-in device skills are a separate, code-owned pipeline described above.)

## Architecture

<img src="https://res.cloudinary.com/dh5uxc6ql/image/upload/v1791167887/Gemini_Generated_Image_h8fdzkh8fdzkh8fd_p0v2qc.jpg" alt="NIX PUCA" width="900">

| Component | Responsibility | Current boundary |
| --- | --- | --- |
| [`nix_core/`](nix_core/) | Routes requests, holds conversation context, composes the final reply, hosts the console/API and WebSocket gateway. | The configured conversation backend generates natural-language output; Core owns the request pipeline. |
| [`nix_knowledge/`](nix_knowledge/) | Persists and retrieves personal knowledge; interprets supported calendar/person-state language. | Structured operations and deterministic validation govern writes; local model artifacts are optional and ignored by Git. |
| [`nix_actions/`](nix_actions/) | Captures, schedules, runs, cancels, and reschedules supported actions. | Deterministic engine; upstream Core/Knowledge decide what to capture. |
| [`nix_decision/`](nix_decision/) | Defines a future trigger/state gate for possible proactive behavior. | Foundation only; no calling rules are enabled. |
| `testing-echo-connect/` | Isolated Echo/Home Assistant integration prototype. | Optional experiment, not required by the main stack. |

### Simple request walkthrough

For “I have robotics practice tomorrow at 5pm; remind me,” the intended pipeline is:

1. Core identifies a personal/action request and forwards it to Knowledge.
2. Knowledge resolves “tomorrow at 5pm” using `NIX_TZ`, validates the event details, and persists structured data.
3. The Knowledge-to-Actions bridge captures a linked reminder, if the request and operation are valid.
4. Core composes a response using the structured result. It does not state that the reminder succeeded unless Actions confirmed it.

The exact outcome depends on the current route, service health, database, timezone, and model configuration. Use isolated test databases for experiments.

### Device skill pipeline

Requests that target a device — for example a ring light's power, color, or brightness — never go straight from model prose to hardware. They run through a staged, code-validated pipeline:

```text
user request
  -> Core router (skill vs chat)
  -> Qwen3-0.6B prompt cleanser   (optional; rewrites vague wording into
                                   explicit device-anchored steps; fail-closed
                                   to the user's own words)
  -> Needle planner               (one step at a time, NIX_SKILL_PLAN envelope,
                                   at most 8 steps)
  -> Core schema validation       (skills, tools, arguments; exact RGB
                                   preservation; intent preservation)
  -> Skill worker execution       (device command + readback)
  -> verified acknowledgment      (sent only after confirmed results)
```

The cleanser is an enhancement, never a gate: any model error or malformed output falls back to the user's own words, and fully grounded requests skip it. A plan that is missing, malformed, incomplete, or invalid executes no action, and the final reply is checked against the device's confirmed results before it is sent.

## High-level concepts, in plain language

| Term | Simple definition |
| --- | --- |
| **Local-first** | Core data and services are designed to run on infrastructure you control; optional backends and integrations may still use network services. |
| **NIX Core** | The request coordinator: chooses a route, calls the needed services, and prepares the final reply. |
| **NIX Knowledge** | The structured memory service: stores and retrieves authorized facts, people, and calendar records. |
| **NIX Actions** | The scheduler: tracks supported reminders and other captured work through their lifecycle. |
| **Nix-Skills** | The planned format/runtime for optional capabilities and integrations; it is not executable in the current repository. |
| **LLM** | A large language model that generates or interprets text. NIX uses it as a replaceable component rather than as the database or scheduler. |
| **Adapter / LoRA / QLoRA** | An adapter adds learned parameters to a base model; LoRA is a low-rank adapter method, and QLoRA trains LoRA adapters while loading a quantized base model. Neither replaces the surrounding application rules or data store. |
| **Prompt cleanser** | A small local model (Qwen3-0.6B) that rewrites a vague device request into explicit steps before planning. It can only rephrase what the user asked for. |
| **Device skill** | A built-in, code-owned capability (such as ring-light control) that executes through the validated plan pipeline above. |
| **Readback** | Reading the device's actual state after a command, so a confirmation reflects reality rather than the model's claim. |
| **Router** | A decision layer that selects a request path, such as chat or Knowledge. It does not generate the final answer; routing latency is not full-response latency. |
| **p50 / p95 latency** | p50 is the median time across samples; p95 is the time at or below which 95% of measured samples completed. |
| **Requests per second (RPS)** | A throughput calculation for the timed benchmark operation; it is not a count of full model-generated conversations per second. |
| **Durable memory / Knowledge** | Structured records that can be read later, rather than relying only on what is in the current chat prompt. |
| **Temporal grounding** | Converting phrases such as “tomorrow at 5” into a timezone-aware date and time before saving or reporting them. |
| **Action scheduler** | A deterministic service that tracks when captured reminders or other supported operations are due. |
| **VRAM** | GPU memory. Requirements depend on model weights, quantization, runtime, context size, and concurrent workloads. |
| **Presence detection** | A signal that may estimate whether a device/person is at home or in a room; it is not automatically identity proof or precise location. |
| **SQLite** | A database stored in a local file; NIX uses it for structured records and service state. |
| **OpenAI-compatible API** | An HTTP request/response format supported by many clients; compatibility does not imply that NIX is OpenAI or uses OpenAI-hosted models. |

## Built with

- **Python 3.10+** service and package code.
- **SQLite** local Knowledge, Actions, and Core/session stores.
- **Python standard library HTTP servers** for the included APIs and dashboard host.
- **NumPy** for an optional CPU-first route predictor.
- **Sentence Transformers** for optional local semantic embeddings.
- **Transformers, PEFT, PyTorch, bitsandbytes, and CUDA** for optional local adapter inference.
- **Ollama** or **TabbyAPI** as optional, separately run model services.

The packages have separate `pyproject.toml` metadata. There is no single root lockfile, one-command production installer, container deployment, or hosted service definition in this repository.

## Requirements

- Python 3.10 or later.
- A compatible Python environment and the dependencies required by the selected package/service.
- `pytest` for tests.
- Optional semantic search: `sentence-transformers`, local BGE model files, and available CPU or CUDA resources.
- Local Transformers inference: compatible PyTorch, Transformers, PEFT and bitsandbytes versions; NVIDIA/CUDA is required by the current local Casper/Luna GPU loaders.
- Ollama or TabbyAPI inference: the backend installed and configured separately, plus a compatible model loaded by that service.

Weights, adapters, datasets, and third-party services are not bundled with this repository. A successful package install alone does not provision model files or external inference servers.

## Getting started

### 1. Create a Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./nix_knowledge -e ./nix_actions
python -m pip install requests websockets numpy
```

The command above installs the sibling packages and basic Core/HTTP dependencies. It does not install `pytest`, Sentence Transformers, a CUDA-enabled PyTorch build, Transformers/PEFT/bitsandbytes, or a model. Add only the optional stack needed for your chosen use case.

### 2. Configure trusted local settings

Use [`.env.example`](.env.example) as a **reference**. The scripts read environment variables from the process environment; they do not automatically load `.env.example`. Configure values through your shell, service manager, or another protected secret mechanism.

At minimum, set a unique `NIX_AUTH_TOKEN` before running the WebSocket gateway. Set `NIX_OPENAI_API_KEY` before allowing OpenAI-compatible clients beyond a trusted local machine/network. Never commit local environment files or secrets.

### 3. Start the dashboard/API

For a local-only console, run:

```bash
NIX_CONSOLE_HOST=127.0.0.1 python nix_core/console_extend.py
```

Open the local URL printed on startup. The console serves the dashboard, `/api/*`, and the OpenAI-compatible `/v1/*` endpoint from one port. It reuses reachable Knowledge/Actions APIs; otherwise, it can host their handlers in-process through an internal bridge.

The console binds `0.0.0.0:49117` by default; the port is fixed across restarts and dashboard reloads. Set `NIX_CONSOLE_HOST=127.0.0.1` to restrict access to this machine; there is no port override. The console exposes private records and administrative operations and is not an internet-facing security boundary by itself. Binding to all interfaces permits LAN access when host firewall/network policy allows, but does not configure internet access or router port forwarding.

### Run Knowledge and Actions separately

If you want separate API processes, start each in its own terminal:

```bash
python nix_knowledge/scripts/knowledge_api.py  # default 127.0.0.1:8100
python nix_actions/scripts/actions_api.py      # default 127.0.0.1:8200
```

Then start the console with service URL settings as needed. Do not run separate services against live personal data during tests unless you intend that behavior.

### Optional WebSocket text gateway

```bash
NIX_AUTH_TOKEN='replace-with-a-long-random-secret' \
  NIX_WS_HOST=127.0.0.1 \
  python nix_core/ws_server.py
```

The gateway listens on port `9000` by default. It accepts authenticated text prompts. A separate voice client may provide speech input/output; the included gateway alone is not speech recognition or speech synthesis.

### Default service bindings

| Service | Default | Purpose |
| --- | --- | --- |
| Knowledge API | `127.0.0.1:8100` | Knowledge HTTP endpoints. |
| Actions API | `127.0.0.1:8200` | Scheduler/session HTTP endpoints. |
| Core WebSocket | `0.0.0.0:9000` | Authenticated text transport; restrict its interface as appropriate. |
| Core console | `0.0.0.0:49117` | Dashboard, `/api/*`, and `/v1/*`; set `NIX_CONSOLE_HOST` to restrict its interface. |

## Configuration, privacy, and security

Common settings are documented in [`.env.example`](.env.example):

| Variable | Description |
| --- | --- |
| `NIX_AUTH_TOKEN` | Shared token required by WebSocket clients. |
| `NIX_OPENAI_API_KEY` | Optional Bearer token for OpenAI-compatible clients; empty means no API-token check and should be restricted to trusted local use. |
| `NIX_DATA_DIR` | Local database directory; defaults to `./data`. |
| `NIX_TZ` | Timezone for resolving/displaying local dates; default example is `America/Chicago`. |
| `NIX_KNOWLEDGE_API_HOST` / `NIX_KNOWLEDGE_API_PORT` | Knowledge service bind host and port. |
| `NIX_ACTIONS_API_HOST` / `NIX_ACTIONS_API_PORT` | Actions service bind host and port. |
| `NIX_WS_HOST` / `NIX_WS_PORT` | WebSocket bind host and port. |
| `NIX_CONSOLE_HOST` | Dashboard/API bind host; port is fixed at `49117`. |
| `NIX_CASPER_BACKEND` | `transformers`, `ollama`, or `tabby` transport selection. |
| `NIX_SKILL_CLEANSE_MODEL` | Ollama model used for skill prompt cleansing; default `qwen3:0.6b`. |
| `NIX_REQUEST_LOG` / `NIX_REQUEST_LOG_DIR` | Core request logging (enabled by default) and its output directory. Logs do not rotate automatically. |

Important operational notes:

- The console contains memory, conversation, deletion, and full-reset controls. Do not expose it to untrusted networks without adding and validating an appropriate authentication/proxy boundary.
- Use API tokens for network clients, protect backend credentials, and scope service binds to trusted interfaces.
- SQLite databases, WAL files, backups, request JSONL files, model files, adapters, and datasets are private/runtime/provenance data, not source files. They are ignored by Git; `.gitignore` is not encryption or access control.
- Core request logs can contain user text and response content. Disable logging with `NIX_REQUEST_LOG=0` if that fits your requirements, or define access controls, retention, and backup policy before enabling it in a shared environment.
- Do not connect broad tests to a real personal database. Use disposable test databases and verify destructive test behavior.

## Models and inference

NIX treats every language model as a replaceable component. All backends below are optional and configured locally; model weights are never bundled with the repository.

| Backend | What it is | Notes |
| --- | --- | --- |
| Transformers + PEFT | Local PyTorch inference with LoRA/QLoRA adapters | Source default selects the Luna V6 adapter (`luna-v6-contextual-v1`). Casper V5 (`casper-puca-qlora-v5`) and Casper V6 (`casper-puca-qlora-v6-final`, beta) are selectable alternatives. |
| Ollama | Separately installed local model server | Configured with `NIX_OLLAMA_MODEL` (default `qwen3.5:4b`); NIX never launches it automatically. |
| TabbyAPI | Separately installed OpenAI-compatible server | Optional HTTP client in Core (`NIX_CASPER_BACKEND=tabby`); verify artifact compatibility before enabling. |
| Qwen3-0.6B (Ollama) | Small utility model for skill prompt cleansing | Rewrites vague device requests into explicit steps; fail-closed to the user's own words. |
| Qwen 2.5 0.5B | Optional Knowledge route gate (`Nix_predictor`) | Disabled by default (`NIX_CORE_USE_KNOWLEDGE_MODEL_GATE=0`). |

**Luna Pro v1 is retired.** It is no longer registered for selection, its former direct API endpoints return HTTP 410 (`model_retired`), and its research artifacts are archived for provenance only. Luna development is limited to the V6 runtime/integration path.

These statements describe source code and configuration, not any particular machine: weights must be downloaded separately, and `/api/health` or `/api/model` on a running instance reports what is actually loaded.

### GPU memory

Conversation models run on GPU; the project was developed on an RTX 4060 (8 GiB). Local loaders default to a `0.68` per-process CUDA memory fraction. Development records measured roughly 3–4.3 GiB for Casper and about 2.3 GiB inference / 3.1 GiB training for Luna-class adapters. Actual use depends on weights, quantization, runtime, context length, and other GPU workloads — measure your own configuration before drawing conclusions. The repository's router microbenchmark measures a Python classifier, not model generation; its timings are not end-to-end latency.

### Optional TabbyAPI / ExLlamaV2

NIX Core includes a TabbyAPI-compatible HTTP **client**, not an ExLlamaV2 server. TabbyAPI must be installed and run separately with a verified compatible model artifact. The current PEFT/QLoRA adapters are not directly verified as ExLlama loadable. Configure this backend only after validating the actual model format and compatibility; see [`.env.example`](.env.example).

## Benchmarks and model evaluation

### Reproducible router-classifier microbenchmark

The repository includes a safe, read-only microbenchmark for `nix_core/router.py::classify`:

```bash
PYTHONPATH=nix_core python nix_core/benchmarks/speed_benchmark.py \
  --benchmark router --repeats 100 --warmup 10
```

A 100-sample run in the current development workspace reported **p50 0.106 ms**, **p95 0.145 ms**, mean **0.103 ms**, minimum **0.031 ms**, and maximum **0.189 ms** on the harness's built-in prompt set (10 warm-up samples; all 100 measured samples succeeded). The harness-calculated throughput was **9,673 requests/second** for this timed operation.

Timing covers only the classifier call and JSON serialization of the route and matched rule — not HTTP/database work, model loading, or response generation — and reflects one local software environment. Rerun the command in your own environment before making comparisons. See [`nix_core/benchmarks/README.md`](nix_core/benchmarks/README.md) for targets, limitations, and usage.

### Archived model-quality and latency evidence

Small project-local evaluation suites are engineering checks, not human-subject studies or standardized language-quality scores. Historical Casper V6 notes report **7/8** in one policy-conditioned single-turn set and **1/5** in the latest documented direct multi-turn check; the V6 candidate is not promoted as the default. A historical isolated generation benchmark reported p50 **1.64 s** and max **4.28 s** over five short prompts after an approximately **8.35 s** load. The environment and protocol are in [`CASPER_V6_RESEARCH.md`](CASPER_V6_RESEARCH.md), and are not a claim of current serving performance.

Historical Luna contextual scores were mixed and were recorded with synthetic injected context as well as isolated mode; synthetic evaluator context is not a full Core/Knowledge/Actions integration test.

NIX does not claim “human-level” or scientifically measured “human-like” speech. The project aims for natural, concise, grounded conversational behavior. Voice input/output is on the roadmap; current WebSocket support is text transport.

## Testing

Run the package suites from the repository root:

```bash
(cd nix_knowledge && python -m pytest tests -q)
(cd nix_actions && PYTHONPATH=../nix_knowledge:. python -m pytest tests -q)
(cd nix_core && python -m pytest -q --ignore=benchmarks)
PYTHONPATH=. python -m pytest -q nix_decision/nix_decision/test_engine.py
```

A hermetic Core run that also excludes the live subprocess integration test is:

```bash
(cd nix_core && python -m pytest -q --ignore=benchmarks --ignore=test_e2e_subprocess.py)
```

Test the API Token Guard and Open WebUI conversation-history isolation with:

```bash
(cd nix_core && python -m pytest -q test_conversations.py)
```

The route benchmark's unit tests can run without model weights:

```bash
PYTHONPATH=nix_core python -m pytest -q nix_core/benchmarks/test_speed_benchmark.py
```

Use temporary SQLite databases for test runs that create, update, cancel, or delete records. Never treat passing unit tests as proof of safe production deployment.

## Project structure

```text
README.md                       Project overview and developer guide
ROADMAP.md                      Planned Nix-Skills, app, voice, and hardware features
nix_core/                       Routing, response composition, console, API, gateway
├── dashboard.html              Local responsive dashboard
├── skill_cleanser.py           Qwen3-0.6B skill prompt cleanser
├── skill_runtime.py            Skill plan validation/execution runtime
└── benchmarks/                 Safe read-only router/classification benchmark harness
nix_knowledge/
├── nix_knowledge/              Durable memory and temporal interpretation
├── scripts/                    Knowledge API and local utilities
├── tests/                      Knowledge tests
└── models/                      Local weights, adapters, and datasets (Git-ignored)
nix_actions/                    Deterministic actions and scheduling
nix_decision/                   Explicit trigger/state-gate foundation (no rules enabled)
testing-echo-connect/           Optional isolated integration prototype
Nix-skills-repo/                Skill packaging guide and example skill
data/                           Local SQLite/runtime state (Git-ignored)
```

## Documentation

- [**Roadmap**](ROADMAP.md) — planned features, status, dependencies, and privacy gates.
- [`nix_core/README.md`](nix_core/README.md) — routing, console, APIs, WebSocket text gateway, device skill pipeline, and model boundaries.
- [`nix_core/MOBILE_APP_API.md`](nix_core/MOBILE_APP_API.md) — NIX Home mobile API blueprint; clearly separates the copy prototype from proposed routes.
- [`nix_knowledge/README.md`](nix_knowledge/README.md) — memory, temporal processing, API, and tests.
- [`nix_actions/README.md`](nix_actions/README.md) — action lifecycle, scheduler, sessions, and API.
- [`nix_decision/README.md`](nix_decision/README.md) — explicit trigger contract and safety gates.
- [`nix_core/benchmarks/README.md`](nix_core/benchmarks/README.md) — benchmark harness usage and limitations.
- [`Nix-skills-repo/README.md`](Nix-skills-repo/README.md) — how to package and publish a NIX skill.
- [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) — detailed architecture, operational boundaries, and archived research.
- [`CASPER_V6_RESEARCH.md`](CASPER_V6_RESEARCH.md) — archived Casper V6 experiments and performance/evaluation caveats.
- [`LUNA_PRO_TRAINING_PLAN.md`](LUNA_PRO_TRAINING_PLAN.md) — archived Luna Pro research record (retired).
- [`model_cards/luna-v6/`](model_cards/luna-v6/) and [`model_cards/casper-v5/`](model_cards/casper-v5/) — draft Hugging Face model cards.
- [`NIX-Modeldev/index.html`](NIX-Modeldev/index.html) — model lineage, dataset provenance, and research archive.

## License

NIX PUCA is open-source software licensed under the **Apache License 2.0**.

See [`LICENSE`](LICENSE) for the complete license text.

Copyright © 2026 Sai Neela.

