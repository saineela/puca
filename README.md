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
  <a href="https://github.com/saineela/puca/stargazers">☆ Star NIX PUCA</a> ·
  <a href="nix_core/README.md">Core docs</a> ·
  <a href="nix_knowledge/README.md">Knowledge docs</a> ·
  <a href="nix_actions/README.md">Actions docs</a> ·
  <a href="nix_decision/README.md">Decision docs</a>
</p>

NIX PUCA is a modular software foundation for a **Personal User Companion Agent (PUCA)**: a self-hosted personal assistant built around explicit boundaries between conversation, durable memory, temporal interpretation, and scheduled actions. It is a Python project with local SQLite persistence and optional local or separately operated model backends.

The design is intentionally more structured than a single prompt wrapped around an LLM. NIX Core routes and composes requests; NIX Knowledge stores and retrieves personal records; NIX Actions manages scheduled work. A configured conversational model can produce natural-language responses, while application code validates and manages persistence and scheduling. Deployment security and feature-specific permissions still require configuration, testing, and review.

**Project identity:** The source handoff identifies Casper as the user-facing NIX PUCA identity, created and built by Sai Neela in NIX. This product identity is separate from the local Transformers selector, which currently defaults to Luna V6. These are source-level settings; they do not prove weights are present, loaded, or used by any hosted service.

> **Project status:** This repository is a development foundation, not a claim of a certified, production-ready, or hosted AI product. Model quality, device compatibility, latency, VRAM use, privacy behavior, and service exposure depend on local configuration and require deployment-specific validation. **Public-release note:** no repository-level `LICENSE` is present; choose and add one before presenting the code as open source or inviting reuse/contributions.

## What is a PUCA?

**PUCA** means **Personal User Companion Agent**. In NIX it describes a personal-assistant architecture where conversational language generation is only one part of the system. Personal facts, event times, reminders, presence signals, device access, and external integrations are handled by separately defined services and policies—not assumed to be safe or accurate because a model generated text.

This is a project term, not a claim that NIX is the first PUCA, the first personal AI assistant, or a formally recognized product category.

## Why NIX

A personal assistant needs to do more than answer one prompt. It needs to distinguish a question from a request to store something, resolve “tomorrow” in the configured timezone, keep a reminder linked to the event that created it, ask when a person is ambiguous, and avoid claiming that an action succeeded before the scheduler confirms it.

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
- **Core console:** browser dashboard, traces, conversations, event and memory views, model settings, and local administration. The dashboard follows NIX PUCA's official light-only black-and-white theme and logo.
- **OpenAI-compatible API:** `GET /v1/models` and `POST /v1/chat/completions` for compatible clients. API Token Guard rejects known Open WebUI follow-up/title/tag metadata jobs before Core/model invocation so they do not pollute conversations.
- **WebSocket text gateway:** token-authenticated text transport for client/voice-gateway integrations. This is not, by itself, a complete speech-recognition or text-to-speech system.
- **Optional semantic retrieval:** local Sentence Transformers/BGE embeddings when dependencies and model files are supplied.
- **Optional model backends:** local Transformers/PEFT adapter selection, Ollama, and a TabbyAPI HTTP client. NIX does not include or launch those separately operated services automatically.
- **Isolated integration prototype:** `testing-echo-connect/` explores a future Echo/Home Assistant boundary and is not on the primary request path.

### Planned, not yet implemented

The [**roadmap**](ROADMAP.md) details proposed Nix-Skills, Android access, real-time voice, open-source Web Search, ESP32/OPNsense presence, opt-in location/timeline integrations, and Echo Dot research. The planned NIX Home mobile client has an [API and screen-flow blueprint](nix_core/MOBILE_APP_API.md); no mobile app or authenticated mobile API suite is implemented. These are future plans, not currently available features. In particular, the existing Skills marketplace saves bounded static files and does **not** execute skills; the catalog's Web Search entry is not a working search connector.

## Architecture

```text
                           NIX PUCA
             Dashboard · OpenAI-compatible clients · WS text clients
                                  |
                                  v
                         +----------------+
                         |    NIX Core    |<----> Optional configured
                         | route/compose |       conversation model
                         +-------+--------+       (Transformers/PEFT,
                                 |                 Ollama, or TabbyAPI)
                                 v
                         +----------------+
                         | NIX Knowledge  |<----> NIX Actions
                         | facts, people, | event  reminders, schedules,
                         | time, retrieval| bridge lifecycle and sessions
                         +----------------+

       NIX Decision: separate explicit-trigger foundation;
       no proactive rules are registered.
```

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
4. Core composes a response using the structured result. It should not state that the reminder succeeded if Actions did not confirm it.

The exact outcome depends on the current route, service health, database, timezone, and model configuration. Use isolated test databases for experiments.

## High-level concepts, in plain language

| Term | Simple definition |
| --- | --- |
| **Local-first** | Core data and services are designed to run on infrastructure you control; optional backends and integrations may still use network services. |
| **NIX Core** | The request coordinator: chooses a route, calls the needed services, and prepares the final reply. |
| **NIX Knowledge** | The structured memory service: stores and retrieves authorized facts, people, and calendar records. |
| **NIX Actions** | The scheduler: tracks supported reminders and other captured work through their lifecycle. |
| **Nix-Skills** | The planned format/runtime for optional capabilities and integrations; it is not executable in the current repository. |
| **LLM** | A large language model that generates or interprets text. NIX uses it as a replaceable component rather than as the database or scheduler. |
| **Router** | A decision layer that selects a request path, such as chat or Knowledge. It does not generate the final answer; routing latency is not full-response latency. |
| **p50 / p95 latency** | p50 is the median time across samples; p95 is the time at or below which 95% of measured samples completed. |
| **Requests per second (RPS)** | A throughput calculation for the timed benchmark operation; it is not a count of full model-generated conversations per second. |
| **Durable memory / Knowledge** | Structured records that can be read later, rather than relying only on what is in the current chat prompt. |
| **Temporal grounding** | Converting phrases such as “tomorrow at 5” into a timezone-aware date and time before saving or reporting them. |
| **Action scheduler** | A deterministic service that tracks when captured reminders or other supported operations are due. |
| **Adapter / LoRA / QLoRA** | An adapter adds learned parameters to a base model; LoRA is a low-rank adapter method, and QLoRA trains LoRA adapters while loading a quantized base model. Neither replaces the surrounding application rules or data store. |
| **VRAM** | GPU memory. Requirements depend on model weights, quantization, runtime, context size, and concurrent workloads. |
| **Presence detection** | A signal that may estimate whether a device/person is at home or in a room; it is not automatically identity proof or precise location. |
| **Skill** | A planned, permissioned capability/connector. In the current repository, marketplace packages are static files and are not executable. |
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

The command above installs the sibling packages and basic Core/HTTP dependencies. It does not install `pytest`, Sentence Transformers, a CUDA-enabled PyTorch build, Transformers/PEFT/bitsandbytes, or a model. Add only the optional stack needed for your chosen use case. The repository does not currently lock every optional dependency to a tested version matrix.

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
| `NIX_CASPER_BACKEND` | `transformers`, `ollama`, or `tabby` transport selection; verify actual runtime configuration. |
| `NIX_REQUEST_LOG` / `NIX_REQUEST_LOG_DIR` | Core request logging (enabled by default) and its output directory. Logs do not rotate automatically. |

Important operational notes:

- The console contains memory, conversation, deletion, and full-reset controls. Do not expose it to untrusted networks without adding and validating an appropriate authentication/proxy boundary.
- Use API tokens for network clients, protect backend credentials, and scope service binds to trusted interfaces.
- SQLite databases, WAL files, backups, request JSONL files, model files, adapters, and datasets are private/runtime/provenance data, not source files. They are ignored by Git; `.gitignore` is not encryption or access control.
- Core request logs can contain user text and response content. Disable logging with `NIX_REQUEST_LOG=0` if that fits your requirements, or define access controls, retention, and backup policy before enabling it in a shared environment.
- Do not connect broad tests to a real personal database. Use disposable test databases and verify destructive test behavior.

## Models and inference

### Model identities and source-level selection

- The **Transformers selector defaults to the official Luna V6 ID** `luna-v6-contextual-v1` in source.
- **Casper V5** (`casper-puca-qlora-v5`) and **Casper V6** (`casper-puca-qlora-v6`, local adapter directory `casper-puca-qlora-v6-final`) are separate selectable Casper choices. Casper V6 is documented as beta and is not the selector default.
- **Luna Pro v1 is retired** from this repository's runtime. Do not reuse or adapt its training approach or data for any model. Luna development is V6 runtime/integration only; no Luna training or new offline evaluation is authorized.
- When `NIX_CASPER_BACKEND` selects Ollama or TabbyAPI, that non-Transformers backend bypasses the Luna/Casper local adapter selector.

These are **source-code facts**, not statements that any listed weights exist on your machine, that a model was loaded during this session, or that a hosted process serves one of those models. Check the running process and its health/configuration. Local models are separately downloaded and excluded from Git.

### GPU memory and the under-3-GB claim

NIX contains deterministic CPU-routed paths, optional CPU-first routing, optional semantic embeddings, and GPU-backed conversation model implementations. These are different workloads. A fast route decision does not mean the full conversation or every model pipeline uses less than 3 GB of VRAM.

This repository does **not** currently establish a complete NIX conversation run under 3 GB of VRAM. Historical notes report a Luna V6 training allocation of about **3.074 GiB** and a Casper development GPU run around **3.8–4.3 GiB**; those are archived environment-specific figures, not a current hardware guarantee. A fixed per-process VRAM fraction is a cap/control, not a measurement of end-to-end peak memory. The local weights, quantization, GPU, runtime, context length, warm/cold state, and other GPU processes all affect actual use.

For a defensible model-memory claim, publish a reproducible run with: exact GPU/driver, backend and package versions, exact base and adapter revisions, quantization, prompt/context/token settings, cold and warm runs, measured peak allocated **and reserved** VRAM, and repeated sample count. Until such a full-stack benchmark is recorded and independently repeatable, NIX does not advertise itself as “the world’s first PUCA under 3 GB VRAM.”

### Optional TabbyAPI / ExLlamaV2

NIX Core includes a TabbyAPI-compatible HTTP **client**, not an ExLlamaV2 server. TabbyAPI must be installed and run separately with a verified compatible model artifact. The current PEFT/QLoRA adapters are not directly verified as ExLlama loadable. Configure this backend only after validating the actual model format and compatibility; see [`.env.example`](.env.example).

## Benchmarks and model evaluation

### Reproducible router-classifier microbenchmark

The repository includes a safe, read-only microbenchmark for `nix_core/router.py::classify`:

```bash
PYTHONPATH=nix_core python nix_core/benchmarks/speed_benchmark.py \
  --benchmark router --repeats 100 --warmup 10
```

A 100-sample run in the current development workspace reported **p50 0.106 ms**, **p95 0.145 ms**, mean **0.103 ms**, minimum **0.031 ms**, and maximum **0.189 ms** on the harness's built-in prompt set (10 warm-up samples; all 100 measured samples succeeded). The harness-calculated throughput was **9,673 requests/second** for this timed operation. Timing includes the classifier call and JSON serialization of the route and matched rule. These are local software-environment measurements, not universal latency guarantees or end-to-end assistant response/throughput. The benchmark does not measure the separate `CoreRoutingEngine.decide` API, HTTP/database work, model loading, or response generation. Its JSON result does not capture the host hardware, OS, Python version, or package versions; rerun the command in your own environment before making a comparison.

The read-only router/predictor/dashboard-classification benchmark harness does not invoke a complete model response. See [`nix_core/benchmarks/README.md`](nix_core/benchmarks/README.md) for targets, limitations, and usage. Do not present its sub-millisecond route timing as total conversation latency.

### Archived model-quality and latency evidence

Small project-local evaluation suites are engineering checks, not human-subject studies or standardized language-quality scores. Historical Casper V6 notes report **7/8** in one policy-conditioned single-turn set and **1/5** in the latest documented direct multi-turn check; the V6 candidate is not promoted as the default. A historical isolated generation benchmark reported p50 **1.64 s** and max **4.28 s** over five short prompts after an approximately **8.35 s** load. The environment and protocol are in [`CASPER_V6_RESEARCH.md`](CASPER_V6_RESEARCH.md), and are not a claim of current serving performance.

Historical Luna contextual scores were mixed and were recorded with synthetic injected context as well as isolated mode. Synthetic evaluator context is not a full Core/Knowledge/Actions integration test. Luna model development is restricted to V6 runtime/integration; no Luna training or new offline evaluation is authorized.

NIX does not claim “human-level” or scientifically measured “human-like” speech. The project aims for natural, concise, grounded conversational behavior. Voice input/output is on the roadmap; current WebSocket support is text transport.

## Testing

Run the package suites from the repository root:

```bash
(cd nix_knowledge && python -m pytest tests -q)
(cd nix_actions && PYTHONPATH=../nix_knowledge:. python -m pytest tests -q)
(cd nix_core && python -m pytest -q)
PYTHONPATH=. python -m pytest -q nix_decision/nix_decision/test_engine.py
```

Core's full suite may require configured services or model backends. A hermetic Core run that excludes the live subprocess integration test is:

```bash
(cd nix_core && python -m pytest -q --ignore=test_e2e_subprocess.py)
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
└── benchmarks/                 Safe read-only router/classification benchmark harness
nix_knowledge/
├── nix_knowledge/              Durable memory and temporal interpretation
├── scripts/                    Knowledge API and local utilities
├── tests/                      Knowledge tests
└── models/                      Local weights, adapters, and datasets (Git-ignored)
nix_actions/                    Deterministic actions and scheduling
nix_decision/                   Explicit trigger/state-gate foundation (no rules enabled)
testing-echo-connect/           Optional isolated integration prototype
data/                           Local SQLite/runtime state (Git-ignored)
```

## Documentation

- [**Roadmap**](ROADMAP.md) — planned features, status, dependencies, and privacy gates.
- [`nix_core/README.md`](nix_core/README.md) — routing, console, APIs, WebSocket text gateway, and model boundaries.
- [`nix_core/MOBILE_APP_API.md`](nix_core/MOBILE_APP_API.md) — NIX Home mobile API blueprint; clearly separates the copy prototype from proposed routes.
- [`nix_knowledge/README.md`](nix_knowledge/README.md) — memory, temporal processing, API, and tests.
- [`nix_actions/README.md`](nix_actions/README.md) — action lifecycle, scheduler, sessions, and API.
- [`nix_decision/README.md`](nix_decision/README.md) — explicit trigger contract and safety gates.
- [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) — detailed architecture, operational boundaries, and historical research.
- [`CASPER_V6_RESEARCH.md`](CASPER_V6_RESEARCH.md) — archived Casper V6 experiments and performance/evaluation caveats.
- [`NIX-Modeldev/index.html`](NIX-Modeldev/index.html) — model lineage, dataset provenance, and research archive.

## GitHub discovery and project metadata

GitHub repository topics classify a repository by its actual purpose, subject area, and language. Before a public launch, consider adding this accurate set through the repository's **About → Topics** control:

`personal-assistant` · `personal-ai-assistant` · `personal-user-companion-agent` · `puca` · `ai-assistant` · `local-ai` · `local-llm` · `self-hosted` · `python` · `sqlite` · `conversational-ai` · `knowledge-management`

These are suggested GitHub repository topics, not a claim that GitHub endorses or ranks NIX. The project-specific `PUCA` term is also useful in the project description and documentation, while generic topic labels help describe the current code to people browsing established topic pages. Because voice and smart-home integrations are roadmap items rather than released features, do not add `voice-assistant` or `smart-home` topics until those capabilities exist. The suggested set also omits `privacy-focused`: a local-first design is not itself a security guarantee, and deployment settings need a security review before making that marketing claim. Remove any topic that no longer accurately describes a released capability. The [GitHub topics documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics) notes that repositories can use up to 20 lowercase topic names (50 characters or fewer).

Suggested concise repository description:

> Local-first Personal User Companion Agent foundation in Python: modular conversation routing, durable personal memory, temporal grounding, and deterministic reminders.

Target repository URL: [`github.com/saineela/puca`](https://github.com/saineela/puca). [Star NIX PUCA](https://github.com/saineela/puca/stargazers) if you find the project useful. Confirm the repository exists, is public, and this link resolves before announcing the launch. Add repository topics and description in GitHub settings; README text alone does not set GitHub's metadata or guarantee search ranking.

## License and contributions

No repository-level `LICENSE` file is currently present. Until a license is chosen and added, do not assume this repository grants permission to reuse or redistribute its code. Model and dataset licenses are separate and must be reviewed independently. Contributions should include focused tests and validation notes and must not add secrets, personal data, model weights, or generated datasets to Git.
