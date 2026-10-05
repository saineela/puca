# nix_core

`nix_core` is the conversational orchestration layer for NIX/Casper. It receives user text, decides whether the request belongs to personal Knowledge or ordinary conversation, calls the required services, and produces the final user-facing response.

## Responsibilities

- Independent `routing_engine.py` decision boundary with sub-millisecond local fast paths and explicit confidence/model-needed metadata.
- Conversation/session context and follow-up continuation.
- Response generation through the configured official local conversation model.
- Knowledge composition: raw user request + grounded Knowledge output + current temporal context are sent to Core for final formatting.
- Websocket gateway for voice/client integrations.
- Browser dashboard for live testing, trace inspection, sessions, schedule, and Knowledge views.
- Extended console launcher and the currently implemented Mobile App home-copy prototype; see [`MOBILE_APP_API.md`](MOBILE_APP_API.md) for the full proposed NIX Home API and client flows.
- Request logging and regression analysis.

Core does **not** own durable personal memory. It asks `nix_knowledge` for that information. It does not decide when reminders fire; `nix_actions` owns scheduling.

## Request flow

The independent Core Routing Engine runs before any model or service call. It
classifies response acts, personal/action intent, world chat, emotional turns,
safety boundaries, and ambiguous input. It never invokes Qwen or Casper. Only
an explicit `requires_model=true` ambiguity may reach the configured fallback;
ordinary social, emotional, reminder, memory, and world-chat requests do not.

```text
CoreRoutingEngine.decide(text)
  -> route + confidence + reason + requires_model + routing_latency_ms
```

The engine is not the response generator: a fast route can still be followed
by official conversation-model generation, and that generation is reported
separately by the Dashboard.

```text
user text
   |
   v
router.py
   |-- personal/temporal request --> nix_knowledge /process
   |                                      |
   |                                      v
   |                              grounded result + absolute times
   |
   `-- conversation/world request --> configured assistant backend
                                      |
                                      v
                              final response to user
```

Core remains the final response boundary after Knowledge. It may use the active official conversation model to format a grounded result; internal route labels such as `CHAT rule: ...` are diagnostics only and must never appear in the user-facing reply.

The former Qwen 2.5 0.5B **Nix_predictor** is not required for deterministic high-confidence routes. The optional `routing_predictor.py` is a tiny CPU-first NumPy classifier for broad chat-vs-Knowledge ambiguity; it can abstain, never executes tools, and does not resolve dates or people. Keep `NIX_CORE_USE_KNOWLEDGE_MODEL_GATE=0`; do not enable a neural model gate for the V6 runtime/integration path. High-confidence social checks and harmless food/drink preference questions bypass neural routing.

## Official conversation-model runtime

The official conversation-model path is non-thinking. Core passes
`think=False`; compatible Ollama transport also sends `"think": false`, and
Tabby does not receive a thinking field. The adaptive Qwen thinking gate is
not consulted for model responses. Complex routing/tool work stays in
structured Core/Knowledge code rather than a hidden reasoning trace.

### Luna V6 runtime and archived research

Luna V6 is the only Luna model in the runtime registry. The Transformers
selector defaults to `luna-v6-contextual-v1`; Casper fallbacks require explicit
dashboard selection. A non-Transformers backend bypasses that selector. This local
configuration is not evidence of what a hosted service is serving.

**Luna Pro v1 is retired from this repository's runtime.** Its adapter is not
registered for selection/loading, and the former Luna model endpoint handlers
return HTTP 410. Historical Pro research, builders, manifests, source
data, and artifacts are kept as an archive only: the Pro workflows and their
training approach or data are not used by any model. Luna work is limited
to the V6 runtime/integration path, and training/offline-evaluation tooling
stays disabled. The historical notes below describe past research.
Archived builder/evaluator CLIs are disabled; the Pro
trainer's CPU-only, read-only `check-data` preflight is the only retained
functional command. Its training/GPU commands fail closed.

Historical V6 research records the public Llama 3.2 3B Instruct base and its
official chat template. The repository retains earlier base-model, dataset,
builder, and evaluation sources for provenance; builder CLI entry points are
disabled. The preserved clean-mix notes
record exclusions and filtering only.

Historical Luna prompts and corpora recorded a role boundary between Nix, its
Casper PUCA, and the independent Luna model. Casper identity and
personal-system facts remain Core-owned concerns for authoritative responses.

Historical trainer implementations and their methods (including rsLoRA,
DoRA, LoRA A/B learning-rate groups, exact-prefix masking, and token-weighted
gradient accumulation) are retained for auditability only.

Preserved research references include QLoRA, LoRA, rsLoRA, LoRA+, DoRA,
LoftQ, NEFTune, LIMA, SFTMix, and GRAPE.

Earlier research notes transcribe 2/9 single-turn and 0/5 multi-turn checks
for an identity candidate, and 1/9 + 1/5 for a balanced rsLoRA checkpoint.
These are historical reports; the creator case tested a Casper fact, not Luna's identity. The
corrected shared-harness Instruct baseline was reported at 5/9 single-turn and
1/5 multi-turn without injected context, and 7/9 + 2/5 in synthetic context.
The preserved `luna-v6-contextual-v1` (60 steps from the Instruct
adapter) was reported at 4/9 + 1/5 without context and 6/9 + 2/5 with synthetic
context, below that reported baseline. A paired fictional
diagnostic (9 cases / 15 turns) reported that context helped some
state/ambiguity retrieval, but the candidate invented collaborator details,
changed a four-day recurrence to four hours, and claimed a failed reminder
succeeded without supplied context. The Instruct baseline also reportedly
hallucinated in no-context action/memory conditions. The notes reported zero
exact prompt collisions against the shared suite and scanned V4/V2/V3 user
turns, but this was not statistically robust evidence or a real
Core/Knowledge/Actions integration test. The cited report is
`models/nixlm/luna-generalization-probes-paired-20260923.json`; current artifact
presence is not verified. The historical V3 corpus is described in the notes
as a corrected **untrained** snapshot, not the data reported for the existing
adapter.

Historical Luna research was separate from Casper and did not execute the
Core → Knowledge → Actions path or write durable memory. The V6 model identity
is registered with the official assistant selector and regular Core path; this
is source configuration, not proof of a loaded or hosted model. The former
`/api/luna/model` and `/api/luna/chat` preview endpoints are retired and their
handlers return HTTP 410; Pro IDs are also rejected by official model/chat
routes. The historical Pro pilot report is cited at
`nix_knowledge/models/nixlm/luna-pro-v1-topical-v8-384-retry2/benchmark_report.md`;
that Git-ignored local artifact's current presence is not verified. The report is
historical evidence only.

**Historical Pro v1 pilot record (pre-retirement):** the
retirement notes described a v8 run using one epoch and a 384-token limit, with
718/1,219 train and 87/165 dev conversations and 90 optimizer steps. Its small
fictional panel was not a quality certification or real stack test. The notes
record quality defects and unresolved provenance/rights. Repository source
keeps Pro retired and non-selectable, and the run has not been reproduced; the
Git-ignored artifacts' current presence is not verified.

Synthetic context injects hand-authored memory/actions messages in an
evaluator; it is not an end-to-end stack test. See the exact historical scores
and lineage in [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md); current local adapter
presence is not verified.

The code supports a Transformers/PEFT adapter when selected and local artifacts
are present. See
[`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md) for code-level backend caveats.

Relevant configuration includes:

```bash
NIX_CASPER_BACKEND=transformers
NIX_CORE_WARMUP_MODELS=1
NIX_CASPER_MAX_VRAM_FRACTION=0.68
CASPER_FAST_MAX_NEW_TOKENS=64
NIX_CORE_USE_KNOWLEDGE_MODEL_GATE=0
NIX_CORE_USE_CUSTOM_ROUTING_PREDICTOR=1
NIX_OPENAI_API_KEY=change-me-for-external-clients
NIX_OPENAI_API_ALLOW_ORIGIN=*
NIX_TZ=America/Chicago
```

See [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md) for exact model paths, adapter lineage, VRAM controls, and runtime caveats.

## Services

| Service | Default role |
|---|---|
| Knowledge API | `nix_knowledge` HTTP service, commonly port `8100` |
| Actions API | `nix_actions` HTTP service, commonly port `8200` |
| Console | Browser test dashboard; the **only** port it listens on (`49117`, fixed) |
| Websocket gateway | Voice/client interface, commonly port `9000` |

Start the console:

```bash
python nix_core/console_extend.py
```

The console binds `0.0.0.0:49117` by default (set `NIX_CONSOLE_HOST` to restrict it) and serves the UI, every `/api/*` route, and the `/v1` OpenAI surface on that one fixed port. There is no port override; the port remains stable across restarts and browser reloads. This enables LAN access subject to firewall/network policy but does not provide internet port forwarding.

Knowledge and Actions run in-process through an internal HTTP bridge, so starting the console opens exactly one listening socket — a browser needs only the console URL. The optional Testing-page runner is the sole exception: while a corpus batch runs it starts two temporary loopback-only subprocess APIs against copied databases.

Useful console endpoints:

- `GET /api/mobile/v1` and `GET /api/mobile/v1/home-copy` — Mobile App category index and randomized home-copy prototype, available only when started with `console_extend.py`; the full mobile API is not implemented. See [`MOBILE_APP_API.md`](MOBILE_APP_API.md).

- `GET /api/health` — service, model, GPU, and timezone status
- `GET /api/feed` — trace, turns, actions, and Knowledge summaries
- `GET /api/sessions` — grouped live chat sessions
- `GET /api/schedule` — events and linked actions
- `POST /api/send` — run a request through the full Brain pipeline
- `POST /api/kb/delete` — delete one Knowledge record and its derived indexes
- `POST /api/reset` with `{"confirm":"RESET_ALL"}` — dashboard complete reset for Knowledge, indexes, actions, and sessions
- `GET /api/memory` — current bounded memory block supplied to the active assistant
- `GET /api/model` — active official conversation model and available local choices
- `POST /api/model` with `{"model":"casper-puca-qlora-v6"}` — switch the local adapter exclusively; Luna IDs are rejected here
- `GET /api/luna/model`, `POST /api/luna/model`, and `POST /api/luna/chat` — retired direct-research endpoints; their handlers return HTTP 410 `model_retired`
- Pro model IDs submitted to `/api/model`, `/api/send`, or `/v1/chat/completions` — rejected by the handlers with HTTP 410 `model_retired`
- `POST /v1/chat/completions` — runs the server's configured official Core model when available; API Token Guard rejects OpenWebUI follow-up/title/tag metadata tasks with `api_token_guard_rejected` before Core/model execution

The dashboard source is [`dashboard.html`](dashboard.html). It provides
responsive Home, Models, Conversations, Memories, People, Events, Skills, API,
Documentation, Release notes, and Settings views. The Skills marketplace can
preview public GitHub skill manifests and install declared static files only;
community code is never executed. When the console is running, chat requests
use the `/api/send` pipeline, report server-side timings, query
sessions/memory/events, and expose confirmation-protected individual delete
and complete-reset controls.

Core routing uses deterministic fast paths for ordinary conversation,
explicit reminders/cancellations, and common indirect storage language. Keep
the Qwen selector disabled for the Luna V6 path.

The console also provides an OpenAI-compatible external API:

```text
GET  /v1/models
POST /v1/chat/completions
```

Configure `NIX_OPENAI_API_KEY` to require Bearer authentication. Open WebUI
should use the console origin plus `/v1` as its OpenAI API base and select the
model reported by that server. API Token Guard rejects structured OpenWebUI
follow-up/title/tag jobs before they reach Core or a model, even when the job
contains no chat-history block or the weekly conversation ID. The repository's
Transformers selector defaults
to Luna V6; Casper V5/V6 are explicit selectable fallbacks. A non-Transformers backend bypasses that selector.
Neither this code default nor local status is proof of a hosted model or
deployment state. Luna Pro v1 is retired from this repository's runtime
registry, and former direct endpoint handlers return HTTP 410; its research is
archived only. Luna V6 follows Core → Knowledge → Actions and shares the
process-wide model slot with Casper. Both non-streaming and OpenAI SSE streaming run the full Core
pipeline; native local token callbacks are not exposed. The API response
includes the standard OpenAI `chat.completion` shape plus a small `nix`
diagnostic object.

Start the websocket gateway:

```bash
python nix_core/ws_server.py
```

The client authenticates with the configured token, then sends text prompts:

```json
{"token":"$NIX_AUTH_TOKEN"}
{"text_prompt":"what do I have tomorrow?","location":"home"}
```

The server returns status and reply messages containing the response, route, conversation ID, and follow-up state.

## Important behavior

### Temporal grounding

Relative expressions are resolved before events are stored or rendered:

- `tmr`, `tmrw`, `tomorrow`
- `in 2 more days`
- `the day after tomorrow`
- `next Monday morning`
- `this Friday`
- timed `this weekend`

Core should prefer absolute local dates and times in final schedule responses.

### Follow-up clarification

If Knowledge/Core cannot identify which person the user means, Core stores the pending clarification by conversation ID. The next answer is consumed as a continuation instead of being treated as a new unrelated request.

### Deterministic identity

Casper creator/identity questions use a deterministic guard so model memory cannot invent an answer. The canonical creator response is:

> Created and Built by Sai Neela, and living in NIX's PUCA system.

## Tests

From the repository root:

```bash
PYTHONPATH=. python -m pytest -q nix_core/test_conversation.py
PYTHONPATH=. python -m pytest -q nix_core/test_temporal_regressions.py
(cd nix_core && python -m pytest -q --ignore=test_e2e_subprocess.py)
```

The websocket subprocess suite may require the local service/model environment. Use temporary databases for full integration tests and do not test destructive operations against live personal data.

## Key files

- `brain.py` — Core request pipeline and response composition.
- `routing_engine.py` — independent typed Core route boundary and confidence metadata.
- `router.py` — compatibility symbolic corpus/rule classifier used as a bounded fallback inside the routing engine.
- `console.py` — unchanged base browser dashboard and API.
- `console_extend.py` — opt-in launcher and additive HTTP route dispatch.
- `mobile_app_api.py` — implemented home-copy catalog prototype for the future mobile category.
- `MOBILE_APP_API.md` — mobile screen flows, proposed APIs, setup, and security contract.
- `ws_server.py` — websocket gateway.
- `followup.py` — short-lived continuation behavior.
- `request_log.py` — JSONL request logging.
- `analyze_logs.py` — log problem summaries and exports.
- `config.py` — environment-backed service/runtime configuration.
- `skill_cleanser.py` — Qwen3-0.6B skill prompt cleanser (fail-closed rewrite of vague device requests).
- `skill_profiles.py` — per-skill profiles and the Needle3 skill decider.
- `skill_runtime.py` — explicitly trusted skill worker runtime.

### Device skill pipeline

Targeted device skills (for example the ring light) use a two-stage pipeline
for both single-action and ordered multi-action requests:

1. **Prompt cleansing (optional).** For vague or conversational requests, a
   small local model — Qwen3-0.6B via Ollama (`NIX_SKILL_CLEANSE_MODEL`, default
   `qwen3:0.6b`) — rewrites the user's words into one to eight short,
   device-anchored imperative steps. Core resolves explicit color descriptions
   first and passes the resolved color as a hint. The cleanser fails closed:
   any model error, malformed JSON, or contract violation falls back to the
   user's own words, and fully grounded requests skip this stage.
2. **Planning, validation, and execution.** The Needle skill decider and
   executor propose a `NIX_SKILL_PLAN:` envelope of at most eight calls. Core
   validates every selected skill, tool, and argument schema before dispatch,
   then executes sequentially through the skill worker and stops at the first
   failed or unconfirmed action. The final acknowledgment is composed only
   after the device readback confirms the results.

Model prose is never evidence of execution, and a plan that is missing,
malformed, incomplete, or invalid executes no action. Explicit package trust,
digest/configuration checks, worker isolation boundaries, and result validation
remain in `SkillRuntime`. The console pipeline view shows every stage,
including “Qwen3-0.6B · skill prompt cleansing”.
