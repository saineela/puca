# nix_core

`nix_core` is the conversational orchestration layer for NIX/Casper. It receives user text, decides whether the request belongs to personal Knowledge or ordinary conversation, calls the required services, and produces the final user-facing response.

## Responsibilities

- Deterministic request routing and a zero-VRAM hybrid route boundary.
- Conversation/session context and follow-up continuation.
- Casper response generation through the configured local backend.
- Knowledge composition: raw user request + grounded Knowledge output + current temporal context are sent to Core for final formatting.
- Websocket gateway for voice/client integrations.
- Browser dashboard for live testing, trace inspection, sessions, schedule, and Knowledge views.
- Request logging and regression analysis.

Core does **not** own durable personal memory. It asks `nix_knowledge` for that information. It does not decide when reminders fire; `nix_actions` owns scheduling.

## Request flow

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
   `-- conversation/world request --> Casper backend
                                      |
                                      v
                              final response to user
```

When Knowledge is used, Casper remains the final response writer. Internal route labels such as `CHAT rule: ...` are diagnostics only and must never appear in the user-facing reply.

The local Qwen 2.5 0.5B model is used as **Nix_predictor**, a constrained Knowledge function selector. Core's deterministic router and Knowledge symbolic rules remain the fast safety layer, while Nix_predictor handles indirect and multi-intent requests. Disable it only for diagnostics with `NIX_CORE_USE_KNOWLEDGE_MODEL_GATE=0`.

## Casper runtime

The verified development runtime is the local Casper v5 QLoRA adapter loaded through Transformers/PEFT on CUDA. The base model and adapter are stored locally under `nix_knowledge/models/` and are intentionally ignored by Git. The optional TabbyAPI/ExLlama path is opt-in and requires a compatible EXL2/EXL3 artifact; it is not the default Casper backend.

Relevant configuration includes:

```bash
NIX_CASPER_BACKEND=transformers
NIX_CORE_WARMUP_MODELS=1
NIX_CASPER_MAX_VRAM_FRACTION=0.68
NIX_KNOWLEDGE_MODEL_GATE=1
NIX_TZ=America/Chicago
```

See [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md) for exact model paths, adapter lineage, VRAM controls, and runtime caveats.

## Services

| Service | Default role |
|---|---|
| Knowledge API | `nix_knowledge` HTTP service, commonly port `8100` |
| Actions API | `nix_actions` HTTP service, commonly port `8200` |
| Console | Browser test dashboard, commonly port `35567` |
| Websocket gateway | Voice/client interface, commonly port `9000` |

Start the console:

```bash
python nix_core/console.py
```

Useful console endpoints:

- `GET /api/health` — service, model, GPU, and timezone status
- `GET /api/feed` — trace, turns, actions, and Knowledge summaries
- `GET /api/sessions` — grouped live chat sessions
- `GET /api/schedule` — events and linked actions
- `POST /api/send` — run a request through the full Brain pipeline

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
- `router.py` — deterministic route classification.
- `console.py` — browser dashboard and API.
- `ws_server.py` — websocket gateway.
- `followup.py` — short-lived continuation behavior.
- `request_log.py` — JSONL request logging.
- `analyze_logs.py` — log problem summaries and exports.
- `config.py` — environment-backed service/runtime configuration.
