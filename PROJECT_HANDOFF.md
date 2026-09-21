# Casper / Nix Project Handoff

This document is the durable context for future agents working in this repository. It records the architecture, model lineage, runtime contracts, data boundaries, testing rules, and operational decisions that previously lived only in conversation history.

> **Product identity:** The repository still uses `nix_*` package and environment names for compatibility. The user-facing PUCA is **Casper**. Casper's creator response is deterministic: **"Created and Built by Sai Neela, and living in NIX's PUCA system."**

## 1. Project purpose

This is a local personal-assistant stack:

- **Casper** is the user-facing Personal User Companion Agent (PUCA).
- **`nix_core`** orchestrates conversations, routes requests, owns final user-facing wording, and maintains conversational/session context.
- **`nix_knowledge`** is the durable personal knowledge base and structured natural-language interface. It stores facts, people, temporal person states, calendar events, and semantic/entity information.
- **`nix_actions`** captures and schedules reminders/alarms/actions produced by Knowledge. It is deterministic and does not interpret natural language.
- **`nix_decision`** is a separate decision-engine prototype and is not on the primary Casper request path.

The central design rule is:

> Neural models propose; deterministic Python validates, resolves, stores, schedules, and enforces policy.

## 2. Repository map

```text
README.md                         Project overview and basic commands
.env.example                      Configuration template; never put secrets here
.gitignore                        Runtime/model/cache exclusion policy
PROJECT_HANDOFF.md                This document

nix_core/
  brain.py                        Main Core pipeline and final response composer
  router.py                       Deterministic chat/knowledge classifier
  ws_server.py                    Authenticated websocket gateway
  console.py                      Local web dashboard and HTTP API
  config.py                       Runtime configuration and backend selection
  casper_model.py                 Local Transformers + PEFT Casper backend
  casper_runtime.py               Backend/artifact health metadata
  runtime_warmup.py               Parallel Casper/Knowledge warm-up
  tabby_client.py                 Optional TabbyAPI/OpenAI-compatible transport
  context.py                      Session-context trimming
  followup.py                     Short follow-up conversation window
  tone_policy.py                  Emotional/conversation-attunement policy
  request_log.py                  Daily JSONL request logging
  analyze_logs.py                 Log analysis utility
  Modelfile.casper-puca-v5        Legacy Ollama merged-model recipe
  test_*.py                       Core, backend, identity, temporal, and session tests

nix_knowledge/
  nix_knowledge/needle.py         Knowledge selector + structured tool execution
  nix_knowledge/temporal.py       Authoritative timezone/date resolver
  nix_knowledge/temporal_hybrid.py Neuro-symbolic temporal repair/validation
  nix_knowledge/context.py        Temporal windows, occurrences, and clock context
  nix_knowledge/rules.py          Deterministic Knowledge function routing
  nix_knowledge/states.py          Person current-state parsing/supersession
  nix_knowledge/keys.py            Durable profile/fact extraction
  nix_knowledge/engine.py          SQLite knowledge persistence
  nix_knowledge/functions/calendar.py Calendar storage/search/update/cancel
  nix_knowledge/semantic/         Embeddings, entity registry, classifier/gate
  nix_knowledge/memory_block.py   Core/Casper memory grounding block
  scripts/knowledge_api.py        Knowledge HTTP API
  scripts/nixlm/                  Dataset, evaluation, QLoRA, and merge tools
  tests/                           Knowledge and temporal tests
  models/                          Local model/adapters/datasets; ignored by Git

nix_actions/
  nix_actions/engine.py           Deterministic action capture/scheduler
  nix_actions/sessions.py         Day/week session store
  scripts/actions_api.py          Actions HTTP API
  tests/                           Scheduler/propagation/session tests

testing-echo-connect/             Optional Echo/Home Assistant prototype
```

## 3. Request flow

### 3.1 Chat and personal-memory flow

```text
client/dashboard/voice
  -> nix_core websocket or /api/send
  -> Brain.handle()
  -> deterministic Core router
  -> Knowledge route OR chat route
```

For a Knowledge request:

```text
Brain
  -> KnowledgeClient.process()
  -> nix_knowledge /process
  -> deterministic route or Qwen2.5-0.5B selector
  -> Python validation/tool execution
  -> structured Knowledge result
  -> Knowledge memory_block() after the operation
  -> Core final composer (Casper, think=False)
  -> user-facing reply
```

Core must remain the final speaker after Knowledge. The final composition input contains:

1. Raw user request.
2. Knowledge-scoped request.
3. Recent bounded session context.
4. Post-operation Knowledge memory block.
5. Authoritative structured Knowledge JSON.
6. Explicit temporal grounding block.
7. Deterministic safe baseline rendering.

If Casper fails or produces unsafe/incomplete wording, Core returns the deterministic baseline. Knowledge does not directly speak to the user.

For an ordinary chat request:

```text
Brain
  -> Casper local Transformers backend (default)
  -> bounded session history + Knowledge memory block
  -> final reply
```

The chat model must not invent personal memory. The memory block and structured Knowledge results are authoritative.

### 3.2 Actions flow

```text
Knowledge creates/updates/cancels calendar event
  -> nix_knowledge.bridge
  -> nix_actions capture/schedule/reschedule/cancel
  -> action linked with source_record_id
```

Do not modify the Actions database directly when testing normal behavior. Use the Knowledge/API path or an isolated temporary database.

## 4. Casper model lineage and files

### 4.1 Base model

The verified local base is the Qwen3.5 4B HF checkpoint:

```text
nix_knowledge/models/qwen3.5-4b-hf/
```

Important metadata:

- Architecture in `config.json`: `Qwen3_5ForConditionalGeneration`.
- The local HF model is a multimodal Qwen3.5 architecture, not a simple legacy causal model.
- The model index reports approximately 9.32 GB of unquantized safetensor data.
- The directory is ignored by Git; it must be provisioned locally.

### 4.2 Casper QLoRA adapter

The active adapter is:

```text
nix_knowledge/models/nixlm/casper-puca-qlora-v5/
```

The adapter metadata is in `adapter_config.json`:

- PEFT type: LoRA.
- Base: local `qwen3.5-4b-hf`.
- Rank `r=8`.
- Alpha `16`.
- Dropout `0.05`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Task type: causal language modeling.
- `use_qalora` is currently false in the stored adapter config; the training pilot uses 4-bit NF4 loading and PEFT LoRA.

The adapter is loaded by:

```text
nix_core/casper_model.py
```

The path can be overridden with:

```text
CASPER_BASE_MODEL_PATH
CASPER_ADAPTER_PATH
CASPER_VRAM_FRACTION
CASPER_MAX_NEW_TOKENS
CASPER_MAX_CONCURRENT_REQUESTS
```

The default runtime uses Transformers + PEFT on CUDA, with `CASPER_VRAM_FRACTION=0.68` and one concurrent decode. The observed deployment uses an RTX 4060 and approximately 3.8–4.3 GiB total observed GPU use depending on warm-up/cache state.

### 4.3 QLoRA/data scripts

Important scripts:

```text
nix_knowledge/scripts/nixlm/build_humanlike_dataset.py
nix_knowledge/scripts/nixlm/build_public_human_dialogue.py
nix_knowledge/scripts/nixlm/train_qwen35_qlora_pilot.py
nix_knowledge/scripts/nixlm/merge_casper_safetensors.py
```

Dataset and run metadata are stored locally under:

```text
nix_knowledge/models/training_data/
```

Known metadata files include:

```text
casper_puca_public_mix_v4.metadata.json
casper_puca_public_mix_v5.metadata.json
nix_humanlike_public_mix_v3.metadata.json
last_training_run.json
```

The style corpus is intended to teach conversational behavior, not durable facts. It emphasizes:

- Concise natural replies.
- No forced interview questions.
- No automatic “How can I help?” endings.
- Honest uncertainty.
- Emotional restraint and tone matching.
- Human conversational rhythm.
- Casper/PUCA identity.
- No fabricated personal memories.

Public-dialogue data must remain subject to its source license and local-use review. Do not commit generated datasets or model weights.

### 4.4 Evaluation artifacts

Existing local evaluation JSON files include:

```text
nix_knowledge/models/nixlm/casper-puca-v5-eval.json
nix_knowledge/models/nixlm/casper-puca-v4-eval.json
nix_knowledge/models/nixlm/qwen35-humanlike-eval.json
nix_knowledge/models/nixlm/qwen35-humanlike-v3-eval.json
```

They are ignored local artifacts. Treat them as evidence, not as an automated guarantee of production quality.

## 5. Active backend and optional ExLlama path

### Active verified backend

```text
NIX_CASPER_BACKEND=transformers
```

Casper is loaded locally through `casper_model.py` with CUDA, PEFT, and the QLoRA adapter. Ollama may be installed and may expose `qwen3.5:4b`, but it is not the active Casper path when the backend is `transformers`.

### Tabby/ExLlama boundary

`nix_core/tabby_client.py` implements an optional OpenAI-compatible TabbyAPI transport. Select it only after a compatible artifact and running Tabby server are verified:

```text
NIX_CASPER_BACKEND=tabby
NIX_TABBY_API_URL=http://127.0.0.1:5000/v1/chat/completions
NIX_TABBY_MODEL=casper-puca-v5
```

The current Casper adapter is a PEFT adapter over `Qwen3_5ForConditionalGeneration`. It is not directly verified for ExLlamaV2. Do not switch the default backend merely because an EXL2/EXL3 artifact exists for a different base or fine-tune.

`nix_core/Modelfile.casper-puca-v5` references a merged model directory:

```text
/root/nix_knowledge/models/qwen35-casper-puca-v5-merged
```

That merged directory is a local deployment artifact and is not part of the source repository. The merge utility is `merge_casper_safetensors.py`. Do not overwrite an existing merged output without an explicit backup/approval.

## 6. Knowledge engine and temporal correctness

### 6.1 Temporal authority

`nix_knowledge/nix_knowledge/temporal.py` and `context.py` are the only date/time authorities. The selector model must never decide absolute dates.

Every Knowledge result should carry `temporal_context`, including:

- Current timezone.
- Current local ISO timestamp.
- Current local date/time display.
- Today’s absolute date.
- Tomorrow’s absolute date.
- Weekday and relevant window boundaries.

Calendar result entries should carry:

- `start` and `end` ISO timestamps with timezone offset.
- `start_date`, `start_time`, `end_date`, `end_time`.
- Human-readable `start_local` and `end_local`.
- Original temporal expression.
- Relative label.
- `temporal_grounding` with absolute values.

### 6.2 Neuro-symbolic calendar boundary

The selector may propose a title and temporal expression. `temporal_hybrid.py` then:

1. Compares proposal coverage against all temporal markers in the original request.
2. Rejects partial proposals that drop offsets, day parts, or clock ranges.
3. Searches symbolic suffix/prefix candidates from the raw request.
4. Resolves the selected expression through `TemporalResolver`.
5. Rejects unresolved or conflicting expressions before mutation.

This protects against failures such as:

```text
5 days after today ...
```

being left in the title while only `from 9am to 11am` is parsed.

Speech-normalized forms currently covered include:

- `tmr`, `tmrw`.
- `calender` calendar typo.
- `in 2 more days`.
- `around 4pm`.
- `day after tomorrow`.
- `next Monday morning from 9am to 11am`.
- `this weekend at 7pm` as one upcoming occurrence.
- `this Friday` as the upcoming Friday when the current week’s Friday has passed.

### 6.3 Person states

Person mood/health/well-being statements are temporal states, not durable facts. The state parser and store live in:

```text
nix_knowledge/nix_knowledge/states.py
```

Examples:

- `my sister is sick`
- `Jane is fine`
- `my sistser is doign alright now`

The parser normalizes common ASR/typo forms, resolves people through the semantic entity registry, and supersedes contradictory prior states. Ambiguous relationships must trigger clarification rather than selecting a person randomly.

Core stores pending clarification by conversation ID in `brain.py`; the next answer is consumed as continuation instead of routed as a new independent request.

## 7. Core identity and metadata protections

Identity questions are deterministic and bypass Casper generation:

- `Who created Casper?`
- `Who created you Casper?`
- `Who made you?`
- `Who are you?`
- `Who are you again?`

Internal fields such as `route` and `rule` are useful for logs/dashboard diagnostics but must not be emitted as the user’s reply. `brain.py` contains route-metadata sanitization for leaked strings such as:

```text
CHAT rule: casper_transformers
```

If this appears in a browser, determine whether it is a diagnostic trace/API field or actual reply text before changing the model.

## 8. Dashboard and services

### Service defaults

```text
Knowledge API: 127.0.0.1:8100
Actions API:   127.0.0.1:8200
Websocket:     127.0.0.1:9000 by default
Console:       127.0.0.1:35567 in the current deployment
Timezone:      America/Chicago
```

The deployed dashboard URL used during development was:

```text
http://100.108.149.71:35567/
```

Do not hard-code that address into source; use environment configuration.

### Useful endpoints

```text
GET  /api/health
POST /api/send              {"text": "...", "conversation_id": "..."}
GET  /api/sessions
GET  /api/schedule
```

Health should report Casper backend, Casper warm-up, Knowledge warm-up, GPU state, and service health.

### Startup

Use the Knowledge virtual environment because it contains Torch/Transformers:

```bash
/root/nix_knowledge/.venv/bin/python /root/nix_knowledge/scripts/knowledge_api.py
/root/nix_knowledge/.venv/bin/python /root/nix_actions/scripts/actions_api.py
/root/nix_knowledge/.venv/bin/python /root/nix_core/console.py
/root/nix_knowledge/.venv/bin/python /root/nix_core/ws_server.py
```

The console/websocket startup binds before model warm-up so cold-start clients do not receive connection refusal while Casper and Knowledge load.

## 9. Sessions and follow-up behavior

`nix_actions/nix_core/sessions.py` and `nix_core/followup.py` implement conversational continuity.

- Actions owns persisted session turns.
- Core uses bounded recent context.
- The websocket opens a short follow-up window after a prompt.
- A pending Knowledge clarification is keyed by conversation ID.
- The answer to a clarification must complete the original request.
- Dashboard sessions should read the live Actions session store, not an unused Core database path.

Never assume a bare follow-up such as `Maanvi` is a new request if Core has a pending clarification.

## 10. Testing and evaluation

### Core tests

From the repository root:

```bash
PYTHONPATH=nix_core:nix_knowledge:nix_actions:. \
  nix_knowledge/.venv/bin/pytest -q nix_core --ignore=nix_core/test_e2e_subprocess.py
```

Focused tests include:

```text
nix_core/test_conversation.py
nix_core/test_fast_path.py
nix_core/test_followup.py
nix_core/test_knowledge_composition.py
nix_core/test_temporal_regressions.py
nix_core/test_tabby_backend.py
```

### Knowledge tests

Run from `nix_knowledge/`:

```bash
cd nix_knowledge
PYTHONPATH=.. .venv/bin/pytest -q tests/test_temporal.py tests/test_temporal_hybrid.py tests/test_rules.py
```

The full Knowledge suite may require the package’s expected environment/import layout. If it reports an import collision around `nix_knowledge.models`, record that as an environment/package-layout issue instead of changing model data blindly.

### Real-model temporal evaluation

Do not test create-event prompts against the real database. Use a temporary database and the actual `KnowledgeNeedle` selector:

```bash
cd nix_knowledge
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.. .venv/bin/python <isolated-evaluation-script>
```

Manually inspect every output for:

- Correct absolute date.
- Correct start/end time.
- Correct timezone offset.
- Full event title and meaningful details.
- Correct `window` for calendar queries.
- No partial selector proposal.
- No internal route metadata in the user-facing reply.

### Full-stack tests

The dashboard’s Testing page is designed to run prompts against sandboxed copies of databases. Prefer that path for broad behavioral matrices. Never use production/live SQLite paths for mutation-heavy tests.

## 11. Runtime data and safety

The following are local and must remain ignored:

- `data/` and all SQLite databases.
- `nix_core/logs/` request JSONL logs.
- `.venv/`, `nix_env/`, Python caches, test caches.
- Downloaded model weights and adapters.
- Training datasets and generated artifacts.
- `.env` and credentials.
- Agent/Freebuff state.

The current development machine contains private shell history, local service state, Ollama files, and downloaded models. Do not add those to GitHub.

## 12. Known operational caveats

1. The repository’s package names remain `nix_*` even though the user-facing identity is Casper.
2. The active Casper backend is Transformers + PEFT CUDA, not ExLlama/Tabby.
3. Ollama may be healthy but is not necessarily the active Casper backend.
4. Existing database records may have been created by older parser versions. Do not silently rewrite calendar records; inspect and ask before migrating user data.
5. A log entry’s `rule=casper_transformers` is expected internal metadata. It is not a model answer unless the same text appears in the reply body.
6. Local model cards contain placeholder fields from the generated Hugging Face template. The authoritative runtime paths are in `casper_model.py`, `casper_runtime.py`, and `config.py`.

## 13. Safe change checklist for future agents

Before changing a routing or temporal rule:

1. Read the corresponding raw request log entry.
2. Reproduce in a temporary Knowledge database.
3. Run the actual selector when the failure involves model-proposed arguments.
4. Compare structured ISO timestamps, not only human relative labels.
5. Check title preservation and timezone offsets.
6. Add a focused regression test.
7. Run Core and Knowledge temporal suites.
8. Restart long-running services before live verification.
9. Verify `/api/health` and at least one read-only `/api/send` request.
10. Never repair/mutate existing user calendar data without explicit approval.
