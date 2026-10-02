# Casper / Nix Project Handoff

This document is the durable context for future agents working in this repository. It records the architecture, model lineage, runtime contracts, data boundaries, testing rules, and operational decisions that previously lived only in conversation history.

> **Current Luna directive (2026-09-27):** Repository source retires Luna Pro v1 from the runtime registry; former Pro API handlers return HTTP 410. This does not establish the state of any hosted process. Preserve the Pro research, datasets, builders, manifests, and artifacts as historical records only. Never execute the Pro training workflow or reuse/adapt its data or training approach for **any** model. Luna work is limited to the V6 runtime/integration path; no Luna training or new offline evaluation is authorized. Later research notes in this handoff are historical context, not operational permission. This repository cannot establish any hosted deployment's serving model or live state.
>
> **Product identity:** The repository still uses `nix_*` package and environment names for compatibility. The PUCA/system identity is **Casper**. Casper's creator response is deterministic: **"Created and Built by Sai Neela, and living in NIX's PUCA system."**

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
  -> active configured backend/model
  -> bounded session history + Knowledge memory block
  -> final reply
```

In the repository's Transformers selector, Luna V6 is the startup default; Casper models require explicit selection. A non-Transformers backend bypasses that selector. This source-level selection is not evidence of the model served by any hosted process. The chat model must not invent personal memory; the memory block and structured Knowledge results are authoritative.

### 3.2 Actions flow

```text
Knowledge creates/updates/cancels calendar event
  -> nix_knowledge.bridge
  -> nix_actions capture/schedule/reschedule/cancel
  -> action linked with source_record_id
```

Do not modify the Actions database directly when testing normal behavior. Use the Knowledge/API path or an isolated temporary database.

## 4. Casper model lineage and files

### 4.1 Base model lineage

The Casper Transformers implementation's source-level default base path is the Qwen3.5 4B HF checkpoint:

```text
nix_knowledge/models/qwen3.5-4b-hf/
```

Important metadata:

- Architecture in `config.json`: `Qwen3_5ForConditionalGeneration`.
- The local HF model is a multimodal Qwen3.5 architecture, not a simple legacy causal model.
- The model index reports approximately 9.32 GB of unquantized safetensor data.
- The directory is ignored by Git; its current presence and contents are not established by this repository snapshot.

### 4.2 Casper V5 adapter lineage

The Casper-specific backend's default adapter path is:

```text
nix_knowledge/models/nixlm/casper-puca-qlora-v5/
```

When the ignored local adapter is present, its recorded `adapter_config.json` metadata is:

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

Relevant source-level path and runtime overrides are:

```text
CASPER_BASE_MODEL_PATH
CASPER_V5_ADAPTER_PATH
CASPER_V6_ADAPTER_PATH
CASPER_VRAM_FRACTION
CASPER_MAX_NEW_TOKENS
CASPER_FAST_MAX_NEW_TOKENS
NIX_CASPER_MAX_CONCURRENT_REQUESTS
```

When the Casper Transformers backend is selected, the implementation lazily loads a selected Casper adapter through Transformers + PEFT on CUDA; its code defaults are a `0.68` VRAM fraction and one concurrent decode. These defaults do not prove local artifact presence, a loaded model, or a hosted serving identity. Earlier development notes recorded an RTX 4060 and approximately 3.8–4.3 GiB GPU use; that is historical telemetry, not a statement about current deployment state.

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

## 5. Luna V6 runtime and archived research

Luna is separate from Casper. Luna V6 is the only Luna model in the runtime
registry and is the configured Transformers-selector startup default. A
non-Transformers backend can
bypass that selector. Local artifacts/status do not prove a hosted serving
identity.

**No Luna training is authorized.** Keep Luna work to the V6 runtime and
integration path; do not run Luna training or dataset-building workflows.
The Pro-specific data and training approach must not be reused or adapted for
V6 or any other model. The research, builders, manifests, evaluation sources,
and artifacts remain preserved for provenance only. The paths below identify
historical source files, not approved commands; builder/trainer CLI entry points
are disabled while helper implementations remain for provenance:

```text
nix_knowledge/scripts/nixlm/build_luna_conversation.py      [CLI disabled]
nix_knowledge/scripts/nixlm/train_luna_qlora.py             [not authorized]
nix_knowledge/scripts/nixlm/evaluate_luna_v6.py             [CLI disabled]
nix_core/luna_runtime.py                                    [current V6 runtime]
nix_knowledge/scripts/nixlm/build_luna_resource_mix.py      [CLI disabled]
```

Historical research used local DailyDialog and filtered FineTome-100k,
No Robots, and UltraFeedback chosen-SFT resources, excluding Ubuntu's
unrelated adjacent-line pairs and adding project-authored behavior controls.
FineTome's card did not declare a clear license and was derived from The-Tome;
source terms remain unresolved. Those notes describe preserved provenance,
not permission to regenerate data, train, or reuse the Pro approach. The base
checkpoint and tokenizer details below are retained as research lineage only.

Historical `evaluate_luna_v6.py` runs used the same single-turn and multi-turn
contracts as Casper V6 and reported failures without relabeling them. The creator
case tested a Casper fact rather than Luna's identity. The historical evaluator
is retained for provenance but its CLI is disabled; these records do not
authorize training or new evaluations.

### Archived Luna identity-correction research

The pre-retirement identity work recorded the following Unsloth, TRL, and
Transformers references. They are provenance only, not current guidance. All
Luna corpus-building/offline-evaluation commands are disabled; retained helper
code is not authorization to regenerate data, train, or evaluate:

- The historical notes discussed exact chat-template rendering and assistant
  completion masking.
- They recorded LoRA target modules across attention and MLP projections.
- They described identity paraphrases, entity-disambiguation cases, and
  Luna/Casper boundary examples in the archived corpus.
- They stated that identity examples should not replace deterministic Core
  identity handling, Knowledge grounding, or memory validation.

Historical references listed in the pre-retirement Luna research notes (archive only; not current guidance or run authorization):

- Unsloth LoRA and completion-only guidance:
  `https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide`
- Unsloth fine-tuning and instruct guidance:
  `https://unsloth.ai/docs/get-started/fine-tuning-llms-guide`
- TRL conversational SFT and assistant/completion-only loss:
  `https://huggingface.co/docs/trl/en/sft_trainer`
- Transformers chat-template and special-token guidance:
  `https://huggingface.co/docs/transformers/en/chat_templating`

Historical implementation sources (preserved for provenance; not approved workflows):

```text
nix_knowledge/scripts/nixlm/luna_format.py
nix_knowledge/scripts/nixlm/luna_role_control.py
nix_knowledge/scripts/nixlm/build_luna_identity_mix.py
nix_knowledge/scripts/nixlm/build_luna_balanced_mix.py
nix_knowledge/scripts/nixlm/train_luna_qlora.py
nix_knowledge/scripts/nixlm/evaluate_luna_v6.py
```

The historical notes describe an identity candidate resumed from checkpoint-80
to checkpoint-160, with LR `5e-5`, one-example micro-batches, gradient
accumulation 8, a 512-token cap, 230 identity-control examples, and a recorded
2.555 GiB peak allocation. Their score transcription gives checkpoint-160
**2/9 single-turn and 0/5 multi-turn**, compared with checkpoint-80's **4/9 and
0/5**, and describes a near-match `Sai Neella` plus regressions elsewhere.
These are archive reports, not independently verified results here; the
candidate is not registered, and no new reproduction, activation, training, or
evaluation is authorized.

The archive also records a balanced joint SFT/adapter-selection attempt and
its non-promotion decision. Any former recommendation for follow-on training is
superseded: no Luna training is authorized, and the Pro training approach must
not be reused for any model. Core's deterministic identity guard remains the
authority for exact creator attribution.

The preserved `luna_format.py` documents historical role-separation research
between Nix, Casper, and Luna. Casper's exact identity and personal facts remain
Core-owned authoritative concerns; the archived material does not authorize
new builds, training, evaluation, or Pro-method reuse.

The historical optimization notes describe an audit of token-weighted
gradient accumulation and record exact-prefix label validation, PEFT rsLoRA,
optional DoRA, and separate A/B LoRA learning rates. These notes and preserved
implementation code are provenance only, not permission to reuse that approach
for Luna V6 or any other model.

Research reviewed for the Luna redesign:

- QLoRA, `arXiv:2305.14314`: NF4, double quantization, paged optimizers,
  all-layer adapters, and small high-quality data.
- LoRA, `arXiv:2106.09685`: low-rank updates and mergeable adapters.
- rsLoRA, `arXiv:2312.03732`: `1/sqrt(r)` scaling avoids rank-related gradient
  collapse.
- LoRA+, `arXiv:2402.12354`: different learning rates for LoRA A and B.
- DoRA, `arXiv:2402.09353`: historical reference on magnitude/direction
  decomposition and LoRA capacity gaps.
- LoftQ, `arXiv:2310.08659`: quantization-aware adapter initialization.
- NEFTune, `arXiv:2310.05914`: historical reference on embedding noise and
  SFT overfitting.
- LIMA, `arXiv:2305.11206`: carefully curated small data can beat indiscriminate
  scale.
- SFTMix, `arXiv:2410.05248`: historical reference on training dynamics.
- GRAPE, `arXiv:2502.04194`: historical reference on response distribution.

The archive lists Unsloth sources that were reviewed for the pre-retirement
research; these references do not authorize a Luna run, new evaluation, or
reuse of the Pro approach.

The historical notes transcribe the first balanced rsLoRA checkpoint at **1/9
single-turn and 1/5 multi-turn** in isolation. It is not registered in the
current runtime source. The shared V6 harness's Casper creator case tests a
Casper fact, not Luna's identity; that is a documented limitation, not an
instruction to train or evaluate Luna.

### Historical V6 score report transcriptions (2026-09-23; not actionable)

The historical `evaluate_luna_v6.py` comparison harness was described as
appending an assistant reply after each user turn and saving full traces.
Earlier multi-turn scores from a harness that misread later user turns as
assistant turns were marked invalid in the archive. The table below transcribes
scores attributed to the corrected harness; they are historical reports, not
independently verified results. No evaluator is authorized to run now.
The `luna-instruct-v1` baseline was described as the trained Instruct adapter,
not the bare Llama base; the unadapted base was reported at 2/9 single and 1/5
multi in isolated mode. Current local adapters/reports are not verified here.

| Candidate/checkpoint | Isolated single | Isolated multi | Synthetic-context single | Synthetic-context multi |
|---|---:|---:|---:|---:|
| Bare Instruct base (`--no-adapter`) | 2/9 | 1/5 | — | — |
| `luna-instruct-v1` baseline adapter | 5/9 | 1/5 | 7/9 | 2/5 |
| `luna-v6-recovery-v1` cp60 | 5/9 | 1/5 | — | — |
| `luna-v6-recovery-v1` cp120 | 4/9 | 1/5 | — | — |
| `luna-v6-targeted-v2` cp30 | 5/9 | 0/5 | — | — |
| `luna-v6-targeted-v2` cp60 | 6/9 | 0/5 | 7/9 | 2/5 |
| `luna-v6-contextual-v1` cp30 | 4/9 | 0/5 | 6/9 | 2/5 |
| `luna-v6-contextual-v1` cp60 | 4/9 | 1/5 | 6/9 | 2/5 |

“Synthetic context” is described as an evaluator simulation that injects
hand-authored Knowledge/Actions system messages for selected cases. The cited
evaluation JSON labels this `synthetic_nix_authoritative_memory_and_actions`
and includes a warning in `context_note`; it is **not** an end-to-end
Core/Knowledge/Actions test. The archived comparison reported the baseline at
7/9 + 2/5 and the latest contextual candidate at 6/9 + 2/5 in that matched
setup, with a reported regression for the candidate without context. The
historical notes concluded that these results did not justify promotion at that
time; this is not a claim about a currently loaded or hosted model.

The pre-moratorium notes describe an experiment using the then-cited
`luna_targeted_v6_contextual_v2_sft.jsonl` snapshot (reported as 2,653 rows;
360 complete multi-turn trajectories; 223 examples with trusted Nix context;
zero exact-normalized V6 user-prompt overlaps). Its metadata
`nix_knowledge/models/nixlm/luna-v6-contextual-v1-training-metadata.json`
was cited for hyperparameters, matched scores, resource use, and promotion
rationale. The archived account says the builder was corrected to emit actual
system-message newlines and that the corpus was checked for exact benchmark
overlap before the run. A post-run trace audit reportedly found that the
positive-ambiguity context in that training snapshot mentioned the user's
sisters before a non-family good-news turn, potentially cueing an unwarranted
family question. A later builder-only correction reportedly scoped sibling
memory to turns mentioning a sister and clarified that general good news is not
about family unless the user says so. The notes place that correction after the
reported training; the cited adapter and scores therefore do **not** represent
that corrected data version. The notes also describe a separate corpus snapshot,
`luna_targeted_v6_contextual_v3_untrained_sft.jsonl` (reported as 2,653 rows;
223 contextual examples; 0 exact V6 prompt overlaps), described as generated
after the fix and not trained or scored. Both corpus metadata files and JSONL
snapshots are cited for provenance; current local presence is not established
here. No new build, evaluation, or training run is authorized. The historical
adapter was reported at `models/nixlm/luna-v6-contextual-v1/`, initialized from
`luna-instruct-v1`, with checkpoints recorded at steps 30 and 60. The notes list
LR `1e-5`, rank-16 LoRA without rsLoRA, micro-batch 1, gradient accumulation 8,
and a reported 3.074 GiB peak allocation on an RTX 4060. This describes a
bounded historical experiment, not current artifact or deployment state; the
notes also flagged historical benchmark-exposure risk from the cited
initialization.

Reports were documented as saved under `nix_knowledge/models/nixlm/`; their
current local presence is not verified. The cited paths were
`luna-v6-contextual-v1-checkpoint{30,60}-{isolated,context}-v6.json`. The notes
recorded reported peak inference allocations of about 2.28–2.30 GiB. Archived
trace summaries described some grounding gains in synthetic-context cases but
also reported cadence errors, mishandled clarifications, invented medical
facts, and unnecessary follow-up questions. In isolated cases they described
invented personal details, a guessed medicine schedule, and failed family
clarification. These are reported historical quality/safety concerns, not
current model observations.

Other archived notes summarize ACT and ReSURE as research references. They do
not describe implemented features or authorize follow-on work. No new Luna
research/evaluation/training is authorized, and Pro-method reuse for any model
is prohibited.

- ACT, Google Research/ICLR 2025: `https://research.google/blog/learning-to-clarify-multi-turn-conversations-with-action-based-contrastive-self-training/`
- ReSURE: `https://arxiv.org/html/2508.19996v1`
- Multi-turn conversational-agent survey: `https://arxiv.org/html/2504.04717v5`

Interpret the transcribed V6 counts cautiously: the historical suite was
described as using strict substring gates, including cases with missing
authoritative context in isolated mode (`How is Maya now?`, reminder
scheduling), rejecting a question mark inside a joke for the multi-intent case,
and requiring the literal word `brief` where `short` is semantically
reasonable. Context-mode results are not end-to-end results, and historical
scores should be read alongside their reported replies/traces (whose current
artifact presence is unverified). The handoff archive separately reported
harness/corpus tests passing **19/19** and four candidate reports completing;
those historical claims are not revalidated here. This maintenance pass did not
repeat model evaluations or verify GPU/process status; it ran only focused
CPU/hermetic code tests, not model evaluation or live-runtime verification.

## 6. Source-level backend selection and optional ExLlama path

### Source-level backend choices

```text
NIX_CASPER_BACKEND=transformers  # code default; environment-overridable
```

Source config defaults the official Transformers selector to Luna V6; Casper V5/V6 require explicit selection. With `NIX_CASPER_BACKEND=transformers`, Casper's fallback can lazily load its configured QLoRA adapter through `casper_model.py`. Ollama may be configured as another backend. These are code paths, not evidence of local artifacts, a loaded process, or a hosted serving identity.

### Tabby/ExLlama boundary

`nix_core/tabby_client.py` implements an optional OpenAI-compatible TabbyAPI transport. Select it only after a compatible artifact and running Tabby server are verified:

```text
NIX_CASPER_BACKEND=tabby
NIX_TABBY_API_URL=http://127.0.0.1:5000/v1/chat/completions
NIX_TABBY_MODEL=casper-puca-v5
```

The documented Casper adapter lineage is a PEFT adapter over `Qwen3_5ForConditionalGeneration`. It is not directly verified for ExLlamaV2. Do not switch the configured backend merely because an EXL2/EXL3 artifact exists for a different base or fine-tune.

`nix_core/Modelfile.casper-puca-v5` references a merged model directory:

```text
/root/nix_knowledge/models/qwen35-casper-puca-v5-merged
```

That merged directory is a local deployment artifact and is not part of the source repository. The merge utility is `merge_casper_safetensors.py`. Do not overwrite an existing merged output without an explicit backup/approval.

## 7. Connected dashboard and external API

`nix_core/dashboard.html` is the responsive dashboard asset. `console.py` serves it at `/` and, when run, wires the UI to the Core, Knowledge, and Actions handlers. This source description is not a claim that a live service is reachable. The Skills page is intentionally disabled. The Memories page exposes the rendered memory block and individual record deletion; reset controls require typing `RESET_ALL` and clear Knowledge, derived indexes, Actions, sessions, and trace state.

The console exposes an OpenAI-compatible API for Open WebUI and SDK clients:

```text
GET  /v1/models
POST /v1/chat/completions
```

Set `NIX_OPENAI_API_KEY` for Bearer authentication. The endpoint is wired to the same `Brain.handle` pipeline as the dashboard, preserving routing, Knowledge, Actions, temporal/person memory, conversation history, and final verification through the configured official conversation-model path. Both `stream: false` and OpenAI SSE `stream: true` are supported. The local Transformers path produces a complete grounded answer first, then emits it in progressive SSE chunks; native token callbacks are not exposed. These source-level details do not establish any hosted or live deployment state.

## 8. Knowledge engine and temporal correctness

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

## 9. Core identity and metadata protections

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

## 10. Dashboard and services

### Source-configured service defaults (not verified live)

```text
Knowledge API: 127.0.0.1:8100
Actions API:   127.0.0.1:8200
Websocket:     127.0.0.1:9000 by default
Console:       0.0.0.0:49117 by default; fixed port (restrict host with NIX_CONSOLE_HOST)
Timezone:      America/Chicago
```

A dashboard URL recorded in historical development notes was:

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
/root/nix_knowledge/.venv/bin/python /root/nix_core/console_extend.py
/root/nix_knowledge/.venv/bin/python /root/nix_core/ws_server.py
```

The startup code is intended to bind before model warm-up; this source-level behavior is not a verified observation of a running process.

## 11. Sessions and follow-up behavior

`nix_actions/nix_core/sessions.py` and `nix_core/followup.py` implement conversational continuity.

- Actions owns persisted session turns.
- Core uses bounded recent context.
- The websocket opens a short follow-up window after a prompt.
- A pending Knowledge clarification is keyed by conversation ID.
- The answer to a clarification must complete the original request.
- Dashboard sessions should read the live Actions session store, not an unused Core database path.

Never assume a bare follow-up such as `Maanvi` is a new request if Core has a pending clarification.

## 12. Testing and evaluation

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

## 13. Runtime data and safety

The following are local and must remain ignored:

- `data/` and all SQLite databases.
- `nix_core/logs/` request JSONL logs.
- `.venv/`, `nix_env/`, Python caches, test caches.
- Downloaded model weights and adapters.
- Training datasets and generated artifacts.
- `.env` and credentials.
- Agent/Freebuff state.

Development environments may contain private shell history, local service state, Ollama files, and downloaded models. Do not add those to GitHub; this handoff does not verify the contents of any current machine.

## 14. Known operational caveats

1. The repository’s package names remain `nix_*` even though the system has separate Casper and Luna identities.
2. Source code defaults the official Transformers selector to Luna V6; Casper fallbacks require explicit selection. Non-Transformers backends bypass that selector. This is not evidence of a hosted or currently loaded model.
3. The active backend or adapter for a live process must be verified from that process; Ollama may be healthy without being the selected backend.
4. Existing database records may have been created by older parser versions. Do not silently rewrite calendar records; inspect and ask before migrating user data.
5. A log entry’s `rule=casper_transformers` is expected internal metadata. It is not a model answer unless the same text appears in the reply body.
6. Local model cards contain placeholder fields from the generated Hugging Face template. The authoritative runtime paths are in `casper_model.py`, `casper_runtime.py`, and `config.py`.

## 15. Safe change checklist for future agents

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
