# Archived: Luna Pro training plan and local pilot record (retired)

> **Retirement record (2026-09-27):** This file is preserved as historical
> research; it is not an active plan. Luna Pro v1 is retired
> in this repository: it is not registered for model selection, and former
> Luna API handlers return HTTP 410. Its research, builders,
> manifests, data sources, and artifacts are preserved, but this plan is not
> executed and its training approach or data is not reused for any model.
> Luna development is V6-only. All commands,
> recommendations, environment notes, and model-run details below are archival.

**Historical pilot record (as documented at retirement):** A local QLoRA pilot
was recorded as completed from the frozen v8 candidate at 384 tokens. The
adapter was described as separate from `luna-instruct-v1`, all V6 checkpoints,
and Casper. At retirement, the record listed data defects and unresolved
provenance/rights questions and stated that distribution was ruled out.
The Git-ignored local `benchmark_report.md` and `training_metadata.json` at
`nix_knowledge/models/nixlm/luna-pro-v1-topical-v8-384-retry2/` were cited as
records of the historical run; their current presence is not verified.

The original design plan below predates the recorded v8 pilot. It describes
a V3 training proposal and explicitly not the data or procedure used by that
historical run.

## Historical paired-context probe record (archived)

A historical 2026-09-23 paired test was recorded for nine fictional cases / 15
user turns, without external context and with injected fictional context,
against the Instruct adapter and `luna-v6-contextual-v1/checkpoint-60`. Its
record reported zero exact prompt collisions with the shared Casper V6 suite or
the scanned Instruct mix, contextual V2 snapshot, and corrected V3 snapshot. It
cited the trace report
`nix_knowledge/models/nixlm/luna-generalization-probes-paired-20260923.json`
and overlap audit at the same path plus `.overlap.json`; current artifact
presence is not verified.

Results are diagnostic, not statistical proof. Context helped some recall and
clarification turns, but both adapters failed important cases. The contextual
candidate invented collaborator biographies/states, confused a four-day
recurrence with four hours, said a failed reminder succeeded without result
context, and claimed the action happened before any simulated result arrived.
The Instruct baseline also hallucinated personal biography and successful
actions when context was missing, and fell back to stale memory in one context
case. This was described as supporting controlled research, **not release as a
trusted memory/action agent**. Injected context must never be represented as
an end-to-end Nix Core/Knowledge/Actions result.

## Historical goals and boundaries

The items in this section document superseded Pro-era research goals only;
they are not current objectives.

The remaining sections are historical archives.
All Luna dataset-builder and offline-evaluation CLIs are disabled. The Pro
trainer's `check-data` command is retained as CPU-only/read-only and does not
load weights; Pro training/GPU smoke and all new Luna evaluation commands fail
closed.

Before retirement, Pro was described as a model-quality experiment for
context-aware, multi-turn conversational behavior. The historical goals were:

- Use relevant, explicit supplied context while abstaining when it is missing.
- Ignore irrelevant or noisy memory rather than forcing it into the answer.
- Prefer recent corrections/current state to superseded historical notes.
- Ask for a missing entity referent instead of guessing among candidates.
- Describe only externally validated action success/failure and preserve exact
  recurrence, units, and timing.
- Maintain Luna's identity boundary: Luna remains a separate conversation
  model; Casper remains the PUCA identity. This archived product note does not
  assert which model is configured or served in any current deployment.

A language model does **not** create, retrieve, or validate durable memory or
perform side effects. Production Core must remain authoritative for Knowledge
and Actions. Pro is not a substitute for system policy, validators, tool result
contracts, or safety handling.

## Historical training-data and contamination notes (archived)

The preserved proposal recorded the following details and review criteria:

- **Base:** local
  `nix_knowledge/models/llama-3.2-3b-unsloth-instruct/`.
- **Initialization proposal:** the historical note described a clean adapter
  from the Instruct baseline, and rejected resuming the contextual V2 adapter,
  Instruct adapter, or prior experiments.
- **Initial curriculum candidate at the time:** corrected
  `nix_knowledge/models/training_data/luna_targeted_v6_contextual_v3_untrained_sft.jsonl`
  (reported as 2,653 rows; 360 complete multi-turn trajectories; 223
  context-bearing examples; zero exact shared-suite overlap at build time).
  The note recorded that snapshot and metadata should remain read-only; it did
  not establish statistical disjointness from other material.
- **License review:** the V3 note listed No Robots CC-BY-NC and FineTome/source
  provenance questions as unresolved and recorded that local research is not a
  general license grant.
- **Audit criteria:** the former proposal listed overlap checks, semantic
  near-duplicate review, split isolation, and separation of evaluation fixtures
  from data and checkpoint selection.
- **Quality criteria:** the historical note listed role/length/template/mask,
  duplicate-hash, and evidence/result checks, plus review of synthetic examples
  and supported action claims. These criteria are preserved as provenance,
  not as work instructions.

### Historical pre-pilot data proposal (superseded; not actionable)

The historical proposal treated V3 as a starting snapshot and described
additional coverage. This proposal is superseded. Its
archival categories were:

1. Correct memory use versus missing/irrelevant/conflicting/noisy context.
2. Entity ambiguity and clarification continuations for varied relationships
   and non-family entities.
3. Explicit correction, revocation, and current-state supersession.
4. Action lifecycle as an event sequence: request → pending (no success claim)
   → explicit success/failure result → faithful summary; varied units and
   recurrence values including negative/decoy numbers.
5. Confidence-calibrated abstention and concise, natural non-intrusive tone.
6. Ordinary general chat, held-out style controls, and Luna/Casper identity
   separation so the model does not overfit to memory/action scripts.

The former proposal also called for balanced held-out splits, counterfactual
pairs, negative controls, and human-reviewed targets. These are preserved as
historical notes.

## Archived Unsloth research notes (archived)

Pre-retirement notes described Unsloth Core (Python API) as the preferred
optimized trainer and recorded project use of Transformers, PEFT, and a custom
token-weighted assistant-only trainer. The Git-ignored
`.venvs/luna-pro-unsloth` environment was smoke-checked on 2026-09-23 without
modifying `nix_knowledge/.venv` or loading model weights:

- CPython 3.13; `unsloth==2026.9.11`, `unsloth-zoo==2026.9.7`,
  `torch==2.12.1+cu132`, `transformers==5.5.0`, `trl==0.24.0`,
  `peft==0.21.0`, `bitsandbytes==0.50.2`, `datasets==4.3.0`.
- Unsloth and `FastLanguageModel` imports passed; PyTorch saw CUDA 13.2 and the
  NVIDIA GeForce RTX 4060.
- The local Llama tokenizer loaded with `local_files_only=True`; the chat
  template produced the expected assistant generation header.
- This was a package/device/template smoke check only. It did **not** load
  model weights, train, or verify a forward/backward pass, assistant-label
  masks, or checkpoint save/reload. That unverified scope is recorded as a
  limitation of the historical check.

The remaining environment, installation, training-configuration, and evaluation
material is retained solely as historical research. The old procedural
recommendations below are archival records.

Historical pre-run notes (transcribed proposal):

1. The proposal advised checking OS, Python, Torch/CUDA build, NVIDIA driver,
   free disk, package compatibility, and `nvidia-smi`; it also advised
   preserving `nix_knowledge/.venv` rather than upgrading it in place or
   replacing its Torch/CUDA.
2. It recommended the isolated venv and then-current official **Linux** install
   route, warned against the Studio `curl | sh` installer and global installs,
   and advised checking package guidance and recording versions.
3. It proposed a bounded model forward/backward and adapter save/load smoke
   test, including inspection of assistant-label masks and EOS/EOT behavior.
   These were proposed checks only.
4. It proposed comparing any changed Unsloth/Llama/Transformers template,
   tokenizer, loss mask, checkpoint format, or inference path against local
   references before a run. This is retained solely as historical text.

Official references listed in the pre-retirement research notes (archive only):

- Unsloth fine-tuning guide: <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide>
- Unsloth LoRA hyperparameter guide:
  <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide>
- Unsloth dataset guide:
  <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide>
- Unsloth current pip/uv install guide:
  <https://unsloth.ai/docs/get-started/install/pip-install>
- Hugging Face TRL SFTTrainer:
  <https://huggingface.co/docs/trl/en/sft_trainer>

The historical reading notes summarized Unsloth's general guidance on Instruct
models, QLoRA, adapter targets, assistant-only supervision, and held-out
selection. They also recorded that TRL's `assistant_only_loss=True` depends on
template-generated masks and discussed prefix/mask validation. This is source
provenance only; the links above are archival references.

## Superseded historical run configuration (proposal; not executed)

The pre-retirement proposal recorded the following comparison values; they are
archival context only:

| Setting | Initial controlled value | Reason / guard |
|---|---:|---|
| Model | Llama 3.2 3B Instruct | Fresh, independent Pro adapter |
| Quantization | NF4 4-bit, double quantization if supported | Leaves headroom for activations/KV |
| Compute dtype | bf16 if local hardware reports support; otherwise fp16 | Verify numerical stability first |
| Adapter | LoRA rank 16, alpha 16; all q/k/v/o + gate/up/down | Known project baseline; an r=32/alpha=32 ablation only if 16 underfits |
| Dropout | 0 initially or 0.05 matched to existing run | Do not stack many speculative techniques |
| Microbatch | 1 | Fits observed RTX 4060 budget |
| Gradient accumulation | 8 (effective batch 8), token-weighted loss | Preserve length-correct loss normalization |
| Sequence cap | 768 initially; inspect truncation and raise only with evidence | Current trajectories fit this scale; avoid wasting VRAM |
| Learning rate | 2e-5 primary, 5e-5 comparison; Unsloth's general LoRA defaults are much higher but not blindly applied to a delicate 3B behavioral correction | LR sweep is a controlled ablation, not a single assumed best value |
| Schedule | short warmup then cosine/linear decay | Record exact steps and seed; compare matched runs |
| Epochs/steps | one data pass or a short pilot with evaluation checkpoints | Do not repeat rows to an arbitrary target step count |
| Checkpoint frequency | 25–50 optimizer steps, to unique output path | Never overwrite an existing adapter/checkpoint |
| Seed | fixed (e.g. 1307) plus a second-seed repeat if results improve | Separate treatment effect from sampling/seed variance |
| Optimizer | Unsloth/TRL supported paged AdamW 8-bit if validated | Do not carry the custom LoRA+ 8x B multiplier into the first Unsloth run |
| Context | no chain-of-thought labels, no reasoning traces | Only observable conversation/tool-result contracts |

The historical contextual experiment was recorded with LR `1e-5`, rank 16
without rsLoRA, and 60 steps, without improvement on its matched checks. The
old note cautioned that more steps, higher rank, rsLoRA, DoRA, DFT loss, packing,
NEFTune, LoRA+, DPO, or GRPO were not proven improvements. Its run-isolation,
output-path, logging, and resource-release advice is retained as history only.

## Historical evaluation/promotion proposal (archived)

These pre-retirement gates are preserved to explain the old research decision.
The list below summarizes what the old proposal specified; its imperative
wording is historical.

1. **Pre-run:** validate data/template/mask integrity; freeze a train/eval/test
   manifest and hashes; audit benchmark/entity/trajectory leakage.
2. **During training:** log supervised-token loss, eval loss, gradient norm,
   LR, tokens/sec, peak allocated/reserved VRAM, elapsed time and sample counts.
   Stop on nonfinite loss, evidence of truncation/mask errors, or worsening held-
   out behavioral results. Training loss is not release quality.
3. **Paired offline tests:** Instruct baseline vs Pro checkpoint; same tokenizer,
   generation config, cases, and context conditions. Use at least a development
   set and a genuinely untouched final test set. Expand beyond the current
   9-case panel with multiple independently authored templates/entities per
   slice; do not tune on final test failures.
4. **Rubric:** blind human review of every safety/grounding/action trace and a
   stratified ordinary-chat sample. Report per-slice rates and confidence
   intervals for paired correctness, unsupported-claim rate, false-success
   rate, unsafe-memory-use rate, clarification precision, relevance, response
   length, latency, and regressions. Keep exact-match/string checks only as
   narrow diagnostics.
5. **Safety gates:** no fabricated personal facts or tool success, no action
   completion before a positive tool result, correct failure/pending handling,
   latest explicit correction honored, ambiguous referents clarified, and no
   regression on core conversation style/identity. A single high-severity false
   success or unsupported sensitive fact blocks promotion pending remediation.
6. **Repeatability:** reproduce any apparent win with a second seed or an
   independent expanded panel; archive raw traces and score adjudications.
7. **Historical proposal:** the pre-retirement note contemplated an opt-in
   research preview after offline evaluation. This proposal is superseded: Pro
   is retired and its preview path is disabled.

## Recorded historical v8 pilot (September 2026; not independently verified here)

The retirement record describes a run using the frozen
`luna_pro_sft_v1_topical_v8` candidate—not the proposed V3 snapshot above—in
the isolated `.venvs/luna-pro-unsloth` environment. According to that record,
the default 768-token attempt OOM'd before an optimizer step and a 384-token
attempt completed 90 steps before a final-save guard bug; a later fresh run
reportedly completed after a bounded sequence-length option and checkpoint-safe
save guard were added. These details are not independently verified by the
current repository snapshot. The record cites output path
`nix_knowledge/models/nixlm/luna-pro-v1-topical-v8-384-retry2/`, 718/1,219 train
and 87/165 dev conversations (longer rows reportedly dropped whole), 4.991 GiB
peak allocated VRAM, and paired traces for eight frozen fictional cases. Its
cited local `benchmark_report.md` reportedly records qualitative wins, misses,
limitations, and hashes; current artifact presence is not verified. None of
this certifies data quality, rights, general model superiority, or integrated
Nix behavior.

The retirement record states that the adapter was removed from the repository's
runtime registry and that former model/API routes return HTTP 410; these are
source-level facts, not verification of a hosted process. The dashboard source
labels Pro as retired. The record states that, at that time, no model had been
published or distributed. The held-out fictional panel was not a real Nix
integration test. Preserve this account for provenance.

## Current retirement status (supersedes every proposal above)

- The retirement record states that no V3 training was started and that the recorded v8 run was a separate historical experiment.
- The retirement record stated that no model had been published to a hub or distributed at that time; this is historical and does not verify current hosted/deployment state.
- This document makes no claim about a live or hosted deployment.
- The retirement record lists dataset provenance/rights and v8 review findings as unresolved.
- Repository source marks Pro retired; its training method, data, and workflow are not reused for any model.
