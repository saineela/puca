# Hugging Face Release Preparation — NIX PUCA Models

**Status: draft only — no model weights have been uploaded.** These notes prepare two PEFT adapter releases for the Hugging Face Hub. They do not grant redistribution rights, constitute legal advice, authorize more Luna training/evaluation, or promise that the adapters are production-ready.

## Proposed model repositories

- Luna V6 adapter: [`saineela/nix_luna-human-conversation-3B-v6`](https://huggingface.co/saineela/nix_luna-human-conversation-3B-v6)
- Casper V5 adapter: [`saineela/nix_casper-assist-conversation-4B-v5`](https://huggingface.co/saineela/nix_casper-assist-conversation-4B-v5)

These destination pages are proposed only. Their existence, ownership, visibility, namespace authorization, and availability have **not** been verified. Create them only after release gates below are closed. Casper's `4B` name reflects the inspected `Qwen/Qwen3.5-4B` base; do not call it a 3B model.

## What is in the local model folders

Both releases are **PEFT LoRA adapters**, not complete standalone model checkpoints. A user needs the matching base model, its access/license, compatible Transformers/PEFT support, and the correct chat template. Do not upload a base model copy with the adapter unless its license and redistribution terms expressly allow that distribution.

### Luna V6 candidate

- Adapter directory in this checkout: `nix_knowledge/models/nixlm/luna-v6-contextual-v1/checkpoint-60/`.
- Base model identity: [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), under Meta's Llama 3.2 Community License and gated Hub access. Each recipient must independently satisfy Meta/Hugging Face access requirements.
- Local artifact originally records an absolute `/root/...` base path; the release copy must use the public base-model ID. **Do not edit the preserved training/runtime artifact in place.** Make a staging copy, normalize paths in the copy, and test that copy against the intended public base revision.
- Recorded PEFT configuration: LoRA, rank 16, alpha 16, dropout 0.05, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; PEFT metadata says 0.20.0.
- It is text-generation/instruct-adapter style. Luna's current runtime uses the repository's explicit Llama chat serialization and Luna system prompt; standalone users need these details. A Hugging Face Transformers model card alone does not install NIX's request router, memory, scheduler, or safeguards.
- **Candidate status:** V6 is a local official-source selector default, but its archived V6 experimental results did not show an improvement over the Instruct baseline on the small shared checks; the documented promotion record says “not promoted.” Treat the release as a research/experimental adapter unless the owner separately establishes a qualified release decision. The repo must disclose both isolated and synthetic-context protocol scopes; injected context was not a live end-to-end NIX stack test.
- Resource evidence is historical and not a public inference guarantee: the stored run metadata says 3.074 GiB peak allocated during one training run and about 2.28 GiB peak during one archived inference evaluation. The exact GPU/software/protocol need verification and disclosure before using those figures. They do not substantiate “under 3 GB end-to-end” or “world's first.”
- Dataset manifest references DailyDialog, No Robots, UltraFeedback chosen-SFT, FineTome-100k, and project-authored controls. The metadata lists non-commercial restrictions for at least DailyDialog and No Robots and unresolved FineTome provenance/rights. Confirm derivative-model redistribution is permitted for the exact source revisions and resulting adapter before public upload. Linking/citing source data is not sufficient clearance.
- No Luna evaluation may be run as part of this preparation. The local policy limits Luna activity to existing V6 runtime/integration work and prohibits new offline evaluation/training.

### Casper V5 candidate

- Adapter directory in this checkout: `nix_knowledge/models/nixlm/casper-puca-qlora-v5/`.
- Base model identity: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), whose current Hub card declares Apache-2.0. The base is multimodal at the model-family level, but Casper V5's adapter/runtime path is text-only (`AutoModelForCausalLM`, text messages); this adapter has **not** been verified for vision/image/video inference. Do not market the V5 adapter as multimodal.
- Local adapter config also embeds an absolute `/root/...` path. Normalize that value only in an isolated release staging copy, to the verified public base identifier/revision, then test the staged adapter. Preserve source artifact provenance.
- Recorded configuration: LoRA rank 8, alpha 16, dropout 0.05, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, PEFT metadata says 0.20.0. Training pilot metadata: 2,941 examples, 20 optimizer steps, sequence cap 1,024, learning rate `2e-5`, seed 42, 4-bit NF4 base loading, double quantization, bfloat16 compute; review whether stored training metadata is complete before publishing these details.
- A local 14-prompt JSON file records base-versus-adapter outputs, but it is not a standardized score report and does not by itself support a quality number. The saved outputs include potentially problematic claims/persona language (e.g. personal preferences, invented relationship suggestions, “we're friends”); review the sample table for privacy, safety, persona, and provenance before publishing examples. Do not omit material known weaknesses from a release card.
- The V5 training metadata names DailyDialog and EmpatheticDialogues with CC BY-NC-SA 4.0 and says to review source terms before redistribution. A public adapter release is still public distribution. Verify exact revisions, source terms, and whether a derived model may be shared under the proposed terms. No blanket open-source claim is established by Unsloth, GitHub citations, or the base-model license alone.
- The recorded local model's inference allocation history is not a controlled V5 benchmark. There is no reproducible model-generation benchmark for this V5 candidate in the current benchmark harness. The root router microbenchmark is not an adapter-generation benchmark and must not be attributed to Casper.
- Do not advertise image/video or other multimodal fine-tuning simply because the Qwen3.5 base supports vision. PEFT target modules are text-backbone projections; adapter training/evaluation is recorded as text dialogue, and current Casper runtime only accepts text messages.

## Required release gates (block public upload until complete)

1. **Rights and license review**
   - Review Meta's Llama 3.2 Community License and gated model restrictions for Luna; confirm that the adapter and all included artifacts may be distributed and identify any required downstream notices.
   - Review each exact training source/revision for both models, including non-commercial restrictions and derivative-model redistribution questions. Resolve FineTome provenance for Luna or exclude the affected material only if an authorized, already available artifact exists; do not rebuild training data here.
   - Choose a truthful license per adapter. Do not declare `apache-2.0`, `mit`, or `open-source` for a derivative merely because one base is Apache-2.0 or training was done with Unsloth. If no approved license can be stated, mark the Hub release blocked and do not upload.
   - Confirm permission to publish project-authored training material, examples, and the model name/branding.
2. **Release artifact review**
   - Stage copies outside tracked model folders; scrub absolute local paths, cache/local-only flags, user names, private paths, and local-only metadata from the staged copy.
   - Include only the adapter weight file, correct adapter config with public `base_model_name_or_path`, compatible tokenizer files/chat template actually required for use, an accurate `README.md` model card, and applicable notices. Avoid base weights, datasets, eval prompts containing personal data, local logs, optimizer files, temporary checkpoints, and private data.
   - Ensure HF model-card YAML has an approved `license` value, `base_model` Hub ID, appropriate task, language and conservative tags. Do not use high-volume but inaccurate tags (e.g. `multimodal`, `speech`, `human-like`, `under-3gb-vram`, or `world-first`).
   - Pin both base model and adapter to a commit hash/revision in card/code when possible; record the checksum of the staged adapter and provenance for the local checkpoint. Current local path values are not a revision.
3. **Correctness and reproducibility**
   - Verify adapter tensor keys/shape compatibility with its stated base architecture; ensure the public `base_model` ID and adapter-config public ID agree.
   - Test the **staged copy** in a clean environment with the documented dependency versions and authorized base-model access. Exercise deterministic one-turn load/generation, correct chat template and EOS behavior, CPU/CUDA expectations, and full adapter loading. Do not modify original weights.
   - For Casper, obtain a controlled, reproducible generation speed/VRAM report before listing any speed figures (GPU/driver, OS, Transformers/PyTorch/PEFT/bitsandbytes versions, base and adapter revisions, quantization, prompt set, warm-up/repeat count, input/output tokens, load vs warm latency, peak allocated/reserved VRAM). The current route benchmark is unrelated to model decoding.
   - Luna must not be evaluated again under the current project directive. The release card may link and accurately summarize existing archived research, clearly labeling it historical and its synthetic context as non-end-to-end.
4. **Hub account and review**
   - Verify the account namespace `saineela`, obtain explicit account-owner confirmation of the exact public repository IDs and visibility, and create repos through the Hub UI/API only after rights clearance.
   - Review the cards, metadata, files, checksums, and license text in a private/local staging review before the first public push. Publishing weights is an external, public, difficult-to-reverse action: obtain separate explicit approval before doing it.
   - Do not put a Hugging Face access token in source files, README, shell history, or chat. Authenticate through the approved local credential mechanism.

## Public claims allowed by current evidence

- “PEFT LoRA adapter for [exact base model].”
- “NIX PUCA conversation-model candidate” with an explicit local research status.
- Recorded adapter rank/targets and archival training settings, identified as project metadata.
- Existing archival measurement values only if the precise protocol/environment is attributed and the run limitations are stated.
- Casper V5 is a **text-only adapter/runtime candidate** even though the Qwen3.5 base model family supports image/video input.

## Claims not established

- No evidence supports “world's first PUCA,” “human-like/human-level speech,” a sub-3-GB full end-to-end NIX claim, an end-to-end model speed score from the router benchmark, or multimodal behavior of these adapters.
- The models are not self-contained weights for their base models.
- No account namespace ownership, destination-repository existence, public visibility, final license, gated base-model access, public access, or upload is verified by this draft.
- Owner-reported manual acceptance testing can be added separately with a date, hardware/software/configuration, sample count, prompts or protocol, and authorship label. It must not replace or contradict the retained historical results silently.

## Sources

- [Hugging Face model cards and metadata](https://huggingface.co/docs/hub/model-cards)
- [Hugging Face PEFT checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)
- [Hugging Face PEFT model loading API](https://huggingface.co/docs/peft/en/package_reference/peft_model)
- [Hugging Face Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Meta Llama 3.2 3B Instruct card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5 Transformers architecture and modalities](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)
- [No Robots dataset card](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [UltraFeedback dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [FineTome-100k dataset card](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- [EmpatheticDialogues dataset card](https://huggingface.co/datasets/facebook/empathetic_dialogues)

All model and dataset terms must be checked at the exact source revisions and at the time of release. This checklist is not legal advice.
