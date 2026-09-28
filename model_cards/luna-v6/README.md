---
language:
- en
library_name: peft
pipeline_tag: text-generation
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- peft
- lora
- llama
- text-generation
- conversational
- nix-puca
# LICENSE METADATA INTENTIONALLY OMITTED: redistribution rights are not cleared.
---

# Luna V6 Contextual LoRA Adapter for NIX PUCA — Draft

> **DRAFT — DO NOT UPLOAD AS A PUBLIC MODEL CARD YET.** This local candidate has no approved redistribution license. The base model is gated and governed by Meta's Llama 3.2 Community License; the training-data manifest includes non-commercial and unresolved-provenance sources. Resolve the rights and license gate in [Hugging Face release checklist](https://github.com/saineela/puca/blob/main/HF_RELEASE_CHECKLIST.md) before creating a public Hub repository. This card does not grant rights to the adapter, base model, or training data.

> **Research status:** The NIX source registry selects `luna-v6-contextual-v1` as its local Transformers default. That software setting is not equivalent to promotion of this particular checkpoint as a validated release-quality model. Archived project notes record this V6 candidate as **not promoted** after small matched checks. The NIX PUCA application's deterministic memory, routing, and action services are separate from this adapter.

## Summary

Luna V6 is a **text-generation PEFT LoRA adapter** for the Llama 3.2 3B Instruct base, fine-tuned for concise conversational behavior and selected context-aware dialogue patterns. It is intended as one replaceable response-generation component in the NIX Personal User Companion Agent (PUCA) architecture.

This repository would contain adapter parameters, not the Llama base-model weights. Loading it requires authorized access to [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), compliance with the base-model license and acceptable-use requirements, and compatible Transformers/PEFT versions. Meta gates access to that base on Hugging Face.

**This is not** a voice model, speech recognizer, speech synthesizer, personal-memory database, reminder engine, multimodal model, or complete standalone PUCA application. It does not call tools or independently access external NIX services.

## Model details

| Field | Record |
| --- | --- |
| Name | Luna V6 contextual adapter |
| Proposed Hub ID | `saineela/nix_luna-human-conversation-3B-v6` — proposed only; repository existence/ownership not verified |
| Adapter format | PEFT LoRA; `adapter_model.safetensors` + `adapter_config.json` |
| Base model | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), gated; confirm exact base revision before release |
| Intended task | English text generation / conversation |
| Adapter configuration | Rank 16, alpha 16, dropout 0.05; targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; PEFT 0.20.0 recorded in the local config |
| Quantization | The adapter file itself is not a quantized full-model checkpoint. NIX's local inference path loads a quantized base with 4-bit NF4, double quantization, and bfloat16 compute. |
| Training source snapshot | `luna_targeted_v6_contextual_v2_sft.jsonl`, 2,653 examples per local metadata; 60 recorded optimizer steps, learning rate `1e-5`, micro-batch 1, gradient accumulation 8, sequence cap 768 |
| License | **Not specified. Public distribution is blocked pending rights review and an approved license.** |

The staged public adapter config must replace the local absolute `/root/...` base path with the verified public base-model ID. Do that in a release staging copy only; preserve the original project artifact. See the checklist.

## Intended uses

- Research on compact text-dialogue adapters and context-aware response generation.
- As a language-generation component in a system that separately supplies user-authorized, validated context.
- Evaluation of how structured application code and a replaceable language model can divide responsibilities.

A model response is not proof that a memory was retrieved, a reminder was scheduled, a person was correctly identified, or an action completed. An application must verify tool/service results independently.

## Out-of-scope uses

Do not use this adapter alone for medical, mental-health, legal, financial, safety-critical, identity-verification, surveillance, or autonomous action decisions. Do not treat generated dialogue as a factual record or as a substitute for verified personal data. Do not represent it as a human, conscious entity, or scientifically measured “human-like” speaker. It emits text; real-time voice requires separate speech-input/output components.

## Known limitations and risks

- This is a small research model plus adapter, not a verified production-quality standalone assistant.
- Without trusted application context, a language model may guess personal details, mishandle ambiguity, misstate a reminder, or produce irrelevant follow-up questions.
- The shared internal evaluation contains only nine single-turn and five multi-turn cases and uses literal/string-based checks. It is not a human-subject study, standardized benchmark, or independent review.
- In the archived matched comparison, Luna V6 scored **4/9 isolated single-turn and 1/5 isolated multi-turn**; with hand-authored synthetic context it scored **6/9 and 2/5**, respectively. The Instruct-adapter baseline in that report scored **5/9 and 1/5 isolated**, and **7/9 and 2/5 with synthetic context**. The synthetic-context cases were not a full Core/Knowledge/Actions test. These are historical reported results, not a new evaluation.
- The project owner reports that manual testing of the individual model and its integration with connected services met the owner's intended accuracy requirements. No dated, repeatable test protocol, prompt set, run log, or sample count was supplied; this is owner-reported qualitative feedback, not a published performance score, and does not replace the archived evaluation record above.
- The training manifest reports no personal chat logs in the V2 target snapshot, but that snapshot included project-authored controls and the run started from an earlier Instruct adapter with historical project controls. A post-training audit also noted a potentially over-broad family-context hint. A separate corrected V3 corpus is documented as untrained and unscored; it did not produce this adapter.
- Exact-string benchmark overlap checks do not rule out semantic overlap or exposure from the initialization adapter.
- The adapter contains no speech I/O, tool-execution sandbox, privacy boundary, or NIX database. These are responsibilities of separate application components.

## Historical speed and resource observations

These are archived, one-environment research observations—not guaranteed performance, a service-level objective, or a comparison against other models.

| Measurement | Archived observation | Scope and caveat |
| --- | ---: | --- |
| Generation time | Median **2.323 s** across nine different single-turn cases; observed range **0.394–7.738 s** | One archived direct-generation evaluation, 80-token generation ceiling, one generation per case. Timer started after tokenization and ended after generation; excludes base/adapter load, tokenization, network/API, NIX routing, Knowledge, Actions, and response composition. Prompt difficulty and output length varied. |
| Peak CUDA allocated memory | **2.280 GiB** in the archived isolated-evaluation JSON | PyTorch allocated-memory field only; the JSON does not report peak reserved memory or a complete deployment memory budget. It is not independently remeasured here. |
| Training peak allocated memory | **3.074 GiB** in training metadata | Historical training run, **not inference**. Do not present this as inference VRAM. |

The local NIX loader also sets an environment-overridable CUDA per-process memory fraction. A fraction is a ceiling/control, not evidence of measured peak usage. None of these figures establish that a complete NIX PUCA conversation, all services, or every compatible runtime fits below 3 GB VRAM. The available archive does not capture a sufficiently complete machine/software manifest to make these figures independently reproducible.

**No end-to-end model speed score is available here.** The sub-millisecond NIX Core route microbenchmark measures a separate Python classifier; it is not Luna generation time and is not included in the values above.

## Training data and provenance

The local V2 snapshot metadata reports 2,653 examples:

- 1,800 broad dialogue/instruction examples (listed source counts: DailyDialog 1,292; No Robots 185; UltraFeedback chosen-SFT 183; FineTome-100k 140);
- 461 project controls, 18 authored single-turn examples, 360 authored multi-turn trajectories, and 14 role controls;
- 223 context-bearing examples are reported as a subset of the total, not additional rows.

The project metadata lists DailyDialog as CC BY-NC-SA 4.0, No Robots as CC BY-NC 4.0, and UltraFeedback as MIT. The FineTome-100k snapshot's exact license/provenance is unresolved in the archived project notes. Meta Llama base terms are separate. These source labels are not a legal determination that distributing this adapter is permitted. Recheck exact dataset/base revisions and derivative-work terms before any public release. Do not upload source corpora or personal logs as part of this adapter repository.

## Reproduction / inference example

This is an illustrative PEFT loading pattern, not a guarantee that a particular hardware/software setup will fit in memory. The base is gated: first request/accept its terms on Hugging Face, then authenticate using Hugging Face's supported credential flow. Do not paste tokens into code.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_ID = "saineela/nix_luna-human-conversation-3B-v6"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    quantization_config=quantization,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base, ADAPTER_ID, is_trainable=False).eval()

messages = [
    {"role": "system", "content": (
        "You are Luna. Be concise; do not invent memories or claim actions "
        "that were not supplied as confirmed context."
    )},
    {"role": "user", "content": "Hello."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
print(tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
))
```

Package/model revisions should be pinned and the exact example tested against the published adapter and authorized base before the card is made public. This example is base-model-plus-adapter generation only; it does not reproduce NIX's complete Core/Knowledge/Actions pipeline.

## Evaluation protocol summary

Existing reports were produced with the repository's archived Luna V6 evaluation code. That command is currently disabled by project policy. The reported small-suite checks use nine single-turn cases and five sequential multi-turn cases, literal/string-based assertions, and a second mode that injects hand-authored context for selected cases. The latter is a simulation, not an end-to-end system test. No new Luna evaluation has been run for this card.

## Citation

If the adapter is cleared for release, cite its immutable Hub revision, for example:

```bibtex
@misc{neela2026lunav6adapter,
  author       = {Neela, Sai},
  title        = {Luna V6 Contextual LoRA Adapter for NIX PUCA},
  year         = {2026},
  howpublished = {Hugging Face model repository},
  url          = {https://huggingface.co/saineela/nix_luna-human-conversation-3B-v6}
}
```

## References

- [Llama 3.2 3B Instruct base model and license](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Hugging Face PEFT adapter checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)
- [NIX PUCA repository](https://github.com/saineela/puca)
- [Hugging Face release checklist](https://github.com/saineela/puca/blob/main/HF_RELEASE_CHECKLIST.md)
