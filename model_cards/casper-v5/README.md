---
language:
- en
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-4B
tags:
- peft
- lora
- qwen3-5
- text-generation
- conversational
- nix-puca
# LICENSE METADATA INTENTIONALLY OMITTED: redistribution rights are not cleared.
---

# Casper PUCA V5 — Qwen3.5-4B LoRA Adapter — Draft

> **DRAFT — DO NOT UPLOAD AS A PUBLIC MODEL CARD YET.** The adapter's training manifest lists non-commercial dialogue sources and asks for review before redistribution. The project has not approved a license for this adapter or documented clearance for distributing the resulting weights. See [Hugging Face release checklist](https://github.com/saineela/puca/blob/main/HF_RELEASE_CHECKLIST.md). This card does not grant rights to the adapter, base model, or training data.

> **Evidence status:** The repository contains a local Casper V5 adapter and a small saved comparison-output file, but no current reproducible V5 generation-speed/VRAM benchmark or standardized quality score. Claims below are limited to the stored configuration and artifacts; the project-specific scores require a separately published protocol before release.

## Summary

Casper PUCA V5 is a **text-generation PEFT LoRA adapter** for [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B). Its local training manifest describes the task as style-oriented conversational SFT—not factual training and not training on personal memory records.

This proposed repository contains adapter parameters, not a complete copy of the Qwen base model. To run it, users need the matching Qwen3.5-4B model, compatible Transformers/PEFT versions, and the correct Qwen chat template. The adapter does not contain NIX's Core router, Knowledge database, Actions scheduler, API service, or privacy/security boundaries.

The Qwen3.5-4B **base model family** supports vision-language inputs, including image/video according to its official model documentation. **This Casper V5 adapter and the NIX runtime path documented here are text-only.** The adapter has not been shown to improve multimodal behavior, and the local Casper loader uses a causal text-generation interface. Do not attribute base-model multimodality to evaluated or adapter-trained multimodal capability.

## Model details

| Field | Record |
| --- | --- |
| Name | Casper PUCA V5 adapter |
| Proposed Hub ID | `saineela/nix_casper-assist-conversation-4B-v5` — proposed only; repository existence/ownership not verified |
| Adapter format | PEFT LoRA; `adapter_model.safetensors` + `adapter_config.json` |
| Base model | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), whose current model card declares Apache-2.0; base license does not automatically clear the adapter's training-data terms |
| Intended task | English text generation / conversational style |
| Adapter configuration | Rank 8, alpha 16, dropout 0.05, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`; PEFT 0.20.0 recorded locally |
| Training metadata | Seed 42, 2,941 training rows, 400 evaluation rows, 20 optimizer steps, learning rate `2e-5`, max sequence length 1,024, micro-batch 1, gradient accumulation 1 |
| Recorded base loading | 4-bit NF4 base loading, double quantization, bfloat16 compute, recorded CUDA memory fraction 0.68; a fraction setting is a configured cap, not observed peak usage |
| License | **Not specified. Public distribution is blocked pending rights review and an approved license.** |

The inspected local `adapter_config.json` points to an absolute `/root/.../qwen3.5-4b-hf` directory. Any release staging copy must identify the public base model and exact revision instead. Do not change the preserved training/runtime artifact in place.

## Intended uses

- Research and experimentation on text-conversation LoRA adapters.
- A selectable text-response component in the NIX PUCA application when combined with its separate deterministic application services.
- Reproducible study of adapter loading, text generation, and integration boundaries—after the rights and testing gates are resolved.

A model response is not evidence that a fact is true, memory was stored, a reminder was scheduled, or an action succeeded. An application must verify those results outside the model.

## Out-of-scope uses

Do not use the adapter alone for medical, mental-health, legal, financial, safety-critical, identity-verification, surveillance, or autonomous action decisions. Do not interpret its generated persona text as factual human biography or proof of consciousness. Do not use it as a speech model, voice-cloning system, or multimodal Casper model. Although the base model family is multimodal, this adapter/runtime release has only text-adapter artifacts and text-oriented inference code; no adapted image/video evaluation is documented.

## Known limitations and risks

- The stored V5 sample file contains 14 prompt/output comparisons. It is a small, local artifact—not an independently scored benchmark or human evaluation.
- There is no reproducible V5 model-generation benchmark in the repository's safe benchmark harness. Do not use the Core router microbenchmark (about tenths of a millisecond on a specific classifier operation) as Casper generation speed or conversational throughput.
- No measured V5-specific peak allocated/reserved VRAM report, cold-start distribution, warm-generation p50/p95, first-token latency, or multi-run hardware manifest was found in the reviewed release records.
- Generated responses can contain unsupported personal claims, inappropriate first-person preferences, excessive questions, or persona framing. Applications should clearly label the system as AI, ground it in trusted context, validate mutations deterministically, and avoid asserting completed operations before confirmation.
- The local dataset metadata says 2,941 training and 400 evaluation examples; it does not provide evidence of broad or standardized language-model capability. Stored example completions are illustrative qualitative samples only.
- The base supports image/video input, but no multimodal fine-tuning or evaluation is documented for this adapter. Do not infer multimodal capability or improved vision performance from the base model alone.
- NIX's tokenizer/chat template and response policy may differ from other inference stacks; pin software versions and test the exact published adapter with the selected base.

## Performance and resource reporting

**No publishable Casper V5 generation speed/VRAM score is established by the reviewed repository evidence.** The current `nix_core/benchmarks/speed_benchmark.py --benchmark router` measures `nix_core/router.py::classify`, a small deterministic routing/classification operation. It does not load Casper, generate tokens, run a full NIX turn, or measure GPU memory. Do not attribute the router's local timing or throughput to this adapter.

The recorded `CASPER_VRAM_FRACTION=0.68` is an inference-time per-process setting in the NIX loader; it is not a measurement. The pilot metadata's 0.68 setting is also not measured peak usage. The historic 3.8–4.3 GiB figure elsewhere in the project is from separate Casper development context and is not a verified V5 benchmark; it is intentionally not presented here as this adapter's resource score.

Before claiming model performance, publish a controlled and repeatable benchmark containing GPU and driver, OS, Python/PyTorch/Transformers/PEFT/bitsandbytes versions, exact base and adapter revisions, quantization, chat template, input/output token lengths, cold load time, warm p50/p95 generation latency, samples and warm-up, peak CUDA allocated and reserved memory, and whether results cover only generation or the full NIX service pipeline.

## Training data and provenance

The project-local `casper_puca_public_mix_v5.metadata.json` describes a style-only mix of DailyDialog and EmpatheticDialogues. It records 2,941 training rows, 400 evaluation rows, seed 42, and filters for boilerplate, URLs, sensitive secrets, crisis/self-harm details, and malformed turns. Metadata says the corpus is not intended as a factual or personal-memory corpus and explicitly says to review source terms before redistribution.

The project manifest records CC BY-NC-SA 4.0 for these sources, and the current EmpatheticDialogues Hugging Face dataset card lists CC BY-NC 4.0. Each exact source revision and license must be checked. A public adapter upload is redistribution of a derived model artifact; citation alone does not establish permission. The Qwen3.5 base model license is separate from all dataset terms. Do not upload the local training/eval JSONL, raw source data, or private logs with the adapter.

## Reproduction / inference example

The example below loads a public text-generation base and a PEFT adapter. **This exact example has not yet been tested against a public staged artifact.** It is for planning only until the adapter config is sanitized, an exact base revision is pinned, compatible package versions are tested, and rights are cleared.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3.5-4B"
ADAPTER_ID = "saineela/nix_casper-assist-conversation-4B-v5"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER_ID, is_trainable=False).eval()
messages = [
    {"role": "system", "content": "You are Casper. Answer concisely and do not invent personal memories."},
    {"role": "user", "content": "Hello."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
))
```

Note: the NIX development loader uses a 4-bit NF4 base and a text-only causal-LM path. This card's example does not reproduce NIX's end-to-end request pipeline, nor does it demonstrate vision/video support. Ensure the current Qwen3.5 Transformers architecture and processor requirements are handled by the exact version selected for release. Prefer the official base-model documentation for image/video applications.

## Evaluation status

No standardized Casper V5 evaluation score is provided in this card because the reviewed file is a small example-output comparison rather than a defined scoring protocol. The owner-reported manual testing is not reproducible from the submitted materials. The existing `CASPER_V6_RESEARCH.md` results concern **Casper V6**, not V5; they are not transferable to this adapter.

Before publishing numerical quality claims, create an authorized, held-out, documented V5 test protocol with sample count, prompt provenance, scoring criteria, evaluator, base comparison, and reproducible run configuration. Do not include private/user data in public evaluation prompts.

## Citation

If this adapter is cleared for release, cite its immutable Hub revision, for example:

```bibtex
@misc{neela2026casperv5adapter,
  author       = {Neela, Sai},
  title        = {Casper PUCA V5 Qwen3.5-4B LoRA Adapter},
  year         = {2026},
  howpublished = {Hugging Face model repository},
  url          = {https://huggingface.co/saineela/nix_casper-assist-conversation-4B-v5}
}
```

## References

- [Qwen3.5-4B base model](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5 multimodal architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5)
- [Hugging Face PEFT adapter checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)
- [DailyDialog paper](https://aclanthology.org/I17-1099/)
- [EmpatheticDialogues paper and dataset](https://aclanthology.org/P19-1534/)
- [NIX PUCA repository](https://github.com/saineela/puca)
- [Hugging Face release checklist](https://github.com/saineela/puca/blob/main/HF_RELEASE_CHECKLIST.md)
