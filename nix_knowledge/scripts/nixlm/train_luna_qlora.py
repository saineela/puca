"""Archived Luna trainer; execution is disabled by the current no-training directive.

The historical implementation is retained for provenance only. Luna work is
limited to the V6 runtime/integration path; no Luna training is authorized.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from luna_format import EOT, render_messages

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("LUNA_BASE_MODEL_PATH", str(ROOT / "models" / "llama-3.2-3b-unsloth-instruct")))
DATA = Path(os.environ.get("LUNA_TRAIN_DATA", str(ROOT / "models" / "training_data" / "luna_balanced_v5_sft.jsonl")))
OUT = Path(os.environ.get("LUNA_ADAPTER_PATH", str(ROOT / "models" / "nixlm" / "luna-instruct-v1")))
MAX_LENGTH = int(os.environ.get("LUNA_MAX_LENGTH", "768"))
MAX_STEPS = int(os.environ.get("LUNA_MAX_STEPS", "120"))
CHECKPOINT_EVERY = int(os.environ.get("LUNA_CHECKPOINT_EVERY", "60"))
LORA_RANK = int(os.environ.get("LUNA_LORA_RANK", "16"))
EPOCHS = int(os.environ.get("LUNA_EPOCHS", "1"))
GRAD_ACCUM = int(os.environ.get("LUNA_GRAD_ACCUMULATION", "8"))
MICRO_BATCH = int(os.environ.get("LUNA_MICRO_BATCH", "1"))
LR = float(os.environ.get("LUNA_LR", "5e-5"))
LORA_USE_RSLORA = os.environ.get("LUNA_USE_RSLORA", "1") == "1"
LORA_USE_DORA = os.environ.get("LUNA_USE_DORA", "0") == "1"
LORA_B_LR_MULTIPLIER = float(os.environ.get("LUNA_LORA_B_LR_MULTIPLIER", "8"))
VRAM_FRACTION = float(os.environ.get("LUNA_VRAM_FRACTION", "0.60"))
RESUME_FROM = Path(os.environ["LUNA_RESUME_FROM"]) if os.environ.get("LUNA_RESUME_FROM") else None
INITIAL_STEP = int(os.environ.get("LUNA_INITIAL_STEP", "0"))


def examples(tokenizer):
    """Tokenize exact chat templates with assistant-only labels.

    Multi-turn rows supervise every assistant span, not only the final reply.
    The old last-message-only behavior made a multi-turn curriculum look like
    context during training while never teaching the model to produce the
    clarification/acknowledgement turns inside that context.
    """
    output = []
    all_assistant_turns = os.environ.get("LUNA_ALL_ASSISTANT_TURNS", "1") == "1"
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line)["messages"]
            full_text = render_messages(messages, generation=False, tokenizer=tokenizer)
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if not full_ids:
                continue

            spans: list[tuple[int, int]] = []
            assistant_indexes = [
                index for index, message in enumerate(messages)
                if str(message.get("role", "")).casefold() == "assistant"
            ]
            if not all_assistant_turns:
                assistant_indexes = assistant_indexes[-1:]
            valid = True
            for index in assistant_indexes:
                prefix_text = render_messages(messages[:index], generation=True, tokenizer=tokenizer)
                end_text = render_messages(messages[: index + 1], generation=False, tokenizer=tokenizer)
                prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                end = tokenizer(end_text, add_special_tokens=False)["input_ids"]
                # The official template must be a literal prefix of the full
                # sequence. This catches duplicated BOS/control tokens and
                # prevents silently shifting assistant-only labels.
                if full_ids[: len(prefix)] != prefix or full_ids[: len(end)] != end:
                    valid = False
                    break
                if len(end) > len(prefix):
                    spans.append((len(prefix), len(end)))
            if not valid or not spans:
                continue

            full = full_ids[:MAX_LENGTH]
            labels = [-100] * len(full)
            for start, end in spans:
                if start >= len(full):
                    continue
                for position in range(start, min(end, len(full))):
                    labels[position] = full[position]
            if not any(label != -100 for label in labels):
                continue
            output.append({"input_ids": full, "labels": labels})
    return output


def pad(batch, pad_id):
    size = max(len(x["input_ids"]) for x in batch)
    ids, labels, mask = [], [], []
    for item in batch:
        n = size - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * n)
        labels.append(item["labels"] + [-100] * n)
        mask.append([1] * len(item["input_ids"]) + [0] * n)
    return tuple(torch.tensor(x, dtype=torch.long, device="cuda") for x in (ids, labels, mask))


def main() -> None:
    raise PermissionError(
        "Luna model training is not authorized. Continue only with V6 "
        "runtime/integration work; no Luna training may be run."
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Luna QLoRA requires CUDA")
    if not BASE.is_dir():
        raise FileNotFoundError(f"Llama 3.2 3B Instruct checkpoint is not present: {BASE}")
    if not DATA.is_file():
        raise FileNotFoundError(f"Luna dataset is not present: {DATA}")
    if os.environ.get("NIX_CORE_USE_KNOWLEDGE_MODEL_GATE") == "1":
        raise RuntimeError("Disable the Qwen Knowledge gate before training Luna")
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    model.config.use_cache = False
    for parameter in model.parameters(): parameter.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.set_float32_matmul_precision("high")
    if RESUME_FROM is not None:
        if not RESUME_FROM.is_dir():
            raise FileNotFoundError(f"Luna resume adapter is not present: {RESUME_FROM}")
        model = PeftModel.from_pretrained(model, RESUME_FROM, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_RANK,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                task_type="CAUSAL_LM",
                use_rslora=LORA_USE_RSLORA,
                use_dora=LORA_USE_DORA,
            ),
        )
    data = examples(tokenizer)
    if not data: raise RuntimeError("Luna corpus produced no supervised examples")
    import bitsandbytes as bnb
    trainable = [p for p in model.parameters() if p.requires_grad]
    # LoRA+ reports that A and B benefit from different learning rates. Keep
    # the multiplier conservative and configurable; rsLoRA remains the default
    # stabilizer for rank-16 adapters.
    a_params, b_params, other_params = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A" in name:
            a_params.append(parameter)
        elif "lora_B" in name:
            b_params.append(parameter)
        else:
            other_params.append(parameter)
    groups = []
    if a_params: groups.append({"params": a_params, "lr": LR})
    if b_params: groups.append({"params": b_params, "lr": LR * LORA_B_LR_MULTIPLIER})
    if other_params: groups.append({"params": other_params, "lr": LR})
    optimizer = bnb.optim.PagedAdamW8bit(groups)
    if MICRO_BATCH < 1:
        raise ValueError("LUNA_MICRO_BATCH must be >= 1")
    if GRAD_ACCUM < 1:
        raise ValueError("LUNA_GRAD_ACCUMULATION must be >= 1")
    steps_target = MAX_STEPS or max(
        1, (len(data) + MICRO_BATCH * GRAD_ACCUM - 1) // (MICRO_BATCH * GRAD_ACCUM)
    ) * max(1, EPOCHS)
    if INITIAL_STEP > steps_target:
        raise ValueError("LUNA_INITIAL_STEP cannot exceed LUNA_MAX_STEPS")
    model.train(); optimizer.zero_grad(set_to_none=True)
    step = INITIAL_STEP
    micro = 0
    loss_sum = 0.0
    # Resume at the corresponding data position. Without this offset, every
    # checkpointed chunk replayed the first examples and did not add coverage.
    first_micro = INITIAL_STEP * GRAD_ACCUM
    final_micro = steps_target * GRAD_ACCUM
    import torch.nn.functional as F
    for micro_index in range(first_micro, final_micro):
        start = (micro_index * MICRO_BATCH) % len(data)
        batch = data[start:start + MICRO_BATCH]
        if len(batch) < MICRO_BATCH:
            batch += data[: MICRO_BATCH - len(batch)]
        # Unsloth documented that averaging per-micro-batch CE losses is wrong
        # when response lengths differ. Normalize every micro loss by the
        # total supervised-token count of its accumulation window instead.
        window_tokens = 0
        accumulation_start = (micro_index // GRAD_ACCUM) * GRAD_ACCUM
        for offset in range(GRAD_ACCUM):
            window_index = accumulation_start + offset
            window_start = (window_index * MICRO_BATCH) % len(data)
            window_batch = data[window_start:window_start + MICRO_BATCH]
            if len(window_batch) < MICRO_BATCH:
                window_batch += data[: MICRO_BATCH - len(window_batch)]
            window_tokens += sum(sum(label != -100 for label in item["labels"]) for item in window_batch)
        input_ids, labels, attention = pad(batch, tokenizer.pad_token_id)
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss = token_loss / max(1, window_tokens)
        loss.backward(); loss_sum += float(loss.item()); micro += 1
        if micro % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                print(
                    f"step={step} loss={loss_sum:.4f} "
                    f"vram={torch.cuda.memory_allocated()/2**30:.2f}GiB",
                    flush=True,
                )
                loss_sum = 0.0
            if CHECKPOINT_EVERY and step % CHECKPOINT_EVERY == 0:
                checkpoint = OUT / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                print(f"checkpoint={checkpoint}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT); tokenizer.save_pretrained(OUT)
    (OUT / "luna_training_config.json").write_text(json.dumps({"model": "Luna", "base": str(BASE), "data": str(DATA), "examples": len(data), "steps": step, "rank": LORA_RANK, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "use_rslora": LORA_USE_RSLORA, "use_dora": LORA_USE_DORA, "lora_b_lr_multiplier": LORA_B_LR_MULTIPLIER, "vram_fraction": VRAM_FRACTION, "micro_batch": MICRO_BATCH, "gradient_accumulation": GRAD_ACCUM, "effective_batch": MICRO_BATCH * GRAD_ACCUM, "resume_from": str(RESUME_FROM) if RESUME_FROM else None, "thinking": False}, indent=2), encoding="utf-8")
    print(json.dumps({"adapter": str(OUT), "examples": len(data), "steps": step, "peak_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 3)}, indent=2))

if __name__ == "__main__": main()
