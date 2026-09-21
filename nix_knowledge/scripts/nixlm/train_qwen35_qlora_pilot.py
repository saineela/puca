"""Conservative QLoRA pilot for Nix's Qwen3.5 4B conversational style.

This trains an adapter only. It never modifies the base checkpoint and refuses
to start when another Ollama model is loaded. The pilot is intentionally tiny:
style first, then evaluation before any larger run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models" / "qwen3.5-4b-hf"
DATA = Path(os.environ.get(
    "NIX_LORA_DATA",
    str(ROOT / "models" / "training_data" / "nix_humanlike_style_v2.jsonl"),
))
OUT = Path(os.environ.get(
    "NIX_LORA_OUT",
    str(ROOT / "models" / "nixlm" / "qwen35-humanlike-qlora-v2"),
))

MAX_LENGTH = int(os.environ.get("NIX_LORA_MAX_LENGTH", "1024"))
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = int(os.environ.get("NIX_LORA_GRAD_ACCUMULATION", "8"))
EPOCHS = int(os.environ.get("NIX_LORA_EPOCHS", "1"))
LEARNING_RATE = float(os.environ.get("NIX_LORA_LR", "3e-5"))
MAX_STEPS = int(os.environ.get("NIX_LORA_MAX_STEPS", "0"))
VRAM_FRACTION = 0.68  # about 5.57 GiB of an 8 GiB card


def _messages_text(tokenizer, messages, *, generation: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
    }
    try:
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_examples(tokenizer) -> list[dict[str, list[int]]]:
    examples = []
    with DATA.open(encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line)["messages"]
            prefix = _messages_text(tokenizer, messages[:-1], generation=True)
            full = _messages_text(tokenizer, messages, generation=False)
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            if len(full_ids) <= len(prefix_ids):
                continue
            full_ids = full_ids[:MAX_LENGTH]
            labels = [-100] * min(len(prefix_ids), len(full_ids))
            labels += full_ids[len(labels):]
            if any(label != -100 for label in labels):
                examples.append({"input_ids": full_ids, "labels": labels})
    return examples


def _pad(batch, pad_id: int):
    length = max(len(item["input_ids"]) for item in batch)
    inputs, labels, masks = [], [], []
    for item in batch:
        padding = length - len(item["input_ids"])
        inputs.append(item["input_ids"] + [pad_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
    device = "cuda"
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA pilot requires CUDA")
    if not BASE.exists():
        raise FileNotFoundError(f"Trainable base checkpoint missing: {BASE}")
    if not DATA.exists():
        raise FileNotFoundError(f"Pilot dataset missing: {DATA}")

    # Do not compete with Ollama or another model on the same GPU.
    if os.environ.get("OLLAMA_NUM_PARALLEL") not in (None, "0", "1"):
        raise RuntimeError("OLLAMA_NUM_PARALLEL must be 0 or 1 for the pilot")
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # Standard PEFT preparation casts several frozen parameters to FP32,
    # which temporarily costs ~2.4 GiB on this 8 GiB card. Keep the frozen
    # Qwen base in its loaded low-precision form for the pilot instead.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    examples = build_examples(tokenizer)
    if not examples:
        raise RuntimeError("No trainable examples were produced")
    print(f"pilot examples: {len(examples)}")
    print(f"allocated before training: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    import bitsandbytes as bnb

    optimizer = bnb.optim.PagedAdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    running = 0.0
    for _epoch in range(EPOCHS):
        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start : start + BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue
            input_ids, labels, attention_mask = _pad(batch, tokenizer.pad_token_id)
            loss = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            ).loss / GRADIENT_ACCUMULATION
            loss.backward()
            running += float(loss.item())
            micro_step += 1
            if micro_step % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if MAX_STEPS and step >= MAX_STEPS:
                    print("reached configured max steps", flush=True)
                    break
                print(
                    f"optimizer step {step} loss "
                    f"{running:.4f} allocated "
                    f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB",
                    flush=True,
                )
                running = 0.0
        if MAX_STEPS and step >= MAX_STEPS:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    (OUT / "pilot_config.json").write_text(
        json.dumps(
            {
                "base": str(BASE),
                "examples": len(examples),
                "rank": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "max_length": MAX_LENGTH,
                "vram_fraction": VRAM_FRACTION,
                "steps": step,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "learning_rate": LEARNING_RATE,
                "dataset": str(DATA),
            },
            indent=2,
        )
    )
    print(f"pilot adapter saved to {OUT}")


if __name__ == "__main__":
    main()
