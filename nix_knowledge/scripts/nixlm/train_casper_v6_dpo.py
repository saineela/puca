"""Memory-conscious DPO-style preference tuning for Casper V6.

A single quantized base model is used twice per batch: once with the trainable
adapter enabled and once with the adapter disabled as the reference policy.
This avoids loading a second 4B model on an 8 GiB RTX 4060.
"""
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models" / "qwen3.5-4b-hf"
SFT_ADAPTER = ROOT / "models" / "nixlm" / "casper-puca-qlora-v6-sft"
DATA = ROOT / "models" / "training_data" / "casper_v6_preferences.jsonl"
OUT = ROOT / "models" / "nixlm" / "casper-puca-qlora-v6"
MAX_LENGTH = 512
BETA = 0.1
VRAM_FRACTION = 0.68


def read_rows(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = ("prompt", "chosen", "rejected")
    if not rows or any(not all(isinstance(row.get(key), str) and row[key].strip() for key in required) for row in rows):
        raise ValueError("preference data must contain non-empty prompt/chosen/rejected fields")
    return rows


def encode(tokenizer, prompt: str, completion: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (prompt_ids + completion_ids)[-(MAX_LENGTH):]
    prompt_length = min(len(prompt_ids), len(ids))
    labels = [-100] * prompt_length + ids[prompt_length:]
    if len(labels) < len(ids):
        labels = [-100] * (len(ids) - len(labels)) + labels
    return (
        torch.tensor([ids], dtype=torch.long, device="cuda"),
        torch.tensor([labels], dtype=torch.long, device="cuda"),
    )


def sequence_logprob(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids, labels=None, use_cache=False)
    logits = output.logits[:, :-1, :]
    next_labels = labels[:, 1:]
    valid = next_labels != -100
    safe_labels = next_labels.masked_fill(~valid, 0)
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--adapter", type=Path, default=SFT_ADAPTER)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({"mode": "check-only", "base": str(args.base), "adapter": str(args.adapter), "data": str(args.data), "output": str(args.output), "cuda": torch.cuda.is_available(), "rows": len(read_rows(args.data))}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("DPO-style training requires CUDA")
    if args.output.resolve() == (ROOT / "models" / "nixlm" / "casper-puca-qlora-v5").resolve():
        raise RuntimeError("refusing to overwrite V5")
    rows = read_rows(args.data)
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-6, weight_decay=0.01)

    step = 0
    losses: list[float] = []
    for epoch in range(args.epochs):
        for row in rows:
            chosen_ids, chosen_labels = encode(tokenizer, row["prompt"], row["chosen"])
            rejected_ids, rejected_labels = encode(tokenizer, row["prompt"], row["rejected"])
            chosen_logp = sequence_logprob(model, chosen_ids, chosen_labels)
            rejected_logp = sequence_logprob(model, rejected_ids, rejected_labels)
            with model.disable_adapter():
                with torch.no_grad():
                    ref_chosen = sequence_logprob(model, chosen_ids, chosen_labels)
                    ref_rejected = sequence_logprob(model, rejected_ids, rejected_labels)
            logits = BETA * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected))
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu()))
            print(f"step {step} epoch {epoch + 1} loss {losses[-1]:.4f} allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "preference_train_log.json").write_text(json.dumps({"rows": len(rows), "epochs": args.epochs, "steps": step, "mean_loss": sum(losses) / max(1, len(losses)), "beta": BETA}, indent=2), encoding="utf-8")
    print(f"preference adapter saved to {args.output}")


if __name__ == "__main__":
    main()
