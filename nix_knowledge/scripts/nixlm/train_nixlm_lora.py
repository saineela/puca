"""NixLM LoRA training (owner-approved NixLM transition, 2026-09-12).

Base: the ALREADY-LOCAL qwen2.5-0.5b-instruct (no download).
Method: LoRA r=16 on attention+MLP projections, bf16.
Data:   models/training_data/nixlm/nixlm_train.jsonl  (built by
        build_nixlm_dataset.py)

Output: models/nixlm/nixlm-lora/  (adapters + tokenizer + log)
Per agent.md: single instance, single GPU, logged in project-details.md.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
BASE = ROOT / "models" / "qwen2.5-0.5b-instruct"
TRAIN = ROOT / "models" / "training_data" / "nixlm" / "nixlm_train.jsonl"
EVAL = ROOT / "models" / "training_data" / "nixlm" / "nixlm_eval.jsonl"
OUT = ROOT / "models" / "nixlm" / "nixlm-lora"

MAX_LEN = 768
# 152k-vocab CE logits dominate VRAM: batch 8 OOMs on the 8 GB card
# even with the 0.5B model. batch 2 + checkpointing + accum 8 keeps
# the same effective batch (16).
BATCH = 2
GRAD_ACCUM = 8
EPOCHS = 2
LR = 1.5e-4
WARMUP = 100


def build_example(tokenizer, messages: list[dict]) -> dict:
    """Tokenize a chat transcript; supervise ONLY assistant content."""
    input_ids: list[int] = []
    labels: list[int] = []
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    for msg in messages:
        header = f"<|im_start|>{msg['role']}\n"
        h_ids = tokenizer.encode(header, add_special_tokens=False)
        c_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        # header + newline are never supervised
        input_ids.extend(h_ids)
        labels.extend([-100] * len(h_ids))
        if msg["role"] == "assistant":
            input_ids.extend(c_ids)
            labels.extend(c_ids)
        else:
            input_ids.extend(c_ids)
            labels.extend([-100] * len(c_ids))
        input_ids.extend([im_end])
        labels.append(im_end if msg["role"] == "assistant" else -100)
        input_ids.extend(nl_ids)
        labels.extend([-100] * len(nl_ids))

    input_ids = input_ids[:MAX_LEN]
    labels = labels[:MAX_LEN]
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                ex = build_example(tokenizer, rec["messages"])
                if any(t != -100 for t in ex["labels"]):
                    self.rows.append(ex)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad_id: int):
    n = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        pad = n - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad)
        labels.append(ex["labels"] + [-100] * pad)
        attn.append([1] * len(ex["input_ids"]) + [0] * pad)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attn),
    )


def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def main():
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    # activation memory for the backward pass
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_ds = SFTDataset(TRAIN, tok)
    eval_ds = SFTDataset(EVAL, tok)
    print(f"train examples: {len(train_ds)}  eval: {len(eval_ds)}")

    pad_id = tok.pad_token_id
    loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id), num_workers=2,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id), num_workers=2,
    )

    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    global TOTAL_STEPS
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"total optimizer steps: {TOTAL_STEPS}")
    model.train()
    step = 0
    micro = 0
    running = 0.0
    best_eval = float("inf")
    log = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        for input_ids, labels, attn in loader:
            input_ids, labels, attn = (
                input_ids.cuda(), labels.cuda(), attn.cuda())
            loss = model(
                input_ids=input_ids, labels=labels,
                attention_mask=attn,
            ).loss / GRAD_ACCUM
            loss.backward()
            running += loss.item()
            micro += 1

            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % 50 == 0:
                    el = time.time() - t0
                    msg = (
                        f"epoch {epoch+1} step {step}/{TOTAL_STEPS} "
                        f"loss {running / (50 * GRAD_ACCUM):.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} "
                        f"({el/60:.1f} min)"
                    )
                    print(msg, flush=True)
                    log.append(msg)
                    running = 0.0

        # ---- end of epoch: eval loss ------------------------------
        model.eval()
        tot, n_batches = 0.0, 0
        with torch.no_grad():
            for input_ids, labels, attn in eval_loader:
                out = model(
                    input_ids=input_ids.cuda(), labels=labels.cuda(),
                    attention_mask=attn.cuda(),
                )
                tot += out.loss.item()
                n_batches += 1
        eval_loss = tot / max(1, n_batches)
        msg = f"== epoch {epoch+1} eval_loss {eval_loss:.4f}"
        print(msg, flush=True)
        log.append(msg)
        if eval_loss < best_eval:
            best_eval = eval_loss
            ckpt = OUT / "checkpoint-best"
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            log.append(f"saved best -> {ckpt}")
        model.train()

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT / "checkpoint-final")
    tok.save_pretrained(OUT / "checkpoint-final")
    with (OUT / "train_log.json").open("w") as fh:
        json.dump(
            {"epochs": EPOCHS, "train_examples": len(train_ds),
             "best_eval_loss": best_eval, "log": log},
            fh, indent=2,
        )
    print(f"DONE best_eval_loss={best_eval:.4f} -> {OUT}")


if __name__ == "__main__":
    main()
