"""Evaluate a policy-conditioned Casper V6 adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models" / "qwen3.5-4b-hf"
DATA = ROOT / "models" / "training_data" / "casper_v6_policy_train.jsonl"
ADAPTER = ROOT / "models" / "nixlm" / "casper-puca-qlora-v6-policy-sft"


def load(base: Path, adapter: Path):
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--output", type=Path, default=Path("/tmp/casper-v6-policy-eval.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = {}
    for row in rows:
        unique[row["messages"][1]["content"]] = row
    tokenizer, model = load(BASE, args.adapter)
    results = []
    for request, row in list(unique.items()):
        reply = generate(tokenizer, model, row["messages"])
        required = [token for token in ("better", "A or B", "2 days", "Sai Neela") if token.casefold() in (row["messages"][2]["content"] or "").casefold()]
        failures = [f"missing expected grounding: {token}" for token in required if token.casefold() not in reply.casefold()]
        if not reply:
            failures.append("empty")
        print(f"{'PASS' if not failures else 'FAIL'} {request}: {reply}", flush=True)
        results.append({"request": request, "reply": reply, "expected": row["messages"][2]["content"], "failures": failures})
    report = {"adapter": str(args.adapter), "passed": sum(not item["failures"] for item in results), "total": len(results), "results": results}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("adapter", "passed", "total")}, indent=2))


if __name__ == "__main__":
    main()
