"""Evaluate final Casper V6 on real multi-turn conversations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models" / "qwen3.5-4b-hf"
ADAPTER = ROOT / "models" / "nixlm" / "casper-puca-qlora-v6-final"


def load():
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, turns):
    messages = [{"role": "system", "content": "You are Casper, a natural and concise PUCA final verbalizer. Follow current state over history, ask only necessary questions, and do not invent memories."}]
    for index, turn in enumerate(turns):
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": turn})
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    global ADAPTER
    sys.path.insert(0, str(ROOT.parent / "nix_core"))
    from casper_v6_multiturn_eval import CASES, check_final
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--output", type=Path, default=Path("/tmp/casper-v6-final-multiturn.json"))
    args = parser.parse_args()
    ADAPTER = args.adapter
    tokenizer, model = load()
    results = []
    for case in CASES:
        reply = generate(tokenizer, model, case.turns)
        failures = check_final(case, reply)
        print(f"{'PASS' if not failures else 'FAIL'} {case.case_id}: {reply}", flush=True)
        results.append({"id": case.case_id, "reply": reply, "failures": failures})
    report = {"adapter": str(ADAPTER), "passed": sum(not item["failures"] for item in results), "total": len(results), "results": results}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("adapter", "passed", "total")}, indent=2))


if __name__ == "__main__":
    main()
