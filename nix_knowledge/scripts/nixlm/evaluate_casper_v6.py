"""Run local Casper adapter generation against the V6 behavioral corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "models" / "qwen3.5-4b-hf"
DEFAULT_ADAPTER = ROOT / "models" / "nixlm" / "casper-puca-qlora-v6"


def load(base: Path, adapter: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, request: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Casper, a warm and grounded PUCA. Be natural and concise. "
                "Do not invent memory, ask unnecessary questions, or claim actions "
                "without confirmed results."
            ),
        },
        {"role": "user", "content": request},
    ]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    from pathlib import Path as _Path
    import sys
    sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "nix_core"))
    from casper_v6_eval import CASES, check

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=Path("/tmp/casper-v6-eval.json"))
    args = parser.parse_args()
    tokenizer, model = load(args.base, args.adapter)
    results = []
    for case in CASES:
        reply = generate(tokenizer, model, case.request)
        failures = check(case, reply)
        print(f"[{case.case_id}] {'PASS' if not failures else 'FAIL'}: {reply}", flush=True)
        results.append({"id": case.case_id, "request": case.request, "reply": reply, "failures": failures})
    report = {"adapter": str(args.adapter), "passed": sum(not row["failures"] for row in results), "total": len(results), "results": results}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("adapter", "passed", "total")}, indent=2))


if __name__ == "__main__":
    main()
