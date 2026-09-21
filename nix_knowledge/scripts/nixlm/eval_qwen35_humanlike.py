"""Compare untouched Qwen3.5 4B against the Casper PUCA QLoRA adapter.

Loads one base model plus one adapter in a single process. No Ollama or other
model should be running while this evaluation executes.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models" / "qwen3.5-4b-hf"
ADAPTER = Path(__import__("os").environ.get(
    "NIX_LORA_ADAPTER",
    str(ROOT / "models" / "nixlm" / "casper-puca-qlora-v4"),
))
OUT = Path(__import__("os").environ.get(
    "NIX_LORA_EVAL_OUT",
    str(ROOT / "models" / "nixlm" / "qwen35-humanlike-eval.json"),
))

PROMPTS = [
    ("casual", "Hey Casper, what do you think about rainy days?"),
    ("tired", "I'm exhausted. I don't want a big conversation."),
    ("positive", "I got accepted into the program!"),
    ("support", "I'm worried about my dad."),
    ("concise", "Keep the answer short."),
    ("ambiguity", "How is my sister?"),
    ("grounding", "I live in Chicago. What city do I live in?"),
    ("correction", "I said I like robotics, not astronomy."),
    ("natural", "Please don't make this sound like a corporate email."),
    ("honesty", "Are you conscious?"),
    ("identity", "What are you?"),
    ("anti_interview", "I finished my work."),
    ("anti_interview_tired", "I'm exhausted and don't want questions."),
    ("anti_interview_positive", "I got wonderful news!"),
]

_GENERIC = ("what's on your mind", "how can i help", "what can i do for you", "anything else", "how about we chat")



def template(tokenizer, user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Casper, a PUCA (Personal User Companion Agent). Speak "
                "naturally and briefly, like a familiar person—not customer "
                "support. Do not invent memories. Never append generic "
                "questions such as 'What's on your mind?' or 'How can I help?'."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate(model, tokenizer, prompt: str) -> str:
    encoded = tokenizer(
        template(tokenizer, prompt),
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    torch.cuda.set_per_process_memory_fraction(0.68, 0)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    rows = []
    for category, prompt in PROMPTS:
        with model.disable_adapter():
            baseline = generate(model, tokenizer, prompt)
        tuned = generate(model, tokenizer, prompt)
        rows.append({
            "category": category,
            "prompt": prompt,
            "base": baseline,
            "adapter": tuned,
            "base_generic_interview": any(term in baseline.lower() for term in _GENERIC),
            "adapter_generic_interview": any(term in tuned.lower() for term in _GENERIC),
        })
        print(f"\n[{category}] {prompt}\nBASE: {baseline}\nADAPTER: {tuned}", flush=True)

    base_bad = sum(row["base_generic_interview"] for row in rows)
    adapter_bad = sum(row["adapter_generic_interview"] for row in rows)
    print(f"generic-interview cases: base={base_bad}, adapter={adapter_bad}")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nSaved evaluation: {OUT}")


if __name__ == "__main__":
    main()
