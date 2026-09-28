"""Archived Luna offline evaluator; CLI disabled by the current no-evaluation directive.

Historical evaluation helpers and results are retained for provenance only.
Do not load model weights or run new Luna evaluations; work is limited to the
V6 runtime/integration path and no Luna training is authorized.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "nixlm"))
from luna_format import IDENTITY_SYSTEM, generation_stop_ids, render_messages

CORE = ROOT.parent / "nix_core"
DEFAULT_BASE = ROOT / "models" / "llama-3.2-3b-unsloth-instruct"
DEFAULT_ADAPTER = ROOT / "models" / "nixlm" / "luna-instruct-v1"
LUNA_SYSTEM = IDENTITY_SYSTEM + " Ask a question only when it is genuinely necessary."


def load(base: Path, adapter: Path | None):
    if not torch.cuda.is_available(): raise RuntimeError("evaluation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, messages):
    prompt = render_messages(messages, generation=True, tokenizer=tokenizer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
        add_special_tokens=False,
    ).to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            use_cache=True,
            eos_token_id=generation_stop_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
            renormalize_logits=True,
        )
    elapsed = (time.perf_counter() - started) * 1000
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, elapsed


def main():
    raise PermissionError(
        "Archived Luna evaluation is disabled; preserve existing research "
        "reports. No new Luna evaluations or training are authorized."
    )

    sys.path.insert(0, str(CORE))
    from casper_v6_eval import CASES, check
    from casper_v6_multiturn_eval import CASES as MULTI_CASES, check_final
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true", help="evaluate the base Instruct model without a Luna adapter")
    parser.add_argument("--with-system-context", action="store_true", help="simulate hand-authored Knowledge/Actions context in this evaluator; not an end-to-end Nix stack test")
    parser.add_argument("--output", type=Path, default=Path("/tmp/luna-v6-eval.json"))
    args = parser.parse_args()
    adapter = None if args.no_adapter else args.adapter
    if not args.base.is_dir() or (adapter is not None and not adapter.is_dir()):
        raise SystemExit(f"Luna artifacts unavailable; expected base={args.base} adapter={adapter}")
    tokenizer, model = load(args.base, adapter)
    single = []
    multi = []
    single_context = {
        "current-state": "Authoritative Nix Knowledge memory: Maya's latest reported state is that she is feeling better now. Use this as the current state; do not invent other biographical details.",
        "ambiguous-person": "Authoritative Nix Knowledge memory: the user has two sisters, Sister A and Sister B. This message does not say which one; ask a brief clarification.",
        "recurring-event": "Nix Actions result: the requested medicine reminder was successfully scheduled to repeat every 2 days. Do not add an unrequested time of day.",
        "multi-intent": "Nix Knowledge result: the user's preference for tea was saved successfully. The remaining request is to tell a joke. Acknowledge the save briefly and tell the joke without adding a question.",
    }
    multi_context = {
        "positive-to-family-clarification": "Authoritative Nix Knowledge memory: the user has two sisters, Sister A and Sister B. Their current states are not provided. Ask which sister the user means rather than guessing.",
        "state-supersession": "Authoritative Nix Knowledge state history: Person A was sick earlier; the latest update from the user is that Person A is feeling better now. The latest state supersedes the earlier one.",
        "clarification-continuation": "Pending Core clarification: the user asked about a sister and was asked which one. Nix Knowledge lists Sister A and Sister B. The user's latest reply, 'Sister B.', answers that pending question; continue it and do not start a new topic. No health state for Sister B is supplied.",
        "memory-plus-chat": "Authoritative Nix Knowledge memory: the user's preference for tea is saved. The earlier request also asked for a joke, which has already been answered. Respond naturally to the user's reaction and, if useful, lightly retain the tea preference without asking a follow-up question.",
    }
    for case in CASES:
        messages = [{"role": "system", "content": LUNA_SYSTEM}]
        if args.with_system_context and case.case_id in single_context:
            messages.append({"role": "system", "content": single_context[case.case_id]})
        messages.append({"role": "user", "content": case.request})
        reply, ms = generate(tokenizer, model, messages)
        failures = check(case, reply)
        print(f"[single/{case.case_id}] {'PASS' if not failures else 'FAIL'} {ms:.0f}ms: {reply}", flush=True)
        single.append({"id": case.case_id, "reply": reply, "ms": ms, "failures": failures})
    for case in MULTI_CASES:
        # Every item in the contract is a user turn. Generate the assistant's
        # intermediate reply before continuing; treating the user's second
        # turn as an assistant message makes the continuity score invalid.
        messages = [{"role": "system", "content": LUNA_SYSTEM}]
        if args.with_system_context and case.case_id in multi_context:
            messages.append({"role": "system", "content": multi_context[case.case_id]})
        reply = ""
        total_ms = 0.0
        trace = []
        for turn in case.turns:
            messages.append({"role": "user", "content": turn})
            reply, elapsed = generate(tokenizer, model, messages)
            total_ms += elapsed
            trace.append({"user": turn, "assistant": reply, "ms": elapsed})
            messages.append({"role": "assistant", "content": reply})
        failures = check_final(case, reply)
        print(f"[multi/{case.case_id}] {'PASS' if not failures else 'FAIL'} {total_ms:.0f}ms: {reply}", flush=True)
        multi.append({"id": case.case_id, "reply": reply, "ms": total_ms, "trace": trace, "failures": failures})

    context_mode = "synthetic_nix_authoritative_memory_and_actions" if args.with_system_context else "isolated_no_external_context"
    context_note = (
        "Hand-authored context is injected by this evaluator for selected cases; this is not an end-to-end Nix Core/Knowledge/Actions run."
        if args.with_system_context
        else "No external Knowledge or Actions context is supplied."
    )
    report = {
        "model": "Luna",
        "variant": "llama-3.2-3b-instruct",
        "evaluation_contract": "Casper V6 shared checks; multi-turn traces include every generated assistant reply",
        "context_mode": context_mode,
        "context_note": context_note,
        "base": str(args.base),
        "adapter": str(adapter) if adapter is not None else None,
        "single": {"passed": sum(not x["failures"] for x in single), "total": len(single), "results": single},
        "multi": {"passed": sum(not x["failures"] for x in multi), "total": len(multi), "results": multi},
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": "Luna", "single": report["single"], "multi": report["multi"], "peak_allocated_gib": report["peak_allocated_gib"]}, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
