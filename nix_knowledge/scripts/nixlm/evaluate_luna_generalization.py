"""Archived Luna generalization evaluator; CLI disabled by current policy.

Historical probes and evaluation helpers are retained for provenance only.
Do not load model weights or run new Luna evaluations; work is limited to the
V6 runtime/integration path and no Luna training is authorized.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from luna_generalization_probes import (
    PROBES,
    audit_probe_overlap,
    default_training_snapshots,
    iter_probe_turns,
)

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT.parent / "nix_core"
DEFAULT_BASE = ROOT / "models" / "llama-3.2-3b-unsloth-instruct"
ADAPTERS = {
    "luna-instruct-v1": ROOT / "models" / "nixlm" / "luna-instruct-v1",
    "luna-v6-contextual-v1/checkpoint-60": ROOT / "models" / "nixlm" / "luna-v6-contextual-v1" / "checkpoint-60",
}
SYSTEM_PROMPT = (
    "You are Luna, an independent conversation model used inside Nix's PUCA "
    "system. Nix is the overall personal-companion system. Casper is Nix's "
    "separate user-facing PUCA identity, and Luna is not Casper. Be natural, "
    "concise, and answer the latest user message directly. Do not invent "
    "memories, actions, relationships, health facts, or a biography. Use "
    "fictional test context only when relevant, and follow newer explicit "
    "conversation corrections over older context."
)


def _load(base: Path, adapter: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("Luna generalization evaluation requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant,
            device_map={"": 0},
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, messages: list[dict[str, str]]) -> tuple[str, float]:
    from luna_format import generation_stop_ids, render_messages

    prompt = render_messages(messages, generation=True, tokenizer=tokenizer)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
    ).to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
            eos_token_id=generation_stop_ids(tokenizer),
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            renormalize_logits=True,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return answer, elapsed_ms


def _run_probe(tokenizer, model, probe: dict[str, Any], use_context: bool) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    injected_contexts: list[dict[str, Any]] = []
    initial_context = probe.get("initial_context")
    if use_context and initial_context:
        messages.append({
            "role": "system",
            "content": initial_context,
        })
        injected_contexts.append({
            "before_turn": 0,
            "kind": "fictional_initial_context",
            "content": initial_context,
        })

    trace = []
    total_ms = 0.0
    for turn_index, turn in enumerate(probe["turns"]):
        dynamic_context = turn.get("context_before") if use_context else None
        if dynamic_context:
            messages.append({"role": "system", "content": dynamic_context})
            injected_contexts.append({
                "before_turn": turn_index,
                "kind": "fictional_simulated_result_arrival",
                "content": dynamic_context,
            })
        user_text = str(turn["user"])
        messages.append({"role": "user", "content": user_text})
        answer, elapsed_ms = _generate(tokenizer, model, messages)
        total_ms += elapsed_ms
        trace.append({
            "turn_index": turn_index,
            "user": user_text,
            "assistant": answer,
            "generation_ms": round(elapsed_ms, 1),
            "review_points": list(turn.get("review_points", ())),
        })
        messages.append({"role": "assistant", "content": answer})

    return {
        "probe_id": probe["id"],
        "category": probe["category"],
        "context_condition": "injected_fictional_context" if use_context else "no_external_context",
        "injected_contexts": injected_contexts,
        "trace": trace,
        "total_generation_ms": round(total_ms, 1),
    }


def main() -> None:
    raise PermissionError(
        "Archived Luna evaluation is disabled; preserve existing research "
        "reports. No new Luna evaluations or training are authorized."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--baseline-adapter", type=Path, default=ADAPTERS["luna-instruct-v1"])
    parser.add_argument("--candidate-adapter", type=Path, default=ADAPTERS["luna-v6-contextual-v1/checkpoint-60"])
    parser.add_argument("--training-v2", type=Path, default=default_training_snapshots()["contextual_adapter_training_v2"])
    parser.add_argument("--training-v3", type=Path, default=default_training_snapshots()["corrected_untrained_v3"])
    parser.add_argument("--baseline-mix", type=Path, default=default_training_snapshots()["instruct_baseline_mix_v4"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-exact-overlap", action="store_true", help="diagnostic only; exact overlap blocks execution by default")
    args = parser.parse_args()

    if not args.base.is_dir():
        raise SystemExit(f"Luna Instruct base is unavailable: {args.base}")
    if not args.baseline_adapter.is_dir():
        raise SystemExit(f"Baseline adapter is unavailable: {args.baseline_adapter}")
    if not args.candidate_adapter.is_dir():
        raise SystemExit(f"Contextual checkpoint is unavailable: {args.candidate_adapter}")

    snapshots = {
        "instruct_baseline_mix_v4": args.baseline_mix,
        "contextual_adapter_training_v2": args.training_v2,
        "corrected_untrained_v3": args.training_v3,
    }
    audit = audit_probe_overlap(PROBES, snapshots)
    if audit["blocking_exact_overlap_count"] and not args.allow_exact_overlap:
        raise SystemExit(
            f"Probe panel has {audit['blocking_exact_overlap_count']} exact overlaps "
            "with training or the shared benchmark; inspect the audit before running. "
            "For explicit diagnostic-only use, pass --allow-exact-overlap."
        )

    # Freeze a complete audit before loading a multi-GiB model. This is a
    # deliberate evaluation artifact and has a unique caller-specified path.
    report: dict[str, Any] = {
        "model_family": "Luna / Llama 3.2 3B Instruct",
        "evaluation": "paired fictional held-out context-generalization diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context_scope": (
            "Injected context consists of hand-authored fictional test fixtures. "
            "It is not an end-to-end Core, Knowledge, Actions, scheduler, or API test."
        ),
        "limitations": (
            "Small hand-authored diagnostic, not a statistically robust proof of generalization. "
            "Exact overlap is blocked; lexical near matches are only surfaced for human review. "
            "No automatic pass score is used; inspect every trace and compare matched conditions."
        ),
        "system_prompt": SYSTEM_PROMPT,
        "probe_count": len(PROBES),
        "probe_turn_count": sum(len(probe["turns"]) for probe in PROBES),
        "overlap_audit": audit,
        "candidate_adapter": str(args.candidate_adapter),
        "baseline_adapter": str(args.baseline_adapter),
        "base": str(args.base),
        "results": [],
    }
    if audit["blocking_exact_overlap_count"]:
        report["overlap_override"] = True
    report_path = args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # This is a new output file only; refuse to overwrite old reports.
    if report_path.exists():
        raise SystemExit(f"Refusing to overwrite an existing report: {report_path}")
    audit_path = report_path.with_suffix(report_path.suffix + ".overlap.json")
    if audit_path.exists():
        raise SystemExit(f"Refusing to overwrite an existing overlap audit: {audit_path}")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    conditions = (
        ("baseline", args.baseline_adapter),
        ("contextual_candidate", args.candidate_adapter),
    )
    for model_name, adapter_path in conditions:
        tokenizer, model = _load(args.base, adapter_path)
        try:
            torch.cuda.reset_peak_memory_stats()
            for use_context in (False, True):
                mode = "with_context" if use_context else "no_context"
                for probe in PROBES:
                    result = _run_probe(tokenizer, model, probe, use_context)
                    result["model_condition"] = model_name
                    result["adapter"] = str(adapter_path)
                    report["results"].append(result)
                    final = result["trace"][-1]["assistant"] if result["trace"] else ""
                    print(
                        f"[{model_name}/{mode}/{probe['id']}] "
                        f"{result['total_generation_ms']:.0f}ms: {final}",
                        flush=True,
                    )
            report.setdefault("peak_allocated_gib_by_model", {})[model_name] = round(
                torch.cuda.max_memory_allocated() / 2**30, 3
            )
        finally:
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(report_path),
        "overlap_audit": str(audit_path),
        "probe_count": report["probe_count"],
        "probe_turn_count": report["probe_turn_count"],
        "result_count": len(report["results"]),
        "peak_allocated_gib_by_model": report.get("peak_allocated_gib_by_model"),
    }, indent=2))


if __name__ == "__main__":
    main()
