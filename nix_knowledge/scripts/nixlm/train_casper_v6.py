"""Casper V6 adapter training entry point.

The command defaults to a dry resource/data check. Training requires the
explicit ``--execute`` flag, a local base checkpoint, an approved JSONL
preference dataset, and a CUDA device. It never downloads models and never
writes over the V5 adapter.

The actual preference optimizer is intentionally imported lazily because the
project's runtime environment does not require TRL for inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "models" / "qwen3.5-4b-hf"
DEFAULT_DATA = ROOT / "models" / "training_data" / "casper_v6_preferences.jsonl"
DEFAULT_OUTPUT = ROOT / "models" / "nixlm" / "casper-puca-qlora-v6"
V5_OUTPUT = ROOT / "models" / "nixlm" / "casper-puca-qlora-v5"
VRAM_FRACTION = 0.68


def read_preferences(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("prompt", "chosen", "rejected"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"line {line_number}: missing non-empty {field}")
            if row["chosen"].strip() == row["rejected"].strip():
                raise ValueError(f"line {line_number}: chosen equals rejected")
            rows.append(row)
    if not rows:
        raise ValueError("preference dataset is empty")
    return rows


def resource_report(base: Path, data: Path, output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "base": str(base),
        "data": str(data),
        "output": str(output),
        "base_present": base.is_dir(),
        "data_present": data.is_file(),
        "output_is_v5": output.resolve() == V5_OUTPUT.resolve(),
        "vram_fraction": VRAM_FRACTION,
        "execute_required": True,
    }
    try:
        import torch
    except ImportError:
        report.update({"torch": False, "cuda": False})
        return report
    report["torch"] = True
    report["cuda"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = props.name
        report["total_vram_gib"] = round(props.total_memory / 2**30, 2)
        report["free_vram_gib"] = round((props.total_memory - torch.cuda.memory_allocated()) / 2**30, 2)
    return report


def validate(base: Path, data: Path, output: Path, *, require_cuda: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = resource_report(base, data, output)
    if report["output_is_v5"]:
        raise ValueError("refusing to write over the active V5 adapter")
    if not report["base_present"]:
        raise FileNotFoundError(f"local base checkpoint missing: {base}")
    if not report["data_present"]:
        raise FileNotFoundError(f"preference dataset missing: {data}")
    rows = read_preferences(data)
    if require_cuda and report.get("torch") and not report.get("cuda"):
        raise RuntimeError("V6 training requires CUDA; no model was loaded")
    if require_cuda and report.get("total_vram_gib", 99) < 7.0:
        raise RuntimeError("V6 training requires an RTX 4060-class 8 GiB device")
    return report, rows


def train(base: Path, rows: list[dict[str, Any]], output: Path, max_steps: int) -> None:
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("V6 execution requires torch, transformers, peft, bitsandbytes, and trl") from exc

    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
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
    ref_model = None
    dataset = [{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]} for row in rows]
    config = DPOConfig(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,
        max_length=768,
        max_prompt_length=512,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max_steps,
        report_to=[],
        bf16=True,
        gradient_checkpointing=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("CASPER_V6_BASE", DEFAULT_BASE)))
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("CASPER_V6_DATA", DEFAULT_DATA)))
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("CASPER_V6_OUTPUT", DEFAULT_OUTPUT)))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--execute", action="store_true", help="Actually load the model and train")
    args = parser.parse_args()
    report, rows = validate(args.base, args.data, args.output, require_cuda=args.execute)
    print(json.dumps({**report, "preference_rows": len(rows), "mode": "execute" if args.execute else "check-only"}, indent=2))
    if args.execute:
        train(args.base, rows, args.output, args.max_steps)


if __name__ == "__main__":
    main()
