"""Retired Luna Pro v1 research trainer; GPU smoke and training are disabled.

Historical candidate validation and run code are retained for auditability,
but this module must not be used to train or load any model. Luna work is
limited to the V6 runtime and integration path; no Luna training is authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from luna_format import EOT, generation_stop_ids, render_messages

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "models" / "training_data" / "luna_pro_sft_v1_topical_v8"
DEFAULT_TRAIN_OUTPUT = ROOT / "models" / "nixlm" / "luna-pro-v1-topical-v8"
DEFAULT_SMOKE_OUTPUT = ROOT / "models" / "nixlm" / "luna-pro-v1-topical-v8-smoke"
ISOLATED_ENV = REPO_ROOT / ".venvs" / "luna-pro-unsloth"

TRAINING_CONFIG = {
    "max_seq_length": 768,
    "smoke_max_seq_length": 384,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "epochs": 1,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "evaluation_steps": 25,
    "checkpoint_steps": 25,
    "seed": 1307,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "use_rslora": False,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "quantization": "bitsandbytes NF4 4-bit QLoRA; double quantization",
    "optimizer": "bitsandbytes PagedAdamW8bit",
    "loss": "assistant-only causal cross-entropy, token-weighted across accumulation windows",
    "overlength_policy": "drop whole conversation; never truncate a target or trajectory",
}
EXPECTED_GPU_ENV = {
    "unsloth": "2026.9.11",
    "unsloth-zoo": "2026.9.7",
    "torch": "2.12.1+cu132",
    "transformers": "5.5.0",
    "peft": "0.21.0",
    "bitsandbytes": "0.50.2",
}


class CandidateValidationError(ValueError):
    """The candidate artifact or its recorded provenance failed validation."""


class EncodedExample:
    def __init__(self, source: str, input_ids: Sequence[int], labels: Sequence[int]):
        self.source = source
        self.input_ids = tuple(input_ids)
        self.labels = tuple(labels)

    @property
    def supervised_tokens(self) -> int:
        # Causal-LM loss shifts labels by one position; input position 0 is not
        # itself a next-token target.
        return sum(label != -100 for label in self.labels[1:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CandidateValidationError(f"Blank JSONL line in {path.name}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateValidationError(
                    f"Invalid JSON in {path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CandidateValidationError(f"Expected object in {path.name}:{line_number}")
            rows.append(row)
    return rows


def _validate_message_row(row: dict[str, Any], where: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise CandidateValidationError(f"Missing conversation messages in {where}")
    if any(
        not isinstance(message, dict)
        or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        for message in messages
    ):
        raise CandidateValidationError(f"Invalid role or empty content in {where}")
    dialogue = messages[1:] if messages[0]["role"] == "system" else messages
    if len(dialogue) < 2 or dialogue[-1]["role"] != "assistant":
        raise CandidateValidationError(f"Conversation does not end with an assistant in {where}")
    if any(
        message["role"] != ("user" if index % 2 == 0 else "assistant")
        for index, message in enumerate(dialogue)
    ):
        raise CandidateValidationError(f"Non-alternating roles in {where}")


def _validate_report_audits(report: dict[str, Any]) -> None:
    if report.get("status") != "candidate_data_built_training_not_started_review_required":
        raise CandidateValidationError("Candidate report is not an untrained Pro data build")
    if report.get("train_dev_group_overlap_count") != 0:
        raise CandidateValidationError("Train/dev source-group overlap is nonzero")
    for split_name in ("train", "dev"):
        audit = report.get(split_name)
        if not isinstance(audit, dict):
            raise CandidateValidationError(f"Missing {split_name} audit")
        if (
            audit.get("role_error_count") != 0
            or audit.get("empty_target_count") != 0
            or audit.get("benchmark_exact_overlap_count") != 0
            or audit.get("unsupported_action_claim_rows")
            or audit.get("topical_sensitive_domain_rows")
            or audit.get("topical_personal_claim_rows")
            or audit.get("benchmark_near_overlap", {}).get("blocking_near_overlap_count") != 0
        ):
            raise CandidateValidationError(f"{split_name} candidate audit has blocking findings")


def load_candidate_data(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read and verify v8 train/dev/review artifacts without loading model weights."""
    directory = data_dir.resolve()
    report_path = directory / "build_report.json"
    if not report_path.is_file():
        raise CandidateValidationError(f"Missing candidate report: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateValidationError(f"Invalid candidate report: {exc}") from exc
    if not isinstance(report, dict):
        raise CandidateValidationError("Candidate report must be a JSON object")
    _validate_report_audits(report)

    manifest_relative = report.get("manifest")
    if not isinstance(manifest_relative, str):
        raise CandidateValidationError("Candidate report does not identify its manifest")
    manifest_path = (REPO_ROOT / manifest_relative).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != report.get("manifest_sha256"):
        raise CandidateValidationError("Candidate manifest is missing or differs from the recorded hash")
    builder_path = SCRIPT_DIR / "build_luna_pro_sft_v1.py"
    if not builder_path.is_file() or sha256_file(builder_path) != report.get("builder_sha256"):
        raise CandidateValidationError("Candidate builder differs from the recorded hash")

    files = report.get("files")
    if not isinstance(files, dict):
        raise CandidateValidationError("Candidate report has no file manifest")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for key in ("train", "dev", "review_samples"):
        file_info = files.get(key)
        if not isinstance(file_info, dict) or not isinstance(file_info.get("path"), str):
            raise CandidateValidationError(f"Candidate report is missing {key} file metadata")
        path = (directory / file_info["path"]).resolve()
        if path.parent != directory or not path.is_file():
            raise CandidateValidationError(f"Invalid or missing candidate {key} file")
        if sha256_file(path) != file_info.get("sha256"):
            raise CandidateValidationError(f"Candidate {key} SHA-256 does not match its report")
        rows = _read_jsonl(path)
        if len(rows) != file_info.get("rows"):
            raise CandidateValidationError(f"Candidate {key} row count does not match its report")
        for index, row in enumerate(rows):
            _validate_message_row(row, f"{key}[{index}]")
        loaded[key] = rows

    for split_name in ("train", "dev"):
        if len(loaded[split_name]) != report[split_name].get("rows"):
            raise CandidateValidationError(f"Candidate {split_name} audit row count is inconsistent")
    if not loaded["train"] or not loaded["dev"]:
        raise CandidateValidationError("Candidate train and dev splits must both be nonempty")
    base = report.get("base_model")
    if not isinstance(base, dict) or base.get("snapshot_revision") != "006f5dcd1393c3add266de40994ba96225e9689d":
        raise CandidateValidationError("Candidate report does not pin the expected Llama 3.2 base snapshot")
    return report, loaded["train"], loaded["dev"], loaded["review_samples"]


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise CandidateValidationError("Tokenizer returned multiple sequences for one conversation")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def validate_tokenizer_terminators(tokenizer: Any) -> dict[str, Any]:
    if not getattr(tokenizer, "chat_template", None):
        raise CandidateValidationError("The local Instruct tokenizer has no chat template")
    try:
        eot_id = tokenizer.convert_tokens_to_ids(EOT)
        stop_ids = generation_stop_ids(tokenizer)
    except Exception as exc:
        raise CandidateValidationError(f"Could not resolve Llama EOS/EOT IDs: {exc}") from exc
    if not isinstance(eot_id, int) or eot_id < 0 or eot_id not in stop_ids:
        raise CandidateValidationError("Llama EOT is missing from the tokenizer's generation stop IDs")
    return {
        "eot_token": EOT,
        "eot_token_id": eot_id,
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "generation_stop_ids": stop_ids,
    }


def encode_conversation(row: dict[str, Any], tokenizer: Any, *, max_seq_length: int) -> tuple[EncodedExample | None, str | None]:
    """Apply the exact Instruct chat template and label only assistant content + EOT.

    The prefix/end token sequences must be exact prefixes of the full render.
    Any discrepancy fails closed instead of silently shifting labels. Whole
    overlength conversations are dropped rather than partially supervised.
    """
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")
    messages = row["messages"]
    terminators = validate_tokenizer_terminators(tokenizer)
    eot_id = terminators["eot_token_id"]
    try:
        full_text = render_messages(messages, generation=False, tokenizer=tokenizer)
        full_ids = _token_ids(tokenizer, full_text)
    except Exception as exc:
        raise CandidateValidationError(f"Could not render/tokenize conversation: {exc}") from exc
    if not full_ids:
        raise CandidateValidationError("A candidate conversation tokenized to an empty sequence")
    if full_ids[-1] != eot_id:
        raise CandidateValidationError("Full Llama conversation does not end with the expected EOT token")

    labels = [-100] * len(full_ids)
    assistant_indexes = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistant_indexes:
        raise CandidateValidationError("A candidate conversation has no assistant target")
    for index in assistant_indexes:
        prefix_ids = _token_ids(
            tokenizer,
            render_messages(messages[:index], generation=True, tokenizer=tokenizer),
        )
        end_ids = _token_ids(
            tokenizer,
            render_messages(messages[:index + 1], generation=False, tokenizer=tokenizer),
        )
        if full_ids[:len(prefix_ids)] != prefix_ids or full_ids[:len(end_ids)] != end_ids:
            raise CandidateValidationError("Chat-template prefix is not an exact prefix of the full token sequence")
        if len(end_ids) <= len(prefix_ids) or end_ids[-1] != eot_id:
            raise CandidateValidationError("Assistant target is empty or is missing its EOT terminator")
        for position in range(len(prefix_ids), len(end_ids)):
            labels[position] = full_ids[position]

    if not any(label != -100 for label in labels[1:]):
        raise CandidateValidationError("Conversation has no next-token assistant labels")
    if len(full_ids) > max_seq_length:
        return None, "overlength"
    return EncodedExample(str(row.get("source") or "unknown"), full_ids, labels), None


def encode_rows(rows: Sequence[dict[str, Any]], tokenizer: Any, *, max_seq_length: int) -> tuple[list[EncodedExample], dict[str, Any]]:
    examples: list[EncodedExample] = []
    overlength = 0
    all_lengths: list[int] = []
    for row in rows:
        example, dropped_reason = encode_conversation(row, tokenizer, max_seq_length=max_seq_length)
        if example is None:
            if dropped_reason != "overlength":
                raise CandidateValidationError(f"Unknown example drop reason: {dropped_reason}")
            overlength += 1
            continue
        examples.append(example)
        all_lengths.append(len(example.input_ids))
    if not examples:
        raise CandidateValidationError("No candidate conversations fit the selected sequence length")
    return examples, {
        "rows_seen": len(rows),
        "rows_kept": len(examples),
        "whole_conversations_dropped_overlength": overlength,
        "supervised_next_tokens": sum(example.supervised_tokens for example in examples),
        "max_kept_sequence_length": max(all_lengths),
        "max_seq_length": max_seq_length,
        "assistant_only_masks": True,
        "assistant_eot_supervised": True,
        "user_and_system_labels_masked": True,
        "truncation": "none; overlength conversations dropped whole",
    }


def _load_tokenizer(base_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_path, local_files_only=True)


def run_data_check(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """CPU-only candidate hash, role, chat-template, and response-mask preflight."""
    report, train_rows, dev_rows, review_rows = load_candidate_data(data_dir)
    base_path = (REPO_ROOT / report["base_model"]["local_path"]).resolve()
    if not base_path.is_dir():
        raise CandidateValidationError(f"Pinned local base checkpoint is missing: {base_path}")
    tokenizer = _load_tokenizer(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    terminators = validate_tokenizer_terminators(tokenizer)
    _, train_audit = encode_rows(train_rows, tokenizer, max_seq_length=TRAINING_CONFIG["max_seq_length"])
    _, dev_audit = encode_rows(dev_rows, tokenizer, max_seq_length=TRAINING_CONFIG["max_seq_length"])
    return {
        "status": "cpu_data_preflight_passed_review_and_gpu_gates_still_required",
        "data_dir": str(Path(data_dir).resolve()),
        "manifest_sha256": report["manifest_sha256"],
        "builder_sha256": report["builder_sha256"],
        "train_sha256": report["files"]["train"]["sha256"],
        "dev_sha256": report["files"]["dev"]["sha256"],
        "review_sample_rows": len(review_rows),
        "terminators": terminators,
        "train_tokenization": train_audit,
        "dev_tokenization": dev_audit,
        "manual_review_status": report.get("manual_review", {}).get("status"),
        "topical_chat_license_review": "unresolved; local training risk and distribution rights require separate review",
        "model_weights_loaded": False,
        "gpu_used": False,
    }


def validate_output_path(path: Path) -> Path:
    """Permit only a new, non-existing output under the ignored local model tree."""
    resolved = path.resolve()
    allowed_root = (ROOT / "models" / "nixlm").resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Adapter output must be under {allowed_root}") from exc
    if resolved == allowed_root:
        raise ValueError("Adapter output must be a unique child path")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing Pro output: {resolved}")
    return resolved


def validate_frozen_eval_manifest(
    path: Path, data_dir: Path, candidate_report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Require an external, hash-pinned evaluation set before any full run."""
    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise CandidateValidationError(f"Frozen evaluation manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateValidationError(f"Invalid frozen evaluation manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "frozen_before_training":
        raise CandidateValidationError("Evaluation manifest must have status=frozen_before_training")
    cases_relative = manifest.get("cases_path")
    cases_hash = manifest.get("cases_sha256")
    case_count = manifest.get("case_count")
    if not isinstance(cases_relative, str) or not isinstance(cases_hash, str):
        raise CandidateValidationError("Evaluation manifest must pin cases_path and cases_sha256")
    cases_path = (manifest_path.parent / cases_relative).resolve()
    if not cases_path.is_file() or sha256_file(cases_path) != cases_hash:
        raise CandidateValidationError("Frozen evaluation case file is missing or its hash changed")
    if cases_path.is_relative_to(data_dir.resolve()):
        raise CandidateValidationError("Held-out evaluation cases must be outside the SFT candidate directory")
    if candidate_report is not None and cases_path.name == "luna_pro_eval_cases.jsonl":
        expected_cases = SCRIPT_DIR / "luna_pro_eval_cases.jsonl"
        expected_manifest = SCRIPT_DIR / "luna_pro_eval_manifest.json"
        if cases_path != expected_cases.resolve() or manifest_path != expected_manifest.resolve():
            raise CandidateValidationError("Default evaluation paths must resolve to the pinned in-project Pro panel")
        if candidate_report.get("manifest_sha256") == sha256_file(expected_manifest):
            raise CandidateValidationError("SFT manifest must be independent of the held-out Pro evaluation set")
    cases = _read_jsonl(cases_path)
    if not cases or len(cases) != case_count:
        raise CandidateValidationError("Frozen evaluation case count does not match its manifest")
    candidate_user_prompts: set[str] = set()
    if candidate_report is not None:
        for split_name in ("train", "dev"):
            split_path = data_dir.resolve() / candidate_report["files"][split_name]["path"]
            for row in _read_jsonl(split_path):
                prompts = row.get("source_user_turns")
                if not isinstance(prompts, list):
                    prompts = [message["content"] for message in row["messages"] if message["role"] == "user"]
                candidate_user_prompts.update(
                    " ".join(re.findall(r"[a-z0-9]+", str(prompt).casefold()))
                    for prompt in prompts
                )
    ids: set[str] = set()
    for index, case in enumerate(cases):
        if set(case) - {"id", "category", "messages", "review_points"}:
            raise CandidateValidationError(f"Unexpected target/answer fields in evaluation row {index}")
        if set(case) != {"id", "category", "messages", "review_points"}:
            raise CandidateValidationError(f"Missing required review fields in evaluation row {index}")
        if not isinstance(case["review_points"], list) or not case["review_points"]:
            raise CandidateValidationError(f"Missing human-review points in evaluation row {index}")
        case_id = str(case.get("id") or "").strip()
        messages = case.get("messages")
        if not case_id or case_id in ids or not isinstance(messages, list) or not messages:
            raise CandidateValidationError(f"Invalid or duplicate frozen evaluation case at row {index}")
        ids.add(case_id)
        if any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            for message in messages
        ):
            raise CandidateValidationError(f"Invalid message in frozen evaluation case {case_id}")
        if messages[0]["role"] not in {"system", "user"} or messages[-1]["role"] != "user":
            raise CandidateValidationError(f"Evaluation case {case_id} must start with optional system and end with user")
        dialogue = messages[1:] if messages[0]["role"] == "system" else messages
        if any(message["role"] != ("user" if turn_index % 2 == 0 else "assistant")
               for turn_index, message in enumerate(dialogue)):
            raise CandidateValidationError(f"Evaluation case {case_id} has invalid role alternation")
        for prompt in (message["content"] for message in messages if message["role"] == "user"):
            normalized_prompt = " ".join(re.findall(r"[a-z0-9]+", prompt.casefold()))
            if normalized_prompt in candidate_user_prompts:
                raise CandidateValidationError(
                    f"Held-out prompt in {case_id} exactly overlaps the SFT candidate"
                )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cases_path": str(cases_path),
        "cases_sha256": cases_hash,
        "case_count": case_count,
        "status": manifest["status"],
    }


def validate_run_acknowledgements(args: argparse.Namespace, *, smoke: bool) -> None:
    if smoke:
        if not args.confirm_gpu_smoke:
            raise PermissionError("gpu-smoke requires --confirm-gpu-smoke")
        _validate_gpu_environment()
        return
    if not args.confirm_training:
        raise PermissionError("train requires --confirm-training")
    if not args.confirm_reviewed_data:
        raise PermissionError("train requires --confirm-reviewed-data")
    if not args.acknowledge_unresolved_topical_chat_rights_risk:
        raise PermissionError(
            "train requires --acknowledge-unresolved-topical-chat-rights-risk; this is not legal clearance"
        )
    if getattr(args, "frozen_eval_manifest", None) is None:
        raise PermissionError("train requires --frozen-eval-manifest for an external held-out case set")
    _validate_gpu_environment()


def _validate_gpu_environment() -> dict[str, Any]:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"Luna Pro GPU work requires CPython 3.13; got {sys.version.split()[0]}")
    if Path(sys.prefix).resolve() != ISOLATED_ENV.resolve():
        raise RuntimeError(f"Run GPU work only in the isolated environment: {ISOLATED_ENV}")
    versions: dict[str, str] = {}
    for package, expected in EXPECTED_GPU_ENV.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required isolated package is missing: {package}") from exc
        versions[package] = actual
        if actual != expected:
            raise RuntimeError(f"Expected {package}=={expected}; found {actual}")
    return {"python": sys.version.split()[0], "versions": versions}


def _check_gpu_prerequisites(torch: Any, *, smoke: bool) -> tuple[dict[str, Any], Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Luna Pro GPU command requires CUDA")
    environment = _validate_gpu_environment()
    if torch.version.cuda != "13.2":
        raise RuntimeError(f"Expected CUDA-enabled Torch 13.2 build; found {torch.version.cuda}")
    environment.update({
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    })
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Luna Pro is configured for one GPU and refuses multi-GPU execution")
    if os.environ.get("NIX_CORE_USE_KNOWLEDGE_MODEL_GATE") == "1":
        raise RuntimeError("Disable the Qwen Knowledge gate before any Luna GPU run")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < 6 * 1024**3:
        raise RuntimeError(
            f"At least 6 GiB of free VRAM is required before model load; found {free_bytes / 1024**3:.2f} GiB"
        )
    environment["free_vram_gib_before_load"] = round(free_bytes / 1024**3, 3)
    environment["total_vram_gib"] = round(total_bytes / 1024**3, 3)
    environment["smoke_mode"] = smoke
    return environment, torch.device("cuda:0")


def _load_unsloth_model(
    base_path: Path,
    *,
    max_seq_length: int,
    seed: int,
    attach_adapter: bool = True,
) -> tuple[Any, Any, Any]:
    import torch
    from unsloth import FastLanguageModel

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_path),
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        device_map={"": 0},
        use_gradient_checkpointing="unsloth" if attach_adapter else False,
        local_files_only=True,
        trust_remote_code=False,
        random_state=seed,
        max_lora_rank=TRAINING_CONFIG["lora_rank"],
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not attach_adapter:
        model.config.use_cache = True
        return model, tokenizer, FastLanguageModel
    model.config.use_cache = False
    model = FastLanguageModel.get_peft_model(
        model,
        r=TRAINING_CONFIG["lora_rank"],
        target_modules=TRAINING_CONFIG["target_modules"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=TRAINING_CONFIG["use_rslora"],
        loftq_config=None,
    )
    return model, tokenizer, FastLanguageModel


def validate_existing_adapter_path(path: Path) -> Path:
    """Resolve an existing local read-only PEFT adapter under Luna's model tree."""
    resolved = path.resolve()
    allowed_root = (ROOT / "models" / "nixlm").resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Baseline adapter must be under {allowed_root}") from exc
    if not resolved.is_dir() or not (resolved / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Expected a local PEFT adapter at {resolved}")
    if not any((resolved / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Adapter weights are missing from {resolved}")
    return resolved


def _load_eval_adapter(
    base_path: Path, adapter_path: Path, *, max_seq_length: int, seed: int
) -> tuple[Any, Any, Any]:
    if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Baseline/evaluation adapter is missing or invalid: {adapter_path}")
    from peft import PeftModel
    model, tokenizer, FastLanguageModel = _load_unsloth_model(
        base_path,
        max_seq_length=max_seq_length,
        seed=seed,
        attach_adapter=False,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    return model, tokenizer, FastLanguageModel


def _generate_eval_turn(
    model: Any,
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    device: Any,
    max_seq_length: int,
    torch: Any,
) -> tuple[str, dict[str, Any]]:
    prompt = render_messages(messages, generation=True, tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    max_new_tokens = min(128, max_seq_length - prompt_tokens)
    if max_new_tokens < 1:
        raise CandidateValidationError(
            f"Frozen evaluation prompt fills the {max_seq_length}-token context; no generation room remains"
        )
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=generation_stop_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - started
    answer_ids = generated[0, prompt_tokens:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(answer_ids.shape[-1]),
        "generation_seconds": round(elapsed, 3),
        "response": answer,
    }


def evaluate_case_panel(
    model: Any,
    tokenizer: Any,
    cases: Sequence[dict[str, Any]],
    *,
    device: Any,
    max_seq_length: int,
    torch: Any,
) -> list[dict[str, Any]]:
    """Generate paired-review traces; intentionally produces no automatic pass score."""
    records: list[dict[str, Any]] = []
    for case in cases:
        history: list[dict[str, str]] = []
        trace: list[dict[str, Any]] = []
        for message in case["messages"]:
            if message["role"] == "user":
                history.append(message)
                answer, stats = _generate_eval_turn(
                    model,
                    tokenizer,
                    history,
                    device=device,
                    max_seq_length=max_seq_length,
                    torch=torch,
                )
                trace.append({"user": message["content"], **stats})
                history.append({"role": "assistant", "content": answer})
            else:
                history.append(message)
        records.append({
            "id": case["id"],
            "category": case["category"],
            "trace": trace,
            "review_points": case["review_points"],
        })
    return records


def _collate(examples: Sequence[EncodedExample], pad_token_id: int, device: Any, torch: Any) -> tuple[Any, Any, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(example.input_ids) for example in examples)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for example in examples:
        padding = length - len(example.input_ids)
        input_ids.append(list(example.input_ids) + [pad_token_id] * padding)
        labels.append(list(example.labels) + [-100] * padding)
        attention_mask.append([1] * len(example.input_ids) + [0] * padding)
    return tuple(
        torch.tensor(values, dtype=torch.long, device=device)
        for values in (input_ids, labels, attention_mask)
    )


def _batches(examples: Sequence[EncodedExample], batch_size: int) -> list[list[EncodedExample]]:
    return [list(examples[index:index + batch_size]) for index in range(0, len(examples), batch_size)]


def _token_weighted_step(
    model: Any,
    optimizer: Any,
    window: Sequence[Sequence[EncodedExample]],
    *,
    pad_token_id: int,
    device: Any,
    max_grad_norm: float,
    torch: Any,
    functional: Any,
) -> tuple[float, int]:
    window_tokens = sum(example.supervised_tokens for batch in window for example in batch)
    if window_tokens < 1:
        raise RuntimeError("Gradient accumulation window has no supervised next-token labels")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in window:
        input_ids, labels, attention_mask = _collate(batch, pad_token_id, device, torch)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        summed_loss = functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(summed_loss.detach().item())
        (summed_loss / window_tokens).backward()
        del outputs, logits, shifted_logits, shifted_labels, summed_loss
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_grad_norm,
    )
    optimizer.step()
    return total_loss / window_tokens, window_tokens


def _evaluate(
    model: Any,
    examples: Sequence[EncodedExample],
    *,
    pad_token_id: int,
    device: Any,
    torch: Any,
    functional: Any,
    FastLanguageModel: Any,
) -> float:
    FastLanguageModel.for_inference(model)
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in _batches(examples, TRAINING_CONFIG["micro_batch_size"]):
            input_ids, labels, attention_mask = _collate(batch, pad_token_id, device, torch)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            token_count = int((shifted_labels != -100).sum().item())
            loss_sum = functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss_sum.item())
            total_tokens += token_count
            del outputs, logits, shifted_logits, shifted_labels, loss_sum
    FastLanguageModel.for_training(model)
    if total_tokens == 0:
        raise RuntimeError("Development split has no supervised assistant tokens")
    return total_loss / total_tokens


def _read_frozen_eval_cases(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Re-read the exact frozen evaluation panel after training before scoring."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_path = (manifest_path.resolve().parent / manifest["cases_path"]).resolve()
    if sha256_file(cases_path) != manifest["cases_sha256"]:
        raise CandidateValidationError("Frozen evaluation cases changed after preflight")
    cases = _read_jsonl(cases_path)
    if len(cases) != manifest["case_count"]:
        raise CandidateValidationError("Frozen evaluation row count changed after preflight")
    return manifest, cases


def _save_adapter(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    allow_checkpoint_dirs: bool = False,
) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Refusing to overwrite non-directory adapter output: {output_dir}")
        children = list(output_dir.iterdir())
        if allow_checkpoint_dirs:
            metadata_path = output_dir / "training_metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FileExistsError(
                    f"Refusing final adapter save without this run's metadata: {output_dir}"
                ) from exc
            if metadata.get("status") != "started":
                raise FileExistsError(
                    f"Refusing final adapter save for a non-active run: {output_dir}"
                )
        for child in children:
            if child.name == "training_metadata.json" and child.is_file():
                continue
            if allow_checkpoint_dirs and re.fullmatch(r"checkpoint-\d{4}", child.name) and child.is_dir():
                has_adapter_config = (child / "adapter_config.json").is_file()
                has_adapter_weights = any(
                    (child / name).is_file()
                    for name in ("adapter_model.safetensors", "adapter_model.bin")
                )
                if has_adapter_config and has_adapter_weights:
                    continue
            raise FileExistsError(f"Refusing to overwrite adapter/checkpoint: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "training_metadata.json"
    }


def _write_metadata(output_dir: Path, record: dict[str, Any]) -> None:
    (output_dir / "training_metadata.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_gpu_command(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    """Former Pro GPU smoke/train entry point; permanently disabled by policy."""
    del args, smoke
    raise PermissionError(
        "Luna Pro v1 is retired. Its GPU smoke and training commands are "
        "disabled; continue only with Luna V6 runtime/integration work. "
        "No Luna training is authorized."
    )

    validate_run_acknowledgements(args, smoke=smoke)
    output_dir = validate_output_path(args.output_dir)
    candidate_report, train_rows, dev_rows, review_rows = load_candidate_data(args.data_dir)
    if smoke:
        frozen_eval = None
        frozen_eval_manifest = None
        frozen_eval_cases = []
        baseline_adapter_path = None
        baseline_traces = []
    else:
        frozen_eval = validate_frozen_eval_manifest(
            args.frozen_eval_manifest, args.data_dir, candidate_report
        )
        frozen_eval_manifest, frozen_eval_cases = _read_frozen_eval_cases(args.frozen_eval_manifest)
        baseline_adapter_path = validate_existing_adapter_path(args.baseline_adapter)
        baseline_traces = []
    import torch
    import bitsandbytes as bnb
    import torch.nn.functional as functional

    environment, device = _check_gpu_prerequisites(torch, smoke=smoke)
    base_path = (REPO_ROOT / candidate_report["base_model"]["local_path"]).resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Pinned local base checkpoint is missing: {base_path}")
    max_seq_length = (
        TRAINING_CONFIG["smoke_max_seq_length"]
        if smoke
        else args.max_seq_length
    )
    if not smoke:
        eval_model, eval_tokenizer, EvalFastLanguageModel = _load_eval_adapter(
            base_path,
            baseline_adapter_path,
            max_seq_length=max_seq_length,
            seed=TRAINING_CONFIG["seed"],
        )
        baseline_traces = evaluate_case_panel(
            eval_model,
            eval_tokenizer,
            frozen_eval_cases,
            device=device,
            max_seq_length=max_seq_length,
            torch=torch,
        )
        del eval_model, eval_tokenizer, EvalFastLanguageModel
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    model, tokenizer, FastLanguageModel = _load_unsloth_model(
        base_path, max_seq_length=max_seq_length, seed=TRAINING_CONFIG["seed"]
    )
    terminators = validate_tokenizer_terminators(tokenizer)
    train_examples, train_audit = encode_rows(train_rows, tokenizer, max_seq_length=max_seq_length)
    if smoke:
        authored_examples = [item for item in train_examples if item.source == "project_authored"]
        if not authored_examples:
            raise CandidateValidationError("No project-authored row fits the bounded GPU smoke length")
        train_examples = authored_examples[:1]
        dev_examples: list[EncodedExample] = []
        dev_audit = {"rows_kept_for_eval": 0, "smoke_uses_project_authored_only": True}
    else:
        dev_examples, dev_audit = encode_rows(dev_rows, tokenizer, max_seq_length=max_seq_length)
    if tokenizer.pad_token_id is None:
        raise CandidateValidationError("Tokenizer has no pad token after mapping pad to EOS")

    run_record: dict[str, Any] = {
        "status": "started",
        "mode": "bounded_project_authored_smoke" if smoke else "single_pass_training",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_label": "Luna Pro",
        "distribution_identifier": "Llama Luna Pro",
        "base_model": candidate_report["base_model"],
        "base_path": str(base_path),
        "data_dir": str(Path(args.data_dir).resolve()),
        "manifest_sha256": candidate_report["manifest_sha256"],
        "builder_sha256": candidate_report["builder_sha256"],
        "train_sha256": candidate_report["files"]["train"]["sha256"],
        "dev_sha256": candidate_report["files"]["dev"]["sha256"],
        "review_sample_sha256": candidate_report["files"]["review_samples"]["sha256"],
        "frozen_evaluation_set": frozen_eval,
        "baseline_adapter": str(baseline_adapter_path) if baseline_adapter_path else None,
        "baseline_adapter_files_sha256": (
            _artifact_hashes(baseline_adapter_path) if baseline_adapter_path else None
        ),
        "baseline_frozen_evaluation_traces": baseline_traces,
        "environment": environment,
        "tokenizer_terminators": terminators,
        "train_tokenization": train_audit,
        "dev_tokenization": dev_audit,
        "frozen_eval_case_count": len(frozen_eval_cases),
        "training_config": {
            **TRAINING_CONFIG,
            "max_seq_length": max_seq_length,
            "optimizer": "bitsandbytes PagedAdamW8bit",
        },
        "data_review_confirmed_by_command_flag": bool(getattr(args, "confirm_reviewed_data", False)),
        "unresolved_topical_chat_rights": "Not legally cleared; local training is an acknowledged unresolved risk and no distribution is authorized.",
        "model_weights_loaded_after_explicit_gpu_confirmation": True,
    }
    output_dir.mkdir(parents=False, exist_ok=False)
    _write_metadata(output_dir, run_record)

    try:
        train_batches = _batches(train_examples, TRAINING_CONFIG["micro_batch_size"])
        accumulation_steps = 1 if smoke else TRAINING_CONFIG["gradient_accumulation_steps"]
        total_steps = 1 if smoke else math.ceil(len(train_batches) / accumulation_steps) * TRAINING_CONFIG["epochs"]
        warmup_steps = int(total_steps * TRAINING_CONFIG["warmup_ratio"])
        optimizer = bnb.optim.PagedAdamW8bit(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=0.0,
        )

        def lr_scale(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1, step + 1) / warmup_steps
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        history: list[dict[str, Any]] = []
        optimizer_step = 0
        for epoch in range(1 if smoke else TRAINING_CONFIG["epochs"]):
            ordered = list(train_batches)
            if not smoke:
                random.Random(TRAINING_CONFIG["seed"] + epoch).shuffle(ordered)
            for start in range(0, len(ordered), accumulation_steps):
                window = ordered[start:start + accumulation_steps]
                mean_loss, supervised_tokens = _token_weighted_step(
                    model,
                    optimizer,
                    window,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
                    torch=torch,
                    functional=functional,
                )
                scheduler.step()
                optimizer_step += 1
                record = {
                    "epoch": epoch + 1,
                    "step": optimizer_step,
                    "loss": mean_loss,
                    "supervised_tokens": supervised_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                if not smoke and (optimizer_step % TRAINING_CONFIG["evaluation_steps"] == 0 or optimizer_step == total_steps):
                    record["dev_token_weighted_loss"] = _evaluate(
                        model,
                        dev_examples,
                        pad_token_id=tokenizer.pad_token_id,
                        device=device,
                        torch=torch,
                        functional=functional,
                        FastLanguageModel=FastLanguageModel,
                    )
                history.append(record)
                if not smoke and optimizer_step % TRAINING_CONFIG["checkpoint_steps"] == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{optimizer_step:04d}"
                    if checkpoint_dir.exists():
                        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
                    _save_adapter(model, tokenizer, checkpoint_dir)
                if smoke or optimizer_step >= total_steps:
                    break
            if smoke:
                break

        _save_adapter(model, tokenizer, output_dir, allow_checkpoint_dirs=not smoke)
        run_record["history"] = history
        run_record["optimizer_steps"] = optimizer_step
        run_record["adapter_artifacts_sha256"] = _artifact_hashes(output_dir)
        del optimizer, scheduler, model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        if smoke:
            from unsloth import FastLanguageModel as ReloadFastLanguageModel
            from peft import PeftModel

            reload_base, reload_tokenizer, _ = _load_unsloth_model(
                base_path,
                max_seq_length=max_seq_length,
                seed=TRAINING_CONFIG["seed"],
                attach_adapter=False,
            )
            reloaded = PeftModel.from_pretrained(
                reload_base, str(output_dir), is_trainable=False, local_files_only=True
            )
            ReloadFastLanguageModel.for_inference(reloaded)
            smoke_item = train_examples[0]
            smoke_ids, _, smoke_attention = _collate(
                [smoke_item], reload_tokenizer.pad_token_id, device, torch
            )
            with torch.inference_mode():
                smoke_output = reloaded(input_ids=smoke_ids, attention_mask=smoke_attention, use_cache=False)
                if not bool(torch.isfinite(smoke_output.logits).all().item()):
                    raise RuntimeError("Reloaded adapter produced non-finite logits")
            del reload_base, reloaded, smoke_output, smoke_ids, smoke_attention
            torch.cuda.empty_cache()
            gc.collect()
            run_record["save_reload_forward_pass"] = "passed"
            run_record["status"] = "bounded_smoke_passed"
        else:
            scored_model, scored_tokenizer, ScoringFastLanguageModel = _load_eval_adapter(
                base_path,
                output_dir,
                max_seq_length=max_seq_length,
                seed=TRAINING_CONFIG["seed"],
            )
            run_record["candidate_frozen_evaluation_traces"] = evaluate_case_panel(
                scored_model,
                scored_tokenizer,
                frozen_eval_cases,
                device=device,
                max_seq_length=max_seq_length,
                torch=torch,
            )
            del scored_model, scored_tokenizer, ScoringFastLanguageModel
            torch.cuda.empty_cache()
            gc.collect()
            run_record["status"] = "training_completed_review_required"
        run_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_record["peak_allocated_vram_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        _write_metadata(output_dir, run_record)
        return run_record
    except Exception as exc:
        run_record["status"] = "failed"
        run_record["failure"] = f"{type(exc).__name__}: {exc}"
        run_record["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_metadata(output_dir, run_record)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-data", help="CPU-only local data preflight; does not train or load model weights")
    check.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    smoke = subparsers.add_parser("gpu-smoke", help="retired and disabled; retained for historical research only")
    smoke.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_SMOKE_OUTPUT)
    smoke.add_argument("--confirm-gpu-smoke", action="store_true")
    train = subparsers.add_parser("train", help="retired and disabled; no Luna training is authorized")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    train.add_argument(
        "--max-seq-length",
        type=int,
        choices=(TRAINING_CONFIG["smoke_max_seq_length"], TRAINING_CONFIG["max_seq_length"]),
        default=TRAINING_CONFIG["max_seq_length"],
        help="Archived training setting retained for argument compatibility; this command is disabled.",
    )
    train.add_argument("--confirm-training", action="store_true")
    train.add_argument("--confirm-reviewed-data", action="store_true")
    train.add_argument("--acknowledge-unresolved-topical-chat-rights-risk", action="store_true")
    train.add_argument(
        "--frozen-eval-manifest",
        type=Path,
        default=SCRIPT_DIR / "luna_pro_eval_manifest.json",
    )
    train.add_argument(
        "--baseline-adapter",
        type=Path,
        default=ROOT / "models" / "nixlm" / "luna-instruct-v1",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-data":
        result = run_data_check(args.data_dir)
    elif args.command == "gpu-smoke":
        result = run_gpu_command(args, smoke=True)
    else:
        result = run_gpu_command(args, smoke=False)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
