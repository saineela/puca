"""Luna artifact metadata and gated model-generation policy.

This module only discovers local artifacts; it never loads weights or CUDA.
Official Luna is the official assistant. Luna Pro v1 is retired from
runtime use; its research is retained separately. Luna V7 remains gated.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LUNA_BASE_PATH = Path(os.environ.get(
    "LUNA_BASE_MODEL_PATH",
    ROOT / "nix_knowledge" / "models" / "llama-3.2-3b-unsloth-instruct",
))
LUNA_ADAPTER_PATH = Path(os.environ.get(
    "LUNA_ADAPTER_PATH",
    ROOT / "nix_knowledge" / "models" / "nixlm" / "luna-instruct-v1",
))
LUNA_V6_ADAPTER_PATH = Path(os.environ.get(
    "LUNA_V6_ADAPTER_PATH",
    ROOT / "nix_knowledge" / "models" / "nixlm" / "luna-v6-contextual-v1" / "checkpoint-60",
))
LUNA_NAME = "Luna"
LUNA_V6_MODEL_ID = "luna-v6-contextual-v1"
# Retain only the historical ID so stale API clients receive an explicit
# retirement response. No adapter path or inference loader is defined for it.
RETIRED_LUNA_MODEL_IDS = frozenset({"luna-pro-v1-topical-v8-384-retry2"})

# V7 has its own artifact path and an explicit final-answer-only contract.
# It stays out of model registries until the gated V7 plan is complete.
LUNA_V7_MODEL_ID = "luna-v7-contextual-v1"
LUNA_V7_ADAPTER_PATH = Path(os.environ.get(
    "LUNA_V7_ADAPTER_PATH",
    ROOT / "nix_knowledge" / "models" / "nixlm" / LUNA_V7_MODEL_ID,
))
LUNA_V7_THINKING_ENABLED = False
LUNA_V7_NO_THINKING_INSTRUCTION = (
    "Do not produce or reveal chain-of-thought, private reasoning, or "
    "intermediate analysis. Return only the direct final answer."
)
LUNA_V7_RUNTIME_PLAN: dict[str, Any] = {
    "model_id": LUNA_V7_MODEL_ID,
    "adapter_path": str(LUNA_V7_ADAPTER_PATH),
    "registered": False,
    "trained": False,
    "training_authorized": False,
    "thinking_enabled": LUNA_V7_THINKING_ENABLED,
    "output_policy": "final_answer_only",
    "status": "gated_not_trained",
}


def system_prompt_for_model(model_id: str, base_prompt: str) -> str:
    """Apply V7's explicit no-thinking policy; leave V6 and Pro unchanged."""
    if model_id != LUNA_V7_MODEL_ID:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{LUNA_V7_NO_THINKING_INSTRUCTION}"


def strip_v7_reasoning_markup(model_id: str, text: object) -> str:
    """Drop tagged private reasoning if a future V7 checkpoint emits it."""
    value = str(text or "")
    if model_id == LUNA_V7_MODEL_ID:
        value = re.sub(
            r"<think>.*?</think>", "", value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        value = re.sub(r"<think>.*$", "", value, flags=re.IGNORECASE | re.DOTALL)
    return value


LUNA_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": LUNA_V6_MODEL_ID,
        "label": "Official Luna",
        "adapter_path": LUNA_V6_ADAPTER_PATH,
        "release_status": "official_assistant_default",
        "official": True,
        "experimental": False,
        "production_default": True,
        "quality_note": "Official local Luna model selected from the model settings.",
    },
)


def _has_peft_adapter(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and any(
            (path / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
    )


def available_models(
    active_model: str | None = None,
    *,
    base_path: Path | None = None,
    adapter_paths: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """Describe the official V6 artifact without loading it."""
    base_path = base_path or LUNA_BASE_PATH
    adapter_paths = adapter_paths or {
        LUNA_V6_MODEL_ID: LUNA_V6_ADAPTER_PATH,
    }
    models: list[dict[str, Any]] = []
    for spec in LUNA_MODEL_SPECS:
        adapter_path = adapter_paths[spec["id"]]
        models.append({
            "id": spec["id"],
            "label": spec["label"],
            "beta": spec["experimental"],
            "experimental": spec["experimental"],
            "official": spec["official"],
            "present": base_path.is_dir() and _has_peft_adapter(adapter_path),
            "active": active_model == spec["id"],
            "production_default": spec["production_default"],
            "release_status": spec["release_status"],
            "quality_note": spec["quality_note"],
        })
    return models


LUNA_V6_MODEL_IDS = frozenset({LUNA_V6_MODEL_ID})
LUNA_MODEL_IDS = LUNA_V6_MODEL_IDS


def runtime_status() -> dict[str, Any]:
    """Describe Luna V6 and the gated V7 plan without forcing a load."""
    models = available_models()
    return {
        "name": LUNA_NAME,
        "model_id": LUNA_V6_MODEL_ID,
        "label": "Official Luna",
        "family": "Llama 3.2 3B Instruct",
        "role": "Official Luna, the Nix PUCA conversation model",
        "base_path": str(LUNA_BASE_PATH),
        "adapter_path": str(LUNA_V6_ADAPTER_PATH),
        "baseline_adapter_path": str(LUNA_ADAPTER_PATH),
        "base_present": LUNA_BASE_PATH.is_dir(),
        "adapter_present": next(
            model["present"] for model in models if model["id"] == LUNA_V6_MODEL_ID
        ),
        "available_models": models,
        "loaded": False,
        "thinking": False,
        "v7_plan": dict(LUNA_V7_RUNTIME_PLAN),
        "release_status": "official_assistant_default",
        "retired_models": [
            {"id": model_id, "status": "retired_not_loadable"}
            for model_id in sorted(RETIRED_LUNA_MODEL_IDS)
        ],
        "shared_v6_suite": {
            "baseline_instruct_v1": {
                "isolated": {"single": "5/9", "multi": "1/5"},
                "synthetic_context": {"single": "7/9", "multi": "2/5"},
            },
            "contextual_candidate_checkpoint_60": {
                "isolated": {"single": "4/9", "multi": "1/5"},
                "synthetic_context": {"single": "6/9", "multi": "2/5"},
            },
            "context_note": (
                "Injected hand-authored evaluator context; not an end-to-end "
                "Core/Knowledge/Actions test."
            ),
        },
        "generalization_report": (
            "nix_knowledge/models/nixlm/luna-generalization-probes-paired-20260923.json"
        ),
    }
