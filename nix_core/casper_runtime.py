"""Backend-aware Casper deployment metadata.

This module reports configuration and artifact readiness without forcing a
model load. Transformers/PEFT is the verified local Casper backend; Tabby is
an optional OpenAI-compatible transport for a separately loaded
ExLlama-compatible artifact.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config import CASPER_BACKEND, OLLAMA_MODEL, TABBY_MODEL

ROOT = Path(__file__).resolve().parent.parent
CASPER_ADAPTER_VERSION = os.environ.get(
    "NIX_CASPER_MODEL", "casper-puca-qlora-v5"
)  # legacy Casper-only compatibility metadata; PUCA defaults to Luna V6
CASPER_ADAPTER_PATH = Path(os.environ.get(
    "CASPER_ADAPTER_PATH",
    ROOT / "nix_knowledge" / "models" / "nixlm" / CASPER_ADAPTER_VERSION,
))
CASPER_V6_ADAPTER_PATH = Path(os.environ.get(
    "CASPER_V6_ADAPTER_PATH",
    ROOT / "nix_knowledge" / "models" / "nixlm" / "casper-puca-qlora-v6-final",
))
CASPER_BASE_PATH = Path(os.environ.get(
    "CASPER_BASE_MODEL_PATH",
    ROOT / "nix_knowledge" / "models" / "qwen3.5-4b-hf",
))
CASPER_MERGER = Path(os.environ.get(
    "CASPER_MERGER_PATH",
    ROOT / "nix_knowledge" / "scripts" / "nixlm" / "merge_casper_safetensors.py",
))


def runtime_status(*, backend_status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report the configured backend and local Casper artifacts truthfully."""
    if CASPER_BACKEND == "transformers":
        model = CASPER_ADAPTER_VERSION
    elif CASPER_BACKEND == "tabby":
        model = TABBY_MODEL
    else:
        model = OLLAMA_MODEL

    status = dict(backend_status or {})
    return {
        "backend": CASPER_BACKEND,
        "model": model,
        "active": True,
        "adapter_version": CASPER_ADAPTER_VERSION,
        "adapter_present": CASPER_ADAPTER_PATH.is_dir(),
        "v6_adapter_present": CASPER_V6_ADAPTER_PATH.is_dir(),
        "base_present": CASPER_BASE_PATH.is_dir(),
        "merge_script_present": CASPER_MERGER.is_file(),
        **status,
    }
