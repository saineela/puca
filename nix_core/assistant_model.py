"""Selection and lazy loading for the official local conversation model.

Official Luna is the default conversation model for Nix.
Casper V5/V6 remain selectable through the dashboard. Luna Pro v1 is retired and not
part of any runtime registry. Weights are loaded only when a chat is requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import CASPER_BACKEND
from luna_runtime import (
    LUNA_BASE_PATH,
    LUNA_V6_ADAPTER_PATH,
    LUNA_V6_MODEL_ID,
    _has_peft_adapter,
)
from model_slot import GPU_SLOT

CASPER_V5_MODEL_ID = "casper-puca-qlora-v5"
CASPER_V6_MODEL_ID = "casper-puca-qlora-v6"
CASPER_MODEL_IDS = frozenset({CASPER_V5_MODEL_ID, CASPER_V6_MODEL_ID})
OFFICIAL_MODEL_IDS = frozenset({LUNA_V6_MODEL_ID, *CASPER_MODEL_IDS})

# PUCA always starts with Luna V6. Other registered models remain available
# only through an explicit dashboard selection.
DEFAULT_MODEL_ID = LUNA_V6_MODEL_ID
_selected_model = DEFAULT_MODEL_ID


def selected_model() -> str:
    """Return the configured assistant model without loading its weights."""
    return _selected_model


def assistant_name_for_model(model_id: str | None = None) -> str:
    """Map an official model ID to its independent conversational identity."""
    model_id = model_id or _selected_model
    return "Luna" if model_id == LUNA_V6_MODEL_ID else "Casper"


def backend_for_model(model_id: str | None = None) -> str:
    """Return the active backend only when the official selector can serve it."""
    model_id = model_id or _selected_model
    if model_id not in OFFICIAL_MODEL_IDS:
        return "unknown"
    return "transformers" if CASPER_BACKEND == "transformers" else CASPER_BACKEND


def _model_inventory() -> list[dict[str, Any]]:
    if CASPER_BACKEND != "transformers":
        return []
    from casper_model import ADAPTERS

    luna_present = (
        LUNA_BASE_PATH.is_dir()
        and _has_peft_adapter(LUNA_V6_ADAPTER_PATH)
    )
    return [
        {
            "id": LUNA_V6_MODEL_ID,
            "label": "Official Luna",
            "name": "Luna",
            "backend": "transformers",
            "present": luna_present,
            "active": _selected_model == LUNA_V6_MODEL_ID,
            "default": DEFAULT_MODEL_ID == LUNA_V6_MODEL_ID,
            "official": True,
            "experimental": False,
            "beta": False,
            "production_default": DEFAULT_MODEL_ID == LUNA_V6_MODEL_ID,
        },
        {
            "id": CASPER_V5_MODEL_ID,
            "label": "Casper · V5",
            "name": "Casper",
            "backend": "transformers",
            "present": ADAPTERS[CASPER_V5_MODEL_ID].is_dir(),
            "active": _selected_model == CASPER_V5_MODEL_ID,
            "default": DEFAULT_MODEL_ID == CASPER_V5_MODEL_ID,
            "official": True,
            "experimental": False,
            "beta": False,
            "production_default": DEFAULT_MODEL_ID == CASPER_V5_MODEL_ID,
        },
        {
            "id": CASPER_V6_MODEL_ID,
            "label": "Casper · V6 (beta)",
            "name": "Casper",
            "backend": "transformers",
            "present": ADAPTERS[CASPER_V6_MODEL_ID].is_dir(),
            "active": _selected_model == CASPER_V6_MODEL_ID,
            "default": DEFAULT_MODEL_ID == CASPER_V6_MODEL_ID,
            "official": True,
            "experimental": False,
            "beta": True,
            "production_default": DEFAULT_MODEL_ID == CASPER_V6_MODEL_ID,
        },
    ]


def available_models() -> list[dict[str, Any]]:
    """List selectable official Transformer choices without allocating GPU memory."""
    return _model_inventory()


def _ensure_model_present(model_id: str) -> None:
    entry = next((model for model in _model_inventory() if model["id"] == model_id), None)
    if entry is None:
        raise ValueError("unsupported official conversation model")
    if not entry["present"]:
        raise FileNotFoundError(f"Conversation model artifacts missing: {model_id}")


def select_model(model_id: str) -> str:
    """Set the official assistant model and evict other resident adapters."""
    global _selected_model
    if CASPER_BACKEND != "transformers":
        raise RuntimeError(
            "The official Luna/Casper adapter selector requires the Transformers backend"
        )
    if model_id not in OFFICIAL_MODEL_IDS:
        raise ValueError("unsupported official conversation model")
    _ensure_model_present(model_id)

    with GPU_SLOT:
        if model_id == LUNA_V6_MODEL_ID:
            from luna_model import _evict_casper_client, unload_luna_client

            unload_luna_client(slot_already_held=True)
            _evict_casper_client(slot_already_held=True)
        else:
            from casper_model import select_model as select_casper_model

            select_casper_model(model_id)
        _selected_model = model_id
    return _selected_model


def get_chat_client():
    """Load and return only the currently selected official model client."""
    if CASPER_BACKEND != "transformers":
        raise RuntimeError(
            "The official Luna/Casper adapter loader requires the Transformers backend"
        )
    model_id = selected_model()
    _ensure_model_present(model_id)
    with GPU_SLOT:
        if model_id == LUNA_V6_MODEL_ID:
            from luna_model import get_luna_client

            return get_luna_client(model_id)

        from casper_model import get_casper_client, select_model as select_casper_model

        select_casper_model(model_id)
        return get_casper_client()
