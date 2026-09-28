"""Local Official Luna inference backend, exclusive with other CUDA models.

The Official Luna adapter serves NIX PUCA conversations.
Retired research models are not selectable or loadable. Only one local model
checkpoint is resident.
"""
from __future__ import annotations

import gc
import os
import sys
import threading
from pathlib import Path
from typing import Any

from config import CONTEXT_MAX_CHARS, CONTEXT_WINDOW
from luna_runtime import (
    LUNA_BASE_PATH,
    LUNA_V6_ADAPTER_PATH,
    LUNA_V6_MODEL_ID,
    LUNA_MODEL_IDS,
    LUNA_V7_MODEL_ID,
    LUNA_V7_THINKING_ENABLED,
    available_models as describe_luna_models,
    strip_v7_reasoning_markup,
    system_prompt_for_model,
)
from model_slot import GPU_SLOT

LUNA_SCRIPTS = Path(__file__).resolve().parent.parent / "nix_knowledge" / "scripts" / "nixlm"
if str(LUNA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LUNA_SCRIPTS))
from luna_format import IDENTITY_SYSTEM
from luna_text import clean_response_text

LUNA_SYSTEM_PROMPT = IDENTITY_SYSTEM

LUNA_VRAM_FRACTION = float(os.environ.get("LUNA_VRAM_FRACTION", "0.48"))
LUNA_MAX_NEW_TOKENS = int(os.environ.get("LUNA_MAX_NEW_TOKENS", "96"))
# Direct callers default to the official V6 assistant.
LUNA_MODEL_ID = LUNA_V6_MODEL_ID
LUNA_ADAPTER = LUNA_V6_ADAPTER_PATH


def _adapter_paths() -> dict[str, Path]:
    return {LUNA_V6_MODEL_ID: LUNA_ADAPTER}


def _has_adapter_weights(path: Path) -> bool:
    return (
        (path / "adapter_config.json").is_file()
        and any(
            (path / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
    )

_lock = threading.RLock()
_client: "LunaTransformersClient | None" = None
_active_luna_model: str | None = None


class LunaTransformersClient:
    """Lazy 4-bit local model client; never allocated at module import."""

    def __init__(self, model_id: str, adapter_path: Path) -> None:
        self.model = model_id
        self.adapter_path = adapter_path
        # Luna generation is final-answer-only. V7's separate gated path keeps
        # this explicit in status and applies an extra prompt/output guard.
        self.thinking_enabled = (
            LUNA_V7_THINKING_ENABLED if model_id == LUNA_V7_MODEL_ID else False
        )
        self.model_instance = None
        self.tokenizer = None
        with GPU_SLOT:
            self._load()

    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Luna Transformers backend requires CUDA")
        if not LUNA_BASE_PATH.is_dir():
            raise FileNotFoundError(f"Luna base model missing: {LUNA_BASE_PATH}")
        if not self.adapter_path.is_dir():
            raise FileNotFoundError(f"Luna adapter missing: {self.adapter_path}")

        torch.cuda.set_per_process_memory_fraction(LUNA_VRAM_FRACTION, 0)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            LUNA_BASE_PATH,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            base = AutoModelForCausalLM.from_pretrained(
                LUNA_BASE_PATH,
                quantization_config=quantization,
                device_map={"": 0},
                dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                LUNA_BASE_PATH,
                quantization_config=quantization,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception:
            # Casper weights have already been evicted; release any partial
            # CUDA base allocation before allowing another backend to load.
            gc.collect()
            torch.cuda.empty_cache()
            raise
        try:
            self.model_instance = PeftModel.from_pretrained(
                base,
                self.adapter_path,
                is_trainable=False,
            )
        except Exception:
            del base
            gc.collect()
            torch.cuda.empty_cache()
            raise
        self.model_instance.eval()

    def chat(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
        timeout: float | None = None,
        think: bool | None = None,
    ) -> str:
        # This Transformers model has no separate reasoning-mode switch. Keep
        # caller requests from enabling one and, for V7, make final-answer-only
        # behavior explicit in the system prompt as well as postprocessing.
        think = False
        from luna_format import generation_stop_ids, render_messages

        effective_system_prompt = system_prompt_for_model(
            self.model,
            system_prompt or LUNA_SYSTEM_PROMPT,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_system_prompt}
        ]
        for turn in history[-CONTEXT_WINDOW:]:
            role = str(turn.get("role") or "")
            content = str(turn.get("content") or "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        prompt = render_messages(messages, generation=True, tokenizer=self.tokenizer)
        with GPU_SLOT, self._torch.inference_mode():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(8192, CONTEXT_MAX_CHARS // 2),
                add_special_tokens=False,
            ).to("cuda")
            output = self.model_instance.generate(
                **inputs,
                max_new_tokens=LUNA_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                eos_token_id=generation_stop_ids(self.tokenizer),
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.08,
                no_repeat_ngram_size=4,
                renormalize_logits=True,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded = strip_v7_reasoning_markup(self.model, decoded)
        return clean_response_text(decoded)


def _evict_casper_client(*, slot_already_held: bool = False) -> None:
    """Unload Casper without allocating Luna or triggering a model load."""
    try:
        import casper_model
    except ImportError:
        return
    guard = _NullContext() if slot_already_held else GPU_SLOT
    with guard, casper_model._inference_slots, casper_model._lock:
        previous = casper_model._instance
        casper_model._instance = None
        if previous is not None:
            del previous
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def unload_luna_client(*, slot_already_held: bool = False) -> None:
    """Evict Luna after in-flight generation completes."""
    global _client, _active_luna_model
    guard = _NullContext() if slot_already_held else GPU_SLOT
    with guard, _lock:
        previous = _client
        _client = None
        _active_luna_model = None
        if previous is not None:
            del previous
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def available_models() -> list[dict[str, Any]]:
    """Report the official Luna V6 artifact without loading it."""
    models = describe_luna_models(
        _active_luna_model,
        base_path=LUNA_BASE_PATH,
        adapter_paths=_adapter_paths(),
    )
    official_model = None
    try:
        from assistant_model import selected_model as selected_official_model
        official_model = selected_official_model()
    except Exception:
        pass
    for model in models:
        model["thinking_enabled"] = False
        if model["id"] == LUNA_V6_MODEL_ID:
            model["label"] = "Official Luna"
            model["active"] = official_model == LUNA_V6_MODEL_ID
            model["default"] = True
            model["production_default"] = True
            model["official"] = True
            model["experimental"] = False
            model["beta"] = False
    return models


def selected_model() -> str | None:
    return _active_luna_model


def get_luna_client(model_name: str = LUNA_MODEL_ID) -> LunaTransformersClient:
    """Load the official Luna V6 client for Core."""
    global _client, _active_luna_model
    if model_name not in _adapter_paths():
        raise ValueError("unsupported Luna model")
    from config import CASPER_BACKEND
    if CASPER_BACKEND != "transformers":
        raise RuntimeError(
            "Luna's local Transformers loader is available only when the "
            "official conversation model uses the Transformers backend"
        )
    if not LUNA_BASE_PATH.is_dir():
        raise FileNotFoundError(f"Luna base model missing: {LUNA_BASE_PATH}")
    adapter_path = _adapter_paths()[model_name]
    if not adapter_path.is_dir() or not _has_adapter_weights(adapter_path):
        raise FileNotFoundError(f"Luna adapter or adapter weights missing: {adapter_path}")
    with GPU_SLOT, _lock:
        if _client is not None and _active_luna_model == model_name:
            return _client
        unload_luna_client(slot_already_held=True)
        _evict_casper_client(slot_already_held=True)
        _client = LunaTransformersClient(model_name, adapter_path)
        _active_luna_model = model_name
        return _client


def select_model(model_name: str | None) -> str | None:
    """Reject retired research IDs; V6 is selected through Core's registry."""
    if model_name is None:
        return None
    raise ValueError("Luna research-model selection is retired")


def runtime_status() -> dict[str, Any]:
    try:
        cuda_available = bool(_client and _client._torch.cuda.is_available())
    except Exception:
        cuda_available = False
    models = available_models()
    active_adapter = (
        _adapter_paths().get(_active_luna_model)
        if _active_luna_model is not None
        else None
    )
    from assistant_model import selected_model as selected_official_model

    official_model = selected_official_model()
    return {
        "backend": "transformers",
        "active": _active_luna_model,
        "selected_model": _active_luna_model,
        "lab_active": False,
        "official_luna_loaded": bool(
            _client is not None and _active_luna_model == LUNA_V6_MODEL_ID
        ),
        "official_model": official_model,
        "production_default": official_model == LUNA_V6_MODEL_ID,
        "official_default": official_model == LUNA_V6_MODEL_ID,
        "base_present": LUNA_BASE_PATH.is_dir(),
        "adapter_present": _has_adapter_weights(LUNA_ADAPTER),
        "active_adapter_present": (
            _has_adapter_weights(active_adapter) if active_adapter else False
        ),
        "loaded": _client is not None,
        "lab_loaded": False,
        "cuda": cuda_available,
        "vram_fraction": LUNA_VRAM_FRACTION,
        "release_status": "official_assistant_default",
        "thinking_enabled": False,
        "active_thinking_enabled": False,
        "v7_plan": {
            "model_id": LUNA_V7_MODEL_ID,
            "registered": False,
            "trained": False,
            "thinking_enabled": LUNA_V7_THINKING_ENABLED,
            "output_policy": "final_answer_only",
        },
    }
