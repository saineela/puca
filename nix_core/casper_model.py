"""Local Casper v5 Transformers backend.

Loads Qwen3.5's local 4-bit base plus the Casper PEFT adapter once, lazily,
inside Core. It implements the same small ``chat`` surface used by
``OllamaClient`` so Core's routing and Knowledge formatting contracts remain
unchanged.
"""
from __future__ import annotations

import gc
import os
import threading
from pathlib import Path
from typing import Any

from config import (
    ASSISTANT_NAME,
    ASSISTANT_ROLE,
    CONTEXT_MAX_CHARS,
    CONTEXT_WINDOW,
    CASPER_MAX_CONCURRENT_REQUESTS,
)
from model_slot import GPU_SLOT

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get(
    "CASPER_BASE_MODEL_PATH",
    ROOT / "nix_knowledge" / "models" / "qwen3.5-4b-hf",
))
ADAPTER_ROOT = ROOT / "nix_knowledge" / "models" / "nixlm"
ADAPTERS = {
    "casper-puca-qlora-v5": Path(os.environ.get(
        "CASPER_V5_ADAPTER_PATH", ADAPTER_ROOT / "casper-puca-qlora-v5"
    )),
    "casper-puca-qlora-v6": Path(os.environ.get(
        "CASPER_V6_ADAPTER_PATH", ADAPTER_ROOT / "casper-puca-qlora-v6-final"
    )),
}
# Retained for direct legacy Casper backend callers only. The official PUCA
# selector starts with Luna V6 and routes here only after explicit selection.
DEFAULT_MODEL = os.environ.get(
    "NIX_CASPER_MODEL", "casper-puca-qlora-v5"
)
if DEFAULT_MODEL not in ADAPTERS:
    DEFAULT_MODEL = "casper-puca-qlora-v5"
VRAM_FRACTION = float(os.environ.get("CASPER_VRAM_FRACTION", "0.68"))
MAX_NEW_TOKENS = int(os.environ.get("CASPER_MAX_NEW_TOKENS", "120"))
# Ordinary PUCA turns should not reserve a long decoding budget. Complex
# requests retain the larger ceiling; this fast ceiling only applies when Core
# explicitly passes think=False.
FAST_MAX_NEW_TOKENS = int(os.environ.get("CASPER_FAST_MAX_NEW_TOKENS", "64"))

_lock = threading.RLock()
_inference_slots = threading.BoundedSemaphore(max(1, CASPER_MAX_CONCURRENT_REQUESTS))
_instance: "CasperTransformersClient | None" = None
_selected_model = DEFAULT_MODEL


class CasperTransformersClient:
    """Lazy local QLoRA inference client; no model is loaded at import time."""

    def __init__(self, model_name: str, adapter: Path) -> None:
        self.model = model_name
        self.adapter_path = adapter
        # Do not overlap a Luna inference/model load with Casper allocation.
        with GPU_SLOT:
            self._load_weights()

    def _load_weights(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Casper Transformers backend requires CUDA")
        if not BASE.is_dir():
            raise FileNotFoundError(f"Casper base model missing: {BASE}")
        if not self.adapter_path.is_dir():
            raise FileNotFoundError(f"Casper adapter missing: {self.adapter_path}")

        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE, local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            base = AutoModelForCausalLM.from_pretrained(
                BASE,
                quantization_config=quantization,
                device_map={"": 0},
                dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            # Compatibility with Transformers versions that still spell this
            # argument torch_dtype.
            base = AutoModelForCausalLM.from_pretrained(
                BASE,
                quantization_config=quantization,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        self.model_instance = PeftModel.from_pretrained(
            base, self.adapter_path, is_trainable=False
        )
        self.model_instance.eval()

    def _template(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def chat(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
        timeout: float | None = None,
        think: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        # Casper never exposes a thinking path. Keep this invariant local to
        # the backend as well as in Core so callers cannot re-enable it by
        # passing think=True.
        think = False
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for turn in history[-CONTEXT_WINDOW:]:
            role = str(turn.get("role") or "")
            content = str(turn.get("content") or "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        prompt = self._template(messages)
        # A single RTX 4060 should not run unbounded overlapping generations
        # from websocket clients. The bounded slot protects both the input
        # tensors and KV cache; the Knowledge API remains independently warm.
        with GPU_SLOT, _inference_slots, self._torch.inference_mode():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(8192, CONTEXT_MAX_CHARS // 2),
            ).to("cuda")
            output = self.model_instance.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 1024) if max_new_tokens is not None else FAST_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()


def available_models() -> list[dict[str, Any]]:
    """Return the selectable local adapters; V6 is explicitly beta-only."""
    return [
        {
            "id": name,
            "label": "Casper V6 (beta)" if name.endswith("v6") else "Casper V5",
            "beta": name.endswith("v6"),
            "present": path.is_dir(),
            "active": name == _selected_model,
        }
        for name, path in ADAPTERS.items()
    ]


def selected_model() -> str:
    return _selected_model


def select_model(model_name: str) -> str:
    """Switch adapters exclusively; never keep V5 and V6 resident together."""
    global _instance, _selected_model
    if model_name not in ADAPTERS:
        raise ValueError("unsupported Casper model")
    if not ADAPTERS[model_name].is_dir():
        raise FileNotFoundError(f"Casper adapter missing: {ADAPTERS[model_name]}")
    # Acquire the process GPU slot before the adapter slot so cross-family
    # requests cannot deadlock while a model is being evicted.
    with GPU_SLOT, _inference_slots:
        with _lock:
            if _selected_model == model_name:
                # A previous Luna V6 request can have evicted Casper's
                # weights while leaving the configured Casper ID unchanged.
                # Selecting the visible active option should still restore
                # the Casper-only resident state without forcing a load.
                try:
                    from luna_model import unload_luna_client
                    unload_luna_client(slot_already_held=True)
                except ImportError:
                    pass
                return _selected_model
            old = _instance
            _instance = None
            _selected_model = model_name
            if old is not None:
                del old
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            # Luna and Casper share this GPU. An opt-in Casper switch evicts
            # Luna's model before the new Casper weights are initialized.
            try:
                from luna_model import unload_luna_client
                unload_luna_client(slot_already_held=True)
            except ImportError:
                pass
    return _selected_model


def get_casper_client() -> CasperTransformersClient:
    global _instance
    with GPU_SLOT, _lock:
        if _instance is None:
            # Core may have explicitly opened Luna Lab since Casper's last
            # request. Evict that experiment before loading Casper's PUCA.
            try:
                from luna_model import unload_luna_client
                unload_luna_client(slot_already_held=True)
            except ImportError:
                pass
            _instance = CasperTransformersClient(
                _selected_model, ADAPTERS[_selected_model]
            )
        return _instance


def runtime_status() -> dict[str, Any]:
    """Report the local adapter state without forcing model loading."""
    try:
        available = bool(__import__("torch").cuda.is_available())
    except Exception:
        available = False
    allocated = 0.0
    reserved = 0.0
    if available:
        try:
            torch = __import__("torch")
            allocated = round(torch.cuda.memory_allocated() / 2**30, 3)
            reserved = round(torch.cuda.memory_reserved() / 2**30, 3)
        except Exception:
            pass
    return {
        "backend": "transformers",
        "model": _selected_model,
        "available_models": available_models(),
        "base_present": BASE.is_dir(),
        "adapter_present": ADAPTERS[_selected_model].is_dir(),
        "loaded": _instance is not None,
        "cuda": available,
        "allocated_gib": allocated,
        "reserved_gib": reserved,
        "vram_fraction": VRAM_FRACTION,
        "max_concurrent_requests": max(1, CASPER_MAX_CONCURRENT_REQUESTS),
    }
