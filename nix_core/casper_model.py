"""Local Casper v5 Transformers backend.

Loads Qwen3.5's local 4-bit base plus the Casper PEFT adapter once, lazily,
inside Core. It implements the same small ``chat`` surface used by
``OllamaClient`` so Core's routing and Knowledge formatting contracts remain
unchanged.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get(
    "CASPER_BASE_MODEL_PATH",
    ROOT / "nix_knowledge" / "models" / "qwen3.5-4b-hf",
))
ADAPTER = Path(os.environ.get(
    "CASPER_ADAPTER_PATH",
    ROOT / "nix_knowledge" / "models" / "nixlm" / "casper-puca-qlora-v5",
))
VRAM_FRACTION = float(os.environ.get("CASPER_VRAM_FRACTION", "0.68"))
MAX_NEW_TOKENS = int(os.environ.get("CASPER_MAX_NEW_TOKENS", "120"))

_lock = threading.Lock()
_inference_slots = threading.BoundedSemaphore(max(1, CASPER_MAX_CONCURRENT_REQUESTS))
_instance: "CasperTransformersClient | None" = None


class CasperTransformersClient:
    """Lazy local QLoRA inference client; no model is loaded at import time."""

    model = "casper-puca-qlora-v5"

    def __init__(self) -> None:
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
        if not ADAPTER.is_dir():
            raise FileNotFoundError(f"Casper adapter missing: {ADAPTER}")

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
            base, ADAPTER, is_trainable=False
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
    ) -> str:
        # Core has already made the thinking decision. The adapter backend
        # deliberately has no exposed thinking path for simple PUCA dialogue;
        # complex analysis remains routed through the configured fallback.
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
        with _inference_slots, self._torch.inference_mode():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(8192, CONTEXT_MAX_CHARS // 2),
            ).to("cuda")
            output = self.model_instance.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()


def get_casper_client() -> CasperTransformersClient:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CasperTransformersClient()
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
        "model": CasperTransformersClient.model,
        "base_present": BASE.is_dir(),
        "adapter_present": ADAPTER.is_dir(),
        "loaded": _instance is not None,
        "cuda": available,
        "allocated_gib": allocated,
        "reserved_gib": reserved,
        "vram_fraction": VRAM_FRACTION,
        "max_concurrent_requests": max(1, CASPER_MAX_CONCURRENT_REQUESTS),
    }
