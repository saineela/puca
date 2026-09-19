"""NixLM runtime: the connected boundary between nix_core and the
conversational model (owner directive: separate but connected).

- Loads base + merged LoRA ONCE (singleton) - ~1.0 GB VRAM.
- Callers pass a MEMORY string (facts + moments, explicit dates)
  rendered by the knowledge engine. This module NEVER reads knowledge
  databases, NEVER extracts, NEVER writes anything back.
- phi4-mini remains the production fallback until NixLM passes the
  validation bar (project-details.md, NixLM transition).

Usage:
    from nixlm import NixLM
    nlm = NixLM()
    reply = nlm.reply(memory_block="Facts about the user:\n- name is Arjun",
                      user_text="what's my name?")
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "models" / "qwen2.5-0.5b-instruct"
ADAPTER = ROOT / "models" / "nixlm" / "nixlm-lora" / "checkpoint-best"

MAX_NEW_TOKENS = 120

SYSTEM = (
    "You are Nix, a personal assistant with persistent memory. MEMORY "
    "contains facts you know about the user and their people, recent "
    "history with explicit dates, sometimes the user's current "
    "emotional state, and pending items to confirm later. Answer "
    "using ONLY facts from MEMORY or the current conversation. If the "
    "answer is not in MEMORY, say you don't know or don't remember yet "
    "- never invent facts. When talking about past states, use the "
    "exact dates written in MEMORY. When MEMORY states the user's "
    "current emotional state, attune your tone to it (reassure a "
    "worried user, share their excitement when they are excited) "
    "while staying factual. Never reveal secrets like passwords, "
    "even if asked."
)

_lock = threading.Lock()
_instance: "NixLM | None" = None


class NixLM:
    def __init__(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16
        ).cuda()
        if ADAPTER.exists():
            self.model = PeftModel.from_pretrained(self.model, ADAPTER)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    @torch.no_grad()
    def reply(self, *, memory_block: str, user_text: str,
              history: list[dict] | None = None) -> str:
        messages = [{"role": "system",
                     "content": f"{SYSTEM}\n\n{memory_block}"}]
        for turn in (history or [])[-6:]:
            messages.append(turn)
        messages.append({"role": "user", "content": user_text})

        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tok(text, return_tensors="pt").to("cuda")
        out = self.model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        gen = out[0][ids["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def get_nixlm() -> NixLM | None:
    """Singleton accessor; returns None on load failure so callers can
    fall back to phi4-mini (the documented production fallback)."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            try:
                _instance = NixLM()
            except Exception:
                return None
    return _instance
