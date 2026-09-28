"""Conversation serialization for Luna's Llama 3.2 base checkpoint.

The base checkpoint is not assumed to provide an instruct chat template. These
are the standard Llama 3 header/eot markers, rendered explicitly so training
and inference use exactly the same format.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from luna_text import luna_identity_control_allowed, validate_luna_training_rows

BOS = "<|begin_of_text|>"
EOT = "<|eot_id|>"

# Luna's identity is self-contained: no prompt text asks her to define herself
# against another model or personality. Core remains responsible for the
# system-supplied memory/action boundary and deterministic identity facts.
IDENTITY_SYSTEM = (
    "You are Luna. You are the user's conversational companion. Be warm, "
    "natural, and concise; answer the latest user message directly. Do not "
    "invent memories, real-world actions, relationships, or a human biography. "
    "Use only facts supplied in this conversation or trusted context. If asked "
    "your name, say Luna. If you do not know something, say so briefly."
)


_CASPER_REFERENCE_RE = re.compile(r"\\bcasper\\b", re.IGNORECASE)


def luna_identity_control_text(text: str) -> bool:
    """True when a row explicitly teaches a Luna/Casper identity relation."""
    return bool(_CASPER_REFERENCE_RE.search(text))


def luna_identity_control_allowed(messages: Iterable[dict[str, Any]]) -> bool:
    """Reject legacy cross-identity examples from new Luna training data."""
    return not any(
        luna_identity_control_text(str(message.get("content") or ""))
        for message in messages
    )


def validate_luna_training_rows(rows: Iterable[dict[str, Any]]) -> int:
    """Fail closed if future SFT data teaches cross-person identity text."""
    rejected = sum(
        not luna_identity_control_allowed(row.get("messages", []))
        for row in rows
    )
    if rejected:
        raise ValueError(
            f"Luna SFT contains {rejected} row(s) mentioning a different model/person identity"
        )
    return rejected


def generation_stop_ids(tokenizer: Any) -> list[int]:
    """Return both Llama EOS and end-of-turn IDs for generation stopping."""
    ids: list[int] = []
    for token in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(EOT)):
        if isinstance(token, int) and token >= 0 and token not in ids:
            ids.append(token)
    if not ids:
        raise ValueError("Luna tokenizer has no usable EOS/EOT token IDs")
    return ids


def render_messages(
    messages: Iterable[dict[str, Any]],
    *,
    generation: bool = False,
    tokenizer: Any | None = None,
) -> str:
    """Render messages using an instruct template when available.

    Base checkpoints use the explicit Llama 3 headers. Instruct checkpoints
    provide the authoritative template, so training and inference delegate to
    it instead of duplicating template behavior.
    """
    messages = list(messages)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
        ))
    parts = [BOS]
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported Luna message role: {role}")
        content = str(message.get("content") or "")
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )
    if generation:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)
