"""D5 - grammar-constrained decoding for the planner's tool-call envelope.

The envelope is the one place where the shape is owned by code, so it is the one
place a grammar may constrain generation. Everything else the planner writes -
the reasoning, the wording - stays unconstrained, because forcing a model off its
high-probability path degrades exactly the output a grammar cannot check.

What this module does *not* do is claim the plan is correct. Zero malformed
envelopes is the whole claim; intent, permissions, confirmation and outcome
verification are the independent checks in ``eval/plan_validation.py`` and they
are unchanged whether the grammar is active or not.

No grammar engine is bundled with this box. The module detects one of the
engines that can supply a Transformers logits processor (XGrammar, then
llm-format-enforcer) and uses it when present; with none installed the planner
generates exactly as before and only the independent checks apply. That is a
deliberate choice, not a fallback: an engine is an extra dependency with its own
build, and the measured validate-retry rate decides whether it is worth adding.
"""
from __future__ import annotations

import json
from typing import Any

#: The envelope, as a schema. One call object per generation: the planner emits
#: a single tool call per step and Core composes multi-step plans from steps it
#: has validated one at a time.
ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "minLength": 1},
        "arguments": {"type": "object"},
    },
    "required": ["tool", "arguments"],
    "additionalProperties": False,
}

#: Engines that can build a logits processor, in the order they are preferred.
_ENGINES = ("xgrammar", "lmformatenforcer")


def available_engine() -> str | None:
    """The installed engine this build can use, or ``None``."""
    import importlib.util

    for name in _ENGINES:
        if importlib.util.find_spec(name) is not None:
            return name
    return None


def grammar_logits_processor(tokenizer: Any, prompt_length: int, *, engine: str | None = None):
    """A logits processor that constrains generation to ``ENVELOPE_SCHEMA``.

    Returns ``None`` when no engine is installed, which is the normal state of
    this box: the planner then runs unconstrained and the five independent
    checks are the guarantee, exactly as they are when the grammar is active.
    """
    name = engine or available_engine()
    if name is None:
        return None
    if name == "xgrammar":
        import xgrammar as xgr

        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
        compiled = compiler.compile_json_schema(json.dumps(ENVELOPE_SCHEMA))
        return xgr.contrib.hf.LogitsProcessor(compiled)
    if name == "lmformatenforcer":
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(ENVELOPE_SCHEMA)
        return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    raise ValueError(f"unknown grammar engine: {name}")


def envelope_is_well_formed(raw: str) -> bool:
    """The deterministic check the grammar is not allowed to replace.

    Accepts what the planner actually emits - a ``<tool_call>`` block, a bare
    JSON object, or a JSON array of them - and refuses anything whose parsed
    shape is not the declared envelope.
    """
    text = (raw or "")
    start = text.find("<tool_call>")
    while start != -1:
        end = text.find("</tool_call>", start)
        if end == -1:
            return False
        if not _shapes_ok(text[start + len("<tool_call>"):end]):
            return False
        start = text.find("<tool_call>", end)
    if "<tool_call>" in text:
        return True
    return _shapes_ok(text)


def _shapes_ok(fragment: str) -> bool:
    try:
        parsed = json.loads(fragment.strip())
    except (TypeError, ValueError):
        return False
    items = parsed if isinstance(parsed, list) else [parsed]
    if not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if set(item) - set(ENVELOPE_SCHEMA["properties"]):
            return False
        if not isinstance(item.get("tool"), str) or not item["tool"]:
            return False
        if not isinstance(item.get("arguments"), dict):
            return False
    return True
