"""Skill prompt cleanser: Qwen3-0.6B turns the user's raw wording into
explicit, device-anchored step tasks that are loaded one by one onto the
Needle planner.

Contract (deliberately narrow):
- Input: the user's (typo-corrected) request text and the device label.
- Output: 1..8 short imperative steps, each naming the device, carrying no
  JSON, no tool names, and no invented actions.
- Fail-closed: any model error, malformed JSON, or contract violation falls
  back to the user's own words as a single step. The cleanser is an
  enhancement, never a gate: Core's intent-preservation checks, Needle's
  schema validation, and the device readback still police every step.
"""
from __future__ import annotations

import json
import re
from typing import Any

_CLEANSE_SYSTEM_PROMPT = (
    "You are the skill prompt cleanser for a local device assistant. The user's "
    "raw request is cleaned up for a separate device planner. Rewrite the request "
    "as one to eight short imperative steps, in the user's own order. Each step "
    "must be one simple device action that names the device (for example 'turn on "
    "the ring light', 'set the ring light color to blue'). Use only the actions "
    "and settings the user asked for: never add, remove, reverse, or reorder "
    "meaning. If the request contains an explicit RGB or hex color value, copy it "
    "exactly into the matching step. When the payload includes resolved_color, it "
    "is the exact settable color Core resolved for the user's color description — "
    "write that color name into the color step instead of the vague wording. Fix "
    "obvious typos in device and color words. Do not invent colors, numbers, "
    "effects, or tool names beyond resolved_color. Do not answer the user. Return "
    'ONLY a JSON object with exactly one key: {"steps": ["...", "..."]}'
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def repair_steps_json(response: Any) -> dict[str, Any] | None:
    """Extract the steps JSON from a possibly fenced or prose-wrapped reply."""
    if not isinstance(response, str) or not response.strip():
        return None
    match = _JSON_OBJECT_RE.search(response)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


class SkillPromptCleanser:
    """Qwen3-0.6B backed prompt cleanser producing explicit step tasks."""

    MAX_STEPS = 8

    def __init__(self, client: Any, model: str = "qwen3:0.6b"):
        self._client = client
        self.model = model

    def cleanse(
        self,
        *,
        user_text: str,
        device_name: str,
        resolved_color: str | None = None,
    ) -> list[str]:
        """Return 1..8 cleaned imperative steps; never raise.

        resolved_color carries Core's concrete color name for the user's
        color description ("light blue" stays "light blue", not "blue") so
        the cleanser can fill in the information the planner needs.
        """
        fallback = [" ".join(str(user_text or "").split())]
        if not fallback[0]:
            return fallback
        payload = {
            "user_request": fallback[0][:600],
            "device": str(device_name or "the device")[:80],
        }
        if resolved_color:
            payload["resolved_color"] = str(resolved_color)[:40]
        try:
            response = self._client.chat(
                system_prompt=_CLEANSE_SYSTEM_PROMPT,
                history=[],
                user_text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                think=False,
                max_new_tokens=200,
            )
        except Exception:
            return fallback
        parsed = repair_steps_json(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("steps"), list):
            return fallback
        steps: list[str] = []
        for step in parsed["steps"][: self.MAX_STEPS]:
            if not isinstance(step, str):
                continue
            cleaned = " ".join(step.split())
            if not cleaned or len(cleaned) > 200:
                continue
            if "{" in cleaned or "}" in cleaned or "steps" == cleaned.casefold():
                continue
            steps.append(cleaned)
        if not steps:
            return fallback
        return steps
