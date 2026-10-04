"""Bounded orchestration for model-proposed, schema-declared skill actions."""
from __future__ import annotations

import json
from typing import Any, Callable

from skill_runtime import SkillRuntimeError


class SkillManager:
    """Validate complete action plans before dispatching any trusted worker call."""

    MAX_STEPS = 8
    MAX_RESPONSE_CHARS = 32768

    def __init__(self, runtime: Any):
        self.runtime = runtime

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON field: {key}")
            value[key] = item
        return value

    def parse(self, response: str) -> list[dict[str, Any]]:
        """Parse exactly one canonical plan marker and JSON object."""
        if not isinstance(response, str) or len(response) > self.MAX_RESPONSE_CHARS:
            raise SkillRuntimeError("Skill plan response exceeded its size limit.")
        marker = "NIX_SKILL_PLAN:"
        if not response.lstrip().startswith(marker):
            raise SkillRuntimeError("Model response did not contain a structured skill plan.")
        encoded = response.lstrip()[len(marker):].strip()
        plan, end = json.JSONDecoder(object_pairs_hook=self._unique_object).raw_decode(encoded)
        if encoded[end:].strip():
            raise SkillRuntimeError("Unexpected text after the structured skill plan.")
        if not isinstance(plan, dict) or set(plan) != {"type", "calls"} or plan.get("type") != "skill_tool_plan":
            raise SkillRuntimeError("Skill plan has an invalid envelope.")
        calls = plan.get("calls")
        if not isinstance(calls, list) or not 1 <= len(calls) <= self.MAX_STEPS:
            raise SkillRuntimeError(f"Skill plan must contain between 1 and {self.MAX_STEPS} actions.")
        for call in calls:
            if not isinstance(call, dict) or set(call) != {"type", "skill_id", "tool", "arguments"}:
                raise SkillRuntimeError("Every plan action must identify one declared skill tool and arguments.")
            if call.get("type") != "skill_tool_call":
                raise SkillRuntimeError("Skill plan contains an action with an invalid type.")
        return calls

    def validate(self, calls: list[dict[str, Any]], allowed_skill_ids: set[str]) -> list[dict[str, Any]]:
        """Validate every action fully before any worker execution can begin."""
        if not isinstance(calls, list) or not 1 <= len(calls) <= self.MAX_STEPS:
            raise SkillRuntimeError(f"Skill plan must contain between 1 and {self.MAX_STEPS} actions.")
        validated: list[dict[str, Any]] = []
        for call in calls:
            if call.get("skill_id") not in allowed_skill_ids:
                raise SkillRuntimeError("A proposed skill was not supplied for this request.")
            matched = self.runtime.validate_proposal(call)
            if matched is None or matched.get("skill_id") not in allowed_skill_ids:
                raise SkillRuntimeError("A proposed skill action did not validate against its installed schema.")
            validated.append(matched)
        return validated

    def execute(
        self,
        validated: list[dict[str, Any]],
        *,
        confirm: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """Run in order, stopping at the first exception or failed readback."""
        outcomes: list[dict[str, Any]] = []
        for index, matched in enumerate(validated):
            try:
                outcome = self.runtime.execute(matched)
            except Exception as exc:
                return {
                    "outcomes": outcomes,
                    "failed_index": index,
                    "completed_count": index,
                    "attempted_count": index + 1,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "confirmed": False,
                }
            outcomes.append(outcome)
            if confirm is not None and not confirm(matched, outcome):
                return {
                    "outcomes": outcomes,
                    "failed_index": index,
                    "completed_count": index,
                    "attempted_count": index + 1,
                    "error": "The returned device state did not confirm this action.",
                    "confirmed": False,
                }
        return {
            "outcomes": outcomes,
            "failed_index": None,
            "completed_count": len(validated),
            "attempted_count": len(validated),
            "error": None,
            "confirmed": True,
        }
