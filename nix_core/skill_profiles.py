"""Two-stage skill pipeline support: per-skill profiles plus a Needle3 decider.

Stage 1 (decider): a dedicated Needle 3 instance receives one bounded tool per
installed skill profile and proposes exactly one profile for the normalized
request. It is proposal-only and never executes anything.

Stage 2 (executor): the existing SkillPlanner droidcall is loaded with the
selected profile's system prompt and proposes the actual device calls, which
Core validates and executes exactly as before. Every stage fails closed.
"""
from __future__ import annotations

import math
import os
import re
import threading
from typing import Any

from config import SKILL_PLANNER_MODEL
from skill_planner import SkillPlannerError, _NEEDLE_INFERENCE_LOCK


SKILL_PROFILE_DECIDER_MODEL = os.environ.get(
    "NIX_SKILL_PROFILE_DECIDER_MODEL", SKILL_PLANNER_MODEL
)

MAX_PROFILE_PROMPT_CHARS = 2000
MAX_DECIDER_PROMPT_CHARS = 1200
MAX_DECIDER_TOOL_DESCRIPTION_CHARS = 420


class SkillProfileError(RuntimeError):
    """The profile decider was unavailable or did not return a valid selection."""


class SkillProfile:
    """A bounded, per-skill executor prompt loaded into the skill executor."""

    __slots__ = ("profile_id", "skill_id", "system_prompt")

    def __init__(self, *, profile_id: str, skill_id: str, system_prompt: str):
        self.profile_id = profile_id
        self.skill_id = skill_id
        self.system_prompt = system_prompt


_EXECUTOR_BASE_PROMPT = (
    "Propose only the schema-declared functions matching the normalized user "
    "request. Do not add actions or arguments; preserve explicit RGB values and "
    "use the exact available effect catalog."
)


def _executor_prompt(
    *,
    profile_id: str,
    skill_id: str,
    spec: dict[str, Any],
) -> SkillProfile:
    device_name = str(spec.get("device_name") or spec.get("name") or "the device")[:64]
    prompt = (
        f"You are the skill executor for the {profile_id} profile (device: {device_name}). "
        + _EXECUTOR_BASE_PROMPT
    ).strip()
    if len(prompt) > MAX_PROFILE_PROMPT_CHARS:
        prompt = prompt[:MAX_PROFILE_PROMPT_CHARS]
    return SkillProfile(
        profile_id=profile_id, skill_id=skill_id, system_prompt=prompt
    )


def profile_for_spec(spec: dict[str, Any]) -> SkillProfile:
    """Return the bounded executor profile for one installed skill."""
    if not isinstance(spec, dict) or not isinstance(spec.get("skill_id"), str):
        raise SkillProfileError("Skill profile data did not identify a skill.")
    skill_id = spec["skill_id"]
    if "ring-light" in skill_id or "ring_light" in skill_id:
        return _executor_prompt(
            profile_id="ring-light",
            skill_id=skill_id,
            spec=spec,
        )
    return _executor_prompt(
        profile_id=skill_id.split(":")[-1][:40].strip("_") or "generic",
        skill_id=skill_id,
        spec=spec,
    )


_DECIDER_SYSTEM_PROMPT = (
    "You are the skill-profile decider in a two-stage device-skill pipeline. "
    "You receive one normalized device request and exactly one tool per "
    "installed skill profile. Your only job is to choose which profile owns "
    "the device the request is about; you do not judge whether the request is "
    "answerable and you never plan or perform actions yourself. Select exactly "
    "one profile whose device the request targets or reads. Questions and read "
    "requests about a profile's device (its state, colors, effects, or "
    "settings) still select that profile: a later stage plans the actual "
    "actions. Do not answer the user, do not invent actions or arguments, and "
    "never select more than one profile. If no installed profile's device "
    "matches the request, withhold the call."
)


def _profile_slug(spec: dict[str, Any], index: int) -> str:
    """Prefer a semantic, device-based tool name the small model can read."""
    source = str(spec.get("device_name") or spec.get("name") or "")[:40]
    if not source.strip():
        source = str(spec.get("skill_id") or "").split(":")[-1][:40]
    slug = re.sub(r"[^A-Za-z0-9_]", "_", source).casefold().strip("_") or f"profile_{index}"
    name = f"route_to_{slug[:40]}_profile"[:64]
    return re.sub(r"_+", "_", name)


def decider_tools(
    capabilities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build one bounded decider tool per installed skill profile."""
    if not isinstance(capabilities, list) or not capabilities or len(capabilities) > 8:
        raise SkillProfileError("Skill profile decider received invalid capability data.")
    tools: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}
    for index, spec in enumerate(capabilities):
        if not isinstance(spec, dict) or not isinstance(spec.get("skill_id"), str):
            raise SkillProfileError("A skill capability did not identify a skill.")
        skill_id = spec["skill_id"]
        device_name = str(spec.get("device_name") or spec.get("name") or "")[:64]
        # Routing-only language: action-planning wording from the manifest
        # ("choose one RGB triplet") makes the decider withhold because it
        # takes no arguments. The decider only routes to the profile.
        description = (
            f"Route the request to the skill executor for the device '{device_name}'. "
            "This one profile covers every request about this device: turning it on "
            "or off, changing its color, brightness, or effects, and reading its "
            "state, colors, or effects. Selecting it only routes the request; a "
            "later stage plans the exact actions."
        )[:MAX_DECIDER_TOOL_DESCRIPTION_CHARS]
        name = _profile_slug(spec, index)
        if name in bindings:
            raise SkillProfileError("Installed skill profiles produced a duplicate decider tool name.")
        tools.append({
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The exact normalized request text being routed.",
                    },
                },
                "required": ["request"],
            },
        })
        bindings[name] = skill_id
    return tools, bindings


class SkillProfileDecider:
    """Proposal-only Needle 3 stage that selects one installed skill profile."""

    MAX_RESPONSE_CHARS = 32768

    def __init__(
        self,
        *,
        model: str = SKILL_PROFILE_DECIDER_MODEL,
        needle_factory: Any = None,
    ):
        self.model = model
        self.needle_factory = needle_factory

    def decide(
        self,
        *,
        user_text: str,
        capabilities: list[dict[str, Any]],
        allowed_skill_ids: list[str] | set[str],
    ) -> str:
        """Return the selected skill_id, or raise SkillProfileError (fail closed)."""
        if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > 1000:
            raise SkillProfileError("Normalized skill request was empty or exceeded its size limit.")
        allowed = {value for value in (allowed_skill_ids or []) if isinstance(value, str)}
        if not allowed:
            raise SkillProfileError("No runnable skill profiles were supplied to the decider.")
        tools, bindings = decider_tools(capabilities)
        try:
            os.environ["NEEDLE_TELEMETRY"] = "0"
            os.environ["DO_NOT_TRACK"] = "1"
            if self.needle_factory is None:
                import needle
                factory = needle.Needle
            else:
                factory = self.needle_factory
            with _NEEDLE_INFERENCE_LOCK:
                agent = factory(
                    tools=tools,
                    system=_DECIDER_SYSTEM_PROMPT[:MAX_DECIDER_PROMPT_CHARS],
                    generation=3,
                    stateless=True,
                    buffer_size=self.MAX_RESPONSE_CHARS,
                )
                try:
                    response = agent.complete(user_text, max_new_tokens=256)
                finally:
                    close = getattr(agent, "close", None)
                    if callable(close):
                        close()
        except SkillProfileError:
            raise
        except Exception as exc:
            raise SkillProfileError(
                f"Needle 3 unavailable: {type(exc).__name__}: {str(exc)[:180]}"
            ) from exc

        if not isinstance(response, dict) or response.get("success") is not True:
            raise SkillProfileError("The skill decider did not return a successful response.")
        if response.get("type") != "call":
            raise SkillProfileError("The skill decider did not return a profile selection.")
        suppressed = response.get("suppressed_calls")
        if not isinstance(suppressed, list) or suppressed:
            raise SkillProfileError(
                "The skill decider withheld the profile selection for confidence or grounding."
            )
        calls = response.get("function_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise SkillProfileError("The skill decider must select exactly one profile.")
        # Needle only attaches a validation block when calls are present.
        validation = response.get("validation")
        if calls and validation is not None:
            if not isinstance(validation, dict) or validation.get("negation"):
                raise SkillProfileError("The skill decider flagged a negated or invalid selection.")
            ungrounded = validation.get("ungrounded")
            if not isinstance(ungrounded, list):
                raise SkillProfileError("The skill decider omitted valid grounding validation.")
            if ungrounded:
                raise SkillProfileError("The skill decider flagged ungrounded arguments.")
        confidence = response.get("confidence")
        if (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence)) or not 0.1 <= float(confidence) <= 1.0
        ):
            raise SkillProfileError("The skill decider's confidence was below its supported range.")
        call = calls[0]
        name = call.get("name") if isinstance(call, dict) else None
        arguments = call.get("arguments") if isinstance(call, dict) else None
        selected = bindings.get(name) if isinstance(name, str) else None
        routed_text = arguments.get("request") if isinstance(arguments, dict) else None
        if (
            selected is None
            or set(arguments or {}) != {"request"}
            or not isinstance(routed_text, str)
            or not routed_text.strip()
            or len(routed_text) > 1000
        ):
            raise SkillProfileError("The skill decider selected an unknown profile or invalid routing arguments.")
        if selected not in allowed:
            raise SkillProfileError("The skill decider selected a profile that is not runnable for this request.")
        return selected
