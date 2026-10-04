"""Dedicated local Needle 3 client for structured skill-plan proposals."""
from __future__ import annotations

import math
import os
import re
import threading
from typing import Any

from config import SKILL_PLANNER_MODEL


_NEEDLE_INFERENCE_LOCK = threading.Lock()


class SkillPlannerError(RuntimeError):
    """The dedicated planner was unavailable or did not return valid calls."""


class SkillPlanner:
    """Use local Needle 3 as a proposal-only planner; never execute returned calls."""

    MAX_CALLS = 8
    MAX_RESPONSE_CHARS = 32768

    def __init__(
        self,
        *,
        model: str = SKILL_PLANNER_MODEL,
        needle_factory: Any = None,
    ):
        self.model = model
        self.needle_factory = needle_factory

    @staticmethod
    def _tool_definitions(
        capabilities: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str | None]]]:
        """Expose schema-declared actions separately before compacting for Needle."""
        tools: list[dict[str, Any]] = []
        aliases: dict[str, tuple[str, str, str | None]] = {}
        for skill_index, capability in enumerate(capabilities[:8]):
            skill_id = capability.get("skill_id")
            declared_tools = capability.get("tools")
            if not isinstance(skill_id, str) or not isinstance(declared_tools, list):
                continue
            for declared in declared_tools[:32]:
                if not isinstance(declared, dict):
                    continue
                tool_name = declared.get("name")
                schema = declared.get("input_schema")
                if (
                    not isinstance(tool_name, str)
                    or re.fullmatch(r"[A-Za-z0-9_-]{1,48}", tool_name) is None
                    or not isinstance(schema, dict)
                ):
                    continue
                variants = schema.get("oneOf")
                variants = variants if isinstance(variants, list) else [schema]
                for variant_index, variant in enumerate(variants[:16]):
                    if not isinstance(variant, dict) or variant.get("type") != "object":
                        continue
                    properties = variant.get("properties")
                    properties = dict(properties) if isinstance(properties, dict) else {}
                    required = variant.get("required", [])
                    required = list(required) if isinstance(required, list) else []
                    action_schema = properties.get("action")
                    actions = action_schema.get("enum") if isinstance(action_schema, dict) else None
                    if not isinstance(actions, list) or not actions or not all(isinstance(x, str) for x in actions):
                        actions = [None]
                    device = capability.get("live_device")
                    effects = device.get("available_effects") if isinstance(device, dict) else None
                    effects = list(dict.fromkeys(
                        value.strip()[:64] for value in effects
                        if isinstance(value, str) and value.strip()
                    ))[:64] if isinstance(effects, list) else []
                    for action_index, action in enumerate(actions):
                        if action == "effect" and not effects:
                            continue
                        action_properties = dict(properties)
                        action_required = list(required)
                        if action is not None:
                            action_properties.pop("action", None)
                            action_required = [name for name in action_required if name != "action"]
                        if action == "effect":
                            effect_schema = action_properties.get("effect")
                            if not isinstance(effect_schema, dict) or effect_schema.get("type") != "string":
                                continue
                            effect_schema = dict(effect_schema)
                            effect_schema["enum"] = effects
                            action_properties["effect"] = effect_schema
                        if action in {"color", "brightness"}:
                            key = "rgb" if action == "color" else "brightness"
                            value_schema = action_properties.get(key)
                            if isinstance(value_schema, dict):
                                action_properties[key] = dict(value_schema)
                                action_properties[key].pop("description", None)
                        alias = re.sub(
                            r"[^A-Za-z0-9_-]", "_",
                            f"skill_{skill_index}_{tool_name}_{action or f'variant{variant_index}_{action_index}'}",
                        )[:64]
                        if alias in aliases:
                            raise SkillPlannerError("Installed skill functions produced a duplicate tool name.")
                        guidance = {
                            "color": "Green is RGB [0,255,0]; retain any explicit RGB or hex value.",
                            "brightness": "Brightness is a fraction from 0.05 through 1.0.",
                            "effect": "Choose an exact effect from the live firmware catalog: " + ", ".join(effects) + ".",
                        }.get(action, f"Use for {action}.") if action else "Include only arguments requested by the user."
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": alias,
                                "description": (str(declared.get("description") or "Perform the declared device action.") + " " + guidance)[:900],
                                "parameters": {
                                    "type": "object",
                                    "properties": action_properties,
                                    "required": action_required,
                                },
                            },
                        })
                        aliases[alias] = (skill_id, tool_name, action)
        if not tools:
            raise SkillPlannerError("No declared function schemas were available to the skill planner.")
        return tools, aliases

    @classmethod
    def _needle_tool_definitions(
        cls,
        capabilities: list[dict[str, Any]],
        selected_groups: set[str] | None = None,
        selected_actions: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, dict[str, str]]]]:
        """Convert declared actions to compact Needle tools and their Core bindings."""
        if selected_groups is not None and not selected_groups:
            raise SkillPlannerError("Could not identify a supported requested action.")
        declared, aliases = cls._tool_definitions(capabilities)
        grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], str | None]]] = {}
        for item in declared:
            func = item["function"]
            skill_id, tool_name, action = aliases[func["name"]]
            group = (
                "power" if action in {"on", "off"}
                else "read" if action in {"catalog", "color_catalog", "state"}
                else action or func["name"]
            )
            grouped.setdefault((skill_id, tool_name, group), []).append((func, action))

        short_names = {
            "color": "set_color", "power": "set_power", "brightness": "set_brightness",
            "effect": "set_effect", "read": "read_ring",
        }
        action_names = {
            "on": "turn_on", "off": "turn_off",
            "catalog": "list_effects", "color_catalog": "list_colors",
            "state": "read_state",
        }
        action_descriptions = {
            "on": "Turn the ring on.",
            "off": "Turn the ring off.",
            "catalog": "List the available firmware effects, patterns, or animations on the ring.",
            "color_catalog": "List the colors supported by the ring firmware. This is not the effects or animations catalog.",
            "state": "Read the ring's current power, brightness, color, and effect state.",
        }
        tools: list[dict[str, Any]] = []
        bindings: dict[str, tuple[str, str, dict[str, str]]] = {}
        for (skill_id, tool_name, group), members in grouped.items():
            if selected_groups is not None and group not in selected_groups:
                continue
            # Measured: Needle3's confidence collapses on the manifest's generic
            # multi-purpose description; a clean, single-purpose color tool
            # description keeps the correct call above the suppression floor.
            first_capability = next(
                (item for item in capabilities if item.get("skill_id") == skill_id),
                {},
            )
            device_label = str(
                first_capability.get("device_name") or first_capability.get("name") or "device"
            )[:48]
            group_descriptions = {
                "color": (
                    f"Set the {device_label} color to one RGB triplet of three "
                    "channels, each 0-255. Copy the exact RGB numbers provided "
                    "in the request."
                ),
                "brightness": (
                    f"Set the {device_label} brightness to one fraction from "
                    "0.05 through 1.0. Copy the exact fraction provided in the "
                    "request."
                ),
                "effect": (
                    f"Run one exact firmware effect on the {device_label}. "
                    "Copy the exact effect name provided in the request."
                ),
            }
            for func, action in members:
                if selected_actions is not None and action not in selected_actions:
                    continue
                if group in {"power", "read"} and action is not None:
                    name = action_names[action]
                    props = {}
                    required = []
                    mapping = {"__call__": action}
                    description = action_descriptions[action]
                else:
                    props = func["parameters"].get("properties", {})
                    required = func["parameters"].get("required", [])
                    mapping = {"__call__": action} if action is not None else {}
                    description = group_descriptions.get(group) or func["description"]
                    base_name = short_names.get(group, "do_" + re.sub(r"[^a-z0-9_]+", "_", group.casefold()).strip("_")[:24])
                    name = base_name
                    suffix = 2
                    while name in bindings:
                        name = f"{base_name}_{suffix}"
                        suffix += 1
                tools.append({
                    "name": name,
                    "description": description[:900],
                    "parameters": {"type": "object", "properties": props, "required": required},
                })
                bindings[name] = (skill_id, tool_name, mapping)
        if not tools:
            raise SkillPlannerError("No declared function schemas matched the requested action.")
        if len(tools) > 8:
            raise SkillPlannerError("The installed skill toolset exceeded Needle's bounded planner limit.")
        return tools, bindings

    @staticmethod
    def _requested_tool_groups(
        user_text: str,
        capabilities: list[dict[str, Any]],
    ) -> set[str] | None:
        """Narrow choices only when unambiguous surface wording maps to declared actions."""
        text = user_text.casefold()
        if re.search(r"\b(?:do not|don't|never|not|without|stop|avoid)\b", text):
            raise SkillPlannerError("Negated or excluded device requests are not eligible for planning.")
        groups: set[str] = set()
        color = re.search(
            r"\b(?:color|colour|hue|shade|rgb|hex|green|blue|red|purple|violet|pink|orange|yellow|cyan|teal|white|amber)\b|#[0-9a-f]{6}",
            text,
        )
        if color and re.search(r"\b(?:set|change|make|switch|turn|apply|paint)\b", text):
            if not (re.search(r"\b(?:turn|switch|power)\b.{0,40}\boff\b", text)
                    and not re.search(r"\b(?:color|colour|hue|shade|rgb|hex)\b|\bto\s+(?:green|blue|red|purple|violet|pink|orange|yellow|cyan|teal|white|amber)\b|#[0-9a-f]{6}", text)):
                groups.add("color")
        if re.search(r"\b(?:brightness|bright|brighten|dim|dimmer|darker|lighter)\b", text) and re.search(
            r"\b(?:set|change|adjust|raise|lower|increase|decrease|brighten|dim|make)\b", text
        ):
            groups.add("brightness")
        if re.search(r"\b(?:effect|animation|pattern)\b", text):
            if re.search(r"\b(?:run|start|activate|play|set|use|apply)\b", text):
                groups.add("effect")
            elif re.search(r"\b(?:list|show|read|what|available|catalog)\b", text):
                groups.add("read")
        elif re.search(r"\b(?:list|show|read|what|which|tell|available|catalog)\b", text):
            groups.add("read")
        if re.search(r"\b(?:animations?|effects?|patterns?)\b", text) and re.search(
            r"\b(?:list|show|read|what|which|tell|available|catalog)\b", text
        ):
            groups.add("read")
        available_effects = [
            effect for capability in capabilities
            for effect in (
                capability.get("live_device", {}).get("available_effects", [])
                if isinstance(capability.get("live_device"), dict) else []
            )
            if isinstance(effect, str) and effect.strip()
        ]
        if any(effect.casefold() in text for effect in available_effects) and re.search(
            r"\b(?:run|start|activate|play|set|use|apply|enable|turn)\b", text
        ):
            groups.add("effect")
        if re.search(r"\b(?:state|status|current state)\b", text) and re.search(
            r"\b(?:show|read|check|what|current|report|get)\b", text
        ):
            groups.add("read")
        if re.search(r"\b(?:catalog|available|list|show|read)\b", text) and re.search(
            r"\b(?:color|colour)\b", text
        ):
            groups.add("read")
        if (
            re.search(r"\b(?:turn|switch|power)\b.{0,60}\b(?:on|off)\b", text)
            or re.search(r"\b(?:on|off)\b.{0,40}\b(?:light|ring|device)\b", text)
        ) and re.search(r"\b(?:turn|switch|power)\b", text):
            groups.add("power")
        if not groups:
            return None
        declared, aliases = SkillPlanner._tool_definitions(capabilities)
        available_groups = set()
        for alias in aliases.values():
            action = alias[2]
            available_groups.add(
                "power" if action in {"on", "off"}
                else "read" if action in {"catalog", "color_catalog", "state"}
                else action
            )
        missing = groups - available_groups
        if missing:
            raise SkillPlannerError(
                "Requested action is unavailable in the installed schema: " + ", ".join(sorted(missing))
            )
        return groups

    _STEP_SPLIT_PATTERN = re.compile(
        r"\s*(?:\band\s+then\b|\bthen\b|\bafter\s+that\b|\band\b|;|,)\s*",
        re.IGNORECASE,
    )
    _STEP_ACTION_PATTERN = re.compile(
        r"\b(?:turn|switch|power|set|make|change|apply|enable|disable|start|stop|"
        r"brighten|dim|run|play|animate|activate|on|off)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _split_plan_steps(cls, user_text: str) -> list[str] | None:
        """Split a compound request so each step gets its own single-call run.

        Needle3's confidence metric is well-calibrated for one call (~0.12-0.74
        when grounded) but collapses on multi-call responses (~0.007 even for a
        fully correct plan), which suppressed correct multi-action proposals.
        Planning step-by-step keeps every call in the calibrated regime; Core's
        multi-action count gate still requires the complete step set.
        """
        if not re.search(r"\band\b|\bthen\b|\bafter\s+that\b|;|,", user_text, re.IGNORECASE):
            return None
        fragments = [
            fragment.strip(" .!?")
            for fragment in cls._STEP_SPLIT_PATTERN.split(user_text)
            if fragment.strip(" .!?")
        ]
        steps = [
            fragment for fragment in fragments
            if cls._STEP_ACTION_PATTERN.search(fragment)
        ]
        if len(steps) >= 2:
            return steps[: cls.MAX_CALLS]
        return None

    def plan(
        self,
        *,
        system_prompt: str,
        user_text: str,
        capabilities: list[dict[str, Any]],
        color_grounding: dict[str, Any] | None = None,
        brightness_grounding: dict[str, Any] | None = None,
        effect_grounding: dict[str, Any] | None = None,
        force_grounded: bool = False,
    ) -> list[dict[str, Any]]:
        """Ask Needle for calls and return proposals; Core owns validation and execution."""
        if not isinstance(capabilities, list) or len(capabilities) > 8:
            raise SkillPlannerError("Skill capability list exceeded its bounds.")
        if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > 1000:
            raise SkillPlannerError("Normalized skill request was empty or exceeded its size limit.")
        if not isinstance(system_prompt, str) or len(system_prompt) > 4000:
            raise SkillPlannerError("Skill planner instructions exceeded their size limit.")
        steps = self._split_plan_steps(user_text)
        if steps is not None:
            proposals: list[dict[str, Any]] = []
            for step in steps:
                proposals.extend(
                    self._plan_single(
                        system_prompt=system_prompt,
                        user_text=step,
                        capabilities=capabilities,
                        color_grounding=color_grounding,
                        brightness_grounding=brightness_grounding,
                        effect_grounding=effect_grounding,
                        force_grounded=force_grounded,
                    )
                )
            return proposals[: self.MAX_CALLS]
        return self._plan_single(
            system_prompt=system_prompt,
            user_text=user_text,
            capabilities=capabilities,
            color_grounding=color_grounding,
            brightness_grounding=brightness_grounding,
            effect_grounding=effect_grounding,
            force_grounded=force_grounded,
        )

    def _plan_single(
        self,
        *,
        system_prompt: str,
        user_text: str,
        capabilities: list[dict[str, Any]],
        color_grounding: dict[str, Any] | None = None,
        brightness_grounding: dict[str, Any] | None = None,
        effect_grounding: dict[str, Any] | None = None,
        force_grounded: bool = False,
    ) -> list[dict[str, Any]]:
        """One bounded Needle run proposing calls for one request step."""
        selected_groups = self._requested_tool_groups(user_text, capabilities)
        # Core-resolved arguments define the bypass groups even when the
        # surface wording ("chartreuse", "a bit brighter") falls outside the
        # regex tool-group detector: the grounding exists only because Core
        # already matched that argument class in the request.
        if force_grounded:
            grounded_groups = {
                "color" if color_grounding is not None else None,
                "brightness" if brightness_grounding is not None else None,
                "effect" if effect_grounding is not None else None,
            }
            grounded_groups.discard(None)
            if grounded_groups and (selected_groups is None or selected_groups <= grounded_groups):
                selected_groups = grounded_groups
        # Core-resolved canonical colors never pass through the model: Needle3
        # garbles 3-digit numbers ([255, 0, 0] -> [2, 55, 0]), so a named-color
        # step is emitted deterministically and still passes schema validation
        # plus the color-matcher and device-readback gates.
        if (
            force_grounded
            and color_grounding is not None
            and selected_groups == {"color"}
            and isinstance(color_grounding.get("rgb"), list)
            and isinstance(color_grounding.get("color_name"), str)
        ):
            _tools, bindings = self._needle_tool_definitions(capabilities, {"color"}, None)
            for _name, (skill_id, declared_tool, action_map) in bindings.items():
                if action_map.get("__call__") == "color":
                    return [{
                        "type": "skill_tool_call",
                        "skill_id": skill_id,
                        "tool": declared_tool,
                        "arguments": {
                            "action": "color",
                            "rgb": [int(channel) for channel in color_grounding["rgb"]],
                        },
                    }]
            raise SkillPlannerError("No declared color function was available for the grounded color step.")
        # Same for brightness: Needle3 computes '50 percent' -> 0.5 correctly
        # but suppresses the call at ~0.0004 confidence, so Core-owned levels
        # bypass the model and still pass schema validation and readback.
        if (
            force_grounded
            and brightness_grounding is not None
            and selected_groups == {"brightness"}
            and isinstance(brightness_grounding.get("brightness"), (int, float))
        ):
            _tools, bindings = self._needle_tool_definitions(capabilities, {"brightness"}, None)
            for _name, (skill_id, declared_tool, action_map) in bindings.items():
                if action_map.get("__call__") == "brightness":
                    return [{
                        "type": "skill_tool_call",
                        "skill_id": skill_id,
                        "tool": declared_tool,
                        "arguments": {
                            "action": "brightness",
                            "brightness": float(brightness_grounding["brightness"]),
                        },
                    }]
            raise SkillPlannerError("No declared brightness function was available for the grounded brightness step.")
        # Effects: the exact firmware name from the live catalog is Core-owned
        # data; case-matching it here removes the last ungrounded argument
        # class from the model's hands.
        if (
            force_grounded
            and effect_grounding is not None
            and selected_groups == {"effect"}
            and isinstance(effect_grounding.get("effect"), str)
            and effect_grounding["effect"].strip()
        ):
            _tools, bindings = self._needle_tool_definitions(capabilities, {"effect"}, None)
            for _name, (skill_id, declared_tool, action_map) in bindings.items():
                if action_map.get("__call__") == "effect":
                    return [{
                        "type": "skill_tool_call",
                        "skill_id": skill_id,
                        "tool": declared_tool,
                        "arguments": {
                            "action": "effect",
                            "effect": effect_grounding["effect"],
                        },
                    }]
            raise SkillPlannerError("No declared effect function was available for the grounded effect step.")
        selected_actions = None
        normalized_text = user_text.casefold()
        if selected_groups is not None and "read" in selected_groups:
            if re.search(r"\b(?:effects?|animations?|patterns?)\b", normalized_text):
                selected_actions = {"catalog"}
            elif re.search(r"\b(?:colors?|colours?|rgb)\b", normalized_text):
                # "what color is the ring right now" asks for the current
                # state; "what colors are available" asks for the catalog.
                if re.search(r"\bcolors\b|\bcolours\b", normalized_text) or not re.search(
                    r"\b(?:is|are|right now|currently)\b", normalized_text
                ):
                    selected_actions = {"color_catalog"}
                else:
                    selected_actions = {"state"}
            elif re.search(r"\b(?:state|status)\b", normalized_text):
                selected_actions = {"state"}
        # Reads are side-effect-free, and Needle3's validation block is
        # nondeterministic on question wording (measured live: omitted
        # validation, false-positive negation flags). Under the grounded
        # retry, a read-only request is emitted deterministically; the
        # confirmation and readback gates still apply to the executed result.
        if (
            force_grounded
            and selected_groups == {"read"}
            and not (color_grounding or brightness_grounding or effect_grounding)
        ):
            read_action = next(iter(selected_actions)) if selected_actions else "state"
            _tools, bindings = self._needle_tool_definitions(capabilities, {"read"}, {read_action})
            for _name, (skill_id, declared_tool, action_map) in bindings.items():
                if action_map.get("__call__") == read_action:
                    return [{
                        "type": "skill_tool_call",
                        "skill_id": skill_id,
                        "tool": declared_tool,
                        "arguments": {"action": read_action},
                    }]
            raise SkillPlannerError("No declared read function was available for the grounded read step.")
        tools, bindings = self._needle_tool_definitions(
            capabilities, selected_groups, selected_actions
        )
        if len(user_text.encode("utf-8")) > self.MAX_RESPONSE_CHARS // 2:
            raise SkillPlannerError("Normalized skill request exceeded Needle's input buffer bound.")
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
                    system=system_prompt.strip(),
                    generation=3,
                    stateless=True,
                    buffer_size=self.MAX_RESPONSE_CHARS,
                )
                try:
                    response = agent.complete(user_text, max_new_tokens=512)
                finally:
                    close = getattr(agent, "close", None)
                    if callable(close):
                        close()
        except Exception as exc:
            raise SkillPlannerError(f"Needle 3 unavailable: {type(exc).__name__}: {str(exc)[:180]}") from exc

        if not isinstance(response, dict) or response.get("success") is not True:
            raise SkillPlannerError(
                "Needle 3 did not return a successful structured response: "
                + repr(response)[:220]
            )
        if response.get("type") != "call":
            raise SkillPlannerError(
                "Needle 3 did not return a function-call response: " + repr(response)[:220]
            )
        suppressed = response.get("suppressed_calls")
        validation = response.get("validation")
        if not isinstance(suppressed, list) or suppressed:
            raise SkillPlannerError("Needle 3 withheld one or more calls for confidence or grounding.")
        if not isinstance(validation, dict):
            # Fail closed: an unvalidated proposal must not execute. Needle3
            # measurably omits the validation block on some question wordings;
            # the grounded read retry below covers the side-effect-free cases.
            raise SkillPlannerError("Needle 3 omitted valid grounding validation.")
        # A negation flag on a plainly non-control read request is a measured
        # false positive ("what color is the device"): Core's own planner gate
        # already rejects true negations ("do not turn ... on") before Needle
        # runs, so a disagreement here only blocks verified reads. Honor the
        # model flag only when Core also sees a negation cue, or the request
        # looks like a control request.
        if validation.get("negation"):
            control_shaped = bool(re.search(
                r"\b(?:turn|switch|power|set|change|make|apply|paint|enable|disable|activate|run|start|play|stop|open|close|brighten|dim)\b",
                user_text,
                re.IGNORECASE,
            ))
            core_sees_negation = bool(re.search(
                r"\b(?:do\s+not|don't|never|not|without|avoid|stop)\b",
                user_text,
                re.IGNORECASE,
            ))
            if core_sees_negation or control_shaped:
                raise SkillPlannerError("Needle 3 flagged a negated or invalid proposal.")
        ungrounded = validation.get("ungrounded")
        if not isinstance(ungrounded, list):
            raise SkillPlannerError("Needle 3 omitted valid grounding validation.")
        if ungrounded:
            raise SkillPlannerError("Needle 3 flagged ungrounded arguments.")
        confidence = response.get("confidence")
        if (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence)) or not 0.1 <= float(confidence) <= 1.0
        ):
            raise SkillPlannerError("Needle 3 call confidence was below its supported range.")
        calls = response.get("function_calls")
        if not isinstance(calls, list) or not 1 <= len(calls) <= self.MAX_CALLS:
            raise SkillPlannerError("Needle 3 did not return 1 to 8 declared function calls.")
        proposals: list[dict[str, Any]] = []
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else None
            arguments = call.get("arguments") if isinstance(call, dict) else None
            binding = bindings.get(name) if isinstance(name, str) else None
            if binding is None or not isinstance(arguments, dict):
                raise SkillPlannerError("Needle 3 selected an undeclared function or invalid arguments.")
            skill_id, declared_tool, action_map = binding
            full_arguments = dict(arguments)
            if "__call__" in action_map:
                if "action" in full_arguments:
                    raise SkillPlannerError("Needle 3 supplied an action field outside its function schema.")
                full_arguments["action"] = action_map["__call__"]
            elif action_map:
                action = full_arguments.pop("action", None)
                if not isinstance(action, str) or action not in action_map:
                    raise SkillPlannerError("Needle 3 selected an undeclared action variant.")
                full_arguments["action"] = action_map[action]
            proposals.append({
                "type": "skill_tool_call", "skill_id": skill_id,
                "tool": declared_tool, "arguments": full_arguments,
            })
        return proposals
