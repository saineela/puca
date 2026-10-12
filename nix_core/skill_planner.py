"""Dedicated local Needle 3 client for structured skill-plan proposals."""
from __future__ import annotations

import math
import os
import re
import threading
from typing import Any

from config import SKILL_PLANNER_MODEL


_NEEDLE_INFERENCE_LOCK = threading.Lock()

# Shuffle and repeat are a music provider's queue preferences rather than queue
# contents: "shuffle the queue", "repeat this song", "stop repeating". The words
# a preference request is built from are checked against a small vocabulary, so a
# request that names a track ("search for shuffle") is left to the planner
# instead of being read as a queue operation. This mirrors the matcher in the
# installed OpenDeck worker, which answers the same phrases when it is the one
# asked, so speech and the page reach the same queue flag either way.
_MUSIC_PREFERENCE_WORD = re.compile(r"\b(?:shuffl\w*|randomi[sz]\w*|repeat\w*|loop\w*)\b", re.IGNORECASE)
_MUSIC_SHUFFLE_WORD = re.compile(r"\b(?:shuffl(?:e|es|ed|ing)|randomi[sz]e[ds]?)\b", re.IGNORECASE)
_MUSIC_REPEAT_WORD = re.compile(r"\b(?:repeat(?:ing|s)?|loop(?:ing)?)\b", re.IGNORECASE)
# "can you shuffle the queue", "shuffle the queue please": the same request
# with manners on it, so a polite frame comes off before the words are checked.
_MUSIC_PREFERENCE_LEAD = re.compile(
    r"^(?:(?:please|hey\s+nix|nix|ok|okay)[,\s]+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?",
    re.IGNORECASE,
)
_MUSIC_PREFERENCE_TRAIL = re.compile(
    r"(?:\s+(?:please|thanks|thank\s+you|for\s+me))+\s*$", re.IGNORECASE
)
_MUSIC_PREFERENCE_OFF_WORD = re.compile(r"\b(?:off|stop|no|cancel|end|un|without)\b", re.IGNORECASE)
_MUSIC_PREFERENCE_ONE_WORD = re.compile(
    r"\b(?:one|single|same|current|this|that|it|song|track|tune)\b", re.IGNORECASE
)
_MUSIC_PREFERENCE_ALL_WORD = re.compile(
    r"\b(?:all|everything|whole|entire|queue|list|playlist|music|songs|tracks|tunes|rest)\b",
    re.IGNORECASE,
)
_MUSIC_PREFERENCE_WORDS = frozenset(
    "shuffle shuffles shuffled shuffling randomize randomizes randomise randomises random randomly"
    " repeat repeating repeats loop loops looping"
    " on off stop start begin end cancel no more turn turning toggle switch"
    " it this that these those them one all everything whole rest same current"
    " queue list playlist music song songs track tracks tune tunes"
    " and then also now again please for me my our the a an of in to up back".split()
)
# "queue up shuffle" names a track that happens to be called Shuffle: an explicit
# add is read as one, and the preference matcher stays out of its way.
_MUSIC_QUEUE_ADD_WORD = re.compile(
    r"\bqueue\s+up\b|"
    r"\b(?:add|put|drop|throw|queue)\b[^,;]{0,60}?\b(?:to|into|onto)\s+(?:the|my)\s+queue\b",
    re.IGNORECASE,
)


def _music_queue_preference(text: str) -> dict[str, Any] | None:
    """A spoken shuffle/repeat request as queue arguments, or None.

    "shuffle the queue" and "turn shuffle off" set the queue's flag; "repeat this
    song", "repeat the queue" and "stop repeating" set its mode. Shuffle is
    switched on rather than toggled - the page's button is the toggle - and a bare
    "repeat" cycles, which is what the provider does with no mode given.
    """
    lowered = _MUSIC_PREFERENCE_TRAIL.sub(
        "", _MUSIC_PREFERENCE_LEAD.sub("", text.casefold()).strip()
    ).strip()
    if _MUSIC_PREFERENCE_WORD.search(lowered) is None:
        return None
    words = re.findall(r"[a-z']+", lowered)
    if not words or any(word not in _MUSIC_PREFERENCE_WORDS for word in words):
        return None
    if _MUSIC_QUEUE_ADD_WORD.search(lowered) is not None:
        return None
    shuffles = _MUSIC_SHUFFLE_WORD.search(lowered)
    repeats = _MUSIC_REPEAT_WORD.search(lowered)
    if repeats is None or (shuffles is not None and shuffles.start() < repeats.start()):
        return {"action": "queue", "op": "shuffle",
                "on": _MUSIC_PREFERENCE_OFF_WORD.search(lowered) is None}
    if _MUSIC_PREFERENCE_OFF_WORD.search(lowered) is not None:
        return {"action": "queue", "op": "repeat", "mode": "off"}
    if _MUSIC_PREFERENCE_ONE_WORD.search(lowered) is not None:
        return {"action": "queue", "op": "repeat", "mode": "one"}
    if _MUSIC_PREFERENCE_ALL_WORD.search(lowered) is not None:
        return {"action": "queue", "op": "repeat", "mode": "all"}
    return {"action": "queue", "op": "repeat"}


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
        # "stop speaking" and "stop the music" are the actions themselves, not a
        # negation of a plan, so they are removed before the negation guard runs.
        negation_text = re.sub(
            r"\bstop\s+(?:speaking|talking|playing|the\s+music|the\s+audio|the\s+song|"
            r"repeating|looping|shuffling|the\s+repeat|the\s+shuffle)\b",
            " ",
            text,
        )
        if re.search(r"\b(?:do not|don't|never|not|without|stop|avoid)\b", negation_text):
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
            # "change to <name> effect" and "switch to <name> effect" are
            # activation wordings (measured live on compound requests), not
            # catalog reads; only question/list wording selects "read".
            if re.search(r"\b(?:run|start|activate|play|set|use|apply|change|switch)\b", text):
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
        # A named effect from the live catalog plus any activation verb —
        # including "show"/"display" ("while showing the rainbow twinkle
        # effect") — is an effect run, never a catalog read: the read verbs
        # only win when no specific effect name is present ("show available
        # effects").
        if any(effect.casefold() in text for effect in available_effects) and re.search(
            r"\b(?:run|start|activate|play|set|use|apply|enable|turn|change|switch|"
            r"show(?:s|ing|n)?|display(?:s|ing)?|put)\b",
            text,
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
        ) and re.search(r"\b(?:turn|switch|power)\b", text) and _music_queue_preference(text) is None:
            # "turn shuffle off" and "turn off repeat" switch a queue preference,
            # not a device: without this they would narrow the request to a power
            # action the provider does not declare.
            groups.add("power")
        # Speaker wording. The speaker skill declares the ring's actions and the
        # speaker's in one tool, so a request that names one of them is narrowed
        # to that action: without this the planner is handed the whole toolset,
        # which is larger than its bounded tool limit and fails closed.
        if re.search(r"\b(?:volume|loudness|louder|quieter|sound\s+level|speaker\s+level)\b", text):
            groups.add("volume")
        if re.search(r"\bun-?mute\b", text):
            groups.add("unmute")
        elif re.search(r"\bmute\b", text):
            groups.add("mute")
        if re.search(r"\b(?:stop\s+speaking|stop\s+talking|be\s+quiet|shut\s+up)\b", text):
            groups.add("stop_speaking")
        elif re.search(r"\b(?:say|speak|announce|read\s+aloud)\b", text):
            groups.add("say")
        if re.search(r"\b(?:pause|unpause|resume)\b", text):
            groups.add("resume" if re.search(r"\b(?:unpause|resume)\b", text) else "pause")
        # Music provider wording (OpenDeck). A music request is narrowed to the
        # one provider action the words name, which is also what keeps the
        # planner's confidence up: the whole provider toolset is offered only
        # when the request names none of them.
        # A question about the current track ("what's playing", "what song is
        # playing") is a read of the provider's now-playing state, not a "play"
        # command: the "playing" in it would otherwise narrow the planner to the
        # play action, which has no valid call for a question and fails closed.
        if re.search(
            r"\b(?:what(?:'s|\s+song\s+is|\s+tune\s+is|\s+is)?\s+playing|"
            r"what\s+(?:song|tune)\s+is\s+this|now\s+playing|currently\s+playing|"
            r"what\s+am\s+i\s+listening\s+to)\b",
            text,
        ):
            groups.add("now_playing")
        elif re.search(r"\b(?:play|put\s+on)\b[^,;]{0,80}\b(?:next|first)\b", text):
            # "play <song> next/first" prioritises that track in the queue; the
            # track itself comes from the request wording, so Core owns the call.
            groups.add("queue")
        elif re.search(
            r"\bmove\b[^,;]{0,60}\b(?:track|song|tune|position|front)\b", text
        ):
            # "move track 3 to position 1" is an explicit queue reorder.
            groups.add("queue")
        elif _music_queue_preference(text) is not None:
            # "shuffle the queue", "repeat this song": the queue's own listening
            # preferences, which the provider drives by op. Read as a preference
            # instead of a play, and only when the wording is all preference
            # words, so a request that names a track is left to the planner.
            groups.add("queue")
        elif re.search(r"\b(?:queue|queued|up\s+next)\b", text):
            groups.add("queue")
        elif re.search(r"\b(?:play|playing|put\s+on|start\s+playing)\b", text):
            groups.add("play")
        elif re.search(r"\b(?:search|find|look\s+for)\b", text):
            # "search for X" / "find X": a catalog search, never a play. Naming
            # the provider's search action keeps the model from choosing play.
            groups.add("search")
        if re.search(
            r"\b(?:playlists?|liked|loved|favorites?|favourites?|history|library|"
            r"listened\s+to|been\s+listening|did\s+i\s+listen)\b",
            text,
        ):
            groups.add("library")
        if re.search(r"\blyrics?\b", text):
            groups.add("lyrics")
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
            # "read" is this vocabulary's word for a state/catalog read on a
            # device whose tool declares one. A provider whose read action is
            # named something else (OpenDeck answers a library listing) is not
            # an unsupported request: narrowing by that hint would hand the
            # planner an action the schema does not have, so the hint is
            # dropped and the skill's own toolset is offered instead.
            if missing == {"read"}:
                groups.discard("read")
                if not groups:
                    return None
                missing = groups - available_groups
        if missing:
            raise SkillPlannerError(
                "Requested action is unavailable in the installed schema: " + ", ".join(sorted(missing))
            )
        return groups

    @classmethod
    def _grounded_music_call(
        cls,
        capabilities: list[dict[str, Any]],
        groups: set[str],
        user_text: str,
    ) -> dict[str, Any] | None:
        """Build the music call the request names, or None when it is ambiguous."""
        if not groups:
            return None
        text = " ".join(str(user_text).casefold().split())
        action: str | None = None
        arguments: dict[str, Any] | None = None
        if "now_playing" in groups:
            # "what's playing" names no argument at all, so Core owns the call.
            action = "now_playing"
            arguments = {"action": "now_playing"}
        elif "search" in groups:
            # "search for X" / "find X" / "look for X": the search text is the
            # user's own words after the verb, never a model-chosen title.
            query = re.sub(
                r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:search|find|look)\s*(?:for|up)?\s+",
                "",
                text,
            )
            query = re.sub(r"\b(?:on|in)\s+opendeck\b", " ", query)
            query = re.sub(r"\b(?:please|thanks|thank\s+you)\b", " ", query)
            query = " ".join(query.split()).strip(" ,.!?")
            if query:
                action = "search"
                arguments = {"action": "search", "query": query[:200]}
        elif "queue" in groups:
            action = "queue"
            # "shuffle the queue", "repeat this song", "stop repeating": the
            # queue's own preferences, which the page drives by op.
            preference = _music_queue_preference(text)
            # "play <song> next/first" prioritises that track in the queue.
            priority = re.search(r"\b(?:play|put\s+on)\s+(?:the\s+)?(.+?)\s+(?:next|first)\b", text)
            # "move track 3 to position 1" / "move <song> to the front".
            move = re.search(
                r"\bmove\s+(?:track\s+|song\s+)?(\d+)\s+(?:to\s+)?(?:position\s+)?(\d+)\b",
                text,
            )
            move_front = re.search(r"\bmove\s+(?:track\s+|song\s+)?(.+?)\s+to\s+(?:the\s+)?front\b", text)
            # "play the next track"/"skip"/"next song": advance the queue. This
            # is checked before the priority wording so "play the next queued
            # track" is not read as "play <song> next".
            advance = re.search(
                r"\b(?:play\s+(?:the\s+)?next|next\s+(?:track|song|tune|one)|"
                r"skip|following)\b",
                text,
            )
            if preference is not None:
                arguments = preference
            elif (
                _MUSIC_PREFERENCE_WORD.search(text) is not None
                and _MUSIC_QUEUE_ADD_WORD.search(text) is None
            ):
                # A shuffle/repeat word Core cannot read exactly (a track is
                # named, or the wording is unusual): the call needs a lookup or a
                # judgement, so it is left to the planner rather than guessed at
                # as an add of the literal words.
                return None
            elif advance is not None:
                arguments = {"action": "queue", "op": "next"}
            elif priority is not None:
                track = " ".join(priority.group(1).split()).strip(" ,.!?")
                op = "next_up" if re.search(r"\bnext\b", priority.group(0)) else "front"
                if not track or re.fullmatch(
                    r"(?:this|that|it|the\s+one|(?:this|that|the|a|my)?\s*(?:current\s+)?"
                    r"(?:song|track|tune|one|music))",
                    track,
                ):
                    # "play this next": the track is the current one.
                    arguments = {"action": "queue", "op": op}
                else:
                    arguments = {"action": "queue", "op": op, "query": track[:200]}
            elif move_front is not None:
                track = " ".join(move_front.group(1).split()).strip(" ,.!?")
                if re.fullmatch(r"(?:\d+\s+)?(?:track|song|tune)", track):
                    index = int(re.match(r"(\d+)", track).group(1)) - 1
                    arguments = {"action": "queue", "op": "front", "index": max(0, index)}
                elif track:
                    arguments = {"action": "queue", "op": "front", "query": track[:200]}
            elif move is not None:
                arguments = {
                    "action": "queue",
                    "op": "move",
                    "index": max(0, int(move.group(1)) - 1),
                    "to": max(0, int(move.group(2)) - 1),
                }
            elif re.search(r"\b(?:clear|empty|wipe)\b", text):
                arguments = {"action": "queue", "op": "clear"}
            elif re.search(r"\b(?:what|show|list|read|see|any|remaining)\b", text):
                arguments = {"action": "queue", "op": "list"}
            else:
                # "queue up X", "add X to the queue", "queue X": the track is the
                # user's own words between the verb and the queue itself.
                track = re.sub(
                    r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:queue|add|put)\s+(?:up\s+)?",
                    "",
                    text,
                )
                track = re.sub(r"\s*(?:,?\s*)?(?:to|onto|into)\s+(?:the|my)\s+queue\b.*$", "", track)
                track = track.strip(" ,.!?")
                if track:
                    arguments = {"action": "queue", "op": "add", "query": track[:200]}
        elif "library" in groups:
            if re.search(r"\b(?:liked|loved|favorite|favourite)s?\b", text):
                arguments = {"action": "library", "kind": "likes"}
                action = "library"
            elif re.search(
                r"\b(?:history|recently\s+played|recent\s+plays?|listened\s+to|been\s+listening|"
                r"did\s+i\s+listen)\b",
                text,
            ):
                arguments = {"action": "library", "kind": "history"}
                action = "library"
            elif re.search(r"\b(?:stats?|statistics)\b", text):
                arguments = {"action": "library", "kind": "stats"}
                action = "library"
            # A singular "playlist" usually names one to open, which the planner
            # resolves from the request; only a listing is Core-owned.
            elif re.search(r"\bplaylists\b|\bmusic\s+library\b", text):
                arguments = {"action": "library", "kind": "playlists"}
                action = "library"
        elif "lyrics" in groups:
            # Lyrics for a track the user names go to the planner, which fills the
            # title/artist fields; Core only owns the no-argument case.
            if not re.search(r"\b(?:for|of|to)\s+\S", text):
                action = "lyrics"
                arguments = {"action": "lyrics"}
        if action is None or arguments is None:
            return None
        _tools, bindings = cls._needle_tool_definitions(capabilities, {action}, {action})
        for _name, (skill_id, declared_tool, action_map) in bindings.items():
            if action_map.get("__call__") == action:
                return {
                    "type": "skill_tool_call",
                    "skill_id": skill_id,
                    "tool": declared_tool,
                    "arguments": arguments,
                }
        return None

    _STEP_SPLIT_PATTERN = re.compile(
        r"\s*(?:\band\s+then\b|\bthen\b|\bafter\s+that\b|\bwhile\b|\balso\b|"
        r"\bplus\b|\band\b|;|,)\s*",
        re.IGNORECASE,
    )
    _STEP_ACTION_PATTERN = re.compile(
        r"\b(?:turn|switch|power|set|make|change|apply|enable|disable|start|stop|"
        r"brighten|dim|run|play|animate|activate|show|showing|on|off)\b",
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
        # The connector gate must list every split connector: a compound
        # request phrased with "while"/"also" ("switch the color to green
        # while showing the rainbow twinkle effect") previously reached one
        # multi-call inference, where Needle3's confidence collapses and the
        # whole plan is withheld.
        if not re.search(
            r"\band\b|\bthen\b|\bafter\s+that\b|\bwhile\b|\balso\b|\bplus\b|;|,",
            user_text,
            re.IGNORECASE,
        ):
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
        volume_grounding: dict[str, Any] | None = None,
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
                        volume_grounding=volume_grounding,
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
            volume_grounding=volume_grounding,
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
        volume_grounding: dict[str, Any] | None = None,
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
                "volume" if volume_grounding is not None else None,
                "effect" if effect_grounding is not None else None,
            }
            grounded_groups.discard(None)
            if grounded_groups and (selected_groups is None or selected_groups <= grounded_groups):
                # Fill lexical gaps only. In step-by-step planning every step
                # receives the full request's groundings; widening a step's
                # own evidence to the full grounded set ("change to rainbow
                # twinkle effect" gaining "color" from step 1) re-creates
                # exactly the multi-call inference this split exists to
                # avoid, and Needle3 truncates on it (measured live).
                if selected_groups is None:
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
        # The speaker's volume is the same Core-owned fraction as brightness and
        # lives on the same device, so it bypasses the model for the same reason
        # (Needle3 suppresses a correctly computed 0.3 level).
        if (
            force_grounded
            and volume_grounding is not None
            and selected_groups == {"volume"}
            and isinstance(volume_grounding.get("volume"), (int, float))
        ):
            _tools, bindings = self._needle_tool_definitions(capabilities, {"volume"}, None)
            for _name, (skill_id, declared_tool, action_map) in bindings.items():
                if action_map.get("__call__") == "volume":
                    return [{
                        "type": "skill_tool_call",
                        "skill_id": skill_id,
                        "tool": declared_tool,
                        "arguments": {
                            "action": "volume",
                            "volume": float(volume_grounding["volume"]),
                        },
                    }]
            raise SkillPlannerError("No declared volume function was available for the grounded volume step.")
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
        # Power: an explicitly worded on/off step is Core-owned wording (the
        # cleanser normalizes it to "turn on/off the <device>"), so under the
        # grounded retry it is emitted deterministically — mirrored on the
        # color/brightness/effect bypasses. Needle3 measurably withholds the
        # zero-argument power call when it is planned as a second step
        # ("set the color to red and turn on the light"), which failed the
        # whole compound plan closed. The confirmation and readback gates
        # still apply to the executed result.
        if force_grounded and selected_groups == {"power"}:
            direction = None
            if re.search(
                r"\b(?:turn|switch|power)\b.{0,60}\boff\b"
                r"|\boff\b.{0,40}\b(?:light|ring|lamp|device)\b",
                user_text,
                re.IGNORECASE,
            ):
                direction = "off"
            elif re.search(
                r"\b(?:turn|switch|power)\b.{0,60}\bon\b"
                r"|\bon\b.{0,40}\b(?:light|ring|lamp|device)\b",
                user_text,
                re.IGNORECASE,
            ):
                direction = "on"
            if direction is not None:
                _tools, bindings = self._needle_tool_definitions(
                    capabilities, {"power"}, None
                )
                for _name, (skill_id, declared_tool, action_map) in bindings.items():
                    if action_map.get("__call__") == direction:
                        return [{
                            "type": "skill_tool_call",
                            "skill_id": skill_id,
                            "tool": declared_tool,
                            "arguments": {"action": direction},
                        }]
                raise SkillPlannerError(
                    "No declared power function was available for the grounded power step."
                )
        # Zero-argument speaker actions (mute/unmute, stop speaking, pause/resume)
        # carry no model-chosen argument at all, so under the grounded retry they
        # are emitted deterministically like power — the validation and readback
        # gates still apply to the executed result.
        if (
            force_grounded
            and selected_groups is not None
            and len(selected_groups) == 1
            and next(iter(selected_groups)) in {"mute", "unmute", "stop_speaking", "pause", "resume"}
        ):
            action = next(iter(selected_groups))
            _tools, bindings = self._needle_tool_definitions(capabilities, {action}, None)
            for _name, (skill_id, declared_tool, action_map) in bindings.items():
                if action_map.get("__call__") == action:
                    return [{
                        "type": "skill_tool_call",
                        "skill_id": skill_id,
                        "tool": declared_tool,
                        "arguments": {"action": action},
                    }]
            raise SkillPlannerError(
                f"No declared {action} function was available for the grounded speaker step."
            )
        # Speech: the words to speak are the user's own, taken from after the
        # verb, so Core owns the argument instead of the model paraphrasing it.
        if force_grounded and selected_groups == {"say"}:
            speech = re.search(
                r"\b(?:say|speak|announce|read\s+aloud)\b[:,]?\s+(.+)$",
                user_text,
                re.IGNORECASE,
            )
            if speech is not None:
                text = speech.group(1).strip().strip("\"'").rstrip(".!? ").strip()
                if text:
                    _tools, bindings = self._needle_tool_definitions(capabilities, {"say"}, None)
                    for _name, (skill_id, declared_tool, action_map) in bindings.items():
                        if action_map.get("__call__") == "say":
                            return [{
                                "type": "skill_tool_call",
                                "skill_id": skill_id,
                                "tool": declared_tool,
                                "arguments": {"action": "say", "text": text[:500]},
                            }]
                    raise SkillPlannerError(
                        "No declared say function was available for the grounded speech step."
                    )
        # Music provider (OpenDeck) requests. Its actions are named after the
        # operations they perform and the arguments come from the request itself
        # (a track to queue, a kind of listing), so the whole call is Core-owned
        # for the same reason a brightness level is: the model is not asked to
        # choose between two plausible actions (Measured: "queue up <song>" was
        # planned as a search) and cannot lose the call to a confidence floor.
        # Anything Core cannot read (opening a named playlist, saving a like) is
        # left to the planner below.
        if selected_groups is not None and selected_groups <= {
            "queue", "library", "lyrics", "now_playing", "search"
        }:
            grounded_music = self._grounded_music_call(capabilities, selected_groups, user_text)
            if grounded_music is not None:
                return [grounded_music]
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
            and not (color_grounding or brightness_grounding or effect_grounding or volume_grounding)
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
                    response = agent.complete(user_text, max_new_tokens=1024)
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
