from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation import LogitsProcessorList

from .context import TemporalContext
from .engine import KnowledgeEngine
from .functions.calendar import (
    cancel_calendar_event,
    create_calendar_event,
    find_calendar_events,
    update_calendar_event,
)
from .functions.facts import create_fact
from .temporal import TemporalResolver

try:
    from nix_actions.engine import ActionsEngine
except ImportError:  # nix_actions is an optional sibling package
    ActionsEngine = None


# ---------------------------------------------------------------------------
# Calendar-domain guard
# ---------------------------------------------------------------------------

_CALENDAR_DOMAIN_RE = re.compile(
    r"\b("
    r"event|events|meeting|appointment|appointments|schedule|scheduled|"
    r"calendar|remind(?:er|ers)?|reminder|plan|plans|planned|"
    r"agenda|deadline|due|reservation|booking|"
    r"today|tomorrow|tmr|tmrw|yesterday|tonight|weekend|week|month|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"am|pm|morning|afternoon|evening"
    r")\b",
    re.I,
)


_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = os.environ.get(
    "NIX_KNOWLEDGE_MODEL_PATH",
    str(_MODEL_DIR / "qwen2.5-0.5b-instruct"),
)

# Terminal editing artifacts (arrow-key history recall) leak escape
# sequences into pasted/typed requests. They must never reach the
# Knowledge Base or the model prompt.
_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])"
)


class _StopAfterToolCall(StoppingCriteria):
    """
    Halt generation as soon as the closing </tool_call> tag appears.

    The selector emits exactly one tool call; without this criterion
    the model keeps "writing" until the token cap, multiplying GPU
    latency (and risking garbage after the parseable call).
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.marker = "</tool_call>"
        self.prompt_length = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if not self.prompt_length:
            return False

        tail = self.tokenizer.decode(
            input_ids[0][self.prompt_length:],
            skip_special_tokens=False,
        )
        return self.marker in tail


class _ForceFirstTokenProcessor(LogitsProcessor):
    """
    Forces generation to begin with a specific token.

    A small selector model sometimes drifts into conversational text,
    which can never be parsed as a function selection. Beginning every
    response with '<tool_call>' keeps it on the function-call pattern;
    the remaining tokens are still chosen by the model itself.
    """

    def __init__(self, token_id: int):
        super().__init__()
        self.token_id = token_id
        self.done = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        if not self.done:
            self.done = True
            scores[:] = float("-inf")
            scores[:, self.token_id] = 0.0

        return scores


# ---------------------------------------------------------------------------
# Knowledge response boundary
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeResponse:
    """
    Authoritative response returned by Nix Knowledge to Nix Core.

    Knowledge does NOT formulate the user-facing response.

    Core receives:
      - original user request
      - selected Knowledge function
      - arguments
      - authoritative function result
      - analysis_required

    Core is responsible for interpreting this result and speaking to the user.
    """

    user_request: str
    function: dict[str, Any] | None
    result: Any
    analysis_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Knowledge Needle
# ---------------------------------------------------------------------------

class KnowledgeNeedle:
    """
    Qwen-powered natural-language interface for Nix Knowledge.

    Qwen selects the appropriate Knowledge function.

    Python remains authoritative for:
      - validation
      - temporal resolution
      - database access
      - record lookup
      - mutation
      - record IDs
      - persistence

    Knowledge returns structured findings to Nix Core.
    Knowledge does NOT generate the final user-facing answer.
    """

    def __init__(
        self,
        engine: KnowledgeEngine,
        timezone: str = "America/Chicago",
        actions_engine: Any | None = None,
    ):
        self.engine = engine
        self.temporal = TemporalResolver(timezone=timezone)
        self.context = TemporalContext(timezone=timezone)
        self.timezone = timezone
        self.actions_engine = actions_engine

        print("Loading Qwen2.5 0.5B...")

        # This API normally runs separately from Core/Casper. Keep its model
        # on an explicit small GPU budget so both services can stay warm. A
        # device-map limit is safer than a process-wide allocator fraction:
        # the console may embed the HTTP handler in the same Python process.
        max_memory = None
        if torch.cuda.is_available():
            max_memory = {
                0: os.environ.get("NIX_KNOWLEDGE_GPU_MEMORY", "2GiB"),
                "cpu": os.environ.get("NIX_KNOWLEDGE_CPU_MEMORY", "16GiB"),
            }

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        load_kwargs: dict[str, Any] = {
            "dtype": "auto",
            "device_map": "auto",
            "local_files_only": True,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            **load_kwargs,
        )

        self.model.eval()

        self.tools = self._build_tool_schemas()
        self.tool_functions = self._build_tool_functions()

        print("Qwen2.5 0.5B loaded.")

    # ------------------------------------------------------------------
    # Tool schemas
    # ------------------------------------------------------------------

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_calendar_event",
                    "description": (
                        "Create a NEW calendar event. Use this when the user "
                        "states that an event exists, is scheduled, is planned, "
                        "or explicitly asks to add/schedule/create an event. "
                        "Do NOT use this for questions about existing events."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": (
                                    "The event title. Do not invent words "
                                    "that the user did not provide."
                                ),
                            },
                            "temporal_expression": {
                                "type": "string",
                                "description": (
                                    "The exact time/date expression supplied "
                                    "by the user, such as 'tomorrow', "
                                    "'Friday at 6pm', or 'next Monday'. "
                                    "Never invent a time."
                                ),
                            },
                        },
                        "required": [
                            "title",
                            "temporal_expression",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_calendar_events",
                    "description": (
                        "Find EXISTING calendar events. Use this for questions "
                        "about what is already scheduled, checking whether an "
                        "event exists, viewing the calendar, or locating an "
                        "event before changing or cancelling it. Use 'window' "
                        "whenever the user names a time frame, as a natural "
                        "expression like 'today', 'tomorrow', 'this week', "
                        "'next week', 'this weekend', or a specific day - "
                        "never convert it to dates yourself."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": (
                                    "Event title or identifying words from "
                                    "the user's request. Omit if none."
                                ),
                            },
                            "window": {
                                "type": "string",
                                "description": (
                                    "Natural-language time frame for the "
                                    "search, exactly as the user said it: "
                                    "'today', 'tomorrow', 'this week', "
                                    "'next week', 'this weekend', 'on friday'. "
                                    "Omit only when the user asks about the "
                                    "calendar as a whole."
                                ),
                            },
                            "start": {
                                "type": "string",
                                "description": (
                                    "Start boundary if the user gave an exact "
                                    "ISO-8601 moment. Prefer 'window'."
                                ),
                            },
                            "end": {
                                "type": "string",
                                "description": (
                                    "End boundary if the user gave an exact "
                                    "ISO-8601 moment. Prefer 'window'."
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_calendar_event",
                    "description": (
                        "Update an EXISTING calendar event. This represents "
                        "a requested modification such as moving, renaming, "
                        "or changing the time of an existing event. The "
                        "Python layer must locate and validate the existing "
                        "record before mutation. Never invent a record ID."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": (
                                    "Existing event title or identifying "
                                    "text from the user."
                                ),
                            },
                            "new_title": {
                                "type": "string",
                            },
                            "new_temporal_expression": {
                                "type": "string",
                                "description": (
                                    "New time/date expression exactly as "
                                    "stated by the user."
                                ),
                            },
                        },
                        "required": [
                            "title",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_calendar_event",
                    "description": (
                        "Cancel an EXISTING calendar event. Use when the user "
                        "asks to cancel, delete, remove, get rid of, or no "
                        "longer wants an existing event. The Python layer "
                        "must locate the event. Never invent a record ID."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": (
                                    "Existing event title or identifying "
                                    "text from the user."
                                ),
                            },
                            "reason": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "title",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_fact",
                    "description": (
                        "Store a durable factual statement that Nix should "
                        "remember. Use for explicit memory requests such as "
                        "'remember that I like robotics'. Do not use this "
                        "for questions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": (
                                    "The factual statement to remember. "
                                    "Preserve the user's meaning and do not "
                                    "invent additional facts."
                                ),
                            },
                        },
                        "required": [
                            "value",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_facts",
                    "description": (
                        "Find facts already stored in Knowledge. Use this when "
                        "the user asks what Nix remembers, whether Nix "
                        "remembers something, or what Nix knows about a "
                        "topic/person/preference/fact. This is READ-ONLY."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "The topic or statement to search for. "
                                    "Use the user's relevant subject. "
                                    "Omit for broad memory retrieval."
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "store_profile_keys",
                    "description": (
                        "Store profile keys the user stated about themselves: "
                        "their name, nickname, birthday, interests, or goals "
                        "(as in: my name is Sai, I am born on February 25 "
                        "2009). Use ONLY when the sentence is mainly a "
                        "self-introduction; keys are also extracted "
                        "automatically from every other request."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "array",
                                "description": "Key objects from extract_keys.",
                                "items": {"type": "object"},
                            },
                        },
                        "required": ["keys"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_state",
                    "description": (
                        "Record the CURRENT STATE of a person close to the "
                        "user (sister, brother, mom, dad, grandma, grandpa, "
                        "friend, ...): health, mood, energy or stress. Use "
                        "when the user says something like 'my sister is "
                        "sick', 'my mom is happy today', 'maanvi's cured "
                        "now', 'my dad is tired'. States CHANGE over time "
                        "- this is different from a permanent fact. The "
                        "previous contradicting state is replaced "
                        "automatically."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "statement": {
                                "type": "string",
                                "description": (
                                    "The state statement, keeping the "
                                    "person reference and the state word, "
                                    "e.g. 'my sister maanvi is sick with "
                                    "flu' or 'maanvi is cured now'."
                                ),
                            },
                        },
                        "required": ["statement"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_states",
                    "description": (
                        "Show the CURRENT STATES of people close to the "
                        "user (how is my sister, is my mom feeling better, "
                        "who is sick, how is everyone doing)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Optional person or filter words from "
                                    "the user's question, e.g. 'sister' or "
                                    "'maanvi'. Omit to list everyone."
                                ),
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool functions
    # ------------------------------------------------------------------

    def _build_tool_functions(self) -> dict[str, Any]:
        return {
            "create_calendar_event": self._create_calendar_event,
            "find_calendar_events": self._find_calendar_events,
            "update_calendar_event": self._update_calendar_event,
            "cancel_calendar_event": self._cancel_calendar_event,
            "create_fact": self._create_fact,
            "find_facts": self._find_facts,
            "store_profile_keys": self._store_profile_keys,
            "create_state": self._create_state,
            "find_states": self._find_states,
        }

    # ------------------------------------------------------------------
    # Current-state tools (close people)
    # ------------------------------------------------------------------

    def _create_state(self, statement: str) -> dict[str, Any]:
        """Store a current state for a close person, superseding the
        previous contradicting state (sick -> cured)."""
        from .states import parse_state_statement, store_state

        parsed = parse_state_statement(statement)
        if parsed is None:
            raise ValueError(
                "Not a recognizable current-state statement about a "
                "close person."
            )
        result = store_state(self.engine, parsed, statement)
        # register entities so later "my sister" queries resolve
        if result.get("data", {}).get("name") and result.get(
            "data", {}
        ).get("role"):
            try:
                self.engine.semantic.entities.register(
                    result["data"]["role"],
                    result["data"]["name"],
                    record_id=result.get("record_id"),
                    evidence=statement[:160],
                )
            except Exception:  # noqa: BLE001
                pass
        return result

    def _find_states(self, query: str | None = None) -> dict[str, Any]:
        """List current states of close people (optionally filtered)."""
        from .states import find_states

        return find_states(self.engine, query)

    # ------------------------------------------------------------------
    # Function implementations
    # ------------------------------------------------------------------

    def _create_calendar_event(
        self,
        title: str,
        temporal_expression: str,
    ) -> dict[str, Any]:

        if not title.strip():
            raise ValueError("Calendar event title cannot be empty.")

        if not temporal_expression.strip():
            raise ValueError(
                "Calendar event temporal_expression cannot be empty."
            )

        # Direct callers are protected by the same neuro-symbolic gate as
        # the routed process path. The proposal is only a candidate; the
        # symbolic parser supplies the authoritative title/expression/date.
        from .temporal_hybrid import parse_event, TemporalProposal

        parsed = parse_event(
            f"{title} {temporal_expression}",
            self.temporal,
            proposal=TemporalProposal(
                title=title,
                expression=temporal_expression,
                confidence=1.0,
                source="direct_tool",
            ),
        )
        if parsed is None:
            raise ValueError(
                "Could not validate the calendar time symbolically: "
                f"'{temporal_expression}'."
            )

        title = parsed.title
        temporal_expression = parsed.expression
        resolved = parsed.resolved

        result = create_calendar_event(
            self.engine,
            title=title,
            start=resolved.start.isoformat(),
            end=resolved.end.isoformat(),
            all_day=resolved.all_day,
            temporal_expression=temporal_expression,
            recurring=resolved.recurring,
            recurrence=resolved.recurrence,
            source="knowledge_qwen",
            reason="Created from natural-language user request.",
        )

        response = {
            "operation": "CREATE",
            "record_type": "event",
            "record_id": result.get("record_id"),
            "data": result.get("data"),
            "temporal": {
                "original_expression": temporal_expression,
                "resolved_start": resolved.start.isoformat(),
                "resolved_end": resolved.end.isoformat(),
                "all_day": resolved.all_day,
                "recurring": resolved.recurring,
                "recurrence": resolved.recurrence,
                "when": self.context.describe(
                    start=resolved.start,
                    end=resolved.end
                    if resolved.end > resolved.start
                    else None,
                    all_day=resolved.all_day,
                ),
            },
        }

        # Hand the scheduled intent to the Actions engine when one is
        # attached. Knowledge stores the fact; Actions triggers it.
        if self.actions_engine is not None:
            from .bridge import schedule_event_actions

            response["actions"] = schedule_event_actions(
                self.actions_engine,
                record_id=result.get("record_id"),
                data=result.get("data"),
            )

        return response

    def _find_calendar_events(
        self,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        window: str | None = None,
        include_cancelled: bool | None = None,
    ) -> dict[str, Any]:
        return find_calendar_events(
            self.engine,
            title=title,
            start=start,
            end=end,
            window=window,
            include_cancelled=include_cancelled,
            context=self.context,
        )

    def _find_facts(
        self,
        query: str | None = None,
    ) -> dict[str, Any]:

        records = self.engine.search("fact")

        query_normalized = (
            query.strip().lower()
            if query
            else None
        )

        # -------------------------------------------------------
        # Referent resolution: "my sister" must connect to the
        # registered name (maanvi), pulled from the entity registry
        # built at ingest time. Resolved names participate in
        # matching AND are returned so Core can speak them.
        # -------------------------------------------------------
        resolved: list[tuple[str, str]] = []
        if query_normalized and self.engine.semantic is not None:
            try:
                entity_map = self.engine.semantic.entities.all()
            except Exception:
                entity_map = {}
            for match in re.finditer(
                r"\bmy\s+(sister|brother|mom|mother|dad|father|wife|"
                r"husband|daughter|son|cousin|uncle|aunt|grandma|"
                r"grandpa|friend|girlfriend|boyfriend|partner|"
                r"roommate|boss|manager|dog|cat|pet)\b",
                query_normalized,
            ):
                role = match.group(1)
                name = entity_map.get(role)
                if name:
                    resolved.append((role, name))

        resolved_names = {name for _, name in resolved}

        # Polarity/state-change words describe a CHANGE, not the
        # subject; they must not break word-overlap matching.
        polarity_words = {
            "not", "no", "isnt", "isn't", "aint", "wasnt", "wasn't",
            "arent", "aren't", "werent", "weren't", "dont", "don't",
            "doesnt", "doesn't", "didnt", "didn't", "cant", "can't",
            "wont", "won't", "never", "anymore", "longer", "stopped",
            "still", "yet", "again", "now",
        }
        state_change = bool(
            query_normalized
            and (polarity_words & set(re.findall(r"[\w']+", query_normalized)))
        )

        # Word-overlap matching: "alarm unlock code" must hit "my
        # alarm unlock code is 1238" even though the exact substring
        # differs. A fact matches when (a) the query is a substring,
        # (b) every query word appears in the fact, or (c) a resolved
        # entity name appears in the fact (referent hit).
        query_words = (
            set(re.findall(r"[\w']+", query_normalized))
            if query_normalized else set()
        )

        results = []

        for record in records:
            value = str(
                record.data.get("value", "")
            )

            value_words = (
                set(re.findall(r"[\w']+", value.lower()))
                if query_normalized else set()
            )

            name_hit = bool(resolved_names & value_words)

            if query_normalized and not name_hit:
                value_lower = value.lower()
                if query_normalized not in value_lower:
                    core_query_words = (
                        query_words - polarity_words
                    )
                    if not core_query_words.issubset(value_words):
                        continue

            results.append({
                "record_id": record.id,
                "value": value,
                "confidence": record.confidence,
                "certainty": record.certainty,
                "source": record.source,
                "status": record.status,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "kind": "fact",
                "about": sorted(resolved_names & value_words),
            })

        # -------------------------------------------------------
        # Keys ride along: micro-facts about people in the user's
        # life ("user's sister is named Maanvi") surface in recall
        # as well. Matching ignores question/function words so
        # "who is Maanvi" and "what do you know about my sister"
        # hit the right keys.
        # -------------------------------------------------------
        stop = {
            "what", "who", "when", "where", "why", "how", "which",
            "is", "are", "was", "were", "do", "does", "did",
            "the", "a", "an", "my", "me", "i", "you", "your",
            "know", "about", "tell", "of", "for", "to", "in",
            "and", "that", "there", "it", "its",
            # polarity words: state changes, not identity
            "not", "no", "isnt", "isn't", "wasnt", "wasn't",
            "anymore", "longer", "never", "stopped", "still",
        }
        key_query_words = query_words - stop if query_words else set()
        # resolved names always count as key-relevant words
        key_query_words |= {
            name for _, name in resolved
        }

        for record in self.engine.search("key"):
            value = str(record.data.get("value", ""))
            if not value:
                continue

            value_lower = value.lower()
            value_words = set(re.findall(r"[\w']+", value_lower))

            if query_normalized:
                if query_normalized not in value_lower:
                    if not (
                        key_query_words.issubset(value_words)
                        or resolved_names & value_words
                    ):
                        continue

            results.append({
                "record_id": record.id,
                "value": value,
                "confidence": record.confidence,
                "certainty": record.certainty,
                "source": record.source,
                "status": record.status,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "kind": "key",
                "about": sorted(resolved_names & value_words),
            })

        return {
            "ok": True,
            "count": len(results),
            "query": query,
            "resolved_entities": [
                {"role": role, "name": name}
                for role, name in resolved
            ],
            "state_change_detected": state_change,
            "facts": results,
        }

    def _create_fact(
        self,
        value: str,
    ) -> dict[str, Any]:

        if not value.strip():
            raise ValueError(
                "Fact value cannot be empty."
            )

        result = create_fact(
            self.engine,
            value=value,
            source="knowledge_qwen",
            reason="Created from natural-language user request.",
        )

        if result.get("duplicate"):
            # Near-exact restatement: the original record already
            # holds this knowledge. Report the dedup so Core can say
            # "already noted" instead of pretending it is new.
            return {
                "operation": "DUPLICATE",
                "record_type": "fact",
                "record_id": result.get("record_id"),
                "data": result.get("data"),
                "duplicate_of": result.get("duplicate_of"),
            }

        # register any (role, name) entities from the fact text so
        # later referential queries ("my sister") resolve to names
        if self.engine.semantic is not None:
            try:
                from .semantic.entities import extract_entities

                for entity in extract_entities(value):
                    self.engine.semantic.entities.register(
                        entity.role,
                        entity.name,
                        record_id=result.get("record_id"),
                        evidence=value[:160],
                    )
            except Exception:
                pass

        return {
            "operation": "CREATE",
            "record_type": "fact",
            "record_id": result.get("record_id"),
            "data": result.get("data"),
        }

    # ------------------------------------------------------------------
    # Mutation orchestration
    # ------------------------------------------------------------------

    def _find_event_for_mutation(
        self,
        title: str,
    ) -> dict[str, Any]:

        result = find_calendar_events(
            self.engine,
            title=title,
        )

        if not result.get("ok"):
            raise RuntimeError(
                "Calendar lookup failed."
            )

        events = result.get(
            "events",
            [],
        )

        # Cancelled events are historical records; when a live event
        # with the same title exists, mutations should target it.
        live = [
            event
            for event in events
            if event.get("status", "scheduled") != "cancelled"
        ]

        if live:
            events = live

        if len(events) == 0:
            return {
                "ok": False,
                "status": "not_found",
                "events": [],
            }

        if len(events) > 1:
            return {
                "ok": False,
                "status": "ambiguous",
                "events": events,
            }

        return {
            "ok": True,
            "status": "found",
            "event": events[0],
        }

    def _update_calendar_event(
        self,
        title: str,
        new_title: str | None = None,
        new_temporal_expression: str | None = None,
    ) -> dict[str, Any]:

        lookup = self._find_event_for_mutation(
            title
        )

        if not lookup["ok"]:
            return {
                "ok": False,
                "operation": "UPDATE",
                "lookup": lookup,
            }

        event = lookup["event"]
        record_id = event["record_id"]

        start = None
        end = None
        temporal_expression = None

        if new_temporal_expression:
            # Updates use the same symbolic validation as creates. Never
            # reschedule an action from an unvalidated neural expression.
            from .temporal_hybrid import parse_event, TemporalProposal

            parsed = parse_event(
                f"{title} {new_temporal_expression}",
                self.temporal,
                proposal=TemporalProposal(
                    title=title,
                    expression=new_temporal_expression,
                    confidence=1.0,
                    source="direct_tool",
                ),
            )
            if parsed is None:
                return {
                    "ok": False,
                    "operation": "UPDATE",
                    "error": (
                        "The new calendar time could not be validated "
                        "symbolically; no change was made."
                    ),
                }

            resolved = parsed.resolved
            start = resolved.start.isoformat()
            end = resolved.end.isoformat()
            temporal_expression = parsed.expression

        result = update_calendar_event(
            self.engine,
            record_id=record_id,
            title=new_title,
            start=start,
            end=end,
            temporal_expression=temporal_expression,
            reason=(
                "Updated from natural-language user request."
            ),
        )

        response = {
            "operation": "UPDATE",
            "lookup": event,
            "result": result,
        }

        # Keep Actions in sync: reschedule when the time moved,
        # cancel actions when the event was cancelled.
        if self.actions_engine is not None and result.get("ok"):
            from .bridge import (
                cancel_event_actions,
                schedule_event_actions,
            )

            if result.get("data", {}).get("status") == "cancelled":
                response["actions"] = cancel_event_actions(
                    self.actions_engine,
                    record_id=record_id,
                    reason="source event cancelled",
                )
            elif start is not None:
                self.actions_engine.cancel(
                    source_record_id=record_id,
                    reason="event rescheduled",
                )
                response["actions"] = schedule_event_actions(
                    self.actions_engine,
                    record_id=record_id,
                    data=result.get("data", {}),
                )

        return response

    def _cancel_calendar_event(
        self,
        title: str,
        reason: str | None = None,
    ) -> dict[str, Any]:

        lookup = self._find_event_for_mutation(
            title
        )

        if not lookup["ok"]:
            return {
                "ok": False,
                "operation": "CANCEL",
                "lookup": lookup,
            }

        event = lookup["event"]
        record_id = event["record_id"]

        result = cancel_calendar_event(
            self.engine,
            record_id=record_id,
            reason=reason,
        )

        response = {
            "operation": "CANCEL",
            "lookup": event,
            "result": result,
        }

        if self.actions_engine is not None and result.get("ok"):
            from .bridge import cancel_event_actions

            response["actions"] = cancel_event_actions(
                self.actions_engine,
                record_id=record_id,
                reason="source event cancelled",
            )

        return response

    # ------------------------------------------------------------------
    # Model prompt
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return """
You are Nix Knowledge's function selector.

You respond ONLY with function calls wrapped in <tool_call></tool_call> tags.
You never write conversational text.

Available functions:

- create_calendar_event(title, temporal_expression)
  Create a NEW calendar event. Use when the user states that an event
  exists, is scheduled, is planned, or asks to add/schedule/create one.
- find_calendar_events(title, window, start, end)
  Find EXISTING calendar events. Use for questions about what is already
  scheduled. Whenever the user names a time frame, pass it in 'window'
  exactly as they said it ('today', 'tomorrow', 'this week', 'next
  week', 'this weekend', 'on friday'); Python resolves it. Never
  convert expressions into absolute dates yourself.
- update_calendar_event(title, new_title, new_temporal_expression)
  Update an EXISTING event: move, rename, or change its time.
- cancel_calendar_event(title, reason)
  Cancel an EXISTING event: cancel, delete, remove.
- create_state(statement)
  Record the CURRENT STATE of someone close to the user (health,
  mood, energy, stress): 'my sister is sick', 'my mom is happy',
  'maanvi is cured now'. States change over time; the old state is
  replaced automatically. Use this instead of create_fact when the
  sentence describes how a person currently IS rather than who they
  are or what they have.
- find_states(query)
  Show current states of close people ('how is my sister', 'who is
  sick').
- multi_action(actions)
  Use this ONLY when one user message contains two or more independent
  Knowledge intents. The actions array must contain independently
  selected function calls, each with its own name and arguments. Never
  merge unrelated titles, people, or time expressions into one action.
- create_fact(value)
  Remember a durable fact or preference: "remember that I like
  robotics", "I love programming but hate robotics", "I am in TSA".
- find_facts(query)
  Find remembered facts and preferences: "what do you remember
  about X", "do I like robotics".

STRICT RULES:

1. "I have a meeting tomorrow" is the user stating a NEW event exists:
   use create_calendar_event, NOT find_calendar_events.

2. Never invent information. Never invent times, dates, titles,
   record IDs, reasons, or event details.

3. Preserve temporal expressions from the user.
   If the user says "tomorrow", use "tomorrow".
   If the user says "next Monday at 6pm", use exactly that.
   Do NOT convert expressions into absolute dates or clocks.

4. Only supply parameters the user actually gave you.
   Omit optional parameters you have no value for.

5. Statements about the user's likes, dislikes, hobbies, memberships
   or preferences ("I love X", "I hate Y", "I am in Z") are facts:
   use create_fact, preserving the user's own words. Never store them
   as calendar events and never look them up with find_facts.

6. "When is my <event>" and "what time is <event>" ask about an
   EXISTING event: use find_calendar_events with its title.

7. Multi-intent requests are valid. For example, a message that says
   "I have robotics tomorrow and remember that I joined TSA" must return
   multi_action with one create_calendar_event action and one create_fact
   action. Keep each action's arguments local to its clause.

8. Do not confuse a conversational question with a Knowledge intent.
   World questions, jokes, explanations, and general advice should not
   be routed here. If a request contains both world chat and personal
   Knowledge, return only the Knowledge action(s).

9. If the request is genuinely ambiguous and no safe function can be
   selected, respond with:
   <tool_call>
   {"name": "find_facts", "arguments": {}}
   </tool_call>

The pattern for every response is:
<tool_call>
{"name": <function-name>, "arguments": {<args>}}
</tool_call>""".strip()

    def _few_shot_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "I have a dentist appointment tomorrow"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "create_calendar_event", '
                    '"arguments": {"title": "dentist appointment", '
                    '"temporal_expression": "tomorrow"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "I have a meeting with Bob tomorrow at 3pm"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "create_calendar_event", '
                    '"arguments": {"title": "meeting with Bob", '
                    '"temporal_expression": "tomorrow at 3pm"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "what do you remember about robotics"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_facts", '
                    '"arguments": {"query": "robotics"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "do you remember anything about keyboards"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_facts", '
                    '"arguments": {"query": "keyboards"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": "remember that I like robotics",
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "create_fact", '
                    '"arguments": {"value": "I like robotics"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": "what is on my calendar",
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_calendar_events", '
                    '"arguments": {}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "I have robotics every sunday, "
                    "remember to wake me up"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "create_calendar_event", '
                    '"arguments": {"title": "robotics", '
                    '"temporal_expression": "every sunday"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "do I have a car inspection coming up"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_calendar_events", '
                    '"arguments": {"title": "car inspection"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "cancel my haircut appointment"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "cancel_calendar_event", '
                    '"arguments": {"title": "haircut appointment"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "reschedule my team standup to monday at 9am"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "update_calendar_event", '
                    '"arguments": {"title": "team standup", '
                    '"new_temporal_expression": "monday at 9am"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "I hate robotics, but love programming"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "create_fact", '
                    '"arguments": {"value": '
                    '"I hate robotics, but love programming"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "what events do I have today"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_calendar_events", '
                    '"arguments": {"window": "today"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "I have robotics practice tomorrow at 5pm and remember "
                    "that I joined TSA"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "multi_action", "arguments": {"actions": '
                    '[{"name": "create_calendar_event", "arguments": '
                    '{"title": "robotics practice", '
                    '"temporal_expression": "tomorrow at 5pm"}}, '
                    '{"name": "create_fact", "arguments": '
                    '{"value": "I joined TSA"}}]}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "What is on my schedule next week, and do you remember "
                    "what I like about robotics?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "multi_action", "arguments": {"actions": '
                    '[{"name": "find_calendar_events", "arguments": '
                    '{"window": "next week"}}, {"name": "find_facts", '
                    '"arguments": {"query": "robotics"}}]}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Tell me a joke, and remind me what appointment I have tomorrow"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_calendar_events", "arguments": '
                    '{"window": "tomorrow"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "what is on my schedule next week"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_calendar_events", '
                    '"arguments": {"window": "next week"}}\n'
                    '</tool_call>'
                ),
            },
            {
                "role": "user",
                "content": (
                    "when is my tsa meeting"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    '<tool_call>\n'
                    '{"name": "find_calendar_events", '
                    '"arguments": {"title": "tsa meeting"}}\n'
                    '</tool_call>'
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Qwen generation
    # ------------------------------------------------------------------

    def _generate_tool_call(
        self,
        user_request: str,
    ) -> dict[str, Any] | None:

        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            *self._few_shot_messages(),
            {
                "role": "user",
                "content": user_request,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value in inputs.items()
        }

        tool_call_token = self.tokenizer.convert_tokens_to_ids(
            "<tool_call>"
        )

        processors = None

        if isinstance(tool_call_token, int) and tool_call_token >= 0:
            processors = LogitsProcessorList(
                [
                    _ForceFirstTokenProcessor(tool_call_token)
                ]
            )

        stopper = _StopAfterToolCall(self.tokenizer)
        stopper.prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )

        generated = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]

        raw = self.tokenizer.decode(
            generated,
            skip_special_tokens=False,
        )

        return self._parse_tool_call(raw)

    # ------------------------------------------------------------------
    # Tool parser
    # ------------------------------------------------------------------

    def _parse_tool_call(
        self,
        raw: str,
    ) -> dict[str, Any] | None:

        match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            raw,
            re.DOTALL,
        )

        if not match:
            return None

        payload_text = match.group(1)

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = self._repair_tool_call_json(
                payload_text
            )

        if payload is None:
            return None

        if not isinstance(payload, dict):
            return None

        name = payload.get("name")
        arguments = payload.get(
            "arguments",
            {},
        )

        if name != "multi_action" and name not in self.tool_functions:
            return None

        if name == "multi_action" and not isinstance(arguments, dict):
            return None
        if name == "multi_action" and not isinstance(arguments.get("actions"), list):
            return None

        if not isinstance(arguments, dict):
            return None

        # Drop arguments that are empty strings so that optional
        # parameters the model filled with "" behave as omitted.
        arguments = {
            key: value
            for key, value in arguments.items()
            if value != ""
        }

        return {
            "name": name,
            "arguments": arguments,
        }

    # ------------------------------------------------------------------
    # Tool call JSON repair
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_tool_call_json(
        payload_text: str,
    ) -> dict[str, Any] | None:
        """
        Recover the common malformed pattern a small model produces
        when it omits the "arguments" key:

            {"name": "find_calendar_events", {}}

        A bare object following the name is treated as the arguments.
        """

        repaired = re.sub(
            r'("name"\s*:\s*"[^"]+")\s*,\s*(\{)',
            r'\1, "arguments": \2',
            payload_text,
            count=1,
        )

        if repaired == payload_text:
            return None

        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    # Argument safety
    # ------------------------------------------------------------------

    def _validate_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        allowed = {
            tool["function"]["name"]: tool["function"]
            for tool in self.tools
        }

        schema = allowed.get(name)

        if schema is None:
            raise ValueError(
                f"Unknown Knowledge function: {name}"
            )

        properties = schema["parameters"].get(
            "properties",
            {},
        )

        required = schema["parameters"].get(
            "required",
            [],
        )

        for key in arguments:
            if key not in properties:
                raise ValueError(
                    f"Unexpected argument '{key}' for {name}"
                )

        for key in required:
            if key not in arguments:
                raise ValueError(
                    f"Missing required argument '{key}' for {name}"
                )

        return arguments

    def _store_profile_keys(
        self,
        keys: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Deterministic profile storage for self-introduction requests.

        The Key Finding Algorithm always runs on every utterance, but a
        pure self-introduction ("my name is Sai ... wish to pursue
        Computer Engineering") carries no other actionable intent: the
        model gate would otherwise hallucinate a function (it guessed
        find_calendar_events) and the reply would be about nothing.
        This function stores the already-extracted keys and reports
        them so Core can acknowledge the introduction.
        """
        from .keys import key_text, store_keys

        stored = store_keys(
            self.engine,
            keys,
            actions_engine=self.actions_engine,
            timezone=self.context.timezone,
        )
        return {
            "ok": True,
            "operation": "STORE_KEYS",
            "count": len(stored),
            "keys": [key_text(k) for k in stored],
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _prepare_calendar_arguments(
        self,
        request: str,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None, str | None]:
        """Run every calendar mutation through the hybrid safety gate.

        Tool selection may be neural or symbolic, but the final temporal
        expression is always repaired and validated by Python before a
        create/update can reach the database or action bridge.
        """
        if name == "create_calendar_event":
            from .temporal_hybrid import repair_calendar_arguments

            repaired, parsed = repair_calendar_arguments(
                request,
                arguments,
                self.temporal,
            )
            if parsed is None:
                return (
                    arguments,
                    None,
                    "The calendar time could not be validated symbolically; "
                    "no event was created.",
                )
            return repaired, parsed, None

        if name == "update_calendar_event" and arguments.get(
            "new_temporal_expression"
        ):
            from .temporal_hybrid import validate_calendar_update_arguments

            repaired, parsed = validate_calendar_update_arguments(
                request,
                arguments,
                self.temporal,
            )
            if parsed is None:
                return (
                    arguments,
                    None,
                    "The new calendar time could not be validated "
                    "symbolically; no change was made.",
                )
            return repaired, parsed, None

        return arguments, None, None

    # ------------------------------------------------------------------
    # Temporal envelope
    # ------------------------------------------------------------------

    def _temporal_envelope(self) -> dict[str, Any]:
        """
        The shared time frame attached to every response so Core can
        interpret "today"-style results without recomputing time.
        """
        now = self.context.now()

        tomorrow = now.date() + timedelta(days=1)

        return {
            # These values are authoritative and are generated once by
            # Knowledge in its configured timezone. Core/Casper must not
            # resolve relative words against their own machine clock.
            "now": now.isoformat(),
            "timezone": self.timezone,
            "current_date": now.date().isoformat(),
            "current_time": now.strftime("%H:%M:%S"),
            "current_datetime_display": now.strftime(
                "%A, %B %-d, %Y at %-I:%M %p"
            ),
            "today": now.date().isoformat(),
            "tomorrow": tomorrow.isoformat(),
            "weekday": now.strftime("%A"),
            "week_start": (now.date() - timedelta(days=now.weekday())).isoformat(),
            "weekend_start": (
                now.date() + timedelta(days=(5 - now.weekday()) % 7)
            ).isoformat(),
        }

    def process(
        self,
        user_request: str,
    ) -> dict[str, Any]:
        """
        Full pipeline entry: Key Finding FIRST, then routing.

        Keys are micro-facts gleaned from ANY utterance, on EVERY
        request, before any routing decision ("my sister, named
        Maanvi is very naughty" -> has-sister, named-Maanvi,
        is-naughty). They are stored immediately and ride along in
        the response under result["keys_found"] - even when the
        routed operation itself fails or finds nothing.
        """
        # ---- 0. injection guard (defense in depth): never store,
        # never read, never route injection text. nix_core/router.py
        # normally filters these before they get here; this mirrors it
        # for direct API callers (knowledge_api /process, ws_server).
        try:
            from .guards import is_injection

            if is_injection(user_request):
                from .guards import REFUSAL_MESSAGE

                return KnowledgeResponse(
                    user_request=user_request,
                    function=None,
                    result={
                        "ok": False,
                        "status": "refused_injection",
                        "reply": REFUSAL_MESSAGE,
                    },
                    analysis_required=False,
                ).to_dict()
        except Exception:  # noqa: BLE001 - guard must never break replies
            pass

        # ---- 1. key extraction: unconditional, before routing -------
        pre_keys: list = []
        try:
            from .keys import extract_keys

            pre_keys = extract_keys(user_request)
        except Exception:  # noqa: BLE001 - keys must never break replies
            pre_keys = []

        # ---- 2. route + execute ---------------------------------------
        payload = self._process_route(user_request)

        # Every response crosses the same temporal contract, including
        # facts, misses, clarifications, and non-calendar operations. This
        # gives Core one authoritative reference clock for words such as
        # "tomorrow"/"tmr" even when the selected function returned no
        # events.
        result = payload.get("result")
        if isinstance(result, dict):
            result.setdefault("temporal_context", self._temporal_envelope())

        # ---- 3. store keys, attach to whatever result came back -------
        try:
            from .keys import key_text, store_keys
            from zoneinfo import ZoneInfo

            new_keys = store_keys(
                self.engine,
                pre_keys,
                actions_engine=self.actions_engine,
                timezone=ZoneInfo(self.timezone),
            )
            if new_keys:
                result = payload.get("result") or {}
                result["keys_found"] = [key_text(k) for k in new_keys]
                payload["result"] = result
        except Exception:  # noqa: BLE001 - keys must never break replies
            pass

        return payload

    def _process_route(
        self,
        user_request: str,
    ) -> dict[str, Any]:

        # sanitize terminal editing artifacts before anything else
        user_request = _ANSI_ESCAPE.sub(
            "",
            user_request,
        ).strip()

        if not user_request:
            return KnowledgeResponse(
                user_request=user_request,
                function=None,
                result={
                    "ok": False,
                    "status": "empty_request",
                },
                analysis_required=True,
            ).to_dict()

        # Deterministic routing first: lexically unambiguous requests
        # never touch the model (faster, perfectly consistent).
        from .rules import decompose, route

        # Compound requests ("... and ...") split into independently
        # routed subtasks, each executed and reported separately.
        subtasks = decompose(
            user_request,
            self.temporal,
        )

        if subtasks is not None:
            subtask_results = []

            for name, arguments in subtasks:
                hybrid_parse = None
                try:
                    arguments, hybrid_parse, hybrid_error = (
                        self._prepare_calendar_arguments(
                            user_request,
                            name,
                            arguments,
                        )
                    )
                    if hybrid_error:
                        function_result = {
                            "ok": False,
                            "operation": "UPDATE"
                            if name == "update_calendar_event"
                            else "CREATE",
                            "error": hybrid_error,
                        }
                    else:
                        arguments = self._validate_arguments(
                            name,
                            arguments,
                        )
                        function_result = self.tool_functions[name](
                            **arguments
                        )
                        if hybrid_parse is not None:
                            function_result["temporal_parse"] = {
                                "source": hybrid_parse.source,
                                "confidence": hybrid_parse.confidence,
                                "slots": hybrid_parse.slots,
                            }
                except Exception as exc:  # noqa: BLE001
                    function_result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                subtask_results.append(
                    {
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                        "result": function_result,
                    }
                )

            return KnowledgeResponse(
                user_request=user_request,
                function=None,
                result={
                    "ok": True,
                    "status": "multi_action",
                    "count": len(subtask_results),
                    "subtasks": subtask_results,
                    "temporal_context": self._temporal_envelope(),
                },
                analysis_required=True,
            ).to_dict()

        routed = route(
            user_request,
            self.temporal,
        )

        if routed is not None:
            name, arguments = routed
        else:
            tool_call = self._generate_tool_call(
                user_request
            )

            if tool_call is None:
                return KnowledgeResponse(
                    user_request=user_request,
                    function=None,
                    result={
                        "ok": False,
                        "status": "no_function_selected",
                        "temporal_context": self._temporal_envelope(),
                    },
                    analysis_required=True,
                ).to_dict()

            name = tool_call["name"]
            arguments = tool_call["arguments"]

        # Nix_predictor may return several independent Knowledge actions for
        # one compound request. Execute each action through the same symbolic
        # validation and tool boundary; never let the model mutate directly.
        if name == "multi_action":
            subtask_results = []
            for action in arguments.get("actions", []):
                if not isinstance(action, dict):
                    continue
                action_name = action.get("name")
                action_args = action.get("arguments") or {}
                if action_name not in self.tool_functions:
                    continue
                try:
                    action_args, hybrid_parse, hybrid_error = (
                        self._prepare_calendar_arguments(
                            user_request,
                            action_name,
                            action_args,
                        )
                    )
                    if hybrid_error:
                        function_result = {
                            "ok": False,
                            "error": hybrid_error,
                        }
                    else:
                        action_args = self._validate_arguments(
                            action_name,
                            action_args,
                        )
                        function_result = self.tool_functions[action_name](
                            **action_args
                        )
                        if hybrid_parse is not None:
                            function_result["temporal_parse"] = {
                                "source": hybrid_parse.source,
                                "confidence": hybrid_parse.confidence,
                                "slots": hybrid_parse.slots,
                            }
                except Exception as exc:  # noqa: BLE001
                    function_result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                subtask_results.append({
                    "function": {
                        "name": action_name,
                        "arguments": action_args,
                    },
                    "result": function_result,
                })
            return KnowledgeResponse(
                user_request=user_request,
                function={"name": "multi_action", "arguments": arguments},
                result={
                    "ok": True,
                    "status": "multi_action",
                    "count": len(subtask_results),
                    "subtasks": subtask_results,
                    "temporal_context": self._temporal_envelope(),
                },
                analysis_required=True,
            ).to_dict()

        # Neuro-symbolic boundary: the selector may propose calendar
        # arguments, but Python must validate/repair the proposal against
        # the original utterance before any mutation. This catches partial
        # parses such as a title containing "5 days after today" while the
        # expression contains only "from 9am to 11am".
        arguments, hybrid_parse, hybrid_error = (
            self._prepare_calendar_arguments(
                user_request,
                name,
                arguments,
            )
        )
        if hybrid_error:
            return KnowledgeResponse(
                user_request=user_request,
                function={"name": name, "arguments": arguments},
                result={
                    "ok": False,
                    "operation": "UPDATE"
                    if name == "update_calendar_event"
                    else "CREATE",
                    "error": hybrid_error,
                    "temporal_context": self._temporal_envelope(),
                },
                analysis_required=True,
            ).to_dict()

        try:
            arguments = self._validate_arguments(
                name,
                arguments,
            )
        except Exception:
            # The model gate hallucinated a function or argument shape
            # (e.g. find_calendar_events(location=...)). That is a
            # result-shaped failure, not a crash: return it so Core
            # gets a structured error and the user still gets a reply
            # (and keys still ride along).
            return KnowledgeResponse(
                user_request=user_request,
                function=name,
                result={
                    "ok": False,
                    "error": (
                        f"model selected {name} with invalid arguments"
                    ),
                    "temporal_context": self._temporal_envelope(),
                },
                analysis_required=True,
            ).to_dict()

        try:
            function_result = self.tool_functions[name](
                **arguments
            )
        except Exception as exc:  # noqa: BLE001
            # Tool failures are results, not crashes: Core receives a
            # structured error and decides what to tell the user.
            function_result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        # ---- Calendar misroute guard -----------------------------------
        # The selector occasionally routes personal knowledge questions
        # ("my sister isnt sick anymore") to the calendar. A bare
        # find_calendar_events() ALSO succeeds (returns every event), so
        # emptiness alone cannot detect the mistake. Trust the calendar
        # route only when the request carries calendar/temporal domain
        # vocabulary; otherwise retry through the fact finder, whose
        # referent resolution connects "my sister" to the stored name.
        function_result = self._maybe_fallback_to_facts(
            name,
            user_request,
            function_result,
        )

        if isinstance(function_result, dict):
            function_result.setdefault(
                "temporal_context",
                self._temporal_envelope(),
            )

        if hybrid_parse is not None and isinstance(function_result, dict):
            function_result["temporal_parse"] = {
                "source": hybrid_parse.source,
                "confidence": hybrid_parse.confidence,
                "slots": hybrid_parse.slots,
            }

        return KnowledgeResponse(
            user_request=user_request,
            function={
                "name": name,
                "arguments": arguments,
            },
            result=function_result,
            analysis_required=True,
        ).to_dict()

    def _maybe_fallback_to_facts(
        self,
        name: str,
        user_request: str,
        function_result: Any,
    ) -> Any:
        """Retrain a misrouted calendar route onto the fact finder.

        Applies only to read-only calendar lookups (never to create /
        update / cancel, which must never fire twice). The request is
        domain-checked: no calendar vocabulary in the text plus an
        irrelevant or empty event list means the selector most likely
        picked the wrong function, and the fact finder gets a second
        chance - it carries entity referent resolution.
        """
        if name != "find_calendar_events":
            return function_result
        if not isinstance(function_result, dict):
            return function_result

        request_lower = user_request.lower()
        if _CALENDAR_DOMAIN_RE.search(request_lower):
            return function_result

        events = function_result.get("events") or []
        if not events:
            return function_result

        # Relevance check: at least one event title must share a
        # content word with the request. A bare misroute returns ALL
        # events, which are almost never about the question asked.
        request_words = set(
            re.findall(r"[a-z0-9']+", request_lower)
        )
        relevant = False
        for event in events:
            title_words = set(
                re.findall(
                    r"[a-z0-9']+",
                    str(event.get("title", "")).lower(),
                )
            )
            if request_words & title_words:
                relevant = True
                break
        if relevant:
            return function_result

        try:
            fallback = self._find_facts(query=user_request)
        except Exception:  # noqa: BLE001 - fallback must never crash
            return function_result

        if fallback.get("count"):
            fallback["fallback_from"] = name
            return fallback

        return function_result
