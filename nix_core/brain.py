"""
Nix Core brain.

One entry point: Brain.handle(text, location).

Pipeline:
  1. deterministic router (router.py) decides knowledge / chat
  2. if the rules are unsure, the model-backed classifier hosted by
     nix_knowledge decides (POST /classify)
  3. knowledge  -> nix_knowledge API /process (structured result;
     scheduling flows through its bridge into nix_actions)
  4. chat       -> Ollama (Qwen3.5 4B + SearXNG web search), seeded with
     session turns and a compact knowledge digest so the chatty model
     can still answer personal questions mid-conversation

nix_core holds no durable state itself: session turns are logged
through the nix_actions API (NixCore SessionStore).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any

import requests

from config import (
    ACTIONS_API_URL,
    CONTEXT_MAX_CHARS,
    CONTEXT_WINDOW,
    HTTP_TIMEOUT,
    KNOWLEDGE_API_URL,
    OLLAMA_API_URL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_BATCH,
    OLLAMA_NUM_CTX,
    OLLAMA_THINK,
    TIMEZONE,
    ASSISTANT_NAME,
    ASSISTANT_ROLE,
    CASPER_BACKEND,
    USE_NEURAL_INTENT,
    USE_KNOWLEDGE_MODEL_GATE,
    WARMUP_MODELS,
)
from context import select_context
from request_log import log_request, make_entry
from router import CHAT, KNOWLEDGE, UNKNOWN, classify
from tone_policy import conversation_policy
from tabby_client import TabbyClient


# ----------------------------------------------------------------------
# Sibling service clients
# ----------------------------------------------------------------------


class ServiceError(RuntimeError):
    pass


class KnowledgeClient:
    """HTTP client for the nix_knowledge API."""

    def __init__(self, base_url: str = KNOWLEDGE_API_URL):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any] | None:
        try:
            response = requests.get(
                f"{self.base_url}/health", timeout=5
            )
            return response.json()
        except Exception:
            return None

    def warmup(self) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/warmup", json={}, timeout=120
        )
        response.raise_for_status()
        return response.json()

    def process(self, text: str) -> dict[str, Any]:
        """Run a request through the knowledge engine (needle)."""
        response = requests.post(
            f"{self.base_url}/process",
            json={"text": text, "timezone": TIMEZONE},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def classify(self, text: str) -> str:
        """
        Model-backed fallback classification (knowledge vs chat).

        Runs on the nix_knowledge host where torch lives. Never
        raises: an unreachable classifier degrades to 'chat' - the
        deterministic rules already caught the personal-data shapes.
        """
        try:
            response = requests.post(
                f"{self.base_url}/classify",
                json={"text": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("route") or CHAT
        except Exception:
            return CHAT

    def digest(self) -> str:
        """
        Compact personal-knowledge block for the chat model.

        Returns '' when the service is down or has nothing relevant;
        the chat model then simply answers from world knowledge.
        """
        try:
            response = requests.post(
                f"{self.base_url}/digest",
                json={},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("digest") or ""
        except Exception:
            return ""

    def memory_block(self, current_text: str | None = None) -> str:
        """
        Full MEMORY block for NixLM: facts + people + current states +
        dated moments + the user's current emotional state + pending
        context. Returns '' when the service is down - callers then
        fall back to digest() (or nothing).
        """
        try:
            response = requests.post(
                f"{self.base_url}/memory_block",
                json={"text": current_text or ""},
                timeout=15,
            )
            response.raise_for_status()
            return response.json().get("memory_block") or ""
        except Exception:
            return ""

    def learn_keys(self, text: str) -> list[str]:
        """
        Key Finding for chat-routed utterances: store any durable
        personal facts the sentence contains ("btw my brother Alex
        loves hiking"). Returns the freshly learned key texts; never
        raises - key learning must not break the chat reply.
        """
        try:
            response = requests.post(
                f"{self.base_url}/keys",
                json={"text": text},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("keys_found") or []
        except Exception:
            return []

    def lookup_key_person(self, name: str) -> list[str]:
        """
        Does the knowledge base know this person? Used before a bare
        "who is <name>" question falls to world chat: a known person
        means the question is personal recall. Never raises.
        """
        try:
            response = requests.post(
                f"{self.base_url}/keys",
                json={"lookup": name},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("matches") or []
        except Exception:
            return []

    def intent(self, text: str) -> dict[str, Any]:
        """
        Fast neural mood read (emotion / valence / category) for the
        waiting filler and the mood sound. Never raises: a failed
        read simply returns ok=False and Core stays neutral.
        """
        try:
            response = requests.post(
                f"{self.base_url}/intent",
                json={"text": text},
                timeout=15,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"ok": False}


class ActionsClient:
    """HTTP client for the nix_actions API (session logging + runs)."""

    def __init__(self, base_url: str = ACTIONS_API_URL):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any] | None:
        try:
            response = requests.get(
                f"{self.base_url}/health", timeout=5
            )
            return response.json()
        except Exception:
            return None

    def log_turn(
        self,
        *,
        role: str,
        content: str,
        refs: dict[str, Any] | None = None,
    ) -> None:
        """Append a turn to the current session (best effort)."""
        try:
            requests.post(
                f"{self.base_url}/log",
                json={
                    "role": role,
                    "content": content,
                    "refs": refs or {},
                },
                timeout=10,
            )
        except Exception:
            pass

    def context(self, limit: int = CONTEXT_WINDOW) -> list[dict[str, Any]]:
        """Un-pruned turns of the current session, oldest first."""
        try:
            response = requests.get(
                f"{self.base_url}/context",
                params={"limit": limit},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("turns", [])
        except Exception:
            return []


# ----------------------------------------------------------------------
# Ollama (chat / internet)
# ----------------------------------------------------------------------


_DEEP_REASONING_RE = re.compile(
    r"\b(?:analy[sz]e|analysis|deep(?:ly)?|reason(?:ing)?|step[- ]by[- ]step|"
    r"compare|contrast|trade[- ]offs?|debug|diagnos(?:e|is)|architect(?:ure)?|"
    r"design|plan|strategy|derive|prove|calculate|code|program|refactor|"
    r"why\s+does|pros\s+and\s+cons|think\s+(?:this|it)\s+through)\b",
    re.IGNORECASE,
)


def request_requires_thinking(text: str) -> bool:
    """Identify explicit high-complexity requests for optional Qwen thinking.

    Casper's default path is non-thinking. This is deliberately conservative:
    ordinary chat, greetings, emotional support, memory, and routine actions
    should remain fast and do not need a reasoning trace.
    """
    normalized = " ".join((text or "").split())
    return bool(_DEEP_REASONING_RE.search(normalized))


class OllamaClient:
    """Chat client for the local Ollama server (Qwen3.5 + SearXNG)."""

    def __init__(
        self,
        api_url: str = OLLAMA_API_URL,
        model: str = OLLAMA_MODEL,
    ):
        self.api_url = api_url
        self.model = model

    def chat(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
        timeout: float | None = None,
        think: bool | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        for turn in select_context(
            history,
            max_turns=CONTEXT_WINDOW,
            max_chars=CONTEXT_MAX_CHARS,
        ):
            messages.append(turn)

        messages.append({"role": "user", "content": user_text})

        request_body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            # Global thinking is only an opt-in ceiling. Simple requests
            # explicitly pass think=False even if the deployment enabled it.
            "think": OLLAMA_THINK if think is None else bool(think),
            "options": {
                "num_ctx": OLLAMA_NUM_CTX,
                "num_batch": OLLAMA_NUM_BATCH,
                "num_predict": 120,
            },
        }
        response = requests.post(
            self.api_url,
            json=request_body,
            timeout=timeout or HTTP_TIMEOUT,
        )
        if (
            response.status_code == 404
            and OLLAMA_FALLBACK_MODEL
            and OLLAMA_FALLBACK_MODEL != self.model
        ):
            # Model selection is deployment configuration. A fresh install
            # may have the previous model but not the preferred one yet, so
            # preserve availability while the operator downloads the new tag.
            request_body["model"] = OLLAMA_FALLBACK_MODEL
            response = requests.post(
                self.api_url,
                json=request_body,
                timeout=timeout or HTTP_TIMEOUT,
            )
        response.raise_for_status()

        data = response.json()
        message = data.get("message") or {}
        return message.get("content") or data.get("response") or ""

    def simple(self, prompt: str) -> str:
        """One-shot generation (used for chat-route error fallbacks)."""
        return self.chat(
            system_prompt=(
                f"You are {ASSISTANT_NAME}, a {ASSISTANT_ROLE}. "
                "Be concise and natural; do not ask generic follow-up questions."
            ),
            history=[],
            user_text=prompt,
            think=False,
        )


# ----------------------------------------------------------------------
# Multi-question handling: split, classify per clause, merge replies
# ----------------------------------------------------------------------

# Clause boundaries that carry a genuinely new question/request:
# "... tonight, oh and who won ...", "... this week; also what's ...",
# "... friday. and tell me ...". Deliberately conservative: plain
# "and" only splits when followed by a question/request word, so
# "meeting with bob and alice" stays whole.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\s*[;,]?\s*\b(?:oh\s+and|and\s+also|also|btw|by\s+the\s+way)\b"
    r"|\s*\band\s+(?=what|who|where|when|why|how|does|do|is|are|can|"
    r"tell|search|show|remind|schedule|cancel|set|add|move|will|"
    r"remember|my|reschedule)\b"
    r"|\s*\?\s*(?=\w)"
    r"|\s*\.\s+(?=(?:what|who|where|when|can|tell|search|show|" 
    r"remind|schedule|cancel|set|add|move|also|will|remember|my|"
    r"reschedule))\b"
    r")\s*",
    re.IGNORECASE,
)

# Follow-up fragments that only make sense with the previous clause or
# turn: "the one with dr sharma", "at 3 instead".
_FRAGMENT_RE = re.compile(
    r"^(?:the\s+one|that\s+one|at\s+\d|on\s+\w+day\s+instead|"
    r"instead|too|please)\b",
    re.IGNORECASE,
)

# Follow-up requests that reference prior conversation state.
_ANAPHORA_RE = re.compile(
    r"\b(?:that|it|the\s+last\s+one|this\s+one|that\s+one)\b",
    re.IGNORECASE,
)


def split_clauses(text: str) -> list[str]:
    """Split a multi-question message into clause candidates."""
    parts = [part.strip(" ,;.") for part in _CLAUSE_SPLIT_RE.split(text)]
    return [part for part in parts if part]


# ----------------------------------------------------------------------
# Knowledge result formatting (deterministic, no model involved)
# ----------------------------------------------------------------------


def _format_event(entry: dict[str, Any]) -> str:
    title = entry.get("title") or "event"
    # Relative labels are useful metadata, but the user-facing baseline must
    # include the absolute local date and time. This prevents "tomorrow" from
    # becoming stale or being interpreted against Casper's clock.
    start_local = entry.get("start_local")
    end_local = entry.get("end_local")
    if start_local:
        when = start_local
        if end_local and end_local != start_local:
            when += f" to {end_local}"
    else:
        when = entry.get("when") or entry.get("start", "")
    status = entry.get("status", "scheduled")
    suffix = f" ({status})" if status != "scheduled" else ""
    occurrence = " (occurrence)" if entry.get("occurrence") else ""
    return f"- {title}: {when}{suffix}{occurrence}"


def format_knowledge_result(payload: dict[str, Any]) -> str:
    """
    Turn a KnowledgeResponse result into one user-facing paragraph.

    Knowledge never speaks for itself; Core formats the authoritative
    function results deterministically.
    """
    result = payload.get("result") or {}
    status = result.get("status")

    lines: list[str] = []

    if status == "multi_action":
        for index, subtask in enumerate(result.get("subtasks", []), 1):
            body = format_knowledge_result(
                {"result": subtask.get("result") or {}}
            )
            lines.append(f"{index}. {body}")
        return "\n".join(lines)

    # Clarification is a successful conversational outcome, not an
    # engine error. Core must ask the exact question returned by the
    # authoritative person resolver and must not let Ollama guess.
    if result.get("operation") == "NEEDS_CLARIFICATION":
        return result.get("question") or "Who do you mean?"

    if not result.get("ok", True):
        error = result.get("error") or result.get("status") or "failed"
        return f"Knowledge engine could not handle that: {error}"

    operation = result.get("operation")

    if operation == "STORE_KEYS":
        # Self-introduction: the keys were extracted from the sentence
        # and stored. Acknowledge what was learned, in second person.
        keys = result.get("keys") or []
        if keys:
            return _keys_reply(keys)
        return "Got it."

    if operation == "CREATE":
        data = result.get("data") or {}

        # Stored fact: "remember that my wifi password is X".
        if result.get("record_type") == "fact" or (
            "value" in data and "title" not in data
        ):
            return f"Stored: {data.get('value', '')}"

        temporal = result.get("temporal") or {}
        title = data.get("title") or "event"
        when = temporal.get("when") or temporal.get(
            "resolved_start", ""
        )
        lines.append(f"Scheduled '{title}' {when}.")
        actions = result.get("actions") or {}
        if actions.get("ok"):
            lines.append(
                f"Action set: {actions.get('action_type')} fires "
                f"{actions.get('first_fire_at', '')}."
            )
        return " ".join(lines)

    if operation == "CANCEL":
        return "Cancelled."

    if operation == "UPDATE":
        return "Updated."

    if operation in ("STORE_STATE", "SUPERSEDE_STATE", "STATE_NOOP"):
        # Current-state statements about close people. SUPERSEDE means
        # an older contradicting state was replaced (sick -> cured).
        subject = result.get("about") or "them"
        state = result.get("state") or ""
        if operation == "STATE_NOOP":
            return f"Already noted: {subject} is {state}."
        if operation == "SUPERSEDE_STATE":
            return f"Updated: {subject} is now {state}."
        return f"Noted: {subject} is {state}."

    if operation == "DUPLICATE":
        return "Already noted earlier."

    if "states" in result:
        states = result["states"]
        if not states:
            return "No current states stored yet."
        lines.append("Current states:")
        for state in states[:10]:
            name = state.get("name") or state.get("subject") or "someone"
            lines.append(f"- {name}: {state.get('state', '?')}")
        # dated history: what happened before the current state
        for moment in (result.get("moments") or [])[:5]:
            text = moment.get("text") if isinstance(moment, dict) else moment
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    if "events" in result:
        events = result["events"]
        if not events:
            return "No matching events found."
        lines.append(
            f"{result.get('count', len(events))} event(s) found:"
        )
        lines.extend(_format_event(entry) for entry in events[:10])
        return "\n".join(lines)

    if "facts" in result:
        facts = result["facts"]
        if not facts:
            return "I don't have that in my knowledge base yet."
        if result.get("query") == "name" and len(facts) == 1:
            value = str(facts[0].get("value") or "")
            match = re.search(r"(?:user's|your)\s+name\s+is\s+(.+)", value, re.I)
            if match:
                return f"Your name is {match.group(1).rstrip('.')}."
        lines.append("From your knowledge base:")
        for fact in facts[:10]:
            lines.append(f"- {fact.get('value', fact)}")
        return "\n".join(lines)

    if result.get("value"):
        return f"Stored: {result['value']}"

    # Unknown shape: show the JSON compactly rather than nothing.
    return json.dumps(result, ensure_ascii=False, default=str)[:800]


# ----------------------------------------------------------------------
# Temporal grounding crossing the Knowledge -> Core boundary
# ----------------------------------------------------------------------


def _format_grounded_moment(value: Any) -> str:
    """Return an ISO moment plus an explicit local date/time rendering."""
    if value is None:
        return ""
    text = str(value)
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return text

    display = moment.strftime("%A, %B %d, %Y at %I:%M:%S %p").replace(
        " 0", " "
    ).lstrip("0")
    return f"{text} ({display})"


def _temporal_grounding_block(payload: dict[str, Any]) -> str:
    """Render an explicit, model-readable time contract from Knowledge.

    Relative expressions are useful conversational labels, but they are not
    authoritative. This block puts the Knowledge clock, timezone, query
    window, and every returned event's absolute start/end beside those labels
    so Casper never has to guess what ``tmr`` or ``tomorrow`` means.
    """
    nested = payload.get("result")
    result = nested if isinstance(nested, dict) else payload
    lines = [
        "TEMPORAL GROUNDING (authoritative Knowledge values; do not guess):"
    ]

    context = result.get("temporal_context") or payload.get("temporal_context")
    if isinstance(context, dict):
        timezone = context.get("timezone") or TIMEZONE
        lines.append(f"- timezone: {timezone}")
        if context.get("current_datetime_display"):
            lines.append(
                f"- current local date/time: {context['current_datetime_display']}"
            )
        if context.get("now"):
            lines.append(f"- current local ISO datetime: {context['now']}")
        if context.get("current_date") or context.get("today"):
            lines.append(
                f"- today: {context.get('current_date') or context.get('today')}"
            )
        if context.get("tomorrow"):
            lines.append(
                f"- tomorrow / tmr means: {context['tomorrow']}"
            )
        if context.get("weekday"):
            lines.append(f"- current weekday: {context['weekday']}")
    else:
        lines.append(
            "- Knowledge supplied no temporal context; do not convert a "
            "relative expression into an absolute date."
        )

    window = result.get("window")
    if isinstance(window, dict):
        lines.append(
            f"- requested window {window.get('expression')!r}: "
            f"{_format_grounded_moment(window.get('start'))} through "
            f"{_format_grounded_moment(window.get('end'))}"
        )

    event_lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    def add_event(
        event: dict[str, Any],
        *,
        title: str | None = None,
        original: str | None = None,
        relative: str | None = None,
    ) -> None:
        grounding = event.get("temporal_grounding")
        if not isinstance(grounding, dict):
            grounding = {}

        start_info = grounding.get("start") or {}
        end_info = grounding.get("end") or {}
        start = (
            start_info.get("iso")
            or event.get("start")
            or event.get("resolved_start")
        )
        end = (
            end_info.get("iso")
            or event.get("end")
            or event.get("resolved_end")
        )
        if not start:
            return

        title = title or event.get("title") or "event"
        original = (
            original
            or event.get("temporal_expression")
            or grounding.get("original_expression")
        )
        relative = (
            relative
            or event.get("when")
            or event.get("relative_label")
            or grounding.get("relative_label")
        )
        signature = (str(title), str(start), str(end or ""))
        if signature in seen:
            return
        seen.add(signature)

        line = f"- {title}:"
        if original:
            line += f" expression={original!r};"
        if relative:
            line += f" relative label={relative!r};"
        line += f" start={_format_grounded_moment(start)}"
        if end and end != start:
            line += f"; end={_format_grounded_moment(end)}"
        timezone = (
            event.get("timezone")
            or grounding.get("timezone")
            or (context or {}).get("timezone")
        )
        if timezone:
            line += f"; timezone={timezone}"
        event_lines.append(line)

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        events = node.get("events")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    add_event(event)

        temporal = node.get("temporal")
        if isinstance(temporal, dict):
            data = node.get("data") if isinstance(node.get("data"), dict) else {}
            add_event(
                temporal,
                title=data.get("title"),
                original=temporal.get("original_expression"),
                relative=temporal.get("when"),
            )

        # Update responses carry the old matched event in ``lookup`` and the
        # new authoritative stored event in ``result.data``.
        lookup = node.get("lookup")
        if isinstance(lookup, dict):
            add_event(lookup)
        nested_result = node.get("result")
        if isinstance(nested_result, dict):
            data = nested_result.get("data")
            if isinstance(data, dict):
                add_event(data)
            visit(nested_result)

        subtasks = node.get("subtasks")
        if isinstance(subtasks, list):
            visit(subtasks)

    visit(result)
    if event_lines:
        lines.append("- resolved event timestamps:")
        lines.extend(event_lines[:20])
    else:
        lines.append("- resolved event timestamps: none in this result")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Key acknowledgment
# ----------------------------------------------------------------------


def _to_second_person(text: str) -> str:
    """'user has a sister' -> 'you have a sister' for replies."""
    text = re.sub(r"\buser's\b", "your", text)
    text = re.sub(r"\buser has\b", "you have", text)
    text = re.sub(r"\buser is\b", "you are", text)
    text = re.sub(r"\buser wakes up\b", "you wake up", text)
    text = re.sub(r"\buser goes to bed\b", "you go to bed", text)
    text = re.sub(r"\buser goes by\b", "you go by", text)
    text = re.sub(r"\buser never eats\b", "you never eat", text)

    def _de_s(match: re.Match) -> str:
        return "you " + match.group(1)[:-1]

    text = re.sub(
        r"\buser (loves|likes|hates|enjoys|prefers|lives|works|drives|"
        r"wants|studies|wears|uses|avoids)\b",
        _de_s,
        text,
    )
    return re.sub(r"\buser\b", "you", text)


def _keys_reply(keys_found: list[str]) -> str:
    """Acknowledge learned keys when nothing else was produced."""
    parts = [_to_second_person(key) for key in keys_found]
    if len(parts) == 1:
        return f"Got it: {parts[0].rstrip('.')}."
    joined = ", ".join(part.rstrip(".") for part in parts)
    return f"Got it: {joined}."


# ----------------------------------------------------------------------
# The brain
# ----------------------------------------------------------------------


_PERSONAL_KNOWLEDGE_RE = re.compile(
    r"^(?:who\s+am\s+i|do\s+you\s+know\s+(?:who\s+)?i\s+am|"
    r"do\s+you\s+know\s+me|what\s+do\s+you\s+know\s+about\s+me|"
    r"what(?:'s|\s+is|\s+are|\s+was|\s+were)\s+my\b(?:\s+[^?]+)?|"
    r"where\s+do\s+i\b|when\s+is\s+my\b|"
    r"do\s+you\s+remember\b|what\s+do\s+you\s+remember\b|"
    r"how\s+(?:is|are|was|were)\s+my\b(?:\s+[^?]+)?|"
    r"(?:remember|don't\s+forget|keep\s+in\s+mind)\s+(?:that\s+)?my\b)\s*\??$",
    re.IGNORECASE,
)


def should_delegate_to_knowledge(text: str) -> bool:
    """Core boundary rule for requests that require personal memory.

    These must reach Knowledge before the chat/model classifier. Otherwise a
    small conversational model can answer from session wording or claim it
    does not know a fact that is actually stored.
    """
    normalized = " ".join((text or "").split())
    if _PERSONAL_KNOWLEDGE_RE.match(normalized):
        return True
    # State statements are personal memory even when they do not contain
    # "my" (for example, "Jane is fine" after the person was learned).
    # Ask the same deterministic parser used by Knowledge before Casper gets
    # a chance to answer with an empty "Okay.".
    try:
        from nix_knowledge.states import parse_state_statement
    except ImportError:
        try:
            from states import parse_state_statement
        except ImportError:
            parse_state_statement = None
    return bool(parse_state_statement and parse_state_statement(normalized))


# Product identity is a protected fact, not a model-generated biography.
# Keeping it deterministic prevents Casper from confusing the Qwen model's
# original developer with the person who built this PUCA system.
_CREATOR_IDENTITY_RE = re.compile(
    r"^(?:"
    r"who\s+(?:created|made|built|developed)\s+"
    r"(?:casper|you|nix|this|it|this\s+puca|the\s+puca|"
    r"you\s+casper|casper\s+you|"
    r"this\s+system|the\s+system)|"
    r"who\s+is\s+(?:your|casper's|nix's)\s+"
    r"(?:creator|developer|author|builder|maker)|"
    r"who\s+is\s+the\s+"
    r"(?:creator|developer|author|builder|maker)\s+of\s+"
    r"(?:casper|nix|you|this|it|this\s+puca|the\s+puca|"
    r"this\s+system|the\s+system)|"
    r"who\s+developed\s+this\s+puca|"
    r"who\s+made\s+you"
    r")\s*(?:casper|nix)?\s*[?!.,]*$",
    re.IGNORECASE,
)

_CREATOR_IDENTITY_REPLY = (
    "Created and Built by Sai Neela, and living in NIX's PUCA system."
)
_ASSISTANT_IDENTITY_RE = re.compile(
    r"^(?:(?:bro|hey|hi|yo)\s+)?(?:who\s+are\s+you(?:\s+again)?|"
    r"who\s+is\s+casper(?:\s+again)?|what\s+are\s+you)\s*[?!.,]*$",
    re.IGNORECASE,
)
_ASSISTANT_IDENTITY_REPLY = (
    "I'm Casper, Sai's PUCA (Personal User Companion Agent) living in NIX."
)


def is_creator_identity_request(text: str) -> bool:
    """Recognize creator questions that must use the canonical identity."""
    normalized = " ".join((text or "").split())
    return bool(_CREATOR_IDENTITY_RE.match(normalized))


def is_assistant_identity_request(text: str) -> bool:
    return bool(_ASSISTANT_IDENTITY_RE.match(" ".join((text or "").split())))


_GENERIC_INTERVIEW_RE = re.compile(
    r"(?:what(?:'s| is) on your mind|how can i help(?: you)?|"
    r"what can i do for you|anything else(?: i can help with)?|"
    r"how about we chat about something else|let me know if you need anything)",
    re.IGNORECASE,
)


_INTERNAL_ROUTE_LEAK_RE = re.compile(
    r"(?:^|\s)(?:CHAT|KNOWLEDGE)\s+rule:\s*[^\n]+",
    re.IGNORECASE,
)


def suppress_internal_route_metadata(reply: str) -> str:
    """Never expose Core routing labels as a user-facing answer."""
    text = re.sub(r"\s+", " ", (reply or "")).strip()
    if not _INTERNAL_ROUTE_LEAK_RE.search(text):
        return text
    cleaned = _INTERNAL_ROUTE_LEAK_RE.sub("", text).strip(" -:;,.")
    return cleaned or _ASSISTANT_IDENTITY_REPLY


def suppress_generic_interview(reply: str) -> str:
    """Remove canned conversation-extending questions from model output.

    This intentionally targets only generic assistant filler. Genuine questions
    required by a task, safety, or identity ambiguity remain untouched.
    """
    text = re.sub(r"\s+", " ", (reply or "")).strip()
    if not text or not _GENERIC_INTERVIEW_RE.search(text):
        return text
    # Generic fillers are normally a final sentence. Remove that sentence,
    # including a leading conjunction, while preserving the useful answer.
    cleaned = re.sub(
        r"(?:[.!?]\s*|\s+)(?:and\s+)?" + _GENERIC_INTERVIEW_RE.pattern + r"[.!?]*$",
        ".",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if cleaned and not _GENERIC_INTERVIEW_RE.search(cleaned):
        return cleaned
    return "I’m not sure how to answer that yet."


class Brain:
    def __init__(
        self,
        *,
        knowledge: KnowledgeClient | None = None,
        actions: ActionsClient | None = None,
        ollama: OllamaClient | None = None,
        log_requests: bool = True,
    ):
        self.knowledge = knowledge or KnowledgeClient()
        self.actions = actions or ActionsClient()
        if ollama is not None:
            self.ollama = ollama
        elif CASPER_BACKEND == "transformers":
            # Lazy wrapper: importing Core does not allocate GPU memory.
            from casper_model import get_casper_client

            class _LazyCasper:
                model = "casper-puca-qlora-v5"
                api_url = "local://transformers"

                def chat(self, **kwargs):
                    return get_casper_client().chat(**kwargs)

            self.ollama = _LazyCasper()
        elif CASPER_BACKEND == "tabby":
            self.ollama = TabbyClient()
        else:
            self.ollama = OllamaClient()
        self.log_requests = log_requests
        # Pending person clarifications are conversational state, not durable
        # knowledge. They are keyed by the voice/dashboard conversation ID so
        # the next answer ("Maanvi", "the second one") can complete the prior
        # request instead of being routed as an unrelated statement.
        self._pending_clarifications: dict[str, dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        # Every structured Knowledge result is handed back to Core's
        # presentation layer. The formatter always uses think=False; if
        # Ollama is unavailable, the deterministic rendering remains the
        # safe response. The old opt-out flag is intentionally no longer
        # used because Knowledge must not speak directly to the user.

    def warmup(self) -> dict[str, str]:
        """Initialize approved neural services concurrently, if enabled."""
        if not WARMUP_MODELS:
            return {"warmup": "disabled"}
        from runtime_warmup import warm_models

        casper_loader = None
        if CASPER_BACKEND == "transformers":
            from casper_model import get_casper_client

            casper_loader = get_casper_client
        return warm_models(
            casper_loader=casper_loader,
            knowledge_health=self.knowledge.warmup,
        )

    # ------------------------------------------------------------------
    # Pending clarification continuation
    # ------------------------------------------------------------------

    def _remember_clarification(
        self,
        conversation_id: str | None,
        request: str,
        result: dict[str, Any],
    ) -> None:
        if not conversation_id or result.get("operation") != "NEEDS_CLARIFICATION":
            return
        candidates = [str(item) for item in (result.get("candidates") or [])]
        with self._pending_lock:
            self._pending_clarifications[conversation_id] = {
                "request": request,
                "role": result.get("role"),
                "candidates": candidates,
                "question": result.get("question") or "Who do you mean?",
            }

    def _take_clarification_answer(
        self,
        conversation_id: str | None,
        answer: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not conversation_id:
            return None, None
        with self._pending_lock:
            pending = self._pending_clarifications.get(conversation_id)
        if not pending:
            return None, None

        normalized = " ".join(answer.lower().strip(" .!? ").split())
        candidates = pending["candidates"]
        selected = None
        for index, candidate in enumerate(candidates, 1):
            if normalized == candidate.lower() or re.search(
                rf"\b(?:option|number|choice)?\s*{index}(?:st|nd|rd|th)?\b",
                normalized,
            ):
                selected = candidate
                break
        if selected is None:
            return pending, None

        with self._pending_lock:
            self._pending_clarifications.pop(conversation_id, None)
        return pending, selected

    @staticmethod
    def _complete_state_clarification(request: str, pending: dict[str, Any], name: str) -> str:
        role = pending.get("role")
        if role:
            pattern = re.compile(rf"\bmy\s+{re.escape(str(role))}\b", re.IGNORECASE)
            if pattern.search(request):
                return pattern.sub(
                    lambda match: f"{match.group(0)} {name}",
                    request,
                    count=1,
                )
        return f"{request} about {name}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(
        self,
        *,
        text: str,
        location: str = "unknown",
        conversation_id: str | None = None,
        session_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        One full request -> routed, handled, logged, replied.

        Returns {"route", "reply", "rule", "details"}.
        """
        clean = (text or "").strip()
        t0 = time.perf_counter()
        entry: dict[str, Any] | None = None
        clauses: list[str] = [clean]

        def _finish(response: dict[str, Any]) -> dict[str, Any]:
            """Log once, then return. Covers every exit path."""
            nonlocal entry
            if entry is None and self.log_requests:
                entry = make_entry(
                    request=clean,
                    location=location,
                    route=response.get("route") or UNKNOWN,
                    rule=response.get("rule"),
                    reply=response.get("reply") or "",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    details=response.get("details"),
                    clauses=clauses,
                    error=response.get("error"),
                )
                log_request(entry)
            return response

        if not clean:
            return _finish(
                {
                    "route": UNKNOWN,
                    "reply": "I didn't catch anything.",
                    "rule": None,
                    "details": {},
                }
            )

        # A clarification answer belongs to the unresolved prior request.
        # Resolve it before ordinary routing, so a bare "Maanvi" cannot become
        # a new fact or a world-chat query.
        pending, selected = self._take_clarification_answer(
            conversation_id,
            clean,
        )
        if pending is not None and selected is None:
            return _finish(
                {
                    "route": KNOWLEDGE,
                    "reply": pending["question"] + " (Please name one of the listed people.)",
                    "rule": "clarification_still_pending",
                    "details": {"pending_clarification": True},
                }
            )
        if pending is not None and selected is not None:
            completed_request = self._complete_state_clarification(
                pending["request"], pending, selected
            )
            response = self._handle_knowledge(
                completed_request,
                session_context=session_context,
                raw_user_request=f"{pending['request']} (clarified as {selected})",
                conversation_id=conversation_id,
            )
            response.setdefault("details", {})["clarification_continuation"] = {
                "answer": selected,
                "original_request": pending["request"],
                "completed_request": completed_request,
            }
            self.actions.log_turn(
                role="assistant",
                content=response["reply"],
                refs={"route": response.get("route"), "continuation": True},
            )
            return _finish(response)

        # Do not ask Casper to recall its own product identity. This is a
        # protected Core fact and must never be replaced by a route label,
        # model name, or the base model's upstream developer.
        if is_creator_identity_request(clean) or is_assistant_identity_request(clean):
            identity_reply = (
                _CREATOR_IDENTITY_REPLY
                if is_creator_identity_request(clean)
                else _ASSISTANT_IDENTITY_REPLY
            )
            identity_rule = (
                "creator_identity"
                if is_creator_identity_request(clean)
                else "assistant_identity"
            )
            self.actions.log_turn(
                role="user",
                content=clean,
                refs={"route": CHAT, "rule": "creator_identity"},
            )
            self.actions.log_turn(
                role="assistant",
                content=identity_reply,
                refs={"route": CHAT, "rule": identity_rule},
            )
            return _finish(
                {
                    "route": CHAT,
                    "reply": identity_reply,
                    "rule": identity_rule,
                    "details": {
                        "deterministic": True,
                        "model_called": False,
                        "internal_metadata_hidden": True,
                    },
                }
            )

        clauses = split_clauses(clean)
        if len(clauses) > 1:
            return _finish(
                self._handle_multi_clause(
                    clean,
                    clauses,
                    location=location,
                    session_context=session_context,
                    conversation_id=conversation_id,
                )
            )

        if should_delegate_to_knowledge(clean):
            route, features = KNOWLEDGE, {"rule": "core_personal_memory_contract"}
        else:
            route, features = classify(clean)
        rule = features.get("rule")

        if route == UNKNOWN:
            if USE_KNOWLEDGE_MODEL_GATE:
                route = self.knowledge.classify(clean)
                rule = "model_classifier"
            else:
                # Casper itself is the only neural runtime by default. An
                # abstaining Core route safely becomes chat rather than
                # starting Knowledge's second selector model on the GPU.
                route = CHAT
                rule = "core_abstained_model_gate_disabled"

        self.actions.log_turn(
            role="user",
            content=clean,
            refs={"route": route, "rule": rule, "location": location},
        )

        if route == KNOWLEDGE:
            response = self._handle_knowledge(
                clean,
                session_context=session_context,
                raw_user_request=clean,
                conversation_id=conversation_id,
            )
        else:
            response = self._handle_chat(
                clean,
                session_context=session_context,
                conversation_id=conversation_id,
            )

        self.actions.log_turn(
            role="assistant",
            content=response["reply"],
            refs={"route": route},
        )

        return _finish(response)

    # ------------------------------------------------------------------
    # Multi-question messages: per-clause routing, merged reply
    # ------------------------------------------------------------------

    def _handle_multi_clause(
        self,
        original: str,
        clauses: list[str],
        *,
        location: str,
        session_context: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        "remind me to defrost the chicken, oh and what's the capital
        of australia" -> the reminder goes to knowledge, the world
        question goes to chat, and the user gets one merged reply.
        """
        routes = []
        for clause in clauses:
            if should_delegate_to_knowledge(clause):
                route, features = KNOWLEDGE, {"rule": "core_personal_memory_contract"}
            else:
                route, features = classify(clause)
            if route == UNKNOWN:
                route = (
                    self.knowledge.classify(clause)
                    if USE_KNOWLEDGE_MODEL_GATE
                    else CHAT
                )
            routes.append(route)

        self.actions.log_turn(
            role="user",
            content=original,
            refs={
                "route": "+".join(dict.fromkeys(routes)),
                "rule": "multi_clause",
                "location": location,
            },
        )

        replies: list[str] = []
        for clause, route in zip(clauses, routes):
            if route == KNOWLEDGE:
                replies.append(
                    self._handle_knowledge(
                        clause,
                        session_context=session_context,
                        raw_user_request=original,
                        conversation_id=conversation_id,
                    )["reply"]
                )
            else:
                chat_result = self._handle_chat(
                    clause,
                    session_context=session_context,
                    conversation_id=conversation_id,
                )
                replies.append(chat_result["reply"])

        merged = " ".join(part.strip() for part in replies if part.strip())

        self.actions.log_turn(
            role="assistant",
            content=merged,
            refs={"route": "multi"},
        )

        return {
            "route": "+".join(dict.fromkeys(routes)),
            "rule": "multi_clause",
            "reply": merged,
            "details": {"clauses": clauses, "routes": routes},
        }

    # ------------------------------------------------------------------
    # Knowledge route
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Reply composition: knowledge executed, Core speaks
    # ------------------------------------------------------------------

    @staticmethod
    def _critical_tokens(
        request: str, deterministic: str
    ) -> tuple[list[str], list[str]]:
        """
        Values that MUST survive composition.

        Anchors come from the user's OWN request (proper nouns like
        Mary, codes like 8817, times like 3pm) plus the event title
        the engine stored. Engine-internal phrasings ("earlier today
        12pm-5:59pm", ISO stamps) are exempt: the model may say
        "this afternoon" instead - that is good rendering, not data
        loss. The unforgivable failures are dropped names, codes and
        titles, or invented ones.
        """
        hard: set[str] = set()

        # proper nouns from the request (skip sentence starters and
        # politeness words)
        words = re.findall(r"\b[A-Z][a-z']*\b", request.strip())
        for word in words[1:] or words:
            if word.lower() not in {
                "can", "could", "would", "will", "please", "hey",
                "nix", "remind", "remember", "what", "when", "where",
                "who", "how", "schedule", "cancel", "set", "my",
                "i",
            }:
                hard.add(word.lower())

        # Distinctive lower-case nouns from a personal statement must not
        # disappear during conversational rewriting ("flowerpot", "robotics",
        # medication names, etc.). Conservative stopwords keep this from
        # forcing every grammatical word into the reply.
        stop = {
            "please", "remember", "just", "know", "that", "my", "your",
            "spare", "key", "under", "with", "from", "about", "this",
            "have", "has", "is", "are", "the", "and", "for", "to",
            "one", "word", "what", "when", "where", "who", "how",
        }
        hard.update(
            word.lower()
            for word in re.findall(r"[a-z][a-z0-9'-]{5,}", request.lower())
            if word.lower() not in stop
        )

        # codes/times as written by the user: 8817, 3, 5
        hard.update(re.findall(r"\d+", request))

        # the stored event title: 'buy flowers for mary's birthday'
        soft: set[str] = set()
        title_match = re.search(r"'([^']{3,})'", deterministic)
        if title_match:
            stop = {
                "for", "the", "and", "with", "a", "an", "to", "of",
                "my", "our",
            }
            soft.update(
                word
                for word in re.findall(r"[a-z']+", title_match.group(1))
                if len(word) >= 3 and word not in stop
            )

        soft -= hard
        return sorted(hard), sorted(soft)

    def _compose_reply(
        self,
        request: str,
        deterministic: str,
        *,
        knowledge_result: dict[str, Any] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
        knowledge_context: str | None = None,
        raw_user_request: str | None = None,
    ) -> str:
        """
        One natural sentence for a completed knowledge operation,
        written by the chat model and GROUNDED on the deterministic
        result. Falls back to the deterministic text when the model is
        unavailable or drops any critical value (time, name, code).

        Knowledge never speaks for itself: the engine returns
        authoritative data, Core phrases the answer.
        """
        if not deterministic:
            return deterministic

        # Carry the complete KnowledgeResponse across the boundary. The
        # nested result is authoritative for policy checks, while function
        # metadata and analysis_required remain available to the formatter.
        knowledge_payload = knowledge_result or {}
        nested_result = knowledge_payload.get("result")
        result = (
            nested_result
            if isinstance(nested_result, dict)
            else knowledge_payload
        )
        operation = str(result.get("operation") or "")
        is_clarification = operation == "NEEDS_CLARIFICATION"
        is_sensitive_write = (
            operation == "CREATE" and result.get("record_type") == "fact"
        )
        is_empty_or_failed = (
            not result.get("ok", True)
            or result.get("query") == "name"
            or is_clarification
            or is_sensitive_write
            or deterministic.startswith(
                (
                    "Knowledge engine could not handle that",
                    "No matching events found",
                    "No current states stored yet",
                    "I don't have that",
                    "Got it",
                )
            )
        )
        if is_clarification:
            format_policy = (
                "Ask exactly one concise clarification question. Preserve every "
                "candidate person or option from the authoritative result and "
                "never choose one."
            )
        elif is_empty_or_failed:
            format_policy = (
                "State the authoritative result honestly. Do not turn an empty, "
                "failed, or sensitive result into a success claim, and preserve "
                "all names, values, numbers, and codes."
            )
        else:
            format_policy = (
                "Do not ask a question; simply present the completed result "
                "naturally and preserve all authoritative details."
            )
        source_json = json.dumps(
            knowledge_payload,
            ensure_ascii=False,
            default=str,
        )[:6000]
        raw_request = raw_user_request or request
        recent_context = select_context(
            conversation_context or [],
            max_turns=CONTEXT_WINDOW,
            max_chars=CONTEXT_MAX_CHARS,
        )
        context_lines = "\n".join(
            f"{turn['role']}: {turn['content']}"
            for turn in recent_context
        ) or "(no earlier conversation context)"
        memory_context = " ".join((knowledge_context or "").split())[:6000]
        memory_context = memory_context or "(no additional Knowledge memory context)"
        temporal_grounding = _temporal_grounding_block(knowledge_payload)

        try:
            composed = self.ollama.chat(
                system_prompt=(
                    f"You are {ASSISTANT_NAME}, a warm, upbeat {ASSISTANT_ROLE} "
                    "running on the user's private home server. Your "
                    f"name is {ASSISTANT_NAME}. Never call yourself an AI, model, "
                    "or assistant system. A background knowledge system "
                    "just completed the user's request and produced a "
                    "confirmed result. The RAW USER REQUEST and recent "
                    "conversation context below are grounding inputs; use "
                    "them to understand the user's intent, but never invent "
                    "facts beyond the authoritative result. Write ONE short, warm, natural "
                    f"reply confirming it to the user, in {ASSISTANT_NAME}'s voice.\n"
                    "Rules:\n"
                    "- ONE sentence, no lists, no markdown.\n"
                    "- Match the emotional tone of the news: celebrate "
                    "good news warmly, be gentle and caring about bad "
                    "news.\n"
                    "- Phrase it fresh; do NOT repeat the confirmed "
                    "result text verbatim.\n"
                    "- Keep every name, title, number and time from "
                    "the user's request (render times naturally, e.g. "
                    "'this afternoon', 'tomorrow at 3pm').\n"
                    "- For relative words such as tomorrow, tmr, today, or "
                    "next week, use the TEMPORAL GROUNDING block below; "
                    "the Knowledge clock and ISO timestamps are authoritative.\n"
                    "- Never invent anything new.\n"
                    f"- {format_policy}\n"
                    "- No offers of further help, no sign-offs."
                ),
                history=recent_context,
                user_text=(
                    f"RAW USER REQUEST (preserve its meaning): {raw_request}\n"
                    f"KNOWLEDGE-SCOPED REQUEST: {request}\n"
                    "RECENT CONVERSATION CONTEXT:\n"
                    f"{context_lines}\n"
                    "KNOWLEDGE MEMORY CONTEXT (read after the operation):\n"
                    f"{memory_context}\n"
                    f"{temporal_grounding}\n"
                    f"AUTHORITATIVE KNOWLEDGE RESULT (JSON): {source_json}\n"
                    f"CORE SAFE BASELINE RENDERING: {deterministic}\n"
                    "Create the final user-facing reply now; output only that reply:"
                ),
                timeout=20,
                think=False,
            ).strip()
        except Exception:
            return deterministic

        if not composed:
            return deterministic

        # bound the ramble: more than ~2 sentences or 320 chars means
        # the chat model wandered; the deterministic text is better.
        if (
            len(composed) > 320
            or len(re.findall(r"[.!?]", composed)) > 2
        ):
            return deterministic

        haystack = composed.lower()

        # Calendar answers must remain date-safe. If Casper omits the
        # authoritative absolute date supplied by Knowledge, use Core's
        # deterministic rendering instead of allowing a stale relative word
        # such as "tomorrow" to stand in for the actual date.
        events = result.get("events")
        if isinstance(events, list) and events:
            for event in events:
                if not isinstance(event, dict):
                    continue
                absolute_date = str(event.get("start_date") or "")
                if absolute_date and absolute_date not in haystack:
                    local = str(event.get("start_local") or "").lower()
                    month_day = " ".join(local.split(", ")[-1:]).split(" at ")[0]
                    if not month_day or month_day not in haystack:
                        return deterministic

        # echo guard: the model repeated the confirmed result text
        # (would double it in merged replies) instead of phrasing fresh
        det_words = re.sub(r"\s+", " ", deterministic).lower().split()
        if len(det_words) >= 7:
            grams = {
                " ".join(det_words[i : i + 7])
                for i in range(len(det_words) - 6)
            }
            if any(gram in haystack for gram in grams):
                return deterministic

        hard, soft = self._critical_tokens(request, deterministic)

        def _missing(text_low: str, tokens: list[str]) -> list[str]:
            return [token for token in tokens if token not in text_low]

        # Hard anchors (names, codes): must be there, no negotiation.
        if _missing(haystack, hard):
            return deterministic

        # Soft anchors (title words): the model may legitimately
        # paraphrase ('buy' -> 'pick up'); give it one explicit retry
        # before falling back to the deterministic text.
        soft_missing = _missing(haystack, soft)
        if soft_missing:
            try:
                composed = self.ollama.chat(
                    system_prompt=(
                        f"You are {ASSISTANT_NAME}, a {ASSISTANT_ROLE}. Confirm a "
                        "completed request in ONE short sentence. "
                        "Mention exactly these details, phrased naturally. "
                        f"{format_policy} No offers."
                    ),
                    history=recent_context,
                    user_text=(
                        f"RAW USER REQUEST: {raw_request}\n"
                        f"KNOWLEDGE-SCOPED REQUEST: {request}\n"
                        "RECENT CONVERSATION CONTEXT:\n"
                        f"{context_lines}\n"
                        "KNOWLEDGE MEMORY CONTEXT (read after the operation):\n"
                        f"{memory_context}\n"
                        f"{temporal_grounding}\n"
                        f"AUTHORITATIVE KNOWLEDGE RESULT (JSON): {source_json}\n"
                        f"CORE SAFE BASELINE RENDERING: {deterministic}\n"
                        f"Your final reply must mention: "
                        f"{', '.join(soft_missing + hard)}"
                    ),
                    timeout=25,
                    think=False,
                ).strip()
            except Exception:
                return deterministic

            if not composed:
                return deterministic

            haystack = composed.lower()
            if _missing(haystack, hard) or _missing(haystack, soft):
                return deterministic

        return composed

    def _handle_knowledge(
        self,
        text: str,
        *,
        session_context: list[dict[str, Any]] | None = None,
        raw_user_request: str | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        raw_request = raw_user_request or text
        try:
            payload = self.knowledge.process(text)
        except Exception as exc:
            deterministic = (
                f"{ASSISTANT_NAME} Alert: the knowledge engine is unreachable "
                f"right now. ({type(exc).__name__})"
            )
            payload = {
                "result": {
                    "ok": False,
                    "operation": "KNOWLEDGE_UNAVAILABLE",
                    "error": deterministic,
                }
            }
            reply = self._compose_reply(
                text,
                deterministic,
                knowledge_result=payload,
                conversation_context=session_context,
                knowledge_context="",
                raw_user_request=raw_request,
            )
            return {
                "route": KNOWLEDGE,
                "rule": "knowledge_api_unreachable_composed",
                "reply": reply,
                "details": {
                    "error": str(exc),
                    "core_formatter": {
                        "attempted": True,
                        "model": self.ollama.model,
                        "think": False,
                        "conversation_id": conversation_id,
                    },
                },
            }

        deterministic = format_knowledge_result(payload)
        result = payload.get("result") or {}
        self._remember_clarification(conversation_id, text, result)
        # Read the post-operation Knowledge memory block as a separate,
        # bounded grounding input. This lets Casper understand both the
        # structured result and the surrounding person/state/profile context.
        try:
            knowledge_context = self.knowledge.memory_block(text)
        except Exception:
            knowledge_context = ""

        # Core owns presentation for every structured Knowledge result.
        # Failed operations, empty lookups, clarifications, and sensitive
        # writes still go through the non-thinking formatter, which is
        # required to preserve the authoritative meaning and can fall back
        # to the deterministic rendering when it is unsafe or unavailable.
        result = payload.get("result") or {}
        failed_or_empty = (
            not result.get("ok", True)
            or result.get("operation") == "NEEDS_CLARIFICATION"
            or result.get("query") == "name"
            or (
                result.get("operation") == "CREATE"
                and result.get("record_type") == "fact"
            )
            or deterministic.startswith(
                (
                    "Knowledge engine could not handle that",
                    "No matching events found",
                    "No current states stored yet",
                    "I don't have that",
                    "Got it",  # intro acknowledgment: deterministic only
                )
            )
        )

        # Keys first: a personal statement ("my sister, named Maanvi
        # is very naughty") teaches durable knowledge even when its
        # routed operation found nothing. Use the learned-key summary as
        # the baseline, but still send it through Core's formatter below.
        keys_found = result.get("keys_found") or []
        if keys_found and failed_or_empty:
            deterministic = _keys_reply(keys_found)
            result = {
                **result,
                "operation": "STORE_KEYS",
                "keys_found": keys_found,
            }
            payload = {**payload, "result": result}

        # Empty recall on a bare "who is <name>" question: the person
        # is not in the knowledge base, so it is probably a world
        # question ("who is einstein"). Fall through to chat. Known
        # persons never reach this point (recall returned data), and
        # possessive shapes ("who is my sister") keep the honest
        # knowledge reply.
        if (
            failed_or_empty
            and deterministic
            == "I don't have that in my knowledge base yet."
            and re.match(
                r"^who(?:'s|\s+is)\s+[a-z][a-z' ]*?\s*\??$",
                text.strip(),
                re.IGNORECASE,
            )
            and not re.match(
                r"^who(?:'s|\s+is)\s+(?:my|your)\b",
                text.strip(),
                re.IGNORECASE,
            )
        ):
            # Knowledge was consulted, so its authoritative miss still
            # crosses the same Casper final-response boundary. Do not silently
            # switch to a second world-answer path after /process.
            reply = self._compose_reply(
                text,
                deterministic,
                knowledge_result=payload,
                conversation_context=session_context,
                knowledge_context=knowledge_context,
                raw_user_request=raw_request,
            )
            return {
                "route": KNOWLEDGE,
                "rule": "knowledge_miss_composed",
                "reply": reply,
                "details": {
                    **dict(payload),
                    "knowledge_fallback": deterministic,
                    "core_formatter": {
                        "attempted": True,
                        "model": self.ollama.model,
                        "think": False,
                        "conversation_id": conversation_id,
                        "fallback_to_deterministic": reply == deterministic,
                    },
                },
            }

        # Knowledge is never the final speaker. Every successful, empty,
        # clarifying, failed, and sensitive result crosses this Core formatter;
        # the formatter's Ollama call is explicitly think=False.
        reply = self._compose_reply(
            text,
            deterministic,
            knowledge_result=payload,
            conversation_context=session_context,
            knowledge_context=knowledge_context,
            raw_user_request=raw_request,
        )

        # Mood detection for the console filler + mood sound: a fast
        # neural read (no LLM). Best-effort - failures stay neutral.
        # For state statements the engine's own valence (sick=bad,
        # cured=good) is ground truth and overrides the neural guess.
        mood: dict[str, Any] | None = None
        intent = self.knowledge.intent(text) if USE_NEURAL_INTENT else {}
        valence = intent.get("valence") if intent.get("ok") else None
        emotion = intent.get("emotion") if intent.get("ok") else None
        category = intent.get("category") if intent.get("ok") else None

        state_valence = (result.get("result") or {}).get("valence")
        if (result.get("result") or {}).get("operation") in (
            "STORE_STATE",
            "SUPERSEDE_STATE",
            "STATE_NOOP",
        ) and state_valence:
            valence = state_valence
            if valence == "good":
                emotion = "joy"
            elif valence == "bad":
                emotion = "sadness"

        if valence or emotion:
            from moodsound import mood_sound

            mood = {
                "emotion": emotion,
                "valence": valence,
                "category": category,
                "sound": mood_sound(valence, emotion),
            }

        details = dict(payload)
        details["core_formatter"] = {
            "attempted": True,
            "model": self.ollama.model,
            "think": False,
            "fallback_to_deterministic": reply == deterministic,
        }

        return {
            "route": KNOWLEDGE,
            "rule": (
                "knowledge_process_composed"
                if reply != deterministic
                else "knowledge_process"
            ),
            "reply": reply,
            "mood": mood,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Chat route (Ollama Qwen3.5 + SearXNG)
    # ------------------------------------------------------------------

    def _chat_system_prompt(self, current_text: str | None = None) -> str:
        # Prefer the full MEMORY block (facts + moments + emotional
        # state + pending); fall back to the plain digest when the
        # knowledge service predates it or is unreachable.
        block = self.knowledge.memory_block(current_text)
        digest = block or self.knowledge.digest()

        # Wall-clock awareness: the chat model has no clock of its own
        # and is constantly asked time-relative questions ("what time
        # is it", "is it too late to call", "weather tomorrow").
        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo(TIMEZONE))
            clock = now.strftime("%A, %B %d, %Y at %I:%M %p")
        except Exception:
            clock = datetime.now().strftime(
                "%A, %B %d, %Y at %I:%M %p"
            )

        intent = (
            self.knowledge.intent(current_text)
            if USE_NEURAL_INTENT and current_text
            else {}
        )
        policy = conversation_policy(current_text or "", intent)

        prompt = (
            f"You are {ASSISTANT_NAME}, a {ASSISTANT_ROLE} running on a private "
            "home server. You can answer general questions, chat, and "
            "you have web search available. Answer in one or two short "
            "sentences. Do not append unnecessary offers or questions; "
            "ask at most one relevant question when the attunement policy "
            "explicitly permits it.\n\n"
            f"Today is {clock} ({TIMEZONE}). This is authoritative: "
            "you DO know the current date, time and day of week - "
            "never say you cannot know or access them. Use it for "
            "any time-relative question (now, today, tomorrow).\n\n"
            "CONVERSATION STYLE (sound natural, not like customer support):\n"
            "You are Casper, a PUCA (Personal User Companion Agent), not a "
            "generic AI assistant. Your identity is steady, warm, honest, and "
            "non-intrusive.\n"
            "Speak like a warm, familiar person: use contractions, natural "
            "rhythm, and simple wording. Do not start every reply with "
            "'Certainly', 'Of course', or 'Absolutely'. Do not over-explain, "
            "over-validate, force empathy, or add a follow-up just to keep "
            "the conversation alive. Never claim to be conscious or invent "
            "personal experiences; warmth must remain honest.\n"
            "ANTI-INTERVIEW RULE: Never append 'What's on your mind?', "
            "'How can I help?', 'What can I do for you?', 'Anything else?', "
            "or 'How about we chat about something else?' as a generic ending. "
            "After a complete reply, stop. Ask a question only when the user "
            "explicitly requests conversation, a required detail is missing, "
            "or safety requires it.\n\n"
            "CONVERSATION ATTUNEMENT (follow this before stylistic instincts):\n"
            f"Detected interaction state: {policy['state']}\n"
            f"Priority: {policy['priority']}\n"
            f"{policy['instruction']}"
        )

        if digest:
            prompt += (
                "\n\nKNOWLEDGE USE CONTRACT: Personal facts, names, relationships, "
                "states, preferences, dates, and schedules come only from the "
                "MEMORY block or a deterministic Knowledge reply. Do not say "
                "you lack access to personal memory when a matching entry is "
                "present. Do not infer identity from casual chat. If the "
                "memory is empty or does not answer the question, say that "
                "plainly instead of guessing.\n\n"
                "MEMORY (from their private knowledge base - use it "
                "when relevant, never invent more; attune your tone to "
                "an emotional state line when one is present, and use "
                "exact dates from recent history):\n"
                f"{digest}"
            )
        else:
            prompt += (
                "\n\nYou have no stored knowledge about this user. "
                "If they ask about their own life, say you don't "
                "have that stored yet."
            )

        return prompt

    def _handle_chat(
        self,
        text: str,
        *,
        session_context: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        # A bare "who is <name>" question about someone the KB knows
        # is personal recall, not world trivia. Probe once and route
        # to knowledge when the person is on file.
        who_match = re.match(
            r"^who(?:'s|\s+is)\s+([a-z][a-z' ]*?)\s*\??$",
            text.strip(),
            re.IGNORECASE,
        )
        if who_match:
            name = who_match.group(1).strip()
            if name.lower() not in (
                "my", "your", "the", "that", "this", "there",
            ) and self.knowledge.lookup_key_person(name):
                return self._handle_knowledge(
                    text,
                    session_context=session_context,
                    raw_user_request=text,
                    conversation_id=conversation_id,
                )

        history = (
            session_context
            if session_context is not None
            else self.actions.context()
        )

        try:
            think = OLLAMA_THINK and request_requires_thinking(text)
            reply = self.ollama.chat(
                system_prompt=self._chat_system_prompt(current_text=text),
                history=history,
                user_text=text,
                think=think,
            )
            reply = suppress_internal_route_metadata(
                suppress_generic_interview(reply)
            )
            # Small local models occasionally answer a direct question with
            # a conversational deflection. Spend one bounded retry on a
            # clearly constrained request instead of teaching the model to
            # hallucinate an answer or making the user repeat themselves.
            if (
                reply.strip().endswith("?")
                and re.search(r"\b(?:one|single)\s+word\b", text, re.I)
            ):
                reply = self.ollama.chat(
                    system_prompt=(
                        "Answer the user's factual question directly. "
                        "Return exactly one word and nothing else. Do not "
                        "ask a question or request more context."
                    ),
                    history=[],
                    user_text=text,
                    timeout=15,
                    think=False,
                )
        except Exception as exc:
            return {
                "route": CHAT,
                "rule": f"{CASPER_BACKEND}_unreachable",
                "reply": (
                    f"{ASSISTANT_NAME} Alert: Unable to reach the chat model "
                    f"({self.ollama.model}) right now. "
                    f"({type(exc).__name__})"
                ),
                "details": {"error": str(exc)},
            }

        result = {
            "route": CHAT,                "rule": (
                    "casper_transformers"
                    if CASPER_BACKEND == "transformers"
                    else f"casper_{CASPER_BACKEND}"
                ),

            "reply": reply or "(empty reply from chat model)",
            "details": {
                "model": self.ollama.model,
                "backend": CASPER_BACKEND,
                "think": bool(OLLAMA_THINK and request_requires_thinking(text)),
                "knowledge_context": "injected_into_system_prompt",
                "conversation_id": conversation_id,
            },
        }
        self._learn_chat_keys(text, result)
        return result

    def _learn_chat_keys(self, text: str, result: dict[str, Any]) -> None:
        """
        Chat-routed utterances teach keys too: 'btw my brother Alex
        loves hiking' stores family knowledge even though the reply
        comes from the chat model. Silent on failure by contract.
        """
        try:
            learned = self.knowledge.learn_keys(text)
            if learned:
                result["details"]["keys_learned"] = learned
        except Exception:  # noqa: BLE001
            pass