"""
Nix Core brain.

One entry point: Brain.handle(text, location).

Pipeline:
  1. deterministic router (router.py) decides knowledge / chat
  2. if the rules are unsure, the model-backed classifier hosted by
     nix_knowledge decides (POST /classify)
  3. knowledge  -> nix_knowledge API /process (structured result;
     scheduling flows through its bridge into nix_actions)
  4. chat       -> Ollama (phi-4 + SearXNG web search), seeded with
     session turns and a compact knowledge digest so the chatty model
     can still answer personal questions mid-conversation

nix_core holds no durable state itself: session turns are logged
through the nix_actions API (NixCore SessionStore).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any

import requests

from config import (
    ACTIONS_API_URL,
    CONTEXT_WINDOW,
    HTTP_TIMEOUT,
    KNOWLEDGE_API_URL,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    TIMEZONE,
)
from request_log import log_request, make_entry
from router import CHAT, KNOWLEDGE, UNKNOWN, classify


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


class OllamaClient:
    """Chat client for the local Ollama server (phi-4 + SearXNG)."""

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
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        for turn in history[-CONTEXT_WINDOW:]:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and content:
                messages.append(
                    {"role": role, "content": str(content)}
                )

        messages.append({"role": "user", "content": user_text})

        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": 60},
            },
            timeout=timeout or HTTP_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message") or {}
        return message.get("content") or data.get("response") or ""

    def simple(self, prompt: str) -> str:
        """One-shot generation (used for chat-route error fallbacks)."""
        return self.chat(
            system_prompt="You are Nix, a concise helpful assistant.",
            history=[],
            user_text=prompt,
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
        lines.append("From your knowledge base:")
        for fact in facts[:10]:
            lines.append(f"- {fact.get('value', fact)}")
        return "\n".join(lines)

    if result.get("value"):
        return f"Stored: {result['value']}"

    # Unknown shape: show the JSON compactly rather than nothing.
    return json.dumps(result, ensure_ascii=False, default=str)[:800]


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
        self.ollama = ollama or OllamaClient()
        self.log_requests = log_requests
        self.compose_replies = os.environ.get(
            "NIX_COMPOSE_KNOWLEDGE_REPLIES", "1"
        ) != "0"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle(
        self,
        *,
        text: str,
        location: str = "unknown",
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

        clauses = split_clauses(clean)
        if len(clauses) > 1:
            return _finish(
                self._handle_multi_clause(
                    clean, clauses, location=location
                )
            )

        route, features = classify(clean)
        rule = features.get("rule")

        if route == UNKNOWN:
            route = self.knowledge.classify(clean)
            rule = "model_classifier"

        self.actions.log_turn(
            role="user",
            content=clean,
            refs={"route": route, "rule": rule, "location": location},
        )

        if route == KNOWLEDGE:
            response = self._handle_knowledge(clean)
        else:
            response = self._handle_chat(clean)

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
    ) -> dict[str, Any]:
        """
        "remind me to defrost the chicken, oh and what's the capital
        of australia" -> the reminder goes to knowledge, the world
        question goes to chat, and the user gets one merged reply.
        """
        routes = []
        for clause in clauses:
            route, features = classify(clause)
            if route == UNKNOWN:
                route = self.knowledge.classify(clause)
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
                replies.append(self._handle_knowledge(clause)["reply"])
            else:
                chat_result = self._handle_chat(clause)
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
    ) -> str:
        """
        One natural sentence for a completed knowledge operation,
        written by the chat model and GROUNDED on the deterministic
        result. Falls back to the deterministic text when the model is
        unavailable or drops any critical value (time, name, code).

        Knowledge never speaks for itself: the engine returns
        authoritative data, Core phrases the answer.
        """
        if not self.compose_replies or not deterministic:
            return deterministic

        try:
            composed = self.ollama.chat(
                system_prompt=(
                    "You are Nix, a warm, upbeat personal assistant "
                    "running on the user's private home server. Your "
                    "name is Nix - never call yourself an AI, model, "
                    "or assistant system. A background knowledge system "
                    "just completed the user's request and produced a "
                    "confirmed result. Write ONE short, warm, natural "
                    "reply confirming it to the user, in Nix's voice.\n"
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
                    "- Never invent anything new.\n"
                    "- No questions, no offers of further help, no "
                    "sign-offs."
                ),
                history=[],
                user_text=(
                    f"The user asked: {request}\n"
                    f"Confirmed result: {deterministic}\n"
                    "Write the reply:"
                ),
                timeout=20,
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
                        "You are Nix, a home assistant. Confirm a "
                        "completed request in ONE short sentence. "
                        "Mention exactly these details, phrased "
                        "naturally. No questions, no offers."
                    ),
                    history=[],
                    user_text=(
                        f"The user asked: {request}\n"
                        f"Confirmed result: {deterministic}\n"
                        f"Your reply must mention: "
                        f"{', '.join(soft_missing + hard)}"
                    ),
                    timeout=25,
                ).strip()
            except Exception:
                return deterministic

            if not composed:
                return deterministic

            haystack = composed.lower()
            if _missing(haystack, hard) or _missing(haystack, soft):
                return deterministic

        return composed

    def _handle_knowledge(self, text: str) -> dict[str, Any]:
        try:
            payload = self.knowledge.process(text)
        except Exception as exc:
            return {
                "route": KNOWLEDGE,
                "rule": "knowledge_api_unreachable",
                "reply": (
                    "Nix Alert: the knowledge engine is unreachable "
                    f"right now. ({type(exc).__name__})"
                ),
                "details": {"error": str(exc)},
            }

        deterministic = format_knowledge_result(payload)

        # Compose only on confirmed success. Failed operations (bad
        # temporal expressions, engine errors) and empty lookups must
        # keep the deterministic text: a chatty rewrite turns "no
        # results" into apologies or invented data.
        result = payload.get("result") or {}
        failed_or_empty = (
            not result.get("ok", True)
            or deterministic.startswith(
                (
                    "Knowledge engine could not handle that",
                    "No matching events found",
                    "I don't have that",
                    "Got it",  # intro acknowledgment: deterministic only
                )
            )
        )

        # Keys first: a personal statement ("my sister, named Maanvi
        # is very naughty") teaches durable knowledge even when its
        # routed operation found nothing. Acknowledge what was learned
        # instead of replying with a dead-end like "No matching
        # events found."
        keys_found = result.get("keys_found") or []
        if keys_found and failed_or_empty:
            return {
                "route": KNOWLEDGE,
                "rule": "knowledge_keys_learned",
                "reply": _keys_reply(keys_found),
                "details": payload,
            }

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
            chat_response = self._handle_chat(text)
            chat_response["details"]["knowledge_fallback"] = deterministic
            return chat_response

        reply = (
            deterministic
            if failed_or_empty
            else self._compose_reply(text, deterministic)
        )

        # Mood detection for the console filler + mood sound: a fast
        # neural read (no LLM). Best-effort - failures stay neutral.
        # For state statements the engine's own valence (sick=bad,
        # cured=good) is ground truth and overrides the neural guess.
        mood: dict[str, Any] | None = None
        intent = self.knowledge.intent(text)
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

        return {
            "route": KNOWLEDGE,
            "rule": (
                "knowledge_process_composed"
                if reply is not deterministic
                else "knowledge_process"
            ),
            "reply": reply,
            "mood": mood,
            "details": payload,
        }

    # ------------------------------------------------------------------
    # Chat route (Ollama phi-4 + SearXNG)
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

        prompt = (
            "You are Nix, a friendly assistant running on a private "
            "home server. You can answer general questions, chat, and "
            "you have web search available. Answer in one or two short "
            "sentences and never append follow-up offers or questions.\n\n"
            f"Today is {clock} ({TIMEZONE}). This is authoritative: "
            "you DO know the current date, time and day of week - "
            "never say you cannot know or access them. Use it for "
            "any time-relative question (now, today, tomorrow)."
        )

        if digest:
            prompt += (
                "\n\nMEMORY (from their private knowledge base - use it "
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

    def _handle_chat(self, text: str) -> dict[str, Any]:
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
                return self._handle_knowledge(text)

        history = self.actions.context()

        try:
            reply = self.ollama.chat(
                system_prompt=self._chat_system_prompt(current_text=text),
                history=history,
                user_text=text,
            )
        except Exception as exc:
            return {
                "route": CHAT,
                "rule": "ollama_unreachable",
                "reply": (
                    "Nix Alert: Unable to reach the chat model "
                    f"({self.ollama.model}) right now. "
                    f"({type(exc).__name__})"
                ),
                "details": {"error": str(exc)},
            }

        result = {
            "route": CHAT,
            "rule": "ollama",
            "reply": reply or "(empty reply from chat model)",
            "details": {"model": self.ollama.model},
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