from __future__ import annotations

"""
Deterministic intent routing for Nix Knowledge.

The selector model handles arbitrary phrasing, but a class of
knowledge requests is lexically unambiguous. Routing those in Python
is faster, perfectly consistent, and keeps the 0.5B model in its
competence zone. This follows the project's own architecture rule:
Python is authoritative; the model proposes, Python disposes.

route() returns (function_name, arguments) or None when no rule
matches - the caller then falls back to the model.
"""

import re
from typing import Any


# ----------------------------------------------------------------------
# Temporal suffix / prefix resolution
# ----------------------------------------------------------------------


def extract_temporal_suffix(
    text: str,
    resolver,
) -> tuple[str, str] | None:
    """
    Split '<title> <temporal expression>' by finding the longest
    word-suffix of the text that the temporal resolver understands.

    Example:
        "dentist appointment sunday at 6:30pm"
        -> ("dentist appointment", "sunday at 6:30pm")

    Returns None when no suffix resolves.
    """
    text = text.strip().rstrip(".!?")

    tokens = text.split()

    # longest first so "sunday at 6:30pm" beats "at 6:30pm"
    for size in range(len(tokens) - 1, 0, -1):
        candidate = " ".join(tokens[-size:])

        if resolver.resolve(candidate) is not None:
            title = " ".join(tokens[:-size]).strip()
            return title, candidate

    return None


def extract_temporal_prefix(
    text: str,
    resolver,
    max_words: int = 5,
) -> tuple[str, str] | None:
    """
    Split '<temporal expression> <title>' by finding the longest
    word-prefix of the text that the temporal resolver understands.

    Example:
        "this weekend i have pizza party"
        -> ("this weekend", "i have pizza party")

    Returns None when no prefix resolves.
    """
    text = text.strip().rstrip(".!?")

    tokens = text.split()

    for size in range(min(max_words, len(tokens) - 1), 0, -1):
        candidate = " ".join(tokens[:size])

        if resolver.resolve(candidate) is not None:
            return candidate, " ".join(tokens[size:]).strip()

    return None


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

_RECALL_ABOUT = re.compile(
    r"^(?:what|anything)\s+do\s+(?:you|i)\s+remember(?:\s+anything)?\s+about\s+(?P<query>.+?)\s*\??$",
    re.IGNORECASE,
)

_RECALL_DIRECT = re.compile(
    r"^do\s+you\s+remember\s+(?P<query>.+?)\s*\??$",
    re.IGNORECASE,
)

# "who is Maanvi" / "who's alex" - recall about a PERSON. The engine
# rule fires and find_facts decides with real data: a known person
# returns their keys, an unknown one returns nothing (which the brain
# uses to fall through to a world-knowledge chat answer).
_WHO_IS = re.compile(
    r"^who(?:'s|\s+is)\s+(?P<name>[a-z][a-z' ]*?)\s*\??$",
    re.IGNORECASE,
)

# Identity recall is personal memory, never open-ended chat or a state
# lookup. Route all natural variants to the deterministic name/fact query.
_IDENTITY_RECALL = re.compile(
    r"^(?:who\s+am\s+i|do\s+you\s+know\s+(?:who\s+)?i\s+am|"
    r"do\s+you\s+know\s+me|what\s+do\s+you\s+know\s+about\s+me)\??$",
    re.IGNORECASE,
)

# Self-introduction detection: a sentence that identifies the user.
# "my name is X" / "call me X" / "i was born ..." / "people call me X"
_INTRO_MARKERS = re.compile(
    r"\b(?:my name is|i am born|i was born|people call me|call me\s+[a-z]|"
    r"i go by|i'm called)\b",
    re.IGNORECASE,
)

_FACT = re.compile(
    r"^(?:please\s+)?(?:remember|don't forget|keep in mind)(?:\s+that|\s*,)?\s+(?P<value>.+?)\s*\.?$",
    re.IGNORECASE,
)

# Preference statements: "I love/hate/like/dont like X". Verbs that
# express a stance; negated stance verbs ("dont like") are themselves
# stances, so they are matched here rather than treated as negation.
# The leading "I" is optional so continuation clauses inside a
# compound sentence ("..., but love programming") also match.
# Incidental notes: "just so you know, my spare key is under the
# mat". Marker phrase + personal pronoun -> durable fact.
_INCIDENTAL_NOTE = re.compile(
    r"^(?:just\s+so\s+you\s+know|fyi|heads\s*up|"
    r"for\s+future\s+reference|in\s+case\s+you\s+need\s+it|btw)"
    r"\s*[,:-]?\s+(?P<value>.+)$",
    re.IGNORECASE,
)

_PREFERENCE = re.compile(
    r"^(?:i\s+)?"
    r"(?P<verb>"
    r"(?:(?:really|truly)\s+)?"
    r"(?:dont\s+like|don't\s+like|do\s+not\s+like|hate|dislike|"
    r"love|like|enjoy|adore|am\s+into|am\s+passionate\s+about)"
    r")\s+(?P<value>.+?)\s*\.?$",
    re.IGNORECASE,
)

# Capturing so re.split() interleaves the conjunctions with the
# clauses: "robotics, but love programming" ->
# ["robotics", "but", "love programming"]
_CONJUNCTION_SPLIT = re.compile(
    r"\s*,?\s*\b(but|and|however)\b\s*",
    re.IGNORECASE,
)

_NEGATIVE_VERBS = {
    "hate",
    "dislike",
    "dont like",
    "don't like",
    "do not like",
}

# "my alarm code is 2009" / "my garage passcode is 8817" - personal
# statements are durable facts even without a store verb.
_PERSONAL_STATEMENT = re.compile(
    r"^my\s+(?P<subject>[\w'\- ]{1,40}?)\s+(?:is|are|was|were)\s+(?P<value>.+?)\s*\.?$",
    re.IGNORECASE,
)

# Spoken address before the real request: "nix remember my code is...".
# "nic"/"nick" are common ASR renderings of "nix".
_ADDRESS_RE = re.compile(
    r"^(?:hey|hi|hello|ok|okay|so|um|uh|alright|yo)?\s*"
    r"(?:nix|nic|nick)\b[,:! ]+",
    re.IGNORECASE,
)

_CALENDAR_BARE = re.compile(
    r"^(?:what(?:'s| is| do i have)(?: on)? (?:my|the) (?:calendar|schedule)|"
    r"show me (?:my|the) (?:calendar|schedule|events)|"
    r"what events do i have)\s*\??$",
    re.IGNORECASE,
)

# Trailing time frames for calendar browsing and existence probes.
_WINDOW_ALT = (
    r"today|tomorrow|tmr|yesterday|day\s+after\s+tomorrow|"
    r"this\s+week|next\s+week|last\s+week|"
    r"this\s+weekend|this\s+month|"
    r"upcoming|coming\s+up|soon|"
    r"(?:on\s+|next\s+|this\s+)?"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
)

# Calendar browsing with a time frame: "what events do I have today",
# "what's on my schedule this week", "what do i have friday".
_CALENDAR_WINDOWED = re.compile(
    r"^(?:"
    r"what(?:'s| is| do i have)(?: on)? (?:my|the)?\s*(?:calendar|schedule|events)"
    r"|what events do i have"
    r"|what do i have"
    r"|show (?:me|my) (?:calendar|schedule|events)"
    r")"
    rf"\s+(?:for\s+|on\s+|in\s+)?(?P<window>{_WINDOW_ALT})"
    r"\s*\??$",
    re.IGNORECASE,
)

# Existence probes with a time frame: "do I have anything tomorrow",
# "is there anything on friday".
_EVENT_LOOKUP_WINDOWED = re.compile(
    r"^(?:do i have|is there|have i got)\s+"
    r"(?:anything|something|any events|any plans)"
    rf"(?:\s+(?:on|for|in))?\s+(?P<window>{_WINDOW_ALT})"
    r"\s*\??$",
    re.IGNORECASE,
)

_WHEN_IS = re.compile(
    r"^(?:when|what\s+time)\s+(?:is|are)\s+"
    r"(?:my|the|a|an)?\s*(?P<title>.+?)\s*\??$",
    re.IGNORECASE,
)

# "what is my wifi password" / "what are my coworkers' emails" -> fact
# recall. Calendar nouns are excluded so "what is my schedule" keeps
# flowing to the calendar-browsing rule below.
_WHAT_IS_MY = re.compile(
    r"^what(?:'s|\s+is|\s+are|\s+was|\s+were)\s+my\s+(?P<query>.+?)\s*\??$",
    re.IGNORECASE,
)

_CALENDAR_NOUNS_RE = re.compile(
    r"\b(?:schedule|calendar|agenda|events?|plans?|meetings?|"
    r"appointments?|classes?)\b",
    re.IGNORECASE,
)

# "remind me to call mom tomorrow at 5pm" -> a scheduled event whose
# reminder action fires through the bridge into nix_actions.
_REMIND_TO = re.compile(
    r"^(?:please\s+)?remind\s+me\s+(?:to\s+|about\s+)?(?P<rest>.+?)\s*\.?$",
    re.IGNORECASE,
)

# "set an alarm for 7am" / "wake me up at 6" -> alarm event; the
# bridge schedules the nix_actions alarm.
_ALARM_SET_RE = re.compile(
    r"^(?:set|create|add)\s+(?:an?\s+)?(?:alarm|wake\s*up\s*call)"
    r"\s+(?:for|at)\s+(?P<expr>.+?)\s*\.?$",
    re.IGNORECASE,
)

_WAKE_RE = re.compile(
    r"^wake\s+me(?:\s+up)?\s+(?:at|for|by)?\s*(?P<expr>.+?)\s*\.?$",
    re.IGNORECASE,
)

_EVENT_LOOKUP = re.compile(
    r"^(?:do i have|is there|have i got)\s+(?:a|an|my)?\s*(?P<title>.+?)"
    r"(?:\s+(?:coming up|scheduled|soon|planned))?\s*\??$",
    re.IGNORECASE,
)

_CANCEL = re.compile(
    r"^(?:cancel|delete|remove|get rid of)\s+(?:my|the|a|an)?\s*(?P<title>.+?)\s*\.?$",
    re.IGNORECASE,
)

_MOVE = re.compile(
    r"^(?:move|reschedule|change|push)\s+(?:my|the|a|an)?\s*(?P<title>.+?)\s+to\s+(?P<expr>.+?)\s*\.?$",
    re.IGNORECASE,
)

_CREATE = re.compile(
    r"^(?:i have|i've got|i got|schedule|add|create)(?:\s+(?:a|an|my))?\s+(?P<rest>.+?)\s*\.?$",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Preference extraction
# ----------------------------------------------------------------------


def _normalize_verb(verb: str) -> str:
    verb = " ".join(verb.lower().split())
    verb = re.sub(r"^(?:really|truly)\s+", "", verb)
    verb = verb.replace("dont", "don't").replace("do not", "don't")
    return verb


def _preference_clauses(sentence: str) -> list[str] | None:
    """
    Build fact clauses from a preference sentence, splitting on
    coordinating conjunctions and clause-marking commas. Clauses with
    an explicit stance verb are normalized ("i dont like x" ->
    "I don't like x"); plain continuation clauses are kept verbatim.

        "i hate robotics, but love programming"
        -> ["I hate robotics", "I love programming"]

        "i love programming, i hate robotics and electrical is fun"
        -> ["I love programming", "I hate robotics",
            "electrical is fun"]

    Returns None when the sentence expresses no stance.
    """
    match = _PREFERENCE.match(sentence)

    if match is None:
        return None

    verb = _normalize_verb(match.group("verb"))
    value = match.group("value").strip()

    # "..., i hate x" - a comma followed by a pronoun starts a new
    # independent clause, not a continuation
    value = re.sub(
        r",\s+(?=(?:i|we)\b)",
        ", and ",
        value,
        flags=re.IGNORECASE,
    )

    clauses: list[str] = []

    parts = _CONJUNCTION_SPLIT.split(value)

    for index, part in enumerate(parts):
        part = part.strip()

        if not part:
            continue

        if index % 2 == 1:
            # the conjunction itself; handled by clause ordering
            continue

        sub_match = _PREFERENCE.match(part)

        if sub_match is not None:
            sub_verb = _normalize_verb(sub_match.group("verb"))
            sub_value = sub_match.group("value").strip()

            if sub_value:
                clauses.append(f"I {sub_verb} {sub_value}")
        elif index == 0:
            # the first clause carries the sentence's own verb
            clauses.append(f"I {verb} {part}")
        else:
            # continuation phrase: kept verbatim, e.g.
            # "... but electrical is fun"
            clauses.append(part)

    if not clauses:
        clauses.append(f"I {verb} {value}")

    return clauses


# Compound requests split into independent clauses before these
# lead-ins. The lookahead keeps "meeting with bob and alice" or
# "pizza and burgers" unsplit while splitting "... and I have ...",
# "... and a meeting ...", "...; cancel my ...".
# Compound splitter. The lookahead also breaks before a bare event
# noun: "... next week and art class this weekend" - two events in
# one breath, each keeps its own time.
_COMPOUND_SPLIT = re.compile(
        r"\s*[;,]\s*\b(?:but|and|then|also)\s+"
        r"|\s*;\s*"
        r"|\s*,\s*(?=(?:i|we)\b)"
        r"|\s*\b(?:and|but|then|also)\s+"
        r"(?="
        r"(?:i|we|please)\b"
        r"|(?:cancel|delete|remove|reschedule|move|schedule|add|create|"
        r"remember|don't|dont|show|find|what|when|where|do|does|is|are)\b"
        r"|(?:a|an|my|the)\b"
        r"|(?:tomorrow|tmr|yesterday|today|next|this|every)\b"
        r"|(?:meeting|appointment|class|event|party|dinner|lunch|"
        r"breakfast|exam|test|interview|flight|call|session|game|match|"
        r"deadline|birthday|hangout|gathering|visit|trip|practice|"
        r"lesson|movie night|game night|study group|haircut|"
        r"inspection|conference|checkup)\b"
        r")",
        re.IGNORECASE,
    )

_CREATE_LEAD_IN = "i have "

_BARE_CONJ = re.compile(
    r"\s*,?\s*\b(?:and|then|also)\b\s*",
    re.IGNORECASE,
)


def _has_resolvable_time(segment: str, resolver) -> bool:
    """True when the segment carries its own resolvable time, leading
    ("monday meeting") or trailing ("art class this weekend")."""
    return (
        extract_temporal_suffix(segment, resolver) is not None
        or extract_temporal_prefix(segment, resolver) is not None
    )


def _split_all_temporal(
    parts: list[str],
    resolver,
) -> list[str]:
    """When the first pass produced one segment, try splitting it at
    bare conjunctions - but only when every resulting segment has its
    own time. That is the reliable signature of multiple events."""
    if len(parts) != 1:
        return parts
    segments = [
        s.strip(" \t;, :")
        for s in _BARE_CONJ.split(parts[0])
        if s.strip(" \t;, :")
    ]
    if len(segments) < 2:
        return parts
    if all(
        _has_resolvable_time(segment, resolver)
        for segment in segments
    ):
        return segments
    return parts


# ----------------------------------------------------------------------
# Compound request decomposition
# ----------------------------------------------------------------------


def decompose(
    request: str,
    resolver,
) -> list[tuple[str, dict[str, Any]]] | None:
    """
    Split a compound request into independently-routed subtasks.

    "i have a dentist appointment tomorrow and a meeting with bob on
    friday" -> two create_calendar_event subtasks.

    Only returns a decomposition when EVERY part routes deterministically
    (trailing fragments may need the "i have" lead-in retried); otherwise
    returns None and the request is handled as a whole. Fragments routed
    only via the lead-in keep the lead-in arguments - the model layer is
    never asked to guess fragment intent.
    """
    text = request.strip().rstrip(".!?")

    # A self-introduction is ONE profile statement, never compound:
    # decomposing it produces calendar subtasks and garbled facts.
    if _INTRO_MARKERS.search(text):
        return None

    parts = [
        part.strip(" \t;, :")
        for part in _COMPOUND_SPLIT.split(text)
    ]

    parts = [part for part in parts if part]

    if len(parts) < 2:
        # Second pass: a conjunction joining two segments that EACH
        # carry their own time ("... next week and art class this
        # weekend") - two events in one breath, even though the word
        # right after "and" is not itself a trigger word. Runs without
        # the _COMPOUND_SPLIT gate: the whole point is catching splits
        # the trigger-word lookahead cannot express.
        parts = _split_all_temporal(parts or [text], resolver)
        if len(parts) < 2:
            return None

    routed: list[tuple[str, dict[str, Any]]] = []
    inherited_expression: str | None = None

    for part in parts:
        result = route(part, resolver)

        if result is None:
            # trailing fragment: "... and a meeting with bob on friday"
            # is a create with the verb elided
            result = route(_CREATE_LEAD_IN + part, resolver)

        if result is None:
            # leading-temporal split: "this weekend i have pizza party
            # and tomorrow i have dentist" splits after the wrong
            # anchor - "this weekend ... party" + "and tomorrow ...".
            # Rejoin the leading temporal of the previous part with
            # this part and retry.
            if inherited_expression is not None:
                recombined = (
                    f"{_CREATE_LEAD_IN}{inherited_expression} {part}"
                )

                result = route(recombined, resolver)

        if result is None:
            # one unroutable part: no safe decomposition
            return None

        name, arguments = result

        if name == "create_calendar_event":
            inherited_expression = arguments.get("temporal_expression")

            if inherited_expression and inherited_expression not in part:
                # arguments came from an inherited anchor: re-route the
                # original fragment so storage reflects what was said
                recombined = f"{_CREATE_LEAD_IN}{part} {inherited_expression}"

                rerouted = route(recombined, resolver)

                if rerouted is not None:
                    name, arguments = rerouted

        routed.append((name, arguments))

    if len(routed) < 2:
        return None

    return routed


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def route(
    request: str,
    resolver,
) -> tuple[str, dict[str, Any]] | None:
    """
    Deterministically select a Knowledge function for lexically
    unambiguous requests. Returns None to defer to the model.
    """
    text = request.strip()
    lowered = text.lower().rstrip(".!?")
    # Common speech-to-text spelling; keep the symbolic calendar rules
    # deterministic instead of delegating a simple window to the selector.
    lowered = re.sub(r"\bcalender\b", "calendar", lowered)

    # Spoken address ("nix remember my ...", "nic what was my ...").
    address_match = _ADDRESS_RE.match(lowered)
    if address_match:
        lowered = lowered[address_match.end():].strip()

    # Voice transcripts often retain conversational lead-ins when the
    # Knowledge service is called directly (Core normally removes some
    # of these first). Keep the service boundary equally tolerant.
    for prefix in ("real quick", "quickly", "ok so", "okay so", "hey", "okay", "ok", "so"):
        marker = prefix + " "
        if lowered.startswith(marker):
            lowered = lowered[len(marker):].lstrip(" ,:-")
            break

    # Polite command wrappers: "can you remind me to...", "could you
    # please cancel my...". The request starts after the wrapper.
    stripped = re.sub(
        r"^(?:can|could|will|would)\s+you\s+(?:please\s+)?"
        r"(?:kindly\s+)?",
        "",
        lowered,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"^please\s+(?:kindly\s+)?", "", stripped,
        flags=re.IGNORECASE,
    )
    if stripped != lowered:
        lowered = stripped.rstrip(".!?").strip()

    # Speech often appends politeness after the actual request: "what
    # events do I have next week, please". It carries no intent data.
    lowered = re.sub(r"\s+(?:please|okay|ok)$", "", lowered).strip()

    if not lowered:
        return None

    # ---- identity recall must be checked before state/world rules.
    # The stored key finder knows the user's name and can return the
    # authoritative answer; Casper must not infer identity from chat.
    if _IDENTITY_RECALL.match(lowered):
        return "find_facts", {"query": "name"}

    # ---- recall: "do you remember X" must be checked BEFORE "remember X"
    match = _RECALL_ABOUT.match(lowered)
    if match:
        query = match.group("query").strip()
        return "find_facts", {"query": query} if query else {}

    match = _RECALL_DIRECT.match(lowered)
    if match:
        query = match.group("query").strip()
        return "find_facts", {"query": query} if query else {}

    # ---- self-introduction: store profile keys deterministically.
    # A pure self-introduction carries no other actionable intent;
    # without this rule the model gate guesses a calendar function
    # and the reply is about nothing.
    if _INTRO_MARKERS.search(lowered):
        from .keys import extract_keys

        intro_keys = extract_keys(text)
        if intro_keys:
            return "store_profile_keys", {"keys": intro_keys}

    # ---- "who is Maanvi" - person recall. Must run BEFORE the
    # what/who world-question fallthrough.
    match = _WHO_IS.match(lowered)
    if match:
        name = match.group("name").strip()
        if name and name.lower() not in (
            "my", "your", "the", "that", "this", "there",
            "he", "she", "it", "they",
        ):
            return "find_facts", {"query": name}

    # ---- current states of close people ("my sister is sick",
    # "maanvi is cured now") must run BEFORE create_fact: a state is
    # temporary and gets superseded, unlike durable facts.
    try:
        from .states import parse_state_statement

        if parse_state_statement(lowered) is not None:
            return "create_state", {"statement": text.strip()}
    except Exception:  # noqa: BLE001 - state parsing must never break routing
        pass

    # ---- state recall: "how is my sister" / "how was my sister
    # doing last time" -> current states + dated moments history -----
    match = re.match(
        r"how\s+(?:is|are|was|were|'s)\s+(.+?)(?:\s+doing|\s+feeling)?\??$",
        lowered,
    )
    if match and not re.match(
        r"(?:you|it|that|this|things|life|everything|the\s+weather)\b",
        match.group(1),
    ):
        return "find_states", {"query": match.group(1).strip()}

    # ---- store fact
    match = _FACT.match(lowered)
    if match:
        value = match.group("value").strip()
        if value:
            return "create_fact", {"value": value}
        return None

    # ---- "my alarm code is 2009" - a fact statement, NOT an alarm
    # request. Runs before the reminder/alarm rules (below) because
    # "alarm" in a possessive statement is data, not a scheduling verb.
    match = _PERSONAL_STATEMENT.match(lowered)
    if (
        match
        and not lowered.endswith("?")
        and not _CALENDAR_NOUNS_RE.search(lowered)
    ):
        subject = match.group("subject").strip()
        value = match.group("value").strip()
        if subject and value:
            return "create_fact", {"value": f"my {subject} is {value}"}

    # ---- incidental notes ("just so you know, my spare key is...")
    match = _INCIDENTAL_NOTE.match(lowered)
    if match:
        value = match.group("value").strip()
        if value and re.search(r"\b(?:my|our)\b", value):
            return "create_fact", {"value": value}

    # ---- preference statements ("I hate X, but love Y")
    preference_clauses = (
        None
        if lowered.endswith("?")
        else _preference_clauses(lowered)
    )
    if preference_clauses:
        return "create_fact", {"value": "; ".join(preference_clauses)}

    # ---- "what is my wifi password" -> fact recall (calendar nouns
    # fall through to the calendar rules below)
    match = _WHAT_IS_MY.match(lowered)
    if match and not _CALENDAR_NOUNS_RE.search(match.group("query")):
        query = match.group("query").strip()
        if query:
            return "find_facts", {"query": query}

    # ---- "when is my tsa meeting"
    match = _WHEN_IS.match(lowered)
    if match:
        title = match.group("title").strip()
        if title:
            return "find_calendar_events", {"title": title}

    # ---- bare calendar lookups
    if _CALENDAR_BARE.match(lowered):
        return "find_calendar_events", {}

    # ---- calendar browsing with a time frame:
    # "what events do I have today", "what's on my schedule this week"
    match = _CALENDAR_WINDOWED.match(lowered)
    if match:
        return "find_calendar_events", {
            "window": match.group("window")
        }

    # ---- existence probes with a time frame:
    # "do I have anything tomorrow"
    match = _EVENT_LOOKUP_WINDOWED.match(lowered)
    if match:
        return "find_calendar_events", {
            "window": match.group("window")
        }

    # ---- cancel
    match = _CANCEL.match(lowered)
    if match:
        title = match.group("title").strip()
        if title:
            return "cancel_calendar_event", {"title": title}

    # ---- move / reschedule
    match = _MOVE.match(lowered)
    if match:
        title = match.group("title").strip()
        expression = match.group("expr").strip()

        if title and expression:
            arguments: dict[str, Any] = {"title": title}

            if resolver.resolve(expression) is not None:
                arguments["new_temporal_expression"] = expression
            elif re.search(
                r"\b(?:monday|tuesday|wednesday|thursday|friday|"
                r"saturday|sunday|today|tomorrow|tmr|yesterday|week)\b",
                expression,
                re.IGNORECASE,
            ):
                # temporal words left over ("... and gym to tuesday"):
                # the expression is really multiple clauses - defer
                return None
            else:
                arguments["new_title"] = expression

            return "update_calendar_event", arguments

    # ---- "set an alarm for 7am" / "wake me up at 6"
    for pattern in (_ALARM_SET_RE, _WAKE_RE):
        match = pattern.match(lowered)
        if match:
            expression = match.group("expr").strip()
            # bare clock times ("7am", "8:30 am") resolve with an
            # explicit "at" prefix
            if expression and resolver.resolve(expression) is None:
                expression = f"at {expression}"
            if expression and resolver.resolve(expression) is not None:
                return (
                    "create_calendar_event",
                    {"title": "alarm", "temporal_expression": expression},
                )
            return None

    # ---- "remind me to call mom tomorrow at 5pm" -> scheduled
    # reminder event; without a resolvable time, defer to the model
    match = _REMIND_TO.match(lowered)
    if match:
        rest = match.group("rest").strip()

        if rest:
            split = extract_temporal_suffix(rest, resolver)

            if split is not None and split[0]:
                title, expression = split
                return (
                    "create_calendar_event",
                    {
                        "title": title,
                        "temporal_expression": expression,
                    },
                )

            return None

    # ---- event lookup by title
    match = _EVENT_LOOKUP.match(lowered)
    if match:
        title = match.group("title").strip()
        if title:
            return "find_calendar_events", {"title": title}

    # ---- create: '<title> <temporal suffix>'
    match = _CREATE.match(lowered)
    if match:
        rest = match.group("rest").strip()

        split = extract_temporal_suffix(rest, resolver)

        if split is not None:
            title, expression = split

            if title:
                return (
                    "create_calendar_event",
                    {
                        "title": title,
                        "temporal_expression": expression,
                    },
                )

    # ---- create with a leading temporal clause:
    # "this weekend i have pizza party with friends",
    # "tomorrow i have a dentist appointment"
    prefix_split = extract_temporal_prefix(lowered, resolver)

    if prefix_split is not None:
        expression, remainder = prefix_split

        create_match = _CREATE.match(remainder)

        if create_match is not None:
            rest = create_match.group("rest").strip()

            if rest:
                # a trailing temporal expression is more specific
                # than the leading one when both are present
                suffix_split = extract_temporal_suffix(rest, resolver)

                if suffix_split is not None and suffix_split[0]:
                    title, trailing = suffix_split
                    return (
                        "create_calendar_event",
                        {
                            "title": title,
                            "temporal_expression": trailing,
                        },
                    )

                return (
                    "create_calendar_event",
                    {
                        "title": rest,
                        "temporal_expression": expression,
                    },
                )

    return None
