"""
Nix Core request classifier.

Three destinations:

  knowledge  - request touches the user's durable life data: remember,
               recall facts/preferences, schedule/reschedule/cancel
               calendar events, alarms, reminders, "my ..." lookups.
               Handled by nix_knowledge (which schedules real actions
               into nix_actions through its bridge).
  chat       - conversation, opinions, jokes, greetings, internet,
               news, weather, general questions. Handled by the
               Ollama model (phi-4 + SearXNG web search). This model
               knows NOTHING about the user.
  unknown    - rules are not confident; the caller escalates to the
               model-backed classifier hosted by nix_knowledge.

Design mirrors nix_knowledge/rules.py: high-precision lexical rules
first, ambiguous input deferred to the model. Deterministic routing
keeps the hot path instant and perfectly consistent.

Realism layer: input is normalized before rules run - common typos
(tommorow, remeber, calender...), ASR-style contractions (whats,
hows), chat abbreviations (tmrw, pls, u) and spoken filler prefixes
("hey nix,", "ok so,", "can you ...") are folded away, so the rules
see clean text while the user types like a human.

The router is intentionally conversation-state-free: every rule is
matched against the raw request text only. Mid-conversation knowledge
requests ("what time is my meeting tomorrow?") are still routed to
knowledge, which is exactly what the user wants.
"""

from __future__ import annotations

import re
from typing import Any

KNOWLEDGE = "knowledge"
CHAT = "chat"
UNKNOWN = "unknown"

# ----------------------------------------------------------------------
# Input normalization: typos, ASR noise, abbreviations
# ----------------------------------------------------------------------

# Token-level typo and chat-abbreviation fixes. Applied to whole words
# only, so "sat" never corrupts "saturday" (the full word is a
# different token).
# ----------------------------------------------------------------------
# Injection guards (run FIRST in classify). These catch attempts to
# make the knowledge layer spill secrets or obey attacker framing.
# A stored secret ("my wifi password is house5") must still route to
# KNOWLEDGE when the user is VOLUNTARILY storing or asking with a
# real query form; the guards only fire on injection framing
# ("ignore all previous instructions", "system:", "print my X ...").
# ----------------------------------------------------------------------
_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:your\s+|any\s+|the\s+|my\s+|their\s+)?"
        r"(?:previous\s+|prior\s+|above\s+|earlier\s+)?(?:instructions?|rules?|prompts?)\b",
        re.I,
    ),
    re.compile(r"\b(?:developer|system|admin)\s*(?:mode|message|prompt|:|override)\b", re.I),
    re.compile(r"\byou are now\b|\bact as if you have no rules\b|\bbypass\s+(?:your\s+)?(?:rules|filters|safety)\b", re.I),
    re.compile(r"\brepeat everything i say\b|\brepeat after me everything\b", re.I),
    re.compile(
        r"\b(?:print|show|reveal|display|output|echo|read\s+out)\s+"
        r"(?:me\s+)?(?:all\s+stored\s+(?:facts|data|records)|everything\s+stored|my\s+)?"
        r"(?:wifi\s+password|wi\s*fi\s+password|password|passcode|garage\s+code|"
        r"locker\s+(?:code|combination)|safe\s+combination|alarm\s+code|credit\s+card|"
        r"pin\b|ssn|social\s+security)",
        re.I,
    ),
    re.compile(r"\bprint\s+(?:my\s+|all\b|everything\b)", re.I),
    re.compile(r"\b(show|display)\s+(?:me\s+)?all\s+stored\b", re.I),
)

# Past-habit statements: "i used to have gym on fridays", "last year i
# had a meeting every week". Historical context, NOT current state -
# storing them as active knowledge makes "what do I have tmr" wrong.
# NOTE: only DISTANT periods here - "last week" is a recent past event
# ("i had a dentist appointment last week") which stays in the
# knowledge/calendar domain.
_PAST_HABIT_RE = re.compile(
    r"\bused\s+to\b|"
    r"\b(?:last|previous)\s+(?:year|month|semester|summer|winter|fall|spring)\b|"
    r"\bback\s+(?:then|in\s+(?:the\s+)?(?:day|high\s+school|college))\b|"
    r"\bin\s+(?:high\s+school|middle\s+school|elementary\s+school|college)\b",
    re.I,
)

# Negative cognition: talking ABOUT not remembering/knowing is not a
# fact to store ("i can't remember if penguins fly" is a question about
# the world, not a personal fact).
_NEGATIVE_COGNITION_RE = re.compile(
    r"\b(?:can'?t|cannot|don'?t|do\s+not|never|hardly)\s+"
    r"(?:remember|recall|figure\s+out|tell)\b|"
    r"\bno\s+idea\b|\bnot\s+sure\s+(?:if|whether|about)\b|"
    r"\bdon'?t\s+know\s+if\b",
    re.I,
)

# Exemptions from the negative-cognition guard:
#  - an embedded wh-clause ("i can never remember WHEN trash day IS")
#    is an implicit reminder/recall request -> knowledge rules handle it
#  - "my <thing>" ("i don't remember my wifi password") is a recall
#    attempt, not world-questioning -> let the normal rules decide
_NEGATIVE_EXEMPTION_RE = re.compile(
    r"\b(?:when|what|where|which|who|how)\b[^.?!]*\b(?:is|are|was|were)\b"
    r"|\bmy\b",
    re.I,
)

# Bare fragments: a lone date/time phrase is temporal context, not
# knowledge. Only fires when the WHOLE (address-stripped) request is
# a fragment.
_FRAGMENT_RE = re.compile(
    r"^(?:next|this|last|the\s+)?\s*"
    r"(?:week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|morning|afternoon|evening|night)"
    r"(?:\s+(?:(?:at|around|on)\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?"
    r"[.!?]*$",
    re.I,
)

_TYPO_FIXES = {
    "tommorow": "tomorrow",
    "tommorrow": "tomorrow",
    "tomorow": "tomorrow",
    "2moro": "tomorrow",
    "2morrow": "tomorrow",
    "tmrw": "tomorrow",
    "tmw": "tomorrow",
    "tonite": "tonight",
    "wknd": "weekend",
    "remeber": "remember",
    "remmber": "remember",
    "rember": "remember",
    "calender": "calendar",
    "calander": "calendar",
    "scedule": "schedule",
    "schdule": "schedule",
    "skedule": "schedule",
    "schedual": "schedule",
    "alaram": "alarm",
    "remider": "reminder",
    "remender": "reminder",
    "apointment": "appointment",
    "appoitment": "appointment",
    "appt": "appointment",
    "rescedule": "reschedule",
    "reschedual": "reschedule",
    "reschdule": "reschedule",
    "pasword": "password",
    "passward": "password",
    "bday": "birthday",
    "mtg": "meeting",
    "msg": "message",
    "wensday": "wednesday",
    "tueseday": "tuesday",
    "tues": "tuesday",
    "thurs": "thursday",
    "weds": "wednesday",
    "mon": "monday",
    "sat": "saturday",
    "sun": "sunday",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "u": "you",
    "ur": "your",
    "r": "are",
}

# ASR-style contraction expansion (only forms that would otherwise
# break \b word boundaries in the rules; dont/im/ive forms are already
# in the lexicons verbatim).
_CONTRACTIONS = {
    "what's": "what is",
    "whats": "what is",
    "when's": "when is",
    "whens": "when is",
    "where's": "where is",
    "wheres": "where is",
    "who's": "who is",
    "whos": "who is",
    "how's": "how is",
    "hows": "how is",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
}

# Spoken/typed filler prefixes, longest first. Stripped repeatedly from
# the front so "hey nix can you remember ..." -> "remember ...".
_ADDRESS_PREFIXES = (
    "one more thing",
    "by the way",
    "real quick",
    "quick question",
    "quick one",
    "so basically",
    "basically",
    "i need you to",
    "i want you to",
    "i was wondering",
    "hey nix",
    "guess what",
    "can you",
    "could you",
    "would you",
    "will you",
    "please",
    "just",
    "also",
    "and then",
    "and",
    "now",
    "hey",
    "hi",
    "yo",
    "hello",
    "nix",
    "ok so",
    "okay",
    "ok",
    "so",
    "um",
    "uh",
    "well",
    "alright",
    "anyway",
    "btw",
    "fyi",
)


def _normalize(text: str) -> str:
    """Fold typos, abbreviations, contractions and whitespace away."""
    lowered = (text or "").lower().strip()

    for contracted, expanded in _CONTRACTIONS.items():
        lowered = re.sub(
            rf"\b{re.escape(contracted)}\b", expanded, lowered
        )

    tokens = [_TYPO_FIXES.get(token, token) for token in lowered.split()]
    lowered = " ".join(tokens)

    return re.sub(r"\s+", " ", lowered).strip()


def _strip_address(lowered: str) -> str:
    """Remove leading filler/address phrases ("hey nix, ...")."""
    changed = True
    while changed:
        changed = False
        for prefix in _ADDRESS_PREFIXES:
            if lowered == prefix:
                # pure address ("hey nix") - signal with empty string
                return ""
            for lead in (prefix + " ", prefix + ","):
                if lowered.startswith(lead):
                    lowered = lowered[len(lead):].lstrip(" ,.:-").strip()
                    changed = True
                    break
    return lowered


# ----------------------------------------------------------------------
# Lexical material
# ----------------------------------------------------------------------

# Verbs that create or modify durable records.
_STORE_VERBS = (
    "remember",
    # "save"/"store" only in storage shapes - bare "save" would hit
    # "how to save money on groceries" (chat advice).
    "save this",
    "save that",
    "save it",
    "save my",
    "save our",
    "store this",
    "store that",
    "store my",
    "note that",
    "note to self",
    "keep in mind",
    "don't forget",
    "dont forget",
    "write down",
    "log that",
)

_RECALL_LEADINS = (
    "what do i",
    "what did i",
    "what am i",
    "where do i",
    "where did i",
    "when do i",
    "when did i",
    "when is my",
    "when are my",
    "when does my",
    "when's my",
    "where is my",
    "where's my",
    "where does my",
    "who is my",
    "who's my",
    "which is my",
    "do i have",
    "do i like",
    "do i hate",
    "did i",
    "did we",
    "have i",
    "have we",
    "am i supposed to",
    "am i free",
    "am i busy",
    "remind me what",
    "remind me who",
    "remind me when",
    "remind me where",
    "find my",
    "locate my",
    "do you remember",
    "do you recall",
    "do you know my",
    "what meetings do i",
    "what events do i",
    "what appointments do i",
    "what classes do i",
    "any plans",
    "any meetings",
    "any events",
    "any appointments",
)

_RECALL_NOUNS = (
    "my schedule",
    "my calendar",
    "my day",
    "my week",
    "my weekend",
    "my plans",
    "my plan",
    "my agenda",
    "my events",
    "my meeting",
    "my meetings",
    "my appointment",
    "my appointments",
    "my class",
    "my classes",
    "my fact",
    "my facts",
    "my preference",
    "my preferences",
    "my notes",
    "my tasks",
    "my task",
    "my todo",
    "my todos",
    "my list",
    "my birthday",
    "my anniversary",
    "the schedule",
    "the calendar",
    "the agenda",
    "our plans",
    "our calendar",
    "our schedule",
    "our meeting",
)

# Scheduling intent: always durable, always knowledge.
_SCHEDULING_VERBS = (
    "schedule",
    "reschedule",
    "set an alarm",
    "set alarm",
    "set a reminder",
    "set reminder",
    "wake me",
    "add an event",
    "add event",
    "add to my calendar",
    "add to calendar",
    "put on my calendar",
    "book",
    "cancel my",
    "cancel the",
    "move my",
    "moved my",
    "postpone my",
    "push my",
    "change my",
    "delete my",
    "remove my",
)

_ALARM_RE = re.compile(
    r"\b(?:alarm|wake\s*up\s*call|snooze)\b", re.IGNORECASE
)

_REMINDER_RE = re.compile(
    r"\b(?:remind(?:er|\s*me)?|reminder)\b", re.IGNORECASE
)

_REMINDER_LEAD_RE = re.compile(
    r"(?:^|\b)(?:remind\s+me|reminder\s*(?::|to|for)|"
    r"set\s+a?\s*reminder|a\s+reminder\s+for)\b",
    re.IGNORECASE,
)

# Preference statements: durable facts the knowledge engine stores.
# "i hate pineapple", "i really love metal music". Addressing the
# assistant ("i like your haircut", "i love you") is chat, not a fact.
_PREFERENCE_RE = re.compile(
    r"^(?:i|we)\s+(?:really\s+|actually\s+|just\s+)?"
    r"(?:love|like|hate|prefer|enjoy|adore|dislike|can'?t\s+stand)\b",
    re.IGNORECASE,
)

# Health facts: "i'm allergic to peanuts".
_ALLERGIC_RE = re.compile(
    r"^(?:i\s+(?:am\s+)?|i'm\s+|im\s+|we\s+are\s+|we're\s+)allergic\b",
    re.IGNORECASE,
)

# Whereabouts: "i left my keys in the car", "i lost my wallet".
_LOST_FOUND_RE = re.compile(
    r"^(?:i|we)\s+(?:just\s+)?(?:left|put|lost|found|placed)\b"
    r".{0,40}?\bmy\b",
    re.IGNORECASE,
)

# Family facts without "my": "mom's birthday is may 4".
# "mom's birthday is may 4" / "uncle joe's birthday is next month" -
# optional first name between relation and occasion.
_FAMILY_EVENT_RE = re.compile(
    r"\b(?:mom|dad|sister|brother|wife|husband|son|daughter|grandma|"
    r"grandpa|uncle|aunt|cousin|niece|nephew)(?:'s|\s+\w+'?s?)?\s+"
    r"(?:birthday|anniversary)\s+(?:is|will\s+be)\b",
    re.IGNORECASE,
)

# Event nouns after "i have / we have / i got" mark scheduling intent.
# Words like "question"/"problem" deliberately stay out: "i have a
# question about black holes" is chat.
_EVENT_NOUNS = (
    "meeting",
    "appointment",
    "class",
    "event",
    "party",
    "dinner",
    "lunch",
    "breakfast",
    "exam",
    "test",
    "interview",
    "flight",
    "call",
    "session",
    "game",
    "match",
    "deadline",
    "birthday",
    "hangout",
    "gathering",
    "visit",
    "trip",
    "practice",
    "lesson",
    "movie night",
    "game night",
    "date night",
    "study group",
    "haircut",
    "car inspection",
    "inspection",
    "parent teacher conference",
    "conference",
    "checkup",
    "dentist",
    "doctor",
)

# "tomorrow i have ... " / "this weekend i have ..." - temporal openers
# followed by event creation.
_TEMPORAL_OPENER_RE = re.compile(
    r"^(?:today|tomorrow|tmr|tonight|yesterday|this|next|every|"
    r"on\s+\w+day|in\s+(?:a\s+)?(?:few|couple|\d+))\b",
    re.IGNORECASE,
)

_I_HAVE_RE = re.compile(
    r"\b(?:i|we)\s+(?:have|got|(?:'ve|ve)\s+got)\b", re.IGNORECASE
)

# Personal-knowledge nouns: "my" + relationship/life entity. Storing or
# asking about these always needs the knowledge engine.
_MY_RE = re.compile(r"\bmy\s+\w+", re.IGNORECASE)

# "remember this: ... " / "remember that i ..." -> store fact.
_REMEMBER_RE = re.compile(
    r"\b(?:remember|don'?t\s*forget|keep\s+in\s+mind)\b"
    r".{0,40}?(?:this|that|:)\b",
    re.IGNORECASE,
)

_RECALL_RE = re.compile(
    r"\b(?:what|where|when|who|which)\b.{0,30}?\bmy\b",
    re.IGNORECASE,
)

# Questions about personal codes/credentials: "what's the wifi
# password" (users say "the", meaning their own).
_PERSONAL_CODE_RE = re.compile(
    r"^(?:what|where|which|do|does|is|can|tell)\b.*"
    r"\b(?:wifi|wi\s?fi|password|passcode|pass\s?code)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Chat-side lexical material
# ----------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|yo|sup|howdy|helo|heyy|greetings|"
    r"good\s*(?:morning|afternoon|evening|night)|"
    r"morning|afternoon|evening|night)\b[\s!,.?]*$",
    re.IGNORECASE,
)

_CHAT_OPENERS = (
    "tell me a joke",
    "tell me another joke",
    "tell me about yourself",
    "tell me something interesting",
    "tell me a fun fact",
    "who are you",
    "what are you",
    "how are you",
    "what is up",
    "what's up",
    "whats up",
    "how is it going",
    "how's it going",
    "what can you do",
    "are you a robot",
    "are you human",
    "who made you",
    "who built you",
    "what model are you",
    "how old are you",
    "thank you",
    "thanks",
    "nice",
    "cool",
    "i'm bored",
    "im bored",
    "i am bored",
    "im sleepy",
    "im tired",
    "cheer me up",
    "say something nice",
    "hi there",
    "hey there",
    "hello there",
    "entertain me",
    "let's chat",
    "lets chat",
    "talk to me",
    "long time no see",
    "guess what",
    "i have news",
    "good vibes only",
    "surprise me",
    "flip a coin",
    "roll a dice",
    "roll the dice",
    "roll a die",
    "make me laugh",
    "pick a number",
    "tell me a riddle",
    "say a tongue twister",
    "play 20 questions",
    "quiz me",
    "write a haiku",
    "write a poem",
    "tell me a dad joke",
    "give me a fun fact",
    "give me a random fact",
    "would you rather",
)

# Internet / world-knowledge question shapes: the phi-4 + SearXNG
# model's home turf.
_WORLD_Q = (
    "who is",
    "who was",
    "who won",
    "who invented",
    "what is",
    "what are",
    "what was",
    "where is",
    "where are",
    "how do i",
    "how do you",
    "how can i",
    "how does",
    "how to",
    "how tall is",
    "how many people live",
    "why is",
    "why are",
    "why does",
    "why do",
    "can you explain",
    "can cats eat",
    "can dogs eat",
    "explain",
    "define",
    "search",
    "google",
    "look up",
    "find online",
    "latest news",
    "news about",
    "weather",
    "temperature outside",
    "stock price",
    "score of",
    "release date",
    "recommend me",
    "recommend a",
    "suggest a",
    "best restaurant",
    "restaurant near me",
    "recipe for",
    "translate",
    "what should i",
    "should i bring",
    "convert ",
    " to c",
    " to f",
    " to km",
    " to miles",
    " to kg",
    " to lbs",
    " to ml",
    " to ist",
    " to cst",
    " to gbp",
    " to eur",
    "percent tip on",
    "% tip on",
    " to celsius",
    " to fahrenheit",
)

_WORLD_RE = re.compile(
    r"\b(?:news|weather|forecast|rain|snow|sunny|humidity|wikipedia|"
    r"internet|online|website|recipe|movie|tv show|song|celebrity|"
    r"president|capital of|population of|history of|stock|crypto|"
    r"bitcoin|score|solar eclipse|translate|fun fact|fun facts|"
    r"outside|win|won|game|match|super bowl|olympics)\b",
    re.IGNORECASE,
)

# Second-person address with no personal data => pure chat.
_YOU_RE = re.compile(
    r"\b(?:you|your|yourself|u r|you're|youre)\b", re.IGNORECASE
)

# "tell me ..." / "give me ..." requests for entertainment / world
# info: jokes, fun facts, stories, suggestions. Requests referencing
# the user's own data ("tell me my wifi password") are caught by the
# recall rules above; these rules only fire on the remainder.
_TELL_ME_RE = re.compile(
    r"^(?:tell|give)\s+me\b", re.IGNORECASE
)

# Opinions / preferences asked of the assistant.
_OPINION_RE = re.compile(
    r"\b(?:do you (?:like|think|prefer|know about)|your (?:opinion|"
    r"favorite)|what do you think)\b",
    re.IGNORECASE,
)

# "do you remember the name of that actor" - world recall wearing a
# remember-word. Only chat when no "my" personal marker is present.
_WORLD_RECALL_RE = re.compile(
    r"\bdo\s+you\s+(?:remember|recall)\s+(?:the\s+)?"
    r"(?:name|title|one|song|movie|show|actor|band)\b",
    re.IGNORECASE,
)

# Comparison questions: "python vs javascript", "which is better pc
# or console". World opinions, never personal data.
_COMPARISON_RE = re.compile(
    r"\bvs\.?\b|\bwhich\s+is\s+better\b", re.IGNORECASE
)

# Sun/almanac questions: "sunrise time tomorrow", "when is the next
# full moon", "is today a federal holiday".
_ALMANAC_RE = re.compile(
    r"\b(?:sunrise|sunset|full\s+moon|federal\s+holiday|"
    r"daylight\s+saving|time\s+zone|thanksgiving)\b",
    re.IGNORECASE,
)

# World questions about user-owned devices: "is my laptop supposed to
# get this hot", "how often should my car get an oil change".
_DEVICE_Q_RE = re.compile(
    r"^(?:is|are|does|do|should|can|how\s+(?:often|much|long|hot)|"
    r"what\s+(?:if|speed|year))\b.*\bmy\s+"
    r"(?:laptop|computer|pc|phone|car|internet|router|wifi|tv|"
    r"printer|headphones|battery|resume|essay|name|model)\b",
    re.IGNORECASE,
)


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _starts_with_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(text.startswith(phrase) for phrase in phrases)


# ----------------------------------------------------------------------
# Rule pipeline
# ----------------------------------------------------------------------


def classify(
    text: str,
) -> tuple[str, dict[str, Any]]:
    """
    Deterministically classify one request.

    Returns (route, features). route is KNOWLEDGE, CHAT or UNKNOWN;
    features documents which rule fired for the audit log.

    Order matters: knowledge-side rules run first because a request
    that mixes personal data with chatty wording ("hey so what time is
    my meeting") must reach the knowledge engine.
    """
    raw = (text or "").strip()
    normalized = _normalize(raw)
    lowered = _strip_address(normalized)

    features: dict[str, Any] = {"route": UNKNOWN, "rule": None}

    # No signal at all: the caller (brain/ws_server) decides what
    # empty input means; the router abstains.
    if not normalized:
        return UNKNOWN, features

    lowered = lowered.rstrip(" !.?").strip()

    # "hey nix" / "ok so" with nothing after: pure address -> chat.
    if not lowered:
        features.update(route=CHAT, rule="address_only")
        return CHAT, features

    # --------------------------------------------------------------
    # -1. Injection guards (highest precedence): exfiltration framing
    # and instruction-override attempts go to chat with a refusal
    # seed, never into the knowledge engine.
    # --------------------------------------------------------------
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(lowered)
        if match:
            features.update(route=CHAT, rule="injection_guard", matched=match.group(0))
            return CHAT, features

    # --------------------------------------------------------------
    # -0.5 Past-habit / negative-cognition / fragment guards:
    # statements that LOOK like durable knowledge but must not become
    # active records. CHAT is the safe destination: the chat model
    # answers conversationally, nothing is stored.
    # --------------------------------------------------------------
    if _PAST_HABIT_RE.search(lowered):
        features.update(route=CHAT, rule="past_habit_guard")
        return CHAT, features

    if _NEGATIVE_COGNITION_RE.search(lowered):
        if _NEGATIVE_EXEMPTION_RE.search(lowered):
            # "i can never remember WHEN trash day IS" / "i don't
            # remember MY wifi password" are implicit recall requests:
            # the knowledge engine's recall rules take them from here.
            features.update(route=KNOWLEDGE, rule="negative_recall")
            return KNOWLEDGE, features
        features.update(route=CHAT, rule="negative_cognition_guard")
        return CHAT, features

    if _FRAGMENT_RE.match(lowered):
        features.update(route=CHAT, rule="fragment_guard")
        return CHAT, features

    # --------------------------------------------------------------
    # 0. Pure greetings / openers (before address-stripping: "hey
    # there" must not lose its "hey" to the stripper)
    # --------------------------------------------------------------
    if _GREETING_RE.match(normalized) or normalized.rstrip(" !.,?") in _CHAT_OPENERS:
        features.update(route=CHAT, rule="greeting")
        return CHAT, features

    # "hi nix!" / "hey nix???" / "ok so, hi nix??" - greetings aimed
    # at the assistant, optionally wrapped in filler. Only greeting
    # words, an optional addressee tail, and punctuation may follow.
    if re.match(
        r"^(?:(?:um|uh|so|ok|okay|alright|well|hey|hi|hello|yo)\b"
        r"[ ,!]*)*(?:hi|hello|hey|yo|sup|howdy|good\s*"
        r"(?:morning|afternoon|evening|night)|morning|evening)\b"
        r"(?:\s+(?:there|nix|everyone|all))?[,!.?\s]*$",
        normalized,
    ):
        features.update(route=CHAT, rule="greeting_address")
        return CHAT, features

    # --------------------------------------------------------------
    # 0.5 Figurative statements: "my sleep schedule is a disaster".
    # The word "schedule" inside hyperbole is not scheduling intent,
    # so this guard must run BEFORE the scheduling rules.
    # --------------------------------------------------------------
    if re.search(
        r"\b(?:is\s+(?:a\s+)?(?:disaster|mess|crime\s+scene|"
        r"nightmare|joke)|is\s+basically\b|like\s+my\b.{0,30}?"
        r"\bwas\s+ever\b)",
        lowered,
    ):
        features.update(route=CHAT, rule="figurative_statement")
        return CHAT, features

    # Advice requests are world knowledge: "how to fix my sleep
    # schedule", "tips to sleep better", "best way to learn guitar".
    # Must precede scheduling ("fix my ...", "change my ...").
    if re.match(
        r"^(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)|tips\s+to|"
        r"ways\s+to|best\s+way\s+to)\b",
        lowered,
    ):
        features.update(route=CHAT, rule="advice_question")
        return CHAT, features

    # Hypotheticals explore, they never create records: "what if i
    # scheduled everything on the same day", "hypothetically, if my
    # dentist was on saturday", "suppose i moved to japan", "if i move
    # my dentist to thursday does friday stay free".
    if re.match(
        r"^(?:what\s+if\b|hypothetically[,:]?\s+if\b|suppose\b)", lowered
    ) or re.search(r"\bhypothetically\b", lowered) or re.match(
        r"^if\s+i\s+(?:move|moved|cancel|canceled|cancelled|reschedule|"
        r"rescheduled|postpone|postponed|push|pushed)\b"
        r".*\b(?:would|will|does|do|is|are|stays?|free)\b",
        lowered,
    ):
        features.update(route=CHAT, rule="hypothetical")
        return CHAT, features

    # --------------------------------------------------------------
    # 1. Scheduling intent (strongest knowledge signal)
    # --------------------------------------------------------------
    _SCHEDULING_RE = re.compile(
        r"\b(?:"
        + "|".join(re.escape(phrase) for phrase in _SCHEDULING_VERBS)
        + r")\b",
        re.IGNORECASE,
    )

    # "what about his schedule" / "did you see her calendar": talk
    # about someone else's (or the assistant's) life data is chat.
    third_person = bool(
        re.search(
            r"\b(?:his|her|their|its|your)\s+(?:schedule|calendar|"
            r"meeting|appointments?|classes?|events?|agenda|plans?)\b",
            lowered,
        )
    )

    # "my alarm code is 2009" - a possessive statement is a fact to
    # store, not an alarm request. The personal_statement rule (below)
    # handles it; the bare-alarm rule must not fire first.
    possessive_statement = bool(
        re.search(
            r"\bmy\s+[\w'\-]+(?:\s+[\w'\-]+)?\s+(?:is|are|was|were)\b",
            lowered,
        )
    )

    if _ALARM_RE.search(lowered) and not third_person and not possessive_statement:
        features.update(route=KNOWLEDGE, rule="alarm")
        return KNOWLEDGE, features

    if not third_person and _SCHEDULING_RE.search(lowered):
        features.update(route=KNOWLEDGE, rule="scheduling_verb")
        return KNOWLEDGE, features

    # "remind me ..." / "reminder: ..." / "set a reminder ..." ->
    # knowledge. A bare "reminds me of a song" (chat about music)
    # lacks the pattern. "Remind me of the name of that movie" is
    # world recall, not a reminder to do something - defer it.
    world_recall = bool(
        re.search(
            r"remind\s+me\s+(?:of\s+|about\s+)?"
            r"(?:the\s+)?(?:name|title|song|movie|show|one)\b",
            lowered,
        )
    )
    # "remind me what the capital of Morocco is": no personal marker,
    # so it is world recall. "remind me what my password is" has one
    # and stays a knowledge request.
    if re.search(
        r"remind\s+me\s+(?:what|who|which)\b", lowered
    ) and not _MY_RE.search(lowered):
        world_recall = True
    if (
        _REMINDER_RE.search(lowered)
        and _REMINDER_LEAD_RE.search(lowered)
        and not world_recall
    ):
        features.update(route=KNOWLEDGE, rule="reminder")
        return KNOWLEDGE, features

    # --------------------------------------------------------------
    # 1.5 World recall wearing knowledge words: "do you remember the
    # name of that actor", "can you remind me what the capital of
    # Morocco is". Must precede the store and recall rules. Presence
    # of "my" ("do you remember MY password") keeps it knowledge.
    # --------------------------------------------------------------
    if _WORLD_RECALL_RE.search(lowered) and not _MY_RE.search(lowered):
        features.update(route=CHAT, rule="world_recall")
        return CHAT, features

    if re.search(
        r"remind\s+me\s+(?:what|who|which)\b", lowered
    ) and not _MY_RE.search(lowered):
        features.update(route=CHAT, rule="world_recall")
        return CHAT, features

    # Incidental notes: "just so you know, my spare key is under the
    # mat", "heads up, my brother is visiting on the 12th". The
    # marker phrase plus a personal pronoun is high-precision storage.
    if re.match(
        r"^(?:just\s+so\s+you\s+know|fyi|heads\s*up|"
        r"for\s+future\s+reference|in\s+case\s+you\s+need\s+it|btw)\b",
        lowered,
    ) and re.search(r"\b(?:my|our)\b", lowered):
        features.update(route=KNOWLEDGE, rule="incidental_note")
        return KNOWLEDGE, features

    # Store intent. "I can't remember if penguins fly" is a world
    # question wearing a remember-word, not a store command. "never"
    # covers self-referential complaints ("i never remember dream
    # plots") which are chat, not storage commands.
    negated_store = bool(
        re.search(
            r"\b(?:can'?t|cannot|don'?t|do\s+not|never)\s+"
            r"(?:remember|recall|store|save)\b",
            lowered,
        )
    )
    if not negated_store and (
        _REMEMBER_RE.search(lowered)
        or _matches_any(lowered, _STORE_VERBS)
    ):
        features.update(route=KNOWLEDGE, rule="store_fact")
        return KNOWLEDGE, features

    # Storing personal statements: "my sister's birthday is may 4",
    # "i hate pineapple", "my wifi password is ..." - statements
    # (not questions) containing "my X is/are/was" are durable facts.
    if re.search(
        r"\bmy\s+\w[\w' ]{0,40}\b\s+(?:is|are|was|were|will\s+be)\b",
        lowered,
    ):
        features.update(route=KNOWLEDGE, rule="personal_statement")
        return KNOWLEDGE, features

    # Family facts without "my": "mom's birthday is may 4".
    if _FAMILY_EVENT_RE.search(lowered):
        features.update(route=KNOWLEDGE, rule="family_fact")
        return KNOWLEDGE, features

    # Whereabouts: "i left my keys in the car", "i lost my wallet".
    if _LOST_FOUND_RE.match(lowered):
        features.update(route=KNOWLEDGE, rule="whereabouts")
        return KNOWLEDGE, features

    # Health facts: "i'm allergic to peanuts".
    if _ALLERGIC_RE.match(lowered):
        features.update(route=KNOWLEDGE, rule="health_fact")
        return KNOWLEDGE, features

    # Preference statements: "i hate pineapple on pizza". Addressing
    # the assistant ("i love you") is chat, not a fact to store.
    if _PREFERENCE_RE.match(lowered) and not re.search(
        r"\b(?:you|your|yourself)\b", lowered
    ):
        features.update(route=KNOWLEDGE, rule="preference_statement")
        return KNOWLEDGE, features

    # Event lead-ins: "i have a dentist appointment tomorrow at 3".
    if _I_HAVE_RE.search(lowered) and _matches_any(
        lowered, _EVENT_NOUNS
    ):
        features.update(route=KNOWLEDGE, rule="event_lead_in")
        return KNOWLEDGE, features

    # Temporal openers with an event: "this weekend i have a pizza
    # party", "tomorrow i have a meeting with sarah".
    if _TEMPORAL_OPENER_RE.match(lowered) and _I_HAVE_RE.search(
        lowered
    ):
        features.update(route=KNOWLEDGE, rule="temporal_event")
        return KNOWLEDGE, features

    # Comparison questions are world opinions: "python vs
    # javascript", "which is better pc or console".
    if _COMPARISON_RE.search(lowered):
        features.update(route=CHAT, rule="comparison_question")
        return CHAT, features

    # Sun/almanac questions: "sunrise time tomorrow".
    if _ALMANAC_RE.search(lowered):
        features.update(route=CHAT, rule="almanac_question")
        return CHAT, features

    # --------------------------------------------------------------
    # 2.5 Personal-shaped WORLD questions: they wear "my" or a
    # remember-word, but the answer lives on the internet, not in the
    # user's records.
    # --------------------------------------------------------------
    if re.match(
        r"^what\s+(?:does|did)\s+my\s+name\s+mean\b", lowered
    ):
        features.update(route=CHAT, rule="name_meaning")
        return CHAT, features

    if re.search(
        r"\bmy\s+(?:senator|representative|congressman|congresswoman|"
        r"governor|mayor)\b",
        lowered,
    ):
        features.update(route=CHAT, rule="civic_world_question")
        return CHAT, features

    if re.match(r"^(?:what|which)\s+is\s+my\s+ip\b", lowered):
        features.update(route=CHAT, rule="network_question")
        return CHAT, features

    # "is my laptop supposed to get this hot" / "how often should my
    # car get an oil change" - device how-to, not personal data.
    if _DEVICE_Q_RE.match(lowered):
        features.update(route=CHAT, rule="device_question")
        return CHAT, features

    # --------------------------------------------------------------
    # 3. Recall intent
    # --------------------------------------------------------------
    if _matches_any(lowered, _RECALL_LEADINS):
        features.update(route=KNOWLEDGE, rule="recall_lead")
        return KNOWLEDGE, features

    if _matches_any(lowered, _RECALL_NOUNS):
        features.update(route=KNOWLEDGE, rule="recall_noun")
        return KNOWLEDGE, features

    if _RECALL_RE.search(lowered) and not re.search(
        r"\bmy\s+(?:car|phone|laptop|computer|internet|tv|router|"
        r"resume|essay|generation|salary)'?s?\b",
        lowered,
    ):
        features.update(route=KNOWLEDGE, rule="recall_my")
        return KNOWLEDGE, features

    if re.match(
        r"^(?:show|list|view|open)\s+(?:me\s+)?(?:my|the)\b", lowered
    ):
        features.update(route=KNOWLEDGE, rule="list_my")
        return KNOWLEDGE, features

    # "what's the wifi password" - personal credentials, "the" = mine.
    # "what is the difference between wifi and bluetooth" is world
    # knowledge, not a credential lookup.
    if _PERSONAL_CODE_RE.match(lowered) and "difference between" not in lowered:
        features.update(route=KNOWLEDGE, rule="personal_code_query")
        return KNOWLEDGE, features

    # --------------------------------------------------------------
    # 4. Clear chat signals
    # --------------------------------------------------------------
    if _GREETING_RE.match(lowered):
        features.update(route=CHAT, rule="greeting")
        return CHAT, features

    # Openers are matched against normalized too: the address
    # stripper eats "would you" in "would you rather fight ...".
    if _matches_any(lowered, _CHAT_OPENERS) or _matches_any(
        normalized, _CHAT_OPENERS
    ):
        features.update(route=CHAT, rule="chat_opener")
        return CHAT, features

    if _OPINION_RE.search(lowered):
        features.update(route=CHAT, rule="opinion_of_assistant")
        return CHAT, features

    # Entertainment / world-info requests: "tell me one fun fact about
    # octopuses". A trailing "my ..." means recall instead.
    if _TELL_ME_RE.match(lowered) and not _MY_RE.search(lowered):
        features.update(route=CHAT, rule="tell_me")
        return CHAT, features

    # "can you tell me my dentist's number" - recall wearing a polite
    # request wrapper. The stripper already ate "can you"; what's left
    # asks for a personal value with tell/give verbs.
    if re.match(r"^(?:tell|give|show)\s+me\s+my\b", lowered):
        features.update(route=KNOWLEDGE, rule="tell_me_my")
        return KNOWLEDGE, features

    # World/internet questions.
    if _matches_any(lowered, _WORLD_Q) or _WORLD_RE.search(lowered):
        features.update(route=CHAT, rule="world_question")
        return CHAT, features

    # Anything addressed at the assistant with no knowledge signal is
    # social: "you're funny", "you are the best".
    if _YOU_RE.search(lowered):
        features.update(route=CHAT, rule="addressing_assistant")
        return CHAT, features

    # --------------------------------------------------------------
    # 5. Tie-breakers
    # --------------------------------------------------------------

    # General questions about the world, with no personal marker at
    # all, are the chat model's job.
    if re.match(
        r"^(?:what|who|why|how|where|when|which|does|do|can|could|will|"
        r"would|is|are|has|did)\b",
        lowered,
    ):
        if not _MY_RE.search(lowered) and " i " not in f" {lowered} ":
            features.update(route=CHAT, rule="general_question")
            return CHAT, features

    # Statements about the user ("i like metal", "i have a meeting")
    # and anything containing personal references stay unresolved for
    # the model layer: too many shapes to lex reliably.
    features.update(route=UNKNOWN, rule=None)
    return UNKNOWN, features
