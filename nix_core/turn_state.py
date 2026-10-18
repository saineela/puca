"""What this turn is doing, predicted before any pipeline is chosen.

The pipeline used to ask one question of every utterance — *which route is
this?* — and a router that must answer for every turn will eventually call
ordinary talk a request. A person does not do that. They predict what the
other person is up to (Friston 2010, predictive processing), weigh how much
it matters right now (the salience network, Seeley et al. 2007), and only
deliberate when the prediction is uncertain or the consequence is real.

This module is that prediction step, and nothing else. It is deterministic,
it never calls a model, it never speaks, and it never writes memory. Its
output is a `TurnExpectation` that the pipeline uses as a *budget*:

* `read` — conversation, question, instruction, correction, answer or
  nothing_expected;
* `about` — the referents the turn is continuing (topic continuity);
* `expect` — what the person will probably do next, so the next turn can be
  scored as prediction error instead of reclassified from scratch;
* `salience` — 0..1, how much deliberation this turn has earned *once its
  intent is known*; never a substitute for intent.

The important property is the inverse one: a turn whose **intent** is
conversation never reaches the request machinery. It does not need a rule
saying "this is not a request". Salience is a separate axis: it budgets how
much reasoning, verification and recovery a turn gets *after* intent has
decided which pipeline it belongs to. A low-salience question can still need
a tool; a calm statement can still need memory.

Neural anchors, all used as control principles rather than decoration:

* predictive processing — the turn opens with a prediction, not a label
  (Friston, Nat. Rev. Neurosci. 11:127, 2010);
* salience network — anterior insula/dACC set the gain on deliberation
  (Seeley et al., J. Neurosci. 27:2349, 2007);
* multiple-demand / global workspace — only the winning content is
  broadcast, so prediction and budget decide what the reply writer sees
  (Duncan 2010; Mashour et al., Neuron 105:776, 2020);
* complementary learning systems — the fast episodic trace is what makes
  continuity possible without writing a semantic fact every turn
  (McClelland et al. 1995; Kumaran et al., TiCS 20:512, 2016);
* event segmentation — a stream of talk is parsed into events, and a topic
  shift is a boundary, not a new request (Zacks et al., Psych. Bull.
  133:273, 2007);
* mentalizing — interpreting a turn means modelling what the speaker thinks
  and expects, which is why `expect` exists at all (Saxe & Kanwisher,
  NeuroImage 19:1835, 2003).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

CONVERSATION = "conversation"
QUESTION = "question"
INSTRUCTION = "instruction"
CORRECTION = "correction"
ANSWER = "answer"
# A statement the pipeline may need to persist ("I live in Austin", "I have a
# meeting tomorrow at 4pm"). It is not an instruction and not conversation:
# it is a distinct intent, so that routing never has to be inferred from the
# salience number. See `conversational` below.
STATEMENT = "statement"
NOTHING_EXPECTED = "nothing_expected"

# A statement is not one thing. What follows from it is decided separately
# from its shape, and never from the effort budget:
#   * ``none``         - an ordinary statement; answer it, store nothing, plan
#                        nothing;
#   * ``memory_write`` - it carries a fact/preference/plan that should persist;
#   * ``clarify``      - it looks like a fact but the value is missing, so the
#                        only correct next step is one question.
STATEMENT_NONE = "none"
STATEMENT_MEMORY_WRITE = "memory_write"
STATEMENT_CLARIFY = "clarify"
STATEMENT_ACTIONS = (STATEMENT_NONE, STATEMENT_MEMORY_WRITE, STATEMENT_CLARIFY)

READS = (
    CONVERSATION,
    QUESTION,
    INSTRUCTION,
    CORRECTION,
    ANSWER,
    STATEMENT,
    NOTHING_EXPECTED,
)

# Words that carry no topic on their own. Kept small on purpose: this is a
# continuity cue, not a linguistic parser.
_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being but by can could did do does doing for from get got had has
    have he her here him his how i if in into is it its just like me more my
    no not now of off on one only or other our out over own said same she
    should so some that the their them then there these they this those to too
    up us very was we were what when where which while who why will with would
    you your yours im ima dont doesnt didnt cant wont yeah yes ok okay hey hi
    hello thanks thank please really thing things stuff going go went make
    made know think want need tell say said ask asked
    """.split()
)

_GREETING_RE = re.compile(
    r"^(?:hey|hi|hello|yo|good\s+(?:morning|afternoon|evening)|"
    r"how\s+are\s+you|how's\s+it\s+going|what's\s+up)\b",
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(
    r"\b(?:thanks|thank\s+you|appreciate\s+it|good\s+night|see\s+you|"
    r"love\s+you|miss\s+you|proud\s+of\s+you|that's\s+(?:funny|sweet|nice)|"
    r"lol|haha|aww|damn|wow|oh\s+no|i'm\s+sorry)\b",
    re.IGNORECASE,
)
_SELF_DISCLOSURE_RE = re.compile(
    r"^(?:i|i'm|im|we|we're)\s+(?:had|have|just|been|feel|felt|got|am|was|"
    r"think|guess|wish|hope|remember)\b|\b(?:today|yesterday|this\s+morning|"
    r"last\s+night)\b",
    re.IGNORECASE,
)
_EMOTION_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+(?:tired|exhausted|overwhelmed|sad|happy|angry|"
    r"upset|stressed|worried|anxious|lonely|scared)|i\s+feel\s+|"
    r"feeling\s+(?:down|low|great|awful|okay|better|worse))\b",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"^(?:no|nope|nah|not\s+that|that's\s+not|thats\s+not|wrong|actually|"
    r"i\s+meant|i\s+mean|sorry,?\s+i\s+meant|wait,?\s+i\s+meant)\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^(?:what|who|whom|whose|when|where|why|how|which|is|are|was|were|"
    r"do|does|did|can|could|would|should|will|have|has|am)\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"^(?:please\s+)?(?:turn|switch|set|make|play|pause|resume|stop|start|"
    r"open|close|lock|unlock|dim|brighten|mute|unmute|remind|schedule|"
    r"cancel|delete|remove|add|create|move|send|call|text|find|search|"
    r"show|tell|give|put|bring|change|rename|remember|save|note|book|"
    r"wake|enable|disable|adjust|increase|decrease|lower|raise|raise)\b",
    re.IGNORECASE,
)
_POLITE_REQUEST_RE = re.compile(
    r"^(?:can|could|would|will)\s+you\b|^would\s+you\s+mind\b|"
    r"\bplease\b",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(
    r"^(?:and|also|what\s+about|how\s+about|same\s+for|then|plus|"
    r"one\s+more|another)\b",
    re.IGNORECASE,
)
_DESTRUCTIVE_RE = re.compile(
    r"\b(?:delete|erase|wipe|remove|forget|clear|cancel|reset|"
    r"all\s+(?:my|the)|everything)\b",
    re.IGNORECASE,
)
_MEMORY_WRITE_RE = re.compile(
    r"\b(?:remember|save|store|keep\s+track|make\s+(?:a\s+)?note)\b|"
    r"\bmy\s+name\s+is\b|"
    r"\b(?:my|our)\s+[\w'’]+(?:'s|’s)?\s+"
    r"(?:name|birthday|age|number|code|address|pin)\s+is\b|"
    r"\bmy\s+\w+\s+is\s+called\b|"
    r"\bmy\s+\w+\s+is\b|"
    r"\bi\s+(?:live|work|like|love|hate|prefer)\b",
    re.IGNORECASE,
)
_REFERENT_RE = re.compile(r"\b(?:it|that|this|they|them|she|he|her|him|those|these)\b", re.IGNORECASE)
# A declarative opener: the shape of a statement, independent of what it is
# about and independent of how much effort it deserves.
_STATEMENT_SHAPE_RE = re.compile(r"^(?:i|i'm|im|we|we're|my|our)\b", re.IGNORECASE)
# A fact-shaped sentence with the value missing ("my sister's name is", "my
# dog is"), or a save instruction with nothing to save. The only correct move
# is one question, so this is its own outcome rather than a failed write.
_MISSING_VALUE_RE = re.compile(
    r"^(?:please\s+)?(?:remember|save|store|note)\s*(?:that)?\s*[.?!]*$|"
    r"^(?:my|our)\s+[\w'’ ]{1,40}?(?:'s|’s)?\s+"
    r"(?:name|birthday|age|number|code|address|pin|state)\s+is\s*[.?!]*$|"
    r"^(?:my|our)\s+\w+\s+is\s*[.?!]*$",
    re.IGNORECASE,
)


def clarification_question(text: str) -> str | None:
    """The one question a value-less fact statement has earned."""
    clean = " ".join((text or "").split()).strip(" .!?")
    if not _MISSING_VALUE_RE.match(clean):
        return None
    lowered = clean.casefold()
    if re.match(r"^(?:please\s+)?(?:remember|save|store|note)", lowered):
        return "What should I remember?"
    slot = re.search(
        r"\b(name|birthday|age|number|code|address|pin|state)\s+is\s*$", lowered
    )
    slot_name = slot.group(1) if slot else "value"
    owner = re.match(r"^(?:my|our)\s+([\w'’ ]{1,40}?)(?:'s|’s)?\s+", lowered)
    who = (owner.group(1).strip() if owner else "").strip()
    if who and slot_name == "name":
        return f"What's {who}'s name?"
    if who:
        return f"What's the {slot_name} for your {who}?"
    return f"What's the {slot_name}?"
# A statement about a future event is conversation in tone and a calendar
# write in consequence. It must not be answered cheaply: "alright bro, I have
# a TSA meeting tmr from 4pm" is a memory event even without "remind me".
_EVENT_STATEMENT_RE = re.compile(
    r"\b(?:meeting|appointment|call|flight|dinner|lunch|breakfast|class|"
    r"doctor|dentist|interview|birthday|party|trip|concert|game|exam)\b",
    re.IGNORECASE,
)
_TIME_EXPRESSION_RE = re.compile(
    r"\b(?:tomorrow|tmr|tmrw|today|tonight|this\s+(?:morning|afternoon|"
    r"evening|weekend)|next\s+(?:week|month|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    re.IGNORECASE,
)


def topic_terms(text: str) -> list[str]:
    """Content words in order of appearance, lower-cased and de-duplicated.

    This is the address used for continuity: "and my father?" carries almost
    no words of its own, so the referent has to come from the recent topic.
    """
    seen: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or ""):
        word = raw.casefold().strip("'-")
        if len(word) < 3 or word in _STOPWORDS:
            continue
        if word not in seen:
            seen.append(word)
    return seen


def continuation_topics(
    text: str,
    history: Iterable[dict[str, Any]] | None = None,
    *,
    window: int = 4,
) -> list[str]:
    """The topic of a short continuation, read from what was just said.

    A turn that is only "and my father?" or "what about tomorrow?" has no
    topic of its own. Instead of treating it as a fresh request, the topic is
    inherited from the last few user turns, which is what a listener does.
    """
    own = topic_terms(text)
    if own and not _is_short_continuation(text):
        return own[:6]
    inherited: list[str] = []
    turns = [turn for turn in (history or []) if isinstance(turn, dict)]
    for turn in reversed(turns[-window * 2 :]):
        if turn.get("role") != "user":
            continue
        for term in topic_terms(str(turn.get("content") or "")):
            if term not in inherited:
                inherited.append(term)
        if len(inherited) >= 6:
            break
    return (own + [term for term in inherited if term not in own])[:6]


def _is_short_continuation(text: str) -> bool:
    clean = " ".join((text or "").split())
    if not clean:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", clean)
    return bool(_CONTINUATION_RE.match(clean)) or (
        len(words) <= 3 and bool(_REFERENT_RE.search(clean))
    )


@dataclass(frozen=True)
class TurnExpectation:
    """A prediction about the turn, made before the turn is handled."""

    read: str = CONVERSATION
    about: tuple[str, ...] = ()
    expect: str = "nothing"
    salience: float = 0.2
    confidence: float = 0.5
    reasons: tuple[str, ...] = ()
    continues_topic: bool = False
    #: Only meaningful when ``read`` is ``statement``: none | memory_write |
    #: clarify. A separate decision, not a budget.
    statement_action: str = STATEMENT_NONE

    @property
    def conversational(self) -> bool:
        """True when the turn needs no request machinery at all.

        This is deliberately independent of salience. Intent decides whether a
        turn needs retrieval or planning; salience only decides how much
        reasoning, verification and recovery it gets once that decision is
        made. A weather question can need a tool while scoring low on
        salience, and a calm statement of a fact can need memory with a
        moderate score. Using the budget as a router conflated the two, which
        is the one thing the review asked us not to do.

        A statement whose action is ``none`` is ordinary talk wearing
        declarative clothing: it is answered, not retrieved for and not
        planned over. A statement that writes memory or needs clarification is
        not conversation, whatever its effort score.
        """
        if self.read == CONVERSATION:
            return True
        return self.read == STATEMENT and self.statement_action == STATEMENT_NONE

    @property
    def wants_deliberation(self) -> bool:
        """Intent classes that always deserve deliberation, regardless of mood.

        A statement only counts when it asks the pipeline to do something
        (write memory or resolve a missing value). An ordinary statement is
        answered, not deliberated over.
        """
        return self.read in (INSTRUCTION, CORRECTION, ANSWER) or (
            self.read == STATEMENT and self.statement_action != STATEMENT_NONE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "about": list(self.about),
            "expect": self.expect,
            "salience": round(self.salience, 3),
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
            "conversational": self.conversational,
            "continues_topic": self.continues_topic,
            "statement_action": self.statement_action,
        }


def _base(read: str) -> tuple[float, float]:
    return {
        CONVERSATION: (0.2, 0.6),
        QUESTION: (0.38, 0.6),
        INSTRUCTION: (0.65, 0.75),
        CORRECTION: (0.55, 0.8),
        ANSWER: (0.6, 0.85),
        STATEMENT: (0.6, 0.7),
        NOTHING_EXPECTED: (0.05, 0.7),
    }[read]


def predict_turn(
    text: str,
    *,
    history: Iterable[dict[str, Any]] | None = None,
    pending_clarification: dict[str, Any] | None = None,
) -> TurnExpectation:
    """Predict what the person is doing with this turn.

    The order of the checks is the order of certainty: an answer to a question
    we just asked is not a new request; a correction is about the previous
    turn; social talk and self-disclosure are conversation even when they
    contain a question mark. Only after those does the turn get to be an
    instruction.
    """
    clean = " ".join((text or "").split())
    reasons: list[str] = []
    turns = list(history or [])
    about = tuple(continuation_topics(clean, turns))
    recent_turns = [turn for turn in turns if isinstance(turn, dict)]
    previous_terms = next(
        (
            topic_terms(str(turn.get("content") or ""))
            for turn in reversed(recent_turns)
            if turn.get("role") == "user"
        ),
        [],
    )
    continues_topic = bool(set(about) & set(previous_terms)) or _is_short_continuation(clean)
    if previous_terms and not continues_topic:
        # A boundary between two conversational events. Do not carry the old
        # topic forward, and do not treat the shift as an error.
        about = tuple(topic_terms(clean))[:6]
        reasons.append("topic_shift")
    if not clean:
        return TurnExpectation(
            read=NOTHING_EXPECTED,
            salience=0.0,
            confidence=0.9,
            reasons=("empty",),
        )

    if pending_clarification is not None:
        return TurnExpectation(
            read=ANSWER,
            about=about,
            expect="resume the pending request",
            salience=0.75,
            confidence=0.85,
            reasons=tuple(("pending_clarification", *reasons)),
            continues_topic=True,
        )

    if _CORRECTION_RE.match(clean):
        return TurnExpectation(
            read=CORRECTION,
            about=about,
            expect="the corrected answer",
            salience=0.62,
            confidence=0.8,
            reasons=tuple(("correction_marker", *reasons)),
            continues_topic=True,
        )

    read = CONVERSATION
    salience, confidence = _base(read)

    if _GREETING_RE.match(clean) or (len(clean) <= 24 and _SOCIAL_RE.search(clean)):
        reasons.append("social_exchange")
        salience = 0.08
        confidence = 0.85
    elif _EMOTION_RE.search(clean) and not _INSTRUCTION_RE.match(clean):
        reasons.append("emotional_disclosure")
        salience = max(salience, 0.3)
        confidence = 0.7
    elif _SELF_DISCLOSURE_RE.search(clean):
        reasons.append("self_disclosure")
        salience = 0.18
    elif _is_short_continuation(clean):
        reasons.append("continuation")
        salience = 0.3

    memory_write = bool(_MEMORY_WRITE_RE.search(clean))
    question = clean.endswith("?") or bool(_QUESTION_START_RE.match(clean))
    instruction = bool(_INSTRUCTION_RE.match(clean)) or bool(_POLITE_REQUEST_RE.match(clean))

    # Continuations and social turns never become instructions on their own
    # wording; "and play some music" is still a continuation, but the verb
    # makes it deliberate.
    statement_action = STATEMENT_NONE
    if instruction and not _is_pure_social(clean):
        read = INSTRUCTION
        salience, confidence = _base(read)
        reasons.append("instruction_shape")
    elif _MISSING_VALUE_RE.match(clean):
        read = STATEMENT
        statement_action = STATEMENT_CLARIFY
        salience, confidence = 0.5, 0.75
        reasons.append("missing_value")
    elif memory_write:
        read = STATEMENT if not _is_imperative_memory(clean) else INSTRUCTION
        salience, confidence = _base(read)
        if read == STATEMENT:
            statement_action = STATEMENT_MEMORY_WRITE
        reasons.append("memory_statement")
    elif _EVENT_STATEMENT_RE.search(clean) and _TIME_EXPRESSION_RE.search(clean):
        read = STATEMENT
        statement_action = STATEMENT_MEMORY_WRITE
        salience, confidence = _base(read)
        reasons.append("schedulable_statement")
    elif question:
        read = QUESTION
        salience, confidence = _base(read)
        reasons.append("question_shape")
    elif _STATEMENT_SHAPE_RE.match(clean) and not _is_short_continuation(clean):
        # An ordinary statement: declarative in form, no action implied. It is
        # answered as talk; it does not trigger retrieval, storage or planning,
        # and the effort budget stays low because there is nothing to spend it
        # on.
        read = STATEMENT
        statement_action = STATEMENT_NONE
        salience, confidence = 0.2, 0.6
        reasons.append("ordinary_statement")

    if _DESTRUCTIVE_RE.search(clean) and read in (INSTRUCTION, QUESTION):
        salience = max(salience, 0.9)
        reasons.append("consequential_scope")
    if _REFERENT_RE.search(clean) and len(topic_terms(clean)) <= 2:
        salience = min(1.0, salience + 0.05)
        reasons.append("unresolved_referent")
    if _CONTINUATION_RE.match(clean) and read != CONVERSATION:
        reasons.append("continues_previous")

    # Prediction error carries the plan forward: if the previous turn ended
    # in a question of ours, the person answering it is a continuation, not a
    # new command.
    expect = {
        CONVERSATION: "another thought, or nothing",
        QUESTION: "an answer or a follow-up",
        INSTRUCTION: "an acknowledgement or a correction",
        CORRECTION: "the corrected answer",
        STATEMENT: "an acknowledgement or a follow-up",
    }.get(read, "nothing")

    return TurnExpectation(
        read=read,
        about=about,
        expect=expect,
        salience=round(min(1.0, max(0.0, salience)), 3),
        confidence=round(confidence, 3),
        reasons=tuple(dict.fromkeys(reasons)),
        continues_topic=continues_topic,
        statement_action=statement_action,
    )


def _is_pure_social(text: str) -> bool:
    clean = " ".join((text or "").split())
    if len(clean) > 40:
        return False
    return bool(_GREETING_RE.match(clean) or _SOCIAL_RE.search(clean))


def _is_imperative_memory(text: str) -> bool:
    """Distinguish "remember X" (an instruction) from "I remember X" (talk)."""
    return bool(re.match(r"^(?:please\s+)?(?:remember|save|note|store)\b", text or "", re.IGNORECASE))


def prediction_error(previous: TurnExpectation | None, current: TurnExpectation) -> float:
    """How wrong the last prediction was, 0..1.

    A correction is the loudest error signal the system gets, and it should
    change retrieval and selection on the *next* turn rather than only
    producing an apology. This is the scalar the pipeline records for that
    purpose; nothing here rewrites memory or the model's prompt.
    """
    if previous is None:
        return 0.0
    if current.read == CORRECTION:
        return 0.9
    if previous.expect.startswith("an answer") and current.read == QUESTION:
        return 0.35
    if previous.read == current.read:
        return 0.05
    if previous.read == QUESTION and current.read == INSTRUCTION:
        return 0.4
    return 0.25
