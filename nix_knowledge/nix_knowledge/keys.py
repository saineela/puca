"""
Key Finding Algorithm: deterministic micro-fact extraction.

Every user utterance is scanned for small, durable pieces of personal
information - Keys - BEFORE the request is routed:

    "my sister, named Maanvi is very naughty"
        -> key 1: user has a sister
        -> key 2: user's sister is named Maanvi
        -> key 3: sister Maanvi is naughty

Attribute families (all deterministic, no model calls):

    identity     "my name is Alex"              (supersedes)
    birthday     "my birthday is June 3"        (+auto reminder action)
    favorites    "my favorite band is X"        (supersedes per favorite)
    allergies    "i'm allergic to peanuts"      (sensitive)
    health       "i have asthma"                (sensitive)
    relations    "my sister works at NASA",
                 "my brother lives in Austin",
                 "my son is 5 years old"
    work/school  "i work as an electrician",
                 "i'm a student at Lincoln High"
    location     "i live in Dallas"             (supersedes)
    age          "i'm 16 years old"             (supersedes)
    goals        "i want to learn Spanish"
    routines     "i wake up at 6am"             (supersedes per routine)
    devices      "my phone is an iPhone 15"     (supersedes per device)
    diet         "i'm vegetarian"
    sizes        "i wear size 10 shoes"
    contact      "my email is ..."              (sensitive)
    prefs        "use metric", "speak spanish"  (supersedes)

Keys are stored in the knowledge base as knowledge_type="key", one
record per key, with context (source sentence, extraction pattern).

Storage is deduplicating: an identical key already on file is not
stored twice. Superseding keys (same subject+predicate, new value)
close the old record with status="superseded" so the profile stays
current instead of accumulating contradictions.

Sensitive keys (allergies, health, contact info) carry
sensitive=True in their data for careful handling and easy audit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

# ----------------------------------------------------------------------
# Relation vocabulary
# ----------------------------------------------------------------------

_RELATIONS = (
    "sister",
    "brother",
    "mom",
    "mum",
    "mother",
    "dad",
    "father",
    "wife",
    "husband",
    "son",
    "daughter",
    "grandma",
    "grandmother",
    "grandpa",
    "grandfather",
    "aunt",
    "uncle",
    "cousin",
    "niece",
    "nephew",
    "girlfriend",
    "boyfriend",
    "partner",
    "roommate",
    "boss",
    "coworker",
    "co-worker",
    "friend",
    "dog",
    "cat",
    "pet",
)

_RELATION_ALT = "|".join(_RELATIONS)

# "my sister", "my older sister"
_POSSESSIVE_RELATION = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?\b",
    re.IGNORECASE,
)

# "named Maanvi" / "called Maanvi"
_NAMED = re.compile(
    r"\b(?:named|called)\s+(?P<name>[A-Za-z]+)\b",
)

# "my sister Maanvi" - name directly after the relation (capitalized
# in the original text; the capital is the cue it is a name)
_RELATION_THEN_NAME = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?\s+([A-Z][a-z]+)\b"
)

# "Maanvi's birthday is June 3" / "john's school is lincoln high"
_NAME_POSSESSIVE = re.compile(
    r"\b([A-Z][a-z]+)(?:'s|\bs)\s+([a-z][a-z ]{2,30}?)\s+"
    r"(?:is|are|was|were)\s+(?P<value>.+?)\s*$",
)

# ----------------------------------------------------------------------
# Attribute statements: "X is very naughty"
# ----------------------------------------------------------------------

_ATTRIBUTE = re.compile(
    r"\b(?:is|are|was|were|loves|likes|hates|enjoys|prefers|"
    r"love|like|hate|enjoy|prefer)\s+(?P<attr>.+?)\s*$"
)

_STANCE_VERBS = ("loves", "likes", "hates", "enjoys", "prefers",
                 "dislikes", "love", "like", "hate", "enjoy", "prefer")

# Questions never yield attribute keys ("what is the capital of
# france" is not a statement about anyone).
_QUESTION_START = re.compile(
    r"^(?:what|who|when|where|why|how|which|is|are|do|does|did|"
    r"can|could|will|would|should)\b"
)

# Attributes that are really events / progressive actions, not traits.
_PROGRESSIVE = re.compile(
    r"\b\w+ing\b|\bgoing to\b|\babout to\b|\bsupposed to\b|"
    r"\bcoming\b|\bvisiting\b",
    re.IGNORECASE,
)

# Storage verbs already handled by the normal pipeline - never keys.
_STORAGE_HINT = re.compile(
    r"\b(password|passcode|pin|code|birthday is my|reminder|alarm)\b",
    re.IGNORECASE,
)

_ARTICLES = ("a ", "an ", "the ", "very ", "really ", "so ", "quite ")

# ----------------------------------------------------------------------
# Segment splitting: "So, My name is Sai Neela, people call me sai,
# I am born on February 25 2009, and love programming"
# -> ["so", "my name is sai neela", "people call me sai", ...]
#
# Profile statements chain with commas; each segment is extracted
# independently so an end-anchored value in one clause does not
# swallow the rest of the sentence.
# ----------------------------------------------------------------------

_SEGMENT_SPLIT = re.compile(
    r",\s*(?=i\b|my\b|people\b|they\b|we\b|and\s+i\b|but\s+i\b"
    r"|love\b|like\b|hate\b|enjoy\b|prefer\b|want\b|wish\b"
    r"|\d{1,3}\s+years?\s+old\b|from\b|born\b|living\b|currently\b)"
    r"|\band\s+(?=i\b|my\b|love\b|like\b|hate\b|enjoy\b|prefer\b|"
    r"want\b|wanna\b|wish\b|allergic\b|lactose\b|terrified\b|"
    r"scared\b|afraid\b)"
    r"|\bbut\s+(?=my\s+(?:friends|family|people|teachers|"
    r"teammates|coworkers|classmates)|people\b|they\b|everyone\b)"
    r"|;\s*",
    re.IGNORECASE,
)


def _split_segments(text: str) -> list[str]:
    """Split a compound self-description into extractable segments."""
    parts = [p.strip() for p in _SEGMENT_SPLIT.split(text or "") if p.strip()]
    # "and love programming" segments carry no subject - reattach the
    # pronoun so first-person patterns still match.
    fixed: list[str] = []
    for part in parts:
        # trailing prepositional fragments ("in college") are the tail
        # of the previous clause, not their own statement
        if re.match(r"^(?:in|at|on|for|with|to)\s+[a-z]{2,20}$",
                    part, re.IGNORECASE):
            continue
        if re.match(
            r"^(?:love|like|hate|enjoy|prefer|want|wanna|wish|hope|plan|"
            r"train|trying|try|save)\b",
            part, re.IGNORECASE,
        ):
            fixed.append("i " + part)
        elif re.match(
            r"^and\s+(?:love|like|hate|enjoy|prefer|want|wish|hope|plan|"
            r"train|trying|try|save)\b",
            part, re.IGNORECASE,
        ):
            fixed.append(re.sub(r"^and\s+", "i ", part, flags=re.IGNORECASE))
        elif re.match(r"^from\s+", part, re.IGNORECASE):
            fixed.append("i am " + part)
        elif re.match(r"^born\s+", part, re.IGNORECASE):
            fixed.append("i was " + part)
        elif re.match(r"^(?:living|currently)\b", part, re.IGNORECASE):
            fixed.append("i am " + part)
        elif re.match(r"^\d{1,3}\s+years?\s+old\b", part, re.IGNORECASE):
            fixed.append("i am " + part)
        elif re.match(r"^(?:allergic|lactose|terrified|scared|afraid)\b",
                      part, re.IGNORECASE):
            fixed.append("i am " + part)
        else:
            fixed.append(part)
    return fixed or [text]

# Addressing prefixes stripped before profile patterns: "hey nix,",
# "ok so", "um".
_ADDRESSING = re.compile(
    r"^(?:hey|hi|hello|yo|ok|okay|um|uh|so|well|by the way|btw)[, ]*"
    r"(?:nix)?[,: ]*",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Identity: name (clause-safe: works mid-sentence too, "my name is
# Sai Neela, people call me sai, ...")
# ----------------------------------------------------------------------

_NAME_PATTERNS = (
    re.compile(
        r"^(?:my\s+(?:real\s+)?name\s+is|i\s+am\s+called|i'm\s+called)\s+"
        r"(?P<name>[^\W\d_]+(?:[ '-][^\W\d_]+){0,4})\s*$",
        re.IGNORECASE,
    ),
    # bare copula form: "i'm Kai" / "i am Serena" / "im Alex" -
    # capitalized in the original text, otherwise "i'm tired" would
    # match everything
    re.compile(
        r"^i(?:\s*am|'m|m)?\s+(?P<name>[A-Z][a-z]{1,15})\s*$",
    ),
)

# mid-clause forms: not anchored to the end, applied per segment
_NAME_CLAUSE = re.compile(
    r"\b(?:my\s+(?:real\s+|legal\s+)?name\s+is|i\s+am\s+called|"
    r"i'm\s+called)\s+"
    r"(?P<name>[^\W\d_]+(?:[ '-][^\W\d_]+){0,4}?)(?:[,.]|$)",
    re.IGNORECASE,
)

# nickname forms: "people call me sai" / "call me Al" / "i go by Al"
_NICKNAME_CLAUSE = re.compile(
    r"\b(?:(?:my\s+)?(?:people|they|everyone|folks|friends|family|"
    r"teammates|coworkers|classmates)(?:\s+\w+){0,2}?\s+"
    r"(?:call|calls)\s+me|"
    r"call\s+me|i\s+go\s+by)\s+"
    r"(?P<name>[a-z][a-z' -]{1,20}?)(?:[,.]|$)",
    re.IGNORECASE,
)

# "my dad's name is Raj" / "my dads name is Raj"
_RELATION_NAME_IS = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?(?:'s)?\s+name\s+is\s+"
    r"(?P<name>[A-Za-z]+)\b",
    re.IGNORECASE,
)

# "i am born on february 25 2009" / "i was born february 25" - defined
# after _MONTH_ALT below (see birthday section)

# ----------------------------------------------------------------------
# Birthday
# ----------------------------------------------------------------------

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(_MONTHS)

_BIRTHDAY_PATTERNS = (
    # "my birthday is june 3" / "my birthday is on 3 june 2009"
    re.compile(
        rf"\bmy\s+(?:birthday|bday)\s+is\s+(?:on\s+)?"
        rf"(?:(?P<m1>{_MONTH_ALT})\s+(?P<d1>\d{{1,2}})"
        rf"(?:(?:st|nd|rd|th)?,?\s+(?P<y1>\d{{4}}))?"
        rf"|(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m2>{_MONTH_ALT})"
        rf")\s*$",
        re.IGNORECASE,
    ),
    # "birthday July 14" (no "is") / "my birthday: July 14"
    re.compile(
        rf"\bmy\s+(?:birthday|bday)\s*(?:is|:)?\s*"
        rf"(?P<m1>{_MONTH_ALT})\s+(?P<d1>\d{{1,2}})"
        rf"(?:(?:st|nd|rd|th)?,?\s+(?P<y1>\d{{4}}))?\s*$",
        re.IGNORECASE,
    ),
    # "i was born on june 3"
    re.compile(
        rf"\bi\s+was\s+born\s+on\s+"
        rf"(?P<m1>{_MONTH_ALT})\s+(?P<d1>\d{{1,2}})\s*$",
        re.IGNORECASE,
    ),
    # "my birthday is 6/3"
    re.compile(
        r"\bmy\s+birthday\s+is\s+(?:on\s+)?(?P<m>\d{1,2})/"
        r"(?P<d>\d{1,2})\s*$",
        re.IGNORECASE,
    ),
)

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
    6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
    11: "November", 12: "December",
}

# "i am born on february 25 2009" / "i was born february 25" /
# "born on July 4th, 2009" (ordinal + comma both optional)
_BORN_CLAUSE = re.compile(
    rf"\bi\s+(?:am|was)\s+born\s+(?:on\s+)?"
    rf"(?P<m>{_MONTH_ALT})\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:,?\s+(?P<y>\d{{4}}))?\b",
    re.IGNORECASE,
)

# "i was born on 5/12" - numeric form
_BORN_NUMERIC = re.compile(
    r"\bi\s+(?:am|was)\s+born\s+(?:on\s+)?(?P<m>\d{1,2})/(?P<d>\d{1,2})"
    r"(?:/(?P<y>\d{4}))?\s*$",
    re.IGNORECASE,
)

# "i turn 17 on may 5" / "i turn 16 this august 21" - forward-looking
# birthday
_TURN = re.compile(
    rf"\bi\s+(?:will\s+)?(?:turn|become)\s+\d{{1,3}}\s+"
    rf"(?:on\s+|this\s+)?"
    rf"(?P<m>{_MONTH_ALT})\s+(?P<d>\d{{1,2}})\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Favorites: "my favorite band is radiohead"
# ----------------------------------------------------------------------

_FAVORITE = re.compile(
    r"\bmy\s+(?:(?:all[ -]?time\s+)?(?:favorite|favourite|fav)\s+)"
    r"(?P<thing>[a-z][a-z ]{1,24}?)\s+"
    r"(?:is|are)\s+(?P<value>.+?)\s*$",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Allergies / health (sensitive)
# ----------------------------------------------------------------------

_ALLERGY = re.compile(
    r"\bi(?:'m|\s+am)\s+(?P<sev>(?:severely|highly|extremely)\s+)?"
    r"allergic\s+to\s+(?P<value>[a-z][a-z ,/-]{1,40}?)"
    r"(?=\s*(?:,|and\b|&|\s*$))",
    re.IGNORECASE,
)

_CONDITIONS = (
    "diabetes", "diabetic", "asthma", "epilepsy", "adhd", "autism",
    "anxiety", "depression", "hypertension", "migraines", "migraine",
    "arthritis", "insomnia", "ocd", "dyslexia",
)
_CONDITION_ALT = "|".join(_CONDITIONS)

_CONDITION = re.compile(
    rf"\bi\s+(?:have|had|suffer\s+from)\s+"
    rf"(?:(?:type\s+[12]|stage\s+\d)\s+)?(?P<value>{_CONDITION_ALT})\b",
    re.IGNORECASE,
)

_LACTOSE = re.compile(
    r"\bi(?:'m|\s+am)\s+lactose\s+intolerant\b",
    re.IGNORECASE,
)

# conditions a RELATIVE may have: "my grandma has dementia"
_REL_CONDITIONS = _CONDITIONS + (
    "dementia", "alzheimers", "alzheimer's", "cancer", "parkinsons",
    "parkinson's", "schizophrenia", "cerebral palsy", "down syndrome",
    "copd", "heart failure", "ptsd", "bipolar disorder",
)
_REL_CONDITION_ALT = "|".join(
    sorted(
        (c.replace(" ", r"\s+") for c in _REL_CONDITIONS),
        key=len,
        reverse=True,
    )
)
_HEALTH_REL = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?\s+"
    rf"(?:has|had|suffers?(?:\s+from)?)\s+(?:(?:type\s+[12]|stage\s+"
    rf"\d)\s+)?"
    rf"(?P<value>{_REL_CONDITION_ALT})\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Relation attributes: "my sister works at NASA"
# ----------------------------------------------------------------------

# The optional `(?:\s+\w+)?` between relation and verb lets an
# interpolated name through: "my sister Maanvi lives in Austin". The
# specific verb alternation makes backtracking safe when the slot
# swallows a verb word ("my sister lives in Austin").
_REL_WORKS_AT = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?(?:\s+\w+)?\s+"
    r"(?:(?:works|jobs?|is employed)\s+at|interns?\s+at)\s+"
    r"(?P<value>[A-Za-z][\w .&'-]{1,40}?)[,.]?\s*$",
    re.IGNORECASE,
)

_REL_LIVES_IN = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?(?:\s+\w+)?\s+"
    r"(?:(?:lives?|is living)\s+in|(?:moved|relocated)\s+to)\s+"
    r"(?P<value>[A-Za-z][\w .,'-]{1,40}?)[,.]?\s*$",
    re.IGNORECASE,
)

_REL_STUDIES_AT = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?(?:\s+\w+)?\s+"
    r"(?:(?:studies|goes to school|attends)\s+at\s+|studies\s+)"
    r"(?P<value>[A-Za-z][\w .'-]{1,40}?)[,.]?\s*$",
    re.IGNORECASE,
)

_REL_AGE = re.compile(
    rf"\bmy\s+(?:\w+\s+)?({_RELATION_ALT})s?(?:\s+\w+)?\s+is\s+"
    r"(?P<age>\d{1,3})\s*(?:years?\s+old)?\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Occupation / school
# ----------------------------------------------------------------------

_OCCUPATIONS = (
    "engineer", "teacher", "student", "nurse", "doctor", "developer",
    "programmer", "electrician", "plumber", "driver", "chef", "cook",
    "lawyer", "accountant", "artist", "writer", "designer", "manager",
    "musician", "athlete", "photographer", "scientist", "researcher",
    "freelancer", "contractor", "soldier", "pilot", "firefighter",
    "mechanic", "barista", "cashier", "dentist", "therapist",
    "stay at home mom", "stay at home dad", "retiree", "surgeon",
)
_OCC_ALT = "|".join(re.escape(o) for o in _OCCUPATIONS)

_OCC_IS_A = re.compile(
    rf"\bi(?:'m|\s+am)\s+an?\s+(?P<value>{_OCC_ALT})\b",
    re.IGNORECASE,
)

_OCC_WORK_AS = re.compile(
    r"\bi\s+work\s+as\s+an?\s+(?P<value>[a-z][a-z ]{2,30}?)[,.]?\s*$",
    re.IGNORECASE,
)

_OCC_WORK_AT = re.compile(
    r"\bi\s+work\s+(?:at|for)\s+(?P<value>[A-Za-z][\w .&'-]{1,40}?)"
    # stop before "as an engineer" / trailing companions
    r"(?=\s*(?:,|but\b|\s+as\b|[,.]?\s*$))",
    re.IGNORECASE,
)

_SCHOOL = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+(?:a\s+)?student\s+at|i\s+study(?:ing)?\s+at|"
    r"i\s+go\s+to\s+(?:school|college|university)\s+at)\s+"
    r"(?P<value>[A-Za-z][\w .'-]{2,40}?)[,.]?\s*$",
    re.IGNORECASE,
)

_STUDY_FIELD = re.compile(
    r"\bi\s+(?:study|studying|major(?:ing)?\s+in)\s+"
    r"(?P<value>[a-z][a-z ]{2,30}?)[,.]?\s*(?:at\s+[\w .'-]+)?\s*$",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Location / age
# ----------------------------------------------------------------------

_LOCATION = re.compile(
    r"\bi\s*(?:'m|am|\s+)?\s*"
    r"(?:live\s+in|am\s+from|from|moved\s+to|relocated\s+to)\s+"
    r"(?P<value>[A-Za-z][\w .,'-]{1,40}?)"
    # stop before trailing companions: "live in Fremont with my
    # parents" / "live in Munich now" / "moved to Austin last year"
    r"(?=\s*(?:,|and\b|but\b|with\b|now\b|these\s+days\b|currently\b"
    r"|last\s+(?:year|month|week|summer|fall|spring)"
    r"|[,.]?\s*$))",
    re.IGNORECASE,
)

_AGE = re.compile(
    r"\bi(?:'m|\s+am)\s+(?P<age>\d{1,3})\s*(?:years?\s*old)?\s*$",
    re.IGNORECASE,
)

# reject "i'm 5 minutes late": a number followed by a time unit is not
# an age.
_AGE_UNIT = re.compile(
    r"^\d{1,3}\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|"
    r"weeks?|months?)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Goals
# ----------------------------------------------------------------------

_GOAL = re.compile(
    r"\b(?:i\s+wanna|i\s+want\s+to|i\s+plan\s+to|i\s+hope\s+to|"
    r"i\s+wish\s+to|i\s+wish\s+to\s+pursue|"
    r"i(?:'m|\s+am)\s+(?:going\s+to|planning\s+to|trying\s+to)|"
    r"my\s+goal\s+is\s+to)\s+"
    r"(?P<value>[a-z][a-z0-9 ,'-]{2,60}?)[,.]?\s*$",
    re.IGNORECASE,
)

# "i'm saving up for a new laptop" - the saving verb belongs in the
# rendering, not the value
_GOAL_SAVING = re.compile(
    r"\bi(?:'m|\s+am)\s+saving\s+(?:up\s+)?for\s+"
    r"(?P<value>[a-z][a-z0-9 ,'&-]{2,50}?)[,.]?\s*$",
    re.IGNORECASE,
)

_TRAINING_FOR = re.compile(
    r"\bi(?:'m|\s+am)\s+training\s+(?:for|to)\s+"
    r"(?P<value>[a-z][a-z0-9 ,'-]{1,60}?)[,.]?\s*$",
    re.IGNORECASE,
)

# "i wish to pursue Computer Engineering in college" - the pursue
# verb belongs in the rendering, not dropped into the value.
_GOAL_PURSUE = re.compile(
    r"\bi\s+(?:wish|want|hope|plan|would\s+like)\s+to\s+"
    r"(?:pursue|study|major\s+in|become)\s+"
    r"(?P<value>[a-z][a-z0-9 ,'-]{2,60}?)(?:\s+in\s+college|"
    r"\s+in\s+university|\s+at\s+college|\s+at\s+university)?[,.]?\s*$",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Routines
# ----------------------------------------------------------------------

_ROUTINE_WAKE = re.compile(
    r"\bi\s+(?:wake\s*up|get\s*up)\s+(?:at|around|by)\s+"
    r"(?P<value>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.IGNORECASE,
)

_ROUTINE_BED = re.compile(
    r"\bi\s+(?:go\s+to\s+bed|sleep|go\s+to\s+sleep)\s+"
    r"(?:at|around|by)\s+"
    r"(?P<value>\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.IGNORECASE,
)

_ROUTINE_HABIT = re.compile(
    r"\bi\s+(?P<activity>[a-z][a-z'-]{1,14}(?:\s+[a-z][a-z'-]{1,14}){0,3})\s+"
    r"every\s+(?P<freq>morning|afternoon|evening|night|day|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# fronted frequency: "every saturday i mow the lawn"
_ROUTINE_HABIT_FRONT = re.compile(
    r"^every\s+(?P<freq>morning|afternoon|evening|night|day|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
    r"i\s+(?P<activity>[a-z][a-z'-]{1,14}(?:\s+[a-z][a-z'-]{1,14}){0,3})"
    r"\b",
    re.IGNORECASE,
)

# a weekday frequency ("every saturday") is only trusted when the
# activity starts with a hobby/chore verb - "i see him every monday"
# is not a routine.
_HOBBY_VERBS = (
    "play", "practice", "go", "do", "run", "train", "volunteer",
    "study", "work", "read", "write", "draw", "code", "cook", "clean",
    "walk", "swim", "bike", "hike", "lift", "stretch", "meditate",
    "skate", "mow", "garden", "journal", "bake", "paint",
)

# third-person conjugation for rendering routine activities:
# "go to the gym" -> "goes to the gym"
_ROUTINE_CONJ = {
    "go": "goes", "work": "works", "run": "runs", "jog": "jogs",
    "do": "does", "practice": "practices", "meditate": "meditates",
    "exercise": "exercises", "study": "studies", "read": "reads",
    "play": "plays", "watch": "watches", "clean": "cleans",
    "wash": "washes", "take": "takes", "visit": "visits",
    "attend": "attends", "call": "calls", "mow": "mows",
    "bake": "bakes", "paint": "paints", "draw": "draws",
    "hike": "hikes", "bike": "bikes", "swim": "swims",
    "skate": "skates", "lift": "lifts", "garden": "gardens",
    "journal": "journals", "sew": "sews", "knit": "knits",
    "fish": "fishes", "camp": "camps", "cook": "cooks",
    "write": "writes", "volunteer": "volunteers", "code": "codes",
    "tutor": "tutors", "stretch": "stretches", "train": "trains",
}

_ROUTINE_ACTIVITIES = (
    "go to the gym", "go to gym", "work out", "workout", "exercise",
    "run", "jog", "meditate", "practice piano", "practice guitar",
)

# ----------------------------------------------------------------------
# Devices / possessions
# ----------------------------------------------------------------------

_DEVICE = re.compile(
    r"\bmy\s+(?P<kind>phone|laptop|computer|pc|desktop|tablet|watch|"
    r"car|bike|bicycle|motorcycle|console|headphones|earbuds|camera)\s+"
    r"is\s+(?:an?\s+)?(?P<value>[\w][\w .-]{0,40}?)[,.]?\s*$",
    re.IGNORECASE,
)

_DRIVE = re.compile(
    r"\bi\s+drive\s+(?:an?\s+)?(?P<value>[\w][\w .'-]{1,40}?)[,.]?\s*$",
    re.IGNORECASE,
)

# social handles: "my instagram is sai_neela"
_SOCIAL = re.compile(
    r"\bmy\s+(?P<kind>instagram|insta|twitter|facebook|snap|snapchat|"
    r"tiktok|discord|telegram|linkedin|github|youtube)\s+is\s+"
    r"(?P<value>[\w.@#-]{2,30})\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Diet
# ----------------------------------------------------------------------

_DIET = re.compile(
    r"\bi(?:'m|\s+am)\s+(?:a\s+)?"
    r"(?P<value>vegetarian|vegan|pescatarian|carnivore|halal|kosher|"
    r"carnivore)\b",
    re.IGNORECASE,
)

_AVOID_FOOD = re.compile(
    r"\bi\s+(?:(?:don'?t|do\s+not|never|can'?t|cannot|cant)\s+"
    r"(?:eat|drink|have)|avoid)\s+(?:an?\s+)?"
    r"(?P<value>[a-z][a-z ]{1,30}?)[,.]?\s*$",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Sizes
# ----------------------------------------------------------------------

_WEAR_SIZE = re.compile(
    r"\bi\s+wear\s+(?:a\s+)?size\s+"
    r"(?P<value>[\w.]+(?:\s+and\s+a\s+half)?)"
    r"(?:\s+(?P<kind>shoes?|shirt|pants|jeans|jacket|dress|ring|hat))?\b",
    re.IGNORECASE,
)

_BODY_SIZE = re.compile(
    r"\bmy\s+(?P<kind>shoe|shirt|pants|jeans|jacket|dress|ring|hat)\s+"
    r"size\s+is\s+(?P<value>[\w.-]+(?:\s+and\s+a\s+half)?)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Contact info (sensitive)
# ----------------------------------------------------------------------

_EMAIL = re.compile(
    r"\bmy\s+(?:e-?mail|email)\s+(?:is|address\s+is)\s+"
    r"(?P<value>[\w.+-]+@[\w-]+\.[\w.-]+)\b",
    re.IGNORECASE,
)

_PHONE = re.compile(
    r"\bmy\s+(?:phone|cell|mobile|cellphone)\s+(?:number|no\.?|num)?\s*"
    r"(?:is\s+)?(?P<value>[\d()+\-. ]{7,20}\d)\b",
    re.IGNORECASE,
)

_ADDRESS = re.compile(
    r"\bmy\s+(?:home\s+)?address\s+is\s+(?P<value>[A-Za-z0-9][\w .,#'-]{4,80}?)"
    r"[,.]?\s*$",
    re.IGNORECASE,
)

# fears: "i am terrified of spiders"
_FEAR = re.compile(
    r"\bi(?:'m|\s+am)\s+(?:terrified|scared|afraid|fearful)\s+of\s+"
    r"(?P<value>[a-z][a-z ]{1,30}?)[,.]?\s*$",
    re.IGNORECASE,
)

# medication (sensitive): "i take adderall in the mornings"
_MEDICATION = re.compile(
    r"\bi\s+(?:take|am\s+on|use)\s+"
    r"(?P<value>(?!metric\b|imperial\b)[a-z][a-z0-9 -]{2,30}?)"
    r"(?:\s+(?:in\s+the\s+)?(?:morning|evening|night|afternoon|daily)s?)?"
    r"[,.]?\s*$",
    re.IGNORECASE,
)

_HEIGHT = re.compile(
    r"\bi(?:'m|\s+am)\s+(?P<ft>\d)\s*(?:ft|feet|foot)\s*"
    r"(?P<inch>\d{1,2})?\s*(?:in|inches)?\b",
    re.IGNORECASE,
)

_WEIGHT = re.compile(
    r"\bi\s+weigh\s+(?P<value>\d{2,3}(?:\.\d)?\s*(?:pounds|lbs|kg|kilos))\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Nix preferences: units, language
# ----------------------------------------------------------------------

_UNITS = re.compile(
    r"\b(?:use|switch\s+to|speak\s+in|give\s+me)\s+"
    r"(?:the\s+)?(?P<value>metric|imperial|24[- ]hour|12[- ]hour)"
    r"(?:\s+system)?\b(?:\s+(?:time|units|measurements|format))?",
    re.IGNORECASE,
)

_LANGUAGES = (
    "spanish", "french", "german", "hindi", "telugu", "tamil",
    "chinese", "mandarin", "japanese", "korean", "italian",
    "portuguese", "russian", "arabic", "vietnamese", "urdu",
    "bengali", "polish", "dutch", "turkish",
)
_LANG_ALT = "|".join(_LANGUAGES)

_LANGUAGE = re.compile(
    rf"(?<!\bi\s)\b(?:speak|reply|answer|respond|talk|chat)\s+"
    rf"(?:to\s+me\s+)?"
    rf"(?:in\s+)?(?P<value>{_LANG_ALT})\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------
# Supersession
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Predicate vocabulary (single-value vs multi-value)
#
# SINGLE_VALUE_PREDICATES: one active truth at a time ("where do you
# live?"). A new value SUPERSEDES the old record (closed with
# status='superseded', valid_until stamped, audit-logged) and any
# auto-reminders hanging off the old record are cancelled.
#
# MULTI_VALUE_PREDICATES: many can coexist ("my sister works at NASA"
# and "my brother lives in Austin"). A new value SUPERSEDES the old
# one only when it CONTRADICTS it (same subject, opposite/changed
# value per contradiction rules).
# ----------------------------------------------------------------------

_SINGLE_VALUE_PREDICATES = {
    "name",
    "goes_by",
    "location",
    "age",
    "occupation",
    "employer",
    "school",
    "study_field",
    "email",
    "phone",
    "address",
    "units_pref",
    "language_pref",
    "diet",
    # people have exactly one of these per subject:
    "birthday",
    "is_named",
    "works_at",
    "lives_in",
    "studies_at",
    "height",
    "weight",
}
_SINGLE_VALUE_PREFIXES = (
    "favorite_",
    "device_",
    "routine_",
    "size_",
    "social_",
)

# existence of people/pets: multiple relations coexist, but "my dog
# Rex" stays one truth per (relation, name) pair.
_EXISTENCE_PREDICATES = ("has_", "is_named")

# person attributes: contradictions handled per-key
_MULTI_VALUE_PREDICATES = {
    "works_at",
    "lives_in",
    "studies_at",
    "allergy",
    "health",
    "avoids_food",
}

# inherently many: goals and one-off facts accumulate
_ACCUMULATING_PREDICATES = {
    "goal",
    "goal_pursue",
    "goal_training",
}


def _is_superseding(key: dict[str, Any]) -> bool:
    """True when a new value must replace (not join) the old one."""
    subject = key.get("subject", "")
    predicate = key.get("predicate", "")
    if subject not in ("user", "nix"):
        # person attributes supersede on contradiction only
        return predicate in _MULTI_VALUE_PREDICATES
    if predicate in _SINGLE_VALUE_PREDICATES:
        return True
    return predicate.startswith(_SINGLE_VALUE_PREFIXES)


# ----------------------------------------------------------------------
# Alias normalization: "im", "im 16" and "i'm 16" must be the same key
# surface so dedupe and contradiction checks see them identically.
# ----------------------------------------------------------------------

_NORMALIZE_ALIASES = (
    (re.compile(r"\bi'm\b"), "im"),
    (re.compile(r"\bi am\b"), "im"),
    (re.compile(r"\bi would\b"), "id"),
    (re.compile(r"\bi'll\b"), "ill"),
    (re.compile(r"\bi've\b"), "ive"),
    (re.compile(r"\bdon't\b"), "dont"),
    (re.compile(r"\bdoes not\b"), "doesnt"),
    (re.compile(r"\bdo not\b"), "dont"),
    (re.compile(r"\bcannot\b"), "cant"),
    (re.compile(r"\bcan't\b"), "cant"),
    (re.compile(r"\bfavorite\b"), "favorite"),
    (re.compile(r"\bfavourite\b"), "favorite"),
)

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20",
}


def _normalize_for_compare(text: str) -> str:
    """Canonical surface form for equality checks."""
    out = (text or "").lower().strip()
    for pattern, replacement in _NORMALIZE_ALIASES:
        out = pattern.sub(replacement, out)
    for word, digit in _NUMBER_WORDS.items():
        out = re.sub(rf"\b{word}\b", digit, out)
    out = re.sub(r"[.,;!?]+", "", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


# A likely human name: 1-2 capitalized words, not a sentence fragment.
# Guards against "my favorite subject is history" style values being
# treated as identity keys.
_NAME_TOKEN = re.compile(r"^[a-z][a-z' -]{0,29}$", re.IGNORECASE)
_NAME_STOPWORDS = {
    "the", "a", "an", "my", "me", "nix", "assistant", "robot",
    "friend", "friend's", "going", "doing", "trying", "looking",
}


def _is_name_token(value: str) -> bool:
    words = value.split()
    if not 1 <= len(words) <= 2:
        return False
    for word in words:
        clean = word.strip("'")
        if clean.lower() in _NAME_STOPWORDS:
            return False
        if not _NAME_TOKEN.match(clean):
            return False
    return True


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _title(value: str) -> str:
    """'alex' -> 'Alex' for names."""
    return " ".join(part.capitalize() for part in value.split())


def _mk(
    subject: str,
    predicate: str,
    value: str,
    context: dict[str, Any],
    *,
    confidence: float = 1.0,
    certainty: str = "known",
    sensitive: bool = False,
) -> dict[str, Any]:
    key: dict[str, Any] = {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "context": context,
        "confidence": confidence,
        "certainty": certainty,
    }
    if sensitive:
        key["sensitive"] = True
    return key


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------


def extract_keys(text: str) -> list[dict[str, Any]]:
    """
    Scan one utterance and return Keys: small dicts with
    {subject, predicate, value, context, ...}. Empty list = nothing
    personal found (the common case for chat/world requests).
    """
    text = (text or "").strip()
    # A question is a request for recall, not a user assertion. This
    # guard applies before relation/attribute extraction so forms such
    # as "my sister is coming over?" cannot become durable memory.
    if not text or text.endswith("?") or _STORAGE_HINT.search(text):
        return []
    # explicit store commands store their own fact - no key extraction
    if re.match(
        r"^(?:please\s+)?(?:remember|save|store|note that|keep in mind|"
        r"don'?t forget|write down|fyi|just so you know|heads up)\b",
        lowered if (lowered := text.lower()) else "",
    ):
        return []

    # profile patterns tolerate addressing: "hey nix, my name is alex"
    profile_text = _ADDRESSING.sub("", text).strip()
    # core: trailing punctuation stripped, case preserved (values are
    # captured from this so brands/names keep their capitals)
    core = profile_text.rstrip(".!?,")
    lowered = core.lower()
    context = {"source": text[:200]}

    # ------------------------------------------------------------------
    # Segment extraction: compound self-descriptions chain clauses with
    # commas/"and"; every segment gets full treatment so an end-anchored
    # pattern in one clause ("my name is Sai Neela, ...") cannot swallow
    # the rest. Results merge into one key list.
    # ------------------------------------------------------------------
    segments = _split_segments(core)
    if len(segments) > 1:
        merged: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        shared_context = context
        for segment in segments:
            seg_lower = segment.lower()
            if not seg_lower or seg_lower in ("so", "and", "but", "well"):
                continue
            for key in extract_keys(segment):
                text_repr = key_text(key)
                if text_repr not in seen_texts:
                    seen_texts.add(text_repr)
                    key.setdefault("context", shared_context)
                    merged.append(key)
        if merged:
            return merged

    keys: list[dict[str, Any]] = []

    # ---- 1. existence: "my sister" mentioned at all ----------------
    relations_found: list[str] = []
    for match in _POSSESSIVE_RELATION.finditer(lowered):
        relation = match.group(1)
        if relation not in relations_found:
            relations_found.append(relation)
        keys.append(
            {
                "subject": "user",
                "predicate": "has_" + relation,
                "value": "true",
                "context": dict(context),
            }
        )

    # ---- 2. relation names: "named Maanvi" / "my sister Maanvi" ----
    named_person: str | None = None
    name_match = _NAMED.search(text)
    if name_match:
        word = name_match.group("name")
        named_person = word.capitalize()
    else:
        m2 = _RELATION_THEN_NAME.search(text)
        if m2 and not _QUESTION_START.match(lowered):
            named_person = m2.group(2).capitalize()

    if named_person and relations_found:
        keys.append(
            {
                "subject": f"user's {relations_found[0]}",
                "predicate": "is_named",
                "value": named_person,
                "context": dict(context),
            }
        )

    # ---- 3. identity: user's own name (clause-safe) -----------------
    name_value: str | None = None
    match = _NAME_CLAUSE.search(core)
    if match:
        name_value = _title(match.group("name").strip())
    else:
        for pattern in _NAME_PATTERNS:
            # the bare-copula pattern is case-sensitive (capital = the
            # cue it is a name), so try the original text first
            match = pattern.match(text) or pattern.match(lowered)
            if match:
                name_value = _title(match.group("name").strip())
                break
    if name_value:
        # stop before trailing companions: "my name is Robert, but my
        # friends call me Bob"
        name_value = re.split(
            r",\s*(?:but|and|however)\b|\s(?:but|however)\s",
            name_value,
            maxsplit=1,
        )[0].strip()
        if name_value:
            keys.append(_mk("user", "name", name_value, context))

    nickname_value: str | None = None
    match = _NICKNAME_CLAUSE.search(core)
    if match:
        nickname_value = _title(match.group("name").strip())
    if nickname_value:
        keys.append(_mk("user", "goes_by", nickname_value, context))

    # ---- 4. birthday -------------------------------------------------
    birthday = None  # (month, day) for auto-reminder
    match = _BORN_CLAUSE.search(lowered)
    if match:
        month = _MONTHS[match.group("m").lower()]
        day = int(match.group("d"))
        year = match.group("y")
        value = f"{_MONTH_NAMES[month]} {day}"
        if year:
            value += f", {year}"
        keys.append(_mk("user", "birthday", value, context))
        birthday = (month, day)
    else:
        for pattern in _BIRTHDAY_PATTERNS:
            match = pattern.search(lowered)
            if match:
                groups = match.groupdict()
                month_name = groups.get("m1") or groups.get("m2")
                if month_name:
                    month = _MONTHS[month_name.lower()]
                    day = int(groups.get("d1") or groups.get("d2") or 0)
                else:
                    month = int(groups["m"])
                    day = int(groups["d"])
                year = groups.get("y1")
                if month and day:
                    value = f"{_MONTH_NAMES[month]} {day}"
                    if year:
                        value += f", {year}"
                    keys.append(_mk("user", "birthday", value, context))
                    birthday = (month, day)
                break

    if not birthday:
        match = _TURN.search(lowered)
        if match:
            month = _MONTHS[match.group("m").lower()]
            day = int(match.group("d"))
            keys.append(
                _mk("user", "birthday",
                    f"{_MONTH_NAMES[month]} {day}", context)
            )
            birthday = (month, day)

    # ---- 5. favorites ------------------------------------------------
    match = _FAVORITE.search(core)
    if match:
        thing = match.group("thing").strip().replace(" ", "_")
        keys.append(
            _mk("user", f"favorite_{thing}", match.group("value").strip(),
                context)
        )

    # ---- 6. allergies / health (sensitive) ---------------------------
    for allergy in _ALLERGY.finditer(lowered):
        severity = (allergy.group("sev") or "").strip()
        value = (
            f"{severity}allergic to {allergy.group('value').strip()}"
        ).strip()
        keys.append(
            _mk("user", "allergy", value, context, sensitive=True)
        )

    match = _CONDITION.search(lowered)
    if match:
        keys.append(
            _mk("user", "health", match.group("value").lower(), context,
                sensitive=True)
        )

    if _LACTOSE.search(lowered):
        keys.append(
            _mk("user", "diet", "lactose intolerant", context,
                sensitive=True)
        )

    match = _HEALTH_REL.search(lowered)
    if match:
        keys.append(
            _mk(f"user's {match.group(1).lower()}", "health",
                match.group("value").lower(), context, sensitive=True)
        )

    # ---- 7. relation attributes --------------------------------------
    rel_attr_matched = False
    rel_name_matched = False
    match = _RELATION_NAME_IS.search(core)
    if match:
        keys.append(
            _mk(f"user's {match.group(1).lower()}", "is_named",
                match.group("name"), context)
        )
        rel_name_matched = True
    for pattern, predicate in (
        (_REL_WORKS_AT, "works_at"),
        (_REL_LIVES_IN, "lives_in"),
        (_REL_STUDIES_AT, "studies_at"),
    ):
        match = pattern.search(core)
        if match:
            relation = match.group(1).lower()
            person = named_person or relation
            subject = f"{relation} {person}" if named_person else \
                f"user's {relation}"
            keys.append(
                _mk(subject, predicate, match.group("value").strip(),
                    context)
            )
            rel_attr_matched = True
            break

    match = _REL_AGE.search(lowered)
    if match:
        relation = match.group(1).lower()
        subject = (
            f"{relation} {named_person}" if named_person
            else f"user's {relation}"
        )
        keys.append(
            _mk(subject, "age", f"{match.group('age')} years old", context)
        )
        rel_attr_matched = True

    # ---- 8. occupation / school ---------------------------------------
    school_value: str | None = None
    match = _SCHOOL.search(core)
    if match:
        school_value = match.group("value").strip()
        keys.append(_mk("user", "school", school_value, context))
    else:
        match = _STUDY_FIELD.search(core)
        if match:
            keys.append(
                _mk("user", "study_field", match.group("value").strip(),
                    context)
            )

    for pattern in (_OCC_WORK_AS, _OCC_IS_A):
        match = pattern.search(core)
        if match:
            value = match.group("value").strip()
            # "i'm a student at Lincoln High" is school, not occupation
            if school_value and value.lower() == "student":
                break
            keys.append(
                _mk("user", "occupation", value, context)
            )
            break

    match = _OCC_WORK_AT.search(core)
    if match:
        keys.append(
            _mk("user", "employer", match.group("value").strip(), context)
        )

    # ---- 9. location / age --------------------------------------------
    match = _LOCATION.search(core)
    if match:
        keys.append(
            _mk("user", "location", _title(match.group("value").strip()),
                context)
        )

    match = _AGE.search(lowered)
    if match:
        # a number followed by a time unit is a duration, not an age
        # ("i'm 5 minutes late"); the $ anchor already rejects most of
        # those, this guard catches the residue.
        tail = lowered[match.start("age"):]
        if not _AGE_UNIT.match(tail):
            keys.append(
                _mk("user", "age", f"{match.group('age')} years old",
                    context)
            )

    # ---- 10. goals -----------------------------------------------------
    # the specific "pursue/study/become" form wins over the generic
    # "i wish to ..." so one clause yields exactly one goal key
    pursue_match = _GOAL_PURSUE.search(core)
    if pursue_match:
        keys.append(
            _mk("user", "goal_pursue", pursue_match.group("value").strip(),
                context, confidence=0.8, certainty="assumed")
        )
    else:
        match = _GOAL.search(core)
        if match:
            keys.append(
                _mk("user", "goal", match.group("value").strip(), context,
                    confidence=0.8, certainty="assumed")
            )
        else:
            match = _GOAL_SAVING.search(core)
            if match:
                keys.append(
                    _mk("user", "goal_saving", match.group("value").strip(),
                        context, confidence=0.8, certainty="assumed")
                )

    match = _TRAINING_FOR.search(core)
    if match:
        keys.append(
            _mk("user", "goal_training", match.group("value").strip(),
                context, confidence=0.8, certainty="assumed")
        )

    # ---- 11. routines ---------------------------------------------------
    match = _ROUTINE_WAKE.search(lowered)
    if match:
        keys.append(
            _mk("user", "routine_wake", match.group("value").lower(),
                context)
        )

    match = _ROUTINE_BED.search(lowered)
    if match:
        keys.append(
            _mk("user", "routine_bed", match.group("value").lower(),
                context)
        )

    for habit in _ROUTINE_HABIT.finditer(lowered):
        activity = habit.group("activity").strip().lower()
        freq = habit.group("freq").lower()
        weekday_freq = freq in (
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "day",
        )
        # a weekday frequency is only trusted for hobby/chore verbs -
        # "i see him every monday" is not a routine
        if weekday_freq and not activity.startswith(_HOBBY_VERBS):
            continue
        if activity in _ROUTINE_ACTIVITIES or freq in (
            "morning", "evening", "night", "weekend",
        ) or weekday_freq:
            predicate = "routine_" + activity.replace(" ", "_")[:24]
            keys.append(
                _mk("user", predicate, f"every {freq}", context)
            )
            # the habit consumed this clause; stop attribute fallback
            attr_match = None

    front = _ROUTINE_HABIT_FRONT.match(lowered)
    if front:
        activity = front.group("activity").strip().lower()
        freq = front.group("freq").lower()
        predicate = "routine_" + activity.replace(" ", "_")[:24]
        keys.append(
            _mk("user", predicate, f"every {freq}", context)
        )
        attr_match = None

    # ---- 12. devices -----------------------------------------------------
    match = _DEVICE.search(core)
    if match:
        kind = match.group("kind").lower()
        keys.append(
            _mk("user", f"device_{kind}", match.group("value").strip(),
                context)
        )

    match = _DRIVE.search(core)
    if match:
        keys.append(
            _mk("user", "device_car", match.group("value").strip(), context)
        )

    match = _SOCIAL.search(core)
    if match:
        kind = match.group("kind").lower()
        kind = {
            "insta": "instagram", "snap": "snapchat",
        }.get(kind, kind)
        keys.append(
            _mk("user", f"social_{kind}", match.group("value"), context)
        )

    # ---- 13. diet ----------------------------------------------------------
    match = _DIET.search(lowered)
    if match:
        keys.append(
            _mk("user", "diet", match.group("value").lower(), context)
        )

    match = _AVOID_FOOD.search(core)
    if match:
        keys.append(
            _mk("user", "avoids_food", match.group("value").strip(), context)
        )

    # ---- 14. sizes ----------------------------------------------------------
    match = _WEAR_SIZE.search(lowered)
    if match:
        kind = (match.group("kind") or "clothes").lower().rstrip("s")
        keys.append(
            _mk("user", f"size_{kind}", match.group("value"), context)
        )
    else:
        match = _BODY_SIZE.search(lowered)
        if match:
            keys.append(
                _mk("user",
                    f"size_{match.group('kind').lower()}",
                    match.group("value"),
                    context)
            )

    # ---- 15. contact info (sensitive) ---------------------------------------
    match = _EMAIL.search(core)
    if match:
        keys.append(
            _mk("user", "email", match.group("value").lower(), context,
                sensitive=True)
        )

    match = _PHONE.search(core)
    if match:
        keys.append(
            _mk("user", "phone", match.group("value").strip(), context,
                sensitive=True)
        )

    match = _ADDRESS.search(core)
    if match:
        keys.append(
            _mk("user", "address", match.group("value").strip(), context,
                sensitive=True)
        )

    match = _FEAR.search(lowered)
    if match:
        keys.append(
            _mk("user", "fear", match.group("value").strip(), context)
        )

    match = _MEDICATION.search(lowered)
    if match:
        keys.append(
            _mk("user", "medication", match.group("value").strip(), context,
                sensitive=True)
        )

    match = _BORN_NUMERIC.search(lowered)
    if match and not any(k.get("predicate") == "birthday" for k in keys):
        month, day = int(match.group("m")), int(match.group("d"))
        if 1 <= month <= 12 and 1 <= day <= 31:
            year = match.group("y")
            value = f"{_MONTH_NAMES[month]} {day}"
            if year:
                value += f", {year}"
            keys.append(_mk("user", "birthday", value, context))
            if not birthday:
                birthday = (month, day)

    match = _HEIGHT.search(lowered)
    if match:
        height = match.group("ft") + " foot " + (match.group("inch") or "")
        keys.append(_mk("user", "height", height.replace("  ", " ").strip(),
                        context))

    match = _WEIGHT.search(lowered)
    if match:
        keys.append(
            _mk("user", "weight", match.group("value").strip(), context)
        )

    # ---- 16. nix preferences --------------------------------------------------
    match = _UNITS.search(lowered)
    if match:
        keys.append(
            _mk("nix", "units_pref", match.group("value").lower(), context)
        )

    match = _LANGUAGE.search(lowered)
    if match:
        keys.append(
            _mk("nix", "language_pref", match.group("value").lower(),
                context)
        )

    # ---- 3. attributes: "... is very naughty" / "I love hiking" ----
    # (skipped when a relation attribute already described the same
    # person - "my son is 5" must not also produce "user's son is 5"
    # - and when a birthday consumed the clause, else "i was born on
    # june 3" becomes a "was born" trait)
    attr_match = _ATTRIBUTE.search(core)
    if attr_match and not _QUESTION_START.match(lowered) \
            and not rel_attr_matched and not rel_name_matched \
            and not birthday:
        predicate_verb = next(
            (v for v in ("loves", "likes", "hates", "enjoys", "prefers",
                         "love", "like", "hate", "enjoy", "prefer",
                         "is", "are", "was", "were")
             if re.search(rf"\b{v}\b", attr_match.group(0))),
            "is",
        )
        # negation flips the stance: "i do not like spicy food" is a
        # dislike, never a like. The negation sits BEFORE the matched
        # verb, so scan the whole clause, not just the match.
        if re.search(
            r"\b(?:do(?:\s+not|n'?t)|never|dont)\s+"
            r"(?:like|love|enjoy|prefer|care\s+for)\b",
            lowered,
            re.IGNORECASE,
        ):
            predicate_verb = "dislikes"
        attribute = attr_match.group("attr").strip().rstrip(".,!")
        for article in _ARTICLES:
            if attribute.startswith(article):
                attribute = attribute[len(article):]
                break
        # Progressive actions are events, not traits - but only for
        # is-type statements ("mom is coming to visit"). After stance
        # verbs a gerund IS the preference: "I love hiking", "I enjoy
        # swimming".
        trait_style = predicate_verb in ("is", "are", "was", "were")
        if (
            attribute
            and (not trait_style or not _PROGRESSIVE.search(attribute))
            and attribute not in ("named",)
            and not attribute.startswith("named ")
            and not attribute.startswith("called ")
            and not re.search(r"\bname\s+is\b", attribute, re.IGNORECASE)
        ):
            # attach to the most specific subject available
            if named_person and relations_found:
                subject = f"{relations_found[0]} {named_person}"
            elif relations_found:
                subject = f"user's {relations_found[0]}"
            elif re.match(r"^(?:i|im|i'm|we)\b", lowered):
                # first-person statement: the key is about the user
                subject = "user"
            else:
                subject = None
            # "my sister is Priya" already produced "is_named Priya" -
            # the bare-copula key would duplicate it
            duplicates_named = (
                named_person is not None
                and attribute.strip(".,!").lower() == named_person.lower()
            )
            if subject and not duplicates_named:
                keys.append(
                    {
                        "subject": subject,
                        "predicate": predicate_verb,
                        "value": attribute,
                        "context": dict(context),
                        # opinions may be venting, not durable truth
                        "confidence": 0.7,
                        "certainty": "assumed",
                    }
                )

    # ---- 4. named-person possessive facts: "Maanvi's birthday is
    # June 3". Requires a capitalized name so "my mom's..." is not
    # double-counted (relation rules above already covered it).
    for match in _NAME_POSSESSIVE.finditer(profile_text):
        name, noun, value = (
            match.group(1),
            match.group(2).strip(),
            match.group("value").strip(),
        )
        if name.lower() in ("my", "our", "his", "her", "their", "its"):
            continue
        if not value:
            continue
        keys.append(
            {
                "subject": name,
                "predicate": noun,
                "value": value,
                "context": dict(context),
            }
        )

    # birthday auto-reminder instruction rides on the key
    if birthday:
        for key in keys:
            if key.get("predicate") == "birthday":
                key["remind_birthday"] = {"month": birthday[0],
                                          "day": birthday[1]}

    return keys


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


# verbs that render as "user <verb>s ..." and flip to second person as
# "you <verb> ..." in replies.
_THIRD_PERSON_VERBS = {
    "love": "loves", "like": "likes", "hate": "hates",
    "enjoy": "enjoys", "prefer": "prefers",
    "live": "lives", "work": "works", "wear": "wears",
    "drive": "drives", "study": "studies", "want": "wants",
    "wake": "wakes", "go": "goes", "avoid": "avoids",
}


def key_text(key: dict[str, Any]) -> str:
    """Human sentence for a key: 'user's sister is named Maanvi'."""
    subject = key.get("subject", "")
    predicate = key.get("predicate", "")
    value = key.get("value", "")

    # first-person stance verbs become third person for the "user"
    # subject: "I love hiking" -> "user loves hiking"
    if subject == "user" and predicate in _THIRD_PERSON_VERBS:
        predicate = _THIRD_PERSON_VERBS[predicate]

    if predicate == "is":
        return f"{subject} is {value}"
    if predicate in _STANCE_VERBS or predicate in _THIRD_PERSON_VERBS.values():
        return f"{subject} {predicate} {value}"
    if predicate.startswith("has_"):
        return f"{subject} has a {predicate[4:]}"
    if predicate == "is_named":
        return f"{subject} is named {value}"
    if predicate == "name":
        return f"{subject}'s name is {value}"
    if predicate == "goes_by":
        return f"{subject} goes by {value}"
    if predicate == "birthday":
        return f"{subject}'s birthday is {value}"
    if predicate.startswith("favorite_"):
        thing = predicate[len("favorite_"):].replace("_", " ")
        return f"{subject}'s favorite {thing} is {value}"
    if predicate == "allergy":
        return f"{subject} is {value}"
    if predicate == "health":
        return f"{subject} has {value}"
    if predicate == "age":
        return f"{subject} is {value}"
    if predicate == "occupation":
        article = "an" if value[:1].lower() in "aeiou" else "a"
        return f"{subject} works as {article} {value}"
    if predicate == "employer":
        return f"{subject} works at {value}"
    if predicate == "school":
        return f"{subject} is a student at {value}"
    if predicate == "study_field":
        return f"{subject} studies {value}"
    if predicate == "location":
        return f"{subject} lives in {value}"
    if predicate == "goal":
        return f"{subject} wants to {value}"
    if predicate == "goal_training":
        return f"{subject} is training for {value}"
    if predicate == "goal_saving":
        return f"{subject} is saving up for {value}"
    if predicate == "goal_pursue":
        return f"{subject} wants to pursue {value}"
    if predicate == "diet":
        return f"{subject} is {value}"
    if predicate == "avoids_food":
        return f"{subject} never eats {value}"
    if predicate.startswith("routine_"):
        what = predicate[len("routine_"):].replace("_", " ")
        if what in ("wake", "bed"):
            label = "wakes up" if what == "wake" else "goes to bed"
            return f"{subject} {label} at {value}"
        # conjugate known verbs anywhere in the phrase so compound
        # activities render correctly: "run and do yoga" ->
        # "runs and does yoga"
        words = [
            _ROUTINE_CONJ.get(w, w) for w in what.split()
        ]
        return f"{subject} {' '.join(words)} {value}"
    if predicate.startswith("device_"):
        kind = predicate[len("device_"):]
        if kind == "car":
            return f"{subject} drives a {value}"
        return f"{subject}'s {kind} is a {value}"
    if predicate.startswith("size_"):
        kind = predicate[len("size_"):]
        return f"{subject}'s {kind} size is {value}"
    if predicate == "email":
        return f"{subject}'s email is {value}"
    if predicate == "phone":
        return f"{subject}'s phone number is {value}"
    if predicate == "address":
        return f"{subject}'s address is {value}"
    if predicate == "fear":
        return f"{subject} fears {value}"
    if predicate == "medication":
        return f"{subject} takes {value}"
    if predicate == "height":
        return f"{subject} is {value} tall"
    if predicate == "weight":
        return f"{subject} weighs {value}"
    if predicate.startswith("social_"):
        kind = predicate[len("social_"):]
        return f"{subject}'s {kind} is {value}"
    if predicate == "units_pref":
        return f"nix uses {value} units"
    if predicate == "language_pref":
        return f"nix speaks {value} with the user"
    if predicate.startswith("works_at"):
        return f"{subject} works at {value}"
    if predicate.startswith("lives_in"):
        return f"{subject} lives in {value}"
    if predicate.startswith("studies_at"):
        return f"{subject} studies at {value}"
    return f"{subject}'s {predicate} is {value}"


# ----------------------------------------------------------------------
# Birthday auto-reminder
# ----------------------------------------------------------------------


def _next_birthday(month: int, day: int, timezone) -> datetime | None:
    """Next occurrence of this month/day at 9am local."""
    try:
        now = datetime.now(timezone)
        candidate = datetime(
            now.year, month, day, 9, 0, 0, tzinfo=timezone
        )
        if candidate <= now:
            candidate = datetime(
                now.year + 1, month, day, 9, 0, 0, tzinfo=timezone
            )
        return candidate
    except (ValueError, TypeError):
        return None  # February 30 etc.


def schedule_birthday_reminder(
    actions_engine,
    *,
    key_record_id: int,
    person: str,
    month: int,
    day: int,
    timezone,
) -> int | None:
    """
    Schedule one reminder ~3 days before the next occurrence of a
    stored birthday. Returns the action id, or None when scheduling
    is impossible. Never raises into the reply path.
    """
    if actions_engine is None:
        return None

    moment = _next_birthday(month, day, timezone)
    if moment is None:
        return None

    remind_at = moment - timedelta(days=3)
    if remind_at <= datetime.now(timezone):
        # birthday is within 3 days: remind now-ish instead of never
        remind_at = datetime.now(timezone) + timedelta(minutes=1)

    label = person.strip().capitalize() or "Someone"
    try:
        action = actions_engine.schedule(
            action_type="reminder",
            scheduled_for=remind_at,
            payload={
                "title": f"{label}'s birthday",
                "message": (
                    f"{label}'s birthday is coming up on "
                    f"{_MONTH_NAMES[month]} {day}!"
                ),
            },
            source="key_finder",
            source_record_id=key_record_id,
            knowledge_type="key",
            metadata={"auto": "birthday_reminder"},
        )
        return getattr(action, "id", None)
    except Exception:  # noqa: BLE001 - reminders must never break keys
        return None


# ----------------------------------------------------------------------
# Deduplicating + superseding storage
# ----------------------------------------------------------------------


def _find_contradictions(
    engine,
    key: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Active key records that CONTRADICT the candidate key.

    Contradiction is per predicate:
      - single-value: any same-subject+predicate record with a
        different value ("lives in Dallas" vs "lives in Austin")
      - multi-value person attributes: same person + same predicate +
        different value ("sister Maanvi works at X" vs "...at Y")
    """
    subject = _normalize_for_compare(key.get("subject", ""))
    predicate = key.get("predicate", "")
    value_norm = _normalize_for_compare(str(key.get("value", "")))

    try:
        rows = engine.database.execute(
            """
            SELECT id, data FROM knowledge
            WHERE knowledge_type = 'key'
              AND status = 'active'
            """,
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []

    contradictions: list[dict[str, Any]] = []
    for row in rows:
        record_id, raw = row[0], row[1]
        try:
            data = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:  # noqa: BLE001
            continue

        if data.get("predicate") != predicate:
            continue

        existing_subject = _normalize_for_compare(
            str(data.get("subject", ""))
        )
        existing_value = _normalize_for_compare(
            str(data.get("value", ""))
        )

        if existing_subject != subject:
            continue
        if existing_value == value_norm:
            continue  # same fact, dedupe handles it

        if predicate in _SINGLE_VALUE_PREDICATES or predicate.startswith(
            _SINGLE_VALUE_PREFIXES
        ):
            contradictions.append({**data, "_record_id": record_id})
        elif predicate in _MULTI_VALUE_PREDICATES:
            # person attribute: same person, changed attribute =
            # contradiction (people rarely have two employers)
            contradictions.append({**data, "_record_id": record_id})

    return contradictions


def _close_record(
    engine,
    record_id: int,
    old_data: dict[str, Any],
    *,
    new_key: dict[str, Any],
    actions_engine=None,
    reason: str,
) -> None:
    """
    Close a superseded key record: status + valid_until stamped
    (Zep-style bi-temporal validity) + audit row with before/after
    (Mem0-style operation log) + cancel any auto-reminders spawned by
    the old record.
    """
    from datetime import datetime, timezone

    try:
        engine.database.execute(
            """
            UPDATE knowledge
            SET status = 'superseded',
                valid_until = ?
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), record_id),
        )
        engine.database.execute(
            """
            INSERT INTO knowledge_changes (
                record_id,
                operation,
                knowledge_type,
                before_data,
                after_data,
                source,
                reason,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                "supersede",
                "key",
                json.dumps(old_data, default=str),
                json.dumps(
                    {**old_data, "status": "superseded"}, default=str
                ),
                "key_finder",
                reason,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except Exception:  # noqa: BLE001
        return

    # auto-reminders (birthday) spawned by the old record must go too
    if actions_engine is not None:
        try:
            actions_engine.cancel(
                source_record_id=record_id,
                knowledge_type="key",
                reason=reason,
            )
        except Exception:  # noqa: BLE001
            pass


def store_keys(
    engine,
    keys: list[dict[str, Any]],
    *,
    actions_engine=None,
    timezone=None,
) -> list[dict[str, Any]]:
    """
    Store keys with Mem0-style memory management, fully deterministic:

    - NOOP when an equivalent key is already active (alias/number
      normalized comparison, so "im 16" == "i'm 16")
    - ADD for new facts
    - SUPERSEDE for contradictions: single-value predicates always;
      person attributes when the same person's attribute changed.
      Old record closed with valid_until + audit row + reminder
      cancellation.

    Birthday keys schedule an auto-reminder ~3 days before the date
    (when an actions engine is provided).

    Returns the freshly stored key dicts (empty on NOOP).
    """
    stored: list[dict[str, Any]] = []
    for key in keys:
        text = key_text(key)
        text_norm = _normalize_for_compare(text)

        # ---- NOOP: equivalent active key exists (normalized) -------
        duplicate = False
        try:
            active_rows = engine.database.execute(
                """
                SELECT data FROM knowledge
                WHERE knowledge_type = 'key' AND status = 'active'
                """,
            ).fetchall()
        except Exception:  # noqa: BLE001
            active_rows = []
        for row in active_rows:
            raw = row[0]
            try:
                data_check = (
                    raw if isinstance(raw, dict) else json.loads(raw)
                )
            except Exception:  # noqa: BLE001
                continue
            if _normalize_for_compare(
                str(data_check.get("value", ""))
            ) == text_norm:
                duplicate = True
                break
        if duplicate:
            continue

        # ---- SUPERSEDE: close contradicting records -----------------
        if _is_superseding(key):
            for old in _find_contradictions(engine, key):
                _close_record(
                    engine,
                    old.get("_record_id") or 0,
                    old,
                    new_key=key,
                    actions_engine=actions_engine,
                    reason=(
                        f"superseded by: {text}"
                    ),
                )

        data = {
            "value": text,
            "subject": key.get("subject"),
            "predicate": key.get("predicate"),
            "context": key.get("context") or {},
        }
        if key.get("sensitive"):
            data["sensitive"] = True

        record = engine.create(
            "key",
            data,
            confidence=float(key.get("confidence", 1.0)),
            certainty=key.get("certainty", "known"),
            source="key_finder",
        )
        stored.append(key)

        if key.get("remind_birthday") and actions_engine is not None:
            schedule_birthday_reminder(
                actions_engine,
                key_record_id=getattr(record, "id", None),
                person=str(data.get("subject") or "user"),
                month=int(key["remind_birthday"]["month"]),
                day=int(key["remind_birthday"]["day"]),
                timezone=timezone,
            )

    return stored
