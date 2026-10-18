"""
Lightweight spell-correction preprocessing layer.

Sits between raw user input (often a noisy speech-to-text transcript)
and the deterministic routing rules / AI model selector. Instead of a
hand-maintained typo map -- which can never cover the effectively
infinite typo space -- unknown words are corrected against an English
frequency dictionary using Levenshtein edit distance (pyspellchecker),
with a small domain vocabulary seeded in so calendar/reminder language
is always in-dictionary.

Pipeline position:

    User Input --> [ Preprocessing Spell Checker ] --> [ AI Model ]

Design goals:

* **Conservative.** A word is only rewritten when it is NOT in the
  dictionary and a single high-confidence candidate exists within the
  edit budget. Dictionary words, acronyms, contractions, numbers, and
  short tokens are never touched, so healthy input passes through
  byte-identical.
* **Cheap.** pyspellchecker is pure Python; correction is lazy and the
  checker instance is built once per process and cached. Typical
  latency for an already-clean sentence is microseconds per token.
* **Deterministic.** No network, no model, no randomness: the same
  transcript always yields the same correction.
* **Self-sufficient.** Pure Levenshtein cannot see transpositions
  ("cancle" -> "cancel" costs 2 swaps as substitutions), so a tiny
  seed of exact-token ASR staples that the dictionary handles poorly
  is applied first. This seed is fundamentally different from the old
  typo map: it is a fixed bootstrap of maybe a dozen entries, while
  the open-ended long tail ("apointment", "tomorow", "rememer",
  "scedule", "htat", ...) is covered by the dictionary itself.
"""

from __future__ import annotations

import re
from functools import lru_cache

__all__ = ["correct_spelling"]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# A word is letters with an optional internal apostrophe ("don't").
# Everything else (digits, punctuation, whitespace) is passed through.
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ---------------------------------------------------------------------------
# Vocabulary guards
# ---------------------------------------------------------------------------

# Domain words seeded INTO the dictionary so they are always considered
# correctly spelled (general corpora are thin on assistant/household
# vocabulary, and rare real words would otherwise attract corrections).
_DOMAIN_VOCAB = frozenset(
    {
        # Assistant core vocabulary
        "nix", "fafsa", "tsa",
        "remind", "reminds", "reminded", "reminding", "reminder",
        "reminders", "reschedule", "reschedules", "rescheduled",
        "rescheduling", "postpone", "postpones", "postponed",
        "postponing", "cancel", "cancels", "canceled", "cancelled",
        "canceling", "cancelling",
        # Time words that are occasionally out-of-vocabulary downstream
        "tonight", "noon", "midday", "weekdays",
        "mon", "tues", "wed", "thur", "thurs", "sat", "sun",
        # Household / calendar nouns seen in this deployment
        "appointment", "appointments", "dentist", "dentists",
        "orthodontist", "orthodontists", "pediatrician",
        "pediatricians", "chiropractor", "chiropractors",
        "practice", "practices", "recital", "recitals",
        "robotics", "piano", "playdate", "playdates",
        "afterschool", "homeschool", "homeschooling",
        "playground", "playgrounds", "library", "libraries",
        "girlscouts", "soccer", "gym", "swim",
        "birthday", "birthdays", "anniversary", "anniversaries",
        "medication", "medications", "meds",
        "grandma", "grandmas", "grandpa", "grandpas",
        "nana", "papa", "sibling", "siblings",
        "cousin", "cousins", "aunt", "aunts", "uncle", "uncles",
        # Real words the shipped en dictionary is missing or only holds in
        # US spelling. Measured: "cafe" snapped to "came" and "colour" was
        # rewritten to "color" in healthy D2 questions - the corrector must
        # not edit a word a person actually wrote.
        "cafe", "cafes", "café", "wi-fi", "wifi", "locker", "landlord",
        "passport", "insurance", "renewal", "optician", "hyphen",
        "colour", "colours", "coloured", "favourite", "favourites",
        "neighbour", "neighbours", "neighbourhood", "honour", "honours",
        "behaviour", "behaviours", "centre", "centres", "metre", "litre",
        "realise", "realised", "organise", "organised", "practise",
        "travelling", "cancelling", "labelled", "modelling",
    }
)

# Tokens that must never be rewritten even though a dictionary
# correction may exist: intentional shorthand and abbreviations.
_NEVER_CORRECT = frozenset(
    {
        "tmr", "tmrw", "2mrw", "2moro", "tomoro",
        "gonna", "wanna", "gotta", "dunno", "kinda", "sorta",
        "lemme", "gimme", "brb", "idk", "lol", "sup", "yo",
        "ok", "kk",
        # Casual address/slang that dictionaries do not know but the
        # lead-in stripper and reply tone rely on ("bro" would snap to
        # "pro" at edit distance 1).
        "bro", "bros", "bruh", "dude", "fam", "sis", "man",
        "nah", "yep", "yup", "yeah", "yeh", "nope",
        "haha", "hahaha", "omg", "wow", "woah",
        "pls", "plz", "thx", "tho", "cuz", "cmon",
    }
)

# Exact-token ASR staples that pure Levenshtein ranking handles poorly
# (transpositions look like 2 substitutions, and contraction-less ASR
# output like "wats"/"im" would otherwise snap to a wrong common word
# such as "was"/"him"). Applied before dictionary correction. This list
# is intentionally tiny; everything else goes through the dictionary.
_SHORTHAND = {
    "wats": "what's",
    "im": "i'm",
    "ive": "i've",
    "dont": "don't",
    "cant": "can't",
    "cancle": "cancel",
    "cancled": "cancelled",
    "scheduel": "schedule",
    "schedueled": "scheduled",
    "shcedule": "schedule",
}

# Frequency boost (added to the corpus frequency during candidate
# ranking) for domain-correct candidates, and an automatic pass on the
# minimum-frequency threshold. This biases budget-2 cases such as
# "rember" toward "remember" instead of an unrelated same-distance
# word like "timber".
_DOMAIN_BOOST = {
    # App vocabulary: rare in general corpora but constant in this
    # household's requests, so it needs a prior floor of its own.
    "schedule": 0.010, "scheduled": 0.010, "reschedule": 0.010,
    "rescheduled": 0.010, "postponed": 0.010,
    "reminder": 0.010, "reminders": 0.010, "remind": 0.010,
    "remember": 0.005, "tomorrow": 0.005, "tonight": 0.005,
    "appointment": 0.005, "appointments": 0.005,
    "calendar": 0.010, "calendar's": 0.010,
    "practice": 0.005, "recital": 0.005, "robotics": 0.005,
    "dentist": 0.005, "orthodontist": 0.005, "pediatrician": 0.005,
    "grandma": 0.005, "grandpa": 0.005, "playdate": 0.005,
    "afterschool": 0.005, "homeschool": 0.005,
}

# A correction is only accepted when the candidate is at least this
# frequent in the corpus (or domain-boosted). This is what stops the
# corrector from mangling names and rare words: "kayla" never snaps to
# a rare lookalike, because rare candidates are rejected outright.
_MIN_CANDIDATE_FREQUENCY = 1e-5

# Edit-distance budget. Candidate generation already caps at 2
# (pyspellchecker distance=2, i.e. Peter Norvig's classic scheme), and
# the frequency floor -- not the budget -- is what keeps short tokens
# safe: "htat" may reach "that" (2 swaps) precisely because "that" is
# enormously more frequent than the d=1 lookalikes. Tokens outside the
# length guard below are never corrected at all.
_BUDGET = 2
_MIN_TOKEN_LENGTH = 3
_MAX_TOKEN_LENGTH = 18


#


@lru_cache(maxsize=1)
def _checker():
    """Build the pyspellchecker instance once per process."""
    from spellchecker import SpellChecker

    checker = SpellChecker(language="en", distance=2)
    checker.word_frequency.load_words(sorted(_DOMAIN_VOCAB))
    return checker


def _levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            substitute = previous[j - 1] + (char_a != char_b)
            insert = previous[j] + 1
            delete = current[j - 1] + 1
            current.append(min(substitute, insert, delete))
        previous = current
    return previous[-1]


def _budget_for(token: str) -> int:
    return _BUDGET


def _frequency(checker, word: str) -> float:
    """Corpus frequency with the domain boost folded in.

    pyspellchecker 0.9.0 removed ``WordFrequency.frequency()``; the
    raw ``Counter`` plus ``total_words`` gives the same ratio.
    """
    wf = checker.word_frequency
    total = wf.total_words or 1
    frequency = wf.dictionary.get(word, 0) / total
    return frequency + _DOMAIN_BOOST.get(word, 0.0)


def _correct_token(checker, token: str) -> str | None:
    """
    Return the corrected form of *token*, or None to keep it as-is.
    *token* is the original-cased word (no surrounding punctuation).
    """
    lower = token.lower()

    # Contractions are either fine already or handled by _SHORTHAND.
    if "'" in lower:
        return None

    if lower in _SHORTHAND:
        return _SHORTHAND[lower]

    if (
        len(lower) < _MIN_TOKEN_LENGTH
        or len(lower) > _MAX_TOKEN_LENGTH
        or lower in _DOMAIN_VOCAB
        or lower in _NEVER_CORRECT
    ):
        return None

    # Uppercase tokens are acronyms (TSA, FAFSA); leave them alone.
    if token.isupper():
        return None

    if checker.known([lower]):
        return None

    candidates = checker.candidates(lower)
    if not candidates:
        return None

    budget = _budget_for(lower)
    viable = [
        candidate
        for candidate in candidates
        if candidate != lower
        and _levenshtein(lower, candidate) <= budget
        and (
            _DOMAIN_BOOST.get(candidate, 0.0) > 0.0
            or checker.word_frequency.dictionary.get(candidate, 0)
            / (checker.word_frequency.total_words or 1)
            >= _MIN_CANDIDATE_FREQUENCY
        )
    ]
    if not viable:
        return None

    # Deterministic ranking: frequency first, alphabetical order as the
    # stable tie-break.
    best = max(sorted(viable), key=lambda word: _frequency(checker, word))
    if _frequency(checker, best) <= _frequency(checker, lower):
        return None

    if token.isupper():
        return best.upper()
    if token[0].isupper():
        return best.capitalize()
    return best


def correct_spelling(text: str) -> str:
    """
    Correct obvious spelling mistakes in *text*, conservatively.

    Healthy text passes through unchanged; only dictionary-unknown
    words with a confident in-budget candidate are rewritten. Case is
    preserved; punctuation, digits, and whitespace are untouched.
    """
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        corrected = _correct_token(_checker(), token)
        return token if corrected is None else corrected

    return _TOKEN_RE.sub(_replace, text)
