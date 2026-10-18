"""Content words for personal questions, shared by the router and the finder.

Two different failures in the D2 evidence had one cause: the words a person says
were compared literally against the words a fact was stored with.

* the selector had no rule for "Where do I work?" or "What colour is my
  laptop?", so no function was chosen at all;
* the fact finder required *every* word of the question to appear in the stored
  sentence, so "which code opens my garage" could not reach "My garage code is
  8042." and "what's the name of my dog" could not reach "My dog is named
  Biscuit."

The fix is one vocabulary, in one place: strip the words that carry no content
in a personal question (question words, auxiliaries, attributive nouns such as
"colour" or "kind"), and treat a small set of genuine paraphrases as the same
word ("rent" = "landlord", "key" = "password", "good for" = "expire").

Nothing here calls a model. It is a deterministic word list, so a test can pin
each decision, and the router and the finder can never disagree about what a
question is asking for.
"""
from __future__ import annotations

import re
from typing import Iterable

#: Words that ask a question, point at a person, or carry no content.
STOP_WORDS: frozenset[str] = frozenset({
    "what", "whats", "which", "when", "where", "who", "whos", "whose", "why",
    "how", "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "done", "can", "could", "will", "would", "should",
    "shall", "may", "might", "must",
    "the", "a", "an", "my", "our", "your", "their", "his", "her", "its",
    "i", "me", "we", "us", "you", "it", "they", "them", "he", "she",
    "of", "for", "to", "in", "on", "at", "from", "with", "about", "into",
    "and", "or", "but", "that", "this", "these", "those", "there", "here",
    "one", "ones", "same", "other", "another", "both", "each", "every",
    "again", "please", "tell", "say", "said", "know", "remember", "think",
    # a past frame selects WHICH record to read (history), it is not part of
    # the subject, so "what was my garage code before" queries "garage code"
    "before", "previously", "earlier", "originally", "used", "formerly",
    "get", "got", "give", "show", "find", "see", "look", "back", "then",
    "now", "still", "just", "also", "very", "really", "actually",
    "usually", "normally", "currently",
    # A time frame selects WHEN to look, never WHICH record: the calendar
    # rules resolve it and the fact finder must not require it in the stored
    # sentence ("do I have anything tomorrow" is a calendar browse).
    "today", "tomorrow", "tmr", "tmrw", "tonight", "yesterday",
    "week", "weekend", "month", "weekday", "weekdays",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
})

#: Verbs that appear in a question but never in the sentence that stores the
#: answer ("which code OPENS my garage" / "My garage code is 8042.").
#: Only verbs that carry *no* subject matter belong here; "work", "live",
#: "rent" and "avoid" stay because the fact finder matches them, directly or
#: through a synonym group.
QUERY_VERBS: frozenset[str] = frozenset({
    "open", "opens", "opened", "unlock", "unlocks",
    "expire", "expires", "expired", "expiring",
    "renew", "renews", "renewed",
    "leave", "leaves", "leaving", "depart", "departs",
    "doing", "feeling", "feel", "feels", "says", "said",
    "go", "goes", "going", "get", "gets", "giving", "gives",
    "keep", "keeps", "hold", "holds", "long",
    "belong", "belongs", "contain", "contains", "rest", "rests",
    "end", "ends", "taking", "using", "use", "uses", "used",
    "wearing", "wear", "wears",
    "come", "comes", "coming", "fall", "falls", "drink", "drinks",
})

#: Attributive words: they name the *kind* of answer wanted, not the subject.
#: "what colour is my laptop" is a question about the laptop; the colour is
#: what the reply supplies, it is not a word stored beside it. The locatives
#: ("at home") belong here for the same reason: the store holds the Wi-Fi key,
#: not the building.
ATTRIBUTE_WORDS: frozenset[str] = frozenset({
    "colour", "color", "kind", "type", "sort", "make", "brand",
    "time", "date", "day", "days", "year", "hour", "hours",
    "minutes", "minute", "long",
    "home", "house", "flat", "apartment",
})

#: Genuine paraphrases seen in measured evidence. Each group is one meaning:
#: any member in a question matches any member in a stored sentence.
SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"name", "named", "called", "call", "nickname"}),
    frozenset({"rent", "rents", "renting", "landlord", "landlady", "lease"}),
    frozenset({"password", "key", "passcode", "passphrase"}),
    frozenset({"expire", "expires", "expiry", "expiration", "good"}),
    frozenset({"medication", "medicine", "drug", "drugs", "allergic", "allergy", "avoid"}),
    frozenset({"wifi", "wi-fi", "wireless", "network"}),
    frozenset({"coffee", "cafe", "brew"}),
    frozenset({"work", "works", "job", "employer", "company"}),
    frozenset({"live", "lives", "home", "city", "town", "residence"}),
    frozenset({"dog", "dogs", "puppy", "pet"}),
    frozenset({"sister", "sibling"}),
    frozenset({"mother", "mum", "mom", "mamma"}),
    frozenset({"father", "dad", "papa"}),
    frozenset({"insurance", "policy"}),
    frozenset({"renew", "renews", "renewal", "due"}),
    frozenset({"gym", "workout", "exercise"}),
    frozenset({"flight", "plane", "airfare"}),
    frozenset({"dentist", "dental"}),
    frozenset({"bike", "bicycle", "cycle"}),
    frozenset({"laptop", "computer", "notebook"}),
    frozenset({"garage", "parking"}),
    frozenset({"bank", "banking", "banked", "branch"}),
)

#: A stored sentence that frames itself as history ("I previously lived in
#: Boise.") is a past value, not today's. The D4 replay showed the old city
#: returned beside the current one for "What city do I live in?" because the
#: finder had no way to tell the two apart. The frame is matched on the stored
#: text, and only a question that asks for the past reads those rows.
PAST_FRAME_RE = re.compile(
    r"\b(?:previously|formerly|used\s+to|no\s+longer|in\s+the\s+past|"
    r"back\s+then|at\s+the\s+time|my\s+old\s+(?:address|city|job|number|"
    r"house|flat|apartment)|before\s+that)\b",
    re.IGNORECASE,
)

#: Question wording that asks about a past value on purpose ("what was my
#: garage code before", "where did I live originally").
QUESTION_PAST_FRAME_RE = re.compile(
    r"\b(?:before|previously|earlier|originally|formerly|used\s+to|"
    r"back\s+then|at\s+the\s+time|old)\b",
    re.IGNORECASE,
)


def has_past_frame(text: str) -> bool:
    """True when a sentence frames its value as historical."""
    return bool(PAST_FRAME_RE.search(str(text or "")))


def asks_for_past(text: str) -> bool:
    """True when a question asks for an earlier value on purpose."""
    return bool(QUESTION_PAST_FRAME_RE.search(str(text or "")))


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")

#: word -> every word that means the same thing
SYNONYMS: dict[str, frozenset[str]] = {}
for _group in SYNONYM_GROUPS:
    for _word in _group:
        SYNONYMS[_word] = SYNONYMS.get(_word, frozenset()) | _group


def words(text: str) -> list[str]:
    """Lower-cased word tokens, apostrophes collapsed away."""
    return [
        match.group(0).replace("'", "").rstrip("-")
        for match in _WORD_RE.finditer((text or "").lower())
    ]


def content_words(text: str) -> list[str]:
    """The words of a personal question that could appear in a stored fact.

    Order is preserved and duplicates are dropped, so the result reads like the
    question's own subject when printed in an error or a note.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for word in words(text):
        if word in STOP_WORDS or word in QUERY_VERBS or word in ATTRIBUTE_WORDS:
            continue
        if word.isdigit() or word.isdecimal():
            continue
        if word in seen:
            continue
        seen.add(word)
        kept.append(word)
    return kept


def matches_words(question_words: Iterable[str], fact_text: str) -> bool:
    """True when every content word of the question is covered by the fact.

    Coverage is by stem or by synonym: "name" matches "named", "rent" matches
    "landlord", "good for" became "good" and matches "expires". A question with
    no content words (something like "what about it") never matches, so it can
    never pull an arbitrary record out of the store.
    """
    wanted = [word for word in question_words if word]
    if not wanted:
        return False
    fact_words: set[str] = set()
    for word in words(fact_text):
        fact_words.add(word)
        fact_words.add(_stem(word))
        for form in SYNONYMS.get(word, ()):
            fact_words.add(form)
            fact_words.add(_stem(form))
    for word in wanted:
        forms = {word, _stem(word)} | set(SYNONYMS.get(word, ()))
        forms |= {_stem(form) for form in tuple(forms)}
        if not (forms & fact_words):
            return False
    return True


def _stem(word: str) -> str:
    """A conservative suffix strip, the same shape the finder already used."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word
