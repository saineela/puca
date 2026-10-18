from __future__ import annotations

"""
Current-state engine for people in the user's life.

A statement like "my sister is sick" is NOT a durable fact - it is a
STATE that is currently true and will change ("she's cured now").
States are stored as knowledge_type="person" records carrying:

    value          the statement as learned
    subject        person name when known, else "user's <role>"
    role           sister/brother/mom/... (when present)
    name           resolved person name (registry or explicit)
    state          canonical state word (sick, cured, happy, ...)
    valence        good | bad | neutral   (drives mood/sound/replies)
    statement_type "current_state"

When a new state CONTRADICTS the person's active state (sick -> cured),
the old records - person records AND the key/fact rows that asserted
the old state about that person - are closed as superseded, so the
knowledge base keeps exactly one current state per person per family.

All deterministic: no model calls anywhere in this module.
"""

import json
import re
from datetime import datetime, timezone
from typing import Any

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

CLOSE_PEOPLE_ROLES = frozenset({
    "sister", "brother", "mom", "mum", "mother", "dad", "father",
    "wife", "husband", "son", "daughter", "grandma", "grandmother",
    "grandpa", "grandfather", "partner", "best friend",
})


def follow_up_eligible(data: dict[str, Any]) -> bool:
    """Return the single authoritative caring-follow-up decision.

    A follow-up is permitted only for a close/loved person whose current
    state is an active problem and whose prior concern has not been answered.
    Older records may derive closeness from a recognized close-family role,
    but a policy label alone can never make an ordinary relationship close.
    """
    closeness = str(data.get("relationship_closeness") or "").strip().lower()
    policy = str(data.get("follow_up_policy") or "").strip().lower()
    role = str(data.get("role") or "").strip().lower()
    close = closeness in {"close", "loved"} or (
        not closeness
        and policy == "only_if_close_and_unwell_or_problem"
        and role in CLOSE_PEOPLE_ROLES
    )
    active_problem = data.get("follow_up_needed") is True and str(
        data.get("valence") or ""
    ).lower() == "bad"
    answered = data.get("follow_up_answered") is True
    return close and active_problem and not answered


PEOPLE_ROLES = (
    "sister", "brother", "mom", "mum", "mother", "dad", "father",
    "wife", "husband", "son", "daughter", "grandma", "grandmother",
    "grandpa", "grandfather", "aunt", "uncle", "cousin", "niece",
    "nephew", "girlfriend", "boyfriend", "partner", "friend",
    "best friend", "roommate",
)
_PEOPLE_ROLES_NORMALIZED = frozenset(PEOPLE_ROLES)
_PEOPLE_ROLES_RE = "|".join(re.escape(role) for role in PEOPLE_ROLES)
_ROLES_RE = _PEOPLE_ROLES_RE

# canonical state -> (valence, group)
STATE_VOCAB: dict[str, tuple[str, str]] = {
    # ill family (bad)
    "sick": ("bad", "ill"),
    "ill": ("bad", "ill"),
    "unwell": ("bad", "ill"),
    "flu": ("bad", "ill"),
    "fever": ("bad", "ill"),
    "cold": ("bad", "ill"),
    "covid": ("bad", "ill"),
    "nauseous": ("bad", "ill"),
    "migraine": ("bad", "ill"),
    # well family (good)
    "cured": ("good", "well"),
    "recovered": ("good", "well"),
    "healed": ("good", "well"),
    "better": ("good", "well"),
    "healthy": ("good", "well"),
    "alright": ("good", "well"),
    "fine": ("good", "well"),
    "doing well": ("good", "well"),
    "okay": ("good", "well"),
    "ok": ("good", "well"),
    "well": ("good", "well"),
    # mood: sad (bad) vs happy (good)
    "sad": ("bad", "sad"),
    "upset": ("bad", "sad"),
    "unhappy": ("bad", "sad"),
    "depressed": ("bad", "sad"),
    "happy": ("good", "happy"),
    "cheerful": ("good", "happy"),
    "excited": ("good", "happy"),
    "joyful": ("good", "happy"),
    # energy: tired (bad) vs rested (good)
    "tired": ("bad", "tired"),
    "exhausted": ("bad", "tired"),
    "sleepy": ("bad", "tired"),
    "drained": ("bad", "tired"),
    "rested": ("good", "rested"),
    "energetic": ("good", "rested"),
    "refreshed": ("good", "rested"),
    # stress: stressed (bad) vs calm (good)
    "stressed": ("bad", "stressed"),
    "anxious": ("bad", "stressed"),
    "worried": ("bad", "stressed"),
    "nervous": ("bad", "stressed"),
    "calm": ("good", "calm"),
    "relaxed": ("good", "calm"),
    "relieved": ("good", "calm"),
}

# physical injury family (bad)
INJURY_WORDS = ("hurt", "injured", "wounded", "sprained", "fractured")
for _w in INJURY_WORDS:
    STATE_VOCAB[_w] = ("bad", "ill")

_STATE_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(STATE_VOCAB, key=len, reverse=True)) + r")\b", re.I
)

# Presence family (whereabouts: "my dad is back from his trip", "my mom
# is out of town"). The bare words are deliberately registered in
# STATE_VOCAB *after* _STATE_WORD_RE is compiled, so they never match
# as standalone state words: "back" and "away" are ordinary adverbs
# ("give the book back", "my dad's back is sore") and must only read
# as states through the be-verb-anchored phrase patterns below.
STATE_VOCAB["back"] = ("good", "presence")
STATE_VOCAB["away"] = ("neutral", "presence")

_PRESENCE_BACK_RE = re.compile(
    r"\b(?:is|are|was|were|'s|\u2019s)\s+"
    r"(?:(?:now|already|finally|just|officially)\s+)?"
    r"back\s+(?:from|home|in\s+town)\b",
    re.I,
)
_PRESENCE_AWAY_RE = re.compile(
    r"\b(?:is|are|was|were|'s|\u2019s)\s+"
    r"(?:(?:still|currently|gonna\s+be|going\s+to\s+be)\s+)?"
    r"(?:away|out\s+of\s+town)\b",
    re.I,
)


def _match_presence(text: str) -> tuple[str, str, str] | None:
    """Be-verb-anchored whereabouts detection on the full sentence.

    Like _RECOVERY_RE, this searches the whole utterance rather than
    the parsed `rest` fragment: the optional name slot in the person
    patterns can steal tokens such as "back" ("my dad is back from
    his trip" parses with name="back", rest="from his trip"), so the
    reliable cue lives in the sentence as a whole.
    """
    if _PRESENCE_BACK_RE.search(text):
        return "back", "good", "presence"
    if _PRESENCE_AWAY_RE.search(text):
        return "away", "neutral", "presence"
    return None

# Common speech-to-text/transcription swaps for relationship words. These
# are deliberately limited to close-role vocabulary and only normalize the
# role after "my", so ordinary names and facts are never silently rewritten.
_ROLE_ALIASES = {
    "sistser": "sister", "sisterr": "sister", "siter": "sister",
    "brohter": "brother", "broter": "brother", "mtoher": "mother",
    "motheer": "mother", "fater": "father", "daugher": "daughter",
}

def _normalize_role_typos(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1).lower()
        return "my " + _ROLE_ALIASES.get(token, token)
    return re.sub(r"\bmy\s+([a-z]+)\b", replace, text, flags=re.I)

# families that supersede each other: new group vs old group
CONTRADICTING_FAMILIES: dict[str, str] = {
    "ill": "well",
    "well": "ill",
    "sad": "happy",
    "happy": "sad",
    "tired": "rested",
    "rested": "tired",
    "stressed": "calm",
    "calm": "stressed",
}

# recovery phrasings: "she isnt sick anymore", "no more flu",
# "flu is gone", "feeling better", "all cured now"
_RECOVERY_RE = re.compile(
    r"\b(?:not\s+\w+\s+anymore|no\s+(?:more|longer)|isn'?t\s+\w+|"
    r"wasn'?t\s+\w+|no\s+longer|(?:is\s+)?gone|cured|recovered|"
    r"all\s+better|feeling\s+better|feels?\s+better)\b",
    re.I,
)

_NAME = r"([a-z][a-z']+)"
# be-verbs including NEGATED forms: consuming them here keeps them out
# of the optional name slot, while recovery detection still sees the
# full sentence text (negation must flip the state family)
_BE = r"(?:is|was|seems|looks|isn'?t|wasn'?t|aren'?t|ain'?t|'s|\u2019s)\b"

# "maanvi my sister is sick with flu"
_P_NAME_ROLE_STATE = re.compile(
    rf"^{_NAME}\s+my\s+({_ROLES_RE})\s+(?:{_BE}\s+)?(.+)$", re.I
)
# "my sister is sick" / "my sister maanvi is sick" /
# "my sister isnt sick anymore" (negated be-verbs stay in `rest` so
# recovery detection sees them)
_P_ROLE_STATE = re.compile(
    rf"^my\s+({_ROLES_RE})\s+(?:{_NAME}\s+)?(?:{_BE}\s+)?(.+)$", re.I
)
# "maanvi's cured now" / "maanvi is cured"
_P_NAME_STATE = re.compile(
    rf"^{_NAME}(?:'s|\u2019s)?(?:\s+{_BE})?\s+(.+)$", re.I
)

_NAME_STOPWORDS = {
    "i", "im", "i'm", "my", "me", "we", "it", "its", "it's", "this",
    "that", "there", "he", "she", "they", "you", "the", "a", "an",
    "and", "but", "so", "also", "now", "today", "yesterday", "btw",
    "hey", "nix", "just", "oh", "um", "well", "update",
    # be-verbs (incl. negated): the optional name group must never eat them
    "is", "was", "seems", "looks",
    # question words: "how is my mother doing" must never register "how"
    # as the mother's name (measured: a person record named "how" rendered
    # as "how: better" on every later state read)
    "how", "what", "who", "when", "where", "why", "which",
    "did", "does", "doing", "feeling", "recently", "lately",
    "isnt", "isn't", "wasnt", "wasn't", "arent", "aren't", "aint",
    # interjections/addressing: "bro my sister is sick" - "bro" is the
    # user addressing the assistant, NOT the sister's name. Left as a
    # name it poisons the entity registry (role sister -> name bro) and
    # every later state supersede anchors to the wrong person.
    "bro", "dude", "man", "sis", "haha", "lol", "omg", "ugh",
    "okay", "ok", "sorry", "hmm", "huh", "wow", "yeah", "yes",
    "no", "wait", "look", "listen", "guys", "someone", "somebody",
    "anyone", "anybody", "everyone", "everybody", "people", "person",
    # whereabouts adverbs/prepositions: the optional name slot in the
    # person patterns would otherwise swallow them ("my mom is out of
    # town" -> name="out", "my dad is back from his trip" ->
    # name="back") and poison the entity registry.
    "back", "away", "out", "home", "in", "from", "town", "trip",
    # negation/do-family: a sentence-initial "dont ..." ("dont ask me
    # questions like in the start alright") is an instruction to the
    # assistant, never a person's name. Left as a name it became a
    # bogus person record with a state scraped from the tail of the
    # sentence. Both apostrophe and unapostrophed spellings occur.
    "do", "does", "did", "dont", "don't", "didnt", "didn't",
    "doesnt", "doesn't", "not", "never", "nope",
    "cant", "can't", "wont", "won't",
    # command/politeness verbs and fillers: an utterance opener like
    # "remind me to call my mom" or "please check on my sister" is an
    # instruction, so its leading verb must never sit in the optional
    # name slot of the person patterns (state_person_name rejects the
    # whole candidate so "dont ask"-style multiword junk dies too).
    "remind", "reminder", "call", "tell", "ask", "text", "say",
    "please", "thanks", "thank", "welcome", "excuse", "hey",
    # state verbs: "feels tired" attributes the feeling to a subject,
    # "feels" is not the subject itself.
    "feels", "feel", "felt", "feeling", "gets", "got", "getting",
    "goes", "went", "gone", "going", "acts", "sounds", "looks_like",
}


def _clean_name(candidate: str | None) -> str | None:
    """'maanvi's' -> 'maanvi' (possessive marker is not part of a name)."""
    if not candidate:
        return None
    return re.sub(r"'s$", "", candidate).strip() or None


def _fuzzy_state_word(token: str) -> str | None:
    """Map a near-miss state word to its canonical form.

    Typed and ASR-mangled input produces typos the generic spell
    corrector cannot safely fix ('sik' snaps to 'sit' there, which
    would erase the illness meaning). Here the candidate is compared
    ONLY against the small closed state vocabulary.

    Safety rules derived from collision analysis against a full
    English dictionary (tests/test_hard_person_prompts.py locks the
    dangerous cases):

    - Substitution-only at distance 1: transpositions/deletions turn
      real words into state words ('tested'->rested, 'headed'->healed,
      'butter'->better), so only tokens the same length as the state
      word with exactly one swapped letter qualify.
    - Length >= 5: at 4 letters the 1-edit neighborhood is a minefield
      ('call'->calm, 'wine'->fine, 'kick'->sick); at 5 it is already
      sparse ('fired'/'hired'->tired are the notable real collisions,
      which the context sentence must decide).
    - Single candidate: if two state words are both 1 edit away the
      token is genuinely ambiguous and is left alone.
    """
    if len(token) < 5 or token in STATE_VOCAB:
        return None
    best: str | None = None
    matches = 0
    for word in STATE_VOCAB:
        if abs(len(word) - len(token)) > 1:
            continue
        distance = _levenshtein(token, word)
        # same-length 1-edit = one swapped letter ('stresed'->stressed).
        # +/-1-length 1-edit = one dropped or doubled letter ('tird' is
        # handled by the >=5 rule; 'exhasted'->exhausted is a doubled
        # letter and IS accepted, but only for tokens >= 6 where the
        # neighborhood is sparse).
        deletions_allowed = len(token) >= 6
        if distance == 1 and (len(word) == len(token) or deletions_allowed):
            matches += 1
            best = word
            if matches > 1:
                return None
    if matches == 1:
        return best
    return None


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 2
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def state_person_name(candidate: Any) -> str | None:
    """Reject state words and conversational fillers captured as names."""
    if not isinstance(candidate, str):
        return None
    name = _clean_name(candidate)
    if not name:
        return None
    normalized = name.casefold()
    # multiword junk ('dont ask', 'remind me') can never be a person:
    # real names in this pipeline are single tokens.
    if " " in name.strip():
        return None
    if normalized in _NAME_STOPWORDS or normalized in STATE_VOCAB:
        return None
    return name


def state_person_label(data: dict[str, Any]) -> str | None:
    """Return a real person's name/relationship, not a state or empty subject."""
    name = state_person_name(data.get("name"))
    if name:
        return name[:1].upper() + name[1:]

    # Older records sometimes kept only a generic subject but retained the
    # original utterance. Recover a relationship/name only from that explicit
    # person-state wording; never use the state itself as an entity label.
    value = data.get("value")
    if isinstance(value, str):
        parsed = parse_state_statement(value)
        if parsed:
            inferred_name = state_person_name(parsed.get("name"))
            if inferred_name:
                return inferred_name[:1].upper() + inferred_name[1:]
            inferred_role = parsed.get("role")
            if isinstance(inferred_role, str) and inferred_role.casefold() in _PEOPLE_ROLES_NORMALIZED:
                return f"Your {inferred_role}"

    role = data.get("role")
    if isinstance(role, str) and role.strip().casefold() in _PEOPLE_ROLES_NORMALIZED:
        return f"Your {role.strip()}"

    subject = data.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        return None
    subject = subject.strip()
    normalized = " ".join(subject.casefold().split())
    role_match = re.fullmatch(r"(?:user's|your)\s+(.+)", subject, re.IGNORECASE)
    if role_match and role_match.group(1).casefold() in _PEOPLE_ROLES_NORMALIZED:
        return f"Your {role_match.group(1)}"
    if role_match and state_person_name(role_match.group(1)) is None:
        return None
    if normalized in STATE_VOCAB or normalized in {
        "someone close", "someone", "person", "unknown", "none"
    } or state_person_name(subject) is None:
        return None
    return subject[:1].upper() + subject[1:]


def valence_of(text: str) -> str | None:
    """'good' | 'bad' | None for any state vocabulary in the text."""
    match = _STATE_WORD_RE.search(text)
    if match:
        return STATE_VOCAB[match.group(1).lower()][0]
    return None


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def _match_state(rest: str) -> tuple[str, str, str] | None:
    """Return (canonical_state, valence, group) for a clause fragment."""
    match = _STATE_WORD_RE.search(rest)
    if match:
        word = match.group(1).lower()
        valence, group = STATE_VOCAB[word]
        return word, valence, group
    # conservative fuzzy pass for near-miss state words ('sik'->sick,
    # 'tird'->tired): only exact 1-edit matches against the closed
    # state vocabulary qualify, so ordinary words and names survive.
    for token in re.findall(r"[a-z']+", rest):
        canonical = _fuzzy_state_word(token.lower())
        if canonical is not None:
            valence, group = STATE_VOCAB[canonical]
            return canonical, valence, group
    return None


def _recover_state(text: str) -> tuple[str, str, str] | None:
    """
    Negation/recovery detection: "she isnt sick anymore", "no more
    flu", "flu is gone" describe the WELL side even without an
    explicit positive word. Checked against the FULL sentence so
    negated be-verbs ("isnt") are still visible.
    """
    if not _RECOVERY_RE.search(text):
        return None
    matched = _match_state(text)
    if matched and matched[2] in ("well", "happy", "rested", "calm"):
        return matched  # explicit positive word wins
    # a negated bad-family word ("isnt sick anymore") is a recovery
    return "better", "good", "well"


_MODAL_CONDITIONAL_RE = re.compile(
    r"\b(?:would|should|could|might|may|must)\b\s+(?:be|have|get|feel|like|love|hate|want)\b|"
    r"\b(?:would|might)\s+\w+\s+if\b",
    re.I,
)


def _is_hypothetical(lowered: str) -> bool:
    """'my mom would be happy if you called' is a hypothetical, not a
    report of her current state. Modal + be/feel (or an explicit 'if'
    clause attached to a modal) marks the whole utterance as
    conditional; storing it would fabricate a present-tense state."""
    return bool(_MODAL_CONDITIONAL_RE.search(lowered)) or bool(
        re.search(r"\b(?:would|should|could|might|may)\b.*\bif\b", lowered)
    )


def _first_clause(text: str) -> str:
    """Cut a multi-person sentence at the clause boundary.

    'my sister is fine but my brother is stressed' must attribute
    'fine' to the sister, never scrape 'stressed' across the 'but' into
    her record. The same guard covers 'and', semicolons, and 'while' —
    each boundary starts a new subject whose state belongs to it alone.
    """
    return re.split(
        r"\b(?:but|however|though|although|while|whereas|meanwhile|and then)\b|;",
        text,
        maxsplit=1,
    )[0].strip()


def parse_state_statement(text: str) -> dict[str, Any] | None:
    """
    Parse a close-person state statement.

    Returns {role, name, state, valence, group, rest} or None when the
    text is not a state statement about a close person.
    """
    original = text.strip()
    lowered = original.lower().rstrip(".!?").strip()
    if not lowered:
        return None
    lowered = _normalize_role_typos(lowered)

    role: str | None = None
    name: str | None = None
    rest: str | None = None

    match = _P_NAME_ROLE_STATE.match(lowered)
    if match:
        candidate, role, rest = match.groups()
        name = state_person_name(candidate)
    if rest is None:
        match = _P_ROLE_STATE.match(lowered)
        if match:
            role, maybe_name, rest = match.groups()
            name = state_person_name(maybe_name)
    if rest is None:
        # name-only possessive ("maanvi's cured now"): the real gate
        # is the state vocabulary in `rest`, not capitalization -
        # casual typed input often lowercases proper nouns. Registry
        # confirmation is applied later when the engine is available.
        match = _P_NAME_STATE.match(lowered)
        if match:
            candidate, rest = match.groups()
            name = state_person_name(candidate)

    if rest is not None and role is not None or name is not None:
        # Multi-person sentences: this clause's state ends at the next
        # subject boundary. Without the cut, "my sister is fine but my
        # brother is stressed" stores 'stressed' on the sister's record.
        cut = _first_clause(rest)
        if cut and cut != rest:
            rest = cut

    if rest is None or (role is None and name is None):
        return None

    if _is_hypothetical(lowered):
        return None

    # explicit state word first; negation/recovery cues in the full
    # sentence flip bad-family words to their opposite side
    # ("my sister isnt sick anymore" -> recovered, not sick)
    parsed = _match_state(rest)
    if (
        parsed
        and parsed[2] in ("ill", "sad", "tired", "stressed")
        and _RECOVERY_RE.search(lowered)
    ):
        parsed = ("better", "good", "well")
    elif (
        parsed
        and re.search(rf"\b(?:is|was|are|were|'s|\u2019s)?\s*not\s+{re.escape(parsed[0])}\b", lowered)
    ):
        # direct predicate negation ("my brother is not stressed",
        # "my sister is not happy"): the matched word is explicitly
        # denied for THIS person, so the opposite side is recorded
        # instead of the denied word. The _RECOVERY_RE branch above
        # cannot catch plain 'not X' (no anymore/no-longer cue), and
        # without this the negated word was stored as if affirmed —
        # the exact misidentification the hard tests exist to stop.
        opposites = {"ill": "well", "sad": "happy", "tired": "rested", "stressed": "calm",
                     "happy": "sad", "excited": "sad", "cheerful": "sad"}
        flipped = opposites.get(parsed[2])
        if flipped is None:
            # good-family word without a listed bad opposite (e.g.
            # negated "doing well"): record the bad side generically.
            parsed = ("upset", "bad", "sad")
        else:
            group_flipped = {"well": "ill", "happy": "sad", "rested": "tired", "calm": "stressed"}
            parsed = (flipped, "bad" if parsed[1] == "good" else "good", group_flipped.get(flipped, parsed[2]))
    elif parsed is None:
        # whereabouts phrases ("is back from", "is out of town") are
        # matched on the full sentence, before recovery repair
        parsed = _match_presence(original) or _recover_state(lowered)
    if parsed is None:
        return None

    state, valence, group = parsed
    return {
        "role": role,
        "name": name,
        "state": state,
        "valence": valence,
        "group": group,
        "rest": rest,
    }


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------


def _person_anchor(text: str, *, name: str | None, role: str | None) -> bool:
    lowered = text.lower()
    if name and re.search(rf"\b{re.escape(name)}\b", lowered):
        return True
    if role and re.search(rf"\b{re.escape(role)}\b", lowered):
        return True
    return False


def _old_state_words(text: str) -> set[str]:
    words = {
        m.group(1).lower() for m in _STATE_WORD_RE.finditer(text)
    }
    # presence words live outside _STATE_WORD_RE by design; their
    # phrase forms are detected here so away -> back supersedes the
    # old whereabouts record like any same-family update
    if _PRESENCE_BACK_RE.search(text):
        words.add("back")
    if _PRESENCE_AWAY_RE.search(text):
        words.add("away")
    return words


def _close_record(
    engine,
    record_id: int,
    before_data: dict,
    reason: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    engine.database.execute(
        """
        UPDATE knowledge
        SET status = 'superseded',
            valid_until = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (now, now, int(record_id)),
    )
    try:
        engine._audit(
            record_id=record_id,
            operation="supersede",
            knowledge_type="person",
            before=before_data,
            after={**before_data, "status": "superseded"},
            source="state_engine",
            reason=reason,
        )
    except Exception:  # noqa: BLE001 - audit must never break supersede
        pass
    try:
        if engine.semantic is not None:
            engine.semantic.delete_record(record_id)
    except Exception:  # noqa: BLE001
        pass


def _contradicts(old_words: set[str], new_group: str) -> bool:
    wanted_group = CONTRADICTING_FAMILIES.get(new_group)
    if wanted_group is None:
        return False
    for word in old_words:
        if STATE_VOCAB.get(word, ("", ""))[1] == wanted_group:
            return True
    return False


def store_state(engine, statement: dict, raw_text: str) -> dict[str, Any]:
    """
    Store (and supersede) a current state for a close person.

    Resolution order for the name: explicit in the statement, then the
    entity registry (role -> registered name), else None and the
    subject becomes "user's <role>".
    """
    role = statement.get("role")
    name = statement.get("name")

    if name is None and role and engine.semantic is not None:
        try:
            candidates = engine.semantic.entities.candidates(role)
            if len(candidates) > 1:
                return {
                    "ok": False,
                    "operation": "NEEDS_CLARIFICATION",
                    "record_type": "person",
                    "about": f"user's {role}",
                    "role": role,
                    "candidates": candidates,
                    "question": (
                        f"Which {role} do you mean: "
                        + " or ".join(candidates)
                        + "?"
                    ),
                    "superseded": [],
                }
            name = candidates[0] if candidates else None
        except Exception:  # noqa: BLE001
            name = None

    # name-only statement ("maanvi's cured"): resolve the role back
    # from the registry so supersede anchoring matches records that
    # mention the role ("my sister is feeling better")
    if role is None and name and engine.semantic is not None:
        try:
            for known_role, known_name in engine.semantic.entities.all().items():
                if known_name == name:
                    role = known_role
                    break
        except Exception:  # noqa: BLE001
            pass

    subject = name or (f"user's {role}" if role else "someone close")

    superseded: list[int] = []

    # ---- close contradicting active records about this person -------
    try:
        rows = engine.database.execute(
            """
            SELECT id, knowledge_type, data FROM knowledge
            WHERE status = 'active'
              AND knowledge_type IN ('person', 'key', 'fact')
            ORDER BY id
            """
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []

    canonical_text = _normalize_role_typos(raw_text.strip())
    norm_new = re.sub(r"[^a-z0-9]", "", canonical_text.lower())

    for row_id, ktype, raw in rows:
        if int(row_id) == 0:
            continue
        try:
            data = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:  # noqa: BLE001
            continue
        value = str(data.get("value", ""))
        if not value:
            continue
        # must be about the same person (name or role anchor)
        if not _person_anchor(value, name=name, role=role):
            continue
        old_words = _old_state_words(value)
        new_words = {statement["state"]}

        # identical restatement -> NOOP (point at the existing record)
        if (
            ktype == "person"
            and data.get("statement_type") == "current_state"
            and re.sub(r"[^a-z0-9]", "", value.lower()) == norm_new
        ):
            return {
                "ok": True,
                "operation": "STATE_NOOP",
                "record_type": "person",
                "record_id": int(row_id),
                "data": data,
                "about": subject,
                "state": statement["state"],
                "valence": statement["valence"],
                "superseded": [],
            }

        # never close a record that already states the NEW state
        if old_words & new_words:
            continue

        old_groups = {
            STATE_VOCAB.get(w, ("", ""))[1] for w in old_words
        }
        # contradicting family -> state changed
        # same family -> newer statement replaces the older one, so
        # the person keeps exactly one current state per family
        if _contradicts(old_words, statement["group"]) or (
            ktype == "person"
            and data.get("statement_type") == "current_state"
            and statement["group"] in old_groups
        ):
            _close_record(
                engine,
                int(row_id),
                data,
                reason=(
                    f"state changed: {subject} now "
                    f"{statement['state']} (was {sorted(old_words)})"
                ),
            )
            superseded.append(int(row_id))

    value_text = canonical_text.rstrip(".!?")
    close_person = bool(role in CLOSE_PEOPLE_ROLES)
    follow_up_needed = statement["valence"] == "bad"
    data = {
        "value": value_text,
        "subject": subject,
        "statement_type": "current_state",
        "state": statement["state"],
        "valence": statement["valence"],
        "learned_from": raw_text.strip()[:200],
        # Follow-up is a separate policy variable, not an inference Casper
        # should rediscover from prose. A new state update answers any prior
        # concern; only a close person's active problem is eligible again.
        "relationship_closeness": "close" if close_person else "ordinary",
        "follow_up_policy": "only_if_close_and_unwell_or_problem",
        "follow_up_needed": follow_up_needed,
        "follow_up_answered": not follow_up_needed,
        "follow_up_eligible": close_person and follow_up_needed,
        "follow_up_reason": (
            "active_unwell_or_problem"
            if close_person and follow_up_needed
            else "already_answered_or_not_close"
        ),
        "follow_up_last_asked_at": None,
    }
    if role:
        data["role"] = role
    if name:
        data["name"] = name

    record = engine.create(
        "person",
        data,
        confidence=1.0,
        certainty="known",
        source="state_engine",
        reason="Current-state statement about a close person.",
    )

    # keep the entity registry fresh (role -> name)
    if role and name and engine.semantic is not None:
        try:
            engine.semantic.entities.register(
                role, name, record_id=record.id, evidence=value_text[:160]
            )
        except Exception:  # noqa: BLE001
            pass

    return {
        "ok": True,
        "operation": (
            "SUPERSEDE_STATE" if superseded else "STORE_STATE"
        ),
        "record_type": "person",
        "record_id": record.id,
        "data": data,
        "about": subject,
        "state": statement["state"],
        "valence": statement["valence"],
        "superseded": superseded,
    }


def find_states(engine, query: str | None = None) -> dict[str, Any]:
    """
    Return active current-state records, optionally filtered by a
    person mention in the query ("how is my sister").
    """
    try:
        rows = engine.database.execute(
            """
            SELECT id, data, created_at, updated_at FROM knowledge
            WHERE status = 'active' AND knowledge_type = 'person'
            ORDER BY id
            """
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []

    q = (query or "").lower()
    clarification = None
    if q and engine.semantic is not None:
        try:
            for role in re.findall(r"\bmy\s+(%s)\b" % _ROLES_RE, q):
                candidates = engine.semantic.entities.candidates(role)
                named = [name for name in candidates if re.search(
                    rf"\b{re.escape(name.lower())}\b", q
                )]
                if len(candidates) > 1 and not named:
                    clarification = {
                        "role": role,
                        "candidates": candidates,
                        "question": (
                            f"Which {role} do you mean: "
                            + " or ".join(candidates)
                            + "?"
                        ),
                    }
                    break
        except Exception:  # noqa: BLE001
            clarification = None

    states = []
    if clarification is None:
        for row_id, raw, created_at, updated_at in rows:
            try:
                data = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            except Exception:  # noqa: BLE001
                continue
            if q:
                # A question about a PERSON is matched on the person, not
                # on any word that happens to appear in the record. "how is
                # my father doing" used to reach the mother's record because
                # "doing" appears in it; the answer was about someone else.
                wanted_roles = {
                    _ROLE_ALIASES.get(role.casefold(), role.casefold())
                    for role in re.findall(r"\bmy\s+(%s)\b" % _ROLES_RE, q)
                }
                if wanted_roles:
                    record_role = str(data.get("role") or "").casefold()
                    if record_role not in wanted_roles:
                        continue
                else:
                    haystack = json.dumps(data, default=str).lower()
                    if not any(
                        token in haystack
                        for token in re.findall(r"[a-z]+", q)
                        if len(token) > 2
                    ):
                        continue
            person_label = state_person_label(data)
            if person_label is None:
                continue
            states.append(
                {
                    "record_id": int(row_id),
                    "value": data.get("value", ""),
                    "subject": person_label,
                    "name": state_person_name(data.get("name")),
                    "role": data.get("role"),
                    "state": data.get("state", ""),
                    "valence": data.get("valence", "neutral"),
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )

    # ---- moments: superseded history rendered with explicit dates ----
    # "my sister is cured" closes the sick records; a later "how is my
    # sister" must be able to say she WAS sick Sep 11-12 and is well
    # now - the history is part of the answer, not deleted truth.
    try:
        from .moments import collect_moments

        moments = collect_moments(engine, query)
    except Exception:  # noqa: BLE001 - history must never break reads
        moments = []

    return {
        "ok": True,
        "count": len(states),
        "states": states,
        "moments": moments,
        "query": query,
        "needs_clarification": clarification is not None,
        "clarification": clarification,
    }
