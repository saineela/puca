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
    "isnt", "isn't", "wasnt", "wasn't", "arent", "aren't", "aint",
    # interjections/addressing: "bro my sister is sick" - "bro" is the
    # user addressing the assistant, NOT the sister's name. Left as a
    # name it poisons the entity registry (role sister -> name bro) and
    # every later state supersede anchors to the wrong person.
    "bro", "dude", "man", "sis", "haha", "lol", "omg", "ugh",
    "okay", "ok", "sorry", "hmm", "huh", "wow", "yeah", "yes",
    "no", "wait", "look", "listen", "guys", "someone", "somebody",
    "anyone", "anybody", "everyone", "everybody", "people", "person",
    # negation/do-family: a sentence-initial "dont ..." ("dont ask me
    # questions like in the start alright") is an instruction to the
    # assistant, never a person's name. Left as a name it became a
    # bogus person record with a state scraped from the tail of the
    # sentence. Both apostrophe and unapostrophed spellings occur.
    "do", "does", "did", "dont", "don't", "didnt", "didn't",
    "doesnt", "doesn't", "not", "never", "nope",
    "cant", "can't", "wont", "won't",
}


def _clean_name(candidate: str | None) -> str | None:
    """'maanvi's' -> 'maanvi' (possessive marker is not part of a name)."""
    if not candidate:
        return None
    return re.sub(r"'s$", "", candidate).strip() or None


def state_person_name(candidate: Any) -> str | None:
    """Reject state words and conversational fillers captured as names."""
    if not isinstance(candidate, str):
        return None
    name = _clean_name(candidate)
    if not name:
        return None
    normalized = name.casefold()
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

    if rest is None or (role is None and name is None):
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
    elif parsed is None:
        parsed = _recover_state(lowered)
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
    return {m.group(1).lower() for m in _STATE_WORD_RE.finditer(text)}


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
