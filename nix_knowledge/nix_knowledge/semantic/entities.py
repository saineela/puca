from __future__ import annotations

"""
Entities: person/animal names attached to relationship roles, and
referent resolution for retrieval.

Why: "My sister maanvi is sick" stores knowledge about MAANVI, but a
later request "my sister isn't sick anymore" never says the name. The
KnowledgeRetrieval layer must therefore:

  1. AT INGEST  extract (role, name) pairs from statements:
       "my sister maanvi"            -> sister = maanvi
       "my sister's name is Maanvi"  -> sister = Maanvi
       "Maanvi is my sister"         -> sister = Maanvi
     and register them persistently (SQLite, alongside the vectors).

  2. AT QUERY   resolve referential phrases ("my sister") through the
     registry and AUGMENT the query with the resolved names, so both
     the vector and lexical retrievers hit the records that actually
     contain the name -- and Core can speak the name in its reply.

Roles covered: family, close relations, and pets.
"""

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROLES = (
    "sister", "brother", "mom", "mother", "dad", "father",
    "wife", "husband", "daughter", "son", "cousin", "uncle", "aunt",
    "grandma", "grandpa", "grandmother", "grandfather",
    "friend", "girlfriend", "boyfriend", "partner", "roommate",
    "boss", "manager", "coworker", "neighbor", "dog", "cat", "pet",
)

_ROLES_RE = "|".join(ROLES)

# "my sister maanvi ..."  /  "my sister Maanvi ..."
# also covers "my sister named maanvi" / "my sister called maanvi"
_P_ROLE_NAME = re.compile(
    rf"\bmy ({_ROLES_RE})\s+(?:named\s+|called\s+)?([A-Za-z][a-z]+)\b(?:'s)?\b",
    re.I,
)
# "my sister's name is Maanvi" / "my sisters name is maanvi"
_P_ROLE_NAME_IS = re.compile(
    rf"\bmy ({_ROLES_RE})'?s?\s+name\s+is\s+([A-Za-z][a-z]+)\b",
    re.I,
)
# "Maanvi is my sister" / "maanvi's my sister"
_P_NAME_IS_ROLE = re.compile(
    r"\b([A-Za-z][a-z]+)\s+(?:is|'s)\s+my\s+(" + _ROLES_RE + r")\b",
    re.I,
)

# Referential phrase used at query time: "my sister", "my brother", ...
_REFERENT = re.compile(rf"\bmy ({_ROLES_RE})\b", re.I)

# Tokens that commonly follow a role but are NOT names.
_NOT_NAMES = {
    "is", "are", "was", "were", "the", "a", "an", "and", "but", "in",
    "at", "on", "to", "with", "my", "me", "not", "so", "very",
    "really", "still", "already", "yet", "again", "here", "there",
    "home", "sick", "ill", "well", "fine", "good", "great", "bad",
    "okay", "ok", "coming", "going", "gone", "back", "away",
    "best", "only", "just", "also", "too", "now", "then", "always",
    "never", "who", "that", "this", "she", "he", "they", "all",
    # negated/contracted verbs that follow a role ("my sister isnt...")
    "isnt", "aint", "wasnt", "arent", "werent", "didnt", "dont",
    "cant", "wont", "doesnt", "hasnt", "havent", "hadnt",
    "shouldnt", "couldnt", "wouldnt", "am", "named", "called",
    "like", "likes", "loves", "loved", "hates", "hated", "needs",
    "wants", "works", "lives", "died", "moved", "called",
}


@dataclass(frozen=True)
class Entity:
    role: str
    name: str


def _clean_name(token: str) -> str | None:
    token = token.strip(" .,!?").lower()
    if not token or len(token) < 2:
        return None
    if token in _NOT_NAMES or token in ROLES:
        return None
    return token


def extract_entities(text: str) -> list[Entity]:
    """Extract (role, name) pairs from a statement."""
    found: dict[str, str] = {}

    for match in _P_ROLE_NAME_IS.finditer(text):
        role, name = match.group(1).lower(), _clean_name(match.group(2))
        if name:
            found[role] = name

    for match in _P_NAME_IS_ROLE.finditer(text):
        name, role = _clean_name(match.group(1)), match.group(2).lower()
        if name:
            found.setdefault(role, name)

    for match in _P_ROLE_NAME.finditer(text):
        role, name = match.group(1).lower(), _clean_name(match.group(2))
        if name:
            found.setdefault(role, name)

    return [Entity(role=role, name=name) for role, name in found.items()]


def find_referents(text: str) -> list[str]:
    """Roles mentioned referentially in a query ('my sister' -> sister)."""
    return [
        match.group(1).lower()
        for match in _REFERENT.finditer(text)
    ]


class EntityRegistry:
    """Persistent role -> name store (SQLite)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities (
        role       TEXT PRIMARY KEY,
        name       TEXT NOT NULL,
        record_id  INTEGER,
        evidence   TEXT,
        updated_at REAL NOT NULL
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def register(
        self,
        role: str,
        name: str,
        *,
        record_id: int | None = None,
        evidence: str | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO entities
                    (role, name, record_id, evidence, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(role) DO UPDATE SET
                    name = excluded.name,
                    record_id = excluded.record_id,
                    evidence = excluded.evidence,
                    updated_at = excluded.updated_at
                """,
                (role.lower(), name.lower(), record_id, evidence,
                 time.time()),
            )

    def get(self, role: str) -> str | None:
        row = self._conn.execute(
            "SELECT name FROM entities WHERE role = ?",
            (role.lower(),),
        ).fetchone()
        return row[0] if row else None

    def all(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT role, name FROM entities"
        ).fetchall()
        return {role: name for role, name in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
