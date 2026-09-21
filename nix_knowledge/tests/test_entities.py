"""Tests for entity extraction, registry, and referent resolution."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nix_knowledge.semantic.entities import (
    EntityRegistry,
    extract_entities,
    find_referents,
)


def test_extract_role_name_adjacent():
    entities = extract_entities("My sister maanvi is sick")
    assert ("sister", "maanvi") in [(e.role, e.name) for e in entities]


def test_extract_role_name_is():
    entities = extract_entities("my sister's name is Maanvi")
    assert ("sister", "maanvi") in [(e.role, e.name) for e in entities]


def test_extract_name_is_role():
    entities = extract_entities("Maanvi is my sister")
    assert ("sister", "maanvi") in [(e.role, e.name) for e in entities]


def test_extract_named():
    entities = extract_entities("my sister named maanvi visited")
    assert ("sister", "maanvi") in [(e.role, e.name) for e in entities]


def test_no_false_name_after_negation():
    """'my sister isnt sick' must NOT register 'isnt'/'sick' as a name."""
    assert extract_entities("my sister isnt sick anymore") == []
    assert extract_entities("my sister is sick") == []


def test_extract_pets_and_roles():
    assert ("dog", "bruno") in [
        (e.role, e.name) for e in extract_entities("My dog bruno loves walks")
    ]
    assert ("mom", "linda") in [
        (e.role, e.name) for e in extract_entities("my mom linda called")
    ]


def test_find_referents():
    assert find_referents("who is my sister") == ["sister"]
    assert find_referents("my sister isnt sick anymore") == ["sister"]
    assert find_referents("what about my dog and my mom") == ["dog", "mom"]


def test_registry_roundtrip():
    db = os.path.join(tempfile.mkdtemp(), "entities_test.db")
    registry = EntityRegistry(db)
    registry.register("sister", "maanvi", record_id=16,
                      evidence="my sister maanvi is sick")
    assert registry.get("sister") == "maanvi"
    assert registry.all() == {"sister": "maanvi"}

    # re-registering updates (people change names/roles)
    registry.register("sister", "jane")
    assert registry.get("sister") == "jane"
    assert registry.candidates("sister") == ["jane", "maanvi"]
    registry.close()


def test_registry_retains_multiple_people_with_same_role():
    db = os.path.join(tempfile.mkdtemp(), "multiple_people.db")
    registry = EntityRegistry(db)
    registry.register("sister", "maanvi")
    registry.register("sister", "jane")
    assert registry.candidates("sister") == ["jane", "maanvi"]
    # The legacy default is still available, but callers requiring
    # correctness must use candidates() and ask for clarification.
    assert registry.get("sister") == "jane"
    registry.close()
