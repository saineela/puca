"""Integration test: referent resolution through KnowledgeNeedle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.needle import KnowledgeNeedle


def test_referent_resolution_end_to_end():
    engine = KnowledgeEngine("/tmp/needle_resolve_test.db")
    try:
        needle = KnowledgeNeedle(engine)
        # user states a fact with a name
        needle._create_fact("my sister maanvi is sick")
        # registry must have the entity
        assert engine.semantic.entities.get("sister") == "maanvi"
        # later referential query resolves to the name
        result = needle._find_facts("my sister isnt sick anymore")
        assert result["resolved_entities"] == [
            {"role": "sister", "name": "maanvi"}
        ]
        assert result["state_change_detected"] is True
        values = [f["value"] for f in result["facts"]]
        assert any("maanvi" in v.lower() for v in values), values
    finally:
        engine.close()
        import os
        for suffix in ("", "_vectors.db", "_entities.db"):
            try:
                os.remove("/tmp/needle_resolve_test.db" + suffix)
            except FileNotFoundError:
                pass
