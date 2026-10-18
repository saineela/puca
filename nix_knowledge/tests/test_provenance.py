from __future__ import annotations

import pytest

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.provenance import (
    ProvenanceError,
    TurnReference,
    require_turn_reference,
)


@pytest.fixture()
def engine(tmp_path):
    instance = KnowledgeEngine(tmp_path / "knowledge.db")
    yield instance
    instance.close()


def test_turn_reference_round_trips_and_rejects_malformed_ids():
    source_id = TurnReference("conv-1", "msg-42", event_id="evt-7").source_id
    parsed = TurnReference.parse(source_id)
    assert parsed.conversation_id == "conv-1"
    assert parsed.message_id == "msg-42"
    assert parsed.event_id == "evt-7"
    with pytest.raises(ProvenanceError):
        TurnReference.parse("source: something")
    with pytest.raises(ProvenanceError):
        TurnReference.parse("")


def test_conversation_writes_require_a_turn_reference():
    with pytest.raises(ProvenanceError):
        require_turn_reference("conversation", None)
    with pytest.raises(ProvenanceError):
        require_turn_reference("conversation", "not-a-turn")
    assert require_turn_reference(
        "conversation", "conversation:c1|message:m2"
    ) == "conversation:c1|message:m2"
    # System writes are not conversational and may stay unreferenced.
    assert require_turn_reference("system", None) == ""


def test_supersede_links_old_record_and_keeps_history(engine):
    old = engine.create(
        "preference",
        {"topic": "city", "value": "Seattle"},
        source="conversation",
        source_id="conversation:c1|message:m1",
    )
    new = engine.supersede(
        old.id,
        data={"topic": "city", "value": "Austin"},
        source="conversation",
        source_id="conversation:c1|message:m9",
    )

    assert new.id != old.id
    reread = engine.get(old.id)
    assert reread.status == "superseded"
    assert reread.superseded_by == new.id
    assert reread.valid_until is not None
    assert reread.provenance["source_id"] == "conversation:c1|message:m1"
    assert new.provenance["source_id"] == "conversation:c1|message:m9"

    # The default search is current-only; history is explicit.
    assert [record.id for record in engine.search("preference")] == [new.id]
    assert [record.id for record in engine.search("preference", status="superseded")] == [old.id]
    assert [record.id for record in engine.supersede_chain(old.id)] == [old.id, new.id]
    assert engine.current_record(old.id).id == new.id


def test_supersede_writes_an_audit_row_with_the_reason(engine):
    old = engine.create("person", {"person": "mother", "value": "sick"}, source="system")
    new = engine.supersede(old.id, data={"person": "mother", "value": "fine"}, reason="owner said she is better")
    row = engine.database.fetchone(
        "SELECT * FROM knowledge_changes WHERE record_id = ? AND operation = 'supersede'",
        (old.id,),
    )
    assert row is not None
    assert row["reason"] == "owner said she is better"
    assert str(new.id) in row["reason"] or row["after_data"]


def test_supersede_of_an_already_superseded_record_is_rejected(engine):
    old = engine.create("fact", {"x": 1}, source="system")
    engine.supersede(old.id, data={"x": 2}, source="system")
    with pytest.raises(ValueError):
        engine.supersede(old.id, data={"x": 3}, source="system")


def test_failed_supersede_rolls_back_without_a_dangling_link(engine):
    old = engine.create("fact", {"x": 1}, source="system")
    with pytest.raises(ProvenanceError):
        engine.supersede(old.id, data={"x": 2}, source="conversation")
    reread = engine.get(old.id)
    assert reread.status == "active"
    assert reread.superseded_by is None


def test_model_person_state_supersede_chain_is_resolvable(engine):
    first = engine.create(
        "person",
        {"person": "mom", "value": "sick"},
        source="conversation",
        source_id="conversation:c2|message:m1",
    )
    second = engine.supersede(
        first.id,
        data={"person": "mom", "value": "recovering"},
        source="conversation",
        source_id="conversation:c2|message:m2",
    )
    third = engine.supersede(
        second.id,
        data={"person": "mom", "value": "fine"},
        source="conversation",
        source_id="conversation:c2|message:m3",
    )
    chain = engine.supersede_chain(first.id)
    assert [record.data["value"] for record in chain] == ["sick", "recovering", "fine"]
    assert engine.current_record(first.id).id == third.id
    assert chain[0].provenance["source_id"] == "conversation:c2|message:m1"
