import importlib


def test_cancel_lookup_matches_typo_and_grammar_variant(monkeypatch):
    needle_module = importlib.import_module("nix_knowledge.needle")
    event = {
        "record_id": 23,
        "title": "take medicines",
        "status": "scheduled",
        "start": "2026-09-22T00:00:00-05:00",
    }

    def fake_find(engine, title=None, **kwargs):
        if title:
            return {"ok": True, "events": []}
        return {"ok": True, "events": [event]}

    monkeypatch.setattr(needle_module, "find_calendar_events", fake_find)
    needle = object.__new__(needle_module.KnowledgeNeedle)
    needle.engine = None

    result = needle._find_event_for_mutation(
        "reminder for taking medicienes"
    )
    assert result["ok"] is True
    assert result["event"]["record_id"] == 23
