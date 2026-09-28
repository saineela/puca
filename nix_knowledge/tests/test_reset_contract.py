from __future__ import annotations

import importlib.util
import sys

from nix_knowledge.engine import KnowledgeEngine


def _load_api():
    path = "scripts/knowledge_api.py"
    spec = importlib.util.spec_from_file_location("test_knowledge_api", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reset_store_clears_records_and_history(tmp_path):
    api = _load_api()
    database = tmp_path / "knowledge.db"
    api.KNOWLEDGE_DB = str(database)
    api._needle = None

    engine = KnowledgeEngine(database)
    engine.create("person", {"subject": "sister", "value": "sick"})
    engine.close()

    result = api._reset_store()

    assert result["knowledge_records"] == 1
    verify = KnowledgeEngine(database)
    try:
        assert verify.search() == []
    finally:
        verify.close()
