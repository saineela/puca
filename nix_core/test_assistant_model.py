from __future__ import annotations

from pathlib import Path

import assistant_model


def test_luna_v6_is_unconditional_default():
    from luna_runtime import LUNA_V6_MODEL_ID

    assert assistant_model.DEFAULT_MODEL_ID == LUNA_V6_MODEL_ID
    assert assistant_model.selected_model() == LUNA_V6_MODEL_ID
    assert assistant_model.assistant_name_for_model() == "Luna"


def test_luna_v6_is_default_and_casper_models_are_fallbacks(monkeypatch, tmp_path):
    from luna_runtime import LUNA_V6_MODEL_ID

    base = tmp_path / "llama"
    base.mkdir()
    adapter = tmp_path / "luna-v6"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    v5 = tmp_path / "casper-v5"
    v6 = tmp_path / "casper-v6"
    v5.mkdir()
    v6.mkdir()

    monkeypatch.setattr(assistant_model, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(assistant_model, "LUNA_BASE_PATH", base)
    monkeypatch.setattr(assistant_model, "LUNA_V6_ADAPTER_PATH", adapter)
    monkeypatch.setattr(assistant_model, "_selected_model", LUNA_V6_MODEL_ID)
    import casper_model
    monkeypatch.setattr(casper_model, "ADAPTERS", {
        assistant_model.CASPER_V5_MODEL_ID: v5,
        assistant_model.CASPER_V6_MODEL_ID: v6,
    })

    models = assistant_model.available_models()
    assert assistant_model.selected_model() == LUNA_V6_MODEL_ID
    assert assistant_model.assistant_name_for_model() == "Luna"
    assert [item["id"] for item in models] == [
        LUNA_V6_MODEL_ID,
        assistant_model.CASPER_V5_MODEL_ID,
        assistant_model.CASPER_V6_MODEL_ID,
    ]
    assert models[0]["official"] is True
    assert models[0]["default"] is True
    assert models[0]["production_default"] is True
    assert models[0]["present"] is True
    assert all(item["present"] for item in models[1:])
    assert all(item["default"] is False for item in models[1:])
    assert all(item["production_default"] is False for item in models[1:])


def test_non_transformers_backend_does_not_advertise_luna_selector(monkeypatch):
    monkeypatch.setattr(assistant_model, "CASPER_BACKEND", "ollama")
    assert assistant_model.available_models() == []
    assert assistant_model.backend_for_model() == "ollama"
