from pathlib import Path

import casper_model


def test_v6_is_opt_in_and_model_inventory_is_explicit(monkeypatch, tmp_path):
    v5 = tmp_path / "v5"
    v6 = tmp_path / "v6"
    v5.mkdir()
    v6.mkdir()
    monkeypatch.setattr(casper_model, "ADAPTERS", {
        "casper-puca-qlora-v5": v5,
        "casper-puca-qlora-v6": v6,
    })
    monkeypatch.setattr(casper_model, "_selected_model", "casper-puca-qlora-v5")
    monkeypatch.setattr(casper_model, "_instance", None)

    models = casper_model.available_models()
    assert models[0]["active"] is True
    assert models[1]["beta"] is True
    assert models[1]["present"] is True


def test_switch_changes_selected_adapter_without_constructing_both(monkeypatch, tmp_path):
    v5 = tmp_path / "v5"
    v6 = tmp_path / "v6"
    v5.mkdir()
    v6.mkdir()
    monkeypatch.setattr(casper_model, "ADAPTERS", {
        "casper-puca-qlora-v5": v5,
        "casper-puca-qlora-v6": v6,
    })
    monkeypatch.setattr(casper_model, "_selected_model", "casper-puca-qlora-v5")
    sentinel = object()
    monkeypatch.setattr(casper_model, "_instance", sentinel)

    assert casper_model.select_model("casper-puca-qlora-v6") == "casper-puca-qlora-v6"
    assert casper_model.selected_model() == "casper-puca-qlora-v6"
    assert casper_model._instance is None
    assert casper_model.available_models()[0]["active"] is False
    assert casper_model.available_models()[1]["active"] is True


def test_switch_rejects_missing_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(casper_model, "ADAPTERS", {
        "casper-puca-qlora-v5": tmp_path / "missing-v5",
        "casper-puca-qlora-v6": tmp_path / "missing-v6",
    })
    monkeypatch.setattr(casper_model, "_selected_model", "casper-puca-qlora-v5")
    try:
        casper_model.select_model("casper-puca-qlora-v6")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing beta adapter must not be selectable")
