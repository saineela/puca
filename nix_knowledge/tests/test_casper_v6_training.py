import json

from scripts.nixlm.train_casper_v6 import V5_OUTPUT, validate


def test_v6_validation_isolated_from_v5(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    data = tmp_path / "preferences.jsonl"
    data.write_text(json.dumps({
        "prompt": "User: hi",
        "chosen": "Hey.",
        "rejected": "How can I assist you today?",
    }) + "\n", encoding="utf-8")
    report, rows = validate(base, data, tmp_path / "v6", require_cuda=False)
    assert report["base_present"] is True
    assert len(rows) == 1


def test_v6_validation_rejects_v5_output(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    data = tmp_path / "preferences.jsonl"
    data.write_text(json.dumps({
        "prompt": "User: hi",
        "chosen": "Hey.",
        "rejected": "How can I assist you today?",
    }) + "\n", encoding="utf-8")
    try:
        validate(base, data, V5_OUTPUT)
    except ValueError as exc:
        assert "V5" in str(exc)
    else:
        raise AssertionError("V5 output must be rejected")
