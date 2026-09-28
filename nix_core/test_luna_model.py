import json

import pytest

import luna_model
import luna_runtime
from model_slot import GPU_SLOT


def test_only_official_luna_v6_is_listed_and_retired_pro_is_hidden(tmp_path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    v6_adapter = tmp_path / "v6-adapter"
    v6_adapter.mkdir()
    (v6_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (v6_adapter / "adapter_model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(luna_model, "LUNA_BASE_PATH", base)
    monkeypatch.setattr(luna_model, "LUNA_ADAPTER", v6_adapter)
    monkeypatch.setattr(luna_model, "_active_luna_model", None)
    monkeypatch.setattr(luna_model, "_client", None)

    models = luna_model.available_models()
    assert [model["id"] for model in models] == [
        luna_runtime.LUNA_V6_MODEL_ID,
    ]
    assert models[0]["present"]
    assert models[0]["official"] is True
    assert models[0]["experimental"] is False
    assert models[0]["production_default"] is True
    monkeypatch.setattr(luna_runtime, "LUNA_BASE_PATH", base)
    monkeypatch.setattr(luna_runtime, "LUNA_V6_ADAPTER_PATH", v6_adapter)
    status = luna_runtime.runtime_status()
    assert status["role"] == "Official Luna, the Nix PUCA conversation model"
    assert status["loaded"] is False
    assert status["label"] == "Official Luna"
    assert status["release_status"] == "official_assistant_default"
    assert status["retired_models"] == [{
        "id": "luna-pro-v1-topical-v8-384-retry2",
        "status": "retired_not_loadable",
    }]
    assert status["v7_plan"]["thinking_enabled"] is False


def test_selecting_unknown_luna_checkpoint_fails_before_load():
    with pytest.raises(ValueError, match="research-model selection is retired"):
        luna_model.select_model("luna-pro")


def test_casper_and_luna_share_the_same_reentrant_gpu_slot():
    assert luna_model.GPU_SLOT is GPU_SLOT


def test_luna_modules_are_discovery_only_until_explicit_client_request():
    assert luna_model._client is None
    assert luna_model.selected_model() is None
    assert luna_model.runtime_status()["loaded"] is False


def test_luna_research_model_selector_is_retired():
    with pytest.raises(ValueError, match="research-model selection is retired"):
        luna_model.select_model("luna-pro-v1-topical-v8-384-retry2")


def test_luna_v7_has_a_separate_gated_final_answer_only_policy():
    from luna_runtime import (
        LUNA_V6_ADAPTER_PATH,
        LUNA_V7_ADAPTER_PATH,
        LUNA_V7_MODEL_ID,
        LUNA_V7_RUNTIME_PLAN,
        LUNA_V7_THINKING_ENABLED,
        system_prompt_for_model,
    )

    assert LUNA_V7_ADAPTER_PATH != LUNA_V6_ADAPTER_PATH
    assert LUNA_V7_RUNTIME_PLAN["registered"] is False
    assert LUNA_V7_RUNTIME_PLAN["trained"] is False
    assert LUNA_V7_RUNTIME_PLAN["training_authorized"] is False
    assert LUNA_V7_THINKING_ENABLED is False

    prompt = system_prompt_for_model(LUNA_V7_MODEL_ID, "You are Luna.")
    assert "Do not produce or reveal chain-of-thought" in prompt
    assert system_prompt_for_model(
        luna_runtime.LUNA_V6_MODEL_ID, "unchanged"
    ) == "unchanged"


def test_luna_v7_runtime_forces_final_answer_only_and_strips_tagged_reasoning():
    from contextlib import nullcontext

    from luna_runtime import LUNA_V7_MODEL_ID

    captured = {}

    class FakeTokens:
        shape = (1, 2)

        def __getitem__(self, key):
            return self

    class FakeBatch(dict):
        def __init__(self):
            super().__init__(input_ids=FakeTokens())

        def to(self, device):
            assert device == "cuda"
            return self

    class FakeTokenizer:
        eos_token_id = 2
        chat_template = None

        def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            return FakeBatch()

        def convert_tokens_to_ids(self, token):
            return 3

        def decode(self, generated, **kwargs):
            return "<think>private analysis</think>Hello    "

    class FakeOutput:
        def __getitem__(self, key):
            return self

    class FakeModel:
        def generate(self, **kwargs):
            captured["generation_kwargs"] = kwargs
            return [FakeOutput()]

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()

    client = object.__new__(luna_model.LunaTransformersClient)
    client.model = LUNA_V7_MODEL_ID
    client.thinking_enabled = False
    client.tokenizer = FakeTokenizer()
    client.model_instance = FakeModel()
    client._torch = FakeTorch()

    reply = client.chat(
        system_prompt="You are Luna.",
        history=[],
        user_text="Hi",
        think=True,
    )

    assert "Do not produce or reveal chain-of-thought" in captured["prompt"]
    assert "think" not in captured["generation_kwargs"]
    assert reply == "Hello"
