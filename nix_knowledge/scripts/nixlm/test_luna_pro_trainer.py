import json
import re
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_luna_pro_qlora as trainer  # noqa: E402


class TinyTemplateTokenizer:
    """Deterministic chat-template tokenizer for CPU-only masking tests."""

    chat_template = "test template (not used as production template)"
    eos_token = "<|end_of_text|>"
    eos_token_id = 1
    pad_token = None
    pad_token_id = None
    padding_side = "right"

    _special_ids = {
        "<|begin_of_text|>": 3,
        "<|start_header_id|>": 4,
        "<|end_header_id|>": 5,
        "<|eot_id|>": 2,
        "<|end_of_text|>": 1,
    }
    _pieces = re.compile(r"<\|[^|]+\|>|.", re.DOTALL)

    def convert_tokens_to_ids(self, token):
        return self._special_ids.get(token, -1)

    def _encode(self, text):
        return [
            self._special_ids[piece] if piece in self._special_ids else ord(piece) + 10
            for piece in self._pieces.findall(text)
        ]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=False,
        **kwargs,
    ):
        rendered = "<|begin_of_text|>"
        for message in messages:
            rendered += (
                f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                f"{message['content']}<|eot_id|>"
            )
        if add_generation_prompt:
            rendered += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return self._encode(rendered) if tokenize else rendered


def _row():
    return {
        "source": "project_authored",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "second answer"},
        ],
    }


def test_retired_trainer_gpu_and_training_commands_fail_closed():
    parser = trainer.build_parser()
    args = parser.parse_args(["train"])
    for command in ("train", "gpu-smoke"):
        with pytest.raises(PermissionError, match="Luna Pro v1 is retired"):
            trainer.main([command])

    with pytest.raises(PermissionError, match="Luna Pro v1 is retired"):
        trainer.run_gpu_command(args, smoke=False)


def test_retired_data_check_is_read_only_and_never_loads_model_or_gpu(
    tmp_path, monkeypatch, capsys
):
    import torch

    data_dir = tmp_path / "candidate-data"
    data_dir.mkdir()
    base_path = tmp_path / "local-base"
    base_path.mkdir()
    monkeypatch.setattr(trainer, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(trainer, "load_candidate_data", lambda _path: (
        {
            "base_model": {"local_path": "local-base"},
            "manifest_sha256": "fixture-manifest",
            "builder_sha256": "fixture-builder",
            "files": {"train": {"sha256": "fixture-train"}, "dev": {"sha256": "fixture-dev"}},
            "manual_review": {"status": "fixture-only"},
        },
        [],
        [],
        [],
    ))
    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

    monkeypatch.setattr(trainer, "_load_tokenizer", lambda _path: FakeTokenizer())
    monkeypatch.setattr(trainer, "validate_tokenizer_terminators", lambda _tokenizer: {})
    monkeypatch.setattr(trainer, "encode_rows", lambda *_args, **_kwargs: ([], {"rows": 0}))

    def unexpected_work(*args, **kwargs):
        pytest.fail("retired Pro data preflight attempted model-weight or CUDA work")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_work)
    import transformers
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        unexpected_work,
    )
    assert trainer.main(["check-data", "--data-dir", str(data_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_weights_loaded"] is False
    assert report["gpu_used"] is False
    assert report["status"].startswith("cpu_data_preflight_passed")


def test_generic_luna_trainer_stops_before_cuda_model_or_output_work(tmp_path, monkeypatch):
    import train_luna_qlora as luna_trainer

    output_dir = tmp_path / "must-not-be-created"
    monkeypatch.setattr(luna_trainer, "BASE", tmp_path / "missing-base")
    monkeypatch.setattr(luna_trainer, "DATA", tmp_path / "missing-data.jsonl")
    monkeypatch.setattr(luna_trainer, "OUT", output_dir)
    monkeypatch.setattr(
        luna_trainer.torch.cuda,
        "is_available",
        lambda: pytest.fail("retired Luna trainer queried CUDA"),
    )
    monkeypatch.setattr(
        luna_trainer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("retired Luna trainer loaded a tokenizer"),
    )
    monkeypatch.setattr(
        luna_trainer.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("retired Luna trainer loaded weights"),
    )

    with pytest.raises(PermissionError, match="Luna model training is not authorized"):
        luna_trainer.main()
    assert not output_dir.exists()


def _span(tokenizer, messages, assistant_index):
    prefix = trainer._token_ids(
        tokenizer,
        trainer.render_messages(messages[:assistant_index], generation=True, tokenizer=tokenizer),
    )
    end = trainer._token_ids(
        tokenizer,
        trainer.render_messages(messages[:assistant_index + 1], generation=False, tokenizer=tokenizer),
    )
    return len(prefix), len(end)


def test_encoder_supervises_all_assistant_content_and_eot_but_masks_context():
    tokenizer = TinyTemplateTokenizer()
    row = _row()
    example, dropped = trainer.encode_conversation(row, tokenizer, max_seq_length=500)

    assert dropped is None
    assert example is not None
    assistant_indexes = [2, 4]
    supervised_positions = set()
    for assistant_index in assistant_indexes:
        start, end = _span(tokenizer, row["messages"], assistant_index)
        assert all(label == example.input_ids[pos] for pos, label in enumerate(example.labels[start:end], start))
        assert example.input_ids[end - 1] == tokenizer.convert_tokens_to_ids("<|eot_id|>")
        supervised_positions.update(range(start, end))

    assert all(
        (label != -100) == (position in supervised_positions)
        for position, label in enumerate(example.labels)
    )
    assert example.supervised_tokens > 0
    assert example.labels[-1] == tokenizer.convert_tokens_to_ids("<|eot_id|>")

    for start, end in (_span(tokenizer, row["messages"], 2), _span(tokenizer, row["messages"], 4)):
        assert all(
            label == example.input_ids[position]
            for position, label in enumerate(example.labels[start:end], start)
        )
        assert example.labels[end - 1] == tokenizer.convert_tokens_to_ids("<|eot_id|>")


def test_encoder_drops_overlength_conversations_without_truncating():
    example, reason = trainer.encode_conversation(
        _row(), TinyTemplateTokenizer(), max_seq_length=8
    )
    assert example is None
    assert reason == "overlength"


def test_encoder_fails_closed_when_template_prefix_does_not_match():
    class BrokenPrefixTokenizer(TinyTemplateTokenizer):
        def apply_chat_template(
            self, messages, *, tokenize=True, add_generation_prompt=False, **kwargs
        ):
            rendered = super().apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            if add_generation_prompt:
                rendered = rendered.replace("first question", "different question")
            return self._encode(rendered) if tokenize else rendered

    with pytest.raises(trainer.CandidateValidationError, match="exact prefix"):
        trainer.encode_conversation(_row(), BrokenPrefixTokenizer(), max_seq_length=500)


def test_candidate_audit_gate_rejects_any_blocking_findings():
    clean = {
        "status": "candidate_data_built_training_not_started_review_required",
        "train_dev_group_overlap_count": 0,
        "train": {
            "role_error_count": 0,
            "empty_target_count": 0,
            "benchmark_exact_overlap_count": 0,
            "unsupported_action_claim_rows": [],
            "topical_sensitive_domain_rows": [],
            "topical_personal_claim_rows": [],
            "benchmark_near_overlap": {"blocking_near_overlap_count": 0},
        },
        "dev": {
            "role_error_count": 0,
            "empty_target_count": 0,
            "benchmark_exact_overlap_count": 0,
            "unsupported_action_claim_rows": [],
            "topical_sensitive_domain_rows": [],
            "topical_personal_claim_rows": [],
            "benchmark_near_overlap": {"blocking_near_overlap_count": 0},
        },
    }
    trainer._validate_report_audits(clean)
    contaminated = json.loads(json.dumps(clean))
    contaminated["dev"]["topical_personal_claim_rows"] = [3]
    with pytest.raises(trainer.CandidateValidationError, match="blocking findings"):
        trainer._validate_report_audits(contaminated)


def test_final_adapter_save_allows_only_active_run_checkpoints(tmp_path):
    class FakeModel:
        def save_pretrained(self, path):
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.safetensors").write_bytes(b"adapter weights")

    class FakeTokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "active-run"
    output_dir.mkdir()
    (output_dir / "training_metadata.json").write_text(
        json.dumps({"status": "started"}), encoding="utf-8"
    )
    checkpoint = output_dir / "checkpoint-0025"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint weights")

    trainer._save_adapter(FakeModel(), FakeTokenizer(), output_dir, allow_checkpoint_dirs=True)
    assert (output_dir / "adapter_model.safetensors").is_file()
    assert (output_dir / "tokenizer.json").is_file()

    stale_output_dir = tmp_path / "stale-run"
    stale_output_dir.mkdir()
    (stale_output_dir / "training_metadata.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="non-active run"):
        trainer._save_adapter(
            FakeModel(), FakeTokenizer(), stale_output_dir, allow_checkpoint_dirs=True
        )


@pytest.mark.parametrize("module_name", [
    "evaluate_luna_v6",
    "evaluate_luna_generalization",
    "build_luna_conversation",
    "build_luna_resource_mix",
    "build_luna_identity_mix",
    "build_luna_balanced_mix",
    "build_luna_clean_mix",
    "build_luna_targeted_v6",
    "build_luna_v6_curriculum",
    "build_luna_pro_sft_v1",
])
def test_archived_luna_dataset_builders_stop_before_source_or_output_work(
    module_name, tmp_path, monkeypatch
):
    import importlib

    builder = importlib.import_module(module_name)
    output_dir = tmp_path / module_name
    if hasattr(builder, "DEFAULT_OUT"):
        monkeypatch.setattr(builder, "DEFAULT_OUT", output_dir / "data.jsonl")
    if hasattr(builder, "DEFAULT_OUTPUT"):
        monkeypatch.setattr(builder, "DEFAULT_OUTPUT", output_dir / "data.jsonl")
    if hasattr(builder, "DEFAULT_CONVERSATION"):
        monkeypatch.setattr(builder, "DEFAULT_CONVERSATION", output_dir / "conversation.jsonl")
    if hasattr(builder, "DEFAULT_IDENTITY"):
        monkeypatch.setattr(builder, "DEFAULT_IDENTITY", output_dir / "identity.jsonl")
    if hasattr(builder, "DATA"):
        monkeypatch.setattr(builder, "DATA", output_dir)
    if hasattr(builder, "DEFAULT_INPUT"):
        monkeypatch.setattr(builder, "DEFAULT_INPUT", output_dir / "input.jsonl")
    if hasattr(builder, "DEFAULT_DATA_DIR"):
        monkeypatch.setattr(builder, "DEFAULT_DATA_DIR", output_dir)
    if hasattr(builder, "REPO_ROOT"):
        monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)

    def unexpected_work(*args, **kwargs):
        pytest.fail("archived Luna builder reached source access")

    for name in (
        "load_dataset",
        "load_topical_chat_records",
        "local_pairs",
        "finetome_pairs",
        "no_robots_pairs",
        "ultrafeedback_pairs",
        "project_authored_rows",
    ):
        if hasattr(builder, name):
            monkeypatch.setattr(builder, name, unexpected_work)

    if module_name == "evaluate_luna_v6":
        monkeypatch.setattr(builder.torch.cuda, "is_available", unexpected_work)
        monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", unexpected_work)
        monkeypatch.setattr(builder.AutoModelForCausalLM, "from_pretrained", unexpected_work)
    if module_name == "evaluate_luna_generalization":
        monkeypatch.setattr(builder.torch.cuda, "is_available", unexpected_work)
        monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", unexpected_work)
        monkeypatch.setattr(builder.AutoModelForCausalLM, "from_pretrained", unexpected_work)
    with pytest.raises(PermissionError, match="(Archived Luna dataset generation|Archived Luna evaluation|Luna Pro v1 is retired)"):
        builder.main()
    assert not output_dir.exists()


@pytest.mark.parametrize("command", ["gpu-smoke", "train"])
def test_retired_gpu_commands_stop_before_any_gpu_or_output_work(
    command, tmp_path, monkeypatch
):
    output_dir = tmp_path / f"{command}-must-not-be-created"
    args = trainer.build_parser().parse_args([
        command, "--output-dir", str(output_dir)
    ])

    def unexpected_work(*args, **kwargs):
        pytest.fail("retired Luna Pro command reached training setup")

    for name in (
        "validate_run_acknowledgements",
        "validate_output_path",
        "load_candidate_data",
        "_load_unsloth_model",
        "_check_gpu_prerequisites",
    ):
        monkeypatch.setattr(trainer, name, unexpected_work)

    with pytest.raises(PermissionError, match="Luna Pro v1 is retired"):
        trainer.main([command, "--output-dir", str(output_dir)])
    assert not output_dir.exists()


def test_existing_baseline_adapter_uses_read_only_validator(tmp_path, monkeypatch):
    fake_root = tmp_path / "nix_knowledge"
    monkeypatch.setattr(trainer, "ROOT", fake_root)
    adapter_path = fake_root / "models" / "nixlm" / "luna-instruct-v1"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"test weights")

    assert trainer.validate_existing_adapter_path(adapter_path) == adapter_path.resolve()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer.validate_output_path(adapter_path)

    with pytest.raises(ValueError, match="must be under"):
        trainer.validate_existing_adapter_path(tmp_path / "outside-adapter")
    (adapter_path / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="weights are missing"):
        trainer.validate_existing_adapter_path(adapter_path)



def test_frozen_eval_manifest_requires_independent_hash_pinned_cases(tmp_path):
    cases = [
        {
            "id": "fictional-case-1",
            "category": "grounding",
            "messages": [
                {"role": "system", "content": "Test-only fictional context."},
                {"role": "user", "content": "What detail is known?"},
            ],
            "review_points": ["Answer only from supplied context."],
        }
    ]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")
    manifest = {
        "status": "frozen_before_training",
        "cases_path": cases_path.name,
        "cases_sha256": trainer.sha256_file(cases_path),
        "case_count": len(cases),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    candidate_dir = tmp_path / "sft-candidate"
    candidate_dir.mkdir()
    candidate_report = {
        "files": {
            "train": {"path": "train.jsonl"},
            "dev": {"path": "dev.jsonl"},
        }
    }
    for split_name in ("train", "dev"):
        (candidate_dir / f"{split_name}.jsonl").write_text(
            json.dumps({"source_user_turns": ["A different training prompt."]}) + "\n",
            encoding="utf-8",
        )
    result = trainer.validate_frozen_eval_manifest(
        manifest_path, candidate_dir, candidate_report
    )
    assert result["case_count"] == 1
    assert result["cases_sha256"] == manifest["cases_sha256"]

    (candidate_dir / "train.jsonl").write_text(
        json.dumps({"source_user_turns": ["What detail is known?"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(trainer.CandidateValidationError, match="exactly overlaps"):
        trainer.validate_frozen_eval_manifest(
            manifest_path, candidate_dir, candidate_report
        )

    cases_path.write_text(json.dumps({**cases[0], "answer": "leaked target"}) + "\n", encoding="utf-8")
    manifest["cases_sha256"] = trainer.sha256_file(cases_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(trainer.CandidateValidationError, match="target/answer fields"):
        trainer.validate_frozen_eval_manifest(
            manifest_path, candidate_dir, candidate_report
        )
