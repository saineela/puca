"""D5 grammar: the envelope constraint is optional, and honest about it."""
from __future__ import annotations

import sys
import types

from nix_knowledge import grammar


def test_no_engine_means_generation_is_unchanged():
    """On a box with no engine the planner must generate exactly as before."""
    assert grammar.available_engine() is None
    assert grammar.grammar_logits_processor(tokenizer=object(), prompt_length=0) is None


def test_envelope_check_accepts_the_shapes_the_planner_emits():
    assert grammar.envelope_is_well_formed(
        '<tool_call>{"tool": "opendeck_music", "arguments": {"action": "play"}}</tool_call>'
    )
    assert grammar.envelope_is_well_formed(
        '{"tool": "opendeck_music", "arguments": {"action": "play"}}'
    )
    assert grammar.envelope_is_well_formed(
        '[{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {"x": 1}}]'
    )


def test_envelope_check_refuses_malformed_or_extra_shaped_calls():
    assert not grammar.envelope_is_well_formed("not json at all")
    assert not grammar.envelope_is_well_formed(
        '<tool_call>{"tool": "a"}</tool_call>'
    ), "arguments are required"
    assert not grammar.envelope_is_well_formed(
        '<tool_call>{"tool": "a", "arguments": {}, "execute": true}</tool_call>'
    ), "the envelope has no room for an undeclared key"
    assert not grammar.envelope_is_well_formed(
        '<tool_call>{"tool": "", "arguments": {}}</tool_call>'
    )
    assert not grammar.envelope_is_well_formed("<tool_call>{}</tool_call>")
    assert not grammar.envelope_is_well_formed(
        '<tool_call>{"tool": "a", "arguments": {}, "arguments2": {}}</tool_call>'
    )


def test_an_installed_engine_is_used_and_a_missing_one_refused(monkeypatch):
    calls: list[str] = []

    def fake_compiler(*args, **kwargs):
        calls.append("compile")

        class _Compiled:
            pass

        return _Compiled()

    class _TokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer):
            return "info"

    class _GrammarCompiler:
        def __init__(self, info):
            assert info == "info"

        def compile_json_schema(self, schema):
            calls.append(schema)
            return fake_compiler()

    class _LogitsProcessor:
        def __init__(self, compiled):
            self.compiled = compiled

    xgrammar = types.ModuleType("xgrammar")
    xgrammar.TokenizerInfo = _TokenizerInfo
    xgrammar.GrammarCompiler = _GrammarCompiler
    xgrammar.contrib = types.SimpleNamespace(hf=types.SimpleNamespace(LogitsProcessor=_LogitsProcessor))
    monkeypatch.setitem(sys.modules, "xgrammar", xgrammar)

    processor = grammar.grammar_logits_processor(
        tokenizer=object(), prompt_length=0, engine="xgrammar"
    )
    assert processor is not None
    assert calls, "the schema was compiled"

    try:
        grammar.grammar_logits_processor(tokenizer=object(), prompt_length=0, engine="nope")
    except ValueError as exc:
        assert "unknown grammar engine" in str(exc)
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("an unknown engine must be refused, not ignored")
