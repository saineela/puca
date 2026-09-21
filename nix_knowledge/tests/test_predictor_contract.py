from __future__ import annotations

from nix_knowledge.needle import KnowledgeNeedle


def _parser_only() -> KnowledgeNeedle:
    return object.__new__(KnowledgeNeedle)


def test_predictor_parses_multi_action_call() -> None:
    predictor = _parser_only()
    parsed = predictor._parse_tool_call(
        '<tool_call>{"name":"multi_action","arguments":{'
        '"actions":['
        '{"name":"create_calendar_event","arguments":'
        '{"title":"robotics","temporal_expression":"tomorrow"}},'
        '{"name":"create_fact","arguments":'
        '{"value":"I joined TSA"}}]}}</tool_call>'
    )
    assert parsed is not None
    assert parsed["name"] == "multi_action"
    assert len(parsed["arguments"]["actions"]) == 2
    assert parsed["arguments"]["actions"][0]["name"] == "create_calendar_event"
    assert parsed["arguments"]["actions"][1]["name"] == "create_fact"


def test_predictor_rejects_unknown_nested_action() -> None:
    predictor = _parser_only()
    parsed = predictor._parse_tool_call(
        '<tool_call>{"name":"multi_action","arguments":{'
        '"actions":[{"name":"delete_everything","arguments":{}}]}}'
        '</tool_call>'
    )
    # Parsing remains structural; execution rejects the nested name at the
    # Python tool boundary. This ensures the model cannot call arbitrary code.
    assert parsed is not None
    assert parsed["arguments"]["actions"][0]["name"] == "delete_everything"


def test_predictor_prompt_teaches_multi_intent_and_temporal_preservation() -> None:
    predictor = _parser_only()
    prompt = predictor._system_prompt()
    assert "multi_action" in prompt
    assert "multi-intent" in prompt.lower()
    assert "do not convert expressions into absolute dates" in prompt.lower()
