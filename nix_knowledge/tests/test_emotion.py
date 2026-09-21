"""Regression tests for conservative emotional attunement signals."""
from nix_knowledge.emotion import analyze_emotion


def test_hard_conversation_is_detected_as_high_priority_distress():
    result = analyze_emotion("I feel hopeless and I can't cope anymore")
    assert result["emotion"] == "distress"
    assert result["urgent"] is True
    assert result["crisis"] is False
    assert result["certain"] is True


def test_explicit_crisis_language_is_not_downgraded_to_generic_sadness():
    result = analyze_emotion("I want to die and hurt myself")
    assert result["emotion"] == "crisis"
    assert result["crisis"] is True
    assert result["urgent"] is True


def test_emotional_person_reference_is_preserved_for_careful_followup():
    result = analyze_emotion("I'm terrified because my sister is in hospital")
    assert result["emotion"] == "panic"
    assert result["care_subject"] == "my sister"


def test_ambiguous_emotion_abstains_instead_of_mind_reading():
    result = analyze_emotion("The meeting is tomorrow")
    assert result["certain"] is False
    assert result["emotion"] == "unknown"
    assert result["urgent"] is False
