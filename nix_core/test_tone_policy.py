"""Core emotional-attunement policy tests."""
import brain as brain_module
from brain import Brain, suppress_nonessential_questions
from tone_policy import conversation_policy


class _Knowledge:
    def intent(self, text):
        return {"ok": False}

    def memory_block(self, current_text=None):
        return ""

    def digest(self):
        return ""


class _Actions:
    def context(self):
        return []


class _Chat:
    model = "test"


def test_core_prompt_contains_fatigue_guardrail(monkeypatch):
    monkeypatch.setattr(
        brain_module,
        "_active_assistant_identity",
        lambda: ("Casper", "PUCA (Personal User Companion Agent)", "test-casper"),
    )
    brain = Brain(knowledge=_Knowledge(), actions=_Actions(), ollama=_Chat(), log_requests=False)
    prompt = brain._chat_system_prompt("I'm exhausted after a very long day")
    assert "Detected interaction state: tired" in prompt
    assert "Do not ask nonessential follow-up questions" in prompt
    assert "not like customer support" in prompt
    assert "MEMORY-PLUS-CHAT RULE" in prompt
    assert "already answered that follow-up" in prompt



def test_generated_tired_follow_up_is_removed_after_model_output():
    reply = suppress_nonessential_questions(
        "I'm sorry to hear that. Do you want to talk about what's going on?",
        avoid=True,
    )
    assert reply == "I'm sorry to hear that"


def test_required_question_is_not_removed_when_policy_allows_questions():
    reply = suppress_nonessential_questions(
        "Which sister do you mean?",
        avoid=False,
    )
    assert reply == "Which sister do you mean?"


def test_tired_user_is_not_given_nonessential_social_questions():
    policy = conversation_policy("I'm exhausted after a very long day")
    assert policy["state"] == "tired"
    assert policy["avoid_nonessential_questions"] is True
    assert "Do not ask nonessential" in policy["instruction"]


def test_unwell_user_gets_gentle_high_priority_guidance():
    policy = conversation_policy(
        "I feel sick and I have a fever",
        {"emotion": "sadness", "valence": "bad"},
    )
    assert policy["state"] == "unwell_or_distressed"
    assert policy["priority"] == "high"
    assert policy["avoid_nonessential_questions"] is True


def test_positive_user_can_receive_relevant_personal_follow_up():
    policy = conversation_policy(
        "I'm so happy, I got great news about my sister!",
        {"emotion": "joy", "valence": "good"},
    )
    assert policy["state"] == "positive"
    assert policy["allow_relevant_positive_follow_up"] is True
    assert policy["avoid_nonessential_questions"] is False


def test_ambiguous_tone_stays_neutral():
    policy = conversation_policy("The appointment is at three")
    assert policy["state"] == "neutral"
    assert policy["avoid_nonessential_questions"] is False
