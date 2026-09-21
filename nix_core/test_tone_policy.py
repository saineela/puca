"""Core emotional-attunement policy tests."""
from brain import Brain
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


def test_core_prompt_contains_fatigue_guardrail():
    brain = Brain(knowledge=_Knowledge(), actions=_Actions(), ollama=_Chat(), log_requests=False)
    prompt = brain._chat_system_prompt("I'm exhausted after a very long day")
    assert "Detected interaction state: tired" in prompt
    assert "Do not ask nonessential follow-up questions" in prompt
    assert "not like customer support" in prompt



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
