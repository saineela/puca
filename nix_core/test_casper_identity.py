from brain import (
    low_stakes_preference_reply,
    social_companion_reply,
    strip_model_control_traces,
    suppress_generic_interview,
)
from config import ASSISTANT_NAME, ASSISTANT_ROLE


def test_casper_identity_defaults():
    assert ASSISTANT_NAME == "Casper"
    assert ASSISTANT_ROLE == "PUCA (Personal User Companion Agent)"


def test_model_control_traces_are_removed():
    assert strip_model_control_traces("<think>private</think> Hello.") == "Hello."
    assert strip_model_control_traces("user\\nHi\\nassistant\\n<think></think>Hey.") == "user\\nHi\\nassistant\\nHey."


def test_low_stakes_food_preference_engages_without_claiming_real_taste():
    prompt = "do you prefer SpongeBob popsicles or strawberry shortcake popsicles?"
    assert low_stakes_preference_reply(prompt) == (
        "I’d pick strawberry shortcake popsicles; that one sounds especially good."
    )
    assert "strawberry" in low_stakes_preference_reply(
        prompt, "Preferences: user is trying to eat healthier"
    ).lower()


def test_alive_check_is_a_short_human_social_reply():
    assert social_companion_reply("hello? are you alive") == "Yeah, I’m here."
    assert social_companion_reply("hey, you there?") == "Yeah, I’m here."


def test_generic_interview_endings_are_removed():
    assert suppress_generic_interview("Nice work. What's on your mind?") == "Nice work."
    assert suppress_generic_interview("Got it. How can I help?") == "Got it."
    assert suppress_generic_interview("That sounds hard. Anything else?") == "That sounds hard."


def test_real_task_question_is_not_removed():
    reply = "Which sister do you mean: Jane or Maanvi?"
    assert suppress_generic_interview(reply) == reply


def test_casper_identity_is_not_replaced_by_generic_ai_language():
    reply = "I'm Casper, a PUCA (Personal User Companion Agent)."
    assert suppress_generic_interview(reply) == reply
