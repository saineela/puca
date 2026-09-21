from brain import suppress_generic_interview
from config import ASSISTANT_NAME, ASSISTANT_ROLE


def test_casper_identity_defaults():
    assert ASSISTANT_NAME == "Casper"
    assert ASSISTANT_ROLE == "PUCA (Personal User Companion Agent)"


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
