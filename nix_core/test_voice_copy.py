"""The pipeline's own replies are written the way a person would say them.

This is the half of a turn that Core writes for Luna. The tests hold two
properties: the engineering vocabulary never reaches the user, and ordinary
conversation is left completely alone.
"""
from __future__ import annotations

import pytest

from voice_copy import (
    KNOWLEDGE_EMPTY,
    KNOWLEDGE_FAILED,
    NO_EVENTS,
    NO_STATES,
    humanize_reply,
)


@pytest.mark.parametrize(
    "jargon",
    [
        "I didn't send a device command because the dedicated skill planner "
        "did not produce a valid, safe plan.",
        "I didn't send any device action because I couldn't validate the "
        "complete plan (TypeError: boom).",
        "I didn't send any device action because I couldn't validate every "
        "requested action as one complete plan.",
        "I didn't send a device command because the model didn't return a "
        "valid skill call.",
        "I didn't send a device command because I couldn't confirm one clear "
        "color choice for that request.",
        "Studio Ring isn't ready yet. Complete its setup in Skills, then "
        "review and explicitly trust the installed package.",
        "Studio Ring is installed but untrusted. Review its source and "
        "permissions in Skills before explicitly trusting it.",
        "The targeted skill isn't currently ready or trusted, so I didn't "
        "send a device command.",
    ],
)
def test_no_engineering_vocabulary_reaches_the_user(jargon):
    said = humanize_reply(jargon)

    for word in (
        "planner", "valid, safe plan", "untrusted", "explicitly trust",
        "installed package", "skill call", "device command",
    ):
        assert word not in said.lower()
    assert said.endswith("?") or said.endswith(".")


def test_a_planner_refusal_says_what_did_not_happen_and_offers_a_next_step():
    said = humanize_reply(
        "I didn't send a device command because the dedicated skill planner "
        "did not produce a valid, safe plan."
    )

    assert "nothing changed" in said
    assert "try again" in said


def test_the_knowledge_failure_is_warm_and_never_claims_success():
    assert "nothing was saved or changed" in KNOWLEDGE_FAILED
    assert humanize_reply(
        "I couldn't complete that request, so nothing was saved or changed. "
        "Could you phrase it another way?"
    ) == KNOWLEDGE_FAILED
    # Short enough to say, and still offers the retry.
    assert len(KNOWLEDGE_FAILED.split()) <= 20
    assert KNOWLEDGE_FAILED.endswith("?")


def test_an_empty_answer_invites_the_fact_instead_of_reciting_a_store():
    assert humanize_reply("I don't have that in my knowledge base yet.") == KNOWLEDGE_EMPTY
    assert "knowledge base" not in KNOWLEDGE_EMPTY
    assert humanize_reply("No current states stored yet.") == NO_STATES
    assert humanize_reply("No matching events found.") == NO_EVENTS
    # A miss that offers nothing is a dead end, so it asks instead.
    assert "?" in NO_EVENTS
    assert len(NO_EVENTS.split()) <= 20


def test_a_skill_name_survives_the_rewrite():
    said = humanize_reply(
        "OpenDeck is installed but untrusted. Review its source and "
        "permissions in Skills before explicitly trusting it."
    )

    assert said.startswith("OpenDeck")
    assert "trust it in Skills" in said


def test_ordinary_conversation_is_untouched():
    for line in (
        "That's a big move — what made you decide?",
        "I'm listening to music.",
        "Your sister's name is Priya.",
        "The ring light is blue now.",
        "I can't take instructions that try to replace my own rules, so "
        "nothing has changed.",
    ):
        assert humanize_reply(line) == line


def test_the_rewrite_is_idempotent():
    once = humanize_reply(
        "I didn't send a device command because the dedicated skill planner "
        "did not produce a valid, safe plan."
    )
    assert humanize_reply(once) == once


def test_a_model_outage_is_not_reported_as_a_system_alert():
    """A person does not hear the name of an error class.

    Measured live (GPU busy): "Luna Alert: Unable to reach the chat model
    (luna-v6-contextual-v1) right now. (AcceleratorError)".
    """
    said = humanize_reply(
        "Luna Alert: Unable to reach the chat model "
        "(luna-v6-contextual-v1) right now. (AcceleratorError)"
    )

    assert "Alert" not in said
    assert "AcceleratorError" not in said
    assert "luna-v6-contextual-v1" not in said
    # It has to ask the user to repeat, even when it ends with a casual tag.
    assert "could you repeat" in said
    assert "?" in said


def test_a_clinical_i_dont_know_is_rephrased():
    """The user's own example of a reply that does not sound like a person."""
    assert humanize_reply(
        "I don't have any information about your mother's current state."
    ) == "I don't know how your mother is doing right now — tell me?"
    assert humanize_reply(
        "I don't have any information about your mother current state."
    ) == "I don't know how your mother is doing right now — tell me?"
    assert humanize_reply("I don't have any information on that.") == (
        "I don't know that one — tell me?"
    )


def test_the_planner_failure_site_never_holds_engineering_copy():
    """The reply string itself is human, so no path can leak the old line.

    Comments and docstrings may still quote the old wording to explain the
    history; what must not exist is a REPLY built from it.
    """
    import inspect
    import brain

    source = inspect.getsource(brain)
    assert "I didn't send a device command because the dedicated skill planner" not in source
    assert "couldn't get that to the device this time" in source
