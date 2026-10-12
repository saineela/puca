"""
Tests for the spell-correction preprocessing layer
(nix_knowledge.spelling): the general edit-distance layer that
replaced the hand-maintained ASR typo map.

Contract under test:

* Confident single-token typos in ASR transcripts are fixed.
* Healthy text, domain vocabulary, names, acronyms, shorthand, and
  contractions pass through byte-identical.
* Correction is deterministic and cheap (<10 ms per sentence).
"""

from __future__ import annotations

import pytest

from nix_knowledge.spelling import correct_spelling


# ----------------------------------------------------------------------
# Corrections: the open-ended typo space the old map could never cover
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The exact failures that motivated the layer
        ("cancle my dentist appointment", "cancel my dentist appointment"),
        ("my apointment got cancled", "my appointment got cancelled"),
        ("rememer to call grandma", "remember to call grandma"),
        ("scheduel robotics practice", "schedule robotics practice"),
        ("tomorow at 6:30am remind me to stretch",
         "tomorrow at 6:30am remind me to stretch"),
        ("i have robotics practice tomorow at 5pm",
         "i have robotics practice tomorrow at 5pm"),
        ("wats on my calendar tomorrow", "what's on my calendar tomorrow"),
        # Transposition: needs the frequency prior, not just d=1
        ("htat is great", "that is great"),
        # Case is preserved
        ("Tomorow we leave", "Tomorrow we leave"),
    ],
)
def test_typos_are_corrected(raw: str, expected: str) -> None:
    assert correct_spelling(raw) == expected


# ----------------------------------------------------------------------
# Guards: what must never be touched
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        # The production cancellation pair must pass through untouched
        ("my robotics practice has been cancelled"),
        ("robotics practice at 5pm tomorrow"),
        # Question openers keep their auxiliary
        ("was my robotics practice cancelled?"),
        # Acronyms / domain vocabulary
        ("TSA precheck at the airport friday"),
        ("remind me to file the fafsa on 11/15"),
        ("remind me in 2 hours to check the oven"),
        # Names and rare words must not snap to lookalikes
        ("meet Kayla and Maanvi at the park"),
        ("my grandma makes the best pancakes"),
        ("I am in the robotics club at school"),
        # Shorthand and slang (lead-in stripper depends on these)
        ("alright bro, I have a TSA meeting tmr from 4pm to 6pm"),
        ("cancel my piano lessons"),
        ("don't forget my meds at 9pm"),
        ("what's on my calendar for next week"),
        # Numbers, times, punctuation, whitespace
        ("123 456 at 7:30"),
        (""),
        ("   "),
    ],
)
def test_healthy_text_is_untouched(raw: str) -> None:
    assert correct_spelling(raw) == raw


def test_beyond_budget_is_passed_through() -> None:
    # Three edits away from anything: honest pass-through beats a guess
    assert correct_spelling("scduedle a playdate") == "scduedle a playdate"


# ----------------------------------------------------------------------
# Non-functional contract
# ----------------------------------------------------------------------


def test_correction_is_deterministic() -> None:
    sample = "cancle my apointment tomorow at 5pm pleas"
    assert correct_spelling(sample) == correct_spelling(sample)


def test_correction_is_cheap() -> None:
    import time

    sentence = "cancle my apointment with the orthodontist tomorow at 5pm"
    start = time.perf_counter()
    for _ in range(50):
        correct_spelling(sentence)
    per_call_ms = (time.perf_counter() - start) / 50 * 1000
    assert per_call_ms < 10, f"spell layer too slow: {per_call_ms:.2f} ms"
