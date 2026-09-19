"""
Mood sounds for Nix.

Nix expresses its mood through short synthesized sounds at the moment
a reply lands in the console:

    good news  -> a rising trumpet fanfare (celebration)
    bad news   -> a soft descending two-tone (sympathy, not sad-trombone)
    neutral    -> a subtle single blip (acknowledgment)

Sounds are synthesized with the Web Audio API and played in the
browser - no audio files, no server round-trip. This module only
decides WHICH sound fits the situation; the mapping is deterministic
so the neural mood read (nix_knowledge /intent) can be wrong without
ever producing a mismatched sound (a wrong valence still lands on one
of the three sane choices).

Python decides, JavaScript plays: /api/mood returns the sound id and
the console runs the corresponding Web Audio snippet.
"""

from __future__ import annotations

from typing import Any

SOUND_IDS = ("trumpet", "sympathy", "blip")

# Emotions that always map to celebration, regardless of valence read
_CELEBRATION_EMOTIONS = {
    "joy", "excitement", "amusement", "gratitude", "pride", "relief",
    "optimism", "love",
}

# Emotions that always map to the gentle sympathy tone
_SYMPATHY_EMOTIONS = {
    "sadness", "grief", "disappointment", "fear", "remorse",
    "caring", "worry", "nervousness",
}


def mood_sound(
    valence: str | None,
    emotion: str | None = None,
) -> str:
    """
    Decide the mood sound id.

    valence: 'good' | 'bad' | 'neutral' (from the neural /intent read
    or the state engine). emotion: optional GoEmotions label that
    overrides borderline valences.
    """
    emotion_low = (emotion or "").lower()

    if emotion_low in _CELEBRATION_EMOTIONS:
        return "trumpet"
    if emotion_low in _SYMPATHY_EMOTIONS:
        return "sympathy"

    if valence == "good":
        return "trumpet"
    if valence == "bad":
        return "sympathy"
    return "blip"


def mood_from_response(response: dict[str, Any]) -> str | None:
    """
    Extract the mood sound id from a Brain.handle() response.

    Returns None when no mood information is present (chat route,
    intent unavailable), so the console stays silent.
    """
    mood = response.get("mood")
    if not isinstance(mood, dict) or not mood:
        return None
    sound = mood_sound(
        mood.get("valence"),
        mood.get("emotion"),
    )
    return sound if sound in SOUND_IDS else None
