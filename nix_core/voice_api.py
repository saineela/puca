"""HTTP face of the console's own speech recogniser.

Two routes, and both are served by the console *and* by the panel's own origin,
because the page that records the audio is the panel and its API calls have to
stay on one origin (no cross-origin permission dance for a microphone).

* ``GET /api/voice`` — can this console hear, with what, and how long may a clip
  be? The answer is also what starts the model loading in the background, so the
  weights are in memory by the time somebody taps the microphone.
* ``POST /api/voice/transcribe`` — the recorded clip, as the request body, and
  the words come back in the reply.

The clip is raw audio, not JSON: a WAV is a few hundred kilobytes of samples and
base64 would grow it by a third for nothing. The body is read with a hard cap
(the same one ``voice`` enforces) so a page cannot make the console buffer
anything it likes.
"""
from __future__ import annotations

from typing import Any

import music_api
import voice


def voice_index(handler, _match=None) -> bool:
    """Whether the console can hear, and what it hears with."""
    voice.warm_in_background()
    return music_api._respond(handler, {
        "ok": True,
        "available": True,
        "max_audio_bytes": voice.MAX_AUDIO_BYTES,
        **voice.available(),
    })


def _read_audio(handler) -> bytes:
    raw_length = handler.headers.get("Content-Length")
    try:
        length = int(raw_length or 0)
    except (TypeError, ValueError) as exc:
        raise voice.VoiceError("Content-Length must be a number.", 400, "bad_length") from exc
    if length <= 0:
        raise voice.VoiceError("Send the recorded audio in the request body.", 400, "no_audio")
    if length > voice.MAX_AUDIO_BYTES:
        raise voice.VoiceError(
            f"That clip is larger than the {voice.MAX_AUDIO_BYTES // (1024 * 1024)} MB the console accepts.",
            413,
            "audio_too_large",
        )
    payload = handler.rfile.read(length)
    if len(payload) != length:
        raise voice.VoiceError("The clip arrived incomplete. Record it again.", 400, "audio_incomplete")
    return payload


def voice_transcribe(handler, _match=None) -> bool:
    """Recognise a recorded clip, here, and answer with the words."""
    try:
        payload = _read_audio(handler)
        result: dict[str, Any] = voice.transcribe(payload)
    except voice.VoiceError as exc:
        return music_api._respond(
            handler, {"ok": False, "code": exc.code, "error": str(exc)}, exc.status
        )
    return music_api._respond(handler, {"ok": True, **result})
