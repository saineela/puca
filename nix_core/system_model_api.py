"""Read-only API for the PUCA system model v3 decisions.

These routes expose measurement state, never personal prompts: inference
metrics, the privacy boundary's published coverage and retention rules, and the
default-off proactivity policy. They are deliberately read-only so a page (or a
test) can inspect the system without changing it.
"""
from __future__ import annotations

import json
from typing import Any

import music_api
import voice
from eval import context_composer, inference_metrics, privacy, proactivity, turn_taking

DECISIONS = {
    "D1": "turn-level provenance and superseded_by",
    "D2": "memory evaluation: development, held-out and adversarial sets",
    "D3": "deterministic context composer with five section budgets",
    "D4": "routing ledger with counterfactual and separate error rates",
    "D5": "five independent plan validations",
    "D6": "paired reliability gate with safety veto",
    "D7": "voice turn-taking: VAD, endpointing and barge-in",
    "D8": "proactivity off by default, with tiers, quiet hours and a budget",
    "D9": "fail-closed pre-model redaction boundary",
    "D10": "inference measurements and immutable-prefix cache",
}


def _respond(handler, payload: dict[str, Any]) -> bool:
    return music_api._respond(handler, {"ok": True, **payload})


def system_model_index(handler, _match=None) -> bool:
    return _respond(handler, {
        "version": 3,
        "decisions": DECISIONS,
        "endpoints": [
            "/api/system-model",
            "/api/system-model/inference",
            "/api/system-model/privacy",
            "/api/system-model/proactivity",
            "/api/system-model/context?text=",
            "/api/system-model/turn-taking",
        ],
    })


def inference_metrics_view(handler, _match=None) -> bool:
    query = music_api._query(handler)
    try:
        limit = int((query.get("limit") or ["50"])[0])
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 500))
    return _respond(handler, {
        "summary": inference_metrics.summary(),
        "recent": inference_metrics.recent(limit),
        "cache": {
            "hit_rate": inference_metrics.PREFIX_CACHE.hit_rate,
            "hits": inference_metrics.PREFIX_CACHE.hits,
            "misses": inference_metrics.PREFIX_CACHE.misses,
        },
    })


def privacy_view(handler, _match=None) -> bool:
    return _respond(handler, {
        "suite": sorted(privacy.PATTERNS),
        "retention": privacy.retention_report(),
        "claim": (
            "Blocks the sensitive values this published suite detects on every "
            "outbound path; a value outside the suite can pass. Redaction that "
            "cannot complete blocks a non-local send."
        ),
    })


def proactivity_view(handler, _match=None) -> bool:
    policy = proactivity.default_policy()
    return _respond(handler, {
        "policy": policy.as_dict(),
        "registry": list(policy.registry),
        "note": "No notification and no side effect while the policy is off.",
    })


def turn_taking_view(handler, _match=None) -> bool:
    """D7: the thresholds, and the last clip's measured turn-taking."""
    return _respond(handler, {
        "policy": {
            "endpoint_ms": turn_taking.TurnTaker().endpoint_ms,
            "frame_ms": voice.VAD_FRAME_MS,
            "silence_rms": voice.SILENCE_RMS,
            "target_p95_ms": 300.0,
        },
        "vad": "energy VAD over 20 ms frames, thresholded above the clip's own quiet tenth",
        "last_clip": voice.last_turn_taking(),
        "barge_in": (
            "Measured where assistant audio and microphone frames meet; a single "
            "upload records none and says so rather than reporting zero."
        ),
        "endpoints": [
            "POST /api/voice/transcribe returns turn_taking for the clip it read",
        ],
    })


def context_view(handler, _match=None) -> bool:
    query = music_api._query(handler)
    text = str((query.get("text") or [""])[0])[:2000]
    composed = context_composer.compose(
        policy="You are a warm, honest companion. " * 4,
        memory=[context_composer.SectionItem(text, key="request")],
        question=text,
    )
    return _respond(handler, {
        "sections": composed.sizes,
        "dropped": composed.dropped,
        "budgets": composed.budgets,
        "provenance": context_composer.BUDGET_PROVENANCE,
    })
