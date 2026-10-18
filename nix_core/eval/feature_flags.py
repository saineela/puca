"""Phased rollout flags for the approved Luna plan (§8.2 of the proposal).

Every phase lands behind ``NIX_<PHASE>_MODE = off | shadow | on``:

* ``off`` - the old path runs, the new one is not computed;
* ``shadow`` - the new path computes and logs, the reply is unchanged;
* ``on`` - the new path is authoritative.

Promotion from ``shadow`` to ``on`` is not a manual edit: it is the output of
``eval.acceptance.decide_feature``. This module only reads the mode and applies
a gate decision; it never decides one.
"""
from __future__ import annotations

import os
from typing import Any

MODES = ("off", "shadow", "on")

#: The phases named in the proposal's migration order.
PHASES = (
    "answer_abstention",
    "execution_verification",
    "semantic_routing",
    "typed_workspace",
    "heldout_evaluation",
    "memory_consolidation",
    "habit_caching",
    "idle_processing",
)


def env_name(phase: str) -> str:
    """The environment variable that carries one phase's mode."""
    return f"NIX_{phase.upper()}_MODE"


def mode(phase: str, *, environ: dict[str, str] | None = None) -> str:
    """The current mode of a phase, validated. Unknown values are refused."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    raw = str((environ if environ is not None else os.environ).get(env_name(phase), "off")).strip().lower()
    if raw not in MODES:
        raise ValueError(f"{env_name(phase)} must be one of {', '.join(MODES)}; got {raw!r}")
    return raw


def all_modes(*, environ: dict[str, str] | None = None) -> dict[str, str]:
    return {phase: mode(phase, environ=environ) for phase in PHASES}


def is_shadow(phase: str, *, environ: dict[str, str] | None = None) -> bool:
    """True when the phase computes and logs but does not change the reply."""
    return mode(phase, environ=environ) == "shadow"


def promoted_mode(phase: str, gate: dict[str, Any], *, environ: dict[str, str] | None = None) -> str:
    """The mode a phase may have after its promotion gate ran.

    The gate is ``eval.acceptance.decide_feature`` output. A phase that has not
    passed stays exactly where it was — a shadow phase stays in shadow, and an
    off phase stays off. ``promote`` is the only action that reaches ``on``.
    """
    current = mode(phase, environ=environ)
    if gate.get("action") == "promote" and gate.get("decision") == "pass":
        return "on"
    if gate.get("action") == "rollback_to_off":
        return "off"
    return current
