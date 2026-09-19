from __future__ import annotations

import os


def clear_terminal():

    os.system(
        "cls"
        if os.name == "nt"
        else "clear"
    )


def render(engine):

    clear_terminal()

    now = engine.clock.now()

    decision = engine.current_decision

    world = engine.world

    print(
        "┌──────────────────────────────────────────────────────────────┐"
    )

    print(
        "│                       NIX DECISION                          │"
    )

    print(
        "│                    Cognitive Engine                         │"
    )

    print(
        "├──────────────────────────────────────────────────────────────┤"
    )

    print(
        f"│ TIME: {now.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'):<51}│"
    )

    print(
        "│                                                              │"
    )

    print(
        "│ CURRENT COGNITIVE STATE                                     │"
    )

    print(
        f"│ State:       {str(decision.state if decision else 'INITIALIZING'):<42} │"
    )

    print(
        f"│ Action:      {str(decision.action if decision else 'NONE'):<42} │"
    )

    print(
        f"│ Attention:   {str(decision.attention if decision else 'LOW'):<42} │"
    )

    print(
        "│                                                              │"
    )

    print(
        "│ WORLD STATE                                                  │"
    )

    print(
        f"│ Phone:       {str(world.phone_location.value):<42} │"
    )

    print(
        f"│ Phone conf:  {world.phone_location.confidence:<42.2f} │"
    )

    print(
        f"│ Room:        {str(world.presence.room):<42} │"
    )

    print(
        f"│ Presence:    {world.presence.confidence:<42.2f} │"
    )

    print(
        f"│ Calendar:    {len(world.calendar_events)} event(s){'':<31} │"
    )

    print(
        f"│ History:     {len(world.recent_events)} observations{'':<24} │"
    )

    print(
        "│                                                              │"
    )

    print(
        "│ REASON                                                       │"
    )

    reason = (
        decision.reason
        if decision
        else "Waiting for observations."
    )

    # Dashboard should never destroy the terminal layout.
    reason = reason[:58]

    print(
        f"│ {reason:<58} │"
    )

    print(
        "└──────────────────────────────────────────────────────────────┘"
    )
