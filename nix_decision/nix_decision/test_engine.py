from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from .clock import SimulatedClock
from .dashboard import render
from .engine import NixDecisionEngine
from .events import Observation


def decision_changed(previous, current):

    old = (
        previous.state
        if previous
        else "START"
    )

    print()
    print(
        f"[DECISION] {old} → {current.state}"
    )

    print(
        f"Reason: {current.reason}"
    )

    print()


def main():

    parser = argparse.ArgumentParser(
        description="Nix Decision cognitive simulator"
    )

    parser.add_argument(
        "--scenario",
        default="normal-day",
        choices=[
            "normal-day",
            "late-arrival",
        ],
    )

    args = parser.parse_args()

    start = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
    )

    clock = SimulatedClock(start)

    engine = NixDecisionEngine(
        clock=clock,
        interval=2.0,
        on_decision_change=decision_changed,
    )

    # ---------------------------------------------------------
    # Initial dashboard.
    # ---------------------------------------------------------

    render(engine)

    # ---------------------------------------------------------
    # Create historical context.
    #
    # CyberPatriot ended 30 minutes ago.
    # ---------------------------------------------------------

    event_end = (
        start - timedelta(minutes=30)
    )

    engine.ingest(
        Observation(
            kind="calendar.event",
            source="simulator",
            timestamp=start,
            data={
                "name": "CyberPatriot",
                "start": (
                    start - timedelta(hours=2)
                ),
                "end": event_end,
            },
        )
    )

    # ---------------------------------------------------------
    # Normal arrival.
    # ---------------------------------------------------------

    if args.scenario == "normal-day":

        engine.ingest(
            Observation(
                kind="phone.location",
                source="simulator",
                timestamp=clock.now(),
                data={
                    "location": "home",
                },
            )
        )

        engine.ingest(
            Observation(
                kind="presence.room",
                source="simulator",
                timestamp=clock.now(),
                data={
                    "room": "bedroom",
                },
                confidence=0.95,
            )
        )

    # ---------------------------------------------------------
    # Late arrival.
    # ---------------------------------------------------------

    elif args.scenario == "late-arrival":

        clock.advance(
            2 * 60 * 60
        )

        engine.ingest(
            Observation(
                kind="phone.location",
                source="simulator",
                timestamp=clock.now(),
                data={
                    "location": "home",
                },
            )
        )

        engine.ingest(
            Observation(
                kind="presence.room",
                source="simulator",
                timestamp=clock.now(),
                data={
                    "room": "bedroom",
                },
                confidence=0.95,
            )
        )

    render(engine)

    print()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
