from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LocationState:
    value: str | None = None
    source: str | None = None
    confidence: float = 0.0
    observed_at: datetime | None = None


@dataclass
class PresenceState:
    room: str | None = None
    source: str | None = None
    confidence: float = 0.0
    observed_at: datetime | None = None


@dataclass
class CalendarEvent:
    name: str
    start: datetime
    end: datetime | None = None


@dataclass
class WorldState:
    """
    Nix's continuously maintained representation of the world.
    """

    phone_location: LocationState = field(
        default_factory=LocationState
    )

    presence: PresenceState = field(
        default_factory=PresenceState
    )

    calendar_events: list[CalendarEvent] = field(
        default_factory=list
    )

    recent_events: list[dict[str, Any]] = field(
        default_factory=list
    )

    values: dict[str, Any] = field(
        default_factory=dict
    )

    last_observation_at: datetime | None = None

    def ingest(self, observation) -> None:
        """
        Convert an observation into persistent world state.
        """

        self.last_observation_at = observation.timestamp

        self.recent_events.append(
            {
                "kind": observation.kind,
                "source": observation.source,
                "timestamp": observation.timestamp,
                "data": dict(observation.data),
                "confidence": observation.confidence,
            }
        )

        # Keep bounded history for v0.1.
        self.recent_events = self.recent_events[-500:]

        if observation.kind == "phone.location":

            self.phone_location = LocationState(
                value=observation.data.get("location"),
                source=observation.source,
                confidence=observation.confidence,
                observed_at=observation.timestamp,
            )

        elif observation.kind == "presence.room":

            self.presence = PresenceState(
                room=observation.data.get("room"),
                source=observation.source,
                confidence=observation.confidence,
                observed_at=observation.timestamp,
            )

        elif observation.kind == "calendar.event":

            event = CalendarEvent(
                name=observation.data["name"],
                start=observation.data["start"],
                end=observation.data.get("end"),
            )

            self.calendar_events.append(event)

            self.calendar_events = self.calendar_events[-100:]

        else:
            self.values[observation.kind] = dict(
                observation.data
            )
