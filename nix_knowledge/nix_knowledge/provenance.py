"""D1 - turn-level provenance helpers.

A fact records *where it came from*, never *whether to believe it*. The
reference is opaque to Knowledge: any consumer can walk ``record -> change ->
source turn`` without this package knowing how conversations are stored.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_TURN = re.compile(r"^[A-Za-z0-9_.:@/-]{1,180}$")


class ProvenanceError(ValueError):
    """A turn reference is missing or malformed."""


@dataclass(frozen=True)
class TurnReference:
    conversation_id: str
    message_id: str
    event_id: str = ""

    def __post_init__(self) -> None:
        for name in ("conversation_id", "message_id", "event_id"):
            value = getattr(self, name)
            if value and not _TURN.match(str(value)):
                raise ProvenanceError(f"invalid {name} in a turn reference")

    @property
    def source_id(self) -> str:
        parts = [f"conversation:{self.conversation_id}", f"message:{self.message_id}"]
        if self.event_id:
            parts.append(f"event:{self.event_id}")
        return "|".join(parts)

    @classmethod
    def parse(cls, source_id: str | None) -> "TurnReference":
        if not source_id:
            raise ProvenanceError("a turn reference needs a source_id")
        fields: dict[str, str] = {}
        for part in str(source_id).split("|"):
            key, _, value = part.partition(":")
            if key in {"conversation", "message", "event"} and value:
                fields[key] = value
        if "conversation" not in fields or "message" not in fields:
            raise ProvenanceError("source_id is not a conversation turn reference")
        return cls(
            conversation_id=fields["conversation"],
            message_id=fields["message"],
            event_id=fields.get("event", ""),
        )


def require_turn_reference(source: str, source_id: str | None) -> str:
    """Conversation writes must carry a turn reference; system writes may not."""
    if source in {"conversation", "chat", "dashboard", "openai", "voice"}:
        if not source_id:
            raise ProvenanceError("a conversation write must name its source turn")
        TurnReference.parse(source_id)
    return source_id or ""
