"""Short-lived voice follow-up sessions.

This state is deliberately ephemeral and per connection. Durable conversation
turns still go to nix_actions, and durable personal facts still go to
nix_knowledge. The session only decides whether a continuation may omit the
wake word and which recent turns should be replayed to Core's chat model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class FollowUpSession:
    """Bounded active conversation window for one authenticated client."""

    timeout_seconds: float = 12.0
    max_turns: int = 12
    max_chars: int = 6000
    conversation_id: str = ""
    active: bool = False
    expires_at: datetime | None = None
    _turns: list[dict[str, str]] = field(default_factory=list)

    def start(self, *, conversation_id: str | None = None, now: datetime | None = None) -> None:
        moment = now or datetime.now(timezone.utc)
        if conversation_id:
            self.conversation_id = conversation_id
        self.active = True
        self.expires_at = moment + timedelta(seconds=self.timeout_seconds)

    def expired(self, *, now: datetime | None = None) -> bool:
        if not self.active or self.expires_at is None:
            return True
        return (now or datetime.now(timezone.utc)) >= self.expires_at

    def can_continue(self, *, now: datetime | None = None) -> bool:
        if self.expired(now=now):
            self.expire()
            return False
        return True

    def touch(self, *, now: datetime | None = None) -> None:
        moment = now or datetime.now(timezone.utc)
        self.expires_at = moment + timedelta(seconds=self.timeout_seconds)
        self.active = True

    def record(self, role: str, content: str, *, now: datetime | None = None) -> None:
        if role not in {"user", "assistant"} or not content:
            return
        self._turns.append({"role": role, "content": " ".join(content.split())})
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
        while sum(len(turn["content"]) for turn in self._turns) > self.max_chars:
            if len(self._turns) <= 1:
                self._turns[0]["content"] = self._turns[0]["content"][-self.max_chars :]
                break
            self._turns.pop(0)
        self.touch(now=now)

    def context(self) -> list[dict[str, str]]:
        return [dict(turn) for turn in self._turns]

    def expire(self) -> None:
        self.active = False
        self.expires_at = None
        self._turns.clear()

    def end(self) -> None:
        self.expire()

    def metadata(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id or None,
            "active": self.active and not self.expired(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "turns": len(self._turns),
        }
