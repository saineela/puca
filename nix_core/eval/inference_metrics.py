"""D10 - report TTFT, decode time and load time separately, per model call.

A single "latency" number hides which part moved. Every call records:

* ``load_ms`` - constructing/loading the model, if it happened in this call;
* ``ttft_ms`` - request start to the first generated token;
* ``decode_ms`` - first token to the final token;
* ``token_count`` - output tokens, so decode can be read per token.

The prefix cache is only allowed to hold immutable text, keyed by model,
tokenizer and policy version. Personal memory must never enter the key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any, Iterable, Mapping

_METRICS_LOCK = threading.Lock()
_RECORDS: list["CallMetrics"] = []
_MAX_RECORDS = 2_000


@dataclass(frozen=True)
class CallMetrics:
    model: str
    backend: str
    load_ms: float
    ttft_ms: float | None
    decode_ms: float | None
    token_count: int
    cache_hit: bool | None = None
    prefix_key: str = ""
    recorded_at: float = field(default_factory=time.time)

    @property
    def decode_ms_per_token(self) -> float | None:
        if self.decode_ms is None or self.token_count <= 0:
            return None
        return round(self.decode_ms / self.token_count, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "load_ms": round(self.load_ms, 3),
            "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 3),
            "decode_ms": None if self.decode_ms is None else round(self.decode_ms, 3),
            "decode_ms_per_token": self.decode_ms_per_token,
            "token_count": self.token_count,
            "cache_hit": self.cache_hit,
            "prefix_key": self.prefix_key,
            "recorded_at": self.recorded_at,
        }


def record(metrics: CallMetrics) -> None:
    with _METRICS_LOCK:
        _RECORDS.append(metrics)
        del _RECORDS[:-_MAX_RECORDS]


def recent(limit: int = 100) -> list[dict[str, Any]]:
    with _METRICS_LOCK:
        return [item.as_dict() for item in _RECORDS[-limit:]]


def summary() -> dict[str, Any]:
    rows = recent(limit=_MAX_RECORDS)
    if not rows:
        return {"calls": 0, "ttft_ms": None, "decode_ms": None, "load_ms": None}
    def mean(values: Iterable[float]) -> float | None:
        values = list(values)
        return round(sum(values) / len(values), 3) if values else None
    return {
        "calls": len(rows),
        "ttft_ms": mean(row["ttft_ms"] for row in rows if row["ttft_ms"] is not None),
        "decode_ms": mean(row["decode_ms"] for row in rows if row["decode_ms"] is not None),
        "load_ms": mean(row["load_ms"] for row in rows),
        "cache_hit_rate": mean(
            1.0 if row["cache_hit"] else 0.0
            for row in rows if row["cache_hit"] is not None
        ),
        "coverage": {
            "ttft": sum(1 for row in rows if row["ttft_ms"] is not None) / len(rows),
            "decode": sum(1 for row in rows if row["decode_ms"] is not None) / len(rows),
            "load": 1.0,
        },
    }


@dataclass(frozen=True)
class ImmutablePrefix:
    """The only text a prefix cache may key on: policy, not personal memory."""

    model: str
    tokenizer: str
    policy_version: str
    policy_text: str

    def __post_init__(self) -> None:
        if not self.policy_text.strip():
            raise ValueError("an immutable prefix needs policy text")
        if _contains_personal_memory(self.policy_text):
            raise ValueError("personal memory may not enter a cache key")

    @property
    def key(self) -> str:
        digest = hashlib.sha256(
            "\x1f".join(
                (self.model, self.tokenizer, self.policy_version, self.policy_text)
            ).encode("utf-8")
        ).hexdigest()
        return f"prefix:{self.policy_version}:{digest[:24]}"


_PERSONAL_MARKERS = (
    "MEMORY (",
    "How your people are doing",
    "Recent history",
    "preferred name",
    "kind:",
    "person_id:",
)


def _contains_personal_memory(text: str) -> bool:
    folded = (text or "").casefold()
    return any(marker.casefold() in folded for marker in _PERSONAL_MARKERS)


class PrefixCache:
    """A tiny in-process cache keyed only by an immutable prefix."""

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, prefix: ImmutablePrefix) -> Any | None:
        if prefix.key in self._entries:
            self.hits += 1
            return self._entries[prefix.key]
        self.misses += 1
        return None

    def put(self, prefix: ImmutablePrefix, value: Any) -> None:
        self._entries[prefix.key] = value

    @property
    def hit_rate(self) -> float | None:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else None


PREFIX_CACHE = PrefixCache()
