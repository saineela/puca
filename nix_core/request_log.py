"""
Persistent request logging for the Nix core.

Every request that flows through Brain.handle() is appended as one
JSON line to logs/requests-YYYY-MM-DD.jsonl (daily files). The log is
the source of truth for post-hoc analysis: routing decisions, model
usage, latencies, replies, and full pipeline payloads.

Disable with NIX_REQUEST_LOG=0. Files never rotate automatically
beyond the daily split; analyze_logs.py summarizes them.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

_LOG_DIR = os.environ.get(
    "NIX_REQUEST_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
)
_ENABLED = os.environ.get("NIX_REQUEST_LOG", "1") != "0"
_LOCK = threading.Lock()


def log_request(entry: dict[str, Any]) -> None:
    """Append one request record as a JSON line. Never raises: logging
    must not take the assistant down."""
    if not _ENABLED:
        return
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(_LOG_DIR, f"requests-{day}.jsonl")
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass


def log_dir() -> str:
    return _LOG_DIR


def make_entry(
    *,
    request: str,
    location: str,
    route: str,
    rule: str | None,
    reply: str,
    latency_ms: float,
    details: dict[str, Any] | None = None,
    clauses: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Uniform record shape so analyze_logs.py can rely on fields."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request": request,
        "location": location,
        "route": route,
        "rule": rule,
        "reply": (reply or "")[:2000],
        "latency_ms": round(latency_ms, 1),
        "clauses": clauses or [],
        "error": error,
        "details": _compact(details or {}),
    }


def _compact(details: dict[str, Any], limit: int = 4000) -> dict[str, Any]:
    """Keep payloads informative but bounded."""
    try:
        text = json.dumps(details, ensure_ascii=False, default=str)
        if len(text) <= limit:
            return details
        return {"_truncated": text[:limit]}
    except Exception:
        return {"_unserializable": str(details)[:limit]}
