"""D9 - a pre-model redaction boundary over every outbound path.

The guarantee is deliberately narrow: this blocks the sensitive values the
published suite detects. It is not a promise that every possible identifier is
caught. What it does promise is fail-closed behaviour: if redaction cannot be
completed for a non-local request, the request is not sent - not with the
original text, and not with a partial mask.

The reversible mapping is a local object. It is never included in a payload, a
log line, a cache key or a prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Callable, Iterable, Mapping

#: The published suite. A value outside these patterns can pass; that boundary
#: is part of the claim, not a hidden gap.
PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "student_id": re.compile(r"\b(?:student|student\s*id|sid)[\s:#-]*[A-Z]?\d{6,10}\b", re.IGNORECASE),
    "account": re.compile(r"\b(?:account|acct)[\s:#-]*\d{6,17}\b", re.IGNORECASE),
    "postal": re.compile(
        r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
        r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b"
    ),
    "course_code": re.compile(r"\b[A-Z]{2,4}\d{3,4}\b"),
}

REPLACEMENTS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ssn": "[SSN]",
    "student_id": "[STUDENT_ID]",
    "account": "[ACCOUNT]",
    "postal": "[ADDRESS]",
    "course_code": "[COURSE]",
    "person": "[PERSON]",
}


class RedactionBlocked(RuntimeError):
    """A non-local request cannot be sent because redaction did not complete."""


@dataclass
class RedactionResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)
    detected: list[str] = field(default_factory=list)
    completed: bool = True
    error: str = ""


class Redactor:
    """Deterministic masking with a local-only reverse mapping.

    ``on_error`` is a caller hook so a failure can be observed in a test; the
    boundary itself still refuses the send.
    """

    def __init__(self, *, on_error: Callable[[str], None] | None = None) -> None:
        self._mapping: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self.on_error = on_error
        self._counter = 0

    def _token(self, kind: str, value: str) -> str:
        if value in self._reverse:
            return self._reverse[value]
        self._counter += 1
        token = f"{REPLACEMENTS[kind]}#{self._counter}"
        self._mapping[token] = value
        self._reverse[value] = token
        return token

    def redact(self, text: str, *, names: Iterable[str] = ()) -> RedactionResult:
        try:
            if not isinstance(text, str):
                raise TypeError("redaction input must be text")
            output = text
            detected: list[str] = []
            # Names first: a person's name may appear inside an address or email
            # and the specific token must win over the generic one.
            for name in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
                if name in output:
                    output = output.replace(name, self._token("person", name))
                    detected.append("person")
            for kind, pattern in PATTERNS.items():
                def replace(match: re.Match[str], kind: str = kind) -> str:
                    detected.append(kind)
                    return self._token(kind, match.group(0))
                output = pattern.sub(replace, output)
            return RedactionResult(
                text=output,
                mapping=dict(self._mapping),
                detected=sorted(set(detected)),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:120]}"
            if self.on_error:
                self.on_error(message)
            return RedactionResult(text="", completed=False, error=message)

    def rehydrate(self, text: str) -> str:
        output = text
        for token, value in sorted(self._mapping.items(), key=lambda item: -len(item[0])):
            output = output.replace(token, value)
        return output

    def clear(self) -> None:
        self._mapping.clear()
        self._reverse.clear()
        self._counter = 0


@dataclass(frozen=True)
class Retention:
    store: str
    days: int

    def expired(self, written_at: datetime, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(timezone.utc)
        return written_at <= current - timedelta(days=self.days)


RETENTION: dict[str, Retention] = {
    "knowledge": Retention("knowledge", 3650),
    "history": Retention("history", 365),
    "request_log": Retention("request_log", 30),
    "semantic_chunk": Retention("semantic_chunk", 3650),
    "embedding": Retention("embedding", 3650),
    "backup": Retention("backup", 30),
}


def retention_report(now: datetime | None = None) -> dict[str, Any]:
    return {
        "stores": {name: rule.days for name, rule in RETENTION.items()},
        "checked_at": (now or datetime.now(timezone.utc)).isoformat(),
    }


def is_local_backend(url: str) -> bool:
    """Only loopback and private-network hosts count as local."""
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").strip("[]").lower()
    if host in {"localhost", "::1"}:
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        first, second = int(parts[0]), int(parts[1])
        return (
            first in {10, 127}
            or (first == 192 and second == 168)
            or (first == 172 and 16 <= second <= 31)
            or (first == 169 and second == 254)
        )
    return host.endswith(".local")


#: Every path text leaves by. The audit checks each one, not only the model call.
OUTBOUND_PATHS = ("model", "retry", "telemetry", "exception")


@dataclass
class OutboundBoundary:
    """The send gate: local passes, non-local must be redacted whole.

    Every path a payload can take is named on the way out (``model``, a
    ``retry``, ``telemetry``, an ``exception`` string), so the audit can check
    the bytes for each one separately. A retry is not a fresh decision: what the
    first attempt did not send, the retry must not send either.
    """

    redactor: Redactor = field(default_factory=Redactor)
    sent_payloads: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, str]] = field(default_factory=list)

    def prepare(
        self,
        *,
        url: str,
        text: str,
        names: Iterable[str] = (),
        path: str = "model",
    ) -> str:
        if path not in OUTBOUND_PATHS:
            raise ValueError(f"unknown outbound path: {path}")
        if is_local_backend(url):
            return text
        result = self.redactor.redact(text, names=names)
        if not result.completed:
            self.blocked.append({"url": url, "path": path, "error": result.error})
            raise RedactionBlocked(
                "redaction could not be completed for a non-local request"
            )
        self.sent_payloads.append(
            {
                "url": url,
                "path": path,
                "text": result.text,
                "detected": result.detected,
            }
        )
        return result.text

    def retry(
        self,
        *,
        url: str,
        text: str,
        first_attempt: str,
        names: Iterable[str] = (),
    ) -> str:
        """A retry may not send what the first attempt did not.

        Two rules, both checked here: an endpoint whose redaction could not be
        completed is never tried again, and an identical text retried after the
        first attempt must redact to the identical payload — a retry that would
        send more than the first attempt is refused rather than sent.
        """
        if any(item["url"] == url for item in self.blocked):
            raise RedactionBlocked(
                "redaction was already refused for this endpoint; a retry may not send"
            )
        retried = self.prepare(url=url, text=text, names=names, path="retry")
        if text == first_attempt and retried != first_attempt:
            raise RedactionBlocked(
                "a retry redacted to a different payload than the first attempt"
            )
        return retried

    def scrub(self, text: str, *, path: str = "exception") -> str:
        """Redact a non-payload string (telemetry or an error message).

        Unlike ``prepare`` there is no request to block here — the string is not
        going to a model — so a value that cannot be masked is removed whole
        rather than allowed through.
        """
        if path not in {"telemetry", "exception"}:
            raise ValueError(f"scrub is for telemetry and exception text, not {path}")
        result = self.redactor.redact(text)
        if not result.completed:
            self.blocked.append({"url": "", "path": path, "error": result.error})
            return ""
        self.sent_payloads.append(
            {"url": "", "path": path, "text": result.text, "detected": result.detected}
        )
        return result.text

    def audit(self) -> dict[str, Any]:
        leaked: list[str] = []
        for item in self.sent_payloads:
            for kind in item.get("detected", []):
                # A payload that still matches its own detected pattern is a leak.
                pattern = PATTERNS.get(kind)
                if pattern and pattern.search(item["text"]):
                    leaked.append(kind)
            # A value outside the detected set can still be a live identifier
            # the suite knows: check every published pattern, not only the ones
            # this payload reported, so a rehydrated reply cannot slip through.
            for kind, pattern in PATTERNS.items():
                if kind not in item.get("detected", []) and pattern.search(item["text"]):
                    leaked.append(kind)
        return {
            "payloads": len(self.sent_payloads),
            "blocked": len(self.blocked),
            "leaked": sorted(set(leaked)),
            "paths": sorted({item["path"] for item in self.sent_payloads}),
            "ok": not leaked and not self.blocked,
        }
