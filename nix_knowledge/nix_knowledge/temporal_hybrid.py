"""Neuro-symbolic temporal parsing for calendar requests.

The neural selector is allowed to *propose* a title and temporal expression,
but it is never allowed to decide what those words mean.  This module joins
that proposal to :class:`TemporalResolver`, the symbolic authority that does
date arithmetic, clock parsing, range handling, and safety validation.

The boundary is intentionally useful without a second model:

* a Qwen/tool proposal can be passed in and validated;
* when the proposal drops part of a temporal expression, symbolic span search
  repairs it from the original utterance;
* unresolved or conflicting temporal language returns ``None`` rather than
  creating a guessed event.

This is the project's neuro-symbolic contract: neural language understanding
proposes structure, Python grammar and constraints dispose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TemporalProposal:
    """A candidate emitted by a neural/function-selection layer."""

    title: str
    expression: str
    confidence: float = 0.0
    source: str = "neural"


@dataclass(frozen=True)
class HybridTemporalParse:
    """A symbolic validation result safe for calendar mutation."""

    title: str
    expression: str
    resolved: Any
    slots: dict[str, Any]
    source: str
    confidence: float

    def arguments(self) -> dict[str, str]:
        return {
            "title": self.title,
            "temporal_expression": self.expression,
        }


# Strong temporal residue is not allowed to remain in an event title.  This
# catches the exact failure where Qwen returned only "from 9am to 11am" and
# left "5 days after today" in the title.
_TEMPORAL_RESIDUE_RE = re.compile(
    r"\b(?:today|tomorrow|tmr|yesterday|tonight|"
    r"day\s+after\s+tomorrow|"
    r"next|this|last|after|before|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"morning|afternoon|evening|"
    r"\d+\s+(?:minutes?|hours?|days?|weeks?)|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)",
    re.IGNORECASE,
)

_COMMAND_PREFIX_RE = re.compile(
    r"^(?:please\s+)?(?:i\s+have|i've\s+got|i\s+got|"
    r"schedule|add|create|book|set\s+up)"
    r"(?:\s+(?:a|an|my))?\s+",
    re.IGNORECASE,
)

_REMINDER_PREFIX_RE = re.compile(
    r"^(?:please\s+)?remind\s+me\s+(?:to|about)\s+",
    re.IGNORECASE,
)


_CASUAL_LEAD_IN_RE = re.compile(
    r"^(?:(?:alright|all\s+right|okay|ok|hey|hi|hello|yo)\s+)?"
    r"(?:bro|dude|man)\s*[,!:]\s*",
    re.IGNORECASE,
)


_MONTHS = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)


def _clean_request(text: str) -> str:
    """Remove only command wrappers; preserve the user's event words."""
    value = " ".join((text or "").strip().rstrip(".!?").split())
    value = re.sub(
        r"^(?:hey|hi|hello)\s+(?:casper|nix)\s*[,!:]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = _CASUAL_LEAD_IN_RE.sub("", value)
    value = _COMMAND_PREFIX_RE.sub("", value)
    value = _REMINDER_PREFIX_RE.sub("", value)
    # Preserve event words but ignore common conversational framing when
    # symbolically recovering title + time from casual voice transcripts.
    value = re.sub(r"^(?:i\s+have|i've\s+got|i\s+got)\s+", "", value, flags=re.IGNORECASE)
    return value.strip(" ,:;")


def _clean_title(title: Any) -> str:
    value = " ".join(str(title or "").strip().rstrip(".!?").split())
    # Speech-style filler often leaks from a selector proposal into the
    # title: "dentist appointment which is happening in 2 days". The date
    # belongs in the temporal expression, not in the event name.
    value = re.sub(
        r"\s+(?:which|that)\s+is\s+(?:happening|scheduled|planned)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+\b(?:happening|scheduled|planned)\b$", "", value, flags=re.IGNORECASE)
    return value.strip(" ,:;")


def _resolve(resolver, expression: str, now: datetime | None):
    if now is None:
        return resolver.resolve(expression)
    return resolver.resolve(expression, now=now)


def _has_temporal_residue(title: str) -> bool:
    return bool(_TEMPORAL_RESIDUE_RE.search(title))


def _temporal_signature(text: str) -> set[str]:
    """Return normalized temporal markers present in an utterance.

    This is a coverage check, not a parser. It prevents a seemingly valid
    neural span (for example, only ``from 9am to 11am``) from silently
    discarding ``5 days after today during the morning`` from the original
    request.
    """
    value = " ".join((text or "").lower().split())
    markers = {
        match.group(0).strip()
        for match in _TEMPORAL_RESIDUE_RE.finditer(value)
    }
    # The residue expression intentionally focuses on semantic words. Keep
    # clock endpoints as distinct markers so a dropped range is detected.
    markers.update(
        match.group(0).replace(" ", "")
        for match in re.finditer(
            r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
            value,
            re.IGNORECASE,
        )
    )
    return markers


def _proposal_covers_request(request: str, expression: str) -> bool:
    """Whether a neural temporal span covers every spoken time marker."""
    # Compare after the same harmless speech normalization used by the
    # symbolic resolver; otherwise "in 2 more days" looks unlike "in 2 days"
    # and a partial neural proposal such as "tomorrow at 4pm" is accepted.
    def normalize(value: str) -> str:
        value = re.sub(
            r"\bin\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"more\s+(minutes?|hours?|days?|weeks?)\b",
            r"in \1 \2",
            value,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\baround\s+(?=\d|noon|midnight)", "at ", value, flags=re.IGNORECASE)

    request_markers = _temporal_signature(normalize(request))
    expression_markers = _temporal_signature(normalize(expression))
    return request_markers <= expression_markers


def _candidate_suffixes(text: str):
    tokens = text.split()
    # Longest first: a complete compound expression wins over a trailing
    # clock range or a lone day-part.
    for size in range(len(tokens) - 1, 0, -1):
        yield " ".join(tokens[-size:]), " ".join(tokens[:-size])


def _candidate_prefixes(text: str):
    tokens = text.split()
    for size in range(min(10, len(tokens) - 1), 0, -1):
        yield " ".join(tokens[:size]), " ".join(tokens[size:])


def _clock_slots(expression: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    range_match = re.search(
        r"(?:from\s+)?(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)"
        r"\s*(?:to|until|till)\s*"
        r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)",
        expression,
        re.IGNORECASE,
    )
    if range_match:
        slots["start_clock"] = range_match.group("start")
        slots["end_clock"] = range_match.group("end")
        return slots

    clock = re.search(
        r"\b(?:at\s+)?(?P<clock>\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)\b",
        expression,
        re.IGNORECASE,
    )
    if clock:
        slots["clock"] = clock.group("clock")
    return slots


def extract_slots(expression: str, resolved: Any) -> dict[str, Any]:
    """Extract auditable semantic slots after symbolic validation."""
    lower = expression.lower()
    slots: dict[str, Any] = {
        "reference": None,
        "offset_amount": None,
        "offset_unit": None,
        "offset_relation": None,
        "day_part": None,
        **_clock_slots(expression),
    }

    reference = re.search(
        r"\b(today|tomorrow|tmr|yesterday|now)\b|"
        r"\b(?:next|this|last)(?:\s+week)?\s+"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lower,
    )
    if reference:
        slots["reference"] = reference.group(0)

    offset = re.search(
        r"\b(?P<amount>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?P<unit>minutes?|hours?|days?|weeks?)\s+"
        r"(?P<relation>after|before|from)\b",
        lower,
    )
    if offset:
        slots["offset_amount"] = offset.group("amount")
        slots["offset_unit"] = offset.group("unit")
        slots["offset_relation"] = offset.group("relation")
    else:
        relative = re.search(
            r"\bin\s+(?P<amount>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?P<unit>minutes?|hours?|days?|weeks?)\b",
            lower,
        )
        if relative:
            slots["offset_amount"] = relative.group("amount")
            slots["offset_unit"] = relative.group("unit")
            slots["offset_relation"] = "in"

    day_part = re.search(r"\b(morning|afternoon|evening|tonight)\b", lower)
    if day_part:
        slots["day_part"] = day_part.group(1)

    # The resolver is the final authority. These are observability fields,
    # not an alternate source of datetime truth.
    slots.update(
        resolved_start=resolved.start.isoformat(),
        resolved_end=resolved.end.isoformat() if resolved.end else None,
        all_day=bool(resolved.all_day),
        recurring=bool(resolved.recurring),
        recurrence=resolved.recurrence,
    )
    return slots


def _validated_candidate(
    *,
    title: str,
    expression: str,
    resolver,
    now: datetime | None,
    source: str,
    confidence: float,
) -> HybridTemporalParse | None:
    title = _clean_title(title)
    expression = " ".join(str(expression or "").split()).strip(" ,:;")
    if not title or not expression or _has_temporal_residue(title):
        return None

    resolved = _resolve(resolver, expression, now)
    if resolved is None:
        return None

    return HybridTemporalParse(
        title=title,
        expression=expression,
        resolved=resolved,
        slots=extract_slots(expression, resolved),
        source=source,
        confidence=max(0.0, min(1.0, confidence)),
    )


def parse_event(
    text: str,
    resolver,
    *,
    proposal: TemporalProposal | dict[str, Any] | None = None,
    now: datetime | None = None,
) -> HybridTemporalParse | None:
    """Parse and validate an event using neural proposal + symbolic rules.

    A valid neural proposal is accepted only when its title contains no
    unresolved temporal language and its expression resolves symbolically.
    Otherwise candidates are regenerated from the original utterance and
    scored by temporal coverage, which repairs partial tool calls safely.
    """
    request = _clean_request(text)

    if proposal is not None:
        if isinstance(proposal, dict):
            proposal = TemporalProposal(
                title=str(proposal.get("title") or ""),
                expression=str(
                    proposal.get("temporal_expression")
                    or proposal.get("expression")
                    or ""
                ),
                confidence=float(proposal.get("confidence") or 0.0),
                source=str(proposal.get("source") or "neural"),
            )
        accepted = _validated_candidate(
            title=proposal.title,
            expression=proposal.expression,
            resolver=resolver,
            now=now,
            source="neural_validated",
            confidence=proposal.confidence,
        )
        if accepted is not None and _proposal_covers_request(
            request,
            proposal.expression,
        ):
            return accepted

    candidates: list[HybridTemporalParse] = []

    for expression, title in _candidate_suffixes(request):
        parsed = _validated_candidate(
            title=title,
            expression=expression,
            resolver=resolver,
            now=now,
            source="symbolic_repair",
            confidence=0.95,
        )
        if parsed is not None:
            candidates.append(parsed)

    for expression, remainder in _candidate_prefixes(request):
        # Prefixes may include the command lead-in after the temporal phrase
        # ("tomorrow I have robotics practice").
        title = _clean_request(remainder)
        parsed = _validated_candidate(
            title=title,
            expression=expression,
            resolver=resolver,
            now=now,
            source="symbolic_repair",
            confidence=0.9,
        )
        if parsed is not None:
            candidates.append(parsed)

    if not candidates:
        return None

    # Prefer the parse that consumes the most temporal words. Since suffix
    # candidates are generated longest-first, this also prevents a broad
    # title from swallowing a compound date.
    candidates.sort(
        key=lambda item: (
            len(item.expression.split()),
            -len(item.title.split()),
        ),
        reverse=True,
    )
    return candidates[0]


def repair_calendar_arguments(
    request: str,
    arguments: dict[str, Any],
    resolver,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], HybridTemporalParse | None]:
    """Repair/validate neural create-event arguments from original text."""
    proposal = TemporalProposal(
        title=str(arguments.get("title") or ""),
        expression=str(arguments.get("temporal_expression") or ""),
        confidence=0.5,
    )
    parsed = parse_event(request, resolver, proposal=proposal, now=now)
    if parsed is None:
        return arguments, None
    return parsed.arguments(), parsed


def validate_calendar_update_arguments(
    request: str,
    arguments: dict[str, Any],
    resolver,
    *,
    now: datetime | None = None,
) -> tuple[dict[str, Any], HybridTemporalParse | None]:
    """Validate a proposed update without changing its event identity.

    Updates retain the existing event title; only the new temporal expression
    is repaired/validated. This keeps the same neural-proposal/symbolic-gate
    contract as creation while preventing a malformed update from reaching the
    database or action rescheduler.
    """
    expression = str(arguments.get("new_temporal_expression") or "").strip()
    if not expression:
        return arguments, None

    proposal = TemporalProposal(
        title=str(arguments.get("title") or "event"),
        expression=expression,
        confidence=0.5,
    )
    parsed = parse_event(request, resolver, proposal=proposal, now=now)
    if parsed is None:
        return arguments, None

    repaired = dict(arguments)
    repaired["new_temporal_expression"] = parsed.expression
    return repaired, parsed


__all__ = [
    "HybridTemporalParse",
    "TemporalProposal",
    "extract_slots",
    "parse_event",
    "repair_calendar_arguments",
    "validate_calendar_update_arguments",
]
