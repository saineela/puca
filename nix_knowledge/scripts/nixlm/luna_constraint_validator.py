"""Fail-closed checks for a limited, mechanically verifiable constraint subset."""
from __future__ import annotations

import re

_COUNT = re.compile(
    r"\b(exactly|at least|at most|no more than|less than|fewer than|more than|"
    r"no fewer than|minimum of|maximum of)\s+(\d+)\s+"
    r"(bullet points?|bullets?|sentences?|words?|paragraphs?)\b",
    re.IGNORECASE,
)
_FREQUENCY = re.compile(
    r"\b(?:the\s+)?(?:word|keyword)\s+(?:['\"]([^'\"]+)['\"]|\[([^\]]+)\]|([\w-]+))"
    r"\s+(?:(?:should|must|needs? to|has to|have to)\s+)?"
    r"(?:(?:appear|occur|be mentioned|show up)\s+)?"
    r"(exactly|at least|at most|no more than|less than|fewer than|more than)\s+(\d+)\s+times?\b",
    re.IGNORECASE,
)
_PLACEHOLDER_COUNT = re.compile(
    r"\b(exactly|at least|at most|no more than|less than|fewer than)\s+(\d+)\s+"
    r"placeholders?(?:\s+represented\s+by\s+square\s+brackets(?:,?\s+such\s+as\s+\[[^\]]+\])?)?",
    re.IGNORECASE,
)
_PLACEHOLDER_LIST = re.compile(
    r"\b(?:use|include|provide|keep|preserve)\s+(?:number\s+)?placeholders?\s+for\s+"
    r"((?:\[[^\]]+\](?:\s*(?:,|and)\s*)?)+)",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"\[([^\]]{1,100})\]")
_KEYWORDS = re.compile(
    r"\b(?:include|use|mention|contain)\s+(?:(?:the|these|following)\s+)?"
    r"keywords?\s*:?\s*([^.!?;\n]+)",
    re.IGNORECASE,
)
_FORBIDDEN = re.compile(
    r"\b(?:do not include|don't include|avoid|exclude|forbidden)\s+"
    r"(?:(?:the|these|following)\s+)?keywords?\s*:?\s*([^.!?;\n]+)",
    re.IGNORECASE,
)
_NO_COMMA = re.compile(r"\b(?:no|without|do not use|don't use)\s+(?:any\s+)?commas?\b", re.IGNORECASE)
_NO_CAPITAL = re.compile(
    r"\b(?:no|do not use|don't use|without)\s+(?:any\s+)?capital letters?\b|"
    r"\bno capital letters?\s+(?:are\s+)?allowed\b",
    re.IGNORECASE,
)
_EXACT_END = re.compile(
    r"\b(?:end|finish)\s+(?:your\s+)?(?:response|answer)\s+with\s+"
    r"(?:this\s+)?exact\s+phrase\s*(?:['\"]([^'\"]+)['\"]|\[([^\]]+)\])",
    re.IGNORECASE,
)
_POSTSCRIPT = re.compile(
    r"\bpostscript\b.{0,100}?\bstarting\s+with\s+(?:['\"])?(P(?:\.P)?\.S\.)",
    re.IGNORECASE,
)
_BULLET_STYLE = re.compile(
    r"\buse\s+(?:the\s+)?(?:markdown\s+)?bullet\s+points?\s+such\s+as\s*:\s*([-*])\s+[^.!?\n]+[.!?]?",
    re.IGNORECASE,
)
_OBJECTIVE_MARKER = re.compile(
    r"\b(?:exactly|at least|at most|no more than|less than|fewer than|more than|"
    r"no fewer than|minimum of|maximum of|lowercase|lower case|uppercase|upper case|"
    r"include|exclude|forbidden|avoid|mention|bullet|sentences?|word count|paragraphs?|"
    r"placeholders?|square brackets?|postscript|exact phrase|first word|last word|"
    r"first letter|last letter|first line|last line|starting with|start with|begin with|"
    r"ending with|end with|finish with|no commas?|without commas?|do not use|don't use|use only|"
    r"must contain|must have|should have|capital letters?|capitalization|english|"
    r"sections?|headings?|titles?|appear.{0,20}times|times?\s+in\s+(?:the\s+)?response|"
    r"json|xml|html|csv|yaml|markdown|code block|alphabetically|chronologically|"
    r"ascending order|descending order|haiku|acrostic|alliteration|rhyme|rhyming|syllables?|"
    r"characters?|special characters?|punctuation|question marks?|exclamation marks?|"
    r"numbered list|one[- ]line|each bullet|every bullet|first person|second person|"
    r"contractions?|acronyms?|pronouns?|first person|second person|third person|"
    r"without using|only use|do not use)\b",
    re.IGNORECASE,
)


def _satisfies(mode: str, actual: int, expected: int) -> bool:
    mode = mode.casefold()
    if mode == "exactly":
        return actual == expected
    if mode in {"at least", "no fewer than", "minimum of"}:
        return actual >= expected
    if mode in {"at most", "no more than", "maximum of"}:
        return actual <= expected
    if mode in {"less than", "fewer than"}:
        return actual < expected
    if mode == "more than":
        return actual > expected
    return False


def _terms(raw: str) -> list[str] | None:
    """Parse only unambiguous keyword lists, leaving extra instructions visible."""
    raw = raw.strip()
    quoted_spans = list(re.finditer(r"['\"]([^'\"]+)['\"]", raw))
    if quoted_spans:
        residue = list(raw)
        terms: list[str] = []
        for match in quoted_spans:
            terms.append(match.group(1).strip().casefold())
            residue[match.start():match.end()] = [" "] * (match.end() - match.start())
        remainder = "".join(residue)
        if not re.fullmatch(
            r"\s*(?:(?:,|\band\b|\bor\b)\s*)*"
            r"(?:in\s+(?:(?:the|your)\s+)?(?:response|answer))?\s*",
            remainder,
            re.IGNORECASE,
        ):
            return None
        return terms if all(terms) else None

    if not re.search(r",|\band\b|\bor\b", raw, re.IGNORECASE):
        return None
    value = re.sub(
        r"\s*,?\s+in\s+(?:(?:the|your)\s+)?(?:response|answer)\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip().rstrip(",").strip()
    if not value or _OBJECTIVE_MARKER.search(value):
        return None
    parts = [part.strip() for part in re.split(r",|\band\b|\bor\b", value, flags=re.IGNORECASE)]
    if not parts or any(
        not re.fullmatch(r"[\w][\w’'./-]*(?:\s+[\w][\w’'./-]*){0,5}", part, re.UNICODE)
        for part in parts
    ):
        return None
    return [part.casefold() for part in parts]


def _count_term(answer: str, term: str) -> int:
    if not term.strip():
        return 0
    return len(re.findall(r"(?<!\w)" + re.escape(term.strip()) + r"(?!\w)", answer, re.I))


def _count_words(answer: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", answer, re.UNICODE))


def _count_sentences(answer: str) -> int:
    # Deliberately reject uncertain abbreviation/format cases rather than guess.
    return len(re.findall(r"[.!?](?=[\"'”’)]*(?:\s|$))", answer))


def _count_bullets(answer: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", answer))


def _count_paragraphs(answer: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", answer.strip()) if part.strip()])


def validate_objective_constraints(prompt: str, answer: str) -> bool:
    """Return true only if each supported objective constraint is satisfied.

    This does not score language, section counts, exact heading formats, or
    arbitrary start/end rules. Any detectable but unsupported rule is rejected.
    """
    checks: list[tuple[int, int, str, object]] = []

    for match in _COUNT.finditer(prompt):
        mode, count, unit = match.groups()
        checks.append((match.start(), match.end(), "count", (mode.casefold(), int(count), unit.casefold())))

    for pattern, kind in ((_KEYWORDS, "include"), (_FORBIDDEN, "forbid")):
        for match in pattern.finditer(prompt):
            values = _terms(match.group(1))
            if values:
                checks.append((match.start(), match.end(), kind, values))

    for match in _FREQUENCY.finditer(prompt):
        term = next((part for part in match.groups()[:3] if part), "").strip(" []\t")
        mode, count = match.groups()[3:]
        checks.append((match.start(), match.end(), "frequency", (term, mode.casefold(), int(count))))

    for match in _PLACEHOLDER_COUNT.finditer(prompt):
        mode, count = match.groups()[:2]
        checks.append((match.start(), match.end(), "placeholder_count", (mode.casefold(), int(count))))

    for match in _PLACEHOLDER_LIST.finditer(prompt):
        values = [part.strip().casefold() for part in _PLACEHOLDER.findall(match.group(1))]
        if values:
            checks.append((match.start(), match.end(), "placeholder_list", values))

    for match in _EXACT_END.finditer(prompt):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            checks.append((match.start(), match.end(), "exact_end", phrase))

    for match in _POSTSCRIPT.finditer(prompt):
        checks.append((match.start(), match.end(), "postscript", match.group(1).upper()))

    for match in _BULLET_STYLE.finditer(prompt):
        checks.append((match.start(), match.end(), "bullet_style", match.group(1)))

    for pattern, kind in (
        (re.compile(r"\b(?:all\s+)?(?:lowercase|lower case)\b", re.I), "lowercase"),
        (re.compile(r"\b(?:all\s+)?(?:uppercase|upper case)\b", re.I), "uppercase"),
        (_NO_COMMA, "no_comma"),
        (_NO_CAPITAL, "no_capital"),
    ):

        for match in pattern.finditer(prompt):
            checks.append((match.start(), match.end(), kind, None))

    if not checks:
        return False

    masked = list(prompt)
    for start, end, _, _ in checks:
        masked[start:end] = [" "] * (end - start)
    if _OBJECTIVE_MARKER.search("".join(masked)):
        return False

    lowered = answer.casefold()
    for _, _, kind, value in checks:
        if kind == "count":
            mode, expected, unit = value  # type: ignore[misc]
            if unit.startswith("word"):
                actual = _count_words(answer)
            elif unit.startswith("sentence"):
                actual = _count_sentences(answer)
            elif unit.startswith("paragraph"):
                actual = _count_paragraphs(answer)
            else:
                actual = _count_bullets(answer)
            if not _satisfies(mode, actual, expected):
                return False
        elif kind == "include" and any(_count_term(answer, term) == 0 for term in value):  # type: ignore[union-attr]
            return False
        elif kind == "forbid" and any(_count_term(answer, term) > 0 for term in value):  # type: ignore[union-attr]
            return False
        elif kind == "frequency":
            term, mode, expected = value  # type: ignore[misc]
            if not _satisfies(mode, _count_term(answer, term), expected):
                return False
        elif kind == "placeholder_count":
            mode, expected = value  # type: ignore[misc]
            if not _satisfies(mode, len(_PLACEHOLDER.findall(answer)), expected):
                return False
        elif kind == "exact_end" and not answer.rstrip().casefold().endswith(str(value).casefold()):
            return False
        elif kind == "postscript" and not re.search(r"(?im)^\s*" + re.escape(str(value)), answer):
            return False
        elif kind == "bullet_style" and not re.search(r"(?m)^\s*" + re.escape(str(value)) + r"\s+", answer):
            return False
        elif kind == "placeholder_list" and any(f"[{term}]" not in lowered for term in value):  # type: ignore[union-attr]
            return False
        elif kind == "lowercase" and answer != answer.lower():
            return False
        elif kind == "uppercase" and answer != answer.upper():
            return False
        elif kind == "no_comma" and "," in answer:
            return False
        elif kind == "no_capital" and re.search(r"[A-Z]", answer):
            return False
    return True
