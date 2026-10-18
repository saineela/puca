"""D5 - five independent validations for a skill plan.

Grammar-constrained decoding is allowed to own the *shape* of the envelope. It
is never allowed to stand in for the checks that decide whether the plan is the
right one. Each check below returns its own failure list, so a caller (and a
test) can see which guarantee held.

The claim is narrow on purpose: zero malformed envelopes, plus unchanged
intent/permission/confirmation/outcome checks. It is not a claim that a valid
plan is a correct, appropriate or authorised action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

CHECKS = ("schema", "intent", "permission", "confirmation", "outcome")


@dataclass
class PlanValidation:
    failures: dict[str, list[str]] = field(
        default_factory=lambda: {name: [] for name in CHECKS}
    )

    @property
    def ok(self) -> bool:
        return not any(self.failures.values())

    def add(self, check: str, message: str) -> None:
        if check not in self.failures:
            raise ValueError(f"unknown validation check: {check}")
        self.failures[check].append(message)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "failures": {k: list(v) for k, v in self.failures.items()}}


# --------------------------------------------------------------------- schema
def check_schema(plan: Any, *, max_steps: int = 8) -> list[str]:
    failures: list[str] = []
    if not isinstance(plan, (list, tuple)):
        return ["plan must be a list of steps"]
    if not plan:
        failures.append("plan must contain at least one step")
    if len(plan) > max_steps:
        failures.append(f"plan has {len(plan)} steps, above the limit of {max_steps}")
    for index, step in enumerate(plan):
        if not isinstance(step, Mapping):
            failures.append(f"step {index} is not an object")
            continue
        if step.get("type") != "skill_tool_call":
            failures.append(f"step {index} has an unknown type")
        for key in ("skill_id", "tool"):
            if not isinstance(step.get(key), str) or not step.get(key):
                failures.append(f"step {index} is missing {key}")
        if not isinstance(step.get("arguments"), Mapping):
            failures.append(f"step {index} is missing an arguments object")
    return failures


# --------------------------------------------------------------------- intent
_ACTION_WORDS = re.compile(
    r"\b(?:turn|switch|set|change|make|apply|paint|enable|disable|activate|"
    r"run|start|play|stop|open|close|brighten|dim|remind|schedule|remember|"
    r"save|store|add|remove|delete|cancel|mute|unmute|pause|resume|next|previous|"
    r"move|reschedule|feel|feels|is|are|am)\b",
    re.IGNORECASE,
)


_ACTION_ALIASES = {
    "update": ("is", "feels", "update", "state"),
    "remind": ("remind", "reminder"),
    "reschedule": ("reschedule", "move"),
    "turn": ("turn", "switch", "set"),
}


def check_intent(request: str, plan: Sequence[Mapping[str, Any]]) -> list[str]:
    """Compare the plan with the user's actual request, not with a prompt."""
    failures: list[str] = []
    if not (request or "").strip():
        return ["a plan needs a non-empty request"]
    asked_actions = set(_ACTION_WORDS.findall(request))
    asked = request.casefold()
    # A request may be value-led rather than verb-led ("brightness to 70",
    # "volume 30"). That is still an actionable request when the plan's own
    # argument names appear in it; requiring a verb rejected those legitimate
    # plans and the expanded D6 sample exposed it as a false negative.
    named_arguments = {
        str(key).casefold()
        for step in plan
        if isinstance(step, Mapping)
        for key in (step.get("arguments") or {})
        if str(key).casefold() in asked
    }
    if not asked_actions and not named_arguments:
        failures.append("request contains no actionable verb")
    for index, step in enumerate(plan):
        arguments = step.get("arguments") if isinstance(step, Mapping) else None
        if not isinstance(arguments, Mapping) or not arguments:
            failures.append(f"step {index} has no action to compare with the request")
            continue
        action = str(arguments.get("action") or "").strip()
        if not action:
            failures.append(f"step {index} has no action to compare with the request")
            continue
        aliases = _ACTION_ALIASES.get(action.casefold(), (action,))
        if not any(alias in asked for alias in aliases):
            failures.append(
                f"step {index} action {action!r} does not appear in the request"
            )
    return failures


# ----------------------------------------------------------------- permission
def check_permission(
    plan: Sequence[Mapping[str, Any]],
    *,
    allowed_skill_ids: Iterable[str] = (),
    argument_limits: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
) -> list[str]:
    """Only runnable skills and declared argument ranges may be used."""
    failures: list[str] = []
    allowed = set(allowed_skill_ids)
    for index, step in enumerate(plan):
        skill_id = str(step.get("skill_id") or "")
        if allowed and skill_id not in allowed:
            failures.append(f"step {index} uses a skill that is not runnable: {skill_id}")
        arguments = step.get("arguments") or {}
        for key, (low, high) in (argument_limits or {}).get(skill_id, {}).items():
            if key not in arguments:
                continue
            value = arguments[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                failures.append(f"step {index} argument {key} is not numeric")
            elif not low <= float(value) <= high:
                failures.append(
                    f"step {index} argument {key}={value} is outside {low}..{high}"
                )
    return failures


# --------------------------------------------------------------- confirmation
CONSEQUENTIAL_ACTIONS = frozenset(
    {"delete", "remove", "cancel", "disable", "stop", "power_off", "uninstall"}
)


def needs_confirmation(plan: Sequence[Mapping[str, Any]]) -> bool:
    for step in plan:
        arguments = step.get("arguments") or {}
        action = str(arguments.get("action") or "").casefold()
        if action in CONSEQUENTIAL_ACTIONS:
            return True
        # Turning something off is consequential even when the verb is "turn".
        if arguments.get("on") is False:
            return True
    return False


def check_confirmation(
    plan: Sequence[Mapping[str, Any]], *, confirmed: bool
) -> list[str]:
    if needs_confirmation(plan) and not confirmed:
        return ["a consequential action was planned without confirmation"]
    return []


# -------------------------------------------------------------------- outcome
def check_outcome(
    plan: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any] | None,
) -> list[str]:
    """Success is only success after the action's own result was checked."""
    if result is None:
        return ["no action result was returned, so success cannot be claimed"]
    if result.get("ok") is not True:
        return ["the action result did not report success"]
    if result.get("verified") is False:
        return ["the action result was returned but not verified"]
    return []


# --------------------------------------------------------------------- facade
def validate_plan(
    *,
    request: str,
    plan: Any,
    allowed_skill_ids: Iterable[str] = (),
    argument_limits: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    confirmed: bool = False,
    result: Mapping[str, Any] | None = None,
) -> PlanValidation:
    validation = PlanValidation()
    schema_failures = check_schema(plan)
    for failure in schema_failures:
        validation.add("schema", failure)
    # The other checks are only meaningful on a well-formed envelope, but they
    # stay independent: a malformed plan never short-circuits a permission
    # finding, and each check can be tested alone.
    steps = plan if isinstance(plan, (list, tuple)) else []
    for name, failures in (
        ("intent", check_intent(request, steps)),
        ("permission", check_permission(
            steps,
            allowed_skill_ids=allowed_skill_ids,
            argument_limits=argument_limits,
        )),
        ("confirmation", check_confirmation(steps, confirmed=confirmed)),
        ("outcome", check_outcome(steps, result)),
    ):
        for failure in failures:
            validation.add(name, failure)
    return validation
