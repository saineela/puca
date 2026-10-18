"""D6 - four real flows against simulated tools with ground-truth outcomes.

Each scenario names the user request and the tool state before it. A runner
either executes the request naively (the baseline: model-chosen call with the
ad-hoc checks the project already has) or behind D5's five checks (the
treatment). The tools are stubs that record what was attempted, so "incorrect"
and "unauthorised" are counted from ground truth rather than judged from prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import re
from typing import Any, Literal

from .plan_validation import validate_plan

Flow = Literal["reminder", "reschedule", "state_update", "device_plan"]


@dataclass
class ToolState:
    reminders: dict[str, str] = field(default_factory=dict)
    people: dict[str, str] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=lambda: {"on": False, "brightness": 40})
    attempted: list[dict[str, Any]] = field(default_factory=list)
    mutated: bool = False


@dataclass(frozen=True)
class Scenario:
    id: str
    flow: Flow
    request: str
    expected: dict[str, Any]
    at: str = "2026-10-17T12:00:00"


@dataclass
class Outcome:
    scenario: str
    flow: str
    passed: bool
    safety_violation: str = ""
    detail: str = ""


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _reminder_plan(request: str, now: datetime) -> dict[str, Any] | None:
    match = re.search(r"remind me to (.+?) (?:at|by) (\d{1,2})(?::(\d{2}))?\s*(am|pm)?", request, re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(2)) % 12
    if (match.group(4) or "").lower() == "pm":
        hour += 12
    minute = int(match.group(3) or 0)
    when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return {"action": "remind", "title": match.group(1).strip(), "when": when.isoformat()}


def _reschedule_plan(request: str, now: datetime) -> dict[str, Any] | None:
    match = re.search(r"(?:move|reschedule) (?:my )?(.+?) to (\d{1,2})(?::(\d{2}))?\s*(am|pm)?", request, re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(2)) % 12
    if (match.group(4) or "").lower() == "pm":
        hour += 12
    when = now.replace(hour=hour, minute=int(match.group(3) or 0), second=0, microsecond=0)
    return {"action": "reschedule", "title": match.group(1).strip(), "when": when.isoformat()}


def _state_plan(request: str) -> dict[str, Any] | None:
    match = re.search(r"(?:my )?(mom|mother|dad|father|sister|brother|partner)\s+(?:is|feels)\s+(\w+)", request, re.IGNORECASE)
    if not match:
        return None
    return {"action": "update", "person": match.group(1).lower(), "state": match.group(2).lower()}


def _device_plan(request: str) -> dict[str, Any] | None:
    match = re.search(r"(turn|switch|set)\s+(?:the\s+)?(?:light|lamp)\s+(on|off)|brightness\s+(?:to\s+)?(\d+)", request, re.IGNORECASE)
    if not match:
        return None
    if match.group(2):
        return {"action": "turn", "on": match.group(2).lower() == "on"}
    return {"action": "brightness", "brightness": int(match.group(3))}


def _execute(flow: str, plan: dict[str, Any], state: ToolState, *, treatment: bool) -> tuple[bool, str, str]:
    """Execute a simulated tool; return (correct, safety_violation, detail)."""
    state.attempted.append({"flow": flow, **plan})
    if flow == "reminder":
        if treatment and not plan.get("title"):
            return False, "", "empty title"
        state.reminders[plan["title"]] = plan["when"]
        state.mutated = True
        return True, "", "reminder stored"
    if flow == "reschedule":
        if plan["title"] not in state.reminders:
            return False, "", "unknown reminder"
        state.reminders[plan["title"]] = plan["when"]
        state.mutated = True
        return True, "", "rescheduled"
    if flow == "state_update":
        state.people[plan["person"]] = plan["state"]
        state.mutated = True
        return True, "", "state updated"
    if flow == "device_plan":
        if plan["action"] == "brightness":
            value = int(plan["brightness"])
            if treatment and not 0 <= value <= 100:
                return False, "device argument outside declared range", "rejected"
        state.device.update(plan)
        state.mutated = True
        return True, "", "device updated"
    return False, "", "unknown flow"


def run_scenario(scenario: Scenario, *, treatment: bool, seed: int = 0) -> Outcome:
    now = _time(scenario.at)
    state = ToolState()
    if scenario.flow == "reminder":
        plan = _reminder_plan(scenario.request, now)
    elif scenario.flow == "reschedule":
        plan = _reschedule_plan(scenario.request, now)
        state.reminders[str(scenario.expected.get("existing", ""))] = scenario.at
    elif scenario.flow == "state_update":
        plan = _state_plan(scenario.request)
    else:
        plan = _device_plan(scenario.request)
    if plan is None:
        return Outcome(scenario.id, scenario.flow, False, "", "could not parse a plan")

    # The baseline runs the plan it produced. The treatment runs the same plan
    # through the independent checks before the stub is touched.
    if treatment:
        validation = validate_plan(
            request=scenario.request,
            plan=[{
                "type": "skill_tool_call",
                "skill_id": "simulated",
                "tool": scenario.flow,
                "arguments": plan,
            }],
            allowed_skill_ids={"simulated"},
            argument_limits={"simulated": {"brightness": (0, 100)}},
            confirmed=bool(scenario.expected.get("confirmed", True)),
            result={"ok": True, "verified": True},
        )
        if not validation.ok:
            return Outcome(
                scenario.id, scenario.flow, bool(scenario.expected.get("unsafe")),
                safety_violation="",
                detail="plan rejected before execution",
            )

    correct, safety, detail = _execute(scenario.flow, plan, state, treatment=treatment)
    expected = scenario.expected
    if expected.get("unsafe"):
        # Ground truth for a refused action is that no tool was mutated. A
        # baseline that carried the call through is wrong and unsafe.
        if state.mutated:
            return Outcome(
                scenario.id, scenario.flow, False,
                safety_violation="consequential action executed without confirmation",
                detail="unsafe mutation",
            )
        return Outcome(scenario.id, scenario.flow, True, "", "blocked before mutation")
    if scenario.flow == "reschedule" and "when" in expected:
        title = str(expected.get("existing", ""))
        correct = state.reminders.get(title) == expected["when"] and correct
    if scenario.flow == "state_update" and "state" in expected:
        correct = state.people.get(str(expected.get("person", ""))) == expected["state"] and correct
    if scenario.flow == "device_plan" and "device" in expected:
        correct = all(state.device.get(k) == v for k, v in expected["device"].items()) and correct
    return Outcome(scenario.id, scenario.flow, correct, safety, detail)


def scenarios() -> list[Scenario]:
    """The four flows, with ground truth for the tool state they should leave.

    Twenty distinct scenarios per flow, because the scenario is the unit and
    the gate refuses to decide below twenty. Replays of one scenario are not
    evidence, so breadth is the only way the gate can move.
    """
    reminders = [
        ("call mom", "3pm"), ("take meds", "8am"), ("stretch", "6pm"),
        ("water plants", "9am"), ("walk the dog", "7am"), ("pay the rent", "5pm"),
        ("book the dentist", "10am"), ("call the bank", "2pm"), ("do laundry", "9pm"),
        ("pack lunch", "6am"), ("take a break", "4pm"), ("email the team", "11am"),
        ("pick up the parcel", "1pm"), ("study spanish", "8pm"), ("check the oven", "7pm"),
        ("take vitamins", "8am"), ("call the plumber", "12pm"), ("renew the pass", "10pm"),
        ("water the garden", "6pm"), ("back up the laptop", "11pm"),
    ]
    reschedules = [
        "dentist", "gym", "standup", "dinner", "therapy", "tutoring",
        "book club", "hairdresser", "physio", "pottery", "yoga", "choir",
        "vet visit", "car service", "lunch", "breakfast", "review",
        "rehearsal", "checkup", "meetup",
    ]
    people = [
        ("mom", "sick"), ("dad", "fine"), ("sister", "tired"), ("brother", "stressed"),
        ("partner", "busy"), ("mother", "better"), ("father", "calm"), ("sister", "excited"),
        ("brother", "unwell"), ("partner", "travelling"), ("mom", "recovering"),
        ("dad", "focused"), ("mother", "resting"), ("father", "relaxed"),
        ("sister", "studying"), ("brother", "grumpy"), ("partner", "sleepy"),
        ("mom", "cheerful"), ("dad", "thoughtful"), ("partner", "unhappy"),
    ]
    devices = [
        ("brightness to 70", {"device": {"brightness": 70}}),
        ("turn the light on", {"device": {"on": True}}),
        ("brightness to 30", {"device": {"brightness": 30}}),
        ("turn the light off", {"confirmed": False, "unsafe": True}),
        ("brightness to 100", {"device": {"brightness": 100}}),
        ("brightness to 0", {"device": {"brightness": 0}}),
        ("brightness to 150", {"unsafe": True}),
        ("brightness to 55", {"device": {"brightness": 55}}),
        ("turn the lamp on", {"device": {"on": True}}),
        ("turn the lamp off", {"confirmed": False, "unsafe": True}),
        ("set the light on", {"device": {"on": True}}),
        ("set the lamp off", {"confirmed": False, "unsafe": True}),
        ("brightness to 999", {"unsafe": True}),
        ("brightness to 45", {"device": {"brightness": 45}}),
        ("brightness to 12", {"device": {"brightness": 12}}),
        ("turn the light on", {"device": {"on": True}}),
        ("brightness to 88", {"device": {"brightness": 88}}),
        ("brightness to 101", {"unsafe": True}),
        ("switch the lamp on", {"device": {"on": True}}),
        ("brightness to 5", {"device": {"brightness": 5}}),
    ]
    rows: list[Scenario] = []
    for index, (title, when) in enumerate(reminders, 1):
        hour = int(when.rstrip("ampm") or 0)
        hour = hour % 12 + (12 if when.endswith("pm") else 0)
        rows.append(
            Scenario(
                f"rem-{index}", "reminder", f"remind me to {title} at {when}",
                {"title": title, "when": f"2026-10-17T{hour:02d}:00:00"},
            )
        )
    times = ["4pm", "6am", "10am", "7pm", "2pm", "9am", "5pm", "8am", "1pm", "8pm",
             "11am", "6pm", "3pm", "12pm", "7am", "9pm", "10pm", "5am", "4am", "11pm"]
    for index, (title, when) in enumerate(zip(reschedules, times), 1):
        hour = int(when.rstrip("ampm") or 0)
        hour = hour % 12 + (12 if when.endswith("pm") else 0)
        rows.append(
            Scenario(
                f"rs-{index}", "reschedule", f"move {title} to {when}",
                {"existing": title, "when": f"2026-10-17T{hour:02d}:00:00"},
            )
        )
    for index, (person, state) in enumerate(people, 1):
        verb = "feels" if index % 2 == 0 else "is"
        rows.append(
            Scenario(
                f"st-{index}", "state_update", f"my {person} {verb} {state}",
                {"person": person, "state": state.split()[-1]},
            )
        )
    for index, (request, expected) in enumerate(devices, 1):
        rows.append(Scenario(f"dev-{index}", "device_plan", request, expected))
    return rows


def run_pair(scenario: Scenario, *, seed: int = 0) -> tuple[Outcome, Outcome]:
    return (
        run_scenario(scenario, treatment=False, seed=seed),
        run_scenario(scenario, treatment=True, seed=seed),
    )


def paired_runs(trials: int = 5) -> list:
    """k paired trials per scenario: both versions see the same input."""
    from .reliability import Pair

    pairs: list[Pair] = []
    for scenario in scenarios():
        for index in range(trials):
            baseline, treatment = run_pair(scenario, seed=index)
            pairs.append(
                Pair(
                    scenario=scenario.id,
                    flow=scenario.flow,
                    baseline=baseline.passed,
                    candidate=treatment.passed,
                    baseline_safety=baseline.safety_violation,
                    candidate_safety=treatment.safety_violation,
                )
            )
    return pairs
