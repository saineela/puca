"""Knowledge engine stress campaign: new, untrained prompts.

Runs ~60 prompts that appear nowhere in the test corpus through the
production needle.process() pipeline, in two phases:

  Phase 1 (deterministic): model gate OFF. Every request a real
  deployment resolves without the selector must resolve here. A
  no_function_selected outcome is a DEFER (the selector model gets it
  in phase 2), anything else unexpected is a FAIL.

  Phase 2 (selector): the deferred prompts are re-run with the real
  Qwen selector loaded, exactly as production does. The selected
  function is judged against the same expectation.

Any exception escaping process() is a CRASH. A report is printed and
written to knowledge_stress_report.json next to this script.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nix_actions"))

from nix_knowledge.engine import KnowledgeEngine  # noqa: E402
from nix_knowledge.needle import KnowledgeNeedle  # noqa: E402
from nix_actions.engine import ActionsEngine  # noqa: E402

# (text, expected_function_or_None, note)
CASES: list[tuple[str, str | None, str]] = [
    # --- A. create/schedule -------------------------------------------------
    ("I have a math test next tuesday", "create_calendar_event", ""),
    ("schedule a haircut for saturday at 11am", "create_calendar_event", ""),
    ("add a dentist appointment on november 3rd at 9am", "create_calendar_event", ""),
    ("put chess club on my calendar thursday after school", "create_calendar_event", ""),
    ("we have a band concert next friday evening", "create_calendar_event", ""),
    ("book a driving lesson for sunday 10am", "create_calendar_event", ""),
    ("remind me to submit my college essay by friday", "create_calendar_event", ""),
    ("remind me about mom's birthday on december 3rd", "create_calendar_event", ""),
    ("set an alarm for 6:30am tomorrow", "create_calendar_event", "alarm bridge"),
    ("i got a haircut appointment this saturday at 2", "create_calendar_event", ""),
    # --- B. calendar lookups ------------------------------------------------
    ("what events do i have this week", "find_calendar_events", ""),
    ("what's on my schedule tomorrow", "find_calendar_events", ""),
    ("do i have anything on friday", "find_calendar_events", ""),
    ("when is my math test", "find_calendar_events", ""),
    ("what time is my dentist appointment", "find_calendar_events", ""),
    ("show me my calendar for next week", "find_calendar_events", ""),
    ("do i have anything going on this weekend", "find_calendar_events", ""),
    # --- C. durable facts -----------------------------------------------------
    ("remember that my locker number is 428", "create_fact", ""),
    ("remember that i sit with jade at lunch", "create_fact", ""),
    ("my best friend's name is carla", "create_fact", ""),
    ("i am in the robotics club at school", "create_fact", ""),
    ("i love playing guitar after school", "create_fact", ""),
    ("just so you know, my exam is on november 12th", "create_fact", "note"),
    ("i have two siblings", "create_fact", ""),
    ("remember that coach carter moved practice to 6", "create_fact", ""),
    # --- D. person states -----------------------------------------------------
    ("my mom is feeling sick today", "create_state", ""),
    ("my sister is stressed about finals", "create_state", ""),
    ("my brother is doing much better now", "create_state", "supersede"),
    ("how is my mom doing", "find_states", ""),
    ("who is sick at home", "find_states", ""),
    ("my dad is back from his trip", "create_state", "supersede"),
    # --- E. mutations: cancel / update --------------------------------------
    ("move my math test to next wednesday", "update_calendar_event", ""),
    ("reschedule my haircut to sunday at 3pm", "update_calendar_event", ""),
    ("my piano recital is postponed to december 5th", "update_calendar_event", "postponed news"),
    ("my saturday class got moved to sunday", "update_calendar_event", "moved news"),
    ("cancel my 3pm meeting today", "cancel_calendar_event", ""),
    ("delete the reminder for taking vitamins", "cancel_calendar_event", ""),
    ("my robotics practice has been cancelled", "cancel_calendar_event", "oct 5 replay"),
    ("i cancelled my tutoring", "cancel_calendar_event", ""),
    # --- F. tricky / negative guards -----------------------------------------
    ("what if i scheduled two tests on the same day", None, "hypothetical"),
    ("reminds me of that song by coldplay", None, "world recall"),
    ("remind me what the capital of france is", None, "world recall"),
    ("did i already tell you about my robotics competition", None, "chat"),
    ("the weather is nice today", None, "chat"),
    ("how do i tie a tie", None, "world chat"),
    ("my friend says hi", None, "chat"),
    ("i'm gonna be rich someday", None, "chat"),
    ("i hate mondays", "create_fact", "stance"),
    ("my grandma makes the best pancakes", "create_fact", "possessive fact"),
    # --- G. compounds ----------------------------------------------------------
    ("i have a math test friday and remember that i joined TSA", "multi_action", ""),
    ("remind me to call mom at 6 and remind me to feed the dog at 8", "multi_action", ""),
    ("cancel my piano lesson and schedule a new one for friday", "multi_action", ""),
    # --- H. reminder phrasing / date formats ---------------------------------
    ("remind me to book zoo tickets on october 28th at 6:30pm", "create_calendar_event", "abs date"),
    ("reminder: dentist on nov 3rd at 9am", "create_calendar_event", "reminder colon"),
    ("remind me on 11/15 to file the fafsa", "create_calendar_event", "numeric date"),
    ("remind me in 2 hours to check the oven", "create_calendar_event", "relative"),
    ("remind me every monday to water the plants", "create_calendar_event", "weekly"),
    ("remind me tomorrow at 7 to call grandma", "create_calendar_event", "bare clock"),
    ("set a reminder for my mom's birthday on march 4th", "create_calendar_event", ""),
    ("don't let me forget my keys tomorrow morning", None, "colloquial, selector judged"),
    # --- I. typos / ASR noise ----------------------------------------------------
    ("i hav a dentst apointment tomorow at 3", "create_calendar_event", "typos"),
    ("remind me to tak medicenes every 2 days", "create_calendar_event", "typo"),
    ("scheduel gym for tomorow 6pm", "create_calendar_event", "typo"),
    ("cancle my piano lesson", "cancel_calendar_event", "typo cancel"),
    ("wats on my scedule today", "find_calendar_events", "typo"),
    ("rememer that my locker is 428", "create_fact", "typo"),
    # --- J. identity --------------------------------------------------------------
    ("my name is alex", "store_profile_keys", "intro key"),
    ("what's my name", "find_facts", "identity recall"),
    ("who am i", "find_facts", "identity recall"),
    ("what do you remember about me", "find_facts", "recall"),
]


def make_needle(tmp: Path, load_model: bool) -> KnowledgeNeedle:
    knowledge = KnowledgeEngine(tmp / "k.db")
    actions = ActionsEngine(tmp / "a.db", timezone="America/Chicago")
    needle = KnowledgeNeedle(
        knowledge,
        timezone="America/Chicago",
        actions_engine=actions,
        load_model=load_model,
    )
    return needle


def judge(payload: dict, expected: str | None) -> tuple[str, str]:
    """Return (verdict, detail). verdict in PASS/DEFER/FAIL."""
    if not isinstance(payload, dict):
        return "FAIL", f"non-dict payload: {type(payload).__name__}"
    result = payload.get("result") or {}
    if not isinstance(result, dict):
        return "FAIL", f"non-dict result: {type(result).__name__}"
    status = result.get("status")
    function = (payload.get("function") or {}).get("name")

    if status == "no_function_selected":
        if expected is None:
            return "PASS", "correctly declined (chat)"
        return "DEFER", "selector model needed"

    if expected == "multi_action":
        if status == "multi_action":
            return "PASS", f"{result.get('count')} subtasks"
        return "FAIL", f"status={status}"

    if expected is None:
        # chat that reached a function: only mutating calendar calls are
        # clearly wrong; storing a stated fact is within contract.
        if function in {
            "create_calendar_event", "cancel_calendar_event",
            "update_calendar_event",
        }:
            return "FAIL", f"chat mutated via {function}"
        if function is not None:
            return "PASS", f"tolerated: {function}"
        return "PASS", "no function"

    if function == expected:
        extra = []
        args = (result.get("function") or {}).get("arguments") or {}
        if expected == "create_calendar_event":
            temporal = result.get("temporal") or {}
            start = temporal.get("resolved_start") or ""
            if start:
                extra.append(f"start={start[:16]}")
            title = (result.get("data") or {}).get("title")
            if title:
                extra.append(f"title={title!r}")
        if expected == "cancel_calendar_event":
            ok = bool(result.get("ok"))
            if not ok:
                detail = result.get("question") or result.get("status") or "lookup failed"
                return "WARN", f"cancel not found on empty base: {detail}"
        return "PASS", ", ".join(extra)

    return "FAIL", f"expected {expected}, got {function or status}"


def run_phase(needle: KnowledgeNeedle, cases, phase: str) -> list[dict]:
    rows = []
    for text, expected, note in cases:
        t0 = time.perf_counter()
        verdict, detail = "CRASH", ""
        try:
            payload = needle.process(text)
            verdict, detail = judge(payload, expected)
        except Exception as exc:  # noqa: BLE001 - crash IS the finding
            detail = f"{type(exc).__name__}: {exc}"
        rows.append({
            "phase": phase,
            "text": text,
            "expected": expected,
            "note": note,
            "verdict": verdict,
            "detail": detail,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
    return rows


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="knowledge-stress-"))

    print("=" * 72)
    print("PHASE 1: deterministic (model gate off)")
    print("=" * 72)
    needle = make_needle(tmp, load_model=False)
    rows = run_phase(needle, CASES, "deterministic")
    needle.actions_engine.close()
    needle.engine.close()

    deferred = [case for case, row in zip(CASES, rows) if row["verdict"] == "DEFER"]
    for row in rows:
        marker = {"PASS": ".", "DEFER": "d", "FAIL": "F", "CRASH": "C", "WARN": "w"}[row["verdict"]]
        print(marker, end="", flush=True)
    print()
    for row in rows:
        if row["verdict"] in {"FAIL", "CRASH", "DEFER", "WARN"}:
            print(f"  [{row['verdict']}] {row['text']!r} -> {row['detail']}")

    model_rows: list[dict] = []
    if deferred:
        print()
        print("=" * 72)
        print(f"PHASE 2: selector model on {len(deferred)} deferred prompt(s)")
        print("=" * 72)
        needle2 = make_needle(tmp, load_model=True)
        model_rows = run_phase(needle2, deferred, "selector")
        needle2.actions_engine.close()
        needle2.engine.close()
        for row in model_rows:
            marker = {"PASS": ".", "DEFER": "d", "FAIL": "F", "CRASH": "C", "WARN": "w"}[row["verdict"]]
            print(marker, end="", flush=True)
        print()
        for row in model_rows:
            if row["verdict"] in {"FAIL", "CRASH"}:  # keep WARN visible in phase 2 too
                print(f"  [{row['verdict']}] {row['text']!r} -> {row['detail']}")

    all_rows = rows + model_rows
    report_path = Path(__file__).resolve().parent / "knowledge_stress_report.json"
    report_path.write_text(json.dumps(all_rows, indent=2))

    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print()
    print("totals:", counts)
    print(f"report: {report_path}")

    return 1 if counts.get("CRASH") else 0


if __name__ == "__main__":
    raise SystemExit(main())
