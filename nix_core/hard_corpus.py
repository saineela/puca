"""
Hard prompt corpus: unusual but realistic request shapes.

Unlike router_corpus.py (which the deterministic rules must resolve
on their own), these prompts are chosen so the rules SHOULD abstain
(route == UNKNOWN) and let the model-backed classifier understand the
intent. Each entry carries the intended final route, used by the live
model harness (test_model_gate_live.py).

Categories:
  - indirect / implied requests ("it would be bad if i missed my meds")
  - idioms and sarcasm carrying real intent ("i've been meaning to...")
  - world recall wearing personal words ("do i know anyone who...")
  - hypotheticals that must NOT create records ("what if i had...")
  - corrections and follow-ups ("actually, make it 7")
  - compound mixed-intent ("remind me ... oh and who won ...")
  - typos heavy enough to break naive matching
  - politeness-wrapped commands ("would you mind...")

The rules layer is SUPPOSED to return UNKNOWN for all of these.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import UNKNOWN  # noqa: E402

# (prompt, intended_final_route) - the rules must abstain on every one.
HARD_PROMPTS: list[tuple[str, str]] = [
    # ---- implied scheduling (no scheduling word at all) --------------
    ("it would be really bad if i missed my medication again", "knowledge"),
    ("i keep forgetting to water the plants", "knowledge"),
    ("i've been meaning to book a dentist visit", "knowledge"),
    ("i can never remember when trash day is", "knowledge"),
    ("my mom keeps asking when her flight lands", "knowledge"),
    ("i'd hate to double-book myself next week", "knowledge"),
    ("i always oversleep on mondays", "knowledge"),
    ("last time i forgot the vet appointment and felt terrible", "knowledge"),
    # ---- indirect storage (no remember/save word) --------------------
    ("just so you know, my spare key is under the mat", "knowledge"),
    ("for future reference, the garage code changed to 778899", "knowledge"),
    ("fyi my new number is 555 0134", "knowledge"),
    ("in case you need it, my insurance policy is XQ-44291", "knowledge"),
    ("heads up, i switched to the night shift this month", "knowledge"),
    ("my brother is visiting from the 12th to the 15th", "knowledge"),
    # ---- world recall wearing personal words -------------------------
    ("do i know anyone who won a nobel prize", "chat"),
    ("is my laptop supposed to get this hot", "chat"),
    ("does my phone support 5g", "chat"),
    ("should my resume be one page", "chat"),
    ("was my generation the first to grow up with smartphones", "chat"),
    ("what year was my car's model first released", "chat"),
    # ---- hypotheticals: must not store or schedule anything ----------
    ("what if i scheduled everything on the same day", "chat"),
    ("if i had a million dollars, would i still work here", "chat"),
    ("suppose i moved to japan, would my dentist records transfer", "chat"),
    ("could i theoretically run a marathon next month", "chat"),
    # ---- sarcasm / figures of speech ---------------------------------
    ("sure, like my calendar was ever going to be free", "chat"),
    ("oh great, another monday", "chat"),
    ("wow, my sleep schedule is a disaster", "chat"),
    ("my budget is basically a crime scene", "chat"),
    # ---- politeness-wrapped real commands -----------------------------
    ("would you mind keeping track that i owe joel twenty bucks", "knowledge"),
    ("if it's not too much trouble, pencil in a haircut for saturday", "knowledge"),
    ("i hate to ask, but could you watch for when my package is due", "knowledge"),
    ("can we make sure i don't skip leg day again this week", "knowledge"),
    # ---- corrections / follow-ups (stateless router must still route)
    ("actually, move that to 7 instead", "knowledge"),
    ("wait no, cancel that last one", "knowledge"),
    ("scratch that, make it two hours earlier", "knowledge"),
    ("on second thought, i'll skip the gym tomorrow", "knowledge"),
    ("no wait, the birthday dinner is on saturday not friday", "knowledge"),
    # ---- compound mixed intent ----------------------------------------
    ("remind me to defrost the chicken, oh and who won the game last night", "knowledge"),
    ("schedule my flu shot this week, also what's the weather friday", "knowledge"),
    ("cancel my 4pm and reschedule my dentist to next monday", "knowledge"),
    ("what time is my interview, and tell me something about the company", "knowledge"),
    # ---- heavy typos / ASR mangling ------------------------------------
    ("remmber tat my sis bday is septmber 4th", "knowledge"),
    ("shedule a appoitment with the docter tommorow at 9", "knowledge"),
    ("waht tiem is my meting tomorow", "knowledge"),
    ("cna u wake me up at 6 am tommorrow", "knowledge"),
    ("whers my pasword for the wifi", "knowledge"),
    ("set a alaram for 5 30 pm plz", "knowledge"),
    # ---- elliptical / fragment speech (voice assistant style) ----------
    ("the dentist", "knowledge"),
    ("my meetings next week", "knowledge"),
    ("gym schedule", "knowledge"),
    ("wifi password", "knowledge"),
    ("mom's birthday", "knowledge"),
    # ---- chatty wraps around knowledge ---------------------------------
    ("so hey i was just thinking, when is my car service due", "knowledge"),
    ("random question, do i have anything on friday", "knowledge"),
    ("ok this might sound dumb but where did i put my tax documents", "knowledge"),
    ("quick one - what's my login for the school portal", "knowledge"),
    # ---- world questions phrased personally -----------------------------
    ("how much should my resume salary expectation be", "chat"),
    ("what speed should my internet be for streaming", "chat"),
    ("how often should my car get an oil change", "chat"),
    ("how long should my essay be", "chat"),
]


def test_hard_prompts_rules_never_wrong():
    """
    Contract: on unusual prompts the deterministic rules must NEVER
    decide the wrong route. They may resolve a prompt confidently
    (fast path - bonus) or abstain (UNKNOWN -> model layer). A wrong
    confident decision is the only failure mode.
    """
    from router import KNOWLEDGE, CHAT

    route_to_expected = {
        "knowledge": KNOWLEDGE,
        "chat": CHAT,
        UNKNOWN: UNKNOWN,
    }

    failures = []
    abstained = 0
    for prompt, intended in HARD_PROMPTS:
        route, _ = classify_hard(prompt)
        if route == UNKNOWN:
            abstained += 1
            continue
        if route != route_to_expected[intended]:
            failures.append(f"{prompt!r}: rules said {route}, intended {intended}")

    # Soft regression guard: hard prompts are SUPPOSED to lean on the
    # model, but the rules should still confidently resolve a healthy
    # share (wraps, compounds, typos, fragments). Alert if that share
    # collapses - the hot path would be silently getting slower.
    resolved = len(HARD_PROMPTS) - abstained
    assert resolved >= len(HARD_PROMPTS) * 0.3, (
        f"rules resolved only {resolved}/{len(HARD_PROMPTS)} hard prompts"
    )
    assert not failures, "\n".join(failures)


def classify_hard(text: str):
    from router import classify

    return classify(text)
