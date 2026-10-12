"""
Ad-hoc verification for the spell-correction preprocessing layer.
Run directly: .venv/bin/python scripts/check_spelling.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nix_knowledge.spelling import correct_spelling

CASES = [
    # (input, expected) -- None means "must pass through unchanged"
    ("cancle my dentist appointment", "cancel my dentist appointment"),
    ("rememer to call grandma", "remember to call grandma"),
    ("scheduel robotics practice", "schedule robotics practice"),
    ("tomorow at 6:30am remind me to stretch", "tomorrow at 6:30am remind me to stretch"),
    ("my apointment got cancled", "my appointment got cancelled"),
    ("wats on my calendar tomorrow", "what's on my calendar tomorrow"),
    ("i have robotics practice tomorow at 5pm", "i have robotics practice tomorrow at 5pm"),
    ("htat is great", "that is great"),
    # 3-edit keyboard mash: outside any trustworthy edit budget, so
    # conservative pass-through is the CORRECT behavior here.
    ("scduedle a playdate", None),
    # --- guards: these must NOT change ---
    ("my robotics practice has been cancelled", None),
    ("robotics practice at 5pm tomorrow", None),
    ("TSA precheck at the airport friday", None),
    ("remind me to file the fafsa on 11/15", None),
    ("remind me in 2 hours to check the oven", None),
    ("my grandma makes the best pancakes", None),
    ("I am in the robotics club at school", None),
    ("meet Kayla and Maanvi at the park", None),
    ("cancel my piano lessons", None),
    ("was my robotics practice cancelled?", None),
    ("don't forget my meds at 9pm", None),
    ("what's on my calendar for next week", None),
    ("", ""),
    ("123 456 at 7:30", None),
]

failed = 0
for raw, expected in CASES:
    want = raw if expected is None else expected
    got = correct_spelling(raw)
    status = "ok  " if got == want else "FAIL"
    if got != want:
        failed += 1
    print(f"{status} {raw!r:55} -> {got!r}")

# Latency probe: worst case = every token unknown
start = time.perf_counter()
for _ in range(200):
    correct_spelling("cancle my apointment with the orthodontist tomorow at 5pm pleas")
elapsed = (time.perf_counter() - start) / 200 * 1000
print(f"\navg latency (4 unknown tokens): {elapsed:.2f} ms/sentence")

print(f"\n{failed}/{len(CASES)} failed")
sys.exit(1 if failed else 0)
