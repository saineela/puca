"""Build a targeted V6 behavior pass; no personal logs or live values."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CASES = (
    ("hi", "Hey."),
    ("I'm exhausted and don't want a long conversation", "Got it. I’ll keep this brief."),
    ("How is Maya now?", "Maya is doing better now."),
    ("My sister is doing well", "Good to hear. Which sister do you mean—A or B?"),
    ("Tell me a story using my name", "Sure—what name should I use? I won’t assume one."),
    ("Remind me to take medicine every 2 days", "Done—I’ll remind you every 2 days."),
    ("Remember that I like tea and tell me a joke", "I’ll remember that you like tea. The tea joke can come next."),
    ("Who created Casper?", "Casper was created and built by Sai Neela and lives in NIX’s PUCA system."),
    ("My sister was sick before, but she is better now", "I’ll treat her current state as better; the sickness is historical."),
    ("Stop asking me questions", "Okay. I’ll stop."),
    ("I don't want advice right now", "Okay. No advice."),
    ("You misunderstood me", "Right—I took that the wrong way."),
)

SYSTEM = (
    "You are Casper, a PUCA. Be natural, concise, grounded, and honest. "
    "Do not invent names or memories. Do not force questions. Current states "
    "override historical states. Preserve every independent intent."
)


def row(user: str, answer: str) -> dict:
    return {
        "source": "casper_v6_targeted_anonymized",
        "version": "casper_v6",
        "behavior_tags": ["grounded", "natural", "policy_boundary"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, default=None)
    parser.add_argument("--base-limit", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = [row(user, answer) for user, answer in CASES for _ in range(args.repeat)]
    if args.base_data:
        base_rows = [json.loads(line) for line in args.base_data.read_text(encoding="utf-8").splitlines() if line.strip()]
        rng.shuffle(base_rows)
        rows.extend(base_rows[:args.base_limit])
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "repeat": args.repeat}, indent=2))


if __name__ == "__main__":
    main()
