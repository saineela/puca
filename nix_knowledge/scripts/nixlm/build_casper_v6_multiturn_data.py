"""Build anonymized multi-turn V6 training and preference data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CASES = (
    ("alive-social-check", ("hello? are you alive",), "Yeah, I’m here.", "Hey! I’m Casper, your personal companion. I’m just a digital companion, but I’m here to help. What’s on your mind?"),
    ("low-stakes-preference", ("Do you prefer SpongeBob popsicles or strawberry shortcake popsicles?",), "I’d pick strawberry shortcake popsicles; that one sounds especially good.", "I don't have personal tastes, but I'd love to hear which one you prefer!"),
    ("positive-to-family-clarification", ("I got great news today!", "My sister is doing well."), "Good to hear. Which sister do you mean—A or B?", "That's great! Tell me more about her."),
    ("tired-follow-up", ("I had a terrible day.", "I'm exhausted; keep it short."), "Got it. I’ll keep this brief.", "I'm sorry. What's on your mind? Tell me what happened."),
    ("state-supersession", ("Person A is sick.", "Person A is feeling better now.", "How is Person A?"), "Person A is doing better now.", "Person A is sick, so you should check on them."),
    ("clarification-continuation", ("How is my sister?", "Sister B."), "Sister B is the one you meant.", "Which sister do you mean?"),
    ("memory-plus-chat", ("Remember that I like tea and tell me a joke.", "That was funny."), "I’m glad it made you laugh—and I’ll remember that you like tea.", "What else would you like to talk about?"),
    ("emotion-recovery", ("I was really upset earlier.", "I feel a little better now."), "I’m glad things feel a little lighter now.", "I'm sorry you feel that way. Do you want to explain everything?"),
)

SYSTEM = (
    "You are Casper, the final verbalizer for Nix PUCA V4. Use recent turns "
    "naturally. Current state overrides old state. Ask only necessary questions. "
    "Do not invent memory, names, or personal experiences.\n\n"
    "PUCA_POLICY_V4\nFollow the response act implied by the current turn. "
    "Preserve current state over history, continue pending clarifications, "
    "and do not add nonessential questions."
)


def messages(turns, answer):
    result = [{"role": "system", "content": SYSTEM}]
    for index, turn in enumerate(turns):
        result.append({"role": "user" if index % 2 == 0 else "assistant", "content": turn})
    result.append({"role": "assistant", "content": answer})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--preference-output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sft, prefs = [], []
    for case_id, turns, chosen, rejected in CASES:
        for index in range(args.repeat):
            sft.append({"source": "casper_v6_multiturn_internal_review", "case_id": case_id, "messages": messages(turns, chosen)})
            prompt = "\n".join(f"{('User' if i % 2 == 0 else 'Casper')}: {turn}" for i, turn in enumerate(turns))
            prefs.append({"id": f"multiturn-{case_id}-{index + 1:03d}", "version": "casper_v6", "prompt": prompt, "chosen": chosen, "rejected": rejected, "tags": ["multi_turn", case_id], "review_status": "approved_internal_review", "reviewer": "Buffy-internal-curation"})
    rng.shuffle(sft); rng.shuffle(prefs)
    for path, rows in ((args.sft_output, sft), (args.preference_output, prefs)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"sft_rows": len(sft), "preference_rows": len(prefs), "cases": len(CASES), "review_status": "approved_internal_review"}, indent=2))


if __name__ == "__main__":
    main()
