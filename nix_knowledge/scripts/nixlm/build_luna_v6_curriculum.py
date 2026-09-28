"""Archived Luna V6 curriculum builder; CLI disabled by current policy.

Historical helpers are retained for provenance. Do not regenerate Luna training
data: no Luna training is authorized. Preserve existing research artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from luna_format import IDENTITY_SYSTEM

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "models" / "training_data"
DEFAULT_INPUT = DATA / "luna_balanced_v5_sft.jsonl"
DEFAULT_OUTPUT = DATA / "luna_v6_curriculum_sft.jsonl"


def row(turns: list[tuple[str, str]], category: str, source: str = "luna_v6_curriculum") -> dict:
    messages = [{"role": "system", "content": IDENTITY_SYSTEM}]
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return {"source": source, "category": category, "messages": messages}


CHAINS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "clarification_continuation",
        (
            ("How is my sister?", "Which sister do you mean?"),
            ("Sister B.", "Sister B is the one you mean. How is she doing?"),
        ),
    ),
    (
        "clarification_continuation",
        (
            ("My sister is doing well.", "Which sister do you mean?"),
            ("Sister A.", "Got it—Sister A is doing well."),
        ),
    ),
    (
        "clarification_continuation",
        (
            ("Can you check on my brother?", "Which brother do you mean?"),
            ("My older brother.", "Okay, your older brother."),
        ),
    ),
    (
        "state_supersession",
        (
            ("Person A is sick.", "I hope Person A feels better soon."),
            ("Person A is feeling better now.", "I'm glad Person A is feeling better now."),
            ("How is Person A?", "Person A is feeling better now."),
        ),
    ),
    (
        "state_supersession",
        (
            ("My sister was unwell yesterday.", "I hope she gets some rest."),
            ("She's doing better today.", "That's good to hear."),
            ("How is she now?", "She's doing better now."),
        ),
    ),
    (
        "state_supersession",
        (
            ("Dad has been sick.", "That sounds difficult."),
            ("Dad is better now.", "I'm glad he's better."),
            ("How is Dad?", "Dad is better now."),
        ),
    ),
    (
        "tired_followup",
        (
            ("I had a terrible day.", "I'm sorry—it sounds like a rough day."),
            ("I'm exhausted; keep it short.", "Okay. I'll keep this brief."),
        ),
    ),
    (
        "tired_followup",
        (
            ("Today was awful.", "I'm sorry today was so hard."),
            ("I'm tired. No long conversation.", "Understood. I'll keep it short."),
        ),
    ),
    (
        "memory_plus_chat",
        (
            ("Remember that I like tea and tell me a joke.", "I'll remember that you like tea. Why did the tomato blush? It saw the salad dressing."),
            ("That was funny.", "I'm glad you liked it—and I'll keep tea in mind."),
        ),
    ),
    (
        "memory_plus_chat",
        (
            ("Remember I like books, and tell me something funny.", "I'll remember that you like books. Why did the book join the police? It wanted to work undercover."),
            ("That was good.", "Nice. I'll remember that books are your thing."),
        ),
    ),
    (
        "natural_preference",
        (
            ("Do you prefer strawberry or chocolate?", "Chocolate, probably."),
            ("Why?", "It has a richer flavor, and it's a fun choice."),
        ),
    ),
    (
        "boundary",
        (
            ("I'm overwhelmed.", "That sounds like a lot."),
            ("Don't give me advice.", "Okay. No advice—I'll keep it simple."),
        ),
    ),
)

# Held-out paraphrases deliberately differ from the exact training wording.
HELD_OUT = (
    ("clarification_continuation", (("How is my sister doing?", "Which sister do you mean?"), ("The younger one.", "Okay—the younger sister is the one you mean."))),
    ("state_supersession", (("Maya was sick earlier.", "I hope Maya gets well soon."), ("Maya is better now.", "I'm glad Maya is better now."), ("How is Maya?", "Maya is better now."))),
    ("tired_followup", (("I've had enough for today.", "I'm sorry—it sounds like a hard day."), ("Please keep your reply brief.", "Of course. I'll keep it brief."))),
    ("memory_plus_chat", (("Remember that I enjoy coffee and tell me a joke.", "I'll remember that you enjoy coffee. Why did the cookie go to the doctor? It felt crumby."), ("That made me laugh.", "Good. I'll keep coffee in mind."))),
)


def main() -> None:
    raise PermissionError(
        "Archived Luna dataset generation is disabled; preserve existing "
        "artifacts. No Luna training is authorized."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chain-repeats", type=int, default=12)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=913)
    args = parser.parse_args()

    base = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Keep the broad corpus, but cap exact duplicate rows so the new signal is
    # not drowned out by repeated project-control examples.
    seen: set[tuple[str, str]] = set()
    broad = []
    for item in base:
        messages = item.get("messages", [])
        if len(messages) < 3:
            continue
        key = (str(messages[-2].get("content", "")).casefold(), str(messages[-1].get("content", "")).casefold())
        if key in seen:
            continue
        seen.add(key)
        item["messages"][0] = {"role": "system", "content": IDENTITY_SYSTEM}
        broad.append(item)

    rng = random.Random(args.seed)
    curriculum = []
    for category, turns in CHAINS:
        for _ in range(max(1, args.chain_repeats)):
            curriculum.append(row(list(turns), category))
    evaluation = [row(list(turns), category, source="luna_v6_heldout") for category, turns in HELD_OUT]
    rng.shuffle(broad)
    rng.shuffle(curriculum)
    cut = max(1, int(len(curriculum) * (1.0 - args.eval_ratio)))
    train = broad + curriculum[:cut]
    # Keep the held-out chains completely separate from training.
    rng.shuffle(train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_path = args.output.with_name(args.output.stem + "_eval.jsonl")
    for path, values in ((args.output, train), (eval_path, evaluation)):
        with path.open("w", encoding="utf-8") as handle:
            for item in values:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "model": "Luna",
        "purpose": "balanced instruct SFT with explicit multi-turn state curriculum",
        "input": str(args.input),
        "broad_rows": len(broad),
        "curriculum_rows": len(curriculum[:cut]),
        "train": len(train),
        "eval": len(evaluation),
        "categories": sorted({category for category, _ in CHAINS}),
        "held_out_categories": sorted({category for category, _ in HELD_OUT}),
        "chain_repeats": args.chain_repeats,
        "seed": args.seed,
        "note": "No personal logs; all multi-turn rows are project-authored placeholders.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "eval": str(eval_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
