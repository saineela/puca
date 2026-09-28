"""Archived Luna clean-mix builder; CLI disabled by current policy.

Historical helpers are retained for provenance. Do not regenerate Luna training
data: no Luna training is authorized. Preserve existing research artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "models" / "training_data"
DEFAULT_INPUT = DATA / "luna_resource_mix_v2_sft.jsonl"
DEFAULT_OUTPUT = DATA / "luna_clean_mix_v3_sft.jsonl"
REJECT = re.compile(
    r"(?:confidence\s*[:=]|confidenc\w*\s*[:=]|as an ai|language model|"
    r"what can i do for you|how can i help you today|please let me know|"
    r"chain of thought|system prompt|https?://|www\.)",
    re.I,
)

CONTROL = [
    ("Hey, are you there?", "Yeah, I'm here.", "social"),
    ("hello? are you alive", "Yep, I'm here.", "social"),
    ("I got some really good news today.", "Oh, nice. That's a good feeling.", "positive"),
    ("I'm exhausted and don't want a long conversation.", "Okay. I'll keep it short.", "tired"),
    ("I'm tired; keep it brief.", "Got it. Short and simple.", "tired"),
    ("I don't want advice right now.", "Okay. No advice.", "boundary"),
    ("I need quiet, not a lecture.", "Fair. I'll keep it quiet.", "boundary"),
    ("Do you prefer SpongeBob popsicles or strawberry shortcake popsicles?", "I'd pick strawberry shortcake.", "preference"),
    ("I am trying to eat healthier. Which popsicle would you choose?", "Strawberry shortcake sounds like the better pick.", "preference"),
    ("I'm worried about my dad.", "That sounds worrying. What happened?", "emotion"),
    ("I'm angry.", "You sound angry. I won't lecture you.", "emotion"),
    ("What are you?", "I'm Luna, a conversational model in the Nix PUCA system.", "identity"),
    ("Who created Casper?", "Casper was created and built by Sai Neela, in NIX's PUCA system.", "identity"),
    ("Do you remember my sister?", "I don't have that stored here, so I won't guess.", "memory"),
    ("My sister is doing well.", "Which sister do you mean?", "clarification"),
    ("Keep this brief.", "Yep.", "brevity"),
    ("You misunderstood me.", "Right—I took that the wrong way.", "repair"),
    ("Tell me a joke.", "Why did the tomato blush? It saw the salad dressing.", "creative"),
    ("Remember that I like tea and tell me a joke.", "I'll remember that you like tea. And here's one: why did the tomato blush? It saw the salad dressing.", "multi_intent"),
]
SYSTEM = (
    "You are Luna, a warm and natural conversational model in the Nix PUCA system. "
    "Answer the user's latest message directly. Keep ordinary replies concise. "
    "Do not continue unrelated example conversations, emit confidence scores, "
    "invent memories or actions, or add a question unless it is necessary."
)

def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()

def usable(row: dict) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    user = compact(messages[-2].get("content"))
    answer = compact(messages[-1].get("content"))
    return bool(user and answer and len(user) <= 500 and len(answer) <= 700 and not REJECT.search(user + "\n" + answer))

def make_row(user: str, answer: str, source: str, category: str) -> dict:
    return {"source": source, "category": category, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]}

def main() -> None:
    raise PermissionError(
        "Archived Luna dataset generation is disabled; preserve existing "
        "artifacts. No Luna training is authorized."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--control-repeats", type=int, default=24)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source = row.get("source", "")
            # Ubuntu extraction was intentionally excluded after the V2
            # benchmark showed unrelated technical continuation and repetition.
            if source == "ubuntu_dialogue" or not usable(row):
                continue
            row["messages"][0] = {"role": "system", "content": SYSTEM}
            rows.append(row)
    controls = [
        make_row(user, answer, "luna_control", category)
        for _ in range(args.control_repeats)
        for user, answer, category in CONTROL
    ]
    rows.extend(controls)
    unique = {}
    retained_controls = []
    for row in rows:
        messages = row["messages"]
        if row["source"] == "luna_control":
            # Repetition is deliberate here: project-authored behavior is the
            # corrective signal, not an accidental duplicate from a crawler.
            retained_controls.append(row)
            continue
        key = (compact(messages[-2]["content"]).casefold(), compact(messages[-1]["content"]).casefold(), row["source"])
        unique[key] = row
    rows = list(unique.values()) + retained_controls
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * (1 - args.eval_ratio)))
    train, evaluation = rows[:cut], rows[cut:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_path = args.output.with_name(args.output.stem + "_eval.jsonl")
    for path, values in ((args.output, train), (eval_path, evaluation)):
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "model": "Luna",
        "base": "unsloth/Llama-3.2-3B",
        "purpose": "clean base-model conversational control candidate",
        "train": len(train), "eval": len(evaluation), "seed": args.seed,
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "excluded": {"ubuntu_dialogue": "unrelated adjacent-line pairing caused technical continuation failures"},
        "included": {"no_robots": "filtered small supplement", "ultrafeedback_binarized_chosen": "filtered chosen SFT supplement", "luna_control": "project-authored and oversampled"},
        "filters": ["confidence/repetition contamination", "AI boilerplate", "URLs and prompt-injection text", "overlong turns"],
        "upstream_license_note": "No Robots is CC-BY-NC-4.0; review terms before redistribution. UltraFeedback is MIT. Keep this generated mix local until all sources are cleared.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "eval": str(eval_path), **metadata}, indent=2))

if __name__ == "__main__":
    main()
