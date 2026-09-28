"""Archived Luna public-resource corpus builder; CLI disabled by policy.

Historical helpers are retained for provenance. Do not regenerate Luna training
data: no Luna training is authorized. Preserve existing research artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from luna_control_data import expanded_controls

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "models" / "training_data"
FINETOME = DATA_ROOT / "public_resources" / "finetome-100k" / "data" / "train-00000-of-00001.parquet"
DEFAULT_OUT = DATA_ROOT / "luna_resource_mix_v2_sft.jsonl"
SYSTEM = (
    "You are Luna, a natural conversational model in the Nix PUCA system. "
    "Answer directly and naturally. Keep casual replies concise. Do not use "
    "generic assistant boilerplate, claim personal memories or real-world "
    "actions, or append an unnecessary question."
)
REJECT = re.compile(
    r"(?:as an ai|language model|ai assistant|what can i do for you|"
    r"how can i help you today|please let me know if you need|customer service|"
    r"https?://|www\.|system prompt|chain of thought|step[- ]by[- ]step reasoning)",
    re.I,
)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def valid(user: str, assistant: str, *, max_answer: int = 700) -> bool:
    return bool(
        user
        and assistant
        and len(user) <= 500
        and len(assistant) <= max_answer
        and not REJECT.search(user)
        and not REJECT.search(assistant)
    )


def pair_row(user: str, assistant: str, source: str, category: str = "") -> dict[str, Any]:
    return {
        "source": source,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def local_pairs() -> Iterable[tuple[str, str, str, str]]:
    daily = DATA_ROOT / "dailydialog_train"
    for file in sorted(daily.rglob("*.txt")):
        for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            turns = [clean(x) for x in line.split("__eou__") if clean(x)]
            for user, answer in zip(turns, turns[1:]):
                yield user, answer, "daily_dialog", "conversation"

    ubuntu = DATA_ROOT / "ubuntu" / "ubuntu_turns.txt"
    if ubuntu.exists():
        turns = [
            clean(x)
            for x in ubuntu.read_text(encoding="utf-8", errors="ignore").splitlines()
            if clean(x)
        ]
        for user, answer in zip(turns, turns[1:]):
            yield user, answer, "ubuntu_dialogue", "conversation"


def _dataset_messages(messages: Any) -> tuple[str, str] | None:
    """Extract the first user→assistant pair from HF chat messages."""
    if not isinstance(messages, list):
        return None
    normalized: list[tuple[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or "").lower()
        content = clean(item.get("content") or item.get("value"))
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "model"}:
            role = "assistant"
        else:
            continue
        normalized.append((role, content))
    for (left_role, left), (right_role, right) in zip(normalized, normalized[1:]):
        if left_role == "user" and right_role == "assistant":
            return left, right
    return None


def finetome_pairs(limit: int) -> Iterable[tuple[str, str, str, str]]:
    if not FINETOME.exists():
        return
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(FINETOME), split="train")
    count = 0
    for item in dataset:
        pair = _dataset_messages(item.get("conversations"))
        if pair and valid(*pair):
            yield pair[0], pair[1], "finetome_100k", "instruction_conversation"
            count += 1
            if count >= limit:
                return


def no_robots_pairs(limit: int) -> Iterable[tuple[str, str, str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/no_robots", split="train")
    count = 0
    for item in dataset:
        pair = _dataset_messages(item.get("messages"))
        if pair and valid(*pair, max_answer=900):
            yield pair[0], pair[1], "no_robots", clean(item.get("category")) or "instruction"
            count += 1
            if count >= limit:
                return


def ultrafeedback_pairs(limit: int) -> Iterable[tuple[str, str, str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")
    count = 0
    for item in dataset:
        # `messages` is the chosen SFT conversation in this split. We do not
        # train on rejected text; preference data is not treated as dialogue.
        pair = _dataset_messages(item.get("messages"))
        if pair and valid(*pair, max_answer=1000):
            yield pair[0], pair[1], "ultrafeedback_binarized_chosen", "chosen_sft"
            count += 1
            if count >= limit:
                return


LUNA_CONTROL: tuple[tuple[str, str, str], ...] = (
    ("Hey, are you there?", "Yeah, I'm here.", "social"),
    ("I got some really good news today.", "Oh, nice. That's a good feeling.", "positive"),
    ("I'm exhausted and don't want a long conversation.", "Okay. I'll keep it short.", "tired"),
    ("I don't want advice right now.", "Okay. No advice.", "boundary"),
    ("Do you prefer strawberry or chocolate?", "I'd go with chocolate.", "preference"),
    ("I'm worried about my dad.", "That sounds worrying. What happened?", "emotion"),
    ("What are you?", "I'm Luna, a conversational model in the Nix PUCA system.", "identity"),
    ("Do you remember my sister?", "I don't have that stored here, so I won't guess.", "memory_honesty"),
    ("Who created Casper?", "Casper was created and built by Sai Neela, in NIX's PUCA system.", "identity_boundary"),
    ("My sister is doing well.", "Which sister do you mean?", "clarification"),
    ("Keep this brief.", "Yep.", "brevity"),
    ("I changed my mind.", "Fair enough.", "natural"),
    ("You misunderstood me.", "Right—I took that the wrong way.", "repair"),
    ("Tell me a joke.", "Why did the tomato blush? It saw the salad dressing.", "creative"),
    ("I finished the thing I was putting off.", "Nice. That's a good feeling.", "positive"),
    ("I need quiet.", "Okay.", "boundary"),
    ("I'm angry.", "You sound angry. I won't lecture you.", "emotion"),
    ("Thanks for remembering.", "Of course.", "natural"),
)


def sample_source(
    rows: list[tuple[str, str, str, str]], limit: int, rng: random.Random
) -> list[tuple[str, str, str, str]]:
    unique = {(u.casefold(), a.casefold()): (u, a, s, c) for u, a, s, c in rows if valid(u, a)}
    values = list(unique.values())
    rng.shuffle(values)
    return values[:limit]


def main() -> None:
    raise PermissionError(
        "Archived Luna dataset generation is disabled; preserve existing "
        "artifacts. No Luna training is authorized."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-limit", type=int, default=9000)
    parser.add_argument("--finetome-limit", type=int, default=1000)
    parser.add_argument("--no-robots-limit", type=int, default=1500)
    parser.add_argument("--ultrafeedback-limit", type=int, default=1800)
    parser.add_argument("--control-limit", type=int, default=240)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    source_rows: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for row in local_pairs():
        source_rows[row[2]].append(row)
    source_rows["finetome_100k"] = list(finetome_pairs(args.finetome_limit))
    source_rows["no_robots"] = list(no_robots_pairs(args.no_robots_limit))
    source_rows["ultrafeedback_binarized_chosen"] = list(
        ultrafeedback_pairs(args.ultrafeedback_limit)
    )
    source_rows["luna_control"] = [
        (user, answer, "luna_control", category)
        for user, answer, category in expanded_controls(repeats=2)
    ]

    limits = {
        "daily_dialog": args.local_limit,
        "ubuntu_dialogue": args.local_limit,
        "finetome_100k": args.finetome_limit,
        "no_robots": args.no_robots_limit,
        "ultrafeedback_binarized_chosen": args.ultrafeedback_limit,
        "luna_control": args.control_limit,
    }
    selected: list[tuple[str, str, str, str]] = []
    selected_by_source: dict[str, list[tuple[str, str, str, str]]] = {}
    for source, rows in source_rows.items():
        picked = sample_source(rows, limits.get(source, len(rows)), rng)
        selected_by_source[source] = picked
        selected.extend(picked)

    # Split within each source so evaluation retains coverage of each dataset.
    train: list[tuple[str, str, str, str]] = []
    evaluation: list[tuple[str, str, str, str]] = []
    for source, rows in selected_by_source.items():
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * (1.0 - args.eval_ratio))) if rows else 0
        train.extend(rows[:cut])
        evaluation.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(evaluation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_path = args.output.with_name(args.output.stem + "_eval.jsonl")
    for path, rows in ((args.output, train), (eval_path, evaluation)):
        with path.open("w", encoding="utf-8") as handle:
            for user, answer, source, category in rows:
                handle.write(json.dumps(pair_row(user, answer, source, category), ensure_ascii=False) + "\n")

    metadata = {
        "model": "Luna",
        "base": "unsloth/Llama-3.2-3B",
        "purpose": "balanced local SFT for base-model conversational control",
        "train": len(train),
        "eval": len(evaluation),
        "source_counts_total": dict(Counter(source for _, _, source, _ in selected)),
        "source_counts_train": dict(Counter(source for _, _, source, _ in train)),
        "source_counts_eval": dict(Counter(source for _, _, source, _ in evaluation)),
        "seed": args.seed,
        "revisions": {
            "no_robots": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
            "ultrafeedback_binarized": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
            "finetome_100k": "c2343c1372ff31f51aa21248db18bffa3193efdb",
        },
        "licenses": {
            "no_robots": "CC-BY-NC-4.0; local/non-commercial unless terms permit otherwise",
            "ultrafeedback_binarized": "MIT",
            "finetome_100k": "not declared on card; derived from The-Tome; local-only",
            "daily_dialog": "CC BY-NC-SA 4.0; review source terms",
            "ubuntu_dialogue": "source terms require review",
            "luna_control": "project-authored",
        },
        "filters": [
            "assistant boilerplate and AI identity phrases",
            "URLs and prompt-injection/chain-of-thought text",
            "overlong or malformed turns",
            "rejected UltraFeedback completions excluded",
        ],
        "redistribution": "Do not redistribute generated mix until every upstream source is cleared.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"train_path": str(args.output), "eval_path": str(eval_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
