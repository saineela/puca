"""Archived Luna conversation-corpus builder; CLI disabled by current policy.

Historical helpers are retained for provenance. Do not regenerate Luna training
data: no Luna training is authorized. Preserve existing research artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "models" / "training_data"
DEFAULT_OUT = DATA_ROOT / "luna_conversation_sft.jsonl"
SYSTEM = (
    "You are Luna, a natural conversational model in the Nix PUCA system. "
    "Speak plainly and naturally. Answer the message directly, keep casual "
    "replies concise, and do not add a generic offer to help or a question "
    "after every response. Do not claim personal memories, feelings, or "
    "real-world actions."
)
REJECT = re.compile(
    r"(?:as an ai|language model|how can i help|what can i do for you|"
    r"please let me know|customer service|https?://|www\.)", re.I
)

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def valid(user: str, assistant: str) -> bool:
    return bool(user and assistant and len(user) <= 400 and len(assistant) <= 500
                and not REJECT.search(user) and not REJECT.search(assistant))

def pairs_from_daily(path: Path) -> Iterable[tuple[str, str, str]]:
    for file in sorted(path.rglob("*.txt")):
        for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            turns = [clean(x) for x in line.split("__eou__") if clean(x)]
            for user, answer in zip(turns, turns[1:]):
                yield user, answer, "daily_dialog"

def pairs_from_ubuntu(path: Path) -> Iterable[tuple[str, str, str]]:
    turns = [clean(x) for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if clean(x)]
    for user, answer in zip(turns, turns[1:]):
        yield user, answer, "ubuntu_dialogue"

def pairs_from_cornell(path: Path) -> Iterable[tuple[str, str, str]]:
    lines: dict[str, str] = {}
    for raw in (path / "movie_lines.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(" +++$+++ ")
        if len(parts) >= 5:
            lines[parts[0]] = clean(parts[4])
    for raw in (path / "movie_conversations.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(" +++$+++ ")
        if len(parts) < 4:
            continue
        ids = re.findall(r"L\d+", parts[3])
        for left, right in zip(ids, ids[1:]):
            if left in lines and right in lines:
                yield lines[left], lines[right], "cornell_movie_dialogue"

def row(user: str, answer: str, source: str) -> dict:
    return {"source": source, "messages": [
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
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    candidates: list[tuple[str, str, str]] = []
    daily = DATA_ROOT / "dailydialog_train"
    ubuntu = DATA_ROOT / "ubuntu" / "ubuntu_turns.txt"
    cornell = DATA_ROOT / "cornell"
    if daily.exists(): candidates.extend(pairs_from_daily(daily))
    if ubuntu.exists(): candidates.extend(pairs_from_ubuntu(ubuntu))
    if (cornell / "movie_lines.txt").exists() and (cornell / "movie_conversations.txt").exists():
        candidates.extend(pairs_from_cornell(cornell))
    unique = {(u.casefold(), a.casefold()): (u, a, s) for u, a, s in candidates if valid(u, a)}
    values = list(unique.values())
    random.Random(args.seed).shuffle(values)
    values = values[:args.limit]
    split = max(1, int(len(values) * (1.0 - args.eval_ratio)))
    train, evaluation = values[:split], values[split:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_path = args.output.with_name(args.output.stem + "_eval.jsonl")
    for path, rows in ((args.output, train), (eval_path, evaluation)):
        with path.open("w", encoding="utf-8") as handle:
            for user, answer, source in rows:
                handle.write(json.dumps(row(user, answer, source), ensure_ascii=False) + "\n")
    metadata = {
        "model": "Luna",
        "base": "unsloth/Llama-3.2-3B",
        "purpose": "conversation-only QLoRA SFT; independent of Casper",
        "sources": sorted({source for _, _, source in values}),
        "total": len(values), "train": len(train), "eval": len(evaluation),
        "seed": args.seed, "network_access": False,
        "note": "Review source licenses before redistribution; generated files remain ignored local artifacts.",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"train": str(args.output), "eval": str(eval_path), **metadata}, indent=2))

if __name__ == "__main__":
    main()
