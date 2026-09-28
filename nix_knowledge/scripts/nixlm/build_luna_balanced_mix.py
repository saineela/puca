"""Archived Luna identity-aware corpus builder; CLI disabled by policy.

Historical helpers are retained for provenance. Do not regenerate Luna training
data: no Luna training is authorized. Preserve existing research artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from luna_format import IDENTITY_SYSTEM
from luna_role_control import make_row, role_rows

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "models" / "training_data"
DEFAULT_CONVERSATION = DATA / "luna_clean_mix_v4_sft.jsonl"
DEFAULT_IDENTITY = DATA / "luna_identity_mix_v1_sft.jsonl"
DEFAULT_OUTPUT = DATA / "luna_balanced_v5_sft.jsonl"


def key(row: dict) -> tuple[str, str]:
    messages = row["messages"]
    return (
        str(messages[-2].get("content", "")).strip().casefold(),
        str(messages[-1].get("content", "")).strip().casefold(),
    )


def main() -> None:
    raise PermissionError(
        "Archived Luna dataset generation is disabled; preserve existing "
        "artifacts. No Luna training is authorized."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", type=Path, default=DEFAULT_CONVERSATION)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--identity-limit", type=int, default=260)
    parser.add_argument("--role-repeat", type=int, default=8)
    parser.add_argument("--identity-train-repeat", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    conversation = [json.loads(line) for line in args.conversation.read_text(encoding="utf-8").splitlines() if line.strip()]
    identity = [json.loads(line) for line in args.identity.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Preserve ordinary conversation rows, but remove the old identity source's
    # Casper/Nix material. Role knowledge is reintroduced below through a
    # deliberately authored, semantically consistent separation set.
    identity = [
        row for row in identity
        if str(row.get("category", "")).casefold() in {"luna_identity", "luna_boundary", "model_boundary", "memory_honesty", "clarification"}
    ][: args.identity_limit]
    conversation = list(conversation)
    role_controls = [make_row(*item) for item in role_rows(repeats=args.role_repeat)]
    # Keep Luna identity and role knowledge as small correction signals rather
    # than replacing the broad conversational distribution.
    merged: dict[tuple[str, str], dict] = {}
    for row in conversation + identity + role_controls:
        if isinstance(row.get("messages"), list) and len(row["messages"]) >= 3:
            row["messages"][0] = {"role": "system", "content": IDENTITY_SYSTEM}
            merged.setdefault(key(row), row)
    rows = list(merged.values())
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Split by source/category so identity is present in both train and eval.
    identity_rows = [row for row in rows if row.get("source") == "luna_identity_control"]
    role_rows_selected = [row for row in rows if row.get("source") == "luna_role_control"]
    other_rows = [row for row in rows if row.get("source") not in {"luna_identity_control", "luna_role_control"}]
    rng.shuffle(identity_rows); rng.shuffle(other_rows)
    identity_cut = max(1, int(len(identity_rows) * (1 - args.eval_ratio))) if identity_rows else 0
    other_cut = max(1, int(len(other_rows) * (1 - args.eval_ratio))) if other_rows else 0
    identity_train = identity_rows[:identity_cut]
    role_cut = max(1, int(len(role_rows_selected) * (1 - args.eval_ratio))) if role_rows_selected else 0
    role_train = role_rows_selected[:role_cut]
    # Deliberate minority oversampling: retain diverse conversational rows,
    # while making role separation learnable. Evaluation remains deduplicated.
    train = (
        other_rows[:other_cut]
        + identity_train * max(1, args.identity_train_repeat)
        + role_train * max(1, args.role_repeat // 2)
    )
    evaluation = other_rows[other_cut:] + identity_rows[identity_cut:] + role_rows_selected[role_cut:]
    rng.shuffle(train); rng.shuffle(evaluation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_path = args.output.with_name(args.output.stem + "_eval.jsonl")
    for path, values in ((args.output, train), (eval_path, evaluation)):
        with path.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "model": "Luna",
        "purpose": "balanced conversational SFT with minority identity correction",
        "conversation_source": str(args.conversation),
        "identity_source": str(args.identity),
        "deduplicated_total": len(rows),
        "train": len(train),
        "eval": len(evaluation),
        "identity_train": sum(row.get("source") == "luna_identity_control" for row in train),
        "identity_eval": sum(row.get("source") == "luna_identity_control" for row in evaluation),
        "role_train": sum(row.get("source") == "luna_role_control" for row in train),
        "role_eval": sum(row.get("source") == "luna_role_control" for row in evaluation),
        "identity_fraction_train": round(sum(row.get("source") == "luna_identity_control" for row in train) / max(1, len(train)), 4),
        "role_fraction_train": round(sum(row.get("source") == "luna_role_control" for row in train) / max(1, len(train)), 4),
        "identity_train_repeat": args.identity_train_repeat,
        "role_repeat": args.role_repeat,
        "seed": args.seed,
        "selection": "deduplicate exact user/answer pairs; retain authored Nix/Casper role separation; preserve Luna identity and role knowledge in held-out split",
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "eval": str(eval_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
