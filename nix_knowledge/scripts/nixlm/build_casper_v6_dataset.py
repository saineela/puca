"""Build Casper V6 datasets from approved public dialogue and anonymized seeds.

This builder writes only V6-named files. It never reads raw request logs and
never copies prior V5 rows automatically. Public source loading is delegated to
the reviewed V5 loader, whose filtering and license metadata are retained.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from build_public_human_dialogue import CASPER_RULES, acceptable, load_sources, make_row
from build_casper_v6_preference_data import SEEDS

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "training_data"

# Generalized log-derived examples. No real names, dates, secrets, or user
# values are included; these teach behavior boundaries rather than memories.
ANONYMIZED_RULES = (
    ("User has two sisters, A and B. User: my sister is doing better now.", "Good to hear. Which sister do you mean—A or B?"),
    ("Current state: person A is better. Historical state: person A was sick. User: how is person A now?", "Person A is doing better now; the older sickness note is historical."),
    ("User: I am tired; do not ask me about family.", "Got it. I’ll keep this brief and won’t add unnecessary questions."),
    ("User: remember that I like tea and tell me a joke.", "I’ll remember that you like tea. And here’s one: I tried to organize my calendar, but it had too many dates."),
    ("Action result: reminder created every 2 days at 08:00.", "Done—I’ll remind you every 2 days at 8:00 AM."),
    ("User: tell me a story using my name.", "Sure—what name should I use? I won’t assume or store one from the request."),
    ("User: who created Casper?", "Casper was created and built by Sai Neela and lives in NIX’s PUCA system."),
    ("User: stop asking questions.", "Okay. I’ll stop."),
)


def _sft_row(user: str, answer: str, source: str) -> dict[str, Any]:
    row = make_row(user, answer, source)
    row["version"] = "casper_v6"
    row["behavior_tags"] = ["natural", "grounded", "bounded"]
    return row


def _preference_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "version": "casper_v6",
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "tags": row.get("tags", []),
    }


def build(limit: int, seed: int, output_dir: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    public_candidates = [pair for pair in load_sources() if acceptable(pair[0], pair[1])]
    unique: dict[tuple[str, str], tuple[str, str, str]] = {}
    for user, answer, source in public_candidates:
        unique[(user.casefold(), answer.casefold())] = (user, answer, source)
    public = list(unique.values())
    rng.shuffle(public)
    public = public[:limit]

    curated = [(user, answer, "casper_v6_anonymized") for user, answer in ANONYMIZED_RULES]
    curated.extend((user, answer, "casper_v6_rule") for user, answer in CASPER_RULES)
    sft_rows = [_sft_row(*pair) for pair in public + curated]
    rng.shuffle(sft_rows)
    eval_count = max(40, min(len(sft_rows) // 5, 300))
    evaluation = sft_rows[:eval_count]
    training = sft_rows[eval_count:]
    preferences = [_preference_row(row) for row in SEEDS]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sft_train": output_dir / "casper_v6_sft_train.jsonl",
        "sft_eval": output_dir / "casper_v6_sft_eval.jsonl",
        "preferences": output_dir / "casper_v6_preferences.jsonl",
        "metadata": output_dir / "casper_v6_dataset.metadata.json",
    }
    for key, rows in (("sft_train", training), ("sft_eval", evaluation), ("preferences", preferences)):
        with paths[key].open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "version": "casper_v6",
        "seed": seed,
        "public_sources": [
            {"dataset": "daily_dialog", "license": "CC BY-NC-SA 4.0"},
            {"dataset": "facebook/empathetic_dialogues", "license": "CC BY-NC-SA 4.0"},
        ],
        "public_candidate_count_after_filtering": len(public_candidates),
        "public_rows_selected": len(public),
        "curated_rows": len(curated),
        "sft_train_rows": len(training),
        "sft_eval_rows": len(evaluation),
        "preference_rows": len(preferences),
        "raw_logs_used": False,
        "personal_values_used": False,
        "v5_files_modified": False,
        "filters": ["assistant boilerplate", "generic interview endings", "URLs", "secrets", "crisis details", "malformed turns"],
        "note": "Review source terms before redistribution; this dataset is for local development.",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()} | metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.limit, args.seed, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
