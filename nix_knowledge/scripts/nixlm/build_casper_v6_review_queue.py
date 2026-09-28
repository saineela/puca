"""Create a human-review queue; this does not claim synthetic rows are approved."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from build_casper_v6_preference_data import SEEDS

VARIANTS = (
    "Keep the answer natural and brief.",
    "The user is tired; avoid a nonessential question.",
    "Preserve the authoritative person, date, and recurrence details.",
    "Do not invent memory or claim an unconfirmed action.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for source_row in SEEDS:
        for index in range(args.repeat):
            rubric = VARIANTS[index % len(VARIANTS)]
            rows.append({
                "id": f"review-{source_row['id']}-{index + 1:03d}",
                "version": "casper_v6",
                "prompt": source_row["prompt"],
                "chosen": source_row["chosen"],
                "rejected": source_row["rejected"],
                "tags": list(source_row.get("tags", [])),
                "rubric": rubric,
                "source": "anonymized_seed_template",
                "review_status": "pending",
                "reviewer": None,
                "review_notes": None,
                "approved_for_training": False,
            })
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "review_status": "pending", "approved": 0}, indent=2))


if __name__ == "__main__":
    main()
