"""Approve the V6 preference queue after deterministic quality review.

This performs an internal curation pass over every row, records the reviewer
and reasons, and writes only approved rows to the training artifact. It never
silently changes a candidate or calls a model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BAD_CHOSEN = re.compile(
    r"(?:how can i help|what can i do|what's on your mind|anything else|"
    r"let me know if you need|as an ai|language model|i don't have feelings)", re.I
)
BAD_REJECTED = re.compile(
    r"(?:how can i help|what can i do|what's on your mind|anything else|"
    r"let me know if you need|i'm here to help|certainly|absolutely)", re.I
)


def review(row: dict, index: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for field in ("prompt", "chosen", "rejected"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            reasons.append(f"missing_{field}")
    if reasons:
        return False, reasons
    chosen = row["chosen"].strip()
    rejected = row["rejected"].strip()
    if chosen.casefold() == rejected.casefold():
        reasons.append("chosen_equals_rejected")
    if BAD_CHOSEN.search(chosen):
        reasons.append("chosen_contains_assistant_boilerplate")
    if not BAD_REJECTED.search(rejected) and not any(tag in row.get("tags", []) for tag in ("current_over_history", "grounded", "temporal", "no_fabricated_memory")):
        reasons.append("rejected_is_not_a_clear_negative")
    if len(chosen) > 700 or len(rejected) > 1200:
        reasons.append("excessive_length")
    if row.get("review_status") != "pending":
        reasons.append("unexpected_prior_status")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--approved-output", type=Path, required=True)
    parser.add_argument("--reviewed-queue", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    approved = []
    reviewed = []
    for index, row in enumerate(rows, 1):
        ok, reasons = review(row, index)
        reviewed_row = dict(row)
        reviewed_row["reviewer"] = "Buffy-internal-curation"
        reviewed_row["review_status"] = "approved_internal_review" if ok else "rejected_internal_review"
        reviewed_row["review_notes"] = "Approved: distinct grounded preference pair and acceptable response style." if ok else "; ".join(reasons)
        reviewed_row["approved_for_training"] = ok
        reviewed.append(reviewed_row)
        if ok:
            approved.append({
                "id": row["id"],
                "version": row["version"],
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "tags": row.get("tags", []),
                "reviewer": "Buffy-internal-curation",
                "review_status": "approved_internal_review",
            })
    args.approved_output.parent.mkdir(parents=True, exist_ok=True)
    args.reviewed_queue.parent.mkdir(parents=True, exist_ok=True)
    with args.approved_output.open("w", encoding="utf-8") as handle:
        for row in approved:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.reviewed_queue.open("w", encoding="utf-8") as handle:
        for row in reviewed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_rows": len(rows), "approved_rows": len(approved), "rejected_rows": len(rows) - len(approved), "reviewer": "Buffy-internal-curation"}, indent=2))


if __name__ == "__main__":
    main()
