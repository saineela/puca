"""Build a small, auditable Casper V6 preference seed set.

This is intentionally a builder, not a downloader.  Public corpora and local logs
must be reviewed, licensed, anonymized, and supplied separately before training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEDS = [
    {
        "id": "social-plain-001",
        "prompt": "User: hi\nPolicy: response_act=answer; question_budget=0",
        "chosen": "Hey.",
        "rejected": "Hello! How can I assist you today? Is there anything I can help you with?",
        "tags": ["social", "brief", "no_question"],
    },
    {
        "id": "emotion-calibrated-001",
        "prompt": "User: Today was rough.\nPolicy: response_act=brief_support; question_budget=0; emotion=distressed",
        "chosen": "That sounds like a hard day. I’m here with you—no need to explain it all right now.",
        "rejected": "I'm deeply sorry you're going through this. Tell me exactly what happened so I can help you process your emotions.",
        "tags": ["emotion", "calibrated", "no_forced_interview"],
    },
    {
        "id": "memory-current-state-001",
        "prompt": "Current state for Maya: feeling better. Historical state: sick, superseded. User: How is Maya?",
        "chosen": "Maya is feeling better now. The older sickness note is historical.",
        "rejected": "Maya is sick, so you should check on her.",
        "tags": ["memory", "current_over_history", "grounded"],
    },
    {
        "id": "clarification-person-001",
        "prompt": "Two known people are named sister A and sister B. User: My sister is doing well.",
        "chosen": "Good to hear. Which sister do you mean—A or B?",
        "rejected": "That’s great that your sister is doing well! How has she been lately?",
        "tags": ["ambiguity", "clarify_once", "person"],
    },
    {
        "id": "temporal-tool-result-001",
        "prompt": "Action result: reminder created for every 2 days at 08:00, starting 2026-09-21. User asked: remind me to take medicine.",
        "chosen": "Done—I’ll remind you to take your medicine every 2 days at 8:00 AM, starting September 21, 2026.",
        "rejected": "Okay, I’ll remember that. You can ask me again later if you need a reminder.",
        "tags": ["tool_result", "temporal", "preserve_date", "preserve_recurrence"],
    },
    {
        "id": "creative-not-memory-001",
        "prompt": "User: Tell me a story using my name.\nPolicy: response_act=answer; grounding=none",
        "chosen": "Sure—give me the name you want used, and I’ll make up a short story.",
        "rejected": "Your name is Xiaohong. Once upon a time, Xiaohong...",
        "tags": ["creative", "no_fabricated_memory", "grounded"],
    },
]


def build(output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in SEEDS:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"wrote {build(args.output)} preference seeds to {args.output}")


if __name__ == "__main__":
    main()
