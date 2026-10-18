"""Build the D4 human-review packet from the frozen replay record.

The agent rubric pass (``d4_human_review_2026-10-18.csv``) is *not* a human
review and is not edited here. This script reads it, keeps every recorded reply
and its agent score verbatim in a companion column, and emits a blank packet for
a human reviewer plus the scoring instructions.

    python3 benchmarks/build_d4_review_packet.py

Outputs (all under benchmarks/results/):

* ``d4_human_review_packet_2026-10-18.csv``      - one row per scored case
* ``d4_human_review_instructions_2026-10-18.md`` - rubric and workflow
* ``d4_review_status_2026-10-18.json``           - status, counts, six misses
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
AGENT_SHEET = RESULTS / "d4_human_review_2026-10-18.csv"
ROUTING_RESULT = RESULTS / "d4_routing_confirm_2026-10-18.json"
KEYS = HERE.parent / "eval" / "data" / "routing_confirm_keys.json"
PACKET = RESULTS / "d4_human_review_packet_2026-10-18.csv"
INSTRUCTIONS = RESULTS / "d4_human_review_instructions_2026-10-18.md"
STATUS = RESULTS / "d4_review_status_2026-10-18.json"

#: What the turn asks for, by set class. The human scores each arm on that
#: request; the routing decision is reported separately and is not the score.
EXPECTED_BEHAVIOUR = {
    "conversation": "Conversation only: the fast chat path is sufficient, and escalating to memory is a false escalation.",
    "ambiguous": "Ambiguous in isolation: either path may be defensible, but the reply must not invent data and should ask when unsure.",
    "memory_read": "Read the user's stored fact: only the memory arm has it, and a stale or invented value is a failure.",
    "memory_write": "Store the new fact: the write must reach the store, and the reply must acknowledge it.",
    "memory_history": "Read the superseded value as history, not as the current value.",
    "memory_delete": "Delete the stored fact only after an explicit confirmation.",
    "world_question": "General/world question: the chat path answers from general knowledge; a memory lookup is not required.",
    "statement_none": "A statement with nothing to store or look up: chat is fine, and no record should be written.",
    "calendar_write": "Create the calendar entry: the turn belongs on the actions/memory path.",
    "state_read": "Read stored conversational state: the memory arm is expected to return it.",
    "calendar_clarify": "A required slot is missing: ask for it rather than guess a date or time.",
    "safety_confirmation": "Destructive or external action: ask for confirmation before acting.",
}

RUBRIC_TEXT = {
    "correctness": "5 answers the turn's request with the right value or act; 4 right answer with a small flaw; 3 partial or misleading; 2 mostly wrong; 1 fails, invents, or does the wrong act.",
    "completeness": "5 covers every part of what was asked; 4 covers it with a minor omission; 3 answers part of it; 2 misses most of it; 1 answers nothing that was asked.",
    "honesty": "5 states plainly when something is unknown or asks for the missing slot; 4 mostly honest with a hedge; 3 vague but not false; 2 overstates what it knows; 1 invents a value or a fact.",
    "naturalness": "5 reads like a person said it; 4 natural with a rough edge; 3 understandable but stiff or repetitive; 2 mechanical or broken; 1 unusable.",
}

HUMAN_COLUMNS = [
    "reviewer_name",
    "review_date_utc",
    "chat_correctness_1_5",
    "chat_completeness_1_5",
    "chat_honesty_1_5",
    "chat_naturalness_1_5",
    "knowledge_correctness_1_5",
    "knowledge_completeness_1_5",
    "knowledge_honesty_1_5",
    "knowledge_naturalness_1_5",
    "per_case_justification",
]

AGENT_COLUMNS = [
    "agent_chat_correctness_1_5",
    "agent_chat_completeness_1_5",
    "agent_chat_honesty_1_5",
    "agent_chat_naturalness_1_5",
    "agent_knowledge_correctness_1_5",
    "agent_knowledge_completeness_1_5",
    "agent_knowledge_honesty_1_5",
    "agent_knowledge_naturalness_1_5",
]


def load_agent_rows() -> list[dict[str, str]]:
    with AGENT_SHEET.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    routing = json.loads(ROUTING_RESULT.read_text(encoding="utf-8"))
    keys = json.loads(KEYS.read_text(encoding="utf-8"))["cases"]
    rows = load_agent_rows()

    scored, not_replayed = [], []
    for row in rows:
        route_correct = (row.get("route_correct") or "").strip()
        if route_correct in {"yes", "no"}:
            scored.append(row)
        else:
            not_replayed.append(row)

    missed_ids = set(routing.get("false_escalation_dispositions") or [])

    packet_rows = []
    for row in scored:
        case_id = row["id"]
        rubric = (keys.get(case_id) or {}).get("rubric", "")
        packet_rows.append(
            {
                "case_id": case_id,
                "set": routing.get("set", {}).get("tag", ""),
                "class": row.get("class", ""),
                "label": row.get("label", ""),
                "rubric": rubric,
                "expected_behaviour": EXPECTED_BEHAVIOUR.get(row.get("class", ""), ""),
                "turn_text": row.get("text", ""),
                "chosen_route": row.get("chosen_route", ""),
                "router_rule": row.get("router_rule", ""),
                "route_was_correct": (row.get("route_correct") or "").strip(),
                "chat_arm_reply": row.get("chat_reply", ""),
                "knowledge_arm_reply": row.get("knowledge_reply", ""),
                **{name: "" for name in HUMAN_COLUMNS},
                **{
                    "agent_chat_correctness_1_5": row.get("chat_correctness_1_5", ""),
                    "agent_chat_completeness_1_5": row.get("chat_completeness_1_5", ""),
                    "agent_chat_honesty_1_5": row.get("chat_honesty_1_5", ""),
                    "agent_chat_naturalness_1_5": row.get("chat_naturalness_1_5", ""),
                    "agent_knowledge_correctness_1_5": row.get("knowledge_correctness_1_5", ""),
                    "agent_knowledge_completeness_1_5": row.get("knowledge_completeness_1_5", ""),
                    "agent_knowledge_honesty_1_5": row.get("knowledge_honesty_1_5", ""),
                    "agent_knowledge_naturalness_1_5": row.get("knowledge_naturalness_1_5", ""),
                },
                "agent_scores_are_not_human_scores": "yes",
                "escalation_miss": "yes" if case_id in missed_ids else "",
                "preserved_disposition": (row.get("false_escalation_disposition") or "").strip(),
            }
        )

    fieldnames = [
        "case_id",
        "set",
        "class",
        "label",
        "rubric",
        "expected_behaviour",
        "turn_text",
        "chosen_route",
        "router_rule",
        "route_was_correct",
        "chat_arm_reply",
        "knowledge_arm_reply",
        *HUMAN_COLUMNS,
        *AGENT_COLUMNS,
        "agent_scores_are_not_human_scores",
        "escalation_miss",
        "preserved_disposition",
    ]
    with PACKET.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in packet_rows:
            writer.writerow(row)

    misses = []
    for row in rows:
        disposition = (row.get("false_escalation_disposition") or "").strip()
        if not disposition:
            continue
        misses.append(
            {
                "case_id": row["id"],
                "turn_text": row.get("text", ""),
                "class": row.get("class", ""),
                "chosen_route": row.get("chosen_route", ""),
                "disposition": disposition,
                "disposition_owner": row.get("reviewer", ""),
                "status": (
                    "retained; update only with new evidence"
                    if "open-defect" in disposition
                    else "retained as written; a regression test and verification must exist for a fixed verdict"
                ),
            }
        )

    INSTRUCTIONS.write_text(_instructions(len(scored), misses, not_replayed), encoding="utf-8")
    (RESULTS / "d4_human_review_packet_2026-10-18.md").write_text(
        _readable_packet(packet_rows), encoding="utf-8"
    )
    STATUS.write_text(
        json.dumps(
            {
                "kind": "d4-human-review-status",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "AWAITING HUMAN SCORES - not complete",
                "packet": str(PACKET.relative_to(HERE.parent)),
                "instructions": str(INSTRUCTIONS.relative_to(HERE.parent)),
                "source_sheet": str(AGENT_SHEET.relative_to(HERE.parent)),
                "source_disclaimer": (
                    "The agent rubric pass in the source sheet was produced by an "
                    "agent, not a human reviewer. It is preserved unedited here and "
                    "is never substituted for the human scores."
                ),
                "scored_cases": len(scored),
                "cases_not_replayed": [row["id"] for row in not_replayed],
                "dimensions": list(RUBRIC_TEXT),
                "escalation_misses": misses,
                "human_fields_to_complete": HUMAN_COLUMNS,
                "recording_requirements": [
                    "reviewer identity (reviewer_name)",
                    "review method: the frozen 1-5 rubric in the instructions",
                    "per-case justification (per_case_justification)",
                    "UTC review date (review_date_utc)",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "packet": str(PACKET),
        "rows": len(packet_rows),
        "not_replayed": len(not_replayed),
        "escalation_misses": len(misses),
        "instructions": str(INSTRUCTIONS),
        "status": str(STATUS),
    }, indent=2))
    return 0


def _readable_packet(rows: list[dict]) -> str:
    """The same 38 cases as a readable sheet, for reviewing away from a CSV."""
    blocks = [
        "# D4 review packet - readable sheet",
        "",
        "Agent scores are shown for reference only; they are not human scores.",
        "",
    ]
    for row in rows:
        blocks.append(f"## {row['case_id']} - {row['class']} ({row['label']}, rubric: {row['rubric']})")
        blocks.append("")
        blocks.append(f"**Turn:** {row['turn_text']}")
        blocks.append("")
        blocks.append(f"**Expected behaviour:** {row['expected_behaviour']}")
        blocks.append("")
        blocks.append(
            f"**Routing:** chose `{row['chosen_route']}` (rule `{row['router_rule']}`); "
            f"route was correct: {row['route_was_correct']}"
        )
        blocks.append("")
        blocks.append("**Chat arm replied:**")
        blocks.append("")
        blocks.append("> " + (row["chat_arm_reply"] or "(empty)").replace("\n", "\n> "))
        blocks.append("")
        blocks.append("**Memory arm replied:**")
        blocks.append("")
        blocks.append("> " + (row["knowledge_arm_reply"] or "(empty)").replace("\n", "\n> "))
        blocks.append("")
        blocks.append(
            "Agent reference scores (chat c/comp/hon/nat, memory c/comp/hon/nat): "
            + ", ".join(
                str(row.get(key, ""))
                for key in (
                    "agent_chat_correctness_1_5",
                    "agent_chat_completeness_1_5",
                    "agent_chat_honesty_1_5",
                    "agent_chat_naturalness_1_5",
                    "agent_knowledge_correctness_1_5",
                    "agent_knowledge_completeness_1_5",
                    "agent_knowledge_honesty_1_5",
                    "agent_knowledge_naturalness_1_5",
                )
            )
        )
        if row.get("preserved_disposition"):
            blocks.append("")
            blocks.append(f"Preserved disposition: {row['preserved_disposition']}")
        blocks.append("")
        blocks.append("Human scores: chat ____ /____ /____ /____  memory ____ /____ /____ /____")
        blocks.append("")
        blocks.append("Justification: ______________________________")
        blocks.append("")
    return "\n".join(blocks) + "\n"


def _instructions(scored: int, misses: list[dict], not_replayed: list[dict]) -> str:
    miss_lines = "\n".join(
        f"- **{miss['case_id']}** - {miss['turn_text']}\n  - disposition (preserved): {miss['disposition']}"
        for miss in misses
    )
    not_replayed_lines = "\n".join(
        f"- {row['id']} ({row.get('class','')}) - {row.get('text','')}" for row in not_replayed
    )
    rubric_lines = "\n".join(f"- **{name}** - {text}" for name, text in RUBRIC_TEXT.items())
    return f"""# D4 human review packet - {scored} scored cases

**Status: awaiting human scores. Nothing in this packet is human-reviewed yet.**
The scores in `agent_*` columns of the packet CSV come from an agent rubric pass
(`benchmarks/results/d4_human_review_2026-10-18.csv`, preserved unedited). They
are a checklist, not an independent review, and must not be copied into the
`reviewer_*` fields.

## What to fill in

Open `d4_human_review_packet_2026-10-18.csv` (any spreadsheet editor, or edit
the text). For every row, replace the blank human columns:

| column | what goes in it |
| --- | --- |
| `reviewer_name` | your name |
| `review_date_utc` | the date you scored it (UTC) |
| `chat_correctness_1_5`, `chat_completeness_1_5`, `chat_honesty_1_5`, `chat_naturalness_1_5` | your score for the **chat arm reply** on that case |
| `knowledge_*_1_5` | your scores for the **knowledge arm reply** on the same case |
| `per_case_justification` | one sentence: why the scores, required whenever any score is 1-3 |

Score **both arms on every case**, including cases where the router picked the
other arm. The point of the packet is what each arm would have said, not which
one was chosen. Leave `agent_*` and `preserved_disposition` alone; they are the
record being compared against.

## The four dimensions

{rubric_lines}

Score what the reply says, not the route. `expected_behaviour` on each row is
the request the turn makes; `route_was_correct` is the separate routing outcome.

## The six escalation misses

The dispositions below are the ones already on record. Retain them. Update a
disposition only if your reading gives new evidence, and say what evidence.

{miss_lines}

## Cases without a replay

These {len(not_replayed)} cases were not replayed with a controlled alternative,
so there is no second arm to score. They are listed here for completeness and
are excluded from the {scored} scored cases.

{not_replayed_lines}

## How to return it

Send back the edited CSV (same file name). Do not rename, reorder or delete
rows, and do not edit the reply columns: the report compares the packet
byte-for-byte against the frozen replay record. Human scores are recorded only
when the CSV comes back with `reviewer_name` and every score filled in.
"""


if __name__ == "__main__":
    raise SystemExit(main())
