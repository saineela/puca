"""D4 - controlled routing replay with answer-quality scoring.

The routing pilot scored a single route. This runs the *other* route on the
same frozen case as well, so the ledger's counterfactual column is backed by an
actual reply:

* **chat arm** - the model alone, with no memory context;
* **knowledge arm** - the full turn path on a dedicated evaluation instance.

Each arm's reply is scored by a deterministic rubric (memory value present,
answer non-empty and not an uncertainty marker, acknowledgement for a write,
a question for clarification/confirmation, a deletion acknowledgement). It is
a coarse proxy, not a human judgement, and the report says so.

    python3 benchmarks/routing_replay.py --base http://127.0.0.1:49151 --json out.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import routing_ledger  # noqa: E402
from routing_engine import CoreRoutingEngine  # noqa: E402

DEFAULT_SET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "routing_heldout.json",
)
DEFAULT_KEYS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "routing_replay_keys.json",
)

_UNCERTAIN = (
    "i don't know", "i do not know", "not sure", "no record", "no information",
    "couldn't find", "could not find", "i don't have", "i do not have",
    "i'm not able", "i am not able", "unknown",
)
_WRITE_MARKERS = (
    "noted", "saved", "remember", "added", "scheduled", "set", "i'll", "i will",
    "done", "recorded", "got it", "stored",
)
_DELETE_MARKERS = ("deleted", "removed", "forgotten", "forgot", "cleared", "confirm", "sure")


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def chat_arm(text: str, *, ollama: str, model: str) -> str:
    """The model alone: no memory, no PUCA context."""
    payload = _post_json(
        f"{ollama.rstrip('/')}/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
            # Qwen3.5 puts its scratchpad in ``thinking``; without this the
            # answer field is empty and the chat arm scores zero for a reason
            # that has nothing to do with memory.
            "think": False,
            # The replay compares arms, not essays; an unbounded generation made
            # one arm minutes slower than the other and added no information.
            "options": {"num_predict": 120},
        },
    )
    return str((payload.get("message") or {}).get("content") or "")


def knowledge_arm(text: str, *, base: str) -> str:
    payload = _post_json(f"{base.rstrip('/')}/api/send", {"text": text, "location": "d4-replay"})
    return str(payload.get("reply") or "")


def score(rubric: str, expected: list[str], reply: str) -> float:
    text = " ".join((reply or "").split())
    if not text:
        return 0.0
    folded = text.casefold()
    uncertain = any(marker in folded for marker in _UNCERTAIN)
    if rubric == "memory":
        return 1.0 if expected and all(value.casefold() in folded for value in expected) else 0.0
    if rubric == "open":
        return 0.0 if uncertain or len(text) < 8 else 1.0
    if rubric == "write":
        return 1.0 if any(marker in folded for marker in _WRITE_MARKERS) else 0.0
    if rubric in ("clarify", "confirm"):
        return 1.0 if "?" in text else 0.0
    if rubric == "delete":
        return 1.0 if any(marker in folded for marker in _DELETE_MARKERS) else 0.0
    return 0.0


def run(
    cases: list[dict],
    keys: dict[str, dict],
    *,
    base: str,
    ollama: str,
    model: str,
    limit: int = 0,
    review: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    engine = CoreRoutingEngine()
    rows: list[routing_ledger.RoutedOutcome] = []
    qualities = {"chosen": [], "chat": [], "knowledge": []}
    by_rubric: dict[str, dict[str, float]] = {}
    unrouted = 0
    for case in cases[: limit or None]:
        spec = keys.get(case["id"], {})
        rubric = str(spec.get("rubric") or "open")
        expected = [str(value) for value in spec.get("expected") or []]
        label = case["label"]
        decision = engine.decide(case["text"])
        chosen = decision.route
        if label == "undetermined":
            rows.append(
                routing_ledger.RoutedOutcome(
                    request_id=case["id"], chosen_route=chosen, label=label,
                    route_correct=None, notes=f"{case['class']} | not scored",
                )
            )
            if review is not None:
                review.append(
                    {
                        "id": case["id"], "class": case["class"], "label": label,
                        "text": case["text"], "chosen_route": chosen,
                        "router_rule": decision.rule,
                        "route_correct": "", "chat_reply": "", "knowledge_reply": "",
                        "note": "undetermined: not replayed",
                    }
                )
            continue
        # Only scored cases enter the rubric denominator: mixing undetermined
        # turns in with answered ones understated every rubric bucket.
        bucket = by_rubric.setdefault(rubric, {"n": 0.0, "chosen": 0.0, "chat": 0.0, "knowledge": 0.0})
        bucket["n"] += 1
        chat_reply = chat_arm(case["text"], ollama=ollama, model=model)
        knowledge_reply = knowledge_arm(case["text"], base=base)
        chat_quality = score(rubric, expected, chat_reply)
        knowledge_quality = score(rubric, expected, knowledge_reply)
        # A decidable turn the router could not route is a miss, not a case to
        # drop: silently excluding it is how an unrouted third of the set
        # disappeared from the first confirmation replay.
        if chosen == "unknown":
            unrouted += 1
            chosen_quality = 0.0
        else:
            chosen_quality = knowledge_quality if chosen == "knowledge" else chat_quality
        alternative = "chat" if chosen == "knowledge" else "knowledge"
        alternative_quality = chat_quality if alternative == "chat" else knowledge_quality
        route_correct = (
            chosen == "knowledge" if label == "escalation_required" else chosen != "knowledge"
        )
        qualities["chosen"].append(chosen_quality)
        qualities["chat"].append(chat_quality)
        qualities["knowledge"].append(knowledge_quality)
        bucket["chosen"] += chosen_quality
        bucket["chat"] += chat_quality
        bucket["knowledge"] += knowledge_quality
        rows.append(
            routing_ledger.RoutedOutcome(
                request_id=case["id"], chosen_route=chosen, label=label,
                route_correct=route_correct, answer_quality=chosen_quality,
                counterfactual="replayed", replayed_route=alternative,
                counterfactual_correct=alternative_quality > 0,
                notes=(
                    f"{case['class']} | rubric={rubric} | "
                    f"chosen={chosen_quality:.1f} chat={chat_quality:.1f} knowledge={knowledge_quality:.1f}"
                ),
            )
        )
        if review is not None:
            review.append(
                {
                    "id": case["id"], "class": case["class"], "label": label,
                    "text": case["text"], "chosen_route": chosen,
                    "router_rule": decision.rule,
                    "route_correct": "yes" if route_correct else "no",
                    "chat_reply": chat_reply,
                    "knowledge_reply": knowledge_reply,
                    "note": f"deterministic proxy: chat={chat_quality:.1f} knowledge={knowledge_quality:.1f}",
                }
            )
    report = routing_ledger.ledger_report(rows)
    count = max(1, len(qualities["chosen"]))
    report.update(
        {
            "experimental_unit": "case",
            "scored_cases": len(qualities["chosen"]),
            "unrouted_cases": unrouted,
            "route_correctness_scored": round(
                sum(1 for row in rows if row.route_correct is not None and row.route_correct) / max(1, len(qualities["chosen"])), 4
            ),
            "answer_quality_proxy": {
                "chosen": round(sum(qualities["chosen"]) / count, 4),
                "chat_arm": round(sum(qualities["chat"]) / count, 4),
                "knowledge_arm": round(sum(qualities["knowledge"]) / count, 4),
                "rubric": "deterministic proxy (value present, non-empty/non-uncertain, acknowledgement, question, deletion)",
            },
            "by_rubric": {
                name: {
                    "n": int(row["n"]),
                    "chosen": round(row["chosen"] / max(1.0, row["n"]), 3),
                    "chat": round(row["chat"] / max(1.0, row["n"]), 3),
                    "knowledge": round(row["knowledge"] / max(1.0, row["n"]), 3),
                }
                for name, row in by_rubric.items()
            },
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default=DEFAULT_SET)
    parser.add_argument("--keys", default=DEFAULT_KEYS)
    parser.add_argument("--base", default="http://127.0.0.1:49151")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", default="")
    parser.add_argument(
        "--review-csv",
        default="",
        help="write the human review sheet for both arms on every replayed case",
    )
    args = parser.parse_args()

    set_text = open(args.set, encoding="utf-8").read()
    keys_text = open(args.keys, encoding="utf-8").read()
    payload = json.loads(set_text)
    keys_payload = json.loads(keys_text)
    keys = keys_payload.get("cases") or {}
    # Frozen before the first reply: the hash names both the cases and their keys.
    freeze = {
        "tag": payload.get("tag"),
        "sha256": hashlib.sha256(
            json.dumps(
                {"set": payload, "keys": keys_payload}, sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "set_file": args.set,
        "keys_file": args.keys,
    }
    review: list[dict[str, Any]] = []
    report = run(
        payload["cases"], keys,
        base=args.base, ollama=args.ollama, model=args.model, limit=args.limit,
        review=review,
    )
    report["set"] = {"tag": payload.get("tag"), "n": len(payload["cases"])}
    report["freeze"] = freeze
    print(f"D4 controlled replay - {report['set']['tag']} (n={report['set']['n']})")
    print(f"  scored cases: {report['scored_cases']} ({report['unrouted_cases']} unrouted)")
    print(f"  route correctness: {report['route_correctness']} "
          f"(n={report['route_correctness_denominator']}, undetermined excluded)")
    print(f"  false escalation: {report['false_escalation']} ({report['false_escalation_numerator']}/{report['false_escalation_denominator']})")
    print(f"  false de-escalation: {report['false_de_escalation']} ({report['false_de_escalation_numerator']}/{report['false_de_escalation_denominator']})")
    print(f"  replayed counterfactuals: {report['replayed']}")
    for row in review:
        if row.get("label") == "escalation_required" and row.get("route_correct") == "no":
            print(f"    escalation miss {row['id']} [{row['class']}] rule={row.get('router_rule')} "
                  f"route={row['chosen_route']}")
    print(f"  answer-quality proxy: chosen {report['answer_quality_proxy']['chosen']} "
          f"chat {report['answer_quality_proxy']['chat_arm']} "
          f"knowledge {report['answer_quality_proxy']['knowledge_arm']}")
    for rubric, row in sorted(report["by_rubric"].items()):
        print(f"    {rubric:<8} n={row['n']:<3} chosen={row['chosen']} chat={row['chat']} knowledge={row['knowledge']}")
    if args.review_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.review_csv)), exist_ok=True)
        columns = [
            "id", "class", "label", "text", "chosen_route", "router_rule", "route_correct",
            "chat_reply", "knowledge_reply",
            # 1-5, human, both arms: 1 = wrong/harmful, 5 = correct and natural.
            "chat_correctness_1_5", "chat_completeness_1_5",
            "chat_honesty_1_5", "chat_naturalness_1_5",
            "knowledge_correctness_1_5", "knowledge_completeness_1_5",
            "knowledge_honesty_1_5", "knowledge_naturalness_1_5",
            "false_escalation_disposition", "reviewer", "notes",
        ]
        with open(args.review_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(review)
        # The deterministic proxy stays in the CSV; the human columns start empty.
        report["human_review"] = {
            "sheet": args.review_csv,
            "cases": len(review),
            "status": "awaiting human scores for both arms",
            "rubric": (
                "1-5 per arm on correctness, completeness, honesty about missing "
                "information and naturalness; every replayed case, not a subset"
            ),
        }
        print(f"wrote review sheet {args.review_csv} ({len(review)} cases)")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
