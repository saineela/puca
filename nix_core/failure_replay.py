"""Failure-log regression harness (round-3 research T13).

Every Brain.handle() turn is already appended to
logs/requests-YYYY-MM-DD.jsonl with routing, rule, reply, latency, and the
full details payload (request_log.make_entry). This harness turns that log
into a regression suite, following the production-failure-replay pattern
(trace-driven replay: mine failures, replay them, fail on regressions):

    python3 failure_replay.py [--log-dir logs] [--days 7] [--limit 40] [--list]

Modes:
  --list   Summarize recent failure records (rule, class, request) only.
  default  Emit a replay corpus of requests that produced failure records,
           deduplicated, newest first, ready to be re-run through Brain in a
           test session after a change.

A failure record is any entry whose rule or failure taxonomy marks it as
non-confirmed: skill planner/validator/execution rejections, unconfirmed
device actions, unavailable skills, and aborted retries. Successful turns
are excluded by rule so a policy change cannot silently shrink the corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import request_log  # noqa: E402

# Rules that always mark a failed or not-fully-confirmed turn.
FAILURE_RULES = {
    "nix_model_skill_tool_rejected",
    "nix_model_skill_plan_rejected",
    "nix_model_skill_tool_unconfirmed",
    "nix_model_skill_plan_partial",
    "trusted_skill_unavailable",
}

# Confirmed-success rules that should never count as failures even if a
# future change renames them: only these prove a turn acted correctly.
SUCCESS_MARKERS = (
    "confirmed",
    "nix_model_skill_tool",
    "nix_model_skill_plan",
)

# detail keys that mark partial failures inside otherwise-neutral rules
FAILURE_DETAIL_KEYS = (
    "skill_error",
    "skill_decider_error",
    "skill_failure_class",
)


def _failure_class(entry: dict[str, Any]) -> str | None:
    details = entry.get("details") or {}
    if isinstance(details, dict):
        for key in FAILURE_DETAIL_KEYS:
            value = details.get(key)
            if value:
                return f"{key}:{str(value)[:60]}"
        if details.get("skill_execution_confirmed") is False and details.get("model_called"):
            return "unconfirmed_execution"
    rule = str(entry.get("rule") or "")
    if rule in FAILURE_RULES:
        return rule
    if "rejected" in rule or "unconfirmed" in rule or "unavailable" in rule:
        return rule
    return None


def collect_failures(
    log_dir: str | None = None,
    days: int = 7,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Return the newest unique failing requests from the request log."""
    directory = log_dir or request_log.log_dir()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    failures: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    if not os.path.isdir(directory):
        return failures
    day = datetime.now(timezone.utc).date()
    files: list[str] = []
    for _ in range(max(1, days)):
        files.append(os.path.join(directory, f"requests-{day.isoformat()}.jsonl"))
        day -= timedelta(days=1)
    for path in reversed(files):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                stamp = str(entry.get("ts") or "")
                try:
                    when = datetime.fromisoformat(stamp)
                except ValueError:
                    when = None
                if when is not None and when < cutoff:
                    continue
                failure = _failure_class(entry)
                if failure is None:
                    continue
                request_text = " ".join(str(entry.get("request") or "").split())
                if not request_text:
                    continue
                key = request_text.casefold()
                if key in seen_requests:
                    continue
                seen_requests.add(key)
                failures.append({
                    "ts": stamp,
                    "request": request_text[:300],
                    "failure": failure,
                    "rule": entry.get("rule"),
                    "reply": str(entry.get("reply") or "")[:200],
                })
    failures.reverse()  # newest first
    return failures[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", default=None, help="Request-log directory (default: nix_core/logs)")
    parser.add_argument("--days", type=int, default=7, help="How many days of logs to scan")
    parser.add_argument("--limit", type=int, default=40, help="Max failure records to list")
    parser.add_argument("--list", action="store_true", help="Only print the failure summary")
    args = parser.parse_args()

    failures = collect_failures(log_dir=args.log_dir, days=args.days, limit=args.limit)
    counts = Counter(item["failure"].split(":", 1)[0] for item in failures)
    print(f"failure records: {len(failures)} unique in the last {args.days} day(s)")
    for name, count in counts.most_common():
        print(f"  {count:4d}  {name}")
    if args.list or not failures:
        for item in failures:
            print(f"- [{item['ts'][:19]}] {item['failure']} :: {item['request']}")
        return 0
    corpus = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": args.log_dir or request_log.log_dir(),
        "cases": [
            {"request": item["request"], "expected_failure": item["failure"]}
            for item in failures
        ],
    }
    print(json.dumps(corpus, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
