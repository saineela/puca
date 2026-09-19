"""
Analyze nix_core request logs (logs/requests-YYYY-MM-DD.jsonl).

Usage:
    python analyze_logs.py                  # today, human-readable summary
    python analyze_logs.py --all            # every log file
    python analyze_logs.py --days 7         # last 7 days
    python analyze_logs.py --problems       # only the interesting stuff
    python analyze_logs.py --export PATH    # export problem records
    python analyze_logs.py --tail 20        # last 20 requests
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

LOG_DIR = os.environ.get(
    "NIX_REQUEST_LOG_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
)

# Reply prefixes that mean the knowledge layer could not serve the
# request properly - each maps to a problem bucket.
_BAD_REPLY_PREFIXES = {
    "knowledge engine could not handle that": "engine_error",
    "nix alert": "api_unreachable",
    "error:": "reply_error",
}

# Rules that mean the deterministic layer abstained and the model
# decided (fine, but worth watching the share).
_MODEL_RULES = {"model_classifier", "model_intent"}


def load(mode: str = "today", days: int = 7) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(LOG_DIR, "requests-*.jsonl")))
    if mode == "today":
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        paths = [p for p in paths if p.endswith(f"requests-{day}.jsonl")]
    elif mode == "days":
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        kept = []
        for p in paths:
            m = re.search(r"requests-(\d{4}-\d{2}-\d{2})\.jsonl$", p)
            if m and datetime.strptime(m.group(1), "%Y-%m-%d") >= cutoff:
                kept.append(p)
        paths = kept

    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn write; skip
    return records


def problems(rec: dict[str, Any]) -> list[str]:
    """Classify one record into problem buckets (or [])."""
    out: list[str] = []
    route = rec.get("route") or ""
    rule = rec.get("rule") or ""
    reply = (rec.get("reply") or "").strip()
    reply_lower = reply.lower()
    latency = rec.get("latency_ms") or 0

    if rec.get("error"):
        out.append("exception")
    if route == "unknown" or route == "error":
        out.append("route_" + str(route))
    for prefix, bucket in _BAD_REPLY_PREFIXES.items():
        if reply_lower.startswith(prefix):
            out.append(bucket)
    if "i didn't catch anything" in reply_lower:
        out.append("empty_request")
    if rule in _MODEL_RULES:
        out.append("model_decided")
    if latency > 8000:
        out.append("slow_over_8s")
    if not reply:
        out.append("empty_reply")
    return out


def summarize(records: list[dict[str, Any]]) -> None:
    if not records:
        print("no log records found")
        return

    total = len(records)
    routes = Counter(r.get("route") or "?" for r in records)
    rules = Counter(r.get("rule") or "(none)" for r in records)
    locations = Counter(r.get("location") or "?" for r in records)
    latencies = sorted(float(r.get("latency_ms") or 0) for r in records)

    def pctl(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(p / 100 * len(latencies)))
        return latencies[idx]

    print(f"records:   {total}")
    print(f"routes:    {dict(routes.most_common())}")
    print(f"locations: {dict(locations.most_common())}")
    print(
        f"latency:   avg={sum(latencies) / len(latencies):.0f}ms "
        f"p50={pctl(50):.0f}ms p90={pctl(90):.0f}ms max={latencies[-1]:.0f}ms"
    )
    print("top rules:")
    for rule, n in rules.most_common(10):
        print(f"  {n:>5}  {rule}")

    bucket_counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    for rec in records:
        for bucket in problems(rec):
            bucket_counts[bucket] += 1
            examples.setdefault(bucket, rec)

    if bucket_counts:
        print("\nproblem buckets:")
        for bucket, n in bucket_counts.most_common():
            print(f"  {n:>5}  {bucket}")
        print("\nfirst example per bucket:")
        for bucket, rec in examples.items():
            print(f"  [{bucket}] {rec.get('request', '')!r:.70}")
            print(f"      route={rec.get('route')} rule={rec.get('rule')} "
                  f"reply={str(rec.get('reply'))[:70]!r}")
    else:
        print("\nno problems detected")


def show_problems(records: list[dict[str, Any]]) -> int:
    flagged = [r for r in records if problems(r)]
    for rec in flagged:
        buckets = ",".join(problems(rec))
        print(f"[{buckets}] {rec.get('ts', '')} {rec.get('request', '')!r}")
        print(
            f"    route={rec.get('route')} rule={rec.get('rule')} "
            f"latency={rec.get('latency_ms')}ms"
        )
        print(f"    reply: {str(rec.get('reply'))[:140]!r}")
    return len(flagged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="every log file")
    parser.add_argument("--days", type=int, default=0, help="last N days")
    parser.add_argument(
        "--problems", action="store_true", help="list problem records only"
    )
    parser.add_argument(
        "--export", metavar="PATH", help="export problem records as JSONL"
    )
    parser.add_argument("--tail", type=int, default=0, help="show last N requests")
    args = parser.parse_args()

    mode = "all" if args.all else "today"
    if args.days:
        mode = "days"
    records = load(mode, days=args.days or 7)

    if args.export:
        flagged = [r for r in records if problems(r)]
        with open(args.export, "w", encoding="utf-8") as fh:
            for rec in flagged:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"exported {len(flagged)} problem records to {args.export}")
        return 0

    if args.tail:
        for rec in records[-args.tail:]:
            print(
                f"{rec.get('ts', '')[11:19]} {rec.get('route', '?'):<22} "
                f"{rec.get('latency_ms', 0):>7}ms  {rec.get('request', '')!r:.60}"
            )
            print(f"           -> {str(rec.get('reply'))[:100]!r}")
        return 0

    if args.problems:
        n = show_problems(records)
        if not n:
            print("no problems found")
        return 0

    summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
