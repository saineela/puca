from __future__ import annotations

"""
CLI for the neural semantic layer.

    nix-knowledge-semantic backfill
    nix-knowledge-semantic search "favorite food"
    nix-knowledge-semantic train
    nix-knowledge-semantic stats
"""

import argparse
import json
import sys


def _load_engine():
    from nix_knowledge.engine import KnowledgeEngine

    return KnowledgeEngine()


def _iter_active_records(engine):
    for record in engine.retrieval.search(status="active"):
        yield record


def cmd_backfill(args) -> int:
    from .engine import semantic_record_text

    engine = _load_engine()
    try:
        records = list(_iter_active_records(engine))
        print(f"backfilling {len(records)} records ...")
        written = 0
        for record in records:
            text = semantic_record_text(record.data)
            if not text:
                continue
            result = engine.semantic.ingest(record.id, text)
            written += result.get("chunks", 0)
        print(f"indexed {written} chunks across {len(records)} records")
        return 0
    finally:
        engine.close()


def cmd_search(engine, query: str, limit: int, filters_json: str) -> int:
    filters = json.loads(filters_json) if filters_json else None
    hits = engine.semantic_search(query, limit=limit, filters=filters)
    if not hits:
        print("no results")
        return 1
    for hit in hits:
        meta = hit["metadata"]
        print(
            f"[{hit['score']:.4f}] rec {hit['record_id']} "
            f"cat={meta.get('category', '-')} "
            f"tone={meta.get('tone', '-')} "
            f"imp={meta.get('importance', '-')} :: "
            f"{hit['text'][:110]}"
        )
    fused = sum(h["score"] for h in hits)
    print(f"-- {len(hits)} hits (sum {fused:.4f})")
    return 0


def cmd_train(args) -> int:
    from .semantic.training import train_semantic_classifier

    result = train_semantic_classifier(
        max_rows=args.max_rows,
        epochs=args.epochs,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_observe(args) -> int:
    """Print (and store) what the semantic layer learns from a statement."""
    engine = _load_engine()
    try:
        result = engine.observe_and_store(" ".join(args.text))
        for entry in result["stored"]:
            print(
                f"STORED    [{entry['category']}/{entry['tone']}] "
                f"imp={entry['importance']:.2f} :: {entry['text'][:80]}"
            )
        for entry in result["escalated"]:
            print(
                f"ESCALATE  [{entry['category']}] "
                f"conf={entry.get('confidence', '-')} :: {entry['text'][:80]}"
            )
        for entry in result["rejected"]:
            print(f"REJECTED  ({entry['reason']}) :: {entry['text'][:80]}")
        if result["contradictions"]:
            print(
                "CONFLICT  "
                + "; ".join(c["text"][:50] for c in result["contradictions"])
            )
        print(f"-- indexed {result['indexed_chunks']} chunks")
        return 0
    finally:
        engine.close()


def cmd_evidence(args) -> int:
    """Evidence pack for a claim (argumentation support)."""
    engine = _load_engine()
    try:
        pack = engine.evidence_pack(" ".join(args.claim))
        print(f"claim     : {pack['claim']}")
        print(f"verdict   : {pack['verdict']} (confidence {pack['confidence']})")
        for e in pack["supporting"]:
            print(f"  support + {e['similarity']:.3f}  {e['text'][:75]}")
        for e in pack["contradicting"]:
            print(f"  against - {e['similarity']:.3f}  {e['text'][:75]}")
        return 0
    finally:
        engine.close()


def cmd_stats(engine) -> int:
    print(json.dumps(engine.semantic.stats(), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="nix-knowledge-semantic",
        description="Neural semantic layer for Nix Knowledge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_backfill = sub.add_parser(
        "backfill", help="index existing knowledge records"
    )
    p_backfill.add_argument("--limit", type=int, default=None)

    p_search = sub.add_parser("search", help="hybrid semantic search")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument(
        "--filters",
        default=None,
        help='JSON metadata filter, e.g. \'{"tone":"urgent"}\'',
    )

    p_train = sub.add_parser(
        "train", help="train the semantic classifier heads"
    )
    p_train.add_argument("--max-rows", type=int, default=20000)
    p_train.add_argument("--epochs", type=int, default=6)

    p_stats = sub.add_parser(
        "stats", help="semantic index statistics"
    )

    p_observe = sub.add_parser(
        "observe",
        help="perceive + store knowledge from a statement",
    )
    p_observe.add_argument("text", nargs="+")

    p_evidence = sub.add_parser(
        "evidence",
        help="evidence pack for a claim (supports/contradicts)",
    )
    p_evidence.add_argument("claim", nargs="+")

    args = parser.parse_args(argv)

    if args.command == "backfill":
        return cmd_backfill(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "observe":
        return cmd_observe(args)
    if args.command == "evidence":
        return cmd_evidence(args)
    if args.command == "stats":
        engine = _load_engine()
        try:
            return cmd_stats(engine)
        finally:
            engine.close()
    if args.command == "search":
        engine = _load_engine()
        try:
            return cmd_search(
                engine,
                " ".join(args.query),
                args.limit,
                args.filters,
            )
        finally:
            engine.close()

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
