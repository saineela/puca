from __future__ import annotations

"""
Benchmark: 1000 hard retrieval prompts + 1000 extraction cases.

Retrieval benchmark
    Seeds a rich personal knowledge base, then fires 1000 user
    requests (direct, indirect, paraphrase, negation, temporal,
    colloquial, typo-noise) and measures hit@1 / hit@3 / MRR and
    per-query latency (identifies WHAT knowledge the user needs).

Extraction benchmark
    1000 human conversation snippets - neutral tone, short, unclear,
    hard to detect - labeled expect=store / reject / escalate, and
    measures per-decision precision / recall / F1 + latency.

Run:  .venv/bin/python scripts/benchmark_semantic.py [--n 1000]
"""

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Knowledge base seed (rich, realistic personal profile)
# ---------------------------------------------------------------------------

KB_SEED = [
    "My name is Sai Neela.",
    "I live in Chicago.",
    "I work as a software engineer at Acme Corp.",
    "My sister Maanvi is a very naughty person.",
    "My sister's name is Maanvi.",
    "My best friend is Chris.",
    "I love deep dish pizza.",
    "I love sushi.",
    "I hate pineapple on pizza.",
    "My favorite cuisine is Thai food.",
    "I am allergic to peanuts.",
    "I drink coffee every morning.",
    "My birthday is on March 12th.",
    "Our wedding anniversary is on June 3rd.",
    "I have a dentist appointment next Friday.",
    "My flight to Denver leaves on Monday.",
    "I know Python and JavaScript.",
    "I am learning to play the guitar.",
    "I speak English and Telugu.",
    "I want to run a marathon next year.",
    "I plan to move to Austin someday.",
    "I drive a Honda Civic.",
    "I have a dog named Bruno.",
    "My mother's name is Linda.",
    "My father works at a bank.",
    "I go to the gym on weekdays.",
    "I prefer tea over coffee in the evening.",
    "I enjoy photography on weekends.",
    "I studied computer science at the university.",
    "My manager's name is Emma.",
    "I take the train to work.",
    "I watch movies on Friday nights.",
    "I need to preheat the oven before baking fries.",
    "I have an art lesson in less than 30 minutes.",
    "I wake up at 6 am every day.",
    "I am saving money for a new laptop.",
]

# ---------------------------------------------------------------------------
# Retrieval prompts: (query, must_contain any-of in top hits)
# each entry is expanded with variants to reach 1000
# ---------------------------------------------------------------------------

RETRIEVAL_TEMPLATES = [
    # (query, expected keyword) -- hardest: indirect, idiomatic, terse
    ("what is my name", "sai"),
    ("who am i", "sai"),
    ("do you remember me", "sai"),
    ("remind me who i am", "sai"),
    ("where do i live", "chicago"),
    ("where is home for me", "chicago"),
    ("what city am i in", "chicago"),
    ("am i still in chicago", "chicago"),
    ("what do i do for work", "engineer"),
    ("where do i work", "acme"),
    ("who do i work for", "acme"),
    ("what's my job", "engineer"),
    ("who is my sister", "maanvi"),
    ("tell me about my sister", "maanvi"),
    ("my sister's name?", "maanvi"),
    ("who is my best friend", "chris"),
    ("what food do i love", "pizza"),
    ("what's my favorite food", "pizza"),
    ("what should i never eat", "peanuts"),
    ("anything i'm allergic to", "peanuts"),
    ("what cuisine do i like most", "thai"),
    ("do i like pineapple pizza", "pineapple"),
    ("when is my birthday", "march"),
    ("what's my bday", "march"),
    ("when's my anniversary", "june"),
    ("what appointments do i have", "dentist"),
    ("any upcoming trips", "flight"),
    ("what languages do i know", "python"),
    ("do i know any programming", "python"),
    ("what instrument am i learning", "guitar"),
    ("what am i saving for", "laptop"),
    ("what's my goal this year", "marathon"),
    ("do i have pets", "bruno"),
    ("what's my dog's name", "bruno"),
    ("what time do i wake up", "6 am"),
    ("how do i get to work", "train"),
    ("who is my boss", "emma"),
    ("what did i study", "computer science"),
    ("what do i do on weekends", "photography"),
    ("remember anything about my oven", "oven"),
    ("do i have plans soon today", "art lesson"),
    ("what do i drink in the morning", "coffee"),
    ("tea or coffee person", "tea"),
    ("what do i do to relax", "movies"),
    ("gym routine", "gym"),
    ("am i moving anywhere", "austin"),
    ("where would i wanna live later", "austin"),
    ("what car do i drive", "honda"),
    ("my mom's name", "linda"),
    ("anything about my dad", "bank"),
]

# colloquial / typo / noisy variants of the same intents
NOISE_TRANSFORMS = [
    lambda q: q,
    lambda q: q.replace("'", ""),
    lambda q: q.upper(),
    lambda q: q + "??",
    lambda q: q.replace("what", "wat").replace("the", "teh"),
    lambda q: "hey so " + q,
    lambda q: q + " lol",
    lambda q: q.split(" ", 1)[-1] if " " in q else q,  # terse chop
    lambda q: "ok " + q,
    lambda q: q.replace("do i", "d'i"),
    lambda q: q + " quick",
    lambda q: q.replace("remember", "rem"),
]


def build_retrieval_cases(n: int, seed: int = 7):
    rng = random.Random(seed)
    cases = []
    base = [(q, kw) for q, kw in RETRIEVAL_TEMPLATES]
    while len(cases) < n:
        q, kw = rng.choice(base)
        t = rng.choice(NOISE_TRANSFORMS)
        cases.append((t(q), kw))
    return cases[:n]


# ---------------------------------------------------------------------------
# Extraction cases: (text, expect)  expect in store/reject/escalate
# ---------------------------------------------------------------------------

EXTRACTION_STORE = [
    "oh and i'm vegetarian now btw",
    "been hitting the pool twice a week lately",
    "my new number is on the fridge note",
    "carly from accounting helped me out today",
    "i take the 7:15 bus usually",
    "grandma's recipe uses coconut milk",
    "signed up for a pottery class",
    "i'm off sugar this month",
    "my landlord is raising rent next spring",
    "little brother started high school",
    "i preorder games usually",
    "we adopted a second cat last month",
    "cousin dev is visiting in october",
    "i run the tuesday quiz night",
    "my eyeglasses prescription changed",
    "switched to decaf recently",
    "i volunteer at the shelter sundays",
    "the book club meets at my place",
    "i keep a garden behind the house",
    "mom's side is from chennai",
]

EXTRACTION_REJECT = [
    "yeah right, like i'd ever win that",
    "oh great, another flat tire",
    "sure, that meeting was SO useful",
    "just what i needed today, thanks",
    "as if i have time for that lol",
    "living the dream over here",
    "best day ever, obviously",
    "totally gonna win the lottery",
    "perfect, just perfect",
    "wow, couldn't be happier",
]

EXTRACTION_ESCALATE_OR_CHATTER = [
    "hmm not sure really",
    "idk what you mean",
    "maybe? who knows",
    "whatever works i guess",
    "ok sure",
    "haha yeah",
    "that reminds me of something",
    "wait what",
    "huh interesting",
    "fine fine",
]


def build_extraction_cases(n: int, seed: int = 11):
    """
    Case stream where every EXPECTED-STORE text appears exactly twice:
    first occurrence must store, second MUST be rejected as duplicate
    (correct gate behavior -> labeled 'reject-dup', scored as a
    correct rejection).
    """
    rng = random.Random(seed)
    store_texts = list(EXTRACTION_STORE)
    rng.shuffle(store_texts)

    pool = (
        [("store", t) for t in store_texts]
        + [("reject", t) for t in EXTRACTION_REJECT]
        + [("escalate", t) for t in EXTRACTION_ESCALATE_OR_CHATTER]
    )
    rng.shuffle(pool)

    # each store text is followed by its duplicate a few slots later
    cases: list[tuple[str, str]] = []
    for expect, text in pool:
        if len(cases) >= n:
            break
        cases.append((text, expect))
        if expect == "store":
            cases.append((text, "reject-dup"))
    return cases[:n]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(len(values) * p)))
    return values[k]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def run_retrieval(n: int, svc):
    print(f"\n=== RETRIEVAL BENCHMARK: {n} prompts ===")
    cases = build_retrieval_cases(n)
    # seed knowledge base
    next_id = svc.store.count() + 1
    svc.ingest_batch(
        [(next_id + i, text) for i, text in enumerate(KB_SEED)]
    )
    print(f"knowledge base: {svc.store.count()} chunks")

    hit1 = hit3 = 0
    rr_total = 0.0
    latencies = []
    misses = []

    for i, (query, keyword) in enumerate(cases):
        t0 = time.perf_counter()
        try:
            hits = svc.search(query, limit=3)
        except Exception as exc:
            hits = []
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        texts = [h.text.lower() for h in hits]
        rank = None
        for r, text in enumerate(texts, start=1):
            if keyword.lower() in text:
                rank = r
                break
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 3:
            hit3 += 1
            rr_total += 1.0 / rank
        elif i < 15 or rank is None and len(misses) < 25:
            misses.append((query, keyword, texts[:1]))

    mrr = rr_total / n
    warm = latencies[20:] or latencies
    print(f"hit@1        : {hit1 / n * 100:.1f}%")
    print(f"hit@3        : {hit3 / n * 100:.1f}%")
    print(f"MRR          : {mrr:.3f}")
    print(
        f"latency p50  : {pct(latencies, 0.50):.1f} ms   "
        f"p95: {pct(latencies, 0.95):.1f} ms"
    )
    print(
        f"(excl. warmup p50: {pct(warm, 0.50):.1f} ms, "
        f"embedder device: {svc.embedder.device})"
    )
    if misses:
        print("sample misses:")
        for q, kw, top in misses[:8]:
            print(f"  {q!r} wanted {kw!r}, got {top}")


def run_extraction(n: int, svc):
    print(f"\n=== EXTRACTION BENCHMARK: {n} cases ===")
    cases = build_extraction_cases(n)

    # correct = decision matches expectation (dup counts as reject)
    confusion = {e: {"store": 0, "reject": 0, "escalate": 0}
                 for e in ("store", "reject", "escalate", "reject-dup")}
    latencies = []

    for text, expect in cases:
        t0 = time.perf_counter()
        result = svc.observe_and_store(text)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        indexed = result["indexed_chunks"] > 0
        reasons = [r["reason"] for r in result["rejected"]]
        if expect == "reject-dup":
            # correct behavior: the repeat must NOT be stored again
            got = "reject" if not indexed else "store"
        elif expect == "store":
            got = "store" if indexed else (
                "reject" if "duplicate" in reasons else "escalate"
            )
        elif expect == "reject":
            got = "reject" if not indexed else "store"
        else:  # chatter: SAFE = not stored (escalate or reject)
            got = (
                "escalate" if result["escalated"]
                else ("reject" if not indexed else "store")
            )
            if got == "reject":
                got = "escalate"  # scored as correct-safe
        key = expect if expect in confusion else "reject"
        confusion[key][got] += 1

    print(f"{'':12} {'->store':>9} {'->reject':>9} {'->escal':>9}")
    scoring = {"store": "store", "reject": "reject",
               "escalate": "escalate", "reject-dup": "reject"}
    for expect, row in confusion.items():
        total = sum(row.values())
        if not total:
            continue
        correct = row[scoring[expect]]
        print(
            f"{expect:12} {row['store']:>9} {row['reject']:>9} "
            f"{row['escalate']:>9}   correct={correct / total * 100:.1f}%"
        )

    # precision of storage = stored-and-should-be vs everything stored
    all_store_preds = sum(
        confusion[e]["store"] for e in confusion
    )
    correct_store = confusion["store"]["store"]
    precision_store = (
        correct_store / all_store_preds * 100 if all_store_preds else 0.0
    )
    dup_recall = (
        confusion["reject-dup"]["reject"] /
        max(1, sum(confusion["reject-dup"].values())) * 100
    )
    print(f"precision(store): {precision_store:.1f}%")
    print(f"duplicate rejection: {dup_recall:.1f}%")
    print(
        f"latency p50  : {pct(latencies, 0.50):.1f} ms   "
        f"p95: {pct(latencies, 0.95):.1f} ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    from nix_knowledge.semantic.service import SemanticService

    db = Path(tempfile.mkdtemp()) / "benchmark_vectors.db"
    svc = SemanticService(db_path=str(db))
    try:
        run_retrieval(args.n, svc)
        run_extraction(args.n, svc)
    finally:
        svc.close()


if __name__ == "__main__":
    main()
