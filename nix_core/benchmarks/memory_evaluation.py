"""Run the D2 memory evaluation against the live API.

PUCA is asked each scenario's question through the full turn path. The two
baselines receive the same token budget, the same model and the same task
framing: one gets the recent transcript, the other a plain dump of the records.

    python3 benchmarks/memory_evaluation.py --sets heldout adversarial --budget 1200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import memory_sets  # noqa: E402

DEFAULT_SETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "memory_scenarios.json",
)


def call_model(base: str, model: str, prompt: str, context: str = "", timeout: float = 90.0) -> str:
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 96}).encode()
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read() or b"{}")
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def call_ollama(base: str, model: str, prompt: str, context: str = "", timeout: float = 90.0) -> str:
    """A model-only baseline.

    ``/v1/chat/completions`` on the console runs the full PUCA pipeline, so using
    it as a "baseline" gave every arm the same knowledge path and hid whether
    the baselines were independent. This calls the model directly instead.
    """
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": 96},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base.rstrip('/')}/api/chat",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read() or b"{}")
    return str((payload.get("message") or {}).get("content") or "")


def call_puca(base: str, prompt: str, timeout: float = 90.0) -> str:
    body = json.dumps({"text": prompt, "location": "memory-eval"}).encode()
    request = urllib.request.Request(
        f"{base}/api/send",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read() or b"{}")
    return str(payload.get("reply") or "")


def scenario_transcript(scenario) -> list[dict[str, str]]:
    """The transcript baseline's context: the same facts, as conversation."""
    facts = _facts(scenario)
    if not facts:
        return []
    return [{"role": "user", "content": "Things I told you: " + "; ".join(facts)}]


def scenario_records(scenario) -> list[dict[str, Any]]:
    """The filesystem baseline's context: the same facts, as stored records."""
    return [
        {"id": f"{scenario.id}:{index}", "data": {"fact": fact}}
        for index, fact in enumerate(_facts(scenario))
    ]


def _facts(scenario) -> list[str]:
    """Every fact a scenario's answer key holds, current and superseded.

    Both baselines are given the same information the system under test would
    have in its store, so the comparison is about retrieval and use, not about
    who was handed less.
    """
    facts: list[str] = []
    for group in (
        scenario.current_facts,
        scenario.expected_facts,
        scenario.superseded_facts,
    ):
        for fact in group:
            if fact and fact not in facts:
                facts.append(fact)
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("NIX_BASE_URL", "http://127.0.0.1:49117"))
    parser.add_argument("--model", default="luna-v6-contextual-v1")
    parser.add_argument("--scenarios", default=DEFAULT_SETS)
    parser.add_argument("--sets", nargs="*", default=["development", "heldout", "adversarial"])
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--baseline-mode",
        choices=("console", "ollama"),
        default="console",
        help=(
            "console posts /v1/chat/completions (the full pipeline, kept for "
            "compatibility); ollama calls the model directly with no memory"
        ),
    )
    parser.add_argument(
        "--ollama",
        default=os.environ.get("NIX_OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--baseline-model",
        default=os.environ.get("NIX_BASELINE_MODEL", "qwen3.5:4b"),
        help=(
            "the model the console actually answers with, for the model-only "
            "baselines; the console exposes it under luna-v6-contextual-v1 but "
            "/api/health reports this ollama model"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help=(
            "independent samples per case (the approved repeat protocol uses 5); "
            "samples of one case are one cluster and never count as extra cases"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "cases answered concurrently (wall-clock only; a case's own samples "
            "stay sequential and independent, and the scores must not change)"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", default="")
    parser.add_argument("--tag", default="frozen-2026-10-17")
    parser.add_argument(
        "--prior-exposure",
        default="none recorded",
        help="any earlier look at these sets, stated in the result note",
    )
    parser.add_argument(
        "--retired",
        nargs="*",
        default=[],
        help="earlier set tags this one replaces, so the freeze hash matches its manifest",
    )
    args = parser.parse_args()

    sets = memory_sets.load_sets(args.scenarios)
    results: dict[str, Any] = {}
    for name in args.sets:
        rows = sets.get(name) or []
        if args.limit:
            rows = rows[: args.limit]
        results[name] = memory_sets.run_set(
            rows,
            samples=args.samples,
            workers=args.workers,
            budget=args.budget,
            answer=lambda scenario, budget: call_puca(args.base, scenario.question),
            transcripts={scenario.id: scenario_transcript(scenario) for scenario in rows},
            records=scenario_records,
            baseline_answer=lambda scenario, context, budget: (
                call_ollama(
                    args.ollama, args.baseline_model, scenario.question, context,
                    timeout=args.timeout,
                )
                if args.baseline_mode == "ollama"
                else call_model(
                    args.base, args.model, scenario.question, context, timeout=args.timeout
                )
            ),
        )
    print(memory_sets.format_table(results))
    if args.workers > 1:
        print(f"workers: {args.workers} concurrent cases (wall-clock only)")
    note = memory_sets.frozen_manifest(
        {name: sets[name] for name in args.sets if name in sets},
        tag=args.tag,
        model=args.model,
        hardware=os.uname().machine,
        prior_exposure=args.prior_exposure,
        retired=args.retired,
        baseline_mode=args.baseline_mode,
        baseline_model=args.baseline_model if args.baseline_mode == "ollama" else "",
        samples=args.samples,
    )
    print(f"\nfrozen set: {note['tag']} ({note['sha256'][:16]}…)")

    rows_by_set = {name: result["scenarios"] for name, result in results.items()}
    failures = memory_sets.sample_failure_report(results)
    print(
        "sample failures: "
        + ", ".join(
            f"{arm} {count}/{failures['planned_samples']}"
            for arm, count in failures["failed_samples"].items()
        )
        + f" -> {'VALID' if failures['valid'] else 'INVALID (rerun in full)'}"
        + (f"; degraded cases: {', '.join(failures['degraded_cases'])}" if failures["degraded_cases"] else "")
    )
    abstention = memory_sets.abstention_report(rows_by_set)
    pooled = abstention["pooled"]
    print(
        f"abstention (pooled {pooled['n']}): {pooled['honest']}/{pooled['n']} honest, "
        f"rate {pooled['rate']}, 95% Wilson [{pooled['wilson_low']}, {pooled['wilson_high']}] "
        f"-> {'PASS' if abstention['passed'] else 'FAIL (undecided)'}"
    )
    for name, block in abstention["per_set"].items():
        print(f"  {name:<14} {block['honest']}/{block['n']} (descriptive)")
    if args.samples > 1:
        spreads = [
            row["fact_recall_stdev"]
            for result in results.values()
            for row in result["scenarios"]
            if row.get("fact_recall_stdev") is not None
        ]
        unstable = sum(1 for value in spreads if value > 0)
        print(
            f"samples per case: {args.samples}; cases whose recall varied across "
            f"samples: {unstable}/{len(spreads)}; mean within-case stdev "
            f"{round(sum(spreads) / max(1, len(spreads)), 3)}"
        )

    if args.baseline_mode == "ollama":
        advantage = memory_sets.memory_advantage_report(rows_by_set)
        for arm in ("transcript", "filesystem"):
            block = advantage[arm]
            print(
                f"memory advantage vs {arm}: {block['difference']:+.3f} "
                f"[{block['low']:+.3f}, {block['high']:+.3f}] n={block['n']} "
                f"-> {'demonstrated' if block['demonstrated'] else 'not demonstrated'}"
            )
    else:
        advantage = {
            "blocked": (
                "baselines were not model-only; a memory advantage cannot be "
                "claimed from a shared pipeline"
            )
        }
        print(f"memory advantage: blocked ({advantage['blocked']})")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "results": results,
                    "freeze": note,
                    "abstention": abstention,
                    "memory_advantage": advantage,
                    "sample_failures": failures,
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.json}")
    return 0 if failures["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
