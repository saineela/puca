from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = (
    "hi",
    "what is on my schedule tomorrow?",
    "I have robotics practice tomorrow at 5pm and remember that I joined TSA",
    "What do you remember about robotics?",
    "My sister is doing alright now",
    "Tell me a short joke",
)


@dataclass(frozen=True)
class BenchmarkSample:
    prompt: str
    elapsed_ms: float
    ok: bool
    result: str = ""
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    benchmark: str
    scenario: str
    samples: int
    warmup_samples: int
    successful_samples: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    requests_per_second: float
    samples_detail: tuple[BenchmarkSample, ...]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 3)


def summarize(
    benchmark: str,
    scenario: str,
    samples: list[BenchmarkSample],
    warmup_samples: int,
) -> BenchmarkSummary:
    successful = [sample.elapsed_ms for sample in samples if sample.ok]
    total_seconds = sum(successful) / 1000
    return BenchmarkSummary(
        benchmark=benchmark,
        scenario=scenario,
        samples=len(samples),
        warmup_samples=warmup_samples,
        successful_samples=len(successful),
        p50_ms=percentile(successful, 0.50),
        p95_ms=percentile(successful, 0.95),
        min_ms=round(min(successful), 3) if successful else 0.0,
        max_ms=round(max(successful), 3) if successful else 0.0,
        mean_ms=round(statistics.mean(successful), 3) if successful else 0.0,
        requests_per_second=round(len(successful) / total_seconds, 3) if total_seconds else 0.0,
        samples_detail=tuple(samples),
    )


def run_samples(
    benchmark: str,
    scenario: str,
    prompts: tuple[str, ...],
    operation: Callable[[str], str],
    *,
    repeats: int,
    warmup: int,
) -> BenchmarkSummary:
    for index in range(warmup):
        operation(prompts[index % len(prompts)])

    samples: list[BenchmarkSample] = []
    for index in range(repeats):
        prompt = prompts[index % len(prompts)]
        started = time.perf_counter()
        try:
            result = operation(prompt)
            samples.append(
                BenchmarkSample(
                    prompt=prompt,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    ok=True,
                    result=str(result)[:300],
                )
            )
        except Exception as exc:  # benchmark failures are recorded, not hidden
            samples.append(
                BenchmarkSample(
                    prompt=prompt,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return summarize(benchmark, scenario, samples, warmup)


def core_router_operation(prompt: str) -> str:
    from router import classify

    route, features = classify(prompt)
    return json.dumps({"route": route, "rule": features.get("rule")}, sort_keys=True)


def http_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def predictor_operation(prompt: str, endpoint: str, timeout: float) -> str:
    """Read-only Nix_predictor classification; never calls /process."""
    result = http_json(endpoint.rstrip("/") + "/classify", {"text": prompt}, timeout)
    return json.dumps(result, sort_keys=True)


def dashboard_classify_operation(prompt: str, endpoint: str, timeout: float) -> str:
    """Read-only Core classification endpoint; never writes a turn/event."""
    result = http_json(endpoint.rstrip("/") + "/api/classify", {"text": prompt}, timeout)
    return json.dumps(result, sort_keys=True)


def load_prompts(path: str | None) -> tuple[str, ...]:
    if not path:
        return DEFAULT_PROMPTS
    prompts = tuple(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
    if not prompts:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated NIX speed benchmarks.")
    parser.add_argument(
        "--benchmark",
        choices=("router", "predictor", "dashboard-classify"),
        default="router",
        help="Read-only benchmark target. No production writes are performed.",
    )
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--prompts", help="Newline-delimited prompt file")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--endpoint", help="Knowledge API or dashboard base URL")
    parser.add_argument("--output", help="Write JSON result to this path")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")

    prompts = load_prompts(args.prompts)
    if args.benchmark == "router":
        operation = core_router_operation
    elif args.benchmark == "predictor":
        endpoint = args.endpoint or os.environ.get("NIX_KNOWLEDGE_API_URL", "http://127.0.0.1:8100")
        operation = lambda prompt: predictor_operation(prompt, endpoint, args.timeout)
    else:
        endpoint = args.endpoint or "http://127.0.0.1:35567"
        operation = lambda prompt: dashboard_classify_operation(prompt, endpoint, args.timeout)

    summary = run_samples(
        args.benchmark,
        args.scenario,
        prompts,
        operation,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    payload = asdict(summary)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return 0 if summary.successful_samples == summary.samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
