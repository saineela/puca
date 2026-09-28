from __future__ import annotations

from .speed_benchmark import (
    BenchmarkSample,
    percentile,
    run_samples,
    summarize,
)


def test_percentile_is_deterministic() -> None:
    assert percentile([40.0, 10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([], 0.95) == 0.0


def test_summary_reports_latency_and_throughput() -> None:
    samples = [
        BenchmarkSample("a", 10.0, True, "ok"),
        BenchmarkSample("b", 20.0, True, "ok"),
        BenchmarkSample("c", 30.0, False, error="failed"),
    ]
    summary = summarize("test", "unit", samples, warmup_samples=1)
    assert summary.samples == 3
    assert summary.successful_samples == 2
    assert summary.p50_ms == 10.0
    assert summary.p95_ms == 20.0
    assert summary.requests_per_second > 0


def test_run_samples_records_failures_without_stopping() -> None:
    calls: list[str] = []

    def operation(prompt: str) -> str:
        calls.append(prompt)
        if prompt == "bad":
            raise RuntimeError("synthetic failure")
        return "ok"

    summary = run_samples(
        "test",
        "failure-recording",
        ("good", "bad"),
        operation,
        repeats=4,
        warmup=1,
    )
    assert len(calls) == 5  # one warm-up plus four measured calls
    assert summary.samples == 4
    assert summary.successful_samples == 2
    assert any(sample.error for sample in summary.samples_detail)
