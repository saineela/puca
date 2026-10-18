"""Benchmark the Luna model path and the PUCA system pipeline.

Both measurements go through the user-facing API, never a private shortcut:

* the **PUCA system model** is the full Core turn (``POST /api/send``), which
  includes routing, Knowledge, Actions and the companion's reply;
* the **Luna model path** is the OpenAI-compatible completion
  (``POST /v1/chat/completions``) that Open WebUI and the SDKs use.

Prompts come from the owner's real chat history (``GET /api/sessions``) unless
``--prompt-file`` or ``--prompt`` is supplied. Every request is timed end to
end, and the routing decision is measured separately with
``POST /api/classify`` so model time is never reported as routing time.

    python3 benchmarks/system_model_benchmark.py --limit 20 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = os.environ.get("NIX_BASE_URL", "http://127.0.0.1:49117")
FIXED_PROMPTS = [
    "Hi",
    "What is my locker code?",
    "Remind me to call mom at 6pm",
    "How is my mother doing right now?",
    "What effect is on the light?",
    "Turn the ring light blue",
    "Where do I live?",
    "Tell me a one-line joke",
    "What is the capital of Australia?",
    "You there?",
]


def http(method: str, url: str, body: dict | None = None, timeout: float = 90.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "accept": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read() or b"{}")
            status = response.status
    except urllib.error.HTTPError as exc:
        payload = {"error": f"HTTP {exc.code}"}
        status = exc.code
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        status = 0
    return status, payload, (time.perf_counter() - started) * 1000


def history_prompts(base: str) -> list[str]:
    status, payload, _ = http("GET", f"{base}/api/sessions", timeout=15)
    if status != 200:
        return []
    prompts: list[str] = []
    for session in payload.get("sessions") or []:
        for turn in session.get("turns") or []:
            if turn.get("role") == "user" and str(turn.get("content") or "").strip():
                prompts.append(str(turn["content"]).strip())
    return list(dict.fromkeys(prompts))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return round(ordered[index], 1)


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == 200]
    latencies = [row["latency_ms"] for row in ok]
    return {
        "name": name,
        "requests": len(rows),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(rows), 4) if rows else 1.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "min": round(min(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "non_empty_replies": sum(
            1 for row in ok if str(row.get("reply") or "").strip()
        ),
    }


def benchmark(base: str, prompts: list[str], model: str, timeout: float) -> dict[str, Any]:
    routes: dict[str, int] = {}
    routing_rows: list[dict[str, Any]] = []
    pipeline_rows: list[dict[str, Any]] = []
    luna_rows: list[dict[str, Any]] = []

    for index, prompt in enumerate(prompts, start=1):
        status, payload, latency = http(
            "POST", f"{base}/api/classify", {"text": prompt}, timeout=timeout
        )
        route = str(
            payload.get("route")
            or payload.get("final")
            or (payload.get("deterministic") or {}).get("route")
            or (payload.get("routing_engine") or {}).get("route")
            or ""
        )
        routes[route or "unknown"] = routes.get(route or "unknown", 0) + 1
        routing_rows.append({"prompt": prompt, "status": status, "latency_ms": latency, "route": route})

        status, payload, latency = http(
            "POST",
            f"{base}/api/send",
            {"text": prompt, "location": "benchmark"},
            timeout=timeout,
        )
        pipeline_rows.append({
            "prompt": prompt,
            "status": status,
            "latency_ms": latency,
            "reply": str(payload.get("reply") or payload.get("error") or "")[:400],
            "route": str(payload.get("route") or ""),
        })

        status, payload, latency = http(
            "POST",
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
            },
            timeout=timeout,
        )
        choices = payload.get("choices") or []
        reply = ""
        if choices and isinstance(choices[0], dict):
            reply = str((choices[0].get("message") or {}).get("content") or "")
        luna_rows.append({
            "prompt": prompt,
            "status": status,
            "latency_ms": latency,
            "reply": reply[:400],
        })

    return {
        "base_url": base,
        "model": model,
        "prompts": len(prompts),
        "routes": routes,
        "routing": {**summarize("routing-classify", routing_rows), "rows": routing_rows},
        "puca_system_model": {**summarize("puca-system-model", pipeline_rows), "rows": pipeline_rows},
        "luna_model_path": {**summarize("luna-model-path", luna_rows), "rows": luna_rows},
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# PUCA system model benchmark — {report['base_url']}",
        "",
        f"Model: `{report['model']}` · prompts: {report['prompts']} (real chat history unless overridden)",
        "",
        "| Measurement | ok/requests | error rate | p50 ms | p95 ms | mean ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, label in (
        ("routing", "routing only (POST /api/classify)"),
        ("puca_system_model", "PUCA system model (POST /api/send)"),
        ("luna_model_path", "Luna model path (POST /v1/chat/completions)"),
    ):
        row = report[key]
        lines.append(
            f"| {label} | {row['ok']}/{row['requests']} | {row['error_rate']:.3f} | "
            f"{row['latency_ms']['p50']} | {row['latency_ms']['p95']} | {row['latency_ms']['mean']} |"
        )
    lines += ["", f"Route mix: {report['routes']}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", default="luna-v6-contextual-v1")
    parser.add_argument("--limit", type=int, default=0, help="0 means every prompt")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    prompts: list[str] = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            prompts.extend(
                line.strip() for line in handle if line.strip() and not line.startswith("#")
            )
    if not prompts:
        prompts = history_prompts(args.base)
    if not prompts:
        prompts = FIXED_PROMPTS
    if args.limit:
        prompts = prompts[: args.limit]

    report = benchmark(args.base, prompts, args.model, args.timeout)
    text = markdown(report)
    print(text)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
