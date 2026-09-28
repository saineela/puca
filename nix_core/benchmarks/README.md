# NIX speed benchmarks

This directory contains **non-production performance tests**. It does not change Core, Knowledge, Actions, model configuration, databases, or live conversations.

## What is measured

The harness reports:

- warm-up count
- successful samples
- minimum, mean, p50, and p95 latency
- maximum latency
- requests per second
- bounded result/error details for each sample

The first benchmark targets are read-only:

| Benchmark | What it measures | Side effects |
|---|---|---|
| `router` | Core deterministic routing | None |
| `predictor` | HTTP latency of Nix_predictor `/classify` | No database writes |
| `dashboard-classify` | Dashboard/Core classification endpoint | No database writes |

The harness deliberately does **not** benchmark `/api/send` by default because that endpoint can log sessions, learn keys, create events, or schedule actions. A future Casper benchmark must use a copied database or a direct isolated model client.

## Run safe benchmarks

From the repository root:

```bash
# Pure Python routing baseline; no services or models needed
PYTHONPATH=nix_core python nix_core/benchmarks/speed_benchmark.py \
  --benchmark router --repeats 100 --warmup 10

# Nix_predictor, read-only classification against the running service
PYTHONPATH=nix_core python nix_core/benchmarks/speed_benchmark.py \
  --benchmark predictor --repeats 10 --warmup 2 \
  --scenario predictor-baseline \
  --output /tmp/nix-speed-predictor.json

# Core dashboard classification, read-only
PYTHONPATH=nix_core python nix_core/benchmarks/speed_benchmark.py \
  --benchmark dashboard-classify --repeats 10 --warmup 2 \
  --scenario core-classify-baseline \
  --output /tmp/nix-speed-core.json
```

The predictor benchmark requires the Knowledge API and may load Nix_predictor. The dashboard benchmark requires the console. Do not run live model benchmarks while another training or inference job is using the GPU.

## Prompt sets

Use a newline-delimited prompt file for repeatable scenarios:

```text
hi
what do I have tomorrow?
I have robotics practice tomorrow at 5pm and remember that I joined TSA
What do you remember about robotics?
My sister is doing alright now
Tell me a short joke
```

Recommended separate scenarios:

- `simple-chat.txt` — greetings and short conversational turns.
- `personal-recall.txt` — read-only memory queries.
- `multi-intent.txt` — mixed calendar/fact/person requests.
- `temporal.txt` — relative date and time queries.

Keep mutation prompts out of live benchmark files. Test event creation/update/delete only against a temporary copied database.

## Optimization phases

Every candidate should be compared with the same prompt file, warm-up count, repeat count, and service state.

### Phase A: baseline

Measure router, predictor, and Core classification independently.

### Phase B: predictor latency

Compare:

- current long prompt/few-shot context
- shorter predictor prompt
- lower `max_new_tokens`
- early stop at `</tool_call>`
- multi-action prompt variants

Accuracy must be reported alongside latency. A faster misroute is not an improvement.

### Phase C: pipeline overlap

Measure a copied/in-process harness for:

- parallel session-context and tone retrieval
- parallel read-only multi-action Knowledge calls
- serialized writes for calendar/person-state mutations

### Phase D: Casper runtime

Only after a safe isolated Casper harness exists, compare:

- dynamic versus static KV cache
- eager versus `torch.compile`
- SDPA versus compatible Flash Attention
- context-length buckets
- reduced output-token ceilings

Each candidate must record cold-start and warm-request latency separately.

### Phase E: serving engines

vLLM, SGLang, TensorRT-LLM, and speculative decoding are separate migration experiments. They must not replace the working Transformers/QLoRA path until output correctness, VRAM, and power usage are measured.

## Result policy

Benchmark outputs belong outside the repository or in ignored result directories. Do not commit:

- model outputs containing personal data
- live database copies
- request logs
- GPU traces containing prompts
- downloaded weights or adapters

The root `.gitignore` excludes benchmark results, caches, databases, logs, model artifacts, and temporary sandboxes while keeping this benchmark source tracked.
