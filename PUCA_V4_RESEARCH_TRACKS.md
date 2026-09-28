# PUCA V4 Research Tracks

Date: 2026-09-21

Four independent research tracks were reviewed before selecting implementation work. Search results were treated as leads; primary papers and engineering reports were preferred over blog claims.

## Track 1 — Data quality, conversational alignment, and model improvement

### Primary sources

- [A Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications](https://arxiv.org/html/2410.15595v4)
- [Fine-tuning Large Language Models with Limited Data](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.627/136154/Fine-tuning-Large-Language-Models-with-Limited)
- [DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset](https://arxiv.org/html/2505.19978)
- [Modeling natural conversational dynamics with Seamless Interaction](https://ai.meta.com/blog/seamless-interaction-natural-conversational-dynamics/)

### Findings

- SFT should establish basic response behavior before preference optimization.
- DPO is relatively simple and stable, but it has alignment-tax, reward-hacking, out-of-distribution, and reference-model sensitivity problems.
- More preference rows are not automatically better. Pair quality, response-act coverage, and hard negatives matter more for Casper's failure modes.
- Instance-level pairs are coarse. Casper needs structured labels for response act, question budget, grounding source, temporal preservation, and emotional calibration.
- Emotion is a multi-turn trajectory, not a permanent label inferred from one message.
- Natural conversation includes turn-taking, stopping, listening, and familiarity—not just warmer wording.

### Adaptive thinking/routing research

[Adaptive Routing for LLM and Reasoning Strategy Selection](https://arxiv.org/html/2505.19435v1), the [Adaptive and Controllable Test-Time Compute survey](https://arxiv.org/html/2507.02076v1), and [Plan and Budget](https://openreview.net/forum?id=ctspw4CqbS) all support assigning reasoning budgets by task difficulty rather than enabling extended reasoning globally. Recent routing work also emphasizes pre-judgment routing: decide before generation, keep the router cheaper than the work it avoids, and measure false escalation and false de-escalation separately. The practical lesson for Nix is conservative routing: deterministic fast paths for ordinary turns, Qwen2.5-0.5B only for already-complex-looking ambiguous requests, explicit boolean parsing, and FAST as the failure fallback. A small router must not be trusted to override obvious safety/product boundaries. The implementation records `thinking_source`, skips Qwen for obvious everyday turns, parses boolean decisions explicitly, and treats malformed values such as the string `"false"` as FAST rather than relying on Python truthiness.

### Decision

V6 data uses:

1. filtered public dialogue for broad style;
2. targeted, anonymized behavior examples for known failures;
3. chosen/rejected pairs with explicit failure tags;
4. held-out cases that are never used as training rows.

We will not optimize for “human-like” at the expense of honesty. Casper must be natural but clearly not claim human experience or consciousness.

## Track 2 — Algorithms and model/inference architecture

### Primary sources

- [Dovetail: CPU/GPU Heterogeneous Speculative Decoding](https://arxiv.org/html/2412.18934v2)
- [Self-Speculative Decoding with Hierarchical Quantized KV Cache](https://machinelearning.apple.com/research/quantspec)
- [BatchLLM: Global Prefix Sharing and Throughput-oriented Token Batching](https://arxiv.org/html/2412.03594v1)
- [Efficient Inference for Edge Large Language Models](https://www.sciopen.com/article/10.26599/TST.2025.9010166)

### Findings

- Speculative decoding helps when the draft model is cheap and has a high acceptance rate; it is not automatically faster on every single-user workload.
- CPU/GPU heterogeneous speculative decoding can help constrained devices, but CPU verification and transfer costs can erase gains.
- Prefix sharing and KV reuse are mainly valuable when multiple requests share a stable prompt prefix. Casper's personal memory block changes frequently, so naive global caching could expose stale personal context.
- Continuous batching and paged KV memory target multi-request serving more than a single voice turn.
- A fixed small output budget, bounded context, 4-bit weights, SDPA, and no-thinking simple turns are currently higher-confidence wins for this RTX 4060.

### Decision

Prioritize, in order:

1. measure TTFT and decode separately;
2. stabilize prompt/policy prefix boundaries;
3. add safe per-session prefix caching only for immutable system/policy text;
4. benchmark speculative decoding with a small draft model as an opt-in experiment;
5. do not introduce CPU/GPU offload or continuous batching into live PUCA until measurements justify it.

No cache may contain personal memory without a session/user key and invalidation version.

## Track 3 — Multi-agent systems, orchestration, and personal assistants

### Primary sources

- [How we built our multi-agent research system — Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- [The Orchestration of Multi-Agent Systems: Architectures, Protocols, and Enterprise Adoption](https://arxiv.org/html/2601.13671v1)
- [AI agent memory frameworks survey](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)

### Findings

- Orchestrator-worker systems are effective when subtasks are independent and can be explored in parallel.
- Multi-agent systems consume substantially more tokens and introduce coordination, state, and reliability failures.
- Parallel research benefits from separate context windows and a final synthesis/citation stage.
- Shared mutable personal memory is a poor fit for free-form agent debate. Agents can duplicate writes, conflict over temporal truth, or amplify hallucinations.
- Tool descriptions, effort budgets, observability, and outcome-based evaluation are more important than simply adding agents.

### Decision

PUCA V4 uses two distinct modes:

#### Live companion mode

```text
Core orchestrator
  ├── deterministic Perception/policy
  ├── Knowledge worker (read/write, temporal truth)
  ├── Actions worker (schedules/reminders)
  ├── verifier
  └── Casper V6 verbalizer
```

Only Knowledge may commit personal memory, only Actions may commit schedules, and Casper may not commit either.

#### Offline research/evaluation mode

```text
Research lead
  ├── data/alignment scout
  ├── inference scout
  ├── agent-architecture scout
  ├── benchmark scout
  └── citation/synthesis verifier
```

This is where parallel agents are appropriate: results are immutable reports until a human or deterministic implementation step accepts them.

## Track 4 — Proven speed techniques not fully implemented

### Strong candidates

1. **Separate TTFT from decode latency.** Current aggregate timing hides which stage is slow.
2. **Immutable prompt-prefix caching.** Cache only the stable policy/system prefix; invalidate on model, policy, or tokenizer changes.
3. **Bounded prompt construction.** Keep memory and history selection deterministic before tokenization.
4. **Static generation limits by response act.** Greetings and acknowledgments need much smaller limits than tool-result composition.
5. **SDPA/Flash Attention compatibility tests.** Enable only after correctness and VRAM measurements.
6. **CUDA graph capture for fixed-shape decode.** A later experiment; dynamic conversation lengths reduce benefit.
7. **Speculative decoding.** An opt-in benchmark, not an assumed production change.
8. **Continuous batching/paged attention.** Useful if Casper becomes a concurrent server; low priority for one voice user.

### Low-confidence or deferred candidates

- CPU/GPU offload for the current 4B QLoRA model: likely slower and adds transfer complexity.
- Loading multiple autonomous models concurrently: increases OOM risk and makes latency less predictable.
- Unbounded multi-agent debate: high token cost and poor fit for personal-state writes.
- Large external memory frameworks: unnecessary while Nix Knowledge already has temporal and semantic stores.

## Implementation order

1. Use V4 policy envelopes in the final Casper prompt.
2. Add integration tests that distinguish raw-model evaluation from policy-grounded evaluation.
3. Instrument prefill/decoding timing and prompt token counts.
4. Add immutable-prefix cache with versioned invalidation.
5. Benchmark speculative decoding and CUDA graph options in isolation.
6. Expand preference data using human review of hard negatives, not automatic synthetic volume.
7. Activate V6 only after it beats V5 on grounding, question discipline, current-state correctness, naturalness, and latency.
