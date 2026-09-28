# Casper V6 / Nix PUCA V4 Research and Design

## Purpose

Casper V5 improved surface warmth, but the request logs show that surface style alone is not enough. The next version must learn conversational behavior as a policy: when to answer, acknowledge, clarify, stop, remember, or invoke a tool. It must sound natural without becoming an artificial imitation of a person or inventing memory.

This document records the research basis and the implementation plan. V5 remains the active runtime until V6 passes the held-out comparison suite.

## Research reviewed

### Human-like response preference tuning

**Enhancing Human-Like Responses in Large Language Models** (2025)
<https://arxiv.org/html/2501.05032v1>

Relevant findings and limitations:

- Human-like style is improved by comparing conversational and formal responses rather than relying on one style prompt.
- DPO can directly prefer natural responses over assistant boilerplate.
- Training must retain separate general-knowledge and conversational examples to avoid making the model casual everywhere.
- The paper's synthetic preference data is a useful method, but its generation models and evaluation setup are not proof that a model is human.

Application to Casper:

- V6 uses chosen/rejected response pairs.
- Rejected answers explicitly include the failures seen in logs: generic questions, fake memory, over-explaining, forced empathy, repeated acknowledgements, and task answers that lose dates.
- Preference data is tagged by response act, emotional state, grounding requirement, and question budget.

### Multi-turn emotional consistency

**DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset** (2025)
<https://arxiv.org/html/2505.19978>

Relevant findings:

- Multi-turn coherence and emotional consistency degrade in smaller models after roughly six turns.
- Concrete domains produce more grounded and meaningful conversations than abstract prompts.
- Emotion should be represented as a progression across turns, not as an isolated label.
- Cross-model dialogue generation can reduce self-reinforced repetitive patterns.

Application to Casper:

- V4 stores compact dialogue state instead of replaying unlimited raw history.
- Evaluation includes 2-, 4-, 6-, and 8-turn tests with changing emotion.
- Emotional state is a policy input, not a command to produce exaggerated empathy.
- Concrete personal domains—family, schedules, school, routines, health-state updates—receive dedicated held-out tests.

### Human conversational dynamics

**Modeling natural conversational dynamics with Seamless Interaction** (Meta FAIR, 2025)
<https://ai.meta.com/blog/seamless-interaction-natural-conversational-dynamics/>

Relevant findings:

- Natural interaction includes listening behavior, turn-taking, interpersonal stance, familiarity, and synchrony.
- The released dataset uses more than 4,000 hours and 4,000 participants, with familiar and unfamiliar relationships and long-tail stances.
- Evaluation should measure both speaking and listening behavior, not only generated text quality.

Application to Casper:

- V4 introduces response acts: `acknowledge`, `answer`, `clarify`, `reflect`, `brief_support`, `tool_result`, `boundary`, and `end`.
- The assistant is allowed to stop. A question is an explicit act with a budget, not a default ending.
- Voice-facing integrations can later map response acts to backchannel timing without contaminating the text model.

### Turn-taking and backchannel

**Predicting Turn-Taking and Backchannel in Human-Machine Conversations** (ACL 2025)
<https://aclanthology.org/2025.acl-long.743/>

The PDF endpoint is not directly machine-readable in this environment, but the published result motivates separating turn-taking decisions from response wording. V4 therefore keeps floor-control policy outside Casper's prose generation.

## Why V5 feels covered-up

The current V5 path has several structural causes:

1. The public corpus builder is mostly single-turn adjacent response pairs.
2. The training objective is SFT; it does not directly penalize a bad response against a better response for the same prompt.
3. The system prompt asks for warmth and familiarity, which can produce a consistent assistant mask without teaching when warmth is inappropriate.
4. The runtime injects large memory/context blocks but does not expose a compact response-act decision to Casper.
5. Log failures were fixed in Core after the fact, but the model was not trained against those exact boundary cases.
6. Model behavior and system policy are mixed: Casper is asked to decide tone, memory use, question use, and task wording at once.

V6 must not be another prompt-only personality patch.

## Casper V6 training design

### Stage A: behavior SFT

Use a small, high-quality conversational corpus with:

- Human-written DailyDialog-style everyday turns.
- Emotionally consistent multi-turn examples inspired by DeepDialogue.
- Curated Casper response-act examples.
- Placeholder personal-memory examples, never real user records.
- Explicit negative examples excluded from SFT and placed in preference training.

SFT teaches the response vocabulary and format, not the entire product policy.

### Stage B: preference optimization

Use DPO/IPO-style chosen/rejected pairs after the SFT adapter. Each pair shares the same user/context input and differs in behavior:

- chosen: short, specific, naturally bounded, grounded
- rejected: assistant boilerplate, forced question, fake memory, emotional overreach, wrong temporal interpretation, or unnecessary explanation

The current repository does not assume `trl` is installed. The V6 builder emits a stable preference JSONL contract first; a training runner can use TRL or a local DPO implementation after the environment is explicitly selected.

### Stage C: regression-directed refresh

Every confirmed log failure becomes:

- a held-out evaluation prompt;
- a chosen response-act target;
- a rejected response pattern;
- a runtime policy regression when a deterministic boundary is required.

No live personal log content or credentials may enter training data. Sensitive values are replaced with placeholders.

## V6 response contract

The generator receives a compact policy envelope:

```json
{
  "response_act": "answer|acknowledge|clarify|brief_support|tool_result|boundary|end",
  "question_budget": 0,
  "emotion": "neutral|positive|distressed|tired|uncertain",
  "grounding": "none|memory|knowledge_result|current_state|event",
  "current_state_over_history": true,
  "must_preserve": ["absolute date", "person name"],
  "raw_user_request": "...",
  "authoritative_context": "..."
}
```

Casper V6 verbalizes the envelope. It does not decide whether a database write happened, which person a clarification refers to, or what `tomorrow` means.

## Nix PUCA V4 architecture

```text
input / voice turn
       |
       v
[Perception]
  normalization, emotion/tone, entities, speech confidence
       |
       v
[Working state]
  bounded dialogue state, pending clarification, current user state
       |
       +--------------------+
       |                    |
       v                    v
[Dialogue policy]      [Event/alert planner]
  response act,         Qwen predictor + event gate +
  question budget,      temporal symbolic validator
  turn/floor policy            |
       |                       v
       |                 Knowledge + Actions
       |                       |
       +----------+------------+
                  v
        [Casper V6 verbalizer]
                  |
                  v
        [Grounding/style verifier]
                  |
                  v
             user reply
```

### Policy boundaries

- Perception may be uncertain; it must expose confidence.
- Current temporal person state outranks superseded history.
- The Event/Alert Gate can correct a mistaken predictor function but cannot resolve time itself.
- Knowledge and Actions return structured facts; Casper is the final speaker.
- The verifier rejects fabricated names, dates, tool success, and generic interview endings.
- A short social turn can use a fast response-act path without loading Nix_predictor.

## Evaluation plan

V6 is not accepted based on loss alone. Compare V5 and V6 on:

1. Naturalness: human preference for the chosen response.
2. Specificity: response addresses the actual turn without generic filler.
3. Question discipline: unnecessary-question rate.
4. Memory honesty: no invented personal facts.
5. Current-state correctness: current state wins over historical state.
6. Temporal correctness: exact local date/time and recurrence.
7. Clarification correctness: ambiguous people produce one clarification.
8. Multi-intent completeness: every independent action is preserved.
9. Emotional calibration: warmth without exaggeration or therapy-like language.
10. Latency and VRAM: p50/p95, peak VRAM, and power samples on the RTX 4060.

A model that is more human-like but less grounded does not pass.

## Data and licensing policy

Public datasets require source-license review before redistribution or training. The project should store source metadata, filtering rules, hashes, and split manifests. Personal chat logs are used only for local error analysis and anonymized regression prompts, never as an unreviewed public training corpus.

## V6 implementation run log

The first isolated training run used the local Qwen 3.5 4B checkpoint and an RTX 4060 with a 68% process memory cap.

Artifacts:

- `models/training_data/casper_v6_sft_train.jsonl` — 1,003 SFT rows.
- `models/training_data/casper_v6_sft_eval.jsonl` — 250 SFT evaluation rows.
- `models/training_data/casper_v6_preferences.jsonl` — 6 chosen/rejected rows.
- `models/nixlm/casper-puca-qlora-v6-sft` — first SFT adapter.
- `models/nixlm/casper-puca-qlora-v6` — preference-tuned adapter.
- `models/training_data/casper_v6_targeted_train.jsonl` — 588 balanced targeted rows.
- `models/nixlm/casper-puca-qlora-v6-targeted-sft` — targeted behavior adapter.

Observed training results:

- General SFT: 125 optimizer steps, final logged loss 1.8452, no OOM.
- Preference pass: 18 steps, mean loss approximately 0.69, no OOM.
- Targeted SFT: 73 steps, final logged loss 1.5254, no OOM.
- Peak observed VRAM remained below the 8 GiB RTX 4060 limit.

The direct model-only behavioral score was 3/8 for the first SFT/DPO adapter and 4/8 for the targeted adapter. Neither adapter is approved for production. The failures prove that identity, people, current-state, and action-result behavior must remain governed by the PUCA V4 structured policy envelope and Core/Knowledge boundaries rather than raw Casper generation.

A policy-conditioned adapter was then trained with 192 examples containing response acts, question budgets, grounding sources, current-state rules, and authoritative result context. It scored 7/8 on the structured-context evaluation; the remaining issue was a literal test matcher that expected `A or B` while the model correctly returned `sister A or sister B`. Core also gained post-generation removal of hidden `<think>` and template-control traces plus nonessential tired-user question suppression. The policy-conditioned adapter remains isolated until a broader integrated evaluation confirms no regression on normal conversation.

The final approved preference manifest contains 144 rows: 48 internally approved single-turn pairs and 96 internally approved multi-turn pairs. The multi-turn SFT manifest contains 96 rows, and the final policy-conditioned SFT manifest contains 288 rows.

The final adapter `models/nixlm/casper-puca-qlora-v6-final` completed 144 preference steps with a final logged loss of 0.6641 and no OOM. Policy-conditioned single-turn evaluation scored 7/8. Real multi-turn model evaluation scored 3/5: state supersession and clarification continuation passed, while emotional continuity, positive-to-ambiguity handling, and memory-plus-chat still need improvement. The final adapter is therefore preserved for further work but is not activated.

V5 remains the production adapter until an integration-level V6 candidate passes the complete gate. A fresh direct run on 2026-09-21 scored the selected V6 evaluation adapter 3/8 (tired-user discipline, current-state grounding, person ambiguity, multi-intent preservation, and identity failed), so V6 was not activated.

## September 22 latency and behavior audit

A live dashboard probe against the active V6 adapter measured these warm/cold examples:

| Prompt | First/warm observed latency | Result |
|---|---:|---|
| `hello? are you alive` | 5.5 s | V6 emitted an AI disclaimer and generic interview question |
| `hi` | 2.0 s | short but still generic greeting |
| `do you prefer SpongeBob popsicles or strawberry shortcake popsicles?` | 2.7 s | refused to express a harmless preference |
| `what is 2 plus 2` | routed through the same model path | requires a separate factual baseline |

The dashboard's old “Core routing” label was misleading: the predictor and Knowledge stages were skipped, while most elapsed time was the final V6 generation/cold-start path. The current Transformers wrapper also used a 120-token ceiling for ordinary turns. Core now bypasses the model for the high-confidence social alive check and bounded food/drink preferences, and ordinary non-thinking generations use `CASPER_FAST_MAX_NEW_TOKENS=64` by default. Complex turns retain the 120-token ceiling.

The new independent `nix_core/routing_engine.py` is the permanent route boundary. It returns a typed decision without HTTP, database, Qwen, or Casper work. A 100-sample local benchmark measured p95 decisions below 0.16 ms for the tested Knowledge paths and below 0.01 ms for social/world-chat paths. Its `requires_model` field makes any future learned router an explicit fallback rather than a hidden latency tax.

The model itself is still not certified by this optimization. The local V6 artifacts contain public-dialogue mixes, internal synthetic preference pairs, and policy-conditioned rows, but a fresh direct run of the current final adapter on the five multi-turn cases scored **1/5**: only state supersession passed. It failed positive-to-family clarification, tired-user question discipline, clarification continuation, and memory-plus-chat. The newly added alive-check and low-stakes-preference examples are regression/training inputs; they are not claimed to be trained into the existing adapter until a new adapter is actually produced and rerun through the held-out gate.

A direct isolated generation benchmark (same system prompt, 64-token ceiling, after one model load) measured approximately 8.35 s to load the base+adapter, then p50 1.64 s and max 4.28 s across five short prompts. Outputs still contained assistant-like behavior, including “How can I help?”, a digital-life disclaimer, and refusal to express a preference. This confirms that lowering the token ceiling helps the budget but cannot repair the adapter's behavior by itself.

Optimization research reviewed for this pass:

- Hugging Face assisted decoding: a smaller same-tokenizer assistant can draft tokens and let the target verify them in one pass; this is a future option, but loading another model is not acceptable until VRAM is measured on the 4060.
- Hugging Face KV-cache guidance: cache strategy and bounded context affect autoregressive latency; the current runtime already enables `use_cache=True`, but should benchmark static cache/compiled variants separately.
- PyTorch Performance Tuning Guide: inference mode, kernel fusion/`torch.compile`, and avoiding unnecessary work are relevant; compilation is not enabled blindly because quantized PEFT models can regress or consume extra VRAM.

Next benchmark gate: compare V5 and V6 with identical prompts after warm-up, record p50/p95, first-token/complete latency when available, peak VRAM, response length, and the 9-case single-turn plus 5-case multi-turn quality suites. Speed alone cannot promote V6.
