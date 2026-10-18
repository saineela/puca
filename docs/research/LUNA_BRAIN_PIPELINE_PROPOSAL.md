# Luna Intelligence & Dynamic Skill Orchestration — revision 6

**Status:** approved · **Author:** automated agent ·
**Responds to:** owner decisions *changes-requested* 2026-10-17T16:13:45Z and
**2026-10-17T16:41:40Z**, and owner approval **2026-10-18T04:45:23Z** ·
**Boundary:** Luna's prompt, persona, generation settings and weights are untouched.
Approved 2026-10-18; staged implementation proceeds under the gates in §8. No
phase is promoted without evidence.

Revision 4 answers the four remaining requests from the second review:
numbering below follows the review's own numbering. Revision 6 (2026-10-18)
aligns this plan with the PUCA evidence completion proposal (WP1–WP5), adds the
explicit `unknown` abstention outcome required by that work, and records that
WP1 found the pipeline already abstains — the 0.000 came from the D2 scorer and
from baselines sharing the PUCA pipeline. It does not change the architecture or
the approval conditions.

### Implementation status (2026-10-18, approved)

The owner approved revision 4 with four conditions. The first two now have
code, and the other two are standing constraints:

1. **The promotion/non-inferiority distinction is documented** in §7.1 and
   §8.3, and enforced by `nix_core/eval/acceptance.py`: non-inferiority
   (lower bound above −0.05) is necessary, promotion requires the stricter
   interval (lower bound at least 0).
2. **The gate is executable and tested.** `acceptance.decide_feature` requires
   the sample size, paired interval, safety count, latency measurement and
   vocabulary count, and returns `promote`, `hold_in_shadow` or
   `rollback_to_off`. `eval/feature_flags.py` implements
   `NIX_<PHASE>_MODE = off | shadow | on`, with promotion only from a passing
   gate; both have tests.
3. **Safety boundaries are unchanged:** lexical retrieval stays authoritative
   until promotion, the workspace cannot write canonical memory, and
   destructive, bulk or external actions keep their confirmation requirement.
4. **No validation is claimed by this design.** Every phase stays off or in
   shadow until its own gate passes; nothing here claims production readiness.

---

## 0. The review, item by item

The owner approved the architecture direction, not the implementation plan as
written, and asked for eight changes. This revision makes each one explicit.
Where a requested change also exposed a defect in something already shipped,
the defect is corrected now and named as such; everything else stays paused.

| # | Review asks for | This revision does | Where |
| --- | --- | --- | --- |
| 1 | Separate intent, capability selection and computational effort. Salience must not decide whether a request deserves a tool. | Intent is its own enum (`conversation`, `question`, `instruction`, `correction`, `answer`, `statement`). Only `conversation` bypasses the request machinery. Salience now budgets verification, replanning, reads and recovery — never routing. The shipped gate was corrected to match. | §2, §4.1 |
| 2 | Define `TurnWorkspace` as a shared typed interface, not a monolithic orchestrator; specialists keep autonomy; conflicts, provenance, partial results and recovery without bypassing safety. | A contribution protocol with typed kinds, precedence rules, provenance, deadlines, typed partial outcomes and per-specialist ownership. One response boundary remains. | §3 |
| 3 | Introduce semantic capability retrieval in shadow mode first; measure recall@k, false-tool, missed-tool, selection accuracy against lexical and hybrid; hard negatives; fallback; no suppression until held-out results. | Shadow-first design with a promotion gate and a fallback rule; the lexical path stays authoritative until the gate passes. | §4.2 |
| 4 | Strengthen execution semantics: idempotency without blocking legitimate repeats; verify before retry; executable postconditions; confirmation; timeouts; typed partial success; never claim success from `ok`. | Ten execution invariants, including request-scoped idempotency and a verify-before-retry rule. | §5 |
| 5 | Reframe Part II as neuroscience-inspired engineering hypotheses; state mechanism and expected software benefit for every mapping; remove unsupported latency claims; every feature independently ablatable. | Part II is now a hypothesis register: mechanism, expected benefit, ablation, falsifier. No latency number is claimed; latency is an acceptance gate that must be measured. | Part II |
| 6 | Strengthen memory safety before consolidation: preserve source, timestamp, speaker, assertion status, temporal validity; repetition is not truth; consolidation yields reviewable proposals; conflicting/superseded records preserved; explicit deletion across stores. | A memory-safety contract and a hold on consolidation until every field and deletion path exists. | §6 |
| 7 | Strengthen the evaluation protocol: freeze baseline, held-out and adversarial sets, per-class counts and CIs, identical cases per ablation, 29-turn suite is regression only, justify sample size, separate failure classes. | A protocol with a frozen baseline hash, per-class reporting, paired comparisons on identical cases, a sample-size rationale and a failure taxonomy. | §7 |
| 8 | Defer habit caching and autonomous idle-time processing until core routing/execution gains; later caching behind flags with invalidation; idle processing must not mutate canonical memory or act. | Both are removed from the near-term plan and listed as deferred with hard constraints. | §8.4 |

### 0.1 Second review (2026-10-17T16:41:40Z) — the four remaining requests

| # | Review asks for | This revision does | Where |
| --- | --- | --- | --- |
| 1 | Reconcile the statistical acceptance criteria; pre-register the interval-based pass criterion and calculate sample size and permitted error counts for both rates | One pre-registered criterion set: 95% Wilson upper bound below the target for each rate, with a permitted-error table and the minimum n. §8.5 now cites §7 instead of restating a different rule. Implemented in `eval/acceptance.py` and tested. | §7.1, §8.5 |
| 2 | Clarify the statement / no-action path; define and test three outcomes without using salience as the decider | A statement is now three outcomes — `none`, `memory_write`, `clarify` — decided by its content, never by its budget. Ordinary statements are answered without retrieval or planning. Implemented in `turn_state.py` with unit and end-to-end tests. | §2, §4.1 |
| 3 | Replace the generic workspace payload with enforceable schemas; define missing provenance, invalid contributions and conflicting authoritative sources | Each contribution kind has a required-field schema, a validator result, and a handler for missing provenance, invalid contributions and conflicts between authoritative sources. | §3.1, §3.5 |
| 4 | Turn rollback criteria into executable decision rules: paired comparison, interval, minimum sample, inconclusive handling; shadow stays when the gate is unmet | Rollback and promotion are now the same executable rule, with the comparison, interval, sample and the `hold_in_shadow` outcome written out. | §8.3 |

**What is already shipped and stays:** the voice pass, the source-level human
copy, the per-turn prediction with an intent gate, the clarification behaviour
that does not trap a topic change, and correction handling. **What is paused:**
semantic retrieval, the workspace refactor, consolidation, habit caching,
idle-time processing, and any change to the model or its prompt.

One correction made as part of item 1, stated plainly because it changes shipped
behaviour: the first version of the prediction used `salience < 0.45` as part of
the "this is conversation" test. That made effort a routing decision, which the
review correctly rejects. The test is now `read == conversation`; a statement
that must be stored is its own intent (`statement`), not a high-salience
conversation. The tests were updated with it.

---

## Part I — the audit this proposal is built on

### 1. What the pipeline did

```
text
 └─ brain.handle()
     ├─ deterministic guards        identity / secrets / injection / bulk delete
     ├─ pending clarification       only when the reply is answer-shaped
     ├─ prediction                  read, about, expect, salience        ← shipped
     ├─ routing decision            regex/symbolic + optional model gate
     ├─ _handle_chat()              targeted skills → planner → execute, or the model
     └─ _handle_knowledge()         rules → needle tool call → formatted answer
```

Three structural facts drove the proposal:

- routing asked "which route is this?" for every utterance, so abstention
  escalated to a model gate;
- capability candidates were selected lexically from device aliases;
- the planner was one-shot, with no observation or postcondition step.

### 2. Measured failures (2026-10-17)

| Observed | Attribution | Class |
| --- | --- | --- |
| "Where do I live?" → "I couldn't complete that request" | no rule, no model asked | architecture |
| "Where did I live before that?" → current city | no history mode | retrieval |
| "What's my dog called?" → not found | inflection mismatch | retrieval |
| "Delete all my reminders." → "Done." | no confirmation gate | safety |
| Injection turn → planner internals | failure text surfaced raw | voice |
| Duplicate events; six identical facts | no idempotency; dedupe unavailable | execution / infrastructure |
| "Luna Alert: Unable to reach the chat model (AcceleratorError)" | embedder took the chat model's GPU | infrastructure → voice |
| "How is my father doing?" → the mother's state | state filter matched any word | retrieval |
| "I didn't send a device command because the dedicated skill planner…" | pipeline wrote the reply | voice |
| "I don't have any information about your mother's current state." | empty state list paraphrased | composition |

The model was rarely the limiting factor; the failures were pipeline.

### 3. Corrections landed before this revision

Residence supersession and history rendering; inflection-aware recall;
idempotent calendar create; bulk-delete confirmation; instruction-override and
secret-display refusals; the voice pass; turn prediction with intent gating;
clarification continuation; correction handling; human copy at the source.

### 4. Research used in Part I

| Source | Finding | Use | Limitation |
| --- | --- | --- | --- |
| ReAct, arXiv:2210.03629 | act/observe beats one-shot planning | observation step | traces are not reliability |
| Toolformer, arXiv:2302.04761 | models can learn when *not* to call a tool | no-tool examples | curation |
| MCP | standard tool description | manifest shape | adoption optional |
| BFCL / Gorilla | measurement and failure taxonomy | metrics | public schemas ≠ ours |
| τ-bench, arXiv:2406.12045 | dual control, pass^k | sandbox | other domains |

**Rejected:** more intent regex; unrestricted autonomy; a bigger model first;
full rewrite.

---

## Part II — neuroscience-inspired engineering hypotheses

Part II is a **hypothesis register**, not a claim that the software reproduces
biology. Each entry states the mechanism, the expected software benefit, the
ablation that tests it, and what would falsify it. A hypothesis that cannot be
ablated does not ship. No biological mapping is offered as evidence of
correctness, and no latency improvement is claimed anywhere in this document.

| # | Hypothesis | Mechanism in the pipeline | Expected software benefit | Ablation |
| --- | --- | --- | --- | --- |
| H1 | Prediction before classification reduces false requests | a `TurnExpectation` is written before routing; next turn scores error | fewer conversation turns in request paths | on/off by flag; false-tool rate |
| H2 | One broadcast voice reduces machine-sounding replies | reply reads a typed broadcast, never planner state | internal vocabulary rate → 0 | voice pass on/off; vocabulary count |
| H3 | Fast/slow stores reduce duplicate and stale facts | episodic log separate from semantic records; offline promotion | duplicate writes, stale answers | store split on/off |
| H4 | Index rows make provenance answerable | every write carries episode/turn/time; retrieval returns them | "where did you learn that" accuracy | index on/off |
| H5 | Recall purity prevents silent rewrites | retrieval cannot write; only statements supersede | lost-fact rate after questions | write-on-read canary |
| H6 | NoGo gating catches unsafe actions earlier | deterministic veto before deliberation | unauthorized actions (must be 0) | guard removal per guard |
| H7 | Forward models reduce false success | declared postcondition checked against state | false-success rate | postcondition check on/off |
| H8 | Attention-style suppression improves selection | top-k capabilities, rest hidden | selection accuracy at equal context | k varied; catalogue dump |
| H9 | Salience budgets improve cost/quality trade-off | effort scalar after intent | verification coverage vs tokens | budget on/off |
| H10 | Event segmentation improves continuity | topic inheritance, boundaries | short follow-up accuracy | inheritance on/off |
| H11 | Mentalizing improves correction recovery | `expect` + prediction error feeds next retrieval | repeat-error rate | error feedback on/off |
| H12 | Reuse/degeneracy improve fallback | capability aliases and declared fallbacks | goal success when first path fails | alias/fallback off |

### H1 — prediction, not classification

**Mechanism.** A turn opens with an intent (`conversation`, `question`,
`instruction`, `correction`, `answer`, `statement`), the referents it continues,
what the speaker will probably do next, and a separate effort budget.
**Benefit.** Conversation stops entering request pipelines because its *intent*
is conversation, not because a score fell below a threshold.
**Ablation.** Prediction off; measure false-tool rate, missed-tool rate and
follow-up accuracy on identical cases.
**Falsifier.** No reduction in false-tool rate, or a rise in missed tools.

### H2 — one broadcast, one voice

**Mechanism.** Modules contribute typed content; only the broadcast is visible
to the reply writer. **Benefit.** Planner jargon and status codes cannot be
spoken, because they are not in the broadcast.
**Ablation.** Voice pass and broadcast rule off; count internal terms per reply.
**Falsifier.** Internal-vocabulary replies persist with the broadcast in place.

### H3–H5 — memory as memory

**Mechanism.** Episodic/semantic split, index rows, pure reads, supersession on
statements only. **Benefit.** Fewer duplicates, answerable provenance,
corrections that replace the right record.
**Ablation.** Each store/feature independently.
**Falsifier.** Duplicate rate or stale-answer rate does not move.

### H6 — NoGo before deliberation

**Mechanism.** Deterministic guards (privacy, injection, bulk destruction,
secret display, unconfirmed side effects) run before any model or planner.
**Benefit.** Unsafe actions are refused at a cost that does not depend on model
behaviour. **Ablation.** Guard-by-guard removal in the sandbox only.
**Falsifier.** Any unauthorized action with the guard enabled; the veto is
absolute.

### H7 — forward models and postconditions

**Mechanism.** A capability declares an executable expected outcome; the
executor checks authoritative state before reporting success.
**Benefit.** "Done" means the effect was observed. **Ablation.** Check disabled.
**Falsifier.** False-success rate unchanged.

### H8–H9 — selection and effort

**Mechanism.** Top-k capability retrieval with suppression; effort budgets for
verification and replanning. **Benefit.** A large catalogue behaves like a small
one; effort is spent where it changes the outcome.
**Ablation.** k varied; budget on/off. **Falsifier.** No selection or cost gain.

### H10–H12 — continuity, mentalizing, reuse

**Mechanism.** Topic inheritance at short continuations, event boundaries,
prediction error carried to the next turn, declared capability aliases and
fallbacks. **Benefit.** Fewer "new request" misreads, better correction
recovery, a working second path when the first fails.
**Ablation.** Each feature independently. **Falsifier.** No change in the
corresponding metric.

### What the brain is not telling us

No central controller, no unbounded autonomy, no single global classifier, no
decorative mirror-neuron story. Imitation is not implementation: every
hypothesis above ships only if its ablation moves a metric.

---

## 3. TurnWorkspace as a shared typed interface

The workspace is a **contract**, not an owner. Specialists keep their internals;
only typed contributions cross the boundary. The reply writer sees the
broadcast, never the components.

### 3.1 Contribution envelope and per-kind schemas

There is no generic `payload`. Every contribution carries the envelope below,
and its `payload` must validate against the schema for its `kind`. A payload
that does not validate is not a contribution.

```
Contribution {
  id:            string            # unique in the turn
  source:        "intent" | "memory" | "capability" | "planner" | "executor" | "safety" | "retrieval"
  kind:          "evidence" | "candidate" | "constraint" | "outcome" | "question"
  payload:       <schema for kind below>   # no free-form prose
  confidence:    0..1
  provenance:    Provenance | null
  deadline_ms:   integer | null
  conflicts_with:[contribution id]
}

Provenance {
  record_ids:  [integer]           # required, non-empty for evidence/outcome
  turn_id:     string              # required
  timestamp:   string (ISO-8601)   # required
  speaker:     "user" | "assistant" | "system"
  assertion:   "asserted" | "inferred" | "confirmed"
  store:       "episodic" | "semantic" | "device"
}
```

| Kind | Payload schema (required fields in bold) |
| --- | --- |
| **evidence** | **provenance**, **record_id**, **value_or_text**, valid_from, valid_until, superseded_by |
| **candidate** | **capability_id**, **score** 0..1, **retrieved_by** (`lexical` \| `semantic` \| `union`), matched_terms |
| **constraint** | **rule** (`nogo_privacy` \| `nogo_injection` \| `nogo_destructive` \| `nogo_secret` \| `user_requirement`), **deny** bool, reason |
| **outcome** | **status** (`ok` \| `ok_unverified` \| `empty` \| `unknown` \| `failed` \| `partial` \| `needs_user` \| `refused`), **capability_id**, observed_state, completed[], failed[], not_attempted[], idempotency_key |
| **question** | **slot** (`person` \| `time` \| `value` \| `confirmation` \| `choice`), **prompt**, candidates[] with ids |

### 3.2 Resolution

1. NoGo constraints apply first and cannot be overridden.
2. An explicit user instruction or correction outranks inferred evidence.
3. Verified evidence outranks model proposals.
4. Between equal-ranked contributions, the workspace asks one question rather
   than guessing.
5. Every resolution records which contributions won, which lost, and why.

### 3.3 Partial results and failure

An executor returns a typed outcome per step. A partial result names exactly
what succeeded and what did not; the broadcast carries both. A specialist
failure is a contribution (`outcome: failed`), not an exception that rewrites
the reply. No specialist may call the reply writer.

**`unknown` is its own outcome.** When retrieval has no supporting record, the
contribution carries `outcome: unknown`; the reply must say it does not know or
ask for the missing fact, and may not turn the absence into a positive claim.
This is the abstention path from the evidence completion proposal (WP1), and it
lives in the PUCA answer path, not in Luna's prompt or weights.

### 3.4 Missing provenance, invalid contributions, conflicting authorities

- **Missing provenance.** An `evidence` or `outcome` contribution without a
  valid `Provenance` is rejected at admission and never broadcast. If the turn
  has no other evidence, the workspace asks for the fact rather than answering
  from an unprovenanced value. There is no "downgrade to low confidence" path;
  unprovenanced content is not a weaker contribution, it is not one.
- **Invalid contribution.** Admission validates envelope + kind schema and
  returns a typed rejection `{status: "invalid", kind, field, reason}` to the
  contributor. The workspace records the rejection in telemetry; the
  contributor may re-propose once with the same id. Rejections never reach the
  reply and never silently disappear.
- **Conflicting authoritative sources.** Two authoritative contributions (two
  `confirmed` records for the same key, or a user instruction and a stored
  preference) are never averaged and never resolved by ordering of arrival.
  The precedence ladder is: safety constraint > explicit user instruction in
  this turn > `confirmed` record > `asserted` record > `inferred` record >
  model proposal. Within the same rank, the **newer timestamp** wins, and if
  timestamps tie the workspace asks one question citing both values. The loser
  is retained with `superseded_by`/`conflicts_with` links, so the decision is
  auditable and reversible.
- **Conflicting specialist outcomes.** Two executors reporting different
  observed states for the same postcondition are both recorded; the turn
  reports partial success and the observed values, and never claims the
  postcondition holds.

### 3.5 Invariants

- **I1** Only the broadcast feeds the reply.
- **I2** NoGo constraints are evaluated before any other contribution.
- **I3** Every evidence and outcome contribution carries provenance.
- **I4** A contribution cannot mutate another specialist's state.
- **I5** A plan step cannot report success without an observed postcondition.
- **I6** A clarification question names the missing slot.
- **I7** Internal identifiers never enter the broadcast's voice field.
- **I8** A specialist that misses its deadline reports `failed`/`partial`, not
  silence.
- **I9** The workspace cannot write canonical memory.
- **I10** The same inputs compose the same broadcast (deterministic ordering).

---

## 4. Intent, capability selection, effort

### 4.1 Three separate decisions

1. **Intent** (`conversation`, `question`, `instruction`, `correction`,
   `answer`, `statement`) decides which pipeline a turn enters.
2. **Capability selection** decides which tools/knowledge are candidates. It is
   independent of intent: a question can require a tool; a statement can
   require memory; a conversation can mention a device without requesting one.
3. **Effort** (salience) decides how much verification, replanning, reading and
   recovery the chosen path gets. It never decides *whether* a path is chosen.

This is the first review's change, implemented in `nix_core/turn_state.py`:
`conversational` no longer reads salience at all; salience only sets budgets.

### 4.1.1 A statement has three outcomes, decided by content

The second review asked for this explicitly, and it is now implemented and
tested. `statement` carries a separate field, `statement_action`:

| Outcome | Recognised by | Pipeline behaviour | Test |
| --- | --- | --- | --- |
| `none` | declarative shape, no persistable content ("I started a pottery class last week") | answered as talk: no retrieval, no planner, no memory write | `test_an_ordinary_statement_never_reaches_retrieval_or_planning` |
| `memory_write` | a fact, preference or plan with a value ("my sister is called Priya", "my flight is Friday at 9am") | routed to memory by its own intent (`statement_memory_write`), not by a wording contract | `test_a_memory_statement_is_not_an_instruction`, `test_a_calendar_statement_is_its_own_intent_not_conversation` |
| `clarify` | fact shape with the value missing ("my sister's name is") | one question, deterministically; nothing stored, no model call | `test_a_statement_with_a_missing_value_asks_once_and_stores_nothing` |

Salience is not consulted for any of the three: the same outcome holds at
opposite effort scores (asserted in `test_the_three_statement_outcomes_are_decided_separately_from_effort`).

### 4.2 Semantic retrieval, shadow first

The lexical path stays authoritative until a promotion gate passes.

- **Shadow run:** for every turn that reaches selection, compute semantic top-k
  alongside lexical. Log both, the overlap, the score margin and the latency —
  never the prompt text.
- **No suppression:** the planner receives the lexical set exactly as today.
- **Promotion gate (all must hold on held-out cases):**
  - recall@6 at least lexical + 5 points;
  - selection accuracy@1 not worse;
  - false-tool rate on conversation not worse;
  - missed-tool rate not worse;
  - no increase in unauthorized actions (hard veto);
  - p95 added latency within the measured budget.
- **Fallback:** if semantic confidence is below the calibrated floor, or the
  two sets disagree with a small margin, the union is used and the turn is
  marked `selection_uncertain` in telemetry.
- **Cases that must be in the set:** hard negatives (device words in
  conversation), ambiguous requests, corrections, device-state questions,
  multi-skill instructions.

| Metric | Lexical | Hybrid (shadow) | Semantic |
| --- | --- | --- | --- |
| recall@6 | measured | measured | measured |
| false-tool on conversation | measured | measured | measured |
| missed-tool on instructions | measured | measured | measured |
| selection accuracy@1 | measured | measured | measured |
| unauthorized actions | 0 required | 0 required | 0 required |

No numbers are claimed here. The table is the evidence the promotion decision
requires.

---

## 5. Execution semantics

- **Idempotency is request-scoped.** Key = (client request id, capability,
  normalized arguments). A retry of the same request reuses the key. A
  legitimate repeated request gets a new request id and is executed again.
  Keys expire after a bounded window; the window is per capability.
- **Verify before retry.** Before retrying a write that may have succeeded
  (timeout, connection reset, ambiguous transport error), read authoritative
  state. If the effect is present, report success; do not write twice.
- **Executable postconditions.** A postcondition is a predicate over
  authoritative state, not prose. A capability without one may not claim
  success; it reports `ok_unverified`.
- **Confirmation.** Destructive, bulk or external-communication actions require
  confirmation from the NoGo channel. No model output can satisfy it.
- **Timeouts.** Per capability, with a declared expected duration. Timeout
  yields `failed` or `partial`, never `ok`.
- **Typed partial outcomes.** `{completed: [...], failed: [...], not_attempted:
  [...], reason}`; the reply states exactly this.
- **Never success from `ok`.** A tool's `ok` is evidence, not proof. The
  executor checks state; the writer only receives the typed outcome.
- **Ordering and dependencies.** Steps declare `depends_on`; independent steps
  may run in parallel; a failed dependency skips dependants as `not_attempted`.
- **Audit.** Every execution records capability, normalized arguments (bounded),
  idempotency key, outcome, observed state and duration.
- **No silent fallback.** A fallback capability is used only if declared, and
  the reply says which path was used.

---

## 6. Memory safety (before any consolidation)

Every record carries: `source`, `source_id`, `timestamp`, `speaker`,
`assertion_status` (`asserted` | `inferred` | `confirmed`), `valid_from`,
`valid_until`, `supersedes`, `superseded_by`. Missing provenance means the
record may not be promoted.

- **Repetition is not truth.** Repeating an inferred fact never upgrades it to
  `confirmed`. Confirmation requires an explicit user statement.
- **Conflicts are preserved.** A conflicting statement supersedes with a link;
  it does not delete the old record. Both remain queryable by time.
- **Consolidation is a proposal.** The offline job writes *candidates* for
  review (merge, promote, prune) with evidence and the rule that produced them.
  It cannot mutate canonical memory, and it cannot act.
- **Deletion is explicit and total.** A delete request removes the episode, the
  semantic record, index rows, derived summaries, cached embeddings where
  derived, and marks backups for expiry. The reply states what was removed and
  what remains (for example, a backup that expires in 30 days).
- **Retention is declared** per store and enforced by a job, not by convention.
- **Reads never write.** Retrieval is pure; reconsolidation is a human-reviewed
  proposal, not an automatic rewrite.

No consolidation work starts until every field and deletion path above exists
and has a test.

---

## 7. Evaluation protocol

- **Frozen baseline.** The revision under test and the baseline are identified
  by commit + configuration hash. The baseline is frozen before the run.
- **Sets.** Development, held-out, adversarial. Held-out and adversarial are
  written after the implementation and frozen before the reported run; any
  prior exposure is disclosed. A set that informs a design decision is retired
  from confirmation use.
- **Unit and n.** The experimental unit is the scenario/turn, never a repeated
  deterministic replay. Per-class sample counts are reported next to every
  rate. The required sample size is computed from the criterion below, not
  chosen by convenience, and is recorded with the freeze.
- **Intervals.** Wilson intervals for descriptive rates; paired bootstrap or
  McNemar on the difference for comparisons; the deciding quantity is the
  difference with its interval.

### 7.1 Pre-registered pass criteria (the second review's item 1)

The previous text stated two different rules: a 2% target with a Wilson upper
bound below 5%, and "false-tool rate below 2% with its interval". They are not
equivalent, and only one is now used.

**Rate criteria (absolute).** A rate passes when the **upper bound of its 95%
Wilson interval** is below the target. This is a claim about the true rate, not
just the observed point estimate, and it is the same rule for both rates:

| Rate | Target | Minimum n (zero errors) |
| --- | --- | --- |
| false-tool on conversation | upper bound < 2% | **189** |
| missed-tool on instructions | upper bound < 5% | **73** |

Permitted error counts at the sample sizes the evaluation will use (a larger n
buys tolerance; it does not raise the target):

| n | false-tool < 2%: permitted errors | missed-tool < 5%: permitted errors |
| --- | --- | --- |
| 73 | — (0 errors → 5.00%, fails) | 0 (4.9994%) |
| 100 | — (0 errors → 3.70%, fails) | 0 (3.70%) |
| 150 | — (0 errors → 2.50%, fails) | 2 (4.73%) |
| 189 | 0 (1.99%) | 3 (4.56%) |
| 250 | 0 (1.51%) | 5 (4.60%) |
| 300 | 1 (1.86%) | 7 (4.74%) |
| 380 | 2 (1.90%) | 10 (4.78%) |
| 500 | 3 (1.75%) | 15 (4.89%) |

A dash means **no error count passes at that n**; it is below the minimum for
that rate. The sample is therefore sized per rate: 189 for false-tool and 73
for missed-tool, not one n used for both.

**Comparison criteria (paired).** An ablation or a new selector is non-inferior
when the **lower bound of the 95% paired interval on the difference** is above
−0.05, using distinct scenarios. Required pairs, for 80% power at a one-sided
97.5% level, by assumed discordance and true difference:

| discordance | true +5 pts | true +10 pts | true +15 pts |
| --- | --- | --- | --- |
| 0.10 | 78 | 34 | 19 |
| 0.15 | 118 | 52 | 29 |
| 0.20 | 157 | 69 | 38 |
| 0.30 | 235 | 104 | 58 |

**Inconclusive is not a pass.** If the interval includes −0.05, the result is
`undecided` and the sample grows in pre-declared blocks; the feature stays in
shadow. An interval that excludes zero but crosses the margin is still
undecided for a promotion decision.

**Non-inferiority is necessary, not sufficient for promotion.** The −0.05
margin answers "is this worse by an unacceptable amount?". Promotion uses the
stricter condition that the interval excludes *any* regression (lower bound at
least 0). A result with a lower bound between −0.05 and 0 is non-inferior but
inconclusive: the feature stays in shadow until the sample is large enough to
show no possible regression. The two thresholds are intentional; they are not
two readings of the same test.

The criteria are implemented as data, not prose, in `nix_core/eval/acceptance.py`
(`required_n`, `permitted_errors`, `passes_rate`, `pairs_needed`) with tests, so
§8.5 and this section can only agree.

- **Identical cases.** Every ablation runs on the same cases, in the same
  order, with the same tool state.
- **The 29-turn suite is a regression suite**, not evidence of general
  reliability. It is reported as pass/fail, never as an accuracy claim.
- **Failure taxonomy.** Separate routing, retrieval, execution, model and
  infrastructure failures; report each with its denominator and the
  undetermined share. Answer quality is scored separately from route
  correctness.
- **Naturalness.** Blind pairwise comparison against the pre-change transcript
  on held-out conversations, plus an automatic count of internal vocabulary per
  reply (target zero).

### 7.2 Evidence completion alignment (2026-10-18)

The evaluation evidence is shared with the **PUCA system model v3 — evidence
completion proposal**, not duplicated:

- **WP1/WP2 (abstention and D2).** The explicit `unknown` outcome is the
  abstention contract in §3.3. WP1's investigation found the pipeline already
  composes the abstention ("I don't have that yet"); the 0.000 result came from
  the D2 scorer not recognising that line and from the "baselines" sharing the
  PUCA pipeline. The scorer and the model-only baseline are fixed with
  failing-then-passing tests, and any answer-path change still lands behind
  `NIX_ANSWER_ABSTENTION_MODE = off | shadow | on`; WP2's fresh 50/25/25 run on
  the production model is the evidence that decides whether it passes.
- **WP3 (D4).** The new confirmation set and the human review of both arms are
  the held-out evidence for the routing phase; route correctness and answer
  quality stay separate.
- **WP4/WP5 (scope).** D6 remains a limited paired gate pass; D5 grammar and
  D10 cache stay unvalidated and are not promoted on this evidence.
- **One gate.** A phase leaves shadow only through
  `acceptance.decide_feature`; the work packages do not add a second promotion
  path.

---

## 8. Migration, rollback and gates

### 8.1 Order (the review's priority)

1. Baseline and safety invariants, including the explicit `unknown`
   abstention path (WP1 of the evidence completion proposal).
2. Execution verification and idempotency.
3. Semantic routing in shadow mode.
4. Typed workspace and unified response boundary.
5. Held-out evaluation.
6. Memory consolidation, salience tuning, habit caching — only after 1–5.

### 8.2 Every phase lands behind a flag

`NIX_<PHASE>_MODE = off | shadow | on`. Shadow computes and logs without
changing the reply. Promotion from shadow to on requires the phase's gate.
The phases include `answer_abstention`, `execution_verification`,
`semantic_routing`, `typed_workspace`, `heldout_evaluation`,
`memory_consolidation`, `habit_caching` and `idle_processing`, registered in
`nix_core/eval/feature_flags.py`.

### 8.3 Rollback and promotion are the same executable rule

Rollback is not a judgement made after the fact; it is the same function that
decides promotion, so a feature cannot be promoted by one reading of the
evidence and rolled back by another. For each flag:

```
decide(feature):
  safety: any candidate unauthorized action  -> rollback_to_off (hard veto)
  vocabulary: any internal term in a held-out reply -> rollback_to_off
  for every affected metric:
      pairs    = distinct scenarios, identical cases, both arms
      if scenarios < 20          -> hold_in_shadow      (never promote on a thin sample)
      diff, [low, high] = 95% paired interval on the difference
      if low < -0.05             -> rollback_to_off      (regression beyond the margin)
      if low <  0                -> hold_in_shadow      (non-inferior is not enough to promote)
  latency: p95 above the pre-registered budget -> hold_in_shadow
  otherwise -> promote
```

Rules that make it executable rather than advisory:

- the comparison is **paired**, on the **difference**, with the interval from
  §7.1 — never two separate rates whose intervals happen to overlap;
- non-inferiority (lower bound above −0.05) is a necessary condition, not the
  promotion rule: promotion requires the lower bound to be at least 0, so a
  possible regression cannot be promoted;
- the margin, level and minimum sample are constants in
  `nix_core/eval/acceptance.py`, not values chosen per run;
- `undecided` and `hold_in_shadow` are the same outcome: the feature stays in
  shadow and the sample grows in pre-declared blocks;
- a rollback is a configuration change (`MODE=off`) and is itself tested;
- there is no path from `undecided` to `on` that does not pass this function.
Rollback is a configuration change, not a revert, and is tested.

### 8.4 Deferred, with hard constraints

- **Habit caching** is deferred until execution verification and idempotency
  show measurable gains. When it lands: feature flag, explicit invalidation key
  (state version + capability version), and a postcondition failure always
  invalidates. A habit may never bypass NoGo or confirmation.
- **Idle-time processing** is deferred. When it lands: it may propose, never
  mutate canonical memory, never send a message, never trigger an action, and
  runs only with an explicit enable flag. Its output is a review queue.

### 8.5 Acceptance gates (the same rule as §7.1, by reference)

- unauthorized actions = 0 (hard veto, any sample);
- false-tool rate: **95% Wilson upper bound < 2%** — minimum n 189, 0 errors
  permitted there, 3 permitted at n = 500 (§7.1);
- missed-tool rate: **95% Wilson upper bound < 5%** — minimum n 73, 0 errors
  permitted there, 15 permitted at n = 500 (§7.1);
- every ablation or selector promotion: paired interval, minimum 20 distinct
  scenarios per flow; −5 points is the regression bound, and promotion
  additionally requires a lower bound of at least 0, so non-inferior but
  inconclusive results stay in shadow (§8.3);
- multi-action goal success ≥ 85% on the sandbox, reported with its interval;
- duplicate writes < 1% (descriptive, with n);
- partial-completion accuracy ≥ 95% (descriptive, with n);
- zero internal-vocabulary replies in the naturalness set;
- p95 latency for single-tool turns within the pre-registered budget — measured
  before anything is claimed about it.

Every number above is a criterion with a sample attached, not a prediction.

---

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Effort leaks back into routing | invariant test: intent decisions are independent of salience; review item 1 is a test, not a comment |
| Workspace becomes an orchestrator | contribution protocol only; specialists own their internals; no specialist calls the writer |
| Shadow retrieval changes behaviour | shadow mode cannot alter the candidate set; promotion is a separate reviewed change |
| Memory safety regresses under consolidation | consolidation writes proposals only; deletion test covers every store |
| Statistics flatter the change | scenario is the unit; repeated replays are collapsed; baseline frozen; intervals on the difference |
| Humanized copy hides failure | copy never claims success; typed outcomes and telemetry keep the internal class |
| Model or prompt drift | hard boundary: no prompt, persona, weights or decoding changes; a model change is a separate experiment |

---

## 10. References

Anderson (2010) *Behav. Brain Sci.* 33:245 · Aston-Jones & Cohen (2005)
*Annu. Rev. Neurosci.* 28:403 · Barrett (2017) *SCAN* 12:1 · Bastos et al.
(2012) *Neuron* 76:695 · Bullmore & Sporns (2009) *Nat. Rev. Neurosci.* 10:186 ·
Craig (2009) *Nat. Rev. Neurosci.* 10:59 · Daw, Niv & Dayan (2005) *Nat.
Neurosci.* 8:1704 · Diekelmann & Born (2010) *Nat. Rev. Neurosci.* 11:114 ·
Duncan (2010) *TiCS* 14:172 · Fedorenko, Ivanova & Regev (2024) *Nat. Rev.
Neurosci.* 25:289 · Frank (2004) *Science* 306:1940 · Friston (2010) *Nat. Rev.
Neurosci.* 11:127 · Frith & Frith (2006) *Neuron* 50:531 · Kumaran, Hassabis &
McClelland (2016) *TiCS* 20:512 · Kurby & Zacks (2008) *TiCS* 12:72 · Mashour et
al. (2020) *Neuron* 105:776 · McClelland, McNaughton & O'Reilly (1995)
*Psych. Rev.* 102:419 · Menon & Uddin (2010) *Brain Struct. Funct.* 214:655 ·
Nader, Schafe & LeDoux (2000) *Nature* 406:722 · Raichle et al. (2001) *PNAS*
98:676 · Rao & Ballard (1999) *Nat. Neurosci.* 2:79 · Saxe & Kanwisher (2003)
*NeuroImage* 19:1835 · Schultz (1997) *Science* 275:1593 · Seeley et al. (2007)
*J. Neurosci.* 27:2349 · Sherman & Guillery (2002) *Phil. Trans. R. Soc. B*
357:1695 · Sporns & Zwi (2004) *Neuroinformatics* 2:145 · Teyler & DiScenna
(1986) *Behav. Neurosci.* 100:147 · Tse et al. (2007) *Science* 316:76 · Wilson &
McNaughton (1994) *Science* 265:676 · Wolpert & Miall (1996) *Neural Networks*
9:1265 · Zacks et al. (2007) *Psych. Bull.* 133:273.
