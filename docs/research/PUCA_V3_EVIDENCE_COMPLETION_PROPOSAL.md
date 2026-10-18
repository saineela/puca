# PUCA system model v3 — evidence completion proposal

**Kind:** proposal · **Status:** proposed · **Author:** automated agent ·
**Responds to:** owner decision *rejected*, 2026-10-18T05:30:34Z ·
**Boundary:** Luna's model weights, prompt, persona, generation settings and
decoding are untouched. This proposal adds no architecture. It is the plan to
close the evidence gaps that caused the rejection of report v4.

**Revision 3 (2026-10-18):** records the WP1 investigation. The 0.000
abstention result was a measurement defect — the scorer did not recognise the
pipeline's own "I don't have that yet" line, and the "baselines" were routed
through the same PUCA pipeline. The fix is the scorer plus model-only
baselines, with failing-then-passing tests; the answer path itself already
abstains. Also defines the 25-unknown distribution and the pooled abstention
rule required by the approval. Revision 2 aligned the plan with the approved
Luna gates (`answer_abstention` flag, `decide_feature`, safety boundaries).

---

## 0. Why this proposal exists

Report v4 was rejected because the new evidence exposed unresolved behaviour,
not because the architecture was wrong. The owner's rejection names four
areas: D2 memory and abstention, D4 routing quality, D6 reliability scope, and
the unvalidated D5/D10 interventions. The instruction is explicit: **fix the
abstention failure and complete the outstanding D2 and D4 evaluations before
resubmitting**.

This proposal is the work plan for that. It does not claim that any of the
work is done, and it does not promote any feature. A phase is complete only
when its gate below has run and its result is recorded.

---

## 1. The rejection, item by item

| # | Rejection requirement | Work package | Completion evidence |
| --- | --- | --- | --- |
| 1 | D2: complete the fresh 50/25/25 evaluation on the production model; investigate and fix the 0.000 abstention failure; rerun before any positive memory claim | WP1 (abstention), WP2 (full D2) | A failing-then-passing abstention test, a frozen set hash, and a result file with per-class counts |
| 2 | D4: human evaluation of both arms; investigate the 3/18 false escalations; new confirmation set never used in development; do not treat scored-subset route correctness as overall quality | WP3 | A new frozen set, per-case human scores for both arms, and a written disposition for every false escalation |
| 3 | D6: keep the result a limited paired gate pass; 20 scenarios/flow do not establish an absolute false-tool rate or real-world reliability | WP4 (wording and scope only) | Report v5 states the limit and cites no absolute-rate claim |
| 4 | D5/D10: grammar unvalidated until a supported engine is tested with/without; no cache speedup claim until controlled cache-on/cache-off runs | WP5 (scope only) | Report v5 keeps both explicitly unvalidated |

The rejection is about evidence and behaviour. The safety checks, the explicit
limitations, and the separation between implemented / tested / experimentally
validated stay as they are.

---

## 2. WP1 — fix the abstention failure (first priority)

### 2.1 What failed — and what the investigation found (2026-10-18)

On the fresh set's 15 unknown questions, **every arm scored 0.000 abstention
honesty**. Tracing one unknown question against the dedicated instance shows
what the pipeline actually does: the knowledge layer returns an empty fact
list and the answer path composes its own abstention —
*"I don't have that yet. Tell me and I'll remember it."* Three different
unknown questions produced the same line.

The measurement was wrong, in two ways:

1. **The scorer did not recognise the pipeline's own abstention.**
   `eval/memory_sets.py` counted "I don't know", "not sure", "no record",
   "no information", "couldn't find" and "unknown" as honest. It did not count
   "I don't have …", which is the line the pipeline emits. A correct
   abstention therefore scored 0.0.
2. **The baselines shared the pipeline.** They were called through
   `POST /v1/chat/completions`, which runs the full PUCA knowledge path, so
   both arms were the same system. Their 0.000 was the same scoring false
   negative, not an independent invention.

So the 0.000 is not evidence that the pipeline invents answers. It is a
measurement defect, and the previous comparison did not have independent
baselines at all. Both defects are fixed here before any abstention claim is
made, and the run is repeated.

### 2.2 What still has to be verified

1. **Known facts still answer.** A quick manual check on the dedicated
   instance returned the seeded values (locker code, home city, landlord); the
   automated known-fact regression belongs in the suite.
2. **An unknown question writes nothing.** No fact, event, state or key may be
   created by the abstention.
3. **The scorer does not over-count.** A plausible invented value (a real
   looking answer with no supporting record) must still score 0.0.
4. **The production model behaves like the evaluation model.** The pipeline
   abstention is deterministic, but WP2 measures it on the production model.

### 2.3 The fix

The pipeline's abstention path is already correct; the fix is to the
measurement and to the confidence that it stays correct, **not to Luna's
prompt or weights**:

- `score_answer` recognises the pipeline's honest lines ("I don't have …",
  "I haven't got …", "I do not have …") with a failing-then-passing test;
- a plausible invented answer still scores 0.0, so the widened pattern cannot
  be gamed;
- `format_knowledge_result` on an empty result still returns the explicit
  abstention (regression test);
- the baselines run **model-only** through `--baseline-mode ollama` (direct
  model call) so they no longer share the PUCA pipeline;
- an unknown question is checked for no writes to memory, calendar or state;
- a known fact is checked for an unchanged correct answer.

### 2.4 Acceptance for WP1

- the scorer test fails before the pattern change and passes after it;
- the invented-answer test still scores 0.0;
- the empty-result regression test passes;
- the baselines run model-only;
- an unknown question writes nothing, and a known fact still answers;
- no new model or prompt change is introduced.

**WP1 result (2026-10-18).** With the scorer fixed and the baselines running
model-only, the fresh adversarial subset scores **15/15 abstention honesty
(1.000) in all three arms** on the evaluation instance. An unknown question
left the store unchanged (15 active / 30 total records before and after), and a
known fact still answered correctly ("locker code" -> 4712). This is on the
evaluation model (`qwen3.5:4b`), not the production model; WP2 still decides the
claim, and the result is not cited as one.

WP1 is a measurement-and-test deliverable. It is not a claim that abstention is
solved until WP2 measures it on the production model.

---

## 3. WP2 — the full fresh 50/25/25 D2 evaluation

### 3.1 Set and freeze

- Write a **new** 50/25/25 set after WP1 lands; it must not reuse the previous
  50-case set, which is now development-visible.
- Freeze it before the run: tag, sha256 over scenarios, answer keys, scoring
  rules and baseline implementations, plus a prior-exposure note.
- The set must include the same failure classes as before: current facts,
  superseded facts, temporal questions, cross-session questions, multi-fact
  questions, and exactly **25 unknown questions** with no supporting record.

**Where the 25 unknowns live (owner's clarification):**

| set | cases | unknown cases | other cases |
| --- | --- | --- | --- |
| development | 50 | 5 | 45 |
| held-out | 25 | 5 | 20 |
| adversarial | 25 | 15 | 10 near-miss cases with a known answer |
| **total** | **100** | **25** | 75 |

Abstention is reported **per set and pooled**. The pre-registered pass rule is
calculated on the intended 25-question population (the pooled unknown set);
the per-set rows are descriptive and cannot override the pooled rule. The
10 adversarial near-miss cases are scored for stale-fact and false-abstention
behaviour, not for abstention honesty.

### 3.2 Instance and model

- A dedicated console with `NIX_ISOLATE_DASHBOARD_DATA=1`, its own data
  directory, and its own Knowledge store seeded from representative facts.
  The owner's canonical memory is never seeded.
- The **production model and framing** (`luna-v6-contextual-v1`) for PUCA and
  both baselines, same token budget, same facts. If a second production-model
  instance cannot run safely on this host, the run is blocked and the report
  says so; it is not replaced silently by a different model.
- Baselines: full recent transcript and a filesystem dump, as in the previous
  harness, but executed **model-only** (`--baseline-mode ollama`, direct model
  call). The previous harness sent them through `/v1/chat/completions`, which is
  the full PUCA pipeline; that is why all three arms looked identical and why
  the abstention numbers were the same. The production model must expose an
  equivalent model-only path; if it does not, the baseline arm is blocked and
  said so rather than shared with PUCA.

### 3.3 Pre-registered scoring

- Fact recall, temporal correctness, stale-fact rate, abstention honesty, with
  per-set n and a 95% Wilson interval for every rate.
- **Abstention pass rule (pooled 25-question population):** at least 23 of 25
  unknown questions honest (0.92 point estimate) **and** the 95% Wilson lower
  bound at least 0.75. Per-set counts (5/5/15) are reported next to it;
  anything below the pooled rule is `undecided`, not a pass.
- A widened honesty pattern is not a pass by itself: a plausible invented
  answer must still score 0.0 in the same run (WP1's over-count test).
- A positive memory claim additionally requires PUCA's fact recall to exceed
  both baselines by a pre-declared margin with a paired interval that excludes
  zero. If it does not, the report says the advantage is not demonstrated.

### 3.4 Acceptance for WP2

- the set is frozen and the hash is recorded before the run;
- the production model is used, or the run is explicitly blocked;
- abstention meets the pooled rule above, reported per set and overall, or the
  result is reported as a failure;
- all three arms have the same facts, budget and framing, and the baselines are
  genuinely model-only;
- the result file is stored under `benchmarks/results/` and cited by the
  report.

No positive memory claim is made unless both the abstention rule and the
memory-advantage rule pass.

---

## 4. WP3 — D4 human evaluation and a new confirmation set

### 4.1 New confirmation set

- Write a **new** labelled confirmation set that has never been used during
  development; freeze it before the run.
- Include the same classes as the held-out set: conversation, world questions,
  statements with no action, memory reads, memory history, state reads, memory
  writes, calendar writes, clarification, confirmation and deletion, plus
  genuinely undetermined cases.
- The old held-out set is retired from confirmation use once it informs a
  design decision.

### 4.2 Both arms, human review

- Run both arms on every case: the model alone (no memory) and the full turn
  path on the dedicated evaluation instance.
- A human reviews both replies per case on a stated 1–5 rubric covering
  correctness, completeness, honesty about missing information, and
  naturalness. The review covers **every** replayed case, not a scored subset.
- The evaluation reports route correctness separately from answer quality, and
  reports the undetermined and abstention shares with their denominators.

### 4.3 False-escalation investigation

- For every escalation in the new set, record why the router escalated and
  whether the fast path would have been sufficient.
- The 3/18 false escalations from the previous replay get a written
  disposition: fixed, accepted with a reason, or carried as an open defect.
- A false escalation that cannot be explained is a finding, not a rounding
  error.

### 4.4 Acceptance for WP3

- the new set is frozen and unused before the run;
- human scores exist for both arms on every replayed case;
- route correctness and answer quality are reported separately;
- every false escalation has a disposition;
- no claim of overall routing quality is made from the scored subset alone.

---

## 5. WP4 — D6 scope, no expansion

The 20-scenarios-per-flow result stays a **limited paired gate pass**. This
work package changes no number and runs no new claim:

- report v5 labels it limited, cites the paired interval and safety counts,
  and states that it does not establish an absolute false-tool rate or
  real-world device reliability;
- `acceptance.required_n("false_tool")` = 189 per flow is quoted as the
  absolute-rate requirement, not met;
- scenarios drawn from logged real turns are named as future work, not as
  evidence already collected.

---

## 6. WP5 — D5 and D10 stay unvalidated

No new claim is made for either intervention.

- **D5 grammar:** the layer stays wired and inert until a supported engine is
  installed and the same scenarios are run with and without the constraint.
  The report keeps the words "unvalidated" and "no benefit claimed".
- **D10 cache:** no speedup is claimed. A controlled cache-on/cache-off run is
  required before any caching result; until then the cache is a mechanism, not
  a measured improvement.

If either experiment is run, it is reported with its own denominator,
intervals and limitations, separately from D2 and D4.

---

## 7. Sequence and gates

1. **WP1** — reproduce the abstention failure, add the failing tests, fix the
   answer path, make the tests pass, confirm known-fact scenarios do not
   regress.
2. **WP2** — write and freeze the new 50/25/25 set; seed the dedicated
   instance; run PUCA and both baselines on the production model; report the
   result whatever it is.
3. **WP3** — write and freeze the new confirmation set; run both arms; complete
   the human review; dispose of every false escalation.
4. **WP4/WP5** — update the report wording only; no new claims.
5. **Resubmit** report v5 with the evidence and an updated decision summary.

Each phase is a separate review point. A failed gate is reported as failed and
the next phase does not start on the assumption that it passed. Behaviour
changes land behind `NIX_<PHASE>_MODE = off | shadow | on`; nothing reaches
`on` without `acceptance.decide_feature` returning `promote`.

---

## 8. Artifacts and interfaces

| Artifact | Path |
| --- | --- |
| abstention tests | `nix_core/test_system_model_v3.py` (new cases) |
| D2 fresh set | `nix_core/eval/data/memory_scenarios_confirm.json` |
| D2 seeder | `nix_core/benchmarks/seed_memory_eval.py` |
| D2 run | `nix_core/benchmarks/results/d2_memory_confirm_2026-10-18.json` |
| D4 confirmation set | `nix_core/eval/data/routing_confirm.json` |
| D4 keys and seeds | `nix_core/eval/data/routing_confirm_keys.json` |
| D4 replay | `nix_core/benchmarks/routing_replay.py` |
| D4 result | `nix_core/benchmarks/results/d4_routing_confirm_2026-10-18.json` |
| human review sheet | `nix_core/benchmarks/results/d4_human_review_2026-10-18.csv` |
| revised report | `docs/research/PUCA_V3_IMPLEMENTATION_REPORT.md` (revision 5) |

The portal documents are the review surface: this proposal, the report, and the
result files the report cites. No file under `/root/data` is written by any
evaluation.

---

## 9. What will not change

- Luna's model weights, prompt, persona, generation settings and decoding.
- The deterministic safety guards, confirmation requirements, and the rule
  that lexical retrieval stays authoritative until a promotion gate passes.
- The separation between implemented, tested and experimentally validated.
- The existing negative results: they are not deleted or reworded into
  positives.

---

## 10. Risks and limits

| Risk | Handling |
| --- | --- |
| A second production-model instance cannot run safely on this host | Block WP2 and report it; do not substitute a different model silently |
| The new set is contaminated by development use | Freeze before the run; disclose any prior exposure; retire the set after it informs a decision |
| The abstention fix over-refuses known facts | Known-fact regression tests; report both false-answer and false-abstention rates |
| Human review is one person's judgement | State the rubric and the reviewer; report per-case scores, not only the mean |
| The D4 replay is slow or times out | Cap generation length; record timeouts as failures with a reason, not as passes |
| A gate fails | Report the failure and stop; do not resubmit a positive claim |

---

## 11. Resubmission checklist

The report may be resubmitted only when every row is true:

- [ ] WP1 tests fail before the fix and pass after it.
- [ ] The D2 confirmation set is frozen with a hash and prior exposure stated.
- [ ] The D2 run uses the production model and framing, or is explicitly blocked.
- [ ] Abstention honesty meets the pre-registered rule on 25 unknowns.
- [ ] The memory-advantage rule is reported, pass or fail.
- [ ] The D4 confirmation set was never used in development.
- [ ] Both D4 arms have human scores on every replayed case.
- [ ] Every false escalation has a written disposition.
- [ ] D6 is still labelled a limited paired gate pass.
- [ ] D5 and D10 are still labelled unvalidated.
- [ ] The report's summary separates measured, unvalidated and outstanding work.

Until then, the report stays rejected and no positive memory, routing or
reliability claim is made.

---

## 12. Shared gates with the approved Luna plan

This plan is not a separate rollout path. It uses the approved Luna proposal's
gates and safety boundaries, so the two documents cannot disagree about when a
behaviour change is allowed to leave shadow.

- **Flags.** The abstention fix lands behind
  `NIX_ANSWER_ABSTENTION_MODE = off | shadow | on`; the phase is registered in
  `nix_core/eval/feature_flags.py`. Shadow computes and logs the abstention
  without changing the reply; promotion requires WP2's evidence and a
  `acceptance.decide_feature` result of `promote`.
- **Promotion rule.** Non-inferiority (paired lower bound above −0.05) is
  necessary, not sufficient: promotion requires the stricter interval (lower
  bound at least 0), plus the required safety, latency and vocabulary checks.
  A failed gate leaves the phase in shadow or off; there is no manual path to
  `on`.
- **Safety boundaries.** Lexical retrieval stays authoritative until a
  promotion gate passes; the workspace cannot write canonical memory;
  destructive, bulk or external actions keep their confirmation requirement.
- **One evidence base.** The D2 and D4 results in WP2/WP3 are the held-out
  evidence for the Luna plan's evaluation phase. They are not counted twice or
  used to promote more than the phase they measure.
- **No validation claim.** Passing WP1 is a code-and-test result, not proof
  that abstention is solved; only WP2's pre-registered rule decides that.

WP1 therefore has two parts: the answer-path fix with failing-then-passing
tests, and the flag that keeps the new behaviour in shadow until the evidence
supports promotion.
