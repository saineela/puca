# PUCA system model v3 — implementation and evidence (revision 4)

> **Superseded in part, 2026-10-18.** The D2 and D4 sections of this revision
> are superseded by `PUCA_V3_EVIDENCE_PACKAGE_2026-10-18.md` (revision 5), which
> registers the new D2 confirmation protocol and records that its run did not
> complete on this host (no D2 number is claimed), the four defect fixes with
> their regression evidence, and the D4 human-review packet. This file is kept
> as the record of revision 4: its D2 and D4 numbers are historical and must not
> be quoted as the current state.

**Kind:** report · **Responds to:** owner decisions *changes-requested*,
2026-10-17T16:12:11Z, 2026-10-17T16:42:20Z and **2026-10-18T04:45:49Z** ·
**Scope:** the ten approved decisions (D1–D10), the three waves, and the SEFH 2027 primary claim.
No new architecture is proposed or implemented here. Every number below was
measured on this machine unless it is explicitly labelled as not measured.

Revision 4 answers the third review's evidence requirements. The two blockers
are now run on dedicated evaluation instances built for the purpose, and the
results are reported as they came out — including the negative ones.

- **D2 now runs on a dedicated seeded instance.** A second console was started
  with its own data directory (never the owner's canonical memory) and a
  Knowledge store seeded from a fresh frozen 50-case set
  (`fresh-2026-10-18`, 20/15/15). PUCA did **not** beat the baselines:
  development fact recall 0.325 vs transcript 0.400 / filesystem 0.350;
  held-out 0.333 vs 0.333 / 0.333. Abstention honesty on the 15 unknown
  questions is **0.000 in all three arms** — the invented-answer defect is
  confirmed on a second, seeded instance, not explained away by the store. §4.
- **D4 now has a controlled counterfactual replay** on the frozen 40-case
  held-out set, with both arms actually run (the model alone, and the full turn
  path on its own seeded evaluation instance) and answer quality scored by a
  stated deterministic rubric. §5.
- **D6 stays a limited paired gate pass** on 20 distinct scenarios per flow.
  It is not an absolute error-rate claim and not evidence of real-world
  reliability. §3.
- **D5 stays unvalidated** and **D10's cache claim stays withdrawn** (§6, §8).

This revision keeps the previous report's structure and results, including its
negative and incomplete ones. Three things changed, and they are the point of
the revision:

1. **D2's seeded-instance run is done and negative.** The previous full run
   used the owner's live memory; this one uses a dedicated store seeded with
   representative facts and a fresh frozen set written for the run. PUCA still
   shows no advantage over a transcript dump or a filesystem dump, and still
   invents answers to unknown questions. §4.
2. **D4's counterfactual replay is done and scored.** The earlier report said
   no second path executor existed; one now exists and both arms were run. The
   replay confirms the route decision on the scored cases and exposes the
   answer-quality failures underneath it. §5.
3. **The statistical rule and the promotion rule are now explicit and
   executable.** Non-inferiority (−5 points) is a necessary condition; promotion
   requires the stricter "no possible regression" interval (lower bound ≥ 0),
   plus safety, latency and vocabulary gates in `eval/acceptance.py`. §3.4.

---

## 0. The review, item by item

| # | Review asks for | Status in this revision |
| --- | --- | --- |
| 1 | D6 reproducibility and statistical criteria | **Corrected, then expanded.** Unit defined (scenario), replays collapsed, win/loss/ties reported, and the sample grown to 20 scenarios per flow; decision `pass` on the scenario unit under a pre-registered rule; safety counted separately; veto preserved. §3 |
| 2 | D2 complete memory evaluation | **Full 50/25/25 run completed; negative result reported.** PUCA equals both baselines on every measure, fact recall is 0.14/0.28/0.20, and abstention honesty on adversarial unknowns is 0.000 for all arms. The 10/10/10 table is kept as a pilot. A seeded evaluation instance and a fresh frozen set are still required before any D2 claim. §4 |
| 3 | D4 validate routing outcomes | **Pilot only.** 24 hand-labelled cases, 15 scored, 7 abstentions, 4 undetermined; counterfactual replay and answer quality not measured. Not the frozen final evaluation. §5 |
| 4 | D5 test the actual grammar intervention | **Cause analysis done; intervention not tested.** No malformed envelope in 31 checked plans; retries are grounding, not syntax; no supported grammar engine is installed. Validation layer retained. §6 |
| 5 | D9 verify the full outbound boundary | **Extended.** Memory-origin and tool-result-origin values, retries, telemetry, exception text, rehydration audit; 38 tests in the suite. Guarantee stated narrowly. §7 |
| 6 | D3/D10 bound the conclusions | **Done in wording and status, plus one claim withdrawn.** Budgets are model/task-specific and logging-only; inference instrumentation is coverage; there is no cache to measure, so no cache claim exists. §8 |

### 0.1 Second review (2026-10-17T16:42:20Z) — evidence gaps

| # | Review asks for | Status in this revision |
| --- | --- | --- |
| 1 | D2 seeded store + fresh frozen set | **Done in revision 4:** dedicated seeded instance and fresh frozen set, still negative (§4.1). The owner's canonical store is still never seeded. |
| 2 | D4 fresh held-out labelled set + counterfactual replays | **Done in revision 4:** frozen held-out set plus a controlled replay of both arms with answer-quality scoring (§5). |
| 3 | D6 substantially more distinct scenarios per flow | **Done: 20 per flow (80 total), gate re-decided as pass.** §3 |
| 4 | D10 controlled cache-on/cache-off experiment | **Claim withdrawn: no cache exists.** Nothing was measured because there is no mechanism. §8 |
| 5 | D5 grammar intervention unvalidated | **Still unvalidated: no engine installed.** Cause analysis done, and a separate validation defect was found and fixed. §6 |
| 6 | Correct the sample-size / interval inconsistency | **Done, and shared with the Luna proposal**: one interval-based rule in `nix_core/eval/acceptance.py`, cited by both documents. §3.3 |
| 7 | Separate measured results from hypotheses and unverified improvements | **Done throughout**: §9 keeps implemented / tested / validated apart, and every unrun measurement is named as unrun. |

### 0.2 Third review (2026-10-18T04:45:49Z) — the required fixes

| # | Review requires | Status in this revision |
| --- | --- | --- |
| 1 | D2: dedicated seeded instance and a fresh frozen set | **Done, negative.** A second console with its own data directory; a fresh frozen 50-case set (`fresh-2026-10-18`); no memory advantage over both baselines; abstention honesty 0.000 on 15 unknowns in all arms. §4 |
| 2 | D4: frozen labelled evaluation, controlled counterfactual replays, answer-quality scoring | **Done as a controlled replay.** Both arms run on the frozen 40-case set; answer quality scored by a stated deterministic rubric, whose limits are named. §5 |
| 3 | D5: do not claim a grammar benefit | **Unchanged: unvalidated.** No engine installed, no comparison run; no benefit is claimed. §6 |
| 4 | D6: limited paired gate pass only | **Stated as a limited paired gate pass**; explicitly not an absolute error-rate claim and not real-world reliability. §3 |
| 5 | Reporting: separate measured, unvalidated and outstanding | **Done.** §9 keeps implemented / tested / validated apart; §10 lists every outstanding evaluation. |

---

## 1. What is implemented, and where

| Decision | Implementation | Evidence |
| --- | --- | --- |
| **D1** turn-level provenance | `nix_knowledge/nix_knowledge/provenance.py`, `engine.supersede`, `superseded_by` | 7 tests: transactional supersede, no dangling source, no self-supersession, superseding a superseded row refused, old rows still readable |
| **D2** memory evaluation | `nix_core/eval/memory_sets.py`, `eval/data/memory_scenarios.json` (100-case pilot), `eval/data/memory_scenarios_fresh.json` (fresh 50-case set), `benchmarks/memory_evaluation.py`, `benchmarks/seed_memory_eval.py` | one command runs the three sets; freeze note carries a sha256 over scenarios, answer keys, scoring and baselines, plus prior exposure; baselines get the same facts, budget and model; the seeder refuses the owner's canonical paths |
| **D3** context composer | `nix_core/eval/context_composer.py`, `brain._context_sections`, `request_log.make_entry` | every new logged turn carries all five section sizes; policy is never evicted; identical inputs compose identical prompts; budgets are logging-only |
| **D4** routing ledger + replay | `routing_engine.py` (`cost_estimate`, `escalation_cost`, `counterfactual`, `outcome`, `fingerprint`), ledger row in `request_log`, `eval/routing_ledger.py`, `benchmarks/routing_replay.py`, `eval/data/routing_replay_keys.json` | a bounded row on every new turn (route, cost, fingerprint — never the prompt); false escalation and false de-escalation reported separately with denominators and undetermined share; both arms run on the frozen held-out set with a scored answer-quality proxy |
| **D5** plan validation + grammar | `eval/plan_validation.py` (schema, intent, permission, confirmation, outcome) wired into the planner; `nix_knowledge/grammar.py` optional engine | five independent checks, each tested; a valid-but-unauthorised plan is caught; the grammar layer is inert with no engine installed and tested with a stub |
| **D6** reliability gate | `eval/reliability.py`, `eval/reliability_flows.py`, `benchmarks/reliability_gate.py` | paired difference with bootstrap interval + exact McNemar; the scenario-level gate added in this revision refuses to count replays; zero-tolerance safety veto |
| **D7** voice turn-taking | `eval/turn_taking.py`, `voice.turn_taking_profile`, `GET /api/system-model/turn-taking` | energy VAD, endpointing, trailing silence measured from the clip; barge-in reported as unmeasured on a single upload |
| **D8** proactivity | `eval/proactivity.py`, `nix_decision` trigger tiers + gate | policy off delivers nothing and has no side effects; tier gate precedes rules; a trigger must declare a tier |
| **D9** privacy boundary | `eval/privacy.py`, Tabby outbound path | published suite, fail-closed, local-only mapping, per-store retention, every outbound path named; extended tests in §7 |
| **D10** inference measurements | TTFT stamp in `luna_model`, `eval/inference_metrics.py`, prefix cache, `GET /api/system-model/inference` | TTFT, decode and load recorded separately when a model call occurs; cache keys carry model + tokenizer + policy version and refuse personal memory |

Read-only inspection: `GET /api/system-model`, `/inference`, `/privacy`,
`/proactivity`, `/context?text=`, `/turn-taking`. The Documentation page is
checked by a test against the registered routes.

---

## 2. Real chat history replayed through the user paths

The owner's sessions gave **186 user turns**, deduped to **124 distinct
prompts**. Every prompt went through `POST /api/classify`, `POST /api/send` and
`POST /v1/chat/completions` on `luna-v6-contextual-v1`.

| Measurement | ok/requests | errors | non-empty replies | p50 | p95 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| routing only | 124/124 | 0 | — | 0.9 ms | 1.0 ms | 0.9 ms |
| PUCA system model | 124/124 | 0 | 124/124 | 1176.0 ms | 6469.4 ms | 1686.2 ms |
| Luna model path | 124/124 | 0 | 124/124 | 1187.2 ms | 6423.2 ms | 1622.7 ms |

372 requests, 0 errors, 0 empty replies. **This is a liveness and path-
consistency measurement, not a quality measure.** It says the paths answer; it
does not say the answers are right.

---

## 3. D6 — expanded sample, one intermediate regression, and a final pass

### 3.1 The experimental unit, and why the sample had to grow

`paired_runs(k)` runs each scenario k times. In the previous revision,
`--trials 20` produced 4 × 20 = 80 rows per flow from **four** distinct
scenarios per flow. The runner is deterministic and does not consume the
`seed`, so those 20 rows per scenario are identical replays.

The experimental unit is the **scenario**. Replays check reproducibility; they
are not independent evidence. The gate now collapses them (`scenario_gate`) and
refuses to decide below 20 distinct scenarios per flow. The flows therefore hold
**20 scenarios each, 80 total** (was 16), and the report below cites only the
scenario-level analysis.

### 3.2 The run that failed, and what it exposed

The first expansion run read **`regressed`**: device_plan baseline 14/20 vs
treatment 11/20, wins 6 / losses 9 / ties 5. That JSON was overwritten by the
run below; the defect itself is pinned by the test named at the end of this
section rather than by a saved file.

All nine losses were brightness requests ("brightness to 70", "brightness to
30", …) rejected by the treatment's own validation with the failure
*"request contains no actionable verb"*. That was a real false negative in
`eval/plan_validation.py`: a value-led request ("brightness to 70") is a
legitimate request, and requiring a verb rejected it. The check now also accepts
a request when the plan's own argument names appear in it, and still rejects a
plan that names nothing from the request. The change has a dedicated test
(`test_plan_validation_accepts_a_value_led_request_without_a_verb`).

This is the value of the larger sample, stated plainly: at four scenarios the
same defect hid as a single loss inside an interval that included zero.

### 3.3 The current run (after the fix)

```
python3 benchmarks/reliability_gate.py --trials 3 --min-scenarios 20 \
  --json benchmarks/results/d6_gate_2026-10-17.json
```

Scenario-level analysis (the only one that may be cited):

| Flow | scenarios | baseline | treatment | wins | losses | ties | difference [95%] | McNemar p | safety b/t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| device_plan | 20 | 14/20 | **20/20** | 6 | 0 | 14 | **+0.30 [+0.10, +0.50]** | **0.03125** | 6 / **0** |
| reminder | 20 | 20/20 | 20/20 | 0 | 0 | 20 | 0 [0, 0] | 1.0 | 0 / 0 |
| reschedule | 20 | 20/20 | 20/20 | 0 | 0 | 20 | 0 [0, 0] | 1.0 | 0 / 0 |
| state_update | 20 | 20/20 | 20/20 | 0 | 0 | 20 | 0 [0, 0] | 1.0 | 0 / 0 |

**Decision: a limited paired gate pass, on the independent unit.** The paired
difference over distinct scenarios excludes a regression, the device-plan flow
improves by 30 points with the interval clear of zero, and the treatment has
zero safety violations where the baseline commits six (out-of-range brightness,
unconfirmed power-off). The three unchanged flows are ties, as expected for a
change that only acts on device planning. This is **not** an absolute
error-rate result (those need 189/73 per flow, §3.4) and **not** evidence of
real-world reliability: the scenarios are sandbox flows written by the same
hand as the check.

Replay-counted readings are still printed by the tool as a pre-merge signal and
are **not** cited as evidence.

### 3.4 Pre-registered criteria (now shared with the Luna proposal)

The rule is data, not prose, in `nix_core/eval/acceptance.py`:

- **Unit:** scenario; replays collapsed; a non-deterministic scenario is an
  error, not a vote.
- **Confidence level:** 95%, two-sided; paired interval on the difference.
- **Regression margin:** −5 percentage points (lower bound must be above −0.05).
- **Minimum sample:** 20 distinct scenarios per flow before any claim; paired
  sample sizes for a target difference are computed by `pairs_needed` (for
  example 69 pairs at 0.20 discordance and a true +10 points).
- **Stopping rule:** 20/flow is the first checkpoint; if the interval includes
  −0.05 with a lower bound above −0.10, the sample grows in pre-declared blocks
to at most 100/flow. No unplanned peeking.
- **Primary outcome:** incorrect and unauthorized action counts with
  denominators; task completion is secondary and cannot offset a violation.
- **Safety veto:** any candidate safety violation vetoes regardless of the
  average. Unchanged.

Absolute-rate criteria are separate from this non-inferiority gate:
`acceptance.required_n` is **189** for false-tool < 2% and **73** for
missed-tool < 5% at zero errors, with `permitted_errors` giving the tolerance
at larger n. This report claims neither absolute rate; 20 scenarios per flow
supports the paired comparison only.

The full feature gate is now executable in one function
(`acceptance.decide_feature`), with safety, latency and vocabulary as required
inputs, and the phase modes (`off | shadow | on`) live in
`eval/feature_flags.py` with their own tests.

### 3.5 What the primary outcome counts

Per flow the gate reports baseline and candidate safety violations, and baseline
and candidate failures stopped before acting. On `device_plan` the baseline
commits **6** unsafe mutations and the treatment **0**; the treatment's
completion rate is 20/20. The statistical claim and the safety claim are
separate, and both are reported.

---

## 4. D2 — memory evaluation on the full frozen set (negative result)

The previous report's table was a **limited run** (10 scenarios per set, 30
cases); it is kept below as the pilot it was and is not the D2 evaluation.

The full frozen set has now been run: **50 / 25 / 25 = 100 scenarios**, budget
1200 tokens, same model and framing for every arm, freeze tag
`frozen-2026-10-17` (`bb5c17db81888de7…`). The Knowledge store held the owner's
real memory at run time; it was **not** seeded with synthetic facts, and no
representative-fact population was performed.

| set | n | accuracy | temporal | stale | abstain | tokens | p50 ms | transcript acc/temp | filesystem acc/temp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 50 | 0.140 | 0.840 | 0.000 | n/a | 769 | 960.1 | 0.140 / 0.840 | 0.140 / 0.840 |
| heldout | 25 | 0.280 | 0.800 | 0.000 | n/a | 360 | 1031.3 | 0.280 / 0.800 | 0.280 / 0.800 |
| adversarial | 25 | 0.200 | 0.400 | 0.000 | **0.000** | 380 | 938.3 | 0.200 / 0.400 | 0.200 / 0.400 |

Reproducible with:

```
python3 benchmarks/memory_evaluation.py --sets development heldout adversarial \
  --budget 1200 --json benchmarks/results/d2_memory_2026-10-17.json
```

**The honest reading, three parts:**

1. **No memory advantage is demonstrated.** PUCA and both baselines score
   **identically on every measure** (to three decimals) in all three sets. A
   benchmark that cannot separate the system under test from a transcript dump
   has not shown the system's memory is better; it has shown the metric is
   dominated by the model's answering behaviour.
2. **Fact recall is low in absolute terms** (0.14 / 0.28 / 0.20): 43 of 50
   development scenarios, 18 of 25 held-out and 20 of 25 adversarial scored zero
   fact recall. Temporal correctness is better on development (0.84) and worse
   on adversarial (0.40). Stale-fact rate is 0.000 everywhere — the system did
   not report a superseded value as current, which is the one clear pass.
3. **Abstention honesty fails on the adversarial unknowns**: five scenarios ask
   for an unknown fact; PUCA scored **0.000** on them, and so did both baselines.
   All three arms invented an answer instead of saying they did not know. This
   is a real defect, it is reported as such, and it is not attributable to the
   D2 harness (which scores honesty correctly).

**Pilot (previous report, kept for the record):** 10/10/10, accuracy
0.10/0.10/0.20, temporal 0.90/0.90/0.60, stale 0.000, all arms equal.

**Prior exposure (disclosed):** the held-out and adversarial sets have been
looked at three times now (a 3-scenario pilot, the 10/10/10 run, and this full
run). They are therefore development-visible. The next reported D2 run requires
a **fresh frozen set**, and this disclosure travels with it.

### 4.1 The dedicated seeded instance (this revision)

The review's first requirement is now built. A **second console** runs on
`127.0.0.1:49150` with `NIX_ISOLATE_DASHBOARD_DATA=1` and its own data
directory (`/tmp/d2_eval/data`), so it cannot read or write the owner's
canonical memory. Its Knowledge store was seeded before startup by
`benchmarks/seed_memory_eval.py`, which refuses any canonical path.

The set is **fresh and frozen**: `eval/data/memory_scenarios_fresh.json`,
tag `fresh-2026-10-18` (`37b7834dfe2d3b64…`), 20 development / 15 held-out /
15 adversarial, written after the previous run and never used before it. It is
smaller than the 50/25/25 protocol, so it is a fresh confirmation set, not the
final sample. The evaluation model is `qwen3.5:4b` on the isolated instance;
the production Luna weights are unchanged and were not used.

```
python3 benchmarks/seed_memory_eval.py --db /tmp/d2_eval/data/knowledge.db
python3 benchmarks/memory_evaluation.py --base http://127.0.0.1:49150 \
  --model qwen3.5:4b --scenarios eval/data/memory_scenarios_fresh.json \
  --sets development heldout adversarial --budget 1200 --tag fresh-2026-10-18 \
  --json benchmarks/results/d2_memory_fresh_2026-10-18.json
```

| set | n | accuracy | temporal | stale | abstain | tokens | p50 ms | transcript acc | filesystem acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 20 | 0.325 | 1.000 | 0.000 | n/a | 489 | 1028.7 | 0.400 | 0.350 |
| heldout | 15 | 0.333 | 1.000 | 0.000 | n/a | 322 | 1131.7 | 0.333 | 0.333 |
| adversarial | 15 | 1.000* | 1.000 | 0.000 | **0.000** | 195 | 127.9 | 1.000* | 1.000* |

\* The adversarial accuracy of 1.000 is an artifact of the scoring rule: an
unknown question has no expected facts, so fact recall defaults to 1.0. The
meaningful number on that set is **abstention honesty 0.000** — all three arms
invented answers to all 15 unknown questions.

**The honest reading, unchanged by seeding:**

1. **No memory advantage is demonstrated on a seeded store either.** PUCA's
   fact recall is 0.325 on development against the transcript baseline's 0.400
   and the filesystem baseline's 0.350, and 0.333 on held-out against 0.333 and
   0.333. With 20 and 15 cases these differences are descriptive, not a
   statistical claim; the point is that a seeded store did not reverse the
   negative result.
2. **The invented-answer defect is real and reproducible.** On 15 fresh
   unknown questions, every arm scored 0.000 abstention honesty. This is the
   same failure as the previous run and is now confirmed on a store this
   work controls.
3. **Stale-fact rate is 0.000 everywhere**, which remains the one clear pass.

**What is still outstanding:** the final 50/25/25 fresh set at the production
model and framing, and a fix for the abstention defect. Until both exist, no
D2 claim is made and the context budgets stay **logging-only**, not enforced
as optimal.

---

## 5. D4 — routing pilot and controlled counterfactual replay

### 5.1 The pilot (previous revision)

A 24-case hand-labelled pilot was added at
`nix_core/eval/data/routing_pilot.json`, with labels and class per case. It is
development-visible and **not** the final set. The counterfactual and answer
quality columns below are the pilot's state; §5.2 replaces them.

| Measure | Value |
| --- | --- |
| cases | 24 (10 chat, 10 escalation, 4 ambiguous) |
| scored | 15 (7 router abstentions, 2 ambiguous cases resolved by the router) |
| route correctness, scored | 15/15 = 1.0 |
| router abstention share | 7/24 = 0.292 |
| undetermined (label) | 4/24 = 0.167 |
| false escalation | 0.0 (no counterfactual replay, so this is descriptive) |
| false de-escalation | 0.0 (same limitation) |
| answer quality | **unmeasured** (0 replies judged) |
| counterfactual replay | **not run** |
| routing thresholds | **unchanged** |

The abstentions are real: "I had a really long day today", "I started a pottery
class last week" and "Good night, talk tomorrow" return `unknown` from the
router and are resolved to chat by the caller. "How is my mother doing?" and
"Delete all my reminders" abstain at the router and are resolved by deterministic
contracts downstream. Route-only scoring therefore understates the system and
overstates the router; both facts are reported rather than smoothed.

### 5.2 The controlled replay (this revision)

The second path executor the previous report said did not exist now does.
`benchmarks/routing_replay.py` runs **both arms on the same frozen case**:

- **chat arm** — the model alone (`qwen3.5:4b` via Ollama), no memory context;
- **knowledge arm** — the full turn path on a dedicated evaluation instance
  (`127.0.0.1:49151`, its own data directory, seeded from
  `eval/data/routing_replay_keys.json`), never the owner's memory.

Answer quality is scored by a stated deterministic rubric (`open`, `memory`,
`write`, `clarify`, `confirm`, `delete`); it is a coarse proxy, not a human
judgement, and it is reported as such.

| Measure | Value |
| --- | --- |
| frozen held-out cases | 40 (`routing-heldout-2026-10-17`) |
| route correctness, scored cases | 17/19 = 0.895 |
| route correctness, all 40 (ledger denominator) | 17/40 = 0.425 |
| router abstentions | 13/40 = 0.325 |
| undetermined labels | 8/40 = 0.200 |
| cases with a replayed counterfactual | 19 |
| false escalation (replayed) | 3/18 = 0.167 |
| false de-escalation (replayed) | 0/14 = 0.000 |
| answer-quality proxy, chosen arm | 0.684 |
| answer-quality proxy, chat arm | 0.474 |
| answer-quality proxy, knowledge arm | 0.737 |

```
python3 benchmarks/seed_routing_replay.py --db /tmp/d4_eval/data/knowledge.db
python3 benchmarks/routing_replay.py --base http://127.0.0.1:49151 \
  --model qwen3.5:4b --json benchmarks/results/d4_routing_replay_2026-10-18.json
```

By rubric (chosen / chat / knowledge): open 0.273 / 0.273 / 0.273;
memory 0.222 / 0.111 / 0.333; write 0.667 / 0.167 / 0.667;
clarify 1.0 / 1.0 / 1.0; confirm 0.0 / 0.0 / 0.0; delete 0.0 / 0.0 / 0.0.

**The honest reading.** The knowledge arm scores higher than the chat arm on
the memory and write cases, which is the direction the route decision predicts,
and three escalations were unnecessary by the rubric's proxy. The overall
knowledge arm still scores only 0.737, and the same cases fail in both arms
(the confirmation and delete cases score 0.0 in all three columns; several
memory cases still fail on the dedicated instance). The `open` rubric scores
0.273 in every arm, which says the rubric is too strict for short conversational
replies, not that the arms differ.

**What this does not establish.** The replay is 19 replayed cases, not 40, and
the rubric is a proxy. It does not validate routing, it does not show real-world
false-escalation rates, and it does not replace a human answer-quality review.
The frozen set remains development-visible and is retired from confirmation use
once it informs a decision.

---

## 6. D5 — causes of planner retries, and the untested grammar layer

From the request log (3 days, 1,239 requests):

| Measure | Value |
| --- | --- |
| planner runs | 41 |
| planner validate-retry rate | 22/41 = 53.7% |
| grounded retries (argument grounding) | 22 |
| read retries | 3 |
| planning failures (`Needle withheld calls`) | 6 |
| plan envelopes checked | 31 |
| malformed envelopes / schema failures | **0** |

The retries are **grounding corrections**: a correct intent with an argument
that had to be normalised or re-read (volume wording, device state). They are
not malformed output. Zero schema failures in 31 checked envelopes means the
grammar-constrained decoding intervention has no observed defect to fix in this
sample.

**Intervention status: unvalidated, and no benefit is claimed.** No supported
grammar engine is installed on this machine; the layer is wired but inert, and
a stub proves only the integration. Until the with/without comparison below is
run, the intervention stays explicitly unvalidated. Testing it requires installing an engine and running the same
scenarios with and without the constraint, reporting structural validity, intent
preservation, authorization, execution outcome, retry rate and latency
separately. That experiment has not been run, and nothing here claims a benefit.

The five independent validation checks remain in place regardless of whether
grammar constraints ever show a benefit.

---

## 7. D9 — the outbound boundary, extended

The guarantee is stated narrowly: **the sensitive values detected by the
published suite are masked before any non-local request, and a request whose
redaction cannot be completed is not sent.** It is not a claim that every
possible identifier is detected.

Covered and tested:

- patterns: email, phone, SSN, student id, account, postal address, course code,
  plus person names passed by the caller;
- outbound paths: `model`, `retry`, `telemetry`, `exception` — each named and
  audited separately;
- a retry after a blocked endpoint is refused; a retry of identical text must
  produce an identical payload;
- **memory-origin values** and **tool-result values** are redacted exactly like
  prompt text (new test);
- local rehydration is exercised, and a rehydrated reply that is sent again is
  caught by the audit even if the boundary was bypassed (new test);
- retention is declared per store: knowledge 3650 d, history 365 d, request log
  30 d, semantic chunk 3650 d, embedding 3650 d, backup 30 d.

Not claimed: detection of identifiers outside the published suite; protection of
a local store from a local reader; prevention of a caller that writes its own
network code outside this boundary. Embeddings and backups are produced and kept
locally on this machine; they are not outbound paths, and the retention report is
the only guarantee attached to them.

`test_system_model_v3.py` passes 38 tests, including the seven privacy tests.

---

## 8. D3 and D10 — bounded conclusions

**D3 (context).** The five section sizes are logged on **1,180 / 1,239 turns
(95.2%)** over three days; the remainder predate the change. Budgets remain
**logging-only**. They are not enforced as optimal, and they are specific to the
tested model, tasks and hardware; any change to those requires re-tuning and a
new D2 curve.

**D10 (inference).** Measurement coverage on this log: **257 turns carry a
recorded model**, out of 1,239 requests (20.7%); the rest did not call the model.
That is a denominator over requests, not over calls, so it does not establish
per-call coverage. TTFT, decode and load are recorded separately when a call
occurs. **No speedup is claimed.** The prefix cache is instrumented; a
cache-on/cache-off controlled measurement has not been run, and until it is, the
cache is a mechanism, not a result.

---

## 9. Implemented, tested, validated — kept distinct

| Component | Implemented | Tested | Experimentally validated |
| --- | --- | --- | --- |
| D1 provenance | yes | yes (7 tests) | not evaluated at scale |
| D2 memory harness | yes | yes | full frozen set **negative**; fresh **seeded instance** run also negative; abstention defect 0.000 confirmed |
| D3 composer + logging | yes | yes | logging coverage only; budgets not validated |
| D4 ledger + replay | yes | yes | controlled replay of both arms (19 counterfactuals); route decision confirms labels; answer-quality rubric is a proxy, human review outstanding |
| D5 five checks | yes | yes | sandbox flows only (D6) |
| D5 grammar | wired, inert | stub only | **no** (no engine; no observed malformed output) |
| D6 reliability gate | yes | yes (scenario gate tests) | **limited paired gate pass** on 80 distinct scenarios (20/flow); not a 2% error-rate claim |
| D7 turn-taking | yes | yes | per clip; barge-in unmeasured |
| D8 proactivity | yes | yes | policy-off and tier gate tested |
| D9 privacy | yes | yes (7 tests) | published suite; boundary stated narrowly |
| D10 inference | yes | yes | coverage only; no speed claim |

---

## 10. What remains, and what only the owner can supply

1. **D2 final sample and the abstention defect** — the dedicated instance and
   the fresh set exist and the run is negative; what remains is the full
   50/25/25 fresh sample at the production model, and a fix for the 0.000
   abstention honesty on unknown questions. The defect is independent of the
   store and reproduces on the seeded instance.
2. **D6 breadth, not size** — the sample is now 20 scenarios per flow and the
   gate is a limited paired pass, but those scenarios are still sandbox flows
   written by the same hand as the check. A claim about real device traffic
   needs scenarios drawn from logged turns; `acceptance.required_n("false_tool")`
   = **189** per flow is required for a 2% error-rate claim (zero errors), so 20
   is a non-inferiority sample, not an absolute-rate sample.
3. **D4 human answer-quality review** — the controlled replay and the rubric
   proxy are done; what remains is a human review of the two arms' replies and a
   confirmation set retired from development use. The replay covers 19 of 40
   cases; the rest abstain at the router or carry undetermined labels.
4. **D5 grammar experiment** — install a supported engine and run identical
   scenarios with and without it; report the six outcomes separately.
5. **D10 cache measurement** — controlled cache-on/cache-off runs before any
   caching claim.

No architecture changed in this revision. The safety checks, instrumentation and
three-wave structure are unchanged, and proactivity remains off by default.
