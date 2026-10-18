# PUCA system model v3 — evidence package (revision 6, 2026-10-18)

> **Correction in revision 6.** Revision 5 said the registered k=5 confirmation
> run had been run and reported. It had not: every attempt was torn down with the
> host session before it wrote a result file. §2.6 now reports the absence and
> the exact command, §5 leaves both D2 gates undecided, and §4 no longer lists
> artifacts that do not exist. No number from revision 5 is withdrawn because
> revision 5 reported no D2 number either — but its scope line and its artifact
> list claimed a run that did not happen, and those are corrected here.

**Kind:** report · **Responds to:** owner work order *PUCA D2/D4 evidence
closure*, 2026-10-18 · **Supersedes:** the D2 and D4 sections of
`PUCA_V3_IMPLEMENTATION_REPORT.md` (revision 4), which stay in the repository as
the record of that revision.

**Scope:** the four documented routing/retrieval defects, the versioned
scoring proposal, the D4 human-review packet, and the repeat-protocol
pre-registration. The registered confirmation run is documented as **not
completed on this host**, with the reason and the exact command to run it
elsewhere (§2.6). No architecture change, no promotion, no new claim.

**Reading rule used throughout:** *measured* means a number produced by a
command recorded here on this machine; *not measured* means it is not claimed.
A smoke check is never called validation, an agent rubric is never called a
human review, and passing tests are never called reliability.

---

## 0. Content of this revision, in one table

| Item | State |
| --- | --- |
| Four open defects (stale current fact, `put X in for`, renewal write, deletion by description) | fixed, with failing-before / passing-after evidence (§1) |
| `Cedars & Co.` substring scoring | versioned correction proposed and compared; the frozen v1 rule is unchanged (§1.5) |
| D2 repeat pre-registration | version 1 was not executable as written; version 2 registered before the run (§2) |
| D2 confirmation set | new 50/25/25 set, sha256 frozen before the first sample (§2.4) |
| D2 k=5 confirmation run | **not completed on this host** — three attempts, none wrote a result file; no D2 number is reported and both D2 gates stay undecided (§2.6) |
| D4 human review | packet prepared, **awaiting human scores**; not complete (§3) |
| D6 / D5 / D10 | unchanged: D6 is a limited paired gate pass; D5 and D10 stay unvalidated (§5) |
| Promotion, memory advantage, production readiness | not claimed (§8) |

---

## 1. Phase 1 — routing and retrieval defects

### 1.1 The four defects

| # | Defect | First observed | Fix |
| --- | --- | --- | --- |
| 1 | A current-fact read could return a stale value (`What city do I live in?` returned the superseded city beside the current one — D4 c17) | D4 controlled replay, 2026-10-18 | the personal-recall selection now excludes historical records unless the question carries a past frame (`previously`, `used to`, `before`) |
| 2 | `put X in for <time>` selected no function (D4 c31) | D4 controlled replay | deterministic `_PUT_IN_FOR` / `_PUT_TITLE_FOR` calendar-write rules; an unresolvable time defers instead of guessing |
| 3 | A durable deadline statement was routed to chat and never stored (D4 c32) | D4 controlled replay | `_DUE_STATEMENT` reaches `create_fact`, and the Core no longer lets a conversational turn prediction demote a boundary decision to chat |
| 4 | Fact deletion by description was missing (`delete the note about my old address` was read as a calendar cancel, D4 c38) | D4 controlled replay | `delete_fact` with a confirmation round-trip, plus fail-closed guards for unknown and too-broad descriptions |

### 1.2 Failing before, passing after

The pre-fix state is reproduced from the last committed revision rather than
from memory: a detached worktree of `HEAD` (`6744651`) with *only* the final
test files copied in, so the tests run against the pre-fix code.

```
git worktree add --detach /tmp/puca_prefix HEAD
cp nix_knowledge/tests/test_evidence_defects.py /tmp/puca_prefix/nix_knowledge/tests/
cp nix_core/test_evidence_defect_regressions.py nix_core/test_scoring_v2.py /tmp/puca_prefix/nix_core/
python -m pytest -q tests/test_evidence_defects.py            # in the worktree
python -m pytest -q test_evidence_defect_regressions.py test_scoring_v2.py
```

| Suite | Before (HEAD worktree) | After (working tree) |
| --- | --- | --- |
| `nix_knowledge/tests/test_evidence_defects.py` | **14 failed, 1 passed** | **15 passed** |
| `nix_core/test_evidence_defect_regressions.py` + `test_scoring_v2.py` | **11 failed, 7 passed** | **18 passed, 0 failed** |
| `nix_core/test_repeat_protocol.py` (new harness behaviour) | not applicable (module added with the amendment) | **6 passed** |

Logs: `nix_core/benchmarks/results/p1_evidence_2026-10-18/`
(`p1_before_knowledge.txt`, `p1_before_core.txt`, `after_knowledge.txt`,
`after_core.txt`).

**One honest gap.** The pre-fix *live* end-to-end transcript was not
reproducible in the worktree: the HEAD chat model path fails with
`FileNotFoundError`, so every turn fell back to chat for an unrelated reason.
That attempt is retained as `prefix_e2e_unusable.txt` and is **not** cited as
evidence. The pre-fix state is pinned by the test runs above, not by that
transcript.

### 1.3 Adjacent cases covered by the regression tests

Current versus superseded facts, past-frame reads, unknown facts, temporal
resolution of a calendar write, write-then-read-back, paraphrased writes
(`book … in for`, `slot … in for`), paraphrased deadline statements, paraphrasd
deletions (polite lead, trailing meta-noun, `forget`), deletion of a calendar
event staying a cancellation, fail-closed deletion (no match, too broad),
multi-fact reads, and the composed-reply date checks in Core.

### 1.4 Safeguards still enforced

* deletion requires an explicit confirmation and a following `yes`; a different
  turn abandons the pending deletion (`test_a_fact_deletion_is_confirmed_before_it_deletes`,
  `test_a_declined_deletion_deletes_nothing`);
* deletion fails closed on an unknown description (`not_found`) and on one that
  matches several records (`refused_too_broad`), with nothing removed;
* an unknown question writes nothing: the D2/E2E check compares the fact-row
  count before and after (`FACT ROWS before=3 after=3`), and
  `test_a_due_question_is_a_read_not_a_write` asserts the store does not grow;
* a composed reply that moves a stored date is rejected
  (`_stated_dates_are_grounded`), including the invented-year case.

### 1.5 The `Cedars & Co.` substring issue — a versioned proposal

The frozen D2 rule (`eval/memory_sets.py score_answer`) counts an expected value
when its literal characters appear in the reply. Two recorded replies in the
WP6 smoke run carry the fact and score 0.0 anyway: `I work at Cedars & Co.`
against an expected `Cedar & Co` (one letter), and `Cedar and Co.` against
`Cedar & Co` (the ampersand spelled out).

The frozen rule was **not** edited, because the already-published WP2 number was
produced with it. Instead:

* `eval/scoring_v2.py` — `scoring-v2-2026-10-18`, a separate rule that
  normalises (NFKC, `&` → `and`, punctuation, whitespace, ordinals, zero
  padding) and matches token runs with a conservative plural fold, while
  copying v1's abstention detector verbatim so the comparison isolates value
  matching;
* the fair comparison is a **single pass over the same recorded replies**:
  `benchmarks/rescore_scoring_v2.py` scores every case with both rules and
  reports paired per-case deltas.

Result (`benchmarks/results/d2_scoring_v2_comparison_2026-10-18.json`, source
`d2_wp6_smoke_2026-10-18.json`, 75 known + 25 unknown cases):

| Quantity | Value |
| --- | --- |
| mean fact recall, v1 (frozen) | 0.5733 |
| mean fact recall, v2 (proposal) | 0.6133 |
| mean paired delta | +0.040 |
| paired case-level bootstrap 95% interval on the delta | [+0.005, +0.065] |
| cases that moved up / down | 4 / 0 |
| full-recall cases, v1 → v2 | 65 → 69 |
| stale cases, v1 → v2 | 0 → 0 |

v2 never scores below v1 on the recorded run (asserted by
`test_v2_never_scores_below_v1_on_the_recorded_smoke_run`), so the movement is
one-directional and attributable to the rule. **The proposal is not adopted:**
any k=5 confirmation run would be scored with the frozen v1 rule, and v2 would
be reported beside it only.

### 1.6 Files modified in Phase 1

| File | Why |
| --- | --- |
| `nix_knowledge/nix_knowledge/rules.py` | rules 2 and 4 (put-in-for, fact deletion) and their paraphrases |
| `nix_knowledge/nix_knowledge/needle.py`, `retrieval.py`, `query`-side selection | rule 1 (current vs historical selection) and the multi-fact action split |
| `nix_knowledge/nix_knowledge/functions/facts.py` | `delete_fact` with the confirmation and fail-closed guards |
| `nix_knowledge/nix_knowledge/functions/calendar.py` | rule 2's event write path and read-back |
| `nix_knowledge/nix_knowledge/keys.py`, `states.py`, `database.py`, `models.py`, `engine.py`, `semantic/embedder.py`, `spelling.py` | supporting changes for the above |
| `nix_core/routing_engine.py` | rule 3's boundary route and the due-question read |
| `nix_core/brain.py` | pending-deletion turn handling, delete reply formatting, grounded-date checks |
| `nix_core/test_evidence_defect_regressions.py` | Core regression tests |
| `nix_knowledge/tests/test_evidence_defects.py` | knowledge regression tests |
| `nix_core/eval/scoring_v2.py`, `nix_core/benchmarks/rescore_scoring_v2.py`, `nix_core/test_scoring_v2.py` | the versioned scoring proposal and its comparison |

No model weights, prompts, persona text or decoding settings were changed.

### 1.7 End-to-end smoke after the fixes

Command: a dedicated isolated console (`NIX_ISOLATE_DASHBOARD_DATA=1`, its own
data directory, no access to the owner's memory) on `127.0.0.1:49152`, driven by
`benchmarks/results/p1_evidence_2026-10-18/p1_e2e_check.py`. Recorded output
(`p1_e2e_after.txt`), with the run's route/rule alongsise each turn:

| Turn | Route / rule | Reply (excerpt) |
| --- | --- | --- |
| `What is my blood type?` | knowledge / knowledge_process | "I don't have that yet." — fact rows 3 → 3 |
| `Put the optician in for Monday at 10am` | knowledge / knowledge_process | "Scheduled 'optician' Monday, October 19, 2026 at 10:00 AM." |
| `The renewal is due March 4` | knowledge / knowledge_process | "Stored: The renewal is due March 4" |
| `Delete the note about my old address` | knowledge / knowledge_process | "I found '…14 willow lane'. Nothing has been deleted yet — say yes and I'll remove it, or no to keep it." |
| `yes` | knowledge / fact_deletion_confirmed | "Deleted 'my old address was 14 willow lane'." (status → deleted) |
| `What was my old address?` | knowledge / knowledge_process | "I don't have that yet." |
| `When is my optician appointment?` | knowledge / knowledge_process | "1 event(s) found: optician — Monday, October 19, 2026 at 10:00:00 AM" |
| `When is the renewal due?` | knowledge / knowledge_process | "The renewal is due March 4." (no invented year) |

This is a **smoke check**, not evidence of general reliability, and it is
reported separately from the formal evaluation in §2.

---

## 2. Phase 3 — the repeat protocol

### 2.1 Why version 1 could not be executed

`benchmarks/results/d2_repeat_prereg_2026-10-18.json` fixed the sample count and
the gates but left three procedures undefined: how five clustered samples become
an interval, what happens when a case's samples tie, and what happens to a
sample the model never answers. Two analysts could have produced two different
numbers from one result file, which is the failure the pre-registration exists
to prevent. Per the work order, execution stopped and a versioned amendment was
registered.

### 2.2 What version 2 adds

`benchmarks/results/d2_repeat_prereg_2026-10-18_v2.json` (registered before the
first sample; version 1 preserved):

1. **Clustered uncertainty.** The case is the unit. Abstention is evaluated with
   the pre-registered Wilson interval on the 25 *case-level* outcomes, never on
   125 samples, with a case-level cluster bootstrap (10,000 resamples, seed
   20261018) reported as a sensitivity beside it. The memory comparison uses a
   paired percentile bootstrap that resamples cases (§2.3).
2. **Ties.** Three or more honest samples of five make a case honest; two or
   fewer does not. At k=5 no tie is possible; under the missing-run rule an
   exact 50/50 split counts as **not** honest.
3. **Missing runs.** A sample that cannot be read scores as a failure (recall 0,
   not honest), keeps its place in the denominator, and is recorded with its
   error. A case with any failed sample is flagged degraded. If any arm exceeds
   a 5% failure share of its planned samples the run is **invalid** and must be
   repeated in full; failed samples are never selectively retried.

The gates themselves are unchanged (≥ 23 of 25 pooled unknown cases honest and
Wilson lower bound ≥ 0.75; paired interval excluding zero against both
baselines).

### 2.3 Harness implementation and tests

* `eval/memory_sets.py`: `_attempt()` scores an unreadable sample as a failure
  and records the error; `FAILURE_SHARE_LIMIT = 0.05`;
  `sample_failure_report()` applies the invalidity rule with planned-sample
  denominators; `run_set(..., workers=…)` answers cases concurrently as a
  wall-clock setting only, with a case's own samples still sequential.
* `benchmarks/memory_evaluation.py`: `--samples`, `--workers`, the failure
  report, and exit status 3 on an invalid run.
* `test_repeat_protocol.py` (6 tests): a failed sample is scored and kept; the
  majority rule and the even-sample tie; an unreadable sample is not honest; the
  abstention denominator counts 25 cases, not 125 samples; the 5% run-validity
  boundary; the paired interval resamples cases; and concurrent cases produce
  identical scores.

### 2.4 The confirmation set: written, hashed, frozen

Set: `eval/data/memory_scenarios_k5_2026-10-18.json`, tag
`k5-confirm-2026-10-18`, built deterministically by
`benchmarks/build_memory_set.py` from an explicit new cluster list:

| Set | Cases | Unknown |
| --- | --- | --- |
| development | 50 (25 current reads, 20 paraphrases, 5 unknown) | 5 |
| held-out | 25 (20 superseded-value history reads, 5 unknown) | 5 |
| adversarial | 25 (10 two-fact reads, 15 unknown) | 15 |
| **total** | **100** | **25** |

* file sha256 of the set on disk:
  `96bee50d8c02a7e8fd7b8ca5115b49e8fb925d65ee1cac25119ec0acb618fbcf`
* frozen manifest (`benchmarks/results/d2_k5_freeze_2026-10-18.json`, written
  **before** the first sample): its `sha256` field is
  `6c3fdb2842f0931ccd8753bc947a14eac5877e23e684ffab9c35bb0a5bebd67e`, the hash
  of the set as canonicalised by `eval/memory_sets.py`. Recomputing that field
  from the set on disk reproduces it exactly, and the artifact's own counts
  (50/25/25, 25 unknown) match the file. The artifact's file bytes hash to
  `1a81e5fb2c3bfbcb096ed17ae3ae0676b96cc3fa7a3abb8b596e25b4e8d5ff3c`.
* prior exposure: **none** — the set was written after the WP6 fix, hashed
  before any sample was requested, and no sample was inspected before the run;
* retired from confirmation use: `confirm-2026-10-18`, `fresh-2026-10-18`;
* the single-sample `d2_memory_confirm_2026-10-18.json` is preserved unchanged.

Seeded into a dedicated store, never the owner's memory:

```
python3 benchmarks/seed_memory_eval.py --db /tmp/d2_k5/data/knowledge.db \
  --scenarios eval/data/memory_scenarios_k5_2026-10-18.json \
  --json benchmarks/results/d2_k5_seed_2026-10-18.json     # 25 active, 25 superseded
```

### 2.5 The registered command

This is the command the pre-registration fixes. It was issued unchanged in every
attempt; §2.6 records that it did not finish here.

```
NIX_DATA_DIR=/tmp/d2_k5/data NIX_ISOLATE_DASHBOARD_DATA=1 \
  python -c "import console_extend as ce; s,p=ce.build_app(port=49150); s.serve_forever()"

python3 benchmarks/memory_evaluation.py --base http://127.0.0.1:49150 \
  --model luna-v6-contextual-v1 --baseline-model qwen3.5:4b --baseline-mode ollama \
  --scenarios eval/data/memory_scenarios_k5_2026-10-18.json \
  --sets development heldout adversarial --budget 1200 --samples 5 --workers 6 \
  --tag k5-confirm-2026-10-18 --json benchmarks/results/d2_k5_confirm_2026-10-18.json
```

Model and framing are the ones the frozen single-sample result used
(`qwen3.5:4b` behind the console's `luna-v6-contextual-v1`), the same 1200-token
budget per arm, and the frozen v1 scoring rule. `--workers 6` is a wall-clock
setting: PUCA's rows are byte-identical at workers 1 and 4 on a six-case check
(`test_concurrent_cases_do_not_change_the_scores` pins the same property), and a
case's own samples stay sequential.

### 2.6 Results — none: the run did not complete on this host

**No D2 confirmation number is reported.** The registered command was started
three times on this machine and every attempt was killed by the host's session
teardown before it finished. No attempt wrote a result file, so **no sample was
scored, read, or reused**; there is nothing to report and nothing was salvaged
from the partial runs.

| Attempt | Started (UTC−5) | Last seen running | Ended | Result file |
| --- | --- | --- | --- | --- |
| 1 | 08:52 | 09:41 (≥ 48 min) | host teardown | none |
| 2 | 10:17 | 10:46 (≥ 29 min) | host teardown | none |
| 3 (development set only) | 10:59 | 11:04 (≈ 5 min) | host teardown | none |

The abort is a property of the machine, not of the measurement: the sessions on
this host are torn down on a cadence shorter than the run, and a background
process does not survive the teardown (it was verified that `setsid` survives
individual tool calls, but not the session end).

**Why it needs longer than a session.** The registered design is 1,500
generations: 100 cases × 5 samples × 3 arms. The PUCA arm is cheap (p50 1.82 s
per case-sample in the smoke run, six cases answered concurrently), but the two
baseline arms are 1,000 model-only calls to a **single-slot** local backend
(`llama-server -np 1`), so they serialize: at the observed per-call cost the
full run is roughly an hour, whatever `--workers` is set to.

What is *not* concluded from this: nothing about honesty, recall or the memory
comparison. An aborted run is not a failure of the system under test, and it is
not evidence in either direction — it is an absent measurement.

**What would produce the number.** One uninterrupted window of about an hour on
a host that does not tear the process down, with the frozen set and this exact
command (the freeze artifact and the set hash are unchanged):

```
NIX_DATA_DIR=/tmp/d2_k5/data NIX_ISOLATE_DASHBOARD_DATA=1 \
  python -c "import console_extend as ce; s,p=ce.build_app(port=49150); s.serve_forever()"

python3 benchmarks/seed_memory_eval.py --db /tmp/d2_k5/data/knowledge.db \
  --scenarios eval/data/memory_scenarios_k5_2026-10-18.json \
  --json benchmarks/results/d2_k5_seed_2026-10-18.json

python3 benchmarks/memory_evaluation.py --base http://127.0.0.1:49150 \
  --model luna-v6-contextual-v1 --baseline-model qwen3.5:4b --baseline-mode ollama \
  --scenarios eval/data/memory_scenarios_k5_2026-10-18.json \
  --sets development heldout adversarial --budget 1200 --samples 5 --workers 6 \
  --tag k5-confirm-2026-10-18 --json benchmarks/results/d2_k5_confirm_2026-10-18.json

python3 benchmarks/analyse_k5_result.py \
  --result benchmarks/results/d2_k5_confirm_2026-10-18.json \
  --json benchmarks/results/d2_k5_analysis_2026-10-18.json
```

The analysis script already implements the pre-registered reading (case as the
unit, Wilson interval on the 25 unknown cases, cluster bootstrap as a
sensitivity, paired case-level bootstrap for the comparison). It has not been
run on a confirmation result, because none exists.

### 2.7 Deviations and observations to record

* **The baseline arms are stochastic.** On a six-case check the transcript arm
  scored 1.0 for a superseded-value question in one run and 0.0 in the next.
  The baselines call the model directly with the frozen runner's decoding, so
  the paired difference carries sampling noise in **both** arms. This is exactly
  why the repeat protocol fixes five samples per case; it is a property of the
  measurement, not a defect, but it means a single-run baseline number should
  not be compared with a single-run PUCA number.
* **Concurrency is not a protocol change** but it is a documented runner
  setting, and it was verified not to change scores.
* The set is synthetic and template-generated from a new cluster list. It is
  new and independently frozen; it is not evidence of production coverage.
* **The registered run was not completed here** (§2.6). The pre-registration is
  unchanged and still unused: nothing has been sampled from the frozen set, so
  a later run on a stable host is still the first confirmation run, not a
  second look at the same data.

---

## 3. Phase 2 — D4 human review

**Status: packet prepared; the review is NOT complete and no human score
exists.** The agent rubric pass recorded on 2026-10-18 remains what it says it
is: an agent rubric pass, not an independent human review.

Artifacts:

| File | Content |
| --- | --- |
| `benchmarks/results/d4_human_review_packet_2026-10-18.csv` | all 38 scored cases: identifiers, class, label, rubric, expected behaviour, the turn, the chosen route and rule, whether the route was correct, **both arms' original replies**, ten blank human columns (identity, date, four dimensions per arm, justification), the preserved agent scores in clearly-named `agent_*` columns, and the preserved escalation disposition where one exists |
| `benchmarks/results/d4_human_review_packet_2026-10-18.md` | the same 38 cases as a readable sheet with both replies, for reviewing away from a CSV |
| `benchmarks/results/d4_human_review_instructions_2026-10-18.md` | the four scoring dimensions with 1–5 anchors, the workflow, the return format |
| `benchmarks/results/d4_review_status_2026-10-18.json` | status `AWAITING HUMAN SCORES`, counts, the six misses with their preserved dispositions, and the recording requirements |
| `benchmarks/results/d4_human_review_2026-10-18.csv` | the original agent sheet, **unedited**, source of every reply and agent score |

Coverage: 38 of 46 cases were replayed with a controlled alternative and are
scored; the 8 cases the router left undetermined have no second arm and are
listed in the packet as not replayed. Both arms' replies are preserved verbatim.

The six escalation misses (c17, c23, c24, c31, c32, c38) keep their written
dispositions. `c23` and `c24` are marked fixed and have regression tests in
`nix_knowledge/tests/test_evidence_defects.py`; `c17`, `c31` and `c38` were
open defects when the dispositions were written and are now fixed with tests
§1.3, so the packet flags them as *open-defect dispositions retained — update
only with new evidence* rather than rewriting them. `c32` (the renewal write) is
covered by the Core and knowledge tests in §1.3.

Per-case human ratings, per-dimension results and denominators **do not exist
yet**. Nothing in this package presents the agent scores as human judgment.

---

## 4. Artifacts and reproduction

Machine-readable evidence, under `nix_core/benchmarks/results/` unless a row
says otherwise:

| Artifact | What it is |
| --- | --- |
| `d2_repeat_prereg_2026-10-18.json` | version 1 pre-registration (superseded, preserved) |
| `d2_repeat_prereg_2026-10-18_v2.json` | version 2 amendment (registered before the run) |
| `eval/data/memory_scenarios_k5_2026-10-18.json` (in `nix_core/eval/data/`, not under `results/`) | the frozen confirmation set |
| `d2_k5_freeze_2026-10-18.json` | freeze artifact with the set hash and prior-exposure statement |
| `d2_k5_seed_2026-10-18.json` | seed result for the dedicated store |
| `d2_k5_smoke_2026-10-18.json` | three-case smoke check on the isolated instance (labelled smoke) |
| `d2_k5_confirm_2026-10-18.json` | **does not exist** — the registered run was never completed here (§2.6); it is the output of the command recorded there |
| `d2_k5_analysis_2026-10-18.json` | **does not exist** — `benchmarks/analyse_k5_result.py` has no confirmation result to read |
| `d2_scoring_v2_comparison_2026-10-18.json` | v1 vs v2 on the same recorded replies |
| `p1_evidence_2026-10-18/` | before/after regression logs and the post-fix E2E transcript |
| `d4_human_review_packet_2026-10-18.{csv,md}`, `d4_human_review_instructions_2026-10-18.md`, `d4_review_status_2026-10-18.json` | the D4 review packet |
| `d4_human_review_2026-10-18.csv`, `d4_routing_confirm_2026-10-18.json` | the original agent sheet and the frozen routing result (unedited) |

Frozen single-run results that this revision does **not** overwrite:
`d2_memory_confirm_2026-10-18.json`, `d2_memory_fresh_2026-10-18.json`,
`d4_routing_heldout.json`, `d6_gate_2026-10-17.json`.

---

## 5. Gate decision table

The table separates what was **measured** on this machine from what was
**decided** by a person from that measurement, and refuses to fill a D2 row:
there is no confirmation result (§2.6).

| Gate | Evidence | State | Decision |
| --- | --- | --- | --- |
| Phase-1 defects (stale current fact, `put X in for`, renewal write, deletion by description) | failing-before / passing-after suites, 15 + 18 tests (§1.2), post-fix E2E smoke (§1.7) | measured | closed as *defects removed*, not as reliability |
| `scoring-v2` proposal | paired v1/v2 pass over the recorded smoke replies, +0.040 mean recall, interval [+0.005, +0.065], 4 up / 0 down (§1.5) | measured | **undecided** — owner chooses adopt / revise / reject |
| D2 abstention honesty (≥ 23 of 25 unknown cases honest, Wilson lower bound ≥ 0.75) | none — no confirmation result exists | **not measured** | **undecided**: cannot pass or fail on an absent run |
| D2 memory advantage (paired interval excluding zero against both baselines) | none — no confirmation result exists | **not measured** | **undecided**: no advantage and no null result is claimed |
| D4 answer quality (human 1–5 rubric, both arms) | packet of 38 cases, all human fields blank (§3) | **not measured** | **undecided** — awaiting a human reviewer |
| D4 routing outcomes | controlled replay of 38 of 46 cases, frozen routing result, 6 escalation misses with dispositions (§3, §6) | measured | retained as written; the three open-defect dispositions now have fixes and tests |
| D6 reliability | paired gate on 80 distinct scenarios (20/flow), `d6_gate_2026-10-17.json` | measured | **limited paired pass**, unchanged; not a 2% error-rate claim |
| D5 grammar | no controlled run | **not measured** | unvalidated |
| D10 inference timings | no controlled run | **not measured** | unvalidated |
| Promotion / production readiness | no promotion criterion has been met | — | **not promoted**, nothing enabled by this revision |

The honesty rule for reading this table: a *not measured* row is not a failure
of the system and not a pass. It is a row that needs a run, a reviewer, or an
owner decision, and each one names which.

---

## 6. Remaining defects and limitations

1. **Deletion and write coverage is partial.** Fixed: `delete the note about …`,
   a polite lead, a trailing meta-noun, `forget`. Still reaching the calendar
   rule or no rule: e.g. `remove the saved reminder note …` shapes not covered by
   the meta-noun list. Reported, not fixed.
2. **Compound questions.** `Tell me my X and my Y` is split into two reads;
   `What is my X and my Y?` is not split and returns nothing. Reported by a test
   docstring, not fixed.
3. **Baselines are stochastic** (§2.7), so any single-run comparison carries
   uncontrolled variance.
4. **The confirmation set is synthetic.** It is a frozen, new, independent
   100-case sample; it does not measure real user traffic.
5. **Weak-host timing.** The evaluation ran on a CPU-bound local backend
   (baseline p50 latency in seconds). The result says nothing about production
   latency.
6. **D4 remains unreviewed by a human.** Until the CSV comes back filled in,
   there is no human answer-quality evidence and the false-escalation
   dispositions are as written on 2026-10-18.
7. **Agent rubric scores are not human scores**, and the answer-quality proxy in
   the routing result is a deterministic heuristic, not a reviewer.

---

## 7. Decisions that require the owner

1. **Adopt, revise or reject `scoring-v2`** before any further D2 number is
   compared with the published v1 baseline (§1.5).
2. **Complete the D4 human review** by returning the packet CSV with identity,
   method, per-case scores and justifications (§3).
3. **Run the D2 confirmation**: it needs one uninterrupted hour on a host that
   does not tear the process down (§2.6). The set is still unused, so the first
   completed run is the first confirmation run. If it also cannot be run, the
   alternative is to record D2 as unmeasured for v3 rather than to shrink the
   protocol after the fact.
4. **D6** stays a limited paired gate pass; expanding it to a 2% error-rate
   claim needs 189 scenarios per flow.
5. **D5 grammar and D10 inference** remain unvalidated; both need their own
   controlled runs before any claim.

---

## 8. What is not claimed

* Not claimed: a memory advantage, routing quality, reliability, or any
  promotion. The system is not promoted, and no shadowed or disabled phase is
  enabled by this revision.
* Not claimed: that the Phase 1 fixes establish general reliability. They remove
  four documented defects and eight known paraphrases; §6 lists what is still
  uncovered.
* Not claimed: any D2 confirmation result at all. The run was pre-registered
  but not completed here; §2.6 reports the absence, not a null result.
* Not claimed: that any agent-generated score is human judgment, that any smoke
  check is formal validation, or that passing tests are production readiness.
