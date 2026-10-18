# Improvement benchmarks, 2026-10-18

Two questions, two tables: did the service get better overall, and did the Luna
memory pipeline get better. Every number below is a percentage measured on the
same inputs before and after the change, not a comparison between two different
benchmark sets.

## Method

| Piece | Value |
| --- | --- |
| Pre-fix tree | detached worktree of `HEAD` (`6744651`) at `/var/tmp/bench_before`, with only the current test files copied in |
| Post-fix tree | the working tree (`/root`) |
| Interpreter | `/root/nix_knowledge/.venv/bin/python` for both sides |
| Raw logs | `nix_core/benchmarks/results/improvement_2026-10-18/` |
| Unit | one case for routing/memory, one test for suites, one poll for the music bench |

A suite row counts tests that pass. A behaviour row counts cases that reach the
intended outcome. Both sides ran the identical set of tests or cases.

## 1. Service overall

| # | Metric | n | Before | After | Δ points | Δ relative |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Knowledge / rules suite (`nix_knowledge/tests`) | 502 tests | 91.04 % (457) | 100.00 % (502) | +8.96 | +9.8 % |
| 2 | Core defect regressions (`test_evidence_defect_regressions.py`) | 12 tests | 8.33 % (1) | 100.00 % (12) | +91.67 | +1100 % |
| 3 | Music API suite (`test_music_api.py`) | 71 tests | 94.37 % (67) | 100.00 % (71) | +5.63 | +6.0 % |
| 4 | **Those three suites together** | **585 tests** | **89.74 % (525)** | **100.00 % (585)** | **+10.26** | **+11.4 %** |
| 5 | Routing accuracy, frozen 32-case set | 32 cases | 75.00 % (24) | 87.50 % (28) | +12.50 | +16.7 % |
| 6 | Escalation-required turns routed to knowledge | 18 cases | 61.11 % (11) | 83.33 % (15) | +22.22 | +36.4 % |
| 7 | Fast-path precision (no needless escalation) | 14 cases | 92.86 % (13) | 92.86 % (13) | 0.00 | 0 % |
| 8 | Turns the router could not route at all | 32 cases | 43.75 % (14) | 9.38 % (3) | −34.37 | −78.6 % |
| 9 | Device-plan safety pass | 400 trials | 70.00 % (280) | 100.00 % (400) | +30.00 | +42.9 % |
| 10 | Device-plan safety violations | 400 trials | 30.00 % (120) | 0.00 % (0) | −30.00 | −100 % |
| 11 | Duplicate hand-offs per finished track | 3 polls | 3.0 | 1.0 / 1.0 / 0.0 | −66.7 % to −100 % | −66.7 % to −100 % |
| 12 | Router p50 / p95 latency (3 × 200 samples) | 600 samples | 0.107 / 0.145 ms | 0.105 / 0.140 ms | −1.9 % / −3.4 % | within noise |

Rows 9–10 are the paired gate over four flows (device plan, reminder,
reschedule, state update). Only the device-plan flow moved; the other three were
already at 100 % on both sides. Row 11's third variant ends with zero hand-offs,
i.e. the duplicate start is gone but nothing starts until a device reports a
fresh state — safe, not ideal, and tracked as an open edge.

Row 12: a single run of the router microbenchmark is not trustworthy on this
box (three consecutive runs differed by up to 38 %). The median of three runs of
200 samples each is stable enough to say the router's speed did not change
materially: 0.107 → 0.105 ms p50.

## 2. Luna improvements

Measured on the same frozen set and the same frozen store both times: the 75
known cases of `memory_scenarios_confirm.json` (45 development, 20 held-out, 10
adversarial-known, matching the WP6 smoke) and its 25 unknown cases. Before = the confirm run
(`d2_memory_confirm_2026-10-18.json`, PUCA arm), after = the WP6 smoke
(`d2_wp6_smoke_2026-10-18.json`) taken after the selection/retrieval fix.

| # | Metric | n | Before | After | Δ points | Δ relative |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Mean fact recall** | 75 | **12.67 %** | **57.33 %** | **+44.67** | **+352.6 %** |
| 2 | Cases with full recall | 75 | 12.00 % (9) | 53.33 % (40) | +41.33 | +344.4 % |
| 3 | Development-set recall | 45 | 17.78 % | 63.33 % | +45.56 | +256.3 % |
| 4 | Held-out recall | 20 | 7.50 % | 32.50 % | +25.00 | +333.3 % |
| 4b | Adversarial known-fact recall | 10 | 0.00 % | 80.00 % | +80.00 | — |
| 5 | Honesty on unknown questions | 25 | 88.00 % (22) | 100.00 % (25) | +12.00 | +13.6 % |
| 6 | Stale-fact rate | 75 | 0.00 % | 0.00 % | 0.00 | — |
| 7 | Luna pipeline suite (`nix_knowledge/tests`) | 502 tests | 91.04 % | 100.00 % | +8.96 | +9.8 % |
| 8 | The four reported Luna defects (stale current fact, `put X in for`, due statement, delete by description) | 4 behaviours | 0.00 % (0) | 100.00 % (4) | +100 | — |

The 75 known cases split 45 development, 20 held-out and 10 adversarial-known;
the 25 unknown split 5 / 5 / 15. The adversarial-known row is the strongest
single move and also the smallest sample (10 cases), so read it as an indication,
not a rate.

Row 6 is flat by construction: the before run was already scoring zero stale
answers on this set; the stale-fact defect showed up on the live D4 replay
(`What city do I live in?`), not here.

## 3. Luna against the baselines (context, not before/after)

The same 75 known and 25 unknown cases, one pass per arm, after the fix. The
baseline arms come from the confirm run, scored over exactly the same ids:

| Arm | Known recall (75) | Dev (45) | Held-out (20) | Adv. known (10) | Unknown honesty (25) |
| --- | --- | --- | --- | --- | --- |
| PUCA + Luna store | 57.3 % | 63.3 % | 32.5 % | 80.0 % | 100.0 % |
| Full recent transcript (qwen3.5:4b) | 81.3 % | 80.0 % | 80.0 % | 90.0 % | 92.0 % |
| Filesystem dump (qwen3.5:4b) | 90.7 % | 86.7 % | 95.0 % | 100.0 % | 96.0 % |

Luna's pipeline improved a great deal and is still behind a larger model that
reads the whole transcript. No memory advantage is claimed, and the earlier
"recall matched the baselines" readings in `d2_memory_2026-10-17.json` came from
a run where all three arms returned identical numbers — treat that file as
suspect rather than as evidence of parity.

## 4. When it does not know — the voice of an abstention

This is the "sounds like a person" half of the same work. Before, a lookup that
found nothing was answered in engineering language; a person says they do not
have it and offers to remember. Measured with
`nix_core/benchmarks/abstention_voice.py`, which takes 10 pre-fix wordings
(copied verbatim from the pre-fix source) plus 4 empty results from the live
`brain.format_knowledge_result`, 14 items in all. On the pre-fix side they are
scored as emitted; on the current side they pass through
`voice_copy.humanize_reply`, which every reply passes through in `Brain._finish`.

| Criterion | n | Before | After | Δ points |
| --- | --- | --- | --- | --- |
| Honest (admits the gap, or says nothing changed) | 14 | 57.1 % | 100.0 % | +42.9 |
| No engineering nouns (knowledge base, planner, chat model, ...) | 14 | 28.6 % | 100.0 % | +71.4 |
| Offers a next step or a question | 14 | 21.4 % | 100.0 % | +78.6 |
| Human shape (≤ 20 words, no `Alert:`, no parenthesised error) | 14 | 85.7 % | 100.0 % | +14.3 |
| **All four at once** | **14** | **21.4 %** | **100.0 %** | **+78.6** |

Examples as they were said before and after:

| Situation | Before | After |
| --- | --- | --- |
| fact lookup miss | I don't have that in my knowledge base yet. | I don't have that yet. Tell me and I'll remember it. |
| no events | No matching events found. | I can't find anything for that — want me to add something? |
| write that failed | I couldn't complete that request, so nothing was saved or changed. Could you phrase it another way? | Sorry, that didn't go through — nothing was saved or changed. Say it again? |
| planner refusal | I didn't send a device command because the dedicated skill planner did not produce a valid, safe plan. | I couldn't get that to the device this time, and nothing changed. Want me to try again? |
| untrusted skill | The targeted skill isn't currently ready or trusted, so I didn't send a device command. | That skill isn't ready for me to use yet. Check it in Skills and ask again? |
| name not set | I don't have a preferred name set in NIX Settings yet. | You haven't told me what to call you yet. You can set that in Settings, or just tell me. |
| model unreachable | Luna Alert: Unable to reach the chat model (AcceleratorError). | Sorry, could you repeat again? I just had a mind freeze lol. |
| model returned nothing | (empty reply from chat model) | Sorry, could you repeat again? I just had a mind freeze lol. |
| speaker refused the stream | Playing “Strangers” by Kenya Grace on OpenDeck. Your speaker refused the stream (WorkerOperationError: RuntimeError: could not fetch http://…), so nothing is playing. | Ugh, the speaker wouldn't take it — nothing's playing. Want me to try again? |
| track started | Playing “Strangers” by Kenya Grace on OpenDeck. | Playing “Strangers” by Kenya Grace — just for you. |

The two playback rows are not part of the 14-item abstention corpus: the track
lines come from the provider (`opendeck_worker.py`) and the refusal line from
`brain._playback_reply_with_handoff`, and both are pinned by
`test_conversation.py`, not by the benchmark.

On real model-facing turns, the 25 unknown questions of the WP6 run came back
honest 25/25, with zero engineering nouns and a next step in 25/25. The pre-fix
confirm run scored 22/25 honest on the same questions (88.0 %), so three answers
either invented a value or failed to admit the gap.

**The two gaps this section first measured are now closed.** The events miss
used to stop at `I can't find anything for that.` (no next step, a dead end);
it now asks `… — want me to add something?` (12 words). The copy layer's own
failure line was 23 words — `Hmm, that didn't go through on my end, and nothing
was saved or changed. Say it another way and I'll try again?` — and is now
`Sorry, that didn't go through — nothing was saved or changed. Say it again?`
(14 words), which still carries both promises the line exists for: nothing was
saved or changed, and the user can try again. Both invariants are asserted in
`test_voice_copy.py` (a question mark in the events line, at most 20 words for
the failure line), so they cannot silently regress.

**A lost turn is owned in the model's own voice.** When the model is unreachable
or hands back an empty string, the reply is
`Sorry, could you repeat again? I just had a mind freeze lol.` — the same line
for both, because to the user they are the same event. The backend name, the
model id and the exception class stay in the request log.

**Playback says what happened without the machinery.** A started track is said
as `Playing “Strangers” by Kenya Grace — just for you.` (the device name is
gone), and a speaker that refused the stream is said as
`Ugh, the speaker wouldn't take it — nothing's playing. Want me to try again?`
The provider's own `Playing …` sentence is dropped in that case rather than
contradicted, and the `WorkerOperationError` plus the stream URL stay in
`playback_handoff.error` for whoever debugs it.

A refusal is also not the end of the attempt. OpenDeck hands out
`/api/media/stream?url=<direct>`, and the Dot's failure was "could not fetch"
that proxy; so a refused hand-off is retried once with the direct stream the URL
wraps, and only a second refusal is reported. Which route worked, and why the
first did not, is recorded in `handoff.fallback` (`url`, `first_error`), so a
fallback is visible in the request log instead of looking like a normal start.
Tests: `test_a_refused_proxied_stream_is_retried_once_on_the_direct_url` and
`test_a_refused_stream_the_direct_url_cannot_fix_is_still_reported`.

**The model's own AI-speak is not covered, and that is measured too.** Four
model-style phrasings ("As an AI language model, I don't have access to that
information.", "I'm just an AI, so I can't know that.", ...) are reported
separately by the benchmark: **0 of 4 are rewritten, before and after.** The copy
layer rewrites the pipeline's own wording; if the model itself introduces AI
self-reference, it passes through untouched. Closing that gap is a copy-layer
change, not a measurement change — and until it is closed, this report does not
claim that every abstention sounds human, only that the pipeline-written ones do.

## 5. What is not claimed

* **No k=5 number.** The pre-registered k=5 confirmation did not complete on
  this host (three attempts, no result file), so the table above is a
  single-pass smoke comparison, not the repeated-sample confirmation.
* **No intervals on the Luna rows.** They are point estimates from one pass;
  the 25-case sets are small, and a one-case swing moves a percentage by 4
  points.
* **The pre-fix live transcript is unusable.** The HEAD worktree's chat model
  path raises `FileNotFoundError`, so every turn fell back to chat for an
  unrelated reason; the pre-fix state is pinned by the test runs instead.
* **D4 human review is still open.** The packet is built and unscored.
* **The music bench is a fixture-level bench**, not a live speaker test: it
  counts provider `play` calls across three polls of one finished track.
* **The abstention-voice bench scores wording, not comprehension.** Its corpus is
  the pipeline's own copy; it does not judge whether a conversational answer was
  helpful. The pre-fix corpus is read from the pre-fix source, because no pre-fix
  run recorded reply text for the model path.

## 6. Reproduce

```bash
# suites
cd /var/tmp/bench_before/nix_knowledge && /root/nix_knowledge/.venv/bin/python -m pytest -q tests
cd /root/nix_knowledge                  && /root/nix_knowledge/.venv/bin/python -m pytest -q tests
cd /var/tmp/bench_before/nix_core       && /root/nix_knowledge/.venv/bin/python -m pytest -q test_evidence_defect_regressions.py test_music_api.py
cd /root/nix_core                       && /root/nix_knowledge/.venv/bin/python -m pytest -q test_evidence_defect_regressions.py test_music_api.py

# routing and speed
/root/nix_knowledge/.venv/bin/python /var/tmp/bench_routing.py /var/tmp/bench_before
/root/nix_knowledge/.venv/bin/python /var/tmp/bench_routing.py /root
cd /var/tmp/bench_before/nix_core && /root/nix_knowledge/.venv/bin/python -m benchmarks.speed_benchmark --benchmark router --repeats 200 --warmup 20
cd /root/nix_core                 && /root/nix_knowledge/.venv/bin/python -m benchmarks.speed_benchmark --benchmark router --repeats 200 --warmup 20

# music hand-offs
cd /var/tmp/bench_before/nix_core && /root/nix_knowledge/.venv/bin/python -m pytest -q -s /var/tmp/bench_music_polls.py
cd /root/nix_core                 && /root/nix_knowledge/.venv/bin/python -m pytest -q -s /var/tmp/bench_music_polls.py

# gate
cd /root/nix_core && /root/nix_knowledge/.venv/bin/python -m benchmarks.reliability_gate --trials 400 --json /var/tmp/gate_after.json

# abstention voice (before / after)
cd /root/nix_core && /root/nix_knowledge/.venv/bin/python benchmarks/abstention_voice.py /var/tmp/bench_before
cd /root/nix_core && /root/nix_knowledge/.venv/bin/python benchmarks/abstention_voice.py /root
```

The gate has no pre-fix run: `benchmarks/reliability_gate.py` does not exist at
`HEAD`, so its baseline column is the harness's own frozen baseline arm, run on
this machine today.
