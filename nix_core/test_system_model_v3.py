from __future__ import annotations

import io
import time as _clock
import wave
from datetime import time

import numpy as np
import pytest

import voice

from eval import context_composer, inference_metrics, memory_sets, plan_validation
from eval import acceptance, feature_flags, proactivity, privacy, reliability, routing_ledger, turn_taking
from routing_engine import CoreRoutingEngine


# --------------------------------------------------------------------- D3
def test_context_composer_reports_all_five_sections_and_never_evicts_policy():
    composed = context_composer.compose(
        policy="policy " * 40,
        memory=[context_composer.SectionItem("fact " * 20, key="m1")],
        records=[context_composer.SectionItem("record " * 20, key="r1")],
        history=[context_composer.SectionItem("history " * 20, key="h1")],
        tools=[context_composer.SectionItem("tool " * 20, key="t1")],
        question="what is my locker code",
        budgets={"policy": 1, "memory": 1, "records": 1, "history": 1, "tools": 1},
    )
    assert set(composed.sizes) == set(context_composer.SECTIONS)
    assert composed.sizes["policy"] > 0
    assert composed.prompt.startswith("policy")


def test_context_composer_prefers_current_fact_for_present_and_history_for_past():
    current = context_composer.SectionItem(
        "You live in Austin", current=True, key="current", relevance=1
    )
    old = context_composer.SectionItem(
        "You lived in Seattle", current=False, superseded=True, key="old",
        valid_from="2026-03-01", valid_until="2026-04-01", relevance=1,
    )
    present = context_composer.rank_items([old, current], "Where do I live?")
    assert present[0] is current
    historical = context_composer.rank_items([current, old], "Where was I living in March?")
    assert historical[0] is old


def test_context_composer_is_deterministic_for_equal_inputs():
    items = [
        context_composer.SectionItem("b", key="b"),
        context_composer.SectionItem("a", key="a"),
    ]
    first = context_composer.compose(policy="p", memory=items, question="anything")
    second = context_composer.compose(policy="p", memory=items, question="anything")
    assert first.prompt == second.prompt
    assert first.sizes == second.sizes


# --------------------------------------------------------------------- D4
def test_routing_decision_carries_a_bounded_ledger_row():
    decision = CoreRoutingEngine().decide("Hi")
    row = decision.ledger_entry(request="Hi")
    assert row["fingerprint"]
    assert "Hi" not in row["fingerprint"]
    assert row["counterfactual"] == "unmeasured"
    assert row["cost_estimate"] >= 0


def test_routing_ledger_reports_separate_denominators_and_undetermined_share():
    rows = [
        routing_ledger.RoutedOutcome(
            "1", "knowledge", "escalation_required", True,
            counterfactual="replayed", replayed_route="chat", counterfactual_correct=False,
        ),
        routing_ledger.RoutedOutcome(
            "2", "knowledge", "escalation_required", True,
            counterfactual="replayed", replayed_route="chat", counterfactual_correct=True,
        ),
        routing_ledger.RoutedOutcome(
            "3", "chat", "fast_path_sufficient", False, answer_quality=0.4
        ),
        routing_ledger.RoutedOutcome("4", "unknown", "undetermined", None),
    ]
    report = routing_ledger.ledger_report(rows)
    assert report["false_escalation"] == 0.5
    assert report["false_escalation_denominator"] == 2
    assert report["false_de_escalation"] == 1.0
    assert report["false_de_escalation_denominator"] == 1
    assert report["undetermined_share"] == 0.25
    assert report["answer_quality_mean"] == 0.4


# --------------------------------------------------------------------- D5
def _plan(action="color", skill="opendeck"):
    return [{
        "type": "skill_tool_call",
        "skill_id": skill,
        "tool": "device_control",
        "arguments": {"action": action, "brightness": 60},
    }]


def test_acceptance_sample_sizes_and_permitted_errors_match_the_proposal():
    """One rule, no drift: the interval bound, both rates, real numbers."""
    assert acceptance.required_n("false_tool") == 189
    assert acceptance.required_n("missed_tool") == 73

    assert acceptance.permitted_errors(150, "false_tool") == -1
    assert acceptance.permitted_errors(189, "false_tool") == 0
    assert acceptance.permitted_errors(300, "false_tool") == 1
    assert acceptance.permitted_errors(500, "false_tool") == 3

    assert acceptance.permitted_errors(100, "missed_tool") == 0
    assert acceptance.permitted_errors(150, "missed_tool") == 2
    assert acceptance.permitted_errors(250, "missed_tool") == 5
    assert acceptance.permitted_errors(500, "missed_tool") == 15

    # A rate that only looks small does not pass.
    assert acceptance.passes_rate(0, 150, "false_tool") is False
    assert acceptance.passes_rate(0, 189, "false_tool") is True
    report = acceptance.rate_report(3, 500, "false_tool")
    assert report["passes"] is True and report["sample_sufficient"] is True


def test_acceptance_comparison_and_rollback_are_one_rule():
    assert acceptance.decide_comparison(low=0.1, high=0.5, scenarios=20)["action"] == "promote"
    assert acceptance.decide_comparison(low=-0.3, high=0.2, scenarios=20)["action"] == "rollback_to_off"
    # Inconclusive is not a pass, and a thin sample cannot promote.
    assert acceptance.decide_comparison(low=-0.02, high=0.3, scenarios=20)["action"] == "hold_in_shadow"
    assert acceptance.decide_comparison(low=0.1, high=0.5, scenarios=4)["action"] == "hold_in_shadow"

    assert acceptance.pairs_needed(0.20, 0.10) == 69
    assert acceptance.pairs_needed(0.10, 0.05) == 78


def test_feature_gate_enforces_every_promotion_condition():
    """The approved plan's gate: safety, sample, interval, latency, vocabulary."""
    promote = acceptance.decide_feature(
        low=0.10,
        high=0.50,
        scenarios=20,
        safety_violations=0,
        latency_p95_ms=120.0,
        latency_budget_ms=200.0,
        internal_vocabulary=0,
    )
    assert promote["action"] == "promote"

    # A single unauthorized action vetoes even a large improvement.
    vetoed = acceptance.decide_feature(
        low=0.40,
        high=0.60,
        scenarios=100,
        safety_violations=1,
        latency_p95_ms=10.0,
        latency_budget_ms=200.0,
        internal_vocabulary=0,
    )
    assert vetoed["decision"] == "veto"
    assert vetoed["action"] == "rollback_to_off"

    # Internal vocabulary is a rollback, not a warning.
    vocabulary = acceptance.decide_feature(
        low=0.10,
        high=0.50,
        scenarios=20,
        safety_violations=0,
        latency_p95_ms=10.0,
        latency_budget_ms=200.0,
        internal_vocabulary=1,
    )
    assert vocabulary["action"] == "rollback_to_off"

    # The interval may be non-inferior but still inconclusive for promotion.
    non_inferior = acceptance.decide_feature(
        low=-0.02,
        high=0.30,
        scenarios=20,
        safety_violations=0,
        latency_p95_ms=10.0,
        latency_budget_ms=200.0,
        internal_vocabulary=0,
    )
    assert non_inferior["action"] == "hold_in_shadow"

    # A thin sample cannot promote, and latency over budget cannot promote.
    thin = acceptance.decide_feature(
        low=0.10,
        high=0.50,
        scenarios=19,
        safety_violations=0,
        latency_p95_ms=10.0,
        latency_budget_ms=200.0,
        internal_vocabulary=0,
    )
    assert thin["action"] == "hold_in_shadow"
    slow = acceptance.decide_feature(
        low=0.10,
        high=0.50,
        scenarios=20,
        safety_violations=0,
        latency_p95_ms=250.0,
        latency_budget_ms=200.0,
        internal_vocabulary=0,
    )
    assert slow["action"] == "hold_in_shadow"
    assert "latency" in slow["reason"]


def test_phase_flags_are_shadow_first_and_gate_promoted(monkeypatch):
    """The approved plan's rollout contract: off | shadow | on per phase."""
    for phase in feature_flags.PHASES:
        monkeypatch.delenv(feature_flags.env_name(phase), raising=False)
    assert feature_flags.mode("semantic_routing") == "off"
    assert feature_flags.is_shadow("semantic_routing") is False

    monkeypatch.setenv("NIX_SEMANTIC_ROUTING_MODE", "shadow")
    assert feature_flags.is_shadow("semantic_routing") is True
    assert feature_flags.mode("answer_abstention") == "off"

    # A phase that has not passed its gate stays exactly where it is.
    hold = acceptance.decide_feature(
        low=-0.02, high=0.30, scenarios=20, safety_violations=0,
        latency_p95_ms=10.0, latency_budget_ms=200.0, internal_vocabulary=0,
    )
    assert feature_flags.promoted_mode("semantic_routing", hold) == "shadow"

    pass_gate = acceptance.decide_feature(
        low=0.10, high=0.50, scenarios=20, safety_violations=0,
        latency_p95_ms=10.0, latency_budget_ms=200.0, internal_vocabulary=0,
    )
    assert feature_flags.promoted_mode("semantic_routing", pass_gate) == "on"
    veto = acceptance.decide_feature(
        low=0.10, high=0.50, scenarios=20, safety_violations=1,
        latency_p95_ms=10.0, latency_budget_ms=200.0, internal_vocabulary=0,
    )
    assert feature_flags.promoted_mode("semantic_routing", veto) == "off"

    monkeypatch.setenv("NIX_SEMANTIC_ROUTING_MODE", "sideways")
    try:
        feature_flags.mode("semantic_routing")
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid mode must be refused")


def test_plan_validation_has_five_independent_checks():
    validation = plan_validation.validate_plan(
        request="set the light to red",
        plan=_plan("red"),
        allowed_skill_ids={"opendeck"},
        argument_limits={"opendeck": {"brightness": (0, 100)}},
        confirmed=False,
        result={"ok": True, "verified": True},
    )
    assert set(validation.failures) == set(plan_validation.CHECKS)
    assert validation.ok


def test_plan_validation_catches_valid_but_unauthorised_and_unconfirmed_actions():
    validation = plan_validation.validate_plan(
        request="delete the reminder",
        plan=_plan("delete", skill="unknown-skill"),
        allowed_skill_ids={"opendeck"},
        confirmed=False,
        result=None,
    )
    assert not validation.ok
    assert validation.failures["permission"]
    assert validation.failures["confirmation"]
    assert validation.failures["outcome"]


def test_plan_validation_accepts_a_value_led_request_without_a_verb():
    """"brightness to 70" is a request; the verb requirement was too strict.

    The expanded D6 sample exposed this as a false negative: nine legitimate
    device plans were rejected because their wording had no verb. A plan whose
    own argument names appear in the request is intent-consistent.
    """
    validation = plan_validation.validate_plan(
        request="brightness to 70",
        plan=[{
            "type": "skill_tool_call",
            "skill_id": "simulated",
            "tool": "device_plan",
            "arguments": {"action": "brightness", "brightness": 70},
        }],
        allowed_skill_ids={"simulated"},
        argument_limits={"simulated": {"brightness": (0, 100)}},
        confirmed=True,
        result={"ok": True, "verified": True},
    )
    assert validation.failures["intent"] == []
    assert validation.ok

    # A plan that names nothing from the request still fails: the check keeps
    # its purpose.
    unrelated = plan_validation.validate_plan(
        request="tell me a joke",
        plan=[{
            "type": "skill_tool_call",
            "skill_id": "simulated",
            "tool": "device_plan",
            "arguments": {"action": "brightness", "brightness": 70},
        }],
        allowed_skill_ids={"simulated"},
        confirmed=True,
        result={"ok": True, "verified": True},
    )
    assert unrelated.failures["intent"]


def test_plan_validation_catches_malformed_envelopes():
    validation = plan_validation.validate_plan(
        request="turn on the light", plan=[{"type": "nope"}], confirmed=True
    )
    assert validation.failures["schema"]
    assert validation.failures["intent"]


# --------------------------------------------------------------------- D6
def _pairs(baseline=10, candidate=15, n=20, safety=""):
    return [
        reliability.Pair(
            scenario=f"s{i}",
            flow="reminder",
            baseline=i < baseline,
            candidate=i < candidate,
            candidate_safety=safety if i == 0 else "",
        )
        for i in range(n)
    ]


def test_reliability_gate_uses_paired_difference_and_reports_wilson():
    report = reliability.gate(_pairs(), iterations=2000)
    assert report["decision"] == "pass"
    flow = report["flows"]["reminder"]
    assert flow["difference"]["diff"] == 0.25
    assert flow["baseline_wilson"] != flow["candidate_wilson"]


def test_reliability_gate_is_undecided_below_k20():
    report = reliability.gate(_pairs(n=5), iterations=200)
    assert report["decision"] == "undecided"
    assert "20" in report["reason"]


def test_reliability_gate_vetoes_any_critical_safety_violation():
    report = reliability.gate(_pairs(safety="sent a message without confirmation"))
    assert report["decision"] == "veto"
    assert "sent a message" in report["reason"] or report["safety_violations"]


def test_scenario_gate_does_not_count_replays_as_independent_trials():
    # Four scenarios, twenty identical replays each: the same four outcomes.
    pairs = [
        reliability.Pair(
            scenario=f"s{i}",
            flow="device_plan",
            baseline=i < 2,
            candidate=True,
        )
        for i in range(4)
        for _ in range(20)
    ]
    replay_report = reliability.gate(pairs, iterations=200)
    assert replay_report["flows"]["device_plan"]["trials"] == 80

    scenario_report = reliability.scenario_gate(pairs, iterations=200)
    assert scenario_report["decision"] == "undecided"
    assert scenario_report["scenarios"] == 4
    assert scenario_report["flows"]["device_plan"]["scenarios"] == 4
    assert "not independent" in scenario_report["reason"]


def test_scenario_gate_reports_wins_losses_and_safety_separately():
    pairs = [
        reliability.Pair(f"s{i}", "reminder", baseline=i < 10, candidate=i < 15)
        for i in range(20)
    ]
    report = reliability.scenario_gate(pairs, iterations=500)
    flow = report["flows"]["reminder"]
    assert flow["mcnemar"]["candidate_only"] == 5
    assert flow["mcnemar"]["ties"] == 15
    assert flow["outcomes"]["candidate_safety_violations"] == 0
    assert flow["outcomes"]["baseline_incorrect_no_violation"] == 10


def test_scenario_gate_refuses_a_scenario_that_changed_between_replays():
    pairs = [
        reliability.Pair("s1", "reminder", baseline=True, candidate=True),
        reliability.Pair("s1", "reminder", baseline=False, candidate=True),
    ]
    try:
        reliability.scenario_gate(pairs)
    except ValueError as exc:
        assert "not deterministic" in str(exc)
    else:  # pragma: no cover - the raise is the contract
        raise AssertionError("a non-deterministic replay must be refused")


def test_pass_power_and_mcnemar():
    trials = [reliability.Trial(f"s{i}", "reminder", True) for i in range(5)]
    assert reliability.pass_power(trials, k=5) == 1.0
    assert reliability.pass_power(trials[:-1] + [reliability.Trial("s4", "reminder", False)], k=5) == 0.0
    pairs = _pairs(baseline=10, candidate=15)
    result = reliability.mcnemar_exact(pairs)
    assert result["candidate_only"] == 5
    assert result["baseline_only"] == 0


# --------------------------------------------------------------------- D7
def test_turn_taking_measures_endpoint_latency_and_barge_in():
    taker = turn_taking.TurnTaker(endpoint_ms=300)
    taker.feed(turn_taking.Frame(0, True))
    assert taker.feed(turn_taking.Frame(200, False))["endpoint"] is False
    ended = taker.feed(turn_taking.Frame(350, False))
    assert ended["endpoint"] is True
    assert ended["endpoint_ms"] == 350
    taker.assistant_speaking()
    assert taker.feed(turn_taking.Frame(400, True, assistant_audio=True))["state"] == "listening"
    summary = taker.summary()
    assert summary["barge_ins"] == 1
    assert summary["end_of_turn_p50_ms"] == 350


# --------------------------------------------------------------------- D8
def test_proactivity_is_off_by_default_with_no_side_effects():
    policy = proactivity.default_policy()
    result = policy.evaluate(tier="critical", at=time(12, 0))
    assert result["notify"] is False
    assert result["side_effects"] == []
    assert policy.registry == []
    assert policy.sent_today == 0


def test_proactivity_enforces_quiet_hours_and_budget_when_enabled():
    policy = proactivity.ProactivityPolicy(enabled=True, daily_budget=2)
    assert policy.evaluate(tier="routine", at=time(12, 0))["reason"] == "tier_not_allowed"
    assert policy.evaluate(tier="important", at=time(23, 0))["reason"] == "quiet_hours"
    assert policy.evaluate(tier="important", at=time(12, 0))["notify"] is True
    assert policy.evaluate(tier="important", at=time(12, 1))["notify"] is True
    assert policy.evaluate(tier="important", at=time(12, 2))["reason"] == "budget_exhausted"
    policy.sent_today = 0
    assert policy.evaluate(tier="critical", at=time(23, 0), urgent=True)["notify"] is True


# --------------------------------------------------------------------- D9
def test_privacy_redacts_the_published_suite_and_rehydrates_locally():
    redactor = privacy.Redactor()
    result = redactor.redact(
        "Mail sai@example.com or call 512-555-1212; locker 4471; "
        "I live at 12 Oak Street; student id S1234567.",
        names=["Sai"],
    )
    assert result.completed
    assert "sai@example.com" not in result.text
    assert "512-555-1212" not in result.text
    assert "12 Oak Street" not in result.text
    assert "[EMAIL]" in result.text
    assert "Sai" not in result.text
    assert redactor.rehydrate(result.text).startswith("Mail sai@example.com")


def test_privacy_boundary_blocks_non_local_when_redaction_cannot_complete():
    boundary = privacy.OutboundBoundary(redactor=privacy.Redactor(
        on_error=lambda _message: None
    ))
    # A non-string input makes the redactor fail; the boundary must not send.
    with pytest.raises(privacy.RedactionBlocked):
        boundary.prepare(url="https://api.example.com/v1/chat", text=None)  # type: ignore[arg-type]
    assert boundary.audit()["blocked"] == 1


def _wav(parts, rate: int = 16000) -> bytes:
    """A mono 16-bit clip from ``(seconds, amplitude)`` pieces."""
    pieces = []
    for seconds, amplitude in parts:
        count = int(rate * seconds)
        if amplitude:
            ticks = np.arange(count) / rate
            pieces.append((amplitude * np.sin(2 * np.pi * 220 * ticks) * 32767).astype("<i2"))
        else:
            pieces.append(np.zeros(count, dtype="<i2"))
    samples = np.concatenate(pieces) if pieces else np.zeros(0, dtype="<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_turn_taking_measures_vad_endpointing_and_trailing_silence_on_a_clip():
    profile = voice.turn_taking_profile(_wav([(0.2, 0.0), (0.5, 0.3), (0.4, 0.0)]))
    assert profile["speech_frames"] > 0
    assert profile["endpoint_reached"] is True
    assert profile["trailing_silence_ms"] == 400.0
    assert profile["end_of_turn_p50_ms"] == 300.0
    assert profile["barge_ins"] is None, "barge-in is unmeasured on a single upload"


def test_turn_taking_measures_a_clip_that_stops_mid_sentence():
    profile = voice.turn_taking_profile(_wav([(0.4, 0.3), (0.1, 0.0)]))
    assert profile["endpoint_reached"] is False, "100 ms of silence is not an end of turn"
    assert profile["trailing_silence_ms"] == 100.0


def test_privacy_boundary_treats_the_lan_as_local_and_keeps_payloads_clean():
    boundary = privacy.OutboundBoundary()
    assert boundary.prepare(
        url="http://127.0.0.1:8080/v1/chat", text="hello sai@example.com"
    ) == "hello sai@example.com"
    assert boundary.prepare(
        url="https://api.example.com/v1/chat", text="hello sai@example.com"
    ) == "hello [EMAIL]#1"
    assert boundary.audit()["ok"] is True
    assert privacy.retention_report()["stores"]["backup"] == 30


# --------------------------------------------------------------------- D10
def test_privacy_covers_every_outbound_path_including_retries_and_exception_text():
    boundary = privacy.OutboundBoundary()
    first = boundary.prepare(
        url="https://api.example.com/v1/chat", text="mail sai@example.com"
    )
    assert first == "mail [EMAIL]#1"
    # A retry of the identical text redacts to the identical payload.
    assert boundary.retry(
        url="https://api.example.com/v1/chat",
        text="mail sai@example.com",
        first_attempt=first,
    ) == first
    boundary.scrub("{\"event\":\"error\",\"detail\":\"call 512-555-1212\"}", path="telemetry")
    boundary.scrub("Traceback: sai@example.com", path="exception")
    audit = boundary.audit()
    assert audit["leaked"] == []
    assert set(audit["paths"]) == {"model", "retry", "telemetry", "exception"}


def test_privacy_refuses_a_retry_after_a_blocked_attempt():
    boundary = privacy.OutboundBoundary(
        redactor=privacy.Redactor(on_error=lambda _message: None)
    )
    with pytest.raises(privacy.RedactionBlocked):
        boundary.prepare(url="https://api.example.com/v1/chat", text=None)  # type: ignore[arg-type]
    with pytest.raises(privacy.RedactionBlocked):
        boundary.retry(
            url="https://api.example.com/v1/chat",
            text="mail sai@example.com",
            first_attempt="mail sai@example.com",
        )
    assert any(item["url"] == "https://api.example.com/v1/chat" for item in boundary.blocked)
    assert boundary.audit()["ok"] is False, "a refused endpoint is reported, not hidden"


def test_privacy_boundary_redacts_values_that_originated_in_memory_and_tools():
    """The boundary is about the bytes leaving, not where they were read from.

    A remembered account number and a tool result carrying a phone number are
    the two paths the review named; neither is a user prompt, and both must be
    redacted exactly like one.
    """
    boundary = privacy.OutboundBoundary()
    remembered = "account 123456789012"
    tool_result = '{"contact": "512-555-1212", "ok": true}'

    sent_remembered = boundary.prepare(
        url="https://api.example.com/v1/chat", text=remembered
    )
    sent_tool = boundary.prepare(
        url="https://api.example.com/v1/chat", text=tool_result, path="retry"
    )

    assert "123456789012" not in sent_remembered
    assert "512-555-1212" not in sent_tool
    assert boundary.audit()["ok"] is True


def test_privacy_boundary_exposes_a_rehydrated_reply_in_the_audit():
    """Rehydration is local; if a reply is sent again, the audit catches it.

    The redactor can put the real value back for display. That is safe only
    while the value stays on this machine; sending the rehydrated text to a
    non-local endpoint must be caught by ``audit`` even when the boundary was
    bypassed on the way out.
    """
    boundary = privacy.OutboundBoundary()
    redacted = boundary.prepare(
        url="https://api.example.com/v1/chat", text="mail sai@example.com"
    )
    restored = boundary.redactor.rehydrate(redacted)
    assert restored == "mail sai@example.com"

    # Simulate a bypass: a payload recorded as sent that still holds the value.
    boundary.sent_payloads.append(
        {"url": "https://api.example.com/v1/chat", "path": "model", "text": restored, "detected": []}
    )
    audit = boundary.audit()
    assert audit["ok"] is False
    assert "email" in audit["leaked"]


def test_inference_metrics_record_ttft_decode_and_load_separately():
    metrics = inference_metrics.CallMetrics(
        model="luna", backend="transformers", load_ms=120.0,
        ttft_ms=80.0, decode_ms=240.0, token_count=12, cache_hit=True,
        prefix_key="prefix:v1:abc",
    )
    assert metrics.decode_ms_per_token == 20.0
    inference_metrics.record(metrics)
    report = inference_metrics.summary()
    assert report["calls"] >= 1
    assert report["coverage"]["ttft"] == 1.0


def test_luna_ttft_is_stamped_at_the_first_decode_step_only():
    pytest.importorskip("torch")
    luna_model = pytest.importorskip("luna_model")
    stamp = luna_model._FirstStepStamp(_clock.perf_counter())
    assert stamp.ttft_ms is None
    stamp(None, None)
    first = stamp.ttft_ms
    assert first is not None and first >= 0
    _clock.sleep(0.01)
    stamp(None, None)
    assert stamp.ttft_ms == first, "later steps do not move time-to-first-token"


def test_prefix_cache_rejects_personal_memory_and_keys_on_immutable_fields():
    with pytest.raises(ValueError):
        inference_metrics.ImmutablePrefix(
            model="luna", tokenizer="llama", policy_version="v1",
            policy_text="MEMORY (from their private knowledge base): locker 4471",
        )
    prefix = inference_metrics.ImmutablePrefix(
        model="luna", tokenizer="llama", policy_version="v1",
        policy_text="You are a warm, honest companion.",
    )
    cache = inference_metrics.PrefixCache()
    assert cache.get(prefix) is None
    cache.put(prefix, {"ok": True})
    assert cache.get(prefix) == {"ok": True}
    assert cache.hit_rate == 0.5


# --------------------------------------------------------------------- D2
def test_memory_sets_have_the_three_frozen_shapes_and_fair_budgets():
    sets = memory_sets.load_sets("eval/data/memory_scenarios.json")
    assert set(sets) == {"development", "heldout", "adversarial"}
    assert len(sets["development"]) == 50
    assert len(sets["heldout"]) == 25
    assert len(sets["adversarial"]) == 25
    budget = 100
    transcript = memory_sets.transcript_baseline(
        [{"role": "user", "content": "word " * 500}], budget=budget
    )
    dump = memory_sets.filesystem_baseline(
        [{"id": 1, "data": {"x": "word " * 500}}], budget=budget
    )
    assert memory_sets.estimate_tokens(transcript) <= budget
    assert memory_sets.estimate_tokens(dump) <= budget


def test_memory_scoring_handles_stale_facts_and_honest_abstention():
    scenario = memory_sets.MemoryScenario(
        id="s", question="Where do I live?", expected_facts=("Austin",),
        expected_timeframe="current", superseded_facts=("Seattle",),
    )
    good = memory_sets.score_answer(scenario, "You live in Austin right now.")
    assert good.fact_recall == 1.0 and good.stale_fact is False
    stale = memory_sets.score_answer(scenario, "You live in Seattle.")
    assert stale.stale_fact is True
    unknown = memory_sets.MemoryScenario(
        id="u", question="passport?", expected_facts=(), unknown=True
    )
    assert memory_sets.score_answer(unknown, "I don't know.").abstention_honesty == 1.0
    # "I don't have that yet" is the pipeline's own abstention line; scoring it
    # as an invention is what made the fresh D2 run read 0.000 honesty.
    assert memory_sets.score_answer(
        unknown, "I don't have that yet. Tell me and I'll remember it."
    ).abstention_honesty == 1.0
    assert memory_sets.score_answer(unknown, "I haven't got that.").abstention_honesty == 1.0
    assert memory_sets.score_answer(unknown, "It is A1234567.").abstention_honesty == 0.0


def test_empty_knowledge_result_is_an_explicit_abstention():
    """The pipeline already abstains; the reply must stay that way."""
    from brain import format_knowledge_result

    payload = {
        "result": {
            "ok": True,
            "count": 0,
            "query": "passport number",
            "facts": [],
            "previous": [],
        }
    }
    reply = format_knowledge_result(payload)
    assert "don't have" in reply.casefold()
    assert "A1234567" not in reply


def test_memory_result_note_names_the_freeze_tag_and_prior_exposure():
    sets = memory_sets.load_sets("eval/data/memory_scenarios.json")
    note = memory_sets.frozen_manifest(
        sets, tag="frozen-2026-10-17", model="luna-v6-contextual-v1",
        hardware="local", prior_exposure="none",
    )
    assert note["tag"] == "frozen-2026-10-17"
    assert len(note["sha256"]) == 64
    assert note["prior_exposure"] == "none"


def test_wp2_confirmation_set_has_the_approved_unknown_distribution():
    """The owner's clarification: 25 unknowns, distributed 5 / 5 / 15."""
    sets = memory_sets.load_sets("eval/data/memory_scenarios_confirm.json")
    assert {name: len(rows) for name, rows in sets.items()} == {
        "development": 50, "heldout": 25, "adversarial": 25,
    }
    unknown = {name: sum(1 for row in rows if row.unknown) for name, rows in sets.items()}
    assert unknown == {"development": 5, "heldout": 5, "adversarial": 15}
    assert sum(unknown.values()) == 25
    # The ten adversarial near-misses have an answer key, so over-abstention
    # scores as a miss rather than as honesty.
    near_miss = [row for row in sets["adversarial"] if not row.unknown]
    assert len(near_miss) == 10
    assert all(row.expected_facts for row in near_miss)


def test_pooled_abstention_rule_uses_the_25_question_population():
    def rows(honest: int, n: int) -> list[dict]:
        return [{"abstention_honesty": 1.0 if i < honest else 0.0} for i in range(n)]

    passing = memory_sets.abstention_report(
        {"development": rows(5, 5), "heldout": rows(5, 5), "adversarial": rows(13, 15)}
    )
    assert passing["pooled"] == {
        "n": 25, "honest": 23, "rate": 0.92,
        "wilson_low": passing["pooled"]["wilson_low"],
        "wilson_high": passing["pooled"]["wilson_high"],
    }
    assert passing["passed"] is True
    assert passing["per_set"]["adversarial"] == {"n": 15, "honest": 13, "rate": 0.867}

    # 22/25 is below the pre-registered minimum even though the rate is 0.88.
    failing = memory_sets.abstention_report(
        {"development": rows(5, 5), "heldout": rows(5, 5), "adversarial": rows(12, 15)}
    )
    assert failing["pooled"]["honest"] == 22
    assert failing["passed"] is False


def test_memory_advantage_needs_an_interval_that_excludes_zero():
    same = memory_sets.paired_difference_interval([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert same["difference"] == 0.0 and same["low"] <= 0 <= same["high"]
    better = memory_sets.paired_difference_interval([1.0] * 8, [0.0] * 8)
    assert better["low"] > 0 and better["high"] == 1.0
    report = memory_sets.memory_advantage_report({
        "heldout": [
            {"fact_recall": 1.0, "transcript": {"fact_recall": 1.0},
             "filesystem": {"fact_recall": 0.0}},
            {"fact_recall": 1.0, "transcript": {"fact_recall": 1.0},
             "filesystem": {"fact_recall": 0.0}},
        ]
    })
    assert report["filesystem"]["demonstrated"] is True
    assert report["transcript"]["demonstrated"] is False
    assert report["demonstrated"] is False


def test_repeated_sampling_keeps_one_case_one_cluster():
    """The approved repeat protocol: k samples per case, and the case is the
    unit — five samples of one question must not count as five answers."""
    scenario = memory_sets.MemoryScenario(
        id="s", question="Where do I live?", expected_facts=("Austin",)
    )
    calls = {"n": 0}

    def answer(scenario, budget):
        calls["n"] += 1
        return "Austin" if calls["n"] == 1 else "Seattle"

    result = memory_sets.run_set([scenario], answer=answer, budget=100, samples=2)
    assert result["scenarios_count"] == 1
    assert result["samples_per_case"] == 2
    row = result["scenarios"][0]
    assert row["fact_recall"] == 0.5
    assert row["fact_recall_samples"] == [1.0, 0.0]
    assert row["fact_recall_stdev"] > 0
    assert result["accuracy"] == 0.5


def test_abstention_majority_rule_stays_case_level():
    unknown = memory_sets.MemoryScenario(
        id="u", question="What is my blood type?", expected_facts=(), unknown=True
    )
    replies = iter(["I don't know.", "I don't know.", "It is A+."])
    result = memory_sets.run_set(
        [unknown], answer=lambda scenario, budget: next(replies), budget=100, samples=3
    )
    row = result["scenarios"][0]
    # Two of three samples honest: the case is honest, and it is one case.
    assert row["abstention_honesty"] == 1.0
    report = memory_sets.abstention_report({"adversarial": result["scenarios"]})
    assert report["pooled"]["n"] == 1
    assert report["pooled"]["honest"] == 1

    split = memory_sets.run_set(
        [unknown],
        answer=lambda scenario, budget: "It is A+"
        if (split_calls.append(1) or len(split_calls) % 2 == 0)
        else "I don't know.",
        budget=100,
        samples=2,
    )
    # A tie (one honest of two) is not an honest case.
    assert split["scenarios"][0]["abstention_honesty"] == 0.0


split_calls: list[int] = []


def test_wp3_confirmation_set_is_new_and_covers_every_class():
    """A confirmation set that reuses held-out wording confirms nothing."""
    import json
    from pathlib import Path

    payload = json.loads(Path("eval/data/routing_confirm.json").read_text())
    keys = json.loads(Path("eval/data/routing_confirm_keys.json").read_text())["cases"]
    heldout = json.loads(Path("eval/data/routing_heldout.json").read_text())
    cases = payload["cases"]

    assert len(cases) == 46
    assert not ({case["text"] for case in cases} & {case["text"] for case in heldout["cases"]})
    assert {case["class"] for case in cases} == {
        "conversation", "world_question", "statement_none", "memory_read",
        "memory_history", "state_read", "memory_write", "calendar_write",
        "calendar_clarify", "safety_confirmation", "memory_delete", "ambiguous",
    }
    for case in cases:
        if case["label"] == "undetermined":
            assert case["id"] not in keys
            continue
        assert case["id"] in keys, case["id"]
        spec = keys[case["id"]]
        if spec["rubric"] == "memory":
            assert spec.get("seed") and spec.get("expected"), case["id"]
