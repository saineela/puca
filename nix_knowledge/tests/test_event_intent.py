from nix_knowledge.event_intent import detect_event_intent
from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.needle import KnowledgeNeedle
from nix_knowledge.context import TemporalContext
from nix_knowledge.temporal import TemporalResolver
from nix_actions.engine import ActionsEngine


def test_reminder_interval_becomes_event_intent():
    intent = detect_event_intent("Remind me like every 2 days to take my medicine")
    assert intent.requires_event is True
    assert intent.kind == "reminder"
    assert intent.temporal_expression == "every 2 days"
    assert "take my medicine" in intent.title


def test_absolute_date_reminder_for_wording_becomes_event_intent():
    # Production failure (Oct 5): all three phrasings returned
    # no_function_selected because the intent gate had no absolute-date
    # time pattern and could not strip the "for/about" connectors.
    intent = detect_event_intent(
        "remind me for October 28th 6:30pm for zoo ticket booking"
    )
    assert intent.requires_event is True
    assert intent.kind == "reminder"
    assert intent.temporal_expression == "October 28th 6:30pm"
    assert intent.title == "zoo ticket booking"


def test_bare_reminder_noun_date_wording_becomes_event_intent():
    intent = detect_event_intent(
        "reminder for October 28th 6:30pm for zoo ticket booking"
    )
    assert intent.requires_event is True
    assert intent.temporal_expression == "October 28th 6:30pm"
    assert intent.title == "zoo ticket booking"


def test_that_is_a_reminder_framing_becomes_event_intent():
    intent = detect_event_intent(
        "that's a reminder for October 28th 6:30 p.m. about zoo ticket booking"
    )
    assert intent.requires_event is True
    assert intent.temporal_expression.startswith("October 28th 6:30")
    assert intent.title == "zoo ticket booking"


def test_at_time_reminder_wording_is_unchanged():
    """"tomorrow at 5pm" must resolve as one span: the full time goes into
    the temporal expression and the title loses the clock words.
    The regex change came out of a measured live failure: "tomorrow at 4pm"
    previously matched only "tomorrow", leaving "4pm" as temporal residue in
    the title, which the symbolic calendar validator then rejected entirely
    ("could not be validated symbolically; no event was created")."""
    intent = detect_event_intent("remind me to call mom tomorrow at 5pm")
    assert intent.requires_event is True
    assert intent.temporal_expression == "tomorrow at 5pm"
    assert intent.title == "call mom"


def test_reminder_with_address_and_from_now():
    intent = detect_event_intent(
        "hey casper, I have to take medicines every 2 days from now remind me"
    )
    assert intent.requires_event is True
    assert intent.temporal_expression == "every 2 days"
    assert "take medicines" in intent.title


def test_plain_repeating_fact_is_not_an_alert():
    intent = detect_event_intent("I take medicine every 2 days")
    assert intent.requires_event is False


def test_event_gate_creates_recurring_action_without_model(tmp_path):
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = {"create_calendar_event": needle._create_calendar_event}

    try:
        payload = needle._process_route("Remind me like every 2 days to take my medicine")
        result = payload["result"]
        assert result["operation"] == "CREATE"
        assert result["data"]["recurring"] is True
        assert result["data"]["recurrence"] == "interval_2_days"
        assert result["actions"]["ok"] is True
        assert result["actions"]["recurrence"] == "interval_2_days"
        assert len(actions.list_actions(status="pending")) == 1
    finally:
        actions.close()
        knowledge.close()


def test_absolute_date_reminder_creates_event_without_model(tmp_path):
    # Full production replay (Oct 5): the exact phrasing that reached no
    # function must now create the zoo-ticket reminder at the spoken time.
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = {"create_calendar_event": needle._create_calendar_event}

    try:
        payload = needle._process_route(
            "remind me for October 28th 6:30pm for zoo ticket booking"
        )
        result = payload["result"]
        assert result["operation"] == "CREATE"
        assert result["data"]["title"] == "zoo ticket booking"
        assert result["temporal"]["resolved_start"].startswith("2026-10-28T18:30")
        assert result["actions"]["ok"] is True
        assert len(actions.list_actions(status="pending")) == 1
    finally:
        actions.close()
        knowledge.close()


# ---------------------------------------------------------------------
# Past-tense cancellation news (Oct 5 production replay)
# ---------------------------------------------------------------------


def test_cancel_news_after_create_disarms_reminder(tmp_path):
    """Production replay: 'I have robotics practice tomorrow at 5pm'
    created an event + reminder; 'my robotics practice has been
    cancelled' used to fall through to chat and leave the reminder
    armed. It must cancel the stored event and remove the action."""
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = needle._build_tool_functions()

    try:
        created = needle._process_route("I have robotics practice tomorrow at 5pm")
        assert created["result"]["data"]["title"] == "robotics practice"
        assert len(actions.list_actions(status="pending")) == 1

        payload = needle._process_route("my robotics practice has been cancelled")
        result = payload["result"]
        assert result["operation"] == "CANCEL"
        assert result["result"]["ok"] is True
        assert result["result"]["data"]["status"] == "cancelled"
        assert result["actions"]["ok"] is True
        assert result["actions"]["cancelled_actions"] == 1
        assert len(actions.list_actions(status="pending")) == 0
    finally:
        actions.close()
        knowledge.close()


def test_cancel_news_variants_route_to_cancel(tmp_path):
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = needle._build_tool_functions()

    try:
        for create_text, cancel_text, title in [
            ("I have a dentist appointment friday at 2pm",
             "my dentist appointment got cancelled",
             "dentist appointment"),
            ("I have tsa meeting on monday",
             "my tsa meeting was cancelled",
             "tsa meeting"),
            ("I have tutoring at 6pm",
             "i cancelled my tutoring",
             "tutoring"),
        ]:
            created = needle._process_route(create_text)
            assert created["result"]["data"]["title"] == title, create_text

            payload = needle._process_route(cancel_text)
            result = payload["result"]
            assert result["operation"] == "CANCEL", cancel_text
            assert result["result"]["ok"] is True, cancel_text
            assert result["result"]["data"]["status"] == "cancelled"
    finally:
        actions.close()
        knowledge.close()


def test_ambiguous_cancel_asks_labeled_question(tmp_path):
    """Two live events with the same title: cancel must ask which one,
    and repeating a labeled candidate back must cancel exactly that one."""
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = needle._build_tool_functions()

    try:
        needle._process_route("I have a piano lesson monday at 4pm")
        needle._process_route("I have a piano lesson wednesday at 6pm")

        payload = needle._process_route("cancel my piano lesson")
        result = payload["result"]
        assert result["operation"] == "NEEDS_CLARIFICATION"
        assert result["status"] == "ambiguous_cancel"
        assert len(result["candidates"]) == 2
        # identical titles must be distinguished by time labels
        assert result["candidates"][0] != result["candidates"][1]
        assert "piano lesson" in result["question"]
        assert result["continuation_template"] == "cancel my {selected}"

        # no event harmed while asking
        pending_before = len(actions.list_actions(status="pending"))
        assert pending_before == 2

        # user picks candidate 2; Core completes 'cancel my <label>'
        completed = "cancel my " + result["candidates"][1]
        follow = needle._process_route(completed)
        follow_result = follow["result"]
        assert follow_result["operation"] == "CANCEL"
        assert follow_result["result"]["ok"] is True
        assert follow_result["result"]["data"]["status"] == "cancelled"
        assert len(actions.list_actions(status="pending")) == 1
    finally:
        actions.close()
        knowledge.close()


def test_cancel_unknown_event_says_not_found(tmp_path):
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = needle._build_tool_functions()

    try:
        payload = needle._process_route("cancel my llama grooming session")
        result = payload["result"]
        assert result["ok"] is False
        assert result["operation"] == "CANCEL"
        assert result["status"] == "not_found"
        assert "llama grooming session" in result["question"]
        # nothing was cancelled and nothing new stored
        assert len(actions.list_actions(status="pending")) == 0
    finally:
        actions.close()
        knowledge.close()


def _needle(tmp_path):
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = needle._build_tool_functions()
    return needle, knowledge, actions


def test_saying_the_same_event_twice_creates_one_event(tmp_path):
    """A repeated sentence is one appointment, not two rows.

    Measured live: "I have robotics practice tomorrow at 5pm" in two turns
    listed the same practice twice, because create never looked for an
    identical (title, start). A different start is still a new occurrence.
    """
    needle, knowledge, actions = _needle(tmp_path)
    try:
        first = needle._process_route("I have robotics practice tomorrow at 5pm")["result"]
        second = needle._process_route("I have robotics practice tomorrow at 5pm")["result"]

        assert first["operation"] == "CREATE" and second["operation"] == "CREATE"
        assert second.get("duplicate") is True
        assert second["record_id"] == first["record_id"]

        found = needle._find_calendar_events(window="tomorrow")
        robotics = [
            event for event in found["events"]
            if "robotics" in str(event.get("title") or "").lower()
        ]
        assert len(robotics) == 1, "the same practice must not appear twice"
    finally:
        actions.close()
        knowledge.close()


def test_the_same_title_at_another_time_is_a_new_occurrence(tmp_path):
    needle, knowledge, actions = _needle(tmp_path)
    try:
        needle._process_route("I have robotics practice tomorrow at 5pm")
        other = needle._process_route("I have robotics practice tomorrow at 7pm")["result"]

        assert other["operation"] == "CREATE"
        assert not other.get("duplicate")
    finally:
        actions.close()
        knowledge.close()


def test_a_reminder_without_a_time_asks_for_the_time(tmp_path):
    """A missing time is asked for, never guessed at.

    Measured live: "Remind me to call mom." was answered "I'll remind you to
    call your mom later." with nothing scheduled - a promise the system had
    not made good on.
    """
    needle, knowledge, actions = _needle(tmp_path)
    try:
        payload = needle._process_route("Remind me to call mom.")
        result = payload["result"]

        assert result["operation"] == "NEEDS_CLARIFICATION"
        assert result["question"].lower().startswith("when should i remind you")
        assert "call mom" in result["question"]
        # nothing was scheduled by asking the question
        assert len(actions.list_actions(status="pending")) == 0
    finally:
        actions.close()
        knowledge.close()


def test_a_reminder_with_a_time_is_still_scheduled(tmp_path):
    needle, knowledge, actions = _needle(tmp_path)
    try:
        payload = needle._process_route("Remind me tomorrow at 8 to call mom.")
        result = payload["result"]

        assert result.get("operation") == "CREATE"
        assert result.get("record_type") == "event"
        assert len(actions.list_actions(status="pending")) >= 1
    finally:
        actions.close()
        knowledge.close()
