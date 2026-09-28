import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_luna_pro_sft_v1 as builder  # noqa: E402


class TinyTokenizer:
    """Small deterministic tokenizer stand-in for data-pipeline unit tests."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        rendered = "<BOS>" + "".join(
            f"<{message['role']}>{message['content']}<EOT>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered) if tokenize else rendered


def test_normalize_preserves_system_and_rejects_invalid_role_sequence():
    row = builder.normalize_messages([
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": "What is evaporation?"},
        {"role": "assistant", "content": "Liquid water changes into vapor."},
    ])
    assert row == [
        {"role": "system", "content": "Answer in one sentence."},
        {"role": "user", "content": "What is evaporation?"},
        {"role": "assistant", "content": "Liquid water changes into vapor."},
    ]
    assert builder.normalize_messages([
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Follow-up"},
    ]) is None
    assert builder.normalize_messages([
        {"role": "user", "content": "Call a tool"},
        {"role": "assistant", "content": "Calling"},
        {"role": "tool", "content": "result"},
    ]) is None


def test_topical_chat_record_parser_handles_single_object_and_jsonl_encodings():
    single = {
        "id-1": {
            "config": "A",
            "content": [{"agent": "agent_1", "message": "Hello"}, {"agent": "agent_2", "message": "Hi"}],
            "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
        }
    }
    parsed = builder.parse_topical_chat_records(json.dumps(single))
    assert len(parsed) == 1
    assert parsed[0]["conversation_id"] == "id-1"

    record = {
        "article_url": "https://example.org/article",
        "config": "A",
        "content": [
            {"agent": "agent_1", "message": "Hello", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
            {"agent": "agent_2", "message": "Hi", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Excellent"},
    }
    actual_jsonl = "\n".join([
        json.dumps(["t_bde29ce2-4153-4056-9eb7-f4ad710505fe", record]),
        json.dumps(["t_6bdc9f24-1f54-4bf9-90ee-2dd1f45712fc", record]),
    ])
    parsed_lines = builder.parse_topical_chat_records(actual_jsonl)
    assert [row["conversation_id"] for row in parsed_lines] == [
        "t_bde29ce2-4153-4056-9eb7-f4ad710505fe",
        "t_6bdc9f24-1f54-4bf9-90ee-2dd1f45712fc",
    ]
    examples, stats = builder.topical_chat_examples(parsed_lines[0])
    assert examples == [[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]]
    assert stats["eligible_exchange_pairs"] == 1

    parsed_lines = builder.parse_topical_chat_records(
        json.dumps(single) + "\n" + json.dumps(single)
    )
    assert len(parsed_lines) == 2
    try:
        builder.parse_topical_chat_records(json.dumps(["conversation-id", {"not": "a record"}]))
    except ValueError as exc:
        assert "no conversation records" in str(exc)
    else:
        raise AssertionError("malformed alternating Topical-Chat data must fail closed")

    try:
        builder.parse_topical_chat_records("{not-json}")
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("invalid Topical-Chat data must fail closed")


def test_topical_chat_sensitive_screen_rejects_if_any_speaker_raises_it():
    raw = {
        "config": "A",
        "content": [
            {"agent": "agent_1", "message": "Have you heard the news?", "turn_rating": "Good"},
            {"agent": "agent_2", "message": "Yes, I heard about a celebrity's health diagnosis.", "turn_rating": "Excellent", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
    }
    assert builder.topical_chat_examples(raw)[0] == []


def test_topical_chat_requires_high_ratings_and_builds_valid_multi_turn_messages():
    raw = {
        "conversation_id": "fixture",
        "config": "A",
        "content": [
            {"agent": "agent_1", "message": "Astronomy has so many objects to observe.", "turn_rating": "Good"},
            {"agent": "agent_2", "message": "The Moon and planets are common targets for beginners.", "turn_rating": "Excellent", "knowledge_source": ["Personal Knowledge"]},
            {"agent": "agent_1", "message": "Some planets are easier to see than others.", "turn_rating": "Good"},
            {"agent": "agent_2", "message": "Venus and Jupiter are often bright enough to spot without a telescope.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Excellent"},
    }
    examples, stats = builder.topical_chat_examples(raw)
    assert examples == [[
        {"role": "user", "content": "Astronomy has so many objects to observe."},
        {"role": "assistant", "content": "The Moon and planets are common targets for beginners."},
        {"role": "user", "content": "Some planets are easier to see than others."},
        {"role": "assistant", "content": "Venus and Jupiter are often bright enough to spot without a telescope."},
    ]]
    assert len(examples[0]) == 4
    assert stats["eligible_exchange_pairs"] == 2

    bad_target = json.loads(json.dumps(raw))
    bad_target["content"][1]["turn_rating"] = "Poor"
    bad_examples, bad_stats = builder.topical_chat_examples(bad_target)
    assert bad_examples == []
    assert bad_stats["assistant_target_rating_rejected"] == 1
    assert bad_stats["pairs_after_rejected_target_dropped"] == 1

    unsupported_fact = json.loads(json.dumps(raw))
    unsupported_fact["content"][1]["knowledge_source"] = ["AS1"]
    source_examples, source_stats = builder.topical_chat_examples(unsupported_fact)
    assert source_examples == []
    assert source_stats["assistant_target_knowledge_source_rejected"] == 1
    assert source_stats["pairs_after_rejected_target_dropped"] == 1

    bad_conversation = json.loads(json.dumps(raw))
    bad_conversation["conversation_rating"]["agent_1"] = "Poor"
    assert builder.topical_chat_examples(bad_conversation) == (
        [], {"conversation_rating_rejected": 1}
    )
    malformed_end = json.loads(json.dumps(raw))
    malformed_end["content"].append({"agent": "agent_1", "message": "Unanswered question", "turn_rating": "Good"})
    malformed_examples, malformed_stats = builder.topical_chat_examples(malformed_end)
    assert malformed_examples == examples
    assert malformed_stats["incomplete_user_turn_dropped"] == 1

    sensitive = json.loads(json.dumps(raw))
    sensitive["content"][1]["message"] = "My cat needs a better diet."
    assert builder.topical_chat_examples(sensitive)[0] == []
    sensitive["content"][1]["message"] = "I heard someone used a surrogate."
    assert builder.topical_chat_examples(sensitive)[0] == []
    sensitive = json.loads(json.dumps(raw))
    sensitive["content"][0]["message"] = "Did you hear the celebrity's health diagnosis?"
    assert builder.topical_chat_examples(sensitive)[0] == []


def test_topical_chat_rejects_suffix_turns_after_first_bad_target_and_keeps_group_split_safe():
    raw = {
        "conversation_id": "same-human-conversation",
        "content": [
            {"agent": "agent_1", "message": "Describe comets."},
            {"agent": "agent_2", "message": "Comets are interesting to learn about.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
            {"agent": "agent_1", "message": "Which comet was discovered in 1705?"},
            {"agent": "agent_2", "message": "A famous comet appeared in a recent article.", "turn_rating": "Good", "knowledge_source": ["AS1"]},
            {"agent": "agent_1", "message": "What is a comet's tail made of?"},
            {"agent": "agent_2", "message": "A comet's tail contains dust and gas.", "turn_rating": "Excellent", "knowledge_source": ["Personal Knowledge"]},
            {"agent": "agent_1", "message": "Does a comet orbit the Sun?"},
            {"agent": "agent_2", "message": "Yes, this target has a poor partner rating.", "turn_rating": "Poor", "knowledge_source": ["Personal Knowledge"]},
            {"agent": "agent_1", "message": "Can a comet have more than one tail?"},
            {"agent": "agent_2", "message": "Yes, some comets can show multiple tails.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
    }
    examples, stats = builder.topical_chat_examples(raw)
    assert len(examples) == 1, stats
    assert [item[0]["content"] for item in examples] == ["Describe comets."]
    assert stats["assistant_target_knowledge_source_rejected"] == 1
    assert stats["assistant_target_rating_rejected"] == 0
    assert stats["rejected_target_exchange_pairs"] == 1
    assert stats["pairs_after_rejected_target_dropped"] == 3

    rows = [
        {
            "group_id": "topical_chat:same-human-conversation",
            "source": "topical_chat",
            "source_conversation_id": "same-human-conversation",
            "messages": example,
        }
        for example in examples
    ]
    independent = {
        "group_id": "topical_chat:independent-conversation",
        "source": "topical_chat",
        "source_conversation_id": "independent-conversation",
        "messages": [
            {"role": "user", "content": "What are nebulae?"},
            {"role": "assistant", "content": "Nebulae are clouds of gas and dust."},
        ],
    }
    rows.append({
        "group_id": "topical_chat:same-human-conversation",
        "source": "topical_chat",
        "source_conversation_id": "same-human-conversation",
        "messages": examples[0] + [
            {"role": "user", "content": "Do you enjoy meteor showers?"},
            {"role": "assistant", "content": "Yes, meteor showers are fascinating."},
        ],
    })
    builder.assign_group_ids(rows + [independent])
    assert rows[0]["group_id"] == rows[1]["group_id"]
    train, dev = builder.split_rows(rows + [independent], 0.25, 7)
    assert {row["group_id"] for row in train}.isdisjoint({row["group_id"] for row in dev})
    same_conversation_splits = [
        [row for row in part if row.get("source_conversation_id") == "same-human-conversation"]
        for part in (train, dev)
    ]
    assert sum(bool(part) for part in same_conversation_splits) == 1
    assert sum(map(len, same_conversation_splits)) == len(rows)


def test_topical_chat_filters_human_persona_claims_and_keeps_neutral_abstentions():
    cases = [
        ("Do you like watching the NBA?", "I watch a bit here and there."),
        ("Do you ever use Amazon?", "I do! Way more than I should."),
        ("Are you a Republican or Democrat?", "I am a Republican."),
        ("Have you ever seen the movie Snowden?", "Not yet."),
        ("What are you cheering for?", "I have to go with one of my alma maters."),
        ("Do you listen to radio?", "I used to have a favorite station."),
        ("What makes this surprising?", "I am surprised, too."),
        ("How many children do you have?", "As a father of two kids, it makes me proud."),
    ]
    for user, assistant in cases:
        assert builder.topical_chat_target_has_personal_claim(user, assistant)
        raw = {
            "content": [
                {"agent": "agent_1", "message": user},
                {"agent": "agent_2", "message": assistant, "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
            ],
            "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
        }
        examples, stats = builder.topical_chat_examples(raw)
        assert examples == []
        assert stats["assistant_target_personal_claim_rejected"] == 1
        assert stats["rejected_target_exchange_pairs"] == 1

    assert not builder.topical_chat_target_has_personal_claim(
        "Why did humans discover fire?", "I wonder why."
    )
    assert not builder.topical_chat_target_has_personal_claim(
        "Why was the result surprising?", "That's surprising because it was unexpected."
    )
    for answer in (
        "That is very surprising; I am surprised too.",
        "As a father of two kids, it made me sad.",
        "I have a lot more juice than that.",
        "Me too! I know that product line is reliable.",
    ):
        assert builder.topical_chat_target_has_personal_claim("A neutral question.", answer)
    assert not builder.topical_chat_target_has_personal_claim(
        "What if the answer is unknown?", "I don't know."
    )
    assert builder.topical_chat_target_has_personal_claim(
        "Do you like music?", "I don't know."
    )
    factual_query = {
        "content": [
            {"agent": "agent_1", "message": "What is a comet?"},
            {"agent": "agent_2", "message": "A comet is an icy body that orbits the Sun.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
    }
    assert len(builder.topical_chat_examples(factual_query)[0]) == 1


def test_topical_chat_rejects_assistant_questions_that_solicit_personal_experience():
    cases = [
        ("Hello there.", "Hello! So what is your favorite album?"),
        ("Good morning!", "Have you seen Inception?"),
        ("Hello", "Hi, do you use Facebook?"),
        ("Do you recognize this logo?", "Did you use Facebook in college?"),
        ("I have plenty of spare time.", "What are your hobbies?"),
        ("I have plenty of spare time.", "Would you be interested in the radio?"),
        ("The album was released decades ago.", "What is your favorite song?"),
        ("Let's talk about music.", "Do you like any of Kanye's music?"),
        ("Finland has a radio station in Latin.", "Are you interested in the radio?"),
        ("The season is almost over.", "Which teams do you expect to win the Super Bowl?"),
        ("I like books.", "Are you familiar with Crazy Rich Asians?"),
        ("I like books.", "Are you aware of the book Crazy Rich Asians?"),
        ("Hello there.", "I don't know. Do you like music?"),
    ]
    for user, assistant in cases:
        assert builder.topical_chat_target_has_personal_claim(user, assistant), (user, assistant)
        raw = {
            "content": [
                {"agent": "agent_1", "message": user},
                {"agent": "agent_2", "message": assistant, "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
            ],
            "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
        }
        examples, stats = builder.topical_chat_examples(raw)
        assert examples == [], (user, assistant)
        assert stats["assistant_target_personal_claim_rejected"] == 1


def test_topical_chat_rejects_unstable_personal_health_speculation():
    raw = {
        "content": [
            {"agent": "agent_1", "message": "I'm worried that something is wrong with a celebrity. They seem a bit unstable lately."},
            {"agent": "agent_2", "message": "They have always been an interesting character.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Good"},
    }
    examples, stats = builder.topical_chat_examples(raw)
    assert examples == []
    assert stats["sensitive_domain_conversation_rejected"] == 1


def test_topical_chat_requires_both_named_conversation_ratings():
    raw = {
        "content": [
            {"agent": "agent_1", "message": "The Moon is bright tonight."},
            {"agent": "agent_2", "message": "I enjoy looking at the Moon.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good"},
    }
    examples, stats = builder.topical_chat_examples(raw)
    assert examples == []
    assert stats["conversation_rating_rejected"] == 1


def test_topical_chat_drops_action_claim_conversations():
    raw = {
        "content": [
            {"agent": "agent_1", "message": "Did you arrange the event?"},
            {"agent": "agent_2", "message": "I scheduled it for tomorrow.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Excellent"},
    }
    examples, stats = builder.topical_chat_examples(raw)
    assert examples == []
    assert stats["action_claim_conversation_rejected"] == 1


def test_topical_chat_selects_only_personal_knowledge_target_turns():
    raw = {
        "content": [
            {"agent": "agent_1", "message": "Let's discuss comets."},
            {"agent": "agent_2", "message": "Comets orbit the Sun.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Excellent"},
    }
    for sources in ([], ["AS1"], ["Personal Knowledge", "FS1"], None):
        raw["content"][1]["knowledge_source"] = sources
        assert builder.topical_chat_examples(raw)[0] == []
    raw["content"][1]["knowledge_source"] = ["Personal Knowledge"]
    assert len(builder.topical_chat_examples(raw)[0]) == 1


def test_public_rows_emits_auditable_topical_segments_without_network_or_disk_writes(monkeypatch):
    raw = {
        "conversation_id": "segmentable-conversation",
        "config": "A",
        "content": [
            {"agent": "agent_1", "message": "What is stargazing?"},
            {"agent": "agent_2", "message": "Stargazing is a way to observe the night sky.", "turn_rating": "Good", "knowledge_source": ["Personal Knowledge"]},
            {"agent": "agent_1", "message": "Which planet is closest to the Sun?"},
            {"agent": "agent_2", "message": "Mercury is closest to the Sun.", "turn_rating": "Good", "knowledge_source": ["FS1"]},
            {"agent": "agent_1", "message": "What makes a telescope useful?"},
            {"agent": "agent_2", "message": "A telescope's wide field of view can make objects easier to locate.", "turn_rating": "Excellent", "knowledge_source": ["Personal Knowledge"]},
        ],
        "conversation_rating": {"agent_1": "Good", "agent_2": "Excellent"},
    }
    source = {
        "name": "topical_chat",
        "repo": "alexa/Topical-Chat",
        "loader": "jsonl_url",
        "url": "https://example.invalid/{revision}/train.jsonl",
        "config": None,
        "split": "train",
        "revision": "fee4a71dd8ce5b471dd46e1b7aab2acbd1b9e1be",
        "license": "fixture",
        "max_scan": 1,
        "shuffle_buffer": 1,
        "target_cap": 10,
        "category": "topical_human_conversation",
    }
    monkeypatch.setattr(builder, "load_topical_chat_records", lambda _: [raw])
    rows, audits = builder.public_rows(
        {"candidate_train_mix": [source]},
        benchmark_prompts=set(),
        tokenizer=TinyTokenizer(),
    )

    assert len(rows) == 1, audits
    assert rows[0]["messages"][-2]["content"] == "What is stargazing?"
    assert rows[0]["source_conversation_id"] == "segmentable-conversation"
    assert rows[0]["source_row_id"].endswith(":segment-0")
    assert rows[0]["group_id"] == "topical_chat:segmentable-conversation"
    filters = audits["topical_chat"]["filters"]
    assert audits["topical_chat"]["selected"] == 1
    assert filters["assistant_target_knowledge_source_rejected"] == 1
    assert filters.get("assistant_target_personal_claim_rejected", 0) == 0
    assert filters["eligible_exchange_pairs_selected"] == 1
    assert filters["rejected_target_exchange_pairs"] == 1
    assert filters["pairs_after_rejected_target_dropped"] == 1


def test_topical_chat_multiple_segments_from_same_conversation_stay_grouped():
    rows = [
        {"group_id": "topical_chat:conversation-42", "source_conversation_id": "conversation-42", "messages": [{"role": "user", "content": "first"}]},
        {"group_id": "topical_chat:conversation-42", "source_conversation_id": "conversation-42", "messages": [{"role": "user", "content": "second"}]},
        {"group_id": "topical_chat:conversation-99", "source_conversation_id": "conversation-99", "messages": [{"role": "user", "content": "third"}]},
    ]
    builder.assign_group_ids(rows)
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert rows[0]["group_id"] != rows[2]["group_id"]
    train, dev = builder.split_rows(rows, 0.34, 7)
    conversation_42_parts = [
        [row for row in part if row.get("source_conversation_id") == "conversation-42"]
        for part in (train, dev)
    ]
    assert sum(bool(part) for part in conversation_42_parts) == 1
    assert sum(map(len, conversation_42_parts)) == 2
    assert {row["group_id"] for row in train}.isdisjoint({row["group_id"] for row in dev})




def test_sharegpt_roles_normalize_and_source_system_prompt_is_preserved():
    messages = builder.normalize_messages([
        {"from": "system", "value": "Rewrite this more warmly."},
        {"from": "human", "value": "I cannot attend."},
        {"from": "gpt", "value": "I'm sorry, but I won't be able to make it."},
    ])
    assert messages is not None
    row = builder.make_row(messages, "smol-rewrite", "rewrite", "fixture")
    assert row is not None
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert "Rewrite this more warmly." in row["messages"][1]["content"]
    assert "I cannot attend." in row["messages"][1]["content"]
    assert row["messages"][2]["role"] == "assistant"


def test_constraint_validator_accepts_supported_and_satisfied_rules():
    assert builder.constraint_target_valid(
        "Your response should contain less than 100 words.",
        "A concise answer has fewer than one hundred words.",
    )
    assert builder.constraint_target_valid(
        "Write exactly 3 bullet points.",
        "- First\n- Second\n- Third",
    )
    assert builder.constraint_target_valid(
        "Include keywords: comet, aqueduct.",
        "A comet passed above the aqueduct.",
    )
    assert builder.constraint_target_valid(
        "Write exactly 3 words.",
        "One two three",
    )
    assert builder.constraint_target_valid(
        "Use lowercase only.",
        "all text is lowercase",
    )
    assert builder.constraint_target_valid(
        "Write without commas.",
        "A clean answer with no commas",
    )
    assert builder.constraint_target_valid(
        'Your response should contain at least 3 sentences. Include keywords "sustainability" and "ecosystem" in the response.',
        "Sustainability helps protect natural systems. A healthy ecosystem supports diverse life. Communities can help preserve it.",
    )
    assert builder.constraint_target_valid(
        "Include keywords: cedar, lilac, marigold, in your response.",
        "Cedar, lilac and marigold bloom in spring.",
    )
    assert builder.constraint_target_valid(
        "Use exactly 2 placeholders represented by square brackets, such as [name].",
        "Hello [name], welcome to [town].",
    )
    assert builder.constraint_target_valid(
        "Use bullet points such as: - item one. Write exactly 2 bullet points.",
        "- First point\n- Second point",
    )
    assert builder.constraint_target_valid(
        'In your response, the word "the" should appear at least 3 times.',
        "The plan is clear. The next step is ready. The final check is complete.",
    )
    assert builder.constraint_target_valid(
        "Use number placeholders for [age] and [year].",
        "At [age], the person began in [year].",
    )
    assert builder.constraint_target_valid(
        "At the end of your response, please explicitly add a postscript starting with P.S.",
        "A concise response.\n\nP.S. This is the postscript.",
    )


def test_constraint_validator_rejects_unparsed_person_perspective_rules():
    assert not builder.constraint_target_valid(
        "Summarize this text without using second or third person pronouns.",
        "Emily is reaching out and asks whether they can discuss it.",
    )


def test_constraint_validator_rejects_violations_and_unparsed_mixed_rules():
    assert not builder.constraint_target_valid(
        "Your response should contain less than 5 words.",
        "This answer contains more than five words in total.",
    )
    assert not builder.constraint_target_valid(
        "Please exactly follow a strange constraint about blue circles.",
        "The answer mentions blue circles.",
    )
    assert not builder.constraint_target_valid(
        "Use exactly 2 sentences and start with the word Hello.",
        "First sentence. Second sentence.",
    )
    assert not builder.constraint_target_valid(
        "Include keywords: comet, aqueduct.",
        "A comet crossed the field.",
    )
    assert not builder.constraint_target_valid(
        "Write at least 3 bullets.",
        "- first\n- second",
    )
    assert not builder.constraint_target_valid(
        "Use lowercase only.",
        "This Has Uppercase",
    )
    assert not builder.constraint_target_valid(
        "Your response should contain at least 3 sentences. The response must have 2 sections. Mark the beginning of each section with SECTION X.",
        "This is one sentence. The second sentence is here. The third completes it.",
    )
    assert not builder.constraint_target_valid(
        "At the end of your response, add this exact phrase [Is there anything else you need?]",
        "No exact ending here.",
    )
    assert not builder.constraint_target_valid(
        "Your entire response should be in English, and in all lowercase letters. no capital letters are allowed.",
        "la respuesta es breve",
    )
    assert not builder.constraint_target_valid(
        "Write exactly 2 words and use exactly 3 placeholders.",
        "One [name] two",
    )
    assert not builder.constraint_target_valid(
        "Include keywords: cedar, lilac, marigold, and keep the answer under 10 words.",
        "Cedar, lilac and marigold bloom in spring.",
    )
    assert not builder.constraint_target_valid(
        "Write exactly 2 sentences as JSON.",
        '{"answer": "First. Second."}',
    )


def test_action_claim_guard_requires_a_positive_system_result():
    pending = [
        {"role": "system", "content": builder.DEFAULT_SYSTEM},
        {"role": "user", "content": "Set a reminder."},
        {"role": "assistant", "content": "I can't claim it was scheduled until Nix Actions confirms."},
    ]
    assert not builder.assistant_claims_action(pending[-1]["content"])
    assert not builder.has_unsupported_action_claim(pending)

    unsupported = [
        {"role": "system", "content": builder.DEFAULT_SYSTEM},
        {"role": "user", "content": "Set a reminder."},
        {"role": "assistant", "content": "I scheduled it for tomorrow."},
    ]
    assert builder.assistant_claims_action(unsupported[-1]["content"])
    assert builder.has_unsupported_action_claim(unsupported)

    successful = [
        {"role": "system", "content": builder.DEFAULT_SYSTEM + "\nSIMULATED TEST ACTION RESULT: Nix Actions successfully created the reminder."},
        {"role": "user", "content": "Did it go through?"},
        {"role": "assistant", "content": "The reminder was scheduled."},
    ]
    assert builder.is_positive_action_result(successful[:2])
    assert not builder.has_unsupported_action_claim(successful)

    failed = [
        {"role": "system", "content": builder.DEFAULT_SYSTEM + "\nSIMULATED TEST ACTION RESULT: Nix Actions failed; no reminder was created."},
        {"role": "user", "content": "Did it go through?"},
        {"role": "assistant", "content": "No, the reminder wasn't created."},
    ]
    assert not builder.is_positive_action_result(failed[:2])
    assert not builder.has_unsupported_action_claim(failed)
    unrelated_result = [
        {"role": "system", "content": builder.DEFAULT_SYSTEM + "\nSIMULATED TEST ACTION RESULT: Nix Actions failed to create one reminder; separately, Nix Actions successfully created another."},
        {"role": "user", "content": "Did the failed reminder go through?"},
        {"role": "assistant", "content": "The reminder was scheduled."},
    ]
    assert not builder.is_positive_action_result(unrelated_result[:2])
    assert builder.has_unsupported_action_claim(unrelated_result)

    mismatched_object = [
        {"role": "system", "content": builder.DEFAULT_SYSTEM + "\nSIMULATED TEST ACTION RESULT: Nix Actions successfully created the email."},
        {"role": "user", "content": "Set a reminder to review the garden."},
        {"role": "assistant", "content": "The reminder was scheduled."},
    ]
    assert builder.assistant_claims_action("I saved your reminder.")
    assert builder.has_unsupported_action_claim(mismatched_object)
    social_preference = [
        {"role": "user", "content": "Remember that I prefer short answers."},
        {"role": "assistant", "content": "Your preference was saved for this conversation."},
    ]
    assert not builder.has_unsupported_action_claim(social_preference)


def test_project_curriculum_is_fictional_and_balances_action_lifecycle():
    rows = builder.project_rows()
    categories = {row["category"] for row in rows}
    assert {"action_pending", "action_success", "action_failure"} <= categories
    assert all(row["messages"][0]["role"] == "system" for row in rows)
    for category in {"memory_grounded", "irrelevant_context", "state_supersession", "action_success", "action_failure"}:
        row = next(item for item in rows if item["category"] == category)
        assert "SIMULATED TEST" in " ".join(message["content"] for message in row["messages"])
    pending = next(row for row in rows if row["category"] == "action_pending")
    assert not builder.has_unsupported_action_claim(pending["messages"])
    assert len(rows) == len(builder.PROJECT_TRAJECTORIES)


def test_repetition_filter_checks_each_message_and_only_long_dominant_repeats():
    assert not builder._has_repetitive_messages([
        {"role": "user", "content": "Please repeat this wording in your answer: choice please enter your name."},
        {"role": "assistant", "content": "The choice is ready. Please enter your name below."},
    ])
    assert not builder._has_repetitive_ngram(
        "Tokyo cafe planning should address cafe location, cafe design, and cafe operations."
    )
    assert builder._has_repetitive_ngram(" ".join(["repeat these eight ordinary words"] * 8))


def test_underresponsive_and_content_anchor_filters_are_conservative():
    assert builder._underresponsive_to_open_request([
        {"role": "user", "content": "What is the difference between = and == in Python?"},
        {"role": "assistant", "content": "Yes."},
    ])
    assert not builder._underresponsive_to_open_request([
        {"role": "user", "content": "Is Python interpreted?"},
        {"role": "assistant", "content": "Yes."},
    ])
    assert not builder._underresponsive_to_open_request([
        {"role": "user", "content": "What is the difference between = and == in Python?"},
        {"role": "assistant", "content": "= assigns a value; == compares two values for equality."},
    ])
    assert builder._has_no_content_anchors([
        {"role": "user", "content": "Explain the prime factorization of these four numbers using standard notation."},
        {"role": "assistant", "content": " ".join(["photosynthesis chlorophyll sunlight carbon oxygen leaves"] * 8)},
    ])
    assert not builder._has_no_content_anchors([
        {"role": "user", "content": "What is statistical modeling and how does it enable data analysis?"},
        {"role": "assistant", "content": "Statistical modeling uses data analysis to estimate relationships between variables and assess uncertainty."},
    ])
    assert not builder._has_no_content_anchors([
        {"role": "user", "content": "What is statistical modeling?"},
        {"role": "assistant", "content": "A concise definition."},
    ])


def test_chat_template_encoding_accepts_transformers_batch_encoding():
    class BatchEncodingLike(dict):
        pass

    class BatchTokenizer(TinyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return BatchEncodingLike(input_ids=[1, 2, 3])

    row = {"messages": [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}
    assert builder.tokenize_row(row, BatchTokenizer()) == (True, 3, 1)


def test_everyday_fixed_greeting_pair_is_removed_without_dropping_system_prompt():
    messages = [
        {"role": "system", "content": "source prompt"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "Why does rain fall?"},
        {"role": "assistant", "content": "Rain falls when water droplets become heavy."},
    ]
    result, removed = builder.strip_standard_everyday_greeting(messages)
    assert removed
    assert result[0] == messages[0]
    assert result[1:] == messages[3:]
    assert builder.normalize_messages(result) is not None


def test_tokenizer_audit_checks_message_and_total_sequence_bounds():
    tokenizer = TinyTokenizer()
    row = {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    fits, total, message_max = builder.tokenize_row(row, tokenizer, max_sequence_tokens=1000, max_message_tokens=100)
    assert fits and total > 0 and message_max == len("question")
    fits, _, _ = builder.tokenize_row(row, tokenizer, max_sequence_tokens=10, max_message_tokens=100)
    assert not fits
    fits, _, message_max = builder.tokenize_row(row, tokenizer, max_sequence_tokens=1000, max_message_tokens=3)
    assert not fits and message_max == len("question")


def test_source_category_and_reliability_screens_reject_known_failure_modes():
    magpie = next(
        item for item in json.loads(builder.MANIFEST_PATH.read_text(encoding="utf-8"))["candidate_train_mix"]
        if item["name"] == "smol_magpie_ultra"
    )
    assert builder.source_category_allowed(magpie, {"category": "coding"})
    assert not builder.source_category_allowed(magpie, {"category": "reasoning"})
    assert builder._EVERYDAY_VOLATILE.search("I am looking for a taxi service near me.")
    assert builder._EVERYDAY_VOLATILE.search("Do you have any coupons I can use?")
    assert builder._EVERYDAY_VOLATILE.search("What amusement parks are near me?")
    assert builder._EVERYDAY_VOLATILE.search("Is a storm coming tonight?")
    assert builder._HIGH_STAKES.search("A 12-year-old patient has abnormal hormone levels.")
    assert builder._TOPICAL_SENSITIVE_DOMAIN.search("My cat needs a different diet.")
    assert builder._TOPICAL_SENSITIVE_DOMAIN.search("I heard someone used a surrogate.")
    assert builder._SANITY_SENSITIVE_ROLEPLAY.search("The EMF meter detects a paranormal presence.")
    assert builder._REVIEW_SENSITIVE_DOMAIN.search("The latest study shows a 10% increase in ice loss.")


def test_high_stakes_and_volatile_claim_screens_cover_known_failure_modes():
    assert builder._HIGH_STAKES.search(
        "I cut my hand while cooking; rinse the wound and consider whether it needs stitches."
    )
    assert builder._HIGH_STAKES.search(
        "Ask a doctor before changing your medication dose."
    )
    assert builder._HIGH_STAKES.search(
        "A lawyer can advise you about filing a lawsuit."
    )
    assert builder._TIME_SENSITIVE.search(
        "The most popular song on Spotify this week is a new release."
    )
    assert builder._TIME_SENSITIVE.search(
        "This series is available on Netflix right now."
    )
    assert builder._TIME_SENSITIVE.search(
        "The top-selling game this year is still changing in the charts."
    )
    assert builder._TIME_SENSITIVE.search(
        "The series is streaming on Netflix."
    )
    assert builder._TIME_SENSITIVE.search(
        "People keep asking whether the series is on Netflix."
    )
    assert builder._EVERYDAY_VOLATILE.search(
        "Can you recommend an affordable place to stay in Budapest?"
    )
    assert builder._EVERYDAY_VOLATILE.search(
        "What are some things to do in Porto?"
    )


def test_exact_eval_prompt_set_and_near_match_audit():
    prompts = builder.build_benchmark_prompt_set(
        ["Shared benchmark prompt"],
        ["Fresh Luna probe"],
        ["Please write exactly 3 bullets about a comet and explain why it matters."],
    )
    assert "shared benchmark prompt" in prompts
    assert "fresh luna probe" in prompts
    assert "please write exactly 3 bullets about a comet and explain why it matters" in prompts
    assert builder._benchmark_overlap(
        {"messages": [{"role": "user", "content": "Please write exactly 3 bullets about a comet and explain why it matters!"}]},
        prompts,
    )
    assert builder._benchmark_overlap(
        {"source_user_turns": ["Please write exactly 3 bullets about a comet and explain why it matters!"], "messages": [{"role": "user", "content": "Task instructions: hidden source prompt"}]},
        prompts,
    )
    rows = [{"source": "test", "messages": [{"role": "user", "content": "Write exactly 3 bullets about a comet and explain its importance."}]}]
    near = builder.benchmark_near_matches(rows, prompts, report_threshold=0.3, block_threshold=0.5)
    assert near["near_overlap_count"] == 1
    assert near["blocking_near_overlap_count"] == 1


def test_group_aware_split_keeps_each_trajectory_wholly_in_one_split():
    rows = [
        {"group_id": "a", "messages": [{"role": "user", "content": "a"}]},
        {"group_id": "a", "messages": [{"role": "user", "content": "a variant"}]},
        {"group_id": "b", "messages": [{"role": "user", "content": "b"}]},
        {"group_id": "c", "messages": [{"role": "user", "content": "c"}]},
        {"group_id": "d", "messages": [{"role": "user", "content": "d"}]},
    ]
    train, dev = builder.split_rows(rows, 0.25, 11)
    train_groups = {row["group_id"] for row in train}
    dev_groups = {row["group_id"] for row in dev}
    assert train_groups.isdisjoint(dev_groups)
    assert train_groups | dev_groups == {"a", "b", "c", "d"}


def test_authored_template_and_near_duplicate_rows_group_before_split():
    rows = [
        {"group_id": "project:memory_pair", "source": "project_authored", "messages": [{"role": "user", "content": "How is Sela?"}]},
        {"group_id": "project:memory_pair", "source": "project_authored", "messages": [{"role": "user", "content": "How is Sela?"}, {"role": "assistant", "content": "No update."}]},
        {"group_id": "conversation:x", "source": "public", "messages": [{"role": "user", "content": "I have a red bicycle and ride it around the neighborhood every weekend with my cousin."}, {"role": "assistant", "content": "I have a red bicycle and ride it around the neighborhood every weekend with my cousin."}]},
        {"group_id": "conversation:y", "source": "public", "messages": [{"role": "user", "content": "I have a red bicycle and ride it around the neighborhood each weekend with my cousin."}, {"role": "assistant", "content": "I have a red bicycle and ride it around the neighborhood every weekend with my cousin."}]},
        {"group_id": "conversation:z", "source": "public", "messages": [{"role": "user", "content": "A completely unrelated note about painting clouds with watercolor."}, {"role": "assistant", "content": "Watercolor clouds can use a soft blend."}]},
    ]
    grouping = builder.assign_group_ids(rows)
    assert grouping["groups"] < len(rows)
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert rows[2]["group_id"] == rows[3]["group_id"]
    train, dev = builder.split_rows(rows, 0.25, 7)
    assert {row["group_id"] for row in train}.isdisjoint({row["group_id"] for row in dev})


def test_manifest_requires_llama_prefixed_distribution_name_and_consistent_caps():
    manifest = json.loads(builder.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["archive_status"] == "retired_non_actionable"
    assert "do not execute" in manifest["retirement_directive"]
    assert "initial_training_hypothesis_archived_not_authorized" in manifest
    builder.validate_manifest(manifest)
    topical = next(item for item in manifest["candidate_train_mix"] if item["name"] == "topical_chat")
    bad_topical = dict(topical)
    bad_topical["url"] = "https://example.org/latest.jsonl"
    assert not bad_topical["url"].endswith("/train.jsonl")
    invalid = dict(manifest)
    invalid["distribution_model_name"] = "Luna Pro"
    try:
        builder.validate_manifest(invalid)
    except ValueError as exc:
        assert "start with 'Llama'" in str(exc) or "must match" in str(exc)
    else:
        raise AssertionError("non-Llama release identifier should be rejected")


def test_ifeval_is_loaded_only_as_a_benchmark_and_text_normalization_is_stable():
    assert "google/IFEval" not in str(builder.SCRIPT_DIR)
    assert builder.normalize_text(" Write 3 bullets! ") == "write 3 bullets"
    prompts = builder.build_benchmark_prompt_set([], [], ["Keep this prompt for evaluation."])
    assert prompts == {"keep this prompt for evaluation"}
