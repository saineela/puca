import json
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import console


FOLLOW_UP_METADATA_PROMPT = """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a user.
Write them from the user's point of view. Response must be a JSON object with a \"follow_ups\" key.
JSON format: { \"follow_ups\": [\"Question 1?\"] }
### Chat History:
<chat_history>USER: Hi ASSISTANT: Hello!</chat_history>"""
TITLE_METADATA_PROMPT = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.
Your response must consist solely of the JSON object.
JSON format: { \"title\": \"your concise title here\" }
### Chat History:
<chat_history>USER: Hi ASSISTANT: Hello!</chat_history>"""
TAGS_METADATA_PROMPT = """### Task:
Generate broad tags categorizing the main themes, along with specific subtopic tags.
Output JSON format: { \"tags\": [\"tag1\", \"tag2\"] }
### Chat History:
<chat_history>USER: Hi ASSISTANT: Hello!</chat_history>"""


@pytest.fixture
def openai_endpoint(monkeypatch):
    monkeypatch.setattr(console, "OPENAI_API_KEY", "")
    brain = FakeBrain()
    monkeypatch.setattr(console, "get_brain", lambda: brain)
    monkeypatch.setattr(console, "_assistant_models_status", lambda: {
        "active": "official-luna",
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", brain
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post_request(base_url, path, body, headers=None):
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode(),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=3) as response:
            status, content = response.status, response.read()
    except HTTPError as response:
        status, content = response.code, response.read()
    return status, json.loads(content.decode())


OPENAI_CHAT_PATH = "/v1/chat/completions"

FOLLOW_UP_METADATA_PROMPT_WITHOUT_HISTORY = """### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a user.
Write them from the user's point of view. Response must be a JSON object with a \"follow_ups\" key.
JSON format: { \"follow_ups\": [\"Question 1?\"] }"""
TITLE_METADATA_PROMPT_WITHOUT_HISTORY = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat.
Your entire response must consist solely of the JSON object.
JSON format: { \"title\": \"your concise title here\" }"""
TAGS_METADATA_PROMPT_WITHOUT_HISTORY = """### Task:
Generate 1-3 broad tags categorizing the main themes, along with 1-3 more specific subtopic tags.
Output JSON format: { \"tags\": [\"tag1\", \"tag2\"] }"""


def api_body(content="hello", **payload):
    return {
        **payload,
        "model": "official-luna",
        "messages": [{"role": "user", "content": content}],
    }


def test_conversation_groups_keep_api_dashboard_and_legacy_history_searchable():
    grouped = console._group_conversation_sessions([
        {
            "session_tag": "week-2026-09-21",
            "turns": [
                {
                    "id": 1,
                    "role": "user",
                    "content": "API question about lilacs",
                    "created_at": "2026-09-22T10:00:00-05:00",
                    "refs": json.dumps({
                        "location": "openai-api",
                        "conversation_id": "open-webui-42",
                    }),
                },
                {
                    "id": 2,
                    "role": "assistant",
                    "content": "Here is the answer.",
                    "created_at": "2026-09-22T10:00:02-05:00",
                    "refs": {
                        "location": "openai-api",
                        "conversation_id": "open-webui-42",
                    },
                },
                {
                    "id": 3,
                    "role": "user",
                    "content": "Dashboard conversation about orchids",
                    "created_at": "2026-09-23T11:00:00-05:00",
                    "refs": {
                        "location": "dashboard",
                        "conversation_id": "dashboard-17",
                    },
                },
                {
                    "id": 4,
                    "role": "assistant",
                    "content": "The dashboard reply.",
                    "created_at": "2026-09-23T11:00:02-05:00",
                    "refs": {
                        "location": "dashboard",
                        "conversation_id": "dashboard-17",
                    },
                },
                {
                    "id": 5,
                    "role": "user",
                    "content": "Older untagged conversation",
                    "created_at": "2026-09-23T12:00:00-05:00",
                    "refs": {},
                },
            ],
        },
    ])

    assert len(grouped) == 3
    api = next(item for item in grouped if item["source"] == "API")
    assert api["conversation_id"] == "open-webui-42"
    assert api["title"] == "API question about lilacs"
    assert [turn["role"] for turn in api["turns"]] == ["user", "assistant"]
    assert api["preview"] == "Here is the answer."

    dashboard = next(item for item in grouped if item["source"] == "Dashboard")
    assert dashboard["conversation_id"] == "dashboard-17"
    assert dashboard["title"] == "Dashboard conversation about orchids"

    legacy = next(item for item in grouped if item["source"] == "Conversation history")
    assert legacy["title"] == "Older untagged conversation"
    assert legacy["turn_count"] == 1


def test_openai_api_assigns_isolated_conversation_ids(openai_endpoint):
    base_url, brain = openai_endpoint

    def send_api_request(body, headers=None):
        status, result = post_request(base_url, OPENAI_CHAT_PATH, body, headers)
        assert status == 200
        return result

    first = send_api_request(api_body(user="open-webui-user-7"))
    second = send_api_request(api_body(user="open-webui-user-7"))
    threaded = send_api_request(api_body(
        conversation_id="open-webui-thread-42",
    ))
    threaded_again = send_api_request(api_body(
        conversation_id="open-webui-thread-42",
    ))
    metadata_thread = send_api_request(api_body(metadata={"chat_id": "metadata-chat-9"}))
    header_thread = send_api_request(
        api_body(conversation_id="body-thread"),
        headers={"X-Conversation-ID": "header-thread"},
    )

    assert first["choices"][0]["message"]["content"] == "hello reply"
    assert second["choices"][0]["message"]["content"] == "hello reply"
    assert threaded["choices"][0]["message"]["content"] == "hello reply"
    assert threaded_again["choices"][0]["message"]["content"] == "hello reply"
    assert [call["location"] for call in brain.calls] == ["openai-api"] * 6
    ids = [call["conversation_id"] for call in brain.calls]
    assert ids[0] != ids[1]  # OpenAI `user` does not merge separate chats.
    assert all(conversation_id.startswith("openai-api:") for conversation_id in ids)
    assert ids[2] == ids[3] == "openai-api:open-webui-thread-42"
    assert ids[4] == "openai-api:metadata-chat-9"
    assert ids[5] == "openai-api:header-thread"


@pytest.mark.parametrize(("kind", "prompt"), [
    ("follow_ups", FOLLOW_UP_METADATA_PROMPT),
    ("title", TITLE_METADATA_PROMPT),
    ("tags", TAGS_METADATA_PROMPT),
])
def test_openai_api_rejects_auxiliary_metadata_generation(openai_endpoint, kind, prompt):
    base_url, brain = openai_endpoint
    assert console._openai_api_token_guard(prompt) == kind

    status, result = post_request(
        base_url,
        OPENAI_CHAT_PATH,
        api_body(prompt, stream=(kind == "follow_ups")),
    )

    assert status == 400
    assert result["error"]["code"] == "api_token_guard_rejected"
    assert result["error"]["message"].startswith("API Token Guard rejected")
    assert brain.calls == []


@pytest.mark.parametrize(("kind", "prompt"), [
    ("follow_ups", FOLLOW_UP_METADATA_PROMPT_WITHOUT_HISTORY),
    ("title", TITLE_METADATA_PROMPT_WITHOUT_HISTORY),
    ("tags", TAGS_METADATA_PROMPT_WITHOUT_HISTORY),
])
def test_openai_api_token_guard_rejects_metadata_tasks_without_history(
    openai_endpoint, kind, prompt,
):
    base_url, brain = openai_endpoint
    assert console._openai_api_token_guard(prompt) == kind

    status, result = post_request(
        base_url,
        OPENAI_CHAT_PATH,
        api_body(prompt, conversation_id="weekly-session-2026-09-21"),
    )

    assert status == 400
    assert result["error"]["code"] == "api_token_guard_rejected"
    assert result["error"]["message"].startswith("API Token Guard rejected")
    assert brain.calls == []


def test_openai_api_strips_previously_embedded_metadata_prompt_and_reply(openai_endpoint):
    base_url, brain = openai_endpoint
    status, _result = post_request(base_url, OPENAI_CHAT_PATH, {
        "model": "official-luna",
        "conversation_id": "open-webui-thread-42",
        "messages": [
            {"role": "user", "content": FOLLOW_UP_METADATA_PROMPT},
            {"role": "assistant", "content": '{"follow_ups": ["What is next?"]}'},
            {"role": "user", "content": "nice to meet you"},
            {"role": "assistant", "content": "Nice to meet you too."},
            {"role": "user", "content": "Can you recommend a new phone?"},
        ],
    })

    assert status == 200
    assert len(brain.calls) == 1
    assert brain.calls[0]["text"] == "Can you recommend a new phone?"
    assert brain.calls[0]["session_context"] == [
        {"role": "user", "content": "nice to meet you"},
        {"role": "assistant", "content": "Nice to meet you too."},
    ]
    assert all(
        "follow-up questions" not in turn["content"].lower()
        and "follow_ups" not in turn["content"]
        for turn in brain.calls[0]["session_context"]
    )
    assert brain.calls[0]["conversation_id"] == "openai-api:open-webui-thread-42"


def test_metadata_recognition_only_applies_to_openai_api(openai_endpoint):
    base_url, brain = openai_endpoint
    assert console._openai_api_token_guard("Can you suggest follow-up questions about phones?") is None
    assert console._openai_api_token_guard(
        "We discussed JSON format for a novel title in chat history."
    ) is None

    status, result = post_request(base_url, "/api/send", {
        "text": FOLLOW_UP_METADATA_PROMPT,
        "location": "dashboard",
        "conversation_id": "dashboard-thread-1",
    })

    assert status == 200
    assert result["reply"] == "hello reply"
    assert len(brain.calls) == 1
    assert brain.calls[0]["location"] == "dashboard"
    assert brain.calls[0]["text"] == FOLLOW_UP_METADATA_PROMPT


def test_api_token_guard_does_not_reject_normal_calendar_or_history_questions():
    for text in (
        "What did you say the date was for tomorrow?",
        "What is on my schedule for tomorrow?",
        "Can you suggest follow-up questions about the Roman Empire?",
    ):
        assert console._openai_api_token_guard(text) is None


def test_api_conversation_history_is_supplied_by_its_own_messages(openai_endpoint):
    base_url, brain = openai_endpoint
    status, _result = post_request(base_url, OPENAI_CHAT_PATH, {
        "model": "official-luna",
        "conversation_id": "phone-chat-22",
        "messages": [
            {"role": "user", "content": "I asked about a phone"},
            {"role": "assistant", "content": "We were comparing phones."},
            {"role": "user", "content": "Which one has better battery life?"},
        ],
    })

    assert status == 200
    assert len(brain.calls) == 1
    assert brain.calls[0]["conversation_id"] == "openai-api:phone-chat-22"
    assert brain.calls[0]["session_context"] == [
        {"role": "user", "content": "I asked about a phone"},
        {"role": "assistant", "content": "We were comparing phones."},
    ]


def test_openai_auxiliary_system_prompt_is_rejected(openai_endpoint):
    base_url, brain = openai_endpoint
    status, result = post_request(base_url, OPENAI_CHAT_PATH, {
        "model": "official-luna",
        "messages": [
            {"role": "system", "content": TITLE_METADATA_PROMPT},
            {"role": "user", "content": "What is the issue here?"},
        ],
    })

    assert status == 400
    assert result["error"]["code"] == "api_token_guard_rejected"
    assert brain.calls == []


class FakeBrain:
    def __init__(self):
        self.calls = []

    def handle(self, **kwargs):
        self.calls.append(kwargs)
        return {"reply": "hello reply", "route": "chat", "details": {}}
