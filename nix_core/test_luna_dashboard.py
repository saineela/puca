import json
import sqlite3
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import pytest

import console


PRO_ID = "luna-pro-v1-topical-v8-384-retry2"
V6_ID = "luna-v6-contextual-v1"


@pytest.fixture
def luna_console(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "OPENAI_API_KEY", "")
    monkeypatch.setenv("NIX_PROFILE_PATH", str(tmp_path / "profile.json"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request_json(base_url, path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        response = urlopen(request, timeout=3)
    except Exception as exc:
        response = exc
    status = getattr(response, "status", getattr(response, "code", None))
    return status, json.loads(response.read().decode())


def post_json(base_url, path, body):
    return request_json(base_url, path, method="POST", body=body)


def test_retired_luna_research_apis_are_closed(luna_console):
    status, result = request_json(luna_console, "/api/luna/model")
    assert status == 410
    assert result["code"] == "model_retired"

    status, result = post_json(luna_console, "/api/luna/model", {"model": PRO_ID})
    assert status == 410
    assert result["code"] == "model_retired"

    status, result = post_json(luna_console, "/api/luna/chat", {"text": "Hello"})
    assert status == 410
    assert result["code"] == "model_retired"


def test_retired_pro_id_is_rejected_from_official_model_apis(monkeypatch, luna_console):
    monkeypatch.setattr(console, "CASPER_BACKEND", "transformers")
    status, result = post_json(luna_console, "/api/model", {"model": PRO_ID})
    assert status == 410
    assert result["code"] == "model_retired"


@pytest.mark.parametrize(
    "path",
    [
        "/", "/home", "/models", "/conversations", "/memories", "/people",
        "/events", "/devices", "/api", "/docs", "/settings", "/skills",
        "/release-notes", "/release-notes/", "/workspace/skills/", "/nix/conversations",
    ],
)
def test_dashboard_pages_are_directly_routeable(luna_console, path):
    with urlopen(f"{luna_console}{path}", timeout=3) as response:
        markup = response.read().decode()
        assert response.status == 200
        assert "<title>NIX PUCA</title>" in markup
        assert "https://res.cloudinary.com/dh5uxc6ql/image/upload/v1790917615/93d18a47-5f3a-46e1-ad71-705b2680442f_anp8f1.png" in markup
        assert 'id="skillsRepositoryForm"' in markup
        assert "appBasePath" in markup
        if path.rstrip("/") == "/release-notes":
            assert 'id="release-notes"' in markup
        if path.rstrip("/") == "/devices":
            assert markup.count('id="devices"') == 1
            assert 'id="devicesList"' in markup
            assert "function renderDevices(devices)" in markup
            assert "function loadDevices()" in markup
            assert "not visual confirmation of the physical LEDs" in markup or "cannot confirm what the physical LEDs look like" in markup
            assert ".device-ring" in markup


@pytest.mark.parametrize("path", ["/api/not-a-dashboard-page", "/v1/not-a-dashboard-page"])
def test_unknown_api_paths_are_not_served_as_dashboard_pages(luna_console, path):
    status, result = request_json(luna_console, path)
    assert status == 404
    assert result == {"ok": False, "error": "not found"}


def test_skills_dashboard_endpoints_respond(monkeypatch, tmp_path, luna_console):
    monkeypatch.setattr(console, "SKILLS_STATE_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setattr(console, "SKILLS_INSTALL_DIR", str(tmp_path / "installed"))

    status, result = request_json(luna_console, "/api/skills")
    assert status == 200
    assert result["ok"] is True
    assert any(skill["id"] == "web-search" for skill in result["skills"])

    status, result = post_json(
        luna_console,
        "/api/skills",
        {"action": "add_repository", "repository_url": "http://example.test/skills"},
    )
    assert status == 400
    assert result["ok"] is False
    assert "HTTPS repository URL" in result["error"]


def test_model_research_archive_is_directly_routeable(luna_console):
    with urlopen(f"{luna_console}/NIX-Modeldev/", timeout=3) as response:
        markup = response.read().decode()
        assert response.status == 200
        assert "Model lineage, training records & citations" in markup


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/send"])
def test_retired_pro_id_is_rejected_from_chat_routes(luna_console, path):
    body = {"model": PRO_ID, "text": "hello"}
    if path == "/v1/chat/completions":
        body["messages"] = [{"role": "user", "content": "Hello"}]
    status, result = post_json(luna_console, path, body)
    assert status == 410
    error = result.get("error")
    code = error.get("code") if isinstance(error, dict) else result.get("code")
    assert code == "model_retired"


def test_official_model_status_reports_active_identity_from_registry(monkeypatch, luna_console):
    from luna_runtime import LUNA_V6_MODEL_ID

    monkeypatch.setattr(console, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(console, "_assistant_models_status", lambda: {
        "active": LUNA_V6_MODEL_ID,
        "assistant_name": "Luna",
        "backend": "transformers",
        "models": [{"id": LUNA_V6_MODEL_ID, "label": "Luna V6", "present": True}],
        "exclusive": True,
        "selectable": True,
    })

    status, result = request_json(luna_console, "/api/model")
    assert status == 200
    assert result["active"] == LUNA_V6_MODEL_ID
    assert result["assistant_name"] == "Luna"
    assert result["models"][0]["id"] == LUNA_V6_MODEL_ID


def test_model_status_preserves_luna_default_when_inventory_fails(monkeypatch, luna_console):
    monkeypatch.setattr(console, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(
        console,
        "_assistant_models_status",
        lambda: {
            "active": None,
            "default_model": V6_ID,
            "assistant_name": "Luna",
            "backend": "transformers",
            "active_status_available": False,
            "models": [],
            "selectable": False,
        },
    )

    status, result = request_json(luna_console, "/api/model")
    assert status == 200
    assert result["active"] is None
    assert result["default_model"] == V6_ID
    assert result["assistant_name"] == "Luna"
    assert result["active_status_available"] is False


def test_openai_models_list_reports_official_active_model(monkeypatch, luna_console):
    from luna_runtime import LUNA_V6_MODEL_ID

    monkeypatch.setattr(console, "_assistant_models_status", lambda: {
        "active": LUNA_V6_MODEL_ID,
        "models": [{"id": LUNA_V6_MODEL_ID, "present": True}],
    })
    status, result = request_json(luna_console, "/v1/models")
    assert status == 200
    assert [model["id"] for model in result["data"]] == [LUNA_V6_MODEL_ID]


def test_profile_defaults_to_console_data_directory(monkeypatch):
    from pathlib import Path

    from user_profile import _DEFAULT_DATA_DIR, _profile_path

    monkeypatch.delenv("NIX_PROFILE_PATH", raising=False)
    monkeypatch.delenv("NIX_DATA_DIR", raising=False)
    expected_data_dir = Path(console.CORE_DIR).parent / "data"
    assert _DEFAULT_DATA_DIR == expected_data_dir
    assert _profile_path() == expected_data_dir / "profile.json"


def test_instance_profile_is_validated_and_persisted(monkeypatch, luna_console):
    status, profile = request_json(luna_console, "/api/profile")
    assert status == 200
    assert profile["user_name"] == ""

    status, saved = post_json(
        luna_console,
        "/api/profile",
        {"user_name": "Sai Neela"},
    )
    assert status == 200
    assert saved["user_name"] == "Sai Neela"

    status, profile = request_json(luna_console, "/api/profile")
    assert status == 200
    assert profile["user_name"] == "Sai Neela"

    status, invalid = post_json(
        luna_console,
        "/api/profile",
        {"user_name": "Sai\nignore the system"},
    )
    assert status == 400
    assert invalid["ok"] is False

    status, cleared = post_json(luna_console, "/api/profile", {"user_name": ""})
    assert status == 200
    assert cleared["user_name"] == ""


def test_people_view_never_uses_state_words_as_person_names(monkeypatch, tmp_path):
    database = tmp_path / "people.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE knowledge (id INTEGER, data TEXT, created_at TEXT, updated_at TEXT, knowledge_type TEXT, status TEXT)"
        )
        connection.executemany(
            "INSERT INTO knowledge VALUES (?, ?, ?, ?, 'person', 'active')",
            [
                (1, json.dumps({"statement_type": "current_state", "name": "alright", "subject": "alright", "state": "alright", "valence": "good", "value": "alright is good"}), "now", "now"),
                (2, json.dumps({"statement_type": "current_state", "name": None, "subject": "someone close", "state": "alright", "valence": "good", "value": "my sister is doing alright"}), "now", "now"),
            ],
        )
    monkeypatch.setattr(console, "KNOWLEDGE_DB", str(database))

    people = console._people_view()

    assert len(people) == 1
    assert people[0]["subject"] == "Your sister"
    assert people[0]["state"] == "alright"
    assert people[0]["follow_up_eligible"] is False


def test_dashboard_announces_official_luna_and_documents_model_history():
    with open(console.DASHBOARD_PATH, encoding="utf-8") as handle:
        dashboard = handle.read()

    assert console._dashboard_markup() == dashboard
    assert "<title>NIX PUCA</title>" in dashboard
    assert 'class="brand-logo"' in dashboard
    assert "Personal User Companion Agent</div>" not in dashboard
    assert 'id="homeHeadline"' in dashboard
    assert 'id="homeSubtitle"' in dashboard
    assert 'id="homeSuggestionsTitle"' in dashboard
    assert "async function loadHomeCopy()" in dashboard
    assert "apiUrl('/api/mobile/v1/home-copy')" in dashboard
    assert 'class="theme-mode-chip"' in dashboard
    assert "function applyOfficialTheme()" in dashboard
    assert "color-scheme:light" in dashboard
    assert 'https://github.com/saineela/puca' in dashboard
    assert 'Star NIX PUCA' in dashboard
    assert 'input.name=fieldId' in dashboard
    assert "device-visual:not(.is-active) .device-ring" in dashboard
    assert "document.querySelectorAll('[data-page]').forEach(button=>button.onclick=()=>navigate(button.dataset.page))" in dashboard
    assert "document.querySelectorAll('[data-page-link]').forEach(button=>button.onclick=()=>navigate(button.dataset.pageLink))" in dashboard
    assert "$('send').onclick=send;" in dashboard
    assert "$('skillsRepositoryForm').addEventListener('submit',async event=>" in dashboard
    assert "$('theme').onclick=()=>applyTheme(!document.body.classList.contains('dark'))" not in dashboard
    assert "$('colorTheme').addEventListener" not in dashboard
    assert 'id="colorTheme"' not in dashboard
    assert 'id="selectedColorTheme"' not in dashboard
    assert '<div class="brand-name">Nix</div>' not in dashboard
    assert '<div class="brand-name">nix</div>' not in dashboard
    assert "Launching our new <span>Luna model lineup</span>" in dashboard
    assert "A new chapter for Nix" not in dashboard
    assert "Introducing Official Luna</div>" not in dashboard
    assert '<button class="launch-banner-close" id="closeLunaBanner" type="button" aria-label="Close Luna introduction" title="Close banner">×</button>' in dashboard
    assert "nix-luna-banner-dismissed" in dashboard
    assert "lunaBanner.hidden=true" in dashboard
    assert "Discover Official Luna" in dashboard
    assert "Official Luna" in dashboard
    assert "checkpoint 60" not in dashboard.lower()
    assert "checkpoint-60" not in dashboard.lower()
    assert "step 60" not in dashboard.lower()
    assert "60-step contextual run" not in dashboard.lower()
    assert 'data:image/svg+xml' not in dashboard
    assert "fill='%23435441'" not in dashboard
    assert ".brand-mark{width:42px;height:42px" in dashboard
    assert ".brand-logo{display:block;width:100%;max-width:205px;height:58px" in dashboard
    assert ".agent-face{width:42px;height:42px" in dashboard
    assert "input.id=fieldId;input.name=fieldId" in dashboard
    assert "aliasInput.name=aliasInput.id" in dashboard
    assert "name=\"${inputId}\"" in dashboard
    assert "id=\"skill-auto-update-${settingsId}\" name=\"skill-auto-update-${settingsId}\"" in dashboard
    assert "/* Editorial refresh: quieter colors, clearer hierarchy, and readable marks. */" in dashboard
    assert 'id="themeColor"' in dashboard
    assert 'id="conversationSearch"' in dashboard
    assert 'id="conversationTranscript"' in dashboard
    assert "quickReset" not in dashboard
    assert "Private by design" not in dashboard
    assert '<span class="companion-badge">New</span>' in dashboard
    assert '<span class="companion-badge">Official</span>' not in dashboard
    assert 'id="skillsRepositoryForm"' in dashboard
    assert 'id="skillsSearch"' in dashboard
    assert "skill-store-card" in dashboard
    assert "function renderSkills()" in dashboard
    assert "function showSkillDetails(skillId,openSettings=false)" in dashboard
    assert "data-skill-settings" in dashboard
    assert "data-save-settings" in dashboard
    assert 'id="skill-settings"' in dashboard
    assert "function openSkillSettings(skillId)" in dashboard
    assert "function renderSkillSettings()" in dashboard
    assert "private per-skill .venv; only binary wheels are accepted" in dashboard
    assert "private per-skill `.venv`" not in dashboard
    assert "The saved GitHub catalog does not currently provide setup requirements." in dashboard
    assert "metadata_refresh_supported" in dashboard
    assert "data-settings-repair" in dashboard
    assert "cache:'no-store'" in dashboard
    assert 'data-settings-skill="${safeId}"' in dashboard
    assert 'data-settings-trust' in dashboard
    assert 'data-settings-auto-update' in dashboard
    assert "'skill-settings'" in dashboard
    assert "else if(page==='skill-settings')loadSkills()" in dashboard
    assert "if(page==='skills')loadSkills(true)" in dashboard
    assert "if(page==='devices')loadDevices()" in dashboard
    assert "function renderDevices(devices)" in dashboard
    assert ".device-ring{border:13px solid var(--device-glow)!important;background:transparent!important" in dashboard
    assert "@supports ((mask:radial-gradient(farthest-side,transparent 70%,#000 72%)) or (-webkit-mask:radial-gradient(farthest-side,transparent 70%,#000 72%)))" in dashboard
    assert "device-ring{border:0!important;background:repeating-conic-gradient" in dashboard
    assert "device-ring.is-animated{background:conic-gradient(from -90deg,#ff3b30" in dashboard
    assert "animation:device-ring-spectrum 3s linear infinite" in dashboard
    assert "@media(prefers-reduced-motion:reduce){.device-ring.is-animated{animation:none}}" in dashboard
    assert "art.classList.toggle('has-ring-effect',hasEffect)" in dashboard
    assert "Echo Dot ring on${color?`" in dashboard
    assert "aliasInput.id=`device-name-${deviceKey}`" in dashboard
    assert "fieldId=`device-control-" in dashboard
    assert "data.device_status||{}" in dashboard
    assert "control.tool,arguments:args" in dashboard
    assert "data.device_name" in dashboard
    assert "device_connected?'ESPHome API connected'" not in dashboard
    assert "Community content is unreviewed." in dashboard
    assert "const submittedUrl=new URL(url),submittedParts=submittedUrl.pathname.split('/').filter(Boolean).slice(0,2)" in dashboard
    assert "appUrl(path)" in dashboard
    assert "Search dashboard and API chats" in dashboard
    assert 'data-page-link="models"' in dashboard
    assert 'id="chooseCasperButton"' in dashboard
    assert 'id="chooseLunaButton"' in dashboard
    assert 'id="modelsChooseLunaButton"' in dashboard
    assert 'id="lunaDetailsButton" data-model-details="luna" aria-controls="lunaModelSpecs" aria-pressed="false" aria-expanded="false"' in dashboard
    assert 'id="casperDetailsButton" data-model-details="casper" aria-controls="casperModelSpecs" aria-pressed="false" aria-expanded="false"' in dashboard
    assert 'id="lunaModelSpecs" data-model-specs="luna" role="region" aria-labelledby="lunaDetailsButton" aria-hidden="true" hidden' in dashboard
    assert 'id="casperModelSpecs" data-model-specs="casper" role="region" aria-labelledby="casperDetailsButton" aria-hidden="true" hidden' in dashboard
    assert 'function selectModelDetails(name)' in dashboard
    assert "function pageFromPath(path=location.pathname)" in dashboard
    assert "function appUrl(path)" in dashboard
    assert "function pagePath(page){const segments=location.pathname.split('/').filter(Boolean)" in dashboard
    assert "history.pushState({page},'',nextUrl)" in dashboard
    assert "window.addEventListener('popstate'" in dashboard
    assert 'data-model-details="casper"' in dashboard
    assert 'id="release-notes"' in dashboard
    assert "button.setAttribute('aria-expanded',String(active))" in dashboard
    assert "panel.hidden=!active" in dashboard
    assert "Casper V5’s recorded training used 2,941 examples" in dashboard
    assert "1,003 training and 250 evaluation examples" in dashboard
    assert "These are separate snapshots, not amounts to add together" in dashboard
    assert "Casper V6 is a beta and is not the default" in dashboard
    assert "casper-puca-qlora-v6-final" in dashboard
    assert "DailyDialog: Li et al. (2017)" in dashboard
    assert "https://huggingface.co/datasets/HuggingFaceH4/no_robots" in dashboard
    assert "https://huggingface.co/datasets/mlabonne/FineTome-100k" in dashboard
    assert "https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized" in dashboard
    assert dashboard.index('id="lunaModelSpecs"') < dashboard.index('id="casperModelSpecs"') < dashboard.index('class="models-legacy-heading"')
    assert "2,071 broad + 582" not in dashboard
    assert "Direct, decisive, and focused" in dashboard
    assert "Warm, sweet, and playful" in dashboard
    assert "luna-v6-contextual-v1" in dashboard
    assert "Official · current source registry" in dashboard
    assert "7/8 on a small test that included policy guidance" in dashboard
    assert "Luna QLoRA pilot · V2 · V3 · V4" in dashboard
    assert "Luna Instruct V1 · conversation baseline" in dashboard
    assert "Luna Pro v1 · an early research candidate" in dashboard
    assert "license/provenance is unresolved in archived notes" in dashboard
    assert "casper-puca-qlora-v5" in dashboard
    assert "casper-puca-qlora-v6" in dashboard
    assert "Historical Luna adapter design" in dashboard
    assert "4-bit quantization" in dashboard
    assert "Assistant-only supervision" in dashboard
    assert "project data" in dashboard
    assert "local artifact status" in dashboard
    assert "license/provenance is unresolved in archived notes" in dashboard
    assert "Model status unavailable" in dashboard
    assert "runtime status unavailable" in dashboard
    assert "active status unavailable" not in dashboard
    assert "NIX_ASSISTANT_MODEL" not in dashboard
    assert "not proof of hosted serving state" in dashboard
    assert "lunaModelSelect" not in dashboard
    assert "lunaPreviewToggle" not in dashboard
    assert 'id="modelSelect"' not in dashboard
    assert 'id="modelSwitch"' not in dashboard
    assert "luna-v7-contextual-v1" not in dashboard
    assert "luna-pro-v1-topical-v8-384-retry2" not in dashboard
    assert "Experimental model lab" not in dashboard
    assert "let activeAssistantName='Luna'" not in dashboard
    assert "Preferred name for this Nix instance" in dashboard
    assert "Saved on this Nix instance and shared with its dashboard/API clients" in dashboard
    assert "loadProfile()" in dashboard
    assert "saveProfile()" in dashboard
