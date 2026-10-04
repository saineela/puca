import base64
import hashlib
import http.client
import io
import json
import os
import subprocess
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

import console
import skill_runtime
from skill_runtime import SkillRuntime, SkillRuntimeError


@pytest.fixture
def skills_state(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "SKILLS_STATE_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setattr(console, "SKILLS_INSTALL_DIR", str(tmp_path / "installed"))
    monkeypatch.setattr(console, "DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def sample_repository():
    root = Path(console.REPO_ROOT) / "Nix-skills-repo"
    return (
        json.loads((root / "nix-skills.json").read_text(encoding="utf-8")),
        json.loads(
            (root / "skills" / "web-search" / "skill.json").read_text(
                encoding="utf-8"
            )
        ),
    )


class FakeResponse:
    def __init__(self, body=b"{}", status_code=200, headers=None):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.status_code = status_code
        self.headers = headers or {"Content-Length": str(len(self.body))}
        self.closed = False

    def iter_content(self, chunk_size=8192):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]

    def close(self):
        self.closed = True


def github_payloads():
    repository_manifest, skill_manifest = sample_repository()
    skill_dir = Path(console.REPO_ROOT) / "Nix-skills-repo" / "skills" / "web-search"
    return {
        "https://api.github.com/repos/example/Nix-skills-repo": {
            "private": False,
            "full_name": "example/Nix-skills-repo",
            "default_branch": "main",
        },
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/nix-skills.json": repository_manifest,
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/skill.json": skill_manifest,
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/README.md": (
            skill_dir / "README.md"
        ).read_bytes(),
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/instructions.md": (
            skill_dir / "instructions.md"
        ).read_bytes(),
    }


def mock_github(monkeypatch, payloads=None):
    payloads = payloads or github_payloads()
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        if url not in payloads:
            return FakeResponse(b"not found", status_code=404)
        payload = payloads[url]
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return FakeResponse(body)

    monkeypatch.setattr(console.requests, "get", get)
    return requested


def test_local_example_uses_the_supported_v1_format():
    repository_manifest, skill_manifest = sample_repository()

    assert repository_manifest["kind"] == "nix-skills-repository"
    assert repository_manifest["schema_version"] == 1
    assert repository_manifest["skills"] == [
        {"manifest": "skills/web-search/skill.json"}
    ]
    parsed = console._parse_skill_manifest(
        skill_manifest, "skills/web-search/skill.json", repository_manifest["publisher"]
    )
    assert parsed["slug"] == "web-search"
    assert parsed["license"] == "MIT"
    assert parsed["files"] == ["README.md", "instructions.md"]


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/example/skills",
        "https://github.com.evil.test/example/skills",
        "https://user@github.com/example/skills",
        "https://github.com:443/example/skills",
        "https://github.com/example/skills/tree/main",
        "https://github.com/example/skills?tab=readme",
        "https://github.com/example/..",
        "file:///etc/passwd",
        "https://raw.githubusercontent.com/example/skills/main/nix-skills.json",
    ],
)
def test_github_repository_url_rejects_non_repository_urls(url):
    with pytest.raises(console.SkillRepositoryError):
        console._github_repository_url(url)


def test_github_repository_url_normalizes_a_public_repo_root():
    assert console._github_repository_url("https://github.com/Example/Skills.git/") == (
        "Example",
        "Skills",
        "https://github.com/Example/Skills",
    )


def test_manifest_rejects_path_traversal_and_invalid_slug():
    _, manifest = sample_repository()
    manifest["files"] = ["../../escape.py"]
    with pytest.raises(console.SkillRepositoryError, match="path"):
        console._parse_skill_manifest(
            manifest, "skills/web-search/skill.json", "NIX Skills Example"
        )

    _, manifest = sample_repository()
    manifest["id"] = "../outside"
    with pytest.raises(console.SkillRepositoryError, match="lowercase slug"):
        console._parse_skill_manifest(
            manifest, "skills/web-search/skill.json", "NIX Skills Example"
        )


def test_github_fetch_rejects_redirects_and_oversized_content(monkeypatch):
    redirect = FakeResponse(b"", status_code=302)
    monkeypatch.setattr(console.requests, "get", lambda *args, **kwargs: redirect)
    with pytest.raises(console.SkillRepositoryError, match="redirected"):
        console._fetch_github_bytes(
            "https://api.github.com/repos/example/skills",
            1024,
            "api.github.com",
        )
    assert redirect.closed

    too_large = FakeResponse(b"12345")
    monkeypatch.setattr(console.requests, "get", lambda *args, **kwargs: too_large)
    with pytest.raises(console.SkillRepositoryError, match="size limit"):
        console._fetch_github_bytes(
            "https://api.github.com/repos/example/skills",
            4,
            "api.github.com",
        )
    assert too_large.closed


def test_import_public_repo_and_install_only_declared_static_files(
    skills_state, monkeypatch
):
    requested = mock_github(monkeypatch)
    result = console._import_skill_repository(
        "https://github.com/example/Nix-skills-repo"
    )

    assert result["ok"] is True
    skill = next(item for item in result["skills"] if item["community"])
    assert skill["id"] == "github:example/nix-skills-repo:web-search"
    assert skill["runnable"] is False
    assert skill["installed"] is False
    assert skill["repository_url"] == "https://github.com/example/Nix-skills-repo"

    result = console._set_skill_installed(skill["id"], True)
    installed_skill = next(item for item in result["skills"] if item["id"] == skill["id"])
    assert installed_skill["installed"] is True
    assert installed_skill["implementation_status"] == "execution_disabled"
    assert requested[-2:] == [
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/README.md",
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/instructions.md",
    ]

    package = Path(console._skill_package_path(skill["id"]))
    assert (package / "README.md").read_bytes() == github_payloads()[
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/README.md"
    ]
    assert (package / "instructions.md").is_file()
    package_manifest = json.loads((package / "skill.json").read_text(encoding="utf-8"))
    assert package_manifest["files"] == ["README.md", "instructions.md"]
    state = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
    assert state["installed"] == [skill["id"]]

    result = console._set_skill_installed(skill["id"], False)
    uninstalled_skill = next(item for item in result["skills"] if item["id"] == skill["id"])
    assert uninstalled_skill["installed"] is False
    assert not package.exists()
    state = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
    assert state["installed"] == []


def test_local_ring_package_uses_github_import_and_install_flow_without_execution(
    skills_state, monkeypatch
):
    package_root = Path(console.CORE_DIR) / "ring_light_package"
    manifest = json.loads(
        (package_root / "skills" / "ring-light" / "skill.json").read_text(encoding="utf-8")
    )
    repository_manifest = json.loads((package_root / "nix-skills.json").read_text(encoding="utf-8"))
    package_dir = package_root / "skills" / "ring-light"
    base = "https://raw.githubusercontent.com/example/ring-light-local/main"
    payloads = {
        "https://api.github.com/repos/example/ring-light-local": {
            "private": False,
            "full_name": "example/ring-light-local",
            "default_branch": "main",
        },
        f"{base}/nix-skills.json": json.dumps(repository_manifest).encode(),
        f"{base}/skills/ring-light/skill.json": json.dumps(manifest).encode(),
    }
    for relative in manifest["files"]:
        payloads[f"{base}/skills/ring-light/{relative}"] = (
            package_dir / relative
        ).read_bytes()

    requested = mock_github(monkeypatch, payloads)
    dependency_commands = []

    def mock_dependency_setup(command, **kwargs):
        dependency_commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[3])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake python", encoding="utf-8")

    monkeypatch.setattr(skill_runtime.subprocess, "run", mock_dependency_setup)
    monkeypatch.setattr(skill_runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("installation must not start a skill worker"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, body):
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode())

    try:
        preview = post("/api/skills", {
            "action": "add_repository",
            "repository_url": "https://github.com/example/ring-light-local",
        })
        skill = next(item for item in preview["skills"] if item["community"])
        assert skill["id"] == "github:example/ring-light-local:ring-light"
        assert skill["installed"] is False
        assert skill["runtime_declared"] is True
        assert skill["device_ui"]["visualization"] == {
            "type": "ring", "color_path": "state.rgb", "active_path": "state.on", "label_path": "device_name"
        }
        assert [control["label"] for control in skill["device_ui"]["controls"]] == ["Turn on", "Turn off", "Set color", "Set brightness"]
        assert skill["trusted"] is False
        assert skill["runnable"] is False

        installed = post("/api/skills", {"skill_id": skill["id"], "action": "install"})
        record = next(item for item in installed["skills"] if item["id"] == skill["id"])
        assert record["installed"] is True
        assert record["implementation_status"] == "execution_disabled"
        assert record["configured"] is False
        assert record["trusted"] is False
        assert record["worker_running"] is False
        assert record["setup_fields"] == manifest["setup_fields"]
        assert record["device_ui"]["visualization"] == manifest["device_ui"]["visualization"]
        assert record["device_ui"]["fields"] == manifest["device_ui"]["fields"]
        assert manifest["version"] == "1.1.1"
        assert manifest["device_ui"]["visualization"] == {
            "type": "ring", "color_path": "state.rgb", "active_path": "state.on", "label_path": "device_name"
        }
        tool_schema = manifest["tools"][0]["input_schema"]
        assert "palette" not in tool_schema["properties"]
        assert "color" not in tool_schema["properties"]
        assert manifest["tools"][0]["result_schema"]["required"] == ["ok", "action", "device_connected", "device_name", "available_effects", "message"]
        assert [command[1:3] for command in dependency_commands] == [["-m", "venv"], ["-m", "pip"]]
        assert "--only-binary=:all:" in dependency_commands[1]
        assert "--user" not in dependency_commands[1]
        environment = (
            Path(console.DATA_DIR) / "skills" / "runtime_envs"
            / hashlib.sha256(skill["id"].encode()).hexdigest()
        )
        assert (environment / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")).is_file()
        assert not runtime_worker_was_started(dependency_commands)

        # Older saved repository records can lack setup_fields even though the
        # installed package manifest contains them; the API must use the package schema.
        saved = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
        saved_skill = saved["repositories"][0]["skills"][0]
        saved_skill["setup_fields"] = []
        Path(console.SKILLS_STATE_PATH).write_text(json.dumps(saved), encoding="utf-8")
        refreshed_record = next(
            item for item in console._skills_payload()["skills"] if item["id"] == skill["id"]
        )
        assert refreshed_record["setup_fields"] == manifest["setup_fields"]
        assert refreshed_record["device_ui"]["visualization"] == manifest["device_ui"]["visualization"]
        assert refreshed_record["device_ui"]["fields"] == manifest["device_ui"]["fields"]
        dashboard = console._dashboard_markup()
        assert "Skill settings" in dashboard
        assert "aliasInput.name=aliasInput.id" in dashboard
        assert 'name="${inputId}" data-settings-field=' in dashboard
        assert "device-ring.is-animated{background:conic-gradient" in dashboard
        assert "device-ring-spectrum 3s linear infinite" in dashboard
        assert "community&&skill.runtime_declared?`<div class=\"skill-details-warning\">" in dashboard
        assert 'id="skill-settings"' in dashboard
        assert "function renderSkillSettings()" in dashboard
        assert "function openSkillSettings(skillId)" in dashboard
        assert 'data-settings-skill="${safeId}"' in dashboard
        assert "if(installing)openSkillSettings(skillId)" in dashboard
        assert "Expected JSON from ${path}, received ${contentType||'an unknown content type'}" in dashboard
        assert "The JSON response from ${path} was invalid (HTTP ${r.status})." in dashboard
        assert 'data-settings-save' in dashboard
        assert 'data-settings-trust' in dashboard
        assert 'data-settings-runtime="connect"' in dashboard
        assert 'data-settings-auto-update' in dashboard
        assert "if(!skill.installed){const installButton=dialog.querySelector('.skill-details-install')" in dashboard
        assert "device_label||device.name" in dashboard
        assert "data-save-settings" in dashboard
        assert "${skill.configured&&skill.trusted?'<button type=\"button\" data-runtime-action=\"connect\">Connect / test</button>':''}" in dashboard

        package = Path(console._skill_package_path(skill["id"]))
        installed_manifest = json.loads((package / "skill.json").read_text(encoding="utf-8"))
        assert installed_manifest["runtime"]["protocol"] == "nix-skill-jsonl-v1"
        assert installed_manifest["assistant_summary"] == manifest["assistant_summary"]
        assert installed_manifest["device_ui"]["visualization"] == {
            "type": "ring", "color_path": "state.rgb", "active_path": "state.on", "label_path": "device_name"
        }
        assert "available_palettes" not in installed_manifest["tools"][0]["result_schema"]["properties"]
        assert skill_runtime_runtime_schema_allows_color_forms(installed_manifest)
        for relative in manifest["files"]:
            assert (package / relative).read_bytes() == (package_dir / relative).read_bytes()
        assert requested[-len(manifest["files"]):] == [
            f"{base}/skills/ring-light/{relative}" for relative in manifest["files"]
        ]

        # Simulate an old runtime which saved this GitHub package as static-only.
        saved = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
        saved_skill = saved["repositories"][0]["skills"][0]
        saved_skill.update(runtime=None, setup_fields=[], tools=[], triggers=[])
        Path(console.SKILLS_STATE_PATH).write_text(json.dumps(saved), encoding="utf-8")
        console._write_skill_package(
            saved_skill,
            [(relative, (package_dir / relative).read_bytes()) for relative in manifest["files"]],
        )
        stale_payload = console._skills_payload()
        stale_record = next(item for item in stale_payload["skills"] if item["id"] == skill["id"])
        assert stale_record["runtime_declared"] is False
        assert stale_record["setup_fields"] == []
        assert stale_record["metadata_refresh_supported"] is True
        assert "installed&&(skill.runtime_declared||skill.metadata_refresh_supported)" in console._dashboard_markup()
        assert "data-settings-repair" in console._dashboard_markup()

        repaired_payload = post("/api/skills", {"action": "repair", "skill_id": skill["id"]})
        repaired_record = next(
            item for item in repaired_payload["skills"] if item["id"] == skill["id"]
        )
        assert repaired_record["runtime_declared"] is True
        assert repaired_record["setup_fields"] == manifest["setup_fields"]
        assert repaired_record["fingerprint"]
        for relative in manifest["files"]:
            assert (package / relative).read_bytes() == (package_dir / relative).read_bytes()

        runtime = SkillRuntime(
            state_path=Path(console.SKILLS_STATE_PATH),
            install_dir=Path(console.SKILLS_INSTALL_DIR),
            data_dir=skills_state / "data",
        )
        assert runtime.trigger_candidate("Turn on the Echo Dot ring") is not None
        assert runtime.trigger_candidate("Could you explain how the Echo Dot ring works?") is None
        saved = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
        saved["device_names"][skill["id"]] = "Studio Ring"
        Path(console.SKILLS_STATE_PATH).write_text(json.dumps(saved), encoding="utf-8")
        # Target resolution is independent of setup/trust so it can provide a
        # clear setup message, but proposal context remains runnable-only.
        specs = runtime.installed_skill_specs()
        assert specs[0]["device_name"] == "Studio Ring"
        assert runtime.targeted_skill_specs("Turn the Studio Ring to red")[0]["skill_id"] == skill["id"]
        assert runtime.targeted_skill_specs("change it to green", [
            {"role": "user", "content": "Turn the Studio Ring on"},
        ])[0]["skill_id"] == skill["id"]
        assert runtime.targeted_skill_specs("Could you explain how the Studio Ring works?") == []
        with pytest.raises(SkillRuntimeError, match="setup is incomplete"):
            runtime.match("Turn on the Echo Dot ring")
        with pytest.raises(SkillRuntimeError, match="Configure required setup fields"):
            runtime.trust(skill["id"], True)
        with pytest.raises(SkillRuntimeError, match="Echo Dot IP address is required"):
            runtime.save_configuration(skill["id"], {"device_ip": "", "api_key": ""})
        test_key = base64.b64encode(b"test-only-key-".ljust(32, b"x")).decode("ascii")
        configured_payload = post("/api/skills", {
            "action": "configure",
            "skill_id": skill["id"],
            "configuration": {"device_ip": "192.168.1.20", "api_key": test_key},
        })
        configured = next(item for item in configured_payload["skills"] if item["id"] == skill["id"])
        assert configured["configured"] is True
        assert configured["configuration"]["api_key"] == {"configured": True}
        assert test_key not in json.dumps(configured_payload)

        trusted_payload = post("/api/skills", {
            "action": "trust",
            "skill_id": skill["id"],
            "confirm_digest": configured["fingerprint"],
        })
        trusted = next(item for item in trusted_payload["skills"] if item["id"] == skill["id"])
        assert trusted["trusted"] is True
        assert trusted["runnable"] is True
        assert trusted["worker_running"] is False
        assert runtime.status(skill["id"])["worker_running"] is False

        class FakeWorker:
            def __init__(self, _package, _entrypoint, _config, _timeout, python_executable):
                self.python_executable = python_executable
                self.process = SimpleNamespace(poll=lambda: -1)

            def stop(self):
                return None

        def prepare_dependencies(command, **kwargs):
            if command[1:3] == ["-m", "venv"]:
                environment = Path(command[3])
                python = environment / "bin" / "python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("fake python", encoding="utf-8")

        monkeypatch.setattr(skill_runtime.subprocess, "run", prepare_dependencies)
        runtime._ensure_dependencies(skill["id"], package, installed_manifest)
        monkeypatch.setattr(skill_runtime.subprocess, "run", lambda *args, **kwargs: pytest.fail("dependency environment should be cached"))
        monkeypatch.setattr(skill_runtime, "_Worker", FakeWorker)
        worker, _worker_manifest = runtime._worker(skill["id"])
        assert worker.python_executable == str(runtime.dependencies_dir / hashlib.sha256(skill["id"].encode()).hexdigest() / ".venv" / "bin" / "python")
        environment = runtime.dependencies_dir / hashlib.sha256(skill["id"].encode()).hexdigest() / ".venv"
        assert (environment / "bin" / "python").is_file()
        assert runtime.status(skill["id"])["worker_running"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_runtime_installs_declared_dependencies_once_in_per_skill_environment(skills_state, monkeypatch):
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    skill_id = "github:example/ring-light:ring-light"
    package = skills_state / "package"
    package.mkdir()
    requirements = package / "requirements.txt"
    requirements.write_text("aioesphomeapi>=45.7,<46", encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[3])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake python", encoding="utf-8")

    monkeypatch.setattr(skill_runtime.subprocess, "run", fake_run)
    manifest = {"files": ["requirements.txt"], "runtime": {"requirements": "requirements.txt"}}

    python = runtime._ensure_dependencies(skill_id, package, manifest)
    assert Path(python).is_file()
    assert len(commands) == 2
    assert commands[0][1:3] == ["-m", "venv"]
    assert commands[1][1:3] == ["-m", "pip"]
    assert commands[1][-2:] == ["--requirement", str(requirements)]
    assert "--only-binary=:all:" in commands[1]
    assert "--user" not in commands[1]
    assert Path(python).parent.parent.name == ".venv"
    assert runtime._ensure_dependencies(skill_id, package, manifest) == python
    assert len(commands) == 2


def test_runtime_routes_direct_generic_light_request_to_ring_skill(skills_state):
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    skill_id = "github:example/ring-light:ring-light"
    package = runtime.package_dir(runtime.install_dir, skill_id)
    package.mkdir(parents=True)
    (package / "ring.py").write_text("# test-only package file\\n", encoding="utf-8")
    manifest = {
        "kind": "nix-skill",
        "id": "ring-light",
        "name": "Ring Light",
        "runtime": {"protocol": skill_runtime.PROTOCOL, "entrypoint": "ring.py"},
        "files": ["ring.py"],
        "tools": [{"name": "control_ring"}],
        "triggers": ["ring", "ring light"],
    }
    (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
    (skills_state / "skills.json").write_text(json.dumps({"installed": [skill_id]}), encoding="utf-8")

    candidate = runtime.trigger_candidate("hey change the light to red")
    assert candidate is not None
    assert candidate[0] == skill_id
    assert runtime.trigger_candidate("How do I change the light to red?") is None


def test_runtime_targeting_uses_device_alias_and_nearest_relevant_history(skills_state, monkeypatch):
    skill_id = "github:example/ring-light:ring-light"
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    package = runtime.package_dir(runtime.install_dir, skill_id)
    package.mkdir(parents=True)
    (package / "worker.py").write_text("# mocked test package\\n", encoding="utf-8")
    manifest = {
        "kind": "nix-skill",
        "id": "ring-light",
        "name": "Ring Light",
        "runtime": {"protocol": skill_runtime.PROTOCOL, "entrypoint": "worker.py"},
        "files": ["worker.py"],
        "tools": [{"name": "control_ring", "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "result_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}],
        "triggers": ["ring", "ring light"],
    }
    (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
    (skills_state / "skills.json").write_text(json.dumps({
        "installed": [skill_id],
        "device_names": {skill_id: "Studio Ring"},
    }), encoding="utf-8")
    monkeypatch.setattr(runtime, "status", lambda _skill_id: {
        "configured": True, "trusted": True, "runnable": True, "worker_running": False,
    })

    assert runtime.targeted_skill_specs("Set the Studio Ring to blue")[0]["skill_id"] == skill_id
    assert runtime.targeted_skill_specs("Can you turn off the bedroom light?")[0]["skill_id"] == skill_id
    assert runtime.trigger_candidate("Can you turn off the Ring Light?")[0] == skill_id
    history = [
        {"role": "user", "content": "Turn the Studio Ring on"},
        {"role": "assistant", "content": "The device is on"},
    ]
    assert runtime.targeted_skill_specs("change it to green", history)[0]["skill_id"] == skill_id
    assert runtime.targeted_skill_specs("Could you change it to blue?", history)[0]["skill_id"] == skill_id
    assert runtime.targeted_skill_specs("Could you explain how the Studio Ring works?") == []
    assert runtime.targeted_skill_specs("Can you tell me a story about a ring?") == []


def test_runtime_match_rewrites_configured_alias_for_skill_parser(skills_state, monkeypatch):
    skill_id = "github:example/ring-light:ring-light"
    source = Path(console.CORE_DIR) / "ring_light_package" / "skills" / "ring-light"
    manifest = json.loads((source / "skill.json").read_text(encoding="utf-8"))
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    package = runtime.package_dir(runtime.install_dir, skill_id)
    package.mkdir(parents=True)
    (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
    for relative in manifest["files"]:
        (package / relative).write_bytes((source / relative).read_bytes())
    (skills_state / "skills.json").write_text(json.dumps({
        "installed": [skill_id],
        "device_names": {skill_id: "bedroom light"},
    }), encoding="utf-8")
    monkeypatch.setattr(runtime, "status", lambda _skill_id: {
        "configured": True, "trusted": True, "runnable": True, "worker_running": False,
    })

    class FakeWorker:
        def __init__(self):
            self.calls = []

        def request(self, operation, payload):
            self.calls.append((operation, payload))
            assert operation == "match"
            return {"match": {"action": "off"}}

    worker = FakeWorker()
    monkeypatch.setattr(runtime, "_worker", lambda _skill_id: (worker, manifest))

    matched = runtime.match("can you turn off the bedroom light")

    assert matched["skill_id"] == skill_id
    assert matched["arguments"] == {"action": "off"}
    assert worker.calls == [("match", {"text": "can you turn off the Ring Light"})]


def test_runtime_binds_single_tool_match_arguments_to_declared_tool(skills_state, monkeypatch):
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    skill_id = "github:example/ring-light:ring-light"
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["color"]},
            "rgb": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 0, "maximum": 255}},
        },
        "required": ["action", "rgb"],
        "additionalProperties": False,
    }
    result_schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
        "additionalProperties": False,
    }
    manifest = {
        "kind": "nix-skill",
        "id": "ring-light",
        "name": "Ring Light",
        "runtime": {"protocol": skill_runtime.PROTOCOL, "entrypoint": "worker.py"},
        "files": ["worker.py"],
        "tools": [{"name": "control_ring", "input_schema": schema, "result_schema": result_schema}],
    }
    package = runtime.package_dir(runtime.install_dir, skill_id)
    package.mkdir(parents=True)
    (package / "worker.py").write_text("# test only\\n", encoding="utf-8")
    (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeWorker:
        def __init__(self):
            self.calls = []

        def request(self, operation, payload):
            self.calls.append((operation, payload))
            if operation == "match":
                return {"match": {"action": "color", "rgb": [0, 255, 0]}}
            return {"result": {"action": payload["arguments"]["action"]}}

    worker = FakeWorker()
    monkeypatch.setattr(runtime, "trigger_candidate", lambda _text: (skill_id, manifest))
    monkeypatch.setattr(runtime, "status", lambda _skill_id: {"configured": True, "trusted": True})
    monkeypatch.setattr(runtime, "_worker", lambda _skill_id: (worker, manifest))
    (skills_state / "skills.json").write_text(json.dumps({"installed": [skill_id]}), encoding="utf-8")

    matched = runtime.match("Switch the ring on light to green")
    assert matched == {
        "skill_id": skill_id,
        "skill_name": "Ring Light",
        "tool": manifest["tools"][0],
        "arguments": {"action": "color", "rgb": [0, 255, 0]},
    }
    saved = json.loads((skills_state / "skills.json").read_text(encoding="utf-8"))
    saved["installed"] = [skill_id]
    (skills_state / "skills.json").write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(runtime, "status", lambda _skill_id: {"runnable": True, "configured": True, "trusted": True})
    outcome = runtime.execute(matched)
    assert outcome["ok"] is True
    assert outcome["tool"] == "control_ring"
    assert worker.calls == [
        ("match", {"text": "Switch the ring on light to green"}),
        ("execute", {"tool": "control_ring", "arguments": {"action": "color", "rgb": [0, 255, 0]}}),
    ]


def test_runtime_rejects_pip_options_without_invoking_installer(skills_state, monkeypatch):
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    package = skills_state / "package"
    package.mkdir()
    (package / "requirements.txt").write_text("--index-url https://example.invalid", encoding="utf-8")
    monkeypatch.setattr(
        skill_runtime.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid requirements must not invoke pip"),
    )

    with pytest.raises(SkillRuntimeError, match="bounded package/version specifiers"):
        runtime._ensure_dependencies("web-search", package, {"files": ["requirements.txt"], "runtime": {"requirements": "requirements.txt"}})


def test_device_ui_schema_rejects_untrusted_or_unbounded_dashboard_metadata():
    manifest = json.loads((Path(console.CORE_DIR) / "ring_light_package" / "skills" / "ring-light" / "skill.json").read_text(encoding="utf-8"))
    manifest["device_ui"]["visualization"]["type"] = "script"
    with pytest.raises(console.SkillRepositoryError, match="unsupported visualization"):
        console._parse_skill_manifest(manifest, "skills/ring-light/skill.json", "test")

    manifest = json.loads((Path(console.CORE_DIR) / "ring_light_package" / "skills" / "ring-light" / "skill.json").read_text(encoding="utf-8"))
    manifest["device_ui"]["controls"][2]["fields"][0]["widget"] = "html"
    with pytest.raises(console.SkillRepositoryError, match="uniquely satisfy"):
        console._parse_skill_manifest(manifest, "skills/ring-light/skill.json", "test")


def test_runtime_device_status_requires_explicit_connect_and_trust(skills_state, monkeypatch):
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    called = []
    monkeypatch.setattr(runtime, "_worker", lambda skill_id: called.append(skill_id))
    with pytest.raises(SkillRuntimeError, match="Configure and explicitly trust"):
        runtime.device_status("github:example/ring-light:ring-light")
    assert called == []


def test_device_controls_sharing_a_tool_select_matching_declared_arguments(skills_state, monkeypatch):
    skill_id = "github:example/ring-light:ring-light"
    source = Path(console.CORE_DIR) / "ring_light_package" / "skills" / "ring-light"
    manifest = json.loads((source / "skill.json").read_text(encoding="utf-8"))
    package = SkillRuntime.package_dir(skills_state / "installed", skill_id)
    package.mkdir(parents=True)
    (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
    for relative in manifest["files"]:
        (package / relative).write_bytes((source / relative).read_bytes())

    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    calls = []
    monkeypatch.setattr(
        runtime,
        "_device_tool_call",
        lambda requested, tool, arguments: calls.append((requested, tool, arguments)) or {"ok": True},
    )

    controls = (
        {"action": "on"},
        {"action": "off"},
        {"action": "color", "rgb": [17, 34, 51]},
        {"action": "brightness", "brightness": 0.5},
    )
    for arguments in controls:
        assert runtime.control_device(skill_id, arguments, "control_ring") == {"ok": True}

    assert calls == [(skill_id, "control_ring", arguments) for arguments in controls]
    with pytest.raises(SkillRuntimeError, match="do not match a declared control"):
        runtime.control_device(skill_id, {"action": "color", "rgb": [17, 34, 51], "unexpected": True}, "control_ring")
    assert len(calls) == len(controls)


def test_runtime_dependency_install_failure_cleans_partial_environment(skills_state, monkeypatch):
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    package = skills_state / "package"
    package.mkdir()
    (package / "requirements.txt").write_text("aioesphomeapi>=45.7,<46", encoding="utf-8")

    def failed_install(command, **kwargs):
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[3])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake python", encoding="utf-8")
            return
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(skill_runtime.subprocess, "run", failed_install)
    with pytest.raises(SkillRuntimeError, match="Could not install skill runtime dependencies"):
        runtime._ensure_dependencies("web-search", package, {"files": ["requirements.txt"], "runtime": {"requirements": "requirements.txt"}})
    environment = runtime.dependencies_dir / hashlib.sha256(b"web-search").hexdigest()
    assert not environment.exists()


def skill_runtime_runtime_schema_allows_color_forms(manifest):
    schema = manifest["tools"][0]["input_schema"]
    skill_runtime.validate_json_schema({"action": "color", "rgb": [255, 0, 0]}, schema)
    with pytest.raises(SkillRuntimeError):
        skill_runtime.validate_json_schema({"action": "color", "color": "red"}, schema)
    with pytest.raises(SkillRuntimeError):
        skill_runtime.validate_json_schema({"action": "on", "rgb": [255, 0, 0]}, schema)
    with pytest.raises(SkillRuntimeError):
        skill_runtime.validate_json_schema({"action": "color", "color": "red", "rgb": [255, 0, 0]}, schema)
    with pytest.raises(SkillRuntimeError):
        skill_runtime.validate_json_schema({"action": "brightness", "brightness": True}, schema)
    return True


def runtime_worker_was_started(commands):
    return any("--nix-skill-worker" in command for command in commands)


def test_install_failure_does_not_mark_package_installed(skills_state, monkeypatch):
    mock_github(monkeypatch)
    result = console._import_skill_repository("https://github.com/example/Nix-skills-repo")
    skill = next(item for item in result["skills"] if item["community"])

    original_get = console.requests.get

    def failed_package_file(url, **kwargs):
        if url.endswith("/instructions.md"):
            return FakeResponse(b"", status_code=404)
        return original_get(url, **kwargs)

    monkeypatch.setattr(console.requests, "get", failed_package_file)
    with pytest.raises(console.SkillRepositoryError, match="HTTP 404"):
        console._set_skill_installed(skill["id"], True)

    saved = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
    assert skill["id"] not in saved["installed"]
    assert not Path(console._skill_package_path(skill["id"])).exists()


def test_private_repository_is_rejected_without_fetching_raw_files(skills_state, monkeypatch):
    requested = mock_github(
        monkeypatch,
        {"https://api.github.com/repos/example/skills": {
            "private": True,
            "full_name": "example/skills",
            "default_branch": "main",
        }},
    )
    with pytest.raises(console.SkillRepositoryError, match="public"):
        console._import_skill_repository("https://github.com/example/skills")
    assert requested == ["https://api.github.com/repos/example/skills"]
    assert not Path(console.SKILLS_STATE_PATH).exists()
    assert not Path(console.SKILLS_INSTALL_DIR).exists()


def _make_skill_zip(manifest=None, files=None):
    if manifest is None:
        _repository, manifest = sample_repository()
    if files is None:
        files = {path: f"contents of {path}".encode() for path in manifest.get("files", [])}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("skill.json", json.dumps(manifest))
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_zip_manifest_upload_rejects_traversal_extras_and_mismatched_file_sets():
    _repository, manifest = sample_repository()
    files = {path: f"declared {path}".encode() for path in manifest["files"]}
    skill, contents = console._read_skill_zip(_make_skill_zip(manifest, files))
    assert skill["id"] == "web-search"
    assert skill["source_type"] == "upload"
    assert contents == list(files.items())

    with pytest.raises(console.SkillRepositoryError, match="path"):
        console._read_skill_zip(_make_skill_zip(manifest, {**files, "../escape.txt": b"no"}))

    with pytest.raises(console.SkillRepositoryError, match="missing"):
        console._read_skill_zip(_make_skill_zip(manifest, {manifest["files"][0]: b"only one"}))

    too_large = b"x" * (console.SKILLS_MAX_UPLOAD_BYTES + 1)
    with pytest.raises(console.SkillRepositoryError, match="2 MiB"):
        console._read_skill_zip(too_large)


def test_svg_logo_is_passive_and_rejects_active_or_external_content():
    safe, mime = console._validate_skill_logo(
        "logo.svg",
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><path d="M0 0L8 8" fill="#123456"/></svg>',
    )
    assert mime == "image/svg+xml"
    assert b"<path" in safe
    for content in (
        b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><image href="https://evil.test/x"/></svg>',
        b'<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg>&x;</svg>',
    ):
        with pytest.raises(console.SkillRepositoryError):
            console._validate_skill_logo("logo.svg", content)


def test_opt_in_github_updates_are_throttled_and_revoke_trust_on_package_change(skills_state, monkeypatch):
    payloads = github_payloads()
    requested = mock_github(monkeypatch, payloads)
    result = console._import_skill_repository("https://github.com/example/Nix-skills-repo")
    skill = next(item for item in result["skills"] if item["community"])
    skill_id = skill["id"]
    console._set_skill_installed(skill_id, True)

    state = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
    state["trusted"][skill_id] = "previous-package-digest"
    Path(console.SKILLS_STATE_PATH).write_text(json.dumps(state), encoding="utf-8")
    assert console._check_skill_auto_updates() == {"checked": False, "updated": []}

    console._set_skill_auto_update(skill_id, True)
    changed_manifest = dict(payloads[
        "https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/skill.json"
    ])
    changed_manifest["version"] = "0.2.0"
    payloads["https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/skill.json"] = changed_manifest
    payloads["https://raw.githubusercontent.com/example/Nix-skills-repo/main/skills/web-search/README.md"] = b"updated instructions"

    replacement_commands = []

    def mock_dependency_setup(command, **kwargs):
        replacement_commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[3])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake python", encoding="utf-8")

    monkeypatch.setattr(skill_runtime.subprocess, "run", mock_dependency_setup)
    checked = console._check_skill_auto_updates()
    assert checked == {"checked": True, "updated": [skill_id]}
    # The web-search package is static-only and declares no Python runtime;
    # Ring Light's install-flow test covers actual per-skill venv provisioning.
    assert replacement_commands == []
    saved = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
    assert skill_id not in saved["trusted"]
    assert saved["auto_updates"][skill_id] is True
    package = Path(console._skill_package_path(skill_id))
    assert (package / "README.md").read_bytes() == b"updated instructions"
    installed_manifest = json.loads((package / "skill.json").read_text(encoding="utf-8"))
    assert installed_manifest["version"] == "0.2.0"

    request_count = len(requested)
    assert console._check_skill_auto_updates() == {"checked": False, "updated": []}
    assert len(requested) == request_count


def test_zip_runtime_install_provisions_private_venv_without_worker(skills_state, monkeypatch):
    _repository, manifest = sample_repository()
    manifest["runtime"] = {
        "protocol": "nix-skill-jsonl-v1",
        "entrypoint": "worker.py",
        "requirements": "requirements.txt",
        "autostart": False,
    }
    manifest["files"] = [*manifest["files"], "worker.py", "requirements.txt"]
    manifest["setup_fields"] = []
    manifest["tools"] = [{
        "name": "run_worker",
        "description": "A test-only worker tool.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "result_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }]
    archive = _make_skill_zip(manifest, {
        **{path: b"static" for path in manifest["files"] if path not in {"worker.py", "requirements.txt"}},
        "worker.py": b"raise RuntimeError('must not execute at install')",
        "requirements.txt": b"example-package>=1.0,<2.0",
    })
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[3])
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake python", encoding="utf-8")

    monkeypatch.setattr(skill_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(skill_runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("ZIP installation must not launch worker"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/skills",
            data=json.dumps({"action": "upload_zip", "zip_base64": base64.b64encode(archive).decode()}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode())
        record = next(item for item in payload["skills"] if item["id"] == "web-search" and item["community"])
        assert record["installed"] is True
        assert record["runtime_declared"] is True
        skill_id = record["id"]
        environment = Path(console.DATA_DIR) / "skills" / "runtime_envs" / hashlib.sha256(skill_id.encode()).hexdigest()
        python_path = environment / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        assert python_path.is_file()
        assert [command[1:3] for command in commands] == [["-m", "venv"], ["-m", "pip"]]
        assert all("worker.py" not in command for command in commands)
        assert not runtime_worker_was_started(commands)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_devices_api_reads_only_existing_worker_state(skills_state, monkeypatch, tmp_path):
    from threading import Thread

    skill_id = "github:example/ring-light:ring-light"
    repository = {
        "key": "example/ring-light",
        "url": "https://github.com/example/ring-light",
        "source_type": "github",
        "skills": [{"id": skill_id, "name": "Test Device Skill", "runtime": {"protocol": "nix-skill-jsonl-v1"}, "device_ui": {"status_tool": "read_status", "status_arguments": {"mode": "current"}, "visualization": {"type": "status"}, "fields": [], "controls": []}, "tools": [{"name": "read_status"}]}],
    }
    saved = console._read_skills_state()
    saved["repositories"] = [repository]
    saved["installed"] = [skill_id]
    console._write_skills_state(saved)

    class FakeRuntime:
        def __init__(self):
            self.status_calls = []
            self.device_calls = []

        def status(self, requested):
            self.status_calls.append(requested)
            return {"configured": True, "trusted": True, "worker_running": False, "runnable": True, "fingerprint": "digest"}

        def device_status(self, requested):
            self.device_calls.append(requested)
            return {"device_connected": True, "status": "ok"}

        def start_skill(self, requested):
            self.connected_skill = requested

        def control_device(self, requested, arguments, tool_name=None):
            self.control_call = (requested, arguments, tool_name)
            return {"result": "ok"}

    fake_runtime = FakeRuntime()
    monkeypatch.setattr(console, "_marketplace_skill_runtime", lambda: fake_runtime)
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/devices", timeout=5) as response:
            result = json.loads(response.read().decode())
        assert result["ok"] is True
        assert result["devices"][0]["device_status"] is None
        assert result["devices"][0]["worker_running"] is False
        assert result["devices"][0]["device_label"] == "Ring Light"
        assert result["devices"][0]["device_alias"] == ""
        assert console._devices_payload()["devices"][0]["device_label"] == "Ring Light"
        assert fake_runtime.status_calls == [skill_id, skill_id]
        assert fake_runtime.device_calls == []

        def post(action, arguments=None, **extra):
            request = Request(
                f"http://127.0.0.1:{server.server_address[1]}/api/skills",
                data=json.dumps({"action": action, "skill_id": skill_id, "arguments": arguments, **extra}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode())

        assert post("device_connect")["device"]["device_connected"] is True
        assert fake_runtime.connected_skill == skill_id
        assert post("device_status")["device"]["status"] == "ok"
        assert post("device_control", {"mode": "current"}, tool="read_status")["device"]["result"] == "ok"
        assert fake_runtime.control_call == (skill_id, {"mode": "current"}, "read_status")
        renamed = post("device_name", device_name="Studio Ring")
        assert renamed["ok"] is True
        assert renamed["device_name"] == "Studio Ring"
        assert console._devices_payload()["devices"][0]["device_label"] == "Studio Ring"
        saved = json.loads(Path(console.SKILLS_STATE_PATH).read_text(encoding="utf-8"))
        assert saved["device_names"][skill_id] == "Studio Ring"

        with pytest.raises(console.SkillRepositoryError, match="64 characters"):
            console._set_skill_device_name(skill_id, "x" * 65)
        with pytest.raises(console.SkillRepositoryError, match="control characters"):
            console._set_skill_device_name(skill_id, "bad" + chr(10) + "name")
        assert console._set_skill_device_name(skill_id, "   ") == ""
        assert skill_id not in console._read_skills_state()["device_names"]
        assert "data-device-name-save" in console._dashboard_markup()
        assert "device_label||device.name" in console._dashboard_markup()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_zip_uploaded_skill_never_accepts_auto_updates(skills_state):
    skill, _contents = console._read_skill_zip(_make_skill_zip())
    skill["id"] = skill["slug"]
    repository = {
        "key": f"upload/{skill['id']}", "source_type": "upload", "skills": [skill],
    }
    saved = console._read_skills_state()
    saved["repositories"].append(repository)
    saved["installed"].append(skill["id"])
    console._write_skills_state(saved)
    with pytest.raises(KeyError):
        console._set_skill_auto_update(skill["id"], True)


def test_zip_upload_serves_logo_and_stays_manual_update_only(skills_state):
    _repository, manifest = sample_repository()
    manifest["logo"] = "logo.svg"
    manifest["files"] = [*manifest["files"], "logo.svg"]
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><circle cx=".5" cy=".5" r=".5"/></svg>'
    archive = _make_skill_zip(manifest, {**{path: b"contents" for path in manifest["files"] if path != "logo.svg"}, "logo.svg": svg})
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, body):
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode())

    try:
        response = post("/api/skills", {"action": "upload_zip", "zip_base64": base64.b64encode(archive).decode()})
        skill = next(item for item in response["skills"] if item["id"] == "web-search" and item["community"])
        assert skill["zip_upload"] is True
        assert skill["auto_update_supported"] is False
        assert skill["auto_update"] is False
        assert "if(uploadedSkill)openSkillSettings(uploadedSkill.id)" in console._dashboard_markup()
        request = Request(f"http://127.0.0.1:{server.server_address[1]}{skill['logo_url']}")
        with urlopen(request, timeout=5) as image:
            assert image.headers["Content-Type"] == "image/svg+xml"
            delivered = image.read()
            assert delivered == console._validate_skill_logo("logo.svg", svg)[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_runtime_manifest_rejects_unhashable_schema_values_as_validation_errors():
    package_root = Path(console.CORE_DIR) / "ring_light_package"
    manifest_path = package_root / "skills" / "ring-light" / "skill.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))

    malformed = []
    item = json.loads(json.dumps(source))
    item["setup_fields"][0]["type"] = []
    malformed.append(item)

    item = json.loads(json.dumps(source))
    item["tools"][0]["input_schema"]["type"] = []
    malformed.append(item)

    item = json.loads(json.dumps(source))
    item["device_ui"]["visualization"]["type"] = []
    malformed.append(item)

    item = json.loads(json.dumps(source))
    item["device_ui"]["fields"][0]["format"] = []
    malformed.append(item)

    item = json.loads(json.dumps(source))
    item["runtime"]["autostart"] = True
    malformed.append(item)

    for manifest in malformed:
        with pytest.raises(console.SkillRepositoryError):
            console._parse_skill_manifest(manifest, "skills/ring-light/skill.json", "test")


def test_oversized_post_body_returns_http_413(skills_state):
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request(
            "POST",
            "/api/skills",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(console.SKILLS_MAX_UPLOAD_BYTES * 2 + 128 * 1024 + 1),
            },
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode())
        assert response.status == 413
        assert payload["ok"] is False
        assert "2 MiB upload limit" in payload["error"]
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_runtime_targeting_resolves_state_questions_and_generic_antecedents(skills_state, monkeypatch):
    skill_id = "github:example/ring-light:ring-light"
    runtime = SkillRuntime(
        state_path=skills_state / "skills.json",
        install_dir=skills_state / "installed",
        data_dir=skills_state / "data",
    )
    package = runtime.package_dir(runtime.install_dir, skill_id)
    package.mkdir(parents=True)
    (package / "worker.py").write_text("# mocked test package\\n", encoding="utf-8")
    manifest = {
        "kind": "nix-skill",
        "id": "ring-light",
        "name": "Ring Light",
        "runtime": {"protocol": skill_runtime.PROTOCOL, "entrypoint": "worker.py"},
        "files": ["worker.py"],
        "tools": [{"name": "control_ring", "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "result_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}],
        "triggers": ["ring", "ring light"],
    }
    (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
    (skills_state / "skills.json").write_text(json.dumps({
        "installed": [skill_id],
        "device_names": {skill_id: "Studio Ring"},
    }), encoding="utf-8")
    monkeypatch.setattr(runtime, "status", lambda _skill_id: {
        "configured": True, "trusted": True, "runnable": True, "worker_running": False,
    })

    # State questions route to the skill path for a verified device read
    # instead of a chat-model guess.
    assert runtime.targeted_skill_specs("is the light on")[0]["skill_id"] == skill_id

    # A pronoun state question resolves against recent device turns.
    device_history = [
        {"role": "user", "content": "turn the ring light on"},
        {"role": "assistant", "content": "The ring is on."},
    ]
    assert runtime.targeted_skill_specs("is it off", device_history)[0]["skill_id"] == skill_id

    # A generic pronoun request after a device-state question still targets
    # the only runnable skill instead of falling back to chat.
    assert runtime.targeted_skill_specs("turn it off", [
        {"role": "user", "content": "is the light on"},
        {"role": "assistant", "content": "The ring is off."},
    ])[0]["skill_id"] == skill_id

    # Non-device questions still stay out of the skill path.
    assert runtime.targeted_skill_specs("is the store open?") == []
