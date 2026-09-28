import json
from pathlib import Path

import pytest

import console


@pytest.fixture
def skills_state(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "SKILLS_STATE_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setattr(console, "SKILLS_INSTALL_DIR", str(tmp_path / "installed"))
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
