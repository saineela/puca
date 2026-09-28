"""The dashboard must expose exactly one listening port.

Sibling Knowledge/Actions APIs are served in-process through the
requests bridge instead of their own sockets, so a browser only ever
talks to the console port.
"""

import pytest
import requests

import console


def _host(url: str) -> str:
    return url.split("://", 1)[1]


def test_embedded_siblings_run_in_process_and_open_no_port(monkeypatch):
    monkeypatch.setattr(
        console,
        "ThreadingHTTPServer",
        lambda *args, **kwargs: pytest.fail("embedding must not bind a port"),
    )

    knowledge_url = console._embed("knowledge_api")

    assert knowledge_url.startswith("http://nix-")
    assert "127.0.0.1" not in knowledge_url
    assert console._local_services[_host(knowledge_url)] is not None

    response = requests.get(f"{knowledge_url}/definitely-not-a-route", timeout=10)
    assert response.status_code == 404
    assert response.json() == {"ok": False, "error": "not found"}


def test_in_process_bridge_preserves_query_params_and_json_bodies(
    monkeypatch, tmp_path
):
    actions_url = console._embed("actions_api")
    module = console._local_services[_host(actions_url)]
    monkeypatch.setattr(module, "ACTIONS_DB", str(tmp_path / "actions.db"))
    monkeypatch.setattr(module, "CORE_DB", str(tmp_path / "core.db"))
    monkeypatch.setattr(module, "_actions", None)
    monkeypatch.setattr(module, "_core", None)

    response = requests.get(
        f"{actions_url}/context", params={"limit": 4}, timeout=10
    )
    assert response.status_code == 200
    assert "turns" in response.json()

    response = requests.post(
        f"{actions_url}/log",
        json={"role": "user", "content": "bridge check", "refs": {"source": "test"}},
        timeout=10,
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
