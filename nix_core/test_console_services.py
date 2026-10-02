"""The dashboard must expose exactly one listening port.

Sibling Knowledge/Actions APIs are served in-process through the
requests bridge instead of their own sockets, so a browser only ever
talks to the console port.
"""

from threading import Thread

import pytest
import requests

import console
import console_extend


def test_console_uses_fixed_port_across_restarts(monkeypatch, capsys):
    calls = []

    class Server:
        def serve_forever(self):
            raise KeyboardInterrupt

    def fake_build_app(*, host, port):
        calls.append((host, port))
        return Server(), port

    monkeypatch.setattr(console, "build_app", fake_build_app)
    monkeypatch.setattr(console, "_lan_ip", lambda: "192.0.2.10")
    monkeypatch.setenv("NIX_CONSOLE_HOST", "0.0.0.0")
    monkeypatch.setenv("NIX_CONSOLE_PORT", "54321")

    assert console.main() == 0
    assert calls == [("0.0.0.0", 49117)]
    assert "http://127.0.0.1:49117" in capsys.readouterr().out


def test_extended_console_routes_fall_through_to_unchanged_base(monkeypatch):
    route = r"/api/extensions/(?P<name>[a-z]+)"

    def extension_status(handler, match):
        handler._json({"ok": True, "extension": match.group("name")})
        return True

    console_extend.register_route("GET", route, extension_status)
    server = console_extend.base_console.ThreadingHTTPServer(
        ("127.0.0.1", 0), console_extend.ExtendedHandler
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        extension = requests.get(f"{base_url}/api/extensions/ring", timeout=3)
        assert extension.status_code == 200
        assert extension.json() == {"ok": True, "extension": "ring"}

        # This response is provided by the inherited base console route.
        legacy = requests.get(f"{base_url}/api/luna/model", timeout=3)
        assert legacy.status_code == 410
        assert legacy.json()["code"] == "model_retired"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        console_extend.unregister_route("GET", route)


def test_mobile_app_home_copy_routes_and_catalog_are_category_scoped():
    server = console_extend.base_console.ThreadingHTTPServer(
        ("127.0.0.1", 0), console_extend.ExtendedHandler
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        index = requests.get(f"{base_url}/api/mobile/v1", timeout=3)
        assert index.status_code == 200
        assert index.json()["category"] == "Mobile App"
        assert index.json()["mobile_application_included"] is False
        assert index.json()["live_endpoints"] == [
            "GET /api/mobile/v1", "GET /api/mobile/v1/home-copy"
        ]

        response = requests.get(f"{base_url}/api/mobile/v1/home-copy", timeout=3)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store, max-age=0"
        payload = response.json()
        assert payload["category"] == "Mobile App"
        assert payload["catalog_size"] == 150
        assert payload["data"]["copy_id"].startswith("home-")
        assert len(payload["data"]["suggestions"]) == 3

        catalog = requests.get(
            f"{base_url}/api/mobile/v1/home-copy?catalog=1", timeout=3
        )
        variants = catalog.json()["variants"]
        assert len(variants) == 150
        assert len({item["copy_id"] for item in variants}) == 150
        assert len({(item["headline"], item["subtitle"]) for item in variants}) == 150
        assert all(len(item["suggestions"]) == 3 for item in variants)

        invalid = requests.get(
            f"{base_url}/api/mobile/v1/home-copy?catalog=yes", timeout=3
        )
        assert invalid.status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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
