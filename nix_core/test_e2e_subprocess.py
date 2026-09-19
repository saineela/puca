"""
Subprocess end-to-end tests: real OS processes, real ports, real
HTTP and WebSocket traffic - no in-process fakes.

Boots, as the deployment would:
  - knowledge_api.py  (subprocess, hermetic temp DB, model gate ON)
  - actions_api.py    (subprocess, hermetic temp DBs)
  - ws_server.py      (subprocess, wired to the two above)

then drives them over the network exactly like the Orange Pi gateway:
token auth, text prompts, routes, replies. The local Qwen model and
the live Ollama server are used for real; tests skip gracefully if
Ollama is unreachable.

Nothing here touches the live databases: every service runs against
fresh files in a temp directory.

Run:
    cd ~/nix_core && ~/nix_knowledge/.venv/bin/python -m pytest \
        test_e2e_subprocess.py -v
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

import pytest

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CORE_DIR)
VENV_PY = os.environ.get("NIX_TEST_PYTHON", sys.executable)
KNOWLEDGE_API = os.path.join(REPO_ROOT, "nix_knowledge", "scripts", "knowledge_api.py")
ACTIONS_API = os.path.join(REPO_ROOT, "nix_actions", "scripts", "actions_api.py")
WS_SERVER = os.path.join(CORE_DIR, "ws_server.py")
AUTH_TOKEN = os.environ.get("NIX_AUTH_TOKEN", "test-token")

OLLAMA_HOST = os.environ.get("NIX_OLLAMA_HOST", "192.168.0.154")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _http(method: str, url: str, body: dict | None = None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{OLLAMA_HOST}:11434/api/tags", timeout=2
        ):
            return True
    except Exception:
        return False


def _wait_http(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            _http("GET", url, timeout=3)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.4)
    raise RuntimeError(f"service never came up at {url}: {last_error}")


def _db_rows(path: str, sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Session fixture: three real subprocesses, hermetic DBs
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster():
    temp = tempfile.TemporaryDirectory(prefix="nix-e2e-")
    root = temp.name

    knowledge_db = os.path.join(root, "knowledge.db")
    actions_db = os.path.join(root, "actions.db")
    core_db = os.path.join(root, "nix_core.db")
    for path in (knowledge_db, actions_db, core_db):
        sqlite3.connect(path).close()  # create empty files

    k_port, a_port, w_port = _free_port(), _free_port(), _free_port()

    base_env = {
        **os.environ,
        "NIX_KNOWLEDGE_DB": knowledge_db,
        "NIX_ACTIONS_DB": actions_db,
        "NIX_CORE_DB": core_db,
        "NIX_KNOWLEDGE_API_URL": f"http://127.0.0.1:{k_port}",
        "NIX_ACTIONS_API_URL": f"http://127.0.0.1:{a_port}",
        "NIX_TZ": "America/Chicago",
        # model gate explicitly ON: we are testing it
        "NIX_KNOWLEDGE_MODEL_GATE": "1",
        "NIX_AUTH_TOKEN": AUTH_TOKEN,
    }

    procs: list[subprocess.Popen] = []
    logs: dict[str, list] = {}

    def _spawn(name, script, extra_env):
        env = {**base_env, **extra_env}
        proc = subprocess.Popen(
            [VENV_PY, "-u", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=CORE_DIR,
        )
        procs.append(proc)
        logs[name] = (proc, [])

    try:
        _spawn(
            "knowledge", KNOWLEDGE_API,
            {"NIX_KNOWLEDGE_API_PORT": str(k_port)},
        )
        _spawn(
            "actions", ACTIONS_API,
            {"NIX_ACTIONS_API_PORT": str(a_port)},
        )
        _spawn(
            "ws", WS_SERVER,
            {
                "NIX_WS_HOST": "127.0.0.1",
                "NIX_WS_PORT": str(w_port),
            },
        )

        _wait_http(f"http://127.0.0.1:{k_port}/health")
        _wait_http(f"http://127.0.0.1:{a_port}/health")
        time.sleep(1.0)  # let the ws server finish binding

        yield {
            "knowledge": f"http://127.0.0.1:{k_port}",
            "actions": f"http://127.0.0.1:{a_port}",
            "ws_port": w_port,
            "knowledge_db": knowledge_db,
            "actions_db": actions_db,
            "core_db": core_db,
            "logs": logs,
            "procs": procs,
        }
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(0.8)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for name, (proc, _) in logs.items():
            if proc.poll() not in (0, None):
                print(f"\n--- {name} exit code {proc.poll()} ---")
        temp.cleanup()


def _ws_session(cluster, token=AUTH_TOKEN):
    """Open an authenticated websocket, yield it, close on exit."""
    from websockets.sync.client import connect

    ws = connect(
        f"ws://127.0.0.1:{cluster['ws_port']}",
        open_timeout=10,
        close_timeout=5,
    )
    ws.send(json.dumps({"token": token}))

    if token != AUTH_TOKEN:
        # server rejects: expect close code 4001
        try:
            ws.recv(timeout=5)
            raise AssertionError("bad token was accepted")
        except Exception as exc:
            code = getattr(getattr(exc, "rcvd", None), "code", None)
            ws.close()
            assert code == 4001, f"expected 4001, got {code}"
            return None

    return ws


def _ask(cluster, text: str, expect_reply_timeout: float = 120.0):
    ws = _ws_session(cluster)
    try:
        ws.send(json.dumps({"text_prompt": text, "location": "e2e-test"}))
        # first frame: {"type": "status", "msg": "thinking"}
        status = json.loads(ws.recv(timeout=10))
        assert status.get("type") == "status", status
        deadline = time.time() + expect_reply_timeout
        while True:
            remaining = max(1.0, deadline - time.time())
            frame = json.loads(ws.recv(timeout=remaining))
            if frame.get("type") == "reply":
                return frame
    finally:
        ws.close()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_knowledge_api_health(cluster):
    health = _http("GET", f"{cluster['knowledge']}/health")
    assert health["ok"] is True
    assert health["knowledge_records"] == 0  # hermetic DB


def test_actions_api_health(cluster):
    health = _http("GET", f"{cluster['actions']}/health")
    assert health["ok"] is True
    assert health["service"] == "nix_actions"


def test_ws_rejects_bad_token(cluster):
    _ws_session(cluster, token="WRONG_TOKEN")


def test_ws_happy_path_knowledge(cluster):
    """
    Full deterministic path over the wire: store -> reply -> row in
    the hermetic knowledge DB -> session turn logged via actions API.
    """
    marker = f"e2e locker code {int(time.time()) % 100000}"
    frame = _ask(cluster, f"remember that my {marker} is 4471")

    assert frame["route"] == "knowledge", frame
    assert "Stored:" in frame["msg"], frame
    assert "4471" in frame["msg"]

    rows = _db_rows(
        cluster["knowledge_db"],
        "SELECT data FROM knowledge WHERE knowledge_type = 'fact'",
    )
    assert any("4471" in json.dumps(row) for row in rows)

    turns = _http(
        "GET", f"{cluster['actions']}/context?limit=10"
    )["turns"]
    assert any(marker in t["content"] for t in turns)


def test_ws_chat_route_live_ollama(cluster):
    if not _ollama_up():
        pytest.skip(f"ollama unreachable at {OLLAMA_HOST}")

    frame = _ask(cluster, "what's the capital of australia, one word")
    assert frame["route"] == "chat", frame
    assert "canberra" in frame["msg"].lower(), frame


def test_model_gate_understands_indirect_intent(cluster):
    """
    The showcase, in two layers:

    1. 'just so you know, my spare key is under the mat' - the
       deterministic layers now resolve it (incidental-note rule);
       the full pipeline through the websocket must store it and
       answer cleanly.
    2. 'would you mind keeping track that i owe joel twenty bucks' -
       no lexical marker anywhere; the Qwen gate must recognize the
       implied storage intent (create_fact) on its own.
    """
    # 1. full pipeline, deterministic layers
    frame = _ask(
        cluster, "just so you know, my spare key is under the flowerpot"
    )
    assert frame["route"] == "knowledge", frame
    reply = frame["msg"].lower()
    assert "pot" in reply or "flowerpot" in reply, frame

    rows = _db_rows(
        cluster["knowledge_db"],
        "SELECT data FROM knowledge WHERE knowledge_type = 'fact'",
    )
    assert any("flowerpot" in json.dumps(row) for row in rows)

    # 2. pure model intent, no rule marker
    verdict = _http(
        "POST",
        f"{cluster['knowledge']}/classify",
        {"text": "would you mind keeping track that i owe joel twenty bucks"},
        timeout=120,
    )
    assert verdict["route"] == "knowledge", verdict
    assert verdict["reason"].startswith("model_selected"), verdict
    # The model must understand STORAGE intent; which mutation it
    # names (create_fact / update_calendar_event) is its choice.
    assert verdict["function"] in {
        "create_fact", "update_calendar_event",
    }, verdict


def test_services_survive_garbage(cluster):
    """Malformed WS JSON and empty prompts must not kill anything."""
    ws = _ws_session(cluster)
    try:
        ws.send("this is not json")
        frame = json.loads(ws.recv(timeout=10))
        assert frame["type"] == "error", frame

        ws.send(json.dumps({"text_prompt": "   "}))
        frame = json.loads(ws.recv(timeout=10))
        assert frame["type"] == "error", frame
    finally:
        ws.close()

    # and the services are still healthy afterwards
    assert _http("GET", f"{cluster['knowledge']}/health")["ok"]
    assert _http("GET", f"{cluster['actions']}/health")["ok"]
