"""
Nix Central Brain - websocket gateway.

Satellite nodes (Orange Pi Zero gateway, room displays, ...) connect
here, authenticate with the shared token, and send voice text. Every
prompt is routed by the nix_core brain:

  knowledge -> nix_knowledge API (durable records, calendar events,
               facts; scheduling propagates into nix_actions)
  chat      -> Ollama model (phi-4 + SearXNG web search) seeded with
               session context and a verified-knowledge digest

Run with the nix_knowledge venv (has websockets + requests):

    ~/nix_knowledge/.venv/bin/python ~/nix_core/ws_server.py

Protocol:
  client -> {"token": "..."}                          first message
  client -> {"text_prompt": "...", "location": "..."}  then prompts
  server -> {"type": "status", "msg": "thinking"}
  server -> {"type": "reply", "msg": "...", "route": "...", "rule": "..."}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets  # noqa: E402

from brain import Brain  # noqa: E402
from config import (  # noqa: E402
    AUTH_TOKEN,
    FOLLOW_UP_MAX_CHARS,
    FOLLOW_UP_MAX_TURNS,
    FOLLOW_UP_TIMEOUT_SECONDS,
    WS_HOST,
    WS_PORT,
)
from followup import FollowUpSession  # noqa: E402

brain = Brain()
WARMUP_STATUS: dict[str, str] | None = None
_EXIT_RE = re.compile(r"^(?:goodbye|bye|stop|that's all|that is all|end session)$", re.I)


async def handle_client(websocket, path=None):
    print("\n📡 [Connection Request]: Satellite node knocking...")

    try:
        auth_message = await websocket.recv()
        try:
            token = json.loads(auth_message).get("token")
        except json.JSONDecodeError:
            token = None

        if not AUTH_TOKEN or token != AUTH_TOKEN:
            await websocket.close(4001, "Unauthorized")
            print("🔴 [Rejected]: bad or missing token.")
            return

        peer = getattr(websocket, "remote_address", ("?", "?"))
        print(f"🟢 [Link Secure]: gateway verified {peer}.")
        conversation_id = str(uuid.uuid4())
        follow_up = FollowUpSession(
            timeout_seconds=FOLLOW_UP_TIMEOUT_SECONDS,
            max_turns=FOLLOW_UP_MAX_TURNS,
            max_chars=FOLLOW_UP_MAX_CHARS,
            conversation_id=conversation_id,
        )
        started = False

        async for message in websocket:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "msg": "malformed json",
                        }
                    )
                )
                continue

            text = str(
                payload.get("text_prompt")
                or payload.get("text")
                or ""
            ).strip()
            location = str(
                payload.get("location", "desk_area")
            )
            start_requested = bool(
                payload.get("start_session")
                or payload.get("wake_word")
                or payload.get("follow_up_start")
            )
            end_requested = bool(payload.get("end_session")) or bool(_EXIT_RE.match(text))

            if not text:
                await websocket.send(
                    json.dumps(
                        {"type": "error", "msg": "empty prompt"}
                    )
                )
                continue

            if not started or start_requested:
                follow_up.start(conversation_id=conversation_id)
                started = True
            elif not follow_up.can_continue():
                await websocket.send(
                    json.dumps({
                        "type": "follow_up_expired",
                        "msg": "Say Casper to start a new conversation.",
                        "conversation_id": conversation_id,
                    })
                )
                continue

            print(f"📥 [Casper Processing Voice Prompt] ({location}): '{text}'")
            await websocket.send(
                json.dumps({"type": "status", "msg": "thinking"})
            )

            # The brain is synchronous (requests + local model calls);
            # keep the socket responsive by running it in a thread.
            session_context = follow_up.context()
            response = await asyncio.to_thread(
                brain.handle,
                text=text,
                location=location,
                conversation_id=conversation_id,
                session_context=session_context,
            )
            follow_up.record("user", text)
            follow_up.record("assistant", response["reply"])
            if end_requested:
                follow_up.end()

            await websocket.send(
                json.dumps(
                    {
                        "type": "reply",
                        "msg": response["reply"],
                        "route": response["route"],
                        "rule": response["rule"],
                        "conversation_id": conversation_id,
                        "follow_up": follow_up.metadata(),
                    },
                    default=str,
                )
            )

            preview = response["reply"].replace("\n", " ")[:140]
            print(
                f"📤 [{response['route']}/{response['rule']}]: {preview}"
            )

    except websockets.exceptions.ConnectionClosed:
        print("🔴 [Pipeline Lost]: Satellite node disconnected.")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  [Handler Error]: {type(exc).__name__}: {exc}")


def _service_report() -> str:
    knowledge = brain.knowledge.health()
    actions = brain.actions.health()

    knowledge_line = (
        f"up ({knowledge.get('knowledge_records', '?')} records)"
        if knowledge
        else "DOWN"
    )
    actions_line = (
        f"up ({actions.get('stats', {}).get('pending', '?')} pending)"
        if actions
        else "DOWN"
    )

    warmup_line = WARMUP_STATUS or {"status": "not_started"}
    return (
        f"   knowledge API: {knowledge_line}\n"
        f"   actions API:   {actions_line}\n"
        f"   chat backend:   {brain.ollama.model} "
        f"@ {getattr(brain.ollama, 'api_url', 'local://transformers')}\n"
        f"   parallel warm-up: {warmup_line}"
    )


async def main():
    global WARMUP_STATUS

    print("🧠 Nix Central Classification Brain Active.")
    # Bind the gateway before model warm-up. Previously the socket was not
    # created until both GPU models finished loading, so clients and the
    # subprocess end-to-end tests saw connection refused during cold start.
    # Requests received during warm-up simply wait in the handler thread.
    server = await websockets.serve(handle_client, WS_HOST, WS_PORT)
    print(f"⚡ Real-time WebSockets listening on ws://{WS_HOST}:{WS_PORT}")

    # Warm Casper and the Knowledge selector concurrently. The two services
    # enforce independent VRAM ceilings; this never duplicates inference for
    # one request and prevents the first user from paying both cold-start costs.
    try:
        WARMUP_STATUS = await asyncio.to_thread(brain.warmup)
    except Exception as exc:  # startup remains available if a model is absent
        WARMUP_STATUS = {"error": f"{type(exc).__name__}: {exc}"}
    print(_service_report())

    try:
        await asyncio.Future()
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
