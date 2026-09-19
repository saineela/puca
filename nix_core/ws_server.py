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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import websockets  # noqa: E402

from brain import Brain  # noqa: E402
from config import AUTH_TOKEN, WS_HOST, WS_PORT  # noqa: E402

brain = Brain()


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

            if not text:
                await websocket.send(
                    json.dumps(
                        {"type": "error", "msg": "empty prompt"}
                    )
                )
                continue

            print(f"📥 [Nix Processing Voice Prompt] ({location}): '{text}'")
            await websocket.send(
                json.dumps({"type": "status", "msg": "thinking"})
            )

            # The brain is synchronous (requests + local model calls);
            # keep the socket responsive by running it in a thread.
            response = await asyncio.to_thread(
                brain.handle, text=text, location=location
            )

            await websocket.send(
                json.dumps(
                    {
                        "type": "reply",
                        "msg": response["reply"],
                        "route": response["route"],
                        "rule": response["rule"],
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

    return (
        f"   knowledge API: {knowledge_line}\n"
        f"   actions API:   {actions_line}\n"
        f"   chat model:    {brain.ollama.model} "
        f"@ {brain.ollama.api_url}"
    )


async def main():
    print("🧠 Nix Central Classification Brain Active.")
    print(_service_report())
    print(f"⚡ Real-time WebSockets listening on ws://{WS_HOST}:{WS_PORT}")

    async with websockets.serve(handle_client, WS_HOST, WS_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
