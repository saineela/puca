import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Configuration ──────────────────────────────────────────────
HA_URL = os.environ.get("HA_URL", "http://127.0.0.1:8123")
HA_TOKEN = os.environ["HA_TOKEN"]

ECHO_ASSIST = os.environ.get(
    "ECHO_ASSIST_ENTITY",
    "assist_satellite.echo_dot_d9785c_assist_satellite",
)
ECHO_MEDIA = os.environ.get(
    "ECHO_MEDIA_ENTITY",
    "media_player.echo_dot_d9785c_speaker",
)

HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

app = FastAPI(title="Echo Dot Dashboard API")


# ── Feature 1 & 2: Speak + listen, or listen-only ──────────────
class AskRequest(BaseModel):
    text: str = ""              # empty = listen only, no speech
    timeout: int = 15
    preannounce: bool = True    # chime before listening

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Mode A (text provided): Dot speaks `text`, listens, returns transcript.
    Mode B (text empty):    Dot silently opens mic, returns transcript.
    """
    async with httpx.AsyncClient(timeout=req.timeout + 10) as client:
        payload = {
            "entity_id": ECHO_ASSIST,
            "question": req.text,
            "answers": [],
            "preannounce": req.preannounce,
        }
        resp = await client.post(
            f"{HA_URL}/api/services/assist_satellite/ask_question",
            headers=HEADERS,
            json=payload,
        )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)

        result = resp.json()
        if isinstance(result, list) and result:
            answer = result[0].get("response", {}) or {}
            return {
                "spoken": req.text,
                "transcript": answer.get("sentence", ""),
                "matched_id": answer.get("id"),
                "slots": answer.get("slots", {}),
            }
        return {"spoken": req.text, "transcript": "", "matched_id": None, "slots": {}}


# ── Feature 3: Just speak (no listening) ───────────────────────
class SpeakRequest(BaseModel):
    text: str

@app.post("/speak")
async def speak(req: SpeakRequest):
    """Make the Dot speak `text` without opening the microphone."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{HA_URL}/api/services/tts/speak",
            headers=HEADERS,
            json={
                "media_player_entity_id": ECHO_MEDIA,
                "message": req.text,
                "cache": True,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, resp.text)
        return {"spoken": req.text}


# ── Health ─────────────────────────────────────────────────────
@app.get("/status")
async def status():
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{HA_URL}/api/", headers=HEADERS)
        ha_ok = r.status_code == 200

        sat = await client.get(
            f"{HA_URL}/api/states/{ECHO_ASSIST}", headers=HEADERS
        )
        mp = await client.get(
            f"{HA_URL}/api/states/{ECHO_MEDIA}", headers=HEADERS
        )

    return {
        "ha_reachable": ha_ok,
        "assist_satellite_state": sat.json().get("state") if sat.status_code == 200 else None,
        "media_player_state": mp.json().get("state") if mp.status_code == 200 else None,
        "entities": {"assist": ECHO_ASSIST, "media": ECHO_MEDIA},
    }
