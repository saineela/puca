import json
import sqlite3
import sys
import urllib.request

BASE = "http://127.0.0.1:49152"
DB = "/tmp/p1_e2e/knowledge.db"


def send(text, conv="p1"):
    req = urllib.request.Request(
        BASE + "/api/send",
        data=json.dumps({"text": text, "location": "p1-e2e", "conversation_id": conv}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read() or b"{}")


def rows():
    con = sqlite3.connect(DB)
    try:
        return con.execute(
            "select id, status, data from knowledge where knowledge_type='fact' order by id"
        ).fetchall()
    finally:
        con.close()


def show(label, text, **kw):
    out = send(text, **kw)
    print(f"--- {label}\nUSER: {text}\nROUTE: {out.get('route')} / {out.get('rule')}")
    print("REPLY:", (out.get("reply") or "").replace("\n", " ")[:400])
    return out


for seed in (
    "Remember that I previously lived in Boise.",
    "Remember that my home city is Flagstaff.",
    "Remember that my old address was 14 Willow Lane.",
):
    show("seed", seed)

print("\n== facts after seeding:", [(r[0], r[1]) for r in rows()])

show("D1 stale value", "What city do I live in?")
before = len(rows())
show("unknown", "What is my blood type?")
after = len(rows())
print(f"FACT ROWS before={before} after={after} (unknown must not write)")

show("D2 put-in-for", "Put the optician in for Monday at 10am")

show("D3 due statement", "The renewal is due March 4")
print("== facts after due:", [(r[0], r[1], json.loads(r[2]).get("value")) for r in rows()])

show("D4 delete request", "Delete the note about my old address")
print("== facts after request:", [(r[0], r[1], json.loads(r[2]).get("value")) for r in rows()])
show("D4 confirm", "yes")
print("== facts after confirm:", [(r[0], r[1], json.loads(r[2]).get("value")) for r in rows()])
show("D4 after delete read", "What was my old address?")

show("event check", "When is my optician appointment?")
show("renewal read", "When is the renewal due?")
