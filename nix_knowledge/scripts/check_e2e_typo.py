"""
End-to-end verification of the production path with ASR typos:
the exact robotics-cancellation scenario from
nix_core/logs/requests-2026-10-06.jsonl, but spoken with typos.

Run: NIX_KNOWLEDGE_MODEL_GATE=0 .venv/bin/python scripts/check_e2e_typo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nix_actions"))

from nix_knowledge.engine import KnowledgeEngine  # noqa: E402
from nix_knowledge.needle import KnowledgeNeedle  # noqa: E402
from nix_actions.engine import ActionsEngine  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="nix-e2e-typo-"))
knowledge = KnowledgeEngine(tmp / "k.db")
actions = ActionsEngine(tmp / "a.db", timezone="America/Chicago")
needle = KnowledgeNeedle(
    knowledge,
    timezone="America/Chicago",
    actions_engine=actions,
    load_model=False,
)

failures: list[str] = []


def step(label: str, payload: dict, checks: dict) -> None:
    result = payload.get("result") or {}
    fn = (payload.get("function") or {}).get("name")
    print(f"\n== {label}")
    print(f"   function: {fn}")
    print(f"   result:   {json.dumps({k: result.get(k) for k in ('ok', 'status', 'operation', 'reply')})[:200]}")
    for key, expected in checks.items():
        if key == "function":
            got = fn
        elif key == "status":
            got = result.get("status")
        elif key.startswith("actions."):
            got = (result.get("actions") or {}).get(key.split(".", 1)[1])
        elif key.startswith("result."):
            got = (result.get("result") or {}).get(key.split(".", 1)[1])
        elif key.startswith("data."):
            got = ((result.get("result") or {}).get("data") or result.get("data") or {}).get(key.split(".", 1)[1])
        else:
            got = result.get(key)
        ok = got == expected
        if not ok:
            failures.append(f"{label}: {key} expected {expected!r}, got {got!r}")
        print(f"   check {key}: {'ok' if ok else f'FAIL (got {got!r})'}")


step(
    # the original Oct-6 compound create, as ASR would mangle it
    "typo'd compound create: 'i have robotics practice tomorow at 5pm "
    "and remember that i joined TSA'",
    needle.process(
        "i have robotics practice tomorow at 5pm and remember that i joined TSA"
    ),
    {"status": "multi_action"},
)
step(
    "typo'd cancel news: 'my robotics practice has been cancled'",
    needle.process("my robotics practice has been cancled"),
    {
        "function": "cancel_calendar_event",
        "data.status": "cancelled",
        "actions.cancelled_actions": 1,
    },
)
step(
    "typo'd imperative on empty base: 'cancle my dentist appointment'",
    needle.process("cancle my dentist appointment"),
    {"function": "cancel_calendar_event", "status": "not_found"},
)

# database truth: the event survives with status=cancelled
rows = knowledge.database.execute(
    "SELECT data FROM knowledge WHERE knowledge_type = 'calendar_event'"
).fetchall()
events = [json.loads(r[0] if isinstance(r[0], str) else json.dumps(r[0])) for r in rows]
robotics = [e for e in events if e.get("title") == "robotics practice"]
print(f"\nstored events: {[(e.get('title'), e.get('status')) for e in events]}")
if not robotics or robotics[0].get("status") != "cancelled":
    failures.append(
        f"robotics practice record not cancelled: {robotics}"
    )

print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES:'}")
for f in failures:
    print(f"  - {f}")
sys.exit(1 if failures else 0)
