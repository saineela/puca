# nix_decision

`nix_decision` is the foundation for future proactive behavior in NIX/Casper. It is intentionally separate from `nix_core`: this package may evaluate explicit future triggers, while Core remains responsible for deciding what to say and generating the final response.

## Current status

This is a foundation only. **No calling rules are registered.** The engine will not invent a trigger, infer availability, diagnose the user, or contact Core on its own.

## Safety contract

A future proactive interaction must satisfy every boundary:

1. A producer explicitly submits a `DecisionTrigger`.
2. A `UserStateSnapshot` explicitly describes the user’s current state.
3. The user must be known to be home (`Presence.HOME`). Away or unknown presence is a hard no-op.
4. A future rule must be explicitly registered for that trigger.
5. The rule must intentionally return a `CorePromptRequest`.
6. Core receives structured context and owns the final user-facing wording.

Tiredness is currently context for Core to use when deciding how gently and briefly to respond. It is not an automatic calling rule.

## State model

`UserStateSnapshot` contains:

- `presence`: `home`, `away`, or `unknown`
- `location`: optional room/location label
- `energy`: `tired`, `not_tired`, or `unknown`
- `confidence`: validated from `0.0` to `1.0`
- `observed_at`: optional timestamp

This is an observation snapshot, not a medical diagnosis.

## Trigger and Core boundary

```python
from nix_decision import (
    DecisionEngine,
    DecisionTrigger,
    Presence,
    UserStateSnapshot,
)

engine = DecisionEngine(core_callback)

# No rules are installed yet. Future code will explicitly register one.
outcome = engine.evaluate(
    DecisionTrigger("future_programmable_event", payload={"source": "sensor"}),
    UserStateSnapshot(presence=Presence.HOME, location="living_room"),
)
```

`CoreBoundary` serializes the trigger, payload, user state, and rule instruction into a narrow structured handoff. It should eventually connect to an explicit Core API rather than importing Core internals.

## Package files

- `user_state.py` — validated presence/location/energy snapshot.
- `triggers.py` — trigger, prompt-request, and outcome contracts.
- `engine.py` — rule registry, home gate, evaluation, and dispatch.
- `core_boundary.py` — structured handoff adapter to Core.
- `test_engine.py` — no-rule, state-gate, dispatch, and no-action tests.

## Tests

From the repository root:

```bash
PYTHONPATH=. python -m pytest -q nix_decision/nix_decision/test_engine.py
```

## Future work

The next phase should define trigger registration and deduplication policy before adding real calling rules. Any future rule should also consider conversation cooldowns, user fatigue, quiet hours, confidence thresholds, and whether Core already has an active follow-up conversation.
