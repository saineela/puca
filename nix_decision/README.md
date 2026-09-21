# nix_decision

`nix_decision` is the foundation for future proactive behavior in Casper/NIX. It is intentionally separate from `nix_core`: decision logic may detect an eligible future trigger, but `nix_core` owns the user-facing conversation and final wording.

## Current scope

- Accept explicit future-programmable `DecisionTrigger` objects.
- Carry a validated `UserStateSnapshot` containing:
  - presence: `home`, `away`, or `unknown`
  - optional location
  - energy: `tired`, `not_tired`, or `unknown`
  - confidence and observation time
- Refuse proactive dispatch when the user is away or unavailable.
- Start with **no registered calling rules**.
- Pass structured trigger and state context through `CoreBoundary`.
- Allow a future rule to decline to speak by returning `None`.

## Important safety boundary

The engine does not infer that the user is available, invent a trigger, diagnose tiredness, or generate speech. A future producer must explicitly provide the trigger and state. A future rule must explicitly be registered. Only then can a request be handed to `nix_core`.

Being tired is currently context for Core to use when deciding how gently and briefly to respond; it is not itself a calling rule.

## Example future integration

```python
from nix_decision import DecisionEngine, UserStateSnapshot, Presence

engine = DecisionEngine(core_callback)

# Future code may register a specific rule here. No rules are installed today.
engine.evaluate(trigger, UserStateSnapshot(presence=Presence.HOME))
```

The callback receives a `CorePromptRequest` directly, or a serialized structured payload when using `CoreBoundary`. It should be connected to an explicit `nix_core` API rather than importing Core internals.

## Tests

```bash
PYTHONPATH=. python -m pytest -q nix_decision/nix_decision/test_engine.py
```
