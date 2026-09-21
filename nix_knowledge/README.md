# nix_knowledge

`nix_knowledge` is NIX's durable personal memory and temporal knowledge engine. It stores verified user information, resolves natural-language time, tracks people and changing states, retrieves relevant context, and exposes the Knowledge API used by `nix_core`.

## Responsibilities

- Durable facts, preferences, profile keys, people, and relationships.
- Temporal person states such as being sick, tired, fine, or doing alright.
- Calendar events and reminders with timezone-aware absolute timestamps.
- Supersession of changing values instead of creating duplicate durable facts.
- Clarification support when a relationship is ambiguous, such as multiple sisters.
- Deterministic rules plus neural/model-assisted parsing for difficult language.
- Knowledge digest and memory-block output for Core response composition.
- Bridge calls to `nix_actions` for scheduled reminders.

Knowledge stores memory; it does not write the final conversational answer. `nix_core` receives the structured result and asks Casper to format the response.

## Memory categories

| Category | Examples | Lifecycle |
|---|---|---|
| Profile/fact | Name, preference, goal, allergy | Durable; selected values may supersede older values. |
| Person | Sister, friend, named individual | Durable identity and relationship information. |
| Temporal state | Someone is sick, happy, fine, tired | Time-sensitive; new state supersedes the active state. |
| Calendar event | Meeting, practice, appointment | Durable until cancelled/updated. |
| Retrieval context | Digest, memory block, query result | Generated from current active records. |

Person-state updates are deliberately checked before generic fact creation. For example, “my sister is doing alright now” should update the active temporal state of the matching sister, not create a generic `FACT` record. If multiple people match, the system should ask for clarification and retain that pending continuation through Core.

## Temporal pipeline

```text
user phrase
   |
   v
symbolic temporal resolver + hybrid parser
   |
   v
absolute timezone-aware datetime/range
   |
   v
validated Knowledge operation
   |
   v
stored event/state + action bridge + Core-readable result
```

Supported and regression-tested language includes:

- `tmr`, `tmrw`, `tomorrow`
- `in 2 more days`
- `5 days after today`
- `the day after tomorrow`
- `next Monday morning from 9am to 11am`
- `this Friday at 6:30pm`
- timed `this weekend at 7pm`
- `show my events tomorrow`

The configured timezone is normally `America/Chicago`. Relative phrases must be resolved at request time and absolute local dates/times must be included in results sent to Core.

## API

The HTTP API is implemented by `scripts/knowledge_api.py`. Common endpoints include:

- `GET /health` — service, database, timezone, and model status.
- `POST /process` — classify and execute a Knowledge request.
- `POST /classify` — model/rule route classification.
- `GET /digest` — compact verified context for chat composition.
- `GET /memory_block` — grounded context for Core.
- `GET /keys` — profile/person key retrieval.

Start it from the repository root:

```bash
python nix_knowledge/scripts/knowledge_api.py
```

The database path is controlled by `NIX_KNOWLEDGE_DB` or `NIX_DATA_DIR`. Do not point tests at a personal live database.

## Neural and symbolic parsing

The engine uses a hybrid approach:

1. Deterministic rules recognize high-confidence operations and safety boundaries.
2. Temporal symbolic parsing resolves dates, offsets, ranges, and day parts.
3. Core's deterministic route and Knowledge symbolic rules handle high-confidence requests.
4. Nix_predictor (Qwen 2.5 0.5B) handles indirect and multi-intent requests through constrained function calls.
5. Validation prevents partial model proposals from silently discarding dates, titles, locations, or time ranges.
6. Structured output is returned to Core with absolute timestamps and operation metadata.

The local model files and Casper/QLoRA resources live under `models/` and are ignored by Git. Training and evaluation scripts are under `scripts/nixlm/`.

## Tests

From the repository root:

```bash
(cd nix_knowledge && python -m pytest tests -q)
(cd nix_knowledge && python -m pytest tests/test_temporal.py tests/test_context.py -q)
```

Temporal and model-gate experiments should use an isolated temporary database. The repository’s dashboard testing flow copies the live databases into a sandbox so prompts can exercise the real pipeline without modifying personal data.

## Key directories

- `nix_knowledge/` — package implementation.
- `tests/` — Knowledge, temporal, state, rule, retrieval, and maintenance tests.
- `scripts/` — API service and model/training utilities.
- `models/` — local model artifacts; ignored by Git.
- `pyproject.toml` — package metadata and console entry points.

## Related packages

- [`nix_core`](../nix_core/) requests and formats Knowledge results.
- [`nix_actions`](../nix_actions/) stores and fires linked reminders.
- [`nix_decision`](../nix_decision/) will eventually submit explicit proactive triggers to Core.
