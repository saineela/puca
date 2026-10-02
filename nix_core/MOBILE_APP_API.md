# NIX Home Mobile App — API and integration blueprint

**Category:** Mobile App  
**Status:** Planning/documentation, except for the explicitly marked home-copy prototype endpoint.  
**Client:** No mobile application is implemented by this document. Do not infer that routes marked *proposed* exist.

NIX Home is the planned mobile companion for the NIX PUCA project. This blueprint gives a future client team a stable direction for screens, API resources, setup, and flows while keeping the existing Core console and APIs as the source of truth. The intended first app is a secure companion UI—not a second assistant backend, database, model runtime, or privileged device controller.

> **Current hosting caveat:** the documented endpoints run on the Core console only when launched with `nix_core/console_extend.py`. A service still launched with `console.py` does not have the extension routes. The endpoint currently runs on the console's fixed port (49117), and is not by itself an internet-safe mobile backend.

## What is implemented now

Only these endpoints are implemented in `console_extend.py` and `mobile_app_api.py`:

| Method | Path | Purpose | Authentication |
| --- | --- | --- | --- |
| `GET` | `/api/mobile/v1` | Public category/status index, implementation status, and link to this document. | None; contains no user data. |
| `GET` | `/api/mobile/v1/home-copy` | Return one random home-screen copy bundle chosen from 150 catalog entries. `Cache-Control: no-store`. | None; contains no user data. |
| `GET` | `/api/mobile/v1/home-copy?catalog=1` | Return all 150 bundles for preview, QA, and copy review. | None; contains no user data. |

The random endpoint returns `category: "Mobile App"`, `api_version: "1"`, `catalog_size: 150`, `selection: "random-per-request"`, and `data`. The catalog response uses `selection: "caller-selects-by-copy_id"` and a `variants` array. Each variant has `copy_id`, `headline`, `subtitle`, `section_title`, and three suggestion items (`icon`, `title`, `description`, `prompt`). There is no account personalization, persistent preference, RNG seed, server-side rotation state, push service, or mobile-specific authentication in this prototype. The dashboard fetches one bundle on page load and retains its built-in copy if the extension/API is unavailable.

Example:

```json
{
  "ok": true,
  "category": "Mobile App",
  "api_version": "1",
  "catalog_size": 150,
  "selection": "random-per-request",
  "data": {
    "copy_id": "home-042",
    "headline": "What would you like to work through?",
    "subtitle": "Take your time. We can start wherever you are.",
    "section_title": "Choose a gentle beginning",
    "suggestions": [
      {"icon": "♡", "title": "Check in with me", "description": "A little space to be heard", "prompt": "Can you check in with me?"},
      {"icon": "☼", "title": "Help me reset", "description": "Find a softer next step", "prompt": "Help me make today feel easier."},
      {"icon": "⌁", "title": "Save a thought", "description": "Keep what matters close", "prompt": "Remember a thought I want to keep."}
    ]
  }
}
```

The sample illustrates the schema; a live response's exact text and options are selected from the code-defined catalog. Treat all returned copy as display text. Do not execute an API-provided prompt automatically; send it only after the user taps its card.

## Proposed application/API boundary

The final public mobile base URL should be a dedicated HTTPS origin, for example `https://<configured-api-host>/api/mobile/v1`. This is a placeholder, not a deployed host. The production mobile API should be a versioned facade/gateway that authenticates and authorizes the user, validates payloads, rate-limits, and calls Core/Knowledge/Actions internally. It must not expose databases, internal service URLs, model filesystem paths, worker processes, ESPHome credentials, or raw administrative console routes.

The existing `/api/send`, `/api/profile`, `/api/sessions`, and related console routes are useful internal capabilities, but they are not a complete mobile security contract. Do not ship the app pointed at an unauthenticated LAN console. Do not embed `NIX_AUTH_TOKEN`, `NIX_OPENAI_API_KEY`, ESPHome PSKs, or other server secrets in a mobile binary. The WebSocket gateway token is not automatically a mobile-user identity system.

### Proposed resource map (not implemented unless marked above)

| Method | Proposed path | Purpose / permission |
| --- | --- | --- |
| `GET` | `/api/mobile/v1` | Version, capabilities, and links to API documentation. Current implementation is a minimal public index. |
| `GET` | `/api/mobile/v1/home-copy` | Home copy rotation. Implemented prototype as above. |
| `GET` | `/api/mobile/v1/bootstrap` | Authenticated profile, capabilities, current companion summary, feature flags, and initial dashboard counts; avoid including full memory. |
| `GET` | `/api/mobile/v1/profile` | Read the authenticated user's app-safe profile and timezone. Proposed. |
| `PATCH` | `/api/mobile/v1/profile` | Update allowlisted profile preferences/name/timezone with validation. Proposed. |
| `GET` | `/api/mobile/v1/companion` | Read active companion/model display status and availability. Proposed; artifact presence must not be described as a running model. |
| `PUT` | `/api/mobile/v1/companion` | Select an allowed companion only if the server runtime supports it. Proposed; never accept filesystem/model paths from clients. |
| `POST` | `/api/mobile/v1/conversations` | Create a conversation ID owned by the authenticated user. Proposed. |
| `GET` | `/api/mobile/v1/conversations?cursor=&limit=` | Page that user's conversations. Proposed; cursor pagination, bounded limits. |
| `GET` | `/api/mobile/v1/conversations/{conversation_id}` | Read one authorized transcript with bounded turns. Proposed. |
| `DELETE` | `/api/mobile/v1/conversations/{conversation_id}` | Delete only that user's conversation history where supported by the retention contract. Proposed and requires explicit confirmation UX. |
| `POST` | `/api/mobile/v1/conversations/{conversation_id}/messages` | Submit one user message to the Core pipeline; returns an assistant reply and request/conversation IDs. Proposed. |
| `GET` | `/api/mobile/v1/conversations/{conversation_id}/messages/{request_id}` | Optional status/polling resource if model responses become asynchronous. Proposed. |
| `GET` | `/api/mobile/v1/memory` | Read a deliberately bounded memory summary for the current user. Proposed and separately permissioned. |
| `GET` | `/api/mobile/v1/memory/records?cursor=&limit=` | Paginated user-owned record index, not an unbounded database dump. Proposed. |
| `GET` | `/api/mobile/v1/memory/records/{record_id}` | Read an authorized record. Proposed. |
| `POST` | `/api/mobile/v1/memory/records` | Create an explicit user-requested memory through Knowledge validation. Proposed; no model-generated arbitrary SQL. |
| `PATCH` | `/api/mobile/v1/memory/records/{record_id}` | Edit a supported user-owned record, if Knowledge defines an edit operation. Proposed. |
| `DELETE` | `/api/mobile/v1/memory/records/{record_id}` | Delete one authorized record and its derived indexes. Proposed; confirm in UI. |
| `GET` | `/api/mobile/v1/events?from=&to=&cursor=` | Read the authenticated user's bounded upcoming/past events. Proposed. |
| `POST` | `/api/mobile/v1/events` | Create an event/reminder through Knowledge/Actions and return confirmed structured state. Proposed. |
| `GET` | `/api/mobile/v1/events/{event_id}` | Read one authorized event and linked action state. Proposed. |
| `PATCH` | `/api/mobile/v1/events/{event_id}` | Reschedule/update through the existing owning service. Proposed. |
| `DELETE` | `/api/mobile/v1/events/{event_id}` | Cancel/delete with explicit confirmation and linked-action semantics. Proposed. |
| `GET` | `/api/mobile/v1/people` | Read explicitly stored people/current-state summaries. Proposed; sensitive and user scoped. |
| `GET` | `/api/mobile/v1/people/{person_id}` | Read one authorized person record. Proposed. |
| `GET` | `/api/mobile/v1/devices` | List only device skills explicitly connected and authorized for this user. Proposed. |
| `GET` | `/api/mobile/v1/devices/{device_id}` | Read live-reported device state with freshness and source metadata. Proposed. |
| `POST` | `/api/mobile/v1/devices/{device_id}/actions` | Execute a declared, schema-validated skill action after a user confirmation. Proposed; deny arbitrary tools/arguments. |
| `GET` | `/api/mobile/v1/notifications/preferences` | Read notification choices. Proposed; not push-delivery capability. |
| `PUT` | `/api/mobile/v1/notifications/preferences` | Update opt-in notification preferences. Proposed. |
| `POST` | `/api/mobile/v1/devices/push-tokens` | Register a platform push token only after an authenticated installation is ready. Proposed. |
| `DELETE` | `/api/mobile/v1/devices/push-tokens/{token_id}` | Revoke one installation's push token. Proposed. |
| `GET` | `/api/mobile/v1/health` | Minimal authenticated capability/readiness status; never expose host, private address, DB path, secrets, or stack traces. Proposed. |

Do not implement every resource by proxying console URLs blindly. Specify ownership and policy for each resource first; reuse Core's existing service layer, not its dashboard's unauthenticated administrative HTTP surface.

## Authentication, sessions, and security requirements

These decisions must be finalized before any private mobile endpoint is exposed:

1. Use a reviewed identity/session design (short-lived access credential plus revocable refresh/session credential, or an established OIDC-compatible flow). Do not invent a long-lived static key embedded in the app. The current repository has no mobile login/account backend.
2. Bind every conversation, memory record, event, person, device, push token, and preference to a server-authenticated user/tenant. The user ID comes from the verified credential, never from an untrusted request-body field.
3. Require TLS; redact bearer/refresh credentials and private payloads from logs; rate-limit authentication, message, and control routes; validate `Origin` where applicable, but do not treat CORS as authentication.
4. Use per-route scopes and least privilege. A chat session must not implicitly authorize memory deletion, system reset, model file selection, arbitrary skill code, or a device command.
5. Protect writes with idempotency keys where retries could duplicate messages, reminders, or device actions. Use request IDs and explicit conflict/validation errors.
6. Bound request/response sizes and pagination, prevent cross-user object-ID access (BOLA), and return stable error codes with no Python traces or secret configuration.
7. Treat voice transcript, notification text, home-copy prompt suggestions, and device state as untrusted data. The user must initiate message submission and explicitly confirm consequential controls.
8. Provide logout/revocation and account-data export/delete policy before release. Data retention, backups, encryption-at-rest, and push provider handling need a separate review.
9. Do not report reminder/device success until Actions/the actual skill returns a confirmed result. For delayed or uncertain operations, return pending/unknown instead of success.

### Proposed response/error envelope

```json
{"ok":true,"data":{},"meta":{"api_version":"1","request_id":"opaque-id"}}
```

```json
{"ok":false,"error":{"code":"validation_error","message":"The request could not be accepted.","request_id":"opaque-id","field_errors":[]}}
```

Proposed status mapping: `400` malformed request, `401` absent/invalid session, `403` insufficient scope, `404` absent or not-owned resource, `409` state/idempotency conflict, `413` body too large, `422` semantic validation failure, `429` rate limited (include `Retry-After`), `5xx` generic retryable/server error. Keep a user's inaccessible object indistinguishable from a nonexistent object where that reduces enumeration risk.

## Screens and API flow

### 1. Welcome / sign in

- Show the NIX PUCA official mark, service origin, privacy notice, and sign-in action.
- Complete the selected authentication flow using system browser/provider UI as appropriate; never ask the user for server secrets or ESPHome keys in general chat.
- On success, store credentials only in the platform's protected credential store. Keep this screen blocked until identity, consent, and API version are known.
- Calls: proposed authentication bootstrap/session endpoints (not implemented); then `GET /bootstrap`.

### 2. Home

- Render greeting/profile name, the rotating home-copy headline/subtitle, three suggestion cards, companion indicator, and small event/memory summaries only when available.
- `GET /api/mobile/v1/home-copy` is currently implemented and public. On transport failure, use a bundled, accessible fallback. Tapping a suggestion fills the composer; it does not send the prompt.
- Production bootstrap/summaries are proposed and must be user-scoped. Avoid fetching a full transcript or full memory block merely to paint the home screen.

### 3. Chat / conversations

- Show conversation list with cursor pagination, loading/empty/error states, and a new-conversation action.
- Open/create a conversation, show bounded transcript, and submit the user-authored message to Core. Keep the conversation ID stable across app restarts and separate across users.
- While generation runs, show pending state and support cancellation only if the server supports cancellation. If asynchronous, poll the request resource with backoff; do not resend the message on a timeout without its idempotency key.
- Show assistant response plus minimal provenance/status where useful. Do not render internal prompts, secret fields, raw tool results, or untrusted HTML.
- Calls: proposed conversations/messages endpoints. Existing dashboard `/api/send` is not this authenticated contract.

### 4. Memory

- Offer a concise user-visible summary, then a searchable, paginated record list and record detail/editor.
- Make creation/edit/delete explicit and explain scope. Individual deletion must update derived indexes; bulk reset is intentionally not part of normal mobile navigation.
- Calls: proposed memory endpoints. The console's `/api/memory`, `/api/kb`, delete, and reset routes are not a mobile permission model.

### 5. Events and reminders

- Calendar/timeline list with date range, timezone display, status chips, create, detail, reschedule, and cancel flows.
- Before creating or changing an event, show the parsed local date/time and ask for correction when ambiguous. Display event and linked action state from owning services.
- On timeout, show pending/unknown and fetch status; never present a guessed success.
- Calls: proposed event endpoints backed by Knowledge and Actions. No client-side scheduler.

### 6. People

- Display only user-stored people and current states with restrained previews, clear privacy affordances, and links to records.
- Do not initiate contact or proactive action. Follow-up eligibility/policy remains server-owned.
- Calls: proposed people endpoints; always user-authorized.

### 7. Devices

- Show devices only when the skill is installed, configured, trusted, explicitly connected, and authorized. Show reported state, last refresh time, connection status, and that status is controller-reported rather than proof of physical output.
- Render only actions declared by the trusted skill manifest. A control tap presents its exact effect and then requires explicit confirmation. No model-proposed background device writes.
- Calls: proposed devices endpoints. Keep ESPHome credentials on the NIX host; the mobile app must never receive them.

### 8. Settings, privacy, and diagnostics

- Profile/timezone, companion choice if supported, notification opt-in, active sessions/devices, logout, export/delete requests, app version, and non-sensitive API status.
- No destructive “reset all” button without a distinct, deliberate confirmation and reauthentication policy.
- Calls: proposed profile/companion/notification/health routes; use current read-only home-copy route independently.

## Setup and environment plan

### Prototype/demo (what exists)

1. Start the Core services required for the current installation.
2. Launch the dashboard through `python nix_core/console_extend.py` so the Mobile App extension routes are registered.
3. Verify `GET /api/mobile/v1` and `GET /api/mobile/v1/home-copy`; inspect the catalog with `?catalog=1`.
4. The existing dashboard consumes one random copy bundle on page load. No mobile build, sign-in, user account, or external deployment is part of this prototype.
5. The endpoint currently shares the console listener and deployment exposure. Bind to loopback while developing; do not expose it to the internet as a mobile backend.

### Future mobile development/deployment (not set up)

- Decide and implement the supported app platforms, repository/build pipeline, signing, app identifier, release channels, crash reporting policy, and privacy/terms before creating a client.
- Provision a dedicated HTTPS API origin, DNS/TLS, authenticated identity/session service, secure secrets manager, rate limits/WAF policy as appropriate, monitored backups, and staging environment. None is created or configured here.
- Add environment configuration only after choosing providers and authorization architecture. Keep server credentials in the server environment; keep mobile public configuration non-secret. Do not add provider keys until required and reviewed.
- Publish an OpenAPI 3.x document generated/validated alongside the actual authenticated API implementation. The tables in this document are a design contract, not an OpenAPI declaration or proof of live routes.
- Add migrations only through the owning Knowledge/Actions services and define user scoping before storing app account data.
- Test with isolated accounts and copied databases. Include auth expiry/revocation, user isolation, pagination, retries/idempotency, offline behavior, accessibility, screen-reader labels, low bandwidth, timezone/DST boundaries, and non-confirmed device/action cases.
- Obtain privacy/security review and product approval before beta distribution. No App Store/Play listing or completed mobile deployment is claimed.

## Versioning and implementation gates

- Current prototype API version is `1`; it covers only the two home-copy routes in this document.
- Future endpoints must be added in a versioned mobile facade, included in an OpenAPI schema, and have auth/scope/error/retention requirements before release.
- Keep the mobile app read-oriented first. Add message writes next only after auth and user isolation are implemented. Add memory/event writes after idempotency and confirmation behavior. Add device actions last, after skill authorization and explicit confirmations.
- A path mentioned under *Proposed resource map* or a future screen is not an available feature until implemented, authenticated, tested, and deployment-verified.
