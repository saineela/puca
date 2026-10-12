# Nix Dev / Echo Dot — Official Registration API

**Audience:** the NIX Home builder agent (the code that configures Echo Dot / `Nix Dev` units
over Bluetooth and then registers them with the hub).
**Status:** every endpoint, field, status code and error sentence below was **re-run against the
live hub on 2026-10-09** and is what the server actually does. Where the older server spec (the
"registration API" note you were given) disagrees with this file, **this file wins** — see
[§2 “Route corrections”](#2-route-corrections-read-before-coding).
**Related docs:** `MOBILE_APP_API.md` (mobile blueprint), `README.md` (project overview).

Your unit already advertises, gets discovered, joins Wi-Fi, and gets configured/trusted/started as
the `echo-connect` skill. The **one thing left** is to create a durable hub-side record for it. That
is what this document covers, per unit, in order.

---

## 1. TL;DR — what to call, for each unit

For **every** Nix Dev you configured, do these three calls with the unit's own values:

```bash
BASE=http://<hub-ip>:49117          # same host/port as the rest of the console

# 1. Register / upsert the unit in the console registry  (idempotent on mac)
curl -s -X POST $BASE/api/devices/register -H 'Content-Type: application/json' -d '{
  "mac":"AA:BB:CC:DD:EE:FF",
  "device_id":"<unit-reported id, may be empty>",
  "ip":"192.168.0.55",
  "name":"Nix Dev",
  "model":"Echo Dot 2 (rooted)",
  "device_alias":"Kitchen Dot",
  "owner":"<hub identity this account pairs under>",
  "skill_id":"echo-connect",
  "api_key":"<base64 Noise psk read from the unit>"
}'

# 2. Verify it is listed
curl -s $BASE/api/devices

# 3. (Optional) officially pair it in the mobile registry too — see §5
```

`api_key` (= the BLE `psk`) is write-only: the hub stores only a digest and **never** returns it.
Repeat safely — re-sending the same `mac` updates the record instead of duplicating it.

---

## 2. Route corrections (read before coding)

The older spec assumed the registry create lived at `POST /api/devices` with
`{"action":"register", …}`. **That is not the deployed route.** Verified today:

| You might expect | Actual live behaviour |
|---|---|
| `POST /api/devices` `{"action":"register",…}` | **`404`** → `{"ok": false, "error": "not found"}` |
| `POST /api/devices/register` | **The real upsert route** — use this |
| `GET /api/devices` | Lists skill devices **+** every registry record |
| `GET /api/devices/{identifier}` | One **registry** record — skill devices are list-only (see §4.3) |
| `PATCH /api/devices/{identifier}` | Rename / repoint |
| `DELETE /api/devices/{identifier}` | Forget the unit |

Only these HTTP verbs are served: `GET, POST, PATCH, DELETE, OPTIONS`. The bare
`POST /api/devices` path is not routed, so send the create to `POST /api/devices/register`.

Also note the server runs two **independent** registration stores:

| Surface | Base path | What it is | Entry point |
|---|---|---|---|
| **A. Mobile / official pairing** | `/api/mobile/v1/devices…` | BLE report → claim → official registration, with mock mode | `console_extend.py` |
| **B. Console device registry** | `/api/devices…` | Durable dashboard/console registry (upsert, rename, forget) | `console.py` |

They do **not** share storage. A unit that must be *officially paired* **and** *visible on the
console* needs calls to **both** (§5). Run the hub with:

```bash
cd /root/nix_core
/root/nix_knowledge/.venv/bin/python console_extend.py     # superset of console.py; port 49117, 0.0.0.0
```

If the hub is started with plain `console.py`, Surface A returns `404`. Confirm at startup with
`GET /api/mobile/v1` (must answer `"ok": true`).

**Transport / auth:** plain HTTP on the LAN, **no token or login**. CORS preflight is answered
(`Access-Control-Allow-Origin: *`; methods `GET, POST, PATCH, DELETE, OPTIONS`). Native mobile HTTP
clients ignore CORS. All JSON, UTF-8.

---

## 3. Where the values come from (the unit's identity card)

Your BLE flow already reads the unit's identity card (`NixDevInfo`):

| Identity card | Send to registry as | Notes |
|---|---|---|
| `mac` | `mac` | **Primary key.** Any of `AA:BB:CC:DD:EE:FF`, `aa-bb-cc-dd-ee-ff`, or lower-case hex; normalized to upper-case colon form. |
| `id` | `device_id` | Unit-reported id; may be blank on older units. |
| `name` | `name` | Advertised/human name; defaults to `Nix Dev`. |
| `model` | `model` | Hardware model string; may be blank. |
| `ip` | `ip` | IPv4 the unit joined; blank if not yet on Wi-Fi. |
| `psk` | `api_key` | **Secret.** Base64 Noise key — same value the `echo-connect` skill uses. |
| `owner` | `owner` | Hub identity this account pairs under. Stored as given (see §8 caveat). |
| `paired` | *not sent* | Unit-local claim flag; the hub derives its own `registered` state. |
| — | `device_alias` | Optional friendly name for the console (e.g. `Kitchen Dot`). |
| — | `skill_id` | Defaults to `echo-connect`. |

The hub enforces the same naming rule the app scans for: an advertised name must contain
**`Nix Dev` (with the space)** to be registrable on Surface A. `NixDev` is refused. For Surface B
the `name` is free text (control characters rejected, whitespace collapsed, ≤64 chars).

---

## 4. Surface B — console device registry (the main call)

Storage: `skills.json` → `"device_registry"`, keyed by MAC. Mutations are serialized under a lock;
writes are atomic with mode `0600`.

### 4.1 `POST /api/devices/register` — create / update (upsert, idempotent on `mac`)

Request body — all fields optional, send everything you know:

```json
{
  "mac": "AA:BB:CC:DD:EE:FF",
  "device_id": "nd-001",
  "ip": "192.168.1.50",
  "name": "Nix Dev",
  "model": "Echo Dot 2 (rooted)",
  "device_alias": "Kitchen Dot",
  "owner": "sam",
  "skill_id": "echo-connect",
  "api_key": "<base64 Noise psk>"
}
```

Rules (all verified):

* **Identity:** at least one of `mac` / `device_id` / `ip` is required, else
  `400 {"ok": false, "error": "Registration needs a mac, device_id, or ip so the unit can be tracked."}`
* `mac` is normalized (upper-case, `-`→`:`) and must match `(?:[0-9A-F]{2}:){5}[0-9A-F]{2}`, else
  `400 {"ok": false, "error": "mac must be a MAC address like AA:BB:CC:DD:EE:FF."}`
* Text fields (`name`, `model`, `device_alias`, `owner`, `skill_id`) ≤64 chars, `device_id` ≤80,
  `ip` ≤64; control characters rejected; whitespace collapsed. **Omitted/empty fields keep their
  previous value** on re-upsert (so a partial re-register never wipes data).
* **`api_key` is never stored or echoed.** The hub keeps only
  `sha256("nix-dev-registry:" + api_key)` as an internal `api_key_digest`; a non-string/blank
  `api_key` is a `400`. No response — including `GET` — ever contains the key or the digest.
* **Match order for idempotency:** exact `mac` → same `device_id` → same `ip`.
* **Both create and update return HTTP `200`.** Distinguish with `created: true` (new) or
  `created: false` (updated).
* Stable id: **`echo-connect:<mac-without-colons-lowercased>`**
  (falls back to `device_id`, then `ip`, when `mac` is blank).

Response (`200`):

```json
{
  "ok": true,
  "created": true,
  "device": {
    "id": "echo-connect:aabbccddeeff",
    "name": "Nix Dev",
    "device_alias": "Kitchen Dot",
    "device_label": "Kitchen Dot",
    "icon": "🔊",
    "registered": true,
    "runtime_declared": true,
    "installed": true,
    "configured": false,
    "trusted": false,
    "worker_running": false,
    "owner": "sam",
    "skill_id": "echo-connect",
    "mac": "AA:BB:CC:DD:EE:FF",
    "device_id": "nd-001",
    "ip": "192.168.1.50",
    "model": "Echo Dot 2 (rooted)",
    "device_status": {
      "device_connected": false,
      "state": { "online": false, "ip": "192.168.1.50" }
    },
    "device_ui": {
      "fields": [
        { "label": "IP address", "path": "state.ip" },
        { "label": "Online", "path": "state.online", "format": "boolean" }
      ],
      "controls": [
        {
          "tool": "control_ring",
          "label": "Ring light",
          "arguments": {},
          "fields": [
            { "name": "action", "label": "Action", "widget": "select",
              "options": ["on", "off", "brightness", "color", "effect"] },
            { "name": "brightness", "label": "Brightness", "widget": "number",
              "minimum": 0, "maximum": 1 },
            { "name": "color", "label": "Colour", "widget": "color" },
            { "name": "effect", "label": "Effect", "widget": "select", "options": [] }
          ]
        }
      ]
    }
  }
}
```

`device` is shaped exactly like one element of `GET /api/devices`, so it can be dropped straight
into the app's device list.

### 4.2 `GET /api/devices` — list (skill devices + registry records)

Returns installed device-skills (with live `device_status` when their worker runs) **plus** every
registry record. Registry records carry `"registered": true`, which is how you tell them apart.
Per-device keys the app reads: `id`, `name`, `device_label`, `icon`, `worker_running`,
`runtime_declared`, `configured`, `trusted`, `device_alias`, `device_status`, `device_ui`.
A record is dropped by the app if `id` is blank — the id scheme above guarantees it is not.

### 4.3 `GET /api/devices/{identifier}` — read one **registry** record

`200 {"ok": true, "device": { …same shape as §4.1… }}`;
unknown → `404 {"ok": false, "error": "No such device."}`.

**Accepted identifiers** (URL-encode when it contains `:`): the record `id`
(`echo-connect:aabbccddeeff`), the `mac` (`AA:BB:CC:DD:EE:FF`), or the `device_id`.

> **Caveat (verified):** this path resolves **only the registry store**. A *skill* device id
> (e.g. `github:saineela/echo-connect-skill:echo-connect`) returns
> `404 {"ok": false, "error": "No such device."}` here — read skill devices, and their live
> `device_status`, from the `GET /api/devices` list instead.

### 4.4 `PATCH /api/devices/{identifier}` — rename / repoint

```json
{ "device_alias": "Kitchen Dot" }
```
Also accepts `{"name": "…"}` and `{"ip": "…"}`. Send `"device_alias": ""` to clear the alias (the
label then falls back to `name`). Returns `{"ok": true, "device": { … }}`; unknown → `404`,
validation → `400`. `PATCH` only exists on `/api/devices/{identifier}` — a `PATCH` to
`/api/devices/register` is `404`.

### 4.5 `DELETE /api/devices/{identifier}` — forget a unit

`200 {"ok": true}` on success; repeating it (or an unknown id) →
`404 {"ok": false, "error": "No such device."}`.
Removes **only** the registry entry; installed skills and the mobile record are untouched.

### 4.6 Tested Surface B sequence (copy-paste)

```bash
BASE=http://<hub-ip>:49117
MAC=AA:BB:CC:DD:EE:FF

# create  -> 200, created:true
curl -s -X POST $BASE/api/devices/register -H 'Content-Type: application/json' \
  -d "{\"mac\":\"$MAC\",\"ip\":\"192.168.1.50\",\"name\":\"Nix Dev\",\"owner\":\"sam\",\"api_key\":\"unit-psk-material\"}"

# re-send same mac -> 200, created:false (upsert, no duplicate)
curl -s -X POST $BASE/api/devices/register -H 'Content-Type: application/json' \
  -d "{\"mac\":\"$MAC\",\"device_alias\":\"Kitchen Dot\"}"

# read back / rename / clear alias
curl -s       $BASE/api/devices/echo-connect:aabbccddeeff
curl -s -X PATCH $BASE/api/devices/echo-connect:aabbccddeeff \
  -H 'Content-Type: application/json' -d '{"device_alias":"Reading Nook"}'
curl -s -X PATCH $BASE/api/devices/echo-connect:aabbccddeeff \
  -H 'Content-Type: application/json' -d '{"device_alias":""}'

# forget + confirm
curl -s -X DELETE $BASE/api/devices/echo-connect:aabbccddeeff      # {"ok": true}
curl -s -X DELETE $BASE/api/devices/echo-connect:aabbccddeeff      # 404 No such device.
```

---

## 5. Surface A — official mobile pairing (optional but authoritative)

Surface B makes the unit visible/manageable on the console. Surface A is the **official pairing
ceremony** and the only place `official_count` exists. Do both if the unit must count as officially
paired *and* appear on the console.

### Lifecycle

```
 (none) ──report──► seen ──discover──► seen(claimed) ──register──► registered ──unregister──► retired
```

* `seen` — advertisement acknowledged; dedup by MAC, `rssi`/`last_seen` refreshed per report.
* `registered` — **official**: claim token **and** device secret both verified; token consumed.
* `retired` — unregistered, kept for audit. **Terminal over HTTP** — a retired MAC cannot be
  re-reported or re-discovered (only a server-side/test reset clears it).

State file: `/root/nix_core/data/mobile_devices.json` (override `NIX_MOBILE_DEVICES_PATH`),
atomic, mode `0600` in a `0700` directory. Every response is `Cache-Control: no-store`.
Errors: `{"ok": false, "category": "Mobile App", "error": "<sentence>"}`, HTTP `400` (validation)
or `404` (unknown MAC on GET).

### Endpoints

| Method | Path | Body → result |
|---|---|---|
| `GET` | `/api/mobile/v1` | capability probe / `live_endpoints` |
| `GET` | `/api/mobile/v1/devices` | flow documentation index (no state) |
| `POST` | `/api/mobile/v1/devices/report` | `{mac, advertised_name, rssi?, mock?}` → public device view |
| `POST` | `/api/mobile/v1/devices/discover` | `{mac}` → `claim_token` (16 upper-case hex) |
| `POST` | `/api/mobile/v1/devices/register` | `{mac, claim_token, device_secret, label?, kind?, mock?}` → `status:"registered"` |
| `POST` | `/api/mobile/v1/devices/unregister` | `{mac}` → `status:"retired"` |
| `GET` | `/api/mobile/v1/devices/list` | all devices, strongest signal first, `official_count` |
| `GET` | `/api/mobile/v1/devices/{mac}` | one device; `404` when unknown |
| `POST` | `/api/mobile/v1/devices/mock/seed` | `{suffix?, rssi?}` → fabricates a mock unit **and returns its secret** |

### Validation the hub enforces (Surface A)

* Advertised name must **contain `Nix Dev` with the space** (case-insensitive), 1–26 chars,
  whitespace collapsed, no control characters.
* MAC: 6 hex bytes, normalised to upper-case colon form.
* RSSI: number in **-127..0** (defaults to `-60` when omitted); listings sort strongest-first.
* `kind` ∈ `echo_dot | ring_light | sensor | speaker | other` (default `other`).
* `claim_token` must match `[0-9A-F]{16}`; `device_secret` must be exactly 32 characters.
* Mock and real devices never mix under one MAC (`mock: true` must be consistent across calls).

Representative error sentences (safe to show in UI):
`"That device has not been seen yet. Report its Bluetooth advertisement first (POST /api/mobile/v1/devices/report)."` ·
`"That device is already officially registered. Unregister it first to pair it again."` ·
`"The claim token does not match. Request a fresh one with /api/mobile/v1/devices/discover."` ·
`"The device secret does not match this device. Registration refused."`

### Secret provisioning — read this before the real-hardware flow

`device_secret` (32 hex chars) is minted **by the hub** on the first `report` for a MAC, and is
**only** ever returned by `POST /api/mobile/v1/devices/mock/seed` (mock units). `report`,
`discover`, `list` and `GET /{mac}` all strip it (and the claim token) from their responses.

**Open item for the hardware team:** for real units there is no HTTP channel yet that delivers the
hub-issued secret to the app, so real-hardware Surface A registration cannot complete over HTTP
alone. The same value must arrive out of band — over the BLE provisioning characteristic or shown
on the unit — matching what the hub stored, or the hub must be extended with an authenticated
secret-delivery step. Build against `mock/seed` now and keep the secret field pluggable.

---

## 6. Recommended per-unit procedure

```text
for each Nix Dev you configured and connected:

  1. read its identity card over BLE          -> mac, id, name, model, ip, psk, owner
  2. install/configure/trust/start the echo-connect skill for it
       (already done by the app; configuration.device_ip = ip,
        configuration.api_key = psk)
  3. POST /api/devices/register   (Surface B)  -> 200, created true|false
       body: mac, device_id=id, ip, name, model, owner, skill_id, api_key=psk,
             device_alias=<friendly name>            (optional but nice)
  4. GET /api/devices                          -> confirm the record is listed
  5. optionally repeat from §5 for official pairing (Surface A)
  6. keep the mapping  device.id (echo-connect:<mac>)  ->  unit  for later control
```

Rename later with Surface B `PATCH`; forget the unit with Surface B `DELETE` **and** Surface A
`unregister` (they are separate stores — forgetting only one leaves the other behind).

---

## 7. Control & rename after registration

The same console API the dashboard uses:

* **Control the unit**
  `POST /api/skills` → `{"action":"device_control","skill_id":"<device id>","tool":"control_ring","arguments":{…}}`
  Arguments are **strict — extra keys are rejected**:
  * `{"action":"on"}` · `{"action":"off"}`
  * `{"action":"brightness","brightness":0.0–1.0}`
  * `{"action":"color","rgb":[R,G,B]}`
  * `{"action":"effect","effect":"<name from available_effects>"}`
  * `{"action":"volume","volume":0.0–1.0}` · `{"action":"mute"}` · `{"action":"unmute"}`
  * `{"action":"say","text":"<non-empty>"}` · `{"action":"stop_speaking"}`
  * Voice-tuned speech (declared control "Speak with a tuned voice" — all five keys required):
    `{"action":"say","text":"…","voice":"amy","speed":1.05,"expressiveness":0.7,"variability":0.8}`
    (`voice` ∈ `amy | lessac | hfc_female | kathleen | jenny | alba`)
  * Every `say` result carries a `tts` object: `engine` (`unrealspeech` or `local`), `key_index`
    (1–3, or 0 when the built-in voice spoke), `voice`, `fallback` (why the cloud was skipped),
    `month`, the per-key limits, and one entry per key with `characters`, `seconds`,
    `characters_percent`, `seconds_percent`, and `ready`. The provider choice, up to three Unreal
    Speech API keys, the cloud voice, and the per-key monthly limits are setup fields, not tool
    arguments.
* **Rename the device** (drives the dashboard label and voice routing)
  `POST /api/skills` → `{"action":"device_name","skill_id":"<device id>","device_name":"Kitchen Dot"}`
  → `{"ok":true,"device_name":"Kitchen Dot"}` (≤64 chars; empty clears the alias).
* **Live state** — read the **skill** entry from the `GET /api/devices` list (find the item whose
  `id` is `github:saineela/echo-connect-skill:echo-connect`) and use its `device_status.state`, which
  carries `on / brightness / rgb / effect`. A registry record's `device_status.state` only holds
  `{online, ip}`, and `GET /api/devices/{skill-id}` is a `404` (see §4.3).

---

## 8. Known gaps (do not assume these work)

* **No authentication on the console surface.** The pasted spec asks `DELETE`/`PATCH` to verify the
  caller owns the device via `owner` and return `403`. **That is not implemented** — the console API
  is LAN-only and unauthenticated, so `owner` is stored but never enforced. Do not expose port 49117
  to the internet; TLS and mobile-user auth are future work (`MOBILE_APP_API.md`).
* **Registry records do not mirror live runtime state.** `configured` / `trusted` /
  `worker_running` on a *registry* record are always `false`; they are not synced from the skill
  runtime. Use the corresponding **skill** entry in `GET /api/devices` for live state.
* **One skill instance per unit is required for live control.** The `echo-connect` skill models a
  single `device_ip`/`api_key` per installed instance. The registry can hold many units (distinct
  `mac` → distinct rows), but backing *several* units at once needs the server to instantiate the
  skill per registered device. Treat multi-unit live control as server-side work, not a given.
* **Retired (Surface A) MACs are terminal over HTTP** — re-pairing needs a server-side reset.
* **Surface A real-hardware secret delivery is missing** (see §5).

---

## 9. Acceptance checklist

- [ ] Create uses **`POST /api/devices/register`** (not bare `POST /api/devices`), upserts on `mac`,
      and returns the device record; a repeated `mac` yields `created:false` and **no** duplicate.
- [ ] Every configured unit appears in `GET /api/devices` immediately after registration, with a
      stable, distinct `id` (`echo-connect:<mac>`).
- [ ] `PATCH` renames/clears the alias; `GET /api/devices/{identifier}` reads one back.
- [ ] `DELETE /api/devices/{identifier}` removes it; `GET` no longer lists it; a repeat is `404`.
- [ ] Errors use `{"ok": false, "error": "<sentence>"}` with the right status
      (`400` validation, `404` unknown, `413` oversized body, `500` unexpected).
- [ ] `api_key` (BLE psk) never appears in any response, and is not stored in plaintext.
- [ ] Multiple units (`distinct mac`) can be registered and listed at once.
- [ ] If official pairing is required, Surface A is completed too and `official_count` reflects it.

---

## 10. Quick reference

```
Hub:  cd /root/nix_core && /root/nix_knowledge/.venv/bin/python console_extend.py   # :49117
Probe: GET /api/mobile/v1   -> "ok": true

Console registry (Surface B, the required one)
  POST   /api/devices/register     upsert on mac        -> 200 {ok, created, device}
  GET    /api/devices              skill devices + registry records
  GET    /api/devices/{identifier} id | mac | device_id -> {ok, device}   (registry records only)
  PATCH  /api/devices/{identifier} {"device_alias","name","ip"}
  DELETE /api/devices/{identifier} -> {ok:true}; repeat 404 "No such device."

Mobile pairing (Surface A, optional/official)
  POST /api/mobile/v1/devices/report       {mac, advertised_name, rssi?, mock?}
  POST /api/mobile/v1/devices/discover     {mac} -> claim_token
  POST /api/mobile/v1/devices/register     {mac, claim_token, device_secret, label?, kind?, mock?}
  POST /api/mobile/v1/devices/unregister   {mac}
  GET  /api/mobile/v1/devices/list         official_count
  POST /api/mobile/v1/devices/mock/seed    (only source of a device_secret)

Secret handling
  api_key   -> stored as sha256 digest only; never returned
  device_secret -> returned only by mock/seed; real units need an out-of-band channel
```
