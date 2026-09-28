# NIX PUCA Roadmap

**Project:** [NIX PUCA](https://github.com/saineela/puca) · **Overview:** [README](README.md)

This page describes the intended product direction, not a release promise. Items marked **planned** are not available as working NIX features today. There are no delivery dates or Play Store listing to announce. Progress should be reflected here only after a feature has an implementation, tests, documentation, and a clearly described privacy/security model.

## Current foundation

The repository currently provides a modular software foundation:

- **NIX Core** routes requests, maintains conversation context, and composes user-facing responses.
- **NIX Knowledge** stores local personal facts, people, changing states, and calendar records, and resolves temporal expressions.
- **NIX Actions** captures and schedules deterministic actions, including reminders linked to Knowledge records.
- **NIX Decision** contains an explicit-trigger and home-presence-gate foundation; no proactive calling rules are registered.
- A local dashboard and an OpenAI-compatible chat-completions API are available from the Core console.
- The console's community Skills marketplace can inspect public GitHub manifests and save declared files as **static packages**. Installed files are not executed. The built-in Web Search entry is a catalog placeholder, not a working search connector.

Model weights are not distributed with the repository. Model/backend selection is local configuration, not evidence that a hosted deployment uses a particular model. See the [README's model and performance notes](README.md#models-and-inference).

## Planned architecture: Nix-Skills

The product plan is to define **Nix-Skills** as the capability/connector layer and have **Nix-Actions** manage skill invocation alongside scheduled actions. The existing Nix-Actions package is currently a deterministic scheduler; it does **not** yet run Skills. The current marketplace only imports bounded static files and intentionally does not execute community code.

Before any skill can run, the project needs a reviewed execution contract: explicit capabilities and user consent, least-privilege access, input/output validation, timeouts, audit logs, error isolation, revocation, and safe upgrade/uninstall behavior. A skill must not gain access to personal databases, location, microphones, network services, or hardware merely because its static files were installed.

## Feature roadmap

| Feature | Status | Intended purpose and dependencies |
| --- | --- | --- |
| Nix-Skills execution and permission model | **Planned · prerequisite** | Define how a skill is reviewed, granted capabilities, invoked by Nix-Actions, monitored, disabled, and removed. Static marketplace installation is not executable skill support. |
| Open-source Web Search skill | **Planned** | Let NIX retrieve public web information through a reviewed open-source search connector. Search provider, deployment, source/citation format, query privacy, and permission controls are not finalized. The current Web Search catalog item does not search the web. |
| Nix App Access skill / official NIX PUCA Android app | **Planned** | Provide phone access through an Android client and authenticated NIX API, with Nix-Actions eventually invoking a permissioned Nix-Skills capability. App identity, auth/session design, release process, and Play Store listing remain future work; no official Play Store app is available from this repository today. |
| Real-time voice system | **Planned** | Support live spoken conversations, selectable/configurable inference backends, and user-customizable interaction/voice settings. The current WebSocket gateway transports text; this repository does not yet provide a complete streaming speech-recognition and speech-synthesis product. |
| ESP32 presence detection | **Planned · hardware integration** | Use opt-in ESP32 sensor events to estimate when a person enters or occupies configured rooms. Sensors, firmware, room-level accuracy, event authentication, and retention policy are not implemented here. |
| OPNsense DNS/home-network presence skill | **Planned · presence dependency** | Use a reviewed OPNsense or home-network signal to estimate whether a specifically authorized device is connected to the user's home network. The DNS/DHCP/API signal is not yet selected and needs design and testing. This is a home/away signal, not precise indoor location. |
| Geolocation / timeline integration | **Planned · depends on Android app and presence controls** | With explicit per-user opt-in, send phone location to NIX or a compatible timeline system. Reitti is a candidate integration, not a current dependency or verified NIX integration. The intended policy is to enable location transfer only while the user/device is confirmed on the configured home Wi-Fi through the presence feature. That gate, consent UX, minimization, retention, export, and deletion behavior must be implemented and tested before location data is collected. |
| Echo Dot (2nd generation) interface | **Planned · hardware research** | Investigate an Echo Dot Gen 2 as a physical voice endpoint for NIX. [EchoLocal](https://github.com/ygelfand/echolocal) is an independent third-party project and a possible starting point, not a NIX dependency, endorsement, or completed integration. Hardware, firmware changes, account/cloud behavior, and security constraints require review. |

### Suggested sequencing

1. **Establish the skill contract:** permissions, signed/validated requests, least privilege, auditability, isolation, revocation, and a clear distinction between installing a package and running a capability.
2. **Add a low-risk connector:** implement a read-only Web Search skill only after choosing and documenting a provider, source attribution, network behavior, and failure handling.
3. **Build the phone access layer:** define account/device enrollment, token lifecycle, API scopes, user controls, and a threat model before publishing an app.
4. **Add voice as a separate interface:** specify speech recognition, synthesis, streaming, model selection, latency, and microphone controls without weakening the existing Core/Knowledge/Actions boundaries.
5. **Introduce presence sources:** implement opt-in, authenticated OPNsense DNS/home-network and ESP32 adapters; represent uncertain/stale signals explicitly rather than silently treating them as truth.
6. **Gate location and hardware integrations:** connect Reitti or another timeline service and investigate Echo Dot support only after their consent, network, device, and deletion controls are testable.

This order is a proposal, not a release schedule. A dependency or privacy review may change it.

## Privacy and safety requirements for planned capabilities

Personal location, room presence, microphone audio, and device identifiers are sensitive data. Before a related feature is considered implemented, its design should include:

- Clear, affirmative opt-in per person, device, and data category; no collection merely because a device is installed or on the network.
- Visible status and an immediate way to pause, revoke, inspect, export, and delete collected data.
- Data minimization: store or transmit only what the feature needs, with documented retention and a deletion path.
- Local-first processing where practical; identify every third-party service and what data it receives.
- Authenticated, scoped device/API credentials, secure transport, replay-resistant event handling, and rotation/revocation procedures.
- Explicit uncertainty, stale-signal handling, and safe behavior when presence or sensor events are unavailable.
- Sandboxed or narrowly permissioned skill execution; no running arbitrary downloaded repository code.
- Tests on synthetic data and isolated databases before any live integration.

The existing Decision package has no calling rules. Presence detection or a new Skill must not silently activate proactive messages until a separate policy explicitly enables and tests that behavior.

## Project terms

- **PUCA:** *Personal User Companion Agent*, the project term for a personal assistant designed around one user's authorized context and controls.
- **Nix-Skills:** The planned capability/connector format and runtime, not yet an executable feature of this repository.
- **Nix-Actions:** The current deterministic scheduler; planned to manage skill invocation after a safe skill contract exists.
- **Presence detection:** An estimate that an authorized person/device is home or in a configured room. It is not identity proof or precise location by default.
- **Local-first:** The architecture favors local storage and local services; optional configured backends and future integrations may still communicate over a network.
- **Natural conversational style:** A qualitative design goal for concise, grounded, context-aware replies. It does not mean the model is human, that speech output is currently implemented, or that a standardized human-likeness score has been established.

## References for planned integrations

- [Reitti](https://github.com/dedicatedcode/reitti) — independent self-hosted location-history/timeline project being considered as an integration target.
- [EchoLocal](https://github.com/ygelfand/echolocal) — independent Echo Dot (2nd generation) / Home Assistant project being considered for hardware research.
- [GitHub repository topics documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics) — guidance for adding accurate repository topics.

External projects are independently maintained. Review their licenses, security model, device requirements, and behavior before integrating them.
