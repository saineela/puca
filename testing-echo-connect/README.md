# testing-echo-connect

Optional integration experiments for connecting NIX to an Echo/Home Assistant-style environment.

This folder is **not required** by the core NIX stack. It is isolated from `nix_core`, `nix_knowledge`, `nix_actions`, and `nix_decision` so experiments here cannot change the main conversational or memory pipeline unless explicitly wired in.

## Contents

- `dashboard_server.py` — prototype dashboard/integration server.

## Usage

Inspect the script configuration and provide any required integration credentials through environment variables. Do not commit credentials, device tokens, webhook URLs, or local runtime databases.

```bash
python testing-echo-connect/dashboard_server.py
```

The prototype may require external device services or network access. It should not be used as a dependency of the main NIX services until its authentication, error handling, and integration tests are complete.

## Relationship to NIX

The intended future flow is:

```text
Echo/Home Assistant event -> explicit integration adapter -> nix_decision trigger
                                              |
                                              v
                                      gated nix_core response
```

No automatic calling rules are defined here today. User presence and state must be supplied through the future `nix_decision` contract before Core can proactively speak.
