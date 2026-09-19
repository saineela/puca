from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable

from .engine import Action


Handler = Callable[[Action], Any]


# ----------------------------------------------------------------------
# Device / hardware API functions
#
# These are the integration points for real hardware. Replace the stub
# bodies with actual device SDK calls (GPIO, MQTT, Home Assistant,
# speaker APIs, etc). Each handler receives the full Action and must
# raise on failure so the engine can record it.
# ----------------------------------------------------------------------


def _log(action: Action, message: str) -> None:
    print(
        f"[nix_actions] {action.action_type}#{action.id} {message}"
    )


def handle_alarm(action: Action) -> None:
    """
    Trigger a physical alarm / wake-up.

    Expected payload:
      message  - what to say/show when the alarm fires
      title    - label of the alarm
      device   - optional target device identifier
    """
    message = action.payload.get(
        "message",
        action.payload.get("title", "Alarm"),
    )
    device = action.payload.get("device", "default")

    # TODO: replace with real hardware call, e.g.:
    #   devices.play_alarm(device, message= message)
    _log(
        action,
        f"ALARM on device '{device}': {message}",
    )


def handle_reminder(action: Action) -> None:
    """
    Deliver a reminder to the user.

    Expected payload:
      message  - reminder text
      device   - optional target device identifier
    """
    message = action.payload.get("message", "Reminder")
    device = action.payload.get("device", "default")

    # TODO: replace with real device call, e.g.:
    #   devices.show_notification(device, message)
    _log(
        action,
        f"REMINDER on device '{device}': {message}",
    )


def handle_notify(action: Action) -> None:
    """
    Fire a passive notification (lights, display, speaker chirp...).

    Expected payload:
      message  - notification text
      device   - optional target device identifier
    """
    message = action.payload.get("message", "")
    device = action.payload.get("device", "default")

    _log(
        action,
        f"NOTIFY on device '{device}': {message}",
    )


def handle_webhook(action: Action) -> None:
    """
    POST the action payload to an HTTP endpoint.

    Expected payload:
      url      - target URL (required)
      headers  - optional dict of headers
      timeout  - optional seconds (default 10)
    """
    url = action.payload.get("url")

    if not url:
        raise ValueError("webhook payload requires 'url'")

    body = json.dumps(
        {
            "action_id": action.id,
            "action_type": action.action_type,
            "scheduled_for": action.scheduled_for.isoformat(),
            "source_record_id": action.source_record_id,
            "payload": action.payload,
        }
    ).encode()

    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **action.payload.get("headers", {}),
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=action.payload.get("timeout", 10),
    ) as response:
        if response.status >= 400:
            raise RuntimeError(
                f"webhook returned {response.status}"
            )


def handle_maintenance(action: Action) -> None:
    """
    Internal action used by the midnight maintenance runner.
    """
    _log(
        action,
        f"MAINTENANCE {action.payload.get('task', '')}",
    )


_REGISTRY: dict[str, Handler] = {
    "alarm": handle_alarm,
    "reminder": handle_reminder,
    "notify": handle_notify,
    "webhook": handle_webhook,
    "maintenance": handle_maintenance,
}


def get_handler(action_type: str) -> Handler | None:
    return _REGISTRY.get(action_type)


def register_handler(
    action_type: str,
    handler: Handler,
) -> None:
    """
    Register or override a handler, e.g. to bind a real device API:

        register_handler("alarm", my_speaker.play_alarm)
    """
    _REGISTRY[action_type] = handler


def list_handlers() -> list[str]:
    return sorted(_REGISTRY)
