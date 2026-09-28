"""Instance-local identity settings shared by Core and the dashboard.

The name is operator-configured profile data, not identity proof or a
credential. An unset profile stays unset; no name is inferred from old memory.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unicodedata
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PROFILE_LOCK = threading.RLock()
_MAX_NAME_LENGTH = 80
_ALLOWED_PUNCTUATION = frozenset(" '-.’")


def _profile_path() -> Path:
    override = os.environ.get("NIX_PROFILE_PATH")
    if override:
        return Path(override)
    data_dir = Path(os.environ.get("NIX_DATA_DIR") or _DEFAULT_DATA_DIR)
    return data_dir / "profile.json"


def validate_user_name(value: object) -> str:
    """Normalize a display name and reject control characters/injected text."""
    if not isinstance(value, str):
        raise ValueError("user_name must be a string")
    name = value.strip()
    if len(name) > _MAX_NAME_LENGTH:
        raise ValueError(f"user_name must be {_MAX_NAME_LENGTH} characters or fewer")
    if any(
        not (
            unicodedata.category(char)[0] in {"L", "M"}
            or char in _ALLOWED_PUNCTUATION
        )
        for char in name
    ):
        raise ValueError(
            "user_name may contain letters, spaces, apostrophes, periods, and hyphens only"
        )
    return name


def get_user_name() -> str:
    """Read the instance profile; tolerate absent or malformed files safely."""
    with _PROFILE_LOCK:
        try:
            data = json.loads(_profile_path().read_text(encoding="utf-8"))
            return (
                validate_user_name(data.get("user_name", ""))
                if isinstance(data, dict)
                else ""
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return ""


def save_user_name(value: object) -> str:
    """Atomically persist a validated name with owner-only file permissions."""
    name = validate_user_name(value)
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    with _PROFILE_LOCK:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                json.dump({"user_name": name}, handle, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
    return name
