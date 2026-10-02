"""Explicitly trusted community skill subprocess runtime.

Installed files are data until a user trusts the current package digest.
Workers run as the NIX user without an OS sandbox; this is process isolation,
not a security boundary. Review code before trust.
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import ipaddress
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

PROTOCOL = "nix-skill-jsonl-v1"
MAX_LINE_BYTES = 256 * 1024
WORKER_TIMEOUT_SECONDS = 20.0


class SkillRuntimeError(RuntimeError):
    """Invalid package, configuration, or worker response."""


def _load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _normal_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> None:
    """Validate the bounded schema subset accepted by the marketplace."""
    if not isinstance(schema, dict):
        raise SkillRuntimeError("Skill declares an invalid JSON schema.")
    kind = schema.get("type")
    alternatives = schema.get("oneOf")
    if alternatives is not None:
        if not isinstance(alternatives, list) or not 1 <= len(alternatives) <= 16:
            raise SkillRuntimeError("Skill declares an invalid oneOf schema.")
        matches = 0
        for alternative in alternatives:
            if not isinstance(alternative, dict) or alternative.get("type") != kind:
                raise SkillRuntimeError("Skill oneOf alternative has an invalid type.")
            try:
                validate_json_schema(value, alternative, path)
                matches += 1
            except SkillRuntimeError:
                continue
        if matches != 1:
            raise SkillRuntimeError(f"{path} must match exactly one declared format.")
    predicates = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)),
        "boolean": lambda item: isinstance(item, bool),
    }
    if kind not in predicates or not predicates[kind](value):
        raise SkillRuntimeError(f"{path} must be {kind or 'a supported JSON type'}.")
    if "enum" in schema and value not in schema["enum"]:
        raise SkillRuntimeError(f"{path} is not an allowed value.")

    if kind in {"string", "array"}:
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SkillRuntimeError(f"{path} is too short.")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SkillRuntimeError(f"{path} is too long.")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SkillRuntimeError(f"{path} has too few items.")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SkillRuntimeError(f"{path} has too many items.")
        if kind == "string" and schema.get("pattern") and re.fullmatch(schema["pattern"], value) is None:
            raise SkillRuntimeError(f"{path} has an invalid format.")
    if kind in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise SkillRuntimeError(f"{path} is below its minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise SkillRuntimeError(f"{path} exceeds its maximum.")
    if kind == "array" and "items" in schema:
        for index, item in enumerate(value):
            validate_json_schema(item, schema["items"], f"{path}[{index}]")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise SkillRuntimeError("Skill declares an invalid object schema.")
        missing = [name for name in required if name not in value]
        if missing:
            raise SkillRuntimeError(f"{path} is missing required fields: {', '.join(missing)}.")
        if schema.get("additionalProperties", False) is False:
            extra = set(value) - set(properties)
            if extra:
                raise SkillRuntimeError(f"{path} has unsupported fields: {', '.join(sorted(extra))}.")
        if alternatives is not None:
            schema_for_match = {key: item for key, item in schema.items() if key != "oneOf"}
            validate_json_schema(value, schema_for_match, path)
        for name, item in value.items():
            if name in properties:
                validate_json_schema(item, properties[name], f"{path}.{name}")


class _Worker:
    """One serialized JSONL client for a trusted long-lived worker."""

    def __init__(self, package: Path, entrypoint: str, config: dict[str, Any], timeout: float, python_executable: str = sys.executable):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.responses: queue.Queue[str | None] = queue.Queue()
        package = package.resolve()
        raw_entry = package / entrypoint
        if raw_entry.is_symlink():
            raise SkillRuntimeError("Skill entrypoint cannot be a symlink.")
        entry = raw_entry.resolve()
        if package not in entry.parents or not entry.is_file():
            raise SkillRuntimeError("Skill entrypoint is missing or outside its package.")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        self.process = subprocess.Popen(
            [python_executable, "-I", str(entry), "--nix-skill-worker"],
            cwd=str(package), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            close_fds=True,
        )
        threading.Thread(target=self._read_loop, daemon=True, name="nix-skill-worker-reader").start()
        try:
            if self.request("initialize", {"config": config}).get("ready") is not True:
                raise SkillRuntimeError("Skill did not confirm initialization.")
        except Exception:
            self.stop()
            raise

    def _read_loop(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.process.poll() is not None or self.process.stdin is None:
                raise SkillRuntimeError("Skill worker is not running.")
            request_id = uuid.uuid4().hex
            encoded = json.dumps({**payload, "id": request_id, "op": operation}, ensure_ascii=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_LINE_BYTES:
                raise SkillRuntimeError("Skill request exceeds the runtime size limit.")
            try:
                self.process.stdin.write(encoded + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise SkillRuntimeError("Skill worker closed its input stream.") from exc
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SkillRuntimeError("Skill worker timed out.")
                try:
                    line = self.responses.get(timeout=remaining)
                except queue.Empty as exc:
                    raise SkillRuntimeError("Skill worker timed out.") from exc
                if line is None:
                    raise SkillRuntimeError("Skill worker exited without a response.")
                if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                    raise SkillRuntimeError("Skill response exceeds the runtime size limit.")
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SkillRuntimeError("Skill worker emitted invalid JSON.") from exc
                if not isinstance(response, dict) or response.get("id") != request_id:
                    raise SkillRuntimeError("Skill worker violated response correlation.")
                if response.get("ok") is not True:
                    raise SkillRuntimeError(str(response.get("error") or "Skill operation failed")[:240])
                return response

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=1.5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.5)


class SkillRuntime:
    """Manage configured/trusted workers; install alone never executes code."""

    def __init__(self, *, state_path=None, install_dir=None, data_dir=None, timeout=WORKER_TIMEOUT_SECONDS):
        root = Path(__file__).resolve().parent.parent
        self.state_path = Path(state_path or os.environ.get("NIX_SKILLS_STATE_PATH", root / "data" / "skills.json"))
        self.install_dir = Path(install_dir or os.environ.get("NIX_SKILLS_INSTALL_DIR", root / "data" / "skills" / "installed"))
        self.data_dir = Path(data_dir or os.environ.get("NIX_DATA_DIR", root / "data"))
        self.config_dir = self.data_dir / "skills" / "config"
        self.dependencies_dir = self.data_dir / "skills" / "runtime_envs"
        self.timeout = timeout
        self._workers: dict[str, _Worker] = {}
        self._lock = threading.RLock()
        self._dependency_lock = threading.Lock()
        atexit.register(self.stop_all)

    @staticmethod
    def package_dir(install_dir: Path, skill_id: str) -> Path:
        return install_dir / hashlib.sha256(skill_id.encode("utf-8")).hexdigest()

    def _state(self) -> dict[str, Any]:
        saved = _load_json(self.state_path, {})
        if not isinstance(saved, dict):
            return {}
        installed, trusted = saved.get("installed", []), saved.get("trusted", {})
        device_names = saved.get("device_names", {})
        return {
            **saved,
            "installed": installed if isinstance(installed, list) else [],
            "trusted": trusted if isinstance(trusted, dict) else {},
            "device_names": {
                key: value for key, value in device_names.items()
                if isinstance(key, str) and isinstance(value, str)
            } if isinstance(device_names, dict) else {},
        }

    def _config_path(self, skill_id: str) -> Path:
        return self.config_dir / f"{hashlib.sha256(skill_id.encode()).hexdigest()}.json"

    @staticmethod
    def fingerprint(package: Path, manifest: dict[str, Any]) -> str:
        if package.is_symlink() or not package.is_dir():
            raise SkillRuntimeError("Skill package directory is missing or unsafe.")
        root = package.resolve()
        digest = hashlib.sha256()
        declared = manifest.get("files", [])
        if not isinstance(declared, list) or len(declared) > 25 or any(not isinstance(item, str) or not item or len(item) > 180 for item in declared):
            raise SkillRuntimeError("Skill manifest declares invalid files.")
        for relative in sorted(set(["skill.json", *declared])):
            parts = relative.split("/")
            if len(parts) > 8 or any(part in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", part) for part in parts):
                raise SkillRuntimeError("Skill manifest contains an unsafe file path.")
            raw = package / relative
            if raw.is_symlink():
                raise SkillRuntimeError("Skill files cannot be symlinks.")
            target = raw.resolve()
            if root not in target.parents or not target.is_file():
                raise SkillRuntimeError("Skill contains a missing or unsafe file.")
            content = target.read_bytes()
            digest.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())
        return digest.hexdigest()

    def _package(self, skill_id: str):
        if not isinstance(skill_id, str) or len(skill_id) > 240 or not re.fullmatch(r"(?:[a-z0-9]+(?:-[a-z0-9]+){0,7}|github:[a-z0-9_.-]{1,100}/[a-z0-9_.-]{1,100}:[a-z0-9]+(?:-[a-z0-9]+){0,7})", skill_id):
            raise SkillRuntimeError("Invalid skill id.")
        package = self.package_dir(self.install_dir, skill_id)
        manifest = _load_json(package / "skill.json", {})
        if not isinstance(manifest, dict) or manifest.get("kind") != "nix-skill":
            raise SkillRuntimeError("Skill is not installed or has an invalid package manifest.")
        if manifest.get("id") not in {skill_id, skill_id.split(":")[-1]}:
            raise SkillRuntimeError("Installed package id does not match the marketplace id.")
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("protocol") != PROTOCOL:
            raise SkillRuntimeError("This installed skill does not declare a supported executable runtime.")
        entrypoint = runtime.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise SkillRuntimeError("Skill runtime entrypoint is missing.")
        return package, manifest, self.fingerprint(package, manifest)

    def setup_fields(self, skill_id: str) -> list[dict[str, Any]] | None:
        """Read setup requirements even when another package file breaks its digest."""
        if not isinstance(skill_id, str) or len(skill_id) > 240 or not re.fullmatch(r"(?:[a-z0-9]+(?:-[a-z0-9]+){0,7}|github:[a-z0-9_.-]{1,100}/[a-z0-9_.-]{1,100}:[a-z0-9]+(?:-[a-z0-9]+){0,7})", skill_id):
            return None
        package = self.package_dir(self.install_dir, skill_id)
        manifest_path = package / "skill.json"
        if package.is_symlink() or manifest_path.is_symlink():
            return None
        manifest = _load_json(manifest_path, {})
        if not isinstance(manifest, dict) or manifest.get("kind") != "nix-skill":
            return None
        if manifest.get("id") not in {skill_id, skill_id.split(":")[-1]}:
            return None
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("protocol") != PROTOCOL:
            return None
        fields = manifest.get("setup_fields", [])
        return fields if isinstance(fields, list) else None

    def configured(self, skill_id: str) -> bool:
        try:
            _package, manifest, _digest = self._package(skill_id)
        except SkillRuntimeError:
            return False
        config = _load_json(self._config_path(skill_id), {})
        if not isinstance(config, dict):
            return False
        fields = manifest.get("setup_fields", [])
        required = [field for field in fields if isinstance(field, dict) and field.get("required") is True]
        if not all(isinstance(config.get(field.get("id")), str) and bool(config[field["id"]]) for field in required):
            return False
        for field in fields:
            if not isinstance(field, dict) or field.get("id") not in config:
                continue
            value = config[field["id"]]
            if not isinstance(value, str):
                return False
            if field.get("type") == "ipv4":
                try:
                    address = ipaddress.ip_address(value)
                except ValueError:
                    return False
                if not isinstance(address, ipaddress.IPv4Address) or not address.is_private or address.is_loopback or address.is_link_local or address.is_multicast:
                    return False
            if field.get("type") == "secret":
                try:
                    if len(base64.b64decode(value, validate=True)) != 32:
                        return False
                except (ValueError, TypeError):
                    return False
        return True

    def status(self, skill_id: str) -> dict[str, Any]:
        try:
            _package, _manifest, digest = self._package(skill_id)
            trusted = self._state()["trusted"].get(skill_id) == digest
            configured = self.configured(skill_id)
            error = None
        except SkillRuntimeError as exc:
            digest, trusted, configured, error = None, False, False, str(exc)
        if not trusted:
            self.stop(skill_id)
        with self._lock:
            worker = self._workers.get(skill_id)
            running = bool(worker and worker.process.poll() is None)
        return {"fingerprint": digest, "runnable": bool(digest and trusted and configured), "trusted": trusted, "configured": configured, "worker_running": running, "runtime_error": error}

    def trust(self, skill_id: str, enabled: bool) -> dict[str, Any]:
        _package, _manifest, digest = self._package(skill_id)
        saved = self._state()
        if skill_id not in saved["installed"]:
            raise SkillRuntimeError("Install this package before changing trust.")
        if enabled and not self.configured(skill_id):
            raise SkillRuntimeError("Configure required setup fields before trusting this package.")
        trusted = saved["trusted"]
        if enabled:
            if trusted.get(skill_id) != digest:
                self.stop(skill_id)
            trusted[skill_id] = digest
        else:
            trusted.pop(skill_id, None)
        saved["trusted"] = trusted
        self._write_state(saved)
        if not enabled:
            self.stop(skill_id)
        return self.status(skill_id)

    def save_configuration(self, skill_id: str, values: dict[str, Any]) -> dict[str, Any]:
        _package, manifest, _digest = self._package(skill_id)
        if skill_id not in self._state()["installed"]:
            raise SkillRuntimeError("Install this package before configuring it.")
        if not isinstance(values, dict):
            raise SkillRuntimeError("Configuration must be an object.")
        fields = manifest.get("setup_fields", [])
        if not isinstance(fields, list) or len(fields) > 12 or any(not isinstance(field, dict) or not isinstance(field.get("id"), str) for field in fields):
            raise SkillRuntimeError("Skill setup schema is invalid.")
        allowed = {field["id"]: field for field in fields}
        if set(values) - set(allowed):
            raise SkillRuntimeError("Configuration contains unsupported fields.")
        old = _load_json(self._config_path(skill_id), {})
        old = old if isinstance(old, dict) else {}
        clean = {}
        for field_id, field in allowed.items():
            value = values.get(field_id)
            if field.get("type") == "secret" and value in (None, ""):
                value = old.get(field_id)
            if value in (None, ""):
                if field.get("required"):
                    raise SkillRuntimeError(f"{field.get('label', field_id)} is required.")
                continue
            if not isinstance(value, str) or len(value) > 512:
                raise SkillRuntimeError("Setup values must be short text.")
            if field.get("type") == "ipv4":
                try:
                    address = ipaddress.ip_address(value)
                except ValueError as exc:
                    raise SkillRuntimeError("Device address must be a private LAN IPv4 address.") from exc
                if not isinstance(address, ipaddress.IPv4Address) or not address.is_private or address.is_loopback or address.is_link_local or address.is_multicast:
                    raise SkillRuntimeError("Device address must be a private LAN IPv4 address.")
            if field.get("type") == "secret":
                try:
                    decoded = base64.b64decode(value, validate=True)
                except (ValueError, TypeError) as exc:
                    raise SkillRuntimeError("API key must be base64.") from exc
                if len(decoded) != 32:
                    raise SkillRuntimeError("API key must encode exactly 32 bytes.")
            clean[field_id] = value
        self.config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.config_dir, 0o700)
        target = self._config_path(skill_id)
        temporary = target.with_name(target.name + f".{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(clean, handle)
                handle.write("\n")
            os.replace(temporary, target)
            os.chmod(target, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.stop(skill_id)
        return self.status(skill_id)

    def clear_configuration(self, skill_id: str) -> dict[str, Any]:
        self.stop(skill_id)
        try:
            self._config_path(skill_id).unlink()
        except FileNotFoundError:
            pass
        return self.status(skill_id)

    def public_configuration(self, skill_id: str) -> dict[str, Any]:
        try:
            _package, manifest, _digest = self._package(skill_id)
        except SkillRuntimeError:
            return {}
        values = _load_json(self._config_path(skill_id), {})
        if not isinstance(values, dict):
            return {}
        return {field["id"]: ({"configured": bool(values.get(field["id"]))} if field.get("type") == "secret" else values[field["id"]]) for field in manifest.get("setup_fields", []) if isinstance(field, dict) and field.get("id") and (field.get("type") == "secret" or field["id"] in values)}

    def _write_state(self, saved: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.state_path.parent, 0o700)
        except OSError:
            pass
        temporary = self.state_path.with_name(self.state_path.name + f".{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(saved, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, self.state_path)
            os.chmod(self.state_path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _read_dependency_stamp(path: Path) -> str | None:
        try:
            return path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            return None

    def _ensure_dependencies(self, skill_id: str, package: Path, manifest: dict[str, Any]) -> str:
        runtime = manifest.get("runtime")
        if not isinstance(runtime, dict):
            raise SkillRuntimeError("Skill does not declare a supported Python runtime.")
        requirements = runtime.get("requirements")
        target = None
        requirements_bytes = b""
        if requirements:
            declared_files = manifest.get("files", [])
            if not isinstance(requirements, str) or requirements not in declared_files or len(requirements) > 180 or "\\" in requirements or requirements.startswith("/") or any(part in {"", ".", ".."} for part in requirements.split("/")):
                raise SkillRuntimeError("Skill requirements path is invalid.")
            raw = package / requirements
            if raw.is_symlink():
                raise SkillRuntimeError("Skill requirements file cannot be a symlink.")
            target = raw.resolve()
            if package.resolve() not in target.parents or not target.is_file():
                raise SkillRuntimeError("Skill requirements file is missing or unsafe.")
            requirements_bytes = target.read_bytes()
            try:
                requirements_text = requirements_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SkillRuntimeError("Skill requirements must be valid UTF-8 text.") from exc
            requirement_re = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]{0,80})(?:===|==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]+(?:,(?:===|==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]+)*")
            lines = [line.strip() for line in requirements_text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            matches = [requirement_re.fullmatch(line) for line in lines]
            if not 1 <= len(lines) <= 64 or any(match is None for match in matches):
                raise SkillRuntimeError("Dependencies must be bounded package/version specifiers; pip options and URLs are forbidden.")

        requirements_digest = hashlib.sha256(
            requirements_bytes + b"\0" + sys.version.encode("utf-8")
        ).hexdigest()
        skill_environment = self.dependencies_dir / hashlib.sha256(skill_id.encode("utf-8")).hexdigest()
        environment = skill_environment / ".venv"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        stamp = environment / ".nix-requirements-sha256"
        with self._dependency_lock:
            if (
                python.is_file()
                and not skill_environment.is_symlink()
                and not environment.is_symlink()
                and not stamp.is_symlink()
                and self._read_dependency_stamp(stamp) == requirements_digest
            ):
                return str(python)

            self.dependencies_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.dependencies_dir, 0o700)
            if skill_environment.is_symlink():
                skill_environment.unlink()
            elif skill_environment.exists():
                shutil.rmtree(skill_environment)
            skill_environment.mkdir(mode=0o700)
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(environment)],
                    check=True, timeout=120, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if not python.is_file():
                    raise SkillRuntimeError("Python did not create the skill dependency environment.")
                if target is not None:
                    subprocess.run(
                        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--only-binary=:all:", "--requirement", str(target)],
                        check=True, timeout=300, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                descriptor = os.open(stamp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                    handle.write(requirements_digest)
            except (OSError, UnicodeDecodeError, subprocess.SubprocessError, SkillRuntimeError) as exc:
                if skill_environment.is_symlink():
                    skill_environment.unlink()
                elif skill_environment.exists():
                    shutil.rmtree(skill_environment)
                if isinstance(exc, SkillRuntimeError):
                    raise
                raise SkillRuntimeError(
                    "Could not install skill runtime dependencies or create its isolated .venv."
                ) from exc
        return str(python)

    def remove_dependencies(self, skill_id: str) -> None:
        """Remove one skill's private virtual environment."""
        skill_environment = self.dependencies_dir / hashlib.sha256(skill_id.encode("utf-8")).hexdigest()
        with self._dependency_lock:
            if skill_environment.is_symlink():
                skill_environment.unlink()
            elif skill_environment.exists():
                shutil.rmtree(skill_environment)

    def provision_installed(self, skill_id: str) -> str | None:
        """Provision an installed executable skill without launching its code."""
        package = self.package_dir(self.install_dir, skill_id)
        manifest = _load_json(package / "skill.json", {})
        runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
        if not isinstance(runtime, dict) or runtime.get("protocol") != PROTOCOL:
            return None
        package, manifest, _digest = self._package(skill_id)
        return self._ensure_dependencies(skill_id, package, manifest)

    def _device_tool_call(self, skill_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        status = self.status(skill_id)
        if not status["runnable"]:
            raise SkillRuntimeError("Configure and explicitly trust this installed package before using the device.")
        with self._lock:
            worker = self._workers.get(skill_id)
        if worker is None or worker.process.poll() is not None:
            raise SkillRuntimeError("Connect the skill before requesting device status or controls.")
        _package, manifest, _digest = self._package(skill_id)
        tools = manifest.get("tools", [])
        tool = next((item for item in tools if isinstance(item, dict) and item.get("name") == tool_name), None) if isinstance(tools, list) else None
        if tool is None or not isinstance(tool.get("input_schema"), dict) or not isinstance(tool.get("result_schema"), dict):
            raise SkillRuntimeError("The skill does not declare the requested device tool and schemas.")
        validate_json_schema(arguments, tool["input_schema"])
        try:
            result = worker.request("execute", {"tool": tool["name"], "arguments": arguments}).get("result")
            validate_json_schema(result, tool["result_schema"])
            return result
        except SkillRuntimeError:
            self.stop(skill_id)
            raise

    def device_status(self, skill_id: str) -> dict[str, Any]:
        """Read status using the explicitly declared device UI status operation."""
        status = self.status(skill_id)
        if not status.get("runnable"):
            raise SkillRuntimeError("Configure and explicitly trust this installed package before using the device.")
        _package, manifest, _digest = self._package(skill_id)
        device_ui = manifest.get("device_ui")
        if not isinstance(device_ui, dict):
            raise SkillRuntimeError("This skill does not declare a Devices-page status operation.")
        return self._device_tool_call(skill_id, device_ui["status_tool"], device_ui["status_arguments"])

    def control_device(self, skill_id: str, arguments: dict[str, Any], tool_name: str | None = None) -> dict[str, Any]:
        """Run one control explicitly declared by the installed skill's UI schema."""
        if not isinstance(arguments, dict):
            raise SkillRuntimeError("Device control arguments must be an object.")
        _package, manifest, _digest = self._package(skill_id)
        device_ui = manifest.get("device_ui")
        controls = device_ui.get("controls", []) if isinstance(device_ui, dict) else []
        declared = None
        if isinstance(tool_name, str) and isinstance(controls, list):
            for control in controls:
                if not isinstance(control, dict) or control.get("tool") != tool_name:
                    continue
                base = control.get("arguments")
                fields = control.get("fields")
                if not isinstance(base, dict) or not isinstance(fields, list):
                    continue
                field_names = {
                    field.get("name") for field in fields if isinstance(field, dict)
                }
                if (
                    len(field_names) == len(fields)
                    and set(arguments) == set(base) | field_names
                    and all(arguments.get(key) == value for key, value in base.items())
                ):
                    declared = control
                    break
        if declared is None:
            raise SkillRuntimeError("Device control arguments do not match a declared control.")
        return self._device_tool_call(skill_id, tool_name, arguments)

    @staticmethod
    def _bounded_device_data(value: Any, *, depth: int = 0) -> Any:
        """Keep validated skill results bounded before adding them to model context."""
        if depth > 8:
            return None
        if isinstance(value, dict):
            return {str(key)[:64]: SkillRuntime._bounded_device_data(item, depth=depth + 1) for key, item in list(value.items())[:64]}
        if isinstance(value, list):
            return [SkillRuntime._bounded_device_data(item, depth=depth + 1) for item in value[:128]]
        if isinstance(value, str):
            return value[:512]
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        return None

    def _worker(self, skill_id: str) -> tuple[_Worker, dict[str, Any]]:
        package, manifest, digest = self._package(skill_id)
        if skill_id not in self._state()["installed"]:
            raise SkillRuntimeError("Install this skill before running it.")
        if self._state()["trusted"].get(skill_id) != digest:
            self.stop(skill_id)
            raise SkillRuntimeError("Skill package is not explicitly trusted; review its current digest first.")
        if not self.configured(skill_id):
            raise SkillRuntimeError("Skill setup is incomplete.")
        python_executable = self._ensure_dependencies(skill_id, package, manifest)
        with self._lock:
            current = self._workers.get(skill_id)
            if current and current.process.poll() is None:
                return current, manifest
            config = _load_json(self._config_path(skill_id), {})
            worker = _Worker(package, manifest["runtime"]["entrypoint"], config, self.timeout, python_executable)
            self._workers[skill_id] = worker
            return worker, manifest

    def trigger_candidate(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """Recognize direct address only; no worker or network is started here."""
        clean = " ".join(str(text).casefold().split())
        if len(clean) > 4000:
            return None
        if re.match(r"^(?:can|could|would) you (?:please )?(?:turn|switch|power|set|make|change|run|start|play|stop|disable|enable|brighten|dim|paint|animate|display|show|list)\b", clean):
            clean = re.sub(r"^(?:can|could|would) you (?:please )?", "", clean)
        elif re.match(r"^(?:(?:please|hey nix)[, ]+)?(?:can|could|would|will|do|does) you\b", clean):
            return None
        clean = clean.rstrip("?.! ")
        if len(clean) > 4000 or re.search(r"\b(?:if|when|unless|whether|should|wonder|how do i|how can i|what if|could you explain|can you explain)\b", clean):
            return None
        if re.search(r"\band (?:then )?(?:turn|switch|power|set|make|run|start|stop|disable|enable|dim|brighten)\b|;", clean):
            return None
        clean = re.sub(r"^(?:(?:please|could you|can you|would you|hey(?: nix)?)[, ]*)+", "", clean)
        direct = bool(re.match(r"(?:turn|switch|power|set|make|change|run|start|play|stop|disable|enable|brighten|dim|paint|animate|display|show)\b", clean))
        state_query = bool(re.match(r"what(?: is|'s)\b", clean) and re.search(r"\b(?:ring|state|status)\b.*\b(?:doing|look|state|status)\b", clean))
        catalog_query = bool(re.match(r"(?:what|which|show|list|tell me|are|can)\b", clean) and re.search(r"\b(?:effects?|animations?|colors?|colours?)\b", clean))
        if catalog_query and re.search(r"\b(?:how do i|how can i|can i|should i|wonder|whether)\b", clean):
            return None
        generic_device = bool(re.search(r"\b(?:the|my|that|this)\s+(?:light|lamp|device|thermostat|lock|camera|speaker|plug)\b", clean))
        if not (direct or state_query or catalog_query):
            return None
        def contains_term(text: str, term: str) -> bool:
            return bool(term and re.search(rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])", text))
        specs = self.installed_skill_specs()
        device_names = self._state().get("device_names", {})
        for spec in specs:
            alias = device_names.get(spec["skill_id"], "") if isinstance(device_names, dict) else ""
            terms = [spec["name"], spec["device_name"], alias, *spec["triggers"]]
            if any(contains_term(clean, term) for term in terms):
                _package, manifest, _digest = self._package(spec["skill_id"])
                return spec["skill_id"], manifest
        if not generic_device:
            return None
        candidates = [spec for spec in self.installed_skill_specs(only_runnable=True) if spec.get("device_ui")]
        if len(candidates) == 1:
            return candidates[0]["skill_id"], self._package(candidates[0]["skill_id"])[1]
        # A single installed skill with direct generic device language is a
        # usable target even before setup/trust; match() will then provide the
        # explicit setup/trust boundary without starting a worker.
        specs = self.installed_skill_specs()
        generic_device = bool(re.search(r"\b(?:light|lamp|device|thermostat|lock|camera|speaker|plug)\b", clean))
        direct_request = bool(re.match(r"(?:turn|switch|power|set|make|change|run|start|play|stop|disable|enable|brighten|dim|paint|animate|display|show)\b", clean))
        if generic_device and direct_request and len(specs) == 1:
            skill_id = specs[0]["skill_id"]
            return skill_id, self._package(skill_id)[1]
        return None

    def installed_skill_specs(self, *, only_runnable: bool = False) -> list[dict[str, Any]]:
        """Return bounded schemas and descriptions for installed skills."""
        specs = []
        state_snapshot = self._state()
        for skill_id in state_snapshot["installed"]:
            try:
                package, manifest, _digest = self._package(skill_id)
                state = self.status(skill_id)
            except SkillRuntimeError:
                continue
            if only_runnable and not state["runnable"]:
                continue
            tools = manifest.get("tools", [])
            if not isinstance(tools, list) or not tools:
                continue
            # Do not load arbitrary README/instruction prose into model context.
            # A separately validated, bounded manifest summary is opt-in data.
            summary = str(manifest.get("assistant_summary") or "")[:1200]
            name = str(manifest.get("name") or skill_id)[:80]
            alias = state_snapshot.get("device_names", {}).get(skill_id)
            triggers = manifest.get("triggers", [])
            spec = {
                "skill_id": skill_id,
                "name": name,
                "device_name": (" ".join(alias.split())[:64] if isinstance(alias, str) and alias.strip() else name),
                "description": str(manifest.get("description") or "")[:500],
                "summary": summary,
                "device_ui": manifest.get("device_ui") if isinstance(manifest.get("device_ui"), dict) else None,
                "worker_running": bool(state["worker_running"]),
                "live_device": None,
                "live_device_error": None,
                "triggers": [value[:48] for value in triggers if isinstance(value, str)][:24] if isinstance(triggers, list) else [],
                "configured": bool(state["configured"]),
                "trusted": bool(state["trusted"]),
                "runnable": bool(state["runnable"]),
                "tools": [
                    {
                        "name": str(tool.get("name") or "")[:48],
                        "description": str(tool.get("description") or "")[:500],
                        "input_schema": tool.get("input_schema"),
                    }
                    for tool in tools if isinstance(tool, dict)
                ][:32],
            }
            specs.append(spec)
        return specs[:24]

    def targeted_skill_specs(self, text: str, history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Return runnable skill data only when the current turn targets a device."""
        specs = self.installed_skill_specs()
        if not specs:
            return []
        current = " ".join(str(text or "").casefold().split())[:1000]
        recent_turns = [
            str(turn.get("content") or "")[:500].casefold()
            for turn in (history or [])[-8:] if isinstance(turn, dict)
        ]
        actionable = not bool(re.search(
            r"\b(?:story|explain|explanation|how does|how do|how can|what if|wonder|whether)\b",
            current,
        )) and bool(re.search(
            r"\b(?:turn|switch|power|set|make|change|run|start|play|stop|disable|enable|brighten|dim|paint|animate|show|list|what|which|tell|apply|status|state)\b",
            current,
        ))
        refers_back = len(current) <= 160 and bool(re.search(
            r"\b(?:it|ti|that|this|they|them|those|there|one|ones)\b", current
        ))

        def contains(text_value: str, term: str) -> bool:
            return bool(term and re.search(
                rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])",
                text_value,
            ))

        targeted = []
        for spec in specs:
            alias = self._state().get("device_names", {}).get(spec["skill_id"], "")
            terms = [spec["name"], spec["device_name"], alias, *spec["triggers"]]
            explicit = actionable and any(contains(current, term) for term in terms)
            if explicit:
                targeted.append(spec)
        if refers_back and not targeted and not bool(re.search(
            r"\b(?:story|explain|explanation|how does|how do|how can|what if|wonder|whether)\b",
            current,
        )):
            # Resolve ellipses against the nearest earlier device mention,
            # rather than a stale match anywhere in a long conversation.
            for turn_text in reversed(recent_turns):
                nearest = [
                    spec for spec in specs
                    if any(contains(turn_text, term) for term in (
                        spec["name"], spec["device_name"],
                        self._state().get("device_names", {}).get(spec["skill_id"], ""),
                        *spec["triggers"],
                    ))
                ]
                if nearest:
                    targeted = nearest
                    break

        # A single installed runnable skill may be selected for a plainly
        # device-directed request even when the user uses a generic noun.
        generic_device_request = actionable and not bool(re.search(
            r"\b(?:story|explain|explanation|how does|how do|how can|what if|wonder|whether)\b",
            current,
        )) and bool(re.search(
            r"\b(?:device|light|lamp|ring|speaker|plug|thermostat|lock|camera)\b",
            current,
        ))
        if not targeted and generic_device_request:
            runnable = [spec for spec in specs if spec["runnable"]]
            if len(runnable) == 1:
                targeted = runnable

        # Device-directed but unready single skills receive a bounded setup
        # explanation; their descriptions/schemas are never sent to the model.
        if not targeted and generic_device_request and len(specs) == 1:
            targeted = specs

        # Preserve the runtime's narrow direct-address recognition, but use it
        # only to select a model capability; it never matches or executes.
        if not targeted and actionable:
            candidate = self.trigger_candidate(current)
            if candidate is not None and any(spec["skill_id"] == candidate[0] for spec in specs):
                targeted = [spec for spec in specs if spec["skill_id"] == candidate[0]]
        targeted = targeted[:8]
        # Read live device data only for a device-targeted request and only from
        # a worker the user explicitly connected. This supports accurate catalogs
        # without background device connections during normal conversation.
        for spec in targeted:
            if spec.get("runnable") and spec.get("worker_running"):
                try:
                    if isinstance(spec.get("device_ui"), dict):
                        live = self.device_status(spec["skill_id"])
                    else:
                        _package, manifest, _digest = self._package(spec["skill_id"])
                        tool = next((item for item in manifest.get("tools", []) if isinstance(item, dict) and item.get("name") == "control_ring"), None)
                        if tool is None:
                            continue
                        live = self._device_tool_call(spec["skill_id"], tool["name"], {"action": "catalog"})
                    spec["live_device"] = self._bounded_device_data(live)
                except Exception as exc:
                    spec["live_device_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return targeted

    def proposal_context(self, skill_ids: list[str] | None = None) -> str:
        """Render bounded skill descriptions and schemas as JSON data."""
        specs = self.installed_skill_specs(only_runnable=True)
        if skill_ids is not None:
            selected = set(skill_ids)
            specs = [spec for spec in specs if spec["skill_id"] in selected]
        encoded = json.dumps(specs[:8], ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 16000:
            raise SkillRuntimeError("Skill capability data exceeds the model context limit.")
        return encoded

    def match(self, text: str) -> dict[str, Any] | None:
        candidate = self.trigger_candidate(text)
        if candidate is None:
            return None
        skill_id, manifest = candidate
        if skill_id not in self._state()["installed"]:
            raise SkillRuntimeError("Install this skill before matching a tool request.")
        state = self.status(skill_id)
        if not state["configured"]:
            raise SkillRuntimeError("Skill setup is incomplete.")
        if not state["trusted"]:
            raise SkillRuntimeError("Skill package is not explicitly trusted; review its current digest first.")
        worker, manifest = self._worker(skill_id)
        match_text = str(text)[:4000]
        alias = self._state().get("device_names", {}).get(skill_id)
        if isinstance(alias, str) and alias.strip():
            canonical_name = str(manifest.get("name") or skill_id.split(":")[-1])[:80]
            match_text = re.sub(
                rf"(?<![a-z0-9]){re.escape(alias.strip())}(?![a-z0-9])",
                canonical_name,
                match_text,
                flags=re.IGNORECASE,
            )
        try:
            response = worker.request("match", {"text": match_text})
        except SkillRuntimeError:
            self.stop(skill_id)
            raise
        found = response.get("match")
        if found is None:
            return None
        if not isinstance(found, dict):
            raise SkillRuntimeError("Skill matcher returned an invalid tool call.")
        tools = manifest.get("tools", [])
        if "tool" in found or "arguments" in found:
            tool = next(
                (item for item in tools if isinstance(item, dict) and item.get("name") == found.get("tool")),
                None,
            )
            if tool is None:
                raise SkillRuntimeError("Skill selected an undeclared tool.")
            arguments = found.get("arguments", {})
        else:
            # Single-tool skills may return their schema-validated argument
            # object directly from `match`; bind it to the only declared tool.
            # Multi-tool packages must name their selected tool explicitly.
            if len(tools) != 1 or not isinstance(tools[0], dict):
                raise SkillRuntimeError("Skill matcher must select one declared tool.")
            tool = tools[0]
            arguments = found
        if not isinstance(tool.get("input_schema"), dict):
            raise SkillRuntimeError("Installed skill tool has no valid input schema.")
        validate_json_schema(arguments, tool["input_schema"])
        return {"skill_id": skill_id, "skill_name": manifest.get("name", skill_id), "tool": tool, "arguments": arguments}

    def validate_proposal(self, proposal: Any) -> dict[str, Any] | None:
        """Validate one model proposal against currently trusted installed schemas."""
        if not isinstance(proposal, dict) or proposal.get("type") != "skill_tool_call":
            return None
        if set(proposal) != {"type", "skill_id", "tool", "arguments"}:
            raise SkillRuntimeError("Skill tool proposal contains unsupported fields.")
        skill_id = proposal.get("skill_id")
        tool_name = proposal.get("tool")
        arguments = proposal.get("arguments")
        if not isinstance(skill_id, str) or not isinstance(tool_name, str):
            raise SkillRuntimeError("Skill tool proposal must identify a skill and tool.")
        if not isinstance(arguments, dict):
            raise SkillRuntimeError("Skill tool proposal arguments must be a JSON object.")
        state = self.status(skill_id)
        if not state.get("runnable"):
            raise SkillRuntimeError("The proposed skill is not configured and trusted at its installed digest.")
        if skill_id not in self._state()["installed"]:
            raise SkillRuntimeError("The proposed skill is not installed.")
        _package, manifest, _digest = self._package(skill_id)
        tools = manifest.get("tools", [])
        tool = next(
            (item for item in tools if isinstance(item, dict) and item.get("name") == tool_name),
            None,
        ) if isinstance(tools, list) else None
        if tool is None:
            raise SkillRuntimeError("The proposed tool is not declared by the installed skill.")
        if not isinstance(tool.get("input_schema"), dict) or not isinstance(tool.get("result_schema"), dict):
            raise SkillRuntimeError("The installed tool has no valid input or result schema.")
        validate_json_schema(arguments, tool["input_schema"])
        return {
            "skill_id": skill_id,
            "skill_name": str(manifest.get("name") or skill_id),
            "tool": tool,
            "arguments": arguments,
        }

    def execute(self, matched: dict[str, Any]) -> dict[str, Any]:
        skill_id = str(matched.get("skill_id", ""))
        current_status = self.status(skill_id)
        if not current_status.get("runnable", bool(current_status.get("configured") and current_status.get("trusted"))):
            raise SkillRuntimeError("The installed skill is no longer configured and trusted.")
        state = self._state()
        if skill_id not in state["installed"]:
            raise SkillRuntimeError("The proposed skill is no longer installed.")
        _package, installed_manifest, _digest = self._package(skill_id)
        requested_tool = matched.get("tool", {}).get("name") if isinstance(matched.get("tool"), dict) else None
        declared_tool = next((item for item in installed_manifest.get("tools", []) if isinstance(item, dict) and item.get("name") == requested_tool), None)
        if declared_tool is None:
            raise SkillRuntimeError("Requested tool is no longer declared by the installed package.")
        if not isinstance(declared_tool.get("input_schema"), dict) or not isinstance(declared_tool.get("result_schema"), dict):
            raise SkillRuntimeError("The installed tool has no valid input or result schema.")
        arguments = matched.get("arguments")
        validate_json_schema(arguments, declared_tool["input_schema"])
        # Check proposal shape/schema before starting any trusted worker process.
        worker, manifest = self._worker(skill_id)
        tool = next((item for item in manifest.get("tools", []) if isinstance(item, dict) and item.get("name") == requested_tool), None)
        if tool is None:
            raise SkillRuntimeError("Requested tool changed before execution.")
        validate_json_schema(arguments, tool["input_schema"])
        try:
            result = worker.request("execute", {"tool": tool["name"], "arguments": arguments}).get("result")
            validate_json_schema(result, tool["result_schema"])
            if isinstance(result, dict) and "ok" in result and result["ok"] is not True:
                raise SkillRuntimeError("The skill did not confirm a successful result.")
        except SkillRuntimeError:
            self.stop(skill_id)
            raise
        return {"ok": True, "result": result, "skill_id": skill_id, "skill_name": manifest.get("name", skill_id), "tool": tool["name"], "arguments": arguments}

    def start_skill(self, skill_id: str) -> dict[str, Any]:
        worker, _manifest = self._worker(skill_id)
        del worker
        return self.status(skill_id)

    def stop(self, skill_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(skill_id, None)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        with self._lock:
            workers, self._workers = self._workers, {}
        for worker in workers.values():
            worker.stop()

    def forget(self, skill_id: str) -> None:
        self.stop(skill_id)
        self.remove_dependencies(skill_id)
        try:
            self._config_path(skill_id).unlink()
        except FileNotFoundError:
            pass
        state = self._state()
        state["trusted"].pop(skill_id, None)
        self._write_state(state)

    def prompt_context(self) -> str:
        return self.proposal_context()

    def status_for_installed(self, skill_id: str) -> dict[str, Any]:
        return self.status(skill_id)


_RUNTIME_CACHE: dict[tuple[str, str, str], SkillRuntime] = {}
_RUNTIME_CACHE_LOCK = threading.Lock()


def get_skill_runtime(*, state_path=None, install_dir=None, data_dir=None) -> SkillRuntime:
    root = Path(__file__).resolve().parent.parent
    state = Path(state_path or os.environ.get("NIX_SKILLS_STATE_PATH", root / "data" / "skills.json")).resolve()
    installed = Path(install_dir or os.environ.get("NIX_SKILLS_INSTALL_DIR", root / "data" / "skills" / "installed")).resolve()
    data = Path(data_dir or os.environ.get("NIX_DATA_DIR", root / "data")).resolve()
    key = (str(state), str(installed), str(data))
    with _RUNTIME_CACHE_LOCK:
        if key not in _RUNTIME_CACHE:
            _RUNTIME_CACHE[key] = SkillRuntime(state_path=state, install_dir=installed, data_dir=data)
        return _RUNTIME_CACHE[key]
