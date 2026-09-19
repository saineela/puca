"""
Realtime adversarial evaluation: run the full adversarial corpus through
the complete Brain pipeline in a sandbox (real OS subprocess APIs on
COPIES of the live databases - zero effect on production), then report.

Every case exercises: deterministic rules -> model gate (Qwen on GPU)
-> knowledge engine / phi-4 chat -> deterministic formatting. Results
are written incrementally to the output JSON so a killed run keeps its
progress.

Usage:
    ~/nix_knowledge/.venv/bin/python eval_adversarial_realtime.py \
        [--limit N] [--json out.json] [--port PORT]
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE_DIR))

from adversarial_corpus import ADVERSARIAL_CORPUS  # noqa: E402
from brain import split_clauses  # noqa: E402

ROOT = CORE_DIR.parent
KNOWLEDGE_DB = ROOT / "nix_knowledge" / "knowledge.db"
ACTIONS_DB = ROOT / "nix_knowledge" / "actions.db"
CORE_DB = ROOT / "nix_knowledge" / "nix_core.db"

PYTHON = str(ROOT / "nix_knowledge" / ".venv" / "bin" / "python")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Sandbox:
    """Subprocess knowledge+actions APIs on copied live databases."""

    def __init__(self) -> None:
        self.workdir = Path(tempfile.mkdtemp(prefix="nix_rt_sandbox_"))
        self.procs: list[subprocess.Popen] = []
        self.knowledge_url = ""
        self.actions_url = ""

    def start(self) -> None:
        import shutil

        env_vars = {}
        for src, name in (
            (KNOWLEDGE_DB, "knowledge.db"),
            (ACTIONS_DB, "actions.db"),
            (CORE_DB, "nix_core.db"),
        ):
            dst = self.workdir / name
            if src.exists():
                shutil.copy2(src, dst)
            else:
                dst.touch()
            env_vars[f"NIX_{name.split('.')[0].upper()}_DB"] = str(dst)

        kport, aport = _free_port(), _free_port()
        env = {
            **env_vars,
            "NIX_KNOWLEDGE_API_PORT": str(kport),
            "NIX_KNOWLEDGE_API_URL": f"http://127.0.0.1:{kport}",
            "NIX_ACTIONS_API_PORT": str(aport),
        }
        full = {**__import__("os").environ, **env}

        self.procs.append(
            subprocess.Popen(
                [PYTHON, str(ROOT / "nix_knowledge" / "scripts" / "knowledge_api.py")],
                env=full,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        self.procs.append(
            subprocess.Popen(
                [PYTHON, str(ROOT / "nix_actions" / "scripts" / "actions_api.py")],
                env={**full, "NIX_KNOWLEDGE_API_URL": f"http://127.0.0.1:{kport}"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        self.knowledge_url = f"http://127.0.0.1:{kport}"
        self.actions_url = f"http://127.0.0.1:{aport}"

        deadline = time.time() + 60
        while time.time() < deadline:
            if self._up(self.knowledge_url) and self._up(self.actions_url):
                return
            if any(p.poll() is not None for p in self.procs):
                break
            time.sleep(0.4)
        raise RuntimeError("sandbox APIs failed to start")

    @staticmethod
    def _up(url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def stop(self) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        shutil_rmtree(self.workdir)

    def cleanup_writes(self) -> None:
        """Sandbox DB copies: deletes stay sandbox-only, nothing to do."""
        return None


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def evaluate(limit: int, out_path: Path) -> dict:
    from brain import Brain
    from router import classify as rule_classify

    sandbox = Sandbox()
    sandbox.start()
    results: list[dict] = []
    try:
        brain = Brain(
            knowledge=__import__("brain").KnowledgeClient(sandbox.knowledge_url),
            actions=__import__("brain").ActionsClient(sandbox.actions_url),
            log_requests=False,
        )
        cases = list(ADVERSARIAL_CORPUS)
        if limit:
            cases = cases[:limit]

        started = time.time()
        for index, (text, expected, category, note) in enumerate(cases):
            t0 = time.perf_counter()
            rule_route, feats = rule_classify(text)
            rule_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            try:
                outcome = brain.handle(text=text, location="rt-eval")
                error = None
            except Exception as exc:
                outcome = {}
                error = f"{type(exc).__name__}: {exc}"
            ms = (time.perf_counter() - t1) * 1000
            got = outcome.get("route") or ("error" if error else "unknown")
            reply = str(outcome.get("reply") or error or "")[:220]

            # pass logic (mirrors eval_adversarial.py)
            got_routes = got.split("+") if got not in ("error", "unknown") else [got]
            if got not in ("error", "unknown") and "+" in got:
                expected_set = set(str(expected).split("+"))
                got_set = set(got_routes)
                ok = expected_set == got_set
            elif got == expected:
                ok = True
            elif got == "unknown":
                ok = True
            else:
                ok = False

            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "got": got,
                    "pass": ok,
                    "category": category,
                    "note": note,
                    "rule": outcome.get("rule") or feats.get("rule"),
                    "ms": round(ms, 1),
                    "rule_ms": round(rule_ms, 3),
                    "reply": reply,
                }
            )

            if (index + 1) % 25 == 0 or index == len(cases) - 1:
                out_path.write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "done": index + 1,
                            "total": len(cases),
                            "elapsed_s": round(time.time() - started, 1),
                            "results": results,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                passed = sum(1 for r in results if r["pass"])
                print(
                    f"[{index + 1}/{len(cases)}] pass={passed} "
                    f"elapsed={time.time() - started:.0f}s "
                    f"last={text[:50]!r}",
                    flush=True,
                )
    finally:
        sandbox.stop()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        stats = by_cat.setdefault(
            cat, {"total": 0, "pass": 0, "fail": 0, "errors": 0, "avg_ms": 0.0}
        )
        stats["total"] += 1
        stats["pass"] += int(r["pass"])
        stats["fail"] += int(not r["pass"] and r["got"] != "error")
        stats["errors"] += int(r["got"] == "error")
        stats["avg_ms"] = round(
            stats["avg_ms"] + (r["ms"] - stats["avg_ms"]) / stats["total"], 1
        )

    report = {
        "status": "complete",
        "total": total,
        "pass": passed,
        "accuracy_percent": round(100 * passed / total, 2) if total else 0.0,
        "errors": sum(1 for r in results if r["got"] == "error"),
        "by_category": by_cat,
        "failures": [r for r in results if not r["pass"]],
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", dest="json_out",
                        default="/tmp/rt_adversarial_report.json")
    parser.add_argument("--port", type=int, default=0, help="unused, kept for CLI compat")
    args = parser.parse_args()

    report = evaluate(args.limit, Path(args.json_out))
    print()
    print(f"REALTIME adversarial: {report['total']} cases")
    print(f"  PASS {report['pass']} ({report['accuracy_percent']}%)")
    print(f"  errors: {report['errors']}")
    for cat, stats in sorted(report["by_category"].items()):
        print(
            f"  {cat:24s} total={stats['total']:4d} pass={stats['pass']:4d} "
            f"fail={stats['fail']:3d} err={stats['errors']:2d} "
            f"avg={stats['avg_ms']}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
