#!/usr/bin/env python3
"""Luna long-run continuous tester.

Runs unattended for a set duration (default 10h, skip with SKIP_SLEEP=1) against
the live Nix console (http://127.0.0.1:49117), continuously reloading test cases
from categories.json. Logs every issue with a unique stable ID (LUNA-NNNN) in
&nbsp;issues.jsonl, all per-case outcomes in results.jsonl, and live aggregates in
report.json so a dashboard can poll progress without stopping the loop.
"""
from __future__ import annotations

import json
import os
import re
import random
import signal
import socket
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE = os.environ.get("LUNA_BASE", "http://127.0.0.1:49117")
RUNHours = float(os.environ.get("LUNA_HOURS", "10"))
SKIP_SLEEP = os.environ.get("SKIP_SLEEP", "0") == "1"
OUT = Path(os.environ.get("LUNA_OUT", "/root/luna_burnin"))
OUT.mkdir(parents=True, exist_ok=True)
CATS_PATH = OUT / "categories.json"
ISSUES_PATH = OUT / "issues.jsonl"
RESULTS_PATH = OUT / "results.jsonl"
REPORT_PATH = OUT / "report.json"
STATE_PATH = OUT / "state.json"

_sess = None

def get_session():
    global _sess
    if _sess is None:
        _sess = requests.Session()
    return _sess

_lock = threading.Lock()
issues_count = 0
cases_run = 0
ok_cases = 0
stopped = threading.Event()
start_ts = time.time()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_issue(category: str, sub: str, case_id: str, severity: str, what: str, evidence: str) -> int:
    """Append one issue record; returns global running issue count."""
    global issues_count
    with _lock:
        issues_count += 1
        idx = issues_count
        rec = {
            "id": f"LUNA-{idx:06d}",
            "ts": now_iso(),
            "epoch": round(time.time() - start_ts, 1),
            "category": category,
            "case": sub,
            "case_id": case_id,
            "severity": severity,
            "what": what,
            "evidence": evidence[:600],
        }
        with ISSUES_PATH.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return idx


def log_result(category: str, sub: str, case_id: str, ok: bool, extra: dict) -> None:
    global cases_run, ok_cases
    with _lock:
        cases_run += 1
        if ok:
            ok_cases = ok_cases + 1
        rec = {
            "ts": now_iso(),
            "epoch": round(time.time() - start_ts, 1),
            "category": category,
            "case": sub,
            "case_id": case_id,
            "ok": ok,
            **extra,
        }
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        REPORT_PATH.write_text(
            json.dumps(
                {
                    "started": now_iso(),
                    "uptime_hours": round((time.time() - start_ts) / 3600, 3),
                    "target_hours": RUNHours,
                    "issues_found": issues_count,
                    "cases_run": cases_run,
                    "ok_cases": ok_cases,
                },
                indent=2,
            )
        )


def send(text, location="luna-burnin", timeout=90, session=None):
    s = session or get_session()
    return s.post(
        f"{BASE}/api/send",
        json={"text": text, "location": location},
        timeout=timeout,
    )


def classify(text, timeout=15):
    try:
        return get_session().post(
            f"{BASE}/api/classify", json={"text": text}, timeout=timeout
        )
    except Exception as exc:
        class _Fail:
            ok = False
            status_code = 0
            text = f"{type(exc).__name__}: {exc}"
            def json(self):
                raise ValueError(self.text)
        return _Fail()


def check_case(case: dict):
    global cases_run
    cat = case["category"]
    sub = case.get("sub", case.get("id", "?"))
    cid = case.get("id", "?")
    text = case["text"]
    t0 = time.perf_counter()
    try:
        resp = send(text, location=case.get("location", "luna-burnin"), timeout=case.get("timeout", 90))
    except Exception as exc:
        log_result(cat, sub, cid, False, {"error": f"{type(exc).__name__}: {exc}", "latency_ms": round((time.perf_counter()-t0)*1000,1)})
        log_issue(cat, sub, cid, "critical", "request raised", str(exc))
        return
    ms = round((time.perf_counter() - t0) * 1000, 1)
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    ok = resp.ok and payload.get("ok") is True
    log_result(cat, sub, cid, ok, {
        "status": resp.status_code,
        "latency_ms": ms,
        "route": payload.get("route", ""),
        "latency_server": (payload.get("details") or {}).get("total_ms", ""),
        "reply": (payload.get("reply") or "")[:140],
    })
    if not ok or resp.status_code >= 400:
        log_issue(cat, sub, cid, "critical", "request failed", f"status={resp.status_code} payload={resp.text[:300]}")
    if case.get("expect_route") and payload.get("route") != case["expect_route"]:
        log_issue(cat, sub, cid, "medium", "route mismatch", f"expected {case['expect_route']}, got {payload.get('route')}")
    if case.get("expect_contains"):
        needle = case["expect_contains"].lower()
        reply = (payload.get("reply") or "").lower()
        if needle not in reply:
            log_issue(cat, sub, cid, "medium", "reply missing expected content", f"expected to contain {needle!r}, got: {reply[:300]}")
    if case.get("expect_not_contains_word"):
        bad = case["expect_not_contains_word"].lower()
        reply = (payload.get("reply") or "").lower()
        if bad in reply:
            log_issue(cat, sub, cid, "medium", "reply leaked forbidden label", f"reply contained {bad!r}: {reply[:300]}")
    if case.get("max_latency_ms") and ms > case["max_latency_ms"]:
        log_issue(cat, sub, cid, "low", "latency breach", f"{ms}ms > {case['max_latency_ms']}ms")


def check_classify(case: dict):
    text = case["text"]
    cat = case["category"]
    cid = case["id"]
    resp = classify(text)
    try:
        payload = resp.json()
    except Exception:
        resp.ok = False
        payload = {}
    lat = (payload.get("routing_engine") or {}).get("latency_ms", 0)
    route = payload.get("final") or payload.get("route") or ((payload.get("deterministic") or {}).get("route") or "")
    ok = bool(resp.ok)
    log_result(cat, cid, cid, ok, {"latency_ms": lat, "route": route})
    if not resp.ok:
        log_issue(cat, cid, cid, "critical", "classify endpoint failed", str(getattr(resp, "text", ""))[:400])
    if case.get("expect_classify") and route != case["expect_classify"]:
        log_issue(cat, cid, cid, "medium", "route topology wrong", f"expected {case['expect_classify']}, got {route}")
    if case.get("max_classify_ms") and float(lat) > case["max_classify_ms"]:
        log_issue(cat, cid, cid, "low", "classify latency too slow", f"{lat}ms > {case['max_classify_ms']}ms")
    if case.get("expect_confidence_min"):
        engine = payload.get("routing_engine") or (payload.get("deterministic") or {}).get("features") or {}
        conf = engine.get("confidence", 0)
        if float(conf) < case["expect_confidence_min"]:
            log_issue(cat, cid, cid, "medium", "confidence too low on clear signal", f"{conf} < {case['expect_confidence_min']} for {text!r}")

_plane = re.compile(r"\S")
def check_skills_probe(case):
    cat, cid = case["category"], case["id"]
    try:
        r = get_session().get(f"{BASE}/api/skills", timeout=30)
        body = r.json()
        skills = body.get("skills") if isinstance(body, dict) else body
        installed = [s for s in (skills or []) if isinstance(s, dict) and s.get("installed")]
        ok = r.ok and bool(installed)
        log_result(cat, cid, cid, ok, {"installed": len(installed)})
        if not ok:
            log_issue(cat, cid, cid, "critical", "skills endpoint failing / no installable skill present", f"status={r.status_code} skills={body if not isinstance(body, dict) else list(body)[:5]}")
    except Exception as exc:
        log_result(cat, cid, cid, False, {"error": str(exc)[:300]})
        log_issue(cat, cid, cid, "critical", "skills probe raised", str(exc))


def check_planner(case):
    cat, cid = case["category"], case["id"]
    t0 = time.perf_counter()
    text = case["text"]
    resp = send(text, location=case.get("location", "luna-burnin"), timeout=case.get("timeout", 90))
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    det = payload.get("details") or {}
    ok = resp.ok and payload.get("ok") is True
    plan = det.get("skill_plan_calls")
    conf = det.get("skill_execution_confirmed")
    tool_calls = plan if isinstance(plan, list) else None
    if case.get("expect_plan_calls"):
        expected = case["expect_plan_calls"]
        if not tool_calls or len(tool_calls) != expected:
            log_issue(cat, cid, cid, "medium", "plan length mismatch", f"expected {expected} tool calls, plan={tool_calls}")
    if case.get("expect_confirmed") and conf is not True:
        log_issue(cat, cid, cid, "critical", "device action was not confirmed", f"skill_execution_confirmed={conf} error={det.get('skill_error')}" )
    if case.get("expect_fail") and conf is True:
        log_issue(cat, cid, cid, "medium", "device action unexpectedly succeeded", f"skill_error={det.get('skill_error')}")
    log_result(cat, cid, cid, ok, {"latency_ms": round((time.perf_counter()-t0)*1000,1), "plan": bool(tool_calls), "confirmed": conf})


def check_memory(case):
    t0 = time.perf_counter()
    cat, cid = case["category"], case["id"]
    # Require the memory block endpoint to mention the fact we just seeded.
    fact = case["text"]
    op = case.get("op", "store")
    if op == "store":
        r = send(fact, location=case.get("location", "luna-burnin"), timeout=case.get("timeout", 90))
        payload = r.json() if r.ok else {}
        ok = r.ok and payload.get("ok") is True
    else:
        r = send(fact, location=case.get("location", "luna-burnin"), timeout=case.get("timeout", 90))
        payload = r.json() if r.ok else {}
        ok = r.ok and payload.get("ok") is True
        # The reply should echo back/restate something linked to this fact.
        if case.get("expect_any"):
            reply = payload.get("reply") or ""
            if not any(word.lower() in reply.lower() for word in case["expect_any"]):
                log_issue(cat, cid, cid, "medium", "follow-up lost memory anchor", f"reply={reply[:200]}")
    log_result(cat, cid, cid, ok, {"latency_ms": round((time.perf_counter()-t0)*1000,1), "reply": (payload.get("reply") or "")[:140]})
    if not ok:
        log_issue(cat, cid, cid, "critical", "memory step failed", f"status={r.status_code} body={r.text[:200]}")


def check_schedule(case):
    cat, cid = case["category"], case["id"]
    t0 = time.perf_counter()
    text = case["text"]
    resp = send(text, location=case.get("location", "luna-burnin"), timeout=case.get("timeout", 90))
    try:
        payload = resp.json()
    except Exception:
        payload = {}
    ok = resp.ok and payload.get("ok") is True
    reply = payload.get("reply") or ""
    verdict = "pass"
    # We expect the reply to state an absolute date/time (Temporal grounding rule).
    # "Oct 9 at 4 pm" / "October 9, 2026 at 4 pm" both count: month + day is an
    # absolute date even without a year when the month name is spelled out.
    if case.get("expect_absolute_datetime") and not (
        re.search(r"\d{4}-\d{2}-\d{2}", reply)
        or re.search(
            r"\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}",
            reply,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
            reply,
            re.IGNORECASE,
        )
        or re.search(
            r"\d{4}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
            reply,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b",
            reply,
            re.IGNORECASE,
        )
    ):
        verdict = "issue"
        log_issue(cat, cid, cid, "medium", "schedule reply not temporally grounded", f"reply={reply[:200]}")
    if case.get("expect_word") and case["expect_word"].lower() not in reply.lower():
        verdict = "issue"
        log_issue(cat, cid, cid, "medium", "schedule reply missing actor name", f"expected {case['expect_word']!r} in {reply[:200]}")
    # Formatting hygiene: no broken time renderings like "19: 00".
    broken_time = re.search(r"\b\d{1,2}:\s+\d{2}\b|\b\d{1,2}\s+:\d{2}\b", reply)
    if broken_time:
        log_issue(cat, cid, cid, "low", "reply has malformed time rendering", f"matched {broken_time.group(0)!r} in {reply[:200]}")
    log_result(cat, cid, cid, ok, {"latency_ms": round((time.perf_counter()-t0)*1000,1), "reply": reply[:160]})


def check_mt(case):
    cat, cid = case["category"], case["id"]
    seq = case["seq"]
    fails = []
    for turn in seq:
        resp = send(turn, location=case.get("location", "luna-burnin"), timeout=case.get("timeout", 90))
        try:
            payload = resp.json()
        except Exception:
            payload = {}
        reply = payload.get("reply") or ""
        if not (resp.ok and payload.get("ok")):
            fails.append(f"{turn[:40]} -> status={resp.status_code}")
        for word in case.get("forbidden", []):
            if word.lower() in reply.lower():
                fails.append(f"leaked {word!r}")
    ok = not fails
    log_result(cat, cid, cid, ok, {"fails": fails[:3], "turns": len(seq)})
    if not ok:
        log_issue(cat, cid, cid, "critical" if len(fails) > 1 else "medium", "multi-turn chain broke", "; ".join(fails))


def once(case):
    kind = case.get("kind", "send")
    if kind == "send":
        check_case(case)
    elif kind == "classify":
        check_classify(case)
    elif kind == "skills":
        check_skills_probe(case)
    elif kind == "planner":
        check_planner(case)
    elif kind == "memory":
        check_memory(case)
    elif kind == "schedule":
        check_schedule(case)
    elif kind == "mt":
        check_mt(case)


def write_state(**extra):
    STATE_PATH.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started": now_iso(),
                "target_hours": RUNHours,
                "running": not stopped.is_set(),
                "issues_found": issues_count,
                "cases_run": cases_run,
                **extra,
            },
            indent=2,
        )
    )


def main():
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    global issues_count, cases_run
    if CATS_PATH.exists():
        cases = json.loads(CATS_PATH.read_text())
    else:
        raise SystemExit(f"missing categories file {CATS_PATH}")
    deadline = time.time() + RUNHours * 3600
    write_state(status="starting_up")
    cycle = 0
    random.seed(1337)
    while not stopped.is_set() and time.time() < deadline:
        cycle += 1
        write_state(status="running", cycle=cycle)
        for case in cases:
            if stopped.is_set() or time.time() >= deadline:
                break
            worker = threading.Thread(target=once, args=(case,), daemon=True)
            worker.start()
            hard_t = case.get("timeout", 90) + 45
            worker.join(hard_t)
            if worker.is_alive():
                log_issue(case.get("category", "?"), case.get("id", "?"), case.get("id", "?"), "high", "case exceeded hard time budget (possible server hang)", f"hard limit {hard_t}s exceeded")
            time.sleep(case.get("post_delay", 2))
        time.sleep(1 if SKIP_SLEEP else 30)
    write_state(status="done", cycle=cycle)


if __name__ == "__main__":
    main()
