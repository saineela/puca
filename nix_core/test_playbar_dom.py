#!/usr/bin/env python3
"""Run the jsdom behaviour harness for the mini playbar.

The strip on every page — how it paints a track, what its buttons call, and
that it hides itself when there is nothing to show — is JavaScript in
``static/playbar.js``, so it is checked by running the real script in a DOM and
driving its ``NixPlaybar`` surface rather than by matching strings in the
source. The harness skips when Node or jsdom is unavailable (exit code 77), and
jsdom is a test-only dependency: point ``NIX_JSDOM_ROOT`` at an install if it is
not in a local ``node_modules``.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "playbar_dom.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_mini_playbar_follows_the_console_on_every_page():
    completed = subprocess.run(
        ["node", HARNESS],
        cwd=HERE,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode == 77:
        pytest.skip(completed.stdout.strip() or "jsdom is not installed")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "playbar checks passed" in completed.stdout
