#!/usr/bin/env python3
"""Run the jsdom behaviour harness for the console's music page shell.

Rewriting covers to the console's own artwork cache is JavaScript in
``static/music-page.js``, so it is checked by running the real shell in a DOM and
driving its ``NixMusic`` surface, not by matching strings in the source. The
harness skips when Node or jsdom is unavailable on this machine (exit code 77),
and jsdom is a test-only dependency: point ``NIX_JSDOM_ROOT`` at an install if it
is not in a local ``node_modules``.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "music_shell_dom.mjs")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_music_shell_serves_covers_from_the_console():
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
    assert "music shell checks passed" in completed.stdout
