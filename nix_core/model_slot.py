"""Process-wide lock for mutually exclusive local GPU model workloads.

Casper and Luna both fit individually on the development GPU, but should not
be resident or generate concurrently. Model loaders, inference, and eviction
share this re-entrant slot so switching cannot free weights during a request.
"""
from __future__ import annotations

import threading

GPU_SLOT = threading.RLock()
