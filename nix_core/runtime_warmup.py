"""Optional parallel startup warm-up for the two approved model services."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


def warm_models(
    *,
    casper_loader: Callable[[], Any] | None = None,
    knowledge_health: Callable[[], Any] | None = None,
) -> dict[str, str]:
    """Warm independent model/service boundaries concurrently.

    This only initializes each singleton once; it does not send duplicate user
    prompts or run Casper and the Knowledge selector for the same request.
    """
    jobs: dict[str, Callable[[], Any]] = {}
    if casper_loader is not None:
        jobs["casper"] = casper_loader
    if knowledge_health is not None:
        jobs["knowledge"] = knowledge_health
    if not jobs:
        return {}

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="casper-warmup") as pool:
        futures = {pool.submit(job): name for name, job in jobs.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                results[name] = "ready"
            except Exception as exc:  # warm-up must not prevent startup
                results[name] = f"error:{type(exc).__name__}"
    return results
