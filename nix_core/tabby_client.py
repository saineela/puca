"""Optional TabbyAPI transport for an ExLlama-served Casper artifact.

This module deliberately contains no model-loading code. TabbyAPI owns the
ExLlama process and its GPU allocation; Core only sends chat-completion
requests. The backend is useful once a Qwen3.5-compatible EXL2/EXL3 artifact
has been verified, but is not selected automatically for the current PEFT
adapter.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from config import (
    CONTEXT_MAX_CHARS,
    CONTEXT_WINDOW,
    HTTP_TIMEOUT,
    TABBY_API_KEY,
    TABBY_API_URL,
    TABBY_MODEL,
)
from context import select_context
from eval import inference_metrics
from eval.privacy import OutboundBoundary, RedactionBlocked


class TabbyClient:
    """OpenAI-compatible chat client for TabbyAPI."""

    def __init__(
        self,
        api_url: str = TABBY_API_URL,
        model: str = TABBY_MODEL,
        api_key: str = TABBY_API_KEY,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        # D9: a non-local Tabby endpoint is behind the redaction boundary. A
        # local one is allowed through unchanged, and redaction that cannot
        # complete blocks the send rather than leaking the original text.
        self.privacy = OutboundBoundary()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_text: str,
        timeout: float | None = None,
        think: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        context = select_context(
            history,
            max_turns=CONTEXT_WINDOW,
            max_chars=CONTEXT_MAX_CHARS,
        )
        # Redact each outbound text segment separately, then rehydrate the reply
        # from the local-only mapping. The mapping never enters the request.
        redacted_system = self.privacy.prepare(
            url=self.api_url, text=system_prompt
        )
        redacted_history = [
            {
                **turn,
                "content": self.privacy.prepare(
                    url=self.api_url, text=str(turn.get("content") or "")
                ),
            }
            for turn in context
        ]
        redacted_user = self.privacy.prepare(url=self.api_url, text=user_text)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": redacted_system}
        ]
        messages.extend(redacted_history)
        messages.append({"role": "user", "content": redacted_user})

        started = time.perf_counter()
        response = requests.post(
            self.api_url,
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "max_tokens": min(max_new_tokens, 1024) if max_new_tokens is not None else 120,
                "temperature": 0.2,
            },
            timeout=timeout or HTTP_TIMEOUT,
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - started) * 1000
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        reply = str(message.get("content") or choices[0].get("text") or "")
        inference_metrics.record(
            inference_metrics.CallMetrics(
                model=self.model,
                backend="tabby",
                load_ms=0.0,
                # A non-streaming transport cannot observe the first token;
                # both fields stay None instead of pretending one is known.
                ttft_ms=None,
                decode_ms=elapsed_ms,
                token_count=max(1, len(reply.split())),
                cache_hit=False,
            )
        )
        return self.privacy.redactor.rehydrate(reply)

    def health(self) -> dict[str, Any]:
        """Return Tabby's advertised model list without generating text."""
        base = self.api_url.split("/v1/", 1)[0].rstrip("/")
        try:
            response = requests.get(
                f"{base}/v1/models",
                headers=self._headers(),
                timeout=3,
            )
            data = response.json()
            names = [str(item.get("id", "")) for item in data.get("data", [])]
            return {
                "ok": response.ok,
                "model": self.model,
                "models": names,
                "model_loaded": self.model in names,
            }
        except Exception as exc:
            return {
                "ok": False,
                "model": self.model,
                "models": [],
                "model_loaded": False,
                "error": type(exc).__name__,
            }
