"""
Nix Core configuration.

nix_core is the lightweight session brain: it classifies incoming
requests and routes them. The heavyweight components run as services:

  - nix_knowledge API  (knowledge records, calendar events, facts,
    and the model-backed fallback classifier)   -> KNOWLEDGE_API_URL
  - nix_actions API    (action scheduler, capture service, sessions)
    -> ACTIONS_API_URL
  - Ollama + SearXNG   (chatty / internet model, zero user knowledge)
    -> OLLAMA_API_URL

This file lives in the user's deployment, so every value can be
overridden with environment variables.
"""

from __future__ import annotations

import os

# ----------------------------------------------------------------------
# Identity
# Casper is the user-facing PUCA identity. NIX_* environment names remain
# deployment-compatible while the product migrates away from the old name.
ASSISTANT_NAME = os.environ.get("CASPER_NAME", "Casper")
ASSISTANT_ROLE = "PUCA (Personal User Companion Agent)"
# ----------------------------------------------------------------------

# Shared secret the Orange Pi Zero gateway presents as its first
# websocket message: {"token": "..."}.
# No usable default: deployments must provide their own secret.
AUTH_TOKEN = os.environ.get("NIX_AUTH_TOKEN", "")

WS_HOST = os.environ.get("NIX_WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("NIX_WS_PORT", "9000"))

TIMEZONE = os.environ.get("NIX_TZ", "America/Chicago")

# ----------------------------------------------------------------------
# Sibling services
# ----------------------------------------------------------------------

KNOWLEDGE_API_URL = os.environ.get(
    "NIX_KNOWLEDGE_API_URL",
    "http://127.0.0.1:8100",
)

ACTIONS_API_URL = os.environ.get(
    "NIX_ACTIONS_API_URL",
    "http://127.0.0.1:8200",
)

# ----------------------------------------------------------------------
# Chat model backends
# ----------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("NIX_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("NIX_OLLAMA_PORT", "11434"))
OLLAMA_API_URL = os.environ.get(
    "NIX_OLLAMA_API_URL",
    f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
)
# TabbyAPI exposes an OpenAI-compatible endpoint for an ExLlama model.
# It is opt-in until a verified Qwen3.5-compatible EXL2/EXL3 Casper artifact
# exists; the current PEFT adapter cannot be loaded by ExLlama directly.
TABBY_HOST = os.environ.get("NIX_TABBY_HOST", "127.0.0.1")
TABBY_PORT = int(os.environ.get("NIX_TABBY_PORT", "5000"))
TABBY_API_URL = os.environ.get(
    "NIX_TABBY_API_URL",
    f"http://{TABBY_HOST}:{TABBY_PORT}/v1/chat/completions",
)
TABBY_MODEL = os.environ.get("NIX_TABBY_MODEL", "casper-puca-v5")
TABBY_API_KEY = os.environ.get("NIX_TABBY_API_KEY", "")
# Backends: transformers (verified local LoRA), tabby (ExLlama server),
# or ollama (legacy compatibility).
CASPER_BACKEND = os.environ.get("NIX_CASPER_BACKEND", "transformers").lower()
# Ollama remains an explicit fallback/diagnostic backend.
OLLAMA_MODEL = os.environ.get("NIX_OLLAMA_MODEL", "qwen3.5:4b")
# Fast conversational mode; deterministic services handle reasoning-heavy
# knowledge operations separately.
OLLAMA_THINK = os.environ.get("NIX_OLLAMA_THINK", "0") == "1"
# Nix_predictor: the local Qwen2.5 0.5B selector handles ambiguous and
# multi-intent routing after deterministic safety rules. It is intentionally
# small, constrained to Knowledge function calls, and capped separately from
# Casper. Set the Core flag to 0 only for a no-selector diagnostic run.
USE_NEURAL_INTENT = os.environ.get("NIX_CORE_USE_NEURAL_INTENT", "0") == "1"
USE_KNOWLEDGE_MODEL_GATE = os.environ.get(
    "NIX_CORE_USE_KNOWLEDGE_MODEL_GATE", "1"
) == "1"
# Parallel warm-up initializes Casper and Nix_predictor concurrently; they
# never perform two generations for the same request.
WARMUP_MODELS = os.environ.get("NIX_CORE_WARMUP_MODELS", "1") == "1"
KNOWLEDGE_VRAM_FRACTION = float(
    os.environ.get("NIX_KNOWLEDGE_VRAM_FRACTION", "0.30")
)
CASPER_MAX_CONCURRENT_REQUESTS = int(
    os.environ.get("NIX_CASPER_MAX_CONCURRENT_REQUESTS", "1")
)
OLLAMA_FALLBACK_MODEL = os.environ.get(
    "NIX_OLLAMA_FALLBACK_MODEL", "phi4-mini:latest"
)
# Runtime KV-cache budget. Keep this bounded so chat inference leaves GPU
# headroom for future connector/task models.
OLLAMA_NUM_CTX = int(os.environ.get("NIX_OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_BATCH = int(os.environ.get("NIX_OLLAMA_NUM_BATCH", "128"))
OLLAMA_KEEP_ALIVE = os.environ.get("NIX_OLLAMA_KEEP_ALIVE", "5m")

# Seconds before we give up on a downstream service.
HTTP_TIMEOUT = float(os.environ.get("NIX_HTTP_TIMEOUT", "120"))

# How many session turns to replay as conversation context.
CONTEXT_WINDOW = int(os.environ.get("NIX_CONTEXT_WINDOW", "24"))
CONTEXT_MAX_CHARS = int(os.environ.get("NIX_CONTEXT_MAX_CHARS", "16000"))

# Ephemeral voice follow-up window. This is separate from durable day/week
# session history and is scoped to one authenticated websocket connection.
FOLLOW_UP_TIMEOUT_SECONDS = float(os.environ.get("CASPER_FOLLOW_UP_TIMEOUT", "12"))
FOLLOW_UP_MAX_TURNS = int(os.environ.get("CASPER_FOLLOW_UP_MAX_TURNS", "12"))
FOLLOW_UP_MAX_CHARS = int(os.environ.get("CASPER_FOLLOW_UP_MAX_CHARS", "6000"))
