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
# Chatty / internet model (Ollama fronting SearXNG web search)
# ----------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("NIX_OLLAMA_HOST", "192.168.0.154")
OLLAMA_PORT = int(os.environ.get("NIX_OLLAMA_PORT", "11434"))
OLLAMA_API_URL = os.environ.get(
    "NIX_OLLAMA_API_URL",
    f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat",
)
# The phi-4 model running on the Ollama host (phi4-mini:latest as of
# 2026-09); set NIX_OLLAMA_MODEL to change.
OLLAMA_MODEL = os.environ.get("NIX_OLLAMA_MODEL", "phi4-mini:latest")

# Seconds before we give up on a downstream service.
HTTP_TIMEOUT = float(os.environ.get("NIX_HTTP_TIMEOUT", "120"))

# How many session turns to replay as conversation context.
CONTEXT_WINDOW = int(os.environ.get("NIX_CONTEXT_WINDOW", "20"))
