"""Nix Core.

The conversational brain of the Nix ecosystem. It decides what to do
with a user request and keeps *session history only* — day/week-bucketed
conversations that clear on schedule. Any durable fact lives in
nix_knowledge; any scheduled outcome lives in nix_actions.

Public API:
    SessionStore   - day/week-bucketed conversation sessions
    KnowledgeProvider - protocol nix_knowledge implements
    NixCore        - orchestrator: decide, capture, prune, clear
    main           - CLI entry point
"""

from .sessions import SessionStore, SessionTurn
from .knowledge import KnowledgeProvider, KnowledgeRecord, StaticKnowledgeProvider
from .core import NixCore

__all__ = [
    "SessionStore",
    "SessionTurn",
    "KnowledgeProvider",
    "KnowledgeRecord",
    "StaticKnowledgeProvider",
    "NixCore",
]
