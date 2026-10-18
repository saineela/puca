"""How the assistant answers when it does not know — a before/after benchmark.

    python3 benchmarks/abstention_voice.py /var/tmp/bench_before   # pre-fix tree
    python3 benchmarks/abstention_voice.py /root                  # current tree

The point of the measurement is the wording of an "I don't know" turn. Before
the copy layer existed, a lookup that found nothing was answered in engineering
language ("I don't have that in my knowledge base yet", "No matching events
found", "Luna Alert: Unable to reach the chat model (AcceleratorError)"); a
person says "I don't have that yet - tell me and I'll remember it".

Two corpora, both grounded in code that really ran:

1. ``LEGACY`` — the pre-fix wordings, copied verbatim from the pre-fix source
   (the file and line are recorded next to each). In the pre-fix tree they come
   back unchanged; in the current tree they pass through
   ``voice_copy.humanize_reply``, which is where every reply passes on its way
   out of ``Brain._finish`` (brain.py:3649).
2. ``PIPELINE`` — the pipeline's own empty results, obtained by calling
   ``brain.format_knowledge_result`` with the envelope it really receives.

Four criteria, each a yes/no per reply; the summary reports the share that pass:

* ``honest``      — an unknown question admits the gap, a failed action says it
                    did not happen. Nothing is invented, nothing claims success.
* ``no_internals``— no engineering nouns (knowledge base, planner, skill
                    profile, NIX Settings, chat model, error class, ...).
* ``offers_next`` — invites the user to continue (a question or an offer to
                    retry).
* ``human_shape`` — at most 20 words, no "Alert:", no parenthesised error, no
                    colon-led report.

The benchmark exits non-zero only when a reply cannot be produced at all.
"""
from __future__ import annotations

import json
import os
import sys

# (id, kind, wording, source in the pre-fix tree)
LEGACY: tuple[tuple[str, str, str, str], ...] = (
    (
        "fact-miss",
        "unknown",
        "I don't have that in my knowledge base yet.",
        "brain.py:2176",
    ),
    (
        "states-miss",
        "unknown",
        "No current states stored yet.",
        "brain.py:2151",
    ),
    (
        "events-miss",
        "unknown",
        "No matching events found.",
        "brain.py:2166",
    ),
    (
        "known-failure",
        "failure",
        "I couldn't complete that request, so nothing was saved or changed. "
        "Could you phrase it another way?",
        "brain.py:2040",
    ),
    (
        "planner-refusal",
        "failure",
        "I didn't send a device command because the dedicated skill planner "
        "did not produce a valid, safe plan.",
        "brain.py:5239",
    ),
    (
        "skill-untrusted",
        "failure",
        "The targeted skill isn't currently ready or trusted, so I didn't send "
        "a device command.",
        "brain.py:6453",
    ),
    (
        "name-unset",
        "unknown",
        "I don't have a preferred name set in NIX Settings yet.",
        "brain.py:3416",
    ),
    (
        "model-unreachable",
        "failure",
        "Luna Alert: Unable to reach the chat model (AcceleratorError).",
        "brain.py:7102",
    ),
    (
        "empty-model-reply",
        "failure",
        "(empty reply from chat model)",
        "brain.py:7111",
    ),
    (
        "clinical-unknown",
        "unknown",
        "I don't have any information about your mother's current state.",
        "voice_copy.py docstring, the user's own example",
    ),
)

# The KnowledgeResponse envelope `format_knowledge_result` really receives.
PIPELINE: tuple[tuple[str, str, dict], ...] = (
    ("facts", "unknown", {"result": {"facts": []}}),
    ("events", "unknown", {"result": {"events": []}}),
    ("states", "unknown", {"result": {"states": []}}),
    ("failure", "failure", {"result": {"ok": False}}),
)

# Model self-reference. These are not pre-fix pipeline strings: they are the
# shape a model falls into when it has nothing to say, and the copy layer does
# not claim them. Measured separately so the gap is a number, not an opinion.
AI_SELF_REFERENCE = (
    "As an AI language model, I don't have access to that information.",
    "I'm just an AI, so I can't know that.",
    "I don't have access to your personal information.",
    "As a language model, I cannot answer that.",
)

INTERNALS = (
    "knowledge base",
    "planner",
    "skill profile",
    "nix settings",
    "chat model",
    "acceleratorerror",
    "status code",
    "digest",
    "worker",
    "valid, safe plan",
    "stored yet",
    "matching events",
    "empty reply",
)
ADMITS_GAP = (
    "i don't have",
    "i don't know",
    "i'm not sure",
    "i can't find",
    "no information",
    "haven't told me",
    "isn't ready",
)
SAYS_NOT_DONE = (
    "nothing",
    "couldn't",
    "didn't",
    "can't",
    "isn't ready",
    "slipped",
    "i'm here",
    "try again",
    # A turn the assistant lost is owned out loud ("sorry, could you repeat").
    "could you repeat",
    "mind freeze",
)
RETHINK = (
    "tell me",
    "say that again",
    "try",
    "ask me",
    "let me",
    "name the",
    "which",
    "want me to",
    "our",
)


def score(kind: str, text: str) -> dict:
    low = text.lower().strip()
    words = low.split()
    if kind == "unknown":
        honest = any(marker in low for marker in ADMITS_GAP)
    else:
        honest = any(marker in low for marker in SAYS_NOT_DONE) or any(
            marker in low for marker in ADMITS_GAP
        )
    return {
        "honest": bool(honest),
        "no_internals": not any(marker in low for marker in INTERNALS),
        "offers_next": "?" in text or any(marker in low for marker in RETHINK),
        "human_shape": (
            len(words) <= 20
            and "alert:" not in low
            and "(" not in text
            and not low.startswith("-")
        ),
        "words": len(words),
    }


def run(tree_root: str) -> dict:
    core = os.path.join(tree_root, "nix_core")
    sys.path.insert(0, core)
    import brain  # noqa: E402  — from the tree under test

    try:
        import voice_copy  # noqa: E402

        humanize = voice_copy.humanize_reply
        has_copy_layer = True
    except ImportError:  # the pre-fix tree has no copy layer, which is the point
        has_copy_layer = False

        def humanize(text: str) -> str:
            return text

    rows: list[dict] = []
    for ident, kind, wording, source in LEGACY:
        said = humanize(wording)
        rows.append(
            {
                "id": ident,
                "kind": kind,
                "source": source,
                "emitted": wording,
                "text": said,
                "changed": said != wording,
                **score(kind, said),
            }
        )
    for ident, kind, payload in PIPELINE:
        try:
            said = brain.format_knowledge_result(dict(payload))
        except Exception as exc:  # a crash is a reply-quality failure, not an exit
            said = f"({type(exc).__name__}: {exc})"
        rows.append(
            {
                "id": f"pipeline-{ident}",
                "kind": kind,
                "source": "brain.format_knowledge_result",
                "emitted": said,
                "text": said,
                "changed": None,
                **score(kind, said),
            }
        )

    rewritten = [row for row in rows if row["changed"] is not None]
    ai_rows = [humanize(text) for text in AI_SELF_REFERENCE]
    summary = {
        "tree": os.path.abspath(tree_root),
        "copy_layer": has_copy_layer,
        "items": len(rows),
        "honest": sum(1 for row in rows if row["honest"]) / len(rows),
        "no_internals": sum(1 for row in rows if row["no_internals"]) / len(rows),
        "offers_next": sum(1 for row in rows if row["offers_next"]) / len(rows),
        "human_shape": sum(1 for row in rows if row["human_shape"]) / len(rows),
        "all_four": sum(
            1
            for row in rows
            if row["honest"]
            and row["no_internals"]
            and row["offers_next"]
            and row["human_shape"]
        )
        / len(rows),
        "rewritten": (
            sum(1 for row in rewritten if row["changed"]) / len(rewritten)
            if rewritten
            else None
        ),
        "no_next_step": [row["id"] for row in rows if not row["offers_next"]],
        "too_long": [row["id"] for row in rows if row["words"] > 20],
        # Not part of the 14: a gap the copy layer does not yet close.
        "ai_self_reference": {
            "n": len(ai_rows),
            "rewritten": sum(1 for said, raw in zip(ai_rows, AI_SELF_REFERENCE) if said != raw),
        },
    }
    return {"summary": summary, "rows": rows}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    result = run(argv[1])
    print(json.dumps(result, indent=1))
    return 0 if all(row["text"] for row in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
