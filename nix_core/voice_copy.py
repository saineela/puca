"""The words Luna uses when the pipeline, not the conversation, writes the reply.

Luna's own voice is the model's; nothing here touches its prompt, its persona or
its generation. What this module fixes is the other half of a turn: the places
where Core has to answer *for* Luna because a tool failed, a record was empty or
a skill was not ready. Those paths used to speak in engineering language -

    "I didn't send a device command because the dedicated skill planner did not
     produce a valid, safe plan."

- which is accurate, and sounds like a machine reading a log line. A person in
that situation says "I couldn't get that to the device, and nothing changed."

Two rules govern every line in this file:

* **Never claim more than happened.** A failure says what did not happen
  ("nothing changed", "nothing was sent") and offers a next step. It never
  says "Done." for work that did not run.
* **Say it the way a person would say it to someone in the room.** Short
  sentences, no internal names (planner, worker, digest, status code), and a
  question only when an answer would actually help.

The map is deliberately a list of exact patterns rather than a model call: this
is the last line of the pipeline, where a wording must be predictable and
testable, and where a model outage must not be able to make the assistant sound
worse than silence.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Copy used by Core itself (also imported by brain.format_knowledge_result so
# the deterministic text and the sentinel tests agree)
# ---------------------------------------------------------------------------

KNOWLEDGE_FAILED = (
    "Sorry, that didn't go through — nothing was saved or changed. "
    "Say it again?"
)

# A turn the assistant itself lost: the model was unreachable, or it handed back
# nothing. A person does not report a backend; they say they lost the thread.
LOST_THREAD = "Sorry, could you repeat again? I just had a mind freeze lol."

KNOWLEDGE_EMPTY = "I don't have that yet. Tell me and I'll remember it."

NO_STATES = "I don't have anything on that yet. How are things with them?"

NO_EVENTS = "I can't find anything for that — want me to add something?"

INTRO_ACK = "Got it."

# ---------------------------------------------------------------------------
# Jargon that other paths can emit, and what a person would say instead.
# Each entry is (pattern, replacement). Patterns are matched against the whole
# reply and may appear anywhere in it.
# ---------------------------------------------------------------------------

_REWRITES: tuple[tuple[str, str], ...] = (
    # --- planner refusals ---------------------------------------------------
    (
        r"I didn't send (?:a|any) device (?:command|action) because the "
        r"dedicated skill planner did not produce a valid, safe plan\.?",
        "I couldn't get that to the device this time, and nothing changed. "
        "Want me to try again?",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because I couldn't "
        r"validate the complete plan(?: \([^)]*\))?\.?",
        "I couldn't line that up safely as one plan, so nothing was sent. "
        "We can do it a step at a time if you like.",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because I couldn't "
        r"validate every requested action as one complete plan\.?",
        "I couldn't take every part of that as one safe plan, so nothing was "
        "sent. Want to do them one at a time?",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because the model "
        r"didn't return a valid skill call\.?",
        "I couldn't turn that into something I can do, so nothing changed. "
        "Try saying it a little differently?",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because the skill "
        r"profile did not produce a valid, safe plan\.?",
        "I couldn't turn that into a device action, so nothing changed. "
        "Try naming the device and what you want it to do?",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because I couldn't "
        r"confirm one clear color choice for that request\.?",
        "I couldn't tell which colour you meant, so I left it alone. "
        "Say the colour and I'll set it.",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because the "
        r"requested color didn't match the immediately preceding color offer\. "
        r"Please state the color you want\.?",
        "That wasn't the colour I offered, so I didn't change anything. "
        "Which colour did you want?",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because the "
        r"proposed device didn't match your (?:color|request)[^.]*\.?",
        "That wasn't the light you meant, so nothing changed. "
        "Tell me which one and I'll set it.",
    ),
    (
        r"I didn't send (?:a|any) device (?:command|action) because I couldn't "
        r"validate (?:the )?(?:complete )?plan[^.]*\.?",
        "I couldn't make that a safe action, so nothing changed. "
        "Want to try it another way?",
    ),
    # --- skills that are not ready -----------------------------------------
    (
        r"(?P<name>[A-Z][\w .'-]{0,60}?) isn't ready yet\. Complete its setup "
        r"in Skills, then review and explicitly trust the installed package\.?",
        "\\g<name> isn't set up yet. Finish that in Skills and ask me again?",
    ),
    (
        r"(?P<name>[A-Z][\w .'-]{0,60}?) is installed but untrusted\. Review "
        r"its source and permissions in Skills before explicitly trusting it\.?",
        "\\g<name> is installed, but I'm not allowed to use it until you trust "
        "it in Skills.",
    ),
    (
        r"(?P<name>[A-Z][\w .'-]{0,60}?) isn't ready or trusted, so I didn't "
        r"send a device command\.?",
        "\\g<name> isn't ready for me to use yet. Check it in Skills and ask "
        "again?",
    ),
    (
        r"The targeted skill isn't currently ready or trusted, so I didn't "
        r"send a device command\.?",
        "That skill isn't ready for me to use yet. Check it in Skills and ask "
        "again?",
    ),
    # --- knowledge results --------------------------------------------------
    (
        r"I couldn't complete that request, so nothing was saved or changed\. "
        r"Could you phrase it another way\?",
        KNOWLEDGE_FAILED,
    ),
    (
        r"I don't have that in my knowledge base yet\.?",
        KNOWLEDGE_EMPTY,
    ),
    (
        r"No current states stored yet\.?",
        NO_STATES,
    ),
    (
        r"No matching events found\.?",
        NO_EVENTS,
    ),
    # --- a clinical way of saying "I don't know" ----------------------------
    # The user's own example of a reply that does not sound like a person.
    # The pipeline no longer hands an empty state list to the model (see
    # brain._compose_reply), and this is the floor for any shape that still
    # reaches speech without content behind it.
    (
        r"I don't have any information about (?P<who>your [a-z]+|the "
        r"[a-z]+)(?:'s)?(?: current)? (?:state|status)\.?",
        r"I don't know how \g<who> is doing right now — tell me?",
    ),
    (
        r"I (?:don't|do not) have (?:any )?information (?:about|on) that\.?",
        "I don't know that one — tell me?",
    ),
    # --- the model itself being unreachable ---------------------------------
    # A person does not say "<Name> Alert: Unable to reach the chat model
    # (AcceleratorError)". They say they lost the thread and ask again.
    (
        r"(?:[A-Z][\w'-]{1,20}\s+)?Alert:\s*Unable to reach the chat model"
        r"[^.]*\.?(?:\s*\([^)]*\))?",
        LOST_THREAD,
    ),
    (
        r"Unable to reach the chat model[^.]*\.?(?:\s*\([^)]*\))?",
        LOST_THREAD,
    ),
    (
        r"\(empty reply from chat model\)",
        LOST_THREAD,
    ),
    (
        r"I don't have a preferred name set in NIX Settings yet\.?",
        "You haven't told me what to call you yet. You can set that in "
        "Settings, or just tell me.",
    ),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _REWRITES
)

_WHITESPACE = re.compile(r"[ \t]{2,}")


def humanize_reply(text: str) -> str:
    """Return the reply as a person would say it.

    Only the pipeline's own phrasing is rewritten. A conversational answer
    from Luna does not match any pattern and is returned unchanged apart from
    collapsed runs of spaces, so this cannot flatten the model's voice.
    """
    if not text:
        return text
    out = str(text)
    for pattern, replacement in _COMPILED:
        out = pattern.sub(replacement, out)
    return _WHITESPACE.sub(" ", out).strip()
