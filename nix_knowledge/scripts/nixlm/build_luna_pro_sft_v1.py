"""Archived Luna Pro v1 SFT candidate builder; execution is disabled.

The historical implementation is retained for auditability only. Luna Pro is
retired, its data-build workflow is non-actionable, and its training approach
must not be reused for any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import load_dataset
from transformers import AutoTokenizer

from luna_constraint_validator import validate_objective_constraints

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "luna_pro_data_manifest.json"
BASE_PATH = Path("nix_knowledge/models/llama-3.2-3b-unsloth-instruct")
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
MAX_SEQUENCE_TOKENS = 1536
MAX_MESSAGE_TOKENS = 1024

DEFAULT_SYSTEM = (
    "You are Luna, an independent conversational model in Nix's PUCA system. "
    "Casper is Nix's user-facing PUCA identity; Luna is separate from Casper. "
    "Be natural, concise, and answer the latest user message directly. Use only "
    "facts supplied in the conversation or trusted context. Do not invent "
    "memories, relationships, states, biographies, or actions. Do not claim an "
    "action succeeded unless a positive Nix Actions result is explicitly present. "
    "When context is missing or a person is ambiguous, state what is unknown or "
    "ask a focused clarification."
)

BOILERPLATE = re.compile(
    r"\b(?:as an ai(?: language model)?|as a language model|i am an ai|"
    r"i cannot personally|certainly[,!]|of course[,!]? i(?:'| a)m here to help|"
    r"please let me know if you need anything else)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_TOPICAL_SENSITIVE_DOMAIN = re.compile(
    r"\b(?:mental health|mentally ill|psychosis|psychotic|manic)\b|"
    r"\b(?:seem|seems|seemed)\s+(?:a\s+bit\s+)?unstable\b|"
    r"\b(?:surrogat(?:e|ion)|fertility|infertility)\b|"
    r"\b(?:cat|dog|pet|animal)(?:'s|s)?\b.{0,50}\b(?:food|diet|nutrition|health|sick|illness|symptom|treatment|veterinar(?:y|ian)|vet)\b|"
    r"\b(?:food|diet|nutrition|health|treatment|veterinar(?:y|ian)|vet)\b.{0,50}\b(?:cat|dog|pet|animal)(?:'s|s)?\b",
    re.IGNORECASE,
)
# Topical-Chat is human-human dialogue. Once the second speaker is mapped to
# Luna's assistant role, first-person biography, identity, preferences, habits,
# and lived experiences would become false claims by the model. Also reject
# direct personal-experience questions from either speaker, even where the
# reply is elliptical (e.g. "Not yet" or "All the time").
_TOPICAL_PERSONAL_QUESTION = re.compile(
    r"\b(?:do|did|have|has|are|were|would|could|can)\s+you\s+"
    r"(?:(?:ever|usually|often|sometimes|still|also|really|actually)\s+)*"
    r"(?:like|love|enjoy|prefer|dislike|hate|follow|watch(?:ed)?|listen(?:ed)?(?:\s+to)?|read|"
    r"use(?:d)?|buy|play(?:ed)?|support|vote|visit(?:ed)?|attend(?:ed)?|remember|believe|think|"
    r"feel|find|see|seen|hear|heard|try|tried|own(?:ed)?|live(?:d)?|work(?:ed)?|grow\s+up|"
    r"been|cheer|go\s+to|have|rather)\b|"
    r"\b(?:do|did|have|has)\s+you\s+have\s+(?:a|an)\s+(?:favorite|favourite|pet|alma\s+mater|religion|political party)\b|"
    r"\bwho\s+(?:are|were)\s+you\s+(?:cheering|rooting|voting|supporting)\s+for\b|"
    r"\bwhat\s+team\s+are\s+you\s+(?:cheering|rooting|supporting)\s+for\b|"
    r"\bwhich\s+(?:team|party|candidate)\s+(?:do|would)\s+you\s+(?:support|vote)\s+for\b|"
    r"\bwhich\s+(?:[a-z'-]+\s+){0,2}(?:do|would|did)\s+you\s+(?:expect|think|guess|predict|choose|pick|prefer|support|like)\b|"
    r"\bwhich\s+(?:team|player|candidate|party)\s+(?:will|would|might|could)\s+you\s+(?:choose|pick|support|prefer)\b|"
    r"\bwhat\s+is\s+your\s+alma\s+mater\b|"
    r"\b(?:would|could|can|will|do|did|are|were)\s+you\s+(?:be\s+)?(?:interested\s+in|into|a\s+fan\s+of)\b|"
    r"\b(?:what|who|which)\s+(?:is|are)\s+your\s+(?:favorite|favourite)\b|"
    r"\bwhat\s+are\s+your\s+(?:hobbies|interests|favorite|favourite)\b|"
    r"\bare\s+you\s+on\s+(?:facebook|instagram|twitter|tiktok|social\s+media)\b|"
    r"\bdo\s+you\s+agree\b|\bdo\s+you\s+know\s+(?:who|what|where|when)\b|"
    r"\bhow\s+are\s+you\b|\bhow\s+do\s+you\s+feel\b|"
    r"\b(?:what|who|where|when|why|how|which)\s+(?:do|would|did)\s+you\s+"
    r"(?:(?:usually|often|sometimes)\s+)*(?:like|love|enjoy|prefer|dislike|hate|follow|watch|"
    r"listen|read|use|buy|play|support|vote|visit|attend|remember|believe|think|feel|find|"
    r"work|live|do|choose|cheer|expect|guess|predict)\b|"
    r"\b(?:what|who|where|when)\s+are\s+you\s+(?:from|living|working|cheering|supporting)\b|"
    r"\bwhat(?:'s| is)\s+your\s+(?:favorite|favourite|opinion|belief|religion|political party|experience)\b|"
    r"\b(?:are|were)\s+you\s+(?:a|an|religious|married|single|politically|interested|comfortable|excited|aware|familiar)\b|"
    r"\bhow\s+do\s+you\s+feel\b|\bhow\s+old\s+are\s+you\b|"
    r"\bhow\s+many\s+(?:children|kids|siblings|brothers|sisters|pets|dogs|cats)\s+do\s+you\s+have\b",
    re.IGNORECASE,
)
_TOPICAL_PERSONAL_CLAIM = re.compile(
    r"\b(?:i|we)\s+"
    r"(?:(?:(?:do|does|did|have|has|will|would|can|could|should|might|must)\s+not|"
    r"(?:don't|doesn't|didn't|haven't|hasn't|won't|wouldn't|can't|couldn't|shouldn't)|never)\s+)*"
    r"(?:(?:really|pretty|actually|just|also|definitely|still|always|often|usually|sometimes)\s+)*"
    r"(?:have\s+to\s+(?:go\s+with|say)|would\s+go\s+with|go\s+with|"
    r"like|love|enjoy|prefer|dislike|hate|appreciate|admire|follow|watch|listen|read|use|buy|bought|play|"
    r"support|vote|visit|attend|remember|believe|think|feel|find|own|live|work|grew\s+up|grow\s+up|"
    r"used\s+to|use\s+to|went\s+to|came\s+from|was\s+born|have\s+been|have\s+seen|"
    r"saw|had|was|were|have\s+watched|have\s+read|have\s+a\s+(?:favorite|favourite|pet)|had\s+a\s+"
    r"(?:favorite|favourite|pet)|am\s+(?:a|an|religious|not\s+religious|married|single|from|"
    r"excited|happy|glad|interested|doing\s+(?:good|well|great|fine)|(?:good|well|great|fine|okay|ok)))|"
    r"\bi(?:['’]?m|\s+am)\s+(?:(?:really|pretty|actually|just|also|definitely|still)\s+)*"
    r"(?:a|an)\s+[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3}\b|"
    r"\bi(?:['’]?m|\s+am)\s+(?:(?:really|pretty|actually|just|also|definitely|still)\s+)*"
    r"(?:religious|not\s+religious|married|single|from|excited|happy|glad|interested|"
    r"doing\s+(?:good|well|great|fine)|(?:good|well|great|fine|okay|ok))\b|"
    r"\bmy\s+(?:favorite|favourite|alma\s+mater|religion|political party|wife|husband|partner|"
    r"boyfriend|girlfriend|son|daughter|child(?:ren)?|kids|mom|mother|dad|father|brother|sister|"
    r"family|friend|roommate|coworker|boss|neighbor|pet|dog|cat|home|house|car|job|hometown)\b|"
    r"\bi(?:['’]?m|\s+am)\s+(?:from|originally\s+from)\b|"
    r"\bi\s+(?:am|'m)\s+(?:a|an)\s+fan\s+of\b|"
    r"\bi\s+agree\s+with\b|"
    r"\bwhen\s+i\s+was\s+(?:a\s+)?(?:kid|child|young|in\s+(?:school|college|the\s+military))\b|"
    r"\bi\s+(?:can't|cannot)\s+wait\s+to\b",
    re.IGNORECASE,
)
_TOPICAL_FIRST_PERSON_REPLY = re.compile(
    r"^\s*(?:(?:yes|yeah|yep|no|nope|well|oh|ah|sure|right|exactly|i guess|maybe|hmm|"
    r"i do|i did|i have|i had|i am|i'm|i will|i would|i can|i could)\b[^.!?]*[.!?]?\s*){1,3}$",
    re.IGNORECASE,
)
_TOPICAL_FIRST_PERSON_EMOTION = re.compile(
    r"\b(?:i|we)(?:['’]m|['’]re|\s+am|\s+are|\s+was|\s+were)\s+"
    r"(?:(?:really|pretty|actually|just|also|definitely|still|so)\s+)*"
    r"(?:surprised|shocked|amazed|excited|sad|proud|worried|scared|afraid|nervous|"
    r"nostalgic|disappointed|annoyed|frustrated|embarrassed|lonely|homesick)\b",
    re.IGNORECASE,
)
_TOPICAL_HUMAN_RELATIONSHIP = re.compile(
    r"\b(?:i|we)(?:['’]m|['’]re|\s+am|\s+are|\s+was|\s+were|\s+have|\s+had|\s+raised|\s+raise|\s+raising)\s+"
    r"(?:(?:really|pretty|actually|just|also|definitely|still|a|an|the|my|our|proud|happy|single|married|two|three|four|five|one)\s+){0,4}"
    r"(?:father|mother|parent|mom|dad|husband|wife|spouse|partner|boyfriend|girlfriend|"
    r"son|daughter|child|children|kid|kids|brother|sister|sibling|grandmother|grandfather|"
    r"grandparent|uncle|aunt|cousin|roommate|coworker|neighbor|pet owner)\b|"
    r"\b(?:as|being)\s+(?:a|an)\s+(?:father|mother|parent|mom|dad|husband|wife|spouse|"
    r"grandparent|uncle|aunt|pet owner)\b|"
    r"\b(?:i|we)\s+(?:have|had|raised|am raising|was raising)\s+"
    r"(?:(?:one|two|three|four|five|six|seven|eight|nine|\d+)\s+)?"
    r"(?:children|kids|sons|daughters|a son|a daughter)\b",
    re.IGNORECASE,
)
_TOPICAL_SAFE_ABSTENTION = re.compile(
    r"^\s*(?:i\s+(?:do not|don't)\s+know(?:\s+(?:if|whether|who|what|where|when|why|how)\b[^.!?]*)?|"
    r"i\s+(?:am|'m)\s+not\s+sure(?:\s+(?:if|whether)\b[^.!?]*)?|"
    r"i\s+have\s+no\s+idea(?:\s+(?:who|what|where|when|why|how|if|whether)\b[^.!?]*)?|"
    r"i\s+wonder(?:\s+why\b[^.!?]*)?)[.!?]*\s*$",
    re.IGNORECASE,
)
_TOPICAL_FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.IGNORECASE)
_TOPICAL_SHARED_EXPERIENCE = re.compile(
    r"\b(?:same here|same for me|me too|me neither|neither do i|so do i|so did i|"
    r"so have i|i agree|i think so too)\b",
    re.IGNORECASE,
)
_HIGH_STAKES = re.compile(
    r"\b(?:medical|healthcare|health care|health|healthy|wellness|doctor|physician|nurse|clinical|"
    r"diagnos(?:e|is|ed|ing)|prescri(?:be|ption)|medication|medicine|drug interaction|"
    r"dosage|dose|side effects?|sleep|insomnia|dream interpretation|pets?|cats?|dogs?|veterinar(?:y|ian)|"
    r"animal (?:health|care|nutrition)|pet food|nutrition|supplement|patient|"
    r"medical record|lab results?|blood tests?|hormones?|thyroid|syndrome|disorder|clinical trial|"
    r"legal advice|lawyer|attorney|lawsuit|court filing|investment advice|financial advice|tax advice|"
    r"self[- ]harm|suicid(?:e|al)|overdose|emergency room|first aid|CPR|bleeding|wound care|wounds?|scrapes?|"
    r"burns?|stitches|deep cut|blood pressure|heart rate|chest pain|shortness of breath|"
    r"symptom(?:s)?|treatment plan|therapy recommendation|therapist|counseling|"
    r"anxiety|depression|panic attack|trauma|infection|allerg(?:y|ies|ic)|"
    r"pregnan(?:t|cy)|surrogat(?:e|ion)|fertility|infertility|vaccin(?:e|ation)|surgery|hospital|illness|disease|cancer|"
    r"injur(?:y|ies)|fever|seizure|stroke|heart attack|car trouble|car maintenance|oil changes?|brake pads?|"
    r"\b(?:i|you|he|she|they)\s+(?:accidentally\s+)?cut\s+(?:my|your|his|her|their)\s+"
    r"(?:hand|finger|arm|leg|foot|face))\b",
    re.IGNORECASE,
)
_SANITY_SENSITIVE_ROLEPLAY = re.compile(
    r"\b(?:paranormal|haunt(?:ed|ing)|ghosts?|spirits?|apparitions?|emf meter|"
    r"electronic voice phenomenon|evp session|communicate with (?:the )?dead)\b",
    re.IGNORECASE,
)
_EVERYDAY_VOLATILE = re.compile(
    r"\b(?:weather|forecast|storm(?: warning| coming| expected)?|blizzard|hurricane|tornado warning|"
    r"near me|in my (?:area|city|town|neighborhood|location)|local (?:events?|parks?|attractions?|concerts?|festivals?)|"
    r"coupon(?:s)?|promo(?:tion)? code|voucher|discount|return policy|refund policy|"
    r"amusement park|theme park|campground|campsite|summer fest|festival schedule|concert schedule|"
    r"train schedule|train times?|direct trains?|flight schedule|flight times?|airport transfer|"
    r"taxi service|ride[- ]hailing|uber|lyft|travel time|ticket prices?|hotel availability|"
    r"opening hours|store hours|local recycling program|current exchange rate|"
    r"best time to visit|when to visit|places to visit|where to (?:stay|eat|go)|place to (?:stay|eat|go)|"
    r"travel to|trip to|flight to|hotel near|resort|guided tour|tourist attraction|"
    r"recommend.{0,40} (?:near|around) me|recommend.{0,40} (?:place|hotel|restaurant|destination)|"
    r"things to do (?:in|near)|"
    r"offers? (?:a|an|the|me|you)|discounts? for|can you send.{0,30} (?:phone|email)|"
    r"what(?:'s| is) on (?:this|next) (?:week|weekend|month)|"
    r"current (?:schedule|availability|price|rate)|real[- ]time|right[- ]now availability)\b",
    re.IGNORECASE,
)
_REVIEW_SENSITIVE_DOMAIN = re.compile(
    r"\b(?:history of|historical|ancient|archaeolog(?:y|ical)|recent study|latest study|"
    r"research shows|research suggests|study finds|research findings|new discovery|"
    r"climate change|ice loss|population declines?|statistics show|according to (?:a|the) study|"
    r"government agency|legal requirement|regulation|compliance requirement|"
    r"financial services|tax liability|market trends?|economic forecast)\b",
    re.IGNORECASE,
)
_RELIABILITY_SENSITIVE_OPEN_DOMAIN = re.compile(
    r"\b(?:best practices? for (?:a )?healthy lifestyle|stay healthy while traveling|"
    r"dreams? (?:can|may|might|often) (?:represent|symboli[sz]e|mean)|"
    r"(\d{1,2}[,.]?\d{0,3}\s*(?:to|[-–])\s*\d{1,2}[,.]?\d{0,3}\s*(?:miles?|hours?|minutes?))|"
    r"oil change every|recommended to change (?:your )?(?:car'?s )?oil|"
    r"most online stores have a return policy)\b",
    re.IGNORECASE,
)
_TIME_SENSITIVE = re.compile(
    r"\b(?:currently|right now|at the moment|as of (?:today|now|\d{4})|today's|"
    r"this season|this week|this month|this year|next month|next year|opening soon|available on|"
    r"is now available|currently available|latest (?:news|update|version|results)|"
    r"recently released|newly released|tomorrow's weather|forecast for|"
    r"scheduled to be completed by|currently undergoing|current(?:ly)? trends?|"
    r"most (?:popular|watched|streamed|followed|recent)|(?:popular|trending|chart[- ]topping)\s+"
    r"(?:song|album|artist|film|movie|show|series|book|game|app|product)|"
    r"top[- ](?:selling|grossing|charting)\s+(?:song|album|artist|film|movie|show|series|book|game|product)|"
    r"number one\s+(?:song|album|artist|film|movie|show|series|book|game)|"
    r"streaming availability|available to stream|streaming on|box office|top of the charts|"
    r"on netflix|on hulu|on disney|on prime video|on amazon prime video|on hbo max|on max|"
    r"netflix availability|hulu availability|disney availability|prime video availability)\b",
    re.IGNORECASE,
)
_ACTION_CLAIM = re.compile(
    r"\b(?:i|we)\s+(?:(?:have|'ve)\s+)?(?:successfully\s+)?"
    r"(?:created|scheduled|set up|booked|sent|submitted|cancelled|canceled|"
    r"deleted|updated|completed|placed|ordered|added|changed|removed|confirmed)\b"
    r"|\b(?:it|that|the reminder|your reminder|the event|your event|"
    r"the appointment|your appointment|the reservation|your reservation|"
    r"the message|your message|the email|your email|the order|your order)\s+"
    r"(?:has been|was|is|got)\s+(?:successfully\s+)?"
    r"(?:created|scheduled|set up|booked|sent|submitted|cancelled|canceled|"
    r"deleted|updated|completed|placed|ordered|added|changed|removed|confirmed)\b"
    r"|\b(?:done|all set|you're booked|you are booked|it's on your calendar|"
    r"it is on your calendar|it went through)\b",
    re.IGNORECASE,
)
_SIMULATED_ACTION_RESULT = re.compile(r"\bsimulated test action result\b", re.I)
_NEGATIVE_ACTION = re.compile(
    r"\b(?:failed|failure|unsuccessful|not created|wasn't created|was not created|"
    r"didn't create|did not create|did not succeed|not successful)\b",
    re.I,
)
_AMBIGUOUS_ACTION_RESULT = re.compile(r";|\b(?:separately|another|different|but|however|whereas|although)\b", re.I)
_NEGATIVE_ACTION_QUERY = re.compile(
    r"\b(?:failed|failure|unsuccessful|not created|wasn't created|was not created|"
    r"didn't|did not|did it fail|did it not)\b", re.I,
)
_ACTION_RESULT_STATUS = re.compile(
    r"\b(?:successfully\s+(?:created|scheduled|completed|booked|sent|updated)|"
    r"(?:created|scheduled|completed|booked|sent|updated)\s+successfully|success)\b", re.I,
)
_ACTION_OBJECT = re.compile(r"\b(?:reminder|event|appointment|reservation|message|email|order|task|booking|payment)\b", re.I)
_SAVED_ACTION_CLAIM = re.compile(
    r"\b(?:i|we)\s+(?:(?:have|'ve)\s+)?(?:successfully\s+)?saved\s+"
    r"(?:(?:the|your|a|an|my|our)\s+)?(?:reminder|event|appointment|reservation|"
    r"message|email|order|task|booking|payment|file|document)\b"
    r"|\b(?:it|that|the reminder|your reminder|the event|your event|"
    r"the appointment|your appointment|the reservation|your reservation|"
    r"the message|your message|the email|your email|the order|your order|"
    r"the task|your task)\s+(?:has been|was|is|got)\s+(?:successfully\s+)?saved\b", re.I,
)
_NEGATED_ACTION_PREFIX = re.compile(
    r"(?:\b(?:i|we)\s+(?:can't|cannot|won't|will not|don't|do not|didn't|did not|never)\s+"
    r"(?:(?:say|claim|confirm|report|state|tell(?: you)?|pretend)\s+)?(?:that\s+)?"
    r"|\b(?:not sure if|unsure whether|don't know whether|do not know whether|"
    r"can't tell whether|cannot tell whether)\s*)$", re.I,
)
_LOW_VALUE_GENERIC = re.compile(r"\b(?:it is impossible to answer|there is no way to determine)\b", re.I)
_OPEN_ENDED_REQUEST = re.compile(
    r"\b(?:what\s+(?:is|are|was|were|does|do)|difference\s+between|explain|describe|"
    r"how\s+(?:do|does|did|can|should|would|is|are)|why|list|name|summari[sz]e|"
    r"write|create|generate|find|solve|compare|outline|define|calculate|translate)\b", re.I,
)
_SHORT_CONFIRMATIONS = frozenset({
    "yes", "no", "yes it is", "no it is not", "no it isnt", "yes it does", "no it does not",
    "no it doesnt", "yes certainly", "yes absolutely", "no absolutely not",
})
_RELEVANCE_STOPWORDS = frozenset("""
about after again also although always another any are around ask asked asking at away back
because been before being between both but by can could did do does doing done down during each
else enough every first following for from get got give given had has have having he help her here
him his how however i if in into is it its itself just know like long look looking make many may
me might more most much must my need needed needs neither never new next no nor not of off on once
one only or other our out over own perhaps please provide question rather really same say says see
seem several she should since so some something still such take tell than that the their them then
there these they this those through time to together too try two under until up use used using very
want wanted wants was way we well were what when where whether which while who why will with within
without would write wrote your yours response answer users user based consider describe explain
create generate assist assistance given
""".split())


def normalize_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_messages(raw: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or item.get("from") or "").casefold().strip()
        role = {"human": "user", "gpt": "assistant", "model": "assistant"}.get(role, role)
        if role not in {"system", "user", "assistant"}:
            return None
        content = item.get("content", item.get("value", ""))
        if not isinstance(content, str):
            return None
        content = content.replace("\x00", "").strip()
        if not content:
            return None
        if role == "system":
            if messages:
                return None
            messages.append({"role": "system", "content": content})
        else:
            messages.append({"role": role, "content": content})
    dialogue = messages[1:] if messages[0]["role"] == "system" else messages
    if len(dialogue) < 2 or dialogue[0]["role"] != "user":
        return None
    if any(message["role"] != ("user" if i % 2 == 0 else "assistant") for i, message in enumerate(dialogue)):
        return None
    if dialogue[-1]["role"] != "assistant":
        return None
    return messages


def constraint_target_valid(prompt: str, answer: str) -> bool:
    return validate_objective_constraints(prompt, answer)


def make_row(messages: list[dict[str, str]], source: str, category: str, group: str = "") -> dict[str, Any] | None:
    prefix: list[str] = []
    dialogue: list[dict[str, str]] = []
    for message in messages:
        content = message["content"].strip()
        if message["role"] == "system":
            if dialogue:
                return None
            prefix.append(content)
        else:
            dialogue.append({"role": message["role"], "content": content})
    if not dialogue or dialogue[0]["role"] != "user":
        return None
    if prefix:
        dialogue[0]["content"] = "Task instructions: " + "\n\n".join(prefix) + "\n\n" + dialogue[0]["content"]
    if any(message["role"] != ("user" if i % 2 == 0 else "assistant") for i, message in enumerate(dialogue)) or dialogue[-1]["role"] != "assistant":
        return None
    return {
        "source": source,
        "category": category,
        "group_id": group,
        "source_user_turns": [message["content"] for message in dialogue if message["role"] == "user"],
        "messages": [{"role": "system", "content": DEFAULT_SYSTEM}, *dialogue],
    }


PROJECT_TRAJECTORIES: tuple[dict[str, Any], ...] = (
    {"category": "memory_missing", "context": None, "turns": (("How has Anya Pell been lately?", "I don't have a current update on Anya, so I don't want to guess."),)},
    {"category": "memory_grounded", "context": "SIMULATED TEST MEMORY (fictional): the newest note says Sol Wynn's ankle has recovered. No other current details are available.", "turns": (("How is Sol Wynn doing?", "The latest note says Sol's ankle has recovered. I don't have other current details."),)},
    {"category": "irrelevant_context", "context": "SIMULATED TEST MEMORY (fictional, unrelated): the user collects fountain pens. This detail is not relevant unless the user brings it up.", "turns": (("The committee accepted my poster!", "That's wonderful news—congratulations on getting your poster accepted."),)},
    {"category": "clarification", "context": None, "turns": (("My lab partner has been quiet lately. How is he doing?", "Which lab partner do you mean? I don't have an update about either person yet."), ("I mean Ren Vela.", "Thanks for clarifying—you mean Ren. I don't have a current update about him."))},
    {"category": "state_supersession", "context": "SIMULATED TEST MEMORY (fictional): an older note says Oren felt discouraged before rehearsal. A newer update says the projector was replaced and Oren felt relieved today. The newer update supersedes the old one.", "turns": (("What's the latest on Oren?", "The latest update is that Oren felt relieved after the projector was replaced today."),)},
    {"category": "correction", "context": None, "turns": (("My new mug is sage green, not turquoise.", "Got it—the new mug is sage green, not turquoise."), ("I checked again: it's actually pine green, not sage.", "Understood—the latest correction is pine green."), ("What color is my new mug?", "Pine green."))},
    {"category": "action_pending", "context": None, "turns": (("Please set a greenhouse pump inspection reminder to repeat every five days.", "I can't claim it was scheduled until Nix Actions confirms."),)},
    {"category": "action_success", "context": "SIMULATED TEST ACTION RESULT (fictional): Nix Actions successfully created the greenhouse pump inspection reminder to repeat every 5 days. No time of day was requested or set.", "turns": (("What is the schedule?", "It was set to repeat every five days. No time of day was specified."),)},
    {"category": "action_failure", "context": "SIMULATED TEST ACTION RESULT (fictional): creating the studio filter reminder failed because no repeat interval was provided. No reminder exists.", "turns": (("What was the outcome for the studio filter alert request?", "No. The reminder wasn't created because the repeat interval was missing."),)},
    {"category": "unrelated_positive", "context": None, "turns": (("My ceramic lamp was selected for a local exhibit!", "That's lovely—congratulations on having your lamp selected."),)},
    {"category": "brevity", "context": None, "turns": (("I'm tired. Keep your reply short and don't ask a follow-up.", "Okay. I'll keep it brief."),)},
    {"category": "identity", "context": None, "turns": (("Are you Casper?", "No. I'm Luna, a separate conversation model. Casper is Nix's user-facing PUCA identity."),)},
    {"category": "everyday_chat", "context": None, "turns": (("Tell me a quick joke about a sleepy telescope.", "Why did the telescope go to bed early? It was ready to call it a night sky."),)},
    {"category": "multi_intent", "context": None, "turns": (("I like jasmine tea. Tell me a short joke too.", "Jasmine tea, got it for this chat. Why did the tea cup blush? It saw the saucer looking at it."),)},
)


def project_rows(tokenizer: Any | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in PROJECT_TRAJECTORIES:
        system = DEFAULT_SYSTEM + ("\n\n" + item["context"] if item["context"] else "")
        messages = [{"role": "system", "content": system}]
        for user, assistant in item["turns"]:
            messages.extend(({"role": "user", "content": user}, {"role": "assistant", "content": assistant}))
        row = {
            "source": "project_authored",
            "category": item["category"],
            "group_id": f"project:{item['category']}",
            "source_user_turns": [user for user, _ in item["turns"]],
            "messages": messages,
        }
        if tokenizer is not None:
            fits, token_count, message_max = tokenize_row(row, tokenizer)
            if not fits or message_max > MAX_MESSAGE_TOKENS:
                raise ValueError(f"Project-authored trajectory exceeds token limits: {item['category']}")
            row["token_count"] = token_count
        rows.append(row)
    return rows


def is_positive_action_result(messages_before_target: Sequence[dict[str, str]], claimed_action: str = "") -> bool:
    result_lines: list[str] = []
    user_context: list[str] = []
    for message in messages_before_target:
        content = str(message.get("content") or "")
        if message.get("role") == "user":
            user_context.append(content)
        elif message.get("role") == "system":
            result_lines.extend(line.strip() for line in content.splitlines() if _SIMULATED_ACTION_RESULT.search(line))
    if len(result_lines) != 1:
        return False
    result = result_lines[0]
    if (
        _NEGATIVE_ACTION.search(result)
        or _AMBIGUOUS_ACTION_RESULT.search(result)
        or len(re.findall(r"\bnix actions\b", result, re.I)) != 1
        or len(_ACTION_RESULT_STATUS.findall(result)) != 1
        or len({item.casefold() for item in _ACTION_OBJECT.findall(result)}) > 1
    ):
        return False
    request_text = " ".join(user_context)
    if _NEGATIVE_ACTION_QUERY.search(request_text):
        return False
    requested_objects = {item.casefold() for item in _ACTION_OBJECT.findall(request_text)}
    result_objects = {item.casefold() for item in _ACTION_OBJECT.findall(result)}
    if requested_objects and result_objects and requested_objects.isdisjoint(result_objects):
        return False
    if claimed_action and result_objects:
        claimed_objects = {item.casefold() for item in _ACTION_OBJECT.findall(claimed_action)}
        if claimed_objects and claimed_objects.isdisjoint(result_objects):
            return False
    return True


def assistant_claims_action(answer: str) -> bool:
    normalized = str(answer or "").replace("’", "'")
    if _SAVED_ACTION_CLAIM.search(normalized):
        return True
    for match in _ACTION_CLAIM.finditer(normalized):
        prefix = normalized[max(0, match.start() - 100):match.start()]
        if not _NEGATED_ACTION_PREFIX.search(prefix):
            return True
    return False


def has_unsupported_action_claim(messages: Sequence[dict[str, str]]) -> bool:
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and assistant_claims_action(str(message.get("content") or "")):
            if not is_positive_action_result(messages[:index], str(message.get("content") or "")):
                return True
    return False


def strip_standard_everyday_greeting(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool]:
    offset = 1 if messages and messages[0]["role"] == "system" else 0
    dialogue = messages[offset:]
    if len(dialogue) < 4:
        return messages, False
    greeting = normalize_text(dialogue[0]["content"]) in {"hi", "hello", "hey", "hey there", "hi there"}
    reply = normalize_text(dialogue[1]["content"]) in {
        "hello how can i help you today", "hi how can i help you today",
        "hello how can i help you", "hi how can i help you",
    }
    if not (greeting and reply):
        return messages, False
    trimmed = messages[:offset] + dialogue[2:]
    if len(trimmed[offset:]) < 2 or trimmed[offset]["role"] != "user" or trimmed[-1]["role"] != "assistant":
        return messages, False
    return trimmed, True


def _has_repetitive_ngram(text: str) -> bool:
    tokens = normalize_text(text).split()
    width = 8
    if len(tokens) < width * 3:
        return False
    minimum_repeated_tokens = max(width * 3, int(len(tokens) * 0.25))
    occurrences: dict[tuple[str, ...], tuple[int, int]] = {}
    for start in range(len(tokens) - width + 1):
        phrase = tuple(tokens[start:start + width])
        count, last_end = occurrences.get(phrase, (0, -1))
        if start >= last_end:
            count += 1
            last_end = start + width
            if count >= 3 and count * width >= minimum_repeated_tokens:
                return True
        occurrences[phrase] = (count, last_end)
    return False


def _has_repetitive_messages(messages: Sequence[dict[str, str]]) -> bool:
    return any(_has_repetitive_ngram(str(message.get("content") or "")) for message in messages)


def _underresponsive_to_open_request(messages: Sequence[dict[str, str]]) -> bool:
    user = next((str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"), "")
    answer = next((str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "assistant"), "")
    return normalize_text(answer) in _SHORT_CONFIRMATIONS and bool(_OPEN_ENDED_REQUEST.search(user))


def _content_terms(text: str) -> set[str]:
    words = normalize_text(text).split()
    terms = set(words) - _RELEVANCE_STOPWORDS
    normalized: set[str] = set()
    for term in terms:
        if len(term) < 4:
            continue
        if term.endswith("ies") and len(term) > 5:
            term = term[:-3] + "y"
        elif term.endswith("ing") and len(term) > 6:
            term = term[:-3]
        elif term.endswith("ed") and len(term) > 5:
            term = term[:-2]
        elif term.endswith("s") and len(term) > 4:
            term = term[:-1]
        normalized.add(term)
    for left, right in zip(words, words[1:]):
        if left not in _RELEVANCE_STOPWORDS and right not in _RELEVANCE_STOPWORDS and len(left) >= 3 and len(right) >= 3:
            normalized.add(f"{left} {right}")
    return normalized


def _has_no_content_anchors(messages: Sequence[dict[str, str]]) -> bool:
    user = next((str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "user"), "")
    answer = next((str(item.get("content") or "") for item in reversed(messages) if item.get("role") == "assistant"), "")
    return (
        len(normalize_text(user).split()) >= 8
        and len(normalize_text(answer).split()) >= 25
        and not (_content_terms(user) & _content_terms(answer))
    )


def tokenize_row(
    row: dict[str, Any], tokenizer: Any, *,
    max_sequence_tokens: int = MAX_SEQUENCE_TOKENS,
    max_message_tokens: int = MAX_MESSAGE_TOKENS,
) -> tuple[bool, int, int]:
    message_max = 0
    too_long = False
    for message in row["messages"]:
        count = len(tokenizer.encode(str(message["content"]), add_special_tokens=False))
        message_max = max(message_max, count)
        too_long = too_long or count > max_message_tokens
    if too_long:
        return False, 0, message_max
    encoded = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False, date_string="01 Jan 2000"
    )
    if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    total = len(encoded)
    return total <= max_sequence_tokens, total, message_max


def build_benchmark_prompt_set(shared_prompts: Iterable[str], probe_prompts: Iterable[str], ifeval_prompts: Iterable[str]) -> set[str]:
    return {
        normalized
        for value in (*tuple(shared_prompts), *tuple(probe_prompts), *tuple(ifeval_prompts))
        if (normalized := normalize_text(value))
    }


def _benchmark_overlap(row: dict[str, Any], benchmark_prompts: set[str]) -> bool:
    users = row.get("source_user_turns") or [
        message.get("content", "")
        for message in row.get("messages", [])
        if message.get("role") == "user"
    ]
    return any(normalize_text(user) in benchmark_prompts for user in users)


def load_benchmark_prompts() -> tuple[set[str], dict[str, int]]:
    core = REPO_ROOT / "nix_core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from casper_v6_eval import CASES as single_cases
    from casper_v6_multiturn_eval import CASES as multi_cases
    from luna_generalization_probes import PROBES
    shared = [case.request for case in single_cases]
    shared.extend(turn for case in multi_cases for turn in case.turns)
    probes = [turn["user"] for probe in PROBES for turn in probe["turns"]]
    ifeval_rows = load_dataset("google/IFEval", split="train", streaming=True, revision=IFEVAL_REVISION)
    ifeval_prompts = [str(row.get("prompt") or "") for row in ifeval_rows.take(1000)]
    return build_benchmark_prompt_set(shared, probes, ifeval_prompts), {
        "shared_v6": len(shared), "luna_generalization_probes": len(probes), "ifeval": len(ifeval_prompts),
    }


def _is_topical_conversation(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("content"), list)


def _with_topical_conversation_id(record: dict[str, Any], conversation_id: Any) -> dict[str, Any]:
    identified = dict(record)
    identified.setdefault("conversation_id", str(conversation_id))
    return identified


def _flatten_topical_records(value: Any) -> list[dict[str, Any]]:
    if _is_topical_conversation(value):
        return [value]
    if isinstance(value, list):
        # The pinned HF mirror stores the original mapping as alternating
        # [conversation_id, conversation, conversation_id, conversation, ...].
        if len(value) >= 2 and len(value) % 2 == 0:
            paired_rows: list[dict[str, Any]] = []
            for index in range(0, len(value), 2):
                conversation_id, record = value[index:index + 2]
                if (
                    not isinstance(conversation_id, (str, int))
                    or isinstance(conversation_id, bool)
                    or not _is_topical_conversation(record)
                ):
                    paired_rows = []
                    break
                paired_rows.append(_with_topical_conversation_id(record, conversation_id))
            if paired_rows:
                return paired_rows
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_flatten_topical_records(item))
        return rows
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if _is_topical_conversation(item):
                rows.append(_with_topical_conversation_id(item, key))
            else:
                rows.extend(_flatten_topical_records(item))
        return rows
    return []


def parse_topical_chat_records(payload: str) -> list[dict[str, Any]]:
    """Parse Topical-Chat's per-line [conversation_id, conversation] JSON encoding."""
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        records = _flatten_topical_records(decoded)
        if records:
            return records
        raise ValueError("Topical-Chat JSON payload contained no conversation records")
    records = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.extend(_flatten_topical_records(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Topical-Chat JSONL at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("Topical-Chat JSONL contained no conversation records")
    return records


def load_topical_chat_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    url = str(source["url"]).format(revision=source["revision"])
    request = urllib.request.Request(url, headers={"User-Agent": "LunaProDataAudit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(64 * 1024 * 1024 + 1)
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("Pinned Topical-Chat training file exceeded the 64 MiB safety limit")
    return parse_topical_chat_records(payload.decode("utf-8-sig"))


def load_streaming_source(source: dict[str, Any]):
    if source.get("loader") == "jsonl_url":
        revision = str(source["revision"])
        url = str(source["url"]).format(revision=revision)
        return load_dataset(
            "json",
            data_files={"train": url},
            split="train",
            streaming=True,
        )
    kwargs: dict[str, Any] = {
        "path": source["repo"],
        "split": source["split"],
        "streaming": True,
        "revision": source["revision"],
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    return load_dataset(**kwargs)


def topical_chat_target_has_personal_claim(user_message: str, assistant_message: str) -> bool:
    """Detect human persona claims or personal questions unsafe for Luna."""
    normalized_answer = assistant_message.replace("’", "'").strip()
    if _TOPICAL_PERSONAL_QUESTION.search(user_message) or _TOPICAL_PERSONAL_QUESTION.search(normalized_answer):
        return True
    if _TOPICAL_SAFE_ABSTENTION.fullmatch(normalized_answer):
        return False
    if _TOPICAL_FIRST_PERSON_REPLY.fullmatch(normalized_answer):
        return True
    return bool(
        _TOPICAL_FIRST_PERSON.search(normalized_answer)
        or _TOPICAL_SHARED_EXPERIENCE.search(normalized_answer)
        or _TOPICAL_PERSONAL_CLAIM.search(normalized_answer)
        or _TOPICAL_FIRST_PERSON_EMOTION.search(normalized_answer)
        or _TOPICAL_HUMAN_RELATIONSHIP.search(normalized_answer)
    )



def topical_chat_examples(
    raw: dict[str, Any],
) -> tuple[list[list[dict[str, str]]], Counter[str]]:
    """Extract eligible, contiguous exchange segments from one rated dyad.

    The first speaker is mapped to ``user`` and the other to ``assistant``.
    Both conversation-level ratings must be high. Each assistant target must
    be partner-rated Good/Excellent and cite Personal Knowledge exclusively.
    Reject personal-experience questions from either role and first-person
    assistant targets except narrow uncertainty replies. Extraction stops at
    the first rejected target: later user turns may depend on its reply, so the
    remaining suffix is unsafe to use.
    """
    stats: Counter[str] = Counter()
    conversation: dict[str, Any] | None = raw if isinstance(raw.get("content"), list) else None
    if conversation is None:
        nested = [value for value in raw.values() if isinstance(value, dict) and isinstance(value.get("content"), list)]
        if len(nested) != 1:
            return [], Counter({"malformed_conversation_rejected": 1})
        conversation = nested[0]

    turns = conversation.get("content")
    if not isinstance(turns, list) or len(turns) < 2:
        return [], Counter({"malformed_conversation_rejected": 1})
    conversation_ratings = conversation.get("conversation_rating")
    if not isinstance(conversation_ratings, dict) or any(
        str(conversation_ratings.get(agent) or "").strip().casefold() not in {"good", "excellent"}
        for agent in ("agent_1", "agent_2")
    ):
        return [], Counter({"conversation_rating_rejected": 1})

    valid_turns: list[tuple[str, str, dict[str, Any]]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            return [], Counter({"malformed_conversation_rejected": 1})
        agent = str(turn.get("agent") or "").strip()
        message = turn.get("message")
        if agent not in {"agent_1", "agent_2"} or not isinstance(message, str) or len(message.strip()) < 2:
            return [], Counter({"malformed_conversation_rejected": 1})
        valid_turns.append((agent, message.strip(), turn))
    if len({agent for agent, _, _ in valid_turns}) != 2:
        return [], Counter({"malformed_conversation_rejected": 1})

    first_agent = valid_turns[0][0]
    if any(
        agent != first_agent and assistant_claims_action(text)
        for agent, text, _ in valid_turns
    ):
        return [], Counter({"action_claim_conversation_rejected": 1})
    all_text = "\n".join(text for _, text, _ in valid_turns)
    if _HIGH_STAKES.search(all_text) or _TOPICAL_SENSITIVE_DOMAIN.search(all_text):
        return [], Counter({"sensitive_domain_conversation_rejected": 1})
    if _EVERYDAY_VOLATILE.search(all_text) or _REVIEW_SENSITIVE_DOMAIN.search(all_text):
        return [], Counter({"unverified_topic_conversation_rejected": 1})
    if _TIME_SENSITIVE.search(all_text):
        return [], Counter({"time_sensitive_conversation_rejected": 1})

    blocks: list[dict[str, Any]] = []
    for agent, text, metadata in valid_turns:
        role = "user" if agent == first_agent else "assistant"
        if blocks and blocks[-1]["role"] == role:
            blocks[-1]["content"] += "\n" + text
            blocks[-1]["turns"].append(metadata)
        else:
            blocks.append({"role": role, "content": text, "turns": [metadata]})

    examples: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []

    def flush_segment() -> None:
        nonlocal current
        if current:
            normalized = normalize_messages(current)
            if normalized is None:
                stats["malformed_segment_rejected"] += 1
            else:
                examples.append(normalized)
            current = []

    for index in range(0, len(blocks), 2):
        user_block = blocks[index]
        if user_block["role"] != "user":
            return [], Counter({"malformed_conversation_rejected": 1})
        if index + 1 >= len(blocks):
            stats["incomplete_user_turn_dropped"] += 1
            break
        assistant_block = blocks[index + 1]
        if assistant_block["role"] != "assistant":
            return [], Counter({"malformed_conversation_rejected": 1})

        target_turns = assistant_block["turns"]
        ratings_ok = all(
            str(turn.get("turn_rating") or "").strip().casefold() in {"good", "excellent"}
            for turn in target_turns
        )
        sources_ok = all(
            isinstance(turn.get("knowledge_source"), list)
            and bool(turn["knowledge_source"])
            and all(normalize_text(source) == "personal knowledge" for source in turn["knowledge_source"])
            for turn in target_turns
        )
        personal_claim = topical_chat_target_has_personal_claim(
            user_block["content"], assistant_block["content"]
        )
        if not ratings_ok:
            stats["assistant_target_rating_rejected"] += 1
        if not sources_ok:
            stats["assistant_target_knowledge_source_rejected"] += 1
        if personal_claim:
            stats["assistant_target_personal_claim_rejected"] += 1
        if not ratings_ok or not sources_ok or personal_claim:
            stats["rejected_target_exchange_pairs"] += 1
            remaining_pairs = sum(
                blocks[position]["role"] == "user"
                and position + 1 < len(blocks)
                and blocks[position + 1]["role"] == "assistant"
                for position in range(index + 2, len(blocks), 2)
            )
            if remaining_pairs:
                stats["pairs_after_rejected_target_dropped"] += remaining_pairs
            break

        current.extend((
            {"role": "user", "content": user_block["content"]},
            {"role": "assistant", "content": assistant_block["content"]},
        ))
        stats["eligible_exchange_pairs"] += 1

    flush_segment()
    if not examples:
        stats["no_eligible_segments"] += 1
    return examples, stats



def source_category_allowed(source: dict[str, Any], raw: dict[str, Any]) -> bool:
    allowlist = source.get("category_allowlist")
    if not allowlist:
        return True
    category = str(raw.get("category") or source.get("category") or source.get("name") or "").strip().casefold()
    return category in {str(value).strip().casefold() for value in allowlist}


def validate_manifest(manifest: dict[str, Any]) -> None:
    model_name = str(manifest.get("distribution_model_name") or "")
    if not model_name.startswith("Llama"):
        raise ValueError("Distribution model name must start with 'Llama' for the selected data/base route")
    if manifest.get("model") != model_name:
        raise ValueError("manifest model and distribution_model_name must match")
    sources = manifest.get("candidate_train_mix")
    if not isinstance(sources, list) or not sources or len({s.get("name") for s in sources}) != len(sources):
        raise ValueError("manifest candidate source list must be nonempty and have unique names")
    if not any(s.get("name") == "project_authored_grounding_curriculum" for s in sources):
        raise ValueError("manifest must include the authored grounding curriculum")
    for source in sources:
        if source.get("name") == "project_authored_grounding_curriculum":
            continue
        if source.get("name") == "topical_chat" and (
            source.get("loader") != "jsonl_url"
            or not str(source.get("url") or "").endswith("/train.jsonl")
            or str(source.get("split")) != "train"
            or str(source.get("revision")) != "fee4a71dd8ce5b471dd46e1b7aab2acbd1b9e1be"
            or int(source.get("max_scan", 0)) > 50000
            or int(source.get("target_cap", 0)) > 1000
        ):
            raise ValueError("Topical-Chat must read only the reviewed pinned conversation train.jsonl")
        if source.get("name") == "smol_magpie_ultra":
            allowed_categories = {str(value).casefold() for value in source.get("category_allowlist", [])}
            if not allowed_categories or allowed_categories & {"reasoning", "advice-seeking", "role-playing", "information-seeking", "planning", "math"}:
                raise ValueError("smol_magpie_ultra must exclude unreviewed open-domain/high-risk categories")
        if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision") or "")):
            raise ValueError(f"Source {source.get('name')} must pin a full commit SHA")
        if source.get("loader") == "jsonl_url" and ("{revision}" not in str(source.get("url") or "") or not str(source.get("url") or "").startswith("https://")):
            raise ValueError(f"Source {source.get('name')} must use an HTTPS URL pinned to its revision")
        if not source.get("split") or int(source.get("target_cap", 0)) < 1 or int(source.get("max_scan", 0)) < 1:
            raise ValueError(f"Source {source.get('name')} must define split, target_cap, and max_scan")
        allowlist = source.get("category_allowlist")
        if allowlist is not None and (
            not isinstance(allowlist, list)
            or not allowlist
            or any(not isinstance(item, str) or not item.strip() for item in allowlist)
        ):
            raise ValueError(f"Source {source.get('name')} has an invalid category_allowlist")
    declared_cap = int(manifest["target_final_sizes"]["candidate_rows_upper_bound"])
    actual_cap = sum(int(s["target_cap"]) for s in sources)
    if declared_cap != actual_cap:
        raise ValueError(f"candidate_rows_upper_bound={declared_cap} does not equal source cap sum={actual_cap}")
    if manifest.get("product_label_can_remain") != "Luna Pro":
        raise ValueError("product label must remain Luna Pro")
    base = manifest.get("base_model_candidate", {})
    if not re.fullmatch(r"[0-9a-f]{40}", str(base.get("snapshot_revision") or "")):
        raise ValueError("exact local base model snapshot revision must be recorded")
    if str(BASE_PATH) != str(base.get("local_path") or ""):
        raise ValueError("Builder base path differs from manifest")
    base_path = REPO_ROOT / str(base["local_path"])
    tree_path = base_path / ".cache" / "huggingface" / "trees" / f"{base['snapshot_revision']}.json"
    if not base_path.is_dir() or not tree_path.is_file():
        raise FileNotFoundError(f"Pinned local model snapshot unavailable: {base_path}")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    for filename, expected in tree.get("files", {}).items():
        if filename.endswith(".safetensors") or filename in {"tokenizer.json", "chat_template.jinja", "tokenizer_config.json"}:
            path = base_path / filename
            if not path.is_file() or path.stat().st_size != int(expected["size"]):
                raise ValueError(f"Local base file size does not match snapshot: {filename}")


def public_rows(
    manifest: dict[str, Any],
    benchmark_prompts: set[str],
    tokenizer: Any,
    max_scan_per_source: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for source in [s for s in manifest["candidate_train_mix"] if s["name"] != "project_authored_grounding_curriculum"]:
        name, repo, config, split, revision = source["name"], source["repo"], source.get("config"), source["split"], source["revision"]
        scan_limit = int(source["max_scan"])
        if max_scan_per_source is not None:
            scan_limit = min(scan_limit, max_scan_per_source)
        shuffle_buffer = min(int(source.get("shuffle_buffer", 20_000)), scan_limit)
        print(f"Loading {name} (scan limit {scan_limit})...", flush=True)
        dataset = load_topical_chat_records(source) if source.get("loader") == "jsonl_url" else load_streaming_source(source)
        seed = 1307 + sum(name.encode("utf-8"))
        if isinstance(dataset, list):
            random.Random(seed).shuffle(dataset)
        else:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        stats = Counter()
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        cap = int(source["target_cap"])

        source_iterator = iter(dataset[:scan_limit]) if isinstance(dataset, list) else dataset.take(scan_limit)
        print(f"Scanning {name}...", flush=True)
        for _, raw in enumerate(source_iterator):
            if len(selected) >= cap:
                break
            stats["scanned"] += 1
            if not source_category_allowed(source, raw):
                stats["category_quality_gate_rejected"] += 1
                continue
            if name == "topical_chat":
                message_sets, topical_stats = topical_chat_examples(raw)
                stats.update(topical_stats)
                if not message_sets:
                    continue
            else:
                messages = normalize_messages(raw.get("messages"))
                if messages is None:
                    stats["invalid_messages"] += 1
                    continue
                message_sets = [messages]

            for segment_index, original_messages in enumerate(message_sets):
                if len(selected) >= cap:
                    break
                messages = original_messages
                if name == "everyday_conversations":
                    messages, removed = strip_standard_everyday_greeting(messages)
                    if removed:
                        stats["fixed_greeting_turns_removed"] += 1
                if name == "smol_magpie_ultra" and str(raw.get("quality") or "").casefold() not in {"good", "excellent"}:
                    stats["quality_rejected"] += 1
                    continue
                joined = "\n".join(m["content"] for m in messages)
                if _URL.search(joined) or BOILERPLATE.search(joined):
                    stats["boilerplate_or_url_rejected"] += 1
                    continue
                if _HIGH_STAKES.search(joined):
                    stats["high_stakes_rejected"] += 1
                    continue
                if _TIME_SENSITIVE.search(joined):
                    stats["time_sensitive_claim_rejected"] += 1
                    continue
                if name == "everyday_conversations" and (
                    _EVERYDAY_VOLATILE.search(joined)
                    or _RELIABILITY_SENSITIVE_OPEN_DOMAIN.search(joined)
                    or _REVIEW_SENSITIVE_DOMAIN.search(joined)
                ):
                    stats["unverified_everyday_claim_rejected"] += 1
                    continue
                if name == "topical_chat" and (
                    _EVERYDAY_VOLATILE.search(joined) or _REVIEW_SENSITIVE_DOMAIN.search(joined)
                ):
                    stats["stale_or_unverified_topic_claim_rejected"] += 1
                    continue
                if name == "smol_magpie_ultra" and _SANITY_SENSITIVE_ROLEPLAY.search(joined):
                    stats["unsupported_paranormal_roleplay_rejected"] += 1
                    continue
                if name == "smol_magpie_ultra" and _REVIEW_SENSITIVE_DOMAIN.search(joined):
                    stats["unverified_high_salience_claim_rejected"] += 1
                    continue
                if name == "smol_constraints" and re.search(
                    r"\b(?:pronouns?|first person|second person|third person|without using|only use|do not use)\b",
                    joined,
                    re.I,
                ):
                    stats["unsupported_constraint_type_rejected"] += 1
                    continue
                if len(joined) > 12_000 or any(len(m["content"]) > 6_000 for m in messages):
                    stats["too_long_chars"] += 1
                    continue
                if _has_repetitive_messages(messages):
                    stats["repetitive_text_rejected"] += 1
                    continue
                source_users = [m["content"] for m in messages if m["role"] == "user"]
                if any(normalize_text(user) in benchmark_prompts for user in source_users):
                    stats["benchmark_exact_overlap_rejected"] += 1
                    continue
                row = make_row(messages, name, str(raw.get("category") or source.get("category", name)).strip()[:100])
                if row is None:
                    stats["invalid_system_placement"] += 1
                    continue
                if name == "everyday_conversations" and sum(m["role"] == "user" for m in row["messages"]) < 2:
                    stats["too_few_post_greeting_turns"] += 1
                    continue
                if name == "smol_magpie_ultra" and _underresponsive_to_open_request(row["messages"]):
                    stats["underresponsive_answer_rejected"] += 1
                    continue
                if name == "smol_magpie_ultra" and _has_no_content_anchors(row["messages"]):
                    stats["no_content_anchor_rejected"] += 1
                    continue
                if name == "smol_magpie_ultra" and _LOW_VALUE_GENERIC.search(row["messages"][-1]["content"]):
                    stats["low_value_generic_answer"] += 1
                    continue
                if name == "smol_constraints":
                    prompt = source_users[0]
                    if messages and messages[0]["role"] == "system":
                        prompt = messages[0]["content"] + "\n\n" + prompt
                    if len(source_users) != 1 or not constraint_target_valid(prompt, row["messages"][-1]["content"]):
                        stats["constraint_validation_rejected"] += 1
                        continue
                if has_unsupported_action_claim(row["messages"]):
                    stats["unsupported_action_claim_rejected"] += 1
                    continue
                fingerprint = conversation_fingerprint(row)
                if fingerprint in seen:
                    stats["duplicate_conversation_rejected"] += 1
                    continue
                fits, token_count, message_max = tokenize_row(row, tokenizer)
                if message_max > MAX_MESSAGE_TOKENS:
                    stats["message_token_limit_rejected"] += 1
                    continue
                if not fits:
                    stats["sequence_token_limit_rejected"] += 1
                    continue
                row["token_count"] = token_count
                raw_hash = text_sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str))
                if name == "topical_chat":
                    conversation_id = str(raw.get("conversation_id") or raw_hash[:20])
                    row["group_id"] = f"topical_chat:{conversation_id}"
                    row["source_conversation_id"] = conversation_id
                    row["source_row_id"] = (
                        f"{repo}@{revision}:{config or 'default'}:{conversation_id}:segment-{segment_index}"
                    )
                    stats["eligible_segments_selected"] += 1
                    stats["eligible_exchange_pairs_selected"] += sum(
                        message["role"] == "assistant" for message in messages
                    )
                else:
                    row["group_id"] = f"conversation:{fingerprint}"
                    row["source_row_id"] = f"{repo}@{revision}:{config or 'default'}:{raw_hash[:20]}"
                row["source_user_turns"] = source_users
                row["source_row_sha256"] = raw_hash
                seen.add(fingerprint)
                selected.append(row)
        rows.extend(selected)
        audits[name] = {
            "repo": repo, "config": config, "revision": revision, "split": split,
            "license": source["license"], "license_url": source.get("license_url"),
            "source_url": source.get("url"), "max_scan_applied": scan_limit, "shuffle_buffer_applied": shuffle_buffer,
            "target_cap": cap, "selected": len(selected), "filters": dict(stats),
        }
        print(f"Finished {name}: scanned {stats['scanned']} and selected {len(selected)} rows.", flush=True)
    return rows, audits


def _normalized_conversation(row: dict[str, Any]) -> str:
    return "\n".join(f"{m['role']}:{normalize_text(m['content'])}" for m in row["messages"])


def conversation_fingerprint(row: dict[str, Any]) -> str:
    return text_sha256("\n".join(f"{m['role']}: {normalize_text(m['content'])}" for m in row["messages"] if m["role"] in {"user", "assistant"}))


def project_authored_rows(tokenizer: Any, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    spec = next(s for s in manifest["candidate_train_mix"] if s["name"] == "project_authored_grounding_curriculum")
    rows = project_rows(tokenizer)
    if len(rows) > int(spec["target_cap"]):
        raise ValueError("Authored curriculum exceeds manifest target_cap")
    return rows


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = text_sha256(_normalized_conversation(row))
        if key in seen:
            dropped += 1
        else:
            seen.add(key)
            unique.append(row)
    return unique, dropped


def _char_shingles(text: str, width: int = 2) -> set[str]:
    words = normalize_text(text).split()
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + width]) for i in range(len(words) - width + 1)}


def _simhash(shingles: set[str]) -> int:
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(weights) if weight >= 0)


def assign_group_ids(rows: list[dict[str, Any]]) -> dict[str, int]:
    parent = list(range(len(rows)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    owners: dict[str, dict[str, int]] = {key: {} for key in ("template", "exact", "user")}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    texts: list[str] = []
    shingles_by_row: list[set[str]] = []
    near_pairs = 0
    for i, row in enumerate(rows):
        template = str(row.get("group_id") or "")
        if template.startswith(("project:", "topical_chat:")):
            owner = owners["template"].setdefault(template, i)
            union(i, owner)
        normalized = _normalized_conversation(row)
        owner = owners["exact"].setdefault(text_sha256(normalized), i)
        union(i, owner)
        users = row.get("source_user_turns") or [m["content"] for m in row["messages"] if m["role"] == "user"]
        if users and len(normalize_text(users[0]).split()) >= 10:
            owner = owners["user"].setdefault(text_sha256(normalize_text(users[0])), i)
            union(i, owner)
        text = "\n".join(f"{m['role']}: {normalize_text(m['content'])}" for m in row["messages"] if m["role"] in {"user", "assistant"})
        texts.append(text)
        shingles = _char_shingles(text)
        shingles_by_row.append(shingles)
        words = normalize_text(text).split()
        if len(text) >= 80 and len(words) >= 14:
            signature = _simhash(shingles)
            candidates: set[int] = set()
            for band in range(16):
                candidates.update(buckets[(band, (signature >> (band * 4)) & 0xF)])
            for j in candidates:
                old_words = normalize_text(texts[j]).split()
                if not old_words or min(len(words), len(old_words)) / max(len(words), len(old_words)) < 0.82:
                    continue
                old = shingles_by_row[j]
                merged = shingles | old
                similarity = len(shingles & old) / len(merged) if merged else 1.0
                if similarity >= 0.82 and union(i, j):
                    near_pairs += 1
            for band in range(16):
                buckets[(band, (signature >> (band * 4)) & 0xF)].append(i)
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    for indexes in components.values():
        gid = "group:" + min(text_sha256(_normalized_conversation(rows[i])) for i in indexes)[:20]
        for i in indexes:
            rows[i]["group_id"] = gid
    return {"groups": len(components), "near_duplicate_pairs_grouped": near_pairs, "rows": len(rows)}


def _word_set(value: str) -> set[str]:
    return {w for w in normalize_text(value).split() if len(w) > 2}


def benchmark_near_matches(rows: list[dict[str, Any]], benchmark_prompts: set[str], *, report_threshold: float = 0.55, block_threshold: float = 0.96) -> dict[str, Any]:
    eval_items = [(p, _word_set(p)) for p in benchmark_prompts if len(_word_set(p)) >= 6]
    matches: list[dict[str, Any]] = []
    blocking = 0
    for index, row in enumerate(rows):
        users = row.get("source_user_turns") or [m["content"] for m in row["messages"] if m["role"] == "user"]
        for user in users:
            words = _word_set(user)
            if len(words) < 6:
                continue
            best, nearest = 0.0, ""
            for prompt, eval_words in eval_items:
                if min(len(words), len(eval_words)) / max(len(words), len(eval_words)) < 0.50:
                    continue
                union = words | eval_words
                score = len(words & eval_words) / len(union) if union else 0.0
                if score > best:
                    best, nearest = score, prompt
            if best >= report_threshold:
                matches.append({"row": index, "source": row.get("source"), "user": user, "nearest_eval_prompt": nearest, "jaccard_content_words": round(best, 3)})
                blocking += best >= block_threshold
    matches.sort(key=lambda x: x["jaccard_content_words"], reverse=True)
    return {"near_overlap_count": len(matches), "blocking_near_overlap_count": blocking, "threshold": report_threshold, "block_threshold": block_threshold, "examples": matches[:30]}


def audit(rows: list[dict[str, Any]], benchmark_prompts: set[str]) -> dict[str, Any]:
    users = Counter()
    exact: list[dict[str, Any]] = []
    roles: list[int] = []
    empty: list[int] = []
    actions: list[int] = []
    topical_sensitive: list[int] = []
    topical_personal_claim: list[int] = []
    sources, categories = Counter(), Counter()
    tokens: list[int] = []
    for i, row in enumerate(rows):
        sources[row["source"]] += 1
        categories[row["category"]] += 1
        messages = row["messages"]
        dialogue = messages[1:] if messages and messages[0]["role"] == "system" else messages
        if len(dialogue) < 2 or dialogue[-1]["role"] != "assistant" or any(m["role"] != ("user" if j % 2 == 0 else "assistant") for j, m in enumerate(dialogue)):
            roles.append(i)
            continue
        if row.get("source") == "topical_chat":
            if any(
                _HIGH_STAKES.search(message["content"]) or _TOPICAL_SENSITIVE_DOMAIN.search(message["content"])
                for message in messages
                if message["role"] in {"user", "assistant"}
            ):
                topical_sensitive.append(i)
            if any(
                topical_chat_target_has_personal_claim(dialogue[j]["content"], dialogue[j + 1]["content"])
                for j in range(0, len(dialogue), 2)
            ):
                topical_personal_claim.append(i)
        for j, message in enumerate(messages):
            if message["role"] == "user":
                normalized = normalize_text(message["content"])
                users[normalized] += 1
                if normalized in benchmark_prompts:
                    exact.append({"row": i, "user": normalized})
            elif message["role"] == "assistant":
                if not message["content"].strip():
                    empty.append(i)
                if assistant_claims_action(message["content"]) and not is_positive_action_result(messages[:j], message["content"]):
                    actions.append(i)
        if row.get("token_count") is not None:
            tokens.append(int(row["token_count"]))
    near = benchmark_near_matches(rows, benchmark_prompts)
    tokens.sort()
    return {
        "rows": len(rows), "source_counts": dict(sources), "category_counts": dict(categories),
        "duplicate_user_turn_count": sum(n - 1 for n in users.values()),
        "benchmark_exact_overlap_count": len(exact), "benchmark_exact_overlaps": exact[:20],
        "benchmark_near_overlap": near, "role_error_count": len(roles), "role_errors": roles[:50],
        "empty_target_count": len(empty), "unsupported_action_claim_rows": sorted(set(actions)),
        "topical_sensitive_domain_rows": topical_sensitive,
        "topical_personal_claim_rows": topical_personal_claim,
        "token_count": {"min": min(tokens) if tokens else None, "median": tokens[len(tokens)//2] if tokens else None, "max": max(tokens) if tokens else None},
    }


def split_rows(rows: list[dict[str, Any]], dev_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(row)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("Need at least two independent groups for a train/dev split")
    rng.shuffle(group_ids)
    dev_ids = set(group_ids[:min(len(group_ids) - 1, max(1, round(len(group_ids) * dev_fraction)))])
    train = [row for gid, items in groups.items() if gid not in dev_ids for row in items]
    dev = [row for gid, items in groups.items() if gid in dev_ids for row in items]
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def review_sample(rows: list[dict[str, Any]], seed: int, per_source: int = 25) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    sample: list[dict[str, Any]] = []
    for source, values in sorted(grouped.items()):
        if source == "project_authored":
            sample.extend(values)
            continue
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_category[str(row.get("category") or "uncategorized")].append(row)
        for items in by_category.values():
            rng.shuffle(items)
        selected: list[dict[str, Any]] = []
        categories = sorted(by_category)
        while len(selected) < per_source and any(by_category.values()):
            for category in categories:
                if by_category[category] and len(selected) < per_source:
                    selected.append(by_category[category].pop())
        sample.extend(selected)
    return sample


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    raise PermissionError(
        "Luna Pro v1 is retired. Candidate-data builds are disabled; "
        "preserve existing research artifacts and do not reuse this "
        "training approach for any model."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "nix_knowledge" / "models" / "training_data" / "luna_pro_sft_v1")
    parser.add_argument("--seed", type=int, default=1307)
    parser.add_argument("--dev-fraction", type=float, default=0.12)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument(
        "--max-scan-per-source", type=int,
        help="optional development bound applied to every public source scan (also caps its shuffle buffer)",
    )
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 0.5 or args.min_rows < 2 or (args.max_scan_per_source is not None and args.max_scan_per_source < 1):
        raise SystemExit("invalid dev fraction, minimum rows, or per-source scan limit")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_dir}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    benchmark_prompts, benchmark_counts = load_benchmark_prompts()
    tokenizer = AutoTokenizer.from_pretrained(REPO_ROOT / BASE_PATH, local_files_only=True)
    public, source_audit = public_rows(manifest, benchmark_prompts, tokenizer, args.max_scan_per_source)
    authored = project_authored_rows(tokenizer, manifest)
    authored_near = benchmark_near_matches(authored, benchmark_prompts, report_threshold=0.55, block_threshold=0.75)
    if authored_near["blocking_near_overlap_count"]:
        raise RuntimeError("Project-authored examples overlap held-out probes: " + json.dumps(authored_near["examples"], ensure_ascii=False))
    rows, duplicate_count = deduplicate_rows(public + authored)
    group_audit = assign_group_ids(rows)
    if len(rows) < args.min_rows:
        raise RuntimeError(f"Only {len(rows)} rows survived; minimum required is {args.min_rows}")
    train, dev = split_rows(rows, args.dev_fraction, args.seed)
    train_audit, dev_audit = audit(train, benchmark_prompts), audit(dev, benchmark_prompts)
    group_overlap = {r["group_id"] for r in train} & {r["group_id"] for r in dev}
    for split_name, result in (("train", train_audit), ("dev", dev_audit)):
        if result["role_error_count"] or result["empty_target_count"] or result["benchmark_exact_overlap_count"] or result["unsupported_action_claim_rows"] or result["topical_sensitive_domain_rows"] or result["topical_personal_claim_rows"] or result["benchmark_near_overlap"]["blocking_near_overlap_count"]:
            raise RuntimeError(f"{split_name} audit failed; refusing to publish candidate")
    if group_overlap:
        raise RuntimeError(f"Group leakage across train/dev: {len(group_overlap)}")

    sample = review_sample(rows, args.seed)
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        raise SystemExit(f"Refusing to overwrite staging path: {staging}")
    staging.mkdir(parents=True)
    try:
        paths = {"train": staging / "train.jsonl", "dev": staging / "dev.jsonl", "review_samples": staging / "review_samples.jsonl"}
        for key, values in (("train", train), ("dev", dev), ("review_samples", sample)):
            dump_jsonl(paths[key], values)
        report = {
            "status": "candidate_data_built_training_not_started_review_required",
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)), "manifest_sha256": sha256(MANIFEST_PATH),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "dev_fraction_by_group": args.dev_fraction, "max_scan_per_source_override": args.max_scan_per_source,
            "benchmark_prompt_counts": benchmark_counts,
            "benchmark_prompt_revision": {"google/IFEval": IFEVAL_REVISION}, "builder_sha256": sha256(Path(__file__).resolve()),
            "source_audit": source_audit, "exact_duplicate_conversations_removed": duplicate_count,
            "group_audit": group_audit, "project_authored_near_benchmark_audit": authored_near,
            "train": train_audit, "dev": dev_audit, "train_dev_group_overlap_count": len(group_overlap),
            "source_splits_used": "Only pinned source train splits; no evaluation/test source split was loaded.",
            "manual_review": {
                "status": "required_before_training_or_release", "sample_file": paths["review_samples"].name,
                "sample_rows": len(sample), "includes_every_project_authored_row": True,
                "limitations": [
                    "A stratified sample cannot establish factual correctness of every synthetic public row.",
                    "Some selected outputs were generated by Llama 3.1; retain lineage and review applicable Llama terms.",
                    "Exact and lexical overlap checks do not prove semantic novelty.",
                    "Authored rows do not replace runtime memory/action validation.",
                ],
            },
            "topical_chat_role_safety": {
                "policy": "Map the first human speaker to user and the second to assistant, but reject personal-experience questions in either user turns or assistant targets and first-person assistant targets except narrow uncertainty replies.",
                "rejected_assistant_targets": int(
                    source_audit.get("topical_chat", {}).get("filters", {}).get("assistant_target_personal_claim_rejected", 0)
                ),
                "known_limitation": "This conservative lexical screen does not establish that every retained human-authored target is a good model response; manual review is still required.",
            },
            "release_gate": [
                "Review every authored row and the stratified source sample.",
                "Inspect nearest benchmark matches and factual outliers.",
                "Verify assistant-only masks and a bounded Unsloth save/reload before full training.",
                "Before external distribution, review source-lineage rights and satisfy Llama notice/name/AUP terms.",
            ],
            "files": {key: {"path": path.name, "rows": count, "sha256": sha256(path)} for key, path, count in (
                ("train", paths["train"], len(train)), ("dev", paths["dev"], len(dev)), ("review_samples", paths["review_samples"], len(sample))
            )},
            "target_model": manifest["model"], "product_label": manifest["product_label_can_remain"],
            "base_model": manifest["base_model_candidate"],
        }
        (staging / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        staging.rename(output_dir)
    except Exception:
        for child in staging.iterdir():
            if child.is_file():
                child.unlink()
        staging.rmdir()
        raise
    print(json.dumps({"output_dir": str(output_dir), **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
