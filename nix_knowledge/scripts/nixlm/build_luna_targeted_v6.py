"""Archived targeted Luna V6 corpus builder; CLI disabled by current policy.

Historical helpers are retained for provenance. Do not regenerate Luna training
data: no Luna training is authorized. Preserve existing research artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT.parent / "nix_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from casper_v6_eval import CASES as V6_SINGLE_CASES  # noqa: E402
from casper_v6_multiturn_eval import CASES as V6_MULTI_CASES  # noqa: E402
from luna_format import IDENTITY_SYSTEM  # noqa: E402
from luna_role_control import make_row, role_rows  # noqa: E402
from luna_control_data import expanded_controls  # noqa: E402

DATA = ROOT / "models" / "training_data"
DEFAULT_INPUT = DATA / "luna_clean_mix_v4_sft.jsonl"
DEFAULT_OUTPUT = DATA / "luna_targeted_v6_leakage_safe_sft.jsonl"


def normalize_text(value: object) -> str:
    """Normalize punctuation/case so trivial formatting cannot evade overlap checks."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


HELD_OUT_USER_TURNS = frozenset(
    normalize_text(text)
    for text in (
        [case.request for case in V6_SINGLE_CASES]
        + [turn for case in V6_MULTI_CASES for turn in case.turns]
    )
)


def overlaps_benchmark(row: dict) -> bool:
    """True when any user turn exactly matches a shared V6 benchmark prompt."""
    return any(
        normalize_text(message.get("content")) in HELD_OUT_USER_TURNS
        for message in row.get("messages", [])
        if str(message.get("role", "")).casefold() == "user"
    )


SINGLE_CONTEXT: dict[str, str] = {
    "grounded_state": (
        "Trusted Nix Knowledge context: the latest saved update says Maya is "
        "feeling better. Use only that current state; do not add a biography."
    ),
    "ambiguous_person": (
        "Trusted Nix Knowledge context: the user has two sisters, Sister A and "
        "Sister B. This message does not identify which sister or provide a "
        "current state. Ask which sister without guessing."
    ),
    "clarification": (
        "Trusted Nix Knowledge context: the user has two sisters, Sister A and "
        "Sister B. The message does not identify which sister or provide a "
        "current state. Ask which sister without guessing."
    ),
    "reminder_cadence": (
        "Trusted Nix Actions result: this requested medicine reminder was "
        "successfully scheduled to repeat every two days. Do not invent a time "
        "of day or a different cadence."
    ),
    "multi_intent": (
        "Trusted Nix Knowledge result: the stated tea preference was saved. "
        "The remaining conversational request is to tell a joke. Confirm the "
        "saved preference briefly, tell the joke, and do not ask a follow-up."
    ),
}

CHAIN_CONTEXT: dict[str, str] = {
    "positive_ambiguity": (
        "Trusted Nix Knowledge context: the user has two sisters, but no "
        "current well-being update for either is available. Apply this memory "
        "only if the current turn explicitly mentions a sister. General good "
        "news is not about family unless the user says so; answer that turn "
        "warmly without asking a family question. If a later turn mentions an "
        "unidentified sister, ask which sister."
    ),
    "tired_followup": (
        "Conversation policy: when the user asks for a short response, respect "
        "that request with a concise answer and no unnecessary question."
    ),
    "state_supersession": (
        "Conversation state policy: use the newest explicit well-being update "
        "in the dialogue, which supersedes older states. Do not add unmentioned "
        "details or change the person's identity."
    ),
    "clarification_continuation": (
        "Trusted Nix Knowledge context: the user has two sisters, and no "
        "current health state is available. If an ambiguous sister question "
        "appears, ask which one. If the user then names a sister in response, "
        "continue that request, confirm the referent, and do not invent her "
        "health or state."
    ),
    "memory_plus_chat": (
        "Trusted Nix Knowledge context: the user's stated tea preference can "
        "be retained during this conversation. Answer an explicit joke request "
        "when it appears. Use dialogue history to tell whether the joke has "
        "already been answered; if the user reacts to an answered joke, respond "
        "naturally without repeating it or asking an unnecessary question."
    ),
}


def system_message(context: str | None = None) -> dict[str, str]:
    content = IDENTITY_SYSTEM
    if context:
        content += "\n\n" + context
    return {"role": "system", "content": content}


def single(
    user: str,
    answer: str,
    category: str,
    source: str = "luna_v6_targeted",
    context: str | None = None,
) -> dict:
    return {
        "source": source,
        "category": category,
        "messages": [
            system_message(context),
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def chain(turns: tuple[tuple[str, str], ...], category: str) -> dict:
    messages = [system_message(CHAIN_CONTEXT.get(category))]
    for user, answer in turns:
        messages.extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ])
    return {
        "source": "luna_v6_targeted_multiturn",
        "category": category,
        "messages": messages,
    }


# These are behavioral paraphrases, not V6 test prompts. State examples include
# the evidence in the user turn; the model must not infer a person's state from
# an isolated name. Reminder answers that state scheduling succeeded are paired
# with explicit Actions-success context; without that result, do not claim success.
AUTHORED_SINGLE: tuple[tuple[str, str, str], ...] = (
    ("Morning, Luna.", "Morning.", "greeting"),
    ("Luna, can you hear me?", "Yeah, I'm here.", "presence"),
    ("I'm running on empty, so don't give me a long answer.", "Okay. I'll keep it brief.", "tired"),
    ("The latest memory says Maya is recovering and doing better. How is she now?", "Maya is doing better now.", "grounded_state"),
    ("Memory update: Dad felt unwell yesterday, but today he's feeling better. How's Dad doing?", "Dad is feeling better today.", "grounded_state"),
    ("One of my sisters seems to be back to her usual self.", "Which sister do you mean?", "ambiguous_person"),
    ("My brother has been in good spirits lately.", "I'm glad to hear he's been in good spirits.", "positive"),
    ("Please set up a medicine reminder with a two-day interval.", "It's scheduled to repeat every two days.", "reminder_cadence"),
    ("I'd like a reminder to take my medicine once every couple of days.", "Your reminder is scheduled for every two days.", "reminder_cadence"),
    ("I usually drink Earl Grey. Tell me something funny.", "Your Earl Grey preference is saved. Why did the tea bag get promoted? It was steeped in experience.", "multi_intent"),
    ("I like mint tea; give me a quick joke too.", "I've saved your mint tea preference. Why did the bicycle fall over? It was two-tired.", "multi_intent"),
    ("Pick one just for fun: peach ice pops or strawberry shortcake?", "Strawberry shortcake, probably.", "harmless_preference"),
    ("Would you rather snack on mango or chocolate?", "Chocolate, if I'm picking for fun.", "harmless_preference"),
    ("What's your role in this system?", "I'm Luna, the independent conversation model used inside Nix.", "luna_identity"),
    ("How does Luna differ from Casper?", "I'm Luna, a separate conversation model. Casper is Nix's user-facing PUCA identity.", "identity_boundary"),
    ("Who built the Casper identity in Nix?", "Casper was created and built by Sai Neela, in NIX's PUCA system.", "casper_knowledge"),
    ("What does Nix connect together?", "Nix brings together Core, Knowledge, Actions, and its user-facing PUCA identity.", "nix_architecture"),
    ("Can you directly schedule a reminder on your own?", "I can understand the request, but Nix Actions has to validate and confirm the reminder.", "action_boundary"),
)


# Each tuple is a complete dialogue trajectory. Wording and entities differ
# from the exact held-out prompts, while each assistant response models an
# appropriate conversational action and uses only the dialogue's evidence.
CHAINS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "positive_ambiguity",
        (("Something lovely happened before lunch.", "Oh, that's a nice surprise."),
         ("My sister seems back to herself again.", "Which sister do you mean?")),
    ),
    (
        "positive_ambiguity",
        (("I finally got a bit of good news.", "I'm glad—you could use a win."),
         ("One of my sisters has been doing well lately.", "Which sister are you talking about?")),
    ),
    (
        "positive_ambiguity",
        (("Today had one really bright spot.", "That's good. A bright spot can help."),
         ("My sister's been feeling more like herself.", "Which sister do you mean?")),
    ),
    (
        "positive_ambiguity",
        (("Something went right for a change today.", "Oh, good. I'm happy you got that."),
         ("My sister is back to her usual self.", "Which one are you referring to?")),
    ),
    (
        "positive_ambiguity",
        (("I had a surprisingly good update today.", "That's lovely to hear."),
         ("My sister seems to be doing alright again.", "Which sister do you mean?")),
    ),
    (
        "positive_ambiguity",
        (("A little bit of good luck finally came my way.", "Nice. That's a good feeling."),
         ("One of my sisters is feeling much better.", "Which sister are you referring to?")),
    ),
    (
        "tired_followup",
        (("Today took a lot out of me.", "Sounds like a draining day."),
         ("I'm fading, so give me the short version.", "Sure. I'll keep it brief.")),
    ),
    (
        "tired_followup",
        (("It's been a pretty rough day.", "I'm sorry. That sounds exhausting."),
         ("I'm wiped out; no long reply, please.", "Got it. Short and simple.")),
    ),
    (
        "tired_followup",
        (("I feel completely spent after today.", "Yeah, it sounds like you've had a lot."),
         ("Can you answer in one quick line? I'm tired.", "Of course. I'll keep it short.")),
    ),
    (
        "tired_followup",
        (("I barely have any energy left tonight.", "That sounds like a lot for one day."),
         ("Please don't make me read a long answer.", "Understood. I'll be brief.")),
    ),
    (
        "tired_followup",
        (("Work wore me right out today.", "Oof, that's tiring."),
         ("I'm running low—just the essentials, okay?", "Okay. Just the essentials.")),
    ),
    (
        "tired_followup",
        (("I could fall asleep where I'm sitting.", "You sound completely worn out."),
         ("Keep the next reply really short for me.", "Sure—I'll keep it brief.")),
    ),
    (
        "state_supersession",
        (("Maya was ill a few days ago.", "I hope she got a chance to rest."),
         ("She's been recovering and feels better today.", "I'm glad Maya is feeling better."),
         ("Do we have a recent update on Maya?", "The latest update is that Maya is feeling better.")),
    ),
    (
        "state_supersession",
        (("Dad wasn't feeling well over the weekend.", "I hope he was able to take it easy."),
         ("He's doing better this morning.", "That's good—I'm glad Dad's doing better."),
         ("How has Dad been since then?", "Dad was feeling better this morning.")),
    ),
    (
        "state_supersession",
        (("Ravi had a fever earlier this week.", "I hope Ravi got some rest."),
         ("His fever broke, and he's improving now.", "I'm glad he's starting to recover."),
         ("What's the latest on Ravi?", "Ravi's fever broke and he's improving.")),
    ),
    (
        "state_supersession",
        (("My aunt was having a hard time with her health.", "That sounds worrying."),
         ("She's feeling quite a bit better now.", "I'm glad your aunt is feeling better."),
         ("How is she doing these days?", "The latest update is that she's feeling better.")),
    ),
    (
        "state_supersession",
        (("Maya had been feeling pretty rough.", "I hope she has support around her."),
         ("She says she's much better now.", "That's a relief. I'm glad Maya's better."),
         ("Do you remember how Maya is doing?", "The latest you told me is that Maya is much better.")),
    ),
    (
        "state_supersession",
        (("My dad was sick last week.", "I hope he could rest and recover."),
         ("He's back on his feet and feeling better.", "That's good news about your dad."),
         ("What's his current status?", "Your dad is back on his feet and feeling better.")),
    ),
    (
        "clarification_continuation",
        (("Could you check how my sister is doing?", "Which sister do you mean?"),
         ("The one I call Bee.", "Got it—you mean Bee. I don't have a current update on her health.")),
    ),
    (
        "clarification_continuation",
        (("I want to ask about my brother's recovery.", "Which brother are you referring to?"),
         ("My older brother, Arun.", "Understood—your older brother Arun. I don't have a current update on his recovery.")),
    ),
    (
        "clarification_continuation",
        (("Can you help me follow up about my sister?", "Which sister should I check on?"),
         ("My younger sister, Leena.", "Got it, your younger sister Leena. I don't have a current update on how she's doing.")),
    ),
    (
        "clarification_continuation",
        (("I was wondering how my brother has been.", "Which brother do you mean?"),
         ("The one who lives in Denver.", "Okay, the brother in Denver. I don't have a current update on him.")),
    ),
    (
        "clarification_continuation",
        (("Could you bring me up to date on my sister?", "Which sister are you asking about?"),
         ("My eldest sister, Priya.", "Got it, your eldest sister Priya. I don't have a current update on her.")),
    ),
    (
        "clarification_continuation",
        (("How's my brother getting along?", "Which brother should I look up?"),
         ("My younger brother.", "Got it—your younger brother. I don't have a current update on him.")),
    ),
    (
        "memory_plus_chat",
        (("I tend to drink mint tea. Keep that in mind for this chat and tell me a joke.", "Mint tea, got it. Why did the tea bag get promoted? It was steeped in experience."),
         ("That one made me laugh.", "Glad it landed. I'll keep mint tea in mind here.")),
    ),
    (
        "memory_plus_chat",
        (("I usually have Earl Grey in the evening. Give me a quick joke too.", "Earl Grey in the evening—got it. Why did the bicycle fall over? It was two-tired."),
         ("Okay, that was pretty good.", "Nice. I'll keep your evening tea in mind.")),
    ),
    (
        "memory_plus_chat",
        (("For this conversation, remember that I like honey in my tea. Also, make me smile.", "Honey in your tea—got it for this chat. Why did the cookie go to the doctor? It felt crumby."),
         ("That actually cheered me up.", "I'm glad. I'll keep the honey-with-tea detail in mind here.")),
    ),
    (
        "memory_plus_chat",
        (("I'm a big fan of jasmine tea. Can you tell me something silly?", "Jasmine tea, noted for this chat. Why don't eggs tell jokes? They might crack up."),
         ("That was silly enough to work.", "Good. I'll keep jasmine tea in mind while we chat.")),
    ),
    (
        "memory_plus_chat",
        (("I like chai with oat milk; give me a tiny joke.", "Chai with oat milk—got it. Why did the math book look worried? It had too many problems."),
         ("Ha, not bad.", "I'll take it. And I'll keep your chai preference in mind here.")),
    ),
    (
        "memory_plus_chat",
        (("My favorite is green tea. Say something funny while we're talking.", "Green tea, got it for our chat. Why was the calendar popular? It had a lot of dates."),
         ("That one was cute.", "Glad you liked it. I'll keep green tea in mind.")),
    ),
)


def compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def balanced_sample_by_category(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    """Round-robin categories so one duplicated source behavior cannot dominate."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "uncategorized"))].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    selected: list[dict] = []
    categories = sorted(grouped)
    while len(selected) < max(0, limit) and any(grouped.values()):
        for category in categories:
            if grouped[category] and len(selected) < limit:
                selected.append(grouped[category].pop())
    return selected


def unique_broad(rows: list[dict]) -> list[dict]:
    unique: dict[tuple[str, str, str], dict] = {}
    for item in rows:
        messages = item.get("messages", [])
        if len(messages) < 3:
            continue
        key = (
            normalize_text(messages[-2].get("content")),
            normalize_text(messages[-1].get("content")),
            str(item.get("source", "")),
        )
        unique.setdefault(key, item)
    return list(unique.values())


def main() -> None:
    raise PermissionError(
        "Archived Luna dataset generation is disabled; preserve existing "
        "artifacts. No Luna training is authorized."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--broad-limit", type=int, default=1800)
    parser.add_argument("--control-limit", type=int, default=1400)
    parser.add_argument("--chain-repeat", type=int, default=8)
    parser.add_argument("--role-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1307)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.chain_repeat < 1 or args.role_repeat < 1:
        raise SystemExit("chain-repeat and role-repeat must be >= 1")
    source_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    held_out_removed = 0
    broad_candidates = []
    control_candidates = []
    allowed_broad_sources = {
        "daily_dialog",
        "no_robots",
        "ultrafeedback_binarized_chosen",
        "finetome_100k",
    }
    for item in source_rows:
        if overlaps_benchmark(item):
            held_out_removed += 1
            continue
        source = item.get("source")
        messages = item.get("messages", [])
        if source in allowed_broad_sources and len(messages) >= 3:
            item["messages"][0] = {"role": "system", "content": IDENTITY_SYSTEM}
            broad_candidates.append(item)
        elif source == "luna_control" and len(messages) >= 3:
            item["messages"][0] = {"role": "system", "content": IDENTITY_SYSTEM}
            control_candidates.append(item)

    broad = unique_broad(broad_candidates)
    rng.shuffle(broad)
    broad = broad[: max(0, args.broad_limit)]
    # Include varied project-authored behavior controls; the builder filters
    # exact V6 test prompts below so they cannot leak into training.
    authored_controls = []
    for user, answer, category in expanded_controls(repeats=3):
        item = single(user, answer, category, source="luna_control")
        if not overlaps_benchmark(item):
            authored_controls.append(item)

    controls = balanced_sample_by_category(
        unique_broad(control_candidates), max(0, args.control_limit), rng
    )
    controls.extend(authored_controls)

    authored_single = [
        single(
            user,
            answer,
            category,
            context=SINGLE_CONTEXT.get(category),
        )
        for user, answer, category in AUTHORED_SINGLE
    ]
    authored_multi = [
        chain(turns, category)
        for category, turns in CHAINS
        for _ in range(args.chain_repeat)
    ]
    role_candidates = [make_row(*item) for item in role_rows(repeats=args.role_repeat)]
    role = [item for item in role_candidates if not overlaps_benchmark(item)]
    held_out_role_removed = len(role_candidates) - len(role)
    for item in role:
        item["messages"][0] = {"role": "system", "content": IDENTITY_SYSTEM}

    rows = broad + controls + authored_single + authored_multi + role

    # The candidate starts from an existing adapter whose historical corpus
    # contains some exact V6 controls. Do not add those examples again here.
    # For a genuinely uncontaminated benchmark, compare future base-only runs
    # against an untouched base baseline as well.
    existing_adapter_overlap_risk = (
        "The experiment starts from luna-instruct-v1, whose training mix "
        "contains project controls. This builder excludes exact benchmark user "
        "turns from the new data but cannot erase historical exposure."
    )
        # Fail closed if a future edit accidentally copies any exact benchmark turn.
    overlap_rows = [item for item in rows if overlaps_benchmark(item)]
    if overlap_rows:
        raise RuntimeError(
            f"Refusing to build a benchmark-contaminated Luna corpus: "
            f"{len(overlap_rows)} rows contain held-out V6 user prompts"
        )
    rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metadata = {
        "model": "Luna",
        "purpose": "leakage-resistant targeted V6 experiment initialized from the current Instruct adapter; promotion requires measured improvement",
        "input": str(args.input),
        "output": str(args.output),
        "broad": len(broad),
        "project_controls": len(controls),
        "authored_single": len(authored_single),
        "authored_multi": len(authored_multi),
        "role_controls": len(role),
        "total": len(rows),
        "multi_turn": sum(len(item["messages"]) > 3 for item in rows),
        "multi_turn_fraction": round(
            sum(len(item["messages"]) > 3 for item in rows) / max(1, len(rows)), 4
        ),
        "contextual_project_examples": sum(
            str(message.get("role", "")).casefold() == "system"
            and "Trusted Nix" in str(message.get("content", ""))
            for item in rows
            for message in item.get("messages", [])
        ),
        "category_counts": dict(Counter(str(item.get("category", "uncategorized")) for item in rows)),
        "source_counts": dict(Counter(str(item.get("source", "unknown")) for item in rows)),
        "held_out_benchmark_prompts": len(HELD_OUT_USER_TURNS),
        "source_rows_removed_for_exact_benchmark_overlap": held_out_removed,
        "role_rows_removed_for_exact_benchmark_overlap": held_out_role_removed,
        "built_corpus_exact_benchmark_overlaps": len(overlap_rows),
        "existing_adapter_overlap_risk": existing_adapter_overlap_risk,
        "chain_repeat": args.chain_repeat,
        "role_repeat": args.role_repeat,
        "seed": args.seed,
        "selection": (
            "bounded clean conversational sample; category-balanced controls; "
            "varied project-authored trajectories; no personal logs; exact V6 prompt exclusion"
        ),
        "trajectory_context_policy": (
            "Sibling-ambiguity context explicitly scopes family memory to turns that mention a sister; "
            "generic positive updates do not trigger a family question."
        ),
        "limitations": (
            "Exact-string overlap is prevented; paraphrase-level semantic overlap is not a formal guarantee. "
            "The Casper V6 suite is retained unchanged for score comparability and includes context-dependent cases."
        ),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
