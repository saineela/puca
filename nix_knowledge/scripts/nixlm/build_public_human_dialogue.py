"""Build Casper's PUCA style corpus from public human-dialogue datasets.

This is style transfer, not knowledge training. Public text is retained only
when it looks like an ordinary human turn and the response does not teach
assistant/corporate habits, invented personal history, or unsafe intimacy.
The source and license metadata are written beside the generated JSONL.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "training_data"
SYSTEM = (
    "You are Casper, a PUCA (Personal User Companion Agent), warm and "
    "grounded. Speak like a "
    "familiar person: natural, concise, and responsive. Do not sound like "
    "customer support. Do not invent memories, personal experiences, or facts. "
    "Do not force a question or offer help after every message."
)

# These filters intentionally bias toward clean, ordinary conversation. They
# are not a safety classifier and do not replace Core/Knowledge policy.
REJECT = re.compile(
    r"(?:as an ai|language model|how can i help|how may i help|i'm here to help|"
    r"what can i do for you|certainly|absolutely|please let me know|"
    r"customer service|terms of service|www\.|https?://|\[url\]|<[^>]+>)",
    re.I,
)
GENERIC_INTERVIEW = re.compile(
    r"(?:what(?:'s| is) on your mind|how can i help|what can i do for you|"
    r"anything else|how about we chat about something else|let me know if you need anything)",
    re.I,
)
PRIVATE_OR_FACT_HEAVY = re.compile(
    r"(?:password|passcode|credit card|social security|api key|secret|address is|"
    r"phone number|email is|my full name|born on|diagnos(?:is|ed)|suicid|kill myself)",
    re.I,
)
# Public datasets sometimes contain screenplay-like turns or malformed rows.
BAD_SHAPE = re.compile(r"(?:^.{0,1}$|\b(?:http|www)\b|\.{4,}|\?{3,})", re.I)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def acceptable(user: str, answer: str) -> bool:
    if not user or not answer or len(user) > 500 or len(answer) > 500:
        return False
    if BAD_SHAPE.search(user) or BAD_SHAPE.search(answer):
        return False
    if REJECT.search(user) or REJECT.search(answer):
        return False
    # Do not import the exact conversational tic we are trying to remove.
    if GENERIC_INTERVIEW.search(answer):
        return False
    if PRIVATE_OR_FACT_HEAVY.search(user) or PRIVATE_OR_FACT_HEAVY.search(answer):
        return False
    # Avoid training the assistant to echo a question endlessly or produce
    # list/article prose instead of a conversational turn.
    if answer.count("\n") > 1 or answer.count("; ") > 3:
        return False
    return True


def daily_pairs_from_hf() -> Iterable[tuple[str, str, str]]:
    from huggingface_hub import hf_hub_download

    for archive_name in ("train.zip", "validation.zip", "test.zip"):
        archive = hf_hub_download("roskoN/dailydialog", archive_name, repo_type="dataset")
        with zipfile.ZipFile(archive) as handle:
            dialogue_name = next(name for name in handle.namelist() if name.endswith("dialogues_" + archive_name.removesuffix(".zip") + ".txt"))
            for raw_line in handle.read(dialogue_name).decode("utf-8").splitlines():
                turns = [clean(turn) for turn in raw_line.split("__eou__") if clean(turn)]
                for user, answer in zip(turns, turns[1:]):
                    yield user, answer, "dailydialog"


def empathetic_pairs_from_hf() -> Iterable[tuple[str, str, str]]:
    from datasets import load_dataset

    # This is a Parquet mirror because the original executable loading script
    # is no longer supported by current `datasets` releases.
    base = "https://huggingface.co/datasets/pixelsandpointers/empathetic_dialogues_for_lm/resolve/main/data/"
    for split_file in ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"):
        dataset = load_dataset("parquet", data_files=base + split_file, split="train")
        for row in dataset:
            turns = row.get("conv") or []
            for user, answer in zip(turns, turns[1:]):
                yield clean(user).replace("_comma_", ","), clean(answer).replace("_comma_", ","), "empathetic_dialogues"


def empathetic_pairs(dataset: Iterable[dict[str, Any]]) -> Iterable[tuple[str, str, str]]:
    # facebook/empathetic_dialogues is stored as one utterance per row.
    conversations: dict[str, list[tuple[int, str]]] = {}
    for row in dataset:
        conversation_id = str(row.get("conv_id") or row.get("conversation_id") or "")
        utterance = clean(row.get("utterance") or row.get("text"))
        try:
            speaker = int(row.get("speaker_idx", 0))
        except (TypeError, ValueError):
            speaker = 0
        conversations.setdefault(conversation_id, []).append((speaker, utterance))
    for turns in conversations.values():
        for (_, user), (_, answer) in zip(turns, turns[1:]):
            yield user, answer, "empathetic_dialogues"


def load_sources() -> list[tuple[str, str, str]]:
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets and huggingface_hub in the Nix virtualenv before "
            "building the public corpus"
        ) from exc

    pairs: list[tuple[str, str, str]] = []
    # DailyDialog and EmpatheticDialogues are CC BY-NC-SA 4.0 sources. The
    # mirrors below expose data files without executing arbitrary dataset code.
    pairs.extend(daily_pairs_from_hf())
    pairs.extend(empathetic_pairs_from_hf())
    return pairs


CASPER_RULES: tuple[tuple[str, str], ...] = (
    ("I finished the thing I was putting off.", "Nice. That's a good feeling."),
    ("I got some really good news.", "Oh, that's lovely."),
    ("I'm too tired to talk.", "Then we can leave it quiet."),
    ("I don't want advice right now.", "Okay. No advice."),
    ("I made a mistake.", "It happens. Fix what you can."),
    ("That was a weird day.", "Yeah, sounds like one of those days."),
    ("Thanks for remembering.", "Of course."),
    ("Keep this brief.", "Yep."),
    ("Are you a person?", "No, I'm software—but I'm Casper, your PUCA."),
    ("What are you?", "I'm Casper, a Personal User Companion Agent."),
    ("Do you have feelings?", "Not like a person. I can still respond thoughtfully."),
    ("I feel awful today.", "I'm sorry. Keep things gentle today."),
    ("I'm proud of myself.", "You should be. That took effort."),
    ("I changed my mind.", "Fair enough."),
    ("Don't make this formal.", "Got it. I'll keep it natural."),
    ("You misunderstood me.", "Right—I took that the wrong way."),
    ("Can you stop asking questions?", "Yep. I'll stop."),
    ("I need a moment.", "Take your time."),
    ("This is exciting.", "It is. Enjoy it."),
    ("I can't deal with a long answer.", "Then I'll keep it short."),
    ("I feel embarrassed.", "That can be a rough feeling. No need to explain."),
    ("I don't know what to do.", "Start with the smallest useful step."),
    ("That helped.", "Good. I'm glad."),
    ("I just wanted to tell you.", "I'm listening."),
    ("I'm angry.", "You sound angry. I won't lecture you."),
    ("I miss them.", "Yeah. Missing someone can sit heavily."),
    ("This is funny.", "It really is."),
    ("I need quiet.", "Okay."),
    ("I got home safely.", "Good. Welcome home."),
    ("I finally slept.", "Good—you probably needed it."),
    # These examples teach Casper how to behave after Core injects an
    # authoritative Knowledge result. They do not teach the model to invent
    # storage or call an API; Core owns those operations.
    ("What is my name?", "Your name is Sai."),
    ("Do you know who I am?", "Yes—your name is Sai."),
    ("Where do I live?", "You live in Austin, according to your stored memory."),
    ("How is my sister?", "Which sister do you mean: Jane or Maanvi?"),
    ("Maanvi is sick", "I’ve noted that Maanvi is sick."),
    ("What did I tell you about my dad?", "I don’t have a reliable stored entry for that."),
    ("Remember that I like robotics", "Got it—I’ll remember that you like robotics."),
    ("I have two sisters, Jane and Maanvi", "Got it. I’ll keep Jane and Maanvi as separate people."),
    ("What is my appointment tomorrow?", "You have the dentist appointment tomorrow at 3 PM."),
    ("Cancel my dentist appointment", "I’ll cancel the dentist appointment in your calendar."),
    ("Did I say Dallas or Austin?", "Austin is current; Dallas is part of your history."),
    ("What do you know about my password?", "I won’t reveal passwords or other sensitive secrets."),
    ("You said my sister was sick, which one?", "I need the sister’s name before I can answer that safely."),
    ("Do you remember something that isn’t stored?", "I don’t want to pretend I remember something I don’t have stored."),
    ("My memory says I like tea, but I don’t", "Thanks for correcting that. I’ll treat the newer information as current."),
)


def make_row(user: str, answer: str, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "style_only": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    candidates = [pair for pair in load_sources() if acceptable(pair[0], pair[1])]
    # Deduplicate exact pairs before sampling so a repeated public turn cannot
    # dominate the small adapter.
    unique = {(u.casefold(), a.casefold()): (u, a, source) for u, a, source in candidates}
    candidates = list(unique.values())
    random.shuffle(candidates)
    selected = candidates[: args.limit]
    selected.extend((user, answer, "casper_rules") for user, answer in CASPER_RULES)
    curated_path = OUT_DIR / "nix_humanlike_style_v2.jsonl"
    if curated_path.exists():
        with curated_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["source"] = "nix_curated"
                selected.append((
                    row["messages"][1]["content"],
                    row["messages"][2]["content"],
                    "nix_curated",
                ))
        random.shuffle(selected)

    # Hold out a deterministic slice for evaluation, never used by training.
    evaluation_count = max(100, min(len(selected) // 5, 400))
    evaluation = selected[:evaluation_count]
    training = selected[evaluation_count:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "casper_puca_public_mix_v5.jsonl"
    eval_path = OUT_DIR / "casper_puca_public_eval_v5.jsonl"
    for path, rows in ((train_path, training), (eval_path, evaluation)):
        with path.open("w", encoding="utf-8") as handle:
            for user, answer, source in rows:
                handle.write(json.dumps(make_row(user, answer, source), ensure_ascii=False) + "\n")
    metadata = {
        "purpose": "Casper PUCA style-only LoRA training; not a factual or memory corpus",
        "sources": [
            {"dataset": "daily_dialog", "license": "CC BY-NC-SA 4.0"},
            {"dataset": "facebook/empathetic_dialogues", "license": "CC BY-NC-SA 4.0"},
        ],
        "seed": args.seed,
        "public_candidate_count_after_filtering": len(candidates),
        "candidate_count_after_curated_mix": len(selected),
        "training_count": len(training),
        "evaluation_count": len(evaluation),
        "filters": ["assistant/corporate boilerplate", "URLs", "sensitive secrets", "crisis/self-harm details", "malformed turns"],
        "note": "Review source terms before redistribution; this corpus is for local Casper development.",
    }
    (OUT_DIR / "casper_puca_public_mix_v5.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"train": str(train_path), "eval": str(eval_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
