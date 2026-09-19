from __future__ import annotations

"""
Training pipeline for the SemanticClassifier.

Sources
-------
  emotion    GoEmotions "simplified" (HF: google-research-datasets/go_emotions)
             -- 58k reddit comments, 27 emotions + neutral, multi-label.
             Column `labels` is a list of ints into the CANONICAL
             GoEmotions order (see GOEMOTIONS_LABEL_ORDER below);
             "neutral" is index 20 in that order.
  sarcasm    tweet_eval/sarcasm (HF: cardiffnlp/tweet_eval) -- binary.
  tone / category / importance
             Purpose-built PERSONAL-domain corpus (first-person
             knowledge statements), generated deterministically below.
             Reddit chatter is the wrong domain for these heads.

Every head is trained by frozen-backbone feature extraction + a tiny
MLP head, so full training runs in minutes on GPU.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .classifier import (
    CATEGORY_LABELS,
    EMOTION_LABELS,
    TONE_LABELS,
    SemanticClassifier,
)
from .knowledge_labels import load_dailydialog, split_label_distribution

DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "models" / "training_data"
)

# ---------------------------------------------------------------------------
# GoEmotions label mapping
# ---------------------------------------------------------------------------

# Canonical GoEmotions column order (27 emotions + neutral at 20).
GOEMOTIONS_LABEL_ORDER = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "neutral", "optimism", "realization", "relief", "remorse",
    "sadness", "surprise",
]

# maps GoEmotions label -> row index in EMOTION_LABELS
GOEMOTIONS_TO_OURS = {
    go_i: EMOTION_LABELS.index(name)
    for go_i, name in enumerate(GOEMOTIONS_LABEL_ORDER)
    if name in EMOTION_LABELS
}


def load_goemotions(max_rows: int | None = None) -> tuple[list[str],
                                                          np.ndarray]:
    """
    Return (texts, multi-hot matrix aligned with EMOTION_LABELS).
    Uses the HF `datasets` library (parquet-backed, no scripts).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "google-research-datasets/go_emotions", "simplified"
    )
    texts: list[str] = []
    rows: list[list[int]] = []

    for split in ("train", "validation"):
        for example in ds[split]:
            text = (example.get("text") or "").strip()
            if not text or len(text) < 8:
                continue
            texts.append(text)
            rows.append(list(example.get("labels") or []))
            if max_rows and len(texts) >= max_rows:
                break
        if max_rows and len(texts) >= max_rows:
            break

    y = np.zeros((len(texts), len(EMOTION_LABELS)), dtype=np.float32)
    for i, go_labels in enumerate(rows):
        for go_i in go_labels:
            ours = GOEMOTIONS_TO_OURS.get(int(go_i))
            if ours is not None:
                y[i, ours] = 1.0
    return texts, y


def load_sarcasm(max_rows: int | None = None) -> tuple[list[str],
                                                       np.ndarray]:
    """
    Binary verbal-irony data from tweet_eval (label 1 = ironic).
    tweet_eval has no 'sarcasm' config; 'irony' is its parent task
    and the standard proxy for sarcasm detection.
    Returns (texts, y) with y aligned to [not_ironic, ironic].
    """
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "irony")
    texts: list[str] = []
    labels: list[int] = []

    for split in ("train", "validation"):
        for example in ds[split]:
            text = (example.get("text") or "").strip()
            if not text or len(text) < 8:
                continue
            texts.append(text)
            labels.append(int(example.get("label") or 0))
            if max_rows and len(texts) >= max_rows:
                break
        if max_rows and len(texts) >= max_rows:
            break

    y = np.array(labels, dtype=np.float32)
    return texts, y


# ---------------------------------------------------------------------------
# Personal-domain corpus (tone / category / importance)
# ---------------------------------------------------------------------------

# Template pools per category. Each entry: (template, importance).
# {name} {city} {food} {person} {topic} are substituted from pools.
_CATEGORIES: dict[str, list[tuple[str, float]]] = {
    "personal_fact": [
        ("My name is {name}.", 0.95),
        ("I live in {city}.", 0.9),
        ("I work as a {job} at {company}.", 0.9),
        ("I study {topic} at {company}.", 0.85),
        ("I was born in {city}.", 0.9),
        ("I have a {pet} named {name}.", 0.8),
        ("I am a {job}.", 0.85),
        ("I drive a {car}.", 0.6),
        ("I am allergic to {food}.", 0.95),
        ("I speak {language}.", 0.85),
    ],
    "preference": [
        ("I love {food}.", 0.8),
        ("My favorite food is {food}.", 0.85),
        ("I really enjoy {topic}.", 0.7),
        ("I like {music} music.", 0.7),
        ("I prefer {topic} over sports.", 0.7),
        ("I hate {food}.", 0.75),
        ("I can't stand {music} music.", 0.7),
        ("I dislike {topic}.", 0.65),
    ],
    "event": [
        ("My birthday is on {date}.", 0.95),
        ("Our wedding is on {date}.", 0.95),
        ("I have a dentist appointment on {date}.", 0.8),
        ("My flight to {city} leaves on {date}.", 0.85),
        ("The meeting with {person} is on {date}.", 0.8),
        ("My graduation is scheduled for {date}.", 0.85),
        ("There is a deadline on {date}.", 0.8),
    ],
    "relationship": [
        ("{person} is my sister.", 0.9),
        ("{person} is my husband.", 0.9),
        ("{person} is my best friend.", 0.85),
        ("My brother {person} works at {company}.", 0.8),
        ("My mother's name is {name2}.", 0.9),
        ("{person} is my coworker.", 0.6),
    ],
    "skill": [
        ("I know {language}.", 0.8),
        ("I am good at {topic}.", 0.75),
        ("I can speak {language} fluently.", 0.8),
        ("I learned {topic} last year.", 0.65),
        ("I am great at {topic}.", 0.75),
    ],
    "opinion": [
        ("I think {topic} is overrated.", 0.5),
        ("In my opinion, {topic} is the best.", 0.5),
        ("I believe {topic} will change everything.", 0.5),
        ("{topic} is boring, if you ask me.", 0.5),
    ],
    "goal": [
        ("I want to learn {language}.", 0.7),
        ("I plan to move to {city}.", 0.8),
        ("My goal is to run a marathon.", 0.75),
        ("I hope to become a {job}.", 0.75),
        ("I need to finish {topic} by {date}.", 0.7),
    ],
    "general": [
        ("The weather is nice today.", 0.2),
        ("I saw a movie last night.", 0.3),
        ("Traffic was terrible this morning.", 0.25),
        ("Just finished reading a book about {topic}.", 0.35),
        ("The coffee here is decent.", 0.2),
    ],
}

_TONE_TRANSFORMS = [
    # (tone, suffix/transform)
    ("urgent", [" right now", " immediately", " asap", " today!"]),
    ("casual", [" lol", " haha", ", kinda", " btw"]),
    ("formal", [", please note", ", kindly remember", ", to be clear",
                ", for your records"]),
    ("neutral", ["", "", "", ""]),
]

POOLS = {
    "name": ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey"],
    "name2": ["Maria", "David", "Linda", "James"],
    "city": ["Chicago", "Austin", "Denver", "Seattle", "Miami"],
    "food": ["sushi", "spicy food", "pineapple pizza", "coffee",
             "peanuts"],
    "person": ["Sarah", "Mike", "Emma", "Chris"],
    "topic": ["machine learning", "gardening", "history", "photography",
              "cooking"],
    "job": ["engineer", "teacher", "designer", "nurse"],
    "company": ["Acme Corp", "the university", "a startup"],
    "pet": ["dog", "cat", "parrot"],
    "car": ["Toyota", "Honda", "old van"],
    "music": ["jazz", "rock", "classical"],
    "language": ["Python", "Spanish", "German", "French"],
    "date": ["June 3rd", "next Friday", "March 12th", "Monday"],
}


def generate_personal_corpus(
    *,
    per_combination: int = 3,
    seed: int = 42,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic corpus: every category template crossed with every
    tone transform. Returns (texts, tone_y, importance_y, category_y).
    """
    rng = random.Random(seed)
    texts: list[str] = []
    tone_y: list[int] = []
    importance_y: list[float] = []
    category_y: list[int] = []

    for category, templates in _CATEGORIES.items():
        cat_index = CATEGORY_LABELS.index(category)
        for template, importance in templates:
            for tone, transforms in _TONE_TRANSFORMS:
                tone_index = TONE_LABELS.index(tone)
                for _ in range(per_combination):
                    text = template
                    for key, pool in POOLS.items():
                        if "{" + key + "}" in text:
                            text = text.replace(
                                "{" + key + "}", rng.choice(pool)
                            )
                    text = text + rng.choice(transforms)
                    # transform shifts urgency into importance
                    imp = min(
                        1.0,
                        importance + (0.15 if tone == "urgent" else 0.0),
                    )
                    texts.append(text)
                    tone_y.append(tone_index)
                    importance_y.append(round(imp, 3))
                    category_y.append(cat_index)

    return (
        texts,
        np.array(tone_y, dtype=np.int64),
        np.array(importance_y, dtype=np.float32),
        np.array(category_y, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_semantic_classifier(
    *,
    max_emotion_rows: int = 40000,
    max_sarcasm_rows: int = 6000,
    epochs_emotion: int = 8,
    epochs_attributes: int = 30,
    epochs_sarcasm: int = 5,
    epochs_knowledge: int = 10,
) -> dict[str, Any]:
    """
    Full supervised training run:
      emotion   head <- GoEmotions (multi-label)
      sarcasm   head <- tweet_eval/irony (binary)
      tone/category/importance <- personal-domain corpus
      knowledge head <- DailyDialog TRAIN split (test split is held out)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    classifier = SemanticClassifier()
    report: dict[str, Any] = {}

    # -- emotion -------------------------------------------------------
    texts, y = load_goemotions(max_rows=max_emotion_rows)
    print(f"goemotions: {len(texts)} examples")
    report["emotion"] = classifier.train_emotion_head(
        texts, y, epochs=epochs_emotion, val_split=0.1
    )

    # -- sarcasm -------------------------------------------------------
    s_texts, s_y = load_sarcasm(max_rows=max_sarcasm_rows)
    print(f"sarcasm: {len(s_texts)} examples")
    report["sarcasm"] = classifier.train_sarcasm_head(
        s_texts, s_y, epochs=epochs_sarcasm
    )

    # -- tone / importance / category ----------------------------------
    p_texts, tone_y, importance_y, category_y = (
        generate_personal_corpus()
    )
    print(f"personal corpus: {len(p_texts)} examples")
    report["attributes"] = classifier.train_attribute_heads(
        p_texts, tone_y, importance_y, category_y,
        epochs=epochs_attributes,
    )

    # -- knowledge-worthiness (DailyDialog TRAIN split only) -----------
    k_examples = load_dailydialog("train")
    k_texts = [e.text for e in k_examples]
    k_y = np.array([e.label for e in k_examples], dtype=np.float32)
    print(
        f"knowledge (DailyDialog train): {split_label_distribution(k_examples)}"
    )
    report["knowledge"] = classifier.train_knowledge_head(
        k_texts, k_y, epochs=epochs_knowledge
    )

    classifier.version = f"supervised-v3-{epochs_emotion}ep"
    classifier.save()

    result = {"rows": {"emotion": len(texts), "sarcasm": len(s_texts),
                       "attributes": len(p_texts),
                       "knowledge": len(k_texts)},
              "metrics": report}
    with (DATA_DIR / "last_training_run.json").open("w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


if __name__ == "__main__":
    train_semantic_classifier()
