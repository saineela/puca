from __future__ import annotations

"""
Knowledge-worthiness labeling from public conversational data.

The DailyDialog corpus (Li et al., 2017) ships human-annotated
dialogue-act labels per turn (Li et al.'s four-way scheme):
    0 = no_intent   1 = inform   2 = question   3 = directive
    4 = commissive
which we use as the backbone signal for the new 'knowledge'
(knowledge-worthiness) head:

    POSITIVE (knowledge-worthy):
      act == inform  (the speaker is conveying information)
      + lexicon support (first-person details, owned things,
        plans, skills, relationships...)

    NEGATIVE (not knowledge-worthy):
      questions, greetings/thanks, phatic no_intent turns,
      directives/commissives without informative content.

A turn is only POSITIVE when BOTH signals agree (high precision);
everything else is NEGATIVE. This mirrors how the production gate
blends a neural head with deterministic evidence.

Train/test discipline:
    train.zip  -> 11,118 dialogues -> label + train
    test.zip   ->  1,000 dialogues -> held out until first-try eval
"""

import re
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "models" / "training_data"
)

_ACT_NAMES = {0: "no_intent", 1: "inform", 2: "question",
              3: "directive", 4: "commissive"}

# ---------------------------------------------------------------------------
# Lexicon: evidence of knowledge-worthy content
# ---------------------------------------------------------------------------

_FIRST_PERSON = re.compile(
    r"\b(i|i'm|im|i've|ive|i'll|ill|i'd|my|me|mine|we|our)\b", re.I
)
_HAVE_OWN = re.compile(
    r"\b(i have|i own|i got|bought|brought|received|was given|"
    r"we have|adopted|signed up|joined|rented|booked)\b", re.I
)
_PLANS = re.compile(
    r"\b(i'm going to|im going to|i will|i'll|we will|i plan to|"
    r"i want to|i need to|i hope to|i'm thinking of|im thinking of|"
    r"planning on|next week|next month|tomorrow|tonight)\b", re.I
)
_SKILLS = re.compile(
    r"\b(i can|i know how|i'm good at|im good at|i learned|"
    r"i study|i studied|i speak|i play|i practice)\b", re.I
)
_RELATIONSHIP = re.compile(
    r"\b(my (wife|husband|sister|brother|mom|mother|dad|father|son|"
    r"daughter|friend|boss|cousin|uncle|aunt|dog|cat)|is my)\b", re.I
)
_DETAILS = re.compile(
    r"\b(\d+ (am|pm|o'clock|oclock|dollars|years|days|weeks|months|"
    r"miles|minutes|hours)|every (day|morning|evening|weekend)|"
    r"on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.I,
)
_PREFERENCE = re.compile(
    r"\b(i love|i like|i enjoy|i hate|i dislike|i prefer|my favorite|"
    r"i can't stand|i cannot stand)\b", re.I
)
_GREETINGS = re.compile(
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|"
    r"thanks|thank you|thank you very much|bye|goodbye|good night|"
    r"see you|ok|okay|oh|yeah|yes|no|right|sure|wow|oh really|"
    r"i see|that's great|thats great|that's nice|thats nice|"
    r"that's too bad|thats too bad|congratulations|really|"
    r"that sounds|sounds good)\b[.!.?]*\s*$",
    re.I,
)

LEXICON_PATTERNS = [
    _HAVE_OWN, _PLANS, _SKILLS, _RELATIONSHIP, _DETAILS, _PREFERENCE,
]


@dataclass(frozen=True)
class KnowledgeExample:
    text: str
    label: int                 # 1 = knowledge-worthy, 0 = not
    act: int                   # DailyDialog dialogue act
    emotion: int               # DailyDialog emotion id (context only)
    lexicon_hits: int          # how many lexicon patterns matched
    dialogue_id: int
    turn_index: int


def _turn_is_knowledge_worthy(turn: str, act: int) -> tuple[int, int]:
    """
    Return (label, lexicon_hits).

    v2 labeling: a PURE FUNCTION OF TEXT. The act annotation was used
    in v1 but is invisible to a text-only model -- turns like "Of
    course, I do." flip between inform/commissive annotators, making
    the task partially unlearnable (first-try held-out F1 43.6%).
    The production question is text-shaped anyway: does this utterance
    carry first-person knowledge evidence? So:

        label = 1  iff  first-person  AND  >=1 lexicon evidence  AND
                       not a phatic/greeting turn  AND  >= 3 words

    The act annotation is retained on the example for analysis only.
    """
    turn_clean = turn.strip()
    if not turn_clean or len(turn_clean.split()) < 3:
        return 0, 0

    # pure phatic/greeting turns are never knowledge
    if _GREETINGS.match(turn_clean):
        return 0, 0

    # knowledge evidence from lexicon
    lexicon_hits = sum(
        1 for pattern in LEXICON_PATTERNS if pattern.search(turn_clean)
    )
    has_first_person = bool(_FIRST_PERSON.search(turn_clean))

    if has_first_person and lexicon_hits >= 1:
        return 1, lexicon_hits

    return 0, lexicon_hits


def load_dailydialog(
    split: str = "train",
    *,
    max_dialogues: int | None = None,
    data_root: Path | None = None,
) -> list[KnowledgeExample]:
    """
    Load DailyDialog turns as KnowledgeExamples.

    split: 'train' (11,118 dialogues) or 'test' (1,000, HELD OUT).
    """
    if data_root is None:
        data_root = DATA_ROOT
    base = data_root / f"dailydialog_{split}" / split
    suffix = split

    dialogues_path = base / f"dialogues_{suffix}.txt"
    act_path = base / f"dialogues_act_{suffix}.txt"
    emotion_path = base / f"dialogues_emotion_{suffix}.txt"

    if not dialogues_path.exists():
        raise FileNotFoundError(
            f"DailyDialog {split} data not found under {base}. "
            "Extract train.zip / test.zip from roskoN/dailydialog "
            "into models/training_data/dailydialog_{split}/."
        )

    examples: list[KnowledgeExample] = []
    with (
        dialogues_path.open(encoding="utf-8") as d_fh,
        act_path.open(encoding="utf-8") as a_fh,
        emotion_path.open(encoding="utf-8") as e_fh,
    ):
        for dialogue_id, (d_line, a_line, e_line) in enumerate(
            zip(d_fh, a_fh, e_fh)
        ):
            turns = [
                t.strip() for t in d_line.strip().split("__eou__")
                if t.strip()
            ]
            acts = [int(a) for a in a_line.split()]
            emotions = [int(e) for e in e_line.split()]
            for turn_index, turn in enumerate(turns):
                act = acts[turn_index] if turn_index < len(acts) else 0
                emotion = (
                    emotions[turn_index]
                    if turn_index < len(emotions) else 0
                )
                label, lexicon_hits = _turn_is_knowledge_worthy(
                    turn, act
                )
                examples.append(
                    KnowledgeExample(
                        text=turn,
                        label=label,
                        act=act,
                        emotion=emotion,
                        lexicon_hits=lexicon_hits,
                        dialogue_id=dialogue_id,
                        turn_index=turn_index,
                    )
                )
            if max_dialogues and dialogue_id + 1 >= max_dialogues:
                break

    return examples


def split_label_distribution(
    examples: list[KnowledgeExample],
) -> dict[str, int]:
    pos = sum(e.label for e in examples)
    return {
        "total": len(examples),
        "knowledge_worthy": pos,
        "not_knowledge": len(examples) - pos,
    }
