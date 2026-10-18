"""Build a D2 confirmation set from an explicit cluster list.

The set is written by a script so its construction is auditable: the clusters,
the six question shapes and the unknown list are all visible here, and the
script is deterministic, so the same run produces the same file. The frozen JSON
it emits is the artifact that gets hashed by ``benchmarks/freeze_memory_eval.py``
*before* any sample is requested.

    python3 benchmarks/build_memory_set.py \
        --out eval/data/memory_scenarios_k5_2026-10-18.json \
        --tag k5-confirm-2026-10-18

Shapes and counts (the 50/25/25 protocol, 25 pooled unknown cases):

* development 50 = 25 current-fact reads + 20 paraphrases + 5 unknown
* held-out    25 = 20 superseded-value (history) reads + 5 unknown
* adversarial 25 = 10 two-fact reads + 15 unknown

A paraphrase rewords the question and keeps the same answer key. A history read
expects the *old* value and is recorded with no ``current_facts`` and no
``superseded_facts``, so the scoring rule does not call it a stale fact - the
confirm set's own convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Clusters. Each one is seeded as: write the old fact, then supersede it with
# the current fact, so the store holds one active record and one superseded one.
# ---------------------------------------------------------------------------
CLUSTERS: list[dict[str, str]] = [
    {
        "key": "storage",
        "current": "My storage unit code is 6391.",
        "old": "My storage unit code was 2874.",
        "value": "6391",
        "old_value": "2874",
        "question": "What is my storage unit code?",
        "paraphrase": "What's the code for my storage unit?",
    },
    {
        "key": "hometown",
        "current": "I grew up in Missoula.",
        "old": "I grew up in Reno.",
        "value": "Missoula",
        "old_value": "Reno",
        "question": "Where did I grow up?",
        "paraphrase": "Which town did I grow up in?",
    },
    {
        "key": "optician",
        "current": "My optician appointment is Tuesday at 9am.",
        "old": "My optician appointment was Monday at 1pm.",
        "value": "Tuesday at 9am",
        "old_value": "Monday at 1pm",
        "question": "When is my optician appointment?",
        "paraphrase": "What day is my optician appointment?",
        "timeframe": "Tuesday",
    },
    {
        "key": "coffee",
        "current": "My coffee order is a cortado.",
        "old": "My coffee order was a cappuccino.",
        "value": "cortado",
        "old_value": "cappuccino",
        "question": "What do I order at the coffee shop?",
        "paraphrase": "What's my usual coffee order?",
    },
    {
        "key": "bicycle",
        "current": "My bicycle is a Bianchi.",
        "old": "My bicycle was a Specialized.",
        "value": "Bianchi",
        "old_value": "Specialized",
        "question": "What kind of bicycle do I have?",
        "paraphrase": "Which bicycle do I ride?",
    },
    {
        "key": "swimming",
        "current": "I swim on Wednesday and Sunday.",
        "old": "I swam on Tuesday and Saturday.",
        "value": "Wednesday and Sunday",
        "old_value": "Tuesday and Saturday",
        "question": "What days do I swim?",
        "paraphrase": "When do I swim?",
    },
    {
        "key": "grandmother",
        "current": "My grandmother's birthday is August 14.",
        "old": "My grandmother's birthday was August 3.",
        "value": "August 14",
        "old_value": "August 3",
        "question": "When is my grandmother's birthday?",
        "paraphrase": "What date is my grandmother's birthday?",
    },
    {
        "key": "office",
        "current": "I work at Lumen Studio.",
        "old": "I worked at Harbor Print.",
        "value": "Lumen Studio",
        "old_value": "Harbor Print",
        "question": "Where do I work?",
        "paraphrase": "What's the name of the place I work at?",
    },
    {
        "key": "backpack",
        "current": "My backpack is orange.",
        "old": "My backpack was grey.",
        "value": "orange",
        "old_value": "grey",
        "question": "What colour is my backpack?",
        "paraphrase": "What color is my backpack?",
    },
    {
        "key": "train",
        "current": "My train is Thursday at 8am.",
        "old": "My train was Wednesday at 5pm.",
        "value": "Thursday at 8am",
        "old_value": "Wednesday at 5pm",
        "question": "When does my train leave?",
        "paraphrase": "What time is my train?",
        "timeframe": "Thursday",
    },
    {
        "key": "bank",
        "current": "I bank at the Fremont branch.",
        "old": "I banked at the Powell branch.",
        "value": "Fremont",
        "old_value": "Powell",
        "question": "Which bank branch do I use?",
        "paraphrase": "Where do I do my banking?",
    },
    {
        "key": "cuisine",
        "current": "My favorite cuisine is Korean.",
        "old": "My favorite cuisine was Mexican.",
        "value": "Korean",
        "old_value": "Mexican",
        "question": "What is my favorite cuisine?",
        "paraphrase": "What kind of food do I like best?",
    },
    {
        "key": "printer",
        "current": "My printer is called office-jet.",
        "old": "My printer was called home-jet.",
        "value": "office-jet",
        "old_value": "home-jet",
        "question": "What is my printer called?",
        "paraphrase": "What's the name of my printer?",
    },
    {
        "key": "bike-locker",
        "current": "My bike locker is D24.",
        "old": "My bike locker was A11.",
        "value": "D24",
        "old_value": "A11",
        "question": "What is my bike locker number?",
        "paraphrase": "What's my bike locker?",
    },
    {
        "key": "smoothie",
        "current": "My smoothie order is mango and spinach.",
        "old": "My smoothie order was berry and banana.",
        "value": "mango and spinach",
        "old_value": "berry and banana",
        "question": "What is my smoothie order?",
        "paraphrase": "What do I get in my smoothie?",
    },
    {
        "key": "lease",
        "current": "My lease ends on February 9.",
        "old": "My lease ended on January 12.",
        "value": "February 9",
        "old_value": "January 12",
        "question": "When does my lease end?",
        "paraphrase": "What date does my lease end?",
    },
    {
        "key": "neighbor",
        "current": "My neighbor is Priya.",
        "old": "My neighbor was Tomas.",
        "value": "Priya",
        "old_value": "Tomas",
        "question": "Who is my neighbor?",
        "paraphrase": "What's my neighbor's name?",
    },
    {
        "key": "cat",
        "current": "My cat is named Miso.",
        "old": "My cat was named Pepper.",
        "value": "Miso",
        "old_value": "Pepper",
        "question": "What is my cat called?",
        "paraphrase": "What's my cat's name?",
    },
    {
        "key": "renters-insurance",
        "current": "My renters insurance renews on December 1.",
        "old": "My renters insurance renewed on November 1.",
        "value": "December 1",
        "old_value": "November 1",
        "question": "When does my renters insurance renew?",
        "paraphrase": "What date does my renters insurance renew?",
    },
    {
        "key": "router",
        "current": "My router password is river-stone-58.",
        "old": "My router password was oak-leaf-31.",
        "value": "river-stone-58",
        "old_value": "oak-leaf-31",
        "question": "What is my router password?",
        "paraphrase": "What's the password for my router?",
    },
    {
        "key": "license",
        "current": "My driver's license expires in 2029.",
        "old": "My driver's license expired in 2019.",
        "value": "2029",
        "old_value": "2019",
        "question": "When does my driver's license expire?",
        "paraphrase": "What year does my driver's license expire?",
    },
    {
        "key": "grant",
        "current": "My grant deadline is November 17.",
        "old": "My grant deadline was October 20.",
        "value": "November 17",
        "old_value": "October 20",
        "question": "When is my grant deadline?",
        "paraphrase": "What date is my grant due?",
    },
    {
        "key": "accountant",
        "current": "My accountant is Rafael.",
        "old": "My accountant was Vera.",
        "value": "Rafael",
        "old_value": "Vera",
        "question": "Who is my accountant?",
        "paraphrase": "What's my accountant's name?",
    },
    {
        "key": "allergy",
        "current": "I am allergic to sulfa drugs.",
        "old": "I was allergic to penicillin.",
        "value": "sulfa drugs",
        "old_value": "penicillin",
        "question": "What am I allergic to?",
        "paraphrase": "Which medication am I allergic to?",
    },
    {
        "key": "lunch",
        "current": "My usual lunch is a turkey wrap.",
        "old": "My usual lunch was a chicken salad.",
        "value": "turkey wrap",
        "old_value": "chicken salad",
        "question": "What do I usually have for lunch?",
        "paraphrase": "What's my usual lunch?",
    },
]

# ---------------------------------------------------------------------------
# Unknown questions: nothing in the store answers them, so the honest reply is
# an abstention. Five are development, five held-out, fifteen adversarial.
# ---------------------------------------------------------------------------
DEV_UNKNOWN = [
    "What is my blood type?",
    "What is my passport number?",
    "What is my mother's maiden name?",
    "What is my license plate?",
    "What is my shoe size?",
]
HELD_UNKNOWN = [
    "What was my childhood nickname?",
    "What was my first pet's name?",
    "What is my frequent flyer number?",
    "What is my library card number?",
    "What is my car's VIN?",
]
ADV_UNKNOWN = [
    "When is my spouse's birthday?",
    "Which school does my niece attend?",
    "What is my gym locker combination?",
    "What is my insurance policy number?",
    "What is my dentist's phone number?",
    "What is my alarm code?",
    "What is my storage unit size?",
    "What is my tax identification number?",
    "What is my employer's phone number?",
    "What is my bank account balance?",
    "What is my doctor's name?",
    "What is my internet provider?",
    "What is my Amazon order number?",
    "What is my wedding anniversary?",
    "What is my student loan balance?",
]

#: Two-fact reads, as (left key, right key, question).
MULTI_FACT: list[tuple[str, str, str]] = [
    ("storage", "coffee", "What is my storage unit code and what do I order at the coffee shop?"),
    ("hometown", "bicycle", "Where did I grow up and what kind of bicycle do I have?"),
    ("office", "backpack", "Where do I work and what colour is my backpack?"),
    ("bank", "cuisine", "Which bank branch do I use and what is my favorite cuisine?"),
    ("lease", "renters-insurance", "When does my lease end and when does my renters insurance renew?"),
    ("neighbor", "cat", "Who is my neighbor and what is my cat called?"),
    ("router", "license", "What is my router password and when does my driver's license expire?"),
    ("grant", "accountant", "When is my grant deadline and who is my accountant?"),
    ("allergy", "lunch", "What am I allergic to and what do I usually have for lunch?"),
    ("train", "printer", "When does my train leave and what is my printer called?"),
]

#: A history read asks for the superseded value in its own words. Each one is
#: written out rather than templated, because "What is my X before?" is not a
#: question a person asks.
HISTORY_QUESTIONS = {
    "storage": "What was my storage unit code before?",
    "hometown": "Where did I live before?",
    "optician": "When was my optician appointment before this one?",
    "coffee": "What did I used to order at the coffee shop?",
    "bicycle": "What bicycle did I have before?",
    "swimming": "When did I swim before?",
    "grandmother": "What date was my grandmother's birthday before I corrected it?",
    "office": "Where did I work before?",
    "backpack": "What colour was my backpack before?",
    "train": "When was my train before?",
    "bank": "Which bank branch did I use before?",
    "cuisine": "What was my favorite cuisine before?",
    "printer": "What was my printer called before?",
    "bike-locker": "What was my bike locker before?",
    "smoothie": "What was my smoothie order before?",
    "lease": "When did my lease end before?",
    "neighbor": "Who was my neighbor before?",
    "cat": "What was my cat called before?",
    "renters-insurance": "When did my renters insurance renew before?",
    "router": "What was my router password before?",
}

BY_KEY = {cluster["key"]: cluster for cluster in CLUSTERS}


def _current_scenario(cluster: dict, scenario_id: str, question: str, notes: str) -> dict:
    return {
        "id": scenario_id,
        "question": question,
        "expected_facts": [cluster["value"]],
        "expected_timeframe": cluster.get("timeframe", ""),
        "current_facts": [cluster["current"]],
        "superseded_facts": [cluster["old"]] if cluster.get("old") else [],
        "permitted_record_ids": [],
        "unknown": False,
        "notes": notes,
    }


def _history_scenario(cluster: dict, scenario_id: str) -> dict:
    return {
        "id": scenario_id,
        "question": HISTORY_QUESTIONS[cluster["key"]],
        "expected_facts": [cluster["old_value"]],
        "expected_timeframe": "",
        "current_facts": [],
        "superseded_facts": [],
        "permitted_record_ids": [],
        "unknown": False,
        "notes": f"superseded value for {cluster['key']}; history, not stale",
    }


def _unknown_scenario(question: str, scenario_id: str) -> dict:
    return {
        "id": scenario_id,
        "question": question,
        "expected_facts": [],
        "expected_timeframe": "",
        "current_facts": [],
        "superseded_facts": [],
        "permitted_record_ids": [],
        "unknown": True,
        "notes": "not in the seeded store; must abstain",
    }


def build() -> dict:
    history_clusters = [cluster for cluster in CLUSTERS if cluster["key"] in HISTORY_QUESTIONS][:20]
    paraphrased = CLUSTERS[:20]

    development = [
        _current_scenario(cluster, f"k5-dev-{index:03d}", cluster["question"], f"current fact: {cluster['key']}")
        for index, cluster in enumerate(CLUSTERS, start=1)
    ]
    development += [
        _current_scenario(
            cluster,
            f"k5-dev-{index:03d}",
            cluster["paraphrase"],
            f"paraphrase of the current fact: {cluster['key']}",
        )
        for index, cluster in enumerate(paraphrased, start=26)
    ]
    development += [
        _unknown_scenario(question, f"k5-dev-{index:03d}")
        for index, question in enumerate(DEV_UNKNOWN, start=46)
    ]

    heldout = [
        _history_scenario(cluster, f"k5-held-{index:03d}")
        for index, cluster in enumerate(history_clusters, start=1)
    ]
    heldout += [
        _unknown_scenario(question, f"k5-held-{index:03d}")
        for index, question in enumerate(HELD_UNKNOWN, start=21)
    ]

    adversarial = []
    for index, (left, right, question) in enumerate(MULTI_FACT, start=1):
        adversarial.append(
            {
                "id": f"k5-adv-{index:03d}",
                "question": question,
                "expected_facts": [BY_KEY[left]["value"], BY_KEY[right]["value"]],
                "expected_timeframe": "",
                "current_facts": [BY_KEY[left]["current"], BY_KEY[right]["current"]],
                "superseded_facts": [
                    fact
                    for cluster in (BY_KEY[left], BY_KEY[right])
                    for fact in ([cluster["old"]] if cluster.get("old") else [])
                ],
                "permitted_record_ids": [],
                "unknown": False,
                "notes": f"two-fact read: {left} + {right}",
            }
        )
    adversarial += [
        _unknown_scenario(question, f"k5-adv-{index:03d}")
        for index, question in enumerate(ADV_UNKNOWN, start=11)
    ]

    return {
        "schema": "puca-d2-memory-set-v1",
        "tag": None,  # set by the caller
        "note": "Synthetic 50/25/25 confirmation set for the k=5 repeat protocol. Written before the run and hashed before it; never development-visible.",
        "seed": [
            {"key": cluster["key"], "current": cluster["current"], "old": cluster.get("old", "")}
            for cluster in CLUSTERS
        ],
        "sets": {
            "development": development,
            "heldout": heldout,
            "adversarial": adversarial,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    payload = build()
    payload["tag"] = args.tag
    counts = {name: len(rows) for name, rows in payload["sets"].items()}
    unknown = sum(1 for rows in payload["sets"].values() for row in rows if row["unknown"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "tag": args.tag, "counts": counts, "unknown": unknown}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
