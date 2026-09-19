"""
Labeled evaluation set for the Nix Core request router.

Structure:

  HAND_WRITTEN - the original curated entries
  GENERATED    - realistic prompt families: natural phrasings, filler
                 prefixes ("hey nix,"), typos and ASR-style spellings,
                 real names/objects/times. Over 1000 entries, deduped,
                 deterministic (seeded), balanced knowledge vs chat.

Each entry: (request, expected_route). Routes:
  K = knowledge   C = chat

Contract: every entry MUST be resolved by the deterministic rules
(route != UNKNOWN) and MUST equal its label - unknown-shape requests
are the model layer's job and live in hard_corpus.py instead.

Used by test_router.py, eval_router.py and the console Testing page.
"""

import random

from router import CHAT, KNOWLEDGE

_HAND_WRITTEN = [
    # -------------------------------------------------- knowledge: store
    ("remember that my sister's birthday is may 4", KNOWLEDGE),
    ("remember my wifi password ishouse5", KNOWLEDGE),
    ("note that i have a dentist appointment friday", KNOWLEDGE),
    ("keep in mind that mom prefers morning calls", KNOWLEDGE),
    ("don't forget that i have a tsa meeting tuesday", KNOWLEDGE),
    ("save that my locker code is 4412", KNOWLEDGE),
    ("my robotics class is tuesday at 6pm", KNOWLEDGE),
    ("my sister's birthday is in two weeks", KNOWLEDGE),
    ("i hate pineapple on pizza", KNOWLEDGE),
    # ------------------------------------------------ knowledge: recall
    ("what time is my meeting tomorrow", KNOWLEDGE),
    ("when is my tsa meeting", KNOWLEDGE),
    ("when is my dentist appointment", KNOWLEDGE),
    ("what do i have this week", KNOWLEDGE),
    ("do i have anything tomorrow", KNOWLEDGE),
    ("where is my car keys", KNOWLEDGE),
    ("where did i put my passport", KNOWLEDGE),
    ("what did i say about the garage code", KNOWLEDGE),
    ("who is my manager", KNOWLEDGE),
    ("do i like spicy food", KNOWLEDGE),
    ("what are my plans for the weekend", KNOWLEDGE),
    ("show me my calendar", KNOWLEDGE),
    ("list my tasks", KNOWLEDGE),
    ("what's on my schedule today", KNOWLEDGE),
    ("remind me when the next robotics class is", KNOWLEDGE),
    # -------------------------------------------- knowledge: scheduling
    ("i have a dentist appointment tomorrow at 3", KNOWLEDGE),
    ("schedule a meeting with bob on friday", KNOWLEDGE),
    ("set an alarm for 6am", KNOWLEDGE),
    ("set alarm for 7:30 am", KNOWLEDGE),
    ("wake me up at 7 tomorrow", KNOWLEDGE),
    ("set a reminder to call mom at 5pm", KNOWLEDGE),
    ("remind me to take out the trash tonight", KNOWLEDGE),
    ("remind me about the robotics class", KNOWLEDGE),
    ("add an event to my calendar for friday", KNOWLEDGE),
    ("book a table for two saturday", KNOWLEDGE),
    ("cancel my dentist appointment", KNOWLEDGE),
    ("reschedule my meeting with bob to monday", KNOWLEDGE),
    ("move my tsa meeting to next week", KNOWLEDGE),
    ("this weekend i have a pizza party", KNOWLEDGE),
    ("tomorrow i have a meeting with sarah", KNOWLEDGE),
    ("i have a meeting with bob and a dentist appointment friday", KNOWLEDGE),
    # ------------------------------------------------------ chat: social
    ("hello", CHAT),
    ("hi", CHAT),
    ("hey there", CHAT),
    ("good morning", CHAT),
    ("who are you", CHAT),
    ("what can you do", CHAT),
    ("tell me a joke", CHAT),
    ("how are you doing today", CHAT),
    ("i'm bored", CHAT),
    ("thank you", CHAT),
    ("you're funny", CHAT),
    ("what do you think about electric cars", CHAT),
    # ------------------------------------------- chat: internet / world
    ("what's the weather like today", CHAT),
    ("will it rain tomorrow", CHAT),
    ("what's the latest news", CHAT),
    ("news about the mars mission", CHAT),
    ("who is the president of france", CHAT),
    ("who won the game last night", CHAT),
    ("what is the capital of australia", CHAT),
    ("how do i fix a leaky faucet", CHAT),
    ("how to make pancakes", CHAT),
    ("search for the best noise cancelling headphones", CHAT),
    ("look up the release date of gta 6", CHAT),
    ("what is quantum computing", CHAT),
    ("define entropy", CHAT),
    ("explain how wifi works", CHAT),
    ("recommend a good sci-fi movie", CHAT),
    ("best restaurant near me", CHAT),
    ("recipe for lasagna", CHAT),
    ("what's the population of tokyo", CHAT),
    ("translate 'good morning' to japanese", CHAT),
    ("what's the stock price of apple", CHAT),
    ("who was cleopatra", CHAT),
    ("why is the sky blue", CHAT),
    ("when is the next solar eclipse", CHAT),
]

# ----------------------------------------------------------------------
# Generated families
# ----------------------------------------------------------------------

_GEN = random.Random(20260906)


def _rotate(items, start):
    """Deterministic rotation over a vocabulary."""
    n = len(items)
    return [items[(start + step) % n] for step in range(n)]


def _knowledge_family() -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []

    # 1. store facts: personal statements "my X is Y"
    items = [
        "wifi password", "garage code", "locker combination", "laptop password",
        "router password", "bike lock code", "safe combination", "alarm code",
        "office address", "passport number", "gym locker code", "desk phone extension",
        "backup email", "storage unit code", "gate code", "parking spot number",
    ]
    values = [
        "4712", "hunter2", "house5", "blue42", "9981", "sunset7", "delta9",
        "8231", "falcon12", "green65",
    ]
    store_templates = [
        "remember that my {i} is {v}",
        "remember my {i} is {v}",
        "save that my {i} is {v}",
        "my {i} is {v}",
        "don't forget my {i} is {v}",
    ]
    for index, item in enumerate(items):
        for offset, value in enumerate(values):
            template = store_templates[(index * 3 + offset) % len(store_templates)]
            prompts.append((template.format(i=item, v=value), KNOWLEDGE))

    # 2. recall facts
    recall_shapes = [
        "what is my {i}",
        "whats my {i}",
        "do you remember my {i}",
        "remind me what my {i} is",
        "can you tell me my {i}",
    ]
    recall_items = [
        "wifi password", "garage code", "locker combination", "boss's name",
        "neighbors' names", "dentist's number", "insurance policy", "passport number",
        "loan rate", "wifi name", "parking spot", "desk phone extension",
    ]
    for index, item in enumerate(recall_items):
        for offset, shape in enumerate(recall_shapes):
            prompts.append((shape.format(i=item), KNOWLEDGE))

    # 3. whereabouts
    objects = [
        "keys", "wallet", "passport", "charger", "glasses", "umbrella",
        "work badge", "headphones", "journal", "toolkit", "medication", "earbuds",
    ]
    verbs = ["left", "put", "lost", "found"]
    places = ["in the car", "at the office", "in the kitchen", "at the gym"]
    for index, obj in enumerate(objects):
        verb = verbs[index % len(verbs)]
        place = places[(index * 3) % len(places)]
        prompts.append((f"i {verb} my {obj} {place}", KNOWLEDGE))
        prompts.append((f"where did i put my {obj}", KNOWLEDGE))

    # 4. event creation
    events = [
        "dentist appointment", "team meeting", "doctor visit", "piano lesson",
        "soccer practice", "job interview", "dinner reservation", "haircut",
        "vet appointment", "parent teacher conference", "car inspection",
        "gym session", "study group", "movie night", "family dinner",
    ]
    whens = [
        "tomorrow at 3", "friday at 5pm", "next monday at 9", "tonight at 7",
        "sunday morning", "on saturday at noon", "this afternoon", "next tuesday at 2pm",
    ]
    for index, event in enumerate(events):
        for offset, when in enumerate(whens):
            lead = "i have a" if (index + offset) % 2 == 0 else "i have an"
            prompts.append((f"{lead} {event} {when}", KNOWLEDGE))

    # 5. reminders
    tasks = [
        "take out the trash", "call the dentist", "water the plants",
        "feed the cat", "defrost the chicken", "pay the electric bill",
        "pick up the dry cleaning", "text the landlord", "renew my prescription",
        "back up my laptop", "change the air filter", "call the pharmacy",
        "move the laundry", "book the hotel",
    ]
    when_suffixes = [
        "tonight", "tomorrow at 6pm", "this evening", "at 8", "on friday morning",
        "later today",
    ]
    for index, task in enumerate(tasks):
        for offset, suffix in enumerate(when_suffixes):
            shape = (
                "remind me to {t} {w}"
                if (index + offset) % 3
                else "set a reminder to {t} {w}"
            )
            prompts.append((shape.format(t=task, w=suffix), KNOWLEDGE))

    # 6. alarms
    alarm_times = [
        "5am", "6:30 am", "7", "5:30 pm", "noon", "6 in the morning",
        "7:15", "8 at night", "4am", "9 on sunday", "6:45 am", "10 tomorrow",
    ]
    for index, moment in enumerate(alarm_times):
        shape = "set an alarm for {t}" if index % 2 else "wake me up at {t}"
        prompts.append((shape.format(t=moment), KNOWLEDGE))

    # 7. cancel / move
    ops = ["cancel my {e}", "move my {e} to monday", "reschedule my {e} to friday",
           "postpone my {e} to next week"]
    cancellables = [
        "dentist appointment", "4pm meeting", "gym session", "piano lesson",
        "haircut", "dinner reservation", "vet appointment", "study group",
        "car inspection", "family dinner", "movie night", "job interview",
    ]
    for index, event in enumerate(cancellables):
        for offset in range(2):
            prompts.append((ops[(index + offset) % len(ops)].format(e=event), KNOWLEDGE))

    # 8. calendar browsing
    browse_shapes = [
        "what's on my schedule {w}",
        "what do i have {w}",
        "show me my calendar",
        "what are my plans {w}",
        "list my events",
        "what meetings do i have {w}",
        "whats on my calendar {w}",
        "do i have anything {w}",
    ]
    windows = ["today", "tomorrow", "this week", "next week", "this weekend", "on friday"]
    for index, shape in enumerate(browse_shapes):
        if "{w}" in shape:
            for window in windows:
                prompts.append((shape.format(w=window), KNOWLEDGE))
        else:
            prompts.append((shape, KNOWLEDGE))

    # 9. preferences
    stances = ["love", "like", "hate", "really enjoy"]
    liked = [
        "spicy food", "horror movies", "jazz music", "morning runs",
        "black coffee", "long drives", "board games", "cold weather",
        "thriller novels", "hiking", "spicy noodles", "vinyl records",
        "road trips", "documentaries",
    ]
    for index, thing in enumerate(liked):
        stance = stances[index % len(stances)]
        prompts.append((f"i {stance} {thing}", KNOWLEDGE))

    # 10. family facts
    relations = [
        "mom", "dad", "sister", "brother", "grandma", "grandpa",
        "aunt carol", "uncle joe", "niece lily", "cousin max",
    ]
    dates = ["may 4", "june 12", "next month", "on the 30th"]
    for index, relation in enumerate(relations):
        for offset, date in enumerate(dates):
            shape = "{r}'s birthday is {d}" if (index + offset) % 2 else "{r}'s birthday is in {d}"
            prompts.append((shape.format(r=relation, d=date), KNOWLEDGE))

    return prompts


def _chat_family() -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []

    # 1. greetings
    for base in (
        "hello", "hi", "hey", "yo", "howdy", "sup", "good morning",
        "good afternoon", "good evening", "good night", "hey there",
        "hello there", "hi nix", "hey nix", "morning",
    ):
        prompts.append((base, CHAT))
        prompts.append((base + "!", CHAT))
        prompts.append((base + "??", CHAT))

    # 2. who / what world questions
    persons = [
        "the president of france", "einstein", "nikola tesla", "shakespeare",
        "ada lovelace", "the ceo of apple", "usain bolt", "messi",
        "the inventor of the telephone", "marie curie",
    ]
    for person in persons:
        prompts.append((f"who is {person}", CHAT))
        prompts.append((f"who was {person}", CHAT))

    capitals = [
        "australia", "canada", "japan", "brazil", "kenya", "norway",
        "india", "mexico", "egypt", "spain",
    ]
    for country in capitals:
        prompts.append((f"what is the capital of {country}", CHAT))
        prompts.append((f"what's the population of {country}", CHAT))

    for topic in (
        "quantum computing", "photosynthesis", "blockchain", "gravity",
        "machine learning", "the stock market", "dna", "volcanoes",
        "the solar system", "black holes",
    ):
        prompts.append((f"what is {topic}", CHAT))
        prompts.append((f"explain {topic}", CHAT))

    # 3. how-to
    for task in (
        "fix a leaky faucet", "make pancakes", "tie a tie", "learn python",
        "unclog a drain", "change a tire", "boil eggs", "remove a stain",
        "start a garden", "fold laundry fast", "build a birdhouse",
        "fix a squeaky door", "sew a button", "clean a keyboard",
        "season a cast iron pan", "write a resume", "jump start a car",
        "calibrate a monitor", "grow tomatoes", "sharp knives",
    ):
        prompts.append((f"how do i {task}", CHAT))
        prompts.append((f"how to {task}", CHAT))

    # 4. weather / news / sports
    for city in ("chicago", "austin", "seattle", "miami", "denver"):
        prompts.append((f"what's the weather in {city}", CHAT))
        prompts.append((f"weather {city}", CHAT))
        prompts.append((f"will it rain tomorrow in {city}", CHAT))

    for topic in (
        "the mars mission", "the election", "ai regulation", "the olympics",
        "space launches", "the housing market",
    ):
        prompts.append((f"latest news about {topic}", CHAT))
        prompts.append((f"what's the latest news on {topic}", CHAT))

    for game in ("cubs game", "lakers game", "world series", "super bowl",
                 "champions league", "us open"):
        prompts.append((f"who won the {game} last night", CHAT))
        prompts.append((f"what was the score of the {game}", CHAT))

    # 5. tell me
    for animal in ("octopuses", "dolphins", "ravens", "axolotls", "ants"):
        prompts.append((f"tell me a fun fact about {animal}", CHAT))
        prompts.append((f"tell me one interesting thing about {animal}", CHAT))
    for extra in ("a joke", "another joke", "something interesting",
                  "a scary story", "about yourself"):
        prompts.append((f"tell me {extra}", CHAT))

    # 6. recommendations
    for genre in ("sci-fi", "comedy", "horror", "anime", "documentary"):
        prompts.append((f"recommend a good {genre} movie", CHAT))
        prompts.append((f"recommend me a {genre} series", CHAT))
    for cuisine in ("italian", "thai", "mexican", "japanese", "indian"):
        prompts.append((f"best {cuisine} restaurant near me", CHAT))
        prompts.append((f"recipe for a classic {cuisine} dish", CHAT))

    # 7. assistant smalltalk
    for base in (
        "how are you", "what can you do", "who made you", "are you a robot",
        "are you human", "what model are you", "thank you", "thanks a lot",
        "you're funny", "you are the best", "what is your favorite color",
        "what do you think about electric cars", "do you like pizza",
        "how old are you", "where do you live", "cheer me up",
        "i'm bored", "im bored", "say something nice", "surprise me",
        "good vibes only", "long time no see",
    ):
        prompts.append((base, CHAT))

    # 8. translate / define / misc
    for phrase, lang in (
        ("good morning", "japanese"), ("thank you", "french"),
        ("hello", "german"), ("where is the station", "spanish"),
        ("cheers", "italian"),
    ):
        prompts.append((f"translate '{phrase}' to {lang}", CHAT))
    for word in ("entropy", "kinematics", "inflation", "heuristic", "osmosis"):
        prompts.append((f"define {word}", CHAT))

    # 9. opinions / comparisons
    for thing in (
        "electric cars", "remote work", "video games", "social media",
        "crypto", "pineapple on pizza", "cats vs dogs", "reading books",
        "gym memberships", "meal prepping",
    ):
        prompts.append((f"what do you think about {thing}", CHAT))
        prompts.append((f"do you like {thing}", CHAT))
        prompts.append((f"is {thing} worth it", CHAT))
    for a, b in (
        ("python", "javascript"), ("ios", "android"), ("pc", "console"),
        ("netflix", "youtube"), ("coffee", "tea"), ("biking", "running"),
        ("handwriting", "typing"), ("renting", "buying"),
    ):
        prompts.append((f"which is better {a} or {b}", CHAT))
        prompts.append((f"{a} vs {b}", CHAT))

    # 10. conversions / math / trivia
    for conv in (
        "100 f to c", "20 c to f", "5 miles to km", "2 km to miles",
        "150 lbs to kg", "70 kg to lbs", "1 cup to ml", "350 f to c oven",
        "9 pm cst to ist", "how many ounces in a liter",
    ):
        prompts.append((f"convert {conv}", CHAT))
        prompts.append((conv, CHAT))
    for pct, amt in (("15", "200"), ("20", "85"), ("10", "1250"), ("25", "64")):
        prompts.append((f"what is {pct}% of {amt}", CHAT))
        prompts.append((f"{pct} percent tip on {amt}", CHAT))
    for base in (
        "how many kilometers in a mile", "how many days until christmas",
        "how many days in a leap year", "how many bones in the human body",
        "how many continents are there", "what time zone is denver in",
        "what is the tallest mountain", "what is the longest river",
        "how big is the sun compared to earth", "how fast does light travel",
    ):
        prompts.append((base, CHAT))

    # 11. jokes / fun / games
    for base in (
        "tell me a dad joke", "say a tongue twister", "make me laugh",
        "give me a pickup line", "do a magic trick", "flip a coin",
        "roll a dice", "pick a number 1 to 10", "tell me a riddle",
        "what would you rather ask", "would you rather fight 1 horse or 100 ducks",
        "write a haiku about rain", "write a poem about coffee",
        "give me a fun fact", "give me a random fact",
        "quiz me on state capitals", "play 20 questions with me",
    ):
        prompts.append((base, CHAT))

    # 12. life advice / wellbeing (world knowledge, not personal data)
    for base in (
        "how do i stay motivated", "tips to sleep better",
        "how to stop procrastinating", "how to be more productive",
        "ways to reduce stress", "how to make friends in a new city",
        "how to start running", "how to drink more water",
        "best way to learn guitar", "how to negotiate a raise",
        "how to write a cover letter", "how to save money on groceries",
        "how to fix my sleep schedule", "how do i get better at chess",
    ):
        prompts.append((base, CHAT))

    # 13. animals / science facts
    for base in (
        "are penguins birds", "do fish sleep", "can pigs fly",
        "why do cats purr", "why are flamingos pink", "do bees die after stinging",
        "how long do turtles live", "can snakes hear",
        "what do owls eat", "why is the ocean salty",
        "what causes thunder", "how do vaccines work",
        "why do we dream", "how do plane wings generate lift",
        "what is inside a black hole", "how do magnets work",
    ):
        prompts.append((base, CHAT))

    # 14. tech / world misc
    for base in (
        "what does gpu stand for", "is python faster than c++",
        "what is the difference between wifi and bluetooth",
        "how does the stock market work", "what is a 401k",
        "how do interest rates work", "what is inflation caused by",
        "who invented the lightbulb", "when did world war 2 end",
        "what language do they speak in brazil",
        "what currency does japan use", "which side of the road do they drive on in the uk",
    ):
        prompts.append((base, CHAT))

    # 15. weather / time / date (world, not user's calendar)
    for city in ("paris", "tokyo", "london", "sydney", "toronto", "berlin"):
        prompts.append((f"is it hot in {city} right now", CHAT))
        prompts.append((f"what time is it in {city}", CHAT))
        prompts.append((f"what's the weather this weekend in {city}", CHAT))
    for base in (
        "what day is thanksgiving this year", "when does daylight saving start",
        "is today a federal holiday", "when is the next full moon",
        "what season is it in australia right now",
        "sunrise time tomorrow", "sunset time today",
    ):
        prompts.append((base, CHAT))

    for base in (
        "when is the next solar eclipse", "when is the next super bowl",
        "how tall is mount everest", "how far is the moon",
        "how many people live in iceland", "why is the sky blue",
        "why does it rain", "can cats eat chocolate", "can dogs eat apples",
        "is my laptop supposed to get this hot", "should my resume be one page",
        "what speed should my internet be for streaming",
        "how often should my car get an oil change",
        "what year was my car's model first released",
        "who is my senator", "what does my name mean",
        "do i know anyone who won a nobel prize",
        "what if i scheduled everything on the same day",
    ):
        prompts.append((base, CHAT))

    return prompts


def _apply_realism(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Realism pass: filler prefixes, typos, ASR spellings, punctuation -
    applied at fixed deterministic positions so the labeled set is
    stable across runs.
    """
    fillers = ["hey nix, ", "um so ", "ok so, ", "nix ", "so, ", "alright, "]
    typo_pairs = [
        ("tomorrow", "tommorow"),
        ("remember", "remeber"),
        ("calendar", "calender"),
        ("appointment", "apointment"),
        ("schedule", "scedule"),
        ("tonight", "tonite"),
        ("whats", "what's"),
    ]

    realistic: list[tuple[str, str]] = []
    for index, (text, label) in enumerate(entries):
        variant = text
        if index % 7 == 3:
            variant = fillers[(index // 7) % len(fillers)] + variant
        if index % 11 == 5:
            for correct, wrong in typo_pairs:
                if correct in variant:
                    variant = variant.replace(correct, wrong, 1)
                    break
        if index % 5 == 2:
            variant = variant + "?"
        realistic.append((variant, label))

    return realistic


def _build() -> list[tuple[str, str]]:
    combined = list(_HAND_WRITTEN)
    combined.extend(_knowledge_family())
    combined.extend(_chat_family())
    combined = _apply_realism(combined)

    # dedupe, deterministic shuffle for mixed ordering
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for text, label in combined:
        key = " ".join(text.lower().split())
        if key not in seen:
            seen.add(key)
            deduped.append((text, label))

    _GEN.shuffle(deduped)
    return deduped


CORPUS: list[tuple[str, str]] = _build()

if __name__ == "__main__":
    knowledge = sum(1 for _, label in CORPUS if label == KNOWLEDGE)
    chat = sum(1 for _, label in CORPUS if label == CHAT)
    print(f"corpus: {len(CORPUS)} prompts ({knowledge} knowledge / {chat} chat)")
