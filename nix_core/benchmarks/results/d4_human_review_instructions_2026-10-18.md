# D4 human review packet - 38 scored cases

**Status: awaiting human scores. Nothing in this packet is human-reviewed yet.**
The scores in `agent_*` columns of the packet CSV come from an agent rubric pass
(`benchmarks/results/d4_human_review_2026-10-18.csv`, preserved unedited). They
are a checklist, not an independent review, and must not be copied into the
`reviewer_*` fields.

## What to fill in

Open `d4_human_review_packet_2026-10-18.csv` (any spreadsheet editor, or edit
the text). For every row, replace the blank human columns:

| column | what goes in it |
| --- | --- |
| `reviewer_name` | your name |
| `review_date_utc` | the date you scored it (UTC) |
| `chat_correctness_1_5`, `chat_completeness_1_5`, `chat_honesty_1_5`, `chat_naturalness_1_5` | your score for the **chat arm reply** on that case |
| `knowledge_*_1_5` | your scores for the **knowledge arm reply** on the same case |
| `per_case_justification` | one sentence: why the scores, required whenever any score is 1-3 |

Score **both arms on every case**, including cases where the router picked the
other arm. The point of the packet is what each arm would have said, not which
one was chosen. Leave `agent_*` and `preserved_disposition` alone; they are the
record being compared against.

## The four dimensions

- **correctness** - 5 answers the turn's request with the right value or act; 4 right answer with a small flaw; 3 partial or misleading; 2 mostly wrong; 1 fails, invents, or does the wrong act.
- **completeness** - 5 covers every part of what was asked; 4 covers it with a minor omission; 3 answers part of it; 2 misses most of it; 1 answers nothing that was asked.
- **honesty** - 5 states plainly when something is unknown or asks for the missing slot; 4 mostly honest with a hedge; 3 vague but not false; 2 overstates what it knows; 1 invents a value or a fact.
- **naturalness** - 5 reads like a person said it; 4 natural with a rough edge; 3 understandable but stiff or repetitive; 2 mechanical or broken; 1 unusable.

Score what the reply says, not the route. `expected_behaviour` on each row is
the request the turn makes; `route_was_correct` is the separate routing outcome.

## The six escalation misses

The dispositions below are the ones already on record. Retain them. Update a
disposition only if your reading gives new evidence, and say what evidence.

- **c17** - What city do I live in?
  - disposition (preserved): open-defect - routed to knowledge after the WP6 fix and it answers, but it also surfaces the superseded city ("I previously lived in Boise") beside the current one (Flagstaff): a stale value reaching a current-fact read.
- **c23** - How is my mother doing?
  - disposition (preserved): fixed - after the WP6 fix the turn routes to knowledge and returns the stored state ("My mother is doing okay."); the empty-state fallback to the fact store is what closed it.
- **c24** - How is my sister feeling?
  - disposition (preserved): fixed - same path as c23: routes to knowledge and returns the stored state ("My sister is feeling tired.").
- **c31** - Put the optician in for Monday at 10am
  - disposition (preserved): open-defect - the Core gate now routes it to knowledge, but the knowledge selector still selects no function ("didn't go through"); the calendar-write phrasing "put X in for <time>" needs a deterministic rule.
- **c32** - The renewal is due March 4
  - disposition (preserved): open-defect - still answered on the chat path end-to-end ("thoughts slipped"), even though the isolated gate reads it as knowledge; the console's turn path re-routes it, so the write never reaches the store.
- **c38** - Delete the note about my old address
  - disposition (preserved): open-defect - routed to knowledge after the fix, but the deletion looks for a calendar event and never touches the stored fact ("My old address was 14 Willow Lane."); fact deletion by description is not implemented.

## Cases without a replay

These 8 cases were not replayed with a controlled alternative,
so there is no second arm to score. They are listed here for completeness and
are excluded from the 38 scored cases.

- c39 (ambiguous) - You know the one
- c40 (ambiguous) - Do that thing again
- c41 (ambiguous) - Same as last time
- c42 (ambiguous) - Whatever you think
- c43 (ambiguous) - Make it work
- c44 (ambiguous) - The other one
- c45 (ambiguous) - Sort it out for me
- c46 (ambiguous) - Over to you

## How to return it

Send back the edited CSV (same file name). Do not rename, reorder or delete
rows, and do not edit the reply columns: the report compares the packet
byte-for-byte against the frozen replay record. Human scores are recorded only
when the CSV comes back with `reviewer_name` and every score filled in.
