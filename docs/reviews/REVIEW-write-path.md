# Write-path review

Scope: `ingest/`, `extract/`, `merge/`, `consolidate/`, `notes/`, `signals/`, `pipeline.py`,
the model clients in `llm/`, and the schema as it serves them. Read end to end on
2026-11-20 against `0486f19`. The read path (`search/`, `report/`, `lint/`, `audit/`,
`bench/`) was reviewed separately and is not changed here.

## How the write path works

One source goes through seven steps. Only extraction runs in parallel; everything that
writes runs one source at a time.

1. **Intake** (`ingest/intake.py`). The original is stored and a `sources` row written,
   keyed by `(source_id, content hash)`. Unchanged bytes are a no-op. Changed bytes under
   the same id become `<id>--v2`, so a ticket that gained a comment is a new version.
2. **Prepare** (`ingest/prepare.py`). Quoted email text and boilerplate are stripped and
   every citable block gets a locator in brackets at the start of its line: a call
   timestamp, a ticket comment number, an email paragraph, a CRM row id. The same step
   returns an index of those locators with their character spans.
3. **Filter** (`ingest/filter.py`). A source that names no known customer is excluded. Two
   percent of excluded sources are extracted anyway to measure what the filter misses.
4. **Extract** (`extract/`). A long source is cut into segments of about 4,000 tokens on
   line boundaries. Each segment is one model call that returns facts: a type, a claim, the
   entities it names, a locator, a verbatim quote for the five evidentiary types. A reply
   that fails the schema is salvaged fact by fact. Segment results are unioned.
5. **Anchor** (`extract/anchor.py`, new). Each fact is tied to its place in the source by
   code: the quote is found in the text, a re-typed quote is replaced by the source's own
   words, and a locator that disagrees with the quote or is not in the source is repaired.
   A fact that cannot be placed is parked for a person.
6. **Merge** (`merge/`). Each fact's entities are resolved against the registry by exact
   name, alias or normalised name, never by guess. The fact gets a home note
   (`signal-<type>-<account>` for most types). The home note's live claims are scored
   against it; below 0.45 it is new with no model call, otherwise one compare call answers
   same, extends, contradicts or different. Same adds a citation to the existing claim.
   Contradicts retires the old claim only when both carry explicit dates and the new one is
   later; otherwise both stay and a person is asked. Everything one source writes is one
   transaction.
7. **Consolidate** (`consolidate/`, a separate pass). When a note passes 30 claims or 12
   updates, a cheap model rewrites its synthesis from the live claims, code checks every
   citation marker it used is real, and older or superseded claims move to an archive note.

The rule that holds it together: a model or a heuristic may propose, and only a person
makes something stick. Facts that cannot be filed wait in `review_queue`; they are not
dropped and not guessed into place.

## Findings

Ranked by what they cost. "Fixed" means changed in this branch with a test.

| # | Finding | Status |
|---|---|---|
| 1 | **Facts vanished with no row anywhere.** A fact that named no entity was dropped silently. A source's second fact under one unknown name was dropped (the review row kept the first fact's words only). In the stress run below, 16% of facts were lost this way or by finding 4. | Fixed (`30eea5f`) |
| 2 | **Answering a review row did nothing for the facts behind it.** Mapping an unknown name saved the alias and closed the row; the facts that had been waiting on the name were never merged. | Fixed: `merge/replay.py`, called by the add-alias route (`30eea5f`) |
| 3 | **The quote gate passed invented quotes on long sources.** It matched a quote's tokens in order anywhere in the whole text. On a 20,000-word transcript an invented cancellation quote scored 0.88 against a 0.75 bar. | Fixed: the matched tokens must sit together in one window (`da3e4e6`) |
| 4 | **In-source dedup folded distinct facts.** Any two same-type facts with a shared entity and alike wording (0.6, or 0.4 within 90 seconds) became one, on every source, single-segment ones included. | Fixed: fuzzy dedup only across segments or passes (`034f7b5`) |
| 5 | **Locators were never checked.** The pipeline had the source's locator index and did not use it. A claim could cite a timestamp its quote was not at, or a locator the source does not have, and "open the original" landed in the wrong place or nowhere. | Fixed (`da3e4e6`) |
| 6 | **Stored quotes were not always the source's words.** A near-verbatim quote passed the gate and was stored as the model typed it. | Fixed: replaced by the source's own span (`da3e4e6`) |
| 7 | **The compare call could not say "unrelated".** With only same, extends and contradicts, a pair the gate matched on shared words was forced into one: a wrong same swallowed the new fact, a wrong contradicts retired a true claim or opened a review row. The code's own comment named this. | Fixed: `different` verdict; the prompt now shows both quotes and dates (`4e9e1d7`) |
| 8 | **Why the review queue floods.** Every unknown name opened a row, including names on facts that were filed anyway ("the night shift lead"), and wrong contradicts verdicts added more. | Fixed, see "The review queue" below (`30eea5f`, `4e9e1d7`) |
| 9 | **Haiku 5.5 would have failed every call.** The first-party client treated any "haiku" as Haiku 4.5 and sent temperature 0, which current models reject. Thinking models also shared a 4,096-token cap with their own thinking, and Bedrock forced a tool choice that Sonnet 5.5, Opus 5.5 and Fable 5.2 reject. No pricing rows for current ids. | Fixed (`2ab76ba`) |
| 10 | **Paid retries for free fixes.** Any schema fault (a null sentiment, an extra key) triggered a second full extraction call. A customer's own fact types always failed validation, cost a retry, and were then dropped. A quote of 41 words rejected its fact. | Fixed (`1885fbf`) |
| 11 | **Schema-rejected facts left no trace** outside a metering field. | Fixed: one `extract_rejected` changelog row per source, counted in the run report (`da3e4e6`) |
| 12 | **Mention linking rewrote the inside of words** ("Arc" in "Arcadia" became `[[arc]]adia`). | Fixed (`a833665`) |
| 13 | **The synthetic vendor's name was in engine code.** Every tenant's product notes said "a product Halvard sells". | Fixed (`a833665`) |
| 14 | **Consolidation drafted from retired claims.** Superseded claims went to the draft model unmarked and undated, so a synthesis could state the old renewal date as current. Archiving went by age, not by superseded first. | Fixed (`a833665`) |
| 15 | **A single-fact merge blanked the source's summary** ("Keep as claim" in review). | Fixed (`30eea5f`) |

### Not fixed, and why

| Finding | Why it is left |
|---|---|
| `Settings` still defaults to `claude-haiku-4-5` and `claude-sonnet-5`. Haiku 5.5 lists at a tenth of Haiku 4.5's price. | The default provider is the Agent SDK, whose bundled CLI lags new model ids. Needs one live call before the default moves. The first-party client's own defaults did move. |
| The prompt's worked email example teaches `¶2` locators; `prepare.py` emits `<message-id>#p2`. | Changing the prompt changes its hash, which stamps the training labels, and cannot be judged without a live bench. The anchor now repairs the shortened form. |
| Segments are 4,000 tokens. | Sized for small-context models. Larger segments mean fewer calls and less cross-segment dedup, but recall on long inputs has to be measured on a real model. |
| Merge compares a fact with one candidate, from its home note only. | A top-three compare in one call would catch more duplicates. It changes the verdict schema again and wants a live measurement. |
| Claim embeddings are recomputed for every source that touches a note. | Fine at corpus size, slow on a note with thousands of claims. Wants a stored vector per claim. |
| Backfills run synchronously. | The Message Batches API halves the cost of a backfill; it needs an asynchronous pipeline mode. |
| `compare_model` is not used by merge; the compare call runs on the extraction model. | Wiring it needs a provider-aware default, or an OpenRouter deployment gets a Claude id. |
| Consolidation checks that citations are real, not that the prose is faithful. The numbers rule the weekly report uses is not applied. | Applying it rejects honest counts ("three late deliveries") and each rejection is a review row. |
| A deterministic failure is retried three times by `_retry_stage`. | Harmless with the fake; with a paid model it bills three times. Needs retryable and non-retryable errors told apart per client. |
| `signals/` | Read, not changed. It already checks quotes and numbers per item; it inherits the stricter quote gate. |

## The review queue

Before: one open row per distinct unknown name, whatever the name was attached to. A cheap
model names many things the registry will never hold, and most of those facts were filed
correctly on their account anyway. The row asked a person a question whose answer changed
nothing.

Now a name opens a row when the answer matters:

- **It blocks a fact.** The fact named only that name and has no home. Open at once, with
  the whole fact stored on the row.
- **It keeps coming back.** Seen in three distinct sources
  (`PORTOLAN_MENTION_PROMOTE_AFTER`). Open.
- **Otherwise it waits** as `watching` (migration 0039). A watching row changes nothing in
  the graph and loses nothing: the mention and its facts are on the row.

A person is still the only path to anything sticky. Watching only delays a question. When
a person maps a name, the facts waiting on it are merged then.

Watching rows are not shown anywhere yet. `/review` and the lint read `status = 'open'`.

## Numbers

**Tests.** Before: 4,614 passed, 9 skipped, 0 failed (Postgres 17 with pgvector, 10 min).
After: 4,690 passed, 9 skipped, 0 failed on the rebased tree (44 new tests from this branch).

**Corpus run, fake extractor** (`python -m portolan.pipeline --llm fake`, 248 sources, 76
planted facts). Unchanged where it should be: 76 claims, 3 superseded, 4 citations, 93
notes, 1 open review row, before and after. The 10 pairs the fake compare used to call
"extends" for lack of a better word are now "different".

**Stress run** (`scripts/bench_write/noisy_extractor.py`). The fake extractor is perfect,
so it cannot show what happens to a cheap model's mistakes. This run damages its output the
same way every time and tags every fact. Three seeds, summed:

| | Before | After |
|---|---|---|
| Facts emitted | 252 | 252 |
| Became claims | 184 | 220 |
| Kept as a citation on an existing claim | 6 | 9 |
| Held for a person | 22 | 23 |
| **Lost** | **40** | **0** |
| Claims citing a locator the source does not have | 19 | 0 |
| Claims citing the wrong place | 5 | 0 |
| Stored quotes that are not the source's words | 10 | 0 |
| Invented quotes that became claims | 1 | 0 |
| Open review rows | 38 | 21 (18 more watching) |
| Extraction calls | 688 | 666 |

**Not measured.** No model key was available, so nothing here ran against a real model.
The 91% recall and zero invented facts from the earlier evaluation are extraction scores;
they do not see facts lost after extraction, and the bench's own `fuzzy_quote` verdict
still uses the whole-text scan from finding 3, so "zero invented" is optimistic on long
sources.

## For whoever owns the read path

- `/review` and `lint/checks.py` count `status = 'open'` only; `watching` rows are invisible.
- `review_queue.payload.facts` is new on unknown and ambiguous rows; `payload.fact` is new
  on `no_home_note` rows. `quote_unsupported` rows can now carry a second reason (a locator
  that is not in the source).
- `bench/extraction.py` scores `fuzzy_quote` with `_fuzzy_quote_containment`; the
  production gate is now stricter than the bench (`quote_gate.locate_quote`).
- `report/researcher.py`, `report/support_judge.py` and `bench/judge.py` pin
  `claude-sonnet-5` and `claude-haiku-4-5`. `pricing.py` lists Sonnet 5 at 3.00 / 15.00;
  `bench/agentic.py` lists it at 2.00 / 10.00.
- `tests/test_no_outside_material.py:127` fails `ruff check` (line length) on `main`.
