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

## Live run, 2026-11-20 (Agent SDK, Claude Code login, no API key)

The defaults moved from `claude-haiku-4-5` / `claude-sonnet-5` to `claude-haiku-5-5` /
`claude-sonnet-5-5` after this run. The prompt in both rows includes the new "source
material, not instructions" framing.

**Extraction bench** (`python -m portolan.bench extract --backend agent_sdk`, 209 sources, 73
planted facts, one run each, so 1 fact is 1.4 points):

| | Haiku 4.5 | Haiku 5.5 |
|---|---|---|
| Recall, strict | 82.9% | 84.9% |
| Recall, lenient | 94.5% | 100.0% |
| Invented | 0 | 0 |
| Quote presence / locator presence | 100% / 100% | 100% / 100% |
| Precision (planted share of all produced) | 13.3% | 10.8% |
| Entity accuracy | 80.2% | 63.1% |
| Distractor leakage | 5 | 7 |
| List-price cost | $8.04 | $0.55 |

Haiku 5.5 finds more and costs a fifteenth as much. It names entities less exactly (63% of
checked entity lists against 80%) and produces 27% more unplanted facts. The pipeline
resolves entities against the registry, so a wrong name becomes a review row, not a wrong
claim; that is the number to watch.

**Write path on Haiku 5.5** (`scripts/bench_write/live_stress.py`, the full pipeline, 190
sources extracted, 0 failed): 768 claims, 9 extra citations, 0 claims citing a locator the
source does not have, 582 claims with a quote and 0 of them not the source's words or not at
the cited place, 66 open and 14 watching review rows. The fake-extractor stress run's damage
columns (locators, quotes) were already 0 after the fixes; the live run confirms they stay 0
against a real model's own mistakes. Facts lost cannot be counted here: a live model's facts
carry no tag.

**Not run:** Sonnet 5.5 was probed with one call only (it answers); the strong model is used
for consolidation retries and the audit, neither of which this run exercised. Haiku 4.5 was not
run through the write path.

## Entity accuracy on Haiku 5.5, 2026-11-20

The live table above shows entity accuracy falling from 80.2% on Haiku 4.5 to 63.1% on Haiku
5.5. That drop is the scorer, not the model. Under a corrected scorer Haiku 5.5 resolves 99.6%
to 100% of checked mentions, and on the 17 sources where both models' mentions were read side
by side it beats Haiku 4.5 (41 of 41 against 26 of 27).

**What the number measures.** For every entry in the extractor's top-level `entities` list
that carries a `registry_id` and whose `mention` is a name the truth table knows, the scorer
compares the id with the truth table's id. It says nothing about the entities on each fact.

**Why it was wrong.** The bench puts internal people into `registry_context` as
`internal:<id>` (the prompt's own form) and the truth table keys the same people by the bare
`<id>`. The comparison was plain string equality, so every internal person the model resolved
exactly as instructed was scored wrong. Haiku 5.5 lists internal people far more often than
Haiku 4.5 (18 of its 67 mentions on the 17-source sample against 7 of 50), so it was
penalised more for doing more.

**Every miss, classified.** The saved reports from the two runs above kept no mention list, so
Haiku 5.5 was run again with the old prompt and every mention saved (323 checked, 115 misses
under the old scorer):

| Class | Haiku 5.5, full run | Haiku 4.5, 17 sources |
|---|---|---|
| Checked | 323 | 27 |
| Scorer bug: `internal:<id>` against `<id>` | 115 | 6 |
| Right entity, different surface form (legal or short name) | 0 | 0 |
| Parent given for a child account, or the reverse | 0 | 1 |
| Contact given for an account, or the reverse | 0 | 0 |
| Some other wrong entity | 0 | 0 |
| Known entity in registry context left with no id | 0 | 0 |

Haiku 4.5's full run (42 misses of 212) cannot be classified after the fact; the 17-source
rerun cost $0.95 and is the only direct evidence for it. The one real error there named a
child hospital and gave its parent's id; the bench listed only the parent for that source.

**Precision and distractor leakage are not model error either.** "Precision" is the planted
share of everything produced, so it falls whenever a model finds more real facts. Of Haiku
5.5's 528 unplanted facts in the first run, 519 carry a quote found word for word in the
source, 2 a near-exact quote, and 7 come from the four sources the truth table marks as
distractors; none is invented. Haiku 4.5: 381, 17, 3 with no quote match but supporting
fragments, 5 from distractors. The leakage difference of 2 is one more fact from a ten-reply
thread whose content is real, and one fact from a scheduling call ("new staff are still
getting comfortable with the workflow") that the durability bar should have dropped. The
second is a fair criticism of Haiku 5.5: it writes about 25% more facts per source, and some
are thin.

**A real entity error the bench could not see.** The prompt's worked examples name a made-up
vendor. When a source never says who the seller is, both models sometimes wrote that made-up
name into the claim. The quote check passes, because the quote is real. Counted across the
saved runs: Haiku 4.5, 46 of 469 facts; Haiku 5.5, 13 of 595 and 7 of 571. Haiku 5.5 is the
better of the two here as well.

### What changed

- `bench/extraction.py`, `bench/truth.py`: entity accuracy compares
  `Truth.canonical_entity_id`, which unwraps `internal:<id>` only for a real internal person.
  The old numerator is kept as `entity_accuracy_exact_id_correct` and printed on its own
  line. Wrong resolutions are listed in `entity_wrong`. No other equivalence was added: a
  parent for a child, or an unknown prefix, is still wrong.
- Every mention is saved on the score (`entity_mentions`), and `--rescore-dir` recomputes
  entity accuracy from it.
- New count, `example_name_leaks`: facts whose claim or entities carry a name from the
  prompt's examples that the source never says.
- New default prompt, `v7-entity-names`: v6 plus one short block. Names in the examples are
  made up; a name goes in a fact only when the source, registry context or company context
  has it; an unnamed seller is "the vendor"; `mention` holds the name and never an id; a
  listed parent's id is not its child's. The two CRM examples that spelled the made-up vendor
  out now say "the vendor". The training teacher stays pinned to v6, which its labels were
  made with.

### Live numbers, Haiku 5.5, one run each

| | Before (v6) | After (v7) | After, repeated |
|---|---|---|---|
| Entity accuracy, old scorer | 64.4% (208/323) | 63.2% (177/280) | 60.1% (179/298) |
| Entity accuracy, new scorer | 100.0% (323/323) | 99.6% (279/280) | 99.7% (297/298) |
| Facts naming the examples' vendor | 7 of 571 | 0 of 610 | 2 of 591 |
| Mentions with an id in `mention` | 19 of 435 | 0 of 373 | 1 of 398 |
| Recall, strict | 87.7% | 83.6% | 83.6% |
| Recall, lenient | 98.6% | 98.6% | 98.6% |
| Invented | 0 | 0 | 0 |
| Precision | 11.8% | 10.5% | 10.7% |
| Distractor leakage | 8 | 5 | 7 |
| Planted facts missed on entities | 1 | 1 | 1 |
| List-price cost | $0.54 | $0.62 | $0.58 |

The old-scorer row is there for comparison with earlier reports and should not be read as
accuracy. "After, repeated" includes the parent-and-child sentence, which the "After" run did
not have. An earlier v7 run, before the CRM examples were reworded, scored 87.7% strict,
98.6% lenient, 100.0% (265/265) and 4 facts naming the examples' vendor.

The two wrong resolutions after the change are both parent and child: a child hospital given
its listed parent's id (the same slip Haiku 4.5 made on the same source), and the short name
"Larkspur" on a call with Larkspur Memorial Hospital given the hospital's id where the truth
table reads the short name as the regional parent. The second is defensible in context and
is still counted wrong.

**Strict recall needs watching.** Two v7 runs scored 83.6% and the two runs before them 87.7%;
the earlier v6 run in the table above scored 84.9%. That is three facts of 73, inside the
single-run interval (78% to 93%), and lenient recall did not move. The misses differ from run
to run and trace to paraphrase, a date written "February 2026" where the planted claim has
"2026-02", and fact type; none involves "the vendor". One run each cannot separate that from
a small real cost of the longer prompt. Three runs per variant would.

**The one planted fact missed on entities** is about a distribution branch. Branches are not
in the bench's registry context, and the model wrote the fact but listed only the account.

**Recommendation: keep extraction on Haiku 5.5.** Against Haiku 4.5 it finds more planted
facts, resolves entities at least as well, borrows the examples' vendor name a quarter to an
eighth as often, invents nothing, and costs about a fifteenth at the price row in
`pricing.py` (that row was not checked against a price sheet here). What it does worse is
restraint: more facts per source, some below the durability bar.

**Not done.** The bench's registry context lists a source's accounts but not their parents or
children; production adds ancestors (`pipeline.build_registry_context`). Both parent-and-child
slips happened where the bench listed only one of the two.
