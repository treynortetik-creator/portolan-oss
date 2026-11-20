# Read path review

2026-11-20. Covers `portolan/search`, `report`, `lint`, `audit`, `bench` and the parts of
`registry` they use: how a question becomes cited notes, and how the graph checks itself.
The write path (ingest, extract, merge, consolidate) has its own review.

Each finding below says whether it was fixed in this pass. Numbers come from runs on a
laptop against local Postgres with pgvector; the commands are at the end.

## How retrieval works

A question goes through two searches at once, and their results are merged.

1. **Keyword leg.** The question is broken into word stems. Every note holding a rare one
   is a candidate; the best 200 are scored with BM25, which rewards rare words and
   discounts long notes. The top 30 or so go forward. This leg is exact: it finds names,
   ids and product codes, and finds nothing when the question uses different words from
   the note.
2. **Vector leg.** The question is turned into a 1,024-number embedding by a local model
   (`BAAI/bge-large-en-v1.5`). Notes are stored as chunks (the title, then sections of the
   body, at most 1,200 characters each), each with its own embedding. The 3k chunks
   nearest the question are fetched and reduced to their notes. This leg finds a note
   worded unlike the question, and always returns something.
3. **Fusion.** Each note scores `1 / (60 + rank)` in each leg it appears in, and the two
   are added (reciprocal rank fusion). Two things adjust the sum:
   - *Routing.* When one note holds 30% or more of the question's rare words, the vector
     leg's weight drops to 0.2: the question is naming something.
   - *Keyword share (new).* A note's keyword rank counts in full only if the note holds at
     least half of the question's rare words, and in proportion below that.
4. **Title boost, then recency.** A note whose title holds the question's words gets up to
   half the top score added. The total is multiplied by 0.85 to 1.0 according to how
   recently the note was written (half-life 180 days).
5. **Optional steps, both off by default.** A cross-encoder can re-read the top 30. `expand`
   follows links out of the top three hits. A separate module (`search/graph.py`) walks the
   entity graph for questions that need more than one note.

Each hit carries the chunk that matched best and the sources the note was built from. If
nothing comes back, a control query for the graph's most common title word tells "nothing
recorded" from "the index is broken".

## How answers stay grounded

The chat path sends the top notes to a model: each note's synthesis, its claims with their
ids, and the matched chunk. The model returns sentences, each tagged with the note slugs,
claim ids and verbatim quotes it rests on. Code then checks every tag against what was
actually sent. A sentence with no valid citation is dropped; so is one stating a number its
cited text does not state. Action cards go further and run a judge model on each sentence
against its evidence.

## How quality is checked

- **Lint** runs eleven checks in SQL over the graph: broken links, orphans, stale notes,
  bad headers, relative dates, citations that do not resolve, missing originals.
- **Audit** re-extracts a sample of sources with a stronger model, diffs the two fact sets,
  and has a judge rule on what the cheap pass missed.
- **Bench** scores extraction against a synthetic corpus whose planted facts are known, and
  scores search against question sets with known answer notes.

## Findings, ranked

### Fixed

**1. Rank-only fusion buried the vector leg's best hit.**
Fusion looked at rank, never at how well a note matched. A note that shared one stray word
with the question, and sat mid-list in both legs, outranked the note the vector leg put
first. On the paraphrase questions the vector leg ranked the right note first five times
and the fused result had it at 5th to 9th. Now a keyword rank counts by the share of the
question's rare words the note holds. This is the largest gain in the pass (table below).

**2. A third of all chunks were not content.**
Every rendered note was chunked with its YAML header ("status: living, schema_version...")
as one chunk and its "Related" link list as another. Both were embedded, both took places
among the nearest chunks, and either could be shown to the chat model as the "matched
passage" or read by the reranker as the note's first body text. They are no longer chunks:
450 chunks became 279 on the demo graph. Long claim lists now split between claims, never
inside one.

**3. A note that gained one claim re-embedded its whole body.**
The index is keyed on a hash of the body, and the body's header carries the day's date, so
any write re-embedded every chunk. On a CPU model that is most of a write's cost. Unchanged
chunks now keep their embeddings. The stored hash carries an index-format version, so
every note re-indexes once after this change with no migration.

**4. The reranker path ignored the recency setting.** It always read the claim date;
the unranked path reads the write date unless a setting says otherwise. Both now read the
same one.

**5. "Unmatched terms" reported words the graph did have.** The list an agent reads as
"nothing on file about this word" compared exact strings against the one chunk shown.
"renewals" was reported missing from a note that says "renewal", and so was any word in a
different chunk of a returned note. It now compares stems against the whole note.

**6. Lint raised 31 warnings on a clean 93-note graph.** `thin_links` counted only links out
of a note, so every product, competitor and contact that other notes point at was flagged.
It now counts links in both directions (18 remain, each a note with one neighbour).

**7. Lint findings nobody could clear.** A superseded or archived note with a past
`stale_after` was a permanent error. A human-edit "contradiction" kept warning after the
later claim was archived or superseded, which is how a person settles it. Both now look at
living notes and live claims only, and the contradiction message says what the check can
see ("may contradict": same note, same fact type, different text).

**8. Relative-date lint missed the common phrases and flagged quotes.** "last quarter",
"next month", "two weeks ago" and "this year" were not on the list; a speaker's quoted
"today" inside a claim was. Both fixed. It found one real case in the demo corpus.

**9. A citation marker could name a real source at a place no claim came from.** Lint only
checked that the source existed. It now warns when a note's `^[source @ locator]` marker
matches no claim, or corroborating citation, on that note.

**10. The audit judge saw one claim, chosen by the wrong measure.** To rule "genuine miss"
or "paraphrase", the judge was shown the single on-file claim closest by wording. A
paraphrase is by definition worded differently, so the right claim was often not the one
shown, and the miss rate read high. The judge now sees every claim on file for the source.
A "paraphrase" verdict naming a claim id it was not shown no longer removes a miss.

**11. The audit's "invention rate" was never judged.** It is the share of on-file claims
the second pass did not restate, matched by wording. The report now calls it "candidate
inventions (not judged)" and says so.

**12. Bench headline had no error bar.** The corpus plants 73 scoreable facts, so one fact is
1.4 points of recall. The report now prints a 95% interval (91% of 73 is 82% to 96%) and a
sentence on what "invented" counts.

**13. Two smaller ones.** Hits fetched their sources one query each (now one read for all).
The search module's own header said routing was off by default; it is on.

### Not fixed, and why

**A. A chat answer's citation is checked for existence, not for support.** *(Largest open
gap.)* The gate confirms the cited claim exists, was sent, and belongs to a cited note. It
does not check that the sentence says what the claim says. "The account plans to leave",
citing a real claim that says the account is happy, passes. Only numbers are compared.
Dates and years are exempt from even that, so a wrong renewal date passes. The typed
support judge that action cards already use (`report/support_judge.py`) would close this
for about one small-model call per answer. Not done here: it lives in the web layer, adds
latency and cost to every answer, and its false-drop rate needs the paid chat exam to
measure. Recommended as the next change.

**B. With a real embedding model, search never returns nothing.** The similarity floor
(0.05) was set for the test embedder. With bge-large, "What is the capital of France?"
scores 0.33 against this graph and returns ten hits; the lowest-scoring correct answer
across 59 questions scores 0.47. A floor near 0.45 would drop the plainly unrelated and
keep every correct answer seen, but the margin is 0.02 on a small sample and the number is
different for every embedding model. Needs a decision and a per-model setting.

**C. Recency reads the note's last write, not the age of what it says.** Any write to a
note resets it, so a note loaded today about a two-year-old call ranks as fresh. The
content-date option exists and is off because it cost one paraphrase question in an
earlier test. Worth re-testing now that fusion has changed.

**D. Graph expansion is not on the default search path.** On the 26 multi-hop questions,
`expand` mode answers 96% against plain search's 81%, but loses 6 points of recall on
single-note questions. Routing by question type would get both; that is a design choice.

**E. The chat context is 12,000 characters.** Notes that do not fit are skipped. Current
models take a hundred times that. Raising it is a one-line change with a cost trade-off.

**F. Model ids are a generation behind.** Defaults name `claude-haiku-4-5`,
`claude-sonnet-5` and `claude-opus-5` in `config.py`, `pricing.py`, `llm/` and in this
lane's `report/`, `bench/` and `audit/`. They were left alone: the price table and the
recorded bench runs are keyed on them, and they should move together.

**G. The audit still does not read the source.** The judge rules on a miss without seeing
the transcript, and candidate inventions are not judged at all. Both need the source text
in the prompt; with prompt caching that is now cheap per candidate.

**H. What the extraction bench's numbers mean.** Three limits, none of them a bug:
"invented" is decided by word overlap, so a claim that misattributes a real quote counts
as grounded; the bench hands the extractor the answer key's own list of which accounts and
contacts each source involves, which production has to work out; and the corpus prose was
written by the same model family being graded.

**I. The embedding model is from 2023.** Several newer 1,024-dimension models would fit the
schema unchanged. Swapping needs a run on a larger exam than the demo graph.

**J. The search questions are few.** 59 single-note and 26 multi-hop questions on a
93-note graph. Recall@10 is near its ceiling; hit@1 and MRR are the informative columns,
and one question is 4 to 7 points. The regression test in CI uses a word-overlap embedder,
so it pins the keyword leg and fusion, not the model.

### Tried and dropped

Each was measured on the two tuning sets with the production embedder and did not help.

| Idea | Paraphrase MRR@10 |
|---|---|
| Final configuration | 0.597 |
| Embed each chunk with its note title and heading in front | 0.544 |
| BGE's query instruction prefix | 0.339 (from 0.405, before other changes) |
| Fetch 6x or 10x chunks instead of 3x | 0.305 and 0.274 (from 0.405, before other changes) |
| Title boost off | 0.517 (demo and held-out rose; mixed, left alone) |
| RRF constant 10, 20, 40 | within one question of 60 |

## Bench numbers

### Search, demo graph, production embedder (bge-large), top 10

| Question set | n | Recall@10 | Hit@1 | MRR@10 |
|---|---:|---|---|---|
| Demo (tuning) | 20 | 100 → 100 | 85.0 → 95.0 | 0.897 → 0.963 |
| Paraphrase (tuning) | 15 | 73.3 → 80.0 | 26.7 → 53.3 | 0.405 → 0.597 |
| **Held-out (new)** | 24 | 91.7 → 95.8 | 58.3 → 79.2 | 0.721 → 0.851 |

The first two sets existed before and chose the fusion weights. The third was written
before any change was measured and chose nothing; its "before" column is the old code run
on the new questions. Old bench and new bench are therefore separate rows, and the
improvement shows on both.

By step, MRR@10 (demo / paraphrase / held-out):

| Configuration | Demo | Paraphrase | Held-out |
|---|---|---|---|
| Before | 0.897 | 0.405 | 0.721 |
| Keyword share only | 0.963 | 0.572 | 0.851 |
| Chunk cleanup only | 0.897 | 0.435 | 0.769 |
| Both (shipped) | 0.963 | 0.597 | 0.851 |

### Multi-hop questions (26, written earlier, not used for tuning), production embedder

| Mode | Answer supported | First-gold MRR |
|---|---|---|
| Plain search | 76.9 → 80.8 | 0.809 → 0.923 |
| Graph `expand` | 88.5 → 96.2 | 0.857 → 0.946 |
| Graph `entity` | 76.9 → 80.8 | 0.862 → 0.952 |

### Same sets, word-overlap test embedder (what CI runs)

| Question set | Recall@10 | MRR@10 |
|---|---|---|
| Demo | 100 → 100 | 0.839 → 0.963 |
| Paraphrase | 33.3 → 23.3 | 0.108 → 0.052 |
| Held-out | 87.5 → 87.5 | 0.630 → 0.707 |
| Multi-hop, answer supported | 80.8 → 92.3 | |

The paraphrase row falls here. This embedder matches literal words only, so on questions
written to share no words with their notes it has no signal, and the weak keyword matches
that used to surface a few are now discounted. It is the one regression in the pass and it
is confined to a test double.

### Other

| | Before | After |
|---|---|---|
| Chunks, demo graph | 450 | 279 |
| Lint warnings, clean demo graph | 31 | 20 (18 thin links, 2 relative dates) |
| Tests (with Postgres) | 4,614 passed, 9 skipped | 4,646 passed, 9 skipped |

The extraction bench (recall and invented facts) needs a paid model and was not re-run; no
change here touches extraction.

## Reproduce

```bash
createdb portolan_test_demo_search
uv run python scripts/bench/demo_search.py --embedder fastembed
uv run python scripts/bench/demo_search.py --embedder fastembed --reuse --questions paraphrase
uv run python scripts/bench/demo_search.py --embedder fastembed --reuse --questions heldout
uv run python scripts/bench/graph_retrieval.py --embedder fastembed   # multi-hop
```

Drop `--embedder fastembed` for the test embedder. The first run downloads the embedding
model (about 1.3 GB).
