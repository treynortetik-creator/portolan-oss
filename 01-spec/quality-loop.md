# The quality loop: catching what extraction missed

2026-09-12, orchestrator session, in answer to the maintainer's second-pass idea, the keyword idea, and the consolidation question.

## The problem, stated honestly

Aggregating this much text into notes loses detail. That is not fixable, it is manageable. The goal is to know the miss rate, know which kinds of facts get missed, and drive it down with prompt and model changes we can measure.

## Two instruments

### 1. The bench (controlled, before customer data)
The synthetic corpus is generated from a truth table, so every planted fact is known. Run a model and prompt over the 200 sources, score:
- **recall**: planted facts that came out as facts
- **precision**: facts that came out but were not planted (invented or over-split)
- **entity accuracy**: registry ids right
- **locator accuracy**: locator lands within the planted span

Every schema version, every prompt change, every candidate model runs the bench. This is where "which model does best" and "how do we prompt better" get answered, cheaply, in minutes, on a laptop if we like.

### 2. The production audit (continuous, on real data)
the maintainer's second-pass idea, with one change. A second pass by the *same* model measures variance (what it happened to miss this time), not systematic misses. So the second pass uses a **stronger model on a sample**:

- each week, sample one to two percent of sources ingested that week, stratified by source type
- re-extract with Sonnet or Opus, full document, same schema
- **diff the two fact sets programmatically**: a fact matches if type matches, entities overlap, and locators are within a window. Claim text similarity breaks ties. Unmatched facts from the strong pass are candidate misses; unmatched facts from the cheap pass are candidate inventions
- a judge call grades only the unmatched remainder (genuine miss, paraphrase of a matched fact, or not a fact)
- report: miss rate by fact type and source type, invention rate, and the ten worst sources with links

That report is programmatic. Facts are structured, so set difference is code. The judge sees only the leftovers. The whole audit costs a couple of percent of the extraction bill.

When the miss rate for a type climbs, the fix is one of: a better definition or example in the checklist, a per-customer vocabulary entry, a segment-size change, or moving that source type up a model tier. The bench confirms the fix before it ships.

## The keyword idea: where it helps, where it does not

**As a pre-filter, yes.** Registry names, product names, and a short list of signal words decide whether an internal-only document is worth extracting at all. Cheap, already in the customer-targeting design.

**As a guide to what to extract, no.** The facts that matter are usually phrased without the keyword. "We've been looking at some other options" is a competitor mention with no competitor word in it. A big keyword list has high recall on the obvious and none on the paraphrased, and the paraphrased is exactly what the model is for. Large lists also fire on noise and make the pre-filter worse.

**The upgraded version of the same instinct** is already in the schema: a closed checklist of signal types, each with a definition, an example and a counter-example, in the extraction prompt. That makes the model look for each kind explicitly, including paraphrases. Plus the per-customer vocabulary, which is where a word list genuinely earns its place: proper nouns and jargon the model would not otherwise know.

**Does the second pass read the whole document?** Yes, on the sample. You cannot find a miss without reading everything the first pass read.

## Segmented extraction

Long transcripts lose details in the middle when read in one call. Extract in roughly 4k-token segments with a running entity list, merge per source. Same token cost, better recall. This is the single most likely lever for transcript miss rates and it needs no model change. In the schema.

## The consolidation pass

Notes updated many times by a cheap model fill with hedges and near-duplicate sentences. the maintainer's read is right: the rewrite is mostly summarisation and a cheap model can draft it. The risk is dropping a claim or a citation. So:

- **Trigger:** a note has taken more than N updates since its last consolidation, or its claim list passes M entries.
- **Draft:** a cheap model rewrites the synthesis (the top of the note) from the claim list. It is told: every claim id must appear, every citation marker must survive verbatim, no new claims.
- **Check, mechanically:** compare the set of citation markers and claim ids before and after. Any missing means reject.
- **Escalate:** on rejection, retry once with Sonnet. On a second rejection, flag for a human.
- **Archive, as the maintainer described:** when the claim list passes the growth threshold, the older entries move to `<slug>-archive.md`, linked from the note. The archive is excluded from default retrieval and included when a query asks for history. The schema's dossier pattern already has this shape and the lint's growth guard already warns at 40 entries.

## Human edits

the maintainer wants humans to fix the graph by hand, attributed. Cleanest model: **a human edit is a source.** It produces an edit record (note, before, after, who, when, reason) and the changed claim carries a citation of the form `^[edit:maintainer 2026-09-12]`, which the viewer renders with the editor's name. The rebuild reapplies edit records on top of regenerated notes, so a human fix survives a schema change. The lint flags a human-edited claim that a later extraction contradicts, so the human is told when reality moved.

## Dig deeper at query time

Notes-first stays the rule, but when a user wants the detail a note left out, the chat can offer a bounded read: open the cited original around the locator, answer from that passage, cite the raw document with its locator. User-initiated, scoped to one source, always cited. It answers "summaries miss details" without turning the product back into query-time RAG over everything.
