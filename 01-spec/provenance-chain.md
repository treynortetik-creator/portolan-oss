# The provenance chain

Draft 2026-09-12. the maintainer's rule: **everything traces back to a phone call, an email, or the original file.** This note makes that rule concrete enough to build.

## Four links, every one clickable

```
answer  →  note  →  source note  →  original document, at the exact spot
```

1. **Answer to note.** Every sentence the chat produces cites the note it came from. Not a footnote list at the end: the citation sits on the sentence. A sentence with no note behind it is not shown.
2. **Note to source note.** Every claim inside a note carries a locator to a source note. A note that has been updated from three sources carries three, on the claims they support.
3. **Source note to original.** The source note records where the raw document lives (S3 key), when it was ingested, its content hash, and which notes were extracted from it.
4. **Original, at the spot.** The viewer opens the raw document and jumps to the locator: a timestamp in a transcript, a message id in a thread, a page and paragraph in a PDF, a row in a CRM export.

The current schema covers links 2 and 3 at the *note* level (`source:` in frontmatter). The rule requires them at the *claim* level. That is the change.

## Claim-level citation format

Inline, in the body, immediately after the claim:

```
Ashgrove Health's Springfield facility reported three late deliveries in August 2026 ^[ashgrove-qbr-2026-08-14 @ 00:14:32].
```

- `^[...]` is the citation marker. It is plain text, readable in any editor, and matched by one regex.
- The first part is the source note's slug.
- After `@` is the **locator**, whose format depends on the source type:

### Locator formats, settled 2026-09-12

The draft table below (the one this note originally shipped with) was never checked against a real corpus, and drifted from what the code actually produced (review `01-spec/reviews/review-wave-a-2026-09-12.md`, finding F8): tickets carried an invented `T-` prefix that didn't match any real ticket id, and email used a `msg:...¶n` shape found nowhere else. This replacement table is read directly off the first planted locator of each source type in `corpus/out/manifest.json` — the fix wave's answer key — so it is the format the code, the bench, and this doc now all agree on. `portolan.ingest.prepare_text` (`portolan/ingest/prepare.py`) is the one place these are produced; `portolan.ingest.locators.LOCATOR_RE`/`parse_locator` is the one place the `[...]` bracket wrapper around them is parsed back out of prepared text.

| Source type | Locator | Example (from `corpus/out/manifest.json`) |
|---|---|---|
| Call transcript | `HH:MM:SS` timestamp | `@ 00:04:15` |
| Email | message id (no `<>`, no `msg:` prefix) + `#p` + 1-indexed paragraph number, counting every paragraph of the cleaned body from the top | `@ S-email-005@halvardmedical.com#p2` |
| Ticket | the ticket id exactly as given (no invented prefix) + `#` + the comment's own `number` | `@ S-ticket-027#1` |
| PDF / Drive doc | page and paragraph (`p<page>¶<para>`) when the source carries page markers, else `L<line>` — unchanged; no doc source exists in the current corpus to verify against, so this is the pre-existing convention, not newly settled | `@ p7¶2` |
| Spreadsheet / CRM export | the CSV's path relative to the corpus root + `#` + the row's id (`crm_id` for an account/contact row, `opportunity_id` for an opportunity row) | `@ crm/opportunities.csv#O-3001` |
| Chat (Slack) | channel and message timestamp — unchanged, not exercised by this corpus | `@ #ops:1755180000.123` |
| RSS / web | `url:` + the item's link + a space + its date | `@ url:https://feeds.example-newswire.test/portolan-corpus/s-rss-004 2026-08-26` |

Frontmatter `source:` becomes a list and stays as the note-level summary. The lint checks that every citation marker resolves to a source note and every source note resolves to a stored original.

### Source versioning (review finding F6)

A source whose *content* changes under the same `source_id` (a ticket that gains a comment, an edited doc — exactly what an incremental re-crawl produces) is not the same source and is not idempotent-by-hash, but it also can't be inserted under the unchanged `source_id`: `sources.slug` is the primary key. `portolan.ingest.intake` versions it instead: the first ingest keeps the plain `source_id` as its slug; each later one that changes the content gets `<source_id>--v<n>`, and its `sources.metadata.supersedes_source` records the immediately prior slug in that chain (`<source_id>` for `--v2`, `<source_id>--v2` for `--v3`, and so on) — a citation to an older version is still resolvable, and the chain is walkable one hop at a time from any point in it.

### A prepared source that couldn't be cleanly stripped (review finding F1)

Quote/boilerplate stripping (email only, today) never returns empty text for non-empty input. When stripping would have left nothing, `prepare_text` falls back to the unstripped original and sets `PreparedText.stripping_skipped = True` (`removed_chars` stays `0` in that case) rather than handing extraction an empty envelope. This surfaces in the pipeline as ordinary prepared text — every paragraph still gets a locator — just with whatever quoting the original had left in place for the model to read past.

## What the extractor has to output

The extraction schema gains a required field per fact: `locator`. The model reads a transcript with timestamps (or line numbers we inject) and returns, for each fact, where in the source it found it. A fact with no locator is rejected at merge time. This is cheap for the model and it is the whole mechanism: provenance is captured at extraction, never reconstructed later.

For sources without natural locators (a plain text file), the connector injects line numbers before extraction and strips them after.

## What the chat UI has to render

- Each cited sentence is a link. Hover shows the note title and `updated:` date. Click opens the note page.
- The note page renders the body with citation markers as links. Click opens the source note.
- The source note page shows the metadata and a **"open original"** button that loads the raw document in a viewer and scrolls or seeks to the locator.
- Raw documents are served by signed URL from S3, behind the org login, with the access logged. This is the one place PHI is visible in the product.

## Two things this rules out

- **Answers synthesised from raw documents at query time.** Every answer comes from notes. If the notes cannot support the answer, the chat says so and offers the nearest notes, rather than reading raw files and inventing a citation.
- **Notes without locators.** A note migrated from an older corpus without claim-level citations is marked `provenance: note-level` in frontmatter and shown with a visible flag until re-extracted.

## Order of work

1. Add `locator` to the extraction schema and the merge validator.
2. Connectors inject or preserve locators (timestamps for transcripts, message ids for mail, page/paragraph for PDFs).
3. Source notes store the S3 key and content hash.
4. Lint: unresolved citation markers, source notes with no stored original.
5. The viewer: note page, source page, raw-document viewer with seek-to-locator.
6. The chat: sentence-level citations rendered as links to step 5.
