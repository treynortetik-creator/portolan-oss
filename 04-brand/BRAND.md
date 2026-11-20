# PORTOLAN

**Brand identity v1 · 2026-09-12 · for the engine working-titled "Living Knowledge Graph"**

---

## The name

**Portolan** (POR-tuh-lan)

## Tagline

**A living chart of what your customers actually said.**

## Why this name

Before there were surveys, there were portolans. A portolan chart was the working map of a coastline, and it was not drawn from theory — it was compiled from what pilots actually reported when they came back: this harbor takes a deep-draft hull, that headland has a shoal a mile out, the factor at this port pays late. Every line on it traced to a voyage. New reports were laid over old ones and the chart was corrected, so it stayed true as the coast and the commerce changed. And the good ones never left the house that owned them — Portugal treated its charts as state secrets, because the chart *was* the advantage. Nodes, edges drawn between them, a boundary around the whole thing, corrected on a cadence, sourced to the person who saw it. That is this product, four hundred years early.

The name carries the four things the engine actually promises. **Provenance:** a portolan is worthless without the reports behind it, the way a note here is worthless without its `source:` line. **Memory that stays true:** a chart carries the date of its last correction, which is exactly what an `updated:` stamp is — a promise that someone looked on that day and it still held. **A graph that stays alive:** the rhumb lines radiating between ports are literally nodes and edges, and a chart nobody corrects is a chart that runs you aground. **Nothing leaves the building:** the boundary in the mark is the point — the corpus, the index and the embeddings all stay inside the customer's walls. There is a bonus in the first market: in a hospital, *the chart* is already the word for the record you trust because of who wrote in it and when. Medtech customers will hear that before we explain it.

## Palette — six tokens

Chart paper, iron-gall ink, and the two colors a portolan actually used for its wind lines: madder red and a copper half-wind green. No teal. No coral. Full tokens with comments in `brand-tokens.css`.

| Token | Role | Light | Dark |
|---|---|---|---|
| `--pl-ground` | page / panel, warm paper never pure white | `#FAF8F4` | `#0F1214` |
| `--pl-ink` | body text, wordmark, anything that must be read | `#16191D` | `#EDE8E0` |
| `--pl-accent` | the one accent: links, active node, the number that matters | `#C0362C` | `#D8543A` |
| `--pl-secondary` | verified / fresh / corroborated | `#4A6B3C` | `#93BE7E` |
| `--pl-muted` | dates, source lines, captions | `#6B6A63` | `#9A958B` |
| `--pl-line` | hairlines, table rules, borders — structure, never text | `#E2DCD2` | `#262B2E` |

Every text token clears 4.5:1 against its ground in both modes, `--muted` included, because dates and source lines are the whole product and must never be decorative grey.

## Type — three stacks, no web fonts

| Stack | Use | Family |
|---|---|---|
| `--pl-display` | headings, labels, nav, the wordmark's voice | `"Avenir Next", Avenir, Futura, "Century Gothic", "Segoe UI", system-ui, sans-serif` |
| `--pl-body` | anything longer than a label | `"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, "Times New Roman", serif` |
| `--pl-mono` | citations, note ids, paths, queries, dates in tables | `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace` |

Geometric caps over a warm serif: a chart legend sitting on a chart-room document. Every family in every stack ships on macOS, iOS, Windows or Android, so a PDF or an email attachment renders on the recipient's machine with nothing to download. Uppercase display gets `letter-spacing: .06em`. Em-dashes capped at two per page.

## The wordmark

`wordmark.svg` — one file, 355×64, every shape `fill="currentColor"`. The letters are drawn as paths, not set in a font, so the lockup is identical on every machine and in every PDF.

**How to place it:** the wordmark has no color of its own. It takes the `color` of whatever contains it.

```html
<!-- light: paper ground, ink mark -->
<div style="background:var(--pl-ground); color:var(--pl-ink)">…wordmark…</div>
<!-- dark: same file, nothing swapped -->
<div data-theme="dark" style="background:var(--pl-ground); color:var(--pl-ink)">…wordmark…</div>
```

- Set `color: var(--pl-ink)` in almost all cases. `--pl-accent` is allowed for a single hero placement, never for body chrome.
- Clear space: the ring's radius on all four sides. Minimum lockup width 120px; below that use the mark alone.
- The mark — one node, four edges, four nodes, one horizon — holds down to **16px** and is verified at 24px. Crop the SVG's viewBox to `0 0 64 64` to use the mark by itself as a favicon or app icon.
- Never re-color the mark and the letters differently, never outline it, never add a gradient, never set the name in a font as a substitute.

## Voice — three rules

**1. Cite or cut.** Every claim we publish carries its source and the date it was last checked, in the sentence, the same contract the product makes. "1,288 notes and 28,129 links, measured 2026-09-11" — not "a massive knowledge base." If a number has no source we can name, it does not ship, including in a deck.

**2. Name the mechanism, not the magic.** "A scheduled re-crawl compares file modification times and re-reads only what changed" — not "AI-powered continuous intelligence." Banned outright: seamless, unlock, transform, revolutionary, effortless, leverage, supercharge, game-changing. A buyer in this market has been sold vapor before and is listening for it.

**3. Say the edges out loud.** What it doesn't do yet, what it can't know, what a number actually measures. Role-based access is a later problem — we write that sentence ourselves rather than let a compliance officer find it. Understating costs a little credibility once; overstating costs the deal.
