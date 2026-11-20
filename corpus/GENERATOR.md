# Corpus generator

Turns `corpus/truth-table.yaml` (the answer key) into a synthetic corpus under
`corpus/out/` plus `corpus/out/manifest.json`, which records the exact file
and locator where every planted fact landed. The extraction bench
(`01-spec/quality-loop.md`) scores a model's output against that manifest.

## Run it

```bash
uv run --with pyyaml --with anthropic python -m corpus.generator.cli generate --out corpus/out --mode auto
```

`--mode auto` (default) detects credentials and picks `api` or `manual`.
Force one explicitly with `--mode api` or `--mode manual`.

Run the tests:

```bash
uv run --with pytest --with pyyaml --with anthropic pytest tests/test_corpus_generator.py
```

## Package layout

```
corpus/generator/__init__.py
corpus/generator/briefs.py       truth table -> one Brief per source (facts, positions, phrasing, forbidden words)
corpus/generator/structured.py   CRM CSVs, RSS JSON, email headers, ticket skeletons, locator-format helpers
corpus/generator/prose.py        brief -> text: API mode, staged hand-written content, or procedural filler
corpus/generator/markers.py      finds ⟦F01⟧...⟦/F01⟧, records locators, verifies, strips
corpus/generator/manifest.py     manifest schema, writer, validate()
corpus/generator/cli.py          python -m corpus.generator.cli generate
corpus/generator/_staging/       hand-written raw source text, markers still in place (see below)
```

## Mode detection

`prose.detect_mode()` checks, in order: `ant auth status` (an active profile
means zero-arg `anthropic.Anthropic()` works), then `ANTHROPIC_API_KEY` in
the environment, then in `.env` at the project root. Neither was present
when this corpus was generated (2026-09-12), so the run used **manual
mode**. No key is ever printed.

## Manual mode: how the prose actually got written

Manual mode does not mean "template-generated." The build instructions are
explicit that a human (here, the orchestrating Claude Code session) writes
the fact-bearing prose by hand, following each brief's facts, positions,
phrasing constraints and forbidden words, with the same marker discipline
API mode would use. That's what happened, in three tiers by cost/benefit:

1. **The 10 fact-bearing call transcripts** (`S-call-003/004/005/006/007/
   008/009/011/013/014`) were each hand-written to spec — participants,
   date, exact facts at exact positions with exact phrasing — by the
   orchestrating session, delegated across five parallel sub-instances of
   itself (still Claude, not a template) to write in parallel rather than
   serially. Every fact's marker, position window, and forbidden-word list
   came from `briefs.py`; each transcript was written directly into
   `corpus/generator/_staging/calls/<id>.txt` with markers in place.
2. **The 5 fact-bearing tickets and 9 fact-bearing emails**, plus the two
   distractors that need real content (D02's out-of-office, D03's ten-reply
   quoted thread), were hand-written directly into `_staging/tickets/` and
   `_staging/email/` the same way — these are short enough (per the reduced
   manual-mode lengths below) that delegating them added more overhead than
   it saved.
3. **Noise sources** (extra calls/tickets/emails with zero planted facts)
   and **RSS items** (always fully structured, never markered — see below)
   use deterministic, seeded-random procedural generation in `prose.py`
   (`generate_filler_call` / `generate_filler_ticket` /
   `generate_filler_email_body`) and inline data in `cli.py`. Nothing here
   needs the marker/phrasing precision a planted fact needs, and the coding
   standard explicitly calls for seeded-random determinism over
   hand-writing for noise.

`cli.py` decides per source: if a staged file exists at
`_staging/<type>/<source_id>.{txt,json}`, use it; otherwise fall back to
procedural filler. This means re-running the CLI is deterministic and
reproducible without needing the staged files to still exist for noise.

### Reduced lengths (manual mode)

Per the build instructions, manual mode uses shorter targets than the full
spec so a human/delegated writer can hit them:

| Source | Full spec | Manual/reduced |
|---|---|---|
| Planted calls | 12k-24k tokens | 6,000-9,000 tokens |
| Noise calls | 6 @ 10k-15k tokens | 2 @ 3,000 tokens |
| Noise tickets | 40 | 15 |
| Noise emails | 60 | 20 |
| Noise RSS | 20 | 8 |
| Tickets / emails (planted) | 200-1,500 / 200-800 tokens | unchanged (already short) |

`manifest.json` records `"mode": "manual", "reduced": true`.

### RSS is never "prose" in this generator

RSS items are short third-person blurbs, not participant dialogue, and the
build instructions list RSS under "Structured outputs (no model needed)."
All RSS content — planted (F18, F29) and distractor (D01) — is written
directly as short data literals in `cli.py::process_rss`, with no marker
pass; a planted RSS fact's locator is the item's own `url:<link> <date>`,
since there is no sub-document structure to point into.

## Locator formats

| Source type | Format | Example |
|---|---|---|
| call | `HH:MM:SS`, the timestamp of the line carrying the marker | `00:14:32` |
| ticket | `<ticket-id>#<comment-number>` | `S-ticket-019#2` |
| email | `<message-id-without-brackets>#p<paragraph-index>` (1-based, blank-line-separated paragraphs) | `S-email-052@halvardmedical.com#p3` |
| rss | `url:<link> <date>` | `url:https://feeds.example-newswire.test/portolan-corpus/s-rss-004 2026-08-26` |
| crm | `<relative-path-under-corpus/out>#<crm-or-opportunity-id>` | `crm/accounts.csv#A-1010` |

Every email's `Message-ID` uses the `@halvardmedical.com` domain regardless
of who sent it (Halvard's system is what would have logged/archived it),
per the exact header format given in the build instructions:
`Message-ID: <S-email-052@halvardmedical.com>`.

`manifest.py::LOCATOR_PATTERNS` enforces these formats mechanically in
`validate()`.

## Manifest shape

```json
{
  "mode": "manual",
  "reduced": true,
  "truth_table_hash": "sha256 of corpus/truth-table.yaml",
  "sources": [
    {
      "id": "S-call-007",
      "type": "call",
      "path": "calls/S-call-007.txt",
      "facts": [{"fact_id": "F01", "locator": "00:xx:xx", "line_or_comment": "42", "excerpt": "..."}]
    }
  ],
  "facts": {"F01": [{"source_id": "S-call-007", "locator": "...", "line_or_comment": "...", "excerpt": "..."}]}
}
```

`line_or_comment` is a generator-internal pointer distinct from the
extraction-schema `locator` string: the raw line number in the clean call
file, the ticket comment number, or the email paragraph index — useful for
debugging a placement without re-deriving it from the locator string.

`validate(manifest, truth_table)` checks: every planted (non-distractor)
fact's placements match the truth table's `sources` list for that fact
exactly (same source ids, no more, no fewer); every distractor has zero
placements; every locator matches its source type's format.

## Phrasing constraints I could not fully honor

**F04 / S-call-011** — the truth table asks for "two separate mentions
twenty minutes apart in the same call," both tagged `position: middle`
(35%-65% of the transcript). A call within the reduced manual-mode band
(capped at 9,000 tokens ≈ 45 minutes of dialogue) has a middle window only
~13.5 minutes wide — not enough room for a 20-minute gap without one
mention spilling into "early" or "late." I pinned this call to the top of
the reduced band (9,000 tokens, the longest of the ten) and relaxed the
constraint to "as much separation as the middle window allows" rather than
a literal 20 minutes, keeping both mentions inside `position: middle` since
that is the constraint the bench's `two_clocks_supersession`-adjacent
`within_source_dedup` test actually exercises (`tests: [theme_week3,
within_source_dedup]`). This is documented in `briefs.py` next to
`MANUAL_LENGTH_OVERRIDE_TOKENS`.

Every other fact's phrasing and forbidden-word constraint was honored as
written (verified by grep across the finished `corpus/out/` files — see the
generation report for the exact check run).
