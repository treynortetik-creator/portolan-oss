# Surfacing ideas: how the weekly report writes itself

Draft 2026-09-12, orchestrator session, in answer to the maintainer's question: "we would have a giant corpus with everything needed to surface ideas, but no way to do that effectively."

## The principle

Never ask a model to "look at everything and find something interesting." It is expensive, it is not repeatable, and it produces different answers every week. Instead, the report is assembled from four mechanical feeds, each cheap, each citable, and the model's job is to write the prose on top of them.

## The four feeds

### 1. The diff: what changed in the last seven days
Every note has `created:` and `updated:`. The changelog has one row per ingest. So the week's activity is a query, not a judgement:
- notes created this week, grouped by type and account
- notes updated this week, with what changed (git diff of the note)
- facts superseded this week (old note now points at a new one)
- accounts with no new signal in 30, 60, 90 days (the silence is a signal)

This is the section a reader trusts most because nothing in it is inferred. Virgil's morning brief already works this way.

### 2. Theme notes: the rollup layer
A `theme` is a node like any other. It links down to many `signal` notes and up to the accounts, products and revenue they touch. "Late deliveries, Midwest hospital accounts" is a theme note; the 14 signals from 14 accounts are its children; the ARR comes from following each signal to its account to the CRM figure.

Themes are maintained by a scheduled pass, same rule as ingest: update, don't duplicate.
- embed every signal created or updated this week
- for each, find the nearest existing themes by cosine similarity plus shared entity links
- if one matches, link the signal in and refresh the theme's counts and synthesis
- if nothing matches above the threshold, it goes to the outlier queue (feed 4)
- once a quarter, or on demand, re-cluster everything to catch themes that drifted

Each theme note carries: signal count, account count, sum of linked ARR, first seen, last seen, trend (this week against the trailing four), and the three most representative quotes with sources.

The report's core section is **themes by movement**: new this week, rising, falling, resolution proposed, resolved. That is the target sentence shape ("late deliveries, 14 accounts, $8.2M, two branches") produced from the graph, with every number traceable.

**How a theme reads "resolved."** As of 2026-09-14 (decisions log, "withdraw content-based automatic resolution"), only two things can make a theme read resolved, and one of them still needs a human:

- **Quiet weeks.** A theme with no new signal for three weeks (`THEME_RESOLVED_AFTER_WEEKS`) reads resolved on its own, recomputed live every time the report runs — no model asked, nothing for a human to confirm, and it also gets written down (`theme_resolutions`, `status='confirmed'`, `resolution_claim_id NULL`, `reason='silence'`) so it's a citable fact rather than only a live computation. If a new complaint shows up later the theme reopens and goes back to "rising."
- **A judge-proposed claim, confirmed by a human.** Automatic resolution BY CONTENT — guessing from shared word stems — is withdrawn. Four build waves tried to make that gate safe (distinctiveness, a theme-only matching vocabulary, per-stem corroboration across members) and the fifth review reproduced the same false positive on a parent/child account pair the mechanical checks alone admit as one customer relationship: a claim from a child account resolved an unrelated theme on the parent's own account, because two different stories happened to share two ordinary words. The mechanism that decides content now is a model, not a heuristic — but only ever as a *proposal*:
  1. **Mechanical prefilter, no content judgement.** A later praise or decision claim from the theme's own member account (or its registry parent/child), dated on or after the theme's own most recent complaint, matching the theme's own product family when it has one, and not already judged against this theme before.
  2. **One judge call** (the cheap model, `COMPARE_MODEL`) reads the theme's own member claims next to the candidate and answers whether the candidate genuinely says THIS theme's problem is over, quoting its exact evidence. A verdict whose quoted evidence isn't a real, verbatim substring of the candidate is treated as `false` — a model cannot invent an unbacked quote its way into a resolution.
  3. A `true` verdict writes a `theme_resolutions` row with `status='proposed'` and drops a queue row for a human, citing the evidence. The theme's movement becomes **"resolution proposed"** — not resolved. One candidate claim can propose several themes at once; that's fine, since a human reviews each.
  4. A human **confirms** it (`status='confirmed'`, the theme now reads resolved) or **rejects** it (`status='rejected'`, the theme's movement falls back to whatever it would have computed anyway). Nothing is deleted either way — the row is kept as history.

If real activity resumes on an already-resolved theme (confirmed OR still an unconfirmed proposal), it reopens: `THEME_REOPEN_MIN_MEMBERS` new members after the resolution's own week flips a confirmed row's `reopened_week` (its own status is untouched — it really was confirmed once) or, for a still-open proposal nobody acted on, marks it `superseded` instead (the proposal is moot; the story visibly isn't over).

**Waves: three accounts, same week, no reading required.** A theme needs the model to agree two claims are "the same story." A wave doesn't — it's a plain count: how many DIFFERENT top-level accounts (a parent and its own children count once, not once each) raised the same kind of claim about the same product family in the same calendar week. Three or more, and it's a wave: "three hospitals pushed back on Tidewater pricing the week of the 24th." No embedding, no clustering, nothing for a model to get wrong — just a `GROUP BY` and a headcount, the same "cheap, citable, mechanical" shape as the diff feed above.

**The confirmation loop.** Every judge-proposed resolution — never the quiet-weeks kind, which has nothing to confirm — drops a row in the review queue, carrying the theme, the candidate claim, and the judge's own quoted evidence: a human either confirms it (stamped with who and when; this is what actually makes the theme read resolved) or rejects it (which marks the row rejected and leaves the theme reading whatever it would have without the proposal). Unlike the pre-2026-09-14 design, whether the theme reads resolved in next week's report genuinely DOES wait on that click — the queue is not just a record of what already happened, it is the human's decision.

### 3. Standing questions: the user's watchlist
Instead of the system guessing what each reader cares about, each reader registers questions once. Examples:
- "Which accounts mentioned a competitor by name this week?"
- "Any renewal-risk language on accounts above $500k ARR?"
- "What did anyone say about the new pump model?"
- "Which contacts changed roles?" (this is where the LinkedIn RSS feed pays off)

The weekly job runs each question through the fused search, restricted to notes touched this week, and writes the answer with citations into that reader's section. Ops gets ops questions, product gets product questions. This is the MCP query made recurring. Adding a question is one line in a config file, and the MCP can expose "add a standing question" as a tool so a user does it from their chat client.

### 4. Outliers: new and unexplained
Signals that matched no theme in feed 2. Listed plainly with source and account. Most weeks this is short. The week it is long is the week something new is happening, and the reader sees it before a theme exists for it.

## The customer's voice: per-account synthesis

the maintainer asked whether surfacing means "looking around at each customer to identify their voice." It does, and the schema already has the shape for it: the dossier pattern. An account note has a rewritable synthesis on top and an append-only dated timeline underneath.

- Every ingested signal, call, ticket, feed item appends one timeline line to the account note.
- When an account's timeline grew this week, the synthesis on top is regenerated: what they care about, what they complain about, open requests, risk read, key contacts, last contact date.
- That synthesis **is** the customer's voice, current as of the `updated:` stamp, and it is the thing a rep reads before a call.

Only accounts with new timeline entries get regenerated. On a quiet week that is a handful. Cost stays proportional to activity, not corpus size.

## The report is a node

The weekly report is written as `reports/2026-W37.md`, type `report`, with `[[links]]` to every theme, account and signal it cites. Consequences:
- it is searchable through the same MCP ("what did last week's report say about Ashgrove?")
- every claim in it carries provenance, same as any note
- reports accumulate into a history the lint can check (a theme that was "rising" for six weeks and never got an owner is itself a finding)

Delivery is a rendering of that note: email, Slack, PDF. The note is the truth; the email is a copy.

## Numbers

Models never compute (decisions-log 2026-09-12, "Numbers rule"). Every count, sum, or percentage in the report — notes created, a theme's signal count, an account's ARR, a trend delta, the sampled miss rate — is computed by ordinary code (a SQL `COUNT`/`SUM`, a Python loop) before any prompt is built, never asked of a model. That figure is handed to the model as a named fact, alongside a plain instruction to use only the figures it was given. Restating a number a cited claim already states verbatim is fine — that's a citation, not a computation — but summing, rounding, estimating, or inventing one is not.

A mechanical check runs on everything a model generates before it reaches the report: any numeral in the text that isn't one of the supplied figures fails the check (dates, years, CRM/source ids, times, and ordinals are never numerals for this purpose, and neither are spelled-out counting words like "a" or "few"). A failure rejects the whole piece of text outright — the same "drop it, don't patch it" rule the citation check already uses — and logs a `review_queue` row (`kind: other`, `reason: numeral_not_supplied`) so a human sees what got rejected and why. This applies to the weekly headline, per-account synthesis, theme syntheses, and standing-question answers alike; a calculator tool is the fallback for the rare case where a model genuinely must combine numbers to answer a question (chat, not this report).

Implementation: `portolan/report/numbers.py` (`NumberFacts`, `check_numerals`).

## Closing the loop: actions

The research doc's action object belongs here. Any theme or outlier in the report can become an `action` note: owner, due date, the metric to watch, links to the theme. Next week's report has a section "actions from prior weeks": open, overdue, resolved, and for resolved ones, whether the theme actually fell. That is the "Act" stage, and it is what turns the report from reading into operating.

## Cost shape

| Job | Runs | Touches | Model cost |
|---|---|---|---|
| Diff | weekly | metadata only | none |
| Theme refresh | weekly | this week's signals + nearest themes | small, proportional to new signals |
| Account synthesis | weekly | accounts with new timeline entries | small |
| Standing questions | weekly | search over this week's notes | small |
| Outliers | weekly | unmatched signals | none |
| Full re-cluster | quarterly / on demand | all signals | one large run |
| Report prose | weekly | the four feeds | one medium call |

Nothing weekly reads the whole corpus. That is what makes it sustainable at ten thousand notes.

## Build order for this piece

1. `theme` and `report` and `action` types in the ontology, with required links
2. Diff query over `created:` / `updated:` / changelog (no model needed; do this first, it is most of the value)
3. Account synthesis regeneration for touched accounts
4. Theme matching pass
5. Standing-questions config + runner
6. Report assembler + one renderer (email or Slack)
7. Action notes and the "prior actions" section

Steps 2 and 3 alone give a useful weekly report. Test all of it against the synthetic corpus: plant a rising theme across three weeks of generated data and check the report catches it in week two.
