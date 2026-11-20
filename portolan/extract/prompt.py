"""Renders the extractor's system and user prompts from the schema module.

Nothing in here defines a fact rule, a type definition, or an example —
those all live in `portolan.extract.schema` (`FACT_RULES`, `FACT_TYPES`,
`merge_customer_extension`). This module is pure rendering: strings in,
strings out, so it can be unit-tested without touching a model.

The system prompt is built to be **stable across calls** so it can be
prompt-cached: no timestamps, no per-call data (that all goes in the user
message), and a customer's extension only changes the prompt when the
extension itself changes — not on every call.

Every example in the prompt (worked examples, locator samples, precedence
rules) names one fictional vendor, Veltrona Water Systems, and its
fictional accounts. None of them appears in an exam corpus, so a tenant's
own vendor reaches the prompt only through `company_context`.
`tests/test_extract_prompt.py` pins both halves of that.
"""

from __future__ import annotations

import hashlib

from portolan.extract.schema import (
    FACT_RULES,
    FACT_TYPE_OTHER_V1,
    TYPE_PRECEDENCE_RULES_V6,
    CustomerExtension,
    EntityKind,
    FactType,
    FactTypeInfo,
    SourceEnvelope,
    TypePrecedenceRule,
    merge_customer_extension,
)
from portolan.util.prompt_safety import SOURCE_FRAMING, clean, frame_source

# The made-up names the examples in this prompt use: the vendor, its
# accounts, people and products. `portolan.bench.extraction` counts a
# produced fact that carries one of them into a source that never says it.
PROMPT_EXAMPLE_NAMES: tuple[str, ...] = (
    "Veltrona",
    "Brackenholt",
    "Harrowfield",
    "Quenby",
    "Pembrake",
    "Glenmoss",
    "Duncarron",
    "Pellworth",
    "Ferrowick",
    "Vantrell",
    "Rillfold",
    "Tarnwick",
    "Rudderham",
    "Brenneck",
    "Sandqvist",
    "Ondrell",
    "Sorvald",
    "Olufsen",
    "Achterberg",
)


def _render_type_checklist(checklist: dict[str, FactTypeInfo]) -> str:
    """Render the signal-type checklist: definition, example, counter-example
    per type, in checklist order. This is "what makes the model look for
    each kind explicitly" per extraction-schema-v0.md."""
    lines: list[str] = []
    for type_name, info in checklist.items():
        lines.append(f"### {type_name}")
        lines.append(f"Definition: {info.definition}")
        lines.append(f"Example: {info.example}")
        lines.append(f"Not this: {info.counter_example} — {info.counter_reason}")
        lines.append("")
    return "\n".join(lines).rstrip()


_DURABILITY_BAR = (
    "## What's worth extracting (the durability bar)\n"
    "Before writing a fact, ask: would someone still want to know this a "
    "month from now? A fact clears that bar when it is what the customer "
    "needs, wants, complains about, decides, commits to, or a durable "
    "change in who's involved. It does NOT clear the bar when it is:\n"
    "- scheduling logistics (a meeting time, a calendar invite, \"let's "
    'find a date that works")\n'
    "- small talk and pleasantries (weekend plans, the weather, a hobby, "
    "how someone's kids are doing)\n"
    "- a restatement of a fact you already wrote from this same source\n"
    "- a vague or undated acknowledgment with no new information (\"I'll "
    'look into it and get back to you", with no deliverable or date)\n'
    "When in doubt, leave it out. A missed fact costs a little coverage; a "
    "trivial or invented one costs a reader's trust in every fact next to "
    "it."
)

_FACT_BUDGET = (
    "## How many facts\n"
    "Extract roughly one fact per 600 words of source text, as a soft "
    "guide, not a hard limit — a dense QBR can reasonably produce more, a "
    "routine check-in call can reasonably produce zero. When two candidate "
    "facts are both marginal, prefer writing the stronger one and dropping "
    "the weaker one over writing both. Fewer, high-value facts beat many "
    "low-value ones."
)

_LOCATOR_INSTRUCTIONS = (
    "## Locators\n"
    "Every source line already carries its own locator in brackets at the "
    "start of the line. Copy ONLY the identifier inside the brackets — "
    "never the brackets themselves, and never a range or a description.\n"
    "- call: the line `[00:06:05] Lorcan Rudderham: that late order last "
    "week...` has locator `00:06:05` — not `[00:06:05]`.\n"
    "- ticket: the line `[S-ticket-5318#1] Lorcan Rudderham: Our Rillfold "
    "R4 filter cartridge order...` has locator `S-ticket-5318#1` — not "
    "`[S-ticket-5318#1]`.\n"
    "- email: the line `[¶2] Quick request from our operator training "
    "team...` has locator `¶2` — not `[¶2]`.\n"
    "- rss: the `[url:<link> <date>]` line's own url.\n"
    "- crm: the `[account:A-7730] ...` line's own id, e.g. `account:A-7730`.\n"
    "Every fact and action needs a locator from a line you actually read. "
    "Never invent one."
)

# --------------------------------------------------------------------------
# Prompt A/B variants (extraction accuracy campaign, 2026-09-18,
# `01-spec/prompt-ab-2026-09-18.md`). Each block below is one variant's
# addition; `build_system_prompt`'s `variant` parameter turns them on
# CUMULATIVELY (v3 includes v1's and v2's additions, etc.) per the
# campaign's "build cumulatively, measure each step" instruction.
# `variant="baseline"` renders none of them.
# --------------------------------------------------------------------------

# Feature flags a variant can turn on. `v5-per-type-sweep` and
# `v6-type-precision` both build on `v4-two-step` but are NOT cumulative
# with each other (v6 was built "on top of v4" per `01-spec/
# facttype-4a-2026-09-18.md`, deliberately skipping v5's per-type sweep --
# a different, recall-oriented addition this wave doesn't touch) -- so
# membership is an explicit per-variant set rather than "level >= N",
# which would force every later-numbered variant to include every
# earlier one's block regardless of lineage.
_OTHER_EXAMPLE = "other_example"
_DECISION_PRECEDENCE = "decision_precedence"
_QUOTE_ALWAYS = "quote_always"
_TWO_STEP = "two_step"
_PER_TYPE_SWEEP = "per_type_sweep"
_TYPE_PRECISION = "type_precision"

_ENTITY_NAMES = "entity_names"

_V4_FEATURES: frozenset[str] = frozenset(
    {_OTHER_EXAMPLE, _DECISION_PRECEDENCE, _QUOTE_ALWAYS, _TWO_STEP}
)

# Order here is display/iteration order only (`PROMPT_VARIANTS` below) --
# it is NOT used to derive cumulative membership; each variant's own entry
# in `_VARIANT_FEATURES` is authoritative.
PROMPT_VARIANTS: tuple[str, ...] = (
    "baseline",
    "v1-other-example",
    "v2-decision-tightening",
    "v3-quote-always",
    "v4-two-step",
    "v5-per-type-sweep",
    "v6-type-precision",
    "v7-entity-names",
)

# Production's variant: `portolan.pipeline.run()`, the bench CLI and silver
# labeling all default to it.
DEFAULT_PROMPT_VARIANT = "v7-entity-names"

_VARIANT_FEATURES: dict[str, frozenset[str]] = {
    "baseline": frozenset(),
    "v1-other-example": frozenset({_OTHER_EXAMPLE}),
    "v2-decision-tightening": frozenset({_OTHER_EXAMPLE, _DECISION_PRECEDENCE}),
    "v3-quote-always": frozenset({_OTHER_EXAMPLE, _DECISION_PRECEDENCE, _QUOTE_ALWAYS}),
    "v4-two-step": _V4_FEATURES,
    "v5-per-type-sweep": _V4_FEATURES | {_PER_TYPE_SWEEP},
    "v6-type-precision": _V4_FEATURES | {_TYPE_PRECISION},
    "v7-entity-names": _V4_FEATURES | {_TYPE_PRECISION, _ENTITY_NAMES},
}


def _variant_features(variant: str) -> frozenset[str]:
    try:
        return _VARIANT_FEATURES[variant]
    except KeyError as exc:
        raise ValueError(
            f"unknown prompt variant {variant!r}; choose one of {list(PROMPT_VARIANTS)}"
        ) from exc


# v7-entity-names (2026-11-20, docs/reviews/REVIEW-write-path.md "Entity
# accuracy"). Two things seen on a live run: 1 to 10 percent of facts named
# this prompt's made-up vendor as the seller, because the source never said
# who the seller was and the CRM example spells one out; and about 5
# percent of top-level mentions carried an id in `mention`. The block names
# no example company, so it cannot teach the first one itself.
_ENTITY_NAMES_TEXT = (
    "## Names: only the ones this source gives you\n"
    "Every company, person and product named in this prompt's examples is "
    "made up to show the format. A name belongs in a claim or an "
    "`entities` list only when the source you are reading, its registry "
    "context or the company context contains it. When none of them names "
    'the company doing the selling, write "the vendor" in the claim; do '
    "not borrow a name from an example.\n"
    "In the top-level `entities` list, `mention` is the name the way the "
    "source writes it and `registry_id` holds the id. An id never goes in "
    "`mention`. Copy a registry id exactly as registry context prints it, "
    "prefix included."
)

_CRM_EXAMPLE_NAMED = "annual recurring revenue with Veltrona is"
_CRM_EXAMPLE_UNNAMED = "annual recurring revenue with the vendor is"

# v1-other-example, second half (the first half is FACT_TYPE_OTHER_V1 in
# schema.py): CRM rows, RSS locators, and email paragraphs each need a
# different reading strategy than "find a sentence to quote". The RSS
# bullet exists because models dropped the trailing date from RSS
# locators (`01-spec/extraction-miss-analysis-2026-09-18.md` rec. 3).
_SOURCE_TYPE_INSTRUCTIONS = (
    "## Reading structured and short sources\n"
    "Not every source is a transcript with dialogue. Three source types "
    'need a different reading strategy than "find a sentence to quote":\n'
    "- **crm**: each `[account:...]`/`[contact:...]`/`[opportunity:...]` "
    "row is itself one or more facts, not something to skip because it "
    "has no prose. Read every populated field. An account row's `arr` "
    "figure is a stored record value (type `other`) — e.g. `[account:"
    "A-7730] crm_id=A-7730, name=Brackenholt Beverage Group, ..., "
    "arr=1875000, ...` is the fact \"Brackenholt Beverage Group's annual "
    'recurring revenue with Veltrona is 1,875,000 dollars." An '
    "`opportunity` row's `amount`/"
    "`close_date` is a live negotiation, not a stored record value — "
    "that is `pricing`, not `other`. A `parent_crm_id` field is an "
    'ownership fact ("X is owned by Y", `other`). A contact row\'s '
    "`role`/`status` change is `contact_change`. The row's own bracketed "
    "id is the locator either way. A CRM source is rarely also given "
    "registry context for its own account, so for the fact's `entities` "
    "value: use the registry id if the account IS listed in registry "
    "context; otherwise use the row's own readable `name=` field (e.g. "
    "`Brackenholt Beverage Group`) as the mention — never the bare row "
    "id (`A-7730`) alone, which is neither a registry id the merge step "
    "can resolve nor the account's name. For the account row above: "
    '`entities: ["Brackenholt Beverage Group"]` is correct; `entities: '
    '["A-7730"]` is wrong.\n'
    "- **rss**: the WHOLE `[url:<link> <date>]` line's bracket contents "
    "are the locator, url AND date together — e.g. `url:https://news."
    "example.test/water-industry/filtration-rule-update 2026-05-12`. "
    "Never drop the trailing date and copy the url alone; the date is "
    "part of the locator, not commentary.\n"
    "- **email**: extract per paragraph (`[¶N]`), not per whole thread — "
    "a four-paragraph email can hold up to four distinct facts if four "
    "paragraphs each say something durable. Do not collapse a multi-"
    "paragraph thread into one fact just because it is one message."
)

# v2-decision-tightening: precedence rule plus one negative example per
# confused type. The first three confusions were observed on the exam
# (`01-spec/extraction-miss-analysis-2026-09-18.md` §3); the sentences
# here restate them with fictional names so the prompt carries no exam
# text. The `pricing` case was never observed, and the prompt says so.
_DECISION_PRECEDENCE_TEXT = (
    "## Type precedence: `decision` is not a catch-all\n"
    "`decision` over-triggers on sentences that sound conclusive (\"we're "
    'not doing X", "we decided to drop Y", "it\'s going to Z") when a MORE '
    "SPECIFIC type already covers the same statement. Rule: if a "
    "statement fits a more specific type, use that type — `decision` is "
    "only for a party stating a choice already made that is not better "
    "described as a competitor mention, a request, praise, pricing, or "
    "any other more specific type in the checklist.\n"
    "Four confusions seen in practice, each with the sentence and the "
    "type that actually fits:\n"
    "- **competitor_mention, not decision**: \"We're not looking at other "
    'suppliers" (a customer saying they are not evaluating competitors) '
    "names the competitive question, even phrased in the negative — it "
    "is `competitor_mention`, not `decision`, because the statement's "
    "subject is competitive evaluation, not an internal choice.\n"
    '- **request, not decision**: "Pembrake Cider Works told Veltrona it '
    "would defer the Rillfold Bridge/Tarnwick MES integration discussion "
    'until 2028" reads like a decision ("we decided to wait"), but the '
    "substance is the customer asking Veltrona to shelve something — "
    "`request`, because the sentence's function is asking for something "
    "(here, a delay), not stating an independent internal choice.\n"
    '- **praise, not decision**: "shipments had arrived on schedule for '
    "three weeks running, and she was shelving the backup-vendor plan "
    'for now" is praise (relief that a prior problem resolved) wearing a '
    'decision\'s clothes ("shelving the plan" sounds like a choice) — '
    "the substance being reported is that the product/service is "
    "working, which is `praise`.\n"
    "- **pricing, not decision** (illustration, not an observed case): "
    '"Brackenholt Beverage Group agreed to an 8 percent discount for the '
    '2028 renewal" states a decision-shaped sentence ("agreed to"), but '
    "the substance is a price figure — `pricing`, because the number is "
    "what a reader needs, not that a decision process concluded.\n"
    "Only write `decision` when none of the more specific types above "
    "capture the statement's real substance — e.g. \"Brackenholt chose "
    'to renew the service contract for two years" has no more specific '
    "type to prefer; that one is a real `decision`."
)

# v3-quote-always: extends rule 5's quote requirement (base: complaint,
# praise, competitor_mention, commitment, pricing only) to every type, as
# a PROMPT instruction only -- this does not change `QUOTE_REQUIRED_
# TYPES`/schema-level validation (the live bench pipeline never passes an
# `effective_schema` into `ExtractionResult.parse_lenient`, so mechanical
# enforcement is out of scope for a prompt-only variant); it is measured
# by how well the model complies on its own, per the campaign brief's own
# framing ("watch for a recall drop and output-token cost").
_QUOTE_ALWAYS_ADDENDUM = (
    "## Quote discipline, every type\n"
    "Every fact — not only complaint/praise/competitor_mention/"
    "commitment/pricing — needs a verbatim quote, forty words or fewer, "
    "from the exact source text, in the `quote` field. For a structured "
    "source with no prose to quote (a CRM row, see 'Reading structured "
    "and short sources'), put the exact field text you read instead, "
    "e.g. `arr=1875000` — the raw key=value snippet, not a paraphrase. A "
    "fact with no supporting quote is a fact a reader cannot verify; "
    "write fewer facts before you write one with no quote."
)

# v4-two-step: "ask the model to first list candidate statements ... then
# classify each into a fact (or drop it), returning only the facts" — a
# single-call internal process instruction (the tool schema has no
# candidates field, so this is a reasoning-order instruction, not a
# second API call).
_TWO_STEP_INSTRUCTIONS = (
    "## Two-step extraction process\n"
    "Do this in two internal steps before you call the tool (you do not "
    "need to show this work — only the final tool call is returned):\n"
    "1. List candidates. Scan the whole source and list every statement "
    "that might be durable: who said it, and its locator. Be generous "
    "here — include anything that might matter, even if you are not "
    "sure yet whether it clears the durability bar.\n"
    "2. Classify or drop. For each candidate: does it clear the "
    "durability bar? If yes, which single type fits best — check every "
    "type in the checklist, in order, before defaulting to `other`? If "
    "no type fits and it does not clear the bar, drop it.\n"
    "Only the facts that survive step 2 go into the `facts` array — "
    "never include intermediate candidate notes in the tool call output."
)

# v5-per-type-sweep (optional, campaign brief): "one extra question per
# fact type in the same request" after the main pass.
_PER_TYPE_SWEEP_INSTRUCTIONS = (
    "## Final check, one pass per type\n"
    "Before finalizing your `facts` array, re-scan the source once per "
    'type in the checklist and ask: "any statement of this type not '
    'already listed?" Add anything you find; do not remove or '
    "re-classify anything already listed. This catches a fact type you "
    "stopped looking for after finding a few of a different kind."
)


def _render_type_precedence_rules(rules: list[TypePrecedenceRule]) -> str:
    """Render `TYPE_PRECEDENCE_RULES_V6` as "`winner`, not `loser`: rule /
    Right: ... / Wrong: ..." blocks, one per confused pair, in list order."""
    lines: list[str] = []
    for r in rules:
        lines.append(f"- **`{r.winning_type.value}`, not `{r.losing_type.value}`**: {r.rule}")
        lines.append(f"  Right (`{r.winning_type.value}`): {r.positive_example}")
        lines.append(
            f"  Wrong (`{r.losing_type.value}`, not `{r.winning_type.value}`): {r.negative_example}"
        )
    return "\n".join(lines)


# v6-type-precision (facttype-4a, 2026-09-18, `01-spec/facttype-4a-2026-09-
# 18.md`): a third step added to the two-step flow, run right after
# `_TWO_STEP_INSTRUCTIONS`'s step 2 -- a final self-check against the four
# confused-pair rules `TYPE_PRECEDENCE_RULES_V6` names in `schema.py`, the
# ones the confusion-matrix diagnosis over `var/bench/2026-09-18-final/`
# found pooling every model's matched-but-wrong-typed facts on this same
# v4-two-step prompt. Kept as its own block (not edited into
# `_TWO_STEP_INSTRUCTIONS` itself) so v4's pinned two-step text never
# changes -- rendered immediately after it, so the two read as one
# continuous three-step flow.
_TYPE_PRECISION_SELF_CHECK = (
    "## Step 3: type precision self-check\n"
    "Before you call the tool, re-read every fact in your draft `facts` "
    "array once more and check its `type` against the precedence rules "
    "below, in order. Four confusions come up often enough on this kind "
    "of source to name explicitly:\n"
    f"{_render_type_precedence_rules(TYPE_PRECEDENCE_RULES_V6)}\n"
    "If a fact's type loses to a more specific rule above, fix the type "
    "before emitting -- do not add or remove facts in this step, only "
    "correct a wrong `type` value."
)

_WORKED_EXAMPLES = (
    "## Worked examples\n"
    "\n"
    "### Example 1 — call\n"
    "Source text:\n"
    "```\n"
    "[00:00:00] Tovah Brenneck: Hey Lorcan, thanks for hopping on. How's "
    "it going over there this morning?\n"
    "[00:00:18] Tovah Brenneck: Can't complain. I was actually down in "
    "Spokane for a couple days for a regional meeting, just got back home "
    "last night.\n"
    "[00:01:31] Tovah Brenneck: Good. How's the family, by the way? "
    "Didn't you say your oldest was starting kindergarten this year?\n"
    "[00:02:40] Tovah Brenneck: Oh, before I forget, did you end up going "
    "to that brewers' guild conference last month?\n"
    "[00:03:12] Lorcan Rudderham: A little, yeah. We're looking at "
    "tightening up our sanitation rotation a bit, nothing dramatic.\n"
    "[00:06:05] Lorcan Rudderham: So, that late order last week — that "
    "shipment showing up three days behind schedule caused a real "
    "headache on our end for a few days.\n"
    "[00:08:14] Tovah Brenneck: Let me dig into it right after this call "
    "and I'll follow up by email, probably tomorrow, with what actually "
    "caused the delay.\n"
    "[00:14:26] Lorcan Rudderham: Same day, yeah, within about two hours "
    "actually. Honestly, that was the quickest turnaround we've gotten "
    "from any supplier.\n"
    "```\n"
    "Facts to extract (two):\n"
    "1. `complaint`, locator `00:06:05`: Quenby Creamery's Rillfold R4 "
    "filter cartridge order arrived three days late. Quote: "
    '"that shipment showing up three days behind schedule caused a real '
    'headache on our end for a few days."\n'
    "2. `praise`, locator `00:14:26`: Quenby Creamery's maintenance lead "
    "said Veltrona's field engineer resolved a skid alarm issue within "
    'about two hours. Quote: "Honestly, that was the quickest turnaround '
    "we've gotten from any supplier.\"\n"
    "Left out (four), and why:\n"
    "- 00:00:00-00:00:18 (Spokane trip, weather) — pleasantries.\n"
    "- 00:01:31 (kindergarten) — pleasantries.\n"
    "- 00:02:40-00:03:12 (brewers' guild conference, sanitation rotation) "
    "— an internal process tweak on the customer's own side, not "
    "involving Veltrona and not a decision; too vague to be durable.\n"
    "- 00:08:14 (Tovah's \"I'll follow up by email\") — no deliverable or "
    "date, and superseded later in the same call by a specific written "
    "commitment; not worth logging on its own.\n"
    "\n"
    "### Example 2 — ticket\n"
    "Source text:\n"
    "```\n"
    "Ticket S-ticket-5318: Late cartridge delivery\n"
    "[S-ticket-5318#1] Lorcan Rudderham: Our Rillfold R4 filter cartridge "
    "order was due March 9th and only showed up March 12th — 3 days late. "
    "We got by on the spares we keep on site, but it was tighter than "
    "I'm comfortable with.\n"
    "[S-ticket-5318#2] Tovah Brenneck: Apologies, Lorcan. I've asked our "
    "shipping team what went wrong and will pass along the root cause "
    "when I have it.\n"
    "[S-ticket-5318#3] Lorcan Rudderham: Thanks. I mostly want to know "
    "this won't keep happening, since our spares run lower next month.\n"
    "```\n"
    "Facts to extract (one):\n"
    "1. `complaint`, locator `S-ticket-5318#1`: Quenby Creamery's Rillfold "
    'R4 filter cartridge order arrived three days late. Quote: "Our '
    "Rillfold R4 filter cartridge order was due March 9th and only showed "
    'up March 12th — 3 days late."\n'
    "Left out (two), and why:\n"
    "- comment #2 — a procedural acknowledgment restating the complaint, "
    "with no new information and no committed date.\n"
    "- comment #3 — restates the same concern without adding a new fact.\n"
    "\n"
    "### Example 3 — email\n"
    "Source text:\n"
    "```\n"
    "[msg:S-email-6614@veltronawater.com]\n"
    "From: Thaddeus Olufsen <tolufsen@ferrowickbottling.com>\n"
    "To: Ysolde Achterberg <y.achterberg@veltronawater.com>\n"
    "Subject: Vietnamese-language troubleshooting guide\n"
    "[¶1] Hi Ysolde,\n"
    "[¶2] Quick request from our operator training team — could Veltrona "
    "produce a Vietnamese-language version of the Rillfold R4 "
    "troubleshooting guide? More and more of our line staff would find it "
    "easier to work from in Vietnamese.\n"
    "[¶3] Could you tell me whether that's doable and about how long it "
    "would take to turn around?\n"
    "[¶4] Thaddeus\n"
    "```\n"
    "Facts to extract (one):\n"
    "1. `request`, locator `¶2`: Ferrowick Bottling Company asked for a "
    "Vietnamese-language troubleshooting guide for the Rillfold R4.\n"
    "Left out (three): ¶1 (greeting), ¶3 (asking for a timeline on the "
    "same request already captured — not a second fact), ¶4 (signature)."
)


def render_labeling_rules(variant: str = DEFAULT_PROMPT_VARIANT) -> str:
    """The production prompt's rules for what a fact is, verbatim: the
    durability bar, the fact rules, the signal-type checklist with its
    counter-examples, and the type precedence rules `variant` turns on.
    `portolan.training.label_judge` shows these to the judge, so the judge
    and the extractor read one text."""
    features = _variant_features(variant)
    checklist = dict(merge_customer_extension(None).fact_type_checklist)
    if _OTHER_EXAMPLE in features:
        checklist[FactType.OTHER.value] = FACT_TYPE_OTHER_V1
    sections = [_DURABILITY_BAR]
    rule_lines = (f"{i}. {rule}" for i, rule in enumerate(FACT_RULES, start=1))
    sections.append("## Fact rules\n" + "\n".join(rule_lines))
    sections.append("## Signal types\n" + _render_type_checklist(checklist))
    if _DECISION_PRECEDENCE in features:
        sections.append(_DECISION_PRECEDENCE_TEXT)
    if _TYPE_PRECISION in features:
        sections.append(
            "## More type precedence rules\n"
            + _render_type_precedence_rules(TYPE_PRECEDENCE_RULES_V6)
        )
    return "\n\n".join(sections)


def build_system_prompt(
    extension: CustomerExtension | None = None,
    *,
    company_context: str | None = None,
    variant: str = DEFAULT_PROMPT_VARIANT,
) -> str:
    """Build the extractor's system prompt: role, the company context (if
    given), the durability bar, the fact budget, fact rules, locator
    instructions, the full signal-type checklist (base types plus this
    customer's extra types), entity kinds, customer vocabulary/aliases if
    an extension is given, and three worked examples.

    Deterministic in every input except `extension`/`company_context`/
    `variant` — call it once per (schema_version, extension, company_
    context, variant) tuple and cache the result, not once per source.
    `company_context` is who the vendor is: the tenant's company-context
    note body in production (`portolan.context.load_company_context`), a
    world's or corpus's own vendor note in training. Plain text with no
    "## Company context" heading of its own — this function adds it.
    `None` renders no block at all, same as a missing context note.

    `variant` selects a prompt A/B preset from the extraction accuracy
    campaign (`01-spec/prompt-ab-2026-09-18.md`, `PROMPT_VARIANTS` above)
    — one of `"baseline"`, `"v1-other-example"`,
    `"v2-decision-tightening"`, `"v3-quote-always"`, `"v4-two-step"`,
    `"v5-per-type-sweep"`, `"v6-type-precision"`, or `"v7-entity-names"`
    (the default, `DEFAULT_PROMPT_VARIANT`: v6 plus one block on where a
    name may come from). `v1` through `v4` are
    CUMULATIVE — `"v3-quote-always"` includes v1's and v2's additions too
    — per the campaign's "build cumulatively, measure each step"
    instruction. `v5-per-type-sweep` and `v6-type-precision` both build on
    `v4-two-step` but NOT on each other (`01-spec/
    facttype-4a-2026-09-18.md`) — each adds its own thing on top of v4's
    feature set, neither includes the other's. Raises `ValueError` for any
    other string.
    """
    effective = merge_customer_extension(extension)
    features = _variant_features(variant)

    checklist = dict(effective.fact_type_checklist)
    if _OTHER_EXAMPLE in features:
        # v1-other-example: the worked example lives on FACT_TYPE_OTHER_V1
        # in schema.py (fact-type content), not here (pure rendering) —
        # see that constant's docstring.
        checklist[FactType.OTHER.value] = FACT_TYPE_OTHER_V1

    sections: list[str] = []

    sections.append(
        "You are Portolan's extraction engine. Read exactly one source (or "
        "one segment of a long source) and return strict JSON matching the "
        "ExtractionResult schema you have been given as a forced tool call. "
        "Return nothing else: no prose before or after the tool call."
    )

    # Added 2026-09-17 (owner decision, "Company context page"): before
    # the vocabulary block, per the build instructions, so the model
    # knows who the company is before it sees the durability bar or the
    # fact rules that follow.
    if company_context:
        from portolan.context import render_company_context_block

        sections.append(render_company_context_block(company_context))

    # v4-two-step: a process instruction, placed right after the role/
    # company-context framing and before the durability bar, fact rules
    # and checklist it refers to — those all still apply, this just
    # tells the model the order to work through them in.
    if _TWO_STEP in features:
        sections.append(_TWO_STEP_INSTRUCTIONS)
        # v6-type-precision: step 3 of the same flow, right after steps 1
        # and 2 — see `_TYPE_PRECISION_SELF_CHECK`'s own comment for why
        # this is a separate block rather than an edit to the pinned v4
        # two-step text.
        if _TYPE_PRECISION in features:
            sections.append(_TYPE_PRECISION_SELF_CHECK)

    sections.append(_DURABILITY_BAR)
    sections.append(_FACT_BUDGET)

    rule_lines = (f"{i}. {rule}" for i, rule in enumerate(FACT_RULES, start=1))
    sections.append("## Fact rules\n" + "\n".join(rule_lines))

    sections.append(_LOCATOR_INSTRUCTIONS)

    # v1-other-example, second half: source-type reading guidance (CRM
    # rows, RSS locators, email paragraphs), right after the base locator
    # instructions it extends.
    if _OTHER_EXAMPLE in features:
        sections.append(_SOURCE_TYPE_INSTRUCTIONS)

    sections.append(
        "## Signal types\n"
        "Every fact you write must use one of these `type` values. Check the "
        "source against every type below in turn before defaulting to "
        '`other` — the counter-example after "Not this" shows a claim that '
        "looks similar but does not qualify.\n\n" + _render_type_checklist(checklist)
    )

    # v2-decision-tightening: precedence rule right after the checklist
    # it disambiguates.
    if _DECISION_PRECEDENCE in features:
        sections.append(_DECISION_PRECEDENCE_TEXT)

    if effective.extra_type_names:
        sections.append(
            "## Customer-specific types\n"
            "These extend the base list for this customer: "
            + ", ".join(f"`{name}`" for name in effective.extra_type_names)
        )

    sections.append(
        "## Entity kinds\n"
        "Every entity mention resolves to one of: "
        + ", ".join(f"`{kind.value}`" for kind in EntityKind)
        + ". Internal people resolve against the customer's own directory "
        '(e.g. "Tovah" becomes `internal:tovah-brenneck`). Each fact\'s '
        "own `entities` list uses the same values: the `registry_id` from "
        "`registry_context` (in the user message) when the entity is "
        "listed there, or the exact mention string — the name as it "
        'appears in the source, e.g. "Brackenholt Harrowfield Brewery" — '
        "when it is not. Never invent a registry id or a slug-shaped guess "
        "(`brackenholt-harrowfield-brewery`) for an entity that isn't in "
        "`registry_context`; an unresolved entity is a plain name, not a "
        "guess at what its id might be."
    )

    if _ENTITY_NAMES in features:
        sections.append(_ENTITY_NAMES_TEXT)

    has_vocabulary = (
        effective.vocabulary.products
        or effective.vocabulary.competitors
        or effective.vocabulary.jargon
    )
    if has_vocabulary:
        vocab_lines = ["## Customer vocabulary"]
        if effective.vocabulary.products:
            vocab_lines.append("Products: " + ", ".join(effective.vocabulary.products))
        if effective.vocabulary.competitors:
            vocab_lines.append("Known competitors: " + ", ".join(effective.vocabulary.competitors))
        if effective.vocabulary.jargon:
            jargon_pairs = ", ".join(f"{k} = {v}" for k, v in effective.vocabulary.jargon.items())
            vocab_lines.append("Jargon: " + jargon_pairs)
        sections.append("\n".join(vocab_lines))

    if effective.aliases:
        alias_lines = ["## Known aliases"]
        for registry_id, names in effective.aliases.items():
            alias_lines.append(f"{registry_id}: " + ", ".join(names))
        sections.append("\n".join(alias_lines))

    # v3-quote-always: right before the worked examples, so it's the last
    # rule the model reads before seeing them applied.
    if _QUOTE_ALWAYS in features:
        sections.append(_QUOTE_ALWAYS_ADDENDUM)

    sections.append(_WORKED_EXAMPLES)

    # v5-per-type-sweep: after the worked examples, as the final
    # instruction before the model reads the actual source (the user
    # message) — a last-pass reminder, not a rule to apply mid-read.
    if _PER_TYPE_SWEEP in features:
        sections.append(_PER_TYPE_SWEEP_INSTRUCTIONS)

    prompt = "\n\n".join(sections)
    if _ENTITY_NAMES in features:
        # A CRM row never says who the seller is, and the two CRM examples
        # spelled the made-up vendor out; models copied the sentence, name
        # and all. The examples now follow the rule the block above states.
        prompt = prompt.replace(_CRM_EXAMPLE_NAMED, _CRM_EXAMPLE_UNNAMED)
    return prompt


def system_prompt_sha256(system_prompt: str) -> str:
    """Hex sha256 of a rendered system prompt's UTF-8 bytes: the stamp a
    silver record carries so a later reader can prove which prompt
    labeled it."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def _render_participants(envelope: SourceEnvelope) -> str:
    if not envelope.participants:
        return "(none provided)"
    lines = []
    for p in envelope.participants:
        bits = [p.name]
        if p.role:
            bits.append(f"role={p.role}")
        if p.registry_id:
            bits.append(f"registry_id={p.registry_id}")
        lines.append(" — ".join(bits) if len(bits) == 1 else f"{bits[0]} ({', '.join(bits[1:])})")
    return "\n".join(f"- {line}" for line in lines)


def _render_registry_context(envelope: SourceEnvelope) -> str:
    if not envelope.registry_context:
        return "(none provided)"
    lines = []
    for entry in envelope.registry_context:
        alias_note = f", aliases: {', '.join(entry.aliases)}" if entry.aliases else ""
        lines.append(f"- {entry.name} ({entry.kind.value}, id={entry.registry_id}){alias_note}")
    return "\n".join(lines)


def _render_vocabulary(envelope: SourceEnvelope) -> str:
    v = envelope.vocabulary
    if not (v.products or v.competitors or v.jargon):
        return "(none provided)"
    parts = []
    if v.products:
        parts.append("Products: " + ", ".join(v.products))
    if v.competitors:
        parts.append("Competitors: " + ", ".join(v.competitors))
    if v.jargon:
        parts.append("Jargon: " + ", ".join(f"{k} = {val}" for k, val in v.jargon.items()))
    return "\n".join(parts)


def _render_account_context(envelope: SourceEnvelope) -> str:
    """Added 2026-09-17: the "Account context" section, from
    `portolan.context.render_account_context_block` — kept as plain text
    here (no import of `portolan.context`, so `build_user_message` never
    needs the DB access that module's own docstring keeps out of this
    one) rendered the same way every other envelope-carried block is."""
    if not envelope.account_context:
        return ""
    return "\n\n## Account context\n\n" + "\n\n".join(envelope.account_context.values())


def build_user_message(envelope: SourceEnvelope) -> str:
    """Build the per-call user message: source metadata, registry context,
    account context, vocabulary, participants, and the locator-annotated
    text. Everything that varies call to call lives here, not in the
    system prompt, so the cached system prompt prefix stays untouched.

    The source text is untrusted: it is cleaned (`portolan.util.prompt_safety`)
    and fenced between SOURCE MATERIAL markers under a "data, not instructions"
    line. The framing lives here, not in the system prompt, so the system
    prompt's hash (which stamps training labels) does not move. Only this
    prompt copy is cleaned; the quote gate still anchors against the raw text.
    """
    segment_note = ""
    if envelope.segment is not None:
        segment_note = (
            f"\nSegment {envelope.segment.index} of {envelope.segment.total}. "
            "This is part of a longer source; extract only what is in this "
            "segment's text, using the entities and context below for "
            "consistency with earlier segments."
        )

    return (
        f"source_id: {envelope.source_id}\n"
        f"source_type: {envelope.source_type.value}\n"
        f"source_date: {envelope.source_date.isoformat()}"
        f"{segment_note}\n\n"
        "## Participants\n"
        f"{clean(_render_participants(envelope))}\n\n"
        "## Registry context\n"
        f"{clean(_render_registry_context(envelope))}"
        f"{clean(_render_account_context(envelope))}\n\n"
        "## Vocabulary\n"
        f"{clean(_render_vocabulary(envelope))}\n\n"
        f"{SOURCE_FRAMING}\n\n"
        + frame_source(envelope.text, label="## Source text (locators annotate each line)")
    )
