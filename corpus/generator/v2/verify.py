"""WP4 helper: the eight verification checks for one written document.

`01-spec/world-generator-v2-2026-09-22.md` section 3.2: "A scripted fact is
*verified* when all eight checks pass":

1. Exactly one marker pair (or the designed count for a multi-mention fact).
2. The span sits inside one locator unit (call line, email paragraph,
   comment, chat message) and is at most 60 words.
3. Every anchor is present after glyph folding. For `reference_mode` other
   than explicit, the anchor may instead appear earlier in the document, so
   the reference can be resolved.
4. The speaker matches (line speaker, From, comment author, chat user).
5. No forbidden word appears.
6. The numeral-and-name rule passes (`portolan.training.overlap_gate`).
7. The locator is well formed (`manifest.LOCATOR_PATTERNS`, widened here
   for v2's per-vendor email domain and hyphenated ids).
8. After `prepare_text`, the span still resolves exactly
   (`portolan.training.quote_span.resolve_quote_span`); the label stores that
   character span.

Decoys go through the same checks except 6 (a decoy has no claim); check 5
also refuses a decoy span that matches one of its class's forbidden patterns
(`DECOY_FORBIDDEN_PATTERNS`).

Beyond the eight, every span must sound like the person saying it
(`corpus.generator.v2.voice`, checks c9 to c13): no ISO date in speech,
never both parties named in one sentence, never the speaker's own
organization in the third person, no restated claim, no pronoun from the
wrong side. The account is therefore not a span anchor: the speaker says
"we" or "you", and the account's full name only has to appear somewhere in
the source (c6:account_not_in_text).

Failure codes are `c<N>:<reason>` strings so a caller can count them per
check and a repair prompt can quote them back to the writer.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ConfigDict

from corpus.generator.manifest import LOCATOR_PATTERNS
from corpus.generator.markers import strip_all_markers
from corpus.generator.v2.assembly import (
    SPEAKER_KINDS,
    AssembledDoc,
    AssembledUnit,
    Cell,
    MarkerSpan,
    PreparedSource,
)
from corpus.generator.v2.models import Fact, Person, PersonSide, ReferenceMode, World
from corpus.generator.v2.planner import DocKind, DocPlan, LabelRecord, PlannedDecoy
from corpus.generator.v2.renderers import RenderedDoc
from corpus.generator.v2.voice import (
    ShortForm,
    account_family_names,
    contains_form,
    fact_gist,
    natural_date_options,
    overlap_tokens,
    resolve_relative_dates,
    short_forms,
    unmarked_text,
    unscripted_signals,
    vendor_names,
    voice_failures,
)
from portolan.extract.quote_gate import _fold_glyphs
from portolan.extract.schema import FactType
from portolan.training.overlap_gate import numerals_and_names_covered, tokenize_claim
from portolan.training.quote_span import resolve_quote_span

__all__ = [
    "CHECKS_VERSION",
    "DECOY_FORBIDDEN_PATTERNS",
    "FORBIDDEN_PATTERN_FAILURE",
    "MAX_SPAN_WORDS",
    "DecoySpan",
    "DocVerification",
    "SlotResult",
    "SlotSpec",
    "WriterLabel",
    "LABEL_SOURCE_WRITER",
    "build_slot_specs",
    "COPY_RUN_WORDS",
    "claim_copy_run",
    "copied_elsewhere",
    "coverage_extras",
    "doc_level_problems",
    "entity_words_for",
    "forbidden_pattern_hits",
    "is_claim_copy",
    "locator_pattern_for",
    "make_labels",
    "pasted_sentence",
    "required_terms_for",
    "unscripted_in",
    "verify_document",
]

MAX_SPAN_WORDS = 60  # spec section 3.2 check 2
# Part of every document's resume fingerprint (`writer.input_fingerprint`):
# bump it when a check changes, so a resumed document is re-verified.
CHECKS_VERSION = "wp4-checks-2026-09-23.11-decoy-patterns"

# `DecoyClass.forbidden_patterns` ids -> what check 5 refuses in a decoy
# span. `competitor_evaluation`: a negated competitor statement ("we're not
# looking at another vendor", "we stuck with what we have") is a
# `competitor_mention` fact per the production prompt (decisions log,
# 2026-09-23), so a decoy that says one is not a decoy.
DECOY_FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "competitor_evaluation": re.compile(
        r"\b(?:competitors?|competing|competition|rfps?|bake[- ]?offs?|shortlist(?:ed|ing)?)\b"
        r"|\b(?:another|other|different|alternative|outside|current)\s+(?:[\w'-]+\s+){0,2}?"
        r"(?:vendors?|suppliers?|providers?|platforms?|systems?|solutions?|products?|options?"
        r"|partners?|marketplaces?|tools?|compan(?:y|ies)|brands?|offerings?)\b"
        r"|\bswitch(?:ing|ed|es)?\s+(?:over\s+)?(?:to|from|away)\b"
        r"|\b(?:evaluat(?:e|ed|es|ing|ion)|trial(?:ed|ing|led|ling)|piloted|piloting)\b"
        r"|\b(?:rul(?:e|ed|es|ing)\s+(?:\w+\s+)?out|stick(?:ing)?\s+with|stuck\s+with"
        r"|stay(?:ing|ed)?\s+with|pass(?:ed|ing)?\s+on|decid(?:e|ed|ing)\s+against"
        r"|went\s+with|go(?:ing)?\s+with|opt(?:ed|ing)\s+for)\b"
        r"|(?:\bnot|n't|\bnever)\s+(?:\w+\s+)?(?:using|buying|considering|looking\s+(?:at|into))\b",
        re.IGNORECASE,
    ),
}


# The check-5 failure code for a decoy span that matches one of its class's
# forbidden patterns; WP5 reads it to know the sentence is a fact, not a decoy.
FORBIDDEN_PATTERN_FAILURE = "c5:forbidden_pattern"


def forbidden_pattern_hits(text: str, pattern_ids: tuple[str, ...] | list[str]) -> list[str]:
    """The `DECOY_FORBIDDEN_PATTERNS` ids `text` matches (an unknown id matches
    nothing: the pack schema's closed vocabulary rejects it first)."""
    folded = _fold_glyphs(text)
    return [
        pid
        for pid in pattern_ids
        if pid in DECOY_FORBIDDEN_PATTERNS and DECOY_FORBIDDEN_PATTERNS[pid].search(folded)
    ]


# A span must carry at least this many words that are not names: a label is
# a statement, not a list of the entities it mentions.
MIN_STATEMENT_WORDS = 3

# Label source for a scripted fact the writer planted and code verified --
# the writer-side counterpart of WP3's `scripted_code` (CRM field labels)
# and WP5's `silver_completeness` (spec section 3.4).
LABEL_SOURCE_WRITER = "scripted_writer"


class WriterLabel(LabelRecord):
    """A `LabelRecord` plus the writer-side provenance WP5 and WP7 need.
    `doc_id` is the id of the SOURCE the span indexes (an email message's
    own id; the plan id for every single-file kind), and `char_start`/
    `char_end` index that source's `prepare_text` output."""

    plan_id: str
    slot_id: str
    speaker_person_id: str | None = None
    reference_mode: str = ReferenceMode.EXPLICIT.value
    writer_model: str
    prepared_locator: str
    span_words: int


class DecoySpan(BaseModel):
    """One verified decoy span (spec section 3.3: WP5 counts a teacher fact
    that overlaps a decoy as a decoy hit, per decoy class)."""

    model_config = ConfigDict(extra="forbid")

    decoy_id: str
    decoy_class_id: str
    near_type: FactType
    plan_id: str
    slot_id: str
    doc_id: str
    source_type: str
    locator: str
    prepared_locator: str
    char_start: int
    char_end: int
    text: str
    account_id: str
    writer_model: str


# --------------------------------------------------------------------------
# Slot specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSpec:
    """Everything the prompt and the checks need about one slot."""

    slot_id: str
    marker_id: str
    is_fact: bool
    speaker_person_id: str | None
    # What the span itself must carry or resolve (product, competitor,
    # person). Never the account: the speaker says "we" or "you" for it.
    anchors: tuple[str, ...]
    reference_mode: ReferenceMode
    reference_anchor: str | None  # the anchor a non-explicit reference stands in for
    reference_form: str | None  # what the span says instead ("Priya", "@pnatarajan", ...)
    claim: str | None
    fact_type: FactType | None
    fact_date: date | None
    account_id: str
    # Check-6 terms that are not party names ("ICU", "QBR", "15%"); they may
    # sit inside or outside the span.
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    directive: str | None = None
    decoy_class_id: str | None = None
    near_type: FactType | None = None
    expected_count: int = 1
    entity_words: frozenset[str] = frozenset()
    # Party names check 6 needs somewhere in the source, outside the span.
    party_terms: tuple[str, ...] = ()
    account_name: str | None = None
    # Nicknames per anchor (`voice.short_forms`).
    anchor_forms: tuple[tuple[str, tuple[ShortForm, ...]], ...] = ()
    # The side the speaker must be on ("customer" / "vendor"); None: either.
    expected_side: str | None = None
    gist: str | None = None
    date_options: tuple[str, ...] = ()
    date_reference: date | None = None  # the date the unit stating it carries
    forbidden_patterns: tuple[str, ...] = ()  # decoys: `DECOY_FORBIDDEN_PATTERNS` ids

    def forms_for(self, anchor: str) -> tuple[ShortForm, ...]:
        return next((forms for a, forms in self.anchor_forms if a == anchor), ())


_PERSON_PRONOUN = "they"


def _account_family(world: World, account_id: str) -> set[str]:
    accounts = world.account_by_id()
    family = {account_id}
    acct = accounts.get(account_id)
    if acct is not None and acct.parent:
        family.add(acct.parent)
    family.update(a.id for a in world.accounts if a.parent == account_id)
    return family


def _other_account_names(world: World, account_id: str) -> list[str]:
    family = _account_family(world, account_id)
    return [a.name for a in world.accounts if a.id not in family]


def _reference_target(
    fact: Fact, world: World, kind: DocKind
) -> tuple[ReferenceMode, str | None, str | None]:
    mode = fact.reference_mode
    if mode == ReferenceMode.EXPLICIT:
        return mode, None, None
    people = world.person_by_id()
    person: Person | None = next((people[e] for e in fact.entities if e in people), None)
    if mode == ReferenceMode.HANDLE and kind == DocKind.CALL:
        mode = ReferenceMode.FIRST_NAME  # a handle is not spoken aloud
    if mode in (ReferenceMode.FIRST_NAME, ReferenceMode.TITLE, ReferenceMode.HANDLE):
        if person is None:
            return ReferenceMode.EXPLICIT, None, None
        form = {
            ReferenceMode.FIRST_NAME: person.name.split()[0],
            ReferenceMode.TITLE: person.title,
            ReferenceMode.HANDLE: person.chat_handle,
        }[mode]
        return mode, person.name, form
    if mode == ReferenceMode.ALIAS:
        # The account is never a span anchor (the speaker says "we" or
        # "you"), so an alias reference is to a product.
        products = {p.id: p for p in world.vendor.products}
        for e in fact.entities:
            if e in products:
                forms = short_forms(products[e].name, world)
                if forms:
                    return mode, products[e].name, forms[0].text
        return ReferenceMode.EXPLICIT, None, None
    if mode == ReferenceMode.PRONOUN and person is not None:
        return mode, person.name, _PERSON_PRONOUN
    return ReferenceMode.EXPLICIT, None, None


_CAP_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _strip_possessive(word: str) -> str:
    low = word.lower()
    if len(word) > 2 and low.endswith("'s"):
        return word[:-2]
    if len(word) > 2 and low.endswith("s'"):
        return word[:-1]
    return word


_DIGIT_POSSESSIVE_RE = re.compile(r"(\d)'s\b")


_NAME_COMPOUND_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*|\w*\d\w*)-([a-z]{3,})\b")


def rule_claim(claim: str) -> str:
    """The claim as check 6 reads it: a possessive after a digit ("Hospital
    2's") is the digit alone, since the numeral rule compares digit
    sequences and the text says "Hospital 2" as often as "Hospital 2's";
    and a name with a lowercase suffix ("T2-related", "Care-specific") is
    the name plus a plain word, since the text names the product, not the
    compound."""
    return _NAME_COMPOUND_RE.sub(r"\1 \2", _DIGIT_POSSESSIVE_RE.sub(r"\1", claim))


def required_terms_for(
    claim: str,
    anchors: list[str],
    doc_date: date,
    fact_date: date,
    *,
    names: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Strings the numeral-and-name rule (check 6) needs somewhere in the
    document that the span anchors do not already supply: a known full name
    from `names` whenever the rule needs any of its words ("Marrowbrook
    Community Hospital 2", "Harborlight Health Systems"), other proper
    names and acronyms ("QBR"), and numbers. Dates are left out: the
    prompt gives the date as people say it (`voice.natural_date_options`)
    and check 6 resolves that phrase back."""
    del doc_date, fact_date  # dates are phrased by the prompt's When line
    claim = rule_claim(claim)
    tokens = tokenize_claim(claim)
    anchor_words = {w.lower() for a in anchors for w in _CAP_WORD_RE.findall(_fold_glyphs(a))}
    needed_names = {n for n in tokens.names if n not in anchor_words}
    needed_numerals = [n for n in tokens.numerals if n not in anchor_words]
    out: list[str] = []
    rest = claim  # the claim with every known name cut out, for the phrase pass
    for name in sorted(set(names), key=len, reverse=True):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?:'s)?(?![A-Za-z0-9])"
        if not re.search(pattern, rest):
            continue
        rest = re.sub(pattern, " , ", rest)
        words = {_strip_possessive(w).lower() for w in _CAP_WORD_RE.findall(_fold_glyphs(name))}
        if words & (needed_names | set(needed_numerals)):
            out.append(name)
            needed_names -= words
            needed_numerals = [n for n in needed_numerals if n not in words]
    words = _CAP_WORD_RE.findall(_fold_glyphs(rest))
    run: list[str] = []

    def _flush() -> None:
        # A sentence-initial "The" or "After" is not part of the name.
        while run and _strip_possessive(run[0]).lower() not in needed_names:
            run.pop(0)
        if run and any(_strip_possessive(w).lower() in needed_names for w in run):
            phrase = " ".join([*run[:-1], _strip_possessive(run[-1])])
            if phrase not in out and not any(phrase in o for o in out):
                out.append(phrase)
        run.clear()

    for w in words:
        if w[0].isupper():
            run.append(w)
        else:
            _flush()
    _flush()
    for num in needed_numerals:
        if _ISO_DATE_RE.match(num):
            continue
        # The whole number, never a digit run inside a date ("20" is not in
        # "2026-02-07").
        whole = re.compile(r"[^\d]*" + re.escape(num) + r"[^\d]*")
        raw = next(
            (w for w in re.findall(r"\S+", claim) if whole.fullmatch(w.replace(",", ""))), num
        )
        raw = raw.rstrip(".,;:")
        if raw not in out and not any(raw in o.split() for o in out):
            out.append(raw)
    return out


def _expected_side(fact: Fact, world: World) -> str:
    """The side a fact's speaker is on: the planned speaker's side, or for
    an unplanned speaker the vendor for a commitment (a promise the vendor
    makes) and the customer for everything else (WP1's `speaker_for`)."""
    speaker = world.person_by_id().get(fact.speaker)
    if speaker is not None:
        return speaker.side.value
    if fact.fact_type == FactType.COMMITMENT:
        return PersonSide.VENDOR.value
    return PersonSide.CUSTOMER.value


def build_slot_specs(
    rendered: RenderedDoc,
    plan: DocPlan,
    world: World,
    decoys_by_id: dict[str, PlannedDecoy],
) -> list[SlotSpec]:
    facts = {f.id: f for f in world.facts}
    accounts = world.account_by_id()
    competitors = [c.name for c in world.vendor.competitors]
    entity_words = entity_words_for(world)
    people = world.person_by_id()
    known_names = [
        world.vendor.name,
        *(p.name for p in world.vendor.products),
        *competitors,
        *(a.name for a in world.accounts),
    ]
    specs: list[SlotSpec] = []
    for slot in rendered.slots:
        if slot.fact_id is not None:
            fact = facts[slot.fact_id]
            mode, ref_anchor, ref_form = _reference_target(fact, world, plan.kind)
            forbidden = _other_account_names(world, fact.account_id)
            if fact.fact_type != FactType.COMPETITOR_MENTION:
                forbidden += [c for c in competitors if c not in fact.anchors]
            family = set(account_family_names(world, fact.account_id))
            span_anchors = [a for a in fact.anchors if a not in family]
            names = [*known_names, *(people[e].name for e in fact.entities if e in people)]
            terms = required_terms_for(fact.claim, span_anchors, plan.date, fact.date, names=names)
            account_name = accounts[fact.account_id].name
            party = {account_name, *family, *vendor_names(world)}
            # An email message is its own source and is dated on its fact
            # (`assembly.unit_dates`), so "today" means the fact's day. A
            # ticket's comments share the ticket's date as their source
            # date, so a later comment names its date outright.
            reference = fact.date if plan.kind == DocKind.EMAIL else plan.date
            relative = plan.kind != DocKind.TICKET or fact.date == plan.date
            party_terms = [account_name, *(t for t in terms if t in party and t != account_name)]
            specs.append(
                SlotSpec(
                    slot_id=slot.slot_id,
                    marker_id=fact.id,
                    is_fact=True,
                    speaker_person_id=slot.speaker_person_id,
                    anchors=tuple(span_anchors),
                    reference_mode=mode,
                    reference_anchor=ref_anchor,
                    reference_form=ref_form,
                    claim=fact.claim,
                    fact_type=fact.fact_type,
                    fact_date=fact.date,
                    account_id=fact.account_id,
                    required_terms=tuple(t for t in terms if t not in party),
                    forbidden_terms=tuple(forbidden),
                    directive=fact.directive,
                    entity_words=entity_words,
                    party_terms=tuple(party_terms),
                    account_name=account_name,
                    anchor_forms=tuple((a, short_forms(a, world)) for a in span_anchors),
                    expected_side=_expected_side(fact, world),
                    gist=fact_gist(fact.claim, world, fact.account_id),
                    date_options=tuple(
                        natural_date_options(fact.date, reference, relative=relative)
                    ),
                    date_reference=reference,
                )
            )
        elif slot.decoy_id is not None:
            decoy = decoys_by_id[slot.decoy_id]
            account_id = decoy.account_id or (plan.account_id or "")
            forbidden = _other_account_names(world, account_id) if account_id else []
            if decoy.near_type != FactType.COMPETITOR_MENTION:
                forbidden += competitors
            specs.append(
                SlotSpec(
                    slot_id=slot.slot_id,
                    marker_id=decoy.id,
                    is_fact=False,
                    speaker_person_id=slot.speaker_person_id,
                    anchors=(),
                    reference_mode=ReferenceMode.EXPLICIT,
                    reference_anchor=None,
                    reference_form=None,
                    claim=None,
                    fact_type=None,
                    fact_date=None,
                    account_id=account_id,
                    required_terms=(),
                    forbidden_terms=tuple(forbidden),
                    decoy_class_id=decoy.decoy_class_id,
                    near_type=decoy.near_type,
                    entity_words=entity_words,
                    forbidden_patterns=tuple(decoy.forbidden_patterns),
                )
            )
    return specs


# --------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------


def _norm(s: str) -> str:
    folded = _fold_glyphs(unicodedata.normalize("NFKC", s))
    return re.sub(r"\s+", " ", folded.casefold()).strip()


def _contains_term(haystack_norm: str, term: str) -> bool:
    needle = _norm(term)
    if not needle:
        return True
    pattern = r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])"
    return re.search(pattern, haystack_norm) is not None


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


# c5 catches a near-verbatim copy of the claim: a contiguous run of this
# many non-entity words, or a sentence of about the claim's length carrying
# `COPY_SHARE` of its words in order. Entity names repeat freely. The bars
# were 6 words and 80% for the short Claude-written templates; WP2's
# open-weight templates are full sentences a speaker restates naturally
# ("... considering Knollwell Biomedical for Alderline Link replacement due
# to ongoing issues" failed at 6), and c12 judges claim shape.
COPY_RUN_WORDS = 10


def _longest_common_run(
    a: list[str], b: list[str], entity_words: frozenset[str]
) -> tuple[int, int]:
    """`(longest shared contiguous run, most non-entity words in any shared
    run)` between token lists `a` and `b`."""
    best = best_ne = 0
    prev = [0] * (len(b) + 1)
    prev_ne = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        cur_ne = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                cur_ne[j] = prev_ne[j - 1] + (0 if a[i - 1] in entity_words else 1)
                best = max(best, cur[j])
                best_ne = max(best_ne, cur_ne[j])
        prev, prev_ne = cur, cur_ne
    return best, best_ne


def entity_words_for(world: World) -> frozenset[str]:
    """Every token of every name in the world: vendor, products and their
    aliases, competitors, accounts and their aliases, people."""
    names: list[str] = [world.vendor.name]
    for p in world.vendor.products:
        names += [p.name, *p.aliases]
    for c in world.vendor.competitors:
        names += [c.name, *c.aliases]
    for a in world.accounts:
        names += [a.name, *a.aliases]
    names += [p.name for p in world.people]
    return frozenset(t for n in names for t in _TOKEN_RE.findall(_norm(n)))


def claim_copy_run(
    span: str, claim: str, entity_words: frozenset[str] = frozenset()
) -> tuple[int, int]:
    """`(longest shared word run, non-entity words in the longest-by-content
    run)` between a text and the code-written claim. Copying the claim's
    wording puts code-written text into a training document through the
    writer (spec Build notes, WP2 review)."""
    a = _TOKEN_RE.findall(_norm(span))
    b = _TOKEN_RE.findall(_norm(claim))
    return _longest_common_run(a, b, entity_words)


COPY_SHARE = 0.85
COPY_MIN_WORDS = 4
# A sentence this much longer than the claim adds its own content: not a copy.
COPY_MAX_LENGTH_RATIO = 1.3
_COPY_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _content_tokens(text: str, entity_words: frozenset[str]) -> list[str]:
    tokens = [re.sub(r"'s$", "", t) for t in _TOKEN_RE.findall(_norm(text))]
    return [t for t in tokens if t and t not in entity_words]


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def is_claim_copy(span: str, claim: str, entity_words: frozenset[str] = frozenset()) -> bool:
    """True when `span` reuses the code-written claim's wording: a
    contiguous run of `COPY_RUN_WORDS` non-entity claim words, or any one
    sentence of `span` that carries `COPY_SHARE` of the claim's non-entity
    words in order. Entity names never count; they must repeat."""
    _run, non_entity = claim_copy_run(span, claim, entity_words)
    if non_entity >= COPY_RUN_WORDS:
        return True
    claim_tokens = _content_tokens(claim, entity_words)
    if len(claim_tokens) < COPY_MIN_WORDS:
        return False
    need = max(COPY_MIN_WORDS, -(-int(COPY_SHARE * 100) * len(claim_tokens) // 100))
    for sentence in _COPY_SENTENCE_RE.split(span):
        tokens = _content_tokens(sentence, entity_words)
        if len(tokens) > COPY_MAX_LENGTH_RATIO * len(claim_tokens):
            continue
        if _lcs_len(tokens, claim_tokens) >= need:
            return True
    return False


# Generator internals that must never reach a shipped document: slot tags,
# marker glyphs, fact/decoy/event ids, entity ids, directive labels.
META_RE = re.compile(
    r"\[\s*SLOT\b|\bSLOT\s+(?:fact|decoy|color)|fact_id|decoy_id|decoy_class|near_type|"
    r"directive\s*=|\b[FDE]\d{5}[a-z]?\b|\b(?:person|acct|product|competitor)-[a-z]|"
    r"hard[- ]negative|\bdecoys?\b|training corpus|[⟨⟩⟦⟧〚〛]",
    re.IGNORECASE,
)


def _repeated_units(assembled: AssembledDoc, times: int = 4) -> str | None:
    """A unit text of three or more words that occurs `times` or more
    times: the writer fell into a loop (observed live 2026-09-22 from
    glm-4.7-flash on a call). Structural, so fatal; code never trims it."""
    counts: dict[str, int] = {}
    for u in assembled.units:
        key = " ".join(u.clean_text.lower().split())
        if len(key.split()) >= 3:
            counts[key] = counts.get(key, 0) + 1
    worst = max(counts.items(), key=lambda kv: kv[1], default=None)
    if worst is not None and worst[1] >= times:
        return worst[0][:80]
    return None


def pasted_sentence(text: str, others: list[str], min_words: int = 8) -> str | None:
    """A sentence of `text` (`min_words` or more words) that appears word
    for word in one of `others` -- a repair reply that pasted another
    unit's words (seen live 2026-09-22: two sentences of an earlier turn
    reappeared in the repaired turn). Not a document-wide rule: real email
    threads repeat signatures and closings."""
    keys = {
        " ".join(_TOKEN_RE.findall(_norm(s)))
        for other in others
        for s in re.split(r"(?<=[.!?])\s+|\n+", strip_all_markers(other))
    }
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", strip_all_markers(text)):
        key = " ".join(_TOKEN_RE.findall(_norm(sentence)))
        if len(key.split()) >= min_words and key in keys:
            return sentence.strip()[:80]
    return None


def doc_level_problems(
    assembled: AssembledDoc, known_marker_ids: set[str]
) -> tuple[list[str], list[str]]:
    """`(fatal, notes)` for the document as a whole. Fatal: generator
    internals leaked into the writer's text (never shipped). Notes:
    markers with an id not in this document's script (their glyphs are
    stripped; the text stays)."""
    fatal: list[str] = []
    notes: list[str] = []
    leak = META_RE.search(assembled.writer_text())
    if leak:
        fatal.append(f"generator internals in the text: {leak.group(0)!r}")
    repeated = _repeated_units(assembled)
    if repeated:
        fatal.append(f"degenerate repetition: {repeated!r}")
    stray = sorted(
        {
            s.marker_id
            for u in assembled.units
            for s in u.spans
            if s.marker_id not in known_marker_ids
        }
        | {m for u in assembled.units for m in u.unbalanced if m not in known_marker_ids}
    )
    if stray:
        notes.append(f"markers not in the script (stripped): {stray}")
    return fatal, notes


# --------------------------------------------------------------------------
# Check 6 coverage: the source date and date / number spellings
# --------------------------------------------------------------------------

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_DAY_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", re.I
)
_DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALT})\b\.?(?:\s+(\d{{4}}))?", re.I
)
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}  # fmt: skip
_NUMBER_PHRASE_RE = re.compile(
    r"\b(" + "|".join(_NUMBER_WORDS) + r")(?:[\s-]+(" + "|".join(_NUMBER_WORDS) + r"))?\b", re.I
)


def _safe_iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _nearest_iso(month: int, day: int, source_date: date) -> str | None:
    """A month and day with no year, read in the year that puts it nearest
    the source date ("November 24" in a February document is last
    November)."""
    candidates = []
    for year in (source_date.year - 1, source_date.year, source_date.year + 1):
        iso = _safe_iso(year, month, day)
        if iso:
            candidates.append(date.fromisoformat(iso))
    if not candidates:
        return None
    return min(candidates, key=lambda d: abs((d - source_date).days)).isoformat()


def coverage_extras(text: str, source_date: date) -> str:
    """Extra tokens check 6 counts as covered: the source's own date (the
    extractor receives it as `source_date`), ISO forms of dates the text
    spells out ("January 20", "20th of Jan", "1/20") or says relative to
    the source date ("yesterday", "last Tuesday", "the 28th"), and digit
    forms of spelled-out numbers ("fifteen" -> 15). Never words the text
    lacks: each extra is a normalization of something the text or its
    envelope already carries."""
    extras = [source_date.isoformat(), *resolve_relative_dates(text, source_date)]
    for m in _MONTH_DAY_RE.finditer(text):
        month, day = _MONTHS[m.group(1).lower()], int(m.group(2))
        iso = (
            _safe_iso(int(m.group(3)), month, day)
            if m.group(3)
            else _nearest_iso(month, day, source_date)
        )
        if iso:
            extras.append(iso)
    for m in _DAY_MONTH_RE.finditer(text):
        month, day = _MONTHS[m.group(2).lower()], int(m.group(1))
        iso = (
            _safe_iso(int(m.group(3)), month, day)
            if m.group(3)
            else _nearest_iso(month, day, source_date)
        )
        if iso:
            extras.append(iso)
    for m in _SLASH_DATE_RE.finditer(text):
        month, day = int(m.group(1)), int(m.group(2))
        if m.group(3):
            year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
            iso = _safe_iso(year, month, day)
        else:
            iso = _nearest_iso(month, day, source_date)
        if iso:
            extras.append(iso)
    for m in _NUMBER_PHRASE_RE.finditer(text):
        first = _NUMBER_WORDS[m.group(1).lower()]
        second = _NUMBER_WORDS[m.group(2).lower()] if m.group(2) else None
        extras.append(str(first))
        if second is not None:
            extras.append(str(second))
            if first >= 20 and second < 10:
                extras.append(str(first + second))
    return " ".join(extras)


# --------------------------------------------------------------------------
# Locator patterns (check 7)
# --------------------------------------------------------------------------

_V2_TICKET_RE = re.compile(r"^S-[\w\-]+#\d+$")


def locator_pattern_for(kind: DocKind, world: World) -> re.Pattern[str]:
    if kind == DocKind.EMAIL:
        return re.compile(rf"^\S+@{re.escape(world.vendor.message_id_domain)}#p\d+$")
    if kind == DocKind.TICKET:
        return _V2_TICKET_RE
    return LOCATOR_PATTERNS[kind.value]


# --------------------------------------------------------------------------
# Per-slot verification
# --------------------------------------------------------------------------


@dataclass
class FoundSpan:
    unit: AssembledUnit
    marker: MarkerSpan
    text: str
    cell: Cell | None
    prepared: PreparedSource | None
    prepared_start: int | None = None
    prepared_end: int | None = None


@dataclass
class SlotResult:
    spec: SlotSpec
    failures: list[str] = field(default_factory=list)
    found: FoundSpan | None = None
    # every unit index this slot's marker tokens appear in (repair targeting)
    marker_units: list[int] = field(default_factory=list)
    # units that state this fact again, unmarked (repair removes it there)
    repeat_units: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class DocVerification:
    results: list[SlotResult]
    fatal: list[str]
    notes: list[str]
    # Check 14: sentences outside the marked spans that likely state a fact
    # nobody scripted (`voice.unscripted_signals`).
    unscripted: list[str] = field(default_factory=list)

    def result_for(self, slot_id: str) -> SlotResult:
        return next(r for r in self.results if r.spec.slot_id == slot_id)

    @property
    def facts_passed(self) -> int:
        return sum(1 for r in self.results if r.spec.is_fact and r.passed)

    @property
    def decoys_passed(self) -> int:
        return sum(1 for r in self.results if not r.spec.is_fact and r.passed)

    def score(self) -> tuple[int, int, int, int]:
        """Ordering key for "which attempt is better": no fatal problem
        first, then verified facts, then fewer unscripted sentences, then
        verified decoys. Facts come first: an unverified fact's text stays
        in the document unlabeled, so trading labels for fewer unscripted
        sentences only moves the problem (live 2026-09-22: a markerless
        regeneration outranked a draft with 6 of 7 facts verified)."""
        return (
            0 if self.fatal else 1,
            self.facts_passed,
            -len(self.unscripted),
            self.decoys_passed,
        )


def _nth_occurrence(
    haystack: str, needle: str, n: int, start: int = 0, end: int | None = None
) -> int:
    end = len(haystack) if end is None else end
    pos = start
    for _ in range(n + 1):
        idx = haystack.find(needle, pos, end)
        if idx == -1:
            return -1
        found, pos = idx, idx + 1
    return found


def _cell_for(unit: AssembledUnit, marker: MarkerSpan) -> Cell | None:
    for cell in unit.cells:
        if cell.start <= marker.start and marker.end <= cell.end:
            return cell
    return None


def _locate_in_prepared(found: FoundSpan) -> tuple[int, int] | None:
    cell, prepared = found.cell, found.prepared
    if cell is None or prepared is None or not found.text:
        return None
    region = prepared.region(cell.prepared_locator)
    if region is None:
        return None
    cell_text = found.unit.clean_text[cell.start : cell.end]
    rel = found.marker.start - cell.start
    j = 0
    idx = cell_text.find(found.text)
    while idx != -1 and idx < rel:
        j += 1
        idx = cell_text.find(found.text, idx + 1)
    pos = _nth_occurrence(prepared.text, found.text, j, region[0], region[1])
    if pos == -1:
        return None
    return pos, pos + len(found.text)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def copied_elsewhere(verification: DocVerification, assembled: AssembledDoc) -> list[str]:
    """Fact ids whose script wording an UNMARKED unit repeats: an unlabeled
    restatement, which WP5's completeness pass labels as a second locator
    (spec section 3.3, "Restatement")."""
    out: list[str] = []
    for r in verification.results:
        if not r.spec.claim:
            continue
        # A verified span's own unit is its label; a failed span's text stays
        # in the document unlabeled, so its unit counts too.
        marked = r.found.unit.index if (r.found is not None and r.passed) else None
        if any(
            u.index != marked and is_claim_copy(u.clean_text, r.spec.claim, r.spec.entity_words)
            for u in assembled.units
        ):
            out.append(r.spec.marker_id)
    return sorted(out)


def _stated_twice(text: str, assembled: AssembledDoc, own_unit: int | None = None) -> list[int]:
    """Units that say the marked sentence again, unmarked: a near-copy
    (difflib ratio >= 0.85), or a statement carrying most of the span's
    content words (a second telling in other words; WP5's blind pass found
    writers state a fact twice, which halves the label's recall). Questions
    are exempt: the turn before a two-turn fact asks about it."""
    target = _norm(text)
    target_tokens = overlap_tokens(text)
    if len(target.split()) < 6:
        return []
    out: list[int] = []
    for u in assembled.units:
        # Only UNMARKED text is compared: another slot's marked sentence can
        # share this one's anchors without restating it.
        body = unmarked_text(u.clean_text, [(sp.start, sp.end) for sp in u.spans])
        for sentence in _SENTENCE_SPLIT_RE.split(body):
            cand = _norm(sentence)
            if len(cand.split()) < 6 or sentence.rstrip().endswith("?"):
                continue
            shared = target_tokens & overlap_tokens(sentence)
            near_copy = difflib.SequenceMatcher(None, target, cand).ratio() >= 0.85
            retold = len(shared) >= 4 and len(shared) >= RESTATED_SHARE * len(target_tokens)
            if near_copy or retold:
                out.append(u.index)
                break
    return out


# Share of the marked span's content words an unmarked sentence must carry
# to count as telling the same fact again.
RESTATED_SHARE = 0.6


def _earlier_text(found: FoundSpan, assembled: AssembledDoc) -> str:
    if found.prepared is not None and found.prepared_start is not None:
        return found.prepared.text[: found.prepared_start]
    parts = [
        u.clean_text
        for u in assembled.units
        if u.source_index == found.unit.source_index and u.index < found.unit.index
    ]
    parts.append(found.unit.clean_text[: found.marker.start])
    return "\n".join(parts)


def _prior_turn_text(found: FoundSpan, assembled: AssembledDoc) -> str:
    """The unit just before the span's unit, in the same source: where a
    fact spread over two turns names what the marked answer refers back
    to. Never the span's own unit: there the name belongs inside the marked
    sentence (seen live 2026-09-22: "Chenping Clayland ... has left." then
    a marked "We have not yet identified a replacement champion")."""
    if assembled.kind not in SPEAKER_KINDS:
        return ""  # a doc paragraph is not a turn: its fact names what it is about
    prev = [
        u.clean_text
        for u in assembled.units
        if u.source_index == found.unit.source_index and u.index == found.unit.index - 1
    ]
    return "\n".join(prev)


def _verify_slot(
    spec: SlotSpec,
    assembled: AssembledDoc,
    prepared: list[PreparedSource],
    occurrences: dict[str, list[tuple[AssembledUnit, MarkerSpan]]],
    unbalanced: dict[str, list[int]],
    overlapping: set[str],
    world: World,
) -> SlotResult:
    result = SlotResult(spec)
    occ = occurrences.get(spec.marker_id, [])
    result.marker_units = sorted(
        {u.index for u, _ in occ} | set(unbalanced.get(spec.marker_id, []))
    )

    # check 1 -- exactly one marker pair (or the designed count)
    if spec.marker_id in unbalanced:
        result.failures.append("c1:unbalanced_marker (an opener or closer without its pair)")
    if not occ and spec.marker_id not in unbalanced:
        result.failures.append("c1:missing_marker")
    elif occ and len(occ) != spec.expected_count:
        result.failures.append(f"c1:marker_count={len(occ)} (expected {spec.expected_count})")
    if not occ:
        return result

    unit, marker = occ[0]
    text = unit.clean_text[marker.start : marker.end]
    cell = _cell_for(unit, marker)
    source = (
        prepared[cell.source_index]
        if cell is not None and cell.source_index < len(prepared)
        else None
    )
    found = FoundSpan(unit, marker, text, cell, source)
    result.found = found
    result.repeat_units = _stated_twice(text, assembled)
    if result.repeat_units:
        result.failures.append(
            f"c1:stated_twice (said again, unmarked, in unit {result.repeat_units[0]})"
        )

    # check 2 -- one locator unit, at most 60 words
    words = len(text.split())
    if not text.strip():
        result.failures.append("c2:empty_span")
    if "\n" in text or cell is None:
        result.failures.append("c2:span_crosses_a_line_or_paragraph")
    if words > MAX_SPAN_WORDS:
        result.failures.append(f"c2:span_too_long ({words} words > {MAX_SPAN_WORDS})")
    if spec.marker_id in overlapping:
        result.failures.append("c2:span_overlaps_another_marker")
    if text.strip() and len(_content_tokens(text, spec.entity_words)) < MIN_STATEMENT_WORDS:
        # Seen live 2026-09-22: a writer wrapped only the must-contain names
        # ("⟦F…⟧Willowwood General Hospital 2, Chenping Clayland⟦/F…⟧") and
        # left the real sentence unmarked. Every other check passed.
        result.failures.append("c2:span_states_nothing (names only, no statement)")

    # check 8 is located first: check 3's "earlier in the document" is
    # measured in the prepared text the extractor actually sees.
    located = _locate_in_prepared(found)
    if located is not None:
        found.prepared_start, found.prepared_end = located

    # check 3 -- anchors. A span names each anchor in full, or by a
    # nickname / reference form with the full name earlier in the source,
    # or refers back to it when the turn just before names it (a fact
    # spread over two turns). The account is never an anchor.
    span_norm = _norm(text)
    earlier_norm = _norm(_earlier_text(found, assembled))
    prior_raw = _prior_turn_text(found, assembled)
    prior_norm = _norm(prior_raw)
    for anchor in spec.anchors:
        if _contains_term(span_norm, anchor):
            continue
        full_earlier = _contains_term(earlier_norm, anchor)
        if spec.reference_anchor == anchor:
            if not full_earlier:
                result.failures.append(
                    f"c3:reference_unresolvable ({anchor!r} not earlier in the text)"
                )
            continue
        forms = spec.forms_for(anchor)
        short_in_span = any(contains_form(text, f) for f in forms)
        prior_mention = _contains_term(prior_norm, anchor) or any(
            contains_form(prior_raw, f) for f in forms
        )
        if full_earlier and (short_in_span or prior_mention):
            continue
        if short_in_span:
            result.failures.append(
                f"c3:reference_unresolvable ({anchor!r} is not named in full earlier)"
            )
        else:
            result.failures.append(
                f"c3:anchor_missing ({anchor!r} not in the span or the turn before it)"
            )

    # check 4 -- speaker, and the speaker's side
    people = world.person_by_id()
    author = people.get(unit.author_person_id or "")
    if (
        assembled.kind in SPEAKER_KINDS
        and spec.speaker_person_id is not None
        and unit.author_person_id != spec.speaker_person_id
    ):
        want = people[spec.speaker_person_id].name if spec.speaker_person_id in people else "?"
        result.failures.append(
            f"c4:wrong_speaker (said by {unit.author_name!r}, expected {want!r})"
        )
    if (
        assembled.kind in SPEAKER_KINDS
        and spec.expected_side is not None
        and author is not None
        and author.side.value != spec.expected_side
    ):
        result.failures.append(
            f"c4:speaker_side ({author.name!r} is on the {author.side.value} side; a "
            f"{spec.expected_side}-side person says this)"
        )

    # check 5 -- forbidden words
    leak = META_RE.search(text)
    if leak:
        result.failures.append(f"c5:forbidden_word ({leak.group(0)!r})")
    for term in spec.forbidden_terms:
        if _contains_term(span_norm, term):
            result.failures.append(f"c5:forbidden_word ({term!r})")
    for pattern_id in forbidden_pattern_hits(text, spec.forbidden_patterns):
        result.failures.append(f"{FORBIDDEN_PATTERN_FAILURE} ({pattern_id})")
    if spec.claim and is_claim_copy(text, spec.claim, spec.entity_words):
        result.failures.append("c5:copies_the_script_wording")

    # checks 9 to 13 -- the span sounds like the person saying it
    other_names = (
        *spec.forbidden_terms,
        *(a for a in spec.anchors if a not in {p.name for p in world.vendor.products}),
    )
    result.failures.extend(
        voice_failures(
            text=text,
            kind=assembled.kind,
            world=world,
            speaker=author,
            account_ids=[spec.account_id] if spec.account_id else [],
            claim=spec.claim,
            entity_words=spec.entity_words,
            other_names=tuple(other_names),
        )
    )

    # check 6 -- numeral-and-name rule (facts only)
    if spec.is_fact and spec.claim is not None and source is not None:
        coverage = source.text + "\n" + coverage_extras(source.text, source.date)
        numerals_ok, names_ok = numerals_and_names_covered(rule_claim(spec.claim), coverage)
        if not numerals_ok:
            result.failures.append("c6:numeral_not_in_text")
        if not names_ok:
            result.failures.append("c6:name_not_in_text")
        if spec.account_name and not _contains_term(_norm(source.text), spec.account_name):
            result.failures.append(
                f"c6:account_not_in_text ({spec.account_name!r} is never named in full here)"
            )

    # check 7 -- locator well formed
    if cell is not None:
        pattern = locator_pattern_for(assembled.kind, world)
        if not pattern.match(cell.locator):
            result.failures.append(f"c7:bad_locator ({cell.locator!r})")

    # check 8 -- resolves exactly in the prepared text
    if located is None:
        result.failures.append("c8:span_not_in_prepared_text")
    else:
        start, end = located
        assert source is not None
        resolved = resolve_quote_span(text, source.text)
        if (
            source.text[start:end] != text
            or resolved is None
            or not resolved.tier.endswith("exact")
        ):
            result.failures.append("c8:span_not_exact_in_prepared_text")
    return result


def verify_document(
    assembled: AssembledDoc,
    prepared: list[PreparedSource],
    specs: list[SlotSpec],
    world: World,
    *,
    extra_fatal: list[str] | None = None,
) -> DocVerification:
    occurrences: dict[str, list[tuple[AssembledUnit, MarkerSpan]]] = {}
    unbalanced: dict[str, list[int]] = {}
    overlapping: set[str] = set()
    for unit in assembled.units:
        for span in unit.spans:
            occurrences.setdefault(span.marker_id, []).append((unit, span))
        for marker_id in unit.unbalanced:
            unbalanced.setdefault(marker_id, []).append(unit.index)
        overlapping.update(unit.overlap)
    results = [
        _verify_slot(spec, assembled, prepared, occurrences, unbalanced, overlapping, world)
        for spec in specs
    ]
    fatal, notes = doc_level_problems(assembled, {s.marker_id for s in specs})
    if specs and not any(u.spans or u.unbalanced for u in assembled.units):
        # Every scripted fact would sit in the text unmarked and unlabeled.
        fatal.append("no markers at all: the writer ignored the marker protocol")
    return DocVerification(
        results,
        [*(extra_fatal or []), *fatal],
        [*assembled.notes, *notes],
        unscripted=unscripted_in(assembled, specs, world),
    )


def unscripted_in(assembled: AssembledDoc, specs: list[SlotSpec], world: World) -> list[str]:
    """Check 14 over the writer's own text outside every marked span.
    Numbers a scripted claim or a world name carries are allowed (a failed
    fact's text is counted by its own slot, not here)."""
    scripted = " ".join(s.claim for s in specs if s.claim)
    dates = " ".join(
        coverage_extras(scripted, src.date) + " " + src.date.isoformat()
        for src in assembled.sources
    )
    names = " ".join(
        [a.name for a in world.accounts]
        + [p.name for p in world.vendor.products]
        + [c.name for c in world.vendor.competitors]
    )
    allowed = set(re.findall(r"\d+", f"{scripted} {dates} {names}"))
    texts = [
        unmarked_text(u.clean_text, [(sp.start, sp.end) for sp in u.spans]) for u in assembled.units
    ]
    return unscripted_signals(texts, world, allowed_numbers=allowed)


def make_labels(
    verification: DocVerification,
    *,
    plan: DocPlan,
    world: World,
    decoys_by_id: dict[str, PlannedDecoy],
    writer_model: str,
) -> tuple[list[WriterLabel], list[DecoySpan]]:
    facts = {f.id: f for f in world.facts}
    labels: list[WriterLabel] = []
    decoy_spans: list[DecoySpan] = []
    for r in verification.results:
        if not r.passed or r.found is None:
            continue
        f = r.found
        assert f.prepared is not None and f.cell is not None
        assert f.prepared_start is not None and f.prepared_end is not None
        text = f.prepared.text[f.prepared_start : f.prepared_end]
        assert text == f.text
        if r.spec.is_fact:
            fact = facts[r.spec.marker_id]
            labels.append(
                WriterLabel(
                    fact_id=fact.id,
                    fact_type=fact.fact_type,
                    doc_id=f.prepared.source_id,
                    source_type=plan.kind.value,
                    locator=f.cell.locator,
                    label_source=LABEL_SOURCE_WRITER,
                    char_start=f.prepared_start,
                    char_end=f.prepared_end,
                    text=text,
                    account_id=fact.account_id,
                    date=fact.date,
                    plan_id=plan.id,
                    slot_id=r.spec.slot_id,
                    speaker_person_id=r.spec.speaker_person_id,
                    reference_mode=r.spec.reference_mode.value,
                    writer_model=writer_model,
                    prepared_locator=f.cell.prepared_locator,
                    span_words=len(text.split()),
                )
            )
        else:
            decoy = decoys_by_id[r.spec.marker_id]
            decoy_spans.append(
                DecoySpan(
                    decoy_id=decoy.id,
                    decoy_class_id=decoy.decoy_class_id,
                    near_type=decoy.near_type,
                    plan_id=plan.id,
                    slot_id=r.spec.slot_id,
                    doc_id=f.prepared.source_id,
                    source_type=plan.kind.value,
                    locator=f.cell.locator,
                    prepared_locator=f.cell.prepared_locator,
                    char_start=f.prepared_start,
                    char_end=f.prepared_end,
                    text=text,
                    account_id=r.spec.account_id,
                    writer_model=writer_model,
                )
            )
    return labels, decoy_spans
