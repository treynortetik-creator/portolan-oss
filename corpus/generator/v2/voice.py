"""WP4 helper: a labeled span must sound like the person saying it.

The first WP4 smoke (2026-09-22) labeled sentences like "Let me confirm:
Marrowbrook Community Hospital 2 is currently using Wayfinder T3 ... as we
discussed during the QBR on 2026-02-02", said by someone who works at
Marrowbrook. The writer had been shown the code-written claim and told the
full account name had to sit inside the span, so every label was a claim
sentence pasted into a dialogue. An extractor trained on those learns to
find sentences that never occur in real data.

This module holds the fix's two halves:

- **Meaning, not a sentence** (the prompt side). `fact_gist` turns a claim
  into a note with the two parties as roles ("[the customer] runs Wayfinder
  T3 across its ICU ..."), `natural_date_options` gives the date the way
  people say it relative to the document's date, and `short_forms` lists
  the nicknames a speaker may use for a product, competitor or person.
- **Voice checks** (the verification side, `voice_failures`): a marked
  span fails when it uses an ISO date in speech or informal writing (c9),
  names both the vendor and the customer in one sentence (c10), names the
  speaker's own organization in the third person (c11), repeats most of the
  claim's words (c12), or uses a pronoun from the wrong side (c13). Code
  never rewords a failing span; the writer repairs or regenerates it.

`resolve_relative_dates` is the check-6 counterpart of
`natural_date_options`: every phrase the prompt offers resolves back to the
fact's date (`tests/test_world_generator_v2_wp4.py` pins that).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from corpus.generator.v2.models import Person, PersonSide, World
from corpus.generator.v2.planner import DocKind

__all__ = [
    "CLAIM_OVERLAP_MIN_TOKENS",
    "CLAIM_OVERLAP_SHARE",
    "ShortForm",
    "claim_overlap",
    "contains_form",
    "fact_gist",
    "is_claim_shaped",
    "overlap_tokens",
    "natural_date_options",
    "own_org_names",
    "party_names",
    "resolve_relative_dates",
    "short_forms",
    "unmarked_text",
    "unscripted_signals",
    "voice_failures",
]

# Kinds a person speaks or types informally: an ISO date, a self-reference
# by company name, or both parties named in one breath never happen there.
# `DOC` (proposals, release notes) is formal and may name the vendor.
SPOKEN_KINDS = frozenset({DocKind.CALL, DocKind.SLACK, DocKind.TEAMS})
PERSON_KINDS = SPOKEN_KINDS | {DocKind.EMAIL, DocKind.TICKET}
VENDOR_NOTE_KINDS = frozenset({DocKind.CRM_NOTE, DocKind.MEETING_NOTES})
VOICE_KINDS = PERSON_KINDS | VENDOR_NOTE_KINDS

_ISO_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_ISO_TOKEN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'&][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")


# --------------------------------------------------------------------------
# Party names
# --------------------------------------------------------------------------


def _vendor_stem(world: World) -> str | None:
    """The vendor's first word ("Harborlight" of "Harborlight Health
    Systems") when it is distinctive: people shorten a company to it."""
    words = world.vendor.name.split()
    if len(words) < 2 or len(words[0]) < 4:
        return None
    product_words = {w.lower() for p in world.vendor.products for w in p.name.split()}
    return None if words[0].lower() in product_words else words[0]


def vendor_names(world: World) -> tuple[str, ...]:
    stem = _vendor_stem(world)
    return (world.vendor.name, stem) if stem else (world.vendor.name,)


def account_family_names(world: World, account_id: str | None) -> tuple[str, ...]:
    """The account's name and aliases plus its parent system's: a person at
    a facility calls the system "us" too."""
    accounts = world.account_by_id()
    acct = accounts.get(account_id or "")
    if acct is None:
        return ()
    names = [acct.name, *acct.aliases]
    parent = accounts.get(acct.parent or "")
    if parent is not None:
        names += [parent.name, *parent.aliases]
    return tuple(dict.fromkeys(names))


def party_names(world: World, account_ids: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(vendor names, customer names)` for the accounts a span can be about."""
    customer: list[str] = []
    for account_id in account_ids:
        customer += account_family_names(world, account_id)
    return vendor_names(world), tuple(dict.fromkeys(customer))


def own_org_names(world: World, speaker: Person | None, kind: DocKind) -> tuple[str, ...]:
    """Names of the organization the span's speaker works for. A vendor
    note's author is vendor staff (`writer.build_cast`)."""
    if kind in VENDOR_NOTE_KINDS or (speaker is not None and speaker.side == PersonSide.VENDOR):
        return vendor_names(world)
    if speaker is not None:
        return account_family_names(world, speaker.account_id)
    return ()


def _name_hits(text: str, names: tuple[str, ...]) -> list[tuple[int, int, str]]:
    hits: list[tuple[int, int, str]] = []
    for name in sorted(names, key=len, reverse=True):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?:'s|’s)?(?![A-Za-z0-9])"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not any(s <= m.start() < e for s, e, _ in hits):
                hits.append((m.start(), m.end(), name))
    return sorted(hits)


# "here at Marrowbrook we ..." is how people do name their own company.
_FIRST_PERSON_FRAME_RE = re.compile(
    r"\b(?:here|we|us|all of us|those of us)(?:\s+here)?\s+(?:at|from)\s+(?:the\s+)?$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Short forms (nicknames) for span anchors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortForm:
    text: str
    case_sensitive: bool  # a derived nickname that is also a common word ("Care")


def short_forms(anchor: str, world: World) -> tuple[ShortForm, ...]:
    """What a speaker may say instead of `anchor`: a product's declared
    aliases and its name without the shared product-line word ("T3",
    "Link"), a competitor's aliases and first word, a person's first name,
    title and chat handle. The full name must still appear earlier in the
    source (spec section 3.2 check 3)."""
    forms: list[ShortForm] = []
    products = world.vendor.products
    line_words = {p.name.split()[0] for p in products if len(p.name.split()) > 1}
    for p in products:
        if p.name != anchor:
            continue
        forms += [ShortForm(a, False) for a in p.aliases]
        words = p.name.split()
        if len(words) > 1 and words[0] in line_words:
            rest = " ".join(words[1:])
            if len(rest) >= 2 and not any(p2.name == rest for p2 in products):
                forms.append(ShortForm(rest, True))
    for c in world.vendor.competitors:
        if c.name == anchor:
            forms += [ShortForm(a, False) for a in c.aliases]
            first = c.name.split()[0]
            if len(c.name.split()) > 1 and len(first) >= 4:
                forms.append(ShortForm(first, True))
    for person in world.people:
        if person.name == anchor:
            forms.append(ShortForm(person.name.split()[0], True))
            forms.append(ShortForm(person.title, False))
            forms.append(ShortForm(person.chat_handle, False))
    seen: dict[str, ShortForm] = {}
    for f in forms:
        if f.text and f.text != anchor and f.text.lower() not in seen:
            seen[f.text.lower()] = f
    return tuple(seen.values())


def contains_form(text: str, form: ShortForm) -> bool:
    flags = 0 if form.case_sensitive else re.IGNORECASE
    pattern = r"(?<![A-Za-z0-9])" + re.escape(form.text) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, flags) is not None


# --------------------------------------------------------------------------
# Dates the way people say them
# --------------------------------------------------------------------------

_WEEKDAYS = [d.lower() for d in calendar.day_name]
_WEEKDAY_RE = re.compile(
    r"\b(?:(last|this|next|on|coming|past)\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE
)
_ORDINAL_RE = re.compile(
    r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b(?!\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))",
    re.IGNORECASE,
)
_RELATIVE_DAY_RE = re.compile(
    r"\b(today|tonight|this morning|this afternoon|this evening|earlier today|yesterday|tomorrow)\b",
    re.IGNORECASE,
)


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _shift_month(d: date, months: int, day: int) -> date | None:
    month_index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(month_index, 12)
    try:
        return date(year, month + 1, day)
    except ValueError:
        return None


def natural_date_options(fact_date: date, doc_date: date, *, relative: bool = True) -> list[str]:
    """How a person dated `doc_date` says `fact_date`: "today", "yesterday",
    "last Tuesday", "the 28th", "January 28". Every option resolves back
    through `resolve_relative_dates` (check 6). `relative=False` gives only
    phrases that name the day, for text whose own date the extractor does
    not see (a ticket comment later than its ticket)."""
    delta = (fact_date - doc_date).days
    weekday = calendar.day_name[fact_date.weekday()]
    month_day = f"{calendar.month_name[fact_date.month]} {fact_date.day}"
    ordinal = f"the {_ordinal(fact_date.day)}"
    months_apart = (fact_date.year - doc_date.year) * 12 + fact_date.month - doc_date.month
    if not relative:
        return [ordinal, month_day] if abs(months_apart) <= 1 else [month_day]
    if delta == 0:
        return ["today"]
    if delta == -1:
        return ["yesterday", f"on {weekday}"]
    if delta == 1:
        return ["tomorrow", f"on {weekday}"]
    if -6 <= delta <= -2:
        return [f"last {weekday}", f"on {weekday}", month_day]
    if 2 <= delta <= 6:
        return [f"this {weekday}", f"on {weekday}", month_day]
    if 7 <= delta <= 13:
        return [f"next {weekday}", month_day]
    if abs(months_apart) <= 1:
        return [ordinal, month_day]
    return [month_day, f"the {_ordinal(fact_date.day)} of {calendar.month_name[fact_date.month]}"]


def resolve_relative_dates(text: str, source_date: date) -> list[str]:
    """ISO dates a reader holding `source_date` can resolve from relative
    phrases in `text` ("yesterday", "last Tuesday", "the 28th"). Ambiguous
    phrases give every reading (a weekday either side, an ordinal in this
    month or either neighbour)."""
    out: list[date] = []
    for m in _RELATIVE_DAY_RE.finditer(text):
        word = m.group(1).lower()
        offset = {"yesterday": -1, "tomorrow": 1}.get(word, 0)
        out.append(source_date + timedelta(days=offset))
    for m in _WEEKDAY_RE.finditer(text):
        qualifier = (m.group(1) or "").lower()
        target = _WEEKDAYS.index(m.group(2).lower())
        back = (source_date.weekday() - target) % 7 or 7
        ahead = (target - source_date.weekday()) % 7 or 7
        if qualifier in ("last", "past"):
            out += [source_date - timedelta(days=back), source_date - timedelta(days=back + 7)]
        elif qualifier == "next":
            out += [source_date + timedelta(days=ahead), source_date + timedelta(days=ahead + 7)]
        else:
            out += [source_date - timedelta(days=back), source_date + timedelta(days=ahead)]
            if target == source_date.weekday():
                out.append(source_date)
    for m in _ORDINAL_RE.finditer(text):
        day = int(m.group(1))
        for months in (-1, 0, 1):
            d = _shift_month(source_date, months, day)
            if d is not None:
                out.append(d)
    return [d.isoformat() for d in dict.fromkeys(out)]


# --------------------------------------------------------------------------
# The claim as a note
# --------------------------------------------------------------------------

_AS_OF_RE = re.compile(r",?\s+as of \d{4}-\d{2}-\d{2}")
_NOTE_DROP_RE = re.compile(r"\b(?:a|an|the|its|their|as)\s+", re.IGNORECASE)
_DATED_NOUN_RE = re.compile(r"(\b(?:the|its|a|an)|'s)\s+\d{4}-\d{2}-\d{2}\s+")


def fact_gist(claim: str, world: World, account_id: str) -> str:
    """The claim as a note for the writer: the observation date dropped
    ("as of 2026-02-02"), other dates as "[the date]", and the two parties
    as "[the customer]" and "[the vendor]" so the writer voices them from
    the speaker's side. Instruction text only; check 12 rejects a span
    that repeats it."""
    gist = _AS_OF_RE.sub("", claim)
    gist = _DATED_NOUN_RE.sub(r"\1 ", gist)
    gist = re.sub(r"\d{4}-\d{2}-\d{2}", "[the date]", gist)
    for name in sorted(account_family_names(world, account_id), key=len, reverse=True):
        gist = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", "[the customer]", gist
        )
    for name in sorted(vendor_names(world), key=len, reverse=True):
        gist = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", "[the vendor]", gist
        )
    gist = gist.replace("[the customer]'s", "[the customer's]").replace(
        "[the vendor]'s", "[the vendor's]"
    )
    gist = gist.replace("[the vendor]s", "[the vendor]")
    # Notes, not a sentence: without articles and "its"/"their" the note is
    # harder to paste (seen live 2026-09-22: "... across our ICU and
    # med-surg units as reported at the QBR today" copied from the note).
    parts = re.split(r"(\[[^\]]*\])", gist)
    parts = [p if p.startswith("[") else _NOTE_DROP_RE.sub("", p) for p in parts]
    return " ".join("".join(parts).split())


# --------------------------------------------------------------------------
# Check 12: a claim-shaped span
# --------------------------------------------------------------------------

# Claim-shaped = the claim's words AND a party named in the third person:
# every smoke label the 2026-09-22 review rejected is both (0.75 to 0.92 of
# the claim's words). A faithful sentence in the speaker's own voice keeps
# the content words too (live: 0.82 for "the way our CFO framed it today,
# the proposed Wayfinder Care renewal price just doesn't line up ..."), so
# without a third-person party name only a paste counts: the template's
# words and not one word of the speaker's own. On 17 live spans rejected
# 2026-09-22, the five pastes had no word of their own; the twelve natural
# ones had one or more ("which was frustrating for everyone").
CLAIM_OVERLAP_SHARE = 0.7
CLAIM_OVERLAP_MIN_TOKENS = 6
TEMPLATE_OVERLAP_SHARE = 0.9
_NOT_OWN_WORDS = frozenset(
    "today yesterday tomorrow morning afternoon january february march april may june july "
    "august september october november december monday tuesday wednesday thursday friday "
    "saturday sunday".split()
)

_OVERLAP_STOPWORDS = frozenset(
    "a an the and or but of to in on at by for with as is are was were be been being has have "
    "had it its it's that this these those from their they them we our us you your i my me he "
    "she his her now not no if so than then into about after before during will would can could "
    "should may might just also still yet very there here what which who whom".split()
)


def _stem(token: str) -> str:
    t = token.lower()
    if t.endswith("'s"):
        t = t[:-2]
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3 and not t.endswith("ss"):
            t = t[: -len(suffix)] + repl
            break
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


def overlap_tokens(text: str) -> set[str]:
    return {
        _stem(t)
        for t in _WORD_RE.findall(text.lower().replace("’", "'"))
        if t not in _OVERLAP_STOPWORDS and not _ISO_TOKEN_RE.match(t)
    }


def claim_overlap(
    span: str, claim: str, entity_words: frozenset[str]
) -> tuple[float, float, int, int, int]:
    """`(share of the claim's words in the span, share of the template's
    words in the span, claim word count, template word count, the span's
    own words)`. Template words are the claim's words minus entity names
    and numbers: the pack template's own wording. Own words are span words
    the claim lacks, dates and numbers aside."""
    claim_tokens = overlap_tokens(claim)
    span_tokens = overlap_tokens(span)
    stemmed_entities = {_stem(w) for w in entity_words}
    template = {t for t in claim_tokens if t not in stemmed_entities and not t[0].isdigit()}
    share = len(claim_tokens & span_tokens) / len(claim_tokens) if claim_tokens else 0.0
    t_share = len(template & span_tokens) / len(template) if template else 0.0
    own = {
        t
        for t in span_tokens - claim_tokens - stemmed_entities
        if not t[0].isdigit() and t not in _NOT_OWN_WORDS
    }
    return share, t_share, len(claim_tokens), len(template), len(own)


def is_claim_shaped(
    span: str, claim: str, entity_words: frozenset[str], *, names_party: bool = True
) -> float | None:
    """The offending share when `span` restates the claim, else None.
    `names_party`: the span names the vendor or the customer in the third
    person."""
    share, t_share, n_claim, n_template, own = claim_overlap(span, claim, entity_words)
    if names_party and n_claim >= CLAIM_OVERLAP_MIN_TOKENS and share >= CLAIM_OVERLAP_SHARE:
        return share
    if n_template >= CLAIM_OVERLAP_MIN_TOKENS and t_share >= TEMPLATE_OVERLAP_SHARE and own == 0:
        return t_share
    return None


# --------------------------------------------------------------------------
# Check 13: pronouns from the wrong side
# --------------------------------------------------------------------------

# Nouns only a care provider owns in a medical-devices world. A vendor's
# "our ICU" or a customer's "your ICU" is a side mix-up (seen in the first
# WP4 smoke). Verticals whose vendor legitimately has these drop them.
_CUSTOMER_NOUNS = (
    "icu",
    "med-surg",
    "nurses",
    "nursing staff",
    "patients",
    "residents",
    "biomed",
    "ehr",
)
_VENDOR_OWNS_BY_VERTICAL: dict[str, frozenset[str]] = {
    "healthcare_staffing": frozenset({"nurses", "nursing staff"}),
    "telehealth": frozenset({"patients"}),
}
# Org-level nouns a customer never calls "their" when the fact is about
# its own organization ("their upcoming EHR migration", first smoke).
_CUSTOMER_ORG_NOUNS = (
    "ehr",
    "migration",
    "go-live",
    "renewal",
    "contract",
    "budget",
    "board",
    "cfo",
)
_VENDOR_NOUNS = ("roadmap", "firmware", "r&d", "developers", "dev team", "product team")
_NOT_A_POSSESSION = r"(?!\s+(?:customers?|clients?|accounts?|sites?|users?|buyers?|market))"


def _noun_re(pronoun: str, nouns: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(n) for n in nouns)
    return re.compile(rf"\b{pronoun}\s+(?:[\w-]+\s+)?(?:{alt})\b{_NOT_A_POSSESSION}", re.IGNORECASE)


def _pronoun_mismatch(
    text: str, side: PersonSide, vertical: str, names_other_party: bool
) -> str | None:
    customer_nouns = tuple(
        n for n in _CUSTOMER_NOUNS if n not in _VENDOR_OWNS_BY_VERTICAL.get(vertical, frozenset())
    )
    checks: list[tuple[re.Pattern[str], str]] = []
    if side == PersonSide.VENDOR:
        checks += [
            (_noun_re("our", customer_nouns), "a vendor person says 'our' about the customer's"),
            (_noun_re("your", _VENDOR_NOUNS), "a vendor person says 'your' about the vendor's"),
        ]
    else:
        checks += [
            (_noun_re("your", customer_nouns), "a customer says 'your' about their own"),
            (_noun_re("our", _VENDOR_NOUNS), "a customer says 'our' about the vendor's"),
        ]
        if not names_other_party:
            checks.append(
                (_noun_re("their", _CUSTOMER_ORG_NOUNS), "a customer says 'their' about their own")
            )
    for pattern, why in checks:
        m = pattern.search(text)
        if m:
            return f"{why}: {m.group(0)!r}"
    return None


# --------------------------------------------------------------------------
# All voice checks for one span
# --------------------------------------------------------------------------


def _sentence_before(text: str, start: int) -> str:
    """`text` from the start of the sentence holding `start` up to it."""
    cut = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
    return text[cut + 1 : start]


def voice_failures(
    *,
    text: str,
    kind: DocKind,
    world: World,
    speaker: Person | None,
    account_ids: list[str],
    claim: str | None,
    entity_words: frozenset[str],
    other_names: tuple[str, ...] = (),
) -> list[str]:
    """Failure codes c9 to c13 for one marked span (`text`). `speaker` is
    the unit's author (None in a vendor note, whose author is vendor
    staff); `claim` is None for a decoy, which skips c12 and c13."""
    if kind not in VOICE_KINDS:
        return []
    failures: list[str] = []
    iso = _ISO_DATE_RE.search(text)
    if iso:
        failures.append(f"c9:iso_date ({iso.group(0)!r}; people say 'the 28th' or 'last Tuesday')")
    vendor, customer = party_names(world, account_ids)
    own = own_org_names(world, speaker, kind)
    # A vendor note names the customer in the third person as a matter of
    # course; only naming the vendor there reads like the claim.
    parties = vendor if kind in VENDOR_NOTE_KINDS else vendor + customer
    third_person_party = any(
        not _FIRST_PERSON_FRAME_RE.search(_sentence_before(text, start))
        for start, _end, _name in _name_hits(text, parties)
    )
    for sentence in _SENTENCE_RE.split(text):
        v_hits = _name_hits(sentence, vendor)
        c_hits = _name_hits(sentence, customer)
        if v_hits and c_hits:
            failures.append(
                f"c10:both_parties_named ({v_hits[0][2]!r} and {c_hits[0][2]!r} in one sentence)"
            )
            break
    for start, _end, name in _name_hits(text, own):
        if _FIRST_PERSON_FRAME_RE.search(_sentence_before(text, start)):
            continue
        failures.append(
            f"c11:own_org_named ({name!r}: the speaker's own organization; say we/our/us)"
        )
        break
    if claim is not None and kind in PERSON_KINDS | VENDOR_NOTE_KINDS:
        share = is_claim_shaped(text, claim, entity_words, names_party=third_person_party)
        if share is not None:
            failures.append(
                f"c12:claim_shaped ({share:.0%} of the script note's words; say it the way "
                "the speaker would)"
            )
    if claim is not None and kind in PERSON_KINDS and speaker is not None:
        names_other = bool(_name_hits(text, other_names))
        why = _pronoun_mismatch(text, speaker.side, world.vertical, names_other)
        if why:
            failures.append(f"c13:pronoun_side ({why})")
    return failures


# --------------------------------------------------------------------------
# Check 14 (document level): content nobody scripted
# --------------------------------------------------------------------------

# WP5's blind pass on the first 15 WP4 smoke documents found 106 unscripted
# facts, 24 and 20 in the two long calls: invented products ("MedFlow",
# "SmartStock"), follow-up promises, figures and dates. Each is an
# unlabeled positive. These are the cheap lexical signs of one, counted in
# the text outside the marked spans.
_PROMISE_RE = re.compile(
    r"\b(?:i['’]ll|i will|we['’]ll|we will|i['’]m going to|we['’]re going to|let me)\s+"
    r"(?:\w+\s+){0,2}?(?:send|forward|follow up|get back|circle back|loop in|email|schedule|"
    r"set up|put together|draft|deliver|escalate|reach out|push|ship|provide|prepare|arrange|"
    r"book)\b(?!\s+(?:my|your|the)\s+screen)",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\bby\s+(?:eod|end of (?:day|the day|the week|the month|next week|the quarter)|tomorrow|"
    r"next (?:week|month|quarter)|monday|tuesday|wednesday|thursday|friday|the \d{1,2}(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)
_NUMERAL_TOKEN_RE = re.compile(r"(?<![\w.])\$?\d[\d,.:/]*(?:st|nd|rd|th)?%?(?![\w])")
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_DATE_PHRASE_RE = re.compile(
    rf"\b(?:{_MONTH_NAMES})\s+\d{{1,2}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{_MONTH_NAMES})\b",
    re.IGNORECASE,
)
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]+\b")
# Capitalized words next to a product noun ("the Epic EHR", "Meditech
# platform") or listed as systems ("including Epic and Cerner").
_NAMED_SYSTEM_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s+(?:EHR|EMR|platform|system|software|app|"
    r"portal|suite|module|tool|dashboard|scanner|pump|monitor)s?\b"
    r"|\bincluding\s+([A-Z][A-Za-z]+)(?:,?\s+(?:and\s+)?([A-Z][A-Za-z]+))*"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def unmarked_text(unit_text: str, spans: list[tuple[int, int]]) -> str:
    """`unit_text` with every marked span cut out (newline-joined pieces)."""
    parts, pos = [], 0
    for start, end in sorted(spans):
        parts.append(unit_text[pos:start])
        pos = max(pos, end)
    parts.append(unit_text[pos:])
    return "\n".join(p for p in parts if p.strip())


def _known_name_words(world: World) -> set[str]:
    names = [world.vendor.name, world.vendor.domain]
    for p in world.vendor.products:
        names += [p.name, *p.aliases]
    for c in world.vendor.competitors:
        names += [c.name, *c.aliases]
    for a in world.accounts:
        names += [a.name, *a.aliases, a.region]
    for person in world.people:
        names += [person.name, person.title, person.role]
    return {w.lower() for n in names for w in re.findall(r"[A-Za-z]+", n)}


def unscripted_signals(texts: list[str], world: World, *, allowed_numbers: set[str]) -> list[str]:
    """Sentences outside the marked spans that likely state a fact nobody
    scripted: an invented product or system name, a number, a calendar
    date, or a promise or deadline. Each item is `<kind>: <sentence>`."""
    known = _known_name_words(world)
    out: list[str] = []
    for text in texts:
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            kinds: list[str] = []
            names = [m.group(0) for m in _CAMEL_RE.finditer(sentence)]
            for m in _NAMED_SYSTEM_RE.finditer(sentence):
                names += [g for g in m.groups() if g]
            if any(
                w.lower() not in known
                for n in names
                for w in re.findall(r"[A-Za-z]+", n)
                if not (w.isupper() and len(w) <= 5)
            ):
                kinds.append("name")
            numbers = [
                n.rstrip(".,:/")
                for n in _NUMERAL_TOKEN_RE.findall(sentence)
                if not set(re.findall(r"\d+", n)) <= allowed_numbers
            ]
            if numbers:
                kinds.append("number")
            if _DATE_PHRASE_RE.search(sentence):
                kinds.append("date")
            if _PROMISE_RE.search(sentence) or _DEADLINE_RE.search(sentence):
                kinds.append("promise")
            if kinds:
                out.append(f"{'+'.join(kinds)}: {sentence[:160]}")
    return out
