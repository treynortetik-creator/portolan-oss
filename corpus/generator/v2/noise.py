"""WP6: the noise stage (`01-spec/world-generator-v2-2026-09-22.md` section 5,
"Noise and realism", and section 9's WP6 row: "every section 5 operation
with span remapping and noise labels, near-duplicate writer calls").

Input: WP4's finished documents (`writer.WrittenDoc`: files, prepared text,
exact-span labels and decoy spans), the `World` and its `WorldDocPlan`.
Output: one `NoisedDoc` per document (a `WrittenDoc` whose labels are
`NoisedLabel`s), the near-duplicate documents, and a noise manifest.

Section 5 operations, and where each one lives:

| Operation | Here | Label policy applied |
|---|---|---|
| Misfiled (5% of docs) | code: every code-rendered account field (ticket and CRM-note JSON, email `X-Related-Account`, call and doc `Account:` header), the manifest's account hint and the CRM link point at another account | labels keep the true account |
| Near-duplicate, one fact changed | `noise_neardup`: an open-weight writer call | both labeled; the new fact `supersedes` the old |
| Superseded facts (natural) | the WP1 timeline; recorded here per document | `supersedes`, `valid_to` |
| Forwarded or quoted (20% of emails) | code: an earlier message of the thread is forwarded (header block plus its body, still visible after `prepare_text`) or quoted as a trailing `>` block (stripped) | labels follow `prepare_text`: a forwarded body's facts get copy labels; quoted text is never labeled |
| ASR errors (30% of calls, 1 to 4% of words) | code: homophones, dropped words, misheard names, `[inaudible]`, `[crosstalk]` | the span follows the noisy text, `asr_in_span: true` |
| Partial or garbled (3% of docs) | code: truncation, a missing comment, mojibake, HTML leftovers | cut spans are dropped; garbled spans kept only if their anchors survive |
| Stale or missing links | code: WP3 leaves about 10% of CRM links unwired; half of those become stale (pointing at another account's document) | recorded in the manifest |
| Off-topic layer | WP4 writes these; recorded here | negative |

**Provenance.** Code only moves, removes, re-encodes or mis-transcribes the
writer's own words, and renders structure (forward headers, quote lines)
from values the document's headers already hold. The one operation that
needs new sentences, the near-duplicate, goes through the open-weight writer.

**Spans.** Every edit runs on `noise_text` units whose characters remember
their original prepared-text positions, so after noise every surviving
label is the exact slice of the new prepared text (`text == prepared[start:
end]`), and a label that cannot be placed is dropped with its reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from corpus.generator.v2.models import Account, World
from corpus.generator.v2.noise_text import (
    DocText,
    Edit,
    Key,
    Located,
    ParseError,
    SpanCopy,
    Unit,
    apply_edits,
    assign_original_keys,
    key_index,
    locate,
    parse_written,
    remap_span,
    remove_source,
    remove_unit,
    render_files,
)
from corpus.generator.v2.planner import (
    _LINK_FIELD_BY_KIND,
    DocKind,
    DocPlan,
    DocRole,
    WorldDocPlan,
    crm_account_source_id,
)
from corpus.generator.v2.verify import (
    DecoySpan,
    WriterLabel,
    _contains_term,
    _norm,
    _reference_target,
    entity_words_for,
)
from corpus.generator.v2.voice import short_forms
from corpus.generator.v2.writer import PreparedSourceRecord, WrittenDoc

__all__ = [
    "ASR_WER_BAND",
    "CUT_OPS",
    "GARBLE_OPS",
    "LabelDrop",
    "NOISE_RATES",
    "NoiseConfig",
    "NoiseOp",
    "NoiseRun",
    "NoisedDecoySpan",
    "NoisedDoc",
    "NoisedLabel",
    "apply_noise",
    "carrier_label",
    "carrier_slot_id",
    "link_noise",
    "noise_completed_world",
    "noise_document",
    "plan_noise",
    "prepared_sha256",
    "select_docs",
]

NOISE_VERSION = "wp6-2026-09-23.2"

OP_MISFILED = "misfiled"
OP_NEAR_DUPLICATE = "near_duplicate"
OP_SUPERSEDED = "superseded_fact"
OP_FORWARD = "forwarded_or_quoted"
OP_ASR = "asr"
OP_PARTIAL = "partial_or_garbled"
OP_STALE_LINK = "stale_link"
OP_MISSING_LINK = "missing_link"
OP_OFF_TOPIC = "off_topic"

PARTIAL_TRUNCATED = "partial_or_garbled:truncated"
PARTIAL_MISSING_COMMENT = "partial_or_garbled:missing_comment"
PARTIAL_MOJIBAKE = "partial_or_garbled:mojibake"
PARTIAL_HTML = "partial_or_garbled:html_leftovers"
PARTIAL_VARIANTS = ("truncated", "missing_comment", "mojibake", "html_leftovers")
# Spec section 5: "Facts whose span is cut are removed; garbled spans are
# kept only if their anchors survive."
CUT_OPS = frozenset({PARTIAL_TRUNCATED, PARTIAL_MISSING_COMMENT})
GARBLE_OPS = frozenset({PARTIAL_MOJIBAKE, PARTIAL_HTML})

# Spec section 5 rates, per document of the eligible kind.
NOISE_RATES: dict[str, float] = {
    OP_MISFILED: 0.05,
    OP_FORWARD: 0.20,
    OP_ASR: 0.30,
    OP_PARTIAL: 0.03,
}
# Share of WP3's unwired ("missing") links that become stale instead, so
# stale plus missing stays at the spec's "about 10% of links".
STALE_LINK_RATE = 0.5
# Spec section 5: "1 to 4% word error".
ASR_WER_BAND = (0.01, 0.04)

# The order operations run in on one document.
DOC_OP_ORDER = (OP_FORWARD, OP_ASR, OP_PARTIAL, OP_MISFILED)

_LINK_FIELDS = (
    "linked_calls",
    "linked_email_threads",
    "linked_tickets",
    "linked_chat_threads",
    "linked_docs",
)


# --------------------------------------------------------------------------
# Output models
# --------------------------------------------------------------------------


class NoiseOp(BaseModel):
    """One entry of `manifest.json`'s `noise: [{type, params}]` (spec
    section 5), also carried on the document record."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class NoisedLabel(WriterLabel):
    """A label after noise. `noise` names every operation that touched the
    span's characters; `copy_of` is set on a verbatim copy of another label
    (a forwarded message body), as `<doc_id>:<start>-<end>` of the original."""

    noise: list[str] = Field(default_factory=list)
    asr_in_span: bool = False
    remapped: bool = False
    copy_of: str | None = None
    supersedes: str | None = None


class NoisedDecoySpan(DecoySpan):
    noise: list[str] = Field(default_factory=list)
    remapped: bool = False
    copy_of: str | None = None


class LabelDrop(BaseModel):
    """A label or decoy span the noise stage removed, and why."""

    model_config = ConfigDict(extra="forbid")

    slot_kind: Literal["fact", "decoy"]
    marker_id: str
    plan_id: str
    doc_id: str
    char_start: int
    char_end: int
    text: str
    reason: str
    noise_op: str | None = None


class NoisedDoc(WrittenDoc):
    """A `WrittenDoc` after the noise stage. `labels` index the NEW
    prepared text of `prepared_sources`. `noised_from_sha256` is
    `prepared_sha256` of the version it was noised from, when the stage ran
    on WP5's output (`noise_completed_world`)."""

    labels: list[NoisedLabel] = Field(default_factory=list)
    decoy_spans: list[NoisedDecoySpan] = Field(default_factory=list)
    noise: list[NoiseOp] = Field(default_factory=list)
    label_drops: list[LabelDrop] = Field(default_factory=list)
    true_account_id: str | None = None
    filed_account_id: str | None = None
    noise_notes: list[str] = Field(default_factory=list)
    noise_counts: dict[str, int] = Field(default_factory=dict)
    noised_from_sha256: str | None = None


# --------------------------------------------------------------------------
# Configuration and selection
# --------------------------------------------------------------------------


@dataclass
class NoiseConfig:
    seed: int
    rates: dict[str, float] = field(default_factory=lambda: dict(NOISE_RATES))
    # Operations applied to EVERY eligible document (the live smoke).
    force: frozenset[str] = frozenset()
    asr_wer: tuple[float, float] = ASR_WER_BAND
    stale_link_rate: float = STALE_LINK_RATE
    # doc id -> partial variant (else the document's rng picks one)
    partial_variants: dict[str, str] = field(default_factory=dict)


def _rank(seed: int, op: str, item: str) -> str:
    return hashlib.sha256(f"wp6:{seed}:{op}:{item}".encode()).hexdigest()


def select_docs(eligible: list[str], rate: float, seed: int, op: str) -> set[str]:
    """Exactly `round(rate * n)` items, the ones with the smallest seeded
    hash: deterministic, and independent of input order."""
    k = int(rate * len(eligible) + 0.5)
    return set(sorted(eligible, key=lambda d: _rank(seed, op, d))[:k])


def _rng(seed: int, op: str, doc_id: str) -> random.Random:
    return random.Random(f"wp6:{seed}:{op}:{doc_id}")


def eligible_ops(doc: WrittenDoc, plan: DocPlan) -> list[str]:
    if doc.status != "written":
        return []
    ops = [OP_PARTIAL]
    if plan.account_id:
        ops.append(OP_MISFILED)
    if doc.kind == DocKind.CALL:
        ops.append(OP_ASR)
    if doc.kind == DocKind.EMAIL and len(doc.files) >= 2:
        ops.append(OP_FORWARD)
    return ops


def plan_noise(
    docs: dict[str, WrittenDoc], plans: dict[str, DocPlan], config: NoiseConfig
) -> dict[str, list[str]]:
    """Which operations run on which document: `rate` of the eligible
    documents per operation (spec section 5), or all of them when forced."""
    eligible: dict[str, list[str]] = {op: [] for op in DOC_OP_ORDER}
    for doc_id, doc in sorted(docs.items()):
        plan = plans.get(doc_id)
        if plan is None or plan.role == DocRole.NEAR_DUPLICATE:
            continue
        for op in eligible_ops(doc, plan):
            eligible[op].append(doc_id)
    chosen: dict[str, set[str]] = {}
    for op, ids in eligible.items():
        chosen[op] = (
            set(ids)
            if op in config.force
            else select_docs(ids, config.rates.get(op, 0.0), config.seed, op)
        )
    out: dict[str, list[str]] = {}
    for doc_id in sorted(docs):
        ops = [op for op in DOC_OP_ORDER if doc_id in chosen.get(op, set())]
        out[doc_id] = ops
    return out


# --------------------------------------------------------------------------
# Per-document context
# --------------------------------------------------------------------------


@dataclass
class NoiseContext:
    world: World
    plan: DocPlan
    config: NoiseConfig
    span_owner: dict[Key, str]
    protected_name_tokens: frozenset[str]
    entity_words: frozenset[str]


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _name_tokens(name: str) -> set[str]:
    return set(_TOKEN_RE.findall(_norm(name)))


def _context(
    world: World,
    plan: DocPlan,
    config: NoiseConfig,
    doc: DocText,
    written: WrittenDoc,
    entity_words: frozenset[str],
) -> NoiseContext:
    owner: dict[Key, str] = {}
    for label in written.labels:
        for k in range(label.char_start, label.char_end):
            owner[(label.doc_id, k)] = label.fact_id
    for span in written.decoy_spans:
        for k in range(span.char_start, span.char_end):
            owner[(span.doc_id, k)] = span.decoy_id
    facts = {f.id: f for f in world.facts}
    protected: set[str] = set()
    for label in written.labels:
        fact = facts.get(label.fact_id)
        if fact is None:
            continue
        _mode, anchor, _form = _reference_target(fact, world, plan.kind)
        if anchor:
            protected |= _name_tokens(anchor)
    return NoiseContext(world, plan, config, owner, frozenset(protected), entity_words)


# --------------------------------------------------------------------------
# ASR errors (calls)
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")

_HOMOPHONE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("their", "there", "they're"),
    ("your", "you're"),
    ("its", "it's"),
    ("then", "than"),
    ("whether", "weather"),
    ("hear", "here"),
    ("write", "right"),
    ("buy", "by"),
    ("week", "weak"),
    ("new", "knew"),
    ("whole", "hole"),
    ("peace", "piece"),
    ("plain", "plane"),
    ("site", "sight"),
    ("wait", "weight"),
    ("would", "wood"),
    ("passed", "past"),
    ("break", "brake"),
    ("accept", "except"),
    ("affect", "effect"),
    ("cell", "sell"),
    ("higher", "hire"),
    ("meet", "meat"),
    ("fair", "fare"),
    ("sale", "sail"),
    ("role", "roll"),
    ("seen", "scene"),
    ("through", "threw"),
    ("allowed", "aloud"),
    ("patients", "patience"),
    ("our", "are"),
    ("we're", "were"),
    ("who's", "whose"),
    ("where", "wear"),
)
HOMOPHONES: dict[str, tuple[str, ...]] = {
    w: tuple(x for x in group if x != w) for group in _HOMOPHONE_GROUPS for w in group
}
# Words an ASR engine drops without changing what was said.
DROPPABLE = frozenset(
    {"the", "a", "an", "um", "uh", "just", "really", "so", "like", "that", "of", "very"}
)
_NEGATIONS = frozenset(
    {"not", "no", "never", "nor", "none", "nobody", "nothing", "neither", "cannot", "without"}
)
_NUMBER_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety", "hundred", "thousand", "million", "billion", "first", "second",
        "third", "half", "quarter", "dozen", "once", "twice", "percent",
    }
)  # fmt: skip
# Name words an ASR engine spells right (dictionary words), never "misheard".
_GENERIC_NAME_WORDS = frozenset(
    {
        "hospital", "hospitals", "community", "health", "healthcare", "system", "systems",
        "medical", "center", "centre", "general", "regional", "memorial", "network", "group",
        "partners", "services", "clinic", "clinics", "university", "care", "county", "valley",
        "saint", "north", "south", "east", "west", "northern", "southern", "eastern",
        "western", "united", "national", "family", "senior", "living", "home", "homes",
        "pharmacy", "labs", "devices", "device", "technologies", "solutions", "link",
        "suite", "cloud", "platform", "monitor", "monitoring", "pump", "pumps", "the", "and",
    }
)  # fmt: skip
_ASR_WEIGHTS = (
    ("homophone", 30),
    ("drop", 25),
    ("inaudible", 20),
    ("misheard_name", 15),
    ("crosstalk", 10),
)
_VOWEL_SHIFT = {"a": "e", "e": "i", "i": "e", "o": "a", "u": "o"}


def mishear(word: str, rng: random.Random) -> str | None:
    """A plausible mis-transcription of a name ("Corvane" -> "Carvane"): one
    vowel shifted, a doubled consonant undoubled, or a spelling swap."""
    options: set[str] = set()
    for i in range(1, len(word)):
        lc = word[i].lower()
        if lc in _VOWEL_SHIFT:
            new = _VOWEL_SHIFT[lc]
            options.add(word[:i] + (new.upper() if word[i].isupper() else new) + word[i + 1 :])
    for i in range(1, len(word) - 1):
        if word[i].lower() == word[i + 1].lower() and word[i].lower() not in "aeiou":
            options.add(word[:i] + word[i + 1 :])
    low = word.lower()
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("k", "c"), ("y", "i"), ("s", "z")):
        idx = low.find(a, 1)
        if idx > 0:
            options.add(word[:idx] + b + word[idx + len(a) :])
    options = {o for o in options if o.lower() != low}
    return rng.choice(sorted(options)) if options else None


@dataclass
class _Tok:
    unit: Unit
    start: int
    end: int
    word: str
    owner: str | None  # the label or decoy the token sits in
    protected: bool


def _is_negation(low: str) -> bool:
    return low in _NEGATIONS or "n't" in low


def _tokens(units: list[Unit], ctx: NoiseContext) -> list[_Tok]:
    out: list[_Tok] = []
    for u in units:
        for m in _WORD_RE.finditer(u.text):
            owners: set[str | None] = set()
            for i in range(m.start(), m.end()):
                if not u.keys[i]:
                    owners.add(None)
                for key in u.keys[i]:
                    owners.add(ctx.span_owner.get(key))
            low = m.group(0).lower().replace("’", "'")
            before = u.text[m.start() - 1] if m.start() > 0 else ""
            after = u.text[m.end()] if m.end() < len(u.text) else ""
            protected = (
                len(owners) != 1
                or any(ch.isdigit() for ch in low)
                or low in _NUMBER_WORDS
                or _is_negation(low)
                or before == "["
                or after == "]"
                or any(not keys for keys in u.keys[m.start() : m.end()])
            )
            owner = next(iter(owners)) if len(owners) == 1 else None
            out.append(_Tok(u, m.start(), m.end(), m.group(0), owner, protected))
    return out


def _is_name(tok: _Tok, ctx: NoiseContext) -> bool:
    low = tok.word.lower().replace("’", "'")
    return tok.word[0].isupper() and low in ctx.entity_words


def _match_case(template: str, word: str) -> str:
    out = word.replace("'", "’") if "’" in template else word
    return out[0].upper() + out[1:] if template[0].isupper() else out


def _asr(doc: DocText, rng: random.Random, ctx: NoiseContext) -> NoiseOp | None:
    units = doc.live_units(role="writer")
    toks = _tokens(units, ctx)
    n_words = len(toks)
    if n_words == 0:
        return None
    lo_rate, hi_rate = ctx.config.asr_wer
    lo = max(1, math.ceil(lo_rate * n_words))
    hi = max(lo, math.floor(hi_rate * n_words))
    target = rng.randint(lo, hi)

    def free(i: int) -> bool:
        return 0 <= i < len(toks) and i not in used and not toks[i].protected

    def candidates(kind: str) -> list[int]:
        out = []
        for i, t in enumerate(toks):
            if not free(i):
                continue
            low = t.word.lower().replace("’", "'")
            if kind == "homophone" and low in HOMOPHONES:
                out.append(i)
            elif kind == "drop" and low in DROPPABLE:
                out.append(i)
            elif kind == "misheard_name":
                if (
                    _is_name(t, ctx)
                    and t.word.isalpha()
                    and len(t.word) >= 4
                    and low not in _GENERIC_NAME_WORDS
                    and (t.owner is not None or low not in ctx.protected_name_tokens)
                ):
                    out.append(i)
            elif kind == "inaudible" and t.word.isalpha() and not _is_name(t, ctx):
                out.append(i)
            elif (
                kind == "crosstalk"
                and t.owner is None
                and not t.unit.text[: t.start].strip()
                and t.word.isalpha()
                and not _is_name(t, ctx)
            ):
                out.append(i)
        return out

    used: set[int] = set()
    edits: dict[int, list[Edit]] = {}
    counts: Counter = Counter()
    affected = 0
    for _ in range(target * 20):
        if affected >= target:
            break
        pools = {kind: candidates(kind) for kind, _w in _ASR_WEIGHTS}
        kinds = [(k, w) for k, w in _ASR_WEIGHTS if pools[k]]
        if not kinds:
            break
        kind = rng.choices([k for k, _ in kinds], weights=[w for _, w in kinds], k=1)[0]
        i = rng.choice(pools[kind])
        t = toks[i]
        width = 1
        if kind in ("inaudible", "crosstalk") and target - affected >= 2 and rng.random() < 0.3:
            j = i + 1
            if (
                free(j)
                and toks[j].unit is t.unit
                and toks[j].owner == t.owner
                and t.unit.text[t.end : toks[j].start] == " "
                and toks[j].word.isalpha()
                and not _is_name(toks[j], ctx)
            ):
                width = 2
        last = toks[i + width - 1]
        if kind == "homophone":
            low = t.word.lower().replace("’", "'")
            new = _match_case(t.word, rng.choice(HOMOPHONES[low]))
            edit = Edit(t.start, t.end, new, OP_ASR)
        elif kind == "drop":
            text = t.unit.text
            if text[t.end : t.end + 1] == " ":
                edit = Edit(t.start, t.end + 1, "", OP_ASR)
            elif t.start > 0 and text[t.start - 1] == " ":
                edit = Edit(t.start - 1, t.end, "", OP_ASR)
            else:
                used.add(i)
                continue
            if not text[: edit.start].strip() and not text[edit.end :].strip():
                used.add(i)
                continue  # never drop a turn's only word
        elif kind == "misheard_name":
            new = mishear(t.word, rng)
            if new is None:
                used.add(i)
                continue
            edit = Edit(t.start, t.end, new, OP_ASR)
        else:
            tag = "[inaudible]" if kind == "inaudible" else "[crosstalk]"
            edit = Edit(t.start, last.end, tag, OP_ASR)
        # Neighbours stay untouched so no two edits merge or overlap.
        used.update(range(i - 1, i + width + 1))
        edits.setdefault(t.unit.uid, []).append(edit)
        counts[kind] += width
        affected += width
    if not edits:
        return None
    by_uid = {u.uid: u for u in units}
    for uid, unit_edits in edits.items():
        apply_edits(doc, by_uid[uid], unit_edits)
    return NoiseOp(
        type=OP_ASR,
        params={
            "words": n_words,
            "errors": affected,
            "wer": round(affected / n_words, 4),
            "by_error": dict(sorted(counts.items())),
        },
    )


# --------------------------------------------------------------------------
# Forwarded or quoted threads (emails)
# --------------------------------------------------------------------------


def _person_display(world: World, address: str) -> str:
    for p in world.people:
        if p.email == address:
            return f"{p.name} <{p.email}>"
    return address


def _forward(doc: DocText, rng: random.Random, ctx: NoiseContext) -> NoiseOp | None:
    msgs = doc.live_sources()
    if len(msgs) < 2:
        return None
    variants = ["forward"]
    quotable = [j for j in range(1, len(msgs)) if not msgs[j].meta["tail"]]
    if quotable:
        variants.append("quote")
    variant = rng.choice(variants)
    if variant == "forward":
        j = rng.randrange(1, len(msgs))
        i = rng.randrange(0, j)
        src, dst = msgs[i], msgs[j]
        src_unit = doc.live_units(src.source_id, "writer")[0]
        # Structure only: the forwarded message's own header values.
        prefix = (
            "\n\n---------- Forwarded message ---------\n"
            f"From: {_person_display(ctx.world, src.meta['from'])}\n"
            f"Date: {src.meta['date_line']}\n"
            f"Subject: {src.meta['subject']}\n"
            f"To: {src.meta['to']}\n\n"
        )
        copied = Unit(
            uid=doc.next_uid(),
            source_id=dst.source_id,
            author=src_unit.author,
            text=src_unit.text,
            keys=list(src_unit.keys),
            ops=list(src_unit.ops),
            role="forward_copy",
            meta={"prefix": prefix, "forwarded_from": src.source_id},
        )
        last = max(k for k, u in enumerate(doc.units) if u.source_id == dst.source_id)
        doc.units.insert(last + 1, copied)
        return NoiseOp(
            type=OP_FORWARD,
            params={
                "variant": "forward",
                "forwarded_message": src.source_id,
                "into_message": dst.source_id,
            },
        )
    j = rng.choice(quotable)
    src, dst = msgs[j - 1], msgs[j]
    src_text = doc.live_units(src.source_id, "writer")[0].text
    quoted = "\n".join(f"> {line}" if line else ">" for line in src_text.split("\n"))
    who = _person_display(ctx.world, src.meta["from"])
    dst.meta["tail"] = f"\n\nOn {src.meta['date_line']}, {who} wrote:\n{quoted}"
    return NoiseOp(
        type=OP_FORWARD,
        params={"variant": "quote", "quoted_message": src.source_id, "into_message": dst.source_id},
    )


# --------------------------------------------------------------------------
# Partial or garbled
# --------------------------------------------------------------------------


def _cp1252_char(byte: int) -> str:
    try:
        return bytes([byte]).decode("cp1252")
    except UnicodeDecodeError:
        return ""  # the five bytes cp1252 leaves undefined show as nothing


def mojibake(ch: str) -> str:
    """`ch` as UTF-8 bytes read back as Windows-1252 ("’" -> "â€™")."""
    raw = ch.encode("utf-8")
    if len(raw) == 1:
        return ch
    return "".join(_cp1252_char(b) for b in raw)


def _has_mojibake(text: str) -> bool:
    return any(mojibake(ch) != ch for ch in text)


def partial_variants(doc: DocText) -> list[str]:
    units = doc.live_units(role="writer")
    out: list[str] = []
    if doc.kind == DocKind.EMAIL:
        if len(doc.live_sources()) >= 2:
            out.append("truncated")
    elif doc.kind in (DocKind.CALL, DocKind.TICKET, DocKind.SLACK, DocKind.TEAMS):
        if len(units) >= 2:
            out.append("truncated")
    elif sum(len(u.text.split()) for u in units) >= 20:
        out.append("truncated")
    if doc.kind == DocKind.TICKET and len(units) >= 3:
        out.append("missing_comment")
    if any(_has_mojibake(u.text) for u in doc.live_units() if u.role != "header"):
        out.append("mojibake")
    if doc.kind != DocKind.CALL:
        out.append("html_leftovers")
    return out


def _truncate(doc: DocText, rng: random.Random) -> NoiseOp:
    op = PARTIAL_TRUNCATED
    if doc.kind == DocKind.EMAIL:
        msgs = doc.live_sources()
        keep = rng.randint(1, len(msgs) - 1)
        for s in msgs[keep:]:
            remove_source(doc, s, op)
        return NoiseOp(
            type=OP_PARTIAL,
            params={"variant": "truncated", "kept_messages": keep, "of": len(msgs)},
        )
    units = doc.live_units(role="writer")
    if doc.kind in (DocKind.CALL, DocKind.TICKET, DocKind.SLACK, DocKind.TEAMS):
        low = math.ceil(len(units) / 2) if doc.kind == DocKind.CALL else 1
        keep = rng.randint(max(1, low), len(units) - 1)
        for u in units[keep:]:
            remove_unit(doc, u, op)
        params: dict[str, Any] = {"variant": "truncated", "kept_units": keep, "of": len(units)}
        last = units[keep - 1]
        words = list(_WORD_RE.finditer(last.text))
        if doc.kind == DocKind.CALL and len(words) >= 4 and rng.random() < 0.5:
            cut_after = rng.randrange(1, len(words) - 1)
            cut_at = words[cut_after - 1].end()
            apply_edits(doc, last, [Edit(cut_at, len(last.text), "", op)])
            params["cut_mid_turn"] = True
        return NoiseOp(type=OP_PARTIAL, params=params)
    # Prose: cut mid-paragraph at a word boundary, 50 to 90% of the way in.
    total = sum(len(u.text) for u in units)
    target = int(total * rng.uniform(0.5, 0.9))
    seen = 0
    for k, u in enumerate(units):
        if seen + len(u.text) < target:
            seen += len(u.text)
            continue
        ends = [m.end() for m in _WORD_RE.finditer(u.text) if m.end() <= target - seen]
        cut_at = ends[-1] if ends else 0
        if cut_at == 0 and k > 0:
            for rest in units[k:]:
                remove_unit(doc, rest, op)
            kept = k
        else:
            cut_at = cut_at or (list(_WORD_RE.finditer(u.text))[0].end())
            if cut_at < len(u.text):
                apply_edits(doc, u, [Edit(cut_at, len(u.text), "", op)])
            for rest in units[k + 1 :]:
                remove_unit(doc, rest, op)
            kept = k + 1
        return NoiseOp(
            type=OP_PARTIAL,
            params={"variant": "truncated", "kept_paragraphs": kept, "of": len(units)},
        )
    return NoiseOp(type=OP_PARTIAL, params={"variant": "truncated", "kept_paragraphs": len(units)})


def _missing_comment(doc: DocText, rng: random.Random) -> NoiseOp:
    units = doc.live_units(role="writer")
    gone = units[rng.randrange(1, len(units) - 1)]
    remove_unit(doc, gone, PARTIAL_MISSING_COMMENT)
    return NoiseOp(
        type=OP_PARTIAL,
        params={"variant": "missing_comment", "comment_number": gone.meta["comment"]["number"]},
    )


def _window(units: list[Unit], rng: random.Random, want: Callable[[Unit], bool]) -> list[Unit]:
    starts = [k for k, u in enumerate(units) if want(u)]
    if not starts:
        return []
    a = rng.choice(starts)
    length = rng.randint(1, max(1, len(units) // 2))
    return units[a : a + length]


def _mojibake(doc: DocText, rng: random.Random) -> NoiseOp | None:
    units = [u for u in doc.live_units() if u.role != "header"]
    chosen = _window(units, rng, lambda u: _has_mojibake(u.text))
    changed = 0
    for u in chosen:
        edits = [
            Edit(i, i + 1, mojibake(ch), PARTIAL_MOJIBAKE)
            for i, ch in enumerate(u.text)
            if mojibake(ch) != ch
        ]
        if edits:
            apply_edits(doc, u, edits)
            changed += len(edits)
    if not changed:
        return None
    return NoiseOp(
        type=OP_PARTIAL,
        params={"variant": "mojibake", "units": len(chosen), "chars": changed},
    )


_HTML_ENTITIES = {"&": "&amp;", "'": "&#39;", "’": "&#39;", '"': "&quot;"}


def _html(doc: DocText, rng: random.Random) -> NoiseOp | None:
    units = [u for u in doc.live_units() if u.role != "header"]
    chosen = _window(units, rng, lambda u: bool(u.text.strip()))
    n_edits = 0
    for u in chosen:
        edits: list[Edit] = []
        text = u.text
        for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text):
            edits.append(Edit(m.start(), m.start(), "<p>", PARTIAL_HTML))
            edits.append(Edit(m.end(), m.end(), "</p>", PARTIAL_HTML))
            for i in range(m.start(), m.end()):
                ch = text[i]
                if ch == "\n":
                    edits.append(Edit(i, i + 1, "<br>\n", PARTIAL_HTML))
                elif ch in _HTML_ENTITIES:
                    edits.append(Edit(i, i + 1, _HTML_ENTITIES[ch], PARTIAL_HTML))
        if edits:
            apply_edits(doc, u, edits)
            n_edits += len(edits)
    if not n_edits:
        return None
    return NoiseOp(
        type=OP_PARTIAL,
        params={"variant": "html_leftovers", "units": len(chosen), "edits": n_edits},
    )


def _partial(doc: DocText, rng: random.Random, ctx: NoiseContext) -> NoiseOp | None:
    variants = partial_variants(doc)
    if not variants:
        return None
    wanted = ctx.config.partial_variants.get(doc.plan_id)
    variant = wanted if wanted in variants else rng.choice(variants)
    if variant == "truncated":
        return _truncate(doc, rng)
    if variant == "missing_comment":
        return _missing_comment(doc, rng)
    if variant == "mojibake":
        return _mojibake(doc, rng)
    return _html(doc, rng)


# --------------------------------------------------------------------------
# Misfiled
# --------------------------------------------------------------------------


def _account_family(world: World, account_id: str) -> set[str]:
    accounts = world.account_by_id()
    family = {account_id}
    acct = accounts.get(account_id)
    if acct is not None and acct.parent:
        family.add(acct.parent)
        family.update(a.id for a in world.accounts if a.parent == acct.parent)
    family.update(a.id for a in world.accounts if a.parent == account_id)
    return family


def _misfile(doc: DocText, rng: random.Random, ctx: NoiseContext) -> NoiseOp | None:
    true_id = ctx.plan.account_id
    if not true_id:
        return None
    family = _account_family(ctx.world, true_id)
    candidates: list[Account] = sorted(
        (a for a in ctx.world.accounts if a.id not in family), key=lambda a: a.id
    )
    if not candidates:
        return None
    wrong = rng.choice(candidates)
    surfaces = ["manifest_accounts"]
    params: dict[str, Any] = {"true_account_id": true_id, "filed_account_id": wrong.id}
    # Every code-rendered account field names the wrong account; the words
    # stay the writer's, so metadata and content disagree.
    account_line = f"Account: {wrong.name} ({wrong.crm_id})"
    for source in doc.live_sources():
        payload = source.meta.get("payload")
        if isinstance(payload, dict):
            for key, value in (("account", wrong.name), ("account_crm_id", wrong.crm_id)):
                if key in payload:
                    payload[key] = value
                    surfaces.append(f"{source.source_type.value}_{key}_field")
        lines = source.meta.get("header_lines")
        if lines:
            for i, line in enumerate(lines):
                if line.startswith("X-Related-Account:"):
                    lines[i] = f"X-Related-Account: {wrong.crm_id}"
                    surfaces.append("email_related_account_header")
        header = source.meta.get("header")
        if header and _ACCOUNT_LINE_RE.search(header):
            source.meta["header"] = _ACCOUNT_LINE_RE.sub(account_line, header)
            surfaces.append("call_account_header")
    for unit in doc.live_units(role="header"):
        m = _ACCOUNT_LINE_RE.search(unit.text)
        if m:
            apply_edits(doc, unit, [Edit(m.start(), m.end(), account_line, OP_MISFILED)])
            surfaces.append("doc_account_header")
    crm = ctx.plan.links.get("crm_account")
    if crm:
        surfaces.append("crm_link")
        params["crm_record_from"] = crm[0]
        params["crm_record_to"] = crm_account_source_id(wrong.crm_id)
    params["surfaces"] = sorted(set(surfaces))
    return NoiseOp(type=OP_MISFILED, params=params)


_ACCOUNT_LINE_RE = re.compile(r"(?m)^Account: [^\n]*$")


_DOC_OPS: dict[str, Callable[[DocText, random.Random, NoiseContext], NoiseOp | None]] = {
    OP_FORWARD: _forward,
    OP_ASR: _asr,
    OP_PARTIAL: _partial,
    OP_MISFILED: _misfile,
}


# --------------------------------------------------------------------------
# One document
# --------------------------------------------------------------------------


def _passthrough(written: WrittenDoc, plan: DocPlan, note: str | None) -> NoisedDoc:
    data = written.model_dump(exclude={"labels", "decoy_spans"})
    labels = [NoisedLabel.model_validate(label.model_dump()) for label in written.labels]
    decoys = [NoisedDecoySpan.model_validate(d.model_dump()) for d in written.decoy_spans]
    return NoisedDoc(
        **data,
        labels=labels,
        decoy_spans=decoys,
        true_account_id=plan.account_id,
        filed_account_id=plan.account_id,
        noise_notes=[note] if note else [],
        noise_counts=_counts(len(labels), len(decoys), labels, decoys, []),
    )


def _counts(
    labels_in: int,
    decoys_in: int,
    labels: list[NoisedLabel],
    decoys: list[NoisedDecoySpan],
    drops: list[LabelDrop],
) -> dict[str, int]:
    primary = [x for x in labels if x.copy_of is None]
    primary_d = [x for x in decoys if x.copy_of is None]
    return {
        "labels_in": labels_in,
        "labels_kept": len(primary),
        "labels_remapped": sum(1 for x in primary if x.remapped),
        "labels_text_changed": sum(1 for x in primary if x.noise),
        "labels_dropped": sum(1 for d in drops if d.slot_kind == "fact"),
        "label_copies_added": len(labels) - len(primary),
        "decoys_in": decoys_in,
        "decoys_kept": len(primary_d),
        "decoys_remapped": sum(1 for x in primary_d if x.remapped),
        "decoys_dropped": sum(1 for d in drops if d.slot_kind == "decoy"),
        "decoy_copies_added": len(decoys) - len(primary_d),
    }


def anchors_survive(before: str, after: str, fact_id: str, world: World) -> bool:
    """Spec section 5: a garbled span is kept only if its anchors survive.
    Each anchor of the fact (or a nickname of it, `voice.short_forms`) that
    the span said before noise must still be said after it."""
    fact = next((f for f in world.facts if f.id == fact_id), None)
    if fact is None:
        return True
    old, new = _norm(before), _norm(after)
    for anchor in fact.anchors:
        forms = [anchor, *(f.text for f in short_forms(anchor, world))]
        said = [form for form in forms if _contains_term(old, form)]
        if said and not any(_contains_term(new, form) for form in said):
            return False
    return True


def _moved(copy: SpanCopy, doc_id: str, start: int, end: int) -> bool:
    return (copy.source_id, copy.start, copy.end) != (doc_id, start, end)


def _new_fields(copy: SpanCopy, kind: DocKind) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "doc_id": copy.source_id,
        "char_start": copy.start,
        "char_end": copy.end,
        "text": copy.text,
        "noise": sorted(copy.ops),
    }
    if copy.prepared_locator is not None:
        fields["prepared_locator"] = copy.prepared_locator
        if kind == DocKind.EMAIL:
            fields["locator"] = copy.prepared_locator
    return fields


def noise_document(
    written: WrittenDoc,
    plan: DocPlan,
    world: World,
    ops: list[str],
    *,
    config: NoiseConfig,
    entity_words: frozenset[str] | None = None,
) -> NoisedDoc:
    """Apply `ops` (in `DOC_OP_ORDER`) to one written document and remap
    every label and decoy span. A document that does not round-trip through
    parse, render and prepare is returned unchanged with a note."""
    if written.status != "written" or not ops:
        return _passthrough(written, plan, None)
    try:
        doc = parse_written(
            plan_id=written.plan_id,
            kind=written.kind,
            doc_date=plan.date,
            files=dict(written.files),
            prepared=[(p.source_id, p.source_type, p.path) for p in written.prepared_sources],
            prepared_text={p.source_id: p.text for p in written.prepared_sources},
        )
    except ParseError as exc:
        return _passthrough(written, plan, f"not noised: {exc}")
    if render_files(doc) != written.files:
        return _passthrough(written, plan, "not noised: files do not round-trip")
    before = locate(doc)
    if [p.text for p in before.prepared] != [p.text for p in written.prepared_sources]:
        return _passthrough(written, plan, "not noised: prepared text does not round-trip")
    assign_original_keys(doc, before)
    index0 = key_index(doc)
    spans = [(x.doc_id, x.char_start, x.char_end) for x in written.labels] + [
        (x.doc_id, x.char_start, x.char_end) for x in written.decoy_spans
    ]
    for sid, s, e in spans:
        if any((sid, k) not in index0 for k in range(s, e)):
            return _passthrough(written, plan, "not noised: a label lies outside the units")

    ctx = _context(world, plan, config, doc, written, entity_words or entity_words_for(world))
    applied: list[NoiseOp] = []
    for op in DOC_OP_ORDER:
        if op not in ops:
            continue
        record = _DOC_OPS[op](doc, _rng(config.seed, op, written.plan_id), ctx)
        if record is not None:
            applied.append(record)
    if not applied:
        return _passthrough(written, plan, "no operation applied")

    after = locate(doc)
    return _finish(written, plan, world, doc, after, applied)


def _finish(
    written: WrittenDoc,
    plan: DocPlan,
    world: World,
    doc: DocText,
    after: Located,
    applied: list[NoiseOp],
) -> NoisedDoc:
    index = key_index(doc)
    prepared_text = {p.source_id: p.text for p in after.prepared}
    labels: list[NoisedLabel] = []
    decoys: list[NoisedDecoySpan] = []
    drops: list[LabelDrop] = []
    notes: list[str] = []
    if after.mismatches:
        notes.append(f"{after.mismatches} character(s) could not be placed after noise")

    def drop(kind, marker_id, x, reason, op=None) -> None:  # noqa: ANN001
        drops.append(
            LabelDrop(
                slot_kind=kind,
                marker_id=marker_id,
                plan_id=written.plan_id,
                doc_id=x.doc_id,
                char_start=x.char_start,
                char_end=x.char_end,
                text=x.text,
                reason=reason,
                noise_op=op,
            )
        )

    for label in written.labels:
        remap = remap_span(
            doc,
            index,
            after,
            source_id=label.doc_id,
            start=label.char_start,
            end=label.char_end,
            cut_ops=CUT_OPS,
        )
        found = [(remap.primary, False)] if remap.primary else []
        found += [(c, True) for c in remap.copies]
        if remap.primary is None:
            drop("fact", label.fact_id, label, remap.drop_reason or "span_lost", remap.drop_op)
        notes.extend(f"{label.fact_id} copy not kept: {f}" for f in remap.copy_failures)
        for copy, is_copy in found:
            data = label.model_dump()
            data.update(_new_fields(copy, written.kind))
            data.update(
                span_words=len(copy.text.split()),
                asr_in_span=OP_ASR in copy.ops,
                remapped=_moved(copy, label.doc_id, label.char_start, label.char_end),
                copy_of=(
                    f"{label.doc_id}:{label.char_start}-{label.char_end}" if is_copy else None
                ),
            )
            new = NoisedLabel.model_validate(data)
            if copy.ops & GARBLE_OPS and not anchors_survive(
                label.text, new.text, label.fact_id, world
            ):
                if is_copy:
                    notes.append(f"{label.fact_id} copy not kept: garbled_anchor_lost")
                else:
                    garble = sorted(copy.ops & GARBLE_OPS)[0]
                    drop("fact", label.fact_id, label, "garbled_anchor_lost", garble)
                continue
            labels.append(new)

    for span in written.decoy_spans:
        remap = remap_span(
            doc,
            index,
            after,
            source_id=span.doc_id,
            start=span.char_start,
            end=span.char_end,
            cut_ops=CUT_OPS,
        )
        if remap.primary is None:
            drop("decoy", span.decoy_id, span, remap.drop_reason or "span_lost", remap.drop_op)
        found = [(remap.primary, False)] if remap.primary else []
        found += [(c, True) for c in remap.copies]
        for copy, is_copy in found:
            data = span.model_dump()
            data.update(_new_fields(copy, written.kind))
            data.update(
                remapped=_moved(copy, span.doc_id, span.char_start, span.char_end),
                copy_of=(f"{span.doc_id}:{span.char_start}-{span.char_end}" if is_copy else None),
            )
            decoys.append(NoisedDecoySpan.model_validate(data))

    for label in labels:
        assert prepared_text[label.doc_id][label.char_start : label.char_end] == label.text
    for span in decoys:
        assert prepared_text[span.doc_id][span.char_start : span.char_end] == span.text

    filed = next(
        (op.params["filed_account_id"] for op in applied if op.type == OP_MISFILED),
        plan.account_id,
    )
    data = written.model_dump(
        exclude={"labels", "decoy_spans", "files", "prepared_sources", "word_count"}
    )
    return NoisedDoc(
        **data,
        files=after.files,
        prepared_sources=[
            PreparedSourceRecord(
                source_id=p.source_id, source_type=p.source_type, path=p.path, text=p.text
            )
            for p in after.prepared
        ],
        word_count=sum(len(u.text.split()) for u in doc.live_units(role="writer")),
        labels=labels,
        decoy_spans=decoys,
        noise=applied,
        label_drops=drops,
        true_account_id=plan.account_id,
        filed_account_id=filed,
        noise_notes=notes,
        noise_counts=_counts(len(written.labels), len(written.decoy_spans), labels, decoys, drops),
    )


# --------------------------------------------------------------------------
# World-level: links, supersession, off-topic
# --------------------------------------------------------------------------


def link_noise(
    world_plan: WorldDocPlan, config: NoiseConfig
) -> tuple[list[NoiseOp], dict[str, dict[str, list[str]]]]:
    """Spec section 5, "Stale or missing links, 10% of links". WP3 already
    leaves about that share unwired (`missing_link:<doc>:<record>` on both
    sides); `stale_link_rate` of those become stale instead: the record
    points at a document of the same kind from another account (a link to
    the same account's own document would be a correct link). The rest stay
    missing. Returns the records and the resulting `linked_*` lists."""
    plans = world_plan.doc_by_id()
    record_of = {r.account_id: r.id for r in world_plan.crm_accounts}
    overrides = {
        r.id: {f: list(getattr(r, f)) for f in _LINK_FIELDS} for r in world_plan.crm_accounts
    }
    by_field: dict[str, list[DocPlan]] = {}
    for plan in world_plan.doc_plans:
        fld = _LINK_FIELD_BY_KIND.get(plan.kind)
        if fld is not None and plan.account_id in record_of:
            by_field.setdefault(fld, []).append(plan)
    missing: list[tuple[str, str]] = []
    for plan in world_plan.doc_plans:
        for note in plan.noise_ops:
            if note.startswith("missing_link:"):
                _tag, doc_id, record_id = note.split(":", 2)
                missing.append((doc_id, record_id))
    chosen = select_docs(
        [f"{r}|{d}" for d, r in missing], config.stale_link_rate, config.seed, OP_STALE_LINK
    )
    ops: list[NoiseOp] = []
    for doc_id, record_id in sorted(missing):
        tplan = plans.get(doc_id)
        fld = _LINK_FIELD_BY_KIND.get(tplan.kind) if tplan else None
        stale: str | None = None
        if f"{record_id}|{doc_id}" in chosen and fld and record_id in overrides:
            current = overrides[record_id][fld]
            candidates = sorted(
                p.id
                for p in by_field.get(fld, [])
                if record_of.get(p.account_id) != record_id and p.id not in current
            )
            if candidates:
                stale = _rng(config.seed, OP_STALE_LINK, doc_id).choice(candidates)
                current.append(stale)
        if stale is None:
            ops.append(
                NoiseOp(type=OP_MISSING_LINK, params={"doc_id": doc_id, "record_id": record_id})
            )
        else:
            ops.append(
                NoiseOp(
                    type=OP_STALE_LINK,
                    params={
                        "record_id": record_id,
                        "field": fld,
                        "true_target": doc_id,
                        "stale_target": stale,
                        "stale_target_account_id": plans[stale].account_id,
                    },
                )
            )
    return ops, overrides


def _bookkeeping_ops(doc: NoisedDoc, plan: DocPlan, world: World) -> list[NoiseOp]:
    ops: list[NoiseOp] = []
    if plan.role == DocRole.OFF_TOPIC:
        ops.append(NoiseOp(type=OP_OFF_TOPIC, params={}))
    facts = {f.id: f for f in world.facts}
    for fact_id in sorted({x.fact_id for x in doc.labels}):
        fact = facts.get(fact_id)
        if fact is not None and (fact.supersedes or fact.valid_to):
            ops.append(
                NoiseOp(
                    type=OP_SUPERSEDED,
                    params={
                        "fact_id": fact.id,
                        "supersedes": fact.supersedes,
                        "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
                    },
                )
            )
    return ops


# --------------------------------------------------------------------------
# The whole world
# --------------------------------------------------------------------------


@dataclass
class NoiseRun:
    docs: dict[str, NoisedDoc]
    manifest: dict[str, Any]
    report: dict[str, Any]


def apply_noise(
    world: World,
    world_plan: WorldDocPlan,
    docs: dict[str, WrittenDoc],
    *,
    config: NoiseConfig,
    near_duplicates: Callable[[dict[str, WrittenDoc]], Any] | None = None,
) -> NoiseRun:
    """Every code operation on every written document, plus the world's
    link noise. `near_duplicates`, when given, is called with the written
    documents and returns `noise_neardup.NearDupResult` (the writer calls)."""
    plans = world_plan.doc_by_id()
    chosen = plan_noise(docs, plans, config)
    entity_words = entity_words_for(world)
    out: dict[str, NoisedDoc] = {}
    for doc_id in sorted(docs):
        written = docs[doc_id]
        plan = plans.get(doc_id)
        if plan is None or written.status != "written":
            continue
        noised = noise_document(
            written,
            plan,
            world,
            chosen.get(doc_id, []),
            config=config,
            entity_words=entity_words,
        )
        noised.noise.extend(_bookkeeping_ops(noised, plan, world))
        out[doc_id] = noised

    new_facts: list[dict[str, Any]] = []
    supersessions: list[dict[str, Any]] = []
    near_report: dict[str, Any] = {}
    if near_duplicates is not None:
        result = near_duplicates(docs)
        for nd in result.docs.values():
            out[nd.plan_id] = nd
            for op in nd.noise:
                shadow_id = op.params.get("shadow_id")
                if op.type == OP_NEAR_DUPLICATE and nd.status == "written" and shadow_id in out:
                    out[shadow_id].noise.append(
                        NoiseOp(
                            type=OP_NEAR_DUPLICATE,
                            params={
                                "role": "original",
                                "near_duplicate_id": nd.plan_id,
                                "superseded_fact_id": op.params.get("changed_fact_id"),
                            },
                        )
                    )
        new_facts = [f.model_dump(mode="json") for f in result.new_facts]
        supersessions = list(result.supersessions)
        near_report = result.report

    link_ops, overrides = link_noise(world_plan, config)
    for doc in out.values():
        for op in doc.noise:
            if op.type == OP_MISFILED and "crm_record_from" in op.params:
                src, dst = op.params["crm_record_from"], op.params["crm_record_to"]
                for fld in _LINK_FIELDS:
                    if doc.plan_id in overrides.get(src, {}).get(fld, []) and dst in overrides:
                        overrides[src][fld].remove(doc.plan_id)
                        overrides[dst][fld].append(doc.plan_id)
    manifest = {
        "noise_version": NOISE_VERSION,
        "seed": config.seed,
        "world_id": world.id,
        "docs": {
            d.plan_id: {
                "kind": d.kind.value,
                "status": d.status,
                "noise": [op.model_dump() for op in d.noise],
                "true_account_id": d.true_account_id,
                "filed_account_id": d.filed_account_id,
            }
            for d in sorted(out.values(), key=lambda d: d.plan_id)
        },
        "links": [op.model_dump() for op in link_ops],
        "crm_linked_overrides": overrides,
        "supersessions": supersessions,
        "new_facts": new_facts,
    }
    report = build_report(out, docs, plans, link_ops, world_plan, near_report)
    return NoiseRun(out, manifest, report)


# --------------------------------------------------------------------------
# Noise after the completeness pass (the runner's order: WP4, WP5, WP6, WP7)
# --------------------------------------------------------------------------


def prepared_sha256(doc: WrittenDoc) -> str:
    """One hash over a document's prepared sources: which version of the
    text a label set indexes."""
    payload = [[p.source_id, p.text] for p in doc.prepared_sources]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def carrier_slot_id(label: dict[str, Any]) -> str:
    """The slot a shipped label travels through the noise stage under: its
    WP4 slot, or for a label WP5 added (no slot) a key built from its
    identity. WP7 joins WP6's output back to the label by this id."""
    slot = label.get("slot_id")
    if slot:
        return str(slot)
    key = json.dumps(
        [
            label.get("fact_id"),
            label.get("doc_id"),
            label.get("char_start"),
            label.get("char_end"),
            label.get("teacher_model"),
            str(label.get("fact_type")),
        ]
    )
    return "silver-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


_CARRIED_FIELDS = tuple(
    f for f in WriterLabel.model_fields if f not in ("slot_id", "writer_model", "reference_mode")
)


def carrier_label(label: BaseModel, writer_model: str | None) -> WriterLabel:
    """A shipped label (WP4's, or WP5's scripted or silver) as the
    `WriterLabel` the code operations remap: its span, ids and locators,
    under `carrier_slot_id`."""
    data = label.model_dump(mode="json")
    fields = {k: data[k] for k in _CARRIED_FIELDS if k in data}
    fields.setdefault("plan_id", data.get("plan_id") or "")
    fields.setdefault("span_words", len(str(data.get("text", "")).split()))
    return WriterLabel(
        **fields,
        slot_id=carrier_slot_id(data),
        writer_model=data.get("writer_model") or writer_model or "",
        reference_mode=data.get("reference_mode") or "explicit",
    )


def noise_completed_world(
    world: World,
    world_plan: WorldDocPlan,
    written: dict[str, WrittenDoc],
    states: dict[str, Any],
    *,
    shipped: frozenset[str],
    config: NoiseConfig,
    near_duplicates: Callable[[dict[str, WrittenDoc]], Any] | None = None,
) -> NoiseRun:
    """The noise stage over WP5's output: each shipped document's final
    version (WP5's rewrite, if any) carrying every label WP5 ships, scripted
    and silver, so their spans move with the text; near-duplicates are
    written from the same final versions with their WP4 labels. `states` are
    WP5's `DocCompleteness` by plan id; `shipped` its shipping statuses.
    Each noised document records the version it came from
    (`noised_from_sha256`)."""
    finals: dict[str, WrittenDoc] = {}
    carried: dict[str, WrittenDoc] = {}
    for plan_id, state in sorted(states.items()):
        original = written.get(plan_id)
        if original is None or state.status not in shipped:
            continue
        final = state.final_doc(original)
        finals[plan_id] = final
        labels = [
            carrier_label(x, final.writer_model)
            for x in [*state.scripted_labels, *state.silver_labels]
        ]
        carried[plan_id] = final.model_copy(update={"labels": labels})
    run = apply_noise(
        world,
        world_plan,
        carried,
        config=config,
        near_duplicates=(lambda _docs: near_duplicates(finals)) if near_duplicates else None,
    )
    for plan_id, doc in run.docs.items():
        if plan_id in finals:
            doc.noised_from_sha256 = prepared_sha256(finals[plan_id])
    return run


def build_report(
    out: dict[str, NoisedDoc],
    docs: dict[str, WrittenDoc],
    plans: dict[str, DocPlan],
    link_ops: list[NoiseOp],
    world_plan: WorldDocPlan,
    near_report: dict[str, Any],
) -> dict[str, Any]:
    written = [d for d in docs.values() if d.status == "written"]
    by_kind = Counter(d.kind.value for d in written)
    op_docs: Counter = Counter()
    variants: Counter = Counter()
    wers: list[float] = []
    totals: Counter = Counter()
    for d in out.values():
        totals.update(d.noise_counts)
        for op in d.noise:
            op_docs[op.type] += 1
            if op.type == OP_PARTIAL:
                variants[op.params.get("variant")] += 1
            if op.type == OP_ASR:
                wers.append(op.params["wer"])
    wired = sum(len(getattr(r, f)) for r in world_plan.crm_accounts for f in _LINK_FIELDS)
    link_counts = Counter(op.type for op in link_ops)
    eligible_links = wired + link_counts[OP_MISSING_LINK] + link_counts[OP_STALE_LINK]
    drops = Counter(x.reason for d in out.values() for x in d.label_drops)
    return {
        "noise_version": NOISE_VERSION,
        "documents_written": len(written),
        "documents_by_kind": dict(sorted(by_kind.items())),
        "documents_per_operation": dict(sorted(op_docs.items())),
        "partial_variants": dict(sorted(variants.items())),
        "rates": {
            OP_MISFILED: _share(op_docs[OP_MISFILED], len(written)),
            OP_FORWARD: _share(op_docs[OP_FORWARD], by_kind.get("email", 0)),
            OP_ASR: _share(op_docs[OP_ASR], by_kind.get("call", 0)),
            OP_PARTIAL: _share(op_docs[OP_PARTIAL], len(written)),
            "stale_or_missing_links": _share(
                link_counts[OP_STALE_LINK] + link_counts[OP_MISSING_LINK], eligible_links
            ),
        },
        "asr_wer": {"min": min(wers), "max": max(wers)} if wers else None,
        "links": dict(link_counts),
        "labels": dict(totals),
        "drop_reasons": dict(sorted(drops.items())),
        "near_duplicates": near_report,
        "per_document": [
            {
                "plan_id": d.plan_id,
                "kind": d.kind.value,
                "status": d.status,
                "operations": [
                    op.type + (f":{op.params['variant']}" if "variant" in op.params else "")
                    for op in d.noise
                ],
                **d.noise_counts,
                "notes": d.noise_notes,
            }
            for d in sorted(out.values(), key=lambda d: d.plan_id)
        ],
    }


def _share(n: int, of: int) -> float | None:
    return round(n / of, 4) if of else None


def write_outputs(run: NoiseRun, out_dir: Path) -> None:
    """`docs/<plan_id>.json`, `files/...`, `labels.jsonl`, `decoys.jsonl`,
    `label_drops.jsonl`, `noise_manifest.json`, `noise_report.json`."""
    from corpus.generator.v2.writer import _atomic_write

    out_dir.mkdir(parents=True, exist_ok=True)
    docs = sorted(run.docs.values(), key=lambda d: d.plan_id)
    for d in docs:
        _atomic_write(out_dir / "docs" / f"{d.plan_id}.json", d.model_dump_json(indent=2))
        if d.status == "written":
            for rel, content in d.files.items():
                _atomic_write(out_dir / "files" / rel, content)
    _atomic_write(
        out_dir / "labels.jsonl",
        "".join(x.model_dump_json() + "\n" for d in docs for x in d.labels),
    )
    _atomic_write(
        out_dir / "decoys.jsonl",
        "".join(x.model_dump_json() + "\n" for d in docs for x in d.decoy_spans),
    )
    _atomic_write(
        out_dir / "label_drops.jsonl",
        "".join(x.model_dump_json() + "\n" for d in docs for x in d.label_drops),
    )
    _atomic_write(out_dir / "noise_manifest.json", json.dumps(run.manifest, indent=2))
    _atomic_write(out_dir / "noise_report.json", json.dumps(run.report, indent=2))


# --------------------------------------------------------------------------
# CLI: the WP6 live smoke (`python -m corpus.generator.v2.noise smoke`)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_docs(paths: list[Path]) -> dict[str, WrittenDoc]:
    docs: dict[str, WrittenDoc] = {}
    for path in paths:
        doc = WrittenDoc.model_validate_json(path.read_text(encoding="utf-8"))
        docs[doc.plan_id] = doc
    return docs


def main(argv: list[str] | None = None) -> int:
    from corpus.generator.v2.models import Split
    from corpus.generator.v2.noise_neardup import NearDupConfig, write_near_duplicates
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.planner import plan_world
    from corpus.generator.v2.simulator import simulate_world
    from corpus.generator.v2.writer import (
        OpenRouterWriterBackend,
        RunBudget,
        WriterConfig,
        WriterPool,
        WriterSession,
    )
    from portolan.training.spend import SpendMeter, default_spend_path

    parser = argparse.ArgumentParser(prog="python -m corpus.generator.v2.noise")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="noise WP4's smoke documents")
    smoke.add_argument("--pack", default="medical_devices")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--months", type=int, default=8)
    smoke.add_argument("--split", default="train")
    smoke.add_argument("--noise-seed", type=int, default=None)
    smoke.add_argument(
        "--in", dest="in_dir", type=Path, default=REPO_ROOT / "var" / "data" / "synth_v2" / "smoke"
    )
    smoke.add_argument("--extra-doc", type=Path, action="append", default=[])
    smoke.add_argument(
        "--out", type=Path, default=REPO_ROOT / "var" / "data" / "synth_v2" / "smoke_wp6"
    )
    smoke.add_argument("--force-all", action="store_true")
    smoke.add_argument(
        "--partial-variant", action="append", default=[], help="doc_id=variant, repeatable"
    )
    smoke.add_argument("--near-duplicates", type=int, default=2)
    smoke.add_argument("--budget-usd", type=float, default=0.50)
    smoke.add_argument("--spend-path", type=Path, default=default_spend_path(REPO_ROOT / "var"))
    smoke.add_argument(
        "--completeness",
        type=Path,
        default=None,
        help="WP5's output for these documents: noise WP5's shipped versions and every "
        "label WP5 ships (the runner's order), instead of WP4's",
    )
    smoke.add_argument(
        "--no-type-cap",
        action="store_true",
        help="plan the world without the plan-time per-type cap (documents written "
        "before 2026-09-23)",
    )
    smoke.add_argument("--env-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    import os

    from corpus.generator.prose import _env_or_dotenv
    from corpus.generator.v2.planner import TYPE_CAP_SHARE
    from portolan.training.pipeline import assert_not_holdout

    assert_not_holdout(args.out)
    api_key = _env_or_dotenv(args.env_root, "OPENROUTER_API_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key  # the writer backend reads it from here
    pack = load_pack(REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / f"{args.pack}.yaml")
    world = simulate_world(pack, seed=args.seed, months=args.months, split=Split(args.split))
    world_plan = plan_world(world, pack, type_cap=None if args.no_type_cap else TYPE_CAP_SHARE)
    docs = _load_docs(sorted((args.in_dir / "docs").glob("*.json")) + list(args.extra_doc))
    config = NoiseConfig(
        seed=args.noise_seed if args.noise_seed is not None else world.seed,
        force=frozenset(DOC_OP_ORDER) if args.force_all else frozenset(),
        partial_variants=dict(item.split("=", 1) for item in args.partial_variant),
    )
    writer_config = WriterConfig()
    pool = WriterPool(writer_config.models, world_doc_total=len(world_plan.doc_plans))
    for d in docs.values():
        if d.status == "written" and d.writer_model in writer_config.models:
            pool.assign(d.writer_model)
    session = WriterSession(
        backend=OpenRouterWriterBackend(),
        pool=pool,
        budget=RunBudget(cap_usd=args.budget_usd, meter=SpendMeter(args.spend_path)),
        config=writer_config,
        call_log=args.out / "calls.jsonl",
    )

    def near(written: dict[str, WrittenDoc]):  # noqa: ANN202
        return write_near_duplicates(
            world,
            world_plan,
            pack,
            written,
            session=session,
            config=NearDupConfig(seed=config.seed, limit=args.near_duplicates),
            out_dir=args.out,
            progress=lambda line: print(line, flush=True),
        )

    if args.completeness is not None:
        from corpus.generator.v2.completeness import SHIPPED_STATUSES, DocCompleteness

        states = {
            s.plan_id: s
            for s in (
                DocCompleteness.model_validate_json(p.read_text(encoding="utf-8"))
                for p in sorted((args.completeness / "docs").glob("*.json"))
            )
        }
        run = noise_completed_world(
            world,
            world_plan,
            docs,
            states,
            shipped=SHIPPED_STATUSES,
            config=config,
            near_duplicates=near,
        )
    else:
        run = apply_noise(world, world_plan, docs, config=config, near_duplicates=near)
    write_outputs(run, args.out)
    print(json.dumps({k: run.report[k] for k in ("documents_per_operation", "labels")}, indent=2))
    print(f"near-duplicate spend ${session.budget.spent_usd:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
