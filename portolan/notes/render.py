"""`slugify` and `render_note` (`build-contracts.md` "Notes and claims").

Body shape, per `portolan-spec-v2.md` sec 2 and the dossier pattern ported
from `02-reference-implementation/wiki-tooling/wiki-schema.md` ("synthesis
above, append-only claims below"):

    ---
    title: ...
    description: "..."
    type: ...
    created: YYYY-MM-DD
    updated: YYYY-MM-DD
    status: living
    source:
      - some-source-slug
    stale_after: YYYY-MM-DD
    schema_version: "0.1"
    crm_id: A-1011          # entity notes
    parent: larkspur        # only on account, when it has a parent
    superseded_by: slug     # only when set
    ---

    ## Synthesis

    <note.synthesis, or a one-line default from note fields, no model call>

    ## Related

    - [[slug]]
    - [[slug]]

    <!-- CLAIMS:APPEND-ONLY -->

    ## Claims

    - **YYYY-MM-DD** | fact_type | claim text ^[source-slug @ locator] (quote: "...")

`## Related` is this build's addition to the contract: the contract's
wikilinks all point up (account->parent, signal->account), so nothing
would link *to* a signal or action note and the lint (spec sec 6.1) would
flag every one as an orphan. The entity a signal/action belongs to keeps a
list of links down to them (`persist.link_related`).

`read_note` parses the synthesis and related list back out of `body_md`, so
a claims-only edit keeps them (`models.py`).
"""

from __future__ import annotations

import re

from portolan.notes.models import Claim, NoteRow, NoteType
from portolan.util import clock

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Matches `^[source-slug @ locator]` (provenance-chain.md).
CITATION_RE = re.compile(r"\^\[\s*([^\]@]+?)\s*@\s*([^\]]*)\]")

# `[[slug]]`, lowercase slugs only (no `[[Title|alias]]`). Allows `_` because
# `signal_note_slug` embeds fact types such as `competitor_mention` as-is.
WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9_-]*)\]\]")

SYNTHESIS_HEADING = "## Synthesis"
RELATED_HEADING = "## Related"
CLAIMS_SEPARATOR = "<!-- CLAIMS:APPEND-ONLY -->"
CLAIMS_HEADING = "## Claims"


def plain_text(text: str) -> str:
    """`text` with every `[[slug]]` wikilink de-hyphenated into plain
    words (`[[tidewater-t2]]` -> `tidewater t2`), so a stored, already-
    wikilinked claim compares fairly against a fresh fact's plain prose
    -- e.g. for an embedding or a `difflib` ratio.
    """
    return WIKILINK_RE.sub(lambda m: m.group(1).replace("-", " "), text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors; `0.0` if either is
    the zero vector (undefined otherwise -- treated as "no similarity"
    rather than raising).
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def slugify(title: str) -> str:
    """A stable, lowercase, hyphenated slug: "Tidewater T2" -> "tidewater-t2".

    Entity note slugs are `slugify(canonical_name)` per the build
    instructions ("ashgrove-health", "maria-chen", "tidewater-t2").
    """
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug or "note"


def signal_note_slug(fact_type: str, account_slug: str) -> str:
    """`signal-<fact_type>-<anchor-slug>`; the anchor may be an account,
    product or competitor note."""
    return f"signal-{fact_type}-{account_slug}"


def action_note_slug(action_id: int, what: str) -> str:
    """`action-<n>-<slug-of-what>`, `n` being the row's `actions.id`
    (`portolan.merge.actions.reserve_action_note_slug`)."""
    return f"action-{action_id}-{slugify(what)[:60]}"


def link_mentions(text: str, mention_slug_pairs: list[tuple[str, str]]) -> str:
    """Replace each resolved entity mention in `text` with `[[slug]]`
    (notes contract: "every entity a claim mentions is written as
    [[slug]] in the claim text when the registry resolves it").

    Longest mentions are replaced first so a short mention that happens to
    be a substring of a longer one already replaced (e.g. "Ashgrove" inside
    "Ashgrove Medical Center") doesn't clobber part of an already-inserted
    `[[slug]]` token. Each mention is replaced once, at its first
    occurrence. A mention not found verbatim in `text` (extraction
    paraphrased it, or it was itself an already-resolved id rather than
    literal prose) is appended parenthetically at the end instead, so the
    entity is still linked from this claim -- silently dropping it would
    both violate the contract and risk leaving the target note with no
    inbound edge (see module docstring on `## Related`).
    """
    out = text
    appended: list[str] = []
    for mention, slug in sorted(mention_slug_pairs, key=lambda p: len(p[0]), reverse=True):
        token = f"[[{slug}]]"
        if token in out:
            continue  # already linked (e.g. two mentions resolved to the same entity)
        span = _find_whole_mention(out, mention) if mention else None
        if span is not None:
            out = out[: span[0]] + token + out[span[1] :]
        else:
            appended.append(token)
    if appended:
        out = f"{out} ({' '.join(appended)})"
    return out


def _find_whole_mention(text: str, mention: str) -> tuple[int, int] | None:
    """The first place `mention` stands in `text` as whole words, outside
    any `[[...]]` already there. A bare substring match rewrote the inside
    of longer words: the product "Arc" in "Arcadia Hospital uses Arc"
    became "[[arc]]adia Hospital uses Arc"."""
    linked = [m.span() for m in WIKILINK_RE.finditer(text)]
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(mention)}(?![A-Za-z0-9])")
    for match in pattern.finditer(text):
        if not any(start <= match.start() < end for start, end in linked):
            return match.span()
    return None


# ---------------------------------------------------------------------
# Parsing an existing body back into (synthesis, related, claims-section
# presence) -- used by `persist.read_note`.
# ---------------------------------------------------------------------


def parse_synthesis(body_md: str) -> str | None:
    """Everything between the `## Synthesis` heading and whichever of
    `## Related`/the claims separator comes first (or the end of the
    body, if neither is present).

    Does not stop at other `## ` headings: a synthesis may carry its own
    (the company-context note's does).
    """
    if SYNTHESIS_HEADING not in body_md:
        return None
    after = body_md.split(SYNTHESIS_HEADING, 1)[1]
    stop_at = len(after)
    for marker in (f"\n{RELATED_HEADING}", f"\n{CLAIMS_SEPARATOR}"):
        idx = after.find(marker)
        if idx != -1:
            stop_at = min(stop_at, idx)
    return after[:stop_at].strip("\n") or None


def parse_related(body_md: str) -> list[str]:
    if RELATED_HEADING not in body_md:
        return []
    after = body_md.split(RELATED_HEADING, 1)[1]
    stop_at = after.find(f"\n{CLAIMS_SEPARATOR}")
    section = after if stop_at == -1 else after[:stop_at]
    return [m.group(1) for m in re.finditer(r"^-\s*\[\[([a-z0-9_-]+)\]\]", section, re.M)]


# ---------------------------------------------------------------------
# Default synthesis for a note with no override set (registry-fields-only,
# "no model call") -- see module docstring.
# ---------------------------------------------------------------------


def _default_synthesis(note: NoteRow) -> str:
    if note.type == NoteType.ACCOUNT:
        line1 = f"{note.title} is an account on file (CRM {note.crm_id})."
        if note.parent_slug:
            line1 += f" Part of [[{note.parent_slug}]]."
        return line1
    if note.type == NoteType.CONTACT:
        return f"{note.title} is a contact on file (CRM {note.crm_id})."
    if note.type == NoteType.PRODUCT:
        return f"{note.title} is a product on file."
    if note.type == NoteType.COMPETITOR:
        return f"{note.title} is a named competitor."
    if note.type == NoteType.SIGNAL:
        return f"{note.title} tracks one fact type across its linked account."
    if note.type == NoteType.ACTION:
        return f"{note.title} is an open action."
    return note.description or note.title


def _yaml_str(value: str) -> str:
    """Always-quoted YAML scalar (wiki-schema.md: quote anything with a
    colon, `#`, quotes, or a leading `[[` -- simplest safe rule is to
    always quote)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_frontmatter(note: NoteRow, source_slugs: list[str]) -> str:
    # Dates print as the deployment's calendar day (`clock.world_date`): a
    # timestamptz read back carries the DB session's zone, which may differ,
    # and would change the rendered date between two identical writes.
    lines = ["---"]
    lines.append(f"title: {_yaml_str(note.title)}")
    lines.append(f"description: {_yaml_str(note.description or note.title)}")
    lines.append(f"type: {note.type.value}")
    lines.append(f"created: {clock.world_date(note.created):%Y-%m-%d}")
    lines.append(f"updated: {clock.world_date(note.updated):%Y-%m-%d}")
    lines.append(f"status: {note.status.value}")
    if source_slugs:
        lines.append("source:")
        for slug in source_slugs:
            lines.append(f"  - {slug}")
    else:
        lines.append("source: []")
    if note.stale_after is not None:
        lines.append(f"stale_after: {clock.world_date(note.stale_after):%Y-%m-%d}")
    if note.schema_version:
        lines.append(f'schema_version: "{note.schema_version}"')
    if note.crm_id:
        lines.append(f"crm_id: {note.crm_id}")
    if note.parent_slug:
        lines.append(f"parent: {note.parent_slug}")
    if note.superseded_by:
        lines.append(f"superseded_by: {note.superseded_by}")
    lines.append("---")
    return "\n".join(lines)


def _render_claim_line(claim: Claim) -> str:
    marker = f"^[{claim.source_slug} @ {claim.locator}]"
    # Extra sightings of the same fact render as more markers on this line.
    extra_markers = "".join(f"^[{c.source_slug} @ {c.locator}]" for c in claim.citations)
    quote = f' (quote: "{claim.quote}")' if claim.quote else ""
    # Plain formatting, not `clock.world_date`: hand-built test claims sit at
    # midnight UTC, which conversion would roll back a day
    # (`tests/test_notes.py::test_render_note_strikes_through_a_superseded_claim`).
    line = (
        f"- **{claim.recorded:%Y-%m-%d}** | {claim.fact_type} | {claim.claim_text} "
        f"{marker}{extra_markers}{quote}"
    )
    if claim.superseded_at is not None:
        line = f"~~{line}~~ → superseded {claim.superseded_at:%Y-%m-%d}"
    return line


def render_note(note: NoteRow, claims: list[Claim]) -> str:
    """Markdown with YAML frontmatter, a `## Synthesis` section, an
    optional `## Related` section, then the append-only `## Claims`
    section (module docstring).

    `claims` must already be filtered to unarchived claims by the caller
    (the contract's `Claim` model carries no `archived` field -- see
    `models.py`).
    """
    # `source:` lists every source the body cites, extra citations included.
    source_slugs = sorted(
        {c.source_slug for c in claims}
        | {citation.source_slug for c in claims for citation in c.citations}
    )
    parts = [_render_frontmatter(note, source_slugs), ""]

    parts.append(SYNTHESIS_HEADING)
    parts.append("")
    parts.append(note.synthesis or _default_synthesis(note))
    parts.append("")

    if note.related:
        parts.append(RELATED_HEADING)
        parts.append("")
        for slug in note.related:
            parts.append(f"- [[{slug}]]")
        parts.append("")

    parts.append(CLAIMS_SEPARATOR)
    parts.append("")
    parts.append(CLAIMS_HEADING)
    parts.append("")

    ordered = sorted(claims, key=lambda c: (c.recorded, c.id or 0))
    if ordered:
        for claim in ordered:
            parts.append(_render_claim_line(claim))
    else:
        parts.append("No claims recorded yet.")

    return "\n".join(parts).rstrip() + "\n"
