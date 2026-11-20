"""The eleven lint checks (portolan-spec-v2.md sec 6.1, build-contracts.md).

Each `check_*` function takes an open `psycopg.Connection` and returns a flat
`list[LintFinding]` for that one check; `run_lint` (portolan/lint/report.py)
calls all eleven.

Severity map (two tiers, `portolan/lint/models.py`):

    error   broken_link, orphan, bad_header, unresolved_citation,
            missing_original, and the `stale_after`-in-the-past half of
            `stale`
    warning thin_links, relative_time, ambiguity, human_edit_contradiction,
            growth_guard, and the `updated`-older-than-90-days half of
            `stale`
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import psycopg

from portolan.context import COMPANY_CONTEXT_BODY_CHAR_CAP
from portolan.lint._store import Store
from portolan.lint.models import LintFinding
from portolan.notes.render import parse_synthesis

# ---------------------------------------------------------------------
# 1. broken_link
# ---------------------------------------------------------------------


def check_broken_link(conn: psycopg.Connection) -> list[LintFinding]:
    """`links.to_slug` (kind body or parent) with no matching `notes.slug`;
    provenance links whose target is not in `sources`."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT l.from_slug, l.to_slug, l.kind
            FROM links l
            LEFT JOIN notes n ON n.slug = l.to_slug
            WHERE l.kind IN ('body', 'parent') AND n.slug IS NULL
            ORDER BY l.from_slug, l.to_slug
            """
        )
        for from_slug, to_slug, kind in cur.fetchall():
            findings.append(
                LintFinding(
                    check="broken_link",
                    severity="error",
                    note_slug=from_slug,
                    message=f"{kind} link to [[{to_slug}]] does not resolve to any note",
                )
            )

        cur.execute(
            """
            SELECT l.from_slug, l.to_slug
            FROM links l
            LEFT JOIN sources s ON s.slug = l.to_slug
            WHERE l.kind = 'provenance' AND s.slug IS NULL
            ORDER BY l.from_slug, l.to_slug
            """
        )
        for from_slug, to_slug in cur.fetchall():
            findings.append(
                LintFinding(
                    check="broken_link",
                    severity="error",
                    note_slug=from_slug,
                    source_slug=to_slug,
                    message=f"provenance link to [[{to_slug}]] does not resolve to any source",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 2. orphan
# ---------------------------------------------------------------------


def check_orphan(conn: psycopg.Connection) -> list[LintFinding]:
    """A living note with no inbound body/parent links, and not of type
    `source`, `report`, or `context`: those are entry points into the graph,
    not things other notes are expected to link to."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug
            FROM notes n
            WHERE n.status = 'living'
              AND n.type NOT IN ('source', 'report', 'context')
              AND NOT EXISTS (
                  SELECT 1 FROM links l
                  WHERE l.to_slug = n.slug AND l.kind IN ('body', 'parent')
              )
            ORDER BY n.slug
            """
        )
        for (slug,) in cur.fetchall():
            findings.append(
                LintFinding(
                    check="orphan",
                    severity="error",
                    note_slug=slug,
                    message="no inbound body or parent links reach this note",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 3. thin_links
# ---------------------------------------------------------------------


def check_thin_links(conn: psycopg.Connection) -> list[LintFinding]:
    """A living note with fewer than 2 outbound body links (warning)."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug, COUNT(l.to_slug) AS n_links
            FROM notes n
            LEFT JOIN links l ON l.from_slug = n.slug AND l.kind = 'body'
            WHERE n.status = 'living'
            GROUP BY n.slug
            HAVING COUNT(l.to_slug) < 2
            ORDER BY n.slug
            """
        )
        for slug, n_links in cur.fetchall():
            findings.append(
                LintFinding(
                    check="thin_links",
                    severity="warning",
                    note_slug=slug,
                    message=f"only {n_links} outbound body link(s); at least 2 expected",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 4. stale
# ---------------------------------------------------------------------


def check_stale(conn: psycopg.Connection, *, today: date) -> list[LintFinding]:
    """`stale_after` in the past (error); or a living note with `updated`
    older than 90 days (warning). The two clocks are independent (spec sec
    2), so both conditions are checked independently rather than one
    shadowing the other — a note can be both declared-overdue and merely
    old."""
    findings: list[LintFinding] = []
    ninety_days_ago = today - timedelta(days=90)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, stale_after
            FROM notes
            WHERE stale_after IS NOT NULL AND stale_after < %s
            ORDER BY slug
            """,
            (today,),
        )
        for slug, stale_after in cur.fetchall():
            findings.append(
                LintFinding(
                    check="stale",
                    severity="error",
                    note_slug=slug,
                    message=f"stale_after ({stale_after:%Y-%m-%d}) is in the past",
                )
            )

        cur.execute(
            """
            SELECT slug, updated
            FROM notes
            WHERE status = 'living' AND updated < %s
            ORDER BY slug
            """,
            (ninety_days_ago,),
        )
        for slug, updated in cur.fetchall():
            age_days = (today - updated.date()).days
            findings.append(
                LintFinding(
                    check="stale",
                    severity="warning",
                    note_slug=slug,
                    message=f"living note not updated in {age_days}d (last: {updated:%Y-%m-%d})",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 5. bad_header
# ---------------------------------------------------------------------


def check_bad_header(conn: psycopg.Connection) -> list[LintFinding]:
    """Status outside the closed list is impossible by constraint, so this
    checks the other header defects: missing description, missing `crm_id` on
    account/contact notes, `parent_slug` pointing at a non-account,
    `superseded` notes with no `superseded_by`, and a context note body over
    its cap.

    `crm_id` is NOT NULL-enforced on account/contact by the DB CHECK
    constraint, but an empty string satisfies NOT NULL, so it is still
    checked here for blankness, not just presence."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug FROM notes
            WHERE description IS NULL OR trim(description) = ''
            ORDER BY slug
            """
        )
        for (slug,) in cur.fetchall():
            findings.append(
                LintFinding(
                    check="bad_header",
                    severity="error",
                    note_slug=slug,
                    message="missing description",
                )
            )

        cur.execute(
            """
            SELECT slug, type FROM notes
            WHERE type IN ('account', 'contact')
              AND (crm_id IS NULL OR trim(crm_id) = '')
            ORDER BY slug
            """
        )
        for slug, note_type in cur.fetchall():
            findings.append(
                LintFinding(
                    check="bad_header",
                    severity="error",
                    note_slug=slug,
                    message=f"{note_type} note has no crm_id",
                )
            )

        cur.execute(
            """
            SELECT n.slug, n.parent_slug, p.type
            FROM notes n
            JOIN notes p ON p.slug = n.parent_slug
            WHERE n.parent_slug IS NOT NULL AND p.type != 'account'
            ORDER BY n.slug
            """
        )
        for slug, parent_slug, parent_type in cur.fetchall():
            findings.append(
                LintFinding(
                    check="bad_header",
                    severity="error",
                    note_slug=slug,
                    message=f"parent_slug [[{parent_slug}]] is a {parent_type}, not an account",
                )
            )

        cur.execute(
            """
            SELECT slug FROM notes
            WHERE status = 'superseded' AND superseded_by IS NULL
            ORDER BY slug
            """
        )
        for (slug,) in cur.fetchall():
            findings.append(
                LintFinding(
                    check="bad_header",
                    severity="error",
                    note_slug=slug,
                    message="status is superseded but superseded_by is not set",
                )
            )

        # The context note's body is capped at `COMPANY_CONTEXT_BODY_CHAR_CAP`.
        # The edit route enforces it on save; this catches a note written any other way.
        cur.execute("SELECT slug, body_md FROM notes WHERE type = 'context' ORDER BY slug")
        for slug, body_md in cur.fetchall():
            body = parse_synthesis(body_md or "") or ""
            if len(body) > COMPANY_CONTEXT_BODY_CHAR_CAP:
                findings.append(
                    LintFinding(
                        check="bad_header",
                        severity="error",
                        note_slug=slug,
                        message=(
                            f"context note body is {len(body)} chars, over the "
                            f"{COMPANY_CONTEXT_BODY_CHAR_CAP}-char cap"
                        ),
                    )
                )

    return findings


# ---------------------------------------------------------------------
# 6. relative_time
# ---------------------------------------------------------------------

# The reference implementation's RELTIME_RE
# (`02-reference-implementation/wiki-tooling/wiki-lint.py`) plus this month,
# next quarter, currently.
_RELATIVE_TIME_RE = re.compile(
    r"\b(yesterday|today|tomorrow|recently|last week|next week|this morning"
    r"|this month|next quarter|currently)\b",
    re.IGNORECASE,
)

# Masked out of note bodies before the relative-time regex runs: code spans, and
# quoted speech (the speaker's words, not an undated claim by the note).
_CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_QUOTE_RE = re.compile(r"^\s*>.*$|“[^”]*”|\"[^\"\n]{0,400}\"", re.M)


def _mask(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def check_relative_time(conn: psycopg.Connection) -> list[LintFinding]:
    """Relative-time language in `notes.body_md` or `claims.claim_text`
    (warning) — a durable graph should read the same in six months."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute("SELECT slug, body_md FROM notes ORDER BY slug")
        for slug, body_md in cur.fetchall():
            prose = _mask(_mask(body_md or "", _CODE_RE), _QUOTE_RE)
            hits = sorted({m.group(0).lower() for m in _RELATIVE_TIME_RE.finditer(prose)})
            if hits:
                findings.append(
                    LintFinding(
                        check="relative_time",
                        severity="warning",
                        note_slug=slug,
                        message="relative-time language: " + ", ".join(hits),
                    )
                )

        cur.execute("SELECT id, note_slug, claim_text FROM claims ORDER BY id")
        for claim_id, note_slug, claim_text in cur.fetchall():
            hits = sorted(
                {m.group(0).lower() for m in _RELATIVE_TIME_RE.finditer(claim_text or "")}
            )
            if hits:
                findings.append(
                    LintFinding(
                        check="relative_time",
                        severity="warning",
                        claim_id=claim_id,
                        note_slug=note_slug,
                        message="relative-time language: " + ", ".join(hits),
                    )
                )

    return findings


# ---------------------------------------------------------------------
# 7. unresolved_citation
# ---------------------------------------------------------------------

# `^[source-slug @ locator]` per build-contracts.md and provenance-chain.md.
_CITATION_RE = re.compile(r"\^\[\s*([^\]@]+?)\s*@\s*([^\]]*)\]")


def check_unresolved_citation(conn: psycopg.Connection) -> list[LintFinding]:
    """Every `^[slug @ locator]` marker in `notes.body_md` must name a slug
    present in `sources`; a claim whose `locator` is blank is an error.
    (`claims.source_slug` is a NOT NULL foreign key to `sources.slug`, so a
    claim citing an unknown source cannot exist.)"""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM sources")
        known_sources = {row[0] for row in cur.fetchall()}

        cur.execute("SELECT slug, body_md FROM notes ORDER BY slug")
        for slug, body_md in cur.fetchall():
            for match in _CITATION_RE.finditer(body_md or ""):
                cited_slug = match.group(1).strip()
                if cited_slug not in known_sources:
                    findings.append(
                        LintFinding(
                            check="unresolved_citation",
                            severity="error",
                            note_slug=slug,
                            message=f"citation marker ^[{cited_slug} @ ...] names no known source",
                        )
                    )

        cur.execute(
            """
            SELECT id, note_slug FROM claims
            WHERE trim(locator) = ''
            ORDER BY id
            """
        )
        for claim_id, note_slug in cur.fetchall():
            findings.append(
                LintFinding(
                    check="unresolved_citation",
                    severity="error",
                    claim_id=claim_id,
                    note_slug=note_slug,
                    message="claim has an empty locator",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 8. missing_original
# ---------------------------------------------------------------------


def check_missing_original(conn: psycopg.Connection, store: Store) -> list[LintFinding]:
    """A `sources` row whose `s3_key` is set but `store.exists(s3_key)` is
    False; or status `done` with no `s3_key`."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute("SELECT slug, s3_key FROM sources WHERE s3_key IS NOT NULL ORDER BY slug")
        rows = cur.fetchall()

    for slug, s3_key in rows:
        if not store.exists(s3_key):
            findings.append(
                LintFinding(
                    check="missing_original",
                    severity="error",
                    source_slug=slug,
                    message=f"original not found in store at key '{s3_key}'",
                )
            )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug FROM sources
            WHERE extraction_status = 'done' AND s3_key IS NULL
            ORDER BY slug
            """
        )
        for (slug,) in cur.fetchall():
            findings.append(
                LintFinding(
                    check="missing_original",
                    severity="error",
                    source_slug=slug,
                    message="extraction_status is 'done' but no s3_key is recorded",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 9. ambiguity
# ---------------------------------------------------------------------

_AMBIGUITY_KINDS = ("ambiguous_entity", "unknown_entity")
_AMBIGUITY_OLDEST_N = 5


def check_ambiguity(conn: psycopg.Connection) -> list[LintFinding]:
    """Open `review_queue` rows of kind `ambiguous_entity` or
    `unknown_entity`, summarised as counts with the oldest five listed
    (warning)."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kind, COUNT(*) FROM review_queue
            WHERE status = 'open' AND kind = ANY(%s)
            GROUP BY kind
            ORDER BY kind
            """,
            (list(_AMBIGUITY_KINDS),),
        )
        counts = cur.fetchall()
        if counts:
            summary = ", ".join(f"{n} {kind}" for kind, n in counts)
            findings.append(
                LintFinding(
                    check="ambiguity",
                    severity="warning",
                    message=f"{sum(n for _, n in counts)} open review_queue rows ({summary})",
                )
            )

        cur.execute(
            """
            SELECT id, kind, source_slug, created
            FROM review_queue
            WHERE status = 'open' AND kind = ANY(%s)
            ORDER BY created ASC
            LIMIT %s
            """,
            (list(_AMBIGUITY_KINDS), _AMBIGUITY_OLDEST_N),
        )
        for row_id, kind, source_slug, created in cur.fetchall():
            findings.append(
                LintFinding(
                    check="ambiguity",
                    severity="warning",
                    source_slug=source_slug,
                    message=f"review_queue #{row_id} ({kind}), open since {created:%Y-%m-%d}",
                )
            )

    return findings


# ---------------------------------------------------------------------
# 10. human_edit_contradiction
# ---------------------------------------------------------------------


def check_human_edit_contradiction(conn: psycopg.Connection) -> list[LintFinding]:
    """A claim with `human_edit_id` set that has a later claim on the same
    note, with the same `fact_type` and overlapping entities, whose text
    differs (warning).

    Entity overlap is approximated: two claims overlap when the notes with a
    'provenance' link to claim A's source and to claim B's source intersect.
    That is a loose filter on top of "same note, same fact_type"; claims carry
    no entity ids to do better.
    """
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, note_slug, fact_type, claim_text, source_slug, recorded
            FROM claims
            WHERE human_edit_id IS NOT NULL
            ORDER BY id
            """
        )
        edited_claims = cur.fetchall()

        for (
            edited_id,
            note_slug,
            fact_type,
            edited_text,
            edited_source,
            edited_recorded,
        ) in edited_claims:
            cur.execute(
                """
                SELECT id, claim_text, source_slug, recorded
                FROM claims
                WHERE note_slug = %s
                  AND fact_type = %s
                  AND id != %s
                  AND recorded > %s
                  AND claim_text != %s
                ORDER BY id
                """,
                (note_slug, fact_type, edited_id, edited_recorded, edited_text),
            )
            later_claims = cur.fetchall()
            if not later_claims:
                continue

            edited_entities = _provenance_notes(cur, edited_source)

            for later_id, later_text, later_source, _later_recorded in later_claims:
                later_entities = _provenance_notes(cur, later_source)
                if edited_entities & later_entities:
                    findings.append(
                        LintFinding(
                            check="human_edit_contradiction",
                            severity="warning",
                            note_slug=note_slug,
                            claim_id=later_id,
                            message=(
                                f"claim #{later_id} ('{later_text}') contradicts "
                                f"human-edited claim #{edited_id} ('{edited_text}'), "
                                f"same fact_type '{fact_type}'"
                            ),
                        )
                    )

    return findings


def _provenance_notes(cur: psycopg.Cursor, source_slug: str) -> set[str]:
    """Notes with a 'provenance' link pointing at `source_slug` (the entity-overlap
    stand-in in `check_human_edit_contradiction`)."""
    cur.execute(
        "SELECT from_slug FROM links WHERE kind = 'provenance' AND to_slug = %s",
        (source_slug,),
    )
    return {row[0] for row in cur.fetchall()}


# ---------------------------------------------------------------------
# 11. growth_guard
# ---------------------------------------------------------------------

_GROWTH_GUARD_LIMIT = 40


def check_growth_guard(conn: psycopg.Connection) -> list[LintFinding]:
    """A note with more than 40 unarchived claims (warning)."""
    findings: list[LintFinding] = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT note_slug, COUNT(*) AS n
            FROM claims
            WHERE archived = false
            GROUP BY note_slug
            HAVING COUNT(*) > %s
            ORDER BY note_slug
            """,
            (_GROWTH_GUARD_LIMIT,),
        )
        for slug, n in cur.fetchall():
            findings.append(
                LintFinding(
                    check="growth_guard",
                    severity="warning",
                    note_slug=slug,
                    message=f"{n} unarchived claims (> {_GROWTH_GUARD_LIMIT}); consolidate",
                )
            )

    return findings
