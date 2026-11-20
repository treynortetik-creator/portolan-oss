"""`needs_consolidation` (`01-spec/quality-loop.md` "The consolidation
pass": "Trigger: a note has taken more than N updates since its last
consolidation, or its claim list passes M").

"Since its last consolidation" is tracked without a new table (this wave
owns no migrations): a successful `consolidate_note` appends a `changelog`
row, `kind="consolidate"`, with `details = {"note_slug": ..., "version":
<the note's version right after consolidating>}` (`run.py`). "Updates
since" is then just `notes.version - version_at_last_consolidation` -- a
note with no consolidation row yet compares against version 0, so any
note with at least `updates_threshold` versions ever (consolidated or
not) is a candidate the first time this runs.
"""

from __future__ import annotations

import psycopg

# N: default update-count trigger (`01-spec/quality-loop.md`: "more than N
# updates since its last consolidation"). Chosen for this build: frequent
# enough that a note under active churn (several merge-stage writes a
# week) gets consolidated on a roughly monthly cadence, without triggering
# on the first handful of ordinary writes every note picks up just from
# being created and touched a few times.
CONSOLIDATION_UPDATES_THRESHOLD = 12

# M: default claim-count trigger, same spec sentence ("or its claim list
# passes M"). Set below the lint's own `growth_guard` warning threshold
# (40, `portolan/lint/checks.py`) so consolidation has a chance to fire and
# bring a note back under control before the lint ever has to warn about
# it, rather than the two thresholds racing.
CONSOLIDATION_CLAIMS_THRESHOLD = 30

__all__ = [
    "CONSOLIDATION_CLAIMS_THRESHOLD",
    "CONSOLIDATION_UPDATES_THRESHOLD",
    "last_consolidated_versions",
    "needs_consolidation",
]


def last_consolidated_versions(conn: psycopg.Connection) -> dict[str, int]:
    """`{note_slug: version}` as of each note's most recent `kind=
    'consolidate'` changelog row. A note never consolidated is absent
    (callers treat that as version 0)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (details->>'note_slug')
                   details->>'note_slug', (details->>'version')::int
            FROM changelog
            WHERE kind = 'consolidate' AND details ? 'note_slug'
            ORDER BY details->>'note_slug', occurred_at DESC
            """
        )
        return {slug: version for slug, version in cur.fetchall() if slug is not None}


def needs_consolidation(
    conn: psycopg.Connection,
    *,
    updates_threshold: int = CONSOLIDATION_UPDATES_THRESHOLD,
    claims_threshold: int = CONSOLIDATION_CLAIMS_THRESHOLD,
) -> list[str]:
    """Note slugs due for consolidation: `notes.version` has advanced by
    more than `updates_threshold` since the note's last consolidation, or
    it carries more than `claims_threshold` unarchived claims. Slugs are
    returned sorted, so a repeated call against an unchanged graph is
    stable (useful for `run_consolidation`'s `--limit` to mean "the same
    N notes" across a re-run, not an arbitrary subset)."""
    with conn.cursor() as cur:
        cur.execute("SELECT slug, version FROM notes")
        current_versions = dict(cur.fetchall())

        cur.execute(
            """
            SELECT note_slug, COUNT(*) FROM claims
            WHERE archived = false
            GROUP BY note_slug
            """
        )
        claim_counts = dict(cur.fetchall())

    last_consolidated = last_consolidated_versions(conn)

    candidates: set[str] = set()
    for slug, version in current_versions.items():
        since_last = version - last_consolidated.get(slug, 0)
        if since_last > updates_threshold:
            candidates.add(slug)

    for slug, count in claim_counts.items():
        if count > claims_threshold:
            candidates.add(slug)

    return sorted(candidates)
