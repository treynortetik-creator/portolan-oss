"""`portolan.report.diff` tests (wave C1) against the real fake-driven
pipeline graph (`tests/test_report_helpers.py`'s session-scoped
`full_graph` fixture).
"""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml
from test_report_helpers import FullGraph, full_graph  # noqa: F401

from portolan.notes.render import slugify
from portolan.report import diff

# `claims.recorded` (and `valid_from`) carry the *fictional* corpus date
# (`portolan.merge.util.recorded_dt`: "recorded = source.source_date") --
# this is the window `quiet_accounts` and the theme/outlier feeds use.
# `notes.created`/`notes.updated` and `claims.superseded_at`, by contrast,
# are always stamped at real wall-clock write time
# (`portolan.notes.persist.write_note`: "note.updated = datetime.now(UTC)",
# `portolan.merge.facts._supersede`: "superseded_at = now()") -- so
# `notes_created`/`notes_updated`/`superseded_facts`, which all read those
# columns, need a window around the actual test-run moment, not the
# corpus's fictional 2026 dates.
WINDOW_START = date(2026, 3, 2)
AS_OF = date(2026, 9, 4)
TODAY_WINDOW_START = date.today() - timedelta(days=1)
TODAY_WINDOW_END = date.today() + timedelta(days=1)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _truth_table() -> dict:
    with (REPO_ROOT / "corpus" / "truth-table.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _quiet_account_slugs_from_truth_table() -> set[str]:
    """Full account-note slugs (`slugify(canonical_name)`) for
    `corpus/truth-table.yaml` `expectations.report.
    quiet_accounts_as_of_2026-09-11` (v1.1, 2026-09-13 -- supersedes v0's
    `..._as_of_2026-09-04`, which named vireo and st-aldric too; both
    picked up new activity in v1.1's August facts and are no longer
    quiet), minus `kestrel` -- see `diff.quiet_accounts`'s own docstring
    for why a churned/inactive account is excluded from this feed on
    purpose."""
    truth = _truth_table()
    names_by_id = {a["id"]: a["name"] for a in truth["accounts"]}
    quiet_ids = truth["expectations"]["report"]["quiet_accounts_as_of_2026-09-11"]
    return {slugify(names_by_id[account_id]) for account_id in quiet_ids if account_id != "kestrel"}


EXPECTED_QUIET_30 = _quiet_account_slugs_from_truth_table()


def test_quiet_accounts_30_day_includes_the_named_accounts(full_graph: FullGraph) -> None:
    result = diff.quiet_accounts(full_graph.conn, AS_OF)
    assert EXPECTED_QUIET_30.issubset(set(result.quiet_30))
    assert result.as_of == AS_OF


def test_quiet_accounts_excludes_the_churned_inactive_account(full_graph: FullGraph) -> None:
    result = diff.quiet_accounts(full_graph.conn, AS_OF)
    assert "kestrel-medical-group" not in result.quiet_30
    assert "kestrel-medical-group" not in result.quiet_90


def test_quiet_accounts_90_day_is_a_subset_of_30_day(full_graph: FullGraph) -> None:
    result = diff.quiet_accounts(full_graph.conn, AS_OF)
    assert set(result.quiet_90).issubset(set(result.quiet_60))
    assert set(result.quiet_60).issubset(set(result.quiet_30))


def test_quiet_accounts_ignores_a_claim_recorded_after_as_of(full_graph: FullGraph) -> None:
    """Bughunt-2026-09-18 finding 5: the lateral join computing each
    account's own last touch had no upper bound at `as_of` at all -- a
    claim recorded AFTER `as_of` (planted between two runs of the same
    past week's report, say) still counted as "last touch," so an
    account genuinely quiet as of `as_of` un-quieted itself the moment
    later material landed. Re-running a PAST week's report must
    reproduce the same quiet-accounts list every time (the knowledge
    clock) -- not change depending on when it happens to be re-run."""
    conn = full_graph.conn
    target = next(iter(EXPECTED_QUIET_30))
    before = diff.quiet_accounts(conn, AS_OF)
    assert target in before.quiet_30, "test needs a genuinely quiet account to start from"

    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM notes WHERE type = 'account' LIMIT 1")
        (home_note_slug,) = cur.fetchone()
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded)
            VALUES (%s, 'praise', %s, %s, 'test#1', %s)
            """,
            (
                home_note_slug,
                f"[[{target}]] said things are great now (test, after as_of).",
                any_source,
                # Far in the future (this file's own existing convention
                # for a "clearly after everything" test claim -- see
                # `test_superseded_facts_reports_a_manually_superseded_
                # pair`'s '2099-01-01') -- not just after `AS_OF`
                # (2026-09-04), but after any "today +/- a day" window
                # another test in this SESSION-scoped `full_graph` might
                # also check quiet_accounts against.
                datetime(2099, 1, 1, tzinfo=UTC),
            ),
        )
    conn.commit()

    after = diff.quiet_accounts(conn, AS_OF)
    assert target in after.quiet_30, (
        "a claim recorded after as_of must not un-quiet an account when re-reading a past as_of"
    )


def test_notes_created_covers_every_account_note(full_graph: FullGraph) -> None:
    created = diff.notes_created(full_graph.conn, since=TODAY_WINDOW_START, as_of=TODAY_WINDOW_END)
    slugs = {c.slug for c in created}
    assert "ashgrove-medical-center" in slugs
    assert all(c.created <= c.updated for c in created)


def test_superseded_facts_reports_a_manually_superseded_pair(full_graph: FullGraph) -> None:
    """`portolan.fake_extractor`'s compare heuristic "never verdicts
    'contradicts'" (its own module docstring) -- a fake-LLM-driven run has
    no naturally-occurring supersession to find (confirmed: F11/F11b, the
    corpus's own planted reversal, land as two independent claims with no
    `superseded_by` link when merged through the fake compare path). So
    this checks `superseded_facts` itself, directly, against one pair this
    test supersedes by hand -- the two clocks (spec sec 2) are a pure-SQL
    read either way, and this is what `merge_fact._supersede` itself does
    under the hood on a real contradiction."""
    conn = full_graph.conn
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, note_slug, source_slug FROM claims WHERE note_slug = 'ashgrove-health' "
            "LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT id, note_slug, source_slug FROM claims LIMIT 1")
            row = cur.fetchone()
        old_id, note_slug, source_slug = row
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded)
            VALUES (%s, 'decision', 'renewal moved to 2099-01-01 (test)', %s, 'test#1', now())
            RETURNING id
            """,
            (note_slug, source_slug),
        )
        new_id = cur.fetchone()[0]
        cur.execute(
            "UPDATE claims SET superseded_at = now(), superseded_by = %s WHERE id = %s",
            (new_id, old_id),
        )
    conn.commit()

    superseded = diff.superseded_facts(conn, since=TODAY_WINDOW_START, as_of=TODAY_WINDOW_END)
    matches = [s for s in superseded if s.new_claim_id == new_id]
    assert len(matches) == 1
    assert matches[0].old_claim_id == old_id
    assert "2099-01-01" in matches[0].new_text


def test_build_diff_assembles_all_four_pieces(full_graph: FullGraph) -> None:
    result = diff.build_diff(full_graph.conn, since=TODAY_WINDOW_START, as_of=TODAY_WINDOW_END)
    assert result.since == TODAY_WINDOW_START
    assert result.as_of == TODAY_WINDOW_END
    assert result.notes_created
    assert EXPECTED_QUIET_30.issubset(set(result.quiet_accounts.quiet_30))


# --------------------------------------------------------------------
# Item 4 (e2e-run follow-up wave): `clock="world"`. This fixture's own
# graph is the perfect demonstration of the bug: `full_graph` runs the
# whole pipeline in one sitting, so every note's `created`/`updated`
# (wall-clock write time) lands at THIS TEST SESSION's real run moment --
# nowhere near the corpus's own fictional 2026 dates -- while `claims.
# recorded` (`portolan.merge.util.recorded_dt`: "= source.source_date")
# carries the corpus's real planted dates throughout. `clock="record"`
# against the corpus's own `[WINDOW_START, AS_OF]` window must therefore
# find nothing; `clock="world"` must find the corpus's real activity.
# --------------------------------------------------------------------


def test_notes_created_record_clock_finds_nothing_in_the_corpus_own_window(
    full_graph: FullGraph,
) -> None:
    result = diff.notes_created(full_graph.conn, since=WINDOW_START, as_of=AS_OF, clock="record")
    assert result == []


def test_notes_created_world_clock_matches_earliest_claim_recorded(full_graph: FullGraph) -> None:
    """Cross-check `notes_created(clock="world")`'s SQL against an
    independent Python recomputation from `claims.recorded` directly,
    rather than a hardcoded slug list that would be fragile to any
    truth-table edit."""
    conn = full_graph.conn
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug, MIN(recorded::date) FROM claims GROUP BY note_slug")
        earliest_by_note = dict(cur.fetchall())
    expected_created = {
        slug for slug, earliest in earliest_by_note.items() if WINDOW_START <= earliest <= AS_OF
    }
    assert expected_created, "expected at least one note with claims in the corpus's own window"

    world_mode = diff.notes_created(full_graph.conn, since=WINDOW_START, as_of=AS_OF, clock="world")
    assert {c.slug for c in world_mode} == expected_created


def test_notes_updated_world_clock_requires_an_earlier_claim_too(full_graph: FullGraph) -> None:
    # A later `since` than `WINDOW_START` (which sits right at the
    # corpus's own earliest claim -- almost nothing predates it, so there
    # is no "already existed before the window" case to find there): a
    # mid-corpus cut instead gives both notes newly claimed on-or-after it
    # (`notes_created`) and notes with an earlier claim too
    # (`notes_updated`).
    since = date(2026, 7, 1)
    as_of = AS_OF

    conn = full_graph.conn
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug, recorded::date FROM claims")
        rows = cur.fetchall()
    by_note: dict[str, list] = {}
    for slug, day in rows:
        by_note.setdefault(slug, []).append(day)
    expected_updated = {
        slug
        for slug, days in by_note.items()
        if any(since <= d <= as_of for d in days) and any(d < since for d in days)
    }
    assert expected_updated, "expected at least one note touched both before and during the window"

    world_mode = diff.notes_updated(conn, since=since, as_of=as_of, clock="world")
    assert {c.slug for c in world_mode} == expected_updated

    # Module docstring: a note is reported once, under `notes_created`,
    # never both -- true in either clock mode.
    created_slugs = {
        c.slug for c in diff.notes_created(conn, since=since, as_of=as_of, clock="world")
    }
    assert created_slugs.isdisjoint(expected_updated)


def test_build_diff_world_clock_finds_the_corpus_own_activity(full_graph: FullGraph) -> None:
    result = diff.build_diff(full_graph.conn, since=WINDOW_START, as_of=AS_OF, clock="world")
    assert result.notes_created or result.notes_updated


def test_clock_argument_rejects_an_unknown_value(full_graph: FullGraph) -> None:
    import pytest

    with pytest.raises(ValueError):
        diff.notes_created(full_graph.conn, since=WINDOW_START, as_of=AS_OF, clock="yesterday")
    with pytest.raises(ValueError):
        diff.build_diff(full_graph.conn, since=WINDOW_START, as_of=AS_OF, clock="yesterday")


# --------------------------------------------------------------------
# Test gap 2 (review finding 1, blocker). `test_notes_created_world_
# clock_matches_earliest_claim_recorded` above recomputes its own
# expectation with `MIN(recorded::date)` -- the exact same bare cast the
# bug is in, so it is structurally blind to a session timezone bug (it
# would shift the SAME (wrong) way on both sides of the comparison). This
# test instead asserts against `sources.source_date` -- the one
# independent ground truth -- and pins the database session to a
# timezone the *test process* is not itself running under, exercising
# both `'UTC'` and `'America/Phoenix'` (the review's own two-session
# demonstration), on a plain, fresh `db` rather than the shared, session-
# scoped `full_graph` fixture (so `SET TIME ZONE` here can't leak into
# any other test).
# --------------------------------------------------------------------


def test_notes_created_world_clock_is_correct_under_any_session_timezone(db) -> None:
    from portolan.merge.util import recorded_dt
    from tests.merge_helpers import make_source_row

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description) VALUES "
            "('tz-test-signal', 'signal', 'TZ Test Signal', "
            "'A note used only for the session-timezone regression test.')"
        )

    source_date = date(2026, 8, 24)
    source = make_source_row(db, "s-tz-test-001", source_type="ticket", source_date=source_date)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "recorded) VALUES (%s, 'complaint', "
            "'a claim used only for the timezone regression test', %s, 'test#1', %s)",
            ("tz-test-signal", source.slug, recorded_dt(source)),
        )

    # The review's own proof, reproduced as an assertion: a 2026-08-24
    # source must land inside a window that includes 08-24, and must NOT
    # be pulled into one ending the day before it (08-23) -- regardless
    # of the database session's own `TimeZone`.
    included_since, included_as_of = source_date - timedelta(days=1), source_date
    excluded_since = included_since - timedelta(days=1)
    excluded_as_of = included_as_of - timedelta(days=1)

    for session_tz in ("UTC", "America/Phoenix"):
        with db.cursor() as cur:
            cur.execute(f"SET TIME ZONE '{session_tz}'")

        included = {
            c.slug
            for c in diff.notes_created(
                db, since=included_since, as_of=included_as_of, clock="world"
            )
        }
        assert "tz-test-signal" in included, (
            f"session TimeZone={session_tz!r}: a source dated {source_date} should land in "
            f"the world-clock window [{included_since}, {included_as_of}]"
        )

        excluded = {
            c.slug
            for c in diff.notes_created(
                db, since=excluded_since, as_of=excluded_as_of, clock="world"
            )
        }
        assert "tz-test-signal" not in excluded, (
            f"session TimeZone={session_tz!r}: a source dated {source_date} leaked into a "
            f"world-clock window ending the day before it ([{excluded_since}, {excluded_as_of}])"
        )
