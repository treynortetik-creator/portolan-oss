"""Tests for portolan/lint/** (spec sec 6.1, build-contracts.md).

Each of the eleven checks gets a positive case (a finding is produced) and
a negative case (a clean graph produces none), using raw SQL against the
`db` fixture from `tests/conftest.py` — the notes/claims/lint modules that
would normally build these rows (wave B1/B2) may not exist yet, per the
build instructions. Fictional names are drawn loosely from
`corpus/truth-table.yaml` (Ashgrove Health, Halvard Medical, etc.).

`portolan/store.py` (wave A2) had not landed when this was written, so
tests use a tiny in-memory `FakeStore` against the `Store` protocol
(`portolan/lint/_store.py`), per the build instructions.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import psycopg

from portolan.lint import render_markdown, run_lint, write_changelog_row
from portolan.lint.models import CHECK_NAMES


class FakeStore:
    """Tiny stand-in for `portolan.store.Store`: `exists()` looks up a
    fixed set of keys given at construction time; the write/read/url
    methods are unused by the lint but implemented so the object is a
    structurally complete Store."""

    def __init__(self, existing_keys: set[str] | None = None) -> None:
        self._existing = set(existing_keys or ())

    def put(self, key: str, data: bytes, content_type: str) -> None:  # noqa: ARG002
        self._existing.add(key)

    def get(self, key: str) -> bytes:
        raise NotImplementedError("FakeStore.get is not exercised by the lint")

    def exists(self, key: str) -> bool:
        return key in self._existing

    def url(self, key: str) -> str:
        return f"fake://{key}"


# --------------------------------------------------------------------
# Raw-SQL row builders
# --------------------------------------------------------------------


def insert_source(
    cur: psycopg.Cursor,
    slug: str,
    *,
    source_type: str = "call",
    s3_key: str | None = "s3://bucket/original.mp4",
    extraction_status: str = "done",
    source_date: str = "2026-08-14",
) -> str:
    cur.execute(
        """
        INSERT INTO sources (slug, source_type, s3_key, source_date, extraction_status)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (slug, source_type, s3_key, source_date, extraction_status),
    )
    return slug


def insert_note(
    cur: psycopg.Cursor,
    slug: str,
    *,
    type: str = "account",  # noqa: A002 - matches the DB column name
    title: str = "Ashgrove Health",
    description: str | None = "A hospital system account in the Midwest.",
    status: str = "living",
    crm_id: str | None = "A-1010",
    parent_slug: str | None = None,
    superseded_by: str | None = None,
    stale_after: datetime | None = None,
    body_md: str = "",
    updated: datetime | None = None,
) -> str:
    columns = [
        "slug",
        "type",
        "title",
        "description",
        "status",
        "crm_id",
        "parent_slug",
        "superseded_by",
        "stale_after",
        "body_md",
    ]
    values: list[object] = [
        slug,
        type,
        title,
        description,
        status,
        crm_id,
        parent_slug,
        superseded_by,
        stale_after,
        body_md,
    ]
    if updated is not None:
        columns.append("updated")
        values.append(updated)
    placeholders = ", ".join(["%s"] * len(values))
    cur.execute(
        f"INSERT INTO notes ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    return slug


def insert_link(cur: psycopg.Cursor, from_slug: str, to_slug: str, kind: str) -> None:
    cur.execute(
        "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, %s)",
        (from_slug, to_slug, kind),
    )


def insert_claim(
    cur: psycopg.Cursor,
    note_slug: str,
    *,
    fact_type: str = "complaint",
    claim_text: str = "Ashgrove reported a late delivery.",
    source_slug: str = "ashgrove-qbr-2026-08-14",
    locator: str = "00:01:00",
    human_edit_id: int | None = None,
    recorded: datetime | None = None,
    archived: bool = False,
) -> int:
    columns = ["note_slug", "fact_type", "claim_text", "source_slug", "locator", "archived"]
    values: list[object] = [note_slug, fact_type, claim_text, source_slug, locator, archived]
    if human_edit_id is not None:
        columns.append("human_edit_id")
        values.append(human_edit_id)
    if recorded is not None:
        columns.append("recorded")
        values.append(recorded)
    placeholders = ", ".join(["%s"] * len(values))
    cur.execute(
        f"INSERT INTO claims ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id",
        values,
    )
    return cur.fetchone()[0]


def insert_edit_record(
    cur: psycopg.Cursor,
    note_slug: str,
    *,
    editor: str = "maintainer",
    before_text: str = "old",
    after_text: str = "new",
    reason: str = "correction from a customer call",
) -> int:
    cur.execute(
        """
        INSERT INTO edit_records (note_slug, editor, before_text, after_text, reason)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """,
        (note_slug, editor, before_text, after_text, reason),
    )
    return cur.fetchone()[0]


def insert_review_queue(
    cur: psycopg.Cursor,
    *,
    kind: str = "ambiguous_entity",
    status: str = "open",
    source_slug: str | None = None,
    created: datetime | None = None,
) -> int:
    columns = ["kind", "status"]
    values: list[object] = [kind, status]
    if source_slug is not None:
        columns.append("source_slug")
        values.append(source_slug)
    if created is not None:
        columns.append("created")
        values.append(created)
    placeholders = ", ".join(["%s"] * len(values))
    cur.execute(
        f"INSERT INTO review_queue ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id",
        values,
    )
    return cur.fetchone()[0]


def findings_for(report, check: str) -> list:
    return report.for_check(check)


# --------------------------------------------------------------------
# 1. broken_link
# --------------------------------------------------------------------


def test_broken_link_positive(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_link(cur, "ashgrove", "no-such-note", "body")
        insert_link(cur, "ashgrove", "no-such-source", "provenance")

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "broken_link")
    assert len(rows) == 2
    assert all(r.severity == "error" for r in rows)
    assert {r.note_slug for r in rows} == {"ashgrove"}


def test_broken_link_negative(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(cur, "ashgrove-medical", type="account", crm_id="A-1011")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_link(cur, "ashgrove", "ashgrove-medical", "body")
        insert_link(cur, "ashgrove", "ashgrove-qbr-2026-08-14", "provenance")

    report = run_lint(db, FakeStore())
    assert findings_for(report, "broken_link") == []


# --------------------------------------------------------------------
# 2. orphan
# --------------------------------------------------------------------


def test_orphan_positive(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "orphan")
    assert [r.note_slug for r in rows] == ["ashgrove"]
    assert rows[0].severity == "error"


def test_orphan_negative(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(cur, "ashgrove-medical", crm_id="A-1011")
        insert_link(cur, "ashgrove-medical", "ashgrove", "parent")
        insert_link(cur, "ashgrove", "ashgrove-medical", "body")
        # source/report/context notes are exempt even with zero inbound
        # links (context added 2026-09-17, owner decision "Company
        # context page": the company-context note is a plain entry
        # point, same as source/report).
        insert_note(cur, "ashgrove-qbr-note", type="source", crm_id=None, description="A call.")
        insert_note(cur, "weekly-report-1", type="report", crm_id=None, description="A report.")
        insert_note(cur, "company-context", type="context", crm_id=None, description="Who we are.")

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "orphan")
    assert [r.note_slug for r in rows] == []


# --------------------------------------------------------------------
# 3. thin_links
# --------------------------------------------------------------------


def test_thin_links_positive(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(cur, "ashgrove-medical", crm_id="A-1011")
        insert_link(cur, "ashgrove", "ashgrove-medical", "body")

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "thin_links")
    assert "ashgrove" in {r.note_slug for r in rows}
    assert all(r.severity == "warning" for r in rows)


def test_thin_links_negative(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(cur, "ashgrove-medical", crm_id="A-1011")
        insert_note(cur, "ashgrove-riverside", crm_id="A-1012")
        insert_link(cur, "ashgrove", "ashgrove-medical", "body")
        insert_link(cur, "ashgrove", "ashgrove-riverside", "body")

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "thin_links")
    assert "ashgrove" not in {r.note_slug for r in rows}


# --------------------------------------------------------------------
# 4. stale
# --------------------------------------------------------------------


def test_stale_positive(db):
    today = date(2026, 9, 12)
    with db.cursor() as cur:
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            stale_after=datetime(2026, 8, 1, tzinfo=UTC),
        )
        insert_note(
            cur,
            "larkspur",
            crm_id="A-1002",
            stale_after=None,
            updated=datetime(2026, 5, 1, tzinfo=UTC),
        )

    report = run_lint(db, FakeStore(), today=today)
    rows = findings_for(report, "stale")
    by_slug = {r.note_slug: r for r in rows}
    assert by_slug["ashgrove"].severity == "error"
    assert by_slug["larkspur"].severity == "warning"


def test_stale_negative(db):
    today = date(2026, 9, 12)
    with db.cursor() as cur:
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            stale_after=datetime(2027, 1, 1, tzinfo=UTC),
            updated=datetime(2026, 9, 1, tzinfo=UTC),
        )

    report = run_lint(db, FakeStore(), today=today)
    assert findings_for(report, "stale") == []


# --------------------------------------------------------------------
# 5. bad_header
# --------------------------------------------------------------------


def test_bad_header_positive(db):
    with db.cursor() as cur:
        insert_note(cur, "no-description", crm_id="A-1010", description=None)
        insert_note(cur, "blank-crm-id", crm_id="", description="Has a description.")
        insert_note(
            cur,
            "maria-chen",
            type="contact",
            crm_id="C-2001",
            description="Director of Procurement.",
        )
        insert_note(
            cur,
            "bad-parent",
            crm_id="A-1099",
            description="An account whose parent is not an account.",
            parent_slug="maria-chen",
        )
        insert_note(
            cur,
            "old-account",
            crm_id="A-1000",
            description="Superseded with no pointer.",
            status="superseded",
            superseded_by=None,
        )

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "bad_header")
    slugs = {r.note_slug for r in rows}
    assert slugs == {"no-description", "blank-crm-id", "bad-parent", "old-account"}
    assert all(r.severity == "error" for r in rows)


def test_bad_header_negative(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(cur, "ashgrove-medical", crm_id="A-1011", parent_slug="ashgrove")
        insert_note(
            cur,
            "ashgrove-superseded",
            crm_id="A-1013",
            status="superseded",
            superseded_by="ashgrove",
        )

    report = run_lint(db, FakeStore())
    assert findings_for(report, "bad_header") == []


# Company-context wave, 2026-09-17: the context note's body cap
# (`portolan.context.COMPANY_CONTEXT_BODY_CHAR_CAP`), the lint's own
# independent check of it (the edit route, portolan.web.routes.notes, is
# what actually stops an over-length save from happening).


def _context_body_md(synthesis: str) -> str:
    return (
        '---\ntitle: "Company context"\n---\n\n'
        f"## Synthesis\n\n{synthesis}\n\n"
        "<!-- CLAIMS:APPEND-ONLY -->\n\n## Claims\n\nNo claims recorded yet.\n"
    )


def test_bad_header_positive_context_note_over_the_cap(db):
    with db.cursor() as cur:
        insert_note(
            cur,
            "company-context",
            type="context",
            crm_id=None,
            description="Who we are.",
            body_md=_context_body_md("x" * 2501),
        )

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "bad_header")
    assert any(r.note_slug == "company-context" and "over the" in r.message for r in rows)
    assert all(r.severity == "error" for r in rows if r.note_slug == "company-context")


def test_bad_header_negative_context_note_under_the_cap(db):
    with db.cursor() as cur:
        insert_note(
            cur,
            "company-context",
            type="context",
            crm_id=None,
            description="Who we are.",
            body_md=_context_body_md("x" * 2500),
        )

    report = run_lint(db, FakeStore())
    rows = [r for r in findings_for(report, "bad_header") if r.note_slug == "company-context"]
    assert rows == []


# --------------------------------------------------------------------
# 6. relative_time
# --------------------------------------------------------------------


def test_relative_time_positive(db):
    with db.cursor() as cur:
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            body_md="Procurement raised this last week and it is currently unresolved.",
        )
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        claim_id = insert_claim(
            cur,
            "ashgrove",
            claim_text="The team is currently evaluating a competitor.",
            source_slug="ashgrove-qbr-2026-08-14",
        )

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "relative_time")
    assert any(r.note_slug == "ashgrove" and r.claim_id is None for r in rows)
    assert any(r.claim_id == claim_id for r in rows)
    assert all(r.severity == "warning" for r in rows)


def test_relative_time_negative(db):
    with db.cursor() as cur:
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            body_md=(
                "Procurement flagged deliveries as of 2026-08-14. "
                'The customer\'s own words: "we need this fixed today" '
                "were logged as a direct quote, not a claim by us. "
                "Also see `today` used only as literal syntax in code."
            ),
        )

    report = run_lint(db, FakeStore())
    assert findings_for(report, "relative_time") == []


# --------------------------------------------------------------------
# 7. unresolved_citation
# --------------------------------------------------------------------


def test_unresolved_citation_positive(db):
    with db.cursor() as cur:
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            body_md="Late deliveries were reported ^[missing-source @ 00:14:32].",
        )
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        empty_locator_claim = insert_claim(
            cur,
            "ashgrove",
            source_slug="ashgrove-qbr-2026-08-14",
            locator="",
        )

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "unresolved_citation")
    assert any("missing-source" in r.message for r in rows if r.note_slug == "ashgrove")
    assert any(r.claim_id == empty_locator_claim for r in rows)
    assert all(r.severity == "error" for r in rows)


def test_unresolved_citation_negative(db):
    with db.cursor() as cur:
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            body_md="Late deliveries were reported ^[ashgrove-qbr-2026-08-14 @ 00:14:32].",
        )
        insert_claim(cur, "ashgrove", source_slug="ashgrove-qbr-2026-08-14", locator="00:14:32")

    report = run_lint(db, FakeStore())
    assert findings_for(report, "unresolved_citation") == []


# --------------------------------------------------------------------
# 8. missing_original
# --------------------------------------------------------------------


def test_missing_original_positive(db):
    with db.cursor() as cur:
        insert_source(
            cur, "ashgrove-qbr-2026-08-14", s3_key="s3://bucket/qbr.mp4", extraction_status="done"
        )
        insert_source(cur, "ashgrove-ticket-118", s3_key=None, extraction_status="done")

    report = run_lint(db, FakeStore(existing_keys=set()))
    rows = findings_for(report, "missing_original")
    slugs = {r.source_slug for r in rows}
    assert slugs == {"ashgrove-qbr-2026-08-14", "ashgrove-ticket-118"}
    assert all(r.severity == "error" for r in rows)


def test_missing_original_negative(db):
    with db.cursor() as cur:
        insert_source(
            cur, "ashgrove-qbr-2026-08-14", s3_key="s3://bucket/qbr.mp4", extraction_status="done"
        )
        insert_source(cur, "ashgrove-ticket-119", s3_key=None, extraction_status="pending")

    report = run_lint(db, FakeStore(existing_keys={"s3://bucket/qbr.mp4"}))
    assert findings_for(report, "missing_original") == []


# --------------------------------------------------------------------
# 9. ambiguity
# --------------------------------------------------------------------


def test_ambiguity_positive(db):
    with db.cursor() as cur:
        insert_review_queue(cur, kind="unknown_entity", created=datetime(2026, 8, 1, tzinfo=UTC))
        insert_review_queue(cur, kind="ambiguous_entity", created=datetime(2026, 8, 5, tzinfo=UTC))
        insert_review_queue(cur, kind="ambiguous_entity", status="resolved")  # excluded: not open

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "ambiguity")
    assert all(r.severity == "warning" for r in rows)
    # one summary row plus one row per open ambiguous/unknown item (2 here)
    assert len(rows) == 3


def test_ambiguity_negative(db):
    with db.cursor() as cur:
        insert_review_queue(cur, kind="failed_consolidation", status="open")

    report = run_lint(db, FakeStore())
    assert findings_for(report, "ambiguity") == []


# --------------------------------------------------------------------
# 10. human_edit_contradiction
# --------------------------------------------------------------------


def test_human_edit_contradiction_positive(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_source(cur, "ashgrove-call-2026-09-01")
        insert_link(cur, "ashgrove", "ashgrove-qbr-2026-08-14", "provenance")
        insert_link(cur, "ashgrove", "ashgrove-call-2026-09-01", "provenance")

        edit_id = insert_edit_record(cur, "ashgrove")
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price is locked at last year's rate.",
            source_slug="ashgrove-qbr-2026-08-14",
            human_edit_id=edit_id,
            recorded=datetime(2026, 8, 14, tzinfo=UTC),
        )
        later_claim_id = insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price increases 4% this cycle.",
            source_slug="ashgrove-call-2026-09-01",
            recorded=datetime(2026, 9, 1, tzinfo=UTC),
        )

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "human_edit_contradiction")
    assert [r.claim_id for r in rows] == [later_claim_id]
    assert rows[0].severity == "warning"


def test_human_edit_contradiction_negative_same_text(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_source(cur, "ashgrove-call-2026-09-01")

        edit_id = insert_edit_record(cur, "ashgrove")
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price is locked at last year's rate.",
            source_slug="ashgrove-qbr-2026-08-14",
            human_edit_id=edit_id,
            recorded=datetime(2026, 8, 14, tzinfo=UTC),
        )
        # Same text as the edited claim -> not a contradiction.
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price is locked at last year's rate.",
            source_slug="ashgrove-call-2026-09-01",
            recorded=datetime(2026, 9, 1, tzinfo=UTC),
        )

    report = run_lint(db, FakeStore())
    assert findings_for(report, "human_edit_contradiction") == []


def test_human_edit_contradiction_negative_no_entity_overlap(db):
    """The v1 heuristic requires the two claims' sources to share a
    provenance-linked note; with no provenance links recorded at all, the
    heuristic finds no overlap and stays silent (documented limitation)."""
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_source(cur, "ashgrove-call-2026-09-01")

        edit_id = insert_edit_record(cur, "ashgrove")
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price is locked at last year's rate.",
            source_slug="ashgrove-qbr-2026-08-14",
            human_edit_id=edit_id,
            recorded=datetime(2026, 8, 14, tzinfo=UTC),
        )
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price increases 4% this cycle.",
            source_slug="ashgrove-call-2026-09-01",
            recorded=datetime(2026, 9, 1, tzinfo=UTC),
        )
        # No provenance links inserted at all.

    report = run_lint(db, FakeStore())
    assert findings_for(report, "human_edit_contradiction") == []


# --------------------------------------------------------------------
# 11. growth_guard
# --------------------------------------------------------------------


def test_growth_guard_positive(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        for i in range(41):
            insert_claim(
                cur,
                "ashgrove",
                fact_type="other",
                claim_text=f"Claim number {i}.",
                source_slug="ashgrove-qbr-2026-08-14",
                locator=f"00:00:{i:02d}",
            )

    report = run_lint(db, FakeStore())
    rows = findings_for(report, "growth_guard")
    assert [r.note_slug for r in rows] == ["ashgrove"]
    assert rows[0].severity == "warning"


def test_growth_guard_negative(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        for i in range(40):
            insert_claim(
                cur,
                "ashgrove",
                fact_type="other",
                claim_text=f"Claim number {i}.",
                source_slug="ashgrove-qbr-2026-08-14",
                locator=f"00:00:{i:02d}",
            )
        # An archived claim past the limit does not count toward it.
        insert_claim(
            cur,
            "ashgrove",
            fact_type="other",
            claim_text="Archived claim.",
            source_slug="ashgrove-qbr-2026-08-14",
            locator="00:01:00",
            archived=True,
        )

    report = run_lint(db, FakeStore())
    assert findings_for(report, "growth_guard") == []


# --------------------------------------------------------------------
# run_lint's default `today` (P2-9, bug hunt 2026-09-18)
# --------------------------------------------------------------------


def test_run_lint_default_today_comes_from_the_deployment_clock(db, monkeypatch):
    """`run_lint(today=None)` must resolve "today" through
    `portolan.util.clock.today()` (the deployment's own `PORTOLAN_
    TIMEZONE`), never the stdlib `date.today()` (the host process's own
    zone) -- otherwise a `stale` finding can flip a day early or late
    depending on where the scheduled lint job happens to run, for no
    reason visible in this module. Proven by monkeypatching `clock.today`
    itself to a deliberately-unreal date and checking the report picked
    it up: before the fix, `portolan.lint.report` called the stdlib
    `date.today()` directly and this monkeypatch would have had no
    effect on the result at all.
    """
    from portolan.util import clock

    sentinel = date(2099, 1, 2)
    monkeypatch.setattr(clock, "today", lambda: sentinel)

    report = run_lint(db, FakeStore())

    assert report.today == sentinel


# --------------------------------------------------------------------
# render_markdown and write_changelog_row
# --------------------------------------------------------------------


def test_render_markdown_has_dated_header_counts_and_findings(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")  # orphan

    report = run_lint(db, FakeStore(), today=date(2026, 9, 12))
    markdown = render_markdown(report)

    assert markdown.startswith("# Lint report — 2026-09-12")
    assert "| Check | Errors | Warnings |" in markdown
    for check in CHECK_NAMES:
        assert f"## {check}" in markdown
    assert "ashgrove" in markdown
    assert "no inbound body or parent links" in markdown


def test_render_markdown_clean_report_says_clean(db):
    report = run_lint(db, FakeStore(), today=date(2026, 9, 12))
    markdown = render_markdown(report)
    assert "0 error(s), 0 warning(s)." in markdown
    assert "Clean." in markdown


def test_write_changelog_row_inserts_a_row(db):
    write_changelog_row(
        db, summary="1 error(s), 0 warning(s)", details={"broken_link": {"errors": 1}}
    )

    with db.cursor() as cur:
        cur.execute("SELECT kind, summary, details FROM changelog ORDER BY id DESC LIMIT 1")
        kind, summary, details = cur.fetchone()

    assert kind == "lint"
    assert summary == "1 error(s), 0 warning(s)"
    assert details == {"broken_link": {"errors": 1}}


def test_has_errors_reflects_error_severity_findings(db):
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")  # orphan -> error

    report = run_lint(db, FakeStore())
    assert report.has_errors is True

    with db.cursor() as cur:
        cur.execute("DELETE FROM notes")

    clean_report = run_lint(db, FakeStore())
    assert clean_report.has_errors is False


# --------------------------------------------------------------------
# Read-path review 2026-11-20: findings a clean graph should not raise,
# findings nobody can clear, and citations that name no claim.
# --------------------------------------------------------------------


def test_thin_links_counts_links_into_a_note_as_well(db):
    """An entity note (a product, a competitor) is linked *to*: three signals
    naming it make it well connected, whatever its own body links out to."""
    with db.cursor() as cur:
        insert_note(cur, "tidewater-t2", type="product", crm_id=None, title="Tidewater T2")
        for n in range(3):
            insert_note(cur, f"signal-{n}", type="signal", crm_id=None, title=f"Signal {n}")
            insert_link(cur, f"signal-{n}", "tidewater-t2", "body")

    report = run_lint(db, FakeStore())
    assert "tidewater-t2" not in {r.note_slug for r in findings_for(report, "thin_links")}
    # each signal has one link in all, so it is still thin
    assert "signal-0" in {r.note_slug for r in findings_for(report, "thin_links")}


def test_stale_after_on_a_note_that_is_no_longer_living_is_not_an_error(db):
    """Nobody can clear it: the note was superseded or archived, not left to rot."""
    today = date(2026, 9, 12)
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(
            cur,
            "ashgrove-old",
            crm_id="A-1011",
            status="superseded",
            superseded_by="ashgrove",
            stale_after=datetime(2026, 8, 1, tzinfo=UTC),
        )
        insert_note(
            cur,
            "ashgrove-shelved",
            crm_id="A-1012",
            status="archived",
            stale_after=datetime(2026, 8, 1, tzinfo=UTC),
        )

    report = run_lint(db, FakeStore(), today=today)
    assert findings_for(report, "stale") == []


def test_relative_time_catches_the_common_phrases_the_first_list_missed(db):
    with db.cursor() as cur:
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_note(cur, "ashgrove", crm_id="A-1010")
        flagged = {
            insert_claim(cur, "ashgrove", claim_text=text, source_slug="ashgrove-qbr-2026-08-14")
            for text in (
                "The renewal slipped last quarter.",
                "They plan to decide next month.",
                "Three pumps failed two weeks ago.",
                "Budget is frozen this year.",
                "A pilot starts later this month.",
            )
        }
        clean = {
            insert_claim(cur, "ashgrove", claim_text=text, source_slug="ashgrove-qbr-2026-08-14")
            for text in (
                "The renewal slipped in 2026-Q2.",
                "The last quarter of 2026 is the renewal window.",
                "Deliveries were late for the week of 2026-07-27.",
            )
        }

    rows = findings_for(run_lint(db, FakeStore()), "relative_time")
    assert {r.claim_id for r in rows} == flagged
    assert not ({r.claim_id for r in rows} & clean)


def test_relative_time_leaves_a_speakers_quoted_words_in_a_claim_alone(db):
    with db.cursor() as cur:
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_claim(
            cur,
            "ashgrove",
            claim_text='The materials manager said on 2026-08-24 that "after last month" '
            "the account expects a credit.",
            source_slug="ashgrove-qbr-2026-08-14",
        )

    assert findings_for(run_lint(db, FakeStore()), "relative_time") == []


def test_relative_time_skips_reports_and_notes_that_are_no_longer_living(db):
    """A weekly report is dated by what it is; a superseded note is history."""
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_note(
            cur,
            "weekly-2026-w37",
            type="report",
            crm_id=None,
            body_md="Recently three accounts raised deliveries.",
        )
        insert_note(
            cur,
            "ashgrove-old",
            crm_id="A-1011",
            status="superseded",
            superseded_by="ashgrove",
            body_md="Currently in renewal talks.",
        )

    assert findings_for(run_lint(db, FakeStore()), "relative_time") == []


def test_a_citation_marker_that_no_claim_on_the_note_backs_is_a_warning(db):
    """The source exists, but nothing on this note was recorded from it at that
    place: the sentence carries a citation its own claims do not support."""
    with db.cursor() as cur:
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            body_md=(
                "Late deliveries were reported ^[ashgrove-qbr-2026-08-14 @ 00:14:32]. "
                "A credit was promised ^[ashgrove-qbr-2026-08-14 @ 00:41:07]."
            ),
        )
        insert_claim(cur, "ashgrove", source_slug="ashgrove-qbr-2026-08-14", locator="00:14:32")

    rows = findings_for(run_lint(db, FakeStore()), "unresolved_citation")
    assert [(r.note_slug, r.severity) for r in rows] == [("ashgrove", "warning")]
    assert "00:41:07" in rows[0].message


def test_a_citation_marker_backed_by_a_corroborating_citation_is_clean(db):
    with db.cursor() as cur:
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_source(cur, "ashgrove-call-2026-09-01")
        insert_note(
            cur,
            "ashgrove",
            crm_id="A-1010",
            body_md="Late deliveries were reported ^[ashgrove-call-2026-09-01 @ 00:03:10].",
        )
        claim_id = insert_claim(
            cur, "ashgrove", source_slug="ashgrove-qbr-2026-08-14", locator="00:14:32"
        )
        cur.execute(
            "INSERT INTO claim_citations (claim_id, source_slug, locator) VALUES (%s, %s, %s)",
            (claim_id, "ashgrove-call-2026-09-01", "00:03:10"),
        )

    assert findings_for(run_lint(db, FakeStore()), "unresolved_citation") == []


def test_a_later_claim_that_was_itself_superseded_or_archived_no_longer_warns(db):
    """Superseding or archiving the later claim is how a person settles the
    conflict; the warning has to clear when they do."""
    with db.cursor() as cur:
        insert_note(cur, "ashgrove", crm_id="A-1010")
        insert_source(cur, "ashgrove-qbr-2026-08-14")
        insert_source(cur, "ashgrove-call-2026-09-01")
        insert_link(cur, "ashgrove", "ashgrove-qbr-2026-08-14", "provenance")
        insert_link(cur, "ashgrove", "ashgrove-call-2026-09-01", "provenance")
        edit_id = insert_edit_record(cur, "ashgrove")
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price is locked at last year's rate.",
            source_slug="ashgrove-qbr-2026-08-14",
            human_edit_id=edit_id,
            recorded=datetime(2026, 8, 14, tzinfo=UTC),
        )
        insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price increases 4% this cycle.",
            source_slug="ashgrove-call-2026-09-01",
            recorded=datetime(2026, 9, 1, tzinfo=UTC),
            archived=True,
        )
        superseded = insert_claim(
            cur,
            "ashgrove",
            fact_type="pricing",
            claim_text="Renewal price increases 6% this cycle.",
            source_slug="ashgrove-call-2026-09-01",
            locator="00:02:00",
            recorded=datetime(2026, 9, 1, tzinfo=UTC),
        )
        cur.execute("UPDATE claims SET superseded_at = now() WHERE id = %s", (superseded,))

    assert findings_for(run_lint(db, FakeStore()), "human_edit_contradiction") == []
