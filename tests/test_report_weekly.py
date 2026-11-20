"""`portolan.report.weekly` tests (wave C1): the report note itself, plus
the sections that don't fit neatly under one of the other feed modules
(actions, coverage, contact changes, headline).
"""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from test_report_helpers import (  # noqa: F401
    FullGraph,
    _report_llm_responder,
    arr_by_note_slug,
    full_graph,
    lint_error_identity,
    report_llm,
)

from portolan.lint.report import run_lint
from portolan.llm.fake import FakeLLMClient
from portolan.metering import PostgresSink, metered
from portolan.notes import NoteRow, NoteStatus, NoteType, link_related, read_note, write_note
from portolan.report import weekly
from portolan.report.models import (
    AccountParagraph,
    ActionRow,
    ActionsFeed,
    CoverageSummary,
    DiffFeed,
    ExecutiveSummary,
    PendingResolution,
    QuietAccounts,
    SignalDetail,
    ThemeReport,
    ThemeSummary,
    WavesFeed,
)
from portolan.report.questions import STANDING_QUESTION_SYSTEM
from portolan.report.synthesis import ACCOUNT_SYNTHESIS_SYSTEM
from portolan.report.themes import THEME_NAMING_SYSTEM
from portolan.report.util import week_bounds
from portolan.report.weekly import HEADLINE_SYSTEM
from portolan.store import LocalStore
from portolan.util import clock as clock_module

WINDOW_START = date(2026, 3, 2)
AS_OF = date(2026, 9, 4)


def test_contact_changes_feed_covers_f28_f29_f30(full_graph: FullGraph) -> None:
    changes = weekly.contact_changes_feed(full_graph.conn, since=WINDOW_START, as_of=AS_OF)
    texts = " ".join(c.claim_text for c in changes)
    assert "Tom Ridley left" in texts  # F28
    assert "Renata Solis" in texts  # F29
    assert "Director of Clinical Operations" in texts  # F30
    assert len(changes) >= 3


def test_open_actions_include_f09_with_owner_and_due_date(full_graph: FullGraph) -> None:
    # Theme identity/resolution wave (2026-09-13, task item 3): `_actions_
    # feed` now gates open/overdue actions on their own creation date, per
    # the active clock -- a commitment made after `as_of` must not print
    # (weekly-report-eval-2026-09-13.md defect #2). `full_graph` ingests
    # this whole corpus in ONE real wall-clock pipeline run, so under the
    # default "record" clock every action's `notes.created` is this test
    # session's own run time, not August 2026 -- `clock="world"` reads
    # each action's own claim's world date instead, which is exactly what
    # a real `--clock world` backfill (this project's own step 5) uses.
    actions = weekly._actions_feed(full_graph.conn, since=WINDOW_START, as_of=AS_OF, clock="world")
    f09 = next((a for a in actions.open if "root-cause summary" in a.what), None)
    assert f09 is not None, [a.what for a in actions.open]
    assert f09.owner == "Sam Okafor"
    assert f09.due == date(2026, 8, 28)

    f32 = next((a for a in actions.open if "quarterly business review" in a.what), None)
    assert f32 is not None
    assert f32.owner == "Dana Whitlock"

    f33 = next((a for a in actions.open if "pilot two Tidewater T3" in a.what), None)
    assert f33 is not None
    assert f33.owner is None


def test_actions_feed_excludes_a_commitment_made_after_as_of_under_world_clock(
    full_graph: FullGraph,
) -> None:
    """Task item 3's own worked example: "a commitment made in September
    must not appear in the 08-16 report." F57 (corpus/truth-table.yaml:
    Priya Natarajan's Tidewater Link/Epic test-plan promise to Wrenfield
    Health, committed 2026-08-24, due 2026-09-10) must be absent from a
    world-clock report window ending 2026-08-16 -- before it was ever
    made -- and present once the window reaches its own commitment date.
    Regression guard for weekly-report-eval-2026-09-13.md defect #2 ("all
    5 open actions print identically in every report ... including
    commitments the story hasn't made yet as of that date")."""
    conn = full_graph.conn
    what_substring = "Priya Natarajan promised Wrenfield"

    before_it_was_made = weekly._actions_feed(
        conn, since=date(2026, 8, 10), as_of=date(2026, 8, 16), clock="world"
    )
    assert not any(what_substring in a.what for a in before_it_was_made.open), [
        a.what for a in before_it_was_made.open
    ]

    after_it_was_made = weekly._actions_feed(
        conn, since=date(2026, 8, 18), as_of=date(2026, 8, 24), clock="world"
    )
    assert any(what_substring in a.what for a in after_it_was_made.open), [
        a.what for a in after_it_was_made.open
    ]


def test_actions_feed_tolerates_an_action_note_with_no_claims(full_graph: FullGraph) -> None:
    """Nit 18 fix: `recorded` comes from a `LEFT JOIN LATERAL` and is
    `NULL` for an action note with no claims at all -- `merge_action`
    always writes exactly one today, so this is defensive only, but it
    used to be one `AttributeError` (`clock_module.world_date(None)`)
    away from the same "the weekly job died" failure blocker 2 fixed for
    themes. Falls back to `notes.created` under `clock="world"` too."""
    conn = full_graph.conn
    now = datetime.now(UTC)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, created, updated) "
            "VALUES ('action-nit18-no-claims', 'action', 'Nit 18 fixture', 'fixture', %s, %s) "
            "ON CONFLICT (slug) DO NOTHING",
            (now, now),
        )
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()
        cur.execute(
            "INSERT INTO actions (note_slug, what, status, source_slug) "
            "VALUES ('action-nit18-no-claims', 'a claimless action fixture', 'open', %s)",
            (any_source,),
        )
    conn.commit()

    # Verify-before-claiming fix (found running this wave's own full-
    # suite pass, unrelated to any review finding): a real, merge-
    # pipeline-created action note always gets an inbound link from its
    # owning account (`_ensure_signal_note`'s own backlink mechanism);
    # this fixture bypasses that pipeline (a raw INSERT, not `merge_
    # action`), so without an explicit backlink it is a permanent orphan
    # `test_report_links_resolve_and_introduces_no_new_lint_errors`
    # (later in this same session-scoped graph) would trip on, forever,
    # regardless of what either test is actually about.
    link_related(
        conn,
        "ashgrove-health",
        "action-nit18-no-claims",
        changed_by="test",
        reason="nit 18 fixture backlink, so this fixture is never a permanent orphan",
    )

    actions = weekly._actions_feed(
        conn, since=date(2026, 1, 1), as_of=clock_module.today(), clock="world"
    )
    ours = next((a for a in actions.open if a.what == "a claimless action fixture"), None)
    assert ours is not None, [a.what for a in actions.open]


def test_actions_feed_gates_on_notes_created_under_record_clock(full_graph: FullGraph) -> None:
    """The other half of task item 3: under the default "record" clock,
    the gate is the action note's own wall-clock `created`, not any
    claim's world date. A `full_graph`-built action's `notes.created` is
    this test session's own run time; an `as_of` far in the future (well
    past "now") must still include it, and one far in the past must
    exclude it -- proving the record-clock branch reads `notes.created`
    at all, independent of the corpus's fictional 2026 dates."""
    conn = full_graph.conn
    far_future = date(2026, 1, 1) + timedelta(days=365 * 5)
    far_past = date(2000, 1, 1)

    future_result = weekly._actions_feed(
        conn, since=far_future - timedelta(days=6), as_of=far_future, clock="record"
    )
    assert future_result.open, "expected at least one action to already exist by now"

    past_result = weekly._actions_feed(
        conn, since=far_past - timedelta(days=6), as_of=far_past, clock="record"
    )
    assert past_result.open == []


def test_actions_feed_counts_a_dropped_action_as_closed_this_week_via_edit_records(
    full_graph: FullGraph,
) -> None:
    """Owners, Phase 1 (01-spec/actions-and-owners-2026-09-19.md build
    item 7): "closed this week" must count `dropped`, not only `done` --
    and, when the new `/actions` Done/Dropped controls wrote an
    `edit_records` row for it, that row's own `edited_at` is preferred
    over the `notes.updated` proxy."""
    conn = full_graph.conn
    now = datetime.now(UTC)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, created, updated) "
            "VALUES ('action-owners-dropped-fixture', 'action', 'Dropped fixture action', "
            "'fixture', %s, %s) ON CONFLICT (slug) DO NOTHING",
            (now, now),
        )
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()
        cur.execute(
            "INSERT INTO actions (note_slug, what, status, source_slug) "
            "VALUES ('action-owners-dropped-fixture', 'a dropped action fixture', 'dropped', %s) "
            "RETURNING id",
            (any_source,),
        )
        (action_id,) = cur.fetchone()
        cur.execute(
            "INSERT INTO edit_records (note_slug, editor, before_text, after_text, reason) "
            "VALUES ('action-owners-dropped-fixture', 'Test Editor', 'open', 'dropped', "
            "'marked dropped')"
        )
    conn.commit()
    link_related(
        conn,
        "ashgrove-health",
        "action-owners-dropped-fixture",
        changed_by="test",
        reason="owners-dropped fixture backlink, so this fixture is never a permanent orphan",
    )

    today = clock_module.today()
    actions = weekly._actions_feed(
        conn, since=today - timedelta(days=6), as_of=today, clock="record"
    )

    dropped = next((a for a in actions.resolved_this_week if a.id == action_id), None)
    assert dropped is not None, [a.what for a in actions.resolved_this_week]
    assert dropped.status == "dropped"


def test_build_weekly_writes_a_report_note_and_row(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )

    assert result.week == "2026-W36"
    assert result.note_slug == "report-2026-W36"
    assert result.headline

    note, _claims = read_note(full_graph.conn, result.note_slug)
    assert note.type.value == "report"
    assert "theme-late-deliveries-midwest" in note.synthesis

    with full_graph.conn.cursor() as cur:
        cur.execute("SELECT note_slug FROM reports WHERE week = %s", (result.week,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == result.note_slug


def test_build_weekly_as_of_a_monday_labels_the_prior_weeks_majority(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Bughunt-2026-09-18 finding 7: `--as-of <a Monday>` (the natural
    "run last week's report" invocation, no `--week`) windows the seven
    days ENDING that Monday -- six of which fall in the ISO week BEFORE
    the Monday's own. Labelling from `as_of` alone used to write
    `report-2026-W39` for a window that was six-sevenths `2026-W38`,
    overwriting whatever W39's own future report would later want that
    slug for. `2026-09-21` is a Monday; its window is `2026-09-15` (a
    Tuesday, in W38) through `2026-09-21` (in W39) -- majority W38."""
    as_of = date(2026, 9, 21)
    assert as_of.isoweekday() == 1  # Monday, the shape the finding names

    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
    )

    assert result.since == date(2026, 9, 15)
    assert result.as_of == as_of
    assert result.week == "2026-W38"
    assert result.note_slug == "report-2026-W38"


def test_build_weekly_writes_judge_health_onto_the_reports_row(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Should-fix 7 (review-theme-wave-5c.md finding 7): the resolution
    judge's own run-health counters must be readable AFTER THE FACT, on
    the `reports` row itself -- not only in that one week's own rendered
    report body, which is all `ThemeReport.judge_calls`/`judge_failures`/
    `judge_cap_hit` fed before this fix. `judge_health.outage` is derived
    (every call failed, not merely some) rather than trusted from the
    caller, so it is checked against the SAME counters `result.themes`
    itself carries, not a hardcoded expectation of this fixture's own
    judge volume."""
    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )

    with full_graph.conn.cursor() as cur:
        cur.execute("SELECT judge_health FROM reports WHERE week = %s", (result.week,))
        (judge_health,) = cur.fetchone()
    assert judge_health is not None
    assert judge_health["judge_calls"] == result.themes.judge_calls
    assert judge_health["judge_failures"] == result.themes.judge_failures
    assert judge_health["judge_cap_hit"] == result.themes.judge_cap_hit
    expected_outage = (
        result.themes.judge_failures == result.themes.judge_calls and result.themes.judge_calls > 0
    )
    assert judge_health["outage"] == expected_outage


def test_build_weekly_world_clock_end_to_end_over_the_august_backfill(
    full_graph: FullGraph, report_llm: FakeLLMClient, arr_by_note_slug: dict[str, float]
) -> None:
    """Test gap 3 (review finding 7): the documented scenario --
    `--as-of 2026-08-23 --clock world` over a same-day backfill of the
    August corpus -- run end to end. Before this fix,
    `test_report_weekly_cli.py` only asserted argparse accepts the flag;
    nothing ran the actual scenario and checked which notes, standing-
    question citations, and coverage/actions figures come out.
    """
    conn = full_graph.conn
    as_of = date(2026, 8, 23)
    since = as_of - timedelta(days=6)  # build_weekly's own default 7-day window

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM standing_questions WHERE question = %s",
            ("World-clock test: which accounts had activity?",),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO standing_questions (reader, question) VALUES (%s, %s) RETURNING id",
                ("ops", "World-clock test: which accounts had activity?"),
            )
            (question_id,) = cur.fetchone()
        else:
            (question_id,) = row
    conn.commit()

    # Independent ground truth, straight off `claims.recorded` -- not
    # re-deriving from anything this fix itself computes.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT note_slug FROM claims WHERE recorded::date BETWEEN %s AND %s",
            (since, as_of),
        )
        expected_world_touched = {r[0] for r in cur.fetchall()}
    assert expected_world_touched, (
        "expected at least one note with world-clock activity in "
        f"[{since}, {as_of}] -- adjust the window if the fixture corpus changes"
    )

    result = weekly.build_weekly(
        conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=as_of,
        arr_by_note_slug=arr_by_note_slug,
        clock="world",
    )

    assert result.since == since
    assert result.as_of == as_of

    # 1. notes_created/notes_updated reflect the corpus's own dates
    #    (finding 1's exact scenario), not this test session's real
    #    wall-clock run time -- some real activity in this window.
    assert result.diff.notes_created or result.diff.notes_updated

    # 2. Standing questions were answered from notes with genuine
    #    world-clock activity in the window, not the whole corpus
    #    (finding 7's questions half: `since=` would have matched
    #    everything, since every note's wall-clock `updated` is "today"
    #    for a same-day backfill).
    ours = next(a for a in result.standing_questions if a.question_id == question_id)
    assert set(ours.citations).issubset(expected_world_touched)

    # 3. Coverage and "resolved this window" are plainly marked
    #    record-time-only under world clock, rather than silently
    #    printing zeros (finding 7's coverage/actions half) -- once for
    #    Actions, once for Coverage.
    assert result.body_md.count("record time only") >= 2


def test_headline_sentence_with_an_unsupplied_numeral_is_dropped(
    full_graph: FullGraph, arr_by_note_slug: dict[str, float]
) -> None:
    """The numbers rule (polish wave 2026-09-12): every other surface uses
    `report_llm`'s normal canned answers, but the headline call gets a
    fabricated count with nothing in the payload's `figures` table to back
    it up -- that sentence must not reach `result.headline`, and a
    `review_queue` row records why."""

    def _fabricating(*, system, user, json_schema, model, surface):
        if surface == "report" and system == HEADLINE_SYSTEM:
            assert '"figures"' in user
            return {"sentences": ["Exactly 733 accounts raised a complaint this window."]}
        return _report_llm_responder(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )

    result = weekly.build_weekly(
        full_graph.conn,
        FakeLLMClient(_fabricating),
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )
    assert not any("733" in sentence for sentence in result.headline)

    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE kind = 'other' AND payload->>'reason' = 'numeral_not_supplied' "
            "AND payload->>'surface' = 'headline' "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
    assert row is not None
    assert "733" in row[0]


def test_headline_sentence_with_a_supplied_figure_is_accepted(
    full_graph: FullGraph, arr_by_note_slug: dict[str, float]
) -> None:
    """A headline sentence that states a figure genuinely present in the
    payload's `figures` table passes the numbers check -- the guard
    rejects fabrication, not digits in general."""

    def _responder(*, system, user, json_schema, model, surface):
        if surface == "report" and system == HEADLINE_SYSTEM:
            import json as json_module

            figures = json_module.loads(user)["figures"]
            notes_created = figures["notes_created"]
            return {"sentences": [f"{notes_created} notes were created this window."]}
        return _report_llm_responder(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )

    result = weekly.build_weekly(
        full_graph.conn,
        FakeLLMClient(_responder),
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )
    assert any("notes were created this window" in sentence for sentence in result.headline)


def test_build_weekly_renders_a_wave_with_its_rollup_clause_and_a_citation(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Should-fix 7 (`01-spec/reviews/review-theme-fix-wave-3.md` finding
    7, test gap 6): "nothing renders a wave in any test." Plants a real
    wave -- three TOP-LEVEL accounts sharing a fact type and product
    family in this report's own window, one of them Wrenfield Health
    (whose own registry parent/child relationship, `corpus/out/crm/
    accounts.csv`, rolls its two children up under it) -- and checks the
    rendered `### Waves` section names the right top-level account count,
    the rollup clause, and at least one `^[source @ locator]` citation
    line (should-fix 3 of the PRIOR wave, "the Waves section cites
    nothing," never pinned by a test either). Also pins nit 14's fix: the
    printed account list matches the printed COUNT (top-level slugs, not
    every rolled-up leaf)."""
    conn = full_graph.conn
    now = datetime.now(UTC)
    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()

    product_slug = "waverender-render-widget"
    product_note = NoteRow(
        slug=product_slug,
        type=NoteType.PRODUCT,
        title="Waverender Render Widget",
        description="Test fixture product for the wave-rendering test.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(conn, product_note, [], changed_by="test", reason="wave-render fixture product")
    # A brand-new product note has no real corpus history pointing at it
    # -- an explicit backlink, same reasoning as the nit16/nit18 fix
    # above, so this fixture is never a permanent orphan.
    link_related(
        conn,
        "wrenfield-health",
        product_slug,
        changed_by="test",
        reason="wave-render fixture backlink",
    )

    accounts = [
        "wrenfield-health",  # parent -- rolls its own two children up
        "wrenfield-medical-center",
        "wrenfield-east-hospital",
        "ashgrove-medical-center",  # top-level stranger #2
        "larkspur-memorial-hospital",  # top-level stranger #3
    ]
    for i, account_slug in enumerate(accounts):
        signal_slug = f"signal-waverender-{i}"
        note = NoteRow(
            slug=signal_slug,
            type=NoteType.SIGNAL,
            title=signal_slug,
            description="Wave-render test fixture signal note.",
            status=NoteStatus.LIVING,
            created=now,
            updated=now,
        )
        write_note(conn, note, [], changed_by="test", reason="wave-render fixture signal")
        link_related(
            conn,
            account_slug,
            signal_slug,
            changed_by="test",
            reason="wave-render fixture backlink, so this fixture is never a permanent orphan",
        )
        quote = "we are watching the render queue closely this week" if i == 0 else None
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO claims "
                "(note_slug, fact_type, claim_text, source_slug, locator, recorded, quote) "
                "VALUES (%s, 'risk', %s, %s, %s, %s, %s)",
                (
                    signal_slug,
                    f"[[{account_slug}]] flagged a [[{product_slug}]] render-queue risk this week.",
                    any_source,
                    f"wave-render#{i}",
                    date(2026, 9, 1),
                    quote,
                ),
            )
        conn.commit()

    result = weekly.build_weekly(
        conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )

    render_waves = [w for w in result.waves.waves if w.product_family == "waverender"]
    assert len(render_waves) == 1, result.waves.waves
    wave = render_waves[0]
    assert set(wave.account_slugs) == set(accounts)
    assert wave.rolled_up == {
        "wrenfield-medical-center": "wrenfield-health",
        "wrenfield-east-hospital": "wrenfield-health",
    }
    assert len(wave.quotes) >= 1
    assert wave.quotes[0].text == "we are watching the render queue closely this week"

    body = result.body_md
    assert "### Waves" in body
    waves_section = body.split("### Waves", 1)[1].split("### Standing questions", 1)[0]
    # Nit 14: the printed count (3 top-level accounts) matches the
    # printed account list (top-level slugs only, not all 5 leaves).
    assert "3 accounts" in waves_section
    assert "[[wrenfield-health]]" in waves_section
    assert "[[ashgrove-medical-center]]" in waves_section
    assert "[[larkspur-memorial-hospital]]" in waves_section
    assert "[[wrenfield-medical-center]]" not in waves_section.split("rolled up")[0]
    assert "wrenfield-medical-center under wrenfield-health" in waves_section
    assert "wrenfield-east-hospital under wrenfield-health" in waves_section
    # Weekly-report wave 2: the fixed "up to N representative quotes"
    # preview is superseded by the expandable "N claims" signals fold
    # (`weekly._signals_fold_block`) -- the quote still appears (as an
    # indented sub-bullet under its own claim line), and the claim line
    # itself still carries the citation, just not glued to the quote text
    # any more.
    assert '"we are watching the render queue closely this week"' in waves_section
    assert "@ wave-render#0]" in waves_section
    assert "<!--signals:" in waves_section
    assert "<!--/signals-->" in waves_section


def _plant_nit16_outlier_pair_and_grown_account_claim(conn) -> None:
    """Nit 16 fix: plants a fresh, unclustered pair (same fact type, two
    accounts, near-identical wording under `FakeEmbedder`) that will form
    a BRAND NEW theme in this run's own clustering pass, plus a fresh
    claim on a grown account -- so `theme_naming` and `account_synthesis`
    fire deterministically, rather than depending on whatever earlier
    tests in this session's shared graph happened to leave unclustered or
    ungrown by the time this test runs."""
    now = datetime.now(UTC)
    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM sources LIMIT 1")
        (any_source,) = cur.fetchone()

    # Verify-before-claiming fix (found running this wave's own full-
    # suite pass, unrelated to any review finding): a real signal note
    # always gets an inbound `## Related` link from its owning account
    # (`_ensure_signal_note`'s own backlink mechanism, part of the normal
    # merge pipeline) -- this fixture writes the note directly, bypassing
    # that pipeline, so without an explicit backlink each one is a
    # permanent orphan regardless of whether it later joins a theme
    # (`_attach` links theme<->account and theme<->product, never
    # theme<->the individual member note) -- `test_report_links_resolve_
    # and_introduces_no_new_lint_errors` (later in this same session-
    # scoped graph) would trip on that, forever.
    account_by_slug = {
        "signal-nit16-outlier-a": "kestrel-medical-group",
        "signal-nit16-outlier-b": "sable-valley-health",
        "signal-nit16-grown": "vireo-surgical-group",
    }
    for slug, account_slug in account_by_slug.items():
        note = NoteRow(
            slug=slug,
            type=NoteType.SIGNAL,
            title=slug,
            description="Nit 16 test fixture signal note.",
            status=NoteStatus.LIVING,
            created=now,
            updated=now,
        )
        write_note(conn, note, [], changed_by="test", reason="nit 16 fixture")
        link_related(
            conn,
            account_slug,
            slug,
            changed_by="test",
            reason="nit 16 fixture backlink, so this fixture is never a permanent orphan",
        )

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded) "
            "VALUES (%s, 'risk', %s, %s, %s, %s)",
            (
                "signal-nit16-outlier-a",
                "[[kestrel-medical-group]] flagged an identityfixturenit16 staffing rotation risk.",
                any_source,
                "nit16#a",
                date(2026, 9, 1),
            ),
        )
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded) "
            "VALUES (%s, 'risk', %s, %s, %s, %s)",
            (
                "signal-nit16-outlier-b",
                "[[sable-valley-health]] also flagged an identityfixturenit16 staffing "
                "rotation risk.",
                any_source,
                "nit16#b",
                date(2026, 9, 2),
            ),
        )
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded) "
            "VALUES (%s, 'complaint', %s, %s, %s, %s)",
            (
                "signal-nit16-grown",
                "[[vireo-surgical-group]] reported a fresh identityfixturenit16grown issue.",
                any_source,
                "nit16#c",
                date(2026, 9, 3),
            ),
        )
    conn.commit()


def test_prose_calls_use_the_strong_model_not_the_default(
    full_graph: FullGraph, arr_by_note_slug: dict[str, float]
) -> None:
    """Task item 4: naming, headline, synthesis, and standing-question
    prose must use STRONG_MODEL (here, an explicit `prose_model` passed
    through `build_weekly` -- the CLI itself computes this from `settings.
    strong_model or settings.extraction_model`, `weekly.main`'s own
    comment) -- never the client's own default model. Verified via the
    fake client's own call log: a wrapper around `_report_llm_responder`
    that records the `model` kwarg every call actually received, keyed by
    which system prompt (surface) made it, so this test fails loudly if a
    prose call site silently drops back to the client's default.

    Nit 16 fix: the docstring always claimed to cover "naming, headline,
    synthesis, and standing-question prose", but two of the four used to
    be asserted only if a session-scoped fixture happened to have left
    work for them. `_plant_nit16_outlier_pair_and_grown_account_claim`
    guarantees a theme-naming call and an account-synthesis call both
    fire this run, so all four are checked unconditionally."""
    _plant_nit16_outlier_pair_and_grown_account_claim(full_graph.conn)

    strong_model = "strong-test-model"
    seen_models: dict[str, str | None] = {}

    def _responder(*, system, user, json_schema, model, surface):
        if system == THEME_NAMING_SYSTEM:
            seen_models["theme_naming"] = model
        elif system == HEADLINE_SYSTEM:
            seen_models["headline"] = model
        elif system == STANDING_QUESTION_SYSTEM:
            seen_models["standing_question"] = model
        elif system == ACCOUNT_SYNTHESIS_SYSTEM:
            seen_models["account_synthesis"] = model
        return _report_llm_responder(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )

    weekly.build_weekly(
        full_graph.conn,
        FakeLLMClient(_responder),
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
        prose_model=strong_model,
    )

    # Headline and standing-question calls fire on every `build_weekly`
    # call (there is always a headline, and `full_graph` always has one
    # active standing question). theme_naming and account_synthesis now
    # fire deterministically too, thanks to the planted outlier pair and
    # grown-account claim above (nit 16 fix) -- all four are checked
    # unconditionally.
    for key in ("headline", "standing_question", "theme_naming", "account_synthesis"):
        assert seen_models.get(key) == strong_model, (key, seen_models)


def test_prose_calls_default_to_the_clients_own_model_when_prose_model_is_none(
    full_graph: FullGraph, arr_by_note_slug: dict[str, float]
) -> None:
    """The other half of item 4: omitting `prose_model` (every pre-
    existing caller in this test suite, and every real deployment before
    `STRONG_MODEL` was wired up) must leave each `llm.complete()` call's
    `model` argument `None`, so the client's own default model applies --
    unchanged behaviour, not a silent new requirement to always pass one."""
    seen_models: list[str | None] = []

    def _responder(*, system, user, json_schema, model, surface):
        if system in (HEADLINE_SYSTEM, STANDING_QUESTION_SYSTEM):
            seen_models.append(model)
        return _report_llm_responder(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )

    weekly.build_weekly(
        full_graph.conn,
        FakeLLMClient(_responder),
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )
    assert seen_models, "expected at least the headline and standing-question calls"
    assert all(m is None for m in seen_models)


def test_render_email_is_markdown_with_the_body(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )
    email = weekly.render_email(result)
    assert email.startswith("# Weekly report")
    assert result.body_md in email


def test_render_body_shows_a_pending_resolution_as_evidence_only_no_reason() -> None:
    """Should-fix 5 (review-theme-wave-5-judge.md finding 5) and test gap
    1 ("no test renders the proposal anywhere -- neither the report
    bullet nor the theme note's own block has a single assertion"): the
    report body's "Resolutions proposed" sub-bullet shows the quoted
    evidence and its citation -- never the judge's own free-text reason.

    Nit 9 (review-theme-wave-5b.md finding 9): `PendingResolution` no
    longer HAS a `reason` field at all -- should-fix 5 already dropped
    every reader of it, so a stray reason making it into `pending_
    resolutions` and onto this model is now a `pydantic.ValidationError`
    (`extra="forbid"`), not something this test has to catch by
    scanning rendered text for a string that must not appear."""
    since = date(2026, 8, 17)
    as_of = date(2026, 8, 24)
    pending = PendingResolution(
        resolution_claim_id=99,
        evidence="the delivery delay is over",
        source_slug="render-body-source",
        locator="p1",
    )
    theme = ThemeSummary(
        slug="theme-render-body-fixture",
        name="Render body fixture theme",
        signal_count=3,
        linked_arr=12345.0,
        first_seen=since,
        last_seen=as_of,
        movement="resolution proposed",
        pending_resolutions=[pending],
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    waves = WavesFeed(since=since, as_of=as_of)
    actions = ActionsFeed()
    coverage = CoverageSummary(since=since, as_of=as_of)

    body = weekly._render_body(
        "2026-W34", [], diff, themes_report, waves, [], [], actions, coverage, []
    )

    assert "- **resolution proposed** — [[theme-render-body-fixture]]" in body
    assert '"the delivery delay is over" ^[render-body-source @ p1]' in body


def test_render_body_theme_line_prints_top_level_accounts_and_arr_unknown_note() -> None:
    """Bughunt-2026-09-18 finding 4: the body used to print `len(t.
    account_slugs)` (every LEAF account) next to `t.linked_arr` (summed
    over TOP-LEVEL accounts only -- a child's ARR rolls up under its
    parent) -- a theme with one rolled-up child read as "$X across 3
    accounts" when $X was really the PARENT's ARR alone, and an
    unknown-ARR account vanished with no explanation at all (unlike the
    theme NOTE's own stats block, `_refresh_theme_stats`, which already
    carries the same `(rolled up: ...)`/`(N accounts excluded, ARR
    unknown)` clauses `### Waves` prints). The body's theme line must
    print the SAME top-level account count/list `linked_arr` was
    actually summed over, with those two clauses -- never a leaf count
    next to a top-level sum."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    theme = ThemeSummary(
        slug="theme-arr-rollup-fixture",
        name="ARR rollup fixture theme",
        signal_count=3,
        account_slugs=[
            "ashgrove-medical-center",
            "ashgrove-riverside-hospital",
            "ashgrove-health",
        ],
        linked_arr=2350000.0,
        arr_unknown_accounts=1,
        rolled_up={
            "ashgrove-medical-center": "ashgrove-health",
            "ashgrove-riverside-hospital": "ashgrove-health",
        },
        first_seen=since,
        last_seen=as_of,
        movement="new",
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    waves = WavesFeed(since=since, as_of=as_of)
    actions = ActionsFeed()
    coverage = CoverageSummary(since=since, as_of=as_of)

    body = weekly._render_body(
        "2026-W34", [], diff, themes_report, waves, [], [], actions, coverage, []
    )

    # 1 TOP-LEVEL account (ashgrove-health), matching the set `linked_arr`
    # was summed over -- not 3 leaf accounts.
    assert "1 account ([[ashgrove-health]])" in body
    # A rolled-up child, never its own bullet account.
    assert "[[ashgrove-medical-center]]" not in body
    assert "[[ashgrove-riverside-hospital]]" not in body
    assert "ashgrove-medical-center under ashgrove-health" in body
    assert "ashgrove-riverside-hospital under ashgrove-health" in body
    assert "1 account excluded, ARR unknown" in body
    assert "$2,350,000 ARR" in body


def test_render_body_prints_run_health_only_when_the_judge_had_trouble() -> None:
    """Should-fix 4 (review-theme-wave-5b.md finding 4): a total judge
    outage used to be invisible on the report itself. A "Run health"
    section now prints one line when `judge_failures` is nonzero or the
    cap was hit -- and, just as importantly, prints NOTHING at all (no
    empty section header) on a clean run, so a reader scanning reports
    week over week isn't trained to skip a section that's always there."""
    since = date(2026, 8, 17)
    as_of = date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    waves = WavesFeed(since=since, as_of=as_of)
    actions = ActionsFeed()
    coverage = CoverageSummary(since=since, as_of=as_of)

    healthy = ThemeReport(since=since, as_of=as_of, judge_calls=3, judge_failures=0)
    healthy_body = weekly._render_body(
        "2026-W34", [], diff, healthy, waves, [], [], actions, coverage, []
    )
    assert "Run health" not in healthy_body

    unhealthy = ThemeReport(
        since=since, as_of=as_of, judge_calls=9, judge_failures=9, judge_cap_hit=True
    )
    unhealthy_body = weekly._render_body(
        "2026-W34", [], diff, unhealthy, waves, [], [], actions, coverage, []
    )
    assert "### Run health" in unhealthy_body
    assert "9 of 9 call(s) failed" in unhealthy_body
    assert "hit its per-run cap" in unhealthy_body


def test_render_body_flags_a_total_judge_outage_but_not_a_partial_failure() -> None:
    """Should-fix 7 (review-theme-wave-5c.md finding 7): a run where
    EVERY judge call failed (`judge_failures == judge_calls > 0`) is
    named an OUTAGE in the report line, not left for an operator to
    compute the ratio themselves; a run with only SOME calls failing is
    real trouble too but is not the same signal and must not be
    over-flagged as one."""
    since = date(2026, 8, 17)
    as_of = date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    waves = WavesFeed(since=since, as_of=as_of)
    actions = ActionsFeed()
    coverage = CoverageSummary(since=since, as_of=as_of)

    total_outage = ThemeReport(since=since, as_of=as_of, judge_calls=9, judge_failures=9)
    outage_body = weekly._render_body(
        "2026-W34", [], diff, total_outage, waves, [], [], actions, coverage, []
    )
    assert "OUTAGE" in outage_body

    partial_failure = ThemeReport(since=since, as_of=as_of, judge_calls=9, judge_failures=3)
    partial_body = weekly._render_body(
        "2026-W34", [], diff, partial_failure, waves, [], [], actions, coverage, []
    )
    assert "3 of 9 call(s) failed" in partial_body
    assert "OUTAGE" not in partial_body


def test_headline_rejects_resolved_language_when_no_theme_is_resolved(db) -> None:
    """Should-fix 6 (review-theme-wave-5-judge.md finding 6): the LLM
    headline must not be able to call a proposal "resolved". A responder
    scripted to (wrongly) say a `"resolution proposed"` theme's problem is
    "resolved" gets that sentence dropped outright, never printed, and
    logged to `review_queue`."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    theme = ThemeSummary(
        slug="theme-headline-fixture",
        name="Headline fixture theme",
        signal_count=3,
        linked_arr=1000.0,
        first_seen=date(2026, 8, 3),
        last_seen=date(2026, 8, 20),
        movement="resolution proposed",
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {"sentences": ["The headline fixture theme is now resolved at a few accounts."]}

    llm = FakeLLMClient(_responder)
    sentences = weekly._headline(db, llm, diff, themes_report, [], [])
    assert sentences == []

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload FROM review_queue WHERE payload->>'reason' = "
            "'resolved_language_without_a_resolved_theme'"
        )
        row = cur.fetchone()
    assert row is not None
    assert "resolved" in row[0]["detail"].lower()


def test_headline_allows_resolved_language_when_a_theme_actually_is_resolved(db) -> None:
    """The other half of should-fix 6: the post-check must not become
    over-eager and reject a headline sentence that genuinely, correctly
    says a theme is resolved when one actually is."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    theme = ThemeSummary(
        slug="theme-headline-genuinely-resolved",
        name="Genuinely resolved fixture theme",
        signal_count=3,
        linked_arr=1000.0,
        first_seen=date(2026, 8, 3),
        last_seen=date(2026, 8, 20),
        movement="resolved",
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {
            "sentences": ["The genuinely resolved fixture theme is now resolved at a few accounts."]
        }

    llm = FakeLLMClient(_responder)
    sentences = weekly._headline(db, llm, diff, themes_report, [], [])
    assert sentences == ["The genuinely resolved fixture theme is now resolved at a few accounts."]


def test_headline_rejects_resolved_language_about_one_theme_even_when_another_is_resolved(
    db,
) -> None:
    """review-theme-wave-5b.md finding 1, reproduced then fixed: the
    pre-fix post-check asked one REPORT-WIDE question ("is ANY theme
    resolved?"), so a report holding one genuinely `resolved` theme
    alongside one merely `resolution proposed` theme switched the
    backstop off for BOTH -- the reviewer's own reproduction kept "Late
    deliveries is now resolved." on the record for a theme that was only
    proposed, because "Portal logins" (a different theme in the same
    report) really was resolved.

    review-theme-wave-5c.md finding 1: the per-sentence-only replacement
    (does this sentence use resolved-language AND name a theme that is
    NOT resolved) is the right check when at least one theme in the
    report actually is resolved -- this test's own shape -- but it must
    combine with (not replace) the report-wide guard for the far more
    common zero-resolved-theme report; see the sibling test below for
    that half. Two themes here, named exactly like the review's own
    reproduction, and only the proposed one's sentence must be dropped
    -- the resolved one's own sentence, about a different theme,
    survives untouched."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    proposed_theme = ThemeSummary(
        slug="theme-headline-mixed-proposed",
        name="Late deliveries",
        signal_count=4,
        linked_arr=2000.0,
        first_seen=date(2026, 8, 1),
        last_seen=date(2026, 8, 20),
        movement="resolution proposed",
    )
    resolved_theme = ThemeSummary(
        slug="theme-headline-mixed-resolved",
        name="Portal logins",
        signal_count=2,
        linked_arr=500.0,
        first_seen=date(2026, 7, 1),
        last_seen=date(2026, 8, 1),
        movement="resolved",
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[proposed_theme, resolved_theme])

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {
            "sentences": [
                "Late deliveries is now resolved.",
                "Portal logins is now resolved at a few accounts.",
            ]
        }

    llm = FakeLLMClient(_responder)
    sentences = weekly._headline(db, llm, diff, themes_report, [], [])
    # The proposed theme's false "resolved" sentence must be dropped even
    # though another theme in the same report genuinely is resolved.
    assert sentences == ["Portal logins is now resolved at a few accounts."]

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload FROM review_queue WHERE payload->>'reason' = "
            "'resolved_language_without_a_resolved_theme'"
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0]["detail"] == "Late deliveries is now resolved."


def test_resolved_language_regex_catches_the_widened_vocabulary() -> None:
    """Nit 11 (review-theme-wave-5b.md finding 11): "Late deliveries are
    fixed", "...is closed out", "...resolves this week" all say a theme's
    problem is over without using the word "resolved" -- the post-check
    must catch them too, not just the original three-word list.

    Blocker 1 / finding 11's regex bullet (review-theme-wave-5c.md): the
    bare stems "resolve"/"resolving" were still missed ("will resolve
    next week", "is resolving"), along with two more ways the corpus
    already says a problem is over without any "resolve" root at all --
    "cleared up" and "behind us" -- and, at the time, the bare word
    "over". This is the full canonical vocabulary review-theme-wave-5c.md
    finding 1 names: resolved, resolve, resolving, resolution, fixed,
    cleared up, behind us, over.

    Bughunt-2026-09-18 finding 2: the bare "over" (and a bare "fixed"
    with no hyphen guard) matched ordinary prose that says nothing about
    a theme's state -- "spread over four Midwest accounts", "escalated
    over invoice credits", "over $8.2M", "A fixed-fee renewal at
    Larkspur" -- and on an ordinary week (no theme actually reading
    `resolved`) that meant the report-wide guard below dropped THOSE
    sentences whole too. `over` is dropped from the vocabulary outright
    (no narrower phrasing built on it stayed clear of the false
    positives above); `fixed` keeps a negative lookahead for a following
    hyphen so "fixed-fee"/"fixed-price" no longer matches while "the
    issue is fixed" still does."""
    assert weekly._RESOLVED_LANGUAGE_RE.search("Late deliveries are fixed now.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("Portal logins is closed out at 3 accounts.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("The delivery delay resolves this week.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("Late deliveries will resolve next week.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("Late deliveries is resolving at two accounts.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("The late-delivery resolution is final.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("The late-delivery issue has been cleared up.")
    assert weekly._RESOLVED_LANGUAGE_RE.search("Late deliveries are behind us now.")
    assert not weekly._RESOLVED_LANGUAGE_RE.search("The team closed 3 tickets this week.")
    assert not weekly._RESOLVED_LANGUAGE_RE.search("Late deliveries remains unresolved.")
    # Finding 2's own repro: ordinary prose using "over"/"fixed" that
    # names no theme's state at all must survive, not just the explicit
    # false positives the guard used to produce.
    assert not weekly._RESOLVED_LANGUAGE_RE.search(
        "The complaint spread over four Midwest accounts this week."
    )
    assert not weekly._RESOLVED_LANGUAGE_RE.search("Ashgrove escalated over invoice credits.")
    assert not weekly._RESOLVED_LANGUAGE_RE.search("Tidewater ARR crossed over $8.2M this quarter.")
    assert not weekly._RESOLVED_LANGUAGE_RE.search("A fixed-fee renewal closed at Larkspur.")


def test_headline_rejects_all_seven_attack_sentences_when_no_theme_is_resolved(db) -> None:
    """review-theme-wave-5c.md finding 1: the report-wide guard ("no
    theme in this report is resolved -> reject every resolved-language
    sentence outright") must survive alongside the per-sentence,
    theme-naming check added by review-theme-wave-5b.md, not be replaced
    by it. One attack sentence per word/phrase in the canonical
    resolved-language vocabulary, on a report with a `resolution
    proposed` theme and nothing actually `resolved` -- every one must be
    dropped, including the ones that name no theme at all, which the
    per-sentence-only check (wave 5b/5c's own regression) would have let
    through.

    Bughunt-2026-09-18 finding 2 dropped the bare "over" from the
    vocabulary (it matched ordinary prose with no bearing on a theme's
    state -- see `test_resolved_language_regex_catches_the_widened_
    vocabulary`'s own docstring), so this is now seven attack sentences,
    not eight; that vocabulary change has its own dedicated coverage,
    this test's job is only the report-wide guard around whatever the
    vocabulary currently is."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    theme = ThemeSummary(
        slug="theme-headline-eight-attack",
        name="Late deliveries",
        signal_count=4,
        linked_arr=2000.0,
        first_seen=date(2026, 8, 1),
        last_seen=date(2026, 8, 20),
        movement="resolution proposed",
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])

    attack_sentences = [
        "Late deliveries is now resolved.",
        "Late deliveries will resolve next week.",
        "Late deliveries is resolving at two accounts.",
        "The late deliveries resolution is final.",
        "Meadowbrook's late-delivery complaints are fixed.",
        "The late-delivery issue has been cleared up.",
        "Late deliveries are behind us now.",
    ]
    assert len(attack_sentences) == 7

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {"sentences": list(attack_sentences)}

    llm = FakeLLMClient(_responder)
    sentences = weekly._headline(db, llm, diff, themes_report, [], [])
    assert sentences == []

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload FROM review_queue WHERE payload->>'reason' = "
            "'resolved_language_without_a_resolved_theme'"
        )
        rows = cur.fetchall()
    assert len(rows) == 7
    logged = {row[0]["detail"] for row in rows}
    assert logged == set(attack_sentences)


def test_headline_sentence_citing_the_quiet_threshold_is_kept_but_a_wrong_count_is_dropped(
    db,
) -> None:
    """Coordinator follow-up (2026-09-18, after bughunt-2026-09-18-report.md
    findings 1/3): a live `--llm agent_sdk` run of the fixed code wrote
    "Four accounts have gone quiet for 30 days" -- 4 was a supplied fact
    (`quiet_30_count`), but the 30-day THRESHOLD itself never was, so
    the whole sentence was rejected (`review_queue` reason=
    numeral_not_supplied, detail contains "30"). `diff.QUIET_ACCOUNT_
    THRESHOLDS` is now handed over as `quiet_threshold_30_days` (etc.)
    so a sentence correctly citing the real threshold passes, while one
    citing a fabricated day count still fails."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(
        since=since,
        as_of=as_of,
        quiet_accounts=QuietAccounts(as_of=as_of, quiet_30=["ashgrove-health"]),
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[])

    def _accepted_responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model, surface
        assert "quiet_threshold_30_days" in user
        return {"sentences": ["Ashgrove Health has gone quiet for 30 days."]}

    accepted = weekly._headline(db, FakeLLMClient(_accepted_responder), diff, themes_report, [], [])
    assert accepted == ["Ashgrove Health has gone quiet for 30 days."]

    def _rejected_responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model, surface
        return {"sentences": ["Ashgrove Health has gone quiet for 45 days."]}

    rejected = weekly._headline(db, FakeLLMClient(_rejected_responder), diff, themes_report, [], [])
    assert rejected == []

    with db.cursor() as cur:
        cur.execute(
            "SELECT payload->>'detail' FROM review_queue "
            "WHERE payload->>'reason' = 'numeral_not_supplied' AND payload->>'surface' = 'headline'"
        )
        rows = [r[0] for r in cur.fetchall()]
    assert any("45" in r for r in rows), rows


def test_build_weekly_with_week_uses_that_weeks_own_bounds_not_a_later_as_of(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Review finding G13: `--week 2026-W37 --as-of <a much later date>`
    used to report a "W37" week that actually spanned since=`week_bounds(
    "2026-W37")[0]` through the caller's own (much later) `as_of` -- three
    weeks or more, not seven days. `since`/`as_of` on the returned
    `ReportResult` must be exactly that week's own Monday and Sunday."""
    later_as_of = date(2026, 9, 30)
    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week="2026-W37",
        as_of=later_as_of,
        arr_by_note_slug=arr_by_note_slug,
    )
    expected_since, expected_as_of = week_bounds("2026-W37")
    assert result.since == expected_since
    assert result.as_of == expected_as_of
    assert (result.as_of - result.since).days == 6


def test_report_links_resolve_and_introduces_no_new_lint_errors(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """`run_lint` "reports zero errors" -- checked as zero *new* errors
    relative to the baseline captured right after the pipeline run, before
    any report code touched the graph (`FullGraph.baseline_lint_errors`'s
    own docstring explains the five pre-existing ones: two orphan contact
    notes and three orphan competitor_mention signal notes, none of them
    introduced by this wave's code)."""
    weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )

    store = LocalStore(full_graph.settings.store_root)
    current_errors = run_lint(full_graph.conn, store).errors()

    baseline_identities = {lint_error_identity(f) for f in full_graph.baseline_lint_errors}
    new_errors = [f for f in current_errors if lint_error_identity(f) not in baseline_identities]

    assert new_errors == [], [(f.check, f.note_slug, f.message) for f in new_errors]


def test_week_bounds_round_trips_iso_week_str() -> None:
    monday, sunday = week_bounds("2026-W36")
    assert monday == date(2026, 8, 31)
    assert sunday == date(2026, 9, 6)


def test_coverage_sample_miss_count_reads_filter_audit_changelog(full_graph: FullGraph) -> None:
    """Review finding G19: `CoverageSummary.sample_miss_count` existed
    before this fix wave but nothing ever populated it -- there was no
    measurement of the filter's own miss rate anywhere in the codebase.
    `portolan.pipeline._run_filter_audit` now logs one `filter_audit`
    changelog row per sampled, excluded source; `_coverage_summary` sums
    `facts_found` across the window into this field."""
    import json

    conn = full_graph.conn
    as_of = AS_OF
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO changelog (occurred_at, kind, summary, details) "
            "VALUES (%s, 'filter_audit', %s, %s::jsonb)",
            (
                datetime.combine(as_of, datetime.min.time(), tzinfo=UTC),
                "test filter-audit row",
                json.dumps({"source_id": "S-test-filter-audit", "facts_found": 3}),
            ),
        )
    conn.commit()

    from portolan.report.weekly import _coverage_summary

    coverage = _coverage_summary(conn, since=WINDOW_START, as_of=as_of)
    assert coverage.sample_miss_count is not None
    assert coverage.sample_miss_count >= 3


def test_cli_writes_the_markdown_file(full_graph: FullGraph, tmp_path) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "reports"
    # The CLI opens its own connection via DATABASE_URL; point it at this
    # session's *actual* database (`full_graph.database_url`, not the
    # process's own `DATABASE_URL` -- the fixture derives its own,
    # differently-named database from that env var specifically so it
    # never collides with `tests/conftest.py`'s per-test schema resets;
    # see `test_report_helpers._session_database_url`), already migrated
    # and populated, rather than standing up a second graph.
    env = dict(os.environ)
    env["DATABASE_URL"] = full_graph.database_url

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "portolan.report",
            "weekly",
            "--as-of",
            "2026-09-04",
            "--llm",
            "fake",
            "--embed",
            "fake",
            "--corpus",
            str(repo_root / "corpus" / "out"),
            "--out",
            str(out_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    written = list(out_dir.glob("report-*.md"))
    assert written, result.stdout
    assert written[0].read_text().startswith("# Weekly report")


def test_cli_prints_a_usage_summary_naming_both_models(full_graph: FullGraph, tmp_path) -> None:
    """Task item 4: "the usage summary the CLI prints must show both
    models." `--llm fake` doesn't change which model NAMES `main()`
    prints (`_build_llm`'s docstring: `fake` is a report-CLI-only replay,
    but `prose_model`/`settings.extraction_model` are still read from the
    real environment/.env regardless of `--llm`) -- this test doesn't
    assert specific model names (those are this dev machine's own .env,
    not this test's business), only that both are named on one line."""
    import os
    import re
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "reports"
    env = dict(os.environ)
    env["DATABASE_URL"] = full_graph.database_url

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "portolan.report",
            "weekly",
            "--as-of",
            "2026-09-04",
            "--llm",
            "fake",
            "--embed",
            "fake",
            "--corpus",
            str(repo_root / "corpus" / "out"),
            "--out",
            str(out_dir),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    models_line = next((ln for ln in result.stdout.splitlines() if ln.startswith("Models:")), None)
    assert models_line is not None, result.stdout
    assert re.match(r"Models: extraction=\S+ strong=\S+$", models_line), models_line
    assert "Usage" in result.stdout, result.stdout


def test_print_usage_summary_counts_every_report_surface_call(
    full_graph: FullGraph, capsys
) -> None:
    """Should-fix 4 / test gap 8: no test of `_print_usage_summary` at all
    previously existed. Plants one `usage_events` row per prose call a
    real report run makes -- theme naming, headline, standing question,
    account synthesis, all `surface='report'` now that the synthesis
    surface fix routes account-synthesis prose there too -- across two
    different models, and asserts every one of them appears in the
    printed per-model breakdown. A `surface='consolidate'` event (the
    OLD, wrong surface account-synthesis prose used to record itself
    under) is planted alongside them and must NOT appear -- proving the
    summary counts by surface, not by tenant/time window alone."""
    conn = full_graph.conn
    tenant = full_graph.settings.tenant
    since_ts = clock_module.now()
    sink = PostgresSink(conn)

    report_calls = [
        ("theme_naming", "usage-summary-model-a"),
        ("headline", "usage-summary-model-a"),
        ("standing_question", "usage-summary-model-b"),
        ("account_synthesis", "usage-summary-model-b"),
    ]
    for label, model_key in report_calls:
        with metered(sink, tenant, "report", model_key) as usage:
            usage.input_tokens = 10
            usage.output_tokens = 5
            usage.extra["call"] = label

    # The old, wrong surface -- must be excluded from the printed summary.
    with metered(sink, tenant, "consolidate", "usage-summary-model-c") as usage:
        usage.input_tokens = 1
        usage.output_tokens = 1

    weekly._print_usage_summary(
        conn,
        tenant=tenant,
        since_ts=since_ts,
        extraction_model="extraction-test-model",
        strong_model="strong-test-model",
        themes=ThemeReport(since=date(2026, 8, 17), as_of=date(2026, 8, 24)),
    )
    out = capsys.readouterr().out

    assert "usage-summary-model-a: 2 calls" in out, out
    assert "usage-summary-model-b: 2 calls" in out, out
    assert "usage-summary-model-c" not in out, out


def test_print_usage_summary_shows_the_resolution_judges_own_counts(
    full_graph: FullGraph, capsys
) -> None:
    """Should-fix 4 (review-theme-wave-5b.md finding 4): the CLI's own
    usage summary must show the resolution judge's call/failure/cap
    counts unconditionally, the same "named explicitly" discipline the
    `Models:` line already gets -- not just when something broke."""
    conn = full_graph.conn
    tenant = full_graph.settings.tenant
    since_ts = clock_module.now()

    themes_report = ThemeReport(
        since=date(2026, 8, 17),
        as_of=date(2026, 8, 24),
        judge_calls=9,
        judge_failures=9,
        judge_cap_hit=True,
    )
    weekly._print_usage_summary(
        conn,
        tenant=tenant,
        since_ts=since_ts,
        extraction_model="extraction-test-model",
        strong_model="strong-test-model",
        themes=themes_report,
    )
    out = capsys.readouterr().out
    assert "Resolution judge: 9 call(s), 9 failed, cap hit" in out, out


# --------------------------------------------------------------------------
# Owner review 2026-09-18 (live PDF export, `report-2026-W36`): rendering
# fixes for the theme line (name once, up to 3 signal bullets), Actions
# (no repeated sentence), the headline (prose paragraph, visible failure
# reason), and internal caveats (one footnote, plain headings). Citation
# markers becoming real links is `portolan.web.markdown`'s job, tested in
# `tests/test_web.py`.
# --------------------------------------------------------------------------


def _bare_feeds(since: date, as_of: date):
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    waves = WavesFeed(since=since, as_of=as_of)
    actions = ActionsFeed()
    coverage = CoverageSummary(since=since, as_of=as_of)
    themes_report = ThemeReport(since=since, as_of=as_of)
    return diff, waves, actions, coverage, themes_report


def test_render_body_renders_the_executive_summary_and_drops_the_old_headline() -> None:
    """Weekly-report wave 2 ("the headline is pretty short. It should be
    like an executive summary"), then the coordinator's own polish pass
    (2026-09-18, real W35 review: "the Epic-integration story said
    twice"): `_render_body` prints "### Executive summary" with all four
    named parts in order, each account's own paragraph blank-line-
    separated from its heading (so it renders as its own block, not
    merged with the "**By account**" label into one run-on paragraph) --
    and the OLD "### Headline" section, now redundant with "This week",
    is gone from the rendered body entirely."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, actions, coverage, themes_report = _bare_feeds(since, as_of)
    exec_summary = ExecutiveSummary(
        this_week=["Delivery delays eased at three Midwest hospitals this week."],
        attention=["Ashgrove Health flagged ongoing delivery delays. ^[S-1 @ loc:1]"],
        by_account=[
            AccountParagraph(account_slug="ashgrove-health", paragraph="Still waiting on a fix."),
            AccountParagraph(account_slug="cobalt-ridge", paragraph="Quiet week."),
        ],
        suggested_actions=["Schedule a call with Ashgrove Health. ^[S-1 @ loc:1]"],
    )

    body = weekly._render_body(
        "2026-W34",
        [],
        diff,
        themes_report,
        waves,
        [],
        [],
        actions,
        coverage,
        [],
        executive_summary=exec_summary,
    )

    assert "### Executive summary" in body
    assert "### Headline" not in body
    assert "**This week.** Delivery delays eased at three Midwest hospitals this week." in body
    assert "**What needs attention**" in body
    assert "- Ashgrove Health flagged ongoing delivery delays. ^[S-1 @ loc:1]" in body
    assert "**By account**" in body
    # Each account heading is its own line, blank-line-separated from its
    # paragraph on both sides -- the exact shape a markdown renderer needs
    # to keep them as two distinct blocks, not one run-on paragraph.
    assert (
        "**[[ashgrove-health]]**\n\nStill waiting on a fix.\n\n**[[cobalt-ridge]]**\n\nQuiet week."
        in body
    )
    assert "**Suggested actions**" in body
    assert "- Schedule a call with Ashgrove Health. ^[S-1 @ loc:1]" in body


def test_render_body_with_an_empty_executive_summary_prints_no_section() -> None:
    """`None`, or an all-empty `ExecutiveSummary` (a total failure --
    the model call raised, or every part was gated out), must print
    nothing at all for this section -- not an empty "### Executive
    summary" heading with nothing under it."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, actions, coverage, themes_report = _bare_feeds(since, as_of)

    body_none = weekly._render_body(
        "2026-W34", [], diff, themes_report, waves, [], [], actions, coverage, []
    )
    assert "### Executive summary" not in body_none

    body_empty = weekly._render_body(
        "2026-W34",
        [],
        diff,
        themes_report,
        waves,
        [],
        [],
        actions,
        coverage,
        [],
        executive_summary=ExecutiveSummary(failure_reason="the model call failed"),
    )
    assert "### Executive summary" not in body_empty
    assert "executive summary unavailable: the model call failed" in body_empty


def test_theme_line_names_the_theme_once_and_prints_signal_bullets() -> None:
    """The theme name is the `[[slug]]` wikilink's own rendered anchor
    text (`portolan.web.markdown._wikilink_pass`) -- `_render_body` must
    not ALSO quote it a second time right after the link (owner review
    2026-09-18: "the name of the company is repeated 2x in each line").
    Weekly-report wave 2: the fixed "up to three representative quotes"
    preview is gone -- every `ThemeSummary.all_signals` entry now prints
    inside an expandable signals-fold block naming its account, its fact
    type, and citing its source."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, actions, coverage, _ = _bare_feeds(since, as_of)
    theme = ThemeSummary(
        slug="theme-names-once-fixture",
        name="Late deliveries",
        signal_count=7,
        account_slugs=["ashgrove-health", "cobalt-ridge-medical-center"],
        linked_arr=4_620_000.0,
        first_seen=date(2026, 8, 3),
        last_seen=date(2026, 8, 20),
        movement="rising",
        all_signals=[
            SignalDetail(
                claim_id=1,
                fact_type="complaint",
                claim_text="Deliveries have been late for three weeks running.",
                source_slug="src-call-001",
                locator="00:12:30",
                account_slug="ashgrove-health",
                note_slug="signal-complaint-ashgrove-health",
            )
        ],
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])

    body = weekly._render_body(
        "2026-W34", [], diff, themes_report, waves, [], [], actions, coverage, []
    )

    assert '[[theme-names-once-fixture]] "Late deliveries"' not in body
    # The theme's own name never appears as literal text in the RAW
    # markdown at all any more -- it lives only in the `[[slug]]` token,
    # resolved to the note's title exactly once at HTML-render time
    # (`portolan.web.markdown`, proven end to end in `tests/test_web.py`'s
    # `test_report_detail_strips_a_stored_repeated_theme_name`).
    assert "Late deliveries" not in body
    assert "7 signals across 2 accounts" in body
    assert "<!--signals:7 signals-->" in body
    assert (
        "  - **complaint** — [[ashgrove-health]]: Deliveries have been late for three "
        "weeks running. ^[src-call-001 @ 00:12:30] · [[signal-complaint-ashgrove-health]]" in body
    )


def test_quote_adds_nothing_catches_identical_and_contained_quotes() -> None:
    """Coordinator review, 2026-09-18 (real W35 signals-fold output): a
    quote sub-bullet identical to the claim sentence above it, or a quote
    that's just the sentence with a little more framing around it, tells
    a reader nothing new."""
    sentence = "Driftwood Community Hospital said it had briefly compared Corvane pumps."
    assert weekly._quote_adds_nothing(sentence, sentence)  # byte-identical
    # Curly quotes / a trailing period difference must not defeat this --
    # compared on a normalised form.
    assert weekly._quote_adds_nothing(
        sentence,
        sentence.rstrip(".") + ".",  # same words, punctuation reformatted
    )
    # The quote is the sentence PLUS extra framing -- the sentence's own
    # words are wholly contained in it.
    assert weekly._quote_adds_nothing(
        "Larkspur wants a 12 percent discount.",
        "on 2026-08-25, Larkspur wants a 12 percent discount given delivery issues.",
    )
    # A genuinely different quote is kept.
    assert not weekly._quote_adds_nothing(
        "Sable Valley praised the HL7 feed.", "flawless since go-live"
    )


def test_signals_fold_block_omits_a_quote_sub_bullet_that_adds_nothing() -> None:
    """`_signals_fold_block` itself must skip the quote line, not just
    the helper function that decides -- proven directly against the
    markdown it emits."""
    duplicate = SignalDetail(
        claim_id=1,
        fact_type="complaint",
        claim_text="Driftwood said the pumps never went anywhere.",
        quote="Driftwood said the pumps never went anywhere.",
        source_slug="S-1",
        locator="l1",
        account_slug="driftwood",
        note_slug="signal-complaint-driftwood",
    )
    distinct = SignalDetail(
        claim_id=2,
        fact_type="praise",
        claim_text="Sable Valley praised the HL7 feed.",
        quote="flawless since go-live",
        source_slug="S-2",
        locator="l2",
        account_slug="sable-valley",
        note_slug="signal-praise-sable-valley",
    )
    block = weekly._signals_fold_block([duplicate, distinct], chip_label="2 signals")
    # The sentence appears once, unquoted, in the main bullet -- never
    # again as a quote sub-bullet.
    assert block.count("Driftwood said the pumps never went anywhere.") == 1
    assert '"Driftwood said the pumps never went anywhere."' not in block
    assert '"flawless since go-live"' in block  # the distinct quote survives


def test_action_line_does_not_repeat_its_own_sentence() -> None:
    """Owner review 2026-09-18 ("the link text is the full sentence,
    then the same sentence is repeated after a colon"): an action note's
    own title IS `a.what` (truncated to 120 chars, `portolan.merge.
    actions.merge_action`) -- the `[[slug]]` link already says it."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, _, coverage, themes_report = _bare_feeds(since, as_of)
    action = ActionRow(
        id=1,
        note_slug="action-1-ping-ashgrove",
        what="Ping Ashgrove about the renewal",
        owner="Dana Whitlock",
        due=date(2026, 8, 30),
        status="open",
    )
    actions = ActionsFeed(open=[action])

    body = weekly._render_body(
        "2026-W34", [], diff, themes_report, waves, [], [], actions, coverage, []
    )

    assert "[[action-1-ping-ashgrove]]: Ping Ashgrove about the renewal" not in body
    assert "[[action-1-ping-ashgrove]] (owner: Dana Whitlock, due 2026-08-30)" in body
    assert body.count("Ping Ashgrove about the renewal") == 0


def test_headline_sentences_never_reach_the_rendered_body() -> None:
    """Superseded by the coordinator's polish pass, 2026-09-18 (real W35
    review: the executive summary's "This week" already told the same
    story the old headline did, word for word) -- `_render_body` no
    longer renders `headline` at all, in prose or bulleted form; the
    parameter is still accepted (every existing positional caller keeps
    working) but has no effect on the output."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, actions, coverage, themes_report = _bare_feeds(since, as_of)

    headline = ["First sentence of the headline.", "Second sentence of the headline."]
    body = weekly._render_body(
        "2026-W34", headline, diff, themes_report, waves, [], [], actions, coverage, []
    )
    assert "First sentence of the headline" not in body
    assert "- First sentence" not in body
    assert "### Headline" not in body


def test_internal_caveats_move_to_a_single_notes_footnote() -> None:
    """Owner review 2026-09-18 ("internal caveats leak into headings and
    bullets"): under `clock='world'`, the record-time caveats no longer
    sit inline in the '### Coverage' heading or the 'Resolved this
    window' bullet -- both stay plain, and both caveats move into one
    '### Notes' footnote at the end."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, actions, coverage, themes_report = _bare_feeds(since, as_of)

    body = weekly._render_body(
        "2026-W34", [], diff, themes_report, waves, [], [], actions, coverage, [], clock="world"
    )
    assert "\n### Coverage\n" in body
    assert "### Coverage (record" not in body
    assert "- Resolved this window: 0\n" in body
    assert "- Resolved this window: 0 (record" not in body
    assert "### Notes" in body
    assert body.count("record time only") >= 2
    assert body.index("### Notes") > body.index("### Coverage")


def test_headline_reason_is_accepted_but_no_longer_rendered() -> None:
    """Coordinator polish pass, 2026-09-18: `_render_body` no longer
    renders the old "### Headline" section (or a failure footnote for
    it) at all -- `headline`/`headline_reason` stay accepted parameters
    (every existing positional caller of this function keeps working
    unchanged) but produce no visible text now, superseding the old
    "narrative unavailable" footnote behaviour this test used to pin."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff, waves, actions, coverage, themes_report = _bare_feeds(since, as_of)

    body = weekly._render_body(
        "2026-W34",
        [],
        diff,
        themes_report,
        waves,
        [],
        [],
        actions,
        coverage,
        [],
        headline_reason="the model call failed (RuntimeError: boom)",
    )
    assert "### Headline" not in body
    assert "(no headline this window)" not in body
    assert "narrative unavailable" not in body


def test_headline_records_a_reason_when_the_model_call_raises(db) -> None:
    """Owner review 2026-09-18: a provider/model failure on the one
    `llm.complete()` call `_headline` makes must not propagate -- it's
    caught, and a plain-language reason lands in `failure_reasons`."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    themes_report = ThemeReport(since=since, as_of=as_of)

    def _raising(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        raise RuntimeError("openrouter auth failed")

    llm = FakeLLMClient(_raising)
    reasons: list[str] = []
    sentences = weekly._headline(db, llm, diff, themes_report, [], [], failure_reasons=reasons)
    assert sentences == []
    assert reasons and "openrouter auth failed" in reasons[0]


def test_headline_records_a_reason_when_every_sentence_is_rejected(db) -> None:
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    theme = ThemeSummary(
        slug="theme-headline-reason-fixture",
        name="Headline reason fixture theme",
        signal_count=3,
        linked_arr=1000.0,
        first_seen=date(2026, 8, 3),
        last_seen=date(2026, 8, 20),
        movement="resolution proposed",
    )
    themes_report = ThemeReport(since=since, as_of=as_of, themes=[theme])

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {"sentences": ["The headline reason fixture theme is now resolved."]}

    llm = FakeLLMClient(_responder)
    reasons: list[str] = []
    sentences = weekly._headline(db, llm, diff, themes_report, [], [], failure_reasons=reasons)
    assert sentences == []
    assert reasons and "resolved-language" in reasons[0]


def test_build_weekly_survives_a_headline_model_failure_silently(
    full_graph: FullGraph, arr_by_note_slug: dict[str, float]
) -> None:
    """Owner review 2026-09-18: a provider/model failure on the headline
    call must not take down the whole report -- the rest of the report
    (themes, actions, coverage, ...) still gets written. Coordinator
    polish pass, 2026-09-18: since `_render_body` no longer renders the
    headline (or a failure footnote for it) at all, that failure is now
    silent from a reader's point of view -- `_headline` is still called
    (`result.headline` still reflects it failed) but nothing about it,
    including the raised exception's own message, reaches `body_md`."""

    def _headline_raises(*, system, user, json_schema, model, surface):
        if surface == "report" and system == HEADLINE_SYSTEM:
            raise RuntimeError("openrouter: 401 unauthorized")
        return _report_llm_responder(
            system=system, user=user, json_schema=json_schema, model=model, surface=surface
        )

    result = weekly.build_weekly(
        full_graph.conn,
        FakeLLMClient(_headline_raises),
        full_graph.embedder,
        week=None,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )
    assert result.headline == []
    assert "### Headline" not in result.body_md
    assert "(no headline this window)" not in result.body_md
    assert "narrative unavailable" not in result.body_md
    assert "openrouter: 401 unauthorized" not in result.body_md
    assert "### Actions" in result.body_md  # the rest of the report still ran


def test_headline_strips_a_stray_wikilink_the_model_wrote_anyway(db) -> None:
    """Owner review 2026-09-18, live regeneration against `portolan_
    demo4`: `HEADLINE_SYSTEM` now says not to, but a model can still
    write `[[slug]]` into a headline sentence -- caught here (never
    reaches `result.headline`/the rendered report) since it duplicates
    an account/theme name the report's own mechanical sections already
    link."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    themes_report = ThemeReport(since=since, as_of=as_of)

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {
            "sentences": [
                "Larkspur Regional Health appointed a new CFO. ([[larkspur-regional-health]])"
            ]
        }

    llm = FakeLLMClient(_responder)
    sentences = weekly._headline(db, llm, diff, themes_report, [], [])
    assert sentences == ["Larkspur Regional Health appointed a new CFO."]


def test_build_weekly_prints_a_line_when_a_card_is_dropped_for_judge_error(
    full_graph: FullGraph, report_llm, capsys
) -> None:
    """S6 (`01-spec/reviews/review-action-researcher-2026-09-20.md`,
    should-fix 6): "A judge outage is invisible in the report ... the
    `research` CLI prints `gate=dropped:judge_error:...`; `build_weekly`
    should surface the same count." Before this fix the only trace
    anywhere of a judge-error drop during a full `--research` weekly run
    was a `changelog` row -- Suggested actions simply got shorter, with
    nothing telling an operator a judge OUTAGE, not a quiet week, was
    why."""
    from test_report_research_cli import _assign_a_vendor_side_owner, _FakeResearcherFromClaims

    _since, as_of = week_bounds("2026-W36")
    researcher = _FakeResearcherFromClaims(full_graph.conn, as_of=as_of)

    def _always_errors(*, sentence, reference_text):
        del sentence, reference_text
        return None

    # Probe run: discover a real candidate with a triggering claim to
    # cite, then give its account a real vendor-side owner -- the exact
    # two-pass shape `test_run_research_harness_end_to_end_with_a_fake_
    # researcher_and_judge` uses, since `find_candidates` reads `owner_
    # entity_id` fresh every call. No card is produced by this pass at
    # all (no account has an owner yet), so which judge is given here
    # makes no difference.
    probe_outcomes, _probe_themes = weekly.run_research_harness(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week="2026-W36",
        clock="record",
        connector=full_graph.connector,
        candidate_slugs=None,
        researcher=researcher,
        judge=_always_errors,
    )
    citable = next(o for o in probe_outcomes if o.candidate.claim_ids)
    _assign_a_vendor_side_owner(full_graph.conn, account_slug=citable.candidate.account_slug)

    capsys.readouterr()  # drain anything the probe run above printed
    weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week="2026-W36",
        as_of=as_of,
        connector=full_graph.connector,
        research=True,
        researcher=researcher,
        judge=_always_errors,
    )
    printed = capsys.readouterr().out

    assert "judge_error" in printed, printed


def test_build_weekly_without_research_shows_a_previously_queued_action_card(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Regen-shows-queued-cards fix (`01-spec/actions-and-owners-2026-09-
    19.md` Phase 2 follow-up, coordinator 2026-09-20): "a plain `weekly
    --week 2026-W36` regeneration erases the Suggested actions cards a
    previous `--research` run queued." `action_card_report_lines` is now
    a cheap, unconditional read (`build_weekly`'s own call-site comment)
    -- a card a PRIOR `--research` run queued for this report's week must
    render even when THIS run passes `research=False` (the default), with
    no researcher re-run.

    Superseded test (this test used to be named `..._ignores_existing_
    action_card_rows` and asserted the opposite): that old assertion only
    ever held because its own fixture row carried no `"week"` key at all,
    so it could never have matched any week's filter regardless of
    `research` -- not because a plain regen genuinely ignored existing
    cards, which was exactly the live-graph bug this fix addresses (W36:
    four queued cards, two confirmed, invisible on a bare regen).

    Uses a week ("2026-W50") no other test in this shared, session-scoped
    `full_graph` database touches, so this test's own queued row can't
    collide with (or be polluted by) action-card rows other test modules
    queue for the weeks (`2026-W34`..`2026-W39`) they already use."""
    import json

    week = "2026-W50"
    _since, week_as_of = week_bounds(week)
    with full_graph.conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('action_card', %s::jsonb)",
            (
                json.dumps(
                    {
                        "card": {
                            "verdict": "action",
                            "what": "UNIQUE-MARKER-queued-before-this-regen",
                            "owner": "sam-okafor",
                            "due": "2026-10-01",
                            "confidence": "high",
                        },
                        "week": week,
                    }
                ),
            ),
        )
    full_graph.conn.commit()

    result = weekly.build_weekly(
        full_graph.conn,
        report_llm,
        full_graph.embedder,
        week=week,
        as_of=week_as_of,
        arr_by_note_slug=arr_by_note_slug,
        research=False,
    )

    assert (
        "[action] UNIQUE-MARKER-queued-before-this-regen (proposed, confidence: high)"
        in result.body_md
    )


def test_build_weekly_research_false_is_the_default(
    full_graph: FullGraph, report_llm, arr_by_note_slug: dict[str, float]
) -> None:
    """Calling `build_weekly` with and without the `research`/`researcher`
    parameters produces the same body -- omitting them entirely (every
    call site that predates this wave) behaves exactly like passing
    `research=False` explicitly."""
    kwargs = dict(week=None, as_of=AS_OF, arr_by_note_slug=arr_by_note_slug, clock="record")
    with_default = weekly.build_weekly(full_graph.conn, report_llm, full_graph.embedder, **kwargs)
    with_explicit_false = weekly.build_weekly(
        full_graph.conn, report_llm, full_graph.embedder, research=False, **kwargs
    )
    assert with_default.body_md == with_explicit_false.body_md


def test_headline_rejects_a_citation_marker_missing_its_leading_caret(db) -> None:
    """Owner review 2026-09-18, live regeneration against `portolan_
    demo4`: a model wrote the right source and locator but dropped the
    leading `^` -- '[S-rss-006 @ url:... 2026-09-02]' instead of
    '^[S-rss-006 @ ...]'. `validate_citations` finds zero REAL `^[...]`
    markers in that text and so passes it vacuously; this sentence must
    still be dropped, not printed with a raw, unlinked bracket."""
    since, as_of = date(2026, 8, 17), date(2026, 8, 24)
    diff = DiffFeed(since=since, as_of=as_of, quiet_accounts=QuietAccounts(as_of=as_of))
    themes_report = ThemeReport(since=since, as_of=as_of)

    def _responder(*, system, user, json_schema, model, surface):
        del user, model, json_schema, surface
        assert system == weekly.HEADLINE_SYSTEM
        return {
            "sentences": [
                "Larkspur Regional Health appointed a new CFO. [S-rss-006 @ url:x 2026-09-02]"
            ]
        }

    llm = FakeLLMClient(_responder)
    sentences = weekly._headline(db, llm, diff, themes_report, [], [])
    assert sentences == []
