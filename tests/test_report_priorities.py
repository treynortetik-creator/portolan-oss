"""`portolan.report.priorities` tests (Phase 3, `01-spec/actions-and-
owners-2026-09-19.md` "ranked priorities").

Like `test_report_waves.py`, these use the plain, fresh `db` fixture with
hand-built accounts/signals/theme fixtures rather than `full_graph`'s
whole-corpus session fixture -- ranking is a pure function of
`theme_members`/`claims`/`registry_entities`, no theme-clustering
machinery involved.
"""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.report import priorities
from portolan.report.models import ThemePriority, ThemeSummary
from portolan.report.numbers import NumberFacts, check_numerals

SINCE = date(2026, 8, 24)
AS_OF = date(2026, 8, 30)
IN_WINDOW = SINCE + timedelta(days=1)


def _make_account_note(conn, slug: str) -> None:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.ACCOUNT,
        title=slug,
        description="Priority test fixture account.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id=f"TEST-{slug}",
    )
    write_note(conn, note, [], changed_by="test", reason="priority test fixture account")


def _make_signal_note(conn, slug: str) -> None:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title=slug,
        description="Priority test fixture signal note.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(conn, note, [], changed_by="test", reason="priority test fixture signal")


def _make_bare_theme_note(conn, slug: str, title: str) -> None:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.THEME,
        title=title,
        description="Priority test fixture theme.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(conn, note, [], changed_by="test", reason="priority test fixture theme")


def _insert_source(conn, slug: str) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sources (slug, source_type) VALUES (%s, 'doc')", (slug,))
    conn.commit()


def _insert_claim(
    conn,
    note_slug: str,
    fact_type: str,
    claim_text: str,
    source_slug: str,
    locator: str,
    recorded: date,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (note_slug, fact_type, claim_text, source_slug, locator, recorded),
        )
        claim_id = cur.fetchone()[0]
    conn.commit()
    return claim_id


def _add_member(conn, theme_slug: str, claim_id: int, added: date) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO theme_members (theme_slug, claim_id, added, attached_on) "
            "VALUES (%s, %s, %s, %s)",
            (theme_slug, claim_id, added, added),
        )
    conn.commit()


def _link_parent_child(conn, parent_slug: str, child_slugs: list[str]) -> None:
    """Same synthetic `registry_entities` parent/child chain
    `test_report_waves.py`'s own `_link_parent_child` builds."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, note_slug) "
            "VALUES ('account', %s, %s) RETURNING id",
            (parent_slug, parent_slug),
        )
        (parent_id,) = cur.fetchone()
        for child_slug in child_slugs:
            cur.execute(
                "INSERT INTO registry_entities (kind, canonical_name, note_slug, parent_id) "
                "VALUES ('account', %s, %s, %s)",
                (child_slug, child_slug, parent_id),
            )
    conn.commit()


def _touch_theme(
    conn,
    *,
    theme_slug: str,
    theme_title: str,
    fact_type: str,
    account_slugs: list[str],
    source_slug: str,
    recorded: date = IN_WINDOW,
) -> None:
    """A theme note plus one member claim per account in `account_slugs`,
    each on its own `signal-<fact_type>-<account>` note (case 2 of
    `portolan.report.accounts.home_account_slug`) -- the same shape
    `test_report_waves.py`'s own fixtures use, so `_touched_accounts_in_
    window` resolves every claim back to its account with no `[[slug]]`
    wikilink text needed."""
    _make_bare_theme_note(conn, theme_slug, theme_title)
    _insert_source(conn, source_slug)
    for i, account_slug in enumerate(account_slugs):
        _make_account_note(conn, account_slug)
        signal_slug = f"signal-{fact_type}-{account_slug}"
        _make_signal_note(conn, signal_slug)
        claim_id = _insert_claim(
            conn,
            signal_slug,
            fact_type,
            f"[[{account_slug}]] raised a {fact_type} issue.",
            source_slug,
            f"{source_slug}#{i}",
            recorded,
        )
        _add_member(conn, theme_slug, claim_id, recorded)


def _theme(slug: str, name: str, movement: str) -> ThemeSummary:
    return ThemeSummary(
        slug=slug,
        name=name,
        signal_count=0,
        linked_arr=0.0,
        first_seen=None,
        last_seen=None,
        movement=movement,
    )


def test_ordering_by_accounts_then_arr_then_movement_then_slug_tiebreak(db) -> None:
    """The full ranking, in one scenario: accounts touched beats ARR
    touched beats movement beats a final alphabetical tie-break on
    `theme_slug` -- exactly the priority order the task text spells out."""
    # theme-big: 3 accounts touched -- must outrank everything below no
    # matter how low its own ARR or how "late" its own movement is.
    _touch_theme(
        db,
        theme_slug="theme-big",
        theme_title="Big theme",
        fact_type="delivery",
        account_slugs=["big-acct-1", "big-acct-2", "big-acct-3"],
        source_slug="src-big",
    )
    # theme-more-arr: 2 accounts (fewer than theme-big), but far more ARR
    # than the "tie group" below -- must beat them on ARR alone.
    _touch_theme(
        db,
        theme_slug="theme-more-arr",
        theme_title="High ARR theme",
        fact_type="pricing",
        account_slugs=["arr-acct-1", "arr-acct-2"],
        source_slug="src-arr",
    )
    # Tie group: three themes, all 2 accounts touched, all the SAME ARR --
    # movement, then slug, decide the order among them.
    for slug, title in (
        ("z-theme-new", "Z new theme"),
        ("a-theme-new", "A new theme"),
        ("m-theme-falling", "M falling theme"),
    ):
        _touch_theme(
            db,
            theme_slug=slug,
            theme_title=title,
            fact_type="risk",
            account_slugs=[f"{slug}-acct-1", f"{slug}-acct-2"],
            source_slug=f"src-{slug}",
        )

    arr_by_note_slug = {
        "big-acct-1": 1_000.0,
        "big-acct-2": 1_000.0,
        "big-acct-3": 1_000.0,
        "arr-acct-1": 400_000.0,
        "arr-acct-2": 100_000.0,
        "z-theme-new-acct-1": 50_000.0,
        "z-theme-new-acct-2": 50_000.0,
        "a-theme-new-acct-1": 50_000.0,
        "a-theme-new-acct-2": 50_000.0,
        "m-theme-falling-acct-1": 50_000.0,
        "m-theme-falling-acct-2": 50_000.0,
    }
    themes = [
        _theme("theme-big", "Big theme", "resolved"),
        _theme("theme-more-arr", "High ARR theme", "falling"),
        _theme("z-theme-new", "Z new theme", "new"),
        _theme("a-theme-new", "A new theme", "new"),
        _theme("m-theme-falling", "M falling theme", "falling"),
    ]

    ranked = priorities.compute_theme_priorities(
        db,
        themes=themes,
        since=SINCE,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
        region_by_note_slug={},
    )

    assert [p.theme_slug for p in ranked] == [
        "theme-big",
        "theme-more-arr",
        "a-theme-new",
        "z-theme-new",
        "m-theme-falling",
    ]
    assert ranked[0].account_count == 3
    assert ranked[1].arr == 500_000.0


def test_arr_rollup_counts_a_facility_once_even_when_the_system_is_also_touched(db) -> None:
    """A theme touching both a parent SYSTEM account and one of its own
    CHILD facilities counts both as accounts touched (leaf-level, task
    text: "accounts touched (distinct leaf accounts)"), but their ARR
    rolls up to the parent alone -- the same antichain collapse
    `portolan.report.waves.compute_waves` already applies, reused here
    via `themes._top_level_accounts`/`_linked_arr`. Double-counting the
    child's own ARR on top of the parent's would overstate the theme's
    real linked revenue."""
    # `_link_parent_child` needs both account notes to already exist
    # (`registry_entities.note_slug`'s own foreign key) -- created first,
    # same order `test_report_waves.py`'s own rollup test uses, rather
    # than through `_touch_theme`'s own (second, redundant) account-note
    # creation.
    _make_account_note(db, "rollup-system")
    _make_account_note(db, "rollup-facility")
    _link_parent_child(db, "rollup-system", ["rollup-facility"])
    _make_bare_theme_note(db, "theme-rollup", "Rollup theme")
    _insert_source(db, "src-rollup")
    for i, account_slug in enumerate(("rollup-system", "rollup-facility")):
        signal_slug = f"signal-delivery-{account_slug}"
        _make_signal_note(db, signal_slug)
        claim_id = _insert_claim(
            db,
            signal_slug,
            "delivery",
            f"[[{account_slug}]] raised a delivery issue.",
            "src-rollup",
            f"src-rollup#{i}",
            IN_WINDOW,
        )
        _add_member(db, "theme-rollup", claim_id, IN_WINDOW)
    arr_by_note_slug = {"rollup-system": 300_000.0, "rollup-facility": 50_000.0}

    ranked = priorities.compute_theme_priorities(
        db,
        themes=[_theme("theme-rollup", "Rollup theme", "rising")],
        since=SINCE,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
        region_by_note_slug={},
    )

    assert len(ranked) == 1
    p = ranked[0]
    assert p.account_count == 2
    assert sorted(p.account_slugs) == ["rollup-facility", "rollup-system"]
    # Counted once (the parent's own $300,000), not the $350,000 sum of
    # both -- the facility's own ARR is rolled up under the system, not
    # added a second time.
    assert p.arr == 300_000.0
    assert p.rolled_up == {"rollup-facility": "rollup-system"}


def test_regions_are_deduplicated(db) -> None:
    """Three touched accounts, two sharing a region -- `regions` lists
    each region once, sorted, never once per account."""
    _touch_theme(
        db,
        theme_slug="theme-regions",
        theme_title="Regions theme",
        fact_type="delivery",
        account_slugs=["region-acct-1", "region-acct-2", "region-acct-3"],
        source_slug="src-regions",
    )
    region_by_note_slug = {
        "region-acct-1": "Midwest",
        "region-acct-2": "Midwest",
        "region-acct-3": "Northeast",
    }

    ranked = priorities.compute_theme_priorities(
        db,
        themes=[_theme("theme-regions", "Regions theme", "new")],
        since=SINCE,
        as_of=AS_OF,
        arr_by_note_slug={},
        region_by_note_slug=region_by_note_slug,
    )

    assert len(ranked) == 1
    assert ranked[0].regions == ["Midwest", "Northeast"]


def test_theme_with_no_claims_in_window_is_dropped(db) -> None:
    """A living theme that this window's `_touch_theme` never touches at
    all contributes nothing -- "needs attention" is about this window,
    not the theme's whole history."""
    _make_bare_theme_note(db, "theme-untouched", "Untouched theme")

    ranked = priorities.compute_theme_priorities(
        db,
        themes=[_theme("theme-untouched", "Untouched theme", "new")],
        since=SINCE,
        as_of=AS_OF,
        arr_by_note_slug={},
        region_by_note_slug={},
    )
    assert ranked == []


def _sample_priority() -> ThemePriority:
    return ThemePriority(
        theme_slug="theme-delivery",
        theme_name="Late deliveries",
        movement="falling",
        account_slugs=["mw-acct-1", "mw-acct-2", "mw-acct-3", "mw-acct-4"],
        account_count=4,
        arr=2_340_000.0,
        arr_unknown_accounts=0,
        regions=["Midwest"],
        rolled_up={},
    )


def test_add_priority_facts_registers_account_count_and_arr() -> None:
    """`add_priority_facts` puts both numbers -- exactly as the report
    would render them -- into the `NumberFacts` table, unconditionally
    (the numbers rule)."""
    p = _sample_priority()
    facts = NumberFacts()
    priorities.add_priority_facts(facts, [p])

    assert "4" in facts.renderings()
    assert "2,340,000" in facts.renderings()


def test_numbers_gate_accepts_the_generated_priority_line() -> None:
    """The target sentence shape ("late deliveries affect N accounts
    representing $X in annual revenue in the Midwest") is the acceptance
    test -- every numeral in the mechanical line this module generates
    must come straight out of the number table, so `check_numerals`
    passes it with zero violations."""
    p = _sample_priority()
    facts = NumberFacts()
    priorities.add_priority_facts(facts, [p])

    line = priorities.format_priority_line(p)
    assert check_numerals(line, facts) == []

    stats = priorities.format_priority_stats(p)
    assert check_numerals(stats, facts) == []


def test_movement_priority_rank_matches_task_order() -> None:
    """`new, rising, falling, resolved` (task text, verbatim) ranked in
    that order, ahead of `resolution proposed`."""
    assert (
        priorities.MOVEMENT_PRIORITY_RANK["new"]
        < priorities.MOVEMENT_PRIORITY_RANK["rising"]
        < priorities.MOVEMENT_PRIORITY_RANK["falling"]
        < priorities.MOVEMENT_PRIORITY_RANK["resolved"]
        < priorities.MOVEMENT_PRIORITY_RANK["resolution proposed"]
    )
