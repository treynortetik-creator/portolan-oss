"""`portolan.report.waves` tests (theme review fix wave, 2026-09-13,
"Wave-based themes" recommendation, `01-spec/reviews/review-theme-fix-
wave.md`).

Unlike `test_report_themes.py`, waves need no theme machinery at all
(no `theme_members`, no theme note) -- a wave is a plain `GROUP BY
fact_type, product_family, week` over `claims`, so these tests use the
plain, fresh `db` fixture with hand-built accounts/products/signals
rather than `full_graph`'s whole-corpus session fixture.
"""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml
from test_report_helpers import FullGraph, full_graph  # noqa: F401

from portolan.notes import NoteRow, NoteStatus, NoteType, write_note
from portolan.report import waves

TRUTH_TABLE_PATH = Path(__file__).resolve().parent.parent / "corpus" / "truth-table.yaml"

# Short truth-table alias -> the full account note slug `compute_waves`
# reports (`slugify(canonical_name)`, same mapping style
# `tests/test_report_themes.py`'s own `REQUIRED_ACCOUNTS` already uses).
_ACCOUNT_SLUG = {
    "northlake-general": "northlake-general-hospital",
    "cobalt-ridge": "cobalt-ridge-medical-center",
    "larkspur-north": "larkspur-north-hospital",
    "ashgrove-riverside": "ashgrove-riverside-hospital",
    "meadowbrook": "meadowbrook-community-hospital",
    "ashgrove-medical": "ashgrove-medical-center",
    "larkspur-memorial": "larkspur-memorial-hospital",
}


def _make_account_note(conn, slug: str) -> None:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.ACCOUNT,
        title=slug,
        description="Test fixture account.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id=f"TEST-{slug}",
    )
    write_note(conn, note, [], changed_by="test", reason="wave test fixture account")


def _make_product_note(conn, slug: str) -> None:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.PRODUCT,
        title=slug,
        description="Test fixture product.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(conn, note, [], changed_by="test", reason="wave test fixture product")


def _make_signal_note(conn, slug: str) -> None:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title=slug,
        description="Wave test fixture signal note.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
    )
    write_note(conn, note, [], changed_by="test", reason="wave test fixture signal")


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


def _link_parent_child(conn, parent_slug: str, child_slugs: list[str]) -> None:
    """Should-fix 3 (`01-spec/reviews/review-theme-fix-wave-3.md` finding
    3): a synthetic `registry_entities` parent/child chain, hand-built
    on the plain `db` fixture -- `_top_level_accounts`
    (`portolan.report.themes`) only ever reads `kind = 'account'`,
    `note_slug`, `parent_id`, so this is everything a real CRM sync would
    have written for this shape, without needing the real corpus's own
    Wrenfield rows or `full_graph`'s session-scoped fixture at all."""
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


def test_wave_forms_when_min_accounts_share_fact_type_and_product_family(db) -> None:
    """The basic case: three distinct accounts, all `pricing` claims about
    the same product family, in the same calendar week -- a wave, with
    the right account count, account list, and linked ARR (each
    account's own ARR summed, `_linked_arr`'s usual roll-up)."""
    _insert_source(db, "wave-source")
    _make_product_note(db, "acme-widget")
    for tag in ("alpha", "beta", "gamma"):
        _make_account_note(db, f"wave-account-{tag}")
        signal_slug = f"signal-pricing-wave-account-{tag}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[wave-account-{tag}]] pushed back on the price of [[acme-widget]] this week.",
            "wave-source",
            f"wave#{tag}",
            date(2026, 8, 24) + timedelta(days=1),
        )

    arr_by_note_slug = {
        "wave-account-alpha": 100_000.0,
        "wave-account-beta": 200_000.0,
        "wave-account-gamma": 50_000.0,
    }
    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug=arr_by_note_slug
    )

    assert len(feed.waves) == 1
    wave = feed.waves[0]
    assert wave.week_of == date(2026, 8, 24)
    assert wave.fact_type == "pricing"
    assert wave.product_family == "acme"
    assert wave.account_slugs == [
        "wave-account-alpha",
        "wave-account-beta",
        "wave-account-gamma",
    ]
    assert wave.linked_arr == 350_000.0
    assert len(wave.claim_ids) == 3


def test_wave_all_signals_is_not_capped_at_the_representative_quote_limit(db) -> None:
    """Weekly-report wave 2: `WaveRow.all_signals` must list EVERY claim
    in the wave -- unlike `WaveRow.quotes` (`_representative_quotes`'s
    own `_REPRESENTATIVE_QUOTES` cap, 3), which stays a short preview.
    Four accounts here (one more than the cap) proves the fold isn't
    silently truncated the same way the old preview was."""
    _insert_source(db, "wave-source-2")
    _make_product_note(db, "beta-widget")
    tags = ("alpha", "beta", "gamma", "delta")
    claim_ids = []
    for tag in tags:
        _make_account_note(db, f"wave2-account-{tag}")
        signal_slug = f"signal-risk-wave2-account-{tag}"
        _make_signal_note(db, signal_slug)
        claim_id = _insert_claim(
            db,
            signal_slug,
            "risk",
            f"[[wave2-account-{tag}]] flagged a risk with [[beta-widget]] this week.",
            "wave-source-2",
            f"wave2#{tag}",
            date(2026, 8, 24) + timedelta(days=1),
        )
        claim_ids.append(claim_id)

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )

    assert len(feed.waves) == 1
    wave = feed.waves[0]
    assert len(wave.claim_ids) == 4
    assert len(wave.quotes) <= 3  # the old preview cap, still in force for `quotes`
    assert len(wave.all_signals) == 4  # NOT capped
    assert {s.claim_id for s in wave.all_signals} == set(claim_ids)
    assert all(s.fact_type == "risk" for s in wave.all_signals)
    assert {s.account_slug for s in wave.all_signals} == {f"wave2-account-{t}" for t in tags}
    assert {s.note_slug for s in wave.all_signals} == {
        f"signal-risk-wave2-account-{t}" for t in tags
    }


def test_wave_does_not_form_below_the_account_floor(db) -> None:
    """Two accounts sharing a fact type and product family in the same
    week is the ordinary background rate, not a pattern -- `WAVE_MIN_
    ACCOUNTS` (3) is a floor, not a suggestion."""
    _insert_source(db, "wave-floor-source")
    _make_product_note(db, "acme-widget")
    for tag in ("alpha", "beta"):
        _make_account_note(db, f"wave-floor-account-{tag}")
        signal_slug = f"signal-pricing-wave-floor-account-{tag}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[wave-floor-account-{tag}]] pushed back on [[acme-widget]] pricing.",
            "wave-floor-source",
            f"wave-floor#{tag}",
            date(2026, 8, 24) + timedelta(days=1),
        )

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    assert feed.waves == []


def test_wave_requires_the_same_product_family_not_just_fact_type_and_week(db) -> None:
    """Should-fix per task item 6 ("sharing `fact_type` AND product
    family"): three accounts complaining about pricing the same week does
    NOT form a wave if they're not about the same product -- a claim
    about a genuinely different product correctly does not join, the same
    discipline that keeps a wave from being "any 3 accounts, same week,
    same fact type" regardless of topic (which the review's own
    `pricing_pushback` example shows is exactly the wrong thing to
    loosen: joining Cobalt Ridge's unrelated HalvardCare invoice dispute
    to Larkspur/Anchor Bay's Tidewater volume-discount pushback)."""
    _insert_source(db, "wave-family-source")
    _make_product_note(db, "acme-widget")
    _make_product_note(db, "zenith-gadget")

    for tag, product in (("alpha", "acme-widget"), ("beta", "acme-widget")):
        _make_account_note(db, f"wave-family-account-{tag}")
        signal_slug = f"signal-pricing-wave-family-account-{tag}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[wave-family-account-{tag}]] pushed back on [[{product}]] pricing.",
            "wave-family-source",
            f"wave-family#{tag}",
            date(2026, 8, 24) + timedelta(days=1),
        )
    # A third account, same week, same fact type -- but a DIFFERENT product.
    _make_account_note(db, "wave-family-account-gamma")
    _make_signal_note(db, "signal-pricing-wave-family-account-gamma")
    _insert_claim(
        db,
        "signal-pricing-wave-family-account-gamma",
        "pricing",
        "[[wave-family-account-gamma]] pushed back on [[zenith-gadget]] pricing.",
        "wave-family-source",
        "wave-family#gamma",
        date(2026, 8, 24) + timedelta(days=1),
    )

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    assert feed.waves == []  # acme-widget group has only 2 accounts; zenith-gadget has 1


def test_wave_excludes_a_claim_with_no_resolvable_account(db) -> None:
    """A claim with no `[[account]]` mention resolvable to a real account
    note contributes to no wave -- `home_account_slug` returning `None`
    must not crash or silently count toward the floor."""
    _insert_source(db, "wave-noaccount-source")
    _make_product_note(db, "acme-widget")
    for tag in ("alpha", "beta"):
        _make_account_note(db, f"wave-noaccount-account-{tag}")
        signal_slug = f"signal-pricing-wave-noaccount-{tag}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[wave-noaccount-account-{tag}]] pushed back on [[acme-widget]] pricing.",
            "wave-noaccount-source",
            f"wave-noaccount#{tag}",
            date(2026, 8, 24) + timedelta(days=1),
        )
    # A third claim, same week/fact_type/product, but no resolvable account.
    orphan_signal_slug = "signal-pricing-wave-noaccount-orphan"
    _make_signal_note(db, orphan_signal_slug)
    _insert_claim(
        db,
        orphan_signal_slug,
        "pricing",
        "Someone pushed back on [[acme-widget]] pricing with no account named.",
        "wave-noaccount-source",
        "wave-noaccount#orphan",
        date(2026, 8, 24) + timedelta(days=1),
    )

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    assert feed.waves == []  # only 2 real accounts -- the orphan claim doesn't count


def test_wave_is_bounded_to_the_window(db) -> None:
    """Claims outside `[since, as_of]` don't count toward the floor, even
    if they'd otherwise complete the group -- a wave is a THIS-WEEK
    phenomenon, not an accumulated membership the way a theme is."""
    _insert_source(db, "wave-window-source")
    _make_product_note(db, "acme-widget")
    for tag in ("alpha", "beta"):
        _make_account_note(db, f"wave-window-account-{tag}")
        signal_slug = f"signal-pricing-wave-window-{tag}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[wave-window-account-{tag}]] pushed back on [[acme-widget]] pricing.",
            "wave-window-source",
            f"wave-window#{tag}",
            date(2026, 8, 24) + timedelta(days=1),
        )
    # Third account, same week, but its claim is OUTSIDE the query window.
    _make_account_note(db, "wave-window-account-gamma")
    _make_signal_note(db, "signal-pricing-wave-window-gamma")
    _insert_claim(
        db,
        "signal-pricing-wave-window-gamma",
        "pricing",
        "[[wave-window-account-gamma]] pushed back on [[acme-widget]] pricing.",
        "wave-window-source",
        "wave-window#gamma",
        date(2026, 8, 24) + timedelta(days=1),
    )

    narrow_feed = waves.compute_waves(
        db, since=date(2026, 9, 1), as_of=date(2026, 9, 7), arr_by_note_slug={}
    )
    assert narrow_feed.waves == []

    wide_feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    assert len(wide_feed.waves) == 1
    assert len(wide_feed.waves[0].account_slugs) == 3


def test_real_corpus_pricing_pushback_does_not_clear_the_wave_floor(
    full_graph: FullGraph,
) -> None:
    """`corpus/truth-table.yaml`'s own `waves` note (v1.2): the corpus's
    `pricing_pushback` scenario does NOT actually form a wave once
    product family is required -- Cobalt Ridge's own claim (F52) resolves
    to `halvardcare`, a genuinely different product from Larkspur/Anchor
    Bay's `tidewater` volume-discount pushback (F53/F54), so only 2
    accounts share `(pricing, tidewater, week of 2026-08-24)` -- one
    short of `WAVE_MIN_ACCOUNTS`. This is the mechanically honest answer
    for this corpus's own data, not a defect: a real customer whose three
    accounts genuinely pushed back on the SAME product's pricing the same
    week would clear the floor without any code change.

    **Should-fix 3** (`01-spec/reviews/review-theme-fix-wave-3.md` finding
    3, the "cheaper" half of its fix, kept alongside the roll-up test's
    own move off `full_graph`): scoped to `tidewater` specifically, not
    every pricing wave regardless of product family -- `full_graph` is a
    SESSION-scoped fixture (`test_report_helpers.py`), so any OTHER test
    anywhere in the suite that plants a `pricing` wave for a DIFFERENT
    product family in this same `(2026-08-24, 2026-08-30)` window would
    otherwise make this assertion depend on which other tests happened to
    run first -- exactly the fragility this review finding is about,
    stated as a permanent guard rather than something to prove once."""
    conn = full_graph.conn
    feed = waves.compute_waves(
        conn, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    tidewater_pricing_waves = [
        w for w in feed.waves if w.fact_type == "pricing" and w.product_family == "tidewater"
    ]
    assert tidewater_pricing_waves == [], tidewater_pricing_waves


def test_real_corpus_waves_match_the_truth_tables_measured_waves(full_graph: FullGraph) -> None:
    """Should-fix 10 (`01-spec/reviews/review-theme-fix-wave-2.md` finding
    10, "the truth table's new waves: block records only what does not
    happen"): the bench that checks the three POSITIVE entries now
    recorded alongside the negative `pricing_week_of_2026_08_24` one --
    each named wave's own accounts and facts must actually appear in a
    real `compute_waves` run over the whole corpus window. `>=`, not
    `==`, for the account/fact counts: a real wave may also pick up a few
    generator-synthesized filler claims from the same account/product/
    week that carry no F-number of their own (the truth table's own
    comment on this block explains why)."""
    conn = full_graph.conn
    with TRUTH_TABLE_PATH.open(encoding="utf-8") as fh:
        truth = yaml.safe_load(fh)
    truth_waves = truth["expectations"]["report"]["waves"]

    feed = waves.compute_waves(
        conn, since=date(2026, 3, 1), as_of=date(2026, 9, 10), arr_by_note_slug={}
    )
    by_key = {(w.week_of, w.fact_type, w.product_family): w for w in feed.waves}

    checked = 0
    for key, entry in truth_waves.items():
        if key == "expected_min_accounts" or "facts" not in entry:
            continue  # the negative pricing_week_of_2026_08_24 entry has no `facts`
        checked += 1
        # Parse "..._week_of_2026_08_10" -> date(2026, 8, 10).
        year, month, day = key.split("_week_of_")[1].split("_")
        week_of = date(int(year), int(month), int(day))
        fact_type = _fact_type_for(key)
        wave = by_key.get((week_of, fact_type, entry["product_family"]))
        assert wave is not None, f"{key}: no wave found for {week_of} / {entry['product_family']}"
        expected_accounts = {_ACCOUNT_SLUG[a] for a in entry["accounts"]}
        assert expected_accounts <= set(wave.account_slugs), (
            f"{key}: expected accounts {expected_accounts} not all present in {wave.account_slugs}"
        )
        assert len(wave.account_slugs) >= len(entry["accounts"])
        assert len(wave.claim_ids) >= len(entry["facts"])
    assert checked == 3  # complaint_08_10, complaint_08_17, praise_08_31


def _fact_type_for(key: str) -> str:
    if key.startswith("complaint_"):
        return "complaint"
    if key.startswith("praise_"):
        return "praise"
    raise ValueError(f"unrecognised waves: key {key!r} -- add its fact_type here")


def test_wave_floor_counts_top_level_accounts_not_a_parent_and_its_own_children(db) -> None:
    """Should-fix 4 (`01-spec/reviews/review-theme-fix-wave-2.md` finding
    4, test gap 2): "three sibling hospitals under one parent currently
    form a wave and report the parent's ARR once" -- the bug this test
    pins closed. Three accounts, same fact type/product family/week, but
    one is the parent of the other two -- ONE customer's signal, not
    three -- so `_top_level_accounts` collapses all three to the parent
    alone, one account, one short of `WAVE_MIN_ACCOUNTS`.

    **Should-fix 3** (`01-spec/reviews/review-theme-fix-wave-3.md` finding
    3): this used to run on `full_graph` against the real corpus's own
    Wrenfield Health parent/child rows, planting six `pricing` claims
    into that SESSION-scoped fixture -- which
    `test_real_corpus_pricing_pushback_does_not_clear_the_wave_floor`
    (this file, above) then broke when it happened to run AFTER this
    test, because its own assertion (`pricing_waves == []`, not scoped to
    a product family) started seeing this test's `wave`-family pricing
    wave too. Rebuilt here on the plain, function-scoped `db` fixture
    with a hand-built `registry_entities` parent/child chain
    (`_link_parent_child`) instead of the real corpus's own accounts --
    own schema, nothing shared with any other test, so which order tests
    in this file run in cannot change either test's answer. Proven by
    running the two in both orders (recorded in this wave's own commit
    message, not re-asserted here as a third test -- pytest's own
    `-p no:randomly`-free default order is file-definition order, and
    `pytest tests/test_report_waves.py::test_wave_floor_counts_top_level_
    accounts_not_a_parent_and_its_own_children tests/test_report_waves.py
    ::test_real_corpus_pricing_pushback_does_not_clear_the_wave_floor`
    (explicit order) and the reverse both pass now that neither touches
    the other's data)."""
    _insert_source(db, "wave-rollup-source")
    _make_product_note(db, "wave-rollup-widget")
    _make_account_note(db, "wave-rollup-parent")
    _make_account_note(db, "wave-rollup-child-a")
    _make_account_note(db, "wave-rollup-child-b")
    _link_parent_child(db, "wave-rollup-parent", ["wave-rollup-child-a", "wave-rollup-child-b"])

    for i, account in enumerate(
        ("wave-rollup-parent", "wave-rollup-child-a", "wave-rollup-child-b")
    ):
        signal_slug = f"signal-wave-rollup-{i}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[{account}]] pushed back on [[wave-rollup-widget]] pricing this week.",
            "wave-rollup-source",
            f"wave-rollup#{i}",
            date(2026, 8, 24) + timedelta(days=1),
        )

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    rollup_waves = [w for w in feed.waves if w.product_family == "wave"]
    assert rollup_waves == [], rollup_waves

    # Confirmed independently: three GENUINELY unrelated accounts (no
    # shared parent) with the same shape DO clear the floor -- the fix is
    # about roll-up specifically, not about the floor breaking generally.
    for i, account in enumerate(
        ("wave-rollup-stranger-a", "wave-rollup-stranger-b", "wave-rollup-stranger-c")
    ):
        _make_account_note(db, account)
        signal_slug = f"signal-wave-rollup-control-{i}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[{account}]] pushed back on [[wave-rollup-widget]] pricing this week too.",
            "wave-rollup-source",
            f"wave-rollup-control#{i}",
            date(2026, 8, 24) + timedelta(days=1),
        )
    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    control_waves = [w for w in feed.waves if w.product_family == "wave"]
    assert len(control_waves) == 1
    assert len(control_waves[0].account_slugs) == 6  # all six leaf accounts still listed
    assert control_waves[0].rolled_up == {
        "wave-rollup-child-a": "wave-rollup-parent",
        "wave-rollup-child-b": "wave-rollup-parent",
    }


def test_wave_floor_counts_three_siblings_under_an_absent_parent_as_one_customer(db) -> None:
    """Three hospitals of one health system raise the same claim; the
    parent itself raises nothing. One customer, so no wave."""
    _insert_source(db, "wave-sibling-source")
    _make_product_note(db, "wavesib-widget")
    _make_account_note(db, "wave-sibling-parent")
    siblings = ("wave-sibling-a", "wave-sibling-b", "wave-sibling-c")
    for account in siblings:
        _make_account_note(db, account)
    _link_parent_child(db, "wave-sibling-parent", list(siblings))

    for i, account in enumerate(siblings):
        signal_slug = f"signal-wave-sibling-{i}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[{account}]] pushed back on [[wavesib-widget]] pricing this week.",
            "wave-sibling-source",
            f"wave-sibling#{i}",
            date(2026, 8, 24) + timedelta(days=1),
        )

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    assert [w for w in feed.waves if w.product_family == "wavesib"] == []


def test_wave_floor_rolls_siblings_up_to_a_parent_that_has_no_note(db) -> None:
    """A parent account nobody has mentioned has a registry row but no
    note; its hospitals are still one customer."""
    _insert_source(db, "wave-nonote-source")
    _make_product_note(db, "wavenonote-widget")
    siblings = ("wave-nonote-a", "wave-nonote-b", "wave-nonote-c")
    for account in siblings:
        _make_account_note(db, account)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name) "
            "VALUES ('account', 'Unmentioned Health System') RETURNING id"
        )
        (parent_id,) = cur.fetchone()
        for account in siblings:
            cur.execute(
                "INSERT INTO registry_entities (kind, canonical_name, note_slug, parent_id) "
                "VALUES ('account', %s, %s, %s)",
                (account, account, parent_id),
            )
    db.commit()
    for i, account in enumerate(siblings):
        signal_slug = f"signal-wave-nonote-{i}"
        _make_signal_note(db, signal_slug)
        _insert_claim(
            db,
            signal_slug,
            "pricing",
            f"[[{account}]] pushed back on [[wavenonote-widget]] pricing this week.",
            "wave-nonote-source",
            f"wave-nonote#{i}",
            date(2026, 8, 24) + timedelta(days=1),
        )

    feed = waves.compute_waves(
        db, since=date(2026, 8, 24), as_of=date(2026, 8, 30), arr_by_note_slug={}
    )
    assert [w for w in feed.waves if w.product_family == "wavenonote"] == []
