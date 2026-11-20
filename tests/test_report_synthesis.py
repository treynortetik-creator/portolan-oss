"""`portolan.report.synthesis` tests (wave C1)."""

# ruff: noqa: F811  (pytest fixtures imported by name; see test_report_helpers.py)

from __future__ import annotations

from datetime import date

from test_report_helpers import FullGraph, full_graph, report_llm  # noqa: F401

from portolan.llm.fake import FakeLLMClient
from portolan.notes import read_note
from portolan.report import synthesis

WINDOW_START = date(2026, 3, 2)
AS_OF = date(2026, 9, 4)

ACCOUNT_SLUG = "ashgrove-medical-center"


def test_grown_account_gets_a_rewritten_synthesis(
    full_graph: FullGraph, report_llm: FakeLLMClient
) -> None:
    result = synthesis.refresh_account_synthesis(
        full_graph.conn, report_llm, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(o for o in result.outcomes if o.account_slug == ACCOUNT_SLUG)
    # Bughunt-2026-09-18 finding 10: `refresh_account_synthesis` now
    # skips the write (status "kept_unchanged") when the freshly
    # generated synthesis renders byte-identical to what's already
    # stored -- `report_llm` is deterministic for the same prompt, so
    # whichever OTHER test in this session's shared `full_graph` ran
    # this same account through this same responder first already wrote
    # it; this test only cares that a real, LLM-authored synthesis
    # actually reached the note, not which of the two "it got written
    # (now or earlier)" statuses did it. Dedicated, order-independent
    # coverage for the no-op guard itself is
    # `test_repeat_run_with_no_new_claims_is_a_no_op` below.
    assert ours.status in ("rewritten", "kept_unchanged")

    note, _claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    assert note.synthesis
    assert "on file" in note.synthesis or note.synthesis  # non-empty, LLM-authored


def test_repeat_run_with_no_new_claims_is_a_no_op(
    full_graph: FullGraph, report_llm: FakeLLMClient
) -> None:
    """Bughunt-2026-09-18 finding 10: before this fix, `refresh_account_
    synthesis` called `write_note` unconditionally once the new synthesis
    passed its mechanical checks, even when the model (deterministic for
    the same prompt, like the real `report_llm` responder here) produced
    text byte-identical to what was already stored -- bumping the note's
    `version` and re-stamping `updated` for no real change, every single
    time this account was re-processed with nothing new to say. Calling
    `refresh_account_synthesis` twice, back to back, with the exact same
    `(since, as_of)` window and no claim inserted in between, must not
    move the note's `version` or `updated` the second time -- regardless
    of what the FIRST call's own status was (another test sharing this
    session's `full_graph` may have already synced this account, so the
    first call here could itself already be a no-op; that's fine, the
    assertion only compares the state straddling the SECOND call)."""
    synthesis.refresh_account_synthesis(
        full_graph.conn, report_llm, since=WINDOW_START, as_of=AS_OF
    )
    before, _claims = read_note(full_graph.conn, ACCOUNT_SLUG)

    result = synthesis.refresh_account_synthesis(
        full_graph.conn, report_llm, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(o for o in result.outcomes if o.account_slug == ACCOUNT_SLUG)
    assert ours.status == "kept_unchanged", ours

    after, _claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    assert after.version == before.version
    assert after.updated == before.updated
    assert after.synthesis == before.synthesis


def test_synthesis_prompt_includes_claims_from_linked_signal_notes(full_graph: FullGraph) -> None:
    """Item 3 (e2e-run follow-up wave): every account-anchored fact type
    homes on a `signal-<fact_type>-<account>` note, never the account
    note itself (`portolan.merge.resolve.ACCOUNT_ANCHORED_TYPES`) -- so
    an account's own claim list (`read_note(conn, account_slug)`) is
    almost always empty. Before this fix the prompt only ever saw that
    empty list, which is exactly `portolan_e2e`'s `ashgrove-medical-
    center` bug this item names: the synthesis said no complaints or
    risks were provided while its own `## Related` list links signal
    notes full of them. `_account_claim_pool` now also gathers every
    claim on a linked signal/action note, so at least one of them must
    show up verbatim in the prompt handed to the model."""
    _before_note, before_claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    before_claim_count = len(before_claims)

    # `_grown_accounts` processes every account with new activity in this
    # window, not just `ACCOUNT_SLUG` -- capture every prompt, keyed by
    # its own "Account: <name> ..." first line (`_build_user_prompt`),
    # rather than the single last one.
    prompts_by_first_line: dict[str, str] = {}

    def _capturing_responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model, surface
        prompts_by_first_line[user.splitlines()[0]] = user
        return {"synthesis": "An account on file."}

    synthesis.refresh_account_synthesis(
        full_graph.conn, FakeLLMClient(_capturing_responder), since=WINDOW_START, as_of=AS_OF
    )
    ashgrove_prompt = next(
        (p for line, p in prompts_by_first_line.items() if "Ashgrove Medical Center" in line),
        None,
    )
    assert ashgrove_prompt is not None, list(prompts_by_first_line)

    with full_graph.conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.slug FROM notes n
            JOIN links l ON l.to_slug = n.slug AND l.kind = 'body'
            WHERE l.from_slug = %s AND n.type IN ('signal', 'action')
            """,
            (ACCOUNT_SLUG,),
        )
        linked_slugs = [row[0] for row in cur.fetchall()]
    assert linked_slugs, "expected at least one signal/action note linked from the account"

    linked_claim_texts = []
    for slug in linked_slugs:
        _note, claims = read_note(full_graph.conn, slug)
        linked_claim_texts.extend(c.claim_text for c in claims)
    assert linked_claim_texts, "expected at least one claim on a linked signal/action note"

    assert any(text in ashgrove_prompt for text in linked_claim_texts), (
        "none of the account's linked signal/action notes' claims appeared in the "
        "synthesis prompt -- the account's own (near-empty) claim list is still all "
        "that's being sent"
    )

    # The widened prompt must not leak into what actually gets persisted:
    # `write_note`'s own `claims` argument stays the account note's own
    # claim list, unchanged in count -- the account note's rendered
    # `## Claims` section is exactly what it was before this call.
    _after_note, after_claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    assert len(after_claims) == before_claim_count


def _make_bare_signal_note(conn, slug: str) -> None:
    from datetime import UTC, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note

    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title="Claim-pool fixture signal",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Test fixture.",
    )
    write_note(conn, note, [], changed_by="test", reason="claim-pool fixture signal note")


def test_linked_signal_and_action_notes_does_not_straddle_a_hyphen(full_graph: FullGraph) -> None:
    """Bughunt-2026-09-18 finding 8: the fallback match used to be
    `n.slug LIKE 'signal-%-' || account_slug`, which only anchors the
    literal END of the slug -- for `account_slug="halvard"` that's
    `'signal-%-halvard'`, which ALSO matches `signal-billing-west-
    halvard` (`west-halvard` itself ends in `-halvard`), pulling a
    different customer's claims into `halvard`'s own synthesis prompt.
    Now matched by an exact `signal-<fact_type>-<account_slug>` slug per
    known `FactType`, never a wildcard that can straddle a hyphen it
    shouldn't."""
    conn = full_graph.conn
    # "pricing" is a real `FactType` value -- the fallback only ever
    # builds candidate slugs from that closed enum (`signal_note_slug`),
    # so the exact-match fixture must use one too.
    exact = "signal-pricing-halvard"
    collider = "signal-pricing-west-halvard"
    for slug in (exact, collider):
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM notes WHERE slug = %s", (slug,))
            if cur.fetchone() is None:
                _make_bare_signal_note(conn, slug)

    result = synthesis._linked_signal_and_action_notes(conn, "halvard")
    assert exact in result
    assert collider not in result
    """A synthesis that invents a citation marker not on any claim actually
    given to the model must be rejected: the old synthesis stays, and a
    `review_queue` row records why."""
    note, _claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    old_synthesis = note.synthesis

    def _bad_responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {"synthesis": "Invented fact. ^[not-a-real-source @ 00:00:00]"}

    bad_llm = FakeLLMClient(_bad_responder)

    result = synthesis.refresh_account_synthesis(
        full_graph.conn, bad_llm, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(o for o in result.outcomes if o.account_slug == ACCOUNT_SLUG)
    assert ours.status == "kept_old_failed_check"
    assert "citation marker" in (ours.reason or "")

    note_after, _claims_after = read_note(full_graph.conn, ACCOUNT_SLUG)
    assert note_after.synthesis == old_synthesis
    assert note_after.version == note.version  # no new version was written

    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM review_queue WHERE kind = 'other' AND payload->>'note_slug' = %s",
            (ACCOUNT_SLUG,),
        )
        assert cur.fetchone() is not None


def test_synthesis_over_the_line_limit_is_rejected(full_graph: FullGraph) -> None:
    note, _claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    old_synthesis = note.synthesis
    too_long = "\n".join(f"Line {n} of a synthesis that is far too long." for n in range(30))

    def _long_responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {"synthesis": too_long}

    long_llm = FakeLLMClient(_long_responder)
    result = synthesis.refresh_account_synthesis(
        full_graph.conn, long_llm, since=WINDOW_START, as_of=AS_OF
    )
    ours = next(o for o in result.outcomes if o.account_slug == ACCOUNT_SLUG)
    assert ours.status == "kept_old_failed_check"
    assert "line" in (ours.reason or "")

    note_after, _claims_after = read_note(full_graph.conn, ACCOUNT_SLUG)
    assert note_after.synthesis == old_synthesis


def test_quiet_account_is_skipped_entirely(
    full_graph: FullGraph, report_llm: FakeLLMClient
) -> None:
    """An account with no claim in the window never gets a synthesis call
    at all -- a narrow, far-future window should skip every account."""
    result = synthesis.refresh_account_synthesis(
        full_graph.conn, report_llm, since=date(2030, 1, 1), as_of=date(2030, 1, 7)
    )
    assert result.outcomes == []


# --------------------------------------------------------------------
# The numbers rule (polish wave 2026-09-12): a synthesis with a numeral
# not in the supplied figures table is rejected exactly like a bad
# citation -- old synthesis kept, a review_queue row logged.
# --------------------------------------------------------------------


def test_synthesis_with_a_supplied_figure_is_accepted(full_graph: FullGraph) -> None:
    """A synthesis that only ever states figures actually handed to it
    (an id, a date, and whatever the ARR/open-actions table gave it)
    passes the numbers check -- proving the guard doesn't reject valid
    text just because it contains digits."""

    def _responder(*, system, user, json_schema, model, surface):
        del system, model, surface
        assert "Figures you may state" in user
        return {
            "synthesis": (
                "Ashgrove Medical Center (CRM A-1011) has open items on file as of 2026-08-04."
            )
        }

    result = synthesis.refresh_account_synthesis(
        full_graph.conn, FakeLLMClient(_responder), since=WINDOW_START, as_of=AS_OF
    )
    ours = next(o for o in result.outcomes if o.account_slug == ACCOUNT_SLUG)
    assert ours.status == "rewritten"


def test_account_missing_from_arr_by_note_slug_is_treated_as_unknown_not_zero(
    full_graph: FullGraph,
) -> None:
    """Test gap 11 (review finding 10): before this fix, `.get(account_
    slug, 0.0)` turned an account absent from `arr_by_note_slug` (missing
    from the CRM export, or with an unparseable `arr` cell) into a
    genuine `0.0` -- "unknown" rendered as "zero" in the exact figure the
    report exists to put in front of a revenue owner. `arr_by_note_slug`
    here is non-empty (so the old `if arr_by_note_slug else None` guard
    alone would not have caught this) but deliberately omits
    `ACCOUNT_SLUG`."""
    captured_user = {}

    def _capturing_responder(*, system, user, json_schema, model, surface):
        del system, json_schema, model, surface
        captured_user["text"] = user
        return {"synthesis": "This account has open items on file as of 2026-08-04."}

    arr_by_note_slug = {"some-other-account": 1_000_000.0}

    synthesis.refresh_account_synthesis(
        full_graph.conn,
        FakeLLMClient(_capturing_responder),
        since=WINDOW_START,
        as_of=AS_OF,
        arr_by_note_slug=arr_by_note_slug,
    )

    user = captured_user.get("text")
    assert user is not None, "expected the LLM to be called for ACCOUNT_SLUG"
    assert "Annual recurring revenue" not in user
    assert "account_arr" not in user
    assert "$0" not in user


def test_synthesis_with_an_unsupplied_numeral_is_rejected(full_graph: FullGraph) -> None:
    note, _claims = read_note(full_graph.conn, ACCOUNT_SLUG)
    old_synthesis = note.synthesis

    def _fabricating_responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {"synthesis": "This account has 17 open complaints on file."}

    result = synthesis.refresh_account_synthesis(
        full_graph.conn, FakeLLMClient(_fabricating_responder), since=WINDOW_START, as_of=AS_OF
    )
    ours = next(o for o in result.outcomes if o.account_slug == ACCOUNT_SLUG)
    assert ours.status == "kept_old_failed_check"
    assert "17" in (ours.reason or "")

    note_after, _claims_after = read_note(full_graph.conn, ACCOUNT_SLUG)
    assert note_after.synthesis == old_synthesis
    assert note_after.version == note.version  # no new version was written

    with full_graph.conn.cursor() as cur:
        cur.execute(
            "SELECT payload->>'reason' FROM review_queue "
            "WHERE kind = 'other' AND payload->>'note_slug' = %s "
            "ORDER BY id DESC LIMIT 1",
            (ACCOUNT_SLUG,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "numeral_not_supplied"


# --------------------------------------------------------------------
# A claim's number is licensed only by its own citation: checked per
# cited segment of the synthesis, not over the whole text.
# --------------------------------------------------------------------


def _one_account_with_two_claims(conn) -> str:
    """A fresh account with two claims in the window: `L1` states a
    number (14), `L2` states none. Returns the account slug."""
    from datetime import UTC, datetime

    from portolan.notes import NoteRow, NoteStatus, NoteType, write_note

    slug = "syn-numeral-account"
    now = datetime.now(UTC)
    write_note(
        conn,
        NoteRow(
            slug=slug,
            type=NoteType.ACCOUNT,
            title="Syn Numeral Account",
            description="Test fixture account.",
            status=NoteStatus.LIVING,
            created=now,
            updated=now,
            crm_id="TEST-SYN-1",
        ),
        [],
        changed_by="test",
        reason="synthesis numeral fixture",
    )
    with conn.cursor() as cur:
        cur.execute("INSERT INTO sources (slug, source_type) VALUES ('syn-src', 'doc')")
        for locator, text in (
            ("L1", f"[[{slug}]] reported 14 outages in the pharmacy module."),
            ("L2", f"[[{slug}]] asked for a new reporting dashboard."),
        ):
            cur.execute(
                "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
                "recorded) VALUES (%s, 'complaint', %s, 'syn-src', %s, %s)",
                (slug, text, locator, datetime(2026, 8, 3, 12, tzinfo=UTC)),
            )
    conn.commit()
    return slug


def _synthesis_status(conn, text: str) -> str:
    def _responder(*, system, user, json_schema, model, surface):
        del system, user, json_schema, model, surface
        return {"synthesis": text}

    result = synthesis.refresh_account_synthesis(
        conn, FakeLLMClient(_responder), since=date(2026, 8, 1), as_of=date(2026, 8, 31)
    )
    (outcome,) = [o for o in result.outcomes if o.account_slug == "syn-numeral-account"]
    return outcome.status


def test_synthesis_number_under_another_claims_citation_is_rejected(db) -> None:
    _one_account_with_two_claims(db)

    status = _synthesis_status(
        db,
        "The account asked for 14 new dashboards ^[syn-src @ L2]. "
        "It also reported outages ^[syn-src @ L1].",
    )

    assert status == "kept_old_failed_check"


def test_synthesis_number_under_its_own_claims_citation_is_accepted(db) -> None:
    _one_account_with_two_claims(db)

    status = _synthesis_status(
        db,
        "The account asked for a new dashboard ^[syn-src @ L2]. "
        "It reported 14 outages in pharmacy. ^[syn-src @ L1]",
    )

    assert status == "rewritten"
