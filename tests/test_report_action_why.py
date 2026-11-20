"""Tests for `portolan.report.action_why` and the two behaviours it adds
to `portolan.report.researcher.confirm_action_card`: persisting the why,
and refusing to create a duplicate action when the card's own evidence
already lives on some other open action's note.
"""

from __future__ import annotations

from datetime import date

from portolan.extract.schema import FactType
from portolan.llm.fake import FakeLLMClient
from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.notes.persist import ensure_entity_note, read_note
from portolan.registry import Registry
from portolan.report.action_why import (
    build_why_context,
    find_duplicate_open_action,
    finding_plain_text,
    merge_duplicate_action,
    persist_action_why,
    why_summary_line,
)
from portolan.report.researcher import CURRENT_BRIEF_VERSION, confirm_action_card
from portolan.report.researcher_tools import record_finding
from portolan.search.embed import FakeEmbedder
from tests.merge_helpers import (
    ASHGROVE_MEDICAL,
    SAM_OKAFOR,
    build_action,
    build_fact,
    loaded_registry_fixture,  # noqa: F401
    make_source_row,
)

# Direct `confirm_action_card` calls pin `as_of`: the card fixtures carry a
# fixed `next_report` due date (2026-10-01), and Confirm's recheck measures
# that window from `as_of`. Left to the wall clock these tests fail from
# 2026-10-01 on (bug hunt 2026-09-22).
CARD_AS_OF = date(2026, 9, 20)


def _merge_complaint(
    db, registry: Registry, *, locator: str = "p1"
) -> tuple[int, str, str, str | None]:
    embedder = FakeEmbedder()
    llm = FakeLLMClient([])
    source = make_source_row(db, f"s-why-{locator}", source_date=date(2026, 8, 20))
    fact = build_fact(
        f"F-why-{locator}",
        FactType.COMPLAINT,
        "Ashgrove Medical Center received a late Tidewater T2 delivery.",
        ["Ashgrove Medical Center", "Tidewater T2"],
        locator=locator,
        quote="late delivery",
        speaker="Rob Tallis",
    )
    outcome = merge_fact(db, llm, embedder, registry, fact, source)
    assert outcome.status == "new"
    with db.cursor() as cur:
        cur.execute(
            "SELECT claim_text, quote, source_slug FROM claims WHERE id = %s", (outcome.claim_id,)
        )
        claim_text, quote, source_slug = cur.fetchone()
    return outcome.claim_id, source_slug, claim_text, quote


def _merge_pipeline_action(
    db, registry: Registry, *, what: str, locator: str = "p-pipeline"
) -> tuple[int, str, int]:
    """A `merge_action`-created (pipeline) action on Ashgrove Medical --
    returns `(action_id, note_slug, claim_id)`, `claim_id` being the
    synthetic claim `merge_action` writes onto the action's own note
    (the thing `find_duplicate_open_action` matches a card's evidence
    against)."""
    source = make_source_row(db, f"s-{locator}", source_date=date(2026, 8, 10))
    action = build_action(what, owner_mention=None, due=date(2026, 9, 1), locator=locator)
    outcome = merge_action(db, registry, action, source, entity_id=ASHGROVE_MEDICAL)
    with db.cursor() as cur:
        cur.execute("SELECT id FROM claims WHERE note_slug = %s", (outcome.note_slug,))
        claim_id = cur.fetchone()[0]
    return outcome.action_id, outcome.note_slug, claim_id


def _card_dict(
    *,
    account: str,
    owner: str,
    claim_id: int,
    claim_text,
    quote,
    source_slug,
    locator: str = "p1",
    what: str = "Follow up on the late delivery.",
    findings: list[int] | None = None,
    inference: list[dict] | None = None,
    confidence: str = "medium",
) -> dict:
    return {
        "run_id": "test-run",
        "candidate_slug": "test-candidate",
        "brief_version": CURRENT_BRIEF_VERSION,
        "account": account,
        "verdict": "action",
        "what": what,
        "evidence": [
            {
                "claim_id": claim_id,
                "text": claim_text,
                "quote": quote,
                "source_slug": source_slug,
                "locator": locator,
            }
        ],
        "findings": findings or [],
        "inference": inference or [],
        "owner": owner,
        "due": "2026-10-01",
        "due_rule": "next_report",
        "theme": None,
        "confidence": confidence,
        "confidence_refs": {"evidence_refs": [0], "finding_refs": []},
    }


# ---------------------------------------------------------------------
# persist_action_why / build_why_context / why_summary_line
# ---------------------------------------------------------------------


def test_persist_and_build_why_context_round_trips_evidence_findings_and_inference(
    loaded_registry: Registry, db
):
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry)
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    account_slug, account_title = account_note.slug, account_note.title
    finding_id = record_finding(
        db,
        run_id="test-run",
        candidate_slug="test-candidate",
        tool="absence_check",
        arguments={
            "account_slug": account_slug,
            "theme_slug": None,
            "fact_types": ["commitment", "response"],
            "after_date": "2026-08-25",
        },
        result={
            "count": 0,
            "latest_date": None,
            "claim_ids": [],
            "count_before": 1,
            "had_matches_before": True,
            "claim_ids_before": [claim_id],
        },
    )
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        findings=[finding_id],
        inference=[
            {
                "sentence": "No commitment or response claims after 2026-08-25.",
                "evidence_refs": [],
                "finding_refs": [0],
            }
        ],
    )

    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)
    assert action_id is not None

    why = build_why_context(db, action_id)
    assert why is not None
    assert why["confidence"] == "medium"
    assert why["confirmed_by"] == "Test Editor"
    assert len(why["inference"]) == 1
    assert why["inference"][0]["sentence"] == "No commitment or response claims after 2026-08-25."
    assert why["inference"][0]["findings"][0]["plain_text"] == (
        f"No commitment or response claims recorded for {account_title} after 2026-08-25; 1 before."
    )
    assert len(why["evidence"]) == 1
    assert why["evidence"][0]["claim_id"] == claim_id
    assert len(why["claims"]) == 1
    assert why["claims"][0].id == claim_id
    assert why["claims"][0].claim_text == claim_text

    summary = why_summary_line(db, action_id)
    assert summary == "No commitment or response claims after 2026-08-25."


def test_why_summary_line_falls_back_to_first_evidence_when_no_inference(
    loaded_registry: Registry, db
):
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry, locator="p2")
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        locator="p2",
    )

    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)

    assert why_summary_line(db, action_id) == claim_text


def test_persist_action_why_is_idempotent(loaded_registry: Registry, db):
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry, locator="p3")
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        locator="p3",
    )
    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)

    persist_action_why(db, action_id=action_id, card=card, editor="Test Editor")
    persist_action_why(db, action_id=action_id, card=card, editor="Test Editor")

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM action_why WHERE action_id = %s", (action_id,))
        assert cur.fetchone()[0] == 1


def test_a_card_confirmed_action_page_also_shows_the_notes_own_claims(
    loaded_registry: Registry, db
):
    """An action note from before cards stopped copying the action text keeps
    that claim; chat cites it (the live chat check of 2026-09-25, q6), so its
    `#claim-N` anchor must be on the page."""
    from starlette.requests import Request

    from portolan.web.note_view import build_note_context
    from portolan.web.templating import build_templates

    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry, locator="p4")
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        locator="p4",
    )
    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        (action_slug,) = cur.fetchone()
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "severity, sentiment, confidence) "
            "VALUES (%s, 'other', 'Follow up on the late delivery.', %s, 'p4', 'low', 0, 0.9) "
            "RETURNING id",
            (action_slug, source_slug),
        )
        (own_claim_id,) = cur.fetchone()
    db.commit()

    context = build_note_context(db, action_slug)
    assert [c.id for c in context["claims_to_show"]] == [claim_id, own_claim_id]
    request = Request(
        {"type": "http", "method": "GET", "path": f"/notes/{action_slug}", "headers": []}
    )
    html = build_templates().env.get_template("note.html").render(request=request, **context)
    assert f'id="claim-{claim_id}"' in html
    assert f'id="claim-{own_claim_id}"' in html


# ---------------------------------------------------------------------
# confirm_action_card no longer writes a redundant "other" claim.
# ---------------------------------------------------------------------


def test_confirm_action_card_writes_no_synthetic_claim_on_the_new_note(
    loaded_registry: Registry, db
):
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry, locator="p4")
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        locator="p4",
    )

    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)

    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (action_id,))
        (note_slug,) = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM claims WHERE note_slug = %s", (note_slug,))
        assert cur.fetchone()[0] == 0


# ---------------------------------------------------------------------
# No duplicates (build instructions, item 3): action-15/action-14, worked
# example. Both branches.
# ---------------------------------------------------------------------


def test_confirm_action_card_upgrades_an_existing_pipeline_action_instead_of_duplicating(
    loaded_registry: Registry, db
):
    pipeline_action_id, pipeline_note_slug, pipeline_claim_id = _merge_pipeline_action(
        db, loaded_registry, what="Review the request and follow up with available pricing options."
    )
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=pipeline_claim_id,
        claim_text="Review the request and follow up with available pricing options.",
        quote=None,
        source_slug=pipeline_note_slug,  # placeholder, overwritten below
        what="Follow up with Ashgrove Medical on pricing.",
    )
    # The evidence item must match the STORED claim byte-for-byte
    # (mechanical_checks) -- pull its real source_slug/locator back out.
    with db.cursor() as cur:
        cur.execute("SELECT source_slug, locator FROM claims WHERE id = %s", (pipeline_claim_id,))
        real_source_slug, real_locator = cur.fetchone()
    card["evidence"][0]["source_slug"] = real_source_slug
    card["evidence"][0]["locator"] = real_locator

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (before_count,) = cur.fetchone()

    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)

    assert action_id == pipeline_action_id
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (after_count,) = cur.fetchone()
    assert after_count == before_count  # no new action row

    with db.cursor() as cur:
        cur.execute(
            "SELECT a.due, a.note_slug, owner.crm_id FROM actions a "
            "JOIN registry_entities owner ON owner.id = a.owner_entity_id WHERE a.id = %s",
            (action_id,),
        )
        due, note_slug, owner_crm_id = cur.fetchone()
    assert due == date(2026, 10, 1)  # from the card
    assert note_slug == pipeline_note_slug  # slug kept
    assert owner_crm_id == SAM_OKAFOR

    why = build_why_context(db, action_id)
    assert why is not None
    assert why["evidence"][0]["claim_id"] == pipeline_claim_id


def test_confirm_action_card_creates_a_new_action_when_no_duplicate_exists(
    loaded_registry: Registry, db
):
    claim_id, source_slug, claim_text, quote = _merge_complaint(db, loaded_registry, locator="p5")
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug
    card = _card_dict(
        account=account_slug,
        owner=SAM_OKAFOR,
        claim_id=claim_id,
        claim_text=claim_text,
        quote=quote,
        source_slug=source_slug,
        locator="p5",
    )

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (before_count,) = cur.fetchone()

    action_id = confirm_action_card(db, card=card, editor="Test Editor", as_of=CARD_AS_OF)

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM actions")
        (after_count,) = cur.fetchone()
    assert after_count == before_count + 1
    assert build_why_context(db, action_id) is not None


def test_find_duplicate_open_action_excludes_a_given_action_id(loaded_registry: Registry, db):
    pipeline_action_id, _pipeline_note_slug, pipeline_claim_id = _merge_pipeline_action(
        db, loaded_registry, what="Send a written root-cause summary.", locator="p-dup1"
    )
    account_slug = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL).slug

    found = find_duplicate_open_action(
        db, account_note_slug=account_slug, evidence_claim_ids=[pipeline_claim_id]
    )
    assert found == pipeline_action_id

    excluded = find_duplicate_open_action(
        db,
        account_note_slug=account_slug,
        evidence_claim_ids=[pipeline_claim_id],
        exclude_action_id=pipeline_action_id,
    )
    assert excluded is None


# ---------------------------------------------------------------------
# merge_duplicate_action -- the backfill script's own merge, and its
# idempotency.
# ---------------------------------------------------------------------


def test_merge_duplicate_action_drops_the_newer_and_archives_its_note(
    loaded_registry: Registry, db
):
    older_id, older_slug, _older_claim = _merge_pipeline_action(
        db, loaded_registry, what="Review pricing options for Ashgrove Medical.", locator="p-older"
    )
    newer_id, newer_slug, _newer_claim = _merge_pipeline_action(
        db,
        loaded_registry,
        what="Send a written root-cause summary for the late Tidewater delivery.",
        locator="p-newer",
    )
    assert older_id < newer_id

    card = {"owner": SAM_OKAFOR, "due": "2026-11-01"}
    changed = merge_duplicate_action(
        db, keep_action_id=older_id, drop_action_id=newer_id, card=card, editor="backfill test"
    )
    assert changed is True

    with db.cursor() as cur:
        cur.execute("SELECT status, due FROM actions WHERE id = %s", (newer_id,))
        drop_status, _drop_due = cur.fetchone()
        cur.execute("SELECT status, due FROM actions WHERE id = %s", (older_id,))
        keep_status, keep_due = cur.fetchone()
    assert drop_status == "dropped"
    assert keep_status == "open"
    assert keep_due == date(2026, 11, 1)

    note, _claims = read_note(db, newer_slug)
    assert note.status.value == "archived"
    older_note, _older_note_claims = read_note(db, older_slug)
    assert older_note.status.value == "living"

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM changelog WHERE kind = 'action_duplicate_merged'")
        (changelog_count,) = cur.fetchone()
    assert changelog_count == 1

    # Idempotent: a second call changes nothing further, no second
    # changelog row.
    changed_again = merge_duplicate_action(
        db, keep_action_id=older_id, drop_action_id=newer_id, card=card, editor="backfill test"
    )
    assert changed_again is False
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM changelog WHERE kind = 'action_duplicate_merged'")
        (changelog_count_again,) = cur.fetchone()
    assert changelog_count_again == 1


# ---------------------------------------------------------------------
# finding_plain_text -- one branch per known tool, numbers straight from
# the stored result.
# ---------------------------------------------------------------------


def test_finding_plain_text_absence_check_zero_after(loaded_registry: Registry, db):
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    text = finding_plain_text(
        db,
        {
            "tool": "absence_check",
            "arguments": {
                "account_slug": account_note.slug,
                "fact_types": ["commitment", "response"],
                "after_date": "2026-08-25",
            },
            "result": {"count": 0, "latest_date": None, "count_before": 3},
        },
    )
    assert text == (
        f"No commitment or response claims recorded for {account_note.title} "
        "after 2026-08-25; 3 before."
    )


def test_finding_plain_text_absence_check_nonzero_after(loaded_registry: Registry, db):
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    text = finding_plain_text(
        db,
        {
            "tool": "absence_check",
            "arguments": {
                "account_slug": account_note.slug,
                "fact_types": ["complaint"],
                "after_date": "2026-08-01",
            },
            "result": {"count": 2, "latest_date": "2026-08-15", "count_before": 0},
        },
    )
    assert text == (
        f"2 complaint claims recorded for {account_note.title} after 2026-08-01 "
        "(latest 2026-08-15); 0 before."
    )


def test_finding_plain_text_count_by(loaded_registry: Registry, db):
    text = finding_plain_text(
        db,
        {
            "tool": "count_by",
            "arguments": {"fact_type": "complaint", "group_by": "account"},
            "result": {
                "groups": [{"group": "ashgrove-medical", "count": 3, "arr": 50000.0}],
                "total": 3,
            },
        },
    )
    assert text == "3 complaint claim(s) by account: ashgrove-medical: 3."


def test_finding_plain_text_timeline(loaded_registry: Registry, db):
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    text = finding_plain_text(
        db,
        {
            "tool": "timeline",
            "arguments": {"account_slug": account_note.slug},
            "result": {
                "count": 2,
                "entries": [{"date": "2026-07-01"}, {"date": "2026-08-01"}],
            },
        },
    )
    assert text == f"2 claim(s) for {account_note.title}, 2026-07-01 to 2026-08-01."


def test_finding_plain_text_renewal(loaded_registry: Registry, db):
    account_note = ensure_entity_note(db, loaded_registry, ASHGROVE_MEDICAL)
    text = finding_plain_text(
        db,
        {
            "tool": "renewal",
            "arguments": {"account_slug": account_note.slug},
            "result": {
                "close_date": "2027-01-15",
                "arr": 125000,
                "stage": "negotiation",
                "renewal_claims": [{"claim_id": 1}],
            },
        },
    )
    assert text == (
        f"Renewal on file for {account_note.title}: close 2027-01-15, stage negotiation, "
        "ARR $125,000; 1 claim(s) mention a renewal date."
    )


def test_finding_plain_text_unknown_tool_falls_back_but_never_raises(db):
    text = finding_plain_text(db, {"tool": "mystery_tool", "arguments": {}, "result": {"x": 1}})
    assert "mystery_tool" in text
