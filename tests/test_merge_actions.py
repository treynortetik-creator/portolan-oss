"""Tests for `portolan.merge.merge_action` (`01-spec/build-contracts.md`
"Merge": "dedup against open actions on the same note by difflib >= 0.7").

Run with this module's own test database:
    createdb portolan_test_merge
    DATABASE_URL=postgresql://localhost/portolan_test_merge \
        uv run pytest tests/test_merge_actions.py -v
"""

from __future__ import annotations

from datetime import date

from portolan.merge.actions import merge_action
from portolan.notes import read_note
from portolan.registry import Registry
from portolan.registry.owners import confirm_owner_proposal
from tests.merge_helpers import (  # noqa: F401
    ASHGROVE,
    SAM_OKAFOR,
    build_action,
    loaded_registry_fixture,
    make_source_row,
)


def test_merge_action_creates_a_note_linked_to_the_entity(loaded_registry: Registry, db):
    source = make_source_row(db, "s-call-013", source_type="call", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention="Sam Okafor",
        due=date(2026, 8, 28),
        locator="00:31:10",
    )

    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    assert outcome.status == "new"
    assert outcome.note_slug is not None
    assert outcome.note_slug.startswith("action-")

    note, claims = read_note(db, outcome.note_slug)
    assert note.type.value == "action"
    assert "ashgrove-health" in note.related
    assert len(claims) == 1
    assert claims[0].locator == "00:31:10"

    with db.cursor() as cur:
        cur.execute(
            "SELECT to_slug FROM links WHERE from_slug = %s AND kind = 'body'", ("ashgrove-health",)
        )
        linked = {row[0] for row in cur.fetchall()}
    assert outcome.note_slug in linked


def test_merge_action_dedups_near_duplicate_open_actions(loaded_registry: Registry, db):
    source = make_source_row(db, "s-call-013", source_type="call", source_date=date(2026, 8, 20))
    first = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention="Sam Okafor",
        due=date(2026, 8, 28),
        locator="00:31:10",
    )
    outcome1 = merge_action(db, loaded_registry, first, source, entity_id=ASHGROVE)
    assert outcome1.status == "new"

    second = build_action(
        "Send Ashgrove Health a written root-cause summary for the deliveries",
        owner_mention="Sam Okafor",
        due=date(2026, 8, 28),
        locator="00:32:00",
    )
    outcome2 = merge_action(db, loaded_registry, second, source, entity_id=ASHGROVE)

    assert outcome2.status == "duplicate"
    assert outcome2.action_id == outcome1.action_id
    assert outcome2.note_slug == outcome1.note_slug

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM actions")
        assert cur.fetchone()[0] == 1


def test_merge_action_dedups_when_no_account_resolves(loaded_registry: Registry, db):
    """Bug hunt 2026-09-18, finding 4: `entity_id=None` (a doc, RSS item,
    or ticket export with no participant email, per `merge.pipeline.
    _primary_account_id`) used to skip dedup entirely, so a retry or
    `redrive()` of the same source duplicated the action note every time.
    Keyed on `(source_slug, locator, normalized what)` -- exactly what a
    retry of the SAME source reprocesses identically."""
    source = make_source_row(db, "s-doc-099", source_type="doc", source_date=date(2026, 8, 20))
    action = build_action(
        "Send the CSV export spec to the customer",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p2",
    )

    outcome1 = merge_action(db, loaded_registry, action, source, entity_id=None)
    assert outcome1.status == "new"

    # Same source reprocessed (a retry attempt, or redrive()) -- same
    # Action object, same locator, same text.
    outcome2 = merge_action(db, loaded_registry, action, source, entity_id=None)
    assert outcome2.status == "duplicate"
    assert outcome2.action_id == outcome1.action_id
    assert outcome2.note_slug == outcome1.note_slug

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM actions")
        assert cur.fetchone()[0] == 1


def test_merge_action_no_account_dedup_is_scoped_to_source_and_locator(
    loaded_registry: Registry, db
):
    """The finding-4 fix must not over-merge: a different source, or the
    same source at a different locator, is a genuinely different action
    even with identical wording and no resolved account."""
    source = make_source_row(db, "s-doc-100", source_type="doc", source_date=date(2026, 8, 20))
    action = build_action(
        "Send the CSV export spec to the customer",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p2",
    )
    outcome1 = merge_action(db, loaded_registry, action, source, entity_id=None)
    assert outcome1.status == "new"

    other_locator = build_action(
        "Send the CSV export spec to the customer",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p9",
    )
    outcome2 = merge_action(db, loaded_registry, other_locator, source, entity_id=None)
    assert outcome2.status == "new"
    assert outcome2.action_id != outcome1.action_id

    other_source = make_source_row(
        db, "s-doc-101", source_type="doc", source_date=date(2026, 8, 21)
    )
    outcome3 = merge_action(db, loaded_registry, action, other_source, entity_id=None)
    assert outcome3.status == "new"
    assert outcome3.action_id not in (outcome1.action_id, outcome2.action_id)

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM actions")
        assert cur.fetchone()[0] == 3


def test_merge_action_does_not_dedup_a_clearly_different_action(loaded_registry: Registry, db):
    source = make_source_row(db, "s-call-013", source_type="call", source_date=date(2026, 8, 20))
    first = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention="Sam Okafor",
        due=date(2026, 8, 28),
        locator="00:31:10",
    )
    outcome1 = merge_action(db, loaded_registry, first, source, entity_id=ASHGROVE)

    second = build_action(
        "Schedule a quarterly business review with Ashgrove Health",
        owner_mention="Sam Okafor",
        due=date(2026, 10, 1),
        locator="00:40:00",
    )
    outcome2 = merge_action(db, loaded_registry, second, source, entity_id=ASHGROVE)

    assert outcome2.status == "new"
    assert outcome2.action_id != outcome1.action_id
    assert outcome2.note_slug != outcome1.note_slug

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM actions")
        assert cur.fetchone()[0] == 2


# ---------------------------------------------------------------------
# Owner inheritance (01-spec/actions-and-owners-2026-09-19.md, Phase 1,
# "Actions inherit the owner").
# ---------------------------------------------------------------------

MARIA_CHEN = "C-2001"  # a real contact crm_id, account: ashgrove (corpus/truth-table.yaml)


def _owner_dbid_for_crm_id(db, crm_id: str) -> int:
    with db.cursor() as cur:
        cur.execute("SELECT id FROM registry_entities WHERE crm_id = %s", (crm_id,))
        return cur.fetchone()[0]


def test_merge_action_inherits_the_account_owner_when_the_fact_names_none(
    loaded_registry: Registry, db
):
    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=SAM_OKAFOR) is True
    source = make_source_row(db, "s-owner-inherit-1", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p1",
    )

    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone()[0] == _owner_dbid_for_crm_id(db, SAM_OKAFOR)


def test_merge_action_prefers_an_explicit_owner_mention_over_inheritance(
    loaded_registry: Registry, db
):
    # Ashgrove's registry owner is a *different* person than the fact's
    # own owner_mention -- the explicit mention must win.
    with db.cursor() as cur:
        cur.execute(
            "SELECT crm_id FROM registry_entities WHERE kind = 'internal_person' "
            "AND canonical_name = 'Priya Natarajan'"
        )
        priya = cur.fetchone()[0]
    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=priya) is True

    source = make_source_row(db, "s-owner-inherit-2", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention="Sam Okafor",
        due=date(2026, 8, 28),
        locator="p2",
    )

    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone()[0] == _owner_dbid_for_crm_id(db, SAM_OKAFOR)


def test_merge_action_treats_a_customer_contact_owner_mention_as_unresolved(
    loaded_registry: Registry, db
):
    """Vendor-side only (Standing rules): an owner_mention that resolves
    to a customer contact is never set as the action's owner -- it falls
    through to the account's own registry owner instead, exactly like a
    fact that named no owner_mention at all."""
    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=SAM_OKAFOR) is True
    source = make_source_row(db, "s-owner-inherit-3", source_date=date(2026, 8, 20))
    action = build_action(
        "Confirm the delivery schedule with Ashgrove Health",
        owner_mention="Maria Chen",  # a real customer contact, not vendor-side
        due=date(2026, 8, 28),
        locator="p3",
    )

    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone()[0] == _owner_dbid_for_crm_id(db, SAM_OKAFOR)


def test_merge_action_inherits_via_a_contacts_own_account(loaded_registry: Registry, db):
    """The action's resolved primary entity is a CONTACT (Maria Chen),
    not an account directly -- inheritance walks up to her own account
    (Ashgrove Health, `registry_entities.parent_id`)."""
    assert confirm_owner_proposal(db, account_crm_id=ASHGROVE, owner_crm_id=SAM_OKAFOR) is True
    source = make_source_row(db, "s-owner-inherit-4", source_date=date(2026, 8, 20))
    action = build_action(
        "Follow up with Maria Chen about the renewal",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p4",
    )

    outcome = merge_action(db, loaded_registry, action, source, entity_id=MARIA_CHEN)

    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone()[0] == _owner_dbid_for_crm_id(db, SAM_OKAFOR)


def test_merge_action_leaves_owner_null_when_the_account_has_none(loaded_registry: Registry, db):
    """No inheritance possible -- Ashgrove has no registry owner set --
    `owner_entity_id` simply stays null, same as before this feature."""
    source = make_source_row(db, "s-owner-inherit-5", source_date=date(2026, 8, 20))
    action = build_action(
        "Send Ashgrove Health a written delivery root-cause summary",
        owner_mention=None,
        due=date(2026, 8, 28),
        locator="p5",
    )

    outcome = merge_action(db, loaded_registry, action, source, entity_id=ASHGROVE)

    with db.cursor() as cur:
        cur.execute("SELECT owner_entity_id FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone()[0] is None


def test_merge_action_slug_number_is_the_actions_row_id(loaded_registry: Registry, db):
    """A rolled-back insert (a retried merge) leaves a gap in the id
    sequence; the note slug still names the row's real id."""
    source = make_source_row(db, "s-call-020", source_type="call", source_date=date(2026, 8, 20))
    with db.cursor() as cur:
        cur.execute("SELECT nextval(pg_get_serial_sequence('actions', 'id'))")

    outcome = merge_action(
        db,
        loaded_registry,
        build_action("Send Ashgrove Health the revised shipping schedule", locator="00:02:00"),
        source,
        entity_id=ASHGROVE,
    )

    assert outcome.status == "new"
    assert outcome.note_slug.startswith(f"action-{outcome.action_id}-")
    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM actions WHERE id = %s", (outcome.action_id,))
        assert cur.fetchone() == (outcome.note_slug,)
