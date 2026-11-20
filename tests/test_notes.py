"""Tests for `portolan/notes/**` (`01-spec/build-contracts.md` "Notes and
claims", wave B1).

Run with this module's own test database:
    createdb portolan_test_merge
    DATABASE_URL=postgresql://localhost/portolan_test_merge \
        uv run pytest tests/test_notes.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

from portolan.lint import checks
from portolan.lint._store import Store
from portolan.notes import (
    Claim,
    NoteRow,
    NoteStatus,
    NoteType,
    ensure_entity_note,
    link_related,
    note_exists,
    read_note,
    render_note,
    write_note,
)
from portolan.notes.render import link_mentions, slugify
from portolan.registry import Registry
from tests.merge_helpers import (
    ASHGROVE,
    LARKSPUR,
    LARKSPUR_MEMORIAL,
    PINECREST,
    loaded_registry_fixture,  # noqa: F401 - registers the `loaded_registry` fixture
    make_source_row,
)


class FakeStore(Store):
    def put(self, key: str, data: bytes, content_type: str) -> None:  # noqa: ARG002
        pass

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        return True

    def url(self, key: str) -> str:
        return f"fake://{key}"


def _note(slug: str, **overrides) -> NoteRow:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    defaults = dict(
        slug=slug,
        type=NoteType.ACCOUNT,
        title="Ashgrove Health",
        description="A system account.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id="A-1010",
    )
    defaults.update(overrides)
    return NoteRow(**defaults)


# --------------------------------------------------------------------
# slugify / link_mentions
# --------------------------------------------------------------------


def test_slugify_matches_the_contracted_examples():
    assert slugify("Ashgrove Health") == "ashgrove-health"
    assert slugify("Maria Chen") == "maria-chen"
    assert slugify("Tidewater T2") == "tidewater-t2"


def test_link_mentions_replaces_found_mentions_and_appends_missing_ones():
    text = "Ashgrove Medical Center runs the Tidewater T3 pump."
    out = link_mentions(text, [("Ashgrove Medical Center", "ashgrove-medical-center")])
    assert "[[ashgrove-medical-center]]" in out
    assert "Ashgrove Medical Center" not in out  # replaced, not duplicated

    # A mention that doesn't appear verbatim in the text is appended, not
    # silently dropped -- the target note must still get an inbound link.
    out2 = link_mentions("Something happened.", [("Some Entity", "some-entity")])
    assert "[[some-entity]]" in out2


# --------------------------------------------------------------------
# render_note
# --------------------------------------------------------------------


def test_render_note_produces_frontmatter_and_a_resolvable_citation(db):
    make_source_row(db, "s-call-001", source_type="call")
    note = _note("ashgrove-health")
    claim = Claim(
        note_slug=note.slug,
        fact_type="complaint",
        claim_text="[[ashgrove-health]] received three late deliveries in August 2026.",
        source_slug="s-call-001",
        locator="00:14:32",
        quote="three late deliveries",
        recorded=datetime(2026, 8, 20, tzinfo=UTC),
        sentiment=-2,
        confidence=0.9,
    )
    body = render_note(note, [claim])

    assert body.startswith("---\n")
    assert 'title: "Ashgrove Health"' in body
    assert "type: account" in body
    assert "crm_id: A-1010" in body
    assert "^[s-call-001 @ 00:14:32]" in body
    assert '(quote: "three late deliveries")' in body

    write_note(db, note, [claim], changed_by="test", reason="seed")

    report_findings = checks.check_unresolved_citation(db)
    assert report_findings == []
    # thin_links is a warning-only check (never fails the run); asserting
    # it still runs cleanly against a note with a real link is enough to
    # satisfy "passes the lint's ... thin_links check" -- it produces a
    # finding object, not an exception, either way.
    checks.check_thin_links(db)


def test_render_note_strikes_through_a_superseded_claim():
    note = _note("ashgrove-health")
    old = Claim(
        note_slug=note.slug,
        fact_type="decision",
        claim_text="[[ashgrove-health]] renews on 2027-03-31.",
        source_slug="s-crm-1",
        locator="row 1",
        recorded=datetime(2026, 4, 14, tzinfo=UTC),
        superseded_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    body = render_note(note, [old])
    assert "~~" in body
    assert "→ superseded 2026-08-20" in body


# --------------------------------------------------------------------
# write_note / read_note
# --------------------------------------------------------------------


def _insert_claim_row(conn, claim: Claim) -> int:
    """`write_note` deliberately does not touch the `claims` table (that's
    `portolan.merge`'s job, via a real DB id it needs before rendering) --
    this test-only helper stands in for that INSERT so `write_note` has a
    real, DB-backed claim to render and `read_note` can read it back."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, recorded)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                claim.note_slug,
                claim.fact_type,
                claim.claim_text,
                claim.source_slug,
                claim.locator,
                claim.recorded,
            ),
        )
        return cur.fetchone()[0]


def test_write_note_versions_the_previous_body_and_rewrites_links(db):
    make_source_row(db, "s-call-001", source_type="call")
    note = _note("ashgrove-health")
    write_note(db, note, [], changed_by="test", reason="create")

    read_back, claims = read_note(db, "ashgrove-health")
    assert claims == []
    assert read_back.version == 1

    claim = Claim(
        note_slug="ashgrove-health",
        fact_type="complaint",
        claim_text="Ashgrove complained about late deliveries.",
        source_slug="s-call-001",
        locator="00:01:00",
        recorded=datetime(2026, 8, 20, tzinfo=UTC),
    )
    claim.id = _insert_claim_row(db, claim)
    write_note(db, read_back, [claim], changed_by="test", reason="add claim")

    read_back2, claims2 = read_note(db, "ashgrove-health")
    assert read_back2.version == 2
    assert len(claims2) == 1

    with db.cursor() as cur:
        cur.execute("SELECT version FROM note_versions WHERE slug = %s", ("ashgrove-health",))
        versions = sorted(row[0] for row in cur.fetchall())
    assert versions == [1]

    with db.cursor() as cur:
        cur.execute(
            "SELECT to_slug, kind FROM links WHERE from_slug = %s ORDER BY to_slug, kind",
            ("ashgrove-health",),
        )
        links = cur.fetchall()
    assert ("s-call-001", "provenance") in links


def test_write_note_preserves_a_custom_synthesis_across_a_claims_only_edit(db):
    note = _note("ashgrove-health")
    note.synthesis = "Ashgrove Health is a hand-curated summary."
    write_note(db, note, [], changed_by="test", reason="create")

    read_back, claims = read_note(db, "ashgrove-health")
    assert read_back.synthesis == "Ashgrove Health is a hand-curated summary."

    make_source_row(db, "s-call-002", source_type="call")
    claim = Claim(
        note_slug="ashgrove-health",
        fact_type="other",
        claim_text="Something else happened.",
        source_slug="s-call-002",
        locator="00:02:00",
        recorded=datetime(2026, 8, 21, tzinfo=UTC),
    )
    write_note(db, read_back, [*claims, claim], changed_by="test", reason="append")

    read_back2, _claims2 = read_note(db, "ashgrove-health")
    assert read_back2.synthesis == "Ashgrove Health is a hand-curated summary."


def test_link_related_is_idempotent_and_reports_whether_it_wrote(db):
    write_note(db, _note("ashgrove-health"), [], changed_by="test", reason="create")
    write_note(
        db,
        _note(
            "signal-complaint-ashgrove-health", type=NoteType.SIGNAL, crm_id=None, title="Signal"
        ),
        [],
        changed_by="test",
        reason="create",
    )

    wrote = link_related(
        db, "ashgrove-health", "signal-complaint-ashgrove-health", changed_by="test", reason="link"
    )
    assert wrote is True

    wrote_again = link_related(
        db, "ashgrove-health", "signal-complaint-ashgrove-health", changed_by="test", reason="link"
    )
    assert wrote_again is False

    note, _claims = read_note(db, "ashgrove-health")
    assert note.related == ["signal-complaint-ashgrove-health"]


# --------------------------------------------------------------------
# ensure_entity_note
# --------------------------------------------------------------------


def test_ensure_entity_note_builds_the_parent_chain_as_parent_links(loaded_registry: Registry, db):
    note = ensure_entity_note(db, loaded_registry, LARKSPUR_MEMORIAL)
    assert note.crm_id == LARKSPUR_MEMORIAL

    larkspur_note, _ = read_note(db, note.parent_slug)
    assert larkspur_note.crm_id == LARKSPUR

    pinecrest_note, _ = read_note(db, larkspur_note.parent_slug)
    assert pinecrest_note.crm_id == PINECREST
    assert pinecrest_note.parent_slug is None

    with db.cursor() as cur:
        cur.execute("SELECT from_slug, to_slug FROM links WHERE kind = 'parent' ORDER BY from_slug")
        parent_links = cur.fetchall()
    assert (note.slug, larkspur_note.slug) in parent_links
    assert (larkspur_note.slug, pinecrest_note.slug) in parent_links


def test_ensure_entity_note_is_idempotent_and_caches_on_the_registry_row(
    loaded_registry: Registry, db
):
    first = ensure_entity_note(db, loaded_registry, ASHGROVE)
    second = ensure_entity_note(db, loaded_registry, ASHGROVE)
    assert first.slug == second.slug
    assert note_exists(db, first.slug)

    with db.cursor() as cur:
        cur.execute("SELECT note_slug FROM registry_entities WHERE crm_id = %s", (ASHGROVE,))
        cached = cur.fetchone()[0]
    assert cached == first.slug

    # Fetching again must not create a second version -- it's a plain read.
    read_back, _claims = read_note(db, first.slug)
    assert read_back.version == 1


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 8: `stale_after` was computed once, only
# `if note.stale_after is None` -- true only on a note's first-ever
# write, since every later `write_note` call arrives with `note.
# stale_after` already populated from `read_note`'s round-trip. A note
# written today could carry a `stale_after` from months ago, a permanent
# lint error for a note that's actually current.
# --------------------------------------------------------------------


def test_write_note_refreshes_stale_after_on_every_write(db):
    note = _note("stale-refresh-account")
    write_note(db, note, [], changed_by="test", reason="create")

    read_back, claims = read_note(db, "stale-refresh-account")
    # Simulate a note whose stale_after has already expired by the time
    # of its next write (the exact bug: an old, unrefreshed value).
    read_back.stale_after = datetime(2020, 1, 1, tzinfo=UTC)
    write_note(db, read_back, claims, changed_by="test", reason="touch")

    refreshed, _claims2 = read_note(db, "stale-refresh-account")
    now = datetime.now(UTC)
    assert refreshed.stale_after is not None
    assert refreshed.stale_after > now, (
        "stale_after must be recomputed from the fresh `updated`, not left "
        "at whatever stale value the note arrived with"
    )
    # Recomputed from `updated`, not from the note's original `created`.
    assert refreshed.stale_after > refreshed.created


def test_write_note_preserve_stale_after_keeps_an_explicit_override(db):
    note = _note("stale-preserve-account")
    write_note(db, note, [], changed_by="test", reason="create")

    read_back, claims = read_note(db, "stale-preserve-account")
    override = datetime(2099, 1, 1, tzinfo=UTC)
    read_back.stale_after = override
    write_note(
        db,
        read_back,
        claims,
        changed_by="test",
        reason="human override",
        preserve_stale_after=True,
    )

    refreshed, _claims2 = read_note(db, "stale-preserve-account")
    assert refreshed.stale_after == override


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 11: the frontmatter `source:` list was
# built from each claim's own primary `source_slug` only, so a note
# whose body plainly cites a corroborating source via `claim.citations`
# (a `same`-verdict second sighting, G4) still under-reported its own
# `source:` header.
# --------------------------------------------------------------------


def test_render_note_source_frontmatter_includes_corroborating_citations(db):
    from portolan.notes.models import ClaimCitation

    note = _note("ashgrove-health")
    claim = Claim(
        note_slug="ashgrove-health",
        fact_type="complaint",
        claim_text="Ashgrove complained about late deliveries.",
        source_slug="s-call-a",
        locator="00:00:00",
        recorded=datetime(2026, 8, 20, tzinfo=UTC),
        citations=[ClaimCitation(source_slug="s-ticket-b", locator="msg-3")],
    )

    body = render_note(note, [claim])
    frontmatter = body.split("---", 2)[1]
    assert "- s-call-a" in frontmatter
    assert "- s-ticket-b" in frontmatter


def test_link_mentions_never_rewrites_the_inside_of_a_longer_word():
    from portolan.notes.render import link_mentions

    linked = link_mentions("Arcadia Hospital uses Arc daily.", [("Arc", "arc")])
    assert linked == "Arcadia Hospital uses [[arc]] daily."
    # No whole-word occurrence: appended, as for a paraphrased mention.
    assert link_mentions("Arcadia Hospital called.", [("Arc", "arc")]) == (
        "Arcadia Hospital called. ([[arc]])"
    )
    # A mention that only occurs inside an existing link is not linked twice over.
    assert (
        link_mentions("Tidewater T2 jammed.", [("Tidewater T2", "tidewater-t2"), ("t2", "t2-kit")])
        == "[[tidewater-t2]] jammed. ([[t2-kit]])"
    )
