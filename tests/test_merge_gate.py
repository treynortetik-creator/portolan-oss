"""Tests for `portolan.merge.gate` (`01-spec/build-contracts.md` "Merge";
review G16, polish wave 2026-09-12): the candidate gate's cross-note half.

`portolan.merge.facts.merge_fact` calls `gate()` home-note-only (no
`conn`/`entity_slugs`), so the live pipeline does not use the cross-note
half yet. These tests exercise
`gate()` directly, proving the mechanism itself: a duplicate fact that
lands on a *different* home note than its twin is still caught once a
caller passes `conn`/`entity_slugs`, and is *not* caught without them
(the pre-G16, backward-compatible default).

Run with this module's own test database:
    createdb portolan_test_polish
    DATABASE_URL=postgresql://localhost/portolan_test_polish \
        uv run pytest tests/test_merge_gate.py -v
"""

from __future__ import annotations

from datetime import date

from portolan.merge.gate import GATE_SKIP_THRESHOLD, gate
from portolan.notes import read_note
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from tests.merge_helpers import make_source_row

ACCOUNT = "gate-test-account"
HOME_A = "gate-test-signal-a"
HOME_B = "gate-test-signal-b"


def _insert_note(
    conn, *, slug: str, type_: str, title: str, description: str, crm_id=None, body_md: str = ""
):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, crm_id, body_md) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (slug, type_, title, description, crm_id, body_md),
        )


def _link(conn, from_slug: str, to_slug: str, kind: str = "body") -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, %s)",
            (from_slug, to_slug, kind),
        )


def _insert_claim(conn, *, note_slug: str, source_slug: str, claim_text: str, locator: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, sentiment)
            VALUES (%s, 'complaint', %s, %s, %s, -1)
            RETURNING id
            """,
            (note_slug, claim_text, source_slug, locator),
        )
        return cur.fetchone()[0]


def _setup_graph(conn, *, home_a_body: str = "", home_b_body: str = "") -> None:
    """One account note and two of its signal notes -- the real shape
    `portolan.merge.facts._ensure_signal_note`/`notes.persist.link_related`
    produce: a signal note's own description wikilinks its account (`body`,
    signal -> account) and the account's `## Related` links back (`body`,
    account -> signal). `home_a_body`/`home_b_body` stand in for what
    `notes.render.render_note` would actually put in `body_md` once a
    claim is attached (a real note's claim lines live in its rendered
    body, which is what `search()`'s full-text/vector index runs over) --
    each test passes the claim text it inserts so the note the claim is
    "recorded on" is also the note `search()` can actually find it
    through, exactly as it would be for a real merge."""
    _insert_note(
        conn,
        slug=ACCOUNT,
        type_="account",
        title="Gate Test Account",
        description="An account used only for gate cross-note tests.",
        crm_id="A-9500",
        body_md="Gate Test Account is a hospital system used only for gate cross-note tests.",
    )
    _insert_note(
        conn,
        slug=HOME_A,
        type_="signal",
        title="Late deliveries -- Gate Test Account",
        description=f"Late-delivery signals for [[{ACCOUNT}]].",
        body_md=f"{home_a_body}\n\n[[{ACCOUNT}]]".strip(),
    )
    _insert_note(
        conn,
        slug=HOME_B,
        type_="signal",
        title="Pricing pressure -- Gate Test Account",
        description=f"Pricing signals for [[{ACCOUNT}]].",
        body_md=f"{home_b_body}\n\n[[{ACCOUNT}]]".strip(),
    )
    _link(conn, HOME_A, ACCOUNT)
    _link(conn, HOME_B, ACCOUNT)
    _link(conn, ACCOUNT, HOME_A)
    _link(conn, ACCOUNT, HOME_B)


def test_cross_note_gate_catches_a_duplicate_on_a_different_home_note(db):
    embedder = FakeEmbedder()
    claim_text = "Gate Test Account reported a late delivery of consumables on 2026-08-04."
    _setup_graph(db, home_a_body=claim_text)
    source = make_source_row(db, "s-ticket-500", source_type="ticket", source_date=date(2026, 8, 5))

    claim_id = _insert_claim(
        db,
        note_slug=HOME_A,
        source_slug=source.slug,
        claim_text=claim_text,
        locator="T-500#1",
    )
    reindex_all(db, embedder)

    _home_b_note, home_b_claims = read_note(db, HOME_B)
    assert home_b_claims == []  # nothing on HOME_B itself yet -- the twin lives on HOME_A

    new_text = "Gate Test Account had that late consumables delivery again on 2026-08-04."

    # Pre-G16 / backward-compatible default: no `conn`/`entity_slugs`, so
    # only HOME_B's own (empty) claims are considered -- nothing to catch.
    old_candidate, old_score = gate(embedder, new_text, home_b_claims)
    assert old_candidate is None
    assert old_score == 0.0

    # G16's fix: search the account's other linked notes too.
    candidate, score = gate(
        embedder,
        new_text,
        home_b_claims,
        conn=db,
        entity_slugs=[ACCOUNT],
        exclude_note_slugs=[HOME_B],
    )
    assert candidate is not None
    assert candidate.id == claim_id
    assert candidate.note_slug == HOME_A
    assert score >= GATE_SKIP_THRESHOLD


def test_cross_note_gate_prefers_a_stronger_home_note_candidate(db):
    """When the home note's own claim is at least as good a match as
    anything found cross-note, the home candidate still wins -- the
    cross-note half only ever adds candidates, never displaces a better
    home-note match with a worse cross-note one."""
    embedder = FakeEmbedder()
    home_b_claim_text = "Gate Test Account pushed back hard on pricing during renewal talks."
    home_a_claim_text = "Gate Test Account reported a late delivery of consumables on 2026-08-04."
    _setup_graph(db, home_a_body=home_a_claim_text, home_b_body=home_b_claim_text)
    source = make_source_row(db, "s-ticket-501", source_type="ticket", source_date=date(2026, 8, 6))

    home_claim_id = _insert_claim(
        db,
        note_slug=HOME_B,
        source_slug=source.slug,
        claim_text=home_b_claim_text,
        locator="T-501#1",
    )
    _insert_claim(
        db,
        note_slug=HOME_A,
        source_slug=source.slug,
        claim_text=home_a_claim_text,
        locator="T-501#2",
    )
    reindex_all(db, embedder)

    _home_b_note, home_b_claims = read_note(db, HOME_B)
    new_text = "Gate Test Account pushed back hard on pricing again during renewal talks."

    candidate, score = gate(
        embedder,
        new_text,
        home_b_claims,
        conn=db,
        entity_slugs=[ACCOUNT],
        exclude_note_slugs=[HOME_B],
    )
    assert candidate is not None
    assert candidate.id == home_claim_id
    assert candidate.note_slug == HOME_B
    assert score >= GATE_SKIP_THRESHOLD


def test_gate_excludes_a_superseded_claim_from_the_candidate_pool(db):
    """Review finding 2b: `read_note` hands back every unarchived claim on
    a note, superseded ones included -- they still need to render, struck
    through, in the note body. `gate()` must filter them back out itself,
    or a fact that merely repeats a claim the graph has already corrected
    would "match" the dead claim instead of the live one that replaced
    it (or, worse, supersede it a second time -- see
    `test_merge_facts.test_supersede_is_idempotent_and_queues_a_review_
    row_on_a_second_call` for that half)."""
    embedder = FakeEmbedder()
    claim_text = "Gate Test Account reported a late delivery of consumables on 2026-08-04."
    _setup_graph(db, home_a_body=claim_text)
    source = make_source_row(db, "s-ticket-502", source_type="ticket", source_date=date(2026, 8, 5))

    old_id = _insert_claim(
        db, note_slug=HOME_A, source_slug=source.slug, claim_text=claim_text, locator="T-502#1"
    )
    replacement_id = _insert_claim(
        db,
        note_slug=HOME_A,
        source_slug=source.slug,
        claim_text="Gate Test Account's late delivery was resolved on 2026-08-10.",
        locator="T-502#2",
    )
    with db.cursor() as cur:
        cur.execute(
            "UPDATE claims SET superseded_at = now(), superseded_by = %s WHERE id = %s",
            (replacement_id, old_id),
        )

    _note, claims = read_note(db, HOME_A)
    assert {c.id for c in claims} == {old_id, replacement_id}  # read_note still hands both back

    # A near-exact duplicate of the now-superseded claim's own text,
    # against a pool containing ONLY that superseded claim: without the
    # finding-2b fix this would score close to 1.0 and match it.
    only_superseded = [c for c in claims if c.id == old_id]
    candidate, score = gate(embedder, claim_text, only_superseded)
    assert candidate is None
    assert score == 0.0
