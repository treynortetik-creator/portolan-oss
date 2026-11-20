"""Tests for `portolan/consolidate/**` (`01-spec/quality-loop.md` "The
consolidation pass", "Human edits").

Own test database: `createdb portolan_test_audit`; run with
`DATABASE_URL=postgresql://localhost/portolan_test_audit` (shared with
`tests/test_audit*.py` -- this wave's one database, per the wave
instructions). Uses the shared `db` fixture (`tests/conftest.py`): a
fresh, migrated schema per test, so notes/claims are hand-built here with
`portolan.notes.persist` and raw SQL rather than a full pipeline run
(`tests/test_audit.py` needs the real graph shape for fact-matching;
these tests need precise control over claim counts, ages and citations).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import psycopg

from portolan.consolidate.archive import archive_note_slug, select_claims_to_archive
from portolan.consolidate.draft import CLAIM_ID_TAG_RE, MIN_SYNTHESIS_CITATIONS, check
from portolan.consolidate.edits import edit_citation, edits_for, record_edit
from portolan.consolidate.run import consolidate_note, run_consolidation
from portolan.consolidate.trigger import (
    CONSOLIDATION_CLAIMS_THRESHOLD,
    CONSOLIDATION_UPDATES_THRESHOLD,
    needs_consolidation,
)
from portolan.lint.report import run_lint
from portolan.llm.fake import FakeLLMClient
from portolan.notes.models import NoteRow, NoteStatus, NoteType
from portolan.notes.persist import read_note, write_note
from portolan.util import clock

# --------------------------------------------------------------------
# Raw-SQL / notes-API fixture builders
# --------------------------------------------------------------------


class FakeStore:
    """Minimal `portolan.store.Store` stand-in for `run_lint`'s
    `missing_original` check -- everything "exists" so that check never
    fires on the hand-built sources these tests use."""

    def put(self, key: str, data: bytes, content_type: str) -> None:  # noqa: ARG002
        pass

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        return True

    def url(self, key: str) -> str:
        return f"fake://{key}"


def _insert_source(cur: psycopg.Cursor, slug: str, *, source_type: str = "call") -> str:
    cur.execute(
        "INSERT INTO sources (slug, source_type, s3_key, source_date, extraction_status) "
        "VALUES (%s, %s, %s, %s, 'done')",
        (slug, source_type, f"originals/{source_type}/{slug}", "2026-08-01"),
    )
    return slug


def _insert_claim(
    cur: psycopg.Cursor,
    note_slug: str,
    *,
    fact_type: str = "complaint",
    claim_text: str,
    source_slug: str,
    locator: str,
    recorded: datetime | None = None,
) -> int:
    columns = ["note_slug", "fact_type", "claim_text", "source_slug", "locator"]
    values: list[object] = [note_slug, fact_type, claim_text, source_slug, locator]
    if recorded is not None:
        columns.append("recorded")
        values.append(recorded)
    placeholders = ", ".join(["%s"] * len(values))
    cur.execute(
        f"INSERT INTO claims ({', '.join(columns)}) VALUES ({placeholders}) RETURNING id",
        values,
    )
    return cur.fetchone()[0]


def _make_note(conn: psycopg.Connection, slug: str, *, crm_id: str = "A-9001") -> NoteRow:
    now = datetime.now(UTC)
    note = NoteRow(
        slug=slug,
        type=NoteType.ACCOUNT,
        title=slug.replace("-", " ").title(),
        description=f"A test account ({slug}).",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id=crm_id,
    )
    write_note(conn, note, [], changed_by="test", reason="setup")
    return note


_CLAIM_LINE_RE = re.compile(
    r"^- claim (\d+) \[([^\]]+)\](?: \([^)]*\))?: (.*?) (\^\[[^\]]+\])\s*$", re.M
)


def _echo_draft_responder(*, system, user, json_schema, model, surface):
    """A cheap-model fake that builds a mechanically valid draft directly
    from whatever claim list `draft_synthesis` put in the prompt --
    whichever claims `consolidate_note` actually asked it to cover (i.e.
    after any pre-draft archiving)."""
    del system, json_schema, model, surface
    lines = [
        f"{text.strip()} {marker} [claim:{claim_id}]"
        for claim_id, _fact_type, text, marker in _CLAIM_LINE_RE.findall(user)
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------
# needs_consolidation
# --------------------------------------------------------------------


def test_needs_consolidation_triggered_by_claims_threshold(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "big-note")
    _make_note(db, "small-note", crm_id="A-9002")

    base = datetime(2026, 1, 1, tzinfo=UTC)
    with db.cursor() as cur:
        for i in range(35):
            _insert_claim(
                cur,
                "big-note",
                claim_text=f"Claim {i} happened.",
                source_slug="s1",
                locator=f"00:{i:02d}:00",
                recorded=base + timedelta(days=i),
            )
        for i in range(5):
            _insert_claim(
                cur,
                "small-note",
                claim_text=f"Claim {i} happened.",
                source_slug="s1",
                locator=f"00:{i:02d}:00",
                recorded=base + timedelta(days=i),
            )

    slugs = needs_consolidation(db)
    assert "big-note" in slugs
    assert "small-note" not in slugs


def test_needs_consolidation_triggered_by_updates_threshold(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "churny-note")

    for i in range(CONSOLIDATION_UPDATES_THRESHOLD + 2):
        note, claims = read_note(db, "churny-note")
        write_note(db, note, claims, changed_by="test", reason=f"touch {i}")

    note, _claims = read_note(db, "churny-note")
    assert note.version > CONSOLIDATION_UPDATES_THRESHOLD

    slugs = needs_consolidation(db)
    assert "churny-note" in slugs


def test_needs_consolidation_resets_after_recorded_consolidation(db):
    """A note just consolidated (a `changelog` row recording its
    post-consolidation version) is not immediately due again."""
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "settled-note")
    note, _claims = read_note(db, "settled-note")

    import json

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO changelog (kind, summary, details) VALUES ('consolidate', 'x', %s::jsonb)",
            (json.dumps({"note_slug": "settled-note", "version": note.version}),),
        )

    assert "settled-note" not in needs_consolidation(db)


# --------------------------------------------------------------------
# draft_synthesis / check
# --------------------------------------------------------------------


def test_check_ok_on_complete_draft(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-1")
    with db.cursor() as cur:
        _insert_claim(cur, "acct-1", claim_text="Fact one.", source_slug="s1", locator="00:00:01")
        _insert_claim(cur, "acct-1", claim_text="Fact two.", source_slug="s1", locator="00:00:02")
    _note, claims = read_note(db, "acct-1")

    good_draft = "\n".join(
        f"{c.claim_text} ^[{c.source_slug} @ {c.locator}] [claim:{c.id}]" for c in claims
    )
    result = check(good_draft, claims)
    assert result.ok, result.reasons


def test_check_rejects_a_draft_below_the_citation_floor(db):
    """Review finding G7: with only 2 claims on file, the floor is
    `min(MIN_SYNTHESIS_CITATIONS, 2) == 2` -- citing just 1 fails, but (per
    the fix) NOT because it omits a specific claim; the same claim list
    consolidated a second time is free to omit either one, so long as the
    floor is met."""
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-1")
    with db.cursor() as cur:
        _insert_claim(cur, "acct-1", claim_text="Fact one.", source_slug="s1", locator="00:00:01")
        _insert_claim(cur, "acct-1", claim_text="Fact two.", source_slug="s1", locator="00:00:02")
    _note, claims = read_note(db, "acct-1")

    bad_draft = (
        f"{claims[0].claim_text} ^[{claims[0].source_slug} @ {claims[0].locator}] "
        f"[claim:{claims[0].id}]"
    )
    result = check(bad_draft, claims)
    assert not result.ok
    assert result.too_few_citations
    assert result.cited_claim_ids == [claims[0].id]
    assert claims[1].id not in result.cited_claim_ids


def test_check_accepts_a_draft_that_omits_some_claims_above_the_floor(db):
    """Review finding G7 / task item 4: the whole point of the fix -- a
    synthesis need not cite every claim, only enough of them (the floor)."""
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-1")
    with db.cursor() as cur:
        for i in range(7):
            _insert_claim(
                cur, "acct-1", claim_text=f"Fact {i}.", source_slug="s1", locator=f"{i:02d}:00:00"
            )
    _note, claims = read_note(db, "acct-1")

    # Cite only 5 of the 7 claims -- the floor, not the full list.
    cited = claims[:5]
    draft = "\n".join(
        f"{c.claim_text} ^[{c.source_slug} @ {c.locator}] [claim:{c.id}]" for c in cited
    )
    result = check(draft, claims)
    assert result.ok, result.reasons
    assert len(result.cited_claim_ids) == 5


def test_check_rejects_relative_time_language(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-1")
    with db.cursor() as cur:
        _insert_claim(cur, "acct-1", claim_text="Fact one.", source_slug="s1", locator="00:00:01")
    _note, claims = read_note(db, "acct-1")

    draft = (
        f"Recently, {claims[0].claim_text} ^[{claims[0].source_slug} @ {claims[0].locator}] "
        f"[claim:{claims[0].id}]"
    )
    result = check(draft, claims)
    assert not result.ok
    assert "recently" in result.relative_time_hits


def test_claim_id_tag_regex_matches_draft_synthesis_output():
    assert CLAIM_ID_TAG_RE.findall("text ^[a @ b] [claim:42]") == ["42"]


# --------------------------------------------------------------------
# consolidate_note: retry, failure, success
# --------------------------------------------------------------------


def test_consolidate_note_retries_with_strong_model_on_bad_draft(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-1")
    with db.cursor() as cur:
        _insert_claim(cur, "acct-1", claim_text="Fact one.", source_slug="s1", locator="00:00:01")
        _insert_claim(cur, "acct-1", claim_text="Fact two.", source_slug="s1", locator="00:00:02")
        _insert_claim(cur, "acct-1", claim_text="Fact three.", source_slug="s1", locator="00:00:03")

    _note, claims = read_note(db, "acct-1")
    bad_draft = " ".join(
        f"{c.claim_text} ^[{c.source_slug} @ {c.locator}] [claim:{c.id}]" for c in claims[:2]
    )
    good_draft = "\n".join(
        f"{c.claim_text} ^[{c.source_slug} @ {c.locator}] [claim:{c.id}]" for c in claims
    )

    cheap_llm = FakeLLMClient([bad_draft], model_key="claude-haiku-4-5")
    strong_llm = FakeLLMClient([good_draft], model_key="claude-sonnet-5")

    outcome = consolidate_note(db, cheap_llm, "acct-1", strong_llm=strong_llm)

    assert outcome.status == "consolidated"
    assert outcome.used_strong is True
    assert outcome.draft_attempts == 2

    note, _claims = read_note(db, "acct-1")
    assert good_draft in (note.synthesis or "")


def test_consolidate_note_fails_both_attempts_queues_review_and_leaves_note_untouched(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-1")
    with db.cursor() as cur:
        _insert_claim(cur, "acct-1", claim_text="Fact one.", source_slug="s1", locator="00:00:01")
        _insert_claim(cur, "acct-1", claim_text="Fact two.", source_slug="s1", locator="00:00:02")

    before, _claims = read_note(db, "acct-1")
    always_bad_draft = "This mentions nothing citable at all."
    cheap_llm = FakeLLMClient([always_bad_draft], model_key="claude-haiku-4-5")
    strong_llm = FakeLLMClient([always_bad_draft], model_key="claude-sonnet-5")

    outcome = consolidate_note(db, cheap_llm, "acct-1", strong_llm=strong_llm)

    assert outcome.status == "failed"
    assert outcome.used_strong is True
    assert outcome.review_queue_id is not None

    with db.cursor() as cur:
        cur.execute(
            "SELECT kind, payload->>'note_slug' FROM review_queue WHERE id = %s",
            (outcome.review_queue_id,),
        )
        kind, slug = cur.fetchone()
    assert kind == "failed_consolidation"
    assert slug == "acct-1"

    after, _claims = read_note(db, "acct-1")
    assert after.version == before.version
    assert after.body_md == before.body_md


def test_consolidate_note_skips_a_note_with_no_claims(db):
    _make_note(db, "empty-note")
    outcome = consolidate_note(db, FakeLLMClient([]), "empty-note")
    assert outcome.status == "skipped_no_claims"


def test_consolidate_note_reindexes_when_given_an_embedder(db):
    """Review finding G21: `consolidate_note` exists specifically to
    rewrite a note's searchable top -- without an `embedder`, the note's
    `chunks` (and the archive note's, when one is written) were left
    stale until some later, separate full reindex."""
    from portolan.search.embed import FakeEmbedder

    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-reindex")
    with db.cursor() as cur:
        _insert_claim(
            cur, "acct-reindex", claim_text="Fact one.", source_slug="s1", locator="00:00:01"
        )

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE note_slug = %s", ("acct-reindex",))
        (before_count,) = cur.fetchone()
    assert before_count == 0

    llm = FakeLLMClient(_echo_draft_responder, model_key="claude-haiku-4-5")
    outcome = consolidate_note(db, llm, "acct-reindex", embedder=FakeEmbedder())
    assert outcome.status == "consolidated"

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE note_slug = %s", ("acct-reindex",))
        (after_count,) = cur.fetchone()
    assert after_count > 0


def test_consolidate_note_archives_oldest_uncited_and_links_parent(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-35")

    base = datetime(2026, 1, 1, tzinfo=UTC)
    with db.cursor() as cur:
        for i in range(35):
            _insert_claim(
                cur,
                "acct-35",
                claim_text=f"Fact number {i} occurred.",
                source_slug="s1",
                locator=f"{i:02d}:00:00",
                recorded=base + timedelta(days=i),
            )

    note, claims = read_note(db, "acct-35")
    assert len(claims) == 35

    to_archive = select_claims_to_archive(note, claims, threshold=CONSOLIDATION_CLAIMS_THRESHOLD)
    expected_archived_ids = {c.id for c in to_archive}
    expected_remaining_ids = {c.id for c in claims} - expected_archived_ids
    assert len(expected_archived_ids) == 35 - CONSOLIDATION_CLAIMS_THRESHOLD

    llm = FakeLLMClient(_echo_draft_responder, model_key="claude-haiku-4-5")
    outcome = consolidate_note(db, llm, "acct-35")

    assert outcome.status == "consolidated"
    assert set(outcome.archived_claim_ids) == expected_archived_ids
    assert outcome.archive_note_slug == archive_note_slug("acct-35")

    with db.cursor() as cur:
        cur.execute("SELECT id FROM claims WHERE note_slug = %s AND archived = true", ("acct-35",))
        archived_in_db = {r[0] for r in cur.fetchall()}
    assert archived_in_db == expected_archived_ids

    # The archive note's own `claims` rows still carry `note_slug ==
    # "acct-35"` (only `archived` flips -- see `archive.py`'s module
    # docstring), so `read_note("acct-35-archive")` legitimately has no
    # claims of its own; the archived claims' text is rendered directly
    # into the archive note's body instead. Check that.
    archive_note, archive_note_claims = read_note(db, "acct-35-archive")
    assert archive_note.type.value == "report"
    assert archive_note.description is not None
    assert archive_note.description.startswith("[archive]")
    assert archive_note_claims == []
    archived_claim_texts = {c.claim_text for c in claims if c.id in expected_archived_ids}
    assert all(text in archive_note.body_md for text in archived_claim_texts)

    parent_note, parent_claims = read_note(db, "acct-35")
    assert "acct-35-archive" in parent_note.related
    assert {c.id for c in parent_claims} == expected_remaining_ids


class _BrokenEmbedder:
    """Stands in for the embedding model being unreachable mid-archive
    (review finding 4): `write_note(..., embedder=...)` calls `.embed()`
    via `index_note` whenever it has any chunks to write, which
    `_write_archive_note`'s call always does (a fresh archive note is
    never empty)."""

    dim = 8

    def embed(self, texts):  # noqa: ARG002
        raise RuntimeError("embedding model unreachable (simulated)")


def test_commit_archive_failure_leaves_claims_visible_and_unarchived(db):
    """Test gap 6 (review finding 4): before this fix, `claims.archived`
    flipped and committed BEFORE the archive note was written, so a
    failure in between (here, the embedder `write_note` calls while
    writing the archive note) left the claims hidden from their parent
    note with nothing yet linking to them. The fix reorders the two
    writes -- this asserts the claims are exactly as they were:
    `archived = false`, still on the parent note's own claim list, and no
    archive note exists at all.
    """
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    note = _make_note(db, "acct-archive-fail")
    with db.cursor() as cur:
        claim_id = _insert_claim(
            cur,
            "acct-archive-fail",
            claim_text="A claim that should never actually get archived.",
            source_slug="s1",
            locator="00:00:01",
        )

    _note, claims = read_note(db, "acct-archive-fail")
    to_archive = [c for c in claims if c.id == claim_id]
    assert to_archive

    from portolan.consolidate.archive import commit_archive

    try:
        commit_archive(db, note, to_archive, embedder=_BrokenEmbedder())
        raise AssertionError("expected the simulated embedding failure to propagate")
    except RuntimeError as exc:
        assert "unreachable" in str(exc)

    with db.cursor() as cur:
        cur.execute("SELECT archived FROM claims WHERE id = %s", (claim_id,))
        (archived,) = cur.fetchone()
    assert archived is False

    _parent_note, parent_claims = read_note(db, "acct-archive-fail")
    assert {c.id for c in parent_claims} == {claim_id}

    # Safely retried, per this fix's own reasoning: the archive note's
    # text itself may already have been written (`write_note`'s own
    # commit happens before its `index_note` call, which is where the
    # simulated failure actually lands), but nothing that matters was
    # lost or left inconsistent -- a retry with a working embedder
    # completes the archive normally.
    from portolan.search.embed import FakeEmbedder

    archive_slug = commit_archive(db, note, to_archive, embedder=FakeEmbedder())
    assert archive_slug == archive_note_slug("acct-archive-fail")

    with db.cursor() as cur:
        cur.execute("SELECT archived FROM claims WHERE id = %s", (claim_id,))
        (archived,) = cur.fetchone()
    assert archived is True

    _parent_note, parent_claims = read_note(db, "acct-archive-fail")
    assert parent_claims == []


def _selective_draft_responder(*, system, user, json_schema, model, surface):
    """A cheap-model fake that cites only the NEWEST `MIN_SYNTHESIS_
    CITATIONS` claims in the prompt, mirroring what `draft.py`'s fixed
    prompt (review finding G7) actually asks a real model for -- "the
    five to ten claims that matter", not `_echo_draft_responder`'s
    everything. `draft_synthesis`'s claims_block is oldest-first, so
    "newest" is just the last N lines."""
    del system, json_schema, model, surface
    lines = _CLAIM_LINE_RE.findall(user)
    newest = lines[-MIN_SYNTHESIS_CITATIONS:] if len(lines) > MIN_SYNTHESIS_CITATIONS else lines
    return "\n".join(
        f"{text.strip()} {marker} [claim:{claim_id}]"
        for claim_id, _fact_type, text, marker in newest
    )


def test_consolidation_archives_older_claims_across_two_consecutive_runs(db):
    """Task item 4 / review finding G7: with a synthesis that cites only
    the newest few claims (not all of them, per the fixed prompt), a
    SECOND consolidation round must archive OLDER claims than the first
    round did. Before the fix, a synthesis forced to cite EVERY remaining
    claim made all of them "load-bearing" after round 1, so round 2 could
    only ever archive the newest excess (the only claims not yet cited),
    never the genuinely old ones the growth threshold exists to clear."""
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-two-rounds")

    base = datetime(2026, 1, 1, tzinfo=UTC)
    with db.cursor() as cur:
        for i in range(35):
            _insert_claim(
                cur,
                "acct-two-rounds",
                claim_text=f"Fact number {i} occurred.",
                source_slug="s1",
                locator=f"{i:02d}:00:00",
                recorded=base + timedelta(days=i),
            )

    llm = FakeLLMClient(_selective_draft_responder, model_key="claude-haiku-4-5")

    # Round 1: 35 claims, threshold 30 -> archives the oldest 5 (days
    # 0..4); of the 30 remaining (days 5..34), the fake cites only the
    # newest 5 (days 30..34).
    outcome1 = consolidate_note(db, llm, "acct-two-rounds")
    assert outcome1.status == "consolidated"
    round1_archived = set(outcome1.archived_claim_ids)
    assert len(round1_archived) == 5

    # New activity: 5 more claims land, recorded after everything so far,
    # bringing the note back over threshold (30 remaining + 5 new = 35).
    with db.cursor() as cur:
        for i in range(35, 40):
            _insert_claim(
                cur,
                "acct-two-rounds",
                claim_text=f"Fact number {i} occurred.",
                source_slug="s1",
                locator=f"{i:02d}:00:00",
                recorded=base + timedelta(days=i),
            )

    outcome2 = consolidate_note(db, llm, "acct-two-rounds")
    assert outcome2.status == "consolidated"
    round2_archived = set(outcome2.archived_claim_ids)
    assert len(round2_archived) == 5

    with db.cursor() as cur:
        cur.execute("SELECT id, recorded FROM claims WHERE note_slug = %s", ("acct-two-rounds",))
        recorded_by_id = dict(cur.fetchall())

    newest_five_ids = {
        cid for cid, rec in recorded_by_id.items() if rec >= base + timedelta(days=35)
    }
    # The bug: round 2 could only archive from the 5 brand-new claims.
    assert round2_archived.isdisjoint(newest_five_ids), (
        f"round 2 archived newly-added claims instead of old ones: "
        f"{round2_archived & newest_five_ids}"
    )
    # The fix: round 2 continues chronologically where round 1 left off --
    # older claims leave the note, never newer ones.
    assert max(recorded_by_id[cid] for cid in round1_archived) < min(
        recorded_by_id[cid] for cid in round2_archived
    )


def test_run_consolidation_respects_limit(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    for n in range(3):
        slug = f"acct-limit-{n}"
        _make_note(db, slug, crm_id=f"A-800{n}")
        with db.cursor() as cur:
            for i in range(CONSOLIDATION_CLAIMS_THRESHOLD + 1):
                _insert_claim(
                    cur,
                    slug,
                    claim_text=f"Fact {i}.",
                    source_slug="s1",
                    locator=f"{i:02d}:00:00",
                )

    llm = FakeLLMClient(_echo_draft_responder, model_key="claude-haiku-4-5")
    outcomes = run_consolidation(db, llm, None, limit=2)
    assert len(outcomes) == 2


# --------------------------------------------------------------------
# Human edits
# --------------------------------------------------------------------


def test_record_edit_writes_record_and_citation_and_lint_passes(db):
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-edit")
    with db.cursor() as cur:
        claim_id = _insert_claim(
            cur, "acct-edit", claim_text="Wrong fact.", source_slug="s1", locator="00:00:01"
        )

    edit_id = record_edit(
        db,
        note_slug="acct-edit",
        editor="maintainer",
        before="Wrong fact.",
        after="Correct fact.",
        reason="customer correction",
        claim_id=claim_id,
    )

    records = edits_for(db, "acct-edit")
    assert len(records) == 1
    assert records[0].id == edit_id
    assert records[0].editor == "maintainer"
    assert records[0].claim_id == claim_id

    _note, claims = read_note(db, "acct-edit")
    edited = next(c for c in claims if c.id == claim_id)
    # Review finding G11: the note must actually show the NEW text, not
    # the old one with a citation marker appended -- `record_edit` applies
    # `after` itself now.
    assert "Correct fact." in edited.claim_text
    assert "Wrong fact." not in edited.claim_text
    # item 2, e2e-run follow-up wave ("one clock"): compare against the
    # same deployment-timezone `today()` `record_edit` itself now stamps
    # with, not stdlib `date.today()` (the system's local timezone,
    # whatever that happens to be on whatever machine runs the suite) --
    # the two disagree for part of every day unless they're the same call.
    assert edit_citation("maintainer", clock.today()) in edited.claim_text

    report = run_lint(db, FakeStore())
    assert report.for_check("unresolved_citation") == []
    assert report.for_check("relative_time") == []


def test_record_edit_refuses_a_stale_before(db):
    """Test gap 7 (review finding 5): a `before` that does not match what
    is actually stored -- someone's UI tab left open while the claim
    changed underneath them -- must be refused outright, not silently
    applied on top of whatever landed in between."""
    from portolan.consolidate.edits import StaleEditError

    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-edit-stale")
    with db.cursor() as cur:
        claim_id = _insert_claim(
            cur,
            "acct-edit-stale",
            claim_text="The actual current text.",
            source_slug="s1",
            locator="00:00:01",
        )

    try:
        record_edit(
            db,
            note_slug="acct-edit-stale",
            editor="maintainer",
            before="A stale, no-longer-true guess at the current text.",
            after="Corrected text.",
            reason="customer correction",
            claim_id=claim_id,
        )
        raise AssertionError("expected StaleEditError")
    except StaleEditError:
        pass

    # Nothing was written: no edit_records row, the claim untouched, no
    # new note version.
    assert edits_for(db, "acct-edit-stale") == []
    _note, claims = read_note(db, "acct-edit-stale")
    edited = next(c for c in claims if c.id == claim_id)
    assert edited.claim_text == "The actual current text."
    assert _note.version == 1

    with db.cursor() as cur:
        cur.execute("SELECT human_edit_id FROM claims WHERE id = %s", (claim_id,))
        (human_edit_id,) = cur.fetchone()
    assert human_edit_id is None


def test_record_edit_stores_the_actual_stored_text_as_before_not_the_callers_claim(db):
    """Test gap 7 (review finding 5): `edit_records.before_text` must be
    what was actually in the database, not merely whatever the caller
    happened to pass -- proven here by editing a claim that already
    carries a prior edit's citation marker (so its real current text is
    more than just the "bare fact" a caller might casually assume), and
    checking the recorded `before_text` is that exact, full string."""
    with db.cursor() as cur:
        _insert_source(cur, "s1")
    _make_note(db, "acct-edit-twice")
    with db.cursor() as cur:
        claim_id = _insert_claim(
            cur,
            "acct-edit-twice",
            claim_text="First fact.",
            source_slug="s1",
            locator="00:00:01",
        )

    record_edit(
        db,
        note_slug="acct-edit-twice",
        editor="alex",
        before="First fact.",
        after="Second fact.",
        reason="first correction",
        claim_id=claim_id,
    )
    _note, claims = read_note(db, "acct-edit-twice")
    actual_text_after_first_edit = next(c for c in claims if c.id == claim_id).claim_text
    assert actual_text_after_first_edit.startswith("Second fact.")
    assert edit_citation("alex", clock.today()) in actual_text_after_first_edit

    second_edit_id = record_edit(
        db,
        note_slug="acct-edit-twice",
        editor="maintainer",
        before=actual_text_after_first_edit,
        after="Third fact.",
        reason="second correction",
        claim_id=claim_id,
    )

    records = edits_for(db, "acct-edit-twice")
    second_record = next(r for r in records if r.id == second_edit_id)
    assert second_record.before_text == actual_text_after_first_edit


def test_record_edit_on_synthesis_when_no_claim_id(db):
    _make_note(db, "acct-edit-2")
    note_before, _claims = read_note(db, "acct-edit-2")
    version_before = note_before.version
    # A fresh account note's synthesis is never actually `None` once
    # written -- `render_note` fills the "## Synthesis" section with
    # `_default_synthesis` when `note.synthesis` was `None` at write
    # time, and `read_note` parses that rendered text straight back out.
    assert note_before.synthesis == "Acct Edit 2 is an account on file (CRM A-9001)."

    record_edit(
        db,
        note_slug="acct-edit-2",
        editor="alex",
        # Review finding 5: `record_edit` now refuses a `before` that
        # doesn't match what is actually stored -- this must be the real
        # (default-rendered) current synthesis, not the fictional "old
        # synthesis" this test used before that fix, when `before` was
        # never checked at all.
        before=note_before.synthesis,
        after="new synthesis",
        reason="human correction",
    )

    note, _claims = read_note(db, "acct-edit-2")
    # Review finding G11: the synthesis is actually replaced with the new
    # text (not the old one plus a marker), via exactly one `write_note`
    # call -- one new note version per edit, not two.
    assert "new synthesis" in (note.synthesis or "")
    assert edit_citation("alex", clock.today()) in (note.synthesis or "")
    assert note.version == version_before + 1


def test_edits_for_empty_when_none_recorded(db):
    _make_note(db, "acct-no-edits")
    assert edits_for(db, "acct-no-edits") == []


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 5: `_archived_claims_for`'s SELECT omitted
# `claim_citations`, so a second consolidation round re-rendered an
# already-archived claim's line without the corroborating citations it
# picked up before archiving -- the provenance chain (spec sec 4) loses a
# link on the very next archive pass, even though the row survives in
# `claim_citations` itself.
# --------------------------------------------------------------------


def test_archive_note_keeps_a_claims_citations_across_a_second_archive_round(db):
    from portolan.consolidate.archive import commit_archive
    from portolan.search.embed import FakeEmbedder

    with db.cursor() as cur:
        _insert_source(cur, "s-call-a")
        _insert_source(cur, "s-ticket-b")
    _make_note(db, "demo-citations")

    base = datetime(2026, 1, 1, tzinfo=UTC)
    with db.cursor() as cur:
        claim1_id = _insert_claim(
            cur,
            "demo-citations",
            claim_text="claim 0",
            source_slug="s-call-a",
            locator="00:00:00",
            recorded=base,
        )
        # A second sighting of the same fact (a `same` merge verdict,
        # G4) -- recorded as an extra `claim_citations` row, never a
        # second `claims` row.
        cur.execute(
            "INSERT INTO claim_citations (claim_id, source_slug, locator) VALUES (%s, %s, %s)",
            (claim1_id, "s-ticket-b", "msg-3"),
        )

    note, claims = read_note(db, "demo-citations")
    assert claims[0].citations and claims[0].citations[0].source_slug == "s-ticket-b"

    embedder = FakeEmbedder()

    # Round 1: archive claim 1. The freshly-written archive note carries
    # both citation markers.
    commit_archive(db, note, claims, embedder=embedder)
    archive_note, _archive_claims = read_note(db, "demo-citations-archive")
    assert "^[s-call-a @ 00:00:00]" in archive_note.body_md
    assert "^[s-ticket-b @ msg-3]" in archive_note.body_md

    # Round 2: a second, unrelated claim gets archived later. Re-writing
    # the archive note pulls claim 1 back in via `_archived_claims_for` --
    # this must not drop its citation.
    with db.cursor() as cur:
        claim2_id = _insert_claim(
            cur,
            "demo-citations",
            claim_text="claim 1",
            source_slug="s-call-a",
            locator="00:10:00",
            recorded=base + timedelta(days=1),
        )
    note2, claims2 = read_note(db, "demo-citations")
    claim2 = next(c for c in claims2 if c.id == claim2_id)
    commit_archive(db, note2, [claim2], embedder=embedder)

    archive_note_round2, _claims = read_note(db, "demo-citations-archive")
    assert "^[s-call-a @ 00:00:00]" in archive_note_round2.body_md
    assert "^[s-ticket-b @ msg-3]" in archive_note_round2.body_md, (
        "claim 1's corroborating citation must survive a second archive round"
    )
    assert "^[s-call-a @ 00:10:00]" in archive_note_round2.body_md

    with db.cursor() as cur:
        cur.execute(
            "SELECT to_slug FROM links WHERE from_slug = %s AND kind = 'provenance'",
            ("demo-citations-archive",),
        )
        provenance_targets = {row[0] for row in cur.fetchall()}
    assert "s-ticket-b" in provenance_targets

    # Finding 11, same repro: the archive note's own frontmatter `source:`
    # list must name the corroborating source too, not just each claim's
    # primary one.
    assert "  - s-ticket-b" in archive_note_round2.body_md
    assert "  - s-call-a" in archive_note_round2.body_md


# ---------------------------------------------------------------------------
# Superseded claims: archived first, and never drafted from.
# ---------------------------------------------------------------------------


def test_superseded_claims_are_archived_first_and_kept_out_of_the_draft(db):
    from datetime import UTC, datetime

    from portolan.consolidate.archive import select_claims_to_archive
    from portolan.consolidate.draft import _claim_line
    from portolan.notes.models import Claim, NoteRow, NoteStatus, NoteType

    def claim(cid: int, day: int, *, superseded: bool = False) -> Claim:
        return Claim(
            id=cid,
            note_slug="n",
            fact_type="decision",
            claim_text=f"claim {cid}",
            source_slug="s",
            locator=f"p{cid}",
            recorded=datetime(2026, 1, day, tzinfo=UTC),
            superseded_at=datetime(2026, 2, 1, tzinfo=UTC) if superseded else None,
        )

    now = datetime(2026, 3, 1, tzinfo=UTC)
    note = NoteRow(
        slug="n",
        type=NoteType.ACCOUNT,
        title="N",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis="Old state ^[s @ p3] [claim:3].",
    )
    # Claim 3 is newer than 1 and 2 and cited by the synthesis, but superseded.
    claims = [claim(1, 1), claim(2, 2), claim(3, 3, superseded=True), claim(4, 4)]
    assert [c.id for c in select_claims_to_archive(note, claims, threshold=2)] == [3, 1]

    assert _claim_line(claim(4, 4)).startswith("- claim 4 [decision] (2026-01-04): claim 4 ")
