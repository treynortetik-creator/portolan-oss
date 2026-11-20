"""Tests for `portolan.ingest.intake.intake` against the real DB
(`portolan_test_ingest`) and a `LocalStore` on `tmp_path`.
"""

from __future__ import annotations

from datetime import date

from portolan.extract.schema import SourceType
from portolan.ingest.intake import (
    SOURCE_COLUMNS,
    content_hash,
    intake,
    original_key,
    row_to_source_row,
)
from portolan.ingest.models import Participant, RawSource, SourceRow
from portolan.store import LocalStore


def _raw(**overrides) -> RawSource:
    defaults = dict(
        source_id="S-email-999",
        source_type=SourceType.EMAIL,
        source_date=date(2026, 8, 1),
        participants=[Participant(name="Maria Chen", email="mchen@ashgrovehealth.org")],
        text="Hello, this is the message body.",
    )
    defaults.update(overrides)
    return RawSource(**defaults)


def test_intake_inserts_a_pending_source_row(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw()

    row = intake(raw, conn=db, store=store)

    assert isinstance(row, SourceRow)
    assert row.slug == "S-email-999"
    assert row.source_id == "S-email-999"
    assert row.source_type == "email"
    assert row.extraction_status == "pending"
    assert row.filter_reason is None
    assert row.content_hash == content_hash(raw)
    assert row.s3_key == original_key(raw)
    assert row.participants[0].email == "mchen@ashgrovehealth.org"


def test_intake_stores_the_original_bytes(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw()

    row = intake(raw, conn=db, store=store)

    assert store.exists(row.s3_key)
    assert store.get(row.s3_key) == raw.text.encode("utf-8")


def test_intake_stores_original_bytes_when_provided_instead_of_text(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw(
        source_id="S-doc-999",
        source_type=SourceType.DOC,
        original_bytes=b"%PDF-1.4 fake pdf bytes",
        original_content_type="application/pdf",
    )

    row = intake(raw, conn=db, store=store)

    assert store.get(row.s3_key) == b"%PDF-1.4 fake pdf bytes"


def test_intake_writes_row_in_sources_table(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw()
    intake(raw, conn=db, store=store)

    with db.cursor() as cur:
        cur.execute(
            "SELECT slug, extraction_status, content_hash FROM sources WHERE slug = %s",
            (raw.source_id,),
        )
        row = cur.fetchone()

    assert row == (raw.source_id, "pending", content_hash(raw))


# --------------------------------------------------------------------
# `metadata["raw"]` (migration 0009, polish wave 2026-09-12): the
# connector's own `RawSource.metadata` is now stored verbatim, nested
# under a "raw" key alongside the pre-existing `supersedes_source`.
# --------------------------------------------------------------------


def test_intake_stores_the_connectors_raw_metadata(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw(metadata={"message_id": "abc-123", "subject": "Re: renewal"})

    row = intake(raw, conn=db, store=store)

    assert row.metadata["raw"] == {"message_id": "abc-123", "subject": "Re: renewal"}
    assert "supersedes_source" not in row.metadata


def test_intake_with_no_connector_metadata_stores_no_raw_key(db, tmp_path):
    store = LocalStore(tmp_path)
    row = intake(_raw(), conn=db, store=store)
    assert "raw" not in row.metadata


def test_intake_versioned_reingest_keeps_both_metadata_keys(db, tmp_path):
    store = LocalStore(tmp_path)
    v1 = _raw(source_id="S-ticket-950", metadata={"comments": [{"number": 1, "body": "hi"}]})
    intake(v1, conn=db, store=store)

    v2 = _raw(
        source_id="S-ticket-950",
        text="a different body entirely, so the content hash differs",
        metadata={"comments": [{"number": 1, "body": "hi"}, {"number": 2, "body": "bye"}]},
    )
    row_v2 = intake(v2, conn=db, store=store)

    assert row_v2.slug == "S-ticket-950--v2"
    assert row_v2.metadata["supersedes_source"] == "S-ticket-950"
    assert row_v2.metadata["raw"]["comments"][1]["body"] == "bye"


def test_row_to_source_row_and_source_columns_are_public(db, tmp_path):
    """Review slop list item, closed this wave: `portolan.audit.sample`
    used to carry a third copy of this exact column list and row mapper --
    it now imports these public names instead."""
    store = LocalStore(tmp_path)
    intake(_raw(metadata={"link": "https://example.com/post"}), conn=db, store=store)

    with db.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(SOURCE_COLUMNS)} FROM sources WHERE slug = %s",
            ("S-email-999",),
        )
        row = cur.fetchone()

    source_row = row_to_source_row(row)
    assert isinstance(source_row, SourceRow)
    assert source_row.metadata["raw"] == {"link": "https://example.com/post"}


def test_intake_is_idempotent_on_content_hash(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw()

    first = intake(raw, conn=db, store=store)
    second = intake(raw, conn=db, store=store)

    assert first == second

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources WHERE content_hash = %s", (content_hash(raw),))
        (count,) = cur.fetchone()
    assert count == 1


def test_intake_idempotent_re_ingest_does_not_rewrite_the_store(db, tmp_path):
    store = LocalStore(tmp_path)
    raw = _raw()

    first = intake(raw, conn=db, store=store)
    # Mutate the stored original directly; a re-ingest of the same
    # content hash must NOT overwrite it (build item 4: "if a sources row
    # with that hash exists, returns it ... and does not re-store").
    store.put(first.s3_key, b"mutated by the test, should survive", "text/plain")

    intake(raw, conn=db, store=store)

    assert store.get(first.s3_key) == b"mutated by the test, should survive"


def test_intake_different_content_produces_a_different_source(db, tmp_path):
    store = LocalStore(tmp_path)
    raw_a = _raw(source_id="S-email-aaa", text="Message A")
    raw_b = _raw(source_id="S-email-bbb", text="Message B")

    row_a = intake(raw_a, conn=db, store=store)
    row_b = intake(raw_b, conn=db, store=store)

    assert row_a.slug != row_b.slug
    assert row_a.content_hash != row_b.content_hash

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources")
        (count,) = cur.fetchone()
    assert count == 2


def test_intake_same_source_id_changed_content_gets_a_versioned_slug(db, tmp_path):
    # F6: a ticket that gained a comment, an edited doc -- same source_id,
    # changed content, exactly what a re-crawl produces. Used to miss the
    # hash lookup and violate the slug primary key with an uncaught
    # IntegrityError.
    store = LocalStore(tmp_path)
    raw_v1 = _raw(source_id="S-ticket-900", text="Original ticket body.")
    raw_v2 = _raw(source_id="S-ticket-900", text="Original ticket body, now with a new comment.")

    row_v1 = intake(raw_v1, conn=db, store=store)
    row_v2 = intake(raw_v2, conn=db, store=store)

    assert row_v1.slug == "S-ticket-900"
    assert row_v2.slug == "S-ticket-900--v2"
    assert row_v2.metadata["supersedes_source"] == "S-ticket-900"
    assert row_v1.content_hash != row_v2.content_hash

    # Each version's original is stored under its own key -- v2 must not
    # overwrite v1's.
    assert store.get(row_v1.s3_key) == raw_v1.text.encode("utf-8")
    assert store.get(row_v2.s3_key) == raw_v2.text.encode("utf-8")
    assert row_v1.s3_key != row_v2.s3_key

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources WHERE slug LIKE %s", ("S-ticket-900%",))
        (count,) = cur.fetchone()
    assert count == 2


def test_intake_three_versions_chain_supersedes_to_the_immediately_prior_one(db, tmp_path):
    store = LocalStore(tmp_path)
    raw_v1 = _raw(source_id="S-ticket-901", text="v1")
    raw_v2 = _raw(source_id="S-ticket-901", text="v2")
    raw_v3 = _raw(source_id="S-ticket-901", text="v3")

    intake(raw_v1, conn=db, store=store)
    intake(raw_v2, conn=db, store=store)
    row_v3 = intake(raw_v3, conn=db, store=store)

    assert row_v3.slug == "S-ticket-901--v3"
    assert row_v3.metadata["supersedes_source"] == "S-ticket-901--v2"


def test_intake_two_different_source_ids_with_identical_bytes_each_get_their_own_row(db, tmp_path):
    # Bug hunt 2026-09-18, finding 1 (P0): idempotency used to key on
    # content_hash ALONE, so two different sources whose bytes happened to
    # collide (two Zendesk tickets auto-created from the same alert text,
    # the same RSS item mirrored into two feeds) collapsed into one row --
    # the second source never got its own row, original, or facts; its
    # content was silently filed under the FIRST source's slug and date.
    # Fixed: idempotency keys on (source_id, content_hash) together, so
    # two different source_ids with identical bytes each get their own
    # row, own stored original, and own slug.
    store = LocalStore(tmp_path)
    raw_a = _raw(source_id="S-ticket-902", text="same text")
    raw_b = _raw(source_id="S-ticket-902-retry", text="same text")

    row_a = intake(raw_a, conn=db, store=store)
    row_b = intake(raw_b, conn=db, store=store)

    assert row_a != row_b
    assert row_a.slug == "S-ticket-902"
    assert row_b.slug == "S-ticket-902-retry"
    assert row_a.source_id == "S-ticket-902"
    assert row_b.source_id == "S-ticket-902-retry"
    # Same content hash (the whole point of the test), but two distinct
    # rows -- not one row silently reused for the second source.
    assert row_a.content_hash == row_b.content_hash

    # Each source got its own stored original, under its own key.
    assert row_a.s3_key != row_b.s3_key
    assert store.exists(row_a.s3_key)
    assert store.exists(row_b.s3_key)

    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM sources WHERE slug IN (%s, %s)",
            ("S-ticket-902", "S-ticket-902-retry"),
        )
        (count,) = cur.fetchone()
    assert count == 2


def test_intake_same_source_id_unchanged_content_is_still_idempotent_regardless_of_hash_key(
    db, tmp_path
):
    # The other half of finding 1's fix: re-ingesting the SAME source_id
    # with byte-identical content is still a no-op -- keying on
    # (source_id, content_hash) together doesn't regress the ordinary
    # re-sync/retried-webhook case build item 4 always covered.
    store = LocalStore(tmp_path)
    raw = _raw(source_id="S-ticket-902", text="same text")
    same_source_same_text = _raw(source_id="S-ticket-902", text="same text")

    first = intake(raw, conn=db, store=store)
    second = intake(same_source_same_text, conn=db, store=store)

    assert first == second
    assert second.slug == "S-ticket-902"

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM sources WHERE slug = %s", ("S-ticket-902",))
        (count,) = cur.fetchone()
    assert count == 1
