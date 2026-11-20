"""Tests for `portolan.context` (owner decision 2026-09-17, "Company
context page"): the `context` note-type closed-list entry, the seed
step, the loaders, and the two renderers every prompt call site shares.
"""

from __future__ import annotations

import logging

import psycopg
import pytest

from portolan import context as context_module
from portolan.context import (
    ACCOUNT_CONTEXT_PARAGRAPH_CAP,
    COMPANY_CONTEXT_BODY_CHAR_CAP,
    COMPANY_CONTEXT_SLUG,
    HALVARD_MEDICAL_CONTEXT_BODY,
    build_account_context,
    first_paragraph,
    load_company_context,
    render_account_context_block,
    render_company_context_block,
    seed_company_context,
)
from portolan.extract.schema import EntityKind, RegistryContextEntry
from portolan.notes.models import NoteRow, NoteStatus, NoteType
from portolan.notes.persist import read_note, write_note
from portolan.util import clock

# ---------------------------------------------------------------------
# The closed type list.
# ---------------------------------------------------------------------


def test_context_is_in_the_note_type_enum() -> None:
    assert NoteType.CONTEXT.value == "context"
    assert "context" in {t.value for t in NoteType}


def test_notes_table_accepts_type_context_and_rejects_a_bogus_type(db: psycopg.Connection) -> None:
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description) VALUES (%s, 'context', %s, %s)",
            ("a-context-note", "A context note", "d"),
        )
    with db.cursor() as cur, pytest.raises(psycopg.errors.CheckViolation):
        cur.execute(
            "INSERT INTO notes (slug, type, title, description) "
            "VALUES (%s, 'not-a-real-type', %s, %s)",
            ("a-bogus-note", "A bogus note", "d"),
        )
    db.rollback()


# ---------------------------------------------------------------------
# Seed.
# ---------------------------------------------------------------------


def test_halvard_medical_context_body_is_under_the_cap() -> None:
    assert len(HALVARD_MEDICAL_CONTEXT_BODY) <= COMPANY_CONTEXT_BODY_CHAR_CAP


def test_seed_company_context_creates_the_note_once_and_is_idempotent(
    db: psycopg.Connection,
) -> None:
    created_first = seed_company_context(db)
    assert created_first is True

    note, _claims = read_note(db, COMPANY_CONTEXT_SLUG)
    assert note.type == NoteType.CONTEXT
    assert note.title == "Company context"
    assert note.synthesis == HALVARD_MEDICAL_CONTEXT_BODY
    first_version = note.version

    created_second = seed_company_context(db)
    assert created_second is False

    note_after, _claims_after = read_note(db, COMPANY_CONTEXT_SLUG)
    assert note_after.version == first_version  # never rewritten by a second seed


def test_seed_company_context_never_overwrites_a_human_edit(db: psycopg.Connection) -> None:
    seed_company_context(db)
    note, claims = read_note(db, COMPANY_CONTEXT_SLUG)
    note.synthesis = "A human rewrote this."
    write_note(db, note, claims, changed_by="a-human", reason="test edit")

    seed_company_context(db)  # must be a no-op

    note_after, _claims_after = read_note(db, COMPANY_CONTEXT_SLUG)
    assert note_after.synthesis == "A human rewrote this."


def test_seed_company_context_rejects_an_over_cap_body(db: psycopg.Connection) -> None:
    with pytest.raises(ValueError, match="over the"):
        seed_company_context(db, body="x" * (COMPANY_CONTEXT_BODY_CHAR_CAP + 1))


# ---------------------------------------------------------------------
# load_company_context.
# ---------------------------------------------------------------------


def test_load_company_context_missing_returns_none_and_logs_one_warning(
    db: psycopg.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=context_module.__name__):
        result = load_company_context(db)
    assert result is None
    assert any("not found" in record.getMessage() for record in caplog.records)


def test_load_company_context_returns_text_and_updated_when_present(db: psycopg.Connection) -> None:
    seed_company_context(db)
    result = load_company_context(db)
    assert result is not None
    assert result.text == HALVARD_MEDICAL_CONTEXT_BODY
    note, _claims = read_note(db, COMPANY_CONTEXT_SLUG)
    assert result.updated == note.updated


def test_load_company_context_empty_body_returns_none(db: psycopg.Connection) -> None:
    # `write_note` always falls back to a non-empty default synthesis
    # (description, then title) when `note.synthesis` is falsy -- so a
    # genuinely empty `## Synthesis` section only happens via a body_md
    # written some other way (a hand rebuild, an old schema version).
    # Built directly here to exercise that path.
    now = clock.now()
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, created, updated, body_md)
            VALUES (%s, 'context', 'Company context', 'd', %s, %s, %s)
            """,
            (
                COMPANY_CONTEXT_SLUG,
                now,
                now,
                '---\ntitle: "Company context"\n---\n\n## Synthesis\n\n'
                "<!-- CLAIMS:APPEND-ONLY -->\n\n## Claims\n\nNo claims recorded yet.\n",
            ),
        )
    assert load_company_context(db) is None


# ---------------------------------------------------------------------
# Renderers.
# ---------------------------------------------------------------------


def test_render_company_context_block_has_the_heading() -> None:
    block = render_company_context_block("Halvard Medical sells pumps.")
    assert block.startswith("## Company context\n")
    assert "Halvard Medical sells pumps." in block


def test_first_paragraph_stops_at_the_blank_line_and_caps_length() -> None:
    assert first_paragraph("one\n\ntwo", 100) == "one"
    long_para = "x" * 500
    capped = first_paragraph(long_para, ACCOUNT_CONTEXT_PARAGRAPH_CAP)
    assert len(capped) == ACCOUNT_CONTEXT_PARAGRAPH_CAP + 1  # +1 for the trailing ellipsis
    assert capped.endswith("…")


# ---------------------------------------------------------------------
# Account context.
# ---------------------------------------------------------------------


def test_build_account_context_includes_registry_line_only_when_no_note_exists(
    db: psycopg.Connection,
) -> None:
    entries = [
        RegistryContextEntry(
            name="Ashgrove Medical Center",
            kind=EntityKind.ACCOUNT,
            registry_id="A-1011",
            aliases=["AMC"],
        )
    ]
    out = build_account_context(db, entries)
    assert out == {"A-1011": "Ashgrove Medical Center (account, id=A-1011), aliases: AMC"}


def test_build_account_context_folds_in_the_capped_first_paragraph_of_the_account_note(
    db: psycopg.Connection,
) -> None:
    now = clock.now()
    note = NoteRow(
        slug="ashgrove-medical-center",
        type=NoteType.ACCOUNT,
        title="Ashgrove Medical Center",
        description="A facility account.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        crm_id="A-1011",
        synthesis="Ashgrove Medical Center is a large hospital.\n\nA second paragraph.",
    )
    write_note(db, note, [], changed_by="test", reason="setup")
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO registry_entities (kind, canonical_name, crm_id, note_slug) "
            "VALUES ('account', 'Ashgrove Medical Center', 'A-1011', 'ashgrove-medical-center')"
        )

    entries = [
        RegistryContextEntry(
            name="Ashgrove Medical Center", kind=EntityKind.ACCOUNT, registry_id="A-1011"
        )
    ]
    out = build_account_context(db, entries)
    assert out["A-1011"] == (
        "Ashgrove Medical Center (account, id=A-1011)\nAshgrove Medical Center is a large hospital."
    )
    assert "second paragraph" not in out["A-1011"]


def test_build_account_context_skips_non_account_entries(db: psycopg.Connection) -> None:
    entries = [
        RegistryContextEntry(
            name="Tidewater T2", kind=EntityKind.PRODUCT, registry_id="tidewater-t2"
        )
    ]
    assert build_account_context(db, entries) == {}


def test_render_account_context_block_empty_when_no_entries() -> None:
    assert render_account_context_block({}) == ""


def test_render_account_context_block_joins_entries() -> None:
    block = render_account_context_block({"a": "line a", "b": "line b"})
    assert block.startswith("## Account context\n\n")
    assert "line a" in block
    assert "line b" in block
