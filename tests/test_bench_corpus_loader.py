"""Tests for `portolan.bench.corpus_loader`: one file of each source type
produces locator-prefixed prepared text."""

from __future__ import annotations

import re
from pathlib import Path

from portolan.bench.corpus_loader import load_prepared_source
from portolan.bench.truth import Manifest
from portolan.extract.schema import SourceType

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"
CORPUS_ROOT = REPO_ROOT / "corpus" / "out"


def _manifest() -> Manifest:
    return Manifest.load(MANIFEST_PATH)


def _first_source_of_type(manifest: Manifest, source_type: str):
    for src in manifest.sources:
        if src.type == source_type:
            return src
    raise AssertionError(f"manifest has no source of type {source_type!r}")


def test_call_text_carries_hhmmss_locators() -> None:
    manifest = _manifest()
    src = _first_source_of_type(manifest, "call")
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    assert prepared.source_type == SourceType.CALL
    first_line = prepared.text.splitlines()[0]
    assert re.match(r"^\[\d{2}:\d{2}:\d{2}\]", first_line)


def test_ticket_text_carries_comment_number_locators() -> None:
    manifest = _manifest()
    src = _first_source_of_type(manifest, "ticket")
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    assert prepared.source_type == SourceType.TICKET
    assert re.search(rf"\[{re.escape(src.id)}#\d+\]", prepared.text)


def test_email_text_carries_message_id_and_paragraph_locators() -> None:
    manifest = _manifest()
    src = _first_source_of_type(manifest, "email")
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    assert prepared.source_type == SourceType.EMAIL
    assert prepared.text.startswith("[msg:")
    assert "[¶1]" in prepared.text


def test_rss_text_carries_url_and_date_locator() -> None:
    manifest = _manifest()
    src = _first_source_of_type(manifest, "rss")
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    assert prepared.source_type == SourceType.RSS
    assert prepared.text.startswith("[url:")
    assert str(prepared.source_date) in prepared.text


def test_crm_text_carries_row_id_locator() -> None:
    manifest = _manifest()
    src = _first_source_of_type(manifest, "crm")
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    assert prepared.source_type == SourceType.CRM
    assert re.search(r"^\[(account|contact|opportunity):[\w-]+\]", prepared.text)


def test_email_paragraph_numbering_matches_manifest_placement() -> None:
    """F28's manifest placement locator is `...#p2`; corpus_loader's own
    paragraph 2 must be that same sentence, so a real extractor's `[¶2]`
    locator lands in the window `graph.py`/`extraction.py` expect."""
    manifest = _manifest()
    src = manifest.source("S-email-052")
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    assert "[¶2] I'm taking over from Tom" in prepared.text


def test_participants_are_rendered_as_display_names() -> None:
    manifest = _manifest()
    src = manifest.source("S-call-003")  # participants: [maria-chen, sam-okafor]
    prepared = load_prepared_source(src, corpus_root=CORPUS_ROOT)
    names = {p.name for p in prepared.participants}
    assert names == {"Maria Chen", "Sam Okafor"}
