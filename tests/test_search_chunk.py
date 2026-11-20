"""Tests for portolan/search/chunk.py.

No database needed -- pure text in, list[Chunk] out.
"""

from __future__ import annotations

from portolan.search.chunk import chunk_note


def test_title_and_description_are_always_chunk_zero():
    chunks = chunk_note(
        "Some body text here.",
        title="Ashgrove Medical Center",
        description="A hospital account in the Midwest.",
    )
    assert chunks[0].is_title is True
    assert "Ashgrove Medical Center" in chunks[0].text
    assert "hospital account" in chunks[0].text
    assert chunks[0].heading is None


def test_no_title_means_no_synthetic_chunk_zero():
    chunks = chunk_note("Some body text here.")
    assert all(not c.is_title for c in chunks)
    assert chunks[0].text == "Some body text here."


def test_title_only_note_with_empty_body_still_yields_chunk_zero():
    chunks = chunk_note("", title="Empty Note")
    assert len(chunks) == 1
    assert chunks[0].is_title is True
    assert chunks[0].text == "Empty Note"


def test_splits_on_headings_and_records_nearest_heading():
    body = (
        "Intro paragraph before any heading.\n\n"
        "# Deliveries\n\n"
        "Ashgrove received a late shipment.\n\n"
        "## Follow up\n\n"
        "We called the branch and they apologized."
    )
    chunks = chunk_note(body)
    headings = [c.heading for c in chunks]
    assert headings == [None, "Deliveries", "Follow up"]
    assert "Intro paragraph" in chunks[0].text
    assert "Ashgrove received" in chunks[1].text
    assert "We called" in chunks[2].text


def test_chunk_offsets_point_at_the_right_source_span():
    body = "# Deliveries\n\nAshgrove received a late shipment.\n\nSecond paragraph follows here."
    chunks = chunk_note(body)
    for c in chunks:
        # start/end must span exactly from the first char of the chunk's
        # content to the last -- whitespace-normalized comparison, since a
        # chunk packing >1 paragraph joins them without reproducing the
        # blank line that separated them in body_md (a cosmetic join
        # detail; the offsets themselves are exact boundaries).
        raw = body[c.start : c.end]
        assert " ".join(raw.split()) == " ".join(c.text.split())


def test_two_paragraphs_under_one_heading_pack_into_one_chunk_when_small():
    body = "# Notes\n\nFirst short paragraph.\n\nSecond short paragraph."
    chunks = chunk_note(body, max_chars=1200)
    assert len(chunks) == 1
    assert "First short paragraph" in chunks[0].text
    assert "Second short paragraph" in chunks[0].text


def test_respects_max_chars_by_splitting_paragraphs_into_separate_chunks():
    # Two paragraphs that together exceed max_chars must land in separate chunks.
    para_a = "Alpha sentence. " * 10  # ~160 chars
    para_b = "Bravo sentence. " * 10  # ~160 chars
    body = f"{para_a}\n\n{para_b}"
    chunks = chunk_note(body, max_chars=200)
    assert len(chunks) == 2
    assert "Alpha" in chunks[0].text and "Bravo" not in chunks[0].text
    assert "Bravo" in chunks[1].text and "Alpha" not in chunks[1].text
    for c in chunks:
        assert len(c.text) <= 200


def test_never_splits_mid_sentence_when_a_sentence_boundary_is_available():
    # One paragraph, several short sentences, forced to split by a tight max_chars.
    para = "One. Two. Three. Four. Five. Six. Seven. Eight."
    chunks = chunk_note(para, max_chars=15)
    for c in chunks:
        stripped = c.text.strip()
        # Every chunk should end at a sentence boundary (a period), never mid-word.
        assert stripped.endswith(".")


def test_oversized_single_sentence_falls_back_to_hard_character_split():
    # No punctuation at all -- there is no sentence boundary to respect, so
    # this must fall back to a raw character split rather than looping forever
    # or exceeding max_chars.
    para = "word " * 50  # ~250 chars, no periods
    chunks = chunk_note(para, max_chars=40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 40


def test_token_estimate_is_positive_and_roughly_chars_over_four():
    chunks = chunk_note("a" * 400)
    assert chunks[0].token_estimate == 100


def test_chunk_note_is_pure_and_deterministic():
    body = "# H\n\nSome text.\n\nMore text here."
    first = chunk_note(body, title="T", description="D")
    second = chunk_note(body, title="T", description="D")
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
