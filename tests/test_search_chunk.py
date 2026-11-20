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


# ---------------------------------------------------------------------
# What a note body holds that is not something to retrieve: the YAML
# header `write_note` renders, link-only sections, HTML comments.
# ---------------------------------------------------------------------

NOTE_BODY = (
    "---\n"
    'title: "Pricing — Ashgrove Health"\n'
    "type: signal\n"
    "status: living\n"
    "source:\n"
    "  - S-call-005\n"
    "---\n"
    "\n"
    "## Synthesis\n"
    "\n"
    "Ashgrove wants a volume discount.\n"
    "\n"
    "## Related\n"
    "\n"
    "- [[ashgrove-health]]\n"
    "- [[tidewater-t2]]\n"
    "\n"
    "<!-- CLAIMS:APPEND-ONLY -->\n"
    "\n"
    "## Claims\n"
    "\n"
    "- **2026-06-09** | pricing | Ashgrove expects a 12 percent discount. "
    "([[ashgrove-health]]) ^[S-call-005 @ 00:08:54]\n"
)


def test_the_yaml_header_is_not_a_chunk():
    chunks = chunk_note(NOTE_BODY, title="Pricing — Ashgrove Health", description="Pricing.")
    assert chunks[0].is_title
    assert not any("status: living" in c.text or c.text.startswith("---") for c in chunks)


def test_a_section_holding_only_links_and_comments_is_not_a_chunk():
    chunks = chunk_note(NOTE_BODY)
    assert [c.heading for c in chunks] == ["Synthesis", "Claims"]
    assert not any("CLAIMS:APPEND-ONLY" in c.text for c in chunks)


def test_offsets_still_point_into_the_body_after_the_header_is_skipped():
    for c in chunk_note(NOTE_BODY):
        assert " ".join(NOTE_BODY[c.start : c.end].split()) == " ".join(c.text.split())


def test_a_rule_line_in_the_body_is_not_mistaken_for_a_header():
    body = "First paragraph.\n\n---\n\nkey: value\n\n---\n\nLast paragraph."
    text = " ".join(c.text for c in chunk_note(body))
    assert "First paragraph." in text and "key: value" in text and "Last paragraph." in text


def test_an_unclosed_header_is_kept_as_text():
    chunks = chunk_note("---\ntitle: x\n\nBody text with no closing rule.")
    assert any("Body text" in c.text for c in chunks)
    assert any("title: x" in c.text for c in chunks)


def test_a_long_list_splits_between_items_never_inside_one():
    items = [
        f"- **2026-06-{day:02d}** | pricing | Account {day} asked for a discount. It said "
        f'"we need twelve points off. No less." ^[S-call-{day:03d} @ 00:01:00]'
        for day in range(1, 13)
    ]
    chunks = chunk_note("## Claims\n\n" + "\n".join(items) + "\n", max_chars=400)
    assert len(chunks) > 1
    seen: list[str] = []
    for c in chunks:
        assert len(c.text) <= 400
        lines = c.text.splitlines()
        assert all(line.startswith("- **2026-06-") for line in lines)
        assert all(line.endswith("]") for line in lines)
        seen.extend(lines)
    assert seen == items
