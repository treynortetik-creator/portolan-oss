"""Tests for `corpus.generator`. No network. Run with:

uv run --with pytest --with pyyaml --with anthropic pytest tests/test_corpus_generator.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from corpus.generator import manifest as manifest_mod
from corpus.generator import markers, structured
from corpus.generator.briefs import build_briefs, load_truth_table

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"


@pytest.fixture(scope="module")
def truth() -> dict:
    return load_truth_table(TRUTH_TABLE_PATH)


@pytest.fixture(scope="module")
def briefs(truth):
    return build_briefs(truth, mode="manual", reduced=True)


@pytest.fixture(scope="module")
def briefs_by_id(briefs):
    return {b.source_id: b for b in briefs}


# --------------------------------------------------------------------------
# Briefs
# --------------------------------------------------------------------------


def test_every_planted_fact_covers_each_listed_source_exactly_once(truth, briefs_by_id):
    for fact_row in truth["facts"]:
        if fact_row["type"] == "distractor":
            continue
        for src in fact_row["sources"]:
            brief = briefs_by_id[src["id"]]
            matches = [f for f in brief.facts if f.fact_id == fact_row["id"]]
            assert len(matches) == 1, (
                f"{fact_row['id']} should appear exactly once in {src['id']}'s brief, "
                f"found {len(matches)}"
            )


def test_distractor_sources_carry_no_facts(truth, briefs_by_id):
    for fact_row in truth["facts"]:
        if fact_row["type"] != "distractor":
            continue
        for src in fact_row["sources"]:
            brief = briefs_by_id[src["id"]]
            assert brief.facts == []
            assert brief.is_distractor
            assert brief.distractor_id == fact_row["id"]


def test_positions_match_truth_table(truth, briefs_by_id):
    checked = 0
    for fact_row in truth["facts"]:
        if fact_row["type"] == "distractor":
            continue
        for src in fact_row["sources"]:
            if "position" not in src:
                continue
            brief = briefs_by_id[src["id"]]
            plant = next(f for f in brief.facts if f.fact_id == fact_row["id"])
            assert plant.position == src["position"]
            assert plant.position in ("early", "middle", "late")
            checked += 1
    assert checked > 10  # sanity: we actually exercised the position-bearing facts


def test_forbidden_words_present_for_known_tricky_facts(briefs_by_id):
    f01_source = briefs_by_id["S-call-007"]
    f01 = next(f for f in f01_source.facts if f.fact_id == "F01")
    assert "Dayton" in f01.forbidden

    f06_source = briefs_by_id["S-call-013"]
    f06 = next(f for f in f06_source.facts if f.fact_id == "F06")
    assert "risk" in f06.forbidden
    assert "churn" in f06.forbidden

    f17_source = briefs_by_id["S-call-008"]
    f17 = next(f for f in f17_source.facts if f.fact_id == "F17")
    assert "competitor" in f17.forbidden


def test_noise_and_reduced_counts(truth):
    manual_briefs = build_briefs(truth, mode="manual", reduced=True)
    noise_calls = [b for b in manual_briefs if b.source_type == "call" and b.is_noise]
    noise_tickets = [b for b in manual_briefs if b.source_type == "ticket" and b.is_noise]
    noise_emails = [b for b in manual_briefs if b.source_type == "email" and b.is_noise]
    noise_rss = [b for b in manual_briefs if b.source_type == "rss" and b.is_noise]
    assert len(noise_calls) == 2
    assert len(noise_tickets) == 15
    assert len(noise_emails) == 20
    assert len(noise_rss) == 8


# --------------------------------------------------------------------------
# Structured writers
# --------------------------------------------------------------------------


def test_crm_csvs_have_expected_rows(truth, tmp_path):
    counts = structured.write_crm(truth, tmp_path)
    # v1.1 (2026-09-13): +5 accounts (Wrenfield Health/two facilities, Driftwood,
    # Anchor Bay) and +6 contacts (Yolanda Pierce, Marcus Diallo, Sophie Lund,
    # Carlos Nunez, Renee Castillo, Diego Ramos) over v0's 15/12.
    assert counts["accounts"] == 20
    assert counts["contacts"] == 18

    with (tmp_path / "accounts.csv").open(encoding="utf-8") as fh:
        account_rows = list(csv.DictReader(fh))
    assert len(account_rows) == 20
    assert {r["crm_id"] for r in account_rows} >= {"A-1001", "A-1002", "A-1080", "A-1090", "A-1100"}

    with (tmp_path / "contacts.csv").open(encoding="utf-8") as fh:
        contact_rows = list(csv.DictReader(fh))
    assert len(contact_rows) == 18
    emails = {r["email"] for r in contact_rows}
    assert "maria.chen.ohio@gmail.com" not in emails
    names = [r["name"] for r in contact_rows]
    assert names.count("Maria Chen") == 1  # only the CRM contact, not the personal one

    with (tmp_path / "opportunities.csv").open(encoding="utf-8") as fh:
        opp_rows_v1 = list(csv.DictReader(fh))
    ashgrove_opp_v1 = next(r for r in opp_rows_v1 if r["account_crm_id"] == "A-1010")
    assert ashgrove_opp_v1["close_date"] == "2027-03-31"

    with (tmp_path / "opportunities-2026-08-21.csv").open(encoding="utf-8") as fh:
        opp_rows_v2 = list(csv.DictReader(fh))
    ashgrove_opp_v2 = next(r for r in opp_rows_v2 if r["account_crm_id"] == "A-1010")
    assert ashgrove_opp_v2["close_date"] == "2027-06-30"


def test_rss_feed_is_parseable_json(tmp_path):
    items = [{"id": "x", "date": "2026-01-01", "title": "t", "link": "https://x", "summary": "s"}]
    out_path = tmp_path / "feed.json"
    structured.write_rss_feed(items, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == items


def test_ticket_is_parseable_json(tmp_path):
    out_path = tmp_path / "S-ticket-999.json"
    structured.write_ticket(
        ticket_id="S-ticket-999",
        requester_name="Jane Doe",
        account_name="Test Health",
        created="2026-01-01",
        subject="Test",
        comments=[{"number": 1, "author": "Jane Doe", "body": "hello"}],
        out_path=out_path,
    )
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["id"] == "S-ticket-999"
    assert loaded["comments"][0]["number"] == 1


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------


SAMPLE_CALL = """[00:00:00] Sam Okafor: Hey, thanks for hopping on today.
[00:00:05] Maria Chen: Of course, good to catch up.
[00:04:12] Maria Chen: ⟦F11⟧Just so it's on your radar, our HalvardCare contract renews March 31st, 2027.⟦/F11⟧
[00:04:20] Sam Okafor: Good to know, I'll note that.
"""


def test_process_call_records_locator_and_strips_markers():
    clean_text, placements = markers.process_call(SAMPLE_CALL)
    assert len(placements) == 1
    p = placements[0]
    assert p.fact_id == "F11"
    assert p.locator == "00:04:12"
    assert p.line_or_comment == "3"
    assert "renews March 31st, 2027" in p.excerpt
    assert "⟦" not in clean_text and "⟧" not in clean_text
    assert "renews March 31st, 2027" in clean_text


def test_process_ticket_comments_records_locator_and_strips_markers():
    comments = [
        {"number": 1, "author": "Felix Arden", "body": "Normal comment, no fact here."},
        {
            "number": 2,
            "author": "Felix Arden",
            "body": "⟦F23⟧Three pumps threw false alarms.⟦/F23⟧",
        },
    ]
    clean, placements = markers.process_ticket_comments("S-ticket-024", comments)
    assert len(placements) == 1
    assert placements[0].locator == "S-ticket-024#2"
    assert placements[0].line_or_comment == "2"
    assert "⟦" not in clean[1]["body"]
    assert "Three pumps threw false alarms." in clean[1]["body"]


def test_process_email_body_records_paragraph_locator_and_strips_markers():
    body = (
        "Hi Sam,\n\n"
        "⟦F03⟧Our delivery arrived three days late and we borrowed from Larkspur.⟦/F03⟧\n\n"
        "Best,\nAisha"
    )
    clean, placements = markers.process_email_body("S-email-052", body)
    assert len(placements) == 1
    assert placements[0].locator == "S-email-052@halvardmedical.com#p2"
    assert placements[0].line_or_comment == "2"
    assert "⟦" not in clean


def test_verify_exactly_one_reports_missing_and_unexpected():
    placements = [markers.Placement("F01", "00:00:01", "1", "excerpt")]
    problems = markers.verify_exactly_one(["F01", "F02"], placements)
    assert any("missing marker for F02" in p for p in problems)

    problems2 = markers.verify_exactly_one(["F01"], [*placements, placements[0]])
    assert any("2 markers for F01" in p for p in problems2)


def test_verify_exactly_one_allows_multi_mention_facts():
    # F04 is deliberately planted twice in S-call-011 (two mentions of the
    # same claim); allow_multiple should let that through with zero
    # problems, while a genuinely missing fact is still flagged.
    placements = [
        markers.Placement("F04", "00:10:00", "10", "first mention"),
        markers.Placement("F04", "00:25:00", "40", "second mention"),
    ]
    problems = markers.verify_exactly_one(
        ["F04", "F15b"], placements, allow_multiple=frozenset({"F04"})
    )
    assert any("missing marker for F15b" in p for p in problems)
    assert not any("F04" in p for p in problems)


# --------------------------------------------------------------------------
# Manifest validation
# --------------------------------------------------------------------------


def _tiny_truth_table() -> dict:
    return {
        "facts": [
            {
                "id": "F01",
                "type": "complaint",
                "sources": [{"id": "S-call-001", "type": "call"}],
            },
            {
                "id": "F02",
                "type": "complaint",
                "sources": [
                    {"id": "S-ticket-001", "type": "ticket"},
                    {"id": "S-call-001", "type": "call"},
                ],
            },
            {"id": "D01", "type": "distractor", "sources": [{"id": "S-rss-001", "type": "rss"}]},
        ]
    }


def _tiny_valid_manifest() -> dict:
    return {
        "sources": [
            {"id": "S-call-001", "type": "call"},
            {"id": "S-ticket-001", "type": "ticket"},
            {"id": "S-rss-001", "type": "rss"},
        ],
        "facts": {
            "F01": [{"source_id": "S-call-001", "locator": "00:04:12"}],
            "F02": [
                {"source_id": "S-ticket-001", "locator": "S-ticket-001#1"},
                {"source_id": "S-call-001", "locator": "00:10:00"},
            ],
        },
    }


def test_manifest_validate_passes_on_correct_manifest():
    problems = manifest_mod.validate(_tiny_valid_manifest(), _tiny_truth_table())
    assert problems == []


def test_manifest_validate_fails_on_missing_placement():
    manifest = _tiny_valid_manifest()
    manifest["facts"]["F02"] = [{"source_id": "S-ticket-001", "locator": "S-ticket-001#1"}]
    problems = manifest_mod.validate(manifest, _tiny_truth_table())
    assert any("F02" in p for p in problems)


def test_manifest_validate_fails_when_distractor_has_a_placement():
    manifest = _tiny_valid_manifest()
    manifest["facts"]["D01"] = [{"source_id": "S-rss-001", "locator": "url:https://x 2026-01-01"}]
    problems = manifest_mod.validate(manifest, _tiny_truth_table())
    assert any("distractor" in p for p in problems)


def test_manifest_validate_fails_on_bad_locator_format():
    manifest = _tiny_valid_manifest()
    manifest["facts"]["F01"][0]["locator"] = "not-a-timestamp"
    problems = manifest_mod.validate(manifest, _tiny_truth_table())
    assert any("does not match" in p for p in problems)
