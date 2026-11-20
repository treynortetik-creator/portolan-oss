"""WP9: the v2 world reader, the span scorer and `extract --world` (fake
backend). Fixture: `tests/bench_world_fixture.py`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portolan.bench.__main__ import main
from portolan.bench.world_score import (
    perfect_world_results,
    quote_tier,
    score_world,
    spans_match,
)
from portolan.bench.world_v2 import load_world, locator_region, noise_type
from tests.bench_world_fixture import (
    CALL_ID,
    CALL_TEXT,
    CRM_ID,
    EMAIL_ID,
    build_world,
    scripted_results,
)

# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def test_reader_loads_sources_labels_decoys_and_counts_exclusions(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    assert world.world_id == "world-test-7"
    assert world.input_format == "sources.jsonl"
    assert sorted(world.sources) == sorted([CALL_ID, CRM_ID, EMAIL_ID])
    assert len(world.labels) == 7
    assert world.excluded_labels == {"span_text_mismatch": 1, "source_missing": 1}
    assert [d.decoy_id for d in world.decoys] == ["DC1"]
    for label in world.labels:
        assert world.sources[label.doc_id].text[label.char_start : label.char_end] == label.text


def test_reader_keeps_wp4_wp5_wp6_extra_fields(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    f1 = next(lab for lab in world.labels if lab.fact_id == "F1")
    assert f1.label_source == "scripted_writer"
    assert f1.noise == ["asr"]
    assert f1.model_extra["asr_in_span"] is True
    assert f1.model_extra["writer_model"] == "qwen/qwen3-32b"
    silver = [lab for lab in world.labels if lab.label_source == "silver_completeness"]
    assert {lab.fact_id for lab in silver} == {"D0001-C1"}
    assert any(lab.model_extra.get("restates_fact_id") == "D0001-C1" for lab in silver)
    crm = [lab for lab in world.labels if lab.doc_id == CRM_ID]
    assert {lab.label_source for lab in crm} == {"scripted_code"}
    assert world.sources[CALL_ID].noise == ["asr"]


def test_directive_and_reference_mode_come_from_label_then_world_yaml(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    by_id = {lab.fact_id: lab for lab in world.labels}
    assert world.directive_for(by_id["F1"]) == "colloquial"
    assert world.directive_for(by_id["F2"]) == "hedged"  # label wins over world.yaml
    assert world.directive_for(by_id["D0001-C1"]) == "none"
    assert world.reference_mode_for(by_id["F2"]) == "pronoun"
    assert world.reference_mode_for(by_id["F5"]) == "first_name"
    assert world.products == ["Tidewater", "TW"]
    assert world.accounts["A-marrow"].name == "Marrow General"


def test_noise_type_folds_dicts_and_variants() -> None:
    assert noise_type({"type": "asr", "params": {}}) == "asr"
    assert noise_type("partial_or_garbled:mojibake") == "partial_or_garbled"
    assert noise_type("") is None


def test_reader_docs_path_dates_from_manifest_and_skips_undated(tmp_path: Path) -> None:
    root = tmp_path / "w"
    (root / "docs").mkdir(parents=True)
    for plan_id, sid in (("D1", CALL_ID), ("D2", "S-call-undated")):
        (root / "docs" / f"{plan_id}.json").write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "kind": "call",
                    "status": "written",
                    "prepared_sources": [
                        {"source_id": sid, "source_type": "call", "path": "c", "text": CALL_TEXT}
                    ],
                    "noise": [{"type": "asr", "params": {}}],
                    "true_account_id": "A-true",
                    "filed_account_id": "A-filed",
                }
            ),
            encoding="utf-8",
        )
    (root / "docs" / "D3.json").write_text(
        json.dumps({"plan_id": "D3", "kind": "email", "status": "dropped"}), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps({"sources": [{"id": "x", "plan_id": "D1", "date": "2026-05-04"}]}),
        encoding="utf-8",
    )
    start = CALL_TEXT.index("The Tidewater")
    (root / "labels.jsonl").write_text(
        json.dumps(
            {
                "fact_id": "F1",
                "fact_type": "complaint",
                "doc_id": CALL_ID,
                "char_start": start,
                "char_end": start + 13,
                "text": "The Tidewater",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    world = load_world(root)
    assert world.input_format == "docs"
    assert list(world.sources) == [CALL_ID]
    assert world.sources[CALL_ID].date.isoformat() == "2026-05-04"
    assert world.sources[CALL_ID].noise == ["asr"]
    # A misfiled document reaches the extractor under the account it was filed to.
    assert world.sources[CALL_ID].account_id == "A-filed"
    assert "S-call-undated" in world.skipped_sources
    assert world.skipped_sources["D3"] == "doc status 'dropped'"
    assert len(world.labels) == 1


def test_reader_requires_labels(tmp_path: Path) -> None:
    (tmp_path / "sources.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_world(tmp_path)


# --------------------------------------------------------------------------
# Span helpers
# --------------------------------------------------------------------------


def test_spans_match_iou_or_containment() -> None:
    assert spans_match((0, 10), (2, 11))  # IoU 8/11
    assert spans_match((0, 100), (40, 45))  # containment
    assert not spans_match((0, 10), (10, 20))  # touching, no overlap
    assert not spans_match((0, 10), (7, 30))  # IoU 3/30, no containment


def test_quote_tier_exact_normalized_missing_unresolved() -> None:
    assert quote_tier("Morning all.", CALL_TEXT)[0] == "exact"
    assert quote_tier("We’ll ship the replacement", CALL_TEXT)[0] == "normalized"
    assert quote_tier(None, CALL_TEXT) == ("missing", None)
    assert quote_tier("never said", CALL_TEXT) == ("unresolved", None)


def test_locator_region_reads_the_injected_marker(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    email = world.sources[EMAIL_ID]
    start, end = locator_region(email, "msg-1@mail.vendor.test#p2")
    assert email.text[start:end].endswith("by Tuesday?")
    assert locator_region(email, "msg-9#p1") is None


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


def test_scripted_output_lands_in_the_expected_categories(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    score = score_world(world, scripted_results(), backend="stub")
    cats = {(p.source_id, p.produced_id): p.category for p in score.produced}
    assert cats == {
        (CALL_ID, "f1"): "matched",
        (CALL_ID, "f2"): "matched",
        (CALL_ID, "f3"): "decoy_hit",
        (CALL_ID, "f4"): "off_span",
        (CALL_ID, "f5"): "off_span",
        (CRM_ID, "f1"): "type_confusion",
        (CRM_ID, "f2"): "matched",
        (EMAIL_ID, "f1"): "matched",
    }
    email = next(p for p in score.produced if p.source_id == EMAIL_ID)
    assert email.span_basis == "locator"
    decoy = next(p for p in score.produced if p.category == "decoy_hit")
    assert decoy.decoy_class_id == "hypothetical"
    assert score.type_confusions == {"pricing -> question": 1}


def test_three_numbers_overall_without_judges(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    o = score_world(world, scripted_results(), backend="stub").overall
    assert (o.produced, o.quote_exact, o.quote_normalized) == (8, 5, 1)
    assert (o.quote_unresolved, o.quote_missing) == (1, 1)
    assert o.quote_verbatim == 6 / 8
    assert (o.matched, o.judge_supported) == (4, 0)
    assert o.substantiated == 4 / 8
    assert (o.labeled, o.labeled_found, o.labeled_confused) == (6, 4, 1)
    assert o.omission == 2 / 6
    assert (o.decoy_hits, o.type_confusions, o.off_span) == (1, 1, 2)
    assert o.invention == 3 / 8


def test_judge_decisions_move_off_span_into_substantiated(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    decisions = {
        (CALL_ID, "f3"): "not_supported",
        (CALL_ID, "f4"): "supported",
        (CALL_ID, "f5"): "not_supported",
    }
    o = score_world(world, scripted_results(), backend="stub", judge_decisions=decisions).overall
    assert (o.judge_supported, o.off_span_overturned, o.decoy_overturned) == (1, 1, 0)
    assert o.substantiated == 5 / 8
    assert o.invention == 2 / 8


def test_breakdowns_by_label_source_directive_noise_and_kind(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    b = score_world(world, scripted_results(), backend="stub").breakdowns
    ls = b["label_source"]
    assert (ls["scripted_writer"].labeled, ls["scripted_writer"].labeled_found) == (3, 3)
    assert (ls["silver_completeness"].labeled, ls["silver_completeness"].labeled_found) == (1, 0)
    assert (ls["scripted_code"].labeled, ls["scripted_code"].labeled_confused) == (2, 1)
    assert ls["unlabeled"].produced == 3  # decoy hit + two off-span
    assert b["directive"]["hedged"].labeled_found == 1
    assert b["directive"]["none"].labeled == 3
    assert b["reference_mode"]["pronoun"].labeled == 1
    assert (b["noise"]["asr"].labeled, b["noise"]["asr"].produced) == (3, 5)
    assert b["noise"]["forwarded_or_quoted"].labeled_found == 1
    assert b["noise"]["none"].produced == 2
    assert set(b["doc_kind"]) == {"call", "crm_account", "email"}
    ft = b["fact_type"]
    assert (ft["pricing"].labeled, ft["pricing"].labeled_found) == (2, 0)
    assert (ft["competitor_mention"].produced, ft["competitor_mention"].decoy_hits) == (2, 1)


def test_perfect_results_score_perfectly(tmp_path: Path) -> None:
    world = load_world(build_world(tmp_path / "w"))
    o = score_world(world, perfect_world_results(world), backend="fake").overall
    assert o.labeled == 6 and o.produced == 6
    assert o.quote_verbatim == 1.0 and o.substantiated == 1.0 and o.omission == 0.0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_extract_world_fake_backend_end_to_end(tmp_path: Path) -> None:
    world_dir = build_world(tmp_path / "w")
    out = tmp_path / "out"
    code = main(
        [
            "extract",
            "--world",
            str(world_dir),
            "--backend",
            "fake",
            "--out",
            str(out),
            "--spend-path",
            str(tmp_path / "spend.json"),
        ]
    )
    assert code == 0
    report = json.loads(next(out.glob("world-bench-*-fake.json")).read_text(encoding="utf-8"))
    assert report["overall"]["labeled"] == 6
    assert report["overall_rates"]["quote_verbatim"] == 1.0
    assert report["overall_rates"]["omission"] == 0.0
    assert report["labels_excluded"] == {"source_missing": 1, "span_text_mismatch": 1}
    md = next(out.glob("world-bench-*-fake.md")).read_text(encoding="utf-8")
    assert "Quote-verbatim**: 100.0% (6/6)" in md
    assert "## By directive (label side)" in md
    assert not (tmp_path / "spend.json").exists()  # a fake run spends nothing
