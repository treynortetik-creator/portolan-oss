"""`scripts/data/local_teacher_bakeoff.py`: the document selection and the
per-model numbers the bake-off report's PASS/FAIL table is read from, on a
hand-built two-document run (no model, no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from portolan.extract.schema import Coverage, ExtractionResult, Fact
from portolan.training.records import (
    LabelAudit,
    TrainingFact,
    TrainingRecord,
    repaired_records_path,
    silver_dir,
)
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "local_teacher_bakeoff.py")
_spec = importlib.util.spec_from_file_location("local_teacher_bakeoff_script", _SCRIPT)
bakeoff = importlib.util.module_from_spec(_spec)
sys.modules["local_teacher_bakeoff_script"] = bakeoff
_spec.loader.exec_module(bakeoff)

TEXT = "[L1] The pump alarmed twice overnight. [L2] Please send a loaner unit."


def test_quotas_spread_the_remainder_over_the_first_corpora() -> None:
    assert bakeoff.quotas(30) == {"fda_maude": 8, "cfpb": 8, "enron": 7, "ubuntu_irc": 7}
    assert sum(bakeoff.quotas(5).values()) == 5


def test_selection_is_seeded_and_refuses_a_short_pool() -> None:
    pool = {c: [f"{c}-{i}" for i in range(20)] for c in bakeoff.CORPORA}
    first = bakeoff.select_documents(pool, 30, seed=7)
    assert first == bakeoff.select_documents(pool, 30, seed=7)
    assert first != bakeoff.select_documents(pool, 30, seed=8)
    assert len(first) == len(set(first)) == 30
    with pytest.raises(ValueError, match="need 8"):
        bakeoff.select_documents({**pool, "cfpb": pool["cfpb"][:3]}, 30, seed=7)


def _span(quote: str) -> tuple[int, int]:
    start = TEXT.index(quote)
    return start, start + len(quote)


def _fact(fact_type: str, quote: str, claim: str) -> TrainingFact:
    return TrainingFact(
        fact_type=fact_type, claim=claim, quote=quote, quote_span=_span(quote), locator="L1"
    )


def _record(source_id: str, facts: list[TrainingFact], status: str) -> TrainingRecord:
    audit = LabelAudit(
        status=status,
        teacher_facts=len(facts),
        kept=len(facts),
        regated=0,
        repaired=0,
        withdrawn=0,
        unresolved=0 if status == "complete" else 1,
        unlogged=0,
        audited="2026-09-22",
    )
    return TrainingRecord(
        record_id=source_id,
        corpus="fda_maude",
        source_id=source_id,
        kind="ticket",
        license="public domain",
        text=TEXT,
        facts=facts,
        teachers=["t"],
        agreement=1.0,
        created="2026-09-22",
        mode="single",
        label_audit=audit,
    )


def _write_run(out: Path, run_slug: str, records: list[TrainingRecord]) -> Path:
    var = out / "runs" / run_slug / "var"
    path = repaired_records_path(var, "fda_maude")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))
    return var


def _answer(source_id: str, facts: list[tuple[str, str, str]]) -> dict:
    return ExtractionResult(
        source_id=source_id,
        summary="s",
        entities=[],
        facts=[
            Fact(id=f"f{i}", type=t, claim=c, locator="L1", quote=q, sentiment=0, confidence=0.9)
            for i, (t, q, c) in enumerate(facts)
        ],
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    ).model_dump(mode="json")


def test_run_metrics_and_verdicts_on_a_small_run(tmp_path: Path) -> None:
    docs = [
        {"corpus": "fda_maude", "source_id": sid, "record_id": sid, "system_prompt_sha256": "x"}
        for sid in ("d1", "d2")
    ]
    (tmp_path / "selection.json").write_text(json.dumps({"documents": docs}))
    alarm = "The pump alarmed twice overnight."
    loaner = "Please send a loaner unit."
    _write_run(
        tmp_path,
        "reference",
        [
            _record("d1", [_fact("complaint", alarm, "Alarm twice.")], "complete"),
            _record("d2", [_fact("request", loaner, "Wants a loaner.")], "complete"),
        ],
    )
    var = _write_run(
        tmp_path,
        "cand",
        [
            _record(
                "d1",
                [
                    _fact("complaint", alarm, "The pump alarmed twice."),
                    _fact("request", loaner, "Asks for a loaner."),
                ],
                "complete",
            ),
            _record("d2", [_fact("risk", loaner, "Loaner risk.")], "incomplete"),
        ],
    )
    run_dir = tmp_path / "runs" / "cand"
    calls = [
        # d1: valid first try; its "alarm" quote was paraphrased, the gate resolved it
        {
            "source_id": "d1",
            "calls": 1,
            "lenient": False,
            "result": _answer(
                "d1",
                [
                    ("complaint", "the pump alarmed 2x overnight", "The pump alarmed twice."),
                    ("request", loaner, "Asks for a loaner."),
                ],
            ),
        },
        # d2: needed the lenient parser
        {
            "source_id": "d2",
            "calls": 2,
            "lenient": True,
            "result": _answer("d2", [("risk", loaner, "Loaner risk.")]),
        },
    ]
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    timings = silver_dir(var, "fda_maude") / "timings.jsonl"
    timings.write_text(
        json.dumps({"source_id": "d1", "seconds": 40.0})
        + "\n"
        + json.dumps({"source_id": "d2", "seconds": 20.0})
        + "\n"
    )
    (run_dir / "run.json").write_text(
        json.dumps({"corpora": {"fda_maude": {"repair_seconds": 60.0}}, "peak_memory_bytes": 4e9})
    )

    reference = bakeoff._load_run(tmp_path, "reference", docs)
    m = bakeoff.run_metrics(bakeoff._load_run(tmp_path, "cand", docs), reference)

    assert m["json_valid"] == 0.5 and m["json_valid_first_try"] == 0.5
    assert m["quote_verbatim_kept"] == pytest.approx(2 / 3)
    assert m["quote_verbatim_kept_counts"] == [2, 3, 0]
    assert m["vs_reference"]["matched"] == 1  # d1 complaint; d2's risk has the wrong type
    assert m["vs_reference"]["precision"] == pytest.approx(1 / 3)
    assert m["vs_reference"]["recall"] == 0.5
    assert m["vs_reference_span_only"]["matched"] == 2
    assert m["complete"] == 0.5
    assert m["docs_per_hour"] == pytest.approx(3600 * 2 / 120)
    assert bakeoff.verdicts(m) == {
        "json_valid": "FAIL",
        "quote_verbatim": "FAIL",
        "precision": "FAIL",
        "recall": "FAIL",
        "complete": "FAIL",
        "throughput": "PASS",
    }


def test_review_hits_counts_checked_labels_a_labeler_also_marks() -> None:
    review = [
        {"fact_id": "a", "doc_id": "s1", "char_start": 0, "char_end": 40, "correct": True},
        {"fact_id": "b", "doc_id": "s1", "char_start": 100, "char_end": 120, "correct": False},
        {"fact_id": "c", "doc_id": "s2", "char_start": 10, "char_end": 30, "correct": True},
    ]
    spans = {"s1": [(5, 40), (111, 200)], "s3": [(10, 30)]}
    report = bakeoff.review_hits(review, spans)
    # a: overlap 35 of 35; b: overlap 9 of 20 (< half); c: no span on s2
    assert [p["labeled"] for p in report["per_fact"]] == [True, False, False]
    assert (report["correct_found"], report["correct_total"]) == (1, 2)
    assert (report["wrong_also_labeled"], report["wrong_total"]) == (0, 1)
    assert report["labels_on_these_sources"] == 3


def _judge_row(variant: str, answer: dict | None, *, faithful=True, added=False, keep=True):
    return {
        "variant": variant,
        "true_type": "complaint",
        "emitted_type": "complaint" if variant != "wrong_type" else "risk",
        "expected_faithful": faithful,
        "expected_added_detail": added,
        "expect_type_check": variant != "quote_swapped",
        "expected_keep": keep,
        "answer": answer,
    }


def test_score_judge_counts_each_question_and_the_keep_rule() -> None:
    good = {"faithful": True, "fact_type": "complaint", "added_detail": False}
    rows = [
        _judge_row("gold", good),
        _judge_row("wrong_type", good, keep=False),  # type right, so the gate drops it
        _judge_row("quote_swapped", good, faithful=False, keep=False),  # judged faithful: wrong
        _judge_row("added_detail", None, added=True, keep=False),  # no usable answer
    ]
    report = bakeoff.score_judge(rows)
    assert report["invalid_answers"] == 1
    assert report["faithful"] == 0.5  # gold and wrong_type right; swapped wrong; invalid wrong
    assert report["fact_type"] == pytest.approx(2 / 3)  # three typed rows, one invalid
    assert report["added_detail"] == 0.75
    assert report["keep_agreement"] == 0.5
    assert report["keep_by_variant"] == {
        "gold": 1.0,
        "wrong_type": 1.0,
        "quote_swapped": 0.0,
        "added_detail": 0.0,
    }
