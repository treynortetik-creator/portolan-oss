"""Tests for `portolan.training.records` -- `TrainingRecord` schema
round-trip, JSONL read/write, and `compute_stats` (silver-labels build
instructions item 3 and item 6: "record schema round-trip")."""

from __future__ import annotations

from pathlib import Path

import pytest

from portolan.training.records import (
    DroppedFact,
    TrainingFact,
    TrainingRecord,
    append_dropped,
    append_record,
    compute_stats,
    dropped_by_corpus,
    dropped_path,
    failures_path,
    iter_all_dropped,
    iter_all_records,
    progress_path,
    read_dropped,
    read_records,
    records_path,
    silver_dir,
)


def _record(
    record_id: str = "S-1",
    corpus: str = "fda_maude",
    source_id: str = "S-1",
    facts: list[dict] | None = None,
    teachers: list[str] | None = None,
    agreement: float = 1.0,
) -> TrainingRecord:
    return TrainingRecord(
        record_id=record_id,
        corpus=corpus,
        source_id=source_id,
        kind="ticket",
        license="public domain",
        text="[L1] the device malfunctioned during use",
        facts=facts or [],
        teachers=teachers or ["qwen/qwen3.8-27b:free", "deepseek/deepseek-v4-flash-0731:free"],
        agreement=agreement,
        created="2026-09-21T00:00:00-07:00",
    )


# --------------------------------------------------------------------------
# schema round-trip
# --------------------------------------------------------------------------


def test_training_record_round_trips_through_json() -> None:
    fact = TrainingFact(
        fact_type="complaint",
        claim="device malfunctioned",
        quote="the device malfunctioned during use",
        locator="L1",
        entities=["Acme Devices"],
    )
    record = _record(facts=[fact])
    raw = record.model_dump_json()
    restored = TrainingRecord.model_validate_json(raw)
    assert restored == record
    assert restored.facts[0].fact_type == "complaint"
    assert restored.facts[0].entities == ["Acme Devices"]


def test_training_record_is_negative_when_no_facts() -> None:
    assert _record(facts=[]).is_negative is True
    fact = TrainingFact(fact_type="complaint", claim="x", quote="q", locator="L1")
    assert _record(facts=[fact]).is_negative is False


def test_training_record_rejects_unknown_field() -> None:
    import pytest
    from pydantic import ValidationError

    data = _record().model_dump()
    data["unexpected_field"] = "nope"
    with pytest.raises(ValidationError):
        TrainingRecord.model_validate(data)


def test_training_record_rejects_agreement_out_of_range() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _record(agreement=1.5)


# --------------------------------------------------------------------------
# JSONL read/write
# --------------------------------------------------------------------------


def test_append_and_read_records_round_trip(tmp_path: Path) -> None:
    path = records_path(tmp_path, "fda_maude")
    r1 = _record(record_id="S-1", source_id="S-1")
    r2 = _record(record_id="S-2", source_id="S-2")
    append_record(path, r1)
    append_record(path, r2)

    restored = list(read_records(path))
    assert restored == [r1, r2]


def test_read_records_missing_file_yields_nothing(tmp_path: Path) -> None:
    path = records_path(tmp_path, "fda_maude")
    assert list(read_records(path)) == []


def test_path_helpers(tmp_path: Path) -> None:
    assert silver_dir(tmp_path, "ami") == tmp_path / "data" / "silver" / "ami"
    assert records_path(tmp_path, "ami").name == "records.jsonl"
    assert failures_path(tmp_path, "ami").name == "failures.jsonl"
    assert progress_path(tmp_path, "ami").name == ".progress.json"


def test_iter_all_records_across_corpora(tmp_path: Path) -> None:
    append_record(records_path(tmp_path, "fda_maude"), _record(corpus="fda_maude"))
    append_record(records_path(tmp_path, "ami"), _record(corpus="ami", record_id="S-2"))
    all_records = list(iter_all_records(tmp_path))
    assert {r.corpus for r in all_records} == {"fda_maude", "ami"}

    only_ami = list(iter_all_records(tmp_path, corpora=["ami"]))
    assert len(only_ami) == 1
    assert only_ami[0].corpus == "ami"


def test_iter_all_records_no_silver_dir_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_all_records(tmp_path)) == []


# --------------------------------------------------------------------------
# compute_stats
# --------------------------------------------------------------------------


def test_compute_stats_counts_per_corpus_and_fact_type_and_negatives() -> None:
    fact_complaint = TrainingFact(fact_type="complaint", claim="c", quote="q", locator="L1")
    fact_request = TrainingFact(fact_type="request", claim="r", locator="L2")
    positive = _record(
        record_id="S-1",
        corpus="fda_maude",
        facts=[fact_complaint, fact_request],
        teachers=["qwen/x:free", "deepseek/y:free"],
    )
    negative = _record(
        record_id="S-2",
        source_id="S-2",
        corpus="fda_maude",
        facts=[],
        teachers=["deepseek/y:free", "qwen/x:free"],
    )
    other_corpus = _record(record_id="S-3", source_id="S-3", corpus="ami", facts=[])

    stats = compute_stats([positive, negative, other_corpus])
    assert stats.total_records == 3
    assert stats.per_corpus["fda_maude"] == 2
    assert stats.per_corpus["ami"] == 1
    assert stats.per_fact_type["complaint"] == 1
    assert stats.per_fact_type["request"] == 1
    assert stats.total_negatives == 2
    assert stats.negatives_per_corpus["fda_maude"] == 1
    assert stats.negatives_per_corpus["ami"] == 1
    # order-independent pairing: "qwen/x:free + deepseek/y:free" counted
    # once, both records
    assert stats.teacher_pairs["deepseek/y:free + qwen/x:free"] == 2
    assert stats.total_facts == 2


# --------------------------------------------------------------------------
# TrainingFact.quote_span (feature/labeling-exact-quotes)
# --------------------------------------------------------------------------


def test_training_fact_quote_span_defaults_to_none() -> None:
    fact = TrainingFact(fact_type="complaint", claim="c", quote="q", locator="L1")
    assert fact.quote_span is None


def test_training_fact_quote_span_round_trips_through_json() -> None:
    fact = TrainingFact(
        fact_type="complaint",
        claim="device malfunctioned",
        quote="the device malfunctioned",
        quote_span=(4, 29),
        locator="L1",
    )
    restored = TrainingFact.model_validate_json(fact.model_dump_json())
    assert restored.quote_span == (4, 29)


# --------------------------------------------------------------------------
# DroppedFact / dropped.jsonl (feature/labeling-exact-quotes, build
# instructions item 2)
# --------------------------------------------------------------------------


def _dropped(
    source_id: str = "S-1",
    reason: str = "quote_not_exact",
    fact: dict | None = None,
    teacher: str = "deepseek/deepseek-v3.2",
) -> DroppedFact:
    return DroppedFact(
        source_id=source_id,
        reason=reason,
        fact=fact or {"fact_type": "complaint", "claim": "c", "quote": "q", "locator": "L1"},
        teacher=teacher,
        created="2026-09-22T06:10:00-07:00",
    )


def test_dropped_path_helper(tmp_path: Path) -> None:
    assert dropped_path(tmp_path, "enron").name == "dropped.jsonl"


def test_append_and_read_dropped_round_trip(tmp_path: Path) -> None:
    path = dropped_path(tmp_path, "enron")
    d1 = _dropped(source_id="S-1")
    d2 = _dropped(source_id="S-2", reason="overlap_gate_numeral")
    append_dropped(path, d1)
    append_dropped(path, d2)
    assert list(read_dropped(path)) == [d1, d2]


def test_read_dropped_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(read_dropped(dropped_path(tmp_path, "enron"))) == []


def test_iter_all_dropped_across_corpora(tmp_path: Path) -> None:
    append_dropped(dropped_path(tmp_path, "enron"), _dropped(source_id="S-1"))
    append_dropped(dropped_path(tmp_path, "ami"), _dropped(source_id="S-2"))
    all_dropped = list(iter_all_dropped(tmp_path))
    assert {d.source_id for d in all_dropped} == {"S-1", "S-2"}

    only_enron = list(iter_all_dropped(tmp_path, corpora=["enron"]))
    assert len(only_enron) == 1
    assert only_enron[0].source_id == "S-1"


def test_dropped_by_corpus_groups_reason_counts(tmp_path: Path) -> None:
    append_dropped(dropped_path(tmp_path, "enron"), _dropped(reason="quote_gate"))
    append_dropped(dropped_path(tmp_path, "enron"), _dropped(reason="quote_gate"))
    append_dropped(dropped_path(tmp_path, "enron"), _dropped(reason="quote_not_exact"))
    append_dropped(dropped_path(tmp_path, "ami"), _dropped(reason="overlap_gate_name"))

    by_corpus = dropped_by_corpus(tmp_path)
    assert by_corpus["enron"]["quote_gate"] == 2
    assert by_corpus["enron"]["quote_not_exact"] == 1
    assert by_corpus["ami"]["overlap_gate_name"] == 1
    # a corpus with no dropped.jsonl at all is simply absent, not a KeyError
    assert "fda_maude" not in by_corpus


def test_dropped_by_corpus_empty_when_no_silver_dir(tmp_path: Path) -> None:
    assert dropped_by_corpus(tmp_path) == {}


# --------------------------------------------------------------------------
# compute_stats with drops / exact_quote_rate
# --------------------------------------------------------------------------


def test_compute_stats_counts_drops_by_reason() -> None:
    fact = TrainingFact(fact_type="complaint", claim="c", quote="q", locator="L1")
    record = _record(facts=[fact])
    dropped = [
        _dropped(reason="quote_gate"),
        _dropped(reason="quote_not_exact"),
        _dropped(reason="quote_not_exact"),
    ]
    stats = compute_stats([record], dropped)
    assert stats.total_dropped == 3
    assert stats.dropped_by_reason["quote_gate"] == 1
    assert stats.dropped_by_reason["quote_not_exact"] == 2


def test_compute_stats_without_dropped_arg_leaves_drop_fields_empty() -> None:
    fact = TrainingFact(fact_type="complaint", claim="c", quote="q", locator="L1")
    stats = compute_stats([_record(facts=[fact])])
    assert stats.total_dropped == 0
    assert stats.dropped_by_reason == {}


def test_exact_quote_rate_none_when_nothing_to_divide() -> None:
    stats = compute_stats([])
    assert stats.exact_quote_rate is None


def test_exact_quote_rate_computed_from_kept_and_quote_not_exact_drops() -> None:
    fact1 = TrainingFact(fact_type="complaint", claim="c", quote="q1", locator="L1")
    fact2 = TrainingFact(fact_type="complaint", claim="c", quote="q2", locator="L1")
    fact3 = TrainingFact(fact_type="request", claim="c", quote=None, locator="L1")
    record = _record(facts=[fact1, fact2, fact3])
    dropped = [
        _dropped(reason="quote_not_exact"),
        _dropped(reason="quote_gate"),  # not counted in the exact-quote denominator
    ]
    stats = compute_stats([record], dropped)
    # 2 kept quote-bearing facts / (2 kept + 1 quote_not_exact drop) = 2/3
    assert stats.kept_facts_with_quote == 2
    assert stats.exact_quote_rate == pytest.approx(2 / 3)


def test_exact_quote_rate_counts_backfill_reason_too() -> None:
    fact = TrainingFact(fact_type="complaint", claim="c", quote="q", locator="L1")
    record = _record(facts=[fact])
    dropped = [_dropped(reason="quote_not_exact_backfill")]
    stats = compute_stats([record], dropped)
    assert stats.exact_quote_rate == pytest.approx(0.5)
