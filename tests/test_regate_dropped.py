"""Tests for `scripts/data/regate_dropped.py` (fix/overlap-gate-
paraphrase, 2026-09-22): re-evaluates every `dropped.jsonl` entry whose
`reason` starts with `overlap_gate` against the CURRENT gate, and for
the ones that now pass, gets the fact into `records.jsonl` (creating or
updating the document's record) without re-paying for a teacher call.
No network, no real corpus -- every fixture here is a small `calls/*.txt`
file built fresh under `tmp_path`, read through the real
`documents_for_corpus`/`document_envelopes` so this test exercises the
same envelope reconstruction the script itself relies on.

`scripts/` isn't a Python package and is excluded from the container
image, so this loads the module by file path through
`tests/_paths.py`'s `require_repo_path`, which skips the module at
collection when the file is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from portolan.training.records import (
    DroppedFact,
    TrainingFact,
    TrainingRecord,
    append_dropped,
    append_record,
    dropped_path,
    read_dropped,
    read_records,
    records_path,
)
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "regate_dropped.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("regate_dropped_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["regate_dropped_script"] = module
    spec.loader.exec_module(module)
    return module


regate = _load_module()

_FIXED_NOW = "2026-09-22T08:00:00-07:00"
_CALL_TEXT = (
    "[00:00:00] Customer: hi, what's the pricing?\n\n"
    "[00:01:00] Customer: Our turnover rate is pretty high because of the onboarding experience\n"
)


def _write_call(var_root: Path, source_id: str, text: str = _CALL_TEXT) -> None:
    calls_dir = var_root / "data" / "converted" / "saas_sales" / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    (calls_dir / f"{source_id}.txt").write_text(text, encoding="utf-8")


def _dropped_fact(
    source_id: str,
    *,
    reason: str,
    claim: str,
    quote: str,
    teacher: str = "test-teacher",
    fact_type: str = "risk",
    regated_at: str | None = None,
) -> DroppedFact:
    return DroppedFact(
        source_id=source_id,
        reason=reason,
        fact={
            "fact_type": fact_type,
            "claim": claim,
            "quote": quote,
            "locator": "00:01:00",
            "entities": ["Customer"],
        },
        teacher=teacher,
        created=_FIXED_NOW,
        regated_at=regated_at,
    )


def _existing_record(source_id: str, text: str = _CALL_TEXT, facts: list | None = None):
    fact = TrainingFact(
        fact_type="complaint",
        claim="Customer asked about pricing.",
        quote="hi, what's the pricing?",
        quote_span=(21, 44),
        locator="00:00:00",
        entities=["Customer"],
    )
    return TrainingRecord(
        record_id=source_id,
        corpus="saas_sales",
        source_id=source_id,
        kind="call",
        license=("Apache 2.0 (DeepMostInnovations/saas-sales-conversations, Hugging Face)"),
        text=text,
        facts=facts if facts is not None else [fact],
        teachers=["test-teacher"],
        agreement=0.5,
        created=_FIXED_NOW,
        mode="single",
    )


def _fixed_now() -> str:
    return _FIXED_NOW


# --------------------------------------------------------------------------
# regate_corpus -- no dropped.jsonl at all
# --------------------------------------------------------------------------


def test_regate_no_dropped_file_is_a_clean_no_op(tmp_path: Path) -> None:
    result = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)
    assert result.dropped_entries_total == 0
    assert result.facts_readmitted == 0
    assert result.records_before == result.records_after == 0
    assert result.facts_before == result.facts_after == 0


# --------------------------------------------------------------------------
# The real Kimi K2 finding: a faithful paraphrase dropped only by the
# old content-overlap rule (now disabled at threshold=0.0) and/or the
# old claim-initial name rule -- must be admitted.
# --------------------------------------------------------------------------


def test_regate_admits_fact_dropped_only_by_old_content_rule_and_creates_record(
    tmp_path: Path,
) -> None:
    _write_call(tmp_path, "S-call-1")
    dropped_file = dropped_path(tmp_path, "saas_sales")
    append_dropped(
        dropped_file,
        _dropped_fact(
            "S-call-1",
            reason="overlap_gate_content",
            claim="Customer's turnover rate is high because of the poor onboarding experience",
            quote="Our turnover rate is pretty high because of the onboarding experience",
        ),
    )

    result = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)

    assert result.dropped_entries_eligible == 1
    assert result.facts_readmitted == 1
    assert result.facts_still_dropped == 0
    assert result.records_created == 1
    assert result.records_before == 0
    assert result.records_after == 1
    assert result.facts_before == 0
    assert result.facts_after == 1
    assert result.backup_path is None  # nothing existed to back up

    records = list(read_records(records_path(tmp_path, "saas_sales")))
    assert len(records) == 1
    record = records[0]
    assert record.record_id == "S-call-1"
    assert len(record.facts) == 1
    fact = record.facts[0]
    assert fact.regated is True
    assert fact.quote_span is not None
    start, end = fact.quote_span
    assert record.text[start:end] == fact.quote

    dropped_after = list(read_dropped(dropped_file))
    assert len(dropped_after) == 1
    assert dropped_after[0].regated_at == _FIXED_NOW


def test_regate_appends_to_an_existing_record_and_backs_it_up(tmp_path: Path) -> None:
    _write_call(tmp_path, "S-call-1")
    records_file = records_path(tmp_path, "saas_sales")
    append_record(records_file, _existing_record("S-call-1"))
    original_records_text = records_file.read_text(encoding="utf-8")
    dropped_file = dropped_path(tmp_path, "saas_sales")
    append_dropped(
        dropped_file,
        _dropped_fact(
            "S-call-1",
            reason="overlap_gate_name",
            claim="Customer's turnover rate is high because of the poor onboarding experience",
            quote="Our turnover rate is pretty high because of the onboarding experience",
        ),
    )

    result = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)

    assert result.facts_readmitted == 1
    assert result.records_created == 0
    assert result.records_updated == 1
    assert result.records_before == 1
    assert result.records_after == 1
    assert result.facts_before == 1
    assert result.facts_after == 2
    assert result.backup_path is not None
    assert result.backup_path.exists()
    # the backup holds the file's content BEFORE this run's rewrite
    assert result.backup_path.read_text(encoding="utf-8") == original_records_text

    records = list(read_records(records_file))
    assert len(records) == 1
    assert len(records[0].facts) == 2
    assert sum(1 for f in records[0].facts if f.regated) == 1


# --------------------------------------------------------------------------
# A drop that still fails the current gate, or whose reason the fix
# doesn't touch, must stay dropped -- but every entry actually
# evaluated still gets `regated_at` so the idempotency marker is honest.
# --------------------------------------------------------------------------


def test_regate_leaves_fact_dropped_when_a_fabricated_name_still_fails(tmp_path: Path) -> None:
    _write_call(tmp_path, "S-call-1")
    dropped_file = dropped_path(tmp_path, "saas_sales")
    append_dropped(
        dropped_file,
        _dropped_fact(
            "S-call-1",
            reason="overlap_gate_name",
            claim="Regarding pricing, Kenji Mori confirmed the turnover concern.",
            quote="Our turnover rate is pretty high because of the onboarding experience",
        ),
    )

    result = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)

    assert result.dropped_entries_eligible == 1
    assert result.facts_readmitted == 0
    assert result.facts_still_dropped == 1
    assert result.records_after == 0

    dropped_after = list(read_dropped(dropped_file))
    assert dropped_after[0].regated_at == _FIXED_NOW


def test_regate_skips_entries_whose_reason_it_does_not_touch(tmp_path: Path) -> None:
    _write_call(tmp_path, "S-call-1")
    dropped_file = dropped_path(tmp_path, "saas_sales")
    append_dropped(
        dropped_file,
        _dropped_fact(
            "S-call-1",
            reason="quote_gate",
            claim="A sentence never present anywhere.",
            quote="a sentence never present anywhere in the document",
        ),
    )

    result = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)

    assert result.dropped_entries_eligible == 0
    assert result.facts_readmitted == 0

    dropped_after = list(read_dropped(dropped_file))
    assert dropped_after[0].regated_at is None  # left untouched, by design


def test_regate_counts_no_envelope_match_without_crashing(tmp_path: Path) -> None:
    _write_call(tmp_path, "S-call-1")
    dropped_file = dropped_path(tmp_path, "saas_sales")
    append_dropped(
        dropped_file,
        _dropped_fact(
            "S-call-1",
            reason="overlap_gate_content",
            claim="A claim whose quote never appears in this document at all.",
            quote="this exact sentence is not in the call transcript anywhere",
        ),
    )

    result = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)

    assert result.dropped_entries_eligible == 1
    assert result.facts_no_envelope_match == 1
    assert result.facts_readmitted == 0

    dropped_after = list(read_dropped(dropped_file))
    assert dropped_after[0].regated_at == _FIXED_NOW


# --------------------------------------------------------------------------
# Idempotency: a second run over the same corpus re-evaluates nothing.
# --------------------------------------------------------------------------


def test_regate_is_idempotent(tmp_path: Path) -> None:
    _write_call(tmp_path, "S-call-1")
    dropped_file = dropped_path(tmp_path, "saas_sales")
    append_dropped(
        dropped_file,
        _dropped_fact(
            "S-call-1",
            reason="overlap_gate_content",
            claim="Customer's turnover rate is high because of the poor onboarding experience",
            quote="Our turnover rate is pretty high because of the onboarding experience",
        ),
    )

    first = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)
    assert first.facts_readmitted == 1

    second = regate.regate_corpus(tmp_path, "saas_sales", overlap_threshold=0.0, now=_fixed_now)
    assert second.dropped_entries_eligible == 0
    assert second.dropped_entries_already_regated == 1
    assert second.facts_readmitted == 0
    assert second.records_before == second.records_after == 1
    assert second.facts_before == second.facts_after == 1
    assert second.backup_path is None  # nothing changed, nothing to back up

    records = list(read_records(records_path(tmp_path, "saas_sales")))
    assert len(records) == 1
    assert len(records[0].facts) == 1  # not duplicated
