"""Tests for `scripts/data/backfill_exact_quotes.py`
(feature/labeling-exact-quotes, build instructions item 3): re-resolves
every fact already on disk in a `records.jsonl` to an exact quote span,
drops what can't resolve, logs the drops, and removes any record left
with zero facts that wasn't already a negative. No network, no real
corpus -- a small fixture file built with `TrainingRecord`/`TrainingFact`
directly.

`scripts/` isn't a Python package (`tests/test_set_owners.py`'s own
docstring: "no `__init__.py`... standalone tools, not modules the test
suite imports normally") -- loaded by file path, same pattern.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from portolan.training.records import (
    TrainingFact,
    TrainingRecord,
    append_record,
    dropped_path,
    read_dropped,
    read_records,
    records_path,
)
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "backfill_exact_quotes.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("backfill_exact_quotes_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_exact_quotes_script"] = module
    spec.loader.exec_module(module)
    return module


# Safe at module level: `require_repo_path` above already skipped this module
# during collection when `scripts/` is absent (the container image).
backfill = _load_module()


def _fixed_now() -> str:
    return "2026-09-22T06:10:00-07:00"


def _fact(quote: str, quote_span: tuple[int, int] | None = None, **kw) -> TrainingFact:
    return TrainingFact(
        fact_type=kw.get("fact_type", "complaint"),
        claim=kw.get("claim", "a claim"),
        quote=quote,
        quote_span=quote_span,
        locator=kw.get("locator", "L1"),
        entities=kw.get("entities", []),
    )


def _record(
    record_id: str,
    text: str,
    facts: list[TrainingFact],
    *,
    teachers: list[str] | None = None,
) -> TrainingRecord:
    return TrainingRecord(
        record_id=record_id,
        corpus="enron",
        source_id=record_id,
        kind="email",
        license="public record",
        text=text,
        facts=facts,
        teachers=teachers or ["deepseek/deepseek-v3.2"],
        agreement=1.0,
        created="2026-09-21T00:00:00-07:00",
        mode="single",
    )


# --------------------------------------------------------------------------
# resolve_record_facts -- pure per-record step
# --------------------------------------------------------------------------


def test_already_exact_fact_with_valid_span_is_untouched() -> None:
    text = "the pump beeped three times"
    fact = _fact("the pump beeped", quote_span=(0, len("the pump beeped")))
    record = _record("S-1", text, [fact])
    kept, dropped, already_exact = backfill.resolve_record_facts(record)
    assert kept == [fact]
    assert dropped == []
    assert already_exact == 1


def test_fact_with_stale_wrong_span_is_still_resolved() -> None:
    # quote_span present but WRONG (doesn't match text[start:end]) -- not
    # trusted blindly, re-resolved from scratch.
    text = "the pump beeped three times"
    fact = _fact("the pump beeped", quote_span=(0, 5))
    record = _record("S-1", text, [fact])
    kept, _dropped, already_exact = backfill.resolve_record_facts(record)
    assert already_exact == 0
    assert len(kept) == 1
    start, end = kept[0].quote_span
    assert text[start:end] == kept[0].quote == "the pump beeped"


def test_fact_needing_normalization_gets_span_and_exact_quote() -> None:
    text = "The Pump Beeped Three Times"
    fact = _fact("the pump beeped three times")
    record = _record("S-1", text, [fact])
    kept, dropped, _already_exact = backfill.resolve_record_facts(record)
    assert dropped == []
    assert kept[0].quote == "The Pump Beeped Three Times"
    assert kept[0].quote_span == (0, len(text))


def test_unresolvable_fact_is_dropped_and_resolvable_sibling_kept() -> None:
    text = "the pump beeped three times and shut off"
    good = _fact("the pump beeped three times")
    bad = _fact("the pump chimed three times")  # substituted word, no exact span exists
    record = _record("S-1", text, [good, bad])
    kept, dropped, _already_exact = backfill.resolve_record_facts(record)
    assert len(kept) == 1
    assert kept[0].quote == "the pump beeped three times"
    assert len(dropped) == 1
    assert dropped[0].quote == "the pump chimed three times"


# --------------------------------------------------------------------------
# backfill_records_file -- the file-level operation
# --------------------------------------------------------------------------


def test_backfill_resolves_and_writes_updated_records(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")

    text1 = "The Pump Beeped Three Times and shut off"
    append_record(records_file, _record("S-1", text1, [_fact("the pump beeped three times")]))

    text2 = "made by=20 organizations such as Enron"
    append_record(
        records_file, _record("S-2", text2, [_fact("made by organizations such as Enron")])
    )

    result = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )

    assert result.records_before == 2
    assert result.records_after == 2
    assert result.facts_before == 2
    assert result.facts_after == 2
    assert result.facts_newly_resolved == 2
    assert result.facts_dropped == 0
    assert result.records_dropped == 0

    written = {r.record_id: r for r in read_records(records_file)}
    s1, s2 = written["S-1"], written["S-2"]
    start, end = s1.facts[0].quote_span
    assert s1.text[start:end] == s1.facts[0].quote == "The Pump Beeped Three Times"
    start, end = s2.facts[0].quote_span
    assert s2.text[start:end] == s2.facts[0].quote == "made by=20 organizations such as Enron"


def test_backfill_drops_fact_and_logs_it(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")

    text = "the pump beeped three times and shut off"
    append_record(
        records_file,
        _record(
            "S-1",
            text,
            [_fact("the pump beeped three times"), _fact("the pump chimed three times")],
        ),
    )

    result = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )
    assert result.facts_before == 2
    assert result.facts_after == 1
    assert result.facts_dropped == 1
    assert result.records_dropped == 0  # one fact survived, record kept

    written = list(read_records(records_file))
    assert len(written) == 1
    assert len(written[0].facts) == 1

    dropped = list(read_dropped(dropped_file))
    assert len(dropped) == 1
    assert dropped[0].reason == "quote_not_exact_backfill"
    assert dropped[0].source_id == "S-1"
    assert dropped[0].fact["quote"] == "the pump chimed three times"
    assert dropped[0].teacher == "deepseek/deepseek-v3.2"


def test_backfill_removes_record_left_with_zero_facts(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")

    text = "the pump beeped three times and shut off"
    append_record(records_file, _record("S-1", text, [_fact("the pump chimed three times")]))
    append_record(records_file, _record("S-2", text, [_fact("the pump beeped three times")]))

    result = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )
    assert result.records_before == 2
    assert result.records_after == 1
    assert result.records_dropped == 1

    written = list(read_records(records_file))
    assert [r.record_id for r in written] == ["S-2"]


def test_backfill_keeps_pre_existing_negative_record_untouched(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")

    text = "nothing extractable here"
    append_record(records_file, _record("S-1", text, []))  # a genuine negative

    result = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )
    assert result.records_before == 1
    assert result.records_after == 1
    assert result.records_dropped == 0

    written = list(read_records(records_file))
    assert len(written) == 1
    assert written[0].is_negative


def test_backfill_writes_backup_before_rewriting(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")
    text = "The Pump Beeped Three Times"
    append_record(records_file, _record("S-1", text, [_fact("the pump beeped three times")]))
    original_bytes = records_file.read_bytes()

    result = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )

    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.backup_path.read_bytes() == original_bytes
    assert result.backup_path.name.startswith("records.jsonl.bak-")
    # the live file was actually rewritten (quote_span added), so it now
    # differs from the backup
    assert records_file.read_bytes() != original_bytes


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")

    text = "the pump beeped three times and shut off"
    append_record(
        records_file,
        _record(
            "S-1",
            text,
            [_fact("the pump beeped three times"), _fact("the pump chimed three times")],
        ),
    )
    append_record(records_file, _record("S-2", text, []))

    first = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )
    after_first = records_file.read_bytes()
    dropped_after_first = list(read_dropped(dropped_file))

    second = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )
    after_second = records_file.read_bytes()
    dropped_after_second = list(read_dropped(dropped_file))

    assert after_first == after_second
    assert second.facts_newly_resolved == 0
    assert second.facts_dropped == 0
    assert second.records_dropped == 0
    assert second.records_before == second.records_after == first.records_after
    # no NEW drop log entries on the second pass
    assert dropped_after_first == dropped_after_second


def test_backfill_missing_file_is_a_noop(tmp_path: Path) -> None:
    records_file = records_path(tmp_path, "enron")
    dropped_file = dropped_path(tmp_path, "enron")
    result = backfill.backfill_records_file(
        records_file, dropped_file, corpus="enron", now=_fixed_now
    )
    assert result.records_before == 0
    assert result.records_after == 0
    assert not records_file.exists()
