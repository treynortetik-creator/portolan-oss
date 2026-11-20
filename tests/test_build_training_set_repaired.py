"""`scripts/data/build_training_set.py` with audited records from
`scripts/data/repair_silver.py`: `records.repaired.jsonl` wins over
`records.jsonl`, incomplete records are excluded, and a record whose
teacher facts were all dropped or withdrawn is never a negative."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from portolan.training.records import (
    LabelAudit,
    TrainingFact,
    TrainingRecord,
    records_path,
    repaired_records_path,
)
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "build_training_set.py")


def _load():
    spec = importlib.util.spec_from_file_location("build_training_set_repaired", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_training_set_repaired"] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


def _words(seed: str) -> str:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return " ".join(digest[i : i + 6] for i in range(0, 48, 6))


def _audit(status: str = "complete", **counts: int | None) -> LabelAudit:
    base = dict(
        teacher_facts=1, kept=1, regated=0, repaired=0, withdrawn=0, unresolved=0, unlogged=0
    )
    return LabelAudit(status=status, audited="2026-09-22T16:00:00-07:00", **{**base, **counts})


def _record(
    record_id: str,
    *,
    n_facts: int = 1,
    agreement: float = 1.0,
    audit: LabelAudit | None = None,
    corpus: str = "enron",
) -> TrainingRecord:
    quotes = [f"quote {record_id} {i} {_words(record_id + str(i))[:20]}" for i in range(n_facts)]
    text = f"{_words(record_id)} " + " ".join(quotes)
    facts = [
        TrainingFact(
            fact_type="complaint",
            claim=f"claim {record_id} {i}",
            quote=q,
            quote_span=(text.index(q), text.index(q) + len(q)),
            locator="L1",
        )
        for i, q in enumerate(quotes)
    ]
    return TrainingRecord(
        record_id=record_id,
        corpus=corpus,
        source_id=record_id,
        kind="email",
        license="public record",
        text=text,
        facts=facts,
        teachers=["moonshotai/kimi-k2"],
        agreement=agreement,
        created="2026-09-22T12:00:00-07:00",
        mode="single",
        label_audit=audit,
    )


def _ids(*splits) -> set[str]:
    return {r.record_id for split in splits for r in split}


def _build(records: list[TrainingRecord]):
    padding = [_record(f"PAD-{i}") for i in range(6)]
    return builder.build_training_set(Path("unused"), records=[*records, *padding])


def test_audited_incomplete_record_is_excluded() -> None:
    incomplete = _record("I-1", audit=_audit("incomplete", teacher_facts=2, unresolved=1))
    train, val, stats = _build([incomplete])
    assert "I-1" not in _ids(train, val)
    assert stats.dropped_by_reason["incomplete_labels"] == 1


def test_audit_overrides_agreement_for_a_complete_record() -> None:
    # agreement says the gate dropped facts; the audit says each one was
    # repaired, re-admitted or withdrawn
    repaired = _record("R-1", agreement=0.5, audit=_audit(teacher_facts=2, withdrawn=1))
    train, val, stats = _build([repaired])
    assert "R-1" in _ids(train, val)
    assert stats.dropped_by_reason["partial_labels"] == 0


def test_a_record_whose_teacher_facts_were_all_withdrawn_is_never_a_negative() -> None:
    withdrawn = _record("W-1", n_facts=0, audit=_audit(teacher_facts=2, kept=0, withdrawn=2))
    unknown = _record(
        "U-1", n_facts=0, audit=_audit("incomplete", teacher_facts=None, kept=0, unlogged=1)
    )
    true_negative = _record("N-1", n_facts=0, audit=_audit(teacher_facts=0, kept=0))
    train, val, stats = _build([withdrawn, unknown, true_negative])
    ids = _ids(train, val)
    assert "W-1" not in ids and "U-1" not in ids and "N-1" in ids
    assert stats.dropped_by_reason["negative_after_gate_drops"] == 2


def test_repaired_file_replaces_records_with_the_same_id(tmp_path: Path) -> None:
    var = tmp_path / "var"
    live = records_path(var, "enron")
    live.parent.mkdir(parents=True)
    unaudited = [
        _record("E-1", agreement=0.5),  # partial, repaired below
        _record("E-2"),
        _record("E-3", agreement=0.5),  # partial, labeled after the repair run
        *[_record(f"E-{i}") for i in range(4, 10)],
    ]
    live.write_text("".join(r.model_dump_json() + "\n" for r in unaudited))
    e1 = _record("E-1", n_facts=2, agreement=0.5, audit=_audit(teacher_facts=2, repaired=1))
    e1.facts[1] = e1.facts[1].model_copy(update={"repaired": True})
    repaired = [e1, _record("E-2", audit=_audit())]
    repaired_records_path(var, "enron").write_text(
        "".join(r.model_dump_json() + "\n" for r in repaired)
    )
    out = tmp_path / "out"
    # the exam-contamination gate is tested in test_build_training_set.py
    args = ["--var-root", str(var), "--out", str(out), "--no-contamination-check"]
    assert builder.main(args) == 0

    kept = [
        json.loads(line)
        for name in ("train.jsonl", "val.jsonl")
        for line in (out / name).read_text().splitlines()
    ]
    by_id = {r["record_id"]: r for r in kept}
    assert len(by_id["E-1"]["facts"]) == 2
    assert by_id["E-1"]["label_audit"]["repaired"] == 1
    assert "E-3" not in by_id
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["input_records"] == 9
    assert manifest["dropped_by_reason"] == {"partial_labels": 1}
    assert manifest["label_audit"]["audited_input_records"] == {"enron": 2}
    paths = [i["path"] for i in manifest["inputs"]]
    assert paths == ["data/silver/enron/records.repaired.jsonl", "data/silver/enron/records.jsonl"]
    card = (out / "data_card.md").read_text()
    assert "## Label completeness" in card
    assert "| enron | 2 | 0 | 1 | 0 | 0 | 1 | 0 |" in card
