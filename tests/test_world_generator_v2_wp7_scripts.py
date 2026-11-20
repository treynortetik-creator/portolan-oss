"""WP7's script side (`01-spec/world-generator-v2-2026-09-22.md` section 6
and section 9's WP7 row): the training-set builder reads exported v2 worlds
as corpus `synthetic_v2`, splits them by world, keeps test worlds and the
held-out vertical out, still drops v1 templated filler, and keeps every
existing gate; the runner's export stage is real; the trainer takes a v2
record's own company context. Every model call is stubbed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from portolan.bench.world_v2 import load_world
from portolan.training.records import LabelAudit, TrainingFact, TrainingRecord
from tests._paths import require_repo_path

_BUILDER = require_repo_path("scripts", "data", "build_training_set.py")
_RUNNER = require_repo_path("scripts", "data", "synth_v2.py")
_TRAINER = require_repo_path("scripts", "train", "mlx_lora.py")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("wp7_build_training_set", _BUILDER)
runner = _load("wp7_synth_v2", _RUNNER)

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
).split()


def _text(n: int) -> str:
    """A distinct text per `n`, so dedup never groups two records."""
    words = [f"{w}{n}" for w in WORDS]
    return "[L1] " + " ".join(words) + f" the pump alarm fired {n} times on the night shift."


def _v2(
    n: int,
    *,
    world_id: str = "world-medical_devices-42",
    split: str | None = "train",
    claim_source: str = "teacher",
    negative: bool = False,
) -> TrainingRecord:
    text = _text(n)
    quote = f"the pump alarm fired {n} times on the night shift."
    start = text.index(quote)
    facts = (
        []
        if negative
        else [
            TrainingFact(
                fact_type="complaint",
                claim=f"The pump alarm fired {n} times on the night shift.",
                quote=quote,
                quote_span=(start, start + len(quote)),
                locator="L1",
                sentiment=-1,
                confidence=0.8,
                label_source="scripted_writer",
                claim_source=claim_source,
            )
        ]
    )
    return TrainingRecord(
        record_id=f"{world_id}:S-{n}",
        corpus="synthetic_v2",
        source_id=f"{world_id}/S-{n}",
        kind="call",
        license="synthetic",
        text=text,
        facts=facts,
        teachers=["qwen/qwen3-32b", "moonshotai/kimi-k2"],
        agreement=1.0,
        created="2026-09-23T00:00:00+00:00",
        mode="v2_world",
        label_audit=LabelAudit(
            status="complete",
            teacher_facts=len(facts),
            kept=len(facts),
            regated=0,
            repaired=0,
            withdrawn=0,
            unresolved=0,
            unlogged=0,
            audited="2026-09-23T00:00:00+00:00",
        ),
        world_id=world_id,
        world_split=split,
    )


def _v1_negative(n: int) -> TrainingRecord:
    return TrainingRecord(
        record_id=f"batch:S-v1-{n}",
        corpus="synthetic",
        source_id=f"S-v1-{n}",
        kind="call",
        license="synthetic",
        text=_text(1000 + n),
        facts=[],
        teachers=["qwen/qwen3-32b"],
        agreement=1.0,
        created="2026-09-20T00:00:00+00:00",
    )


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def test_builder_splits_v2_by_world_and_keeps_test_and_held_out_worlds_out() -> None:
    records = [
        *[_v2(i) for i in range(6)],
        *[_v2(10 + i, world_id="world-telehealth-7", split="validation") for i in range(3)],
        _v2(20, world_id="world-telehealth-9", split="test"),
        _v2(21, world_id="world-lab_and_imaging_services-3", split="train"),
        _v2(22, world_id="world-telehealth-11", split=None),
        _v2(23, claim_source="scripted"),
        _v1_negative(1),
    ]
    config = builder.CurationConfig(negative_cap=0.5, max_corpus_share=1.0)
    train, val, stats = builder.build_training_set(
        Path("/nonexistent"), records=records, config=config
    )
    assert sorted(r.record_id for r in train) == sorted(
        f"world-medical_devices-42:S-{i}" for i in range(6)
    )
    assert sorted(r.record_id for r in val) == sorted(
        f"world-telehealth-7:S-{10 + i}" for i in range(3)
    )
    assert stats.provenance_violations["v2_world_not_trainable"] == 2  # test world, no split
    assert stats.provenance_violations["v2_held_out_vertical"] == 1
    assert stats.provenance_violations["v2_claim_not_from_teacher"] == 1
    assert stats.provenance_violations["v1_templated_negative"] == 1
    assert stats.v2_worlds["world-medical_devices-42"] == {
        "split": "train",
        "input": 7,
        "train": 6,
        "val": 0,
    }
    assert stats.v2_worlds["world-telehealth-7"]["val"] == 3


def test_provenance_failures_is_what_the_trainer_rechecks() -> None:
    assert builder.provenance_failures(_v2(1)) == []
    assert builder.provenance_failures(_v2(2, split="test")) == ["v2_world_not_trainable"]
    assert "v2_held_out_vertical" in builder.provenance_failures(
        _v2(3, world_id="world-lab_and_imaging_services-1")
    )
    assert builder.provenance_failures(_v2(4, claim_source="scripted")) == [
        "v2_claim_not_from_teacher"
    ]
    assert builder.provenance_failures(_v2(5, negative=True)) == []
    assert builder.held_out_packs() == frozenset({"lab_and_imaging_services"})


def test_v2_records_keep_every_existing_gate() -> None:
    """The contamination name gate is hard for v2 (a generated source), and
    dedup, quote verification and the label-completeness rule still apply."""
    from portolan.training import contamination as C

    blocklisted = _v2(30)
    blocklisted = blocklisted.model_copy(update={"text": blocklisted.text + " Zanthoria Veldt"})
    duplicate = _v2(31).model_copy(update={"record_id": "world-medical_devices-42:S-31b"})
    bad_quote = _v2(32)
    bad_quote.facts[0] = bad_quote.facts[0].model_copy(update={"quote_span": (0, 5)})
    incomplete = _v2(33)
    incomplete = incomplete.model_copy(
        update={
            "label_audit": incomplete.label_audit.model_copy(
                update={"status": "incomplete", "unresolved": 1, "teacher_facts": 2}
            )
        }
    )
    params = C.ContaminationParams(
        minhash_permutations=32, lsh_bands=8, lsh_rows=4, minhash_seed=12345
    )
    refs = C.ContaminationReferences(
        np.zeros((0, params.minhash_permutations), dtype=np.uint64),
        [],
        np.array([], dtype=np.uint64),
        C.build_name_blocklist([("erb_gmail", "Zanthoria Veldt")], params, background_doc_freq={}),
        params,
    )
    records = [_v2(34), blocklisted, _v2(31), duplicate, bad_quote, incomplete]
    train, val, stats = builder.build_training_set(
        Path("/nonexistent"),
        records=records,
        contamination_refs=refs,
        config=builder.CurationConfig(negative_cap=0.5, max_corpus_share=1.0),
    )
    kept = {r.record_id for r in [*train, *val]}
    assert kept == {"world-medical_devices-42:S-34", "world-medical_devices-42:S-31"}
    assert stats.dropped_by_reason["contaminated_blocklisted_name"] == 1
    assert stats.dropped_by_reason["near_duplicate_document"] == 1
    assert stats.dropped_by_reason["quote_verification_failed"] == 1
    assert stats.dropped_by_reason["incomplete_labels"] == 1


def _write_world(
    root: Path, folder: str, world_id: str, records: list[TrainingRecord], **manifest
) -> Path:
    wdir = root / "data" / "synth_v2" / "worlds" / folder / world_id
    wdir.mkdir(parents=True)
    (wdir / "training_records.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records)
    )
    (wdir / "manifest.json").write_text(
        json.dumps({"export_version": "wp7", "split": folder, "dry_run": False, **manifest})
    )
    return wdir


def test_builder_finds_exported_worlds_and_skips_dry_runs_and_test_worlds(tmp_path) -> None:
    _write_world(tmp_path, "train", "world-medical_devices-42", [_v2(i) for i in range(4)])
    _write_world(
        tmp_path,
        "validation",
        "world-telehealth-7",
        [_v2(10 + i, world_id="world-telehealth-7", split="validation") for i in range(2)],
    )
    _write_world(tmp_path, "train", "world-telehealth-8", [_v2(20)], dry_run=True)
    _write_world(tmp_path, "test", "world-telehealth-9", [_v2(30, split="test")])
    _write_world(tmp_path, "train", "world-telehealth-10", [_v2(40)], split="validation")
    files, skipped = builder.v2_world_files(tmp_path)
    assert [p.parent.name for p in files] == ["world-medical_devices-42", "world-telehealth-7"]
    assert skipped == {
        "data/synth_v2/worlds/train/world-telehealth-10": "manifest split 'validation'",
        "data/synth_v2/worlds/train/world-telehealth-8": "dry run: fixture writer text",
    }
    train, val, stats = builder.build_training_set(
        tmp_path, config=builder.CurationConfig(negative_cap=0.5, max_corpus_share=1.0)
    )
    assert len(train) == 4 and len(val) == 2
    assert {r.corpus for r in [*train, *val]} == {"synthetic_v2"}
    assert any("synth_v2/worlds/train/world-medical_devices-42" in x["path"] for x in stats.inputs)
    assert stats.v2_worlds_skipped == skipped
    (tmp_path / "data" / "synth_v2" / "worlds" / "DRY_RUN").write_text("fixture")
    assert builder.v2_world_files(tmp_path) == (
        [],
        {str(tmp_path / "data/synth_v2/worlds"): "dry-run root"},
    )


def test_builder_cli_writes_the_v2_world_table(tmp_path) -> None:
    var = tmp_path / "var"
    _write_world(var, "train", "world-medical_devices-42", [_v2(i) for i in range(3)])
    out = tmp_path / "out"
    code = builder.main(
        [
            "--var-root",
            str(var),
            "--out",
            str(out),
            "--no-contamination-check",
            "--max-corpus-share",
            "1.0",
        ]
    )
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["v2_worlds"]["worlds"]["world-medical_devices-42"]["train"] == 3
    card = (out / "data_card.md").read_text()
    assert "## v2 worlds (split by world)" in card
    assert "| world-medical_devices-42 | train | 3 | 3 | 0 |" in card


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_runner_dry_run_exports_a_world_the_bench_reads(tmp_path, capsys) -> None:
    out = tmp_path / "dry"
    code = runner.main(
        [
            "--dry-run",
            "--worlds",
            "pilot",
            "--max-docs",
            "6",
            "--no-contamination-check",
            "--no-name-guard",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "export (WP7)" not in capsys.readouterr().err
    wdir = out / "train" / "world-pharmacy_services-1"
    manifest = json.loads((wdir / "manifest.json").read_text())
    assert manifest["dry_run"] is True and manifest["training_ready"] is False
    assert manifest["training_ready_reasons"] == ["dry run: fixture writer text"]
    assert manifest["training"]["records"] == manifest["training"]["sft_ready"] > 0
    assert manifest["training"]["documents_without_records"] == {}
    world = load_world(wdir)
    assert world.excluded_labels == {} and world.labels
    report = json.loads((wdir / "run_report.json").read_text())
    stage = next(s for s in report["stages"] if s["name"] == "export")
    assert stage["status"] == "done" and stage["detail"]["labels"] == len(world.labels)
    assert report["labels"]["exported"] == len(world.labels)
    assert any(x.startswith("export: not training-ready") for x in report["warnings"])
    # The builder never reads a dry-run root.
    assert builder.v2_world_files(tmp_path)[0] == []


def test_a_whole_world_is_capped_at_plan_time_and_the_export_cap_barely_bites(tmp_path) -> None:
    """The WP7 dry run dropped 55 of 222 written documents for the 12% cap;
    the planner now leaves the excess facts unscripted, so the export's cap
    is a safety check that drops almost nothing."""
    out = tmp_path / "dry"
    args = ["--dry-run", "--worlds", "pilot", "--no-contamination-check", "--no-name-guard"]
    assert runner.main([*args, "--out", str(out)]) == 0
    wdir = out / "train" / "world-pharmacy_services-1"
    plan_cap = json.loads((wdir / "world_plan.json").read_text())["type_cap"]
    assert plan_cap["applied"] and plan_cap["capped_fact_ids"] and not plan_cap["unmet_types"]
    manifest = json.loads((wdir / "manifest.json").read_text())
    cap = manifest["caps"]["per_type"]
    assert cap["applied"] is True and cap["met"] is True
    assert cap["max_share_after"] <= 0.12
    written = manifest["upstream"]["writer"]["documents_written"]
    assert len(cap["documents_dropped"]) <= 0.02 * written
    assert manifest["documents"]["excluded_by_reason"].get("type_cap", 0) == len(
        cap["documents_dropped"]
    )


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


def test_trainer_takes_synthetic_v2_and_the_records_own_company_context() -> None:
    trainer = _load("wp7_mlx_lora", _TRAINER)
    assert "synthetic_v2" in trainer.ALLOWED_CORPORA
    record = _v2(1).model_copy(update={"company_context": "Company: Ryecrest MedTech"})
    assert trainer.company_context_for(record) == "Company: Ryecrest MedTech"
    legacy = _v2(2).model_copy(update={"corpus": "ubuntu_irc"})
    assert trainer.company_context_for(legacy) == trainer.CORPUS_CONTEXTS.get("ubuntu_irc")
    trainer.assert_provenance([_v2(3)], Path("x"))
    with pytest.raises(RuntimeError, match="v2_world_not_trainable"):
        trainer.assert_provenance([_v2(4, split="test")], Path("x"))
