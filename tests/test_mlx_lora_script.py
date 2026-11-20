"""Tests for `scripts/train/mlx_lora.py`'s conversion half (fine-tune
shakedown, 2026-09-22): the hold-out guards, the exam-overlap check, the
record -> chat-example conversion (prompt builders, envelope rebuild,
locator remap, strict target). No MLX, no network, no real corpus: a tiny
synthetic batch is written under `tmp_path`.

`scripts/` isn't a package -- loaded by file path through
`tests/_paths.py`'s `require_repo_path` (the container image excludes
`scripts/`)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from portolan.extract.prompt import (
    DEFAULT_PROMPT_VARIANT,
    build_system_prompt,
    system_prompt_sha256,
)
from portolan.extract.schema import ExtractionResult, extraction_result_json_schema
from portolan.llm.mlx_local import effective_system_prompt
from portolan.training.records import TeacherAnswer, TrainingFact, TrainingRecord
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "train", "mlx_lora.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("mlx_lora_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlx_lora_script"] = module
    spec.loader.exec_module(module)
    return module


script = _load_module()

EMAIL = """Message-ID: <S-email-001@example.test>
From: Ana Ruiz <ana@buyer.test>
To: Sam Cole <sam@seller.test>
Date: 2026-02-01 09:00:00 -0700
Subject: Renewal

Hi Sam,

We need a 15 percent discount on the T2 renewal or we will look at other vendors.

Thanks,
Ana
"""


# The system prompt a `synthetic` record's teacher saw: no corpus context.
TEACHER_SYSTEM = build_system_prompt(company_context=None, variant=DEFAULT_PROMPT_VARIANT)


def _record(facts: list[TrainingFact], *, record_id: str = "batch-x:S-email-001") -> TrainingRecord:
    return TrainingRecord(
        record_id=record_id,
        corpus="synthetic",
        source_id="S-email-001",
        kind="email",
        license="synthetic-owned",
        text=EMAIL,
        facts=facts,
        teachers=["qwen/qwen3-32b"],
        agreement=1.0,
        created="2026-09-22T00:00:00+00:00",
        prompt_variant=DEFAULT_PROMPT_VARIANT,
        system_prompt_sha256=system_prompt_sha256(TEACHER_SYSTEM),
        teacher_answer=TeacherAnswer(
            summary="Ana asked for a renewal discount.",
            coverage={"segments_read": 1, "segments_total": 1},
        ),
    )


def _pricing_fact(**overrides) -> TrainingFact:
    base = {
        "fact_type": "pricing",
        "claim": "Ana asked for a 15 percent discount on the T2 renewal.",
        "quote": "We need a 15 percent discount on the T2 renewal",
        "locator": "S-email-001@example.test#p9",  # wrong on purpose: remap target
        "entities": ["Ana Ruiz"],
        "sentiment": -1,
        "confidence": 0.9,
    }
    base.update(overrides)
    return TrainingFact(**base)


@pytest.fixture
def var_root(tmp_path: Path) -> Path:
    email_dir = tmp_path / "data" / "synth" / "batch-x" / "email"
    email_dir.mkdir(parents=True)
    (email_dir / "S-email-001.txt").write_text(EMAIL, encoding="utf-8")
    return tmp_path


# -- guards ----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "corpus/out",
        "corpus/out/email/S-email-005.txt",
        "corpus/truth-table.yaml",
        "~/src/EnterpriseRAG-Bench/data",
        "/tmp/enterprise_rag_bench/train.jsonl",
    ],
)
def test_training_path_guard_refuses_the_exams(path: str) -> None:
    with pytest.raises(RuntimeError, match="hold-out violation"):
        script.assert_training_path_allowed(Path(path))


def test_training_path_guard_allows_the_training_set(tmp_path: Path) -> None:
    script.assert_training_path_allowed(tmp_path / "var" / "data" / "training" / "2026-09-22")


def test_models_dir_must_be_gitignored() -> None:
    script.assert_git_ignored(script.REPO_ROOT / "var" / "models")
    with pytest.raises(RuntimeError, match="not gitignored"):
        script.assert_git_ignored(script.REPO_ROOT / "portolan")


# -- exam overlap ----------------------------------------------------------


def test_exam_overlap_flags_a_shared_13_word_run() -> None:
    exam = script._shingles(
        "one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    shared = _record([], record_id="batch-x:a")
    shared = shared.model_copy(
        update={
            "text": "prefix One two three four five six seven eight nine ten eleven twelve THIRTEEN"
        }
    )
    assert script.exam_overlap(shared, exam) == 1
    assert script.exam_overlap(_record([]), exam) == 0


def test_exam_shingles_read_the_real_exam() -> None:
    shingles = script.exam_shingles()
    assert len(shingles) > 1000


# -- conversion ------------------------------------------------------------


def test_segment_of_parses_the_record_id_suffix() -> None:
    assert script.segment_of("EN2001a#seg2of5") == (2, 5)
    assert script.segment_of("S-email-001") is None


def test_locator_for_quote_takes_the_nearest_label_above() -> None:
    text = "[L1] a: hi\n[L2] b: first line\ncontinued here\n[L3] c: bye"
    assert script.locator_for_quote(text, "continued here") == "L2"
    assert script.locator_for_quote(text, "absent") is None


def test_convert_rebuilds_the_envelope_and_uses_the_production_prompts(var_root: Path) -> None:
    stats = script.ConversionStats()
    rows = script.convert_records(
        [_record([_pricing_fact()])], script.EnvelopeIndex(var_root), set(), stats
    )
    assert len(rows) == 1
    system_msg, user_msg, answer_msg = rows[0]["messages"]
    assert system_msg == {
        "role": "system",
        "content": effective_system_prompt(TEACHER_SYSTEM, extraction_result_json_schema()),
    }
    user = user_msg["content"]
    assert "source_date: 2026-02-01" in user
    assert "- Ana Ruiz" in user
    assert "[S-email-001@example.test#p2] We need a 15 percent discount" in user
    answer = json.loads(answer_msg["content"])
    ExtractionResult.model_validate(answer)
    (fact,) = answer["facts"]
    assert fact["locator"] == "S-email-001@example.test#p2"
    assert fact["quote"] in user
    assert stats.locators_remapped == 1
    assert stats.records_out == 1


def test_converted_answer_is_the_teachers_own_and_parses_without_coercion(var_root: Path) -> None:
    stats = script.ConversionStats()
    rows = script.convert_records(
        [_record([_pricing_fact(entities=["Ana Ruiz", "w-buyer-1"])])],
        script.EnvelopeIndex(var_root),
        set(),
        stats,
    )
    answer = json.loads(rows[0]["messages"][2]["content"])
    parsed, rejected = ExtractionResult.parse_lenient(answer)
    assert rejected == [] and parsed.coercions == []
    assert answer["summary"] == "Ana asked for a renewal discount."
    (fact,) = answer["facts"]
    assert (fact["sentiment"], fact["confidence"]) == (-1, 0.9)
    assert fact["entities"] == ["Ana Ruiz"]  # the slug is in no prompt


def test_gate_facts_drops_bad_facts_and_refuses_an_emptied_positive() -> None:
    from types import SimpleNamespace

    long_quote = " ".join(f"w{i}" for i in range(41))
    envelope = SimpleNamespace(
        source_id="S-1", text=f"[L1] a: {long_quote}\n[L2] b: fine quote here\n"
    )
    keep = _pricing_fact(quote="fine quote here", locator="L2")
    over_40 = _pricing_fact(quote=long_quote, locator="L1")
    missing = _pricing_fact(quote="not in the text", locator="L1")
    stats = script.ConversionStats()
    gated = script.gate_facts(_record([keep, over_40, missing]), envelope, stats)
    assert [f.quote for f in gated.facts] == ["fine quote here"]
    assert stats.fact_drops == {"quote_over_40_words": 1, "quote_not_in_prompt_text": 1}
    assert script.gate_facts(_record([over_40]), envelope, stats) is None


def test_convert_skips_a_positive_record_whose_facts_all_drop(var_root: Path) -> None:
    stats = script.ConversionStats()
    bad = _pricing_fact(quote="this sentence is not in the email")
    rows = script.convert_records([_record([bad])], script.EnvelopeIndex(var_root), set(), stats)
    assert rows == []
    assert stats.skipped["all_facts_dropped"] == 1


def test_convert_keeps_a_true_negative(var_root: Path) -> None:
    stats = script.ConversionStats()
    rows = script.convert_records([_record([])], script.EnvelopeIndex(var_root), set(), stats)
    assert len(rows) == 1
    assert json.loads(rows[0]["messages"][2]["content"])["facts"] == []
    assert stats.negatives_out == 1


def test_convert_skips_records_without_teacher_values_or_with_another_prompt(
    var_root: Path,
) -> None:
    legacy = _record([_pricing_fact()]).model_copy(update={"teacher_answer": None})
    no_sentiment = _record([_pricing_fact(sentiment=None)])
    other_prompt = _record([_pricing_fact()]).model_copy(
        update={"system_prompt_sha256": system_prompt_sha256("a different prompt")}
    )
    stats = script.ConversionStats()
    rows = script.convert_records(
        [legacy, no_sentiment, other_prompt], script.EnvelopeIndex(var_root), set(), stats
    )
    assert rows == []
    assert stats.skipped == {"no_teacher_answer": 2, "system_prompt_mismatch": 1}


def test_frozen_prefix_refuses_a_set_with_more_than_one_system_prompt() -> None:
    def row(system: str) -> dict:
        return {"messages": [{"role": "system", "content": system}]}

    assert script.single_system_prompt([row("A"), row("A")]) == "A"
    with pytest.raises(SystemExit, match="one shared system prompt"):
        script.single_system_prompt([row("A"), row("B")])


def test_convert_excludes_exam_overlap_and_unknown_corpora(var_root: Path) -> None:
    record = _record([_pricing_fact()])
    exam = script._shingles(record.text)
    stats = script.ConversionStats()
    assert script.convert_records([record], script.EnvelopeIndex(var_root), exam, stats) == []
    assert stats.exam_overlap_records == ["batch-x:S-email-001"]
    odd = record.model_copy(update={"corpus": "halvard"})
    with pytest.raises(RuntimeError, match="not allowed"):
        script.convert_records([odd], script.EnvelopeIndex(var_root), set(), stats)


def test_convert_skips_a_source_it_cannot_rebuild(tmp_path: Path) -> None:
    stats = script.ConversionStats()
    rows = script.convert_records(
        [_record([_pricing_fact()])], script.EnvelopeIndex(tmp_path), set(), stats
    )
    assert rows == []
    assert stats.skipped["source_not_rebuilt"] == 1


# -- the trainer's own door (guardrails review 2026-09-22, rows 30-31) ------


def test_saas_sales_is_not_an_allowed_corpus(var_root: Path) -> None:
    assert "saas_sales" not in script.ALLOWED_CORPORA
    record = _record([_pricing_fact()]).model_copy(update={"corpus": "saas_sales"})
    with pytest.raises(RuntimeError, match="not allowed"):
        script.convert_records(
            [record], script.EnvelopeIndex(var_root), set(), script.ConversionStats()
        )


def _gated_record(**overrides) -> TrainingRecord:
    """A record the builder's provenance check accepts: a real corpus, an allowed teacher."""
    fields = {"corpus": "enron", "teachers": ["deepseek/deepseek-v3.2"], **overrides}
    return _record([_pricing_fact()]).model_copy(update=fields)


def _write_set(
    root: Path,
    records: list[TrainingRecord],
    *,
    manifest: bool = True,
    gate_enabled: bool = True,
    hashes: bool = True,
) -> Path:
    train_set = root / "set"
    train_set.mkdir()
    for name in ("train.jsonl", "val.jsonl"):
        (train_set / name).write_text(
            "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
        )
    if manifest:
        body: dict = {"contamination": {"enabled": gate_enabled}}
        if hashes:
            body["outputs"] = {
                name: {"sha256": script.sha256_file(train_set / name)}
                for name in ("train.jsonl", "val.jsonl")
            }
        (train_set / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
    return train_set


def _convert_only(tmp_path: Path, train_set: Path) -> int:
    return script.main(
        [
            "--train-set",
            str(train_set),
            "--run-id",
            "door-test",
            "--var-root",
            str(tmp_path / "var"),
            "--models-root",
            str(tmp_path / "models"),
            "--convert-only",
        ]
    )


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"manifest": False}, "manifest.json is missing"),
        ({"gate_enabled": False}, "without the contamination gate"),
        ({"hashes": False}, "records no sha256"),
    ],
)
def test_main_refuses_a_set_the_builder_did_not_write_with_the_gate_on(
    tmp_path: Path, kwargs: dict, fragment: str
) -> None:
    train_set = _write_set(tmp_path, [_gated_record()], **kwargs)
    with pytest.raises(RuntimeError, match=fragment):
        _convert_only(tmp_path, train_set)
    assert not (tmp_path / "models" / "door-test").exists()


def test_main_refuses_a_train_file_edited_after_the_build(tmp_path: Path) -> None:
    train_set = _write_set(tmp_path, [_gated_record()])
    with (train_set / "train.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(_gated_record(record_id="hand-added").model_dump_json() + "\n")
    with pytest.raises(RuntimeError, match="sha256 differs"):
        _convert_only(tmp_path, train_set)


@pytest.mark.parametrize(
    "overrides",
    [
        {"teachers": ["anthropic/claude-sonnet-5"]},  # a closed teacher
        {"teachers": []},  # no provenance at all
        {"corpus": "saas_sales"},  # GPT-4 document text
    ],
)
def test_main_refuses_a_record_that_fails_the_builders_provenance_check(
    tmp_path: Path, overrides: dict
) -> None:
    train_set = _write_set(tmp_path, [_gated_record(), _gated_record(**overrides)])
    with pytest.raises(RuntimeError, match="provenance check"):
        _convert_only(tmp_path, train_set)
    assert not (tmp_path / "models" / "door-test").exists()


def test_main_converts_a_gated_set_the_builder_wrote(tmp_path: Path) -> None:
    train_set = _write_set(tmp_path, [_gated_record()])
    assert _convert_only(tmp_path, train_set) == 0
    conversion = json.loads((tmp_path / "models" / "door-test" / "conversion.json").read_text())
    assert conversion["splits"]["train"]["records_in"] == 1
