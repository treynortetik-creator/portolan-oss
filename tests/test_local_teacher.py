"""Local open-weight teachers (`portolan.training.local_teacher`,
`01-spec/local-teachers-bakeoff-2026-09-22.md`): the `local:` id and its
license table, the builder's allowlist, the $0 spend line, per-document
timings, and `MLXClient` answering through the real extraction chain.

No model is downloaded and MLX is not needed: `fake_mlx` installs stand-in
`mlx` / `mlx_lm` modules whose generator replays canned text, so
`MLXClient`'s own code (chat template, prefix cache, stop rule, metering)
runs unchanged around it."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from portolan.extract.backends import FakeBackend, LLMClientBackend
from portolan.extract.schema import Coverage, ExtractionResult, Fact, SourceType
from portolan.ingest.models import RawSource
from portolan.llm import mlx_local
from portolan.llm.mlx_local import JsonObjectStop, MLXClient, strip_leading_think
from portolan.metering import InMemorySink
from portolan.training import local_teacher
from portolan.training.local_teacher import (
    LocalModel,
    is_local_teacher,
    local_model_id,
    make_local_client,
    permitted_local_teacher_ids,
    require_local_teacher,
)
from portolan.training.pipeline import Document, document_envelopes, run_corpus_single
from portolan.training.records import (
    dropped_path,
    failures_path,
    progress_path,
    read_records,
    records_path,
)
from portolan.training.spend import SpendMeter, default_spend_path
from tests._paths import require_repo_path

QWEN = "local:mlx-community/Qwen3.5-4B-MLX-4bit"
_DOC_TEXT = "the pump beeped three times and shut off completely during the site visit"


# --------------------------------------------------------------------------
# ids and the license table
# --------------------------------------------------------------------------


def test_local_ids_parse_and_keep_repo_case() -> None:
    assert is_local_teacher(QWEN)
    assert is_local_teacher("LOCAL:x/y")
    assert not is_local_teacher("moonshotai/kimi-k2")
    assert local_model_id(QWEN) == "mlx-community/Qwen3.5-4B-MLX-4bit"
    with pytest.raises(ValueError, match="not a local teacher"):
        local_model_id("moonshotai/kimi-k2")
    with pytest.raises(ValueError, match="owner"):
        local_model_id("local:no-owner")


def test_require_local_teacher_refuses_unlisted_and_unlicensed(monkeypatch) -> None:
    assert require_local_teacher(QWEN).license == "Apache-2.0"
    with pytest.raises(ValueError, match="not in"):
        require_local_teacher("local:someone/unlisted-4b")
    monkeypatch.setitem(
        local_teacher.LOCAL_MODELS,
        "someone/no-training-7b",
        LocalModel("someone/no-training-7b", "abc", "custom", "https://example.invalid", False),
    )
    with pytest.raises(ValueError, match="does not permit"):
        require_local_teacher("local:someone/no-training-7b")
    assert "local:someone/no-training-7b" not in permitted_local_teacher_ids()
    with pytest.raises(ValueError):
        make_local_client("local:someone/no-training-7b")


def test_every_table_entry_names_a_license() -> None:
    keys = [k.lower() for k in local_teacher.LOCAL_MODELS]
    assert len(keys) == len(set(keys))
    for info in local_teacher.LOCAL_MODELS.values():
        assert info.license and info.license_url.startswith("https://")
        assert len(info.revision) == 40  # a pinned Hub commit, not a branch


# --------------------------------------------------------------------------
# the builder's allowlist (scripts/ is absent from the docker image)
# --------------------------------------------------------------------------


def _load_script(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_local_allowlist_matches_the_runtime_table() -> None:
    builder = _load_script(
        "build_training_set_local", require_repo_path("scripts", "data", "build_training_set.py")
    )
    permitted = {teacher.lower() for teacher in permitted_local_teacher_ids()}
    assert frozenset(builder.ALLOWED_LOCAL_TEACHERS) == permitted
    for teacher_id, license_note in builder.ALLOWED_LOCAL_TEACHERS.items():
        assert teacher_id in builder.ALLOWED_TEACHER_IDS
        assert license_note.strip()


def test_builder_accepts_only_listed_local_ids() -> None:
    builder = _load_script(
        "build_training_set_local", require_repo_path("scripts", "data", "build_training_set.py")
    )
    assert builder.normalize_model_id(QWEN) == QWEN.lower()
    assert builder.teacher_allowed(QWEN)
    assert builder.record_teachers_allowed([QWEN])
    assert not builder.teacher_allowed("local:someone/unlisted-4b")
    assert not builder.teacher_allowed("local")
    # the OpenRouter variant suffix is still stripped for hosted ids
    assert builder.teacher_allowed("moonshotai/kimi-k2:free")


# --------------------------------------------------------------------------
# spend and timings
# --------------------------------------------------------------------------


def test_spend_meter_writes_a_zero_line_only_when_asked(tmp_path: Path) -> None:
    meter = SpendMeter(path=tmp_path / "spend.json")
    meter.record(model="m", corpus="c", amount_usd=0.0)
    assert meter.snapshot()["per_model_usd"] == {}
    meter.record(model=QWEN, corpus="fda_maude", amount_usd=0.0, allow_zero=True)
    snap = meter.snapshot()
    assert snap["per_model_usd"] == {QWEN: 0.0}
    assert snap["per_corpus_usd"] == {"fda_maude": 0.0}
    assert snap["total_usd"] == 0.0
    meter.record(model=QWEN, corpus="fda_maude", amount_usd=-1.0, allow_zero=True)
    assert meter.total_usd() == 0.0


def _doc(source_id: str, text: str = _DOC_TEXT) -> Document:
    raw = RawSource(
        source_id=source_id,
        source_type=SourceType.DOC,
        source_date=date(2026, 1, 1),
        participants=[],
        text=text,
        metadata={},
    )
    return Document(source_id=source_id, raw=raw, kind="ticket")


def _result(source_id: str, quote: str = "beeped three times and shut off") -> ExtractionResult:
    return ExtractionResult(
        source_id=source_id,
        summary="A pump shut off.",
        entities=[],
        facts=[
            Fact(
                id="f1",
                type="complaint",
                claim="The pump beeped three times and shut off.",
                locator="L1",
                quote=quote,
                sentiment=-1,
                confidence=0.9,
            )
        ],
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )


class _Limiter:
    def acquire(self) -> None:
        pass


class _FreeBackend:
    """Answers every envelope and meters nothing, like a local model's $0 call."""

    def extract(self, envelope, system_prompt, sink, tenant) -> ExtractionResult:
        return _result(envelope.source_id)


def _run_single(tmp_path: Path, teacher: str, backend, documents: list[Document], **kwargs):
    corpus = "fda_maude"
    return run_corpus_single(
        corpus=corpus,
        kind="ticket",
        license_text="public domain",
        documents=documents,
        system_prompt="SYSTEM",
        prompt_variant="v6-type-precision",
        teacher_model=teacher,
        overlap_threshold=0.0,
        backend_factory=lambda model: backend,
        limiter=_Limiter(),
        sink=InMemorySink(),
        tenant="test",
        records_file=records_path(tmp_path, corpus),
        failures_file=failures_path(tmp_path, corpus),
        progress_file=progress_path(tmp_path, corpus),
        dropped_file=dropped_path(tmp_path, corpus),
        spend=SpendMeter(path=default_spend_path(tmp_path)),
        budget_usd=float("inf"),
        judges={},
        sleep=lambda _s: None,
        now=lambda: "2026-09-22T00:00:00-07:00",
        **kwargs,
    )


def test_local_teacher_run_writes_zero_spend_and_document_timings(tmp_path: Path) -> None:
    ticks = iter([10.0, 12.5, 20.0, 21.25])
    timings = tmp_path / "timings.jsonl"
    stats = _run_single(
        tmp_path,
        QWEN,
        _FreeBackend(),
        [_doc("d1"), _doc("d2")],
        timings_file=timings,
        clock_seconds=lambda: next(ticks),
    )
    assert stats.facts_kept == 2
    records = read_records(records_path(tmp_path, "fda_maude"))
    assert [r.teachers for r in records] == [[QWEN], [QWEN]]
    spend = SpendMeter(path=default_spend_path(tmp_path)).snapshot()
    assert spend["per_model_usd"] == {QWEN: 0.0}
    assert spend["total_usd"] == 0.0
    lines = [json.loads(line) for line in timings.read_text().splitlines()]
    assert [(t["source_id"], t["seconds"], t["envelopes"]) for t in lines] == [
        ("d1", 2.5, 1),
        ("d2", 1.25, 1),
    ]
    assert {t["teacher"] for t in lines} == {QWEN}


def test_hosted_teacher_writes_no_zero_line(tmp_path: Path) -> None:
    _run_single(tmp_path, "moonshotai/kimi-k2", _FreeBackend(), [_doc("d1")])
    assert SpendMeter(path=default_spend_path(tmp_path)).snapshot()["per_model_usd"] == {}
    # a paid call's real cost is still recorded under the hosted id
    _run_single(
        tmp_path, "moonshotai/kimi-k2", FakeBackend(lambda e: _result(e.source_id)), [_doc("d2")]
    )
    assert (
        SpendMeter(path=default_spend_path(tmp_path)).snapshot()["per_model_usd"][
            "moonshotai/kimi-k2"
        ]
        > 0
    )


# --------------------------------------------------------------------------
# MLXClient through stand-in mlx modules
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Chat template as token ids: system turn [1, 2, 3], user turn [4, 5 ...],
    generation prompt [9]. The system turn is a stable prefix, as with a
    real template, so the client's prefix cache path runs. `folds=True`
    mimics a template with no system turn of its own: it refuses a
    system-only conversation and prints the system text inside the user turn."""

    def __init__(self, folds: bool = False) -> None:
        self.folds = folds
        self.template_calls: list[dict] = []

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        self.template_calls.append(kwargs)
        if self.folds and all(m["role"] == "system" for m in messages):
            raise ValueError("Conversation roles must alternate user/assistant")
        ids: list[int] = [7] if self.folds else []
        for m in messages:
            ids += [1, 2, 3] if m["role"] == "system" else [4, 5, len(m["content"]) % 97]
        return ids + ([9] if add_generation_prompt else [])


class _FakeCache:
    def __init__(self) -> None:
        self.offset = 0
        self.state = None


@pytest.fixture
def fake_mlx(monkeypatch) -> Iterator[SimpleNamespace]:
    """Stand-in `mlx.core`, `mlx_lm.load`, `stream_generate`, cache and
    sampler. `harness.replies` is a list of lists of text pieces, one list
    per generation call; `harness.calls` records what each call received."""
    harness = SimpleNamespace(
        replies=[],
        calls=[],
        tokenizer=_FakeTokenizer(),
        loads=[],
        tokenizer_configs=[],
        trims=[],
        trimmable=True,
    )

    core = types.ModuleType("mlx.core")
    core.array = np.array
    core.eval = lambda *_a, **_k: None
    core.get_peak_memory = lambda: 3_500_000_000
    core.clear_cache = lambda: None
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = core

    def model(tokens, cache):
        cache[0].offset += tokens.shape[1]

    def load(model_id, adapter_path=None, revision=None, tokenizer_config=None):
        harness.loads.append((model_id, revision))
        harness.tokenizer_configs.append(tokenizer_config)
        return model, harness.tokenizer

    def stream_generate(model, tokenizer, *, prompt, max_tokens, sampler, prompt_cache):
        harness.calls.append(
            {
                "prompt": list(prompt),
                "max_tokens": max_tokens,
                "temp": sampler,
                "cache": prompt_cache,
            }
        )
        pieces = harness.replies.pop(0)
        if prompt_cache is not None:
            prompt_cache[0].offset += len(prompt) + len(pieces)
        for i, piece in enumerate(pieces, start=1):
            last = i == len(pieces)
            yield SimpleNamespace(
                text=piece, generation_tokens=i, finish_reason="stop" if last else None
            )

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = load
    generate = types.ModuleType("mlx_lm.generate")
    generate.stream_generate = stream_generate
    models = types.ModuleType("mlx_lm.models")
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda _model: [_FakeCache()]

    def trim_prompt_cache(prompt_cache, n):
        harness.trims.append(n)
        prompt_cache[0].offset -= n
        return n

    cache.trim_prompt_cache = trim_prompt_cache
    cache.can_trim_prompt_cache = lambda _cache: harness.trimmable
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda temp: temp

    for name, module in {
        "mlx": mlx_pkg,
        "mlx.core": core,
        "mlx_lm": mlx_lm,
        "mlx_lm.generate": generate,
        "mlx_lm.models": models,
        "mlx_lm.models.cache": cache,
        "mlx_lm.sample_utils": sample_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(mlx_local, "mlx_available", lambda: True)
    yield harness


def test_free_text_is_not_cut_at_a_closing_brace(fake_mlx) -> None:
    fake_mlx.replies = [["Dear team,", " the {draft} ", "closes } here", " and goes on."]]
    client = MLXClient("x/writer-4b")
    result = client.complete(
        system="S", user="write", surface="chat", max_tokens=100, temperature=0.8
    )
    assert result.text == "Dear team, the {draft} closes } here and goes on."
    assert fake_mlx.calls[0]["temp"] == 0.8
    assert fake_mlx.calls[0]["prompt"] == [4, 5, 5, 9]  # after the cached system prefix
    event = client.sink.events[-1]
    assert event.cost_estimate == 0.0
    assert event.extra["mlx"]["finish_reason"] == "stop"
    assert event.extra["mlx"]["peak_memory_bytes"] == 3_500_000_000
    assert all(call["enable_thinking"] is False for call in fake_mlx.tokenizer.template_calls)


def test_json_answer_stops_at_the_first_object_after_a_think_block(fake_mlx) -> None:
    fake_mlx.replies = [["<think>plan: {a} then {b}", "</think>\n", '{"k": 1}', " trailing"]]
    client = MLXClient("x/labeler-4b")
    result = client.complete(
        system="S", user="u", json_schema={"type": "object"}, surface="extract", max_tokens=50
    )
    assert result.data == {"k": 1}
    assert "trailing" not in result.text and "<think>" not in result.text
    assert client.sink.events[-1].extra["mlx"]["finish_reason"] == "json_complete"


def test_the_prefix_cache_is_trimmed_back_after_each_call(fake_mlx) -> None:
    fake_mlx.replies = [["{}"], ["{}"]]
    client = MLXClient("x/labeler-4b")
    for user in ("one", "second"):
        client.complete(system="S", user=user, json_schema={"type": "object"}, surface="extract")
    assert [c["cache"] for c in fake_mlx.calls] == [client._prefix_cache] * 2
    assert fake_mlx.trims == [5, 5]  # 4 prompt tokens + 1 generated, each time
    assert client._prefix_cache[0].offset == 3


def test_an_untrimmable_cache_runs_each_call_on_a_copy(fake_mlx) -> None:
    fake_mlx.trimmable = False  # recurrent or sliding-window layers
    fake_mlx.replies = [["{}"], ["{}"]]
    client = MLXClient("x/hybrid-4b")
    for user in ("one", "second"):
        client.complete(system="S", user=user, json_schema={"type": "object"}, surface="extract")
    used = [c["cache"] for c in fake_mlx.calls]
    assert all(cache is not client._prefix_cache for cache in used)
    assert used[0] is not used[1]
    assert fake_mlx.trims == []
    assert client._prefix_cache[0].offset == 3  # the stored prefix never moved
    assert [e.cached_tokens for e in client.sink.events] == [3, 3]


def test_a_template_without_a_system_turn_caches_the_shared_prefix(fake_mlx) -> None:
    fake_mlx.tokenizer = _FakeTokenizer(folds=True)
    fake_mlx.replies = [["{}"]]
    client = MLXClient("x/folding-4b")
    client.complete(system="S", user="abc", json_schema={"type": "object"}, surface="extract")
    # renders share [7, 1, 2, 3, 4, 5]; the last shared token is left out
    assert client._prefix_ids == [7, 1, 2, 3, 4]
    assert fake_mlx.calls[0]["prompt"] == [5, 3, 9]
    assert client.sink.events[-1].cached_tokens == 5


def test_json_object_stop_waits_out_a_split_think_tag() -> None:
    stop = JsonObjectStop()
    assert not stop.feed("<thi")
    assert not stop.feed("nk>{x}")
    assert not stop.feed("</think>")
    assert stop.feed('{"a": "}"}')
    assert strip_leading_think("  <think>{x}</think>\n{}") == "{}"
    assert strip_leading_think("{} <think>x</think>") == "{} <think>x</think>"


def test_unload_releases_the_model(fake_mlx) -> None:
    fake_mlx.replies = [["{}"]]
    client = MLXClient("x/labeler-4b")
    client.complete(system="S", user="u", json_schema={"type": "object"}, surface="extract")
    assert client._model is not None
    client.unload()
    assert client._model is None and client._prefix_cache is None


def test_local_client_answers_through_the_production_extraction_chain(fake_mlx) -> None:
    doc = _doc("d1")
    envelope = document_envelopes(doc)[0]
    fake_mlx.replies = [[_result("d1").model_dump_json()]]
    client = make_local_client(QWEN)
    result = LLMClientBackend(client, model=client.model_id).extract(
        envelope, "SYSTEM", InMemorySink(), "t"
    )
    assert [f.quote for f in result.facts] == ["beeped three times and shut off"]
    assert fake_mlx.loads == [
        ("mlx-community/Qwen3.5-4B-MLX-4bit", "32f3e8ecf65426fc3306969496342d504bfa13f3")
    ]


def test_a_models_tokenizer_settings_reach_the_loader(fake_mlx) -> None:
    fake_mlx.replies = [["{}"], ["{}"]]
    for teacher in (QWEN, "local:mlx-community/Ministral-3-8B-Instruct-2512-4bit"):
        client = make_local_client(teacher)
        client.complete(system="S", user="u", json_schema={"type": "object"}, surface="extract")
    assert fake_mlx.tokenizer_configs == [None, {"fix_mistral_regex": True}]


# --------------------------------------------------------------------------
# scripts/data/silver_label.py --teacher local:<id>
# --------------------------------------------------------------------------


def test_silver_label_runs_a_local_teacher_end_to_end(fake_mlx, monkeypatch, tmp_path) -> None:
    silver = _load_script(
        "silver_label_local", require_repo_path("scripts", "data", "silver_label.py")
    )
    monkeypatch.setattr(
        silver, "documents_for_corpus", lambda corpus, root, window_size: iter([_doc("d1")])
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_mlx.replies = [[_result("d1").model_dump_json()]]

    code = silver.main(
        [
            "--corpus",
            "fda_maude",
            "--mode",
            "single",
            "--teacher",
            QWEN,
            "--var-root",
            str(tmp_path),
        ]
    )

    assert code == 0
    (record,) = read_records(records_path(tmp_path, "fda_maude"))
    assert record.teachers == [QWEN]
    assert record.teacher_answer is not None and record.teacher_answer.summary == "A pump shut off."
    assert record.facts[0].sentiment == -1 and record.facts[0].confidence == 0.9
    assert record.system_prompt_sha256
    spend = json.loads(default_spend_path(tmp_path).read_text())
    assert spend["per_model_usd"] == {QWEN: 0.0}
    timing = json.loads((records_path(tmp_path, "fda_maude").parent / "timings.jsonl").read_text())
    assert timing["source_id"] == "d1" and timing["seconds"] >= 0


def test_silver_label_refuses_an_unlisted_local_teacher(monkeypatch, tmp_path, capsys) -> None:
    silver = _load_script(
        "silver_label_local", require_repo_path("scripts", "data", "silver_label.py")
    )
    code = silver.main(
        [
            "--corpus",
            "fda_maude",
            "--mode",
            "single",
            "--teacher",
            "local:someone/unlisted-4b",
            "--var-root",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "not in" in capsys.readouterr().err
    assert not records_path(tmp_path, "fda_maude").exists()
