"""Tests for `portolan.bench.__main__`'s CLI wiring: argument defaults and
an end-to-end `extract` run through the `fake` backend (no network), which
is also Step 4's proof that `--concurrency` works without a live model."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import portolan.llm as llm_module
from portolan.bench.__main__ import _live_backend, build_parser, main
from portolan.extract.backends import LLMClientBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus" / "out"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"


def test_extract_defaults_to_concurrency_1_and_judge_off() -> None:
    parser = build_parser()
    args = parser.parse_args(["extract", "--backend", "fake"])
    assert args.concurrency == 1
    assert args.judge is False


def test_extract_defaults_samples_1_temperature_0_no_rescore_dir() -> None:
    parser = build_parser()
    args = parser.parse_args(["extract", "--backend", "fake"])
    assert args.samples == 1
    assert args.temperature == 0.0
    assert args.rescore_dir is None


def test_extract_parses_samples_and_temperature_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["extract", "--backend", "fake", "--samples", "3", "--temperature", "0.2"]
    )
    assert args.samples == 3
    assert args.temperature == 0.2


def test_extract_parses_concurrency_and_judge_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["extract", "--backend", "fake", "--concurrency", "4", "--judge"])
    assert args.concurrency == 4
    assert args.judge is True


def test_cmd_extract_runs_end_to_end_with_fake_backend_and_concurrency(tmp_path: Path) -> None:
    """No network, no model: exercises `main()` -> `_cmd_extract` ->
    `run_extraction(..., concurrency=3)` -> `score_extraction` -> `write`
    on a handful of real sources, proving the `--concurrency` flag is
    wired end to end through the actual CLI entry point."""
    source_ids = ["S-ticket-019", "S-ticket-024", "S-ticket-027", "S-ticket-031", "S-ticket-038"]
    exit_code = main(
        [
            "extract",
            "--backend",
            "fake",
            "--corpus",
            str(CORPUS_ROOT),
            "--manifest",
            str(MANIFEST_PATH),
            "--truth-table",
            str(TRUTH_TABLE_PATH),
            "--out",
            str(tmp_path),
            "--source-ids",
            *source_ids,
            "--concurrency",
            "3",
        ]
    )
    assert exit_code == 0

    json_path = tmp_path / "bench-*-fake.json"
    matches = list(tmp_path.glob("bench-*-fake.json"))
    assert len(matches) == 1, f"expected one JSON report, found {matches} (looked for {json_path})"
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    assert payload["extraction"]["sources_scored"] == len(source_ids)
    # Every one of these five tickets has exactly one planted fact (F13,
    # F23, F19, F02, F05) and the fake backend emits exactly what's
    # planted, so this run should score perfectly.
    assert payload["extraction"]["recall"] == 1.0


def test_cmd_extract_wires_samples_and_records_temperature(tmp_path: Path) -> None:
    """`--samples 3` runs end to end through the real CLI entry point (the
    `fake` backend returns the same canned result every call, so the
    union is identical to one sample here -- this proves the flag is
    wired through to `run_extraction`, not that fake backend output
    varies), and `--temperature` is recorded into the written report."""
    source_ids = ["S-ticket-019"]
    exit_code = main(
        [
            "extract",
            "--backend",
            "fake",
            "--corpus",
            str(CORPUS_ROOT),
            "--manifest",
            str(MANIFEST_PATH),
            "--truth-table",
            str(TRUTH_TABLE_PATH),
            "--out",
            str(tmp_path),
            "--source-ids",
            *source_ids,
            "--samples",
            "3",
            "--temperature",
            "0",
        ]
    )
    assert exit_code == 0
    matches = list(tmp_path.glob("bench-*-fake.json"))
    assert len(matches) == 1
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    assert payload["extraction"]["recall"] == 1.0
    assert payload["temperature_requested"] == 0.0
    assert payload["samples"] == 3


def test_cmd_extract_rescore_dir_full_rescore_end_to_end(tmp_path: Path) -> None:
    """`--rescore-dir` finds every saved `bench-*.json` under a directory,
    fully re-scores each from its own `produced_by_source` (no backend, no
    model call), and writes a summary. Uses two of the corpus's real
    RSS-planted-fact sources so the RSS locator fix is what visibly moves
    recall on the rescore."""
    saved_dir = tmp_path / "saved"
    out_dir = tmp_path / "rescored"

    # A url-only RSS locator (no date) -- what the old, exact-string RSS
    # rule rejected and the current one (portolan.bench.locators._rss_url)
    # accepts. Simulate "already scored and saved" by running one real
    # extract pass (fake backend, hand-built canned result) and letting
    # the CLI write it the normal way.
    from portolan.extract.backends import FakeBackend
    from portolan.extract.schema import Coverage, ExtractionResult, Fact, FactType

    source_id = "S-rss-006"
    produced = Fact(
        id="a",
        type=FactType.CONTACT_CHANGE,
        claim=(
            "Renata Solis became Chief Financial Officer at Larkspur Regional Health "
            "effective 2026-09-01."
        ),
        entities=["Larkspur Regional Health"],
        locator="https://feeds.example-newswire.test/portolan-corpus/s-rss-006",
        quote=None,
        speaker="customer",
        sentiment=0,
        confidence=0.8,
    )
    canned = ExtractionResult(
        source_id=source_id,
        summary="",
        entities=[],
        facts=[produced],
        actions=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    from portolan.bench.extraction import run_extraction, score_extraction
    from portolan.bench.report import BenchReport, write
    from portolan.bench.truth import Truth

    truth = Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)
    run_result = run_extraction(
        truth.manifest,
        FakeBackend({source_id: canned}),
        source_ids=[source_id],
        segment=False,
        corpus_root=CORPUS_ROOT,
    )
    score = score_extraction(truth, run_result)
    write(BenchReport(backend="fake", date=dt.date.today(), extraction=score), saved_dir)

    exit_code = main(
        [
            "extract",
            "--corpus",
            str(CORPUS_ROOT),
            "--manifest",
            str(MANIFEST_PATH),
            "--truth-table",
            str(TRUTH_TABLE_PATH),
            "--out",
            str(out_dir),
            "--rescore-dir",
            str(saved_dir),
        ]
    )
    assert exit_code == 0

    summary_path = out_dir / "rescore-summary.json"
    assert summary_path.exists()
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["recall_after"] == 1.0
    assert rows[0]["lenient_recall_after"] == 1.0


# --------------------------------------------------------------------------
# Fix instructions item 4: `_live_backend` primes an OpenRouterClient's
# process-wide JSON-strategy cache up front (once, before the first source
# is ever extracted) via `client.warm_json_strategy(model)`, when the
# client `get_llm()` resolved to provides that method. No real
# `OpenRouterClient` here (no network in tests) -- a bare double proves
# `_live_backend` calls it correctly and doesn't require it.
# --------------------------------------------------------------------------


def test_live_backend_warms_json_strategy_when_the_client_supports_it(monkeypatch) -> None:
    warmed: list[str | None] = []

    class _FakeOpenRouterClient:
        def warm_json_strategy(self, model: str | None) -> str:
            warmed.append(model)
            return "json_schema"

    # `**kwargs` (not a bare zero-arg lambda) because `_live_backend` now
    # calls `get_llm(provider=name)` (infra-2b wave, 2026-09-18) --
    # asserted on directly in `test_live_backend_passes_its_own_name_as_
    # the_provider` below.
    monkeypatch.setattr(llm_module, "get_llm", lambda **kwargs: _FakeOpenRouterClient())

    backend = _live_backend("openrouter", "deepseek/deepseek-v4-flash-0731")

    assert warmed == ["deepseek/deepseek-v4-flash-0731"]
    assert isinstance(backend, LLMClientBackend)


def test_live_backend_skips_warm_up_for_a_client_without_it(monkeypatch) -> None:
    class _PlainClient:
        def complete(self, **kwargs):  # pragma: no cover - never called here
            raise NotImplementedError

    monkeypatch.setattr(llm_module, "get_llm", lambda **kwargs: _PlainClient())

    # Must not raise (no warm_json_strategy attribute) and must still
    # build the backend normally.
    backend = _live_backend("agent_sdk", None)
    assert isinstance(backend, LLMClientBackend)


# --------------------------------------------------------------------------
# infra-2b wave, 2026-09-18: real bug found live -- `_live_backend` used to
# call `get_llm()` with NO arguments, which obeys only `settings.
# llm_provider` (the `LLM_PROVIDER` env var / `.env`), completely ignoring
# its own `name` parameter (the CLI's `--backend` choice). A checkout with
# `.env` set to `LLM_PROVIDER=openrouter` silently ran every `--backend
# agent_sdk` request through OpenRouter instead -- three Claude "via
# subscription" runs cost $9.98 in real OpenRouter dollars against a bare
# `claude-sonnet-5` model id before anyone noticed the unusable output.
# --------------------------------------------------------------------------


def test_live_backend_passes_its_own_name_as_the_provider(monkeypatch) -> None:
    seen_providers: list[str | None] = []

    class _PlainClient:
        def complete(self, **kwargs):  # pragma: no cover - never called here
            raise NotImplementedError

    def fake_get_llm(**kwargs):
        seen_providers.append(kwargs.get("provider"))
        return _PlainClient()

    monkeypatch.setattr(llm_module, "get_llm", fake_get_llm)

    _live_backend("agent_sdk", None)

    assert seen_providers == ["agent_sdk"]


def test_live_backend_with_agent_sdk_constructs_agent_sdk_client_even_when_env_says_openrouter(
    monkeypatch, tmp_path: Path
) -> None:
    # The real end-to-end scenario, through the REAL (unmocked) get_llm():
    # the environment/`.env` says openrouter, but `--backend agent_sdk`
    # must still win.
    from portolan.config import Settings
    from portolan.llm import factory as factory_module
    from portolan.llm.agent_sdk import AgentSDKClient

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-not-be-needed")
    cfg = Settings(
        _env_file=None, database_url="postgresql://localhost/x", store_root=tmp_path / "store"
    )
    assert cfg.llm_provider == "openrouter"  # confirms the env really is set the "wrong" way
    monkeypatch.setattr(factory_module, "default_settings", cfg)

    backend = _live_backend("agent_sdk", None)

    assert isinstance(backend, LLMClientBackend)
    assert isinstance(backend.client, AgentSDKClient)


def test_cmd_extract_prints_the_resolved_backend_and_model(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """`_cmd_extract` names the resolved provider/model at run start
    (infra-2b wave, 2026-09-18) so a provider/env mismatch is visible at a
    glance instead of discovered from unusable output after real spend.
    `_live_backend` itself is stubbed (no network, no get_llm call) --
    this test is about the print line, not backend construction (see
    `test_live_backend_passes_its_own_name_as_the_provider` for that)."""
    import portolan.bench.__main__ as main_module
    from portolan.bench.fixtures import perfect_results
    from portolan.bench.truth import Manifest, Truth
    from portolan.extract.backends import FakeBackend

    manifest = Manifest.load(MANIFEST_PATH)
    truth = Truth.for_manifest(manifest, truth_table_path=TRUTH_TABLE_PATH)
    source_ids = ["S-ticket-019"]
    canned = perfect_results(truth)
    canned = {sid: canned[sid] for sid in source_ids if sid in canned}

    monkeypatch.setattr(main_module, "_live_backend", lambda name, model: FakeBackend(canned))

    exit_code = main(
        [
            "extract",
            "--backend",
            "agent_sdk",
            "--model",
            "claude-haiku-4-5",
            "--corpus",
            str(CORPUS_ROOT),
            "--manifest",
            str(MANIFEST_PATH),
            "--truth-table",
            str(TRUTH_TABLE_PATH),
            "--out",
            str(tmp_path),
            "--source-ids",
            *source_ids,
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "backend resolved: provider='agent_sdk' model='claude-haiku-4-5'" in captured.err


def test_cmd_extract_prints_provider_default_when_no_model_given(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import portolan.bench.__main__ as main_module
    from portolan.bench.fixtures import perfect_results
    from portolan.bench.truth import Manifest, Truth
    from portolan.extract.backends import FakeBackend

    manifest = Manifest.load(MANIFEST_PATH)
    truth = Truth.for_manifest(manifest, truth_table_path=TRUTH_TABLE_PATH)
    source_ids = ["S-ticket-019"]
    canned = perfect_results(truth)
    canned = {sid: canned[sid] for sid in source_ids if sid in canned}

    monkeypatch.setattr(main_module, "_live_backend", lambda name, model: FakeBackend(canned))

    exit_code = main(
        [
            "extract",
            "--backend",
            "agent_sdk",
            "--corpus",
            str(CORPUS_ROOT),
            "--manifest",
            str(MANIFEST_PATH),
            "--truth-table",
            str(TRUTH_TABLE_PATH),
            "--out",
            str(tmp_path),
            "--source-ids",
            *source_ids,
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "backend resolved: provider='agent_sdk' model='<provider default>'" in captured.err
