"""Tests for `portolan.llm.mlx_local` (fine-tune shakedown, 2026-09-22) and
the bench's `--backend mlx` wiring. None of these need `mlx`: the client's
MLX imports are lazy, and everything here exercises the pure helpers, the
constructor, and the CLI plumbing. The MLX numerics tests live in
`tests/test_mlx_numerics.py`, which skips at collection when `mlx` is
absent (CI, the Docker image)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import portolan.bench.__main__ as bench_main
from portolan.extract.schema import EntityCoercion, ExtractionResult, extraction_result_json_schema
from portolan.llm.mlx_local import (
    PREFIX_MODE_FROZEN_BASE,
    PREFIX_MODE_STANDARD,
    JsonObjectStop,
    MLXClient,
    effective_system_prompt,
    lora_disabled,
    read_prefix_mode,
)
from portolan.llm.openrouter import JSON_INSTRUCTION_TEMPLATE
from portolan.metering import InMemorySink, UsageEvent


def test_effective_system_prompt_appends_the_same_schema_block_as_openrouter() -> None:
    schema = extraction_result_json_schema()
    out = effective_system_prompt("SYSTEM", schema)
    assert out == "SYSTEM" + JSON_INSTRUCTION_TEMPLATE.format(schema=json.dumps(schema))
    assert effective_system_prompt("SYSTEM", None) == "SYSTEM"


def test_json_object_stop_closes_on_the_first_top_level_object() -> None:
    stop = JsonObjectStop()
    assert not stop.feed('Sure! {"a": {"b": "}"}, ')
    assert not stop.feed('"c": "x\\"}"')
    assert stop.feed("} trailing {")
    assert stop.done


def test_json_object_stop_ignores_prose_quotes_before_the_object() -> None:
    stop = JsonObjectStop()
    assert not stop.feed('He said "hi" and then ')
    assert stop.feed("{}")


def test_read_prefix_mode(tmp_path: Path) -> None:
    assert read_prefix_mode(None) == PREFIX_MODE_STANDARD
    assert read_prefix_mode(tmp_path) == PREFIX_MODE_STANDARD
    (tmp_path / "adapter_config.json").write_text(json.dumps({"num_layers": 4}))
    assert read_prefix_mode(tmp_path) == PREFIX_MODE_STANDARD
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"portolan_prefix_mode": PREFIX_MODE_FROZEN_BASE})
    )
    assert read_prefix_mode(tmp_path) == PREFIX_MODE_FROZEN_BASE
    (tmp_path / "adapter_config.json").write_text(json.dumps({"portolan_prefix_mode": "odd"}))
    with pytest.raises(ValueError, match="unknown portolan_prefix_mode"):
        read_prefix_mode(tmp_path)


def test_lora_disabled_zeroes_and_restores_every_lora_scale() -> None:
    lora = SimpleNamespace(lora_a=object(), scale=20.0)
    plain = SimpleNamespace(scale=3.0)
    model = SimpleNamespace(named_modules=lambda: [("a", lora), ("b", plain)])
    with lora_disabled(model) as switched:
        assert switched == 1
        assert lora.scale == 0.0
        assert plain.scale == 3.0
    assert lora.scale == 20.0


def test_mlx_client_constructs_without_loading_mlx(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"portolan_prefix_mode": PREFIX_MODE_FROZEN_BASE})
    )
    client = MLXClient("some/model", adapter_path=tmp_path)
    assert client.prefix_mode == PREFIX_MODE_FROZEN_BASE
    assert client.model_key == "mlx:some/model"
    assert client._model is None


def test_mlx_client_using_sink_swaps_and_restores() -> None:
    client = MLXClient("some/model")
    original = client.sink
    other = InMemorySink()
    with client.using_sink(other, "bench"):
        assert client.sink is other and client.tenant == "bench"
    assert client.sink is original


def test_mlx_client_refuses_a_different_model_id() -> None:
    client = MLXClient("some/model")
    with pytest.raises(ValueError, match="holds 'some/model'"):
        client.complete(system="s", user="u", model="other/model", surface="extract")


def test_bench_parser_accepts_mlx_backend_flags() -> None:
    args = bench_main.build_parser().parse_args(
        [
            "extract",
            "--backend",
            "mlx",
            "--model",
            "m",
            "--adapter-path",
            "var/models/x/adapters",
            "--max-tokens",
            "512",
        ]
    )
    assert (args.backend, args.model, args.adapter_path, args.max_tokens) == (
        "mlx",
        "m",
        "var/models/x/adapters",
        512,
    )


def test_mlx_backend_refuses_without_mlx_installed(monkeypatch) -> None:
    import portolan.llm.mlx_local as mlx_local

    monkeypatch.setattr(mlx_local, "mlx_available", lambda: False)
    with pytest.raises(SystemExit, match="uv sync --group mlx"):
        bench_main._mlx_backend("m", None, None)
    with pytest.raises(SystemExit, match="needs --model"):
        bench_main._mlx_backend(None, None, None)


def test_mlx_backend_wraps_the_client_in_llm_client_backend(monkeypatch) -> None:
    import portolan.llm.mlx_local as mlx_local
    from portolan.extract.backends import LLMClientBackend

    monkeypatch.setattr(mlx_local, "mlx_available", lambda: True)
    backend = bench_main._mlx_backend("m", None, 256)
    assert isinstance(backend, LLMClientBackend)
    assert isinstance(backend.client, MLXClient)
    assert backend.client.max_tokens_cap == 256


def test_cmd_extract_refuses_mlx_with_concurrency(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="concurrency 1"):
        bench_main.main(
            [
                "extract",
                "--backend",
                "mlx",
                "--model",
                "m",
                "--concurrency",
                "2",
                "--source-ids",
                "S-ticket-019",
                "--out",
                str(tmp_path),
            ]
        )


def test_parser_counts_reads_rejections_coercions_and_call_labels() -> None:
    result = ExtractionResult(
        source_id="S-1",
        summary="",
        coverage={"segments_read": 1, "segments_total": 1},
        rejected=[{"index": 0, "error": "bad", "raw": {}}],
        coercions=[
            EntityCoercion(index=0, mention="", field="sentiment", original=None, applied=0),
            EntityCoercion(index=1, mention="", field="sentiment", original="neg", applied=-1),
        ],
    )
    sink = InMemorySink()

    def _event(tokens: int, extra: dict) -> UsageEvent:
        return UsageEvent(
            tenant="t",
            surface="extract",
            model_key="mlx:m",
            input_tokens=tokens,
            output_tokens=1,
            cached_tokens=0,
            cost_estimate=0.0,
            request_id="r",
            timestamp="now",
            extra=extra,
        )

    sink.record(_event(10, {"mlx": {"retry": False, "finish_reason": "json_complete"}}))
    sink.record(_event(10, {"mlx": {"retry": True, "finish_reason": "length"}}))
    sink.record(_event(0, {"coercions": 2}))  # the backend's zero-token lenient event
    run_result = SimpleNamespace(sink=sink, failed_sources={"S-2": "no JSON"})
    counts = bench_main._parser_counts(_Mapping({"S-1": result}, run_result))
    assert counts["rejected_facts"] == 1
    assert counts["coercions_total"] == 2
    assert counts["coercions_by_field"] == {"sentiment": 2}
    assert counts["failed_sources"] == {"S-2": "no JSON"}
    assert counts["model_calls"] == 2
    assert counts["retry_calls"] == 1
    assert counts["finish_reasons"] == {"json_complete": 1, "length": 1}


class _Mapping(dict):
    """A `dict[source_id, ExtractionResult]` carrying `sink`/`failed_sources`,
    the shape `ExtractionRunResult` has."""

    def __init__(self, data: dict, extras: SimpleNamespace) -> None:
        super().__init__(data)
        self.sink = extras.sink
        self.failed_sources = extras.failed_sources
