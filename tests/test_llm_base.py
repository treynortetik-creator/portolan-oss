"""Tests for portolan.llm.base: the LLMResult shape and the exception
hierarchy every implementation shares."""

from __future__ import annotations

import pytest

from portolan.llm.base import (
    DEFAULT_DETERMINISTIC_TEMPERATURE,
    DETERMINISTIC_TEMPERATURE_SURFACES,
    TEMPERATURE_ENV_VAR,
    LLMError,
    LLMJsonError,
    LLMResult,
    claude_takes_forced_tool_choice,
    claude_takes_sampling_params,
    resolve_temperature,
)


def test_llm_result_round_trips_fields() -> None:
    result = LLMResult(
        text="hello",
        data={"a": 1},
        input_tokens=10,
        output_tokens=5,
        cached_tokens=2,
        model="claude-haiku-4-5",
    )
    assert result.text == "hello"
    assert result.data == {"a": 1}
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cached_tokens == 2
    assert result.model == "claude-haiku-4-5"


def test_llm_result_allows_none_text_and_data() -> None:
    result = LLMResult(
        text=None, data=None, input_tokens=0, output_tokens=0, cached_tokens=0, model="x"
    )
    assert result.text is None
    assert result.data is None


def test_llm_json_error_is_an_llm_error() -> None:
    assert issubclass(LLMJsonError, LLMError)
    assert isinstance(LLMJsonError("boom"), LLMError)


# --------------------------------------------------------------------------
# resolve_temperature (infra-2b wave, 2026-09-18):
# `01-spec/harness-review-2026-09-18.md` found temperature not settable
# anywhere in the LLM client stack; this is the shared rule every client
# now uses.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_temperature_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TEMPERATURE_ENV_VAR, raising=False)
    yield
    monkeypatch.delenv(TEMPERATURE_ENV_VAR, raising=False)


def test_deterministic_surfaces_are_extract_audit_and_merge() -> None:
    # `merge`: the compare verdict decides what is written to the graph.
    assert DETERMINISTIC_TEMPERATURE_SURFACES == frozenset({"extract", "audit", "merge"})


@pytest.mark.parametrize(
    ("model", "sampling", "forced_tool"),
    [
        ("claude-haiku-4-5", True, True),
        ("anthropic.claude-haiku-4-5", True, True),
        ("claude-haiku-5-5", False, True),
        ("claude-sonnet-5", False, True),
        ("claude-sonnet-5-5", False, False),
        ("claude-opus-5-5", False, False),
        ("anthropic.claude-opus-5-5", False, False),
        ("claude-fable-5-2", False, False),
        ("claude-fable-5", False, True),
    ],
)
def test_claude_request_shape_by_model(model: str, sampling: bool, forced_tool: bool) -> None:
    assert claude_takes_sampling_params(model) is sampling
    assert claude_takes_forced_tool_choice(model) is forced_tool


@pytest.mark.parametrize("surface", ["extract", "audit", "merge"])
def test_resolve_temperature_defaults_to_zero_for_deterministic_surfaces(surface: str) -> None:
    assert resolve_temperature(None, surface) == DEFAULT_DETERMINISTIC_TEMPERATURE == 0.0


@pytest.mark.parametrize("surface", ["report", "chat", "consolidate", "embed"])
def test_resolve_temperature_is_none_for_every_other_surface(surface: str) -> None:
    assert resolve_temperature(None, surface) is None


def test_resolve_temperature_explicit_argument_always_wins() -> None:
    assert resolve_temperature(0.7, "extract") == 0.7
    assert resolve_temperature(0.7, "report") == 0.7
    assert resolve_temperature(0.0, "chat") == 0.0


def test_resolve_temperature_env_var_overrides_default_for_deterministic_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TEMPERATURE_ENV_VAR, "0.5")
    assert resolve_temperature(None, "extract") == 0.5
    assert resolve_temperature(None, "audit") == 0.5


def test_resolve_temperature_env_var_has_no_effect_on_other_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TEMPERATURE_ENV_VAR, "0.5")
    assert resolve_temperature(None, "chat") is None
    assert resolve_temperature(None, "report") is None


def test_resolve_temperature_ignores_an_unparseable_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TEMPERATURE_ENV_VAR, "not-a-number")
    assert resolve_temperature(None, "extract") == DEFAULT_DETERMINISTIC_TEMPERATURE


def test_resolve_temperature_explicit_argument_wins_over_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TEMPERATURE_ENV_VAR, "0.5")
    assert resolve_temperature(0.9, "extract") == 0.9
