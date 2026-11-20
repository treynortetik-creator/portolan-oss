"""Tests for `portolan.training.paid_teacher` (coordinator course-
correction, 2026-09-21, item 1: cheapest qualifying open-weight paid
teacher). No network: `candidates_from_models_response`/
`select_cheapest_paid_teacher` operate on a plain `data` list a test
hands them directly."""

from __future__ import annotations

from portolan.training.paid_teacher import (
    MIN_CONTEXT_LENGTH,
    PAID_TEACHER_FAMILIES,
    candidates_from_models_response,
    select_cheapest_paid_teacher,
    select_two_cheapest_different_families,
)


def _row(
    model_id: str,
    *,
    context_length: int = 131072,
    prompt: str = "0.0000001",
    completion: str = "0.0000002",
    supported_parameters: list[str] | None = None,
) -> dict:
    return {
        "id": model_id,
        "context_length": context_length,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": (
            supported_parameters if supported_parameters is not None else ["structured_outputs"]
        ),
    }


def test_excludes_free_ids() -> None:
    rows = [_row("qwen/qwen3.7-flash:free")]
    assert candidates_from_models_response(rows) == []


def test_excludes_ids_outside_named_families() -> None:
    rows = [
        _row("anthropic/claude-haiku-4-5"),
        _row("openai/gpt-oss-20b"),
        _row("google/gemma-3-4b-it"),
    ]
    assert candidates_from_models_response(rows) == []


def test_includes_every_named_family() -> None:
    ids = [
        "qwen/qwen3.7-flash",
        "deepseek/deepseek-v4-flash-0731",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-nemo",
        "z-ai/glm-4.7-flash",
        "moonshotai/kimi-k2.5",
    ]
    rows = [_row(mid) for mid in ids]
    candidates = candidates_from_models_response(rows)
    assert {c.model_id for c in candidates} == set(ids)
    assert {c.family for c in candidates} == set(PAID_TEACHER_FAMILIES)


def test_excludes_context_length_at_or_under_32k() -> None:
    rows = [_row("qwen/small", context_length=MIN_CONTEXT_LENGTH)]
    assert candidates_from_models_response(rows) == []
    rows_ok = [_row("qwen/small", context_length=MIN_CONTEXT_LENGTH + 1)]
    assert len(candidates_from_models_response(rows_ok)) == 1


def test_excludes_no_structured_output_support() -> None:
    rows = [_row("qwen/plain", supported_parameters=["temperature", "top_p"])]
    assert candidates_from_models_response(rows) == []


def test_response_format_alone_also_qualifies() -> None:
    rows = [_row("qwen/rf-only", supported_parameters=["response_format"])]
    assert len(candidates_from_models_response(rows)) == 1


def test_excludes_unparseable_or_non_positive_pricing() -> None:
    rows = [
        _row("qwen/nan-price", prompt="not-a-number"),
        _row("qwen/zero-price", prompt="0"),
        _row("qwen/negative-price", prompt="-0.001"),
    ]
    assert candidates_from_models_response(rows) == []


def test_select_cheapest_by_prompt_price() -> None:
    rows = [
        _row("qwen/expensive", prompt="0.0000005"),
        _row("mistralai/mistral-nemo", prompt="0.000000019"),
        _row("deepseek/mid", prompt="0.00000004"),
    ]
    candidates = candidates_from_models_response(rows)
    cheapest = select_cheapest_paid_teacher(candidates)
    assert cheapest is not None
    assert cheapest.model_id == "mistralai/mistral-nemo"


def test_select_cheapest_ties_broken_by_completion_price_then_id() -> None:
    rows = [
        _row("qwen/b", prompt="0.00000005", completion="0.0000002"),
        _row("qwen/a", prompt="0.00000005", completion="0.0000001"),
    ]
    cheapest = select_cheapest_paid_teacher(candidates_from_models_response(rows))
    assert cheapest is not None
    assert cheapest.model_id == "qwen/a"


def test_select_cheapest_empty_list_is_none() -> None:
    assert select_cheapest_paid_teacher([]) is None


def test_real_mistral_nemo_row_qualifies() -> None:
    # The row this module's own docstring cites, live-confirmed 2026-09-21.
    row = {
        "id": "mistralai/mistral-nemo",
        "context_length": 131072,
        "pricing": {"prompt": "0.000000019", "completion": "0.00000003"},
        "supported_parameters": ["response_format", "structured_outputs", "tools"],
    }
    candidates = candidates_from_models_response([row])
    assert len(candidates) == 1
    assert candidates[0].family == "mistralai/"


def test_select_two_cheapest_different_families() -> None:
    rows = [
        _row("mistralai/mistral-nemo", prompt="0.000000019"),
        _row("mistralai/ministral-8b", prompt="0.00000005"),  # cheaper 2nd mistral, must be skipped
        _row("qwen/qwen3.7-flash", prompt="0.00000003"),
        _row("deepseek/mid", prompt="0.00000004"),
    ]
    pair = select_two_cheapest_different_families(candidates_from_models_response(rows))
    assert pair is not None
    first, second = pair
    assert first.model_id == "mistralai/mistral-nemo"
    assert second.model_id == "qwen/qwen3.7-flash"
    assert first.family != second.family


def test_select_two_cheapest_different_families_none_when_one_family_only() -> None:
    rows = [_row("qwen/a", prompt="0.00000003"), _row("qwen/b", prompt="0.00000005")]
    assert select_two_cheapest_different_families(candidates_from_models_response(rows)) is None


def test_select_two_cheapest_different_families_none_when_empty() -> None:
    assert select_two_cheapest_different_families([]) is None
