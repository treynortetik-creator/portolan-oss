"""Tests for `portolan.training.single_teacher` (coordinator course-
correction, 2026-09-21, final design item 1: preference-order paid
teacher selection for `--mode single`). No network: `select_single_
teacher` operates on a plain `data` list a test hands it directly."""

from __future__ import annotations

from portolan.training.single_teacher import (
    MAX_COST_PER_DOCUMENT_USD,
    SINGLE_TEACHER_PREFERENCE,
    estimate_document_cost,
    select_single_teacher,
)


def _row(model_id: str, *, prompt: str, completion: str = "0.0000004") -> dict:
    return {
        "id": model_id,
        "context_length": 131072,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": ["structured_outputs"],
    }


def test_estimate_document_cost() -> None:
    # 4000 * prompt + 500 * completion
    assert estimate_document_cost(0.0000001, 0.0000002) == 4000 * 0.0000001 + 500 * 0.0000002


def test_picks_first_preference_family_when_under_budget() -> None:
    rows = [
        _row("deepseek/deepseek-v3.2", prompt="0.000000269", completion="0.0000004"),
        _row("qwen/qwen3-235b-a22b", prompt="0.0000001", completion="0.0000001"),
    ]
    chosen = select_single_teacher(rows)
    assert chosen is not None
    assert chosen.model_id == "deepseek/deepseek-v3.2"


def test_falls_through_to_next_preference_when_first_over_budget() -> None:
    rows = [
        _row("deepseek/deepseek-v3.2", prompt="0.001", completion="0.001"),  # way over budget
        _row("qwen/qwen3-235b-a22b", prompt="0.0000001", completion="0.0000001"),
    ]
    chosen = select_single_teacher(rows)
    assert chosen is not None
    assert chosen.model_id == "qwen/qwen3-235b-a22b"


def test_falls_through_to_third_preference() -> None:
    rows = [
        _row("deepseek/deepseek-v3.2", prompt="0.001", completion="0.001"),
        _row("qwen/qwen3-235b-a22b", prompt="0.001", completion="0.001"),
        _row("moonshotai/kimi-k2-instruct", prompt="0.0000001", completion="0.0000001"),
    ]
    chosen = select_single_teacher(rows)
    assert chosen is not None
    assert chosen.model_id == "moonshotai/kimi-k2-instruct"


def test_none_when_nothing_qualifies() -> None:
    rows = [_row("deepseek/deepseek-v3.2", prompt="0.001", completion="0.001")]
    assert select_single_teacher(rows) is None


def test_none_when_no_preference_family_present() -> None:
    rows = [_row("mistralai/mistral-nemo", prompt="0.000000019")]
    assert select_single_teacher(rows) is None


def test_excludes_free_ids() -> None:
    rows = [_row("deepseek/deepseek-v3.2:free", prompt="0.0000001")]
    assert select_single_teacher(rows) is None


def test_cheapest_within_family_wins() -> None:
    rows = [
        _row("deepseek/deepseek-v3.2", prompt="0.0000005", completion="0.0000005"),
        _row("deepseek/deepseek-v3.1", prompt="0.0000002", completion="0.0000002"),
    ]
    chosen = select_single_teacher(rows)
    assert chosen is not None
    assert chosen.model_id == "deepseek/deepseek-v3.1"


def test_preference_order_matches_coordinator_spec() -> None:
    assert SINGLE_TEACHER_PREFERENCE == (
        "deepseek/deepseek-v3",
        "qwen/qwen3-235b-a22b",
        "moonshotai/kimi-k2",
    )


def test_real_deepseek_row_qualifies_under_budget() -> None:
    # live-confirmed 2026-09-21: deepseek/deepseek-v3.2 at $0.269/M prompt,
    # $0.40/M completion -- estimated cost/doc ~$0.00128, under the $0.0015 ceiling.
    row = _row("deepseek/deepseek-v3.2", prompt="0.000000269", completion="0.0000004")
    chosen = select_single_teacher([row])
    assert chosen is not None
    cost = estimate_document_cost(chosen.prompt_price_per_token, chosen.completion_price_per_token)
    assert cost < MAX_COST_PER_DOCUMENT_USD
