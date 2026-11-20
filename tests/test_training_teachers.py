"""Tests for `portolan.training.teachers` -- the open-weight family
filter and two-teacher pair rotation (silver-labels build instructions
item 2 and item 6: "the open-weight family filter"). No network:
`filter_open_weight`/`TeacherPairPicker` operate on plain id lists a test
hands them directly, never `discover_open_weight_free_models` itself."""

from __future__ import annotations

import pytest

from portolan.training.teachers import (
    OPEN_WEIGHT_FAMILIES,
    TeacherPairPicker,
    family_of,
    filter_open_weight,
    is_open_weight,
)

# A representative slice of what a live `GET /api/v1/models` response
# could contain: open-weight ids from every named family, plus closed
# and unknown ones that must be excluded.
_CANDIDATE_IDS = [
    "qwen/qwen3.8-27b:free",
    "meta-llama/llama-4.2-70b:free",
    "deepseek/deepseek-v4-flash-0731:free",
    "mistralai/mistral-medium-4:free",
    "moonshotai/kimi-k2.5:free",
    "z-ai/glm-5.2:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-120b:free",
    # closed / unknown -- must be excluded
    "anthropic/claude-haiku-4-5:free",
    "openai/gpt-5.6-luna:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemini-3.5-flash:free",
]


def test_is_open_weight_accepts_every_named_family() -> None:
    for family in OPEN_WEIGHT_FAMILIES:
        # every family substring must actually match at least one of the
        # representative ids above, otherwise this test's own fixture is
        # out of sync with the filter it exercises
        assert any(family in cid for cid in _CANDIDATE_IDS), family


def test_is_open_weight_excludes_closed_and_unknown() -> None:
    assert is_open_weight("anthropic/claude-haiku-4-5:free") is False
    assert is_open_weight("openai/gpt-5.6-luna:free") is False
    assert is_open_weight("google/gemini-3.5-flash:free") is False


def test_is_open_weight_excludes_nemotron() -> None:
    # nvidia/nemotron is in corpus.generator.prose's PREFERRED_FREE_
    # TEACHER_FAMILIES for a different purpose, but is not one of the
    # eight families this build instruction names -- must stay excluded.
    assert is_open_weight("nvidia/nemotron-3-super-120b-a12b:free") is False


def test_filter_open_weight_keeps_only_allowed_families_in_order() -> None:
    result = filter_open_weight(_CANDIDATE_IDS)
    assert result == [
        "qwen/qwen3.8-27b:free",
        "meta-llama/llama-4.2-70b:free",
        "deepseek/deepseek-v4-flash-0731:free",
        "mistralai/mistral-medium-4:free",
        "moonshotai/kimi-k2.5:free",
        "z-ai/glm-5.2:free",
        "google/gemma-4-31b-it:free",
        "openai/gpt-oss-120b:free",
    ]


def test_filter_open_weight_gpt_oss_not_confused_with_openai_closed() -> None:
    # "openai/gpt-oss-120b" must pass (open-weight family) while
    # "openai/gpt-5.6-luna" (closed) must not -- same vendor prefix,
    # different family substring.
    assert is_open_weight("openai/gpt-oss-120b:free") is True
    assert is_open_weight("openai/gpt-5.6-luna:free") is False


def test_family_of_returns_none_for_unknown() -> None:
    assert family_of("anthropic/claude-haiku-4-5:free") is None


# --------------------------------------------------------------------------
# TeacherPairPicker
# --------------------------------------------------------------------------


def test_teacher_pair_picker_requires_two_families() -> None:
    with pytest.raises(ValueError):
        TeacherPairPicker(["qwen/qwen3.8-27b:free"])
    with pytest.raises(ValueError):
        # two ids, same family -- still not enough
        TeacherPairPicker(["qwen/qwen3.8-27b:free", "qwen/qwen3.8-14b:free"])


def test_teacher_pair_picker_always_different_families() -> None:
    models = filter_open_weight(_CANDIDATE_IDS)
    picker = TeacherPairPicker(models)
    for _ in range(20):
        first, second = picker.next_pair()
        assert first in models
        assert second in models
        assert first != second
        assert family_of(first) != family_of(second)


def test_teacher_pair_picker_rotates_first_slot_through_every_model() -> None:
    models = filter_open_weight(_CANDIDATE_IDS)
    picker = TeacherPairPicker(models)
    firsts = [picker.next_pair()[0] for _ in range(len(models))]
    assert firsts == models


def test_teacher_pair_picker_deterministic_given_same_models() -> None:
    models = filter_open_weight(_CANDIDATE_IDS)
    picker_a = TeacherPairPicker(list(models))
    picker_b = TeacherPairPicker(list(models))
    pairs_a = [picker_a.next_pair() for _ in range(10)]
    pairs_b = [picker_b.next_pair() for _ in range(10)]
    assert pairs_a == pairs_b
