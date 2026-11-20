"""Tests for portolan.llm.json_util: pure JSON extraction and validation,
no network, no SDK."""

from __future__ import annotations

from portolan.llm.json_util import extract_and_validate, extract_json_object, validate_json_object

SCHEMA = {
    "type": "object",
    "properties": {
        "greeting": {"type": "string"},
        "count": {"type": "integer"},
    },
    "required": ["greeting", "count"],
    "additionalProperties": False,
}


def test_extract_json_object_unfenced_exact() -> None:
    text = '{"greeting": "hi", "count": 3}'
    assert extract_json_object(text) == {"greeting": "hi", "count": 3}


def test_extract_json_object_unfenced_with_surrounding_prose() -> None:
    text = 'Sure, here you go: {"greeting": "hi", "count": 3} -- let me know if that works.'
    assert extract_json_object(text) == {"greeting": "hi", "count": 3}


def test_extract_json_object_fenced_with_json_language_tag() -> None:
    text = '```json\n{"greeting": "hi", "count": 3}\n```'
    assert extract_json_object(text) == {"greeting": "hi", "count": 3}


def test_extract_json_object_fenced_without_language_tag() -> None:
    text = '```\n{"greeting": "hi", "count": 3}\n```'
    assert extract_json_object(text) == {"greeting": "hi", "count": 3}


def test_extract_json_object_fenced_with_prose_before_and_after() -> None:
    text = 'Here is the JSON:\n```json\n{"greeting": "hi", "count": 3}\n```\nHope that helps!'
    assert extract_json_object(text) == {"greeting": "hi", "count": 3}


def test_extract_json_object_handles_nested_objects_and_braces_in_strings() -> None:
    text = '{"greeting": "hi {not a nested object}", "count": 3, "nested": {"a": 1}}'
    assert extract_json_object(text) == {
        "greeting": "hi {not a nested object}",
        "count": 3,
        "nested": {"a": 1},
    }


def test_extract_json_object_returns_none_for_no_json() -> None:
    assert extract_json_object("I can't help with that, sorry.") is None


def test_extract_json_object_returns_none_for_bare_list_or_scalar() -> None:
    assert extract_json_object("[1, 2, 3]") is None
    assert extract_json_object('"just a string"') is None


def test_extract_json_object_skips_unbalanced_brace_and_finds_the_real_one() -> None:
    text = 'That { looks broken. Anyway: {"greeting": "hi", "count": 3}'
    assert extract_json_object(text) == {"greeting": "hi", "count": 3}


def test_validate_json_object_success_returns_none() -> None:
    assert validate_json_object({"greeting": "hi", "count": 3}, SCHEMA) is None


def test_validate_json_object_failure_returns_message() -> None:
    error = validate_json_object({"greeting": "hi"}, SCHEMA)
    assert error is not None
    assert "count" in error


def test_extract_and_validate_success() -> None:
    data, error = extract_and_validate('```json\n{"greeting": "hi", "count": 3}\n```', SCHEMA)
    assert data == {"greeting": "hi", "count": 3}
    assert error is None


def test_extract_and_validate_no_json_found() -> None:
    data, error = extract_and_validate("no json here at all", SCHEMA)
    assert data is None
    assert error == "no JSON object found in the response"


def test_extract_and_validate_schema_invalid() -> None:
    data, error = extract_and_validate('{"greeting": "hi"}', SCHEMA)
    assert data is None
    assert error is not None
    assert "count" in error
