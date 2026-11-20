"""Tests for portolan.extract.schema."""

from __future__ import annotations

import pydantic
import pytest

from portolan.extract.schema import (
    FACT_RULES,
    FACT_TYPES,
    QUOTE_REQUIRED_TYPES,
    TYPE_PRECEDENCE_RULES_V6,
    Action,
    Coverage,
    CustomerExtension,
    EntityCoercion,
    EntityKind,
    EntityMention,
    ExtractionResult,
    Fact,
    FactType,
    RejectedFact,
    SourceEnvelope,
    SourceType,
    TypePrecedenceRule,
    extraction_result_json_schema,
    merge_customer_extension,
)


def _base_fact(**overrides) -> dict:
    fact = {
        "id": "f1",
        "type": "usage",
        "claim": "Springfield runs eleven X-series pumps across two ICUs.",
        "entities": ["ashgrove-medical", "x-series-pump"],
        "locator": "[00:14:32]",
        "sentiment": 0,
        "confidence": 0.9,
    }
    fact.update(overrides)
    return fact


def test_fact_types_cover_the_full_closed_list() -> None:
    assert set(FACT_TYPES.keys()) == set(FactType)
    for info in FACT_TYPES.values():
        assert info.definition
        assert info.example
        assert info.counter_example
        assert info.counter_reason


def test_quote_required_types_match_the_rule() -> None:
    assert QUOTE_REQUIRED_TYPES == {
        FactType.COMPLAINT,
        FactType.PRAISE,
        FactType.COMPETITOR_MENTION,
        FactType.COMMITMENT,
        FactType.PRICING,
    }


def test_fact_rules_has_nine_rules() -> None:
    assert len(FACT_RULES) == 9


def test_fact_requires_non_empty_locator() -> None:
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(locator=""))


def test_fact_rejects_sentiment_out_of_range() -> None:
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(sentiment=3))
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(sentiment=-3))
    Fact.model_validate(_base_fact(sentiment=2))
    Fact.model_validate(_base_fact(sentiment=-2))


def test_fact_rejects_confidence_out_of_range() -> None:
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(confidence=1.5))
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(confidence=-0.1))


def test_fact_rejects_unknown_type() -> None:
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(type="not_a_real_type"))


@pytest.mark.parametrize("fact_type", sorted(t.value for t in QUOTE_REQUIRED_TYPES))
def test_quote_is_required_for_evidentiary_types(fact_type: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(type=fact_type, quote=None))
    # Providing a quote makes it valid.
    Fact.model_validate(_base_fact(type=fact_type, quote="we had three late deliveries last month"))


def test_quote_not_required_for_other_types() -> None:
    # `usage` is not in QUOTE_REQUIRED_TYPES.
    Fact.model_validate(_base_fact(type="usage", quote=None))


def test_quote_over_forty_words_is_rejected() -> None:
    long_quote = " ".join(["word"] * 41)
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(type="complaint", quote=long_quote))
    ok_quote = " ".join(["word"] * 40)
    Fact.model_validate(_base_fact(type="complaint", quote=ok_quote))


def test_entity_mention_kind_is_closed_list() -> None:
    EntityMention.model_validate({"mention": "Ashgrove", "kind": "account", "confidence": 0.9})
    with pytest.raises(pydantic.ValidationError):
        EntityMention.model_validate(
            {"mention": "Ashgrove", "kind": "not_a_kind", "confidence": 0.9}
        )


def test_entity_kind_closed_list_values() -> None:
    assert {k.value for k in EntityKind} == {
        "account",
        "contact",
        "product",
        "competitor",
        "internal_person",
        "other",
    }


def test_action_requires_locator() -> None:
    with pytest.raises(pydantic.ValidationError):
        Action.model_validate({"what": "Send a summary", "locator": ""})
    Action.model_validate({"what": "Send a summary", "locator": "[00:31:10]"})


def test_extraction_result_round_trips() -> None:
    result = ExtractionResult.model_validate(
        {
            "schema_version": "0.1",
            "source_id": "ashgrove-qbr-2026-08-14",
            "summary": "Ashgrove raised delivery delays and asked about Epic integration.",
            "entities": [
                {
                    "mention": "Ashgrove Medical",
                    "kind": "account",
                    "registry_id": "ashgrove-medical",
                    "confidence": 0.95,
                }
            ],
            "facts": [_base_fact()],
            "actions": [
                {
                    "what": "Send Ashgrove a delivery root-cause summary",
                    "owner_mention": "our rep, Sam",
                    "due": "2026-08-28",
                    "locator": "[00:31:10]",
                }
            ],
            "unknown_entities": ["Corvane Medical Center (possibly a competitor)"],
            "coverage": {"segments_read": 1, "segments_total": 1},
        }
    )
    assert result.source_id == "ashgrove-qbr-2026-08-14"
    assert len(result.facts) == 1
    assert result.facts[0].type == FactType.USAGE


def test_extraction_result_json_schema_is_strict() -> None:
    schema = extraction_result_json_schema()
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is False
    # Nested models should also forbid extra properties.
    defs = schema.get("$defs", {})
    assert defs, "expected nested model definitions in $defs"
    for name, definition in defs.items():
        if definition.get("type") == "object":
            assert definition.get("additionalProperties") is False, name


def test_source_envelope_requires_non_empty_source_id() -> None:
    with pytest.raises(pydantic.ValidationError):
        SourceEnvelope.model_validate(
            {
                "source_id": "   ",
                "source_type": "call",
                "source_date": "2026-08-14",
                "text": "[00:00:01] hello",
            }
        )


def test_source_envelope_source_type_is_closed_list() -> None:
    envelope = SourceEnvelope.model_validate(
        {
            "source_id": "s1",
            "source_type": "call",
            "source_date": "2026-08-14",
            "text": "[00:00:01] hello",
        }
    )
    assert envelope.source_type == SourceType.CALL
    with pytest.raises(pydantic.ValidationError):
        SourceEnvelope.model_validate(
            {
                "source_id": "s1",
                "source_type": "phone_call",
                "source_date": "2026-08-14",
                "text": "hi",
            }
        )


def test_merge_customer_extension_with_none_returns_base_types() -> None:
    effective = merge_customer_extension(None)
    assert set(effective.fact_type_checklist.keys()) == {t.value for t in FactType}
    assert effective.extra_type_names == []


def test_merge_customer_extension_adds_extra_types_and_vocabulary() -> None:
    extension = CustomerExtension.model_validate(
        {
            "customer": "mercy-medical-supply",
            "schema_version": "0.1",
            "extra_types": [
                {
                    "name": "clinical_event",
                    "definition": (
                        "A patient-safety or clinical-workflow event involving our product"
                    ),
                    "quote_required": True,
                }
            ],
            "vocabulary": {
                "products": ["X-series pump", "X-series pump v2", "the X"],
                "competitors": ["Corvane", "Brightline", "BD"],
                "jargon": {
                    "IDN": "integrated delivery network",
                    "GPO": "group purchasing organization",
                },
            },
            "aliases": {
                "ashgrove-medical": [
                    "Ashgrove Medical",
                    "Ashgrove - AMC",
                    "Ashgrove Health System Medical Center",
                ]
            },
            "skip_rules": [
                {
                    "source_type": "email",
                    "when": "subject matches /^(Out of Office|Automatic reply)/",
                }
            ],
            "severity_overrides": {
                "complaint": {"high_if": ["delivery", "patient", "safety", "outage"]}
            },
        }
    )
    effective = merge_customer_extension(extension)
    assert "clinical_event" in effective.fact_type_checklist
    assert effective.extra_type_names == ["clinical_event"]
    assert "Corvane" in effective.vocabulary.competitors
    assert effective.vocabulary.jargon["IDN"] == "integrated delivery network"
    assert "ashgrove-medical" in effective.aliases
    # Base types are still present alongside the extension's extra type.
    assert "complaint" in effective.fact_type_checklist


def test_coverage_requires_non_negative_ints() -> None:
    with pytest.raises(pydantic.ValidationError):
        Coverage.model_validate({"segments_read": -1, "segments_total": 1})


# --------------------------------------------------------------------
# F13: Fact.type open to customer-extension types
# --------------------------------------------------------------------


def _extension(**overrides) -> CustomerExtension:
    data = {
        "customer": "mercy-medical-supply",
        "extra_types": [
            {
                "name": "clinical_event",
                "definition": "A patient-safety or clinical-workflow event",
                "quote_required": True,
            }
        ],
        "severity_overrides": {
            "complaint": {"high_if": ["patient", "safety"]},
            "clinical_event": {"medium_if": ["near miss"]},
        },
    }
    data.update(overrides)
    return CustomerExtension.model_validate(data)


def test_extension_fact_type_rejected_without_effective_schema_context() -> None:
    # Without a context, only the base FactType list is honoured -- an
    # extension type is exactly as invalid as any other made-up string.
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(_base_fact(type="clinical_event"))


def test_extension_fact_type_accepted_with_effective_schema_context() -> None:
    effective = merge_customer_extension(_extension())
    fact = Fact.model_validate(
        _base_fact(type="clinical_event", quote="a near miss with the infusion pump"),
        context={"effective_schema": effective},
    )
    assert fact.type == "clinical_event"


def test_extension_quote_required_is_honoured() -> None:
    effective = merge_customer_extension(_extension())
    with pytest.raises(pydantic.ValidationError):
        Fact.model_validate(
            _base_fact(type="clinical_event", quote=None),
            context={"effective_schema": effective},
        )


def test_extension_severity_override_promotes_to_high() -> None:
    effective = merge_customer_extension(_extension())
    fact = Fact.model_validate(
        _base_fact(
            type="complaint",
            quote="the patient safety issue recurred",
            claim="A patient safety issue recurred at the Springfield ICU.",
        ),
        context={"effective_schema": effective},
    )
    assert fact.severity == "high"


def test_extension_severity_override_promotes_to_medium_when_high_absent() -> None:
    effective = merge_customer_extension(_extension())
    fact = Fact.model_validate(
        _base_fact(
            type="clinical_event",
            quote="a near miss during shift change",
            claim="A near miss occurred during shift change.",
        ),
        context={"effective_schema": effective},
    )
    assert fact.severity == "medium"


# --------------------------------------------------------------------
# Group 3: ExtractionResult.parse_lenient
# --------------------------------------------------------------------


def _base_result_dict(**overrides) -> dict:
    data = {
        "schema_version": "0.1",
        "source_id": "ashgrove-qbr-2026-08-14",
        "summary": "summary",
        "entities": [],
        "facts": [_base_fact()],
        "actions": [],
        "unknown_entities": [],
        "coverage": {"segments_read": 1, "segments_total": 1},
    }
    data.update(overrides)
    return data


def test_parse_lenient_coerces_year_month_valid_from() -> None:
    data = _base_result_dict(facts=[_base_fact(valid_from="2025-08")])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert len(result.facts) == 1
    assert result.facts[0].valid_from.isoformat() == "2025-08-01"


def test_parse_lenient_coerces_year_only_valid_to() -> None:
    data = _base_result_dict(facts=[_base_fact(valid_to="2025")])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert result.facts[0].valid_to.isoformat() == "2025-01-01"


def test_parse_lenient_drops_one_bad_fact_and_keeps_the_rest() -> None:
    # The exact real-world failure from the review: a commitment fact with
    # no quote (rule 5) alongside an otherwise-good fact.
    good = _base_fact(id="f1")
    bad = _base_fact(id="f2", type="commitment", quote=None)
    data = _base_result_dict(facts=[good, bad])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert len(result.facts) == 1
    assert result.facts[0].id == "f1"
    assert result.rejected == rejected
    assert len(rejected) == 1
    assert rejected[0].index == 1
    assert rejected[0].kind == "fact"
    assert "quote" in rejected[0].error.lower()
    assert rejected[0].raw["id"] == "f2"


def test_parse_lenient_drops_a_bad_action_too() -> None:
    good_action = {"what": "Send a summary", "locator": "[00:31:10]"}
    bad_action = {"what": "", "locator": "[00:31:10]"}
    data = _base_result_dict(actions=[good_action, bad_action])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert len(result.actions) == 1
    assert len(rejected) == 1
    assert rejected[0].kind == "action"


def test_parse_lenient_never_drops_the_whole_result_for_one_bad_fact() -> None:
    data = _base_result_dict(facts=[_base_fact(type="commitment", quote=None)])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert result.source_id == "ashgrove-qbr-2026-08-14"
    assert result.facts == []
    assert len(rejected) == 1


def test_parse_lenient_still_raises_for_a_top_level_problem() -> None:
    data = _base_result_dict()
    del data["source_id"]
    with pytest.raises(pydantic.ValidationError):
        ExtractionResult.parse_lenient(data)


def test_extraction_result_rejected_defaults_to_empty() -> None:
    result = ExtractionResult.model_validate(_base_result_dict())
    assert result.rejected == []


def test_extraction_result_json_schema_excludes_rejected() -> None:
    schema = extraction_result_json_schema()
    assert "rejected" not in schema["properties"]


def test_rejected_fact_model() -> None:
    rf = RejectedFact(index=0, error="boom", raw={"id": "f1"})
    assert rf.kind == "fact"


# --------------------------------------------------------------------
# Fix instructions item 2: real Nova Lite/Micro bench losses (12 and 13
# sources) -- parse_lenient must ignore unknown top-level/per-fact keys and
# default a missing entity confidence / an out-of-list entity kind instead
# of raising.
# --------------------------------------------------------------------


def test_parse_lenient_ignores_unknown_top_level_keys() -> None:
    # Real observed extras (2026-09-12 bench, amazon/nova-lite-v1 and
    # amazon/nova-micro-v1): the model echoed source_type/source_date at
    # the top level, or unknown_competitors/unknown_products/rejected_facts
    # -- none of these are ExtractionResult fields, and pre-fix this raised
    # pydantic.ValidationError straight out of parse_lenient.
    data = _base_result_dict(
        source_type="call",
        source_date="2026-08-05",
        rejected_facts=[],
        unknown_competitors=[],
        unknown_products=[],
    )
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert result.source_id == "ashgrove-qbr-2026-08-14"
    assert len(result.facts) == 1


def test_parse_lenient_ignores_unknown_per_fact_keys() -> None:
    # A fact carrying a plausible extra key (e.g. a stray registry_id
    # echoed onto the fact itself) must be kept, not rejected, once the
    # unknown key is stripped before validation.
    fact = _base_fact(source_type="call", extra_note="not a real field")
    data = _base_result_dict(facts=[fact])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert len(result.facts) == 1
    assert result.facts[0].id == "f1"


def test_parse_lenient_still_raises_for_a_top_level_problem_with_extras_present() -> None:
    # Unknown-key stripping must never paper over a genuinely MISSING
    # required field -- only ignore keys the schema doesn't recognize at
    # all.
    data = _base_result_dict(source_type="call")
    del data["source_id"]
    with pytest.raises(pydantic.ValidationError):
        ExtractionResult.parse_lenient(data)


def test_parse_lenient_defaults_missing_entity_confidence() -> None:
    entity = {"mention": "Tidewater T3", "kind": "product"}
    data = _base_result_dict(entities=[entity], facts=[])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert len(result.entities) == 1
    assert result.entities[0].confidence == 0.5
    assert len(result.coercions) == 1
    coercion = result.coercions[0]
    assert coercion.field == "confidence"
    assert coercion.original is None
    assert coercion.applied == 0.5
    assert coercion.mention == "Tidewater T3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("high", 0.9),
        ("HIGH", 0.9),
        ("very high", 0.9),
        ("Very High", 0.9),
        ("medium", 0.6),
        ("moderate", 0.6),
        ("low", 0.3),
        ("very low", 0.3),
        ("0.8", 0.8),
        (" 0.42 ", 0.42),
        ("1", 1.0),
        (1.5, 1.0),  # clamped, not rejected
        (-0.3, 0.0),  # clamped, not rejected
        ("not a confidence word", 0.5),  # falls through to the default
        (True, 0.5),  # bool is not a sane confidence, falls through
    ],
)
def test_parse_lenient_coerces_non_numeric_entity_confidence(raw: object, expected: float) -> None:
    """Fix wave 2026-09-18 (Kimi K2.5 bench): a `confidence` value that is
    not already a clean in-range float -- a word, an out-of-range number,
    a numeric string, or an unrecognized value -- is coerced rather than
    left to fail the FINAL strict `model_validate` and take the whole
    source down (entities, unlike facts, are never validated item by item
    in `parse_lenient`)."""
    entity = {"mention": "Tidewater T3", "kind": "product", "confidence": raw}
    data = _base_result_dict(entities=[entity], facts=[])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert result.entities[0].confidence == pytest.approx(expected)
    coercion = next(c for c in result.coercions if c.field == "confidence")
    assert coercion.original == raw
    assert coercion.applied == pytest.approx(expected)


def test_parse_lenient_defaults_unrecognized_entity_kind_to_other() -> None:
    entity = {"mention": "Support Plan", "kind": "service", "confidence": 0.9}
    data = _base_result_dict(entities=[entity], facts=[])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert len(result.entities) == 1
    assert result.entities[0].kind == EntityKind.OTHER
    kind_coercions = [c for c in result.coercions if c.field == "kind"]
    assert len(kind_coercions) == 1
    assert kind_coercions[0].original == "service"
    assert kind_coercions[0].applied == "other"


def test_parse_lenient_ignores_unknown_entity_keys() -> None:
    entity = {
        "mention": "Ashgrove",
        "kind": "account",
        "confidence": 0.9,
        "source_type": "call",  # not an EntityMention field
    }
    data = _base_result_dict(entities=[entity], facts=[])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert len(result.entities) == 1
    assert result.coercions == []


def test_parse_lenient_valid_entity_produces_no_coercions() -> None:
    entity = {"mention": "Ashgrove", "kind": "account", "confidence": 0.9}
    data = _base_result_dict(entities=[entity], facts=[])
    result, rejected = ExtractionResult.parse_lenient(data)
    assert rejected == []
    assert result.coercions == []


def test_extraction_result_coercions_defaults_to_empty() -> None:
    result = ExtractionResult.model_validate(_base_result_dict())
    assert result.coercions == []


def test_extraction_result_json_schema_excludes_coercions() -> None:
    schema = extraction_result_json_schema()
    assert "coercions" not in schema["properties"]


def test_entity_coercion_model() -> None:
    ec = EntityCoercion(index=0, mention="Ashgrove", field="confidence", applied=0.5)
    assert ec.original is None
    assert ec.applied == 0.5


# --------------------------------------------------------------------
# Fix wave 2026-09-12: real Nova Lite (fact.entities item shaped like an
# EntityMention) and gpt-oss-120b (null sentiment/severity/confidence,
# missing coverage/summary/source_id) bench losses.
# --------------------------------------------------------------------


def test_parse_lenient_coerces_fact_entity_object_to_registry_id() -> None:
    fact = _base_fact(
        entities=[
            {
                "mention": "Priya Natarajan",
                "kind": "internal_person",
                "registry_id": "internal:priya-natarajan",
            },
            "x-series-pump",
        ]
    )
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert len(result.facts) == 1
    assert result.facts[0].entities == ["internal:priya-natarajan", "x-series-pump"]
    entity_coercions = [c for c in result.coercions if c.field == "entities"]
    assert len(entity_coercions) == 1
    assert entity_coercions[0].applied == "internal:priya-natarajan"
    assert entity_coercions[0].original == {
        "mention": "Priya Natarajan",
        "kind": "internal_person",
        "registry_id": "internal:priya-natarajan",
    }


def test_parse_lenient_coerces_fact_entity_object_to_mention_when_registry_id_absent() -> None:
    # Real Nova Lite bench shapes: no registry_id key at all, or an
    # explicit `null` one -- both fall back to `mention`.
    fact = _base_fact(
        entities=[
            {"mention": "Tidewater T3", "kind": "product"},
            {"mention": "Cerner", "kind": "product", "registry_id": None},
        ]
    )
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].entities == ["Tidewater T3", "Cerner"]


def test_parse_lenient_drops_unusable_fact_entity_object() -> None:
    fact = _base_fact(entities=[{"kind": "product"}, "x-series-pump"])
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].entities == ["x-series-pump"]
    entity_coercions = [c for c in result.coercions if c.field == "entities"]
    assert len(entity_coercions) == 1
    assert entity_coercions[0].applied is None
    assert entity_coercions[0].original == {"kind": "product"}


def test_parse_lenient_drops_empty_string_fact_entity_item() -> None:
    fact = _base_fact(entities=["", "x-series-pump"])
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].entities == ["x-series-pump"]
    entity_coercions = [c for c in result.coercions if c.field == "entities"]
    assert len(entity_coercions) == 1
    assert entity_coercions[0].original == ""
    assert entity_coercions[0].applied is None


def test_parse_lenient_good_fact_entities_produce_no_coercion() -> None:
    fact = _base_fact()  # entities are already good strings
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert [c for c in result.coercions if c.field == "entities"] == []


def test_parse_lenient_defaults_null_sentiment_to_zero() -> None:
    fact = _base_fact(sentiment=None)
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].sentiment == 0
    coercion = next(c for c in result.coercions if c.field == "sentiment")
    assert coercion.original is None
    assert coercion.applied == 0
    assert result.coerced_sentiment_count == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("negative", -1),
        ("Negative", -1),
        ("NEGATIVE", -1),
        ("neg", -1),
        ("neutral", 0),
        ("positive", 1),
        ("pos", 1),
        ("very negative", -2),
        ("very positive", 2),
        ("+1", 1),
        ("-1", -1),
        ("-2", -2),
        ("2", 2),
        (" 0 ", 0),
        ("1.0", 1),
        ("unsure", 0),  # unrecognized -- falls through to the same
        # default a `null` sentiment maps to, never a fact rejection.
        ("", 0),
    ],
)
def test_parse_lenient_coerces_non_numeric_fact_sentiment(raw: str, expected: int) -> None:
    """DeepSeek V4.1 Flash bench, 2026-09-22
    (`01-spec/bench-deepseek-v41-flash-2026-09-22.md`): the model sends
    `sentiment` as a word or a signed numeric string almost every time,
    and before this fix only an explicit `null` was coerced -- every
    string value failed `Fact.sentiment`'s int validation and took the
    whole fact down with it. This is the fact-level counterpart of
    `test_parse_lenient_coerces_non_numeric_fact_confidence` above."""
    fact = _base_fact(sentiment=raw)
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].sentiment == expected
    coercion = next(c for c in result.coercions if c.field == "sentiment")
    assert coercion.original == raw
    assert coercion.applied == expected
    assert result.coerced_sentiment_count == 1


def test_parse_lenient_good_fact_sentiment_produces_no_coercion() -> None:
    fact = _base_fact(sentiment=-2)  # already a good in-range int
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].sentiment == -2
    assert [c for c in result.coercions if c.field == "sentiment"] == []
    assert result.coerced_sentiment_count == 0


def test_parse_lenient_still_rejects_a_fact_with_a_bad_type() -> None:
    """String-sentiment leniency must not widen into leniency for a
    genuinely malformed fact -- an unknown `type` is still rejected, same
    as before this fix."""
    fact = _base_fact(type="not_a_real_type", sentiment="negative")
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert result.facts == []
    assert len(rejected) == 1
    assert rejected[0].kind == "fact"


def test_parse_lenient_drops_null_severity_to_none() -> None:
    fact = _base_fact(severity=None)
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].severity is None
    severity_coercions = [c for c in result.coercions if c.field == "severity"]
    assert len(severity_coercions) == 1
    assert severity_coercions[0].applied is None


def test_parse_lenient_omitted_severity_produces_no_coercion() -> None:
    fact = _base_fact()  # no severity key at all
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].severity is None
    assert [c for c in result.coercions if c.field == "severity"] == []


def test_parse_lenient_defaults_null_fact_confidence_to_half() -> None:
    fact = _base_fact(confidence=None)
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].confidence == 0.5
    coercion = next(c for c in result.coercions if c.field == "confidence")
    assert coercion.original is None
    assert coercion.applied == 0.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("high", 0.9),
        ("very high", 0.9),
        ("medium", 0.6),
        ("moderate", 0.6),
        ("low", 0.3),
        ("very low", 0.3),
        ("0.75", 0.75),
        (2.0, 1.0),  # clamped, not rejected
        ("unsure", 0.5),  # falls through to the default
    ],
)
def test_parse_lenient_coerces_non_numeric_fact_confidence(raw: object, expected: float) -> None:
    """Fact-level counterpart of the entity-level fix wave 2026-09-18
    test above -- a bad fact `confidence` was already caught by this
    function's per-fact try/except before this fix (it only ever
    rejected that one fact, never the whole source), but it was rejected
    outright rather than coerced; this recovers the fact instead."""
    fact = _base_fact(confidence=raw)
    data = _base_result_dict(facts=[fact])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.facts[0].confidence == pytest.approx(expected)
    coercion = next(c for c in result.coercions if c.field == "confidence")
    assert coercion.original == raw
    assert coercion.applied == pytest.approx(expected)


def test_parse_lenient_defaults_missing_coverage() -> None:
    data = _base_result_dict()
    del data["coverage"]

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.coverage.segments_read == 1
    assert result.coverage.segments_total == 1
    coercion = next(c for c in result.coercions if c.field == "coverage")
    assert coercion.index == -1
    assert coercion.applied == {"segments_read": 1, "segments_total": 1}


def test_parse_lenient_defaults_missing_summary() -> None:
    data = _base_result_dict()
    del data["summary"]

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.summary == ""
    assert any(c.field == "summary" for c in result.coercions)


def test_parse_lenient_fills_missing_source_id_from_default() -> None:
    data = _base_result_dict()
    del data["source_id"]

    result, rejected = ExtractionResult.parse_lenient(data, default_source_id="s-from-envelope")

    assert rejected == []
    assert result.source_id == "s-from-envelope"
    coercion = next(c for c in result.coercions if c.field == "source_id")
    assert coercion.applied == "s-from-envelope"


def test_parse_lenient_default_source_id_does_not_override_existing() -> None:
    data = _base_result_dict()

    result, rejected = ExtractionResult.parse_lenient(data, default_source_id="should-not-be-used")

    assert rejected == []
    assert result.source_id == "ashgrove-qbr-2026-08-14"
    assert [c for c in result.coercions if c.field == "source_id"] == []


def test_parse_lenient_still_raises_missing_source_id_without_default() -> None:
    data = _base_result_dict()
    del data["source_id"]
    with pytest.raises(pydantic.ValidationError):
        ExtractionResult.parse_lenient(data)


def test_parse_lenient_survives_the_minimal_incomplete_result_shape() -> None:
    # The real gpt-oss-120b bench failure: a model that returned almost
    # nothing (`{"entities": [], "facts": [], "actions": []}`), missing
    # source_id/summary/coverage entirely -- previously this raised
    # pydantic.ValidationError straight out of parse_lenient.
    data = {"entities": [], "facts": [], "actions": []}

    result, rejected = ExtractionResult.parse_lenient(data, default_source_id="s-min")

    assert rejected == []
    assert result.source_id == "s-min"
    assert result.summary == ""
    assert result.coverage.segments_read == 1
    assert result.coverage.segments_total == 1


# --------------------------------------------------------------------------
# Fix wave 2026-09-18 (infra-2b): non-fact bookkeeping leniency, real
# failing payloads copied from the saved bench reports.
# --------------------------------------------------------------------------


def test_parse_lenient_coerces_malformed_coverage_dict() -> None:
    # openai/gpt-5.6-luna, var/bench/2026-09-18-models/openai_gpt-5.6-luna/
    # run1/bench-2026-09-18-openrouter.json, source S-call-003 -- the
    # ValidationError message that used to abort the whole result:
    # "coverage.segments_read Field required ... coverage.source_segment_
    # reviewed Extra inputs are not permitted".
    data = _base_result_dict(
        coverage={
            "source_segment_reviewed": True,
            "high_value_signals_captured": True,
            "omitted_low_durability_content": True,
        }
    )

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.coverage.segments_read == 1
    assert result.coverage.segments_total == 1
    coercion = next(c for c in result.coercions if c.field == "coverage")
    assert coercion.original == {
        "source_segment_reviewed": True,
        "high_value_signals_captured": True,
        "omitted_low_durability_content": True,
    }
    assert coercion.applied == {"segments_read": 1, "segments_total": 1}


def test_parse_lenient_coerces_a_second_malformed_coverage_shape() -> None:
    # Same run, source S-call-004 -- a THIRD run produced yet another
    # distinct reshaping, confirming this is "whatever shape it feels
    # like" rather than one fixed alternate key set a targeted mapping
    # could special-case.
    data = _base_result_dict(
        coverage={
            "source_lines_considered": 137,
            "facts_extracted": 30,
            "high_value_signals": 30,
        }
    )

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.coverage.segments_read == 1
    assert result.coverage.segments_total == 1


def test_parse_lenient_coerces_coverage_that_is_a_bare_string() -> None:
    # Same run, source S-email-030/S-email-063 -- 17 of the 207 failures
    # had `coverage` as a bare string ("complete") rather than a dict at
    # all: "coverage Input should be a valid dictionary or instance of
    # Coverage [type=model_type, input_value='complete', ...]".
    data = _base_result_dict(coverage="complete")

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.coverage.segments_read == 1
    assert result.coverage.segments_total == 1
    coercion = next(c for c in result.coercions if c.field == "coverage")
    assert coercion.original == "complete"


def test_parse_lenient_coverage_that_already_validates_is_unchanged() -> None:
    data = _base_result_dict(coverage={"segments_read": 2, "segments_total": 3})

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.coverage.segments_read == 2
    assert result.coverage.segments_total == 3
    assert [c for c in result.coercions if c.field == "coverage"] == []


def test_parse_lenient_synthesizes_entity_from_a_bare_mention_string() -> None:
    # nvidia/nemotron-3-nano-30b-a3b, var/bench/2026-09-18-models/
    # nvidia_nemotron-3-nano-30b-a3b/run1/, source S-crm-opp-A1010 -- the
    # bare ValueError that used to abort the whole result: "dictionary
    # update sequence element #0 has length 1; 2 is required" (raised
    # from the old `dict(raw_entity)` call trying to build a dict out of
    # a plain multi-character string's own individual characters).
    data = _base_result_dict(entities=["Ashgrove Health"])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert len(result.entities) == 1
    entity = result.entities[0]
    assert entity.mention == "Ashgrove Health"
    assert entity.kind == EntityKind.OTHER
    assert entity.confidence == pytest.approx(0.5)
    assert any(c.field == "entity" and c.original == "Ashgrove Health" for c in result.coercions)


def test_parse_lenient_synthesizes_multiple_bare_mention_entities() -> None:
    # nvidia/nemotron-3-nano-30b-a3b, source S-email-n015 -- same defect,
    # confirmed with more than one bare string in the list.
    data = _base_result_dict(entities=["Ashgrove Health", "Northlake Health System"])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert [e.mention for e in result.entities] == ["Ashgrove Health", "Northlake Health System"]
    assert all(e.kind == EntityKind.OTHER for e in result.entities)


def test_parse_lenient_rejects_a_fact_item_that_is_a_bare_string() -> None:
    # Same class of defect as the entity one above, on `facts` instead --
    # not independently observed in a real run, but the old
    # `dict(raw_fact)` call would crash the same way. A fact genuinely
    # cannot be parsed from a bare string, so it is rejected, not
    # defaulted -- the source itself must still succeed.
    data = _base_result_dict(facts=[_base_fact(), "not a fact object"])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert len(result.facts) == 1
    assert len(rejected) == 1
    assert rejected[0].kind == "fact"
    assert rejected[0].raw == {"value": "not a fact object"}


def test_parse_lenient_rejects_an_action_item_that_is_a_bare_string() -> None:
    data = _base_result_dict(actions=[{"what": "send a summary", "locator": "[00:01:00]"}, 42])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert len(result.actions) == 1
    assert len(rejected) == 1
    assert rejected[0].kind == "action"
    assert rejected[0].raw == {"value": 42}


def test_parse_lenient_coerces_unknown_entities_that_is_not_a_list() -> None:
    data = _base_result_dict(unknown_entities="Dr. Alvarez")

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.unknown_entities == []
    coercion = next(c for c in result.coercions if c.field == "unknown_entities")
    assert coercion.original == "Dr. Alvarez"


def test_parse_lenient_drops_non_string_items_from_unknown_entities() -> None:
    data = _base_result_dict(unknown_entities=[123, "Dr. Alvarez", None, ""])

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.unknown_entities == ["Dr. Alvarez"]


def test_parse_lenient_coerces_non_string_summary() -> None:
    data = _base_result_dict(summary={"note": "structured, not prose"})

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.summary == "{'note': 'structured, not prose'}"
    assert any(c.field == "summary" for c in result.coercions)


def test_parse_lenient_coerces_numeric_source_id() -> None:
    data = _base_result_dict(source_id=1010)

    result, rejected = ExtractionResult.parse_lenient(data)

    assert rejected == []
    assert result.source_id == "1010"
    coercion = next(c for c in result.coercions if c.field == "source_id")
    assert coercion.original == 1010


# --------------------------------------------------------------------------
# v6-type-precision (facttype-4a, 2026-09-18,
# 01-spec/facttype-4a-2026-09-18.md): TYPE_PRECEDENCE_RULES_V6
# --------------------------------------------------------------------------

# The four confused pairs the confusion-matrix diagnosis over
# var/bench/2026-09-18-final/ found, pooling Sonnet/Haiku/Kimi K2/Kimi
# K2.5/Nova Lite's matched-but-wrong-typed facts on v4-two-step:
# decision->other (40), commitment->decision (15), request<->question
# (10), other->risk (6). praise/other (2) was too small a sample to
# chase and has no rule here -- see the doc.
_EXPECTED_V6_PAIRS = {
    (FactType.DECISION, FactType.OTHER),
    (FactType.COMMITMENT, FactType.DECISION),
    (FactType.REQUEST, FactType.QUESTION),
    (FactType.RISK, FactType.OTHER),
}

# The literal planted claims (or their most distinctive substrings) that
# drove each rule -- corpus/truth-table.yaml F11/F11b, F33, F39/F67, F07.
# None of these may appear verbatim in a rule's examples: the rule must
# teach the distinction in the corpus's own domain, not repeat the answer
# to a specific planted fact (the task's own "not copied from planted
# facts verbatim" instruction).
_PLANTED_CLAIM_FRAGMENTS = (
    "HalvardCare contract renews on 2027-03-31",
    "HalvardCare renewal was moved to 2027-06-30",
    "agreed to pilot two Tidewater T3 units",
    "Tidewater Link could integrate with its Epic instance",
    "Tidewater Link integration timeline applied to its facility",
    "Dayton distribution branch",
)


def test_type_precedence_rules_v6_covers_exactly_the_four_confused_pairs() -> None:
    pairs = {(r.winning_type, r.losing_type) for r in TYPE_PRECEDENCE_RULES_V6}
    assert pairs == _EXPECTED_V6_PAIRS
    assert len(TYPE_PRECEDENCE_RULES_V6) == 4


def test_type_precedence_rules_v6_are_well_formed() -> None:
    for rule in TYPE_PRECEDENCE_RULES_V6:
        assert isinstance(rule, TypePrecedenceRule)
        assert rule.winning_type != rule.losing_type
        assert rule.rule.strip()
        assert rule.positive_example.strip()
        assert rule.negative_example.strip()
        # The positive example names the winning type, the negative
        # example names the losing type -- each worked example is
        # self-labeled so the model can check its own answer against it.
        assert f"`{rule.winning_type.value}`" in rule.positive_example
        assert f"`{rule.losing_type.value}`" in rule.negative_example


def test_type_precedence_rules_v6_examples_do_not_copy_a_planted_claim_verbatim() -> None:
    for rule in TYPE_PRECEDENCE_RULES_V6:
        for fragment in _PLANTED_CLAIM_FRAGMENTS:
            assert fragment not in rule.positive_example
            assert fragment not in rule.negative_example
