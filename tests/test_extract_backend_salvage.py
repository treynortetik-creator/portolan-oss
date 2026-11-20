"""`LLMClientBackend`: salvage before retrying, and honour the customer's
own fact types."""

from __future__ import annotations

from datetime import date

from portolan.extract.backends import LLMClientBackend
from portolan.extract.schema import (
    CustomerExtension,
    ExtraFactType,
    SourceEnvelope,
    SourceType,
    merge_customer_extension,
)
from portolan.llm.fake import FakeLLMClient
from portolan.metering import InMemorySink


def _envelope() -> SourceEnvelope:
    return SourceEnvelope(
        source_id="S-1", source_type=SourceType.CALL, source_date=date(2026, 5, 1), text="[p1] x"
    )


def _fact(**overrides) -> dict:
    fact = {
        "id": "f1",
        "type": "request",
        "claim": "The account asked for a quiet-hours setting.",
        "entities": ["A-1"],
        "locator": "p1",
        "sentiment": 0,
        "confidence": 0.9,
    }
    fact.update(overrides)
    return fact


def _reply(*facts: dict) -> dict:
    return {
        "source_id": "S-1",
        "summary": "s",
        "facts": list(facts),
        "coverage": {"segments_read": 1, "segments_total": 1},
    }


def _extract(responses, **backend_kwargs):
    client = FakeLLMClient(responses)
    backend = LLMClientBackend(client, **backend_kwargs)
    result = backend.extract(_envelope(), "system", InMemorySink(), "t")
    return result, client._call_count


def test_a_reply_that_only_needs_coercing_is_not_paid_for_twice():
    result, calls = _extract([_reply(_fact(sentiment="negative", confidence="high"))])
    assert calls == 1
    assert result.facts[0].sentiment == -1 and not result.rejected
    assert {c.field for c in result.coercions} == {"sentiment", "confidence"}


def test_a_reply_that_loses_a_fact_is_retried_once():
    bad = _fact(id="f2", type="complaint", claim="It broke.")  # complaint needs a quote
    good = _fact(id="f2", type="complaint", claim="It broke.", quote="it broke")
    result, calls = _extract([_reply(_fact(), bad), _reply(_fact(), good)])
    assert calls == 2 and len(result.facts) == 2


def test_the_attempt_that_kept_more_facts_wins():
    bad = _fact(id="f2", type="complaint", claim="It broke.")
    result, calls = _extract([_reply(_fact(), _fact(id="f3"), bad), _reply(bad)])
    assert calls == 2
    assert [f.id for f in result.facts] == ["f1", "f3"]
    assert len(result.rejected) == 1


def test_a_quote_over_forty_words_is_cut_not_rejected():
    long_quote = " ".join(f"w{i}" for i in range(55))
    result, calls = _extract([_reply(_fact(type="complaint", claim="It broke.", quote=long_quote))])
    assert calls == 1 and not result.rejected
    assert result.facts[0].quote == " ".join(f"w{i}" for i in range(40))


def test_a_customer_extension_type_is_kept_when_the_backend_knows_the_extension():
    extension = CustomerExtension(
        customer="acme",
        extra_types=[ExtraFactType(name="regulatory", definition="A regulator is involved")],
    )
    reply = _reply(_fact(type="regulatory", claim="The state inspector visits in June."))

    result, calls = _extract([reply], effective_schema=merge_customer_extension(extension))
    assert calls == 1 and [f.type for f in result.facts] == ["regulatory"]

    # Without it the type is unknown: a retry, then the fact is rejected.
    result, calls = _extract([reply, reply])
    assert calls == 2 and not result.facts and len(result.rejected) == 1
