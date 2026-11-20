"""Tests for `portolan.training.repair`: mapping logged drops to records,
re-gating, one stubbed repair call per failed fact, and the record audit.
No network: every teacher call is a stub."""

from __future__ import annotations

from typing import Any

import pytest

from portolan.training.records import DroppedFact, TrainingFact, TrainingRecord
from portolan.training.repair import (
    FailedFact,
    GateVerdict,
    assemble_record,
    build_repair_messages,
    gate_fact,
    judge_reply,
    map_drops_to_records,
    plan_record,
    recover_unwritten_records,
    repair_fact,
)

KIMI = "moonshotai/kimi-k2"
TEXT = (
    "[L1] Consumer: Comerica sent my debit card loaded with funds to somewhere else. "
    "[L2] They put me on a 5 year payment plan and charged $5k per month."
)


def _record(
    record_id: str = "S-1",
    *,
    source_id: str = "S-1",
    text: str = TEXT,
    facts: list[TrainingFact] | None = None,
    agreement: float = 1.0,
    created: str = "2026-09-22T12:00:00-07:00",
    teacher: str = KIMI,
) -> TrainingRecord:
    return TrainingRecord(
        record_id=record_id,
        corpus="cfpb",
        source_id=source_id,
        kind="ticket",
        license="public domain",
        text=text,
        facts=facts or [],
        teachers=[teacher],
        agreement=agreement,
        created=created,
        mode="single",
    )


def _kept(claim: str, quote: str, text: str = TEXT) -> TrainingFact:
    start = text.index(quote)
    return TrainingFact(
        fact_type="complaint",
        claim=claim,
        quote=quote,
        quote_span=(start, start + len(quote)),
        locator="L1",
    )


def _drop(
    claim: str,
    quote: str,
    *,
    reason: str = "overlap_gate_name",
    source_id: str = "S-1",
    created: str = "2026-09-22T11:59:59-07:00",
    teacher: str = KIMI,
) -> DroppedFact:
    return DroppedFact(
        source_id=source_id,
        reason=reason,
        fact={
            "fact_type": "complaint",
            "claim": claim,
            "quote": quote,
            "locator": "L1",
            "entities": [],
        },
        teacher=teacher,
        created=created,
    )


# --------------------------------------------------------------------------
# gate_fact
# --------------------------------------------------------------------------


def test_gate_fact_passes_and_resolves_the_exact_span() -> None:
    verdict = gate_fact("Comerica sent the card elsewhere", "Comerica sent my debit card", TEXT)
    assert verdict.reason is None
    assert verdict.span is not None
    assert TEXT[verdict.span[0] : verdict.span[1]] == verdict.quote


def test_gate_fact_reports_the_unsupported_name() -> None:
    verdict = gate_fact("The card went to Wells Fargo", "Comerica sent my debit card", TEXT)
    assert verdict.reason == "overlap_gate_name"
    assert verdict.detail == ("Wells", "Fargo")


def test_gate_fact_reports_the_unsupported_numeral() -> None:
    verdict = gate_fact("The plan costs $6,000 per month", "charged $5k per month", TEXT)
    assert verdict.reason == "overlap_gate_numeral"
    assert verdict.detail == ("6,000",)


# --------------------------------------------------------------------------
# map_drops_to_records
# --------------------------------------------------------------------------


def test_drop_maps_to_the_next_record_of_its_source_by_time() -> None:
    seg1 = _record(
        "S-2#seg1of2",
        source_id="S-2",
        text="alpha beta gamma quote one here",
        created="2026-09-22T12:00:00-07:00",
    )
    seg2 = _record(
        "S-2#seg2of2",
        source_id="S-2",
        text="delta epsilon quote two here",
        created="2026-09-22T12:01:00-07:00",
    )
    d1 = _drop("c1", "quote one here", source_id="S-2", created="2026-09-22T11:59:30-07:00")
    d2 = _drop("c2", "quote two here", source_id="S-2", created="2026-09-22T12:00:30-07:00")
    mapping = map_drops_to_records([seg2, seg1], [d1, d2])
    assert mapping.by_record == {"S-2#seg1of2": [d1], "S-2#seg2of2": [d2]}
    assert mapping.orphans == []


def test_drop_from_an_unwritten_envelope_is_an_orphan() -> None:
    # the segment whose drops these are was capped as a negative and never
    # written; the next record written does not hold the drop's quote
    seg2 = _record(
        "S-3#seg2of2",
        source_id="S-3",
        text="delta epsilon",
        created="2026-09-22T12:01:00-07:00",
    )
    drop = _drop("c", "a quote from segment one", source_id="S-3")
    mapping = map_drops_to_records([seg2], [drop])
    assert mapping.orphans == [drop]


def test_drop_of_another_teacher_does_not_map() -> None:
    record = _record(teacher="deepseek/deepseek-v3.2")
    mapping = map_drops_to_records([record], [_drop("c", "Comerica sent my debit card")])
    assert mapping.by_record == {}


def test_backfill_drop_maps_by_quote_not_time() -> None:
    record = _record(created="2026-09-22T05:00:00-07:00")
    drop = _drop(
        "c",
        "Comerica sent my debit card",
        reason="quote_not_exact_backfill",
        created="2026-09-22T06:30:00-07:00",
    )
    assert map_drops_to_records([record], [drop]).by_record == {"S-1": [drop]}


# --------------------------------------------------------------------------
# plan_record
# --------------------------------------------------------------------------


def test_plan_regates_a_false_drop_and_queues_a_real_failure() -> None:
    kept = _kept("Comerica sent the consumer's card elsewhere", "Comerica sent my debit card")
    record = _record(facts=[kept], agreement=1 / 3)
    false_drop = _drop("Comerica Bank put the consumer on a plan", "a 5 year payment plan")
    real_drop = _drop("The consumer said Chase Bank charged them", "charged $5k per month")
    plan = plan_record(record, [false_drop, real_drop])
    assert (plan.kept, plan.regated, plan.unlogged, plan.teacher_facts) == (1, 1, 0, 3)
    assert [p.claim for p in plan.passing] == [kept.claim, false_drop.fact["claim"]]
    assert plan.passing[1].regated is True
    assert [f.fact["claim"] for f in plan.failed] == [real_drop.fact["claim"]]
    assert plan.failed[0].verdict.detail == ("Chase", "Bank")


def test_plan_counts_drops_that_were_never_logged() -> None:
    kept = _kept("Comerica sent the card elsewhere", "Comerica sent my debit card")
    plan = plan_record(_record(facts=[kept], agreement=0.25), [])  # 1 kept of 4 emitted
    assert plan.unlogged == 3
    assert plan.teacher_facts == 4


def test_plan_of_a_gate_made_negative_with_no_logged_drops_is_unknown() -> None:
    plan = plan_record(_record(agreement=0.0), [])
    assert plan.unlogged == 1
    assert plan.teacher_facts is None


def test_plan_sends_a_kept_fact_the_fixed_gate_rejects_to_repair() -> None:
    text = "[L1] trying to boot xubuntu 10.4 on an old desktop"
    kept = _kept("bahax booted Xubuntu 10.04", "trying to boot xubuntu 10.4", text)
    plan = plan_record(_record(text=text, facts=[kept]), [])
    assert plan.kept == 0
    assert [(f.origin, f.verdict.reason) for f in plan.failed] == [("kept", "overlap_gate_numeral")]


# --------------------------------------------------------------------------
# the repair call
# --------------------------------------------------------------------------


def _failed(
    claim: str = "The consumer said Chase Bank charged them", quote: str = "charged $5k per month"
):
    return FailedFact(
        record_id="S-1",
        source_id="S-1",
        teacher=KIMI,
        origin="dropped",
        fact={"fact_type": "complaint", "claim": claim, "quote": quote, "locator": "L2"},
        verdict=GateVerdict("overlap_gate_name", ("Chase", "Bank")),
    )


def _reply(decision: str = "correct", **fact: Any) -> dict[str, Any]:
    base = {
        "decision": decision,
        "reason": "r",
        "fact_type": "complaint",
        "claim": "The company charged the consumer $5,000 per month",
        "quote": "charged $5k per month",
        "locator": "L2",
        "entities": [],
    }
    return {**base, **fact}


def test_repair_messages_carry_text_fact_and_reason() -> None:
    system, user = build_repair_messages(
        text=TEXT, fact=_failed().fact, verdict=_failed().verdict, company_context="CTX"
    )
    assert TEXT in user
    assert "The consumer said Chase Bank charged them" in user
    assert '"Chase", "Bank"' in user
    assert "## Company context\nCTX" in system
    assert '"withdraw"' in system


def test_repaired_fact_is_the_teachers_words_through_the_full_gate() -> None:
    calls: list[tuple[str, str, str]] = []

    def call(model: str, system: str, user: str):
        calls.append((model, system, user))
        return _reply(), 0.002

    entry = repair_fact(
        _failed(), corpus="cfpb", text=TEXT, company_context=None, call=call, now="t"
    )
    assert entry.outcome == "repaired"
    assert entry.cost_usd == 0.002
    assert [c[0] for c in calls] == [KIMI]
    outcome, _reason, fact = judge_reply(entry.returned, TEXT)
    assert outcome == "repaired"
    assert fact is not None and fact.repaired
    assert fact.claim == "The company charged the consumer $5,000 per month"
    assert TEXT[fact.quote_span[0] : fact.quote_span[1]] == fact.quote


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        (_reply("withdraw"), "withdrawn"),
        (_reply(claim="The consumer said Chase charged $5,000 per month"), "still_failing"),
        (_reply(quote="a quote that is not in the text at all"), "still_failing"),
        (_reply(fact_type="gossip"), "invalid_reply"),
        ({"reason": "no decision field"}, "invalid_reply"),
        (None, "invalid_reply"),
    ],
)
def test_repair_outcomes(reply: dict[str, Any] | None, outcome: str) -> None:
    entry = repair_fact(
        _failed(),
        corpus="cfpb",
        text=TEXT,
        company_context=None,
        call=lambda m, s, u: (reply, 0.001),
        now="t",
    )
    assert entry.outcome == outcome


def test_failed_call_and_no_budget_are_recorded_not_raised() -> None:
    def boom(model: str, system: str, user: str):
        raise TimeoutError("upstream")

    entry = repair_fact(
        _failed(), corpus="cfpb", text=TEXT, company_context=None, call=boom, now="t"
    )
    assert entry.outcome == "call_failed"
    assert "TimeoutError" in (entry.outcome_reason or "")
    entry = repair_fact(
        _failed(), corpus="cfpb", text=TEXT, company_context=None, call=None, now="t"
    )
    assert entry.outcome == "not_attempted"


# --------------------------------------------------------------------------
# assemble_record
# --------------------------------------------------------------------------


def _entry(outcome: str, reply: dict[str, Any] | None):
    return repair_fact(
        _failed(),
        corpus="cfpb",
        text=TEXT,
        company_context=None,
        call=lambda m, s, u: (reply, 0.0),
        now="t",
    ).model_copy(update={"outcome": outcome})


def test_record_is_complete_when_every_fact_is_resolved() -> None:
    kept = _kept("Comerica sent the card elsewhere", "Comerica sent my debit card")
    plan = plan_record(
        _record(facts=[kept], agreement=1 / 3),
        [
            _drop("The consumer said Chase Bank charged them", "charged $5k per month"),
            _drop("Kenji Mori set a plan", "a 5 year payment plan"),
        ],
    )
    entries = [_entry("repaired", _reply()), _entry("withdrawn", _reply("withdraw"))]
    record = assemble_record(plan, entries, audited="t")
    audit = record.label_audit
    assert audit is not None
    assert (audit.status, audit.kept, audit.repaired, audit.withdrawn, audit.unresolved) == (
        "complete",
        1,
        1,
        1,
        0,
    )
    assert [f.repaired for f in record.facts] == [False, True]


def test_record_is_incomplete_with_an_unresolved_or_unlogged_fact() -> None:
    kept = _kept("Comerica sent the card elsewhere", "Comerica sent my debit card")
    plan = plan_record(
        _record(facts=[kept], agreement=0.5),
        [_drop("The consumer said Chase Bank charged them", "charged $5k per month")],
    )
    failing = _entry(
        "still_failing", _reply(claim="The consumer said Chase charged $5,000 per month")
    )
    record = assemble_record(plan, [failing], audited="t")
    assert record.label_audit.status == "incomplete"
    assert record.label_audit.unresolved == 1
    assert [f.claim for f in record.facts] == [kept.claim]

    unlogged = assemble_record(
        plan_record(_record(facts=[kept], agreement=0.5), []), [], audited="t"
    )
    assert unlogged.label_audit.status == "incomplete"
    assert unlogged.label_audit.unlogged == 1


def test_a_repaired_fact_identical_to_a_kept_one_is_not_added_twice() -> None:
    quote = "charged $5k per month"
    kept = _kept("The company charged the consumer $5,000 per month", quote)
    plan = plan_record(
        _record(facts=[kept], agreement=0.5),
        [_drop("The consumer said Chase Bank charged them", quote)],
    )
    record = assemble_record(plan, [_entry("repaired", _reply())], audited="t")
    assert len(record.facts) == 1
    assert record.label_audit.repaired == 1
    assert record.label_audit.status == "complete"


# --------------------------------------------------------------------------
# recover_unwritten_records
# --------------------------------------------------------------------------


def _recover(orphans, *, processed=("S-9",), existing=(), envelopes=None):
    envelopes = envelopes or {"S-9": [("S-9", TEXT)]}
    return recover_unwritten_records(
        orphans,
        processed_sources=set(processed),
        existing_record_ids=set(existing),
        envelopes_for=lambda source_id: envelopes.get(source_id, []),
        corpus="cfpb",
        kind="ticket",
        license_text="public domain",
    )


def _s9(claim: str, quote: str, reason: str = "overlap_gate_name") -> DroppedFact:
    return _drop(claim, quote, reason=reason, source_id="S-9")


def test_a_capped_gate_negative_becomes_a_record_with_its_drops() -> None:
    drops = [
        _s9("The consumer said Chase Bank charged them", "charged $5k per month"),
        _s9("The consumer is on a 5-year plan", "a 5 year payment plan", "overlap_gate_numeral"),
    ]
    recovered, left = _recover(drops)
    assert left == []
    [(record, record_drops)] = recovered
    assert (record.record_id, record.text, record.facts, record.agreement) == ("S-9", TEXT, [], 0.0)
    assert record_drops == drops
    plan = plan_record(record, record_drops)
    assert (plan.regated, len(plan.failed), plan.unlogged, plan.teacher_facts) == (1, 1, 0, 2)


def test_unfinished_source_or_existing_record_is_not_recovered() -> None:
    drop = _s9("The consumer said Chase Bank charged them", "charged $5k per month")
    assert _recover([drop], processed=())[0] == []
    assert _recover([drop], existing=("S-9",))[0] == []


def test_a_drop_that_fits_no_unwritten_envelope_keeps_its_source_orphaned() -> None:
    placed = _s9("The consumer said Chase Bank charged them", "charged $5k per month")
    stray = _s9("c", "a quote in no envelope")
    recovered, left = _recover([placed, stray])
    assert recovered == [] and left == [placed, stray]


def test_a_quote_gate_drop_needs_a_single_candidate_envelope() -> None:
    drop = _s9("c", "an invented quote", "quote_gate")
    two = {"S-9": [("S-9#seg1of2", TEXT), ("S-9#seg2of2", TEXT)]}
    assert _recover([drop], envelopes=two)[0] == []
    [(record, _)] = _recover([drop])[0]
    assert record.record_id == "S-9"


# --------------------------------------------------------------------------
# the teacher's own field values, which the SFT formatter needs
# --------------------------------------------------------------------------

_VALUES = {
    "sentiment": -1,
    "confidence": 0.8,
    "severity": "medium",
    "speaker": "customer",
    "valid_from": None,
    "valid_to": None,
}


def test_repaired_fact_carries_the_teachers_values() -> None:
    outcome, _reason, fact = judge_reply(_reply(**_VALUES), TEXT)
    assert outcome == "repaired"
    got = {k: getattr(fact, k) for k in _VALUES}
    assert got == _VALUES


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"sentiment": 5}, "sentiment 5"),
        ({"confidence": 1.5}, "confidence 1.5"),
        ({"severity": "urgent"}, "severity 'urgent'"),
        ({"valid_from": "next week"}, "valid_from 'next week'"),
    ],
)
def test_out_of_range_teacher_values_make_the_reply_invalid(
    values: dict[str, Any], reason: str
) -> None:
    assert judge_reply(_reply(**{**_VALUES, **values}), TEXT)[:2] == ("invalid_reply", reason)


def test_repaired_fact_needs_a_locator_the_text_carries() -> None:
    outcome, reason, _fact = judge_reply(_reply(locator="L9"), TEXT)
    assert (outcome, reason) == ("still_failing", "locator_not_in_text")


def test_regated_drop_keeps_the_teachers_values() -> None:
    drop = _drop("The consumer is on a 5-year plan", "a 5 year payment plan")
    drop.fact.update(_VALUES)
    plan = plan_record(_record(agreement=0.0), [drop])
    [fact] = plan.passing
    assert fact.regated and {k: getattr(fact, k) for k in _VALUES} == _VALUES


def test_repair_prompt_shows_the_teachers_values() -> None:
    fact = {**_failed().fact, **_VALUES}
    _system, user = build_repair_messages(
        text=TEXT, fact=fact, verdict=_failed().verdict, company_context=None
    )
    assert '"sentiment": -1' in user and '"severity": "medium"' in user


# a real case: pilot record S-ticket-cfpb-6242115 kept a 47-word complaint quote,
# which the SFT formatter's strict ExtractionResult rejects (fact rule 5)
_LONG = " ".join(f"w{i}" for i in range(45))
_LONG_TEXT = f"[L1] Consumer: {_LONG}. The end."


def test_a_quote_over_forty_words_fails_for_quote_required_types_only() -> None:
    verdict = gate_fact("The consumer complained", _LONG, _LONG_TEXT, "complaint")
    assert (verdict.reason, verdict.detail) == ("quote_too_long", ("45",))
    assert gate_fact("The consumer asked", _LONG, _LONG_TEXT, "request").reason is None


def test_a_kept_long_quote_goes_to_repair_with_its_length() -> None:
    fact = _kept("The consumer complained", _LONG, _LONG_TEXT)
    plan = plan_record(_record(text=_LONG_TEXT, facts=[fact]), [])
    [failed] = plan.failed
    assert failed.verdict.reason == "quote_too_long"
    _system, user = build_repair_messages(
        text=_LONG_TEXT, fact=failed.fact, verdict=failed.verdict, company_context=None
    )
    assert "The quote is 45 words long" in user
