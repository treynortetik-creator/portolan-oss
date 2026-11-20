"""Tests for `portolan.extract.review_pass` (reviewer prompt/parsing) and
`portolan.bench.review_bench` (combine + scoring). Wave 1D,
`01-spec/extraction-accuracy-campaign-2026-09-18.md`.

No network calls anywhere in this file. The score_combined/sweep-recovery
tests use the real, committed `corpus/truth-table.yaml` +
`corpus/out/manifest.json` fixtures (same convention as
`tests/test_bench_extraction.py`) rather than any saved bench run under
`var/` (gitignored, not guaranteed to exist in a fresh checkout).
"""

from __future__ import annotations

from pathlib import Path

from portolan.bench.extraction import (
    ExtractionScore,
    MatchedFact,
    ProducedFactRecord,
    UnplantedFact,
)
from portolan.bench.review_bench import (
    combine_source,
    rejected_fact_classification,
    score_combined,
    sweep_recovered_fact_ids,
)
from portolan.bench.truth import Truth
from portolan.extract.review_pass import (
    FactVerdict,
    ReviewedMissedFact,
    ReviewResult,
    build_review_system_prompt,
    build_review_user_message,
    parse_review_lenient,
    review_result_json_schema,
)
from portolan.extract.schema import (
    EntityKind,
    RegistryContextEntry,
    SourceEnvelope,
    SourceType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"


def _truth() -> Truth:
    return Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)


def _envelope(**overrides) -> SourceEnvelope:
    base = dict(
        source_id="S-test-001",
        source_type=SourceType.CALL,
        source_date="2026-08-01",
        text="[00:00:00] Felix Arden: that shipment was four days late.",
    )
    base.update(overrides)
    return SourceEnvelope(**base)


def _pass1_fact(**overrides) -> ProducedFactRecord:
    base = dict(
        id="f1",
        type="complaint",
        claim="Larkspur Memorial Hospital's Tidewater T2 order arrived four days late.",
        entities=["larkspur-memorial"],
        locator="00:00:00",
        quote="that shipment was four days late",
        confidence=0.8,
    )
    base.update(overrides)
    return ProducedFactRecord(**base)


# --------------------------------------------------------------------------
# review_result_json_schema / parse_review_lenient
# --------------------------------------------------------------------------


def test_review_result_json_schema_has_verdicts_and_missed() -> None:
    schema = review_result_json_schema()
    assert set(schema["properties"]) == {"verdicts", "missed"}


def test_parse_review_lenient_keeps_good_items() -> None:
    data = {
        "verdicts": [{"fact_id": "f1", "verdict": "keep"}],
        "missed": [{"type": "pricing", "claim": "X paid Y", "locator": "00:01:00"}],
    }
    review, dropped = parse_review_lenient(data, valid_fact_ids={"f1"})
    assert dropped == []
    assert len(review.verdicts) == 1
    assert review.verdicts[0].fact_id == "f1"
    assert len(review.missed) == 1


def test_parse_review_lenient_drops_malformed_verdict() -> None:
    data = {"verdicts": [{"fact_id": "f1", "verdict": "not-a-real-verdict"}], "missed": []}
    review, dropped = parse_review_lenient(data, valid_fact_ids={"f1"})
    assert review.verdicts == []
    assert len(dropped) == 1
    assert dropped[0].kind == "verdict"


def test_parse_review_lenient_drops_verdict_for_unknown_fact_id() -> None:
    data = {"verdicts": [{"fact_id": "f99", "verdict": "keep"}], "missed": []}
    review, dropped = parse_review_lenient(data, valid_fact_ids={"f1"})
    assert review.verdicts == []
    assert len(dropped) == 1
    assert "f99" in dropped[0].error


def test_parse_review_lenient_drops_missed_item_with_unknown_type() -> None:
    data = {
        "verdicts": [],
        "missed": [{"type": "not-a-type", "claim": "X", "locator": "p1"}],
    }
    review, dropped = parse_review_lenient(data)
    assert review.missed == []
    assert len(dropped) == 1
    assert dropped[0].kind == "missed"


def test_parse_review_lenient_drops_non_object_items_without_raising() -> None:
    data = {"verdicts": ["not an object"], "missed": [42]}
    review, dropped = parse_review_lenient(data)
    assert review.verdicts == []
    assert review.missed == []
    assert len(dropped) == 2


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------


def test_build_review_system_prompt_includes_company_context() -> None:
    prompt = build_review_system_prompt(company_context="Halvard Medical makes infusion pumps.")
    assert "## Company context" in prompt
    assert "Halvard Medical makes infusion pumps." in prompt
    assert "VERIFY" in prompt
    assert "SWEEP" in prompt


def test_build_review_system_prompt_omits_company_context_block_when_none() -> None:
    prompt = build_review_system_prompt()
    assert "## Company context" not in prompt


def test_build_review_user_message_includes_source_text_facts_and_sweep() -> None:
    envelope = _envelope(
        registry_context=[
            RegistryContextEntry(
                name="Larkspur Memorial Hospital",
                kind=EntityKind.ACCOUNT,
                registry_id="larkspur-memorial",
            )
        ]
    )
    fact = _pass1_fact()
    message = build_review_user_message(envelope, [fact])
    assert "four days late" in message  # source text
    assert "Larkspur Memorial Hospital" in message  # registry context
    assert "fact_id=f1" in message  # pass-1 fact block
    assert "Sweep questions" in message


def test_build_review_user_message_notes_when_pass1_produced_nothing() -> None:
    envelope = _envelope()
    message = build_review_user_message(envelope, [])
    assert "pass 1 produced no facts for this source" in message


# --------------------------------------------------------------------------
# combine_source
# --------------------------------------------------------------------------


def test_combine_keep_survives_unchanged() -> None:
    facts, diag = combine_source(
        "S-test-001",
        [_pass1_fact()],
        ReviewResult(verdicts=[FactVerdict(fact_id="f1", verdict="keep")]),
        mode="combined",
    )
    assert len(facts) == 1
    assert facts[0].claim == _pass1_fact().claim
    assert diag.rejected == []
    assert diag.fixed == []


def test_combine_reject_drops_the_fact() -> None:
    facts, diag = combine_source(
        "S-test-001",
        [_pass1_fact()],
        ReviewResult(
            verdicts=[FactVerdict(fact_id="f1", verdict="reject", reject_reason="not_in_source")]
        ),
        mode="combined",
    )
    assert facts == []
    assert diag.rejected == [("f1", "not_in_source")]


def test_combine_fix_applies_corrected_fields() -> None:
    facts, diag = combine_source(
        "S-test-001",
        [_pass1_fact()],
        ReviewResult(
            verdicts=[
                FactVerdict(
                    fact_id="f1",
                    verdict="fix",
                    fixed_type="risk",
                    fixed_claim="Larkspur Memorial Hospital's late delivery threatens renewal.",
                )
            ]
        ),
        mode="combined",
    )
    assert len(facts) == 1
    assert facts[0].type == "risk"
    assert facts[0].claim == "Larkspur Memorial Hospital's late delivery threatens renewal."
    # quote untouched by the fix (fixed_quote was left null) -- risk isn't
    # quote-required, so the carried-over complaint quote is still legal.
    assert facts[0].quote == _pass1_fact().quote
    assert diag.fixed == ["f1"]


def test_combine_fix_that_produces_an_invalid_fact_is_dropped_not_kept() -> None:
    # commitment/pricing/complaint/praise/competitor_mention all require a
    # <=40-word quote (fact rule 5). Fixing the type to one of those with
    # no quote at all makes the resulting fact invalid -- combine_source
    # must drop it (diagnosed), not silently keep the pre-fix version.
    facts, diag = combine_source(
        "S-test-001",
        [_pass1_fact(quote=None, type="usage")],
        ReviewResult(verdicts=[FactVerdict(fact_id="f1", verdict="fix", fixed_type="pricing")]),
        mode="combined",
    )
    assert facts == []
    assert len(diag.dropped_invalid) == 1
    assert diag.rejected == []  # this was a bad fix, not an explicit reject


def test_combine_missed_fact_is_added_in_combined_mode() -> None:
    missed = ReviewedMissedFact(
        type="pricing", claim="X was quoted $500", quote="quoted $500", locator="00:02:00"
    )
    facts, diag = combine_source("S-test-001", [], ReviewResult(missed=[missed]), mode="combined")
    assert len(facts) == 1
    # Not an id-prefix check: `_merge_facts` (portolan.extract.segment,
    # reused unmodified) renumbers every surviving fact `f1, f2, ...`
    # after dedup regardless of what id it arrived with -- see
    # sweep_recovered_fact_ids's docstring for how sweep provenance is
    # actually tracked (a matched-fact_id set diff against a baseline
    # score, not an id convention).
    assert facts[0].claim == "X was quoted $500"
    assert diag.missed_added == 1


def test_combine_verify_only_ignores_the_sweep() -> None:
    missed = ReviewedMissedFact(
        type="pricing", claim="X was quoted $500", quote="quoted $500", locator="00:02:00"
    )
    facts, diag = combine_source(
        "S-test-001",
        [_pass1_fact()],
        ReviewResult(verdicts=[FactVerdict(fact_id="f1", verdict="keep")], missed=[missed]),
        mode="verify_only",
    )
    assert len(facts) == 1
    assert facts[0].id == "f1"
    assert diag.missed_added == 0


def test_combine_sweep_only_ignores_verdicts_entirely() -> None:
    # Even an explicit reject is ignored in sweep_only -- VERIFY is fully
    # switched off, per the module docstring's three combine modes.
    missed = ReviewedMissedFact(
        type="pricing", claim="X was quoted $500", quote="quoted $500", locator="00:02:00"
    )
    facts, diag = combine_source(
        "S-test-001",
        [_pass1_fact()],
        ReviewResult(
            verdicts=[FactVerdict(fact_id="f1", verdict="reject", reject_reason="not_in_source")],
            missed=[missed],
        ),
        mode="sweep_only",
    )
    assert len(facts) == 2
    assert diag.rejected == []  # verdicts not consulted in this mode


def test_combine_dedups_a_missed_fact_that_duplicates_a_kept_one() -> None:
    # Same type, near-identical claim, close locators -- the existing
    # extract.segment dedup rule (_facts_match) should merge these into
    # one. `ReviewedMissedFact` carries no `entities` field (DESIGN: a
    # missed fact is type/claim/quote/locator only), so `_missed_to_fact`
    # always gives it `entities=[]` -- _facts_match's entity-overlap check
    # only passes when BOTH sides are empty, so this test's pass-1 fact
    # is built with entities=[] too. A kept fact that still carries a
    # resolved entity list won't dedup against a same-claim sweep fact
    # through the entity check alone; documented in
    # 01-spec/two-pass-2026-09-18.md as a known limitation of reusing
    # segment.py's dedup unmodified.
    pass1 = _pass1_fact(entities=[])
    duplicate_missed = ReviewedMissedFact(
        type="complaint",
        claim="Larkspur Memorial Hospital's Tidewater T2 shipment was four days late.",
        quote="that shipment was four days late",
        locator="00:00:05",
    )
    keep_verdict = FactVerdict(fact_id="f1", verdict="keep")
    facts, diag = combine_source(
        "S-test-001",
        [pass1],
        ReviewResult(verdicts=[keep_verdict], missed=[duplicate_missed]),
        mode="combined",
    )
    assert len(facts) == 1
    assert diag.missed_added == 1  # the sweep DID report it; dedup merged it away


# --------------------------------------------------------------------------
# score_combined / sweep_recovered_fact_ids -- real corpus fixtures
# --------------------------------------------------------------------------


def test_sweep_recovers_a_fact_pass1_never_produced() -> None:
    """F35 (`other`, no quote required) is planted in the bare CRM row
    `S-crm-A1010` -- exactly the "CRM silence" gap
    `01-spec/extraction-miss-analysis-2026-09-18.md` found (a real model
    produces ZERO facts from these rows). Pass 1 here produced nothing
    for it; the sweep finds it verbatim."""
    truth = _truth()
    planted = truth.planted("S-crm-A1010")
    assert len(planted) == 1
    f35 = planted[0]

    missed = ReviewedMissedFact(
        type=f35.type, claim=f35.claim, quote=None, locator="account:A-1010", entities=["ashgrove"]
    )
    crm_row = "[account:A-1010] crm_id=A-1010, name=Ashgrove Health, arr=2350000"
    source_texts = {"S-crm-A1010": crm_row}

    combined_facts, _diag = combine_source(
        "S-crm-A1010", [], ReviewResult(missed=[missed]), mode="combined"
    )
    score = score_combined(truth, {"S-crm-A1010": combined_facts}, source_texts)

    baseline_facts, _diag2 = combine_source("S-crm-A1010", [], ReviewResult(), mode="verify_only")
    baseline_score = score_combined(truth, {"S-crm-A1010": baseline_facts}, source_texts)
    assert baseline_score.matched_count == 0  # pass 1 produced nothing here -- the baseline

    assert score.planted_total == 1
    assert score.matched_count == 1
    assert score.recall == 1.0
    assert sweep_recovered_fact_ids(score, baseline_score) == ["F35"]


def test_verify_only_does_not_recover_the_same_fact() -> None:
    truth = _truth()
    crm_row = "[account:A-1010] crm_id=A-1010, name=Ashgrove Health, arr=2350000"
    source_texts = {"S-crm-A1010": crm_row}
    missed = ReviewedMissedFact(
        type="other", claim="irrelevant", quote=None, locator="account:A-1010"
    )

    combined_facts, _diag = combine_source(
        "S-crm-A1010", [], ReviewResult(missed=[missed]), mode="verify_only"
    )
    score = score_combined(truth, {"S-crm-A1010": combined_facts}, source_texts)

    assert combined_facts == []
    assert score.planted_total == 1
    assert score.matched_count == 0
    assert score.recall == 0.0


# --------------------------------------------------------------------------
# rejected_fact_classification
# --------------------------------------------------------------------------


def _pass1_score_fixture() -> ExtractionScore:
    matched_rec = ProducedFactRecord(
        id="f1",
        type="risk",
        claim="real match",
        entities=[],
        locator="p1",
        quote=None,
        confidence=0.8,
    )
    invented_rec_id = "f2"
    plausible_rec_id = "f3"
    return ExtractionScore(
        sources_scored=1,
        planted_total=1,
        matched_count=1,
        produced_total=3,
        recall=1.0,
        precision=1 / 3,
        distractor_leakage=0,
        quote_presence_rate=None,
        locator_presence_rate=1.0,
        entity_accuracy=None,
        entity_accuracy_checked=0,
        matched=[
            MatchedFact(
                fact_id="F01",
                source_id="S-1",
                fact_type="risk",
                produced_type="risk",
                credit=1.0,
                similarity=0.9,
                produced_claim="real match",
                produced=matched_rec,
            )
        ],
        unmatched=[],
        unplanted=[
            UnplantedFact(
                source_id="S-1",
                fact_type="other",
                claim="fabricated",
                locator="p2",
                produced_id=invented_rec_id,
                in_source_text=False,
                evidence_basis="none",
            ),
            UnplantedFact(
                source_id="S-1",
                fact_type="other",
                claim="real but unplanted",
                locator="p3",
                produced_id=plausible_rec_id,
                in_source_text=True,
                evidence_basis="fragments",
            ),
        ],
    )


def test_rejected_fact_classification_matched() -> None:
    score = _pass1_score_fixture()
    assert rejected_fact_classification(score, "S-1", "f1") == "matched"


def test_rejected_fact_classification_invented() -> None:
    score = _pass1_score_fixture()
    assert rejected_fact_classification(score, "S-1", "f2") == "invented"


def test_rejected_fact_classification_unplanted_plausible() -> None:
    score = _pass1_score_fixture()
    assert rejected_fact_classification(score, "S-1", "f3") == "unplanted_plausible"


def test_rejected_fact_classification_unknown() -> None:
    score = _pass1_score_fixture()
    assert rejected_fact_classification(score, "S-1", "not-a-real-id") == "unknown"
