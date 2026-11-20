"""Tests for `portolan.bench.extraction` and `portolan.bench.fixtures`."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from portolan.bench.extraction import (
    EvidenceContext,
    ExtractionRunResult,
    _evidence_basis,
    _fragment_vote,
    _fragments_supported,
    _fuzzy_quote_containment,
    _plausible_in_source,
    _quote_supported_by_text,
    _registry_context_for,
    _salient_claim_fragments,
    _window_supported,
    evidence_context_for_source,
    rescore_full,
    run_extraction,
    score_extraction,
)
from portolan.bench.fixtures import lossy_results, perfect_results
from portolan.bench.truth import Truth
from portolan.extract.backends import FakeBackend, LLMClientBackend
from portolan.extract.schema import (
    Coverage,
    EntityCoercion,
    EntityKind,
    EntityMention,
    ExtractionResult,
    Fact,
    FactType,
)
from portolan.llm.base import LLMResult
from portolan.metering import InMemorySink, metered

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"
CORPUS_ROOT = REPO_ROOT / "corpus" / "out"


def _truth() -> Truth:
    return Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)


def test_perfect_results_score_one_and_one() -> None:
    truth = _truth()
    results = perfect_results(truth)
    score = score_extraction(truth, results)
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)

    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.planted_total == planted_total
    assert score.matched_count == planted_total
    assert score.distractor_leakage == 0
    assert score.unplanted == []
    assert score.unmatched == []
    assert score.quote_presence_rate == 1.0
    assert score.locator_presence_rate == 1.0


def test_dropping_five_facts_gives_recall_68_of_73() -> None:
    truth = _truth()
    results = lossy_results(truth, drop=5, seed=0)
    score = score_extraction(truth, results)
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)

    assert score.planted_total == planted_total
    assert score.matched_count == planted_total - 5
    assert score.recall == (planted_total - 5) / planted_total
    assert len(score.unmatched) == 5
    for u in score.unmatched:
        assert u.failed_criteria  # something concrete explains the miss


def test_inventing_three_facts_gives_three_unplanted() -> None:
    truth = _truth()
    results = lossy_results(truth, invent=3, seed=0)
    score = score_extraction(truth, results)
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)

    # Recall is unaffected -- nothing planted was removed.
    assert score.matched_count == planted_total
    assert score.recall == 1.0
    # Every invented fact lands in the non-distractor "unplanted" bucket,
    # not counted as distractor leakage or as an "error".
    assert len(score.unplanted) == 3
    assert score.distractor_leakage == 0
    assert all(not u.is_distractor_leak for u in score.unplanted)
    assert score.precision is not None and score.precision < 1.0


def test_wrong_locator_fails_locator_criterion_with_reason() -> None:
    truth = _truth()
    results = lossy_results(truth, wrong_locator=1, seed=1)
    score = score_extraction(truth, results)
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)

    assert len(score.unmatched) == 1
    miss = score.unmatched[0]
    assert "locator" in miss.failed_criteria
    # Everything else about the candidate matched: high claim similarity to
    # the best candidate is how we know the *only* problem was the locator.
    assert miss.best_similarity >= 0.99
    assert score.matched_count == planted_total - 1


def test_coerced_sentiment_is_summed_across_sources() -> None:
    """Fix wave 2026-09-22 (`01-spec/bench-deepseek-v41-flash-2026-09-22.md`):
    `ExtractionResult.coerced_sentiment_count` (per source) rolls up into
    `ExtractionScore.coerced_sentiment` (whole run), the number `report.py`
    prints in the Extraction table -- previously nothing tracked this at
    all, so a run like the DeepSeek V4.1 Flash bench (every fact's
    `sentiment` coerced from a string) was invisible in the scorecard."""
    truth = _truth()
    results = perfect_results(truth)
    source_ids = list(results)
    coercion = EntityCoercion(
        index=0, mention="f1", field="sentiment", original="negative", applied=-1
    )
    results[source_ids[0]] = results[source_ids[0]].model_copy(update={"coercions": [coercion]})
    # A second source's coercion confirms this sums across sources, not
    # just reads the last one.
    results[source_ids[1]] = results[source_ids[1]].model_copy(update={"coercions": [coercion]})

    score = score_extraction(truth, results)

    assert score.coerced_sentiment == 2


def test_distractor_source_leakage_is_reported_separately_from_unplanted() -> None:
    truth = _truth()
    results = perfect_results(truth)
    distractor_source_id = next(iter(truth.distractor_source_ids()))
    leaked_fact = Fact(
        id="leak-1",
        type=FactType.OTHER,
        claim="Something extracted from a source that should have yielded nothing.",
        entities=[],
        locator="00:00:01",
        sentiment=0,
        confidence=0.5,
    )
    result = results[distractor_source_id]
    results[distractor_source_id] = result.model_copy(update={"facts": [leaked_fact]})

    score = score_extraction(truth, results)
    assert score.distractor_leakage == 1
    assert any(u.is_distractor_leak for u in score.unplanted)


def test_type_other_gets_partial_credit_against_a_real_type() -> None:
    truth = _truth()
    results = perfect_results(truth)
    # F01 is planted in S-call-007 as a `complaint`. Relabel the produced
    # fact `other` -- compatible-but-imprecise, per fact rule 8.
    source_id = "S-call-007"
    facts = [
        f.model_copy(update={"type": FactType.OTHER}) if f.id == "F01" else f
        for f in results[source_id].facts
    ]
    results[source_id] = results[source_id].model_copy(update={"facts": facts})

    score = score_extraction(truth, results)
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)
    f01_match = next(m for m in score.matched if m.fact_id == "F01")
    assert f01_match.credit == 0.5
    assert f01_match.produced_type == "other"
    # Still counted as recalled (with partial credit), not as a miss.
    assert score.matched_count == planted_total
    assert score.recall == pytest.approx(((planted_total - 1) + 0.5) / planted_total)


def test_run_extraction_segments_and_merges_through_fake_backend() -> None:
    """End-to-end: corpus_loader -> segment -> FakeBackend -> merge, wired
    through `run_extraction`, scored against the truth table. S-call-011
    covers the within-source-dedup fact (F04) too."""
    truth = _truth()
    canned = perfect_results(truth)
    backend = FakeBackend(canned)

    run_result = run_extraction(
        truth.manifest,
        backend,
        source_ids=["S-call-005", "S-call-011"],
        segment=True,
        corpus_root=CORPUS_ROOT,
    )

    assert set(run_result) == {"S-call-005", "S-call-011"}
    assert run_result.sink.events  # metering fired at least once per segment
    assert run_result.total_wall_time_s >= 0.0

    score = score_extraction(truth, run_result)
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.tokens_input > 0
    assert score.wall_time_s >= 0.0


class _SpyBackend:
    """Records every envelope it's asked to extract, then delegates to a
    real `FakeBackend` for the canned result."""

    def __init__(self, canned) -> None:
        self._inner = FakeBackend(canned)
        self.envelopes: list = []

    def extract(self, envelope, system_prompt, sink, tenant):
        self.envelopes.append(envelope)
        return self._inner.extract(envelope, system_prompt, sink, tenant)


def test_run_extraction_without_truth_leaves_envelope_registry_context_empty() -> None:
    """Default (unchanged) behavior: no `truth` -> every existing caller
    (including every other test in this file) keeps seeing an empty
    `registry_context`/`vocabulary`, exactly as before this fix."""
    truth = _truth()
    canned = perfect_results(truth)
    spy = _SpyBackend(canned)

    run_extraction(
        truth.manifest,
        spy,
        source_ids=["S-call-009"],
        segment=True,
        corpus_root=CORPUS_ROOT,
    )

    assert spy.envelopes
    for envelope in spy.envelopes:
        assert envelope.registry_context == []
        assert envelope.vocabulary.products == []


def test_run_extraction_with_truth_populates_registry_context_and_vocabulary() -> None:
    """Diagnosis: with no registry_context, a real produced fact about
    S-call-009 never named the account (`larkspur-memorial`) at all -- nothing
    in the envelope said which account the call belonged to. `truth=truth`
    fills `registry_context` from the manifest's own `accounts`/
    `participants` (exactly what a real connector already knows before
    reading the transcript) and `vocabulary` from the truth table's static
    product/competitor list."""
    truth = _truth()
    canned = perfect_results(truth)
    spy = _SpyBackend(canned)

    run_extraction(
        truth.manifest,
        spy,
        source_ids=["S-call-009"],
        segment=True,
        corpus_root=CORPUS_ROOT,
        truth=truth,
    )

    assert spy.envelopes
    for envelope in spy.envelopes:
        ids = {entry.registry_id for entry in envelope.registry_context}
        # S-call-009: account larkspur-memorial, participants felix-arden
        # (a truth-table contact) and sam-okafor (an internal person).
        assert "larkspur-memorial" in ids
        assert "felix-arden" in ids
        assert "internal:sam-okafor" in ids
        account_entry = next(
            e for e in envelope.registry_context if e.registry_id == "larkspur-memorial"
        )
        assert account_entry.kind.value == "account"
        assert "Larkspur Memorial Hospital" == account_entry.name
        # Static, source-independent vocabulary -- same on every segment.
        assert "Tidewater T2" in envelope.vocabulary.products
        assert "Corvane Systems" in envelope.vocabulary.competitors


def test_run_extraction_records_failed_sources_and_continues() -> None:
    # Group 3 of the review fix wave: one source's extraction raising must
    # not crash the whole bench run.
    truth = _truth()
    canned = perfect_results(truth)

    def flaky(envelope):
        if envelope.source_id == "S-call-011":
            raise RuntimeError("boom")
        return canned[envelope.source_id]

    backend = FakeBackend(flaky)

    run_result = run_extraction(
        truth.manifest,
        backend,
        source_ids=["S-call-005", "S-call-011"],
        segment=True,
        corpus_root=CORPUS_ROOT,
    )

    assert set(run_result) == {"S-call-005"}
    assert set(run_result.failed_sources) == {"S-call-011"}
    assert "boom" in run_result.failed_sources["S-call-011"]

    score = score_extraction(truth, run_result)
    assert score.failed_sources == run_result.failed_sources
    assert score.sources_scored == 1


def test_fake_backend_canned_result_shape_is_a_valid_extraction_result() -> None:
    # Sanity: perfect_results' Fact objects satisfy the schema's own
    # validators (locator required, quote required for evidentiary types).
    truth = _truth()
    results = perfect_results(truth)
    assert isinstance(results["S-ticket-019"], ExtractionResult)
    assert isinstance(results["S-ticket-019"].coverage, Coverage)
    for fact in results["S-call-013"].facts:
        assert fact.locator


# --------------------------------------------------------------------------
# Diagnosis-driven scoring fixes: near-match types, weighted claim
# similarity, per-source produced-fact persistence, unplanted
# classification, judge-assisted promotion, and thread-pool concurrency.
# See var/bench/diag/diagnosis.md.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fact_id", "source_id", "near_type"),
    [
        ("F09", "S-call-013", FactType.DECISION),  # commitment <-> decision
        ("F01", "S-call-007", FactType.RISK),  # complaint <-> risk
        ("F19", "S-call-007", FactType.QUESTION),  # request <-> question
    ],
)
def test_adjacent_type_pairs_get_partial_credit(fact_id, source_id, near_type) -> None:
    truth = _truth()
    results = perfect_results(truth)
    facts = [
        f.model_copy(update={"type": near_type}) if f.id == fact_id else f
        for f in results[source_id].facts
    ]
    results[source_id] = results[source_id].model_copy(update={"facts": facts})

    score = score_extraction(truth, results)
    match = next(m for m in score.matched if m.fact_id == fact_id)
    assert match.credit == 0.5
    assert match.produced_type == near_type.value


def test_unrelated_type_mismatch_is_still_a_hard_miss() -> None:
    truth = _truth()
    results = perfect_results(truth)
    # F01 is a complaint; "usage" is not one of the near-match pairs.
    facts = [
        f.model_copy(update={"type": FactType.USAGE}) if f.id == "F01" else f
        for f in results["S-call-007"].facts
    ]
    results["S-call-007"] = results["S-call-007"].model_copy(update={"facts": facts})

    score = score_extraction(truth, results)
    assert not any(m.fact_id == "F01" for m in score.matched)
    assert any(u.fact_id == "F01" and "type" in u.failed_criteria for u in score.unmatched)


def test_claim_similarity_credits_a_real_paraphrase_the_old_metric_missed() -> None:
    """A produced claim phrased much closer to the source's own words than
    the truth table's durable third-person claim -- close to how F02's own
    ticket text reads -- shares almost no long character run with the
    planted claim (the old `difflib.SequenceMatcher` ratio scores well
    below its own 0.45 cutoff here), but shares nearly every salient token
    (the entities, the day count, the fact that it's a delivery running
    late). Weighted token-set overlap credits it; the old metric did not."""
    truth = _truth()
    results = perfect_results(truth)
    ticket_phrasing = (
        "Larkspur Memorial's Tidewater T2 consumable order that was due "
        "August 1st didn't arrive until August 4th - 4 days late"
    )
    facts = [
        f.model_copy(update={"claim": ticket_phrasing}) if f.id == "F02" else f
        for f in results["S-ticket-031"].facts
    ]
    results["S-ticket-031"] = results["S-ticket-031"].model_copy(update={"facts": facts})

    score = score_extraction(truth, results)
    assert any(m.fact_id == "F02" and m.source_id == "S-ticket-031" for m in score.matched)


def test_unmatched_best_candidate_carries_full_produced_record() -> None:
    """The diagnosis fix: a miss's closest candidate is persisted with every
    field (type, claim, entities, locator, quote, confidence), not just its
    claim text, so a diagnosis pass never needs to re-run the extractor to
    see what the model actually produced."""
    truth = _truth()
    results = lossy_results(truth, drop=1, seed=2)
    score = score_extraction(truth, results)
    assert score.unmatched
    for miss in score.unmatched:
        if miss.best_candidate is not None:
            assert miss.best_candidate.claim == miss.best_candidate_claim
            assert isinstance(miss.best_candidate.entities, list)
            assert miss.best_candidate.locator


def test_produced_by_source_persists_every_produced_fact_with_full_fields() -> None:
    truth = _truth()
    results = perfect_results(truth)
    score = score_extraction(truth, results)

    assert "S-call-013" in score.produced_by_source
    produced = score.produced_by_source["S-call-013"]
    assert len(produced) == len(results["S-call-013"].facts)
    for record, fact in zip(produced, results["S-call-013"].facts, strict=True):
        assert record.id == fact.id
        assert record.type == fact.type.value
        assert record.claim == fact.claim
        assert record.entities == list(fact.entities)
        assert record.locator == fact.locator
        assert record.quote == fact.quote
        assert record.confidence == fact.confidence


def test_unplanted_fact_classifies_plausible_vs_invented_against_source_text() -> None:
    """D04 (`S-call-002`): a routine check-in with no planted facts, but
    real scheduling chatter the extractor might still (wrongly) surface.
    `in_source_text` tells that kind of leak (real content, just not a
    planted fact) apart from an outright invention with no basis in the
    transcript at all -- both count as leakage, but the report should say
    which is which."""
    truth = _truth()
    # v1.1 (2026-09-13): S-call-002 was regenerated through the live
    # `--mode openrouter` writer (corpus/generator/prose.py), so the exact
    # wording of its genuine scheduling chatter changed -- this quote is
    # verbatim from the CURRENT corpus/out/calls/S-call-002.txt, not the old
    # hand-written v0 text.
    plausible = Fact(
        id="f1",
        type=FactType.COMMITMENT,
        claim=(
            "Nina Vale committed to get back to Priya Natarajan once she checked with her team."
        ),
        entities=[],
        locator="09:12:11",
        # Curly apostrophes, verbatim from the source line -- `_normalize_for_
        # containment` (portolan/bench/extraction.py) lowercases and collapses
        # whitespace but does NOT fold quote-mark styles, so a straight "'"
        # here would silently fail the containment check.
        quote="I’ll get back to you once I’ve checked with the team.",
        speaker="customer",
        sentiment=0,
        confidence=0.7,
    )
    invented = Fact(
        id="f2",
        type=FactType.OTHER,
        claim=(
            "Ashgrove Medical Center signed a five-year contract worth 10 "
            "million dollars on 2026-01-01."
        ),
        entities=[],
        locator="00:01:00",
        sentiment=0,
        confidence=0.5,
    )
    canned = ExtractionResult(
        source_id="S-call-002",
        summary="fixture",
        entities=[],
        facts=[plausible, invented],
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    backend = FakeBackend({"S-call-002": canned})

    run_result = run_extraction(
        truth.manifest,
        backend,
        source_ids=["S-call-002"],
        segment=False,
        corpus_root=CORPUS_ROOT,
    )
    score = score_extraction(truth, run_result)

    by_id = {u.produced_id: u for u in score.unplanted if u.source_id == "S-call-002"}
    assert by_id["f1"].in_source_text is True
    assert by_id["f1"].classification == "unplanted_plausible"
    assert by_id["f2"].in_source_text is False
    assert by_id["f2"].classification == "invented"
    # Both are real leaks (the source is D04, a distractor) regardless of
    # classification -- distractor leakage and textual plausibility are
    # independent axes.
    assert by_id["f1"].is_distractor_leak
    assert by_id["f2"].is_distractor_leak


class _FakeJudge:
    """Minimal `JudgeClient`-shaped double: returns a fixed verdict string
    from every `.complete()` call and counts how many times it was asked."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    def complete(self, *, system: str, user: str, surface: str, max_tokens: int = 4096):
        del system, user, surface, max_tokens
        self.calls += 1
        return type("Result", (), {"text": self.verdict})()


def test_judge_promotes_paraphrase_when_only_claim_similarity_failed() -> None:
    truth = _truth()
    results = perfect_results(truth)
    # Only F01, reworded -- not the full S-call-007 produced list (F01, F19,
    # F25). v1.1's regenerated S-call-007 (corpus/generator/prose.py's live
    # writer) happens to place F25 closer to F01's own locator than the old
    # hand-written transcript did, so with all three produced facts present,
    # F25 (wrong type, but now within F01's locator window, similarity 0.31)
    # ties F01's own reworded record on failure count and wins the "best
    # candidate" tie-break on raw similarity -- `failed_criteria` for the
    # winning candidate becomes `["type"]`, not `["claim_similarity"]`, so
    # the promotion this test exists to check never becomes eligible. This
    # test is about the claim-similarity-only path specifically, so it
    # isolates that path by dropping the other two produced facts rather
    # than depending on their locators never drifting close to F01's again.
    facts = [
        f.model_copy(
            update={
                "claim": "There were some issues with getting supplies to the "
                "hospital on time recently."
            }
        )
        for f in results["S-call-007"].facts
        if f.id == "F01"
    ]
    results["S-call-007"] = results["S-call-007"].model_copy(update={"facts": facts})

    judge = _FakeJudge("PARAPHRASE")
    score = score_extraction(truth, results, judge=judge)

    assert judge.calls >= 1
    match = next(m for m in score.matched if m.fact_id == "F01")
    assert match.via_judge is True
    assert match.credit == 1.0
    assert not any(u.fact_id == "F01" for u in score.unmatched)
    assert not any(u.produced_id == "f1" and u.source_id == "S-call-007" for u in score.unplanted)


def test_judge_does_not_promote_a_miss_verdict_or_a_non_text_failure() -> None:
    truth = _truth()
    results = perfect_results(truth)
    facts = [
        f.model_copy(
            update={
                "claim": "There were some issues with getting supplies to the "
                "hospital on time recently."
            }
        )
        if f.id == "F01"
        else f
        for f in results["S-call-007"].facts
    ]
    results["S-call-007"] = results["S-call-007"].model_copy(update={"facts": facts})

    # A genuine MISS verdict leaves the fact unmatched.
    score_miss = score_extraction(truth, results, judge=_FakeJudge("MISS"))
    assert not any(m.fact_id == "F01" for m in score_miss.matched)
    assert any(u.fact_id == "F01" for u in score_miss.unmatched)

    # Dropping the fact entirely (source produces nothing to compare
    # against) is never eligible, even with a PARAPHRASE-happy judge.
    dropped = lossy_results(truth, drop=1, seed=0)
    score_dropped = score_extraction(truth, dropped, judge=_FakeJudge("PARAPHRASE"))
    planted_total = len(truth.planted_fact_ids())  # 73 as of truth-table v1.1 (2026-09-13)
    assert score_dropped.matched_count == planted_total - 1


def test_run_extraction_concurrency_matches_sequential_with_fake_backend() -> None:
    """Step 4: `--concurrency N` must produce the same scored result as the
    sequential path. `FakeBackend` has no per-instance state to race on, so
    this proves the thread-pool wiring (source dispatch, result collection,
    `source_texts` capture) rather than backend-per-worker construction,
    which the CLI (`--backend agent_sdk`) exercises via `backend_factory`
    instead."""
    truth = _truth()
    canned = perfect_results(truth)
    source_ids = ["S-call-005", "S-call-007", "S-call-009", "S-ticket-019", "S-ticket-031"]

    sequential = run_extraction(
        truth.manifest,
        FakeBackend(canned),
        source_ids=source_ids,
        segment=True,
        corpus_root=CORPUS_ROOT,
        concurrency=1,
    )
    concurrent = run_extraction(
        truth.manifest,
        FakeBackend(canned),
        source_ids=source_ids,
        segment=True,
        corpus_root=CORPUS_ROOT,
        concurrency=4,
    )

    assert set(concurrent) == set(sequential) == set(source_ids)
    assert concurrent.source_texts == sequential.source_texts
    for source_id in source_ids:
        assert concurrent[source_id].facts == sequential[source_id].facts

    sequential_score = score_extraction(truth, sequential)
    concurrent_score = score_extraction(truth, concurrent)
    assert concurrent_score.recall == sequential_score.recall
    assert concurrent_score.precision == sequential_score.precision


# --------------------------------------------------------------------------
# Fix wave 2026-09-12 (OpenRouter bench telemetry bug): every OpenRouter
# bench report showed 0 tokens / $0.00 cost even though real calls
# happened -- `LLMClientBackend.extract()` used to ignore the `sink`/
# `tenant` `run_extraction` passed it (see backends.py's own docstring),
# so `OpenRouterClient` always metered into whatever sink it was
# constructed with (Postgres in production via `get_llm()`), never into
# `run_extraction`'s own `InMemorySink` that `score_extraction` reads
# tokens/cost back off of. This is the end-to-end proof, through
# `run_extraction` -> `LLMClientBackend` -> a client -- not a real
# `OpenRouterClient` (no network in tests), but anything shaped like one:
# `.complete()` that meters itself, plus the optional `using_sink()`
# extension `portolan.llm.base` documents.
# --------------------------------------------------------------------------


class _ReroutableLLMClient:
    """Mirrors `tests/test_llm_backend.py`'s `_ReroutableFakeClient`:
    meters into `self.sink`/`self.tenant` by default (like
    `OpenRouterClient` metering into whatever sink/tenant `get_llm()`
    constructed it with), and `using_sink()` temporarily reroutes both --
    the same contract `OpenRouterClient.using_sink` implements for real."""

    def __init__(self, canned: dict, *, model_key: str = "nova-lite") -> None:
        self._canned = canned
        self.model_key = model_key
        self.sink = InMemorySink()  # the client's own "default" sink
        self.tenant = "client-default-tenant"

    def complete(self, *, system, user, json_schema, model, surface, max_tokens=4096):
        del system, user, json_schema, model, max_tokens
        with metered(self.sink, self.tenant, surface, self.model_key) as usage:
            usage.input_tokens = 1234
            usage.output_tokens = 567
        return LLMResult(
            text=None,
            data=self._canned,
            input_tokens=1234,
            output_tokens=567,
            cached_tokens=0,
            model=self.model_key,
        )

    @contextmanager
    def using_sink(self, sink, tenant):
        previous_sink, previous_tenant = self.sink, self.tenant
        self.sink, self.tenant = sink, tenant
        try:
            yield
        finally:
            self.sink, self.tenant = previous_sink, previous_tenant


def test_run_extraction_rolls_up_real_tokens_and_cost_through_using_sink() -> None:
    truth = _truth()
    canned = perfect_results(truth)
    source_id = "S-call-009"
    canned_dict = canned[source_id].model_dump(mode="json")

    client = _ReroutableLLMClient(canned_dict)
    backend = LLMClientBackend(client)

    run_result = run_extraction(
        truth.manifest,
        backend,
        source_ids=[source_id],
        segment=False,
        corpus_root=CORPUS_ROOT,
    )

    assert set(run_result) == {source_id}
    # The BENCH's own sink (run_result.sink, what score_extraction reads)
    # saw the usage -- not the client's own default sink.
    assert len(run_result.sink.events) == 1
    assert run_result.sink.events[0].tenant != "client-default-tenant"
    assert len(client.sink.events) == 0

    score = score_extraction(truth, run_result)
    assert score.tokens_input == 1234
    assert score.tokens_output == 567
    assert score.cost_estimate > 0.0


def test_run_extraction_concurrency_uses_a_fresh_backend_per_worker() -> None:
    """`backend_factory` (what the CLI passes for a live backend) is called
    once per worker thread, never once per source -- proven here by a
    counting factory shared across a thread pool."""
    import threading

    truth = _truth()
    canned = perfect_results(truth)
    source_ids = ["S-ticket-019", "S-ticket-024", "S-ticket-027", "S-ticket-031", "S-ticket-038"]
    build_count = 0
    lock = threading.Lock()

    def factory() -> FakeBackend:
        nonlocal build_count
        with lock:
            build_count += 1
        return FakeBackend(canned)

    run_result = run_extraction(
        truth.manifest,
        FakeBackend(canned),  # unused when backend_factory is given
        source_ids=source_ids,
        segment=True,
        corpus_root=CORPUS_ROOT,
        concurrency=3,
        backend_factory=factory,
    )

    assert set(run_result) == set(source_ids)
    # At most one backend per worker thread, and at least one (some source
    # had to run) -- never one per source (5 sources, <=3 workers).
    assert 1 <= build_count <= 3


# --------------------------------------------------------------------------
# `_evidence_basis` / `_plausible_in_source` -- 2026-09-18 invented-
# classifier fix (`01-spec/model-bench-2026-09-18.md` "Corrected scoring").
#
# The old `_plausible_in_source` returned `False` (invented) the instant a
# *given* quote failed exact substring containment, never trying the
# fragment fallback a no-quote fact always got. A near-verbatim quote (one
# changed word, punctuation, a curly apostrophe, an ellipsis, a dropped
# filler word) routinely fails exact containment even though the fact
# plainly has a textual basis. These are small, hand-built fixtures for
# each path `_evidence_basis` can now take, independent of the full
# `score_extraction` pipeline (`test_full_pipeline_...` below covers that
# separately, against real corpus text).
# --------------------------------------------------------------------------

_SOURCE_TEXT = (
    "[00:01:00] Sam Okafor: We can provide packing information and a "
    "delivery schedule.\n"
    "[00:01:05] Derek Hollis: Include the dimensions and weight of the "
    "shipping cases.\n"
    "[00:01:08] Sam Okafor: I will.\n"
    "[00:01:10] Derek Hollis: And whether the cases can be reused for "
    "returns.\n"
    "[00:01:15] Sam Okafor: We serve Ashgrove Medical Center and shipped "
    "40 pumps in March 2026.\n"
)


def test_evidence_basis_exact_quote() -> None:
    quote = "Include the dimensions and weight of the shipping cases."
    assert _quote_supported_by_text(quote, _SOURCE_TEXT) is True
    assert _evidence_basis("irrelevant claim text", quote, _SOURCE_TEXT) == "exact_quote"
    assert _plausible_in_source("irrelevant claim text", quote, _SOURCE_TEXT) is True


def test_evidence_basis_curly_apostrophe_still_counts_as_exact() -> None:
    """A quote that differs from the source ONLY by apostrophe glyph style
    (straight `'` vs. the source's curly `'`) must not be penalized at
    all -- glyph-folding makes this an exact match, not merely a fuzzy
    one."""
    text_with_curly = (
        "[09:12:11] Nina Vale: I’ll get back to you once I’ve checked with the team.\n"
    )
    quote_with_straight = "I'll get back to you once I've checked with the team."
    assert _quote_supported_by_text(quote_with_straight, text_with_curly) is True
    assert _evidence_basis("claim", quote_with_straight, text_with_curly) == "exact_quote"


def test_evidence_basis_one_word_changed_falls_back_to_fuzzy_quote() -> None:
    """The extractor's quote swaps "checked" for "verified" -- fails exact
    containment outright, but every other token lines up in order, so the
    fuzzy signal should catch it rather than the fact being labeled
    invented."""
    text = "[09:12:11] Nina Vale: I’ll get back to you once I’ve checked with the team.\n"
    quote = "I'll get back to you once I've verified with the team."
    assert _quote_supported_by_text(quote, text) is False
    assert _fuzzy_quote_containment(quote, text) >= 0.75
    assert _evidence_basis("claim", quote, text) == "fuzzy_quote"
    assert _plausible_in_source("claim", quote, text) is True


def test_evidence_basis_dropped_filler_word_falls_back_to_fuzzy_quote() -> None:
    """The extractor drops "just" -- a filler word -- from an otherwise
    verbatim quote."""
    text = "[00:02:00] Priya Natarajan: I was just checking in on the delivery schedule.\n"
    quote = "I was checking in on the delivery schedule."
    assert _quote_supported_by_text(quote, text) is False
    assert _evidence_basis("claim", quote, text) == "fuzzy_quote"


def test_evidence_basis_ellipsis_bridged_quote_falls_back_to_fuzzy_quote() -> None:
    """The extractor bridges two non-adjacent turns from `_SOURCE_TEXT`
    with an ellipsis (Derek's two lines, with Sam's "I will." elided) --
    exactly the merged-turn shape a real extractor produces when quoting
    a back-and-forth exchange as one claim's evidence."""
    quote = (
        "Include the dimensions and weight of the shipping cases... "
        "And whether the cases can be reused for returns."
    )
    assert _quote_supported_by_text(quote, _SOURCE_TEXT) is False
    assert _evidence_basis("claim", quote, _SOURCE_TEXT) == "fuzzy_quote"


def test_evidence_basis_quote_fails_both_containment_checks_falls_through_to_fragments() -> None:
    """The regression this fix targets: a quote is GIVEN, it fails both
    exact and fuzzy containment (unrelated wording entirely), but the
    claim's own salient fragments (the account name, the count, the date)
    are genuinely in the source. The old code returned `False` the moment
    the quote failed -- it never reached the fragment fallback at all."""
    claim = "Ashgrove Medical Center received 40 pumps in March 2026."
    quote = "We are extremely satisfied with the onboarding experience overall."
    assert _quote_supported_by_text(quote, _SOURCE_TEXT) is False
    assert _fuzzy_quote_containment(quote, _SOURCE_TEXT) < 0.75
    assert _fragments_supported(claim, _SOURCE_TEXT) is True
    assert _evidence_basis(claim, quote, _SOURCE_TEXT) == "fragments"
    assert _plausible_in_source(claim, quote, _SOURCE_TEXT) is True


def test_evidence_basis_quote_and_fragments_both_fail_is_invented() -> None:
    """Every signal fails: a genuine invention, quote included."""
    claim = "Meridian Health signed a ten-year exclusivity deal worth $9 million."
    quote = "We are thrilled to announce this exclusivity agreement."
    assert _evidence_basis(claim, quote, _SOURCE_TEXT) == "none"
    assert _plausible_in_source(claim, quote, _SOURCE_TEXT) is False


def test_evidence_basis_no_quote_still_uses_fragment_fallback() -> None:
    """Unchanged pre-existing behavior: a fact with no quote at all (never
    touched by this fix) still gets the fragment fallback."""
    claim = "Ashgrove Medical Center received 40 pumps in March 2026."
    assert _evidence_basis(claim, None, _SOURCE_TEXT) == "fragments"
    assert _evidence_basis(claim, "", _SOURCE_TEXT) == "fragments"


def test_evidence_basis_no_quote_no_fragments_is_invented() -> None:
    claim = "Meridian Health signed a ten-year exclusivity deal worth $9 million."
    assert _evidence_basis(claim, None, _SOURCE_TEXT) == "none"
    assert _plausible_in_source(claim, None, _SOURCE_TEXT) is False


def test_evidence_basis_empty_source_text_is_always_none() -> None:
    assert _evidence_basis("claim", "some quote", "") == "none"
    assert _plausible_in_source("claim", "some quote", "") is False


# ---------------------------------------------------------------------------
# 2026-09-18 sanity audit (`01-spec/sanity-audit-2026-09-18.md`,
# `01-spec/opus-invented-audit.md`): every one of Opus's 51, Sonnet's 16,
# and Haiku's 8 `invented`-labeled facts was read against its source text.
# ALL 75 turned out to be scorer false positives, not fabrications, falling
# into a handful of mechanical classes -- one test fixture per class below,
# each drawn directly from the audit (claim/source text paraphrased down to
# the minimum that reproduces the failure, not retyped verbatim). Plus two
# constructed real-fabrication controls that must stay `invented` under
# every one of these new signals combined -- the audit found no real
# fabrication among the 75 to use as a fixture, so these are built.
# ---------------------------------------------------------------------------


def test_evidence_basis_account_name_never_spoken_credited_via_registry_context() -> None:
    """Class (a): "account name not spoken in transcript." A call
    transcript between a rep and a customer contact routinely never says
    the customer's own account name out loud (`01-spec/extraction-miss-
    analysis-2026-09-18.md`) -- but the account name is exactly what
    `_registry_context_for`/`_build_envelope` already told the model up
    front for this source, real context, not an invention. Modeled on
    Opus's `S-call-004`/`f1` finding (`01-spec/sanity-audit-2026-09-18.md`)."""
    text = (
        "[00:22:01] Sam Okafor: What is your decision process after the "
        "demonstration?\n"
        "[00:22:04] Derek Hollis: I gather feedback, write a "
        "recommendation, and take it through our capital review.\n"
    )
    claim = (
        "Northlake Health System's purchase process requires Derek Hollis "
        "to gather feedback and write a recommendation that goes through a "
        "capital review."
    )
    # Without context, the fragment check alone can't credit the account
    # name (it isn't in the raw text at all) -- whether or not some OTHER
    # signal (the window fallback, tested separately below) happens to
    # also cover this particular pair is not what this test is about.
    no_context = EvidenceContext()
    assert _fragments_supported(claim, text, no_context) is False

    with_context = EvidenceContext(context_strings=frozenset({"northlake health system"}))
    assert _fragments_supported(claim, text, with_context) is True
    assert _evidence_basis(claim, None, text, with_context) == "fragments"
    assert _plausible_in_source(claim, None, text, with_context) is True


def test_evidence_basis_supplier_own_name_credited_via_context() -> None:
    """Same class, the supplier's OWN name rather than the customer's: a
    rep almost never says their own employer's name to the customer either.
    Modeled on Sonnet's `S-call-004`/`f29` finding."""
    text = (
        "[00:27:28] Sam Okafor: We provide product-change notices when a "
        "change affects function, compatibility, service, or ordering.\n"
        "[00:27:35] Derek Hollis: That needs to be in the contract or "
        "support terms.\n"
    )
    claim = (
        "Derek Hollis requested that Halvard Medical's product-change "
        "notice policy be documented in the contract or support terms for "
        "Northlake Health System."
    )
    context = EvidenceContext(
        context_strings=frozenset({"halvard medical", "northlake health system"})
    )
    assert _fragments_supported(claim, text, EvidenceContext()) is False
    assert _fragments_supported(claim, text, context) is True
    assert _evidence_basis(claim, None, text, context) == "fragments"


def test_evidence_basis_crm_as_of_date_credited_via_envelope_source_date() -> None:
    """Class (b)/date: a CRM row (`accounts.csv`) has no date column at
    all -- the "as of <date>" a produced fact states can only be the
    source's own envelope `source_date`, never the row text. Modeled on
    Opus's `S-crm-A1010`/`f1` finding: the CRM row text carries no date,
    but the envelope told the model this source is dated 2026-09-11."""
    text = (
        "[account:A-1010] crm_id=A-1010, name=Ashgrove Health, "
        "kind=system, parent_crm_id=, domain=ashgrovehealth.org, "
        "arr=2350000, region=Midwest, status="
    )
    claim = (
        "Ashgrove Health (CRM account A-1010) carries annual recurring "
        "revenue of $2,350,000 as of 2026-09-11."
    )
    no_date_context = EvidenceContext(context_strings=frozenset({"ashgrove health"}))
    dated_context = EvidenceContext(
        context_strings=frozenset({"ashgrove health"}),
        date_tokens=frozenset({"2026", "09", "11"}),
    )
    assert _fragments_supported(claim, text, no_date_context) is False
    assert _fragments_supported(claim, text, dated_context) is True
    assert _evidence_basis(claim, None, text, dated_context) == "fragments"


def test_evidence_basis_trailing_sentence_punctuation_on_a_number_fragment() -> None:
    """Class (c): `_NUMBER_LITERAL_RE` includes `.`/`,` in its character
    class, so a number at the very end of a claim sentence ("...as of
    2026-08-21.") picks up the period as part of the extracted fragment --
    a fragment that then never matches the source's own bare digits.
    Isolated from the registry-context/date-token classes above: this
    fragment is the KIND of trailing-punctuation artifact fixed by
    `_fragment_grounding`'s signal (2), with no context needed at all."""
    text = "export_date=2026-08-21"
    claim = "The account snapshot was exported on 2026-08-21."
    assert "21." in _salient_claim_fragments(claim)  # the trailing-period artifact
    assert _fragments_supported(claim, text, EvidenceContext()) is True
    assert _evidence_basis(claim, None, text, EvidenceContext()) == "fragments"


def test_evidence_basis_currency_comma_formatting_vs_bare_digits() -> None:
    """Class (d): a claim's own durable-language figure ("$2,350,000")
    against a CRM row's bare digits ("arr=2350000") -- the same number,
    formatted two different ways. `_fragment_grounding`'s digit-only comparison
    (signal (4), 3+ digits to avoid a coincidental short-run match) closes
    this without touching the currency-insensitive claim-similarity scorer
    at all."""
    text = "arr=2350000"
    claim = "This account's ARR is $2,350,000."
    assert _fragments_supported(claim, text, EvidenceContext()) is True


def test_evidence_basis_case_insensitive_abbreviation_containment() -> None:
    """Class (e), found rescoring a non-Claude model's saved output during
    this audit (not one of the 75 Claude facts, but the same mechanism):
    a produced claim's "CRM" (a proper-noun-shaped, all-caps abbreviation)
    against a row's own lowercase `crm_id=...` field never matched the
    ORIGINAL case-sensitive literal check even though the row is
    unmistakably a CRM record."""
    text = "[account:A-1080] crm_id=A-1080, name=Kestrel Medical Group, arr=0"
    claim = "Kestrel Medical Group has an ARR of zero recorded as of the CRM record."
    assert _fragments_supported(claim, text, EvidenceContext()) is True
    assert _evidence_basis(claim, None, text, EvidenceContext()) == "fragments"


def test_window_supported_credits_a_claim_with_no_salient_fragments() -> None:
    """Fix instructions item 3's third bullet: a purely descriptive claim
    (no number, no proper noun -- `_salient_claim_fragments` returns
    empty, and `_fragments_supported` returns False outright with nothing
    to check) can still be genuinely grounded. Modeled on the audit's
    `S-call-014`/`f13` finding (content-word overlap against the best
    3-sentence window well above `_WINDOW_CONTENT_THRESHOLD`)."""
    text = (
        "[09:33:18] Aisha Bell: The internal operating group will make the "
        "recommendation, with input from clinical engineering, the "
        "department, and procurement. "
        "[09:33:25] Sam Okafor: Does senior leadership approve the final "
        "recommendation? "
        "[09:33:30] Aisha Bell: They approve the business decision, but the "
        "operating group needs to produce a clear recommendation first.\n"
    )
    claim = (
        "Ultimately, the operating group will produce the recommendation, "
        "with input from clinical engineering, the department, and "
        "procurement, while senior leadership approves the business "
        "decision."
    )
    assert _salient_claim_fragments(claim) == {"Ultimately"}  # one, unrelated word
    assert _fragments_supported(claim, text, EvidenceContext()) is False
    assert _window_supported(claim, text) is True
    assert _evidence_basis(claim, None, text, EvidenceContext()) == "window"


def test_window_supported_rejects_dense_unrelated_text() -> None:
    """The window fallback must NOT fire just because a window shares a
    handful of common words with the claim."""
    text = (
        "[09:00:00] Sam Okafor: Let's talk about the delivery schedule for "
        "next quarter and who owns the follow-up.\n"
    )
    claim = "The board approved a merger with a competitor next quarter."
    assert _window_supported(claim, text) is False


# ---------------------------------------------------------------------------
# Follow-up fix, same day (`01-spec/sanity-audit-2026-09-18.md` "closing the
# vacuous-fragment-vote hole"): spot-checking Nova Lite's 24 flipped facts
# found the fragment vote can pass with ZERO words of the claim ever
# checked against the source text -- when the claim's only salient
# fragments are names (`EvidenceContext.context_strings`) and it has no
# number at all, every fragment is satisfied by context alone. A soft
# fabrication ("<real account> is unhappy with support") would pass.
# ---------------------------------------------------------------------------


def test_fragment_vote_context_only_account_name_is_not_text_grounded() -> None:
    """The building block the guard is built on: a claim whose only
    fragment is a real account name, absent from the text, passes the raw
    60% vote (fragments_ok) but `text_grounded` is False -- nothing about
    the claim was ever checked against the actual source."""
    text = "[09:00:00] Sam Okafor: Let's review the open items for this week.\n"
    claim = "Ashgrove Health is unhappy with the current level of support."
    context = EvidenceContext(context_strings=frozenset({"ashgrove health"}))
    fragments_ok, text_grounded = _fragment_vote(claim, text, context)
    assert fragments_ok is True  # the account name alone satisfies the 60% vote
    assert text_grounded is False  # but it's the ONLY fragment, and it's context-only


def test_evidence_basis_rejects_constructed_soft_fabrication_context_only_account() -> None:
    """The exact adversarial control the coordinator's follow-up asked
    for: a real, registered account name (legitimate context), no number,
    a claim never said anywhere in this transcript. Must stay `invented`
    -- the fragment vote alone is not enough once every hit is context-
    only, and nothing here corroborates via the window either."""
    text = (
        "[09:00:00] Sam Okafor: Let's review the open items for this week.\n"
        "[09:00:04] Maria Chen: Sounds good, I only have the contract "
        "renewal question from last time.\n"
        "[09:00:09] Sam Okafor: I'll have an answer by Friday.\n"
    )
    claim = "Ashgrove Health is unhappy with the current level of support."
    context = EvidenceContext(context_strings=frozenset({"ashgrove health"}))
    fragments_ok, text_grounded = _fragment_vote(claim, text, context)
    assert fragments_ok is True
    assert text_grounded is False
    assert _window_supported(claim, text) is False
    assert _evidence_basis(claim, None, text, context) == "none"
    assert _plausible_in_source(claim, None, text, context) is False


def test_evidence_basis_context_only_vote_still_credited_when_window_corroborates() -> None:
    """The other half of the guard: a context-only fragment vote is not
    thrown away outright -- if the window fallback independently agrees
    (most of the claim's content words really do reappear together
    nearby), it's still real grounding, just reported as `window` rather
    than `fragments` (the fragment vote alone didn't earn it). Modeled on
    a real Nova Lite flip, `S-call-004/f14` (`01-spec/sanity-audit-
    2026-09-18.md`): the account name is the only fragment and is never
    spoken, but the rest of the claim is a close paraphrase of one
    contiguous passage."""
    text = (
        "[00:18:04] Derek Hollis: Good.\n"
        "[00:18:06] Sam Okafor: Do your teams need a device-tracking "
        "capability?\n"
        "[00:18:08] Derek Hollis: We track equipment, but not always "
        "elegantly. Some departments use labels, some use spreadsheets, "
        "and our central inventory system is not always updated at the "
        "moment a device moves.\n"
    )
    claim = (
        "Northlake Health System tracks equipment but not always "
        "elegantly, using labels, spreadsheets, and a central inventory "
        "system that is not always updated."
    )
    context = EvidenceContext(context_strings=frozenset({"northlake health system"}))
    fragments_ok, text_grounded = _fragment_vote(claim, text, context)
    assert fragments_ok is True
    assert text_grounded is False  # "Northlake Health System" is the only fragment
    assert _window_supported(claim, text) is True
    assert _evidence_basis(claim, None, text, context) == "window"
    assert _plausible_in_source(claim, None, text, context) is True


def test_window_supported_does_not_bridge_widely_separated_passages() -> None:
    """Known, deliberate limitation, found and reported rather than
    papered over (`01-spec/sanity-audit-2026-09-18.md`): a claim
    synthesizing two DISTANT, non-adjacent turns of the same real call
    (`S-call-n03`, ~70 sentences apart -- "busy winter" at 00:02:10 and
    "more requests than there are comfortable hours" at 00:04:02) cannot
    be credited by ANY single contiguous window, no matter its size, and
    is not credited here either. Widening this to a multi-window union
    (checking against the constructed fabrication above) reopens
    something close to the same vacuous-vote risk this whole fix exists
    to close -- scattered common words across a long transcript would
    start to look like coverage. Left as strict (reverts to `invented`
    under the follow-up guard); flagged, not silently forced to pass."""
    text = (
        "[00:00:40] Sam Okafor: How has the team been? Busy?\n"
        "[00:00:45] Derek Hollis: Busy, but manageable.\n"
        "[00:02:05] Sam Okafor: How are things outside the department? I "
        "know you've had a busy winter.\n"
        "[00:02:10] Derek Hollis: Busy winter is accurate. We had a few "
        "weather-related delays, nothing dramatic, but enough to make "
        "scheduling awkward.\n"
        "[00:03:56] Sam Okafor: What's the mood among your leadership "
        "group? Are people feeling stretched, or is it more a matter of "
        "keeping pace?\n"
        "[00:04:02] Derek Hollis: A little of both. We're keeping pace, "
        "but everyone is watching resources closely. There are more "
        "requests than there are comfortable hours in the week.\n"
    )
    claim = (
        "Larkspur North Hospital's team is feeling stretched due to a "
        "busy winter and more requests than available hours."
    )
    context = EvidenceContext(context_strings=frozenset({"larkspur north hospital"}))
    fragments_ok, text_grounded = _fragment_vote(claim, text, context)
    assert fragments_ok is True
    assert text_grounded is False
    assert _window_supported(claim, text) is False
    assert _evidence_basis(claim, None, text, context) == "none"


def test_evidence_basis_still_credits_a_second_real_nova_flip_via_text_grounded_fragment() -> None:
    """Third named Nova Lite flip (`S-email-n020/f1`): unaffected by the
    guard because "Paul Strand" is a SECOND fragment that is genuinely
    literal in the text (the email's own `From:` header), alongside the
    context-only "Sable Valley Health" -- `text_grounded` is True, so the
    fragment vote is accepted outright, exactly as before this fix."""
    text = (
        "[msg:S-email-n020@halvardmedical.com]\n"
        "From: Paul Strand <pstrand@sablevalley.org>\n"
        "To: Kenji Mori <k.mori@halvardmedical.com>\n"
        "Subject: Calendar invite: biomed refresher training\n"
        "Date: 2026-03-02 09:00:00 -0700\n"
        "[¶1] Hi Kenji,\n"
        "[¶2] Attaching nothing new here, just closing the loop from "
        "our call.\n"
        "[¶3] Best, Paul Strand\n"
    )
    claim = (
        "Paul Strand from Sable Valley Health sent a calendar invite for biomed refresher training."
    )
    context = EvidenceContext(context_strings=frozenset({"sable valley health", "paul strand"}))
    fragments_ok, text_grounded = _fragment_vote(claim, text, context)
    assert fragments_ok is True
    assert text_grounded is True  # "Paul Strand" is literal in the header
    assert _evidence_basis(claim, None, text, context) == "fragments"


def test_evidence_basis_rejects_two_constructed_real_fabrications() -> None:
    """The audit's real-fabrication detector, kept honest: two constructed
    controls (the audit found none among the real 75 to use instead) that
    MUST stay `invented` under every new signal at once -- fragments
    (context-aware), window, and everything else. Both name at least one
    REAL, legitimately-known context string (an account/supplier the model
    really was given) to prove that alone is not enough to launder an
    otherwise invented claim -- the rest of each claim (a competitor, a
    dollar figure, a future date; a dollar figure and a bribery
    allegation) has no source, no context, and no window support."""
    text = (
        "[00:01:01] Derek Hollis: We have several people who need to see "
        "the information. Biomedical engineering, nursing operations, our "
        "interfaces group, supply chain, and finance all have a hand in "
        "this.\n"
    )
    context = EvidenceContext(
        context_strings=frozenset({"northlake health system", "halvard medical", "tidewater"})
    )

    fabrication_1 = (
        "Northlake Health System signed an exclusive five-year agreement "
        "with Meridian BioSystems worth $4.75 million to replace all "
        "Tidewater devices by 2028."
    )
    assert _fragments_supported(fabrication_1, text, context) is False
    assert _window_supported(fabrication_1, text) is False
    assert _evidence_basis(fabrication_1, None, text, context) == "none"
    assert _plausible_in_source(fabrication_1, None, text, context) is False

    fabrication_2 = (
        "An anonymous whistleblower alleged that Halvard Medical bribed a "
        "regional purchasing director with $50,000 to win a sole-source "
        "contract in 2029."
    )
    assert _fragments_supported(fabrication_2, text, context) is False
    assert _window_supported(fabrication_2, text) is False
    assert _evidence_basis(fabrication_2, None, text, context) == "none"
    assert _plausible_in_source(fabrication_2, None, text, context) is False


def test_evidence_context_for_source_reads_registry_vocabulary_and_supplier() -> None:
    """Integration: `evidence_context_for_source` against the REAL corpus
    (not a hand-built `EvidenceContext`) for a known real source pulls in
    that source's own account/contact, the corpus-wide supplier name, and
    the product vocabulary -- exactly what `_build_envelope` put in front
    of the model for `S-call-004` (Northlake, Derek Hollis)."""
    truth = _truth()
    context = evidence_context_for_source("S-call-004", truth)
    assert "northlake health system" in context.context_strings
    assert "derek hollis" in context.context_strings
    assert "halvard medical" in context.context_strings
    assert "tidewater" in context.context_strings
    assert context.date_tokens  # this source has a manifest date


def test_evidence_context_for_source_unknown_source_id_is_empty() -> None:
    truth = _truth()
    assert evidence_context_for_source("S-does-not-exist", truth) == EvidenceContext()


def test_score_extraction_credits_account_name_via_real_truth_end_to_end() -> None:
    """Full pipeline (not just the private helper): an unplanted fact
    naming `S-call-004`'s real account by name, with the account name
    itself absent from the raw call text but a second fragment ("Derek
    Hollis") that IS literally there, is credited `unplanted_plausible`
    (`evidence_basis == "fragments"`) once scored through `score_extraction`
    with the REAL `Truth` -- proving `EvidenceContext` is actually wired
    into the call site, not just reachable as a helper function. Needs a
    text-grounded fragment alongside the context-grounded one: a claim
    whose ONLY fragments are context-only doesn't clear the fragment vote
    alone any more (`test_evidence_basis_context_only_fragment_vote_...`
    below) -- that is not what this particular test is checking."""
    truth = _truth()
    fact = Fact(
        id="f-test",
        type=FactType.OTHER,
        claim=(
            "Northlake Health System's purchase process requires Derek "
            "Hollis to gather feedback and write a recommendation that "
            "goes through a capital review."
        ),
        entities=["Northlake Health System"],
        locator="00:22:04",
        quote=None,
        confidence=0.9,
        sentiment=0,
    )
    text = (
        "[00:22:01] Sam Okafor: What is your decision process after the "
        "demonstration?\n"
        "[00:22:04] Derek Hollis: I gather feedback, write a "
        "recommendation, and take it through our capital review.\n"
    )
    result = ExtractionResult(
        source_id="S-call-004",
        summary="",
        entities=[],
        facts=[fact],
        actions=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    run_result = ExtractionRunResult(
        {"S-call-004": result},
        wall_time_s={},
        sink=InMemorySink(),
        total_wall_time_s=0.0,
        source_texts={"S-call-004": text},
    )
    score = score_extraction(truth, run_result)
    unplanted_for_source = [u for u in score.unplanted if u.source_id == "S-call-004"]
    matching = [u for u in unplanted_for_source if u.produced_id == "f-test"]
    assert len(matching) == 1
    assert matching[0].evidence_basis == "fragments"
    assert matching[0].classification == "unplanted_plausible"


def test_full_pipeline_one_word_changed_quote_is_no_longer_invented() -> None:
    """End-to-end regression test through `run_extraction` +
    `score_extraction` (not just the private helper): a produced fact
    whose quote changes one word from the real S-call-002 line, and swaps
    the source's curly apostrophes for straight ones, must score
    `evidence_basis == "fuzzy_quote"` and `in_source_text is True` -- the
    exact shape of fact 2026-09-18 found mislabeled `invented`."""
    truth = _truth()
    near_verbatim = Fact(
        id="f1",
        type=FactType.COMMITMENT,
        claim=(
            "Nina Vale committed to get back to Priya Natarajan once she verified with her team."
        ),
        entities=[],
        locator="09:12:11",
        quote="I'll get back to you once I've verified with the team.",
        speaker="customer",
        sentiment=0,
        confidence=0.7,
    )
    canned = ExtractionResult(
        source_id="S-call-002",
        summary="fixture",
        entities=[],
        facts=[near_verbatim],
        actions=[],
        unknown_entities=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    backend = FakeBackend({"S-call-002": canned})

    run_result = run_extraction(
        truth.manifest,
        backend,
        source_ids=["S-call-002"],
        segment=False,
        corpus_root=CORPUS_ROOT,
    )
    score = score_extraction(truth, run_result)

    by_id = {u.produced_id: u for u in score.unplanted if u.source_id == "S-call-002"}
    assert by_id["f1"].evidence_basis == "fuzzy_quote"
    assert by_id["f1"].in_source_text is True
    assert by_id["f1"].classification == "unplanted_plausible"


# --------------------------------------------------------------------------
# 2026-09-18 harness review (01-spec/harness-review-2026-09-18.md):
# lenient recall / near-miss-credited, --samples union, and rescore_full.
# --------------------------------------------------------------------------


def test_perfect_results_lenient_recall_matches_strict() -> None:
    """Lenient never disagrees with strict on a genuinely perfect run --
    it can only credit MORE, never fewer, planted facts."""
    truth = _truth()
    score = score_extraction(truth, perfect_results(truth))
    assert score.lenient_recall == 1.0
    assert score.near_miss_credited_count == 0
    assert score.near_miss_credited_fact_ids == []


def test_wrong_fact_type_is_a_strict_miss_but_a_lenient_near_miss_credit() -> None:
    """A produced fact with the WRONG type (not one of `_NEAR_MATCH_TYPE_
    PAIRS`, not `other`) is a real extractor error and must stay a strict
    miss (matches `_evaluate_pair`'s unchanged type-compatibility rule) --
    but the lenient matcher (type-agnostic) credits it, and it shows up in
    `near_miss_credited_fact_ids`, exactly the F13/F15/F40/F63 pattern
    `01-spec/extraction-miss-analysis-2026-09-18.md` section 3 found for
    Kimi K2 (a `decision` produced where `competitor_mention`/`request`/
    `praise` was planted)."""
    truth = _truth()
    results = perfect_results(truth)
    # F11 is planted as a `decision` (S-crm-opp-A1010 / S-call-003) --
    # retype every produced copy of it to `risk`, a type with no near-
    # match relationship to `decision`. Rebuilt via `model_validate`
    # (rather than a plain attribute assignment) so `Fact.type`'s own
    # validator runs and produces a proper `_FactTypeValue` (review
    # finding F13's `.value`-bearing wrapper), not a bare `str`.
    for result in results.values():
        result.facts = [
            Fact.model_validate({**fact.model_dump(), "type": "risk"}) if fact.id == "F11" else fact
            for fact in result.facts
        ]

    score = score_extraction(truth, results)
    unmatched_ids = {u.fact_id for u in score.unmatched}
    assert "F11" in unmatched_ids
    unmatched_f11 = next(u for u in score.unmatched if u.fact_id == "F11")
    assert "type" in unmatched_f11.failed_criteria

    assert score.recall < 1.0  # strict: still a miss
    assert score.lenient_recall == 1.0  # lenient: type-agnostic, credited
    assert "F11" in score.near_miss_credited_fact_ids
    assert score.near_miss_credited_count == len(score.near_miss_credited_fact_ids)


def test_unresolved_entity_is_a_miss_in_strict_and_lenient() -> None:
    """An entity the lenient matcher can't resolve at all (blanked out
    entirely, not just mistyped) stays a miss under BOTH matchers -- per
    the harness review's matcher decision, entity resolution is a real
    extraction gap, not a scorer-strictness artifact, so lenient does not
    relax it."""
    truth = _truth()
    results = perfect_results(truth)
    for result in results.values():
        for fact in result.facts:
            if fact.id == "F11":
                fact.entities = ["definitely-not-a-real-entity-id"]

    score = score_extraction(truth, results)
    assert "F11" not in {m.fact_id for m in score.matched}
    assert "F11" not in score.near_miss_credited_fact_ids
    assert score.lenient_recall < 1.0


def test_samples_unions_facts_across_independent_extraction_passes() -> None:
    """`run_extraction(..., samples=3)` runs three independent passes per
    source and unions the facts (`merge_segment_results`'s existing
    dedup): a source whose extractor finds fact A on pass 1 and a
    DIFFERENT fact B on pass 2 ends up with both, not just the last pass's
    output."""
    truth = _truth()
    source_id = "S-ticket-019"  # single planted fact, F13
    calls = {"n": 0}

    def canned(envelope):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            claim = "St. Aldric Health reported a pricing complaint about the HalvardCare invoice."
            fact = Fact(
                id="a",
                type=FactType.PRICING,
                claim=claim,
                entities=[],
                locator="S-ticket-019#1",
                quote="a duplicate line item",  # PRICING is QUOTE_REQUIRED
                speaker="customer",
                sentiment=0,
                confidence=0.8,
            )
        else:
            fact = Fact(
                id="b",
                type=FactType.OTHER,
                claim="A completely unrelated second fact never seen on other passes.",
                entities=[],
                locator="S-ticket-019#1",
                quote=None,
                speaker="customer",
                sentiment=0,
                confidence=0.8,
            )
        return ExtractionResult(
            source_id=envelope.source_id,
            summary="",
            entities=[],
            facts=[fact],
            actions=[],
            coverage=Coverage(segments_read=1, segments_total=1),
        )

    backend = FakeBackend(canned)

    run_1 = run_extraction(
        truth.manifest, backend, source_ids=[source_id], segment=False, samples=1
    )
    assert len(run_1[source_id].facts) == 1

    calls["n"] = 0
    run_3 = run_extraction(
        truth.manifest, backend, source_ids=[source_id], segment=False, samples=3
    )
    claims = {f.claim for f in run_3[source_id].facts}
    # Three passes alternate A, B, A -- the union has both distinct facts,
    # not three (dedup) and not one (last-write-wins).
    assert len(run_3[source_id].facts) == 2
    assert any("pricing complaint" in c for c in claims)
    assert any("unrelated second fact" in c for c in claims)


def test_samples_defaults_to_one_pass_unchanged_behavior() -> None:
    truth = _truth()
    source_ids = ["S-ticket-019"]
    backend = FakeBackend(
        {sid: _r for sid, _r in perfect_results(truth).items() if sid in source_ids}
    )
    run_default = run_extraction(truth.manifest, backend, source_ids=source_ids, segment=False)
    run_explicit = run_extraction(
        truth.manifest, backend, source_ids=source_ids, segment=False, samples=1
    )
    assert len(run_default[source_ids[0]].facts) == len(run_explicit[source_ids[0]].facts)


def test_rescore_full_round_trips_a_freshly_scored_report() -> None:
    """`rescore_full`, fed a report `score_extraction` just produced
    (nothing to fix), reproduces the same strict recall, lenient recall,
    and matched/unmatched/unplanted counts -- proves the reconstruction
    (`_fact_from_record` -> `ExtractionResult` -> `score_extraction`) is
    faithful, not just "runs without crashing"."""
    truth = _truth()
    original = score_extraction(truth, lossy_results(truth, drop=3, invent=2, seed=2))

    rescored = rescore_full(original, truth.manifest, truth, corpus_root=CORPUS_ROOT)

    assert rescored.recall == original.recall
    assert rescored.lenient_recall == original.lenient_recall
    assert rescored.matched_count == original.matched_count
    assert len(rescored.unmatched) == len(original.unmatched)
    assert len(rescored.unplanted) == len(original.unplanted)
    # Carried forward, not recomputed (a saved report has no raw usage
    # events or entity mentions to recompute them from).
    assert rescored.tokens_input == original.tokens_input
    assert rescored.entity_accuracy == original.entity_accuracy


def test_rescore_full_picks_up_the_rss_locator_fix_from_saved_output() -> None:
    """The scenario `rescore_full` exists for: F29 (S-rss-006) is planted
    with locator `url:<link> 2026-09-02`; a produced fact carrying just the
    url (no date -- exactly what `extract/prompt.py`'s own instructions
    tell a model to emit for rss, `01-spec/harness-review-2026-09-18.md`)
    failed the OLD `planted.lower() == produced.lower()` rule. Build an
    `ExtractionScore` that looks like a saved report from that old rule
    (this fact sitting in `unmatched`, matching nothing) and confirm
    `rescore_full` -- which re-derives everything from `produced_by_source`
    against the CURRENT `portolan.bench.locators` rule -- recovers it as a
    match, with recall going up, no model call."""
    truth = _truth()
    source_id = "S-rss-006"
    url_only_locator = "https://feeds.example-newswire.test/portolan-corpus/s-rss-006"
    produced = Fact(
        id="a",
        type=FactType.CONTACT_CHANGE,
        claim=(
            "Renata Solis became Chief Financial Officer at Larkspur Regional Health "
            "effective 2026-09-01."
        ),
        entities=["Larkspur Regional Health"],
        locator=url_only_locator,
        quote=None,
        speaker="customer",
        sentiment=0,
        confidence=0.8,
    )
    result = ExtractionResult(
        source_id=source_id,
        summary="",
        entities=[],
        facts=[produced],
        actions=[],
        coverage=Coverage(segments_read=1, segments_total=1),
    )
    saved_like = score_extraction(truth, {source_id: result})

    # With the CURRENT (fixed) locator rule already in effect, this
    # url-only produced locator matches straightaway -- score_extraction
    # itself proves the fix is live.
    assert saved_like.recall == 1.0
    assert "F29" in {m.fact_id for m in saved_like.matched}

    # rescore_full from the same saved produced_by_source reproduces it
    # (no regression on the reconstruction round trip).
    rescored = rescore_full(saved_like, truth.manifest, truth, corpus_root=CORPUS_ROOT)
    assert rescored.recall == 1.0
    assert "F29" in {m.fact_id for m in rescored.matched}


def test_score_keeps_every_entity_mention_for_later_reading() -> None:
    truth = _truth()
    mentions = [
        EntityMention(
            mention="Ashgrove Health",
            kind=EntityKind.ACCOUNT,
            registry_id="ashgrove",
            confidence=0.9,
        ),
        EntityMention(mention="some vendor", kind=EntityKind.OTHER, confidence=0.4),
    ]
    results = {
        sid: r.model_copy(update={"entities": []}) for sid, r in perfect_results(truth).items()
    }
    results["S-call-003"] = results["S-call-003"].model_copy(update={"entities": mentions})
    score = score_extraction(truth, results)

    kept = [(m.source_id, m.mention, m.kind, m.registry_id) for m in score.entity_mentions]
    assert kept == [
        ("S-call-003", "Ashgrove Health", "account", "ashgrove"),
        ("S-call-003", "some vendor", "other", None),
    ]


def test_rescore_full_recomputes_entity_accuracy_from_saved_mentions() -> None:
    truth = _truth()
    results = {
        sid: r.model_copy(update={"entities": []}) for sid, r in perfect_results(truth).items()
    }
    results["S-call-003"] = results["S-call-003"].model_copy(
        update={
            "entities": [
                EntityMention(
                    mention="Ashgrove Health",
                    kind=EntityKind.ACCOUNT,
                    registry_id="ashgrove",
                    confidence=0.9,
                ),
                EntityMention(
                    mention="Vireo", kind=EntityKind.ACCOUNT, registry_id="ashgrove", confidence=0.9
                ),
            ]
        }
    )
    original = score_extraction(truth, results)
    assert (original.entity_accuracy_correct, original.entity_accuracy_checked) == (1, 2)

    # A saved report whose stored rate is stale: the mentions win.
    stale = original.model_copy(update={"entity_accuracy": 0.0, "entity_accuracy_correct": 0})
    rescored = rescore_full(stale, truth.manifest, truth, corpus_root=CORPUS_ROOT)

    assert (rescored.entity_accuracy_correct, rescored.entity_accuracy_checked) == (1, 2)
    assert rescored.entity_accuracy == 0.5
    assert len(rescored.entity_mentions) == 2


def _score_with_mentions(source_id: str, mentions: list[EntityMention]):
    truth = _truth()
    results = {
        sid: r.model_copy(update={"entities": []}) for sid, r in perfect_results(truth).items()
    }
    results[source_id] = results[source_id].model_copy(update={"entities": mentions})
    return score_extraction(truth, results)


def test_internal_person_id_the_bench_hands_out_is_a_correct_resolution() -> None:
    """The bench's own registry context lists an internal person as
    `internal:<id>`; the truth table's id is the bare `<id>`. Returning the
    id exactly as given is right, and was scored wrong."""
    truth = _truth()
    source = truth.manifest.source("S-call-003")
    issued = {
        e.registry_id for e in _registry_context_for(source, truth) if e.kind == "internal_person"
    }
    assert issued == {"internal:sam-okafor"}

    score = _score_with_mentions(
        "S-call-003",
        [
            EntityMention(
                mention="Sam Okafor",
                kind=EntityKind.INTERNAL_PERSON,
                registry_id="internal:sam-okafor",
                confidence=0.9,
            ),
            EntityMention(
                mention="Ashgrove Health",
                kind=EntityKind.ACCOUNT,
                registry_id="ashgrove",
                confidence=0.9,
            ),
        ],
    )

    assert (score.entity_accuracy_correct, score.entity_accuracy_checked) == (2, 2)
    assert score.entity_accuracy == 1.0
    # The old scorer's numerator (raw string equality) stays readable.
    assert score.entity_accuracy_exact_id_correct == 1
    assert score.entity_wrong == []


def test_a_wrong_entity_is_still_wrong_and_is_listed() -> None:
    score = _score_with_mentions(
        "S-call-003",
        [
            # Another internal person's id: the prefix is no free pass.
            EntityMention(
                mention="Sam Okafor",
                kind=EntityKind.INTERNAL_PERSON,
                registry_id="internal:priya-natarajan",
                confidence=0.9,
            ),
            # A prefix the bench never hands out.
            EntityMention(
                mention="Ashgrove Health",
                kind=EntityKind.ACCOUNT,
                registry_id="account:ashgrove",
                confidence=0.9,
            ),
            # A child account is not its parent.
            EntityMention(
                mention="Ashgrove Health",
                kind=EntityKind.ACCOUNT,
                registry_id="ashgrove-medical",
                confidence=0.9,
            ),
        ],
    )

    assert (score.entity_accuracy_correct, score.entity_accuracy_checked) == (0, 3)
    assert score.entity_accuracy_exact_id_correct == 0
    assert [(w.mention, w.registry_id, w.expected) for w in score.entity_wrong] == [
        ("Sam Okafor", "internal:priya-natarajan", "sam-okafor"),
        ("Ashgrove Health", "account:ashgrove", "ashgrove"),
        ("Ashgrove Health", "ashgrove-medical", "ashgrove"),
    ]


def test_a_name_borrowed_from_the_prompts_examples_is_counted() -> None:
    """The prompt's worked examples name a made-up vendor. A fact that
    carries that name into a claim about a source that never says it has
    the wrong company in it, quote or no quote."""
    truth = _truth()
    results = dict(perfect_results(truth))
    result = results["S-call-003"]
    borrowed = result.facts[0].model_copy(
        update={"id": "fx", "claim": "Ashgrove Health asked Veltrona for monthly account notes."}
    )
    in_entities = result.facts[0].model_copy(
        update={"id": "fy", "claim": "Ashgrove Health wants a recap.", "entities": ["Rillfold R4"]}
    )
    results["S-call-003"] = result.model_copy(
        update={"facts": [*result.facts, borrowed, in_entities]}
    )

    score = score_extraction(truth, results)

    assert score.example_name_leaks == ["S-call-003:fx: Veltrona", "S-call-003:fy: Rillfold"]
    assert score_extraction(truth, perfect_results(truth)).example_name_leaks == []


def test_an_example_name_the_source_itself_says_is_not_a_leak() -> None:
    truth = _truth()
    results = dict(perfect_results(truth))
    result = results["S-call-003"]
    fact = result.facts[0].model_copy(
        update={"id": "fx", "claim": "Ashgrove Health asked Veltrona for monthly account notes."}
    )
    results["S-call-003"] = result.model_copy(update={"facts": [*result.facts, fact]})
    run = ExtractionRunResult(
        results,
        wall_time_s={},
        sink=InMemorySink(),
        total_wall_time_s=0.0,
        source_texts={"S-call-003": "[09:10:02] Maria Chen: we also buy from Veltrona."},
    )

    assert score_extraction(truth, run).example_name_leaks == []
