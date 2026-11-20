"""Tests for `portolan.bench.judge` -- the layered evidence/match judge.

Covers, per the build instructions: cache key stability, human override,
the sentence-check downgrade (including the 2026-09-18 calibration fix --
a claim's context-only account/product name must not count against the
string check), both `--judge`/`--judge-matches` flags (`off`/`cached`/
`live`), with fixtures (the real corpus, same as `test_bench_extraction.py`,
plus a `FakeJudge` double so no real model call is made in a test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from portolan.bench.extraction import (
    EvidenceContext,
    ExtractionScore,
    ProducedFactRecord,
    UnmatchedFact,
    UnplantedFact,
)
from portolan.bench.judge import (
    VerdictStore,
    apply_evidence_judge,
    apply_match_judge,
    evidence_cache_key,
    match_cache_key,
    resolve_evidence_verdict,
    resolve_match_verdict,
    select_sentence_window,
    sentence_check,
    split_sentences,
)
from portolan.bench.truth import Truth

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"
MANIFEST_PATH = REPO_ROOT / "corpus" / "out" / "manifest.json"
CORPUS_ROOT = REPO_ROOT / "corpus" / "out"


def _truth() -> Truth:
    return Truth.load(TRUTH_TABLE_PATH, MANIFEST_PATH)


def _score(**overrides: Any) -> ExtractionScore:
    base: dict[str, Any] = dict(
        sources_scored=1,
        planted_total=1,
        matched_count=0,
        produced_total=1,
        recall=0.0,
        precision=None,
        distractor_leakage=0,
        quote_presence_rate=None,
        locator_presence_rate=1.0,
        entity_accuracy=None,
        entity_accuracy_checked=0,
    )
    base.update(overrides)
    return ExtractionScore(**base)


class FakeJudge:
    """Duck-typed stand-in for `judge.Judge` -- no real model call, no
    `AgentSDKClient`/CLI subprocess. `apply_evidence_judge`/`apply_match_
    judge` only ever call `.judge_evidence_raw`/`.judge_match_raw` and
    read `.model`; that's all this needs to provide."""

    def __init__(self, evidence_raw: dict | None = None, match_raw: dict | None = None) -> None:
        self.model = "fake-judge"
        self.evidence_raw = evidence_raw
        self.match_raw = match_raw
        self.evidence_calls = 0
        self.match_calls = 0

    def judge_evidence_raw(self, **kwargs: Any) -> dict:
        self.evidence_calls += 1
        assert self.evidence_raw is not None
        return self.evidence_raw

    def judge_match_raw(self, **kwargs: Any) -> dict:
        self.match_calls += 1
        assert self.match_raw is not None
        return self.match_raw


# --------------------------------------------------------------------------
# cache key stability
# --------------------------------------------------------------------------


def test_evidence_cache_key_stable_for_identical_input() -> None:
    key1 = evidence_cache_key("some source text", "A claim about four days")
    key2 = evidence_cache_key("some source text", "A claim about four days")
    assert key1 == key2


def test_evidence_cache_key_normalises_claim_number_words_and_whitespace() -> None:
    key1 = evidence_cache_key("some source text", "the order was  four days late")
    key2 = evidence_cache_key("some source text", "the order was 4 days late")
    assert key1 == key2


def test_evidence_cache_key_changes_with_source_or_claim() -> None:
    base = evidence_cache_key("source A", "claim one")
    assert base != evidence_cache_key("source B", "claim one")
    assert base != evidence_cache_key("source A", "claim two")


def test_match_cache_key_stable_and_order_sensitive() -> None:
    k1 = match_cache_key("source text", "planted claim", "produced claim")
    k2 = match_cache_key("source text", "planted claim", "produced claim")
    assert k1 == k2
    reversed_key = match_cache_key("source text", "produced claim", "planted claim")
    assert k1 != reversed_key


# --------------------------------------------------------------------------
# VerdictStore: round trip + human override
# --------------------------------------------------------------------------


def test_verdict_store_round_trips_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.yaml"
    store = VerdictStore(path)
    entry = {
        "kind": "evidence",
        "source_id": "S-test",
        "claim": "a claim",
        "verdict": "supported",
        "sentence_indices": [1, 2],
        "reason": "because",
        "judge_model": "claude-sonnet-5",
        "date": "2026-09-18",
        "human": None,
    }
    store.put("key1", entry)
    store.save()

    reloaded = VerdictStore(path)
    assert reloaded.get("key1") == entry
    assert len(reloaded) == 1


def test_verdict_store_put_never_clobbers_an_existing_human_label(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.yaml")
    store.put(
        "k",
        {
            "kind": "evidence",
            "source_id": "s",
            "claim": "c",
            "verdict": "unsupported",
            "sentence_indices": [],
            "reason": "r1",
            "judge_model": "m1",
            "date": "2026-09-18",
            "human": "supported",
        },
    )
    # A fresh model call re-answers the same question with a DIFFERENT raw
    # verdict -- the human label must survive.
    store.put(
        "k",
        {
            "kind": "evidence",
            "source_id": "s",
            "claim": "c",
            "verdict": "unsupported",
            "sentence_indices": [],
            "reason": "r2 (a re-run)",
            "judge_model": "m2",
            "date": "2026-09-19",
            "human": None,
        },
    )
    entry = store.get("k")
    assert entry is not None
    assert entry["human"] == "supported"
    assert entry["reason"] == "r2 (a re-run)"  # everything else DOES update


def test_verdict_store_human_override_evidence_kind(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.yaml")
    store.put(
        "k",
        {
            "kind": "evidence",
            "source_id": "s",
            "claim": "c",
            "verdict": "unsupported",
            "sentence_indices": [],
            "reason": "r",
            "judge_model": "m",
            "date": "2026-09-18",
            "human": None,
        },
    )
    assert store.effective_verdict("k")["verdict"] == "unsupported"

    store.put("k", {**store.get("k"), "human": "supported"})
    assert store.effective_verdict("k")["verdict"] == "supported"

    store.put("k", {**store.get("k"), "human": "fabricated"})
    assert store.effective_verdict("k")["verdict"] == "unsupported"


def test_verdict_store_human_override_match_kind(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.yaml")
    store.put(
        "k",
        {
            "kind": "match",
            "source_id": "s",
            "claim": "planted || produced",
            "verdict": "not_same",
            "sentence_indices": None,
            "reason": "r",
            "judge_model": "m",
            "date": "2026-09-18",
            "human": "supported",
        },
    )
    assert store.effective_verdict("k")["verdict"] == "same"

    store.put("k", {**store.get("k"), "human": "fabricated"})
    assert store.effective_verdict("k")["verdict"] == "not_same"


def test_verdict_store_missing_key_returns_none(tmp_path: Path) -> None:
    store = VerdictStore(tmp_path / "verdicts.yaml")
    assert store.get("nope") is None
    assert store.effective_verdict("nope") is None


# --------------------------------------------------------------------------
# sentence-check downgrade (+ the context-word calibration fix)
# --------------------------------------------------------------------------


def test_sentence_check_true_when_cited_sentences_cover_the_claim() -> None:
    sentences = ["The order arrived four days late on 2026-08-04.", "Unrelated filler sentence."]
    assert sentence_check("the order arrived four days late", sentences, [0]) is True


def test_sentence_check_false_when_indices_missing_or_out_of_range() -> None:
    sentences = ["The order arrived four days late."]
    assert sentence_check("the order arrived four days late", sentences, []) is False
    assert sentence_check("the order arrived four days late", sentences, [5]) is False


def test_sentence_check_false_when_cited_sentences_dont_cover_the_claim() -> None:
    sentences = ["Completely unrelated content about something else entirely."]
    assert sentence_check("the order arrived four days late", sentences, [0]) is False


def test_sentence_check_excludes_context_only_account_name_from_required_words() -> None:
    """The 2026-09-18 calibration finding: a claim's account name is
    legitimately context-only (never spoken in a real call transcript,
    same root cause as the deterministic classifier's own fragment fix)
    -- requiring the cited sentences to literally contain "Cobalt Ridge
    Medical Center" re-creates the false positive one layer up. Words
    that are ALSO in `context.context_strings` don't count against the
    coverage ratio."""
    context = EvidenceContext(context_strings=frozenset({"cobalt ridge medical center"}))
    sentences = ["Our new coordinator is still learning the terminology."]
    claim = (
        "Cobalt Ridge Medical Center has a new supply coordinator who is still "
        "learning the terminology."
    )
    # Fails WITHOUT context (the account name drags real coverage under 60%)...
    assert sentence_check(claim, sentences, [0]) is False
    # ...and passes WITH it.
    assert sentence_check(claim, sentences, [0], context) is True


def test_sentence_check_vacuous_context_only_claim_still_fails() -> None:
    """A claim that is NOTHING BUT a context name (no other content at
    all) must not pass on zero real checking -- the same anti-vacuous-
    vote principle `extraction._fragment_vote` already enforces."""
    context = EvidenceContext(context_strings=frozenset({"cobalt ridge medical center"}))
    sentences = ["Some completely unrelated sentence about something else."]
    claim = "Cobalt Ridge Medical Center."
    assert sentence_check(claim, sentences, [0], context) is False


def test_resolve_evidence_verdict_unsupported_ignores_indices() -> None:
    sentences = ["The order arrived four days late."]
    raw = {"supported": False, "sentence_indices": [0], "reason": "no basis"}
    resolved = resolve_evidence_verdict("the order arrived four days late", sentences, raw)
    assert resolved["verdict"] == "unsupported"


def test_resolve_evidence_verdict_supported_with_good_citation() -> None:
    sentences = ["The order arrived four days late on 2026-08-04."]
    raw = {"supported": True, "sentence_indices": [0], "reason": "matches"}
    resolved = resolve_evidence_verdict("the order arrived four days late", sentences, raw)
    assert resolved["verdict"] == "supported"


def test_resolve_evidence_verdict_downgrades_to_unverifiable_on_bad_citation() -> None:
    sentences = ["Completely unrelated content."]
    raw = {"supported": True, "sentence_indices": [0], "reason": "trust me"}
    resolved = resolve_evidence_verdict("the order arrived four days late", sentences, raw)
    assert resolved["verdict"] == "unverifiable"


def test_resolve_evidence_verdict_downgrades_when_no_indices_cited() -> None:
    sentences = ["The order arrived four days late."]
    raw = {"supported": True, "sentence_indices": [], "reason": "trust me"}
    resolved = resolve_evidence_verdict("the order arrived four days late", sentences, raw)
    assert resolved["verdict"] == "unverifiable"


def test_resolve_match_verdict() -> None:
    assert resolve_match_verdict({"same": True, "reason": "x"})["verdict"] == "same"
    assert resolve_match_verdict({"same": False, "reason": "x"})["verdict"] == "not_same"


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def test_select_sentence_window_returns_everything_under_budget() -> None:
    sentences = ["one.", "two.", "three."]
    assert select_sentence_window(sentences, "one", max_chars=1000) == [0, 1, 2]


def test_select_sentence_window_focuses_on_claim_fragment_when_over_budget() -> None:
    sentences = [f"filler sentence number {i}." for i in range(20)]
    sentences[10] = "The Northlake account requested a Tidewater shipment manifest."
    idxs = select_sentence_window(sentences, "Tidewater shipment manifest", max_chars=80)
    assert 10 in idxs
    assert sum(len(sentences[i]) for i in idxs) <= 80 or len(idxs) == 1


def test_split_sentences_matches_extraction_window_split() -> None:
    text = "First sentence. Second sentence.\nThird sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence.", "Third sentence."]


# --------------------------------------------------------------------------
# apply_evidence_judge -- both flags, with fixtures (the real corpus)
# --------------------------------------------------------------------------


def _real_unplanted(source_id: str, claim: str, produced_id: str = "p1") -> UnplantedFact:
    return UnplantedFact(
        source_id=source_id,
        fact_type="other",
        claim=claim,
        locator="[00:00:00]",
        is_distractor_leak=False,
        produced_id=produced_id,
        in_source_text=False,
        evidence_basis="none",
    )


def test_apply_evidence_judge_off_mode_leaves_unplanted_unchanged_but_reports_counts() -> None:
    truth = _truth()
    manifest = truth.manifest
    u = _real_unplanted("S-call-004", "A claim with no textual basis at all.")
    score = _score(unplanted=[u], produced_by_source={"S-call-004": []})

    result = apply_evidence_judge(score, manifest, truth, mode="off", corpus_root=CORPUS_ROOT)

    assert result.invented_deterministic == 1
    assert result.invented_adjudicated == 1
    assert result.unplanted[0].evidence_basis == "none"


def test_apply_evidence_judge_cached_mode_never_calls_and_uses_the_store(tmp_path: Path) -> None:
    truth = _truth()
    manifest = truth.manifest
    claim = "Northlake Health System asked for a shipment manifest before delivery."
    u = _real_unplanted("S-call-004", claim)
    score = _score(unplanted=[u], produced_by_source={"S-call-004": []})

    from portolan.bench.corpus_loader import load_prepared_source

    text = load_prepared_source(manifest.source("S-call-004"), corpus_root=CORPUS_ROOT).text
    key = evidence_cache_key(text, claim)
    store = VerdictStore(tmp_path / "verdicts.yaml")
    store.put(
        key,
        {
            "kind": "evidence",
            "source_id": "S-call-004",
            "claim": claim,
            "verdict": "supported",
            "sentence_indices": [],
            "reason": "pre-seeded",
            "judge_model": "test",
            "date": "2026-09-18",
            "human": None,
        },
    )

    fake_judge = FakeJudge()
    result = apply_evidence_judge(
        score,
        manifest,
        truth,
        mode="cached",
        store=store,
        judge=fake_judge,
        corpus_root=CORPUS_ROOT,
    )

    assert fake_judge.evidence_calls == 0  # cached mode never calls, even with a judge in hand
    assert result.judge_calls == 0
    assert result.judge_cache_hits == 1
    assert result.invented_deterministic == 1
    assert result.invented_adjudicated == 0
    assert result.unplanted[0].evidence_basis == "judge_supported"
    assert result.unplanted[0].in_source_text is True


def test_apply_evidence_judge_cached_mode_miss_stays_invented() -> None:
    truth = _truth()
    manifest = truth.manifest
    u = _real_unplanted("S-call-004", "A claim never seen before, so the cache misses.")
    score = _score(unplanted=[u], produced_by_source={"S-call-004": []})

    result = apply_evidence_judge(
        score,
        manifest,
        truth,
        mode="cached",
        store=VerdictStore(Path("/tmp/nonexistent-verdicts.yaml")),
        corpus_root=CORPUS_ROOT,
    )
    assert result.judge_calls == 0
    assert result.invented_adjudicated == 1
    assert result.unplanted[0].evidence_basis == "none"


def test_apply_evidence_judge_live_mode_calls_and_caches(tmp_path: Path) -> None:
    truth = _truth()
    manifest = truth.manifest
    claim = "Northlake Health System asked for a shipment manifest before delivery."
    u = _real_unplanted("S-call-004", claim)
    score = _score(unplanted=[u], produced_by_source={"S-call-004": []})

    store = VerdictStore(tmp_path / "verdicts.yaml")
    fake_judge = FakeJudge(
        evidence_raw={"supported": False, "sentence_indices": [], "reason": "no basis"}
    )

    result = apply_evidence_judge(
        score, manifest, truth, mode="live", store=store, judge=fake_judge, corpus_root=CORPUS_ROOT
    )

    assert fake_judge.evidence_calls == 1
    assert result.judge_calls == 1
    assert result.judge_cache_hits == 0
    assert result.invented_adjudicated == 1
    assert result.unplanted[0].evidence_basis == "judge_unsupported"

    # A second call, same cache: no new model call.
    result2 = apply_evidence_judge(
        score, manifest, truth, mode="live", store=store, judge=fake_judge, corpus_root=CORPUS_ROOT
    )
    assert fake_judge.evidence_calls == 1  # unchanged
    assert result2.judge_cache_hits == 1


def test_apply_evidence_judge_live_mode_downgrades_to_unverifiable(tmp_path: Path) -> None:
    truth = _truth()
    manifest = truth.manifest
    claim = "Northlake Health System asked for a shipment manifest before delivery."
    u = _real_unplanted("S-call-004", claim)
    score = _score(unplanted=[u], produced_by_source={"S-call-004": []})

    store = VerdictStore(tmp_path / "verdicts.yaml")
    # The judge says "supported" but cites nothing -- must downgrade, not pass.
    fake_judge = FakeJudge(
        evidence_raw={"supported": True, "sentence_indices": [], "reason": "trust me"}
    )

    result = apply_evidence_judge(
        score, manifest, truth, mode="live", store=store, judge=fake_judge, corpus_root=CORPUS_ROOT
    )

    assert result.unplanted[0].evidence_basis == "judge_unverifiable"
    assert result.unplanted[0].is_unverifiable is True
    assert result.unverifiable_count == 1
    # Unverifiable is its own bucket -- never counted as adjudicated-invented.
    assert result.invented_adjudicated == 0


def test_apply_evidence_judge_human_override_skips_the_model_even_live(tmp_path: Path) -> None:
    truth = _truth()
    manifest = truth.manifest
    claim = "Northlake Health System asked for a shipment manifest before delivery."
    u = _real_unplanted("S-call-004", claim)
    score = _score(unplanted=[u], produced_by_source={"S-call-004": []})

    from portolan.bench.corpus_loader import load_prepared_source

    text = load_prepared_source(manifest.source("S-call-004"), corpus_root=CORPUS_ROOT).text
    key = evidence_cache_key(text, claim)
    store = VerdictStore(tmp_path / "verdicts.yaml")
    store.put(
        key,
        {
            "kind": "evidence",
            "source_id": "S-call-004",
            "claim": claim,
            "verdict": "unsupported",
            "sentence_indices": [],
            "reason": "model got this wrong",
            "judge_model": "test",
            "date": "2026-09-18",
            "human": "supported",
        },
    )
    fake_judge = FakeJudge(
        evidence_raw={"supported": False, "sentence_indices": [], "reason": "should not be called"}
    )

    result = apply_evidence_judge(
        score, manifest, truth, mode="live", store=store, judge=fake_judge, corpus_root=CORPUS_ROOT
    )

    assert fake_judge.evidence_calls == 0  # cache hit (human-overridden) short-circuits the call
    assert result.unplanted[0].evidence_basis == "judge_supported"


# --------------------------------------------------------------------------
# apply_match_judge -- both flags, with fixtures
# --------------------------------------------------------------------------


def _unmatched_with_candidate(
    fact_id: str, source_id: str, claim: str, candidate_claim: str
) -> UnmatchedFact:
    return UnmatchedFact(
        fact_id=fact_id,
        source_id=source_id,
        source_type="call",
        fact_type="request",
        claim=claim,
        locator="[00:00:00]",
        failed_criteria=["claim_similarity"],
        best_candidate_claim=candidate_claim,
        best_similarity=0.25,
        best_candidate=ProducedFactRecord(
            id="p1",
            type="request",
            claim=candidate_claim,
            entities=[],
            locator="[00:00:00]",
            confidence=0.9,
        ),
    )


def test_apply_match_judge_off_mode_is_a_noop_on_recall() -> None:
    truth = _truth()
    score = _score(recall=0.5, near_miss_credited_fact_ids=[])
    result = apply_match_judge(score, truth.manifest, truth, mode="off")
    assert result.adjudicated_recall == 0.5
    assert result.judge_matches_calls == 0


def test_apply_match_judge_live_mode_confirms_same_and_boosts_recall(tmp_path: Path) -> None:
    truth = _truth()
    entry = _unmatched_with_candidate(
        "F1", "S-call-004", "planted claim text", "produced claim text"
    )
    score = _score(
        recall=0.5,
        planted_total=2,
        near_miss_credited_fact_ids=["F1"],
        unmatched=[entry],
    )
    store = VerdictStore(tmp_path / "verdicts.yaml")
    fake_judge = FakeJudge(match_raw={"same": True, "reason": "same fact, different words"})

    result = apply_match_judge(
        score,
        truth.manifest,
        truth,
        mode="live",
        store=store,
        judge=fake_judge,
        corpus_root=CORPUS_ROOT,
    )

    assert fake_judge.match_calls == 1
    assert result.judge_matches_calls == 1
    assert result.adjudicated_matched_fact_ids == ["F1"]
    assert result.adjudicated_recall == pytest.approx(0.5 + 1 / 2)

    # Cached re-run: no new call, same result.
    result2 = apply_match_judge(
        score,
        truth.manifest,
        truth,
        mode="cached",
        store=store,
        judge=fake_judge,
        corpus_root=CORPUS_ROOT,
    )
    assert fake_judge.match_calls == 1
    assert result2.judge_matches_cache_hits == 1
    assert result2.adjudicated_matched_fact_ids == ["F1"]


def test_apply_match_judge_not_same_does_not_boost_recall(tmp_path: Path) -> None:
    truth = _truth()
    entry = _unmatched_with_candidate(
        "F1", "S-call-004", "planted claim text", "an unrelated claim"
    )
    score = _score(
        recall=0.5, planted_total=2, near_miss_credited_fact_ids=["F1"], unmatched=[entry]
    )
    store = VerdictStore(tmp_path / "verdicts.yaml")
    fake_judge = FakeJudge(match_raw={"same": False, "reason": "different facts"})

    result = apply_match_judge(
        score,
        truth.manifest,
        truth,
        mode="live",
        store=store,
        judge=fake_judge,
        corpus_root=CORPUS_ROOT,
    )
    assert result.adjudicated_matched_fact_ids == []
    assert result.adjudicated_recall == 0.5
