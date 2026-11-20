"""Tests for `scripts/data/build_training_set.py`
(feature/labeling-exact-quotes, build instructions item 4): the teacher
allowlist, per-fact quote verification, the two caps (`other`-typed
facts, negative records), and the never-split-a-source 90/10 hash split.
No network, no real corpus -- `build_training_set(..., records=[...])`
takes an in-memory list instead of reading `var/data/silver/`.

`scripts/` isn't a package -- loaded by file path, same pattern as
`tests/test_set_owners.py`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from portolan.training import contamination
from portolan.training.records import TrainingFact, TrainingRecord
from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "build_training_set.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_training_set_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_training_set_script"] = module
    spec.loader.exec_module(module)
    return module


# Safe at module level: `require_repo_path` above already skipped this module
# during collection when `scripts/` is absent (the container image).
builder = _load_module()


# The builder now drops repeated documents, claim+quote pairs and quotes
# (review 2026-09-22), so by default every fixture fact gets its own claim
# and quote and every fixture record its own text containing those
# quotes. With one shared text and quote, every test below would measure
# the dedup step instead of the behavior it names.
_AUTO = object()
_SEQ = itertools.count()


def _fact(
    quote: str | object | None = _AUTO,
    quote_span: tuple[int, int] | None = None,
    fact_type: str = "complaint",
    claim: str | None = None,
) -> TrainingFact:
    n = next(_SEQ)
    if quote is _AUTO:
        quote = f"device {n} malfunctioned"
    return TrainingFact(
        fact_type=fact_type,
        claim=claim or f"claim {n}",
        quote=quote,
        quote_span=quote_span,
        locator="L1",
    )


def _unique_words(seed: str) -> str:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return " ".join(digest[i : i + 6] for i in range(0, 36, 6))


def _record(
    record_id: str,
    source_id: str,
    facts: list[TrainingFact],
    *,
    corpus: str = "enron",
    text: str | None = None,
    teachers: list[str] | None = None,
    agreement: float = 1.0,
) -> TrainingRecord:
    if text is None:
        quotes = " ".join(f.quote for f in facts if f.quote)
        text = f"{_unique_words(record_id)} {quotes}".strip()
    return TrainingRecord(
        record_id=record_id,
        corpus=corpus,
        source_id=source_id,
        kind="email",
        license="public record",
        text=text,
        facts=facts,
        teachers=teachers if teachers is not None else ["deepseek/deepseek-v3.2"],
        agreement=agreement,
        created="2026-09-21T00:00:00-07:00",
    )


# --------------------------------------------------------------------------
# teacher allowlist
# --------------------------------------------------------------------------


def test_teacher_allowed_for_every_named_open_weight_family() -> None:
    allowed_ids = [
        "qwen/qwen3.8-27b:free",
        "deepseek/deepseek-v3.2",
        "mistralai/mistral-nemo",
        "z-ai/glm-5.2:free",
        "moonshotai/kimi-k2:free",
        "openai/gpt-oss-120b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ]
    for model_id in allowed_ids:
        assert builder.record_teachers_allowed([model_id]) is True, model_id


def test_teacher_not_allowed_for_cohere() -> None:
    assert builder.record_teachers_allowed(["cohere/north-mini-code:free"]) is False


def test_record_excluded_when_any_teacher_disallowed() -> None:
    assert builder.record_teachers_allowed(["deepseek/deepseek-v3.2", "cohere/x:free"]) is False


def test_build_drops_record_with_disallowed_teacher_and_counts_it() -> None:
    good = _record("S-1", "S-1", [_fact()], teachers=["deepseek/deepseek-v3.2"])
    bad = _record("S-2", "S-2", [_fact()], teachers=["cohere/north-mini-code:free"])
    train, val, stats = builder.build_training_set(Path("unused"), records=[good, bad])
    all_ids = {r.record_id for r in train + val}
    assert all_ids == {"S-1"}
    assert stats.dropped_by_reason["teacher_not_allowed"] == 1


# --------------------------------------------------------------------------
# per-fact quote verification
# --------------------------------------------------------------------------


def test_fact_verified_via_quote_span() -> None:
    text = "the device malfunctioned during use"
    fact = _fact(quote="device malfunctioned", quote_span=(4, 24))
    record = _record("S-1", "S-1", [fact], text=text)
    train, val, stats = builder.build_training_set(Path("unused"), records=[record])
    (out,) = train + val
    assert len(out.facts) == 1
    assert stats.dropped_by_reason["quote_verification_failed"] == 0


def test_fact_with_wrong_span_is_dropped() -> None:
    # The whole record goes: keeping it with the fact removed would teach
    # the student that the quoted span is not a fact (a false negative).
    text = "the device malfunctioned during use"
    fact = _fact(quote="device malfunctioned", quote_span=(0, 6))  # wrong span
    record = _record("S-1", "S-1", [fact], text=text)
    padding = [_record(f"P-{i}", f"P-{i}", [_fact()]) for i in range(9)]
    train, val, stats = builder.build_training_set(Path("unused"), records=[record, *padding])
    assert "S-1" not in {r.record_id for r in train + val}
    assert stats.dropped_by_reason["quote_verification_failed"] == 1


def test_fact_without_span_falls_back_to_quote_in_text(tmp_path: Path) -> None:
    text = "the device malfunctioned during use"
    ok_fact = _fact(quote="device malfunctioned", quote_span=None)
    bad_fact = _fact(quote="never appears here at all", quote_span=None)
    good = _record("S-0", "S-0", [_fact(quote="device malfunctioned", quote_span=None)], text=text)
    mixed = _record("S-1", "S-1", [ok_fact, bad_fact], text=text + " again, twice over")
    train, val, stats = builder.build_training_set(Path("unused"), records=[good, mixed])
    # `quote in text` verifies the good record; one unverifiable fact
    # drops the mixed record whole rather than shipping it with one fact.
    assert [r.record_id for r in train + val] == ["S-0"]
    assert stats.dropped_by_reason["quote_verification_failed"] == 1


@pytest.mark.parametrize("quote", [None, "", "   "])
def test_a_fact_with_no_or_blank_quote_drops_its_record(quote: str | None) -> None:
    # Guardrails review 2026-09-22, row 32: `"" in text` is always true, so a blank quote
    # passed verification, and a missing one was never checked at all.
    record = _record("S-1", "S-1", [_fact(), _fact(quote=quote, fact_type="other")])
    padding = [_record(f"P-{i}", f"P-{i}", [_fact()]) for i in range(9)]
    train, val, stats = builder.build_training_set(Path("unused"), records=[record, *padding])
    assert "S-1" not in {r.record_id for r in train + val}
    assert stats.dropped_by_reason["fact_without_quote"] == 1
    assert stats.dropped_by_reason["quote_verification_failed"] == 0


# --------------------------------------------------------------------------
# caps
# --------------------------------------------------------------------------


def test_other_type_cap_drops_excess_at_fifteen_percent() -> None:
    # The cap now drops whole records, never single facts (a record with
    # its `other` fact cut out would teach that span is not a fact).
    # 4 all-"other" records + 1 mixed + 10 complaint-only: 5 of 16 facts
    # are "other" (31%). All-"other" records go first: 4/15, 3/14, 2/13
    # (15.4%, still over), then 1/12 (8.3%) <= 15% -- 4 records dropped,
    # the mixed record keeps its "other" fact.
    records = [_record(f"O-{i}", f"O-{i}", [_fact(fact_type="other")]) for i in range(4)]
    records.append(_record("M-0", "M-0", [_fact(fact_type="other"), _fact(fact_type="complaint")]))
    records += [_record(f"C-{i}", f"C-{i}", [_fact()]) for i in range(10)]
    by_id = {r.record_id: r for r in records}
    train, val, stats = builder.build_training_set(Path("unused"), records=records)
    kept = train + val
    remaining_other = sum(1 for r in kept for f in r.facts if f.fact_type == "other")
    assert stats.dropped_by_reason["other_type_cap"] == 4
    assert remaining_other == 1
    assert remaining_other / sum(len(r.facts) for r in kept) <= 0.15
    assert stats.facts_per_type["complaint"] == 11  # complaint facts untouched
    for r in kept:  # every kept record is exactly its input
        assert r.model_dump() == by_id[r.record_id].model_dump()


def test_other_type_cap_is_a_noop_when_already_under_ratio() -> None:
    # 1 "other" fact among 10 total facts (10%) -- already under the 15%
    # cap, nothing to drop.
    records = [_record("S-0", "S-0", [_fact(fact_type="other")])]
    records += [_record(f"S-{i}", f"S-{i}", [_fact()]) for i in range(1, 10)]
    _train, _val, stats = builder.build_training_set(Path("unused"), records=records)
    assert stats.dropped_by_reason["other_type_cap"] == 0


def test_negative_cap_drops_excess_negative_records() -> None:
    # 10 records: 6 negatives (60%), 4 positives. Cap solves keep <=
    # ratio * positives / (1 - ratio) = 0.30 * 4 / 0.70 = 1.71 ->
    # max_negatives=1, dropping 5 -- so the FINAL pool (4 positive + 1
    # negative = 5 records) actually sits at <=30% negative, not the
    # pre-drop pool (a naive int(10 * 0.30) = 3 would leave 3/7 = 43%).
    records = []
    for i in range(6):
        records.append(_record(f"NEG-{i}", f"NEG-{i}", []))
    for i in range(4):
        records.append(_record(f"POS-{i}", f"POS-{i}", [_fact()]))
    train, val, stats = builder.build_training_set(Path("unused"), records=records)
    remaining = train + val
    remaining_negatives = sum(1 for r in remaining if r.is_negative)
    assert remaining_negatives == 1
    assert len(remaining) == 5  # 10 - 5 dropped negatives
    assert stats.dropped_by_reason["negative_cap"] == 5
    assert stats.negatives_total == 1
    assert remaining_negatives / len(remaining) <= 0.30


def test_negative_cap_is_deterministic_across_runs() -> None:
    records = [_record(f"NEG-{i}", f"NEG-{i}", []) for i in range(10)]
    records += [_record(f"POS-{i}", f"POS-{i}", [_fact()]) for i in range(2)]
    train1, val1, _ = builder.build_training_set(Path("unused"), records=records)
    train2, val2, _ = builder.build_training_set(Path("unused"), records=records)
    ids1 = sorted(r.record_id for r in train1 + val1)
    ids2 = sorted(r.record_id for r in train2 + val2)
    assert ids1 == ids2


# --------------------------------------------------------------------------
# split: 90/10 by source_id hash, never splitting one source
# --------------------------------------------------------------------------


def test_split_never_separates_segments_of_the_same_source() -> None:
    seg1 = _record("S-1#seg1of2", "S-1", [_fact()])
    seg2 = _record("S-1#seg2of2", "S-1", [_fact()])
    train, val, _ = builder.build_training_set(Path("unused"), records=[seg1, seg2])
    in_train = {r.record_id for r in train}
    in_val = {r.record_id for r in val}
    # both segments landed in the SAME split
    assert ({"S-1#seg1of2", "S-1#seg2of2"} <= in_train) or (
        {"S-1#seg1of2", "S-1#seg2of2"} <= in_val
    )


def test_split_is_deterministic_for_a_given_source_id() -> None:
    record = _record("S-1", "a-fixed-source-id", [_fact()])
    train1, val1, _ = builder.build_training_set(Path("unused"), records=[record])
    train2, val2, _ = builder.build_training_set(Path("unused"), records=[record])
    assert (len(train1), len(val1)) == (len(train2), len(val2))


def test_split_roughly_ninety_ten_over_many_sources() -> None:
    records = [_record(f"S-{i}", f"source-{i}", [_fact()]) for i in range(500)]
    _train, _val, stats = builder.build_training_set(Path("unused"), records=records)
    assert stats.train_records + stats.val_records == 500
    val_fraction = stats.val_records / 500
    assert 0.05 < val_fraction < 0.15  # roughly 10%, hash-bucketed not exact


# --------------------------------------------------------------------------
# stats aggregation
# --------------------------------------------------------------------------


def test_stats_count_records_per_prompt_stamp() -> None:
    legacy = _record("A-1", "A-1", [_fact()])
    stamped = _record("B-1", "B-1", [_fact()]).model_copy(
        update={"prompt_variant": "v6-type-precision", "system_prompt_sha256": "ab" * 32}
    )
    _train, _val, stats = builder.build_training_set(Path("unused"), records=[legacy, stamped])
    assert stats.records_per_prompt == {
        builder.UNSTAMPED_PROMPT: 1,
        f"v6-type-precision {'ab' * 8}": 1,
    }


def test_stats_aggregate_per_corpus_and_teachers() -> None:
    a = _record("A-1", "A-1", [_fact()], corpus="enron", teachers=["deepseek/deepseek-v3.2"])
    b = _record("B-1", "B-1", [_fact()], corpus="ami", teachers=["qwen/qwen3.8-27b:free"])
    _train, _val, stats = builder.build_training_set(Path("unused"), records=[a, b])
    assert stats.records_per_corpus["enron"] == 1
    assert stats.records_per_corpus["ami"] == 1
    assert stats.teachers == {"deepseek/deepseek-v3.2", "qwen/qwen3.8-27b:free"}
    assert stats.train_records + stats.val_records == 2


# --------------------------------------------------------------------------
# provenance (review 2026-09-22)
# --------------------------------------------------------------------------


def _ids(train, val) -> set[str]:
    return {r.record_id for r in train + val}


def test_allowlist_is_exact_ids_not_family_substrings() -> None:
    # closed-weight API models in open families, and families whose
    # output-use terms are unread (v2 spec section 4), fail closed
    for model_id in (
        "qwen/qwen-max",
        "qwen/qwen3.7-flash",
        "mistralai/mistral-medium-3",
        "meta-llama/llama-3.3-70b:free",
        "google/gemma-3-27b:free",
        "openai/gpt-4o",
    ):
        assert builder.record_teachers_allowed([model_id]) is False, model_id
    assert builder.record_teachers_allowed(["MoonshotAI/Kimi-K2:free"]) is True


def test_record_with_no_teacher_fails_closed_and_is_counted() -> None:
    assert builder.record_teachers_allowed([]) is False
    rec = _record("S-1", "S-1", [_fact()], teachers=[])
    keep = _record("S-2", "S-2", [_fact()])
    train, val, stats = builder.build_training_set(Path("unused"), records=[rec, keep])
    assert _ids(train, val) == {"S-2"}
    assert stats.dropped_by_reason["no_teacher_recorded"] == 1
    assert stats.dropped_by_teacher[builder.NO_TEACHER] == 1


def test_drops_are_counted_per_teacher_id() -> None:
    recs = [
        _record("A", "A", [_fact()], teachers=["cohere/north-mini-code:free"]),
        _record("B", "B", [_fact()], teachers=["cohere/north-mini-code:free"]),
        _record("C", "C", [_fact()], teachers=["deepseek/deepseek-v3.2", "openai/gpt-4o"]),
        _record("D", "D", [_fact()]),
    ]
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    assert _ids(train, val) == {"D"}
    assert stats.dropped_by_teacher == {"cohere/north-mini-code:free": 2, "openai/gpt-4o": 1}


def test_closed_model_document_text_is_dropped() -> None:
    # SaaS sales dialogue was written by GPT-4 (dataset card), so its
    # text breaks the provenance rule even though Kimi K2 labeled it.
    saas = _record("S-1", "S-1", [_fact()], corpus="saas_sales", teachers=["moonshotai/kimi-k2"])
    enron = _record("E-1", "E-1", [_fact()])
    train, val, stats = builder.build_training_set(Path("unused"), records=[saas, enron])
    assert _ids(train, val) == {"E-1"}
    assert stats.dropped_by_reason["document_text_not_open_weight"] == 1


def test_unknown_corpus_fails_closed() -> None:
    rec = _record("X-1", "X-1", [_fact()], corpus="some_new_corpus")
    train, val, stats = builder.build_training_set(Path("unused"), records=[rec])
    assert train + val == []
    assert stats.dropped_by_reason["corpus_provenance_unknown"] == 1


def test_v1_synthetic_negatives_and_template_claims_are_dropped() -> None:
    writer = ["qwen/qwen3-32b"]
    neg = _record("w:S-ticket-n001", "S-ticket-n001", [], corpus="synthetic", teachers=[])
    pos = _record("w:S-call-001", "S-call-001", [_fact()], corpus="synthetic", teachers=writer)
    real = _record("E-1", "E-1", [_fact()])
    train, val, stats = builder.build_training_set(Path("unused"), records=[neg, pos, real])
    assert _ids(train, val) == {"E-1"}
    assert stats.dropped_by_reason["v1_templated_negative"] == 1
    assert stats.dropped_by_reason["v1_template_claim"] == 1
    assert stats.provenance_violations["no_teacher_recorded"] == 1


# --------------------------------------------------------------------------
# label completeness
# --------------------------------------------------------------------------


def test_negative_made_by_gate_drops_is_dropped() -> None:
    # agreement 0.0 with no facts: the teacher emitted facts and the gate
    # dropped every one; that is not a verified negative
    gate_made = _record("N-1", "N-1", [], agreement=0.0)
    teacher_empty = _record("N-2", "N-2", [], agreement=1.0)
    padding = [_record(f"P-{i}", f"P-{i}", [_fact()]) for i in range(5)]
    train, val, stats = builder.build_training_set(
        Path("unused"), records=[gate_made, teacher_empty, *padding]
    )
    ids = _ids(train, val)
    assert "N-1" not in ids and "N-2" in ids
    assert stats.dropped_by_reason["negative_after_gate_drops"] == 1


def test_partial_records_are_dropped_by_default_and_optionally_kept() -> None:
    partial = _record("P-1", "P-1", [_fact()], agreement=0.25)  # 1 kept of 4 emitted
    full = _record("F-1", "F-1", [_fact()])
    train, val, stats = builder.build_training_set(Path("unused"), records=[partial, full])
    assert _ids(train, val) == {"F-1"}
    assert stats.dropped_by_reason["partial_labels"] == 1
    config = builder.CurationConfig(drop_partial_records=False)
    train, val, stats = builder.build_training_set(
        Path("unused"), records=[partial, full], config=config
    )
    assert _ids(train, val) == {"P-1", "F-1"}
    assert stats.partial_records_per_corpus["enron"] == 1
    assert stats.est_dropped_facts_per_corpus["enron"] == 3


# --------------------------------------------------------------------------
# dedup and repeat caps
# --------------------------------------------------------------------------

_THREAD = (
    "This could interfere with Enron gas sales over the interstates and would certainly "
    "threaten any Transwestern plans for building across the border into California. The "
    "large customers have to commit to take firm service on a five year basis with exit fees."
)


def test_near_duplicate_documents_keep_the_record_with_most_facts() -> None:
    q = "would certainly threaten any Transwestern plans"
    a = _record("E-071454", "E-071454", [_fact(quote=q, fact_type="risk")], text=_THREAD)
    b = _record("E-071455", "E-071455", [], text="Cc: someone. " + _THREAD)
    train, val, stats = builder.build_training_set(Path("unused"), records=[a, b])
    assert _ids(train, val) == {"E-071454"}
    assert stats.dropped_by_reason["near_duplicate_document"] == 1
    assert stats.duplicate_groups == 1


def test_near_siblings_share_a_split() -> None:
    from portolan.training.curation import jaccard, shingle_hashes

    # 20 threads, each with a sibling that keeps its first 40 of 50 words
    # (shingle Jaccard about 0.64: over the 0.5 split line, under the 0.8
    # dedup line). Every thread and its sibling must land in one split.
    def words(seed: str, n: int) -> list[str]:
        return [hashlib.sha256(f"{seed}-{k}".encode()).hexdigest()[:7] for k in range(n)]

    recs = []
    for i in range(20):
        base = words(f"thread{i}", 50)
        head = " ".join(base)
        sibling = " ".join(base[:40] + words(f"tail{i}", 10))
        j = jaccard(shingle_hashes(head), shingle_hashes(sibling))
        assert 0.5 <= j < 0.8, j
        quote = " ".join(base[:2])
        recs.append(_record(f"A-{i}", f"A-{i}", [_fact(quote=quote)], text=head))
        recs.append(_record(f"B-{i}", f"B-{i}", [_fact(quote=quote)], text=sibling))
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    assert len(train + val) == 40
    assert stats.split_groups_multi == 20
    in_val = {r.record_id for r in val}
    for i in range(20):
        assert (f"A-{i}" in in_val) == (f"B-{i}" in in_val), i


def test_pair_repeat_cap_keeps_one_copy_of_an_identical_claim_and_quote() -> None:
    recs = [
        _record(
            f"F-{i}",
            f"F-{i}",
            [
                _fact(
                    quote="missing high/low glucose alarm",
                    claim="A customer reported a missing alarm",
                )
            ],
        )
        for i in range(3)
    ]
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    assert len(train + val) == 1
    assert stats.dropped_by_reason["pair_repeat_cap"] == 2
    assert stats.max_pair_repeat_out == 1


def test_quote_repeat_cap_limits_one_quote_to_three_records() -> None:
    recs = [
        _record(f"S-{i}", f"S-{i}", [_fact(quote="It's $299/month", fact_type="pricing")])
        for i in range(6)
    ]
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    assert len(train + val) == 3
    assert stats.dropped_by_reason["quote_repeat_cap"] == 3
    assert stats.max_quote_repeat_out == 3


def test_conflicting_types_keep_a_clear_majority_and_drop_the_rest() -> None:
    canned = "Company has responded to the consumer and the CFPB"
    recs = [
        _record(f"C-{i}", f"C-{i}", [_fact(quote=canned, fact_type="commitment")], corpus="cfpb")
        for i in range(3)
    ]
    recs.append(_record("C-9", "C-9", [_fact(quote=canned, fact_type="decision")], corpus="cfpb"))
    config = builder.CurationConfig(max_quote_repeats=10)
    train, val, stats = builder.build_training_set(Path("unused"), records=recs, config=config)
    assert _ids(train, val) == {"C-0", "C-1", "C-2"}
    assert stats.dropped_by_reason["conflicting_type_for_same_quote"] == 1


def test_conflicting_types_without_majority_drop_every_carrier() -> None:
    q = "we might have to look elsewhere"
    recs = [
        _record("R-1", "R-1", [_fact(quote=q, fact_type="risk")]),
        _record("R-2", "R-2", [_fact(quote=q, fact_type="competitor_mention")]),
        _record("R-3", "R-3", [_fact()]),
    ]
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    assert _ids(train, val) == {"R-3"}
    assert stats.dropped_by_reason["conflicting_type_for_same_quote"] == 2


# --------------------------------------------------------------------------
# corpus share cap
# --------------------------------------------------------------------------


def test_corpus_share_cap_limits_the_largest_corpus() -> None:
    recs = [_record(f"E-{i}", f"E-{i}", [_fact()]) for i in range(10)]
    recs += [_record(f"C-{i}", f"C-{i}", [_fact()], corpus="cfpb") for i in range(3)]
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    kept = train + val
    assert sum(r.corpus == "enron" for r in kept) == 3
    assert stats.dropped_by_reason["corpus_share_cap"] == 7
    assert max(stats.records_per_corpus.values()) / len(kept) <= 0.5


def test_corpus_share_cap_is_skipped_for_a_single_corpus() -> None:
    recs = [_record(f"E-{i}", f"E-{i}", [_fact()]) for i in range(10)]
    train, val, stats = builder.build_training_set(Path("unused"), records=recs)
    assert len(train + val) == 10
    assert "skipped" in stats.caps["corpus_share"]["applied"]


# --------------------------------------------------------------------------
# never edit, end to end on disk, data card
# --------------------------------------------------------------------------


def test_every_output_record_is_byte_identical_to_its_input() -> None:
    recs = [_record(f"E-{i}", f"E-{i}", [_fact(), _fact(fact_type="other")]) for i in range(6)]
    recs += [_record(f"C-{i}", f"C-{i}", [_fact()], corpus="cfpb") for i in range(6)]
    recs += [_record(f"N-{i}", f"N-{i}", []) for i in range(4)]
    by_id = {r.record_id: r.model_dump_json() for r in recs}
    train, val, _ = builder.build_training_set(Path("unused"), records=recs)
    assert train + val
    for r in train + val:
        assert r.model_dump_json() == by_id[r.record_id]


def _write_var_root(tmp_path: Path) -> Path:
    var_root = tmp_path / "var"
    enron = var_root / "data" / "silver" / "enron" / "records.jsonl"
    cfpb = var_root / "data" / "silver" / "cfpb" / "records.jsonl"
    synth = var_root / "data" / "silver" / "synthetic" / "world-1.jsonl"
    for path in (enron, cfpb, synth):
        path.parent.mkdir(parents=True, exist_ok=True)
    enron.write_text(
        "".join(_record(f"E-{i}", f"E-{i}", [_fact()]).model_dump_json() + "\n" for i in range(4))
    )
    cfpb.write_text(
        "".join(
            _record(f"C-{i}", f"C-{i}", [_fact()], corpus="cfpb").model_dump_json() + "\n"
            for i in range(4)
        )
    )
    synth.write_text(
        _record("w:n1", "n1", [], corpus="synthetic", teachers=[]).model_dump_json() + "\n"
    )
    return var_root


def test_main_writes_sets_manifest_and_data_card(tmp_path: Path) -> None:
    var_root = _write_var_root(tmp_path)
    out = tmp_path / "out"
    # This test is about manifest/data-card assembly, not the WP8
    # contamination gate -- `--no-contamination-check` skips that gate
    # explicitly rather than requiring a fixture `var/contamination/`
    # this test has no other reason to build. The gate's own refusal and
    # gating behavior are covered by the `test_main_*contamination*` and
    # `test_contaminat*`/`test_*_gate` tests below.
    assert (
        builder.main(["--var-root", str(var_root), "--out", str(out), "--no-contamination-check"])
        == 0
    )
    train = (out / "train.jsonl").read_text().splitlines()
    val = (out / "val.jsonl").read_text().splitlines()
    assert len(train) + len(val) == 8
    manifest = json.loads((out / "manifest.json").read_text())
    inputs = {i["path"]: i for i in manifest["inputs"]}
    enron_bytes = (var_root / "data" / "silver" / "enron" / "records.jsonl").read_bytes()
    assert (
        inputs["data/silver/enron/records.jsonl"]["sha256"]
        == hashlib.sha256(enron_bytes).hexdigest()
    )
    assert inputs["data/silver/enron/records.jsonl"]["lines"] == 4
    assert manifest["dropped_by_reason"] == {"v1_templated_negative": 1}
    card = (out / "data_card.md").read_text()
    for heading in (
        "## Totals",
        "## By corpus",
        "## By fact type",
        "## By teacher",
        "## Dropped records",
        "## Caps and dedup",
        "## Inputs",
        "## Settings",
    ):
        assert heading in card, heading
    assert hashlib.sha256(enron_bytes).hexdigest() in card
    assert "| v1_templated_negative | 1 | 0 | synthetic 1 |" in card


def test_main_refuses_to_write_under_the_exam_corpus(tmp_path: Path) -> None:
    var_root = _write_var_root(tmp_path)
    with pytest.raises(RuntimeError, match="hold-out"):
        builder.main(["--var-root", str(var_root), "--out", str(tmp_path / "corpus" / "out" / "x")])


# --------------------------------------------------------------------------
# contamination gate (WP8, 01-spec/world-generator-v2-2026-09-22.md
# section 6; coordinator review 2026-09-22 on scope and distinctiveness)
# --------------------------------------------------------------------------

_TEST_CONTAMINATION_PARAMS = contamination.ContaminationParams(
    minhash_permutations=32, lsh_bands=8, lsh_rows=4, minhash_seed=12345
)


def _contamination_refs(
    *,
    documents: list[tuple[str, str, str]] | None = None,
    name_entries: list[tuple[str, str]] | None = None,
    background_doc_freq: dict[str, int] | None = None,
) -> contamination.ContaminationReferences:
    params = _TEST_CONTAMINATION_PARAMS
    if documents:
        matrix, doc_meta, ngram_sorted = contamination.build_reference_matrix(documents, params)
    else:
        matrix = np.zeros((0, params.minhash_permutations), dtype=np.uint64)
        doc_meta = []
        ngram_sorted = np.array([], dtype=np.uint64)
    blocklist = contamination.build_name_blocklist(
        name_entries or [], params, background_doc_freq=background_doc_freq
    )
    return contamination.ContaminationReferences(matrix, doc_meta, ngram_sorted, blocklist, params)


def test_blocklisted_name_hit_does_not_block_a_public_record() -> None:
    # Coordinator review, 2026-09-22, rule 2: a blocklisted-name hit is
    # review-only for real public corpora -- counted, never dropped by
    # itself. "HalvardCare" is a genuinely distinctive Halvard product
    # name (background_doc_freq={} says it never appears in real text),
    # but the record below is `corpus="enron"`, not a generated corpus,
    # so the SCOPE rule -- not the name's own commonness -- is what
    # keeps it in.
    refs = _contamination_refs(name_entries=[("halvard", "HalvardCare")], background_doc_freq={})
    record = _record(
        "E-1",
        "E-1",
        [_fact(quote="we renewed")],
        corpus="enron",
        text="we renewed the HalvardCare plan",
    )
    train, val, stats = builder.build_training_set(
        Path("unused"), records=[record], contamination_refs=refs
    )
    assert {r.record_id for r in train + val} == {"E-1"}
    assert stats.contamination_per_corpus.get("enron", 0) == 0
    assert stats.contamination_review_per_corpus.get("enron", 0) == 1
    assert stats.contamination_review_by_reason["blocklisted_name"] == 1


def test_halvardcare_blocks_a_synthetic_record() -> None:
    # Same blocklisted name, but corpus="synthetic" -- inside
    # CONTAMINATION_HARD_GATE_CORPORA, so the name hit IS a hard gate.
    refs = _contamination_refs(name_entries=[("halvard", "HalvardCare")], background_doc_freq={})
    record = _record(
        "S-1",
        "S-1",
        [_fact(quote=None)],
        corpus="synthetic",
        text="we renewed the HalvardCare plan",
    )
    train, val, stats = builder.build_training_set(
        Path("unused"), records=[record], contamination_refs=refs
    )
    assert {r.record_id for r in train + val} == set()
    assert stats.contamination_per_corpus.get("synthetic", 0) == 1
    assert stats.contamination_by_reason["blocklisted_name"] == 1
    (entry,) = stats.contamination_dropped
    assert entry["record_id"] == "S-1"
    assert entry["reasons"] == ["blocklisted_name"]


# A stand-in for the real-corpus background (var/data/converted): ordinary
# writing in which the fictional exam names never occur.
_BACKGROUND_TEXTS = [
    "The customer called support about her account and asked for a manager.",
    "Procurement sent the invoice to accounts payable on Monday.",
    "Our customer success review with the director is on the calendar.",
]
_NAME_ENTRIES = [
    ("erb_fireflies", "Customer Success"),
    ("erb_gmail", "Procurement Manager"),
    ("erb_gmail", "Accounts Payable"),
    ("halvard", "HalvardCare"),
    ("halvard", "Tidewater T3"),
]


def _refs_against_background() -> contamination.ContaminationReferences:
    params = _TEST_CONTAMINATION_PARAMS
    phrases = [contamination.normalize_name_words(name) for _source, name in _NAME_ENTRIES]
    freq, docs = contamination.background_frequency(_BACKGROUND_TEXTS, phrases)
    blocklist = contamination.build_name_blocklist(
        _NAME_ENTRIES, params, background_doc_freq=freq, background_doc_count=docs
    )
    matrix = np.zeros((0, params.minhash_permutations), dtype=np.uint64)
    return contamination.ContaminationReferences(
        matrix, [], np.array([], dtype=np.uint64), blocklist, params
    )


def _gate_generated(text: str) -> builder.BuildStats:
    record = _record(
        "V-1", "V-1", [_fact(quote=None)], corpus=builder.SYNTHETIC_V2_CORPUS, text=text
    )
    _train, _val, stats = builder.build_training_set(
        Path("unused"), records=[record], contamination_refs=_refs_against_background()
    )
    return stats


def test_job_title_does_not_block_a_generated_record() -> None:
    stats = _gate_generated(
        "From: Tovin Marle, Procurement Manager\nCc: Ola Brand, customer success lead\n"
        "Tovin: the pharmacy wants the rollout moved to March."
    )
    assert stats.contamination_enabled is True
    assert stats.contamination_dropped == []
    assert stats.contamination_per_corpus.get(builder.SYNTHETIC_V2_CORPUS, 0) == 0


def test_common_phrase_in_background_does_not_block_a_generated_record() -> None:
    stats = _gate_generated("Tovin: I sent the credit memo to accounts payable this morning.")
    assert stats.contamination_dropped == []


@pytest.mark.parametrize(
    "text", ["we renewed the HalvardCare plan", "the Tidewater T3 alarmed twice overnight"]
)
def test_exam_names_still_block_a_generated_record(text: str) -> None:
    stats = _gate_generated(text)
    assert stats.contamination_per_corpus[builder.SYNTHETIC_V2_CORPUS] == 1
    (entry,) = stats.contamination_dropped
    assert entry["reasons"] == ["blocklisted_name"]


def test_exact_ngram_overlap_blocks_any_source() -> None:
    # Coordinator review: "The 13-gram overlap ... checks stay hard
    # gates for every source." A 20-word shared sentence gives 8
    # thirteen-word windows -- comfortably over min_shared_ngrams (3) --
    # embedded in a longer, otherwise-unrelated document so the overall
    # MinHash score stays below the near-duplicate threshold and this
    # isolates the exact-overlap check.
    shared = (
        "the annual renewal for the enterprise contract must be signed "
        "before the end of the fiscal quarter to avoid a lapse"
    )
    refs = _contamination_refs(documents=[("ref-1", "erb_jira", shared)])
    padding = " ".join(f"filler{i} token{i} unrelated{i}" for i in range(40))
    real = _record(
        "C-1", "C-1", [_fact(quote=None)], corpus="cfpb", text=f"{padding} {shared} {padding}"
    )
    synthetic = _record(
        "S-1",
        "S-1",
        [_fact(quote=None)],
        corpus="synthetic",
        text=f"{padding} {shared} {padding}",
    )
    train, val, stats = builder.build_training_set(
        Path("unused"), records=[real, synthetic], contamination_refs=refs
    )
    assert {r.record_id for r in train + val} == set()
    assert stats.contamination_per_corpus.get("cfpb", 0) == 1
    assert stats.contamination_per_corpus.get("synthetic", 0) == 1
    assert stats.contamination_by_reason["exact_ngram_overlap"] == 2


def test_contamination_gate_is_a_noop_when_refs_not_passed() -> None:
    # Default (`contamination_refs=None`, every pre-WP8 call site) --
    # text that would trip the blocklist above is kept, since the gate
    # never ran at all. Confirms every existing test in this file is
    # unaffected by WP8. Uses the default real corpus (not "synthetic"):
    # every "synthetic" record, positive or negative, fails an unrelated
    # provenance check in this builder (template claims / templated
    # negatives), which would confound this test's own assertion.
    record = _record(
        "E-1", "E-1", [_fact(quote="we renewed")], text="we renewed the HalvardCare plan"
    )
    train, val, stats = builder.build_training_set(Path("unused"), records=[record])
    assert {r.record_id for r in train + val} == {"E-1"}
    assert stats.contamination_enabled is False


def test_main_refuses_when_contamination_signatures_missing(tmp_path: Path) -> None:
    var_root = _write_var_root(tmp_path)
    out = tmp_path / "out"
    assert builder.main(["--var-root", str(var_root), "--out", str(out)]) == 1
    assert not (out / "train.jsonl").exists()


def test_main_writes_contamination_logs_when_signatures_present(tmp_path: Path) -> None:
    var_root = _write_var_root(tmp_path)
    out = tmp_path / "out"
    matrix = np.zeros((0, _TEST_CONTAMINATION_PARAMS.minhash_permutations), dtype=np.uint64)
    contamination.save_references(
        var_root / "contamination",
        matrix,
        [],
        np.array([], dtype=np.uint64),
        {},
        _TEST_CONTAMINATION_PARAMS,
        {},
        source_fingerprints=_exam_fingerprint(tmp_path),
    )
    assert builder.main(["--var-root", str(var_root), "--out", str(out)]) == 0
    assert (out / "contamination_dropped.jsonl").exists()
    assert (out / "contamination_review.jsonl").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["contamination"]["enabled"] is True
    assert manifest["contamination"]["dropped_total"] == 0
    card = (out / "data_card.md").read_text()
    assert "## Contamination (WP8)" in card


def _exam_fingerprint(root: Path) -> dict:
    exam = root / "exam"
    exam.mkdir(exist_ok=True)
    (exam / "S-call-001.txt").write_text("an exam call", encoding="utf-8")
    return contamination.fingerprint_sources({"halvard": exam})


def _save_empty_signatures(var_root: Path, fingerprints: dict) -> None:
    contamination.save_references(
        var_root / "contamination",
        np.zeros((0, _TEST_CONTAMINATION_PARAMS.minhash_permutations), dtype=np.uint64),
        [],
        np.array([], dtype=np.uint64),
        {},
        _TEST_CONTAMINATION_PARAMS,
        {},
        source_fingerprints=fingerprints,
    )


def test_main_records_the_sha256_of_the_files_it_wrote(tmp_path: Path) -> None:
    # scripts/train/mlx_lora.py refuses a train/val file that does not hash to these.
    var_root = _write_var_root(tmp_path)
    _save_empty_signatures(var_root, _exam_fingerprint(tmp_path))
    out = tmp_path / "out"
    assert builder.main(["--var-root", str(var_root), "--out", str(out)]) == 0
    outputs = json.loads((out / "manifest.json").read_text())["outputs"]
    for name in ("train.jsonl", "val.jsonl"):
        assert outputs[name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_main_refuses_signatures_built_from_an_older_exam(tmp_path: Path) -> None:
    var_root = _write_var_root(tmp_path)
    _save_empty_signatures(var_root, _exam_fingerprint(tmp_path))
    (tmp_path / "exam" / "S-call-002.txt").write_text("a regenerated call", encoding="utf-8")
    out = tmp_path / "out"
    assert builder.main(["--var-root", str(var_root), "--out", str(out)]) == 1
    assert not (out / "train.jsonl").exists()


def test_provenance_failures_is_the_check_the_builder_drops_on() -> None:
    ok = _record("E-1", "E-1", [_fact()])
    assert builder.provenance_failures(ok) == []
    closed = _record("E-2", "E-2", [_fact()], teachers=["anthropic/claude-sonnet-5"])
    saas = _record("E-3", "E-3", [_fact()], corpus="saas_sales")
    assert builder.provenance_failures(closed) == ["teacher_not_allowed"]
    assert builder.provenance_failures(saas) == ["document_text_not_open_weight"]
    train, val, stats = builder.build_training_set(Path("unused"), records=[ok, closed, saas])
    assert {r.record_id for r in train + val} == {"E-1"}
    assert stats.dropped_by_reason["teacher_not_allowed"] == 1
    assert stats.dropped_by_reason["document_text_not_open_weight"] == 1
