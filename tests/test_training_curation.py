"""Tests for `portolan.training.curation`: normalized repeat keys and
near-duplicate grouping (MinHash + LSH, confirmed by exact Jaccard).
Pure functions, no network, no corpus."""

from __future__ import annotations

import pytest

from portolan.training.curation import (
    NUM_PERM,
    jaccard,
    minhash_signature,
    normalize_for_dedup,
    pair_key,
    quote_key,
    shingle_hashes,
    similarity_groups,
)

BASE = (
    "Mike Day warned that the bill could interfere with gas sales over the interstates "
    "and would certainly threaten any pipeline plans for building across the border into "
    "California, and that large customers would have to commit to firm service on a five "
    "year basis with exit fees or accept interruptible service from the utilities instead."
)
OTHER = (
    "The consumer reported that the servicer never credited twelve monthly student loan "
    "payments of two hundred ten dollars that were withdrawn automatically from checking, "
    "and that nobody could say who placed the account into a deferment period in the spring."
)


def test_normalize_folds_case_glyphs_and_punctuation_but_keeps_dollar_and_digits() -> None:
    assert normalize_for_dedup("It’s $299/month!") == normalize_for_dedup("it's $299 / month")
    assert normalize_for_dedup("It’s $299/month!") == "it s $299 month"
    assert normalize_for_dedup("$299") != normalize_for_dedup("299")
    assert normalize_for_dedup(None) == ""


def test_pair_key_and_quote_key_match_across_formatting() -> None:
    assert pair_key("InsightViz costs $2000 per month", "it’s $2000/month") == pair_key(
        "insightviz costs $2000 per month.", "It's $2000 / month"
    )
    assert quote_key("It's $5k/month.") == quote_key("it’s $5k / month")
    assert quote_key("It's $5k/month.") != quote_key("It's $5,000/month.")


def test_shingles_short_and_empty_texts() -> None:
    assert shingle_hashes("") == frozenset()
    assert len(shingle_hashes("two words")) == 1
    assert len(shingle_hashes("one two three four five six")) == 2


def test_minhash_signature_is_deterministic_and_full_width() -> None:
    s = shingle_hashes(BASE)
    sig1 = minhash_signature(s)
    sig2 = minhash_signature(shingle_hashes(BASE))
    assert sig1 == sig2
    assert len(sig1) == NUM_PERM
    assert minhash_signature(frozenset()) is None


def test_minhash_agreement_tracks_jaccard() -> None:
    a = shingle_hashes(BASE)
    b = shingle_hashes(BASE.replace("five year", "ten year"))
    c = shingle_hashes(OTHER)
    sa, sb, sc = minhash_signature(a), minhash_signature(b), minhash_signature(c)
    agree_ab = sum(x == y for x, y in zip(sa, sb, strict=True)) / NUM_PERM
    agree_ac = sum(x == y for x, y in zip(sa, sc, strict=True)) / NUM_PERM
    assert jaccard(a, b) > 0.7
    assert agree_ab == pytest.approx(jaccard(a, b), abs=0.2)
    assert agree_ac < 0.1


def test_exact_duplicates_after_normalization_group_together() -> None:
    groups = similarity_groups([BASE, BASE.upper() + "!!", OTHER])
    assert [0, 1] in groups.duplicate_groups
    assert [2] in groups.duplicate_groups


def test_near_duplicate_groups_at_dedup_threshold_and_distinct_texts_stay_apart() -> None:
    near = BASE.replace("five year", "5 year")  # one-token edit
    groups = similarity_groups([BASE, OTHER, near])
    assert [0, 2] in groups.duplicate_groups
    assert [1] in groups.duplicate_groups
    (pair,) = [p for p in groups.pairs if p[:2] == (0, 2)]
    assert pair[2] >= 0.8


def test_split_groups_join_siblings_below_the_dedup_line() -> None:
    # Same thread, second half replaced: shingle Jaccard lands between the
    # split line (0.5) and the dedup line (0.8).
    words = BASE.split()
    sibling = " ".join(words[: int(len(words) * 0.75)] + OTHER.split()[:10])
    j = jaccard(shingle_hashes(BASE), shingle_hashes(sibling))
    assert 0.5 <= j < 0.8, j
    groups = similarity_groups([BASE, sibling, OTHER])
    assert [0] in groups.duplicate_groups and [1] in groups.duplicate_groups
    assert [0, 1] in groups.split_groups
    assert [2] in groups.split_groups


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError):
        similarity_groups(["a"], dedup_threshold=0.5, split_threshold=0.8)
