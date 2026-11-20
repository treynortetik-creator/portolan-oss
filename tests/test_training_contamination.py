"""Tests for `portolan.training.contamination` (`01-spec/world-generator-
v2-2026-09-22.md` section 6, work package WP8): every check fires on a
planted overlap and stays quiet on clean text, using small in-memory
fixtures built through `build_reference_matrix`/`build_name_blocklist` --
never by reading `corpus/out/` or EnterpriseRAG-Bench. The real corpora
are only ever read by `scripts/data/build_contamination_signatures.py`,
not by anything under test here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from portolan.training import contamination as C

# A tiny parameter set for fast, deterministic tests -- same shape rules
# as the production default (`lsh_bands * lsh_rows == minhash_permutations`)
# but far fewer permutations, so signatures build in microseconds.
_TEST_PARAMS = C.ContaminationParams(
    minhash_permutations=32,
    lsh_bands=8,
    lsh_rows=4,
    shingle_size=5,
    ngram_window=13,
    min_shared_ngrams=3,
    jaccard_exclude_threshold=0.5,
    jaccard_review_threshold=0.3,
    minhash_seed=12345,
    max_name_words=4,
)

REFERENCE_SENTENCE = (
    "the annual renewal for the enterprise contract must be signed before "
    "the end of the fiscal quarter to avoid a lapse in coverage"
)
UNRELATED_TEXT = (
    "sunlight filtered through the canopy while the hikers paused to check "
    "their map before continuing along the ridge toward the overlook"
)


def _build_refs(
    documents: list[tuple[str, str, str]],
    name_entries: list[tuple[str, str]] | None = None,
    params: C.ContaminationParams = _TEST_PARAMS,
) -> C.ContaminationReferences:
    matrix, doc_meta, ngram_sorted = C.build_reference_matrix(documents, params)
    blocklist = C.build_name_blocklist(name_entries or [], params)
    return C.ContaminationReferences(matrix, doc_meta, ngram_sorted, blocklist, params)


# --------------------------------------------------------------------------
# normalize_words / word_shingles
# --------------------------------------------------------------------------


def test_normalize_words_lowercases_and_strips_punctuation() -> None:
    assert C.normalize_words("Hello, World! It's 2026.") == ["hello", "world", "it's", "2026"]


def test_word_shingles_five_word_windows() -> None:
    words = "a b c d e f".split()
    assert C.word_shingles(words, 5) == ["a b c d e", "b c d e f"]


def test_word_shingles_empty_when_shorter_than_k() -> None:
    assert C.word_shingles(["a", "b"], 5) == []


# --------------------------------------------------------------------------
# MinHash near-duplicate
# --------------------------------------------------------------------------


def test_minhash_signature_identical_shingles_give_jaccard_one() -> None:
    a, b = C.permutation_coefficients(_TEST_PARAMS.minhash_permutations, _TEST_PARAMS.minhash_seed)
    words = C.normalize_words(REFERENCE_SENTENCE)
    shingles = C.word_shingles(words, _TEST_PARAMS.shingle_size)
    sig1 = C.minhash_signature(shingles, a, b)
    sig2 = C.minhash_signature(shingles, a, b)
    assert C.jaccard_estimate(sig1, sig2) == 1.0


def test_check_text_fires_near_duplicate_on_planted_overlap() -> None:
    refs = _build_refs([("ref-1", "erb_fireflies", REFERENCE_SENTENCE)])
    hits = C.check_text(REFERENCE_SENTENCE, refs)
    reasons = {h.reason: h for h in hits}
    assert "near_duplicate" in reasons
    assert reasons["near_duplicate"].exclusionary is True
    assert reasons["near_duplicate"].detail["jaccard"] == 1.0
    assert C.is_contaminated(hits) is True


def test_check_text_clean_text_has_no_near_duplicate_hit() -> None:
    refs = _build_refs([("ref-1", "erb_fireflies", REFERENCE_SENTENCE)])
    hits = C.check_text(UNRELATED_TEXT, refs)
    assert not any(h.reason == "near_duplicate" for h in hits)
    assert C.is_contaminated(hits) is False


def test_max_jaccard_against_review_band_is_not_exclusionary() -> None:
    # Hand-built matrix, bypassing text entirely, to land the score
    # precisely in the spec's 0.3-0.5 "hand review" band (here: 12/32 =
    # 0.375) rather than relying on a probabilistic text overlap.
    matrix = np.arange(32, dtype=np.uint64).reshape(1, 32) + 100  # all distinct, none zero
    query = np.arange(32, dtype=np.uint64) + 200  # all distinct from matrix's row
    matrix[0, :12] = query[:12]  # exactly 12 of 32 slots agree
    score, idx = C.max_jaccard_against(query, matrix)
    assert score == pytest.approx(12 / 32)
    assert _TEST_PARAMS.jaccard_review_threshold <= score < _TEST_PARAMS.jaccard_exclude_threshold
    assert idx == 0


def test_max_jaccard_against_empty_matrix_returns_zero() -> None:
    empty = np.zeros((0, 32), dtype=np.uint64)
    score, idx = C.max_jaccard_against(np.zeros(32, dtype=np.uint64), empty)
    assert (score, idx) == (0.0, -1)


# --------------------------------------------------------------------------
# Exact 13-word overlap
# --------------------------------------------------------------------------


def test_check_text_fires_exact_ngram_overlap_on_planted_shared_sentence() -> None:
    # The reference document IS the shared sentence (20 words -> 8
    # thirteen-word windows). The query embeds that exact sentence
    # inside a long, otherwise-unrelated document, diluting the overall
    # shingle overlap well below the near-duplicate threshold, so this
    # isolates the exact-overlap check.
    refs = _build_refs([("ref-1", "erb_jira", REFERENCE_SENTENCE)])
    padding = " ".join(f"filler{i} token{i} unrelated{i}" for i in range(60))
    query_text = f"{padding} {REFERENCE_SENTENCE} {padding}"
    hits = C.check_text(query_text, refs)
    reasons = {h.reason for h in hits}
    assert "exact_ngram_overlap" in reasons
    near_dup = next((h for h in hits if h.reason == "near_duplicate"), None)
    assert near_dup is None, "the shared sentence should be diluted below the near-dup threshold"


def test_check_text_clean_text_has_no_exact_ngram_overlap_hit() -> None:
    refs = _build_refs([("ref-1", "erb_jira", REFERENCE_SENTENCE)])
    hits = C.check_text(UNRELATED_TEXT, refs)
    assert not any(h.reason == "exact_ngram_overlap" for h in hits)


def test_ngram_window_hash_array_skips_stoplisted_boilerplate() -> None:
    words = C.normalize_words("please let me know if you have any questions about the pricing")
    hashes = C.ngram_window_hash_array(words, 13, stoplist=C.STOP_PHRASES)
    assert hashes.size == 0  # the only 13-word window IS the stoplisted phrase


def test_count_shared_ngrams_counts_only_matches() -> None:
    ref_sorted = np.array([1, 5, 9, 20], dtype=np.uint64)
    query = np.array([5, 5, 9, 999], dtype=np.uint64)
    assert C.count_shared_ngrams(query, ref_sorted) == 2  # 5 and 9, deduplicated


# --------------------------------------------------------------------------
# Canary
# --------------------------------------------------------------------------


def test_check_canary_fires_on_planted_guid() -> None:
    assert C.check_canary(f"see benchmark notes, canary {C.ERB_CANARY_GUID} appended") is True


def test_check_canary_case_insensitive() -> None:
    assert C.check_canary(C.ERB_CANARY_GUID.upper()) is True


def test_check_canary_clean_text_no_hit() -> None:
    assert C.check_canary(UNRELATED_TEXT) is False


def test_check_text_fires_canary_string() -> None:
    refs = _build_refs([])
    text = f"internal note: {C.ERB_CANARY_GUID} must never appear in training data"
    hits = C.check_text(text, refs)
    reasons = {h.reason: h for h in hits}
    assert "canary_string" in reasons
    assert reasons["canary_string"].exclusionary is True


# --------------------------------------------------------------------------
# Name blocklist
# --------------------------------------------------------------------------


def test_build_name_blocklist_buckets_by_source_and_length() -> None:
    # "orion" is a single token, so it needs an EXPLICIT background
    # table showing it's never seen in real text -- an empty dict says
    # exactly that (zero occurrences of everything).
    blocklist = C.build_name_blocklist(
        [("halvard", "Ashgrove Medical Center"), ("erb_hubspot", "Orion")],
        _TEST_PARAMS,
        background_doc_freq={},
    )
    assert set(blocklist["halvard"][3]) == {C.word_sequence_hash(("ashgrove", "medical", "center"))}
    assert set(blocklist["erb_hubspot"][1]) == {C.word_sequence_hash(("orion",))}


def test_build_name_blocklist_skips_short_single_word_names() -> None:
    blocklist = C.build_name_blocklist([("halvard", "Hi")], _TEST_PARAMS, background_doc_freq={})
    assert blocklist == {}


def test_build_name_blocklist_skips_names_longer_than_max_words() -> None:
    blocklist = C.build_name_blocklist([("halvard", "one two three four five")], _TEST_PARAMS)
    assert blocklist == {}


def test_build_name_blocklist_drops_single_token_with_no_background_table() -> None:
    # Coordinator review, 2026-09-22: fail CLOSED when nobody checked the
    # word against real text -- a multi-token name is unaffected.
    blocklist = C.build_name_blocklist(
        [("halvard", "Ashgrove Medical Center"), ("erb_hubspot", "Orion")], _TEST_PARAMS
    )
    assert "erb_hubspot" not in blocklist
    assert set(blocklist["halvard"][3]) == {C.word_sequence_hash(("ashgrove", "medical", "center"))}


def test_build_name_blocklist_drops_single_token_seen_in_background() -> None:
    # "memorial" -- one of the coordinator's own named examples -- is
    # dropped once the background table shows it occurring in real text,
    # even well below any percentage-style threshold (10 of 10,948
    # documents in the real background this module calibrates against).
    blocklist = C.build_name_blocklist(
        [("halvard", "Memorial")], _TEST_PARAMS, background_doc_freq={"memorial": 10}
    )
    assert blocklist == {}


def test_build_name_blocklist_drops_generic_role_phrase() -> None:
    # "the medical center" is a real alias in corpus/truth-table.yaml for
    # Ashgrove Medical Center -- generic on purpose, per the coordinator.
    blocklist = C.build_name_blocklist([("halvard", "the medical center")], _TEST_PARAMS)
    assert blocklist == {}


def test_is_distinctive_name_multitoken_passes_without_background_table() -> None:
    assert C.is_distinctive_name("Ashgrove Medical Center", None) is True


def test_is_distinctive_name_rejects_article_prefix_even_with_zero_background() -> None:
    assert C.is_distinctive_name("the new pump", {}) is False


def test_is_distinctive_name_single_token_needs_zero_background_count() -> None:
    assert C.is_distinctive_name("Tidewater", {"tidewater": 0}) is True
    assert C.is_distinctive_name("Link", {"link": 194}) is False


def test_check_text_fires_blocklisted_name_on_planted_name() -> None:
    refs = _build_refs([], name_entries=[("halvard", "Ashgrove Medical Center")])
    hits = C.check_text("We spoke with Ashgrove Medical Center about the renewal.", refs)
    reasons = {h.reason: h for h in hits}
    assert "blocklisted_name" in reasons
    assert reasons["blocklisted_name"].exclusionary is True
    assert reasons["blocklisted_name"].detail["source"] == "halvard"


def test_check_text_clean_text_has_no_blocklisted_name_hit() -> None:
    refs = _build_refs([], name_entries=[("halvard", "Ashgrove Medical Center")])
    hits = C.check_text(UNRELATED_TEXT, refs)
    assert not any(h.reason == "blocklisted_name" for h in hits)


# --------------------------------------------------------------------------
# Name blocklist: titles and ordinary phrases (WP11 pilot dry run: 25 of 432
# sources dropped on "customer success" and "procurement manager")
# --------------------------------------------------------------------------

# Stand-in for var/data/converted: ordinary writing in which the fictional
# exam names never occur.
BACKGROUND_TEXTS = [
    "The customer called support about her account and asked for a manager.",
    "Procurement sent the invoice to accounts payable on Monday.",
    "Our customer success review with the director is on the calendar.",
    "The engineer said the pump alarm was a false positive.",
    "Please forward the medical records to the nursing office.",
]
NAME_ENTRIES = [
    ("erb_fireflies", "Customer Success"),
    ("erb_gmail", "Procurement Manager"),
    ("erb_gmail", "Accounts Payable"),
    ("halvard", "HalvardCare"),
    ("halvard", "Tidewater T3"),
    ("halvard", "Halvard Medical"),
]


def _background() -> tuple[dict[str, int], int]:
    phrases = [C.normalize_name_words(name) for _source, name in NAME_ENTRIES]
    return C.background_frequency(BACKGROUND_TEXTS, phrases)


def _refs_with_background() -> C.ContaminationReferences:
    freq, docs = _background()
    blocklist = C.build_name_blocklist(
        NAME_ENTRIES, _TEST_PARAMS, background_doc_freq=freq, background_doc_count=docs
    )
    empty = np.zeros((0, _TEST_PARAMS.minhash_permutations), dtype=np.uint64)
    return C.ContaminationReferences(
        empty, [], np.array([], dtype=np.uint64), blocklist, _TEST_PARAMS
    )


def test_background_frequency_counts_candidate_phrases_once_per_document() -> None:
    freq, docs = C.background_frequency(
        ["Accounts payable, then accounts   payable again.", "accounts only", "payable accounts"],
        [("accounts", "payable"), ("never", "seen")],
    )
    assert docs == 3
    assert freq["accounts payable"] == 1
    assert freq["accounts"] == 3
    assert "never seen" not in freq


def test_job_titles_are_role_phrases() -> None:
    freq = {"procurement": 24, "manager": 556, "customer": 4152, "success": 133,
            "director": 117, "of": 9000, "nursing": 30}  # fmt: skip
    for title in ("Procurement Manager", "Customer Success", "Director of Nursing", "CISO"):
        assert C.name_drop_reason(title, freq, 10_948) == "role_phrase", title


def test_team_and_department_phrases_are_role_phrases() -> None:
    """The WP7 mini world lost records to ERB Gmail's "engineering team"."""
    freq = {"engineering": 212, "billing": 95, "it": 3000, "finance": 140, "front": 300,
            "nursing": 30, "clinical": 88, "help": 900, "support": 1405}  # fmt: skip
    for phrase in (
        "engineering team",
        "Billing Department",
        "IT Group",
        "Finance Office",
        "Front Desk",
        "Nursing Staff",
        "Clinical Committee",
        "Help Desk Support",
    ):
        assert C.name_drop_reason(phrase, freq, 10_948) == "role_phrase", phrase
    # A distinctive word keeps the entry a name even next to a team word.
    assert C.name_drop_reason("Tidewater Group", {"group": 700}, 10_948) is None
    assert C.name_drop_reason("Halvard Engineering Team", freq, 10_948) is None


def test_check_text_ignores_team_phrases_but_still_finds_exam_names() -> None:
    entries = [("erb_gmail", "engineering team"), ("halvard", "Halvard Medical")]
    freq, docs = C.background_frequency(
        ["The engineering group met on Monday.", "Ask the team about it."],
        [C.normalize_name_words(n) for _s, n in entries],
    )
    blocklist = C.build_name_blocklist(
        entries, _TEST_PARAMS, background_doc_freq=freq, background_doc_count=docs
    )
    empty = np.zeros((0, _TEST_PARAMS.minhash_permutations), dtype=np.uint64)
    refs = C.ContaminationReferences(
        empty, [], np.array([], dtype=np.uint64), blocklist, _TEST_PARAMS
    )
    assert C.check_text("I'll loop in the engineering team today.", refs) == []
    assert [h.reason for h in C.check_text("Halvard Medical called", refs)] == ["blocklisted_name"]


def test_a_title_word_does_not_hide_a_distinctive_name() -> None:
    # "tidewater" never occurs in the background, so the entry is a name, not a title.
    assert C.name_drop_reason("Tidewater Support", {"support": 1405}, 10_948) is None


def test_phrase_seen_in_background_is_dropped_even_when_its_words_are_rare() -> None:
    freq = {"accounts": 30, "payable": 28, "accounts payable": 7}
    assert C.name_drop_reason("Accounts Payable", freq, 10_948) == "phrase_in_background"


def test_name_made_of_common_words_is_dropped() -> None:
    # Both words in at least 1% of 10,948 documents (110), the phrase itself never.
    freq = {"legal": 404, "review": 1694}
    assert C.name_drop_reason("Legal Review", freq, 10_948) == "common_words"
    # "park" (63 documents) is under 1%, so June Park stays a name.
    assert C.name_drop_reason("June Park", {"june": 176, "park": 63}, 10_948) is None


def test_exam_names_stay_on_the_blocklist_against_a_real_background() -> None:
    freq, docs = _background()
    for name in ("HalvardCare", "Tidewater T3", "Halvard Medical"):
        assert C.name_drop_reason(name, freq, docs) is None, name
    for name in ("Customer Success", "Procurement Manager", "Accounts Payable"):
        assert C.name_drop_reason(name, freq, docs) is not None, name


def test_check_text_ignores_job_titles_but_still_finds_exam_names() -> None:
    refs = _refs_with_background()
    header = "From: Tovin Marle, Procurement Manager\nCc: Ola Brand (Customer Success)\n"
    assert C.check_text(header + "Sending this to accounts payable today.", refs) == []
    for text in ("we renewed the HalvardCare plan", "the Tidewater T3 alarms again"):
        hits = C.check_text(text, refs)
        assert [h.reason for h in hits] == ["blocklisted_name"], text
        assert hits[0].detail["source"] == "halvard"


# --------------------------------------------------------------------------
# LSH band hashes (spec parameters, not used by the gate itself)
# --------------------------------------------------------------------------


def test_lsh_band_hashes_deterministic_and_correct_length() -> None:
    sig = np.arange(32, dtype=np.uint64)
    bands1 = C.lsh_band_hashes(sig, bands=8, rows=4)
    bands2 = C.lsh_band_hashes(sig, bands=8, rows=4)
    assert len(bands1) == 8
    assert bands1 == bands2  # deterministic across calls


def test_lsh_band_hashes_rejects_mismatched_shape() -> None:
    sig = np.arange(32, dtype=np.uint64)
    with pytest.raises(ValueError):
        C.lsh_band_hashes(sig, bands=8, rows=5)


def test_contamination_params_rejects_mismatched_bands_rows() -> None:
    with pytest.raises(ValueError):
        C.ContaminationParams(minhash_permutations=128, lsh_bands=16, lsh_rows=7)


# --------------------------------------------------------------------------
# Save / load round-trip
# --------------------------------------------------------------------------


def test_save_and_load_references_round_trip(tmp_path) -> None:
    documents = [
        ("ref-1", "halvard", REFERENCE_SENTENCE),
        ("ref-2", "erb_gmail", UNRELATED_TEXT),
    ]
    matrix, doc_meta, ngram_sorted = C.build_reference_matrix(documents, _TEST_PARAMS)
    blocklist = C.build_name_blocklist(
        [("halvard", "Ashgrove Medical Center"), ("erb_hubspot", "Orion")],
        _TEST_PARAMS,
        background_doc_freq={},
    )
    out_dir = tmp_path / "contamination"
    source_counts = {"halvard": 1, "erb_gmail": 1}
    fingerprints = C.fingerprint_sources({"halvard": _exam_dir(tmp_path)})
    manifest = C.save_references(
        out_dir,
        matrix,
        doc_meta,
        ngram_sorted,
        blocklist,
        _TEST_PARAMS,
        source_counts,
        source_fingerprints=fingerprints,
    )
    assert manifest["reference_docs_total"] == 2
    assert manifest["source_fingerprints"] == fingerprints
    assert C.references_exist(out_dir) is True

    loaded = C.load_references(out_dir)
    assert loaded.params == _TEST_PARAMS
    assert loaded.doc_meta == doc_meta
    np.testing.assert_array_equal(loaded.minhash_matrix, matrix)
    np.testing.assert_array_equal(loaded.ngram_sorted, np.sort(ngram_sorted))
    assert loaded.name_blocklist == blocklist

    # And the loaded references behave identically to the in-memory ones.
    hits = C.check_text(REFERENCE_SENTENCE, loaded)
    assert any(h.reason == "near_duplicate" for h in hits)


def test_references_exist_false_for_empty_directory(tmp_path) -> None:
    assert C.references_exist(tmp_path / "nothing-here") is False


def test_load_references_or_refuse_raises_when_missing_and_not_allowed(tmp_path) -> None:
    with pytest.raises(C.ContaminationSignaturesMissing):
        C.load_references_or_refuse(tmp_path / "missing", allow_missing=False)


def test_load_references_or_refuse_returns_none_when_allowed(tmp_path) -> None:
    assert C.load_references_or_refuse(tmp_path / "missing", allow_missing=True) is None


def test_load_references_or_refuse_loads_when_present(tmp_path) -> None:
    documents = [("ref-1", "halvard", REFERENCE_SENTENCE)]
    matrix, doc_meta, ngram_sorted = C.build_reference_matrix(documents, _TEST_PARAMS)
    out_dir = tmp_path / "contamination"
    C.save_references(
        out_dir,
        matrix,
        doc_meta,
        ngram_sorted,
        {},
        _TEST_PARAMS,
        {"halvard": 1},
        source_fingerprints=C.fingerprint_sources({"halvard": _exam_dir(tmp_path)}),
    )
    refs = C.load_references_or_refuse(out_dir, allow_missing=False)
    assert refs is not None
    assert refs.minhash_matrix.shape[0] == 1


# --------------------------------------------------------------------------
# Source fingerprints (guardrails review 2026-09-22, row 31): a regenerated exam used to
# leave stale signatures in force silently.
# --------------------------------------------------------------------------


def _exam_dir(root: Path) -> Path:
    exam = root / "exam"
    (exam / "calls").mkdir(parents=True, exist_ok=True)
    (exam / "calls" / "S-call-001.txt").write_text(REFERENCE_SENTENCE, encoding="utf-8")
    return exam


def _saved_signatures(tmp_path: Path) -> tuple[Path, Path]:
    exam = _exam_dir(tmp_path)
    matrix, doc_meta, ngram_sorted = C.build_reference_matrix(
        [("ref-1", "halvard", REFERENCE_SENTENCE)], _TEST_PARAMS
    )
    out_dir = tmp_path / "contamination"
    C.save_references(
        out_dir,
        matrix,
        doc_meta,
        ngram_sorted,
        {},
        _TEST_PARAMS,
        {"halvard": 1},
        source_fingerprints=C.fingerprint_sources({"halvard": exam}),
    )
    return out_dir, exam


def test_fingerprint_follows_content_and_file_set_but_not_dotfiles(tmp_path: Path) -> None:
    exam = _exam_dir(tmp_path)
    first = C.fingerprint_path(exam)
    assert first["files"] == 1
    (exam / ".DS_Store").write_bytes(b"finder")
    assert C.fingerprint_path(exam) == first
    (exam / "calls" / "S-call-001.txt").write_text(REFERENCE_SENTENCE + " edited")
    edited = C.fingerprint_path(exam)
    assert edited["sha256"] != first["sha256"]
    (exam / "calls" / "S-call-002.txt").write_text("another call")
    assert C.fingerprint_path(exam)["files"] == 2
    assert C.fingerprint_path(tmp_path / "missing")["files"] == 0


def test_fresh_signatures_have_no_stale_sources(tmp_path: Path) -> None:
    out_dir, _ = _saved_signatures(tmp_path)
    assert C.stale_signature_sources(out_dir) == []


def test_signatures_are_refused_once_the_exam_changes(tmp_path: Path) -> None:
    out_dir, exam = _saved_signatures(tmp_path)
    (exam / "calls" / "S-call-002.txt").write_text("a regenerated exam call", encoding="utf-8")
    (problem,) = C.stale_signature_sources(out_dir)
    assert problem.startswith("halvard") and "1 files then, 2 now" in problem
    with pytest.raises(C.ContaminationSignaturesStale, match="stale"):
        C.load_references_or_refuse(out_dir, allow_missing=False)
    # The builder's existing handler catches the parent class and exits 1.
    assert issubclass(C.ContaminationSignaturesStale, C.ContaminationSignaturesMissing)
    assert C.load_references_or_refuse(out_dir, allow_missing=True) is None


def test_signatures_with_no_fingerprints_are_refused(tmp_path: Path) -> None:
    # Every manifest built before fingerprints existed looks like this.
    out_dir, _ = _saved_signatures(tmp_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["source_fingerprints"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(C.ContaminationSignaturesStale, match="no fingerprint"):
        C.load_references_or_refuse(out_dir, allow_missing=False)
