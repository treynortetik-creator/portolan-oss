"""Tests for portolan.extract.segment."""

from __future__ import annotations

from portolan.extract.schema import (
    Coverage,
    EntityCoercion,
    ExtractionResult,
    Fact,
    RejectedFact,
)
from portolan.extract.segment import merge_segment_results, segment


def _transcript_lines(n: int, seconds_per_line: int = 5) -> list[str]:
    lines = []
    for i in range(n):
        total_seconds = i * seconds_per_line
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] line number {i} of the transcript")
    return lines


def test_segment_short_text_is_a_single_segment() -> None:
    text = "[00:00:01] hello\n[00:00:05] world"
    segments = segment(text, max_tokens=4000)
    assert len(segments) == 1
    assert segments[0].index == 1
    assert segments[0].total == 1
    assert segments[0].start_locator == "00:00:01"
    assert segments[0].end_locator == "00:00:05"


def test_segment_never_splits_a_line() -> None:
    lines = _transcript_lines(500)
    text = "\n".join(lines)
    segments = segment(text, max_tokens=200, overlap_tokens=20)
    assert len(segments) > 1
    all_segment_lines: set[str] = set()
    for seg in segments:
        for line in seg.text.splitlines():
            all_segment_lines.add(line)
    # Every line that appears in any segment is an exact, complete original
    # line -- never a fragment of one.
    assert all_segment_lines.issubset(set(lines))


def test_segment_covers_every_line_at_least_once() -> None:
    lines = _transcript_lines(500)
    text = "\n".join(lines)
    segments = segment(text, max_tokens=200, overlap_tokens=20)
    covered: set[str] = set()
    for seg in segments:
        covered.update(seg.text.splitlines())
    assert covered == set(lines)


def test_segment_consecutive_segments_overlap() -> None:
    lines = _transcript_lines(500)
    text = "\n".join(lines)
    segments = segment(text, max_tokens=200, overlap_tokens=40)
    assert len(segments) >= 2
    first_lines = set(segments[0].text.splitlines())
    second_lines = set(segments[1].text.splitlines())
    assert first_lines & second_lines, "expected overlap between consecutive segments"


def test_segment_zero_overlap_means_no_shared_lines() -> None:
    lines = _transcript_lines(500)
    text = "\n".join(lines)
    segments = segment(text, max_tokens=200, overlap_tokens=0)
    assert len(segments) >= 2
    first_lines = set(segments[0].text.splitlines())
    second_lines = set(segments[1].text.splitlines())
    assert not (first_lines & second_lines)


def test_segment_respects_injected_token_counter() -> None:
    # A token counter that always returns 1 means every line is "cheap", so
    # a much larger number of lines fits per segment than the default
    # len//4 approximation would allow.
    lines = _transcript_lines(100)
    text = "\n".join(lines)
    segments_default = segment(text, max_tokens=50)
    segments_custom = segment(text, max_tokens=50, count_tokens=lambda line: 1)
    assert len(segments_custom) < len(segments_default)


def test_segment_hard_splits_a_single_line_that_exceeds_max_tokens() -> None:
    # Bug hunt 2026-09-18, finding 11: RSS items and CRM rows are prepared
    # as a single line -- a line longer than max_tokens used to become
    # its own unbounded segment (the exact repro from the bug-hunt doc:
    # segment("x"*400_000, max_tokens=4000) -> 1 segment, ~100,000
    # estimated tokens).
    text = "x" * 400_000
    segments = segment(text, max_tokens=4000)

    assert len(segments) > 1
    for seg in segments:
        # "Roughly max_tokens", same tolerance the whole module already
        # promises: a bucket's real budget is the SUM of its lines' own
        # counter() calls, not counter() of the "\n"-joined segment text
        # -- a handful of join newlines can nudge the joined estimate a
        # token or two over. What this fix actually closes is the
        # UNBOUNDED case (previously ~25x over budget, not ~1 token).
        assert len(seg.text) // 4 <= 4000 + 10, f"segment {seg.index} is {len(seg.text)} chars"
    # Nothing lost -- the first segment starts at the beginning of the
    # line and the last ends at its end (segments legitimately overlap
    # each other by design, so a naive full-text join isn't the right
    # check here).
    assert segments[0].text[0] == text[0]
    assert segments[-1].text[-1] == text[-1]
    assert segments[0].index == 1
    assert segments[-1].index == segments[-1].total == len(segments)


def test_segment_hard_split_respects_an_injected_token_counter() -> None:
    # The hard split must fit whatever counter is injected, not just the
    # default len//4 estimate.
    text = "y" * 1000
    segments = segment(text, max_tokens=50, count_tokens=lambda line: len(line))

    assert len(segments) > 1
    for seg in segments:
        # Same small join-newline slack as the test above.
        assert len(seg.text) <= 50 + 5
    assert segments[0].text[0] == text[0]
    assert segments[-1].text[-1] == text[-1]


def test_segment_line_within_budget_is_never_split() -> None:
    # No regression for the common case: a line already within max_tokens
    # passes through untouched, one segment.
    text = "a short line well within budget"
    segments = segment(text, max_tokens=4000)
    assert len(segments) == 1
    assert segments[0].text == text


def test_segment_indices_and_totals_are_consistent() -> None:
    lines = _transcript_lines(300)
    text = "\n".join(lines)
    segments = segment(text, max_tokens=150, overlap_tokens=15)
    assert [s.index for s in segments] == list(range(1, len(segments) + 1))
    assert all(s.total == len(segments) for s in segments)


def _fact(
    id_: str, claim: str, locator: str, entities: list[str], type_: str = "complaint"
) -> Fact:
    return Fact.model_validate(
        {
            "id": id_,
            "type": type_,
            "claim": claim,
            "entities": entities,
            "locator": locator,
            "quote": "we had three late deliveries last month" if type_ == "complaint" else None,
            "sentiment": -2 if type_ == "complaint" else 0,
            "confidence": 0.9,
        }
    )


def _result(
    source_id: str, facts: list[Fact], segments_total: int, summary: str = "summary"
) -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "schema_version": "0.1",
            "source_id": source_id,
            "summary": summary,
            "entities": [],
            "facts": [f.model_dump(mode="json") for f in facts],
            "actions": [],
            "unknown_entities": [],
            "coverage": {"segments_read": 1, "segments_total": segments_total},
        }
    )


def test_merge_segment_results_dedups_overlapping_facts_and_keeps_earliest_locator() -> None:
    # The same complaint appears in segment 1 (near its end) and segment 2
    # (near its start), as it would from overlap -- same type, same
    # entities, close timestamps, near-identical claim text.
    fact_seg1 = _fact(
        "f1",
        "Ashgrove Medical Center received three late deliveries in August 2026.",
        "00:14:32",
        ["ashgrove-medical", "x-series-pump"],
    )
    fact_seg2 = _fact(
        "f1",
        "Ashgrove Medical Center received three late deliveries in August 2026.",
        "00:14:40",
        ["ashgrove-medical", "x-series-pump"],
    )
    unique_to_seg2 = _fact(
        "f2",
        "Ashgrove asked whether the pump supports Epic integration.",
        "00:20:00",
        ["ashgrove-medical"],
        type_="question",
    )

    result1 = _result("call-1", [fact_seg1], segments_total=2)
    result2 = _result("call-1", [fact_seg2, unique_to_seg2], segments_total=2)

    merged = merge_segment_results([result1, result2])

    assert merged.source_id == "call-1"
    assert merged.coverage.segments_read == 2
    assert merged.coverage.segments_total == 2
    assert len(merged.facts) == 2  # the duplicate collapsed to one

    complaint = next(f for f in merged.facts if f.type.value == "complaint")
    assert complaint.locator == "00:14:32"  # earliest of the two kept


def test_merge_segment_results_keeps_distinct_facts_separate() -> None:
    fact_a = _fact(
        "f1",
        "Ashgrove Medical Center received three late deliveries in August 2026.",
        "00:14:32",
        ["ashgrove-medical"],
    )
    fact_b = _fact(
        "f1",
        "Ashgrove's CFO froze new equipment purchases until Q1 2027.",
        "01:02:00",
        ["ashgrove-medical"],
        type_="risk",
    )
    result1 = _result("call-1", [fact_a], segments_total=2)
    result2 = _result("call-1", [fact_b], segments_total=2)

    merged = merge_segment_results([result1, result2])
    assert len(merged.facts) == 2
    ids = {f.id for f in merged.facts}
    assert ids == {"f1", "f2"}  # renumbered sequentially


def test_merge_segment_results_merges_entities_preferring_resolved_registry_id() -> None:
    result1 = ExtractionResult.model_validate(
        {
            "schema_version": "0.1",
            "source_id": "call-1",
            "summary": "s",
            "entities": [
                {
                    "mention": "Dr. Alvarez",
                    "kind": "contact",
                    "registry_id": None,
                    "confidence": 0.6,
                }
            ],
            "facts": [],
            "actions": [],
            "unknown_entities": [],
            "coverage": {"segments_read": 1, "segments_total": 2},
        }
    )
    result2 = ExtractionResult.model_validate(
        {
            "schema_version": "0.1",
            "source_id": "call-1",
            "summary": "s2",
            "entities": [
                {
                    "mention": "Dr. Alvarez",
                    "kind": "contact",
                    "registry_id": "contact:dr-alvarez",
                    "confidence": 0.8,
                }
            ],
            "facts": [],
            "actions": [],
            "unknown_entities": [],
            "coverage": {"segments_read": 1, "segments_total": 2},
        }
    )
    merged = merge_segment_results([result1, result2])
    assert len(merged.entities) == 1
    assert merged.entities[0].registry_id == "contact:dr-alvarez"


def test_merge_segment_results_unions_unknown_entities_without_duplicates() -> None:
    result1 = _result("call-1", [], segments_total=2)
    result1 = result1.model_copy(
        update={"unknown_entities": ["Corvane Medical Center (possibly a competitor)"]}
    )
    result2 = _result("call-1", [], segments_total=2)
    result2 = result2.model_copy(
        update={
            "unknown_entities": [
                "Corvane Medical Center (possibly a competitor)",
                "Acme Corp",
            ]
        }
    )

    merged = merge_segment_results([result1, result2])
    assert merged.unknown_entities == [
        "Corvane Medical Center (possibly a competitor)",
        "Acme Corp",
    ]


def test_merge_segment_results_carries_rejected_and_coercions_through() -> None:
    # Bug hunt 2026-09-18, finding 6: `merge_segment_results` used to
    # rebuild the merged `ExtractionResult` without `rejected`/
    # `coercions` at all -- every segment's own `parse_lenient` call had
    # already computed them, but the merge silently dropped both, even
    # though fact rule 4 requires a rejected fact to be "counted in the
    # coverage report."
    rejected1 = RejectedFact(index=0, error="missing locator", raw={"claim": "bad fact"})
    rejected2 = RejectedFact(index=1, error="bad type", raw={"claim": "also bad"}, kind="action")
    coercion1 = EntityCoercion(
        index=0,
        mention="Priya Natarajan",
        field="entities",
        original={"mention": "x"},
        applied=None,
    )

    result1 = _result("call-1", [], segments_total=2).model_copy(
        update={"rejected": [rejected1], "coercions": [coercion1]}
    )
    result2 = _result("call-1", [], segments_total=2).model_copy(
        update={"rejected": [rejected2], "coercions": []}
    )

    merged = merge_segment_results([result1, result2])

    assert merged.rejected == [rejected1, rejected2]
    assert merged.coercions == [coercion1]


def test_merge_segment_results_picks_the_longest_summary() -> None:
    result1 = _result("call-1", [], segments_total=2, summary="short")
    result2 = _result(
        "call-1", [], segments_total=2, summary="a much longer and more informative summary"
    )
    merged = merge_segment_results([result1, result2])
    assert merged.summary == "a much longer and more informative summary"


def test_merge_segment_results_uses_caller_supplied_segments_total() -> None:
    # F10: three segments actually ran, but segment 1 hallucinated "1" as
    # its own self-reported total. The caller that did the splitting knows
    # the truth and must win over trusting results[0].
    result1 = _result("call-1", [], segments_total=1)  # wrong: hallucinated
    result2 = _result("call-1", [], segments_total=1)
    result3 = _result("call-1", [], segments_total=1)

    merged = merge_segment_results([result1, result2, result3], segments_total=3)

    assert merged.coverage.segments_total == 3
    assert merged.coverage.segments_read == 3


def test_merge_segment_results_falls_back_to_max_without_an_explicit_total() -> None:
    # F10's "at minimum max(...)": when no caller-supplied total is given,
    # the largest self-reported total among the segments beats trusting
    # segment 1 alone.
    result1 = _result("call-1", [], segments_total=1)  # segment 1 undercounts
    result2 = _result("call-1", [], segments_total=3)  # segment 2 got it right
    result3 = _result("call-1", [], segments_total=2)

    merged = merge_segment_results([result1, result2, result3])

    assert merged.coverage.segments_total == 3


# ---------------------------------------------------------------------------
# Two facts one extraction call wrote are two facts.
# ---------------------------------------------------------------------------


def _alike_fact(fid: str, language: str, locator: str) -> Fact:
    return Fact(
        id=fid,
        type="request",
        claim=f"Ferrowick Bottling asked for a {language}-language troubleshooting guide.",
        entities=["A-1"],
        locator=locator,
        sentiment=0,
        confidence=0.9,
    )


def _one_result(facts: list[Fact]) -> ExtractionResult:
    return ExtractionResult(
        source_id="S-1",
        summary="s",
        facts=facts,
        coverage=Coverage(segments_read=1, segments_total=1),
    )


def test_similar_facts_from_one_call_are_not_folded_together():
    result = merge_segment_results(
        [_one_result([_alike_fact("f1", "Vietnamese", "p2"), _alike_fact("f2", "Spanish", "p4")])]
    )
    assert [f.locator for f in result.facts] == ["p2", "p4"]


def test_an_exact_repeat_within_one_call_is_still_collapsed():
    result = merge_segment_results(
        [
            _one_result(
                [_alike_fact("f1", "Vietnamese", "p2"), _alike_fact("f2", "Vietnamese", "p2")]
            )
        ]
    )
    assert len(result.facts) == 1


def test_the_same_fact_from_two_passes_is_one_fact():
    first = _one_result([_alike_fact("f1", "Vietnamese", "p2")])
    reworded = _alike_fact("f1", "Vietnamese", "p2").model_copy(
        update={
            "claim": "Ferrowick Bottling requested a Vietnamese-language troubleshooting guide."
        }
    )
    result = merge_segment_results([first, _one_result([reworded])])
    assert len(result.facts) == 1


def test_a_merged_fact_keeps_its_quote_with_the_locator_it_came_from():
    late = Fact(
        id="f1",
        type="complaint",
        claim="Quenby's cartridge order arrived three days late.",
        entities=["A-1"],
        locator="00:06:40",
        quote="three days behind schedule",
        sentiment=-1,
        confidence=0.9,
    )
    early = late.model_copy(update={"locator": "00:06:05", "quote": "that late order last week"})
    result = merge_segment_results([_one_result([late]), _one_result([early])])
    assert len(result.facts) == 1
    assert (result.facts[0].locator, result.facts[0].quote) == (
        "00:06:05",
        "that late order last week",
    )
