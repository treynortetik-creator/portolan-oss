"""The cited-answer repair measurement (`portolan.bench.erb_answers`,
2026-09-23): gate reasons, the numbers guard over verified quotes, the
support judge, one repair call whose restatements are gated again, and the
per-question evaluation. No network (MockTransport only), no ERB text."""

from __future__ import annotations

import json

import httpx
import pytest

from portolan.bench.erb_answers import (
    ANSWER_MODES,
    NOT_IN_SOURCES,
    CitedSentence,
    OpenRouterChat,
    estimate_evaluation_usd,
    estimate_generation_usd,
    evaluate_question,
    evaluation_item,
    gate_cited_answer,
    make_chat_support_judge,
    parse_numbered,
    parse_support_verdict,
    run_cited,
    support_reference,
)
from portolan.bench.erb_search import BudgetExceeded

D1 = "dsid_" + "a" * 32
D2 = "dsid_" + "b" * 32
D3 = "dsid_" + "c" * 32

DOCS = {
    D1: "Rollout plan\n\nGo-live is 2028-07-22. The canary runs 7 days at a 10/90 split.",
    D2: "Pump survey\n\nAcme runs 47 pumps across three sites.",
}


def _routed_chat(route, calls, *, run_cap=None):
    """A chat whose reply is `route(system, user)`; every request body is
    appended to `calls`."""

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), None)
        user = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": route(system, user)}}],
                "usage": {"cost": 0.01},
            },
        )

    return OpenRouterChat(
        "sk-test",
        meter=None,
        corpus="erb_exam",
        budget_usd=100.0,
        run_cap_usd=run_cap,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


def test_gate_names_the_reason_for_every_segment():
    sentences = [
        CitedSentence("Go-live is on 2028-07-22", [(D1, "Go-live is 2028-07-22.")]),
        CitedSentence("An uncited claim"),
        CitedSentence("Cites a stranger", [(D3, "anything")]),
        CitedSentence("Misquotes", [(D1, "Go-live is 2029-01-01.")]),
        CitedSentence("Acme runs 48 pumps", [(D2, "Acme runs 47 pumps")]),
        CitedSentence(NOT_IN_SOURCES),
        CitedSentence("", [(D1, "Go-live is 2028-07-22.")]),
    ]
    result = gate_cited_answer(sentences, DOCS)
    assert result.reasons == [
        None,
        "uncited",
        "doc_not_in_context",
        "quote_not_found",
        "number_not_in_quote",
        "abstained",
        "no_text",
    ]
    assert [s.text for s in result.kept] == ["Go-live is on 2028-07-22"]
    assert result.abstained == 1
    assert len(result.dropped) == 5


def test_numbers_guard_reads_only_the_quotes_that_passed():
    """47 is only in the misquote, not in the verified quote: dropped."""
    s = CitedSentence("Acme runs 47 pumps", [(D2, "Acme runs"), (D1, "Acme runs 47 pumps")])
    assert gate_cited_answer([s], DOCS).reasons == ["number_not_in_quote"]
    assert gate_cited_answer([s], DOCS, numbers=False).reasons == [None]


def test_support_judge_drops_what_its_quotes_do_not_entail_and_errors_count_as_no():
    seen = []

    def judge(*, sentence, reference_text):
        seen.append((sentence, reference_text))
        return {"Acme runs 47 pumps": "supported", "Acme is growing fast": "not_supported"}.get(
            sentence
        )

    sentences = [
        CitedSentence("Acme runs 47 pumps", [(D2, "Acme runs 47 pumps")]),
        CitedSentence("Acme is growing fast", [(D2, "Acme runs 47 pumps")]),
        CitedSentence("Acme has three sites", [(D2, "across three sites")]),
        CitedSentence("Uncited, never judged"),
    ]
    result = gate_cited_answer(sentences, DOCS, support=judge, titles={D2: "Pump survey"})
    assert result.reasons == [None, "not_supported", "not_supported", "uncited"]
    assert result.support_verdicts == ["supported", "not_supported", None, None]
    assert len(seen) == 3
    assert all('from "Pump survey"' in ref for _s, ref in seen)


def test_support_reference_lists_each_quote_with_its_title():
    ref = support_reference([(D1, "q one"), (D2, "q two")], {D1: "Rollout plan"})
    assert ref == '- from "Rollout plan": "q one"\n- "q two"'


def test_support_reference_can_show_each_quote_inside_its_passage():
    docs = {D1: "Call notes\n\nJonas: I will create\n  JIRA SUP-3421 to track infra changes. Bye."}
    ref = support_reference(
        [(D1, "i will create JIRA SUP-3421"), (D1, "not in the text")],
        {D1: "Call notes"},
        doc_texts=docs,
        context_chars=9,
    )
    # Found through whitespace and case folding, shown with the original text;
    # a quote that cannot be located is shown bare.
    assert ref.splitlines() == [
        '- from "Call notes": ...Jonas: ⟦I will create JIRA SUP-3421⟧ to track...',
        '- from "Call notes": "not in the text"',
    ]


def test_parse_support_verdict_falls_back_to_the_bare_field():
    assert parse_support_verdict('{"verdict": "supported", "reason": "ok"}') == "supported"
    assert (
        parse_support_verdict('{"verdict": "not_supported", "reason": "says "x" not y"}')
        == "not_supported"
    )
    assert parse_support_verdict('{"verdict": "partly"}') is None
    assert parse_support_verdict("no verdict") is None


def test_parse_numbered_joins_continuation_lines():
    text = "1. first line\ncontinued\n2) second\n\n3. NOT_IN_SOURCES\nnoise after"
    assert parse_numbered(text) == {
        1: "first line continued",
        2: "second",
        3: "NOT_IN_SOURCES noise after",
    }


def _repair_route(reply):
    def route(system, user):
        assert "## Removed sentences" in user
        return reply

    return route


def test_repair_restates_in_place_and_regates_every_restatement():
    raw = (
        f'Acme is huge [{D2}: "not in the document"]. '
        f'Go-live is 2028-07-22 [{D1}: "Go-live is 2028-07-22."]. '
        f'Acme runs 47 pumps at 9 sites [{D2}: "Acme runs 47 pumps"]. '
        "2028-07-15"
    )
    reply = (
        f"1. {NOT_IN_SOURCES}\n"
        f'2. Acme runs 47 pumps at 12 sites [{D2}: "Acme runs 47 pumps across three sites."]\n'
        f"3. The canary starts 7 days before the 2028-07-22 go-live "
        f'[{D1}: "Go-live is 2028-07-22."] [{D1}: "The canary runs 7 days"]'
    )
    calls: list = []
    chat = _routed_chat(_repair_route(reply), calls)
    run = run_cited(chat, "gen/model", "When does it start?", "CTX", DOCS, raw=raw, repair=True)

    assert run.first.reasons == ["quote_not_found", None, "number_not_in_quote", "uncited"]
    assert len(calls) == 1
    user = calls[0]["messages"][-1]["content"]
    assert "2. Acme runs 47 pumps at 9 sites -- removed because it states a number" in user
    assert "3. 2028-07-15 -- removed because it carried no citation" in user
    assert "- Go-live is 2028-07-22" in user  # kept sentences shown, not re-asked
    # 1 abstains, 2's restatement states "12", which its quote does not (the
    # numbers guard applies to repairs unchanged), 3 is restored in place.
    assert [s.text for s in run.final] == [
        "Go-live is 2028-07-22",
        "The canary starts 7 days before the 2028-07-22 go-live",
    ]
    assert (run.repair_attempted, run.repaired, run.abstained, run.dropped_final) == (3, 1, 1, 1)
    assert run.repair_reasons == ["abstained", "number_not_in_quote", None]
    assert run.usd == pytest.approx(0.01) and run.calls == 1


def test_uncited_prose_before_a_citation_is_judged_with_that_citations_segment():
    """The parser keeps a sentence and the prose before it as one segment,
    so the support judge grades the whole of it against the quotes."""
    seen = []

    def judge(*, sentence, reference_text):
        seen.append(sentence)
        return "not_supported"

    raw = f'Nothing covers Beta. Acme runs 47 pumps [{D2}: "Acme runs 47 pumps"].'
    run = run_cited(
        _routed_chat(lambda s, u: "", []), "g", "q", "CTX", DOCS, raw=raw, support=judge
    )
    assert seen == ["Nothing covers Beta. Acme runs 47 pumps"]
    assert run.first.reasons == ["not_supported"]


def test_repair_is_skipped_when_nothing_was_dropped_or_repair_is_off():
    raw = f'Go-live is 2028-07-22 [{D1}: "Go-live is 2028-07-22."].'
    calls: list = []
    chat = _routed_chat(lambda s, u: pytest.fail("no call expected"), calls)
    run = run_cited(chat, "g", "q", "CTX", DOCS, raw=raw, repair=True)
    assert calls == [] and run.repair_raw is None and len(run.final) == 1
    run = run_cited(chat, "g", "q", "CTX", DOCS, raw="uncited only", repair=False)
    assert calls == [] and run.final == [] and run.dropped_final == 1


def test_repair_restatements_go_through_the_support_judge_too():
    reply = f'1. Acme is growing fast [{D2}: "Acme runs 47 pumps"]'
    calls: list = []
    chat = _routed_chat(_repair_route(reply), calls)
    run = run_cited(
        chat,
        "g",
        "q",
        "CTX",
        DOCS,
        raw="A bare uncited claim.",
        repair=True,
        support=lambda *, sentence, reference_text: "not_supported",
    )
    assert run.final == [] and run.repaired == 0 and run.dropped_final == 1
    assert run.repair_reasons == ["not_supported"]


def test_chat_support_judge_sends_the_system_prompt_parses_and_caches():
    from portolan.report.support_judge import ANSWER_SUPPORT_SYSTEM

    replies = iter(['{"verdict": "supported", "reason": "r"}', "no json at all"])
    calls: list = []
    chat = _routed_chat(lambda s, u: next(replies), calls)
    tally = {"usd": 0.0, "calls": 0}
    judge = make_chat_support_judge(chat, "judge/model", tally=tally, cache={})
    assert judge(sentence="s", reference_text="r") == "supported"
    assert judge(sentence="s", reference_text="r") == "supported"
    assert len(calls) == 1
    assert calls[0]["messages"][0] == {"role": "system", "content": ANSWER_SUPPORT_SYSTEM}
    assert calls[0]["messages"][1]["content"].endswith("Sentence to grade:\ns")
    assert judge(sentence="other", reference_text="r") is None
    assert tally == {"usd": pytest.approx(0.02), "calls": 2}


def test_chat_refuses_once_this_runs_cap_is_spent():
    calls: list = []
    chat = _routed_chat(lambda s, u: "x", calls, run_cap=0.015)
    chat.complete("m", "one")
    chat.complete("m", "two")
    with pytest.raises(BudgetExceeded):
        chat.complete("m", "three")
    assert len(calls) == 2


def test_evaluate_question_runs_every_mode_and_judges_each_distinct_text_once():
    """v1 drops its only (uncited) sentence and repair restores it; v2 is
    cited from the start; the recorded answer replays with a seeded score.
    Identical final texts share one set of judge calls."""
    cited = f'The canary runs 7 days [{D1}: "The canary runs 7 days"].'
    recorded = f'Go-live is 2028-07-22 [{D1}: "Go-live is 2028-07-22."].'
    counts = {"correctness": 0, "fact": 0, "support": 0, "repair": 0}

    def route(system, user):
        if system is not None:
            counts["support"] += 1
            return '{"verdict": "supported", "reason": "ok"}'
        if user.startswith("You are a wholistic"):
            counts["correctness"] += 1
            return '{"reason": "r", "aligned": "yes"}'
        if user.startswith("You are an answer validator"):
            counts["fact"] += 1
            return "yes"
        if "## Removed sentences" in user:
            counts["repair"] += 1
            return f'1. The canary runs 7 days [{D1}: "The canary runs 7 days"]'
        raise AssertionError(user[:80])

    calls: list = []
    chat = _routed_chat(route, calls)
    q = {
        "question_id": "qst_0001",
        "question_type": "basic",
        "question": "How long is the canary?",
        "gold_answer": "7 days",
        "answer_facts": ["The canary runs 7 days."],
    }
    gens = {
        "plain": {"text": "Seven days.", "s": 1.0, "usd": 0.001},
        "v1": {"text": "The canary runs 7 days", "s": 2.0, "usd": 0.002},
        "v2": {"text": cited, "s": 3.0, "usd": 0.003},
    }
    seeded = {"answer_correct": False, "completeness_pct": 0.0, "facts_valid": []}
    row, record = evaluate_question(
        chat,
        gen_model="g",
        judge_model="j",
        question=q,
        docs=[(D1, "Rollout plan", DOCS[D1].split("\n\n", 1)[1])],
        gens=gens,
        recorded_raw=recorded,
        recorded_cited_score=seeded,
    )
    assert set(ANSWER_MODES) <= set(row)
    assert row["cited_before"]["shipped"] == 0 and row["cited_before"]["answer_correct"] is False
    assert row["cited_repair"]["repaired"] == 1 and row["cited_repair"]["answer_correct"] is True
    assert row["cited_v2"]["repair_attempted"] == 0
    assert row["cited_v2_support"]["support_gate"]["not_supported_first_pass"] == 0
    # The recorded text is unchanged by the replay, so its seeded score stands.
    assert row["recorded_before"]["answer_correct"] is False
    assert row["recorded_repair"]["answer_correct"] is False
    # Judged: the plain answer and the one cited sentence that repair, v2 and
    # v2+support all end with; the empty v1 answer needs no call.
    assert counts["correctness"] == 2 and counts["fact"] == 2
    assert counts["repair"] == 1
    assert row["cited_repair"]["latency_s"] >= 2.0
    assert row["cited_v2"]["usd"] == pytest.approx(0.003)
    assert row["cited_v2_support"]["usd"] == pytest.approx(0.013)  # one support call
    assert record["modes"]["cited_repair"]["final_text"] == "The canary runs 7 days."


def test_estimate_counts_repair_calls_only_where_the_free_gate_drops_something():
    gen, judge = "openai/gpt-5.6-luna", "anthropic/claude-haiku-4.5"
    one = estimate_generation_usd([30_000], gen)
    assert one > 0 and estimate_generation_usd([30_000, 30_000], gen) == pytest.approx(2 * one)

    q = {"question": "q?", "gold_answer": "g", "answer_facts": ["f1", "f2"]}
    cited = f'Go-live is 2028-07-22 [{D1}: "Go-live is 2028-07-22."].'
    clean = {k: {"text": cited} for k in ("plain", "v1", "v2")}
    dirty = {**clean, "v1": {"text": "An uncited sentence."}}
    kw = {"question": q, "context_chars": 30_000, "doc_texts": DOCS, "recorded_raw": None}
    clean_item = evaluation_item(gens=clean, **kw)
    dirty_item = evaluation_item(gens=dirty, **kw)
    # v2 + support may always repair; v1 repairs only when its gate drops.
    assert clean_item["repair_calls"] == 1 and dirty_item["repair_calls"] == 2
    assert len(clean_item["support_chars"]) == 2 and len(dirty_item["support_chars"]) == 1
    assert len(dirty_item["judged_chars"]) == len(clean_item["judged_chars"]) + 1
    assert estimate_evaluation_usd([dirty_item], gen, judge) > estimate_evaluation_usd(
        [clean_item], gen, judge
    )
