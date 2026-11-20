"""Tests for the ERB answer-evaluation port (`portolan.bench.erb_answers`,
`portolan.bench.erb_prompts`, checklist item 28). No network (MockTransport
only), no ERB text; the verbatim-prompt check skips without the clone."""

from __future__ import annotations

import ast
import json

import httpx
import pytest

from portolan.bench import erb_prompts
from portolan.bench.erb import default_erb_root
from portolan.bench.erb_answers import (
    CITED_ANSWER_PROMPT,
    OpenRouterChat,
    format_context_documents,
    gate_cited_answer,
    parse_aligned,
    parse_cited_answer,
    parse_fact_valid,
    sample_questions,
    score_answer_text,
    strip_dsid_references,
    summarize,
)
from portolan.bench.erb_search import BudgetExceeded
from portolan.training.spend import SpendMeter

D1 = "dsid_" + "a" * 32
D2 = "dsid_" + "b" * 32


def test_prompts_are_byte_identical_to_upstream_when_clone_present():
    root = default_erb_root()
    files = {
        "src/prompts/answer_evaluation.py": [
            "ANSWER_CITATION_STRIPPING_PROMPT",
            "ANSWER_WHOLISTIC_EVALUATION_PROMPT",
            "INDIVIDUAL_FACT_VALIDATOR_PROMPT",
        ],
        "src/prompts/vector_search_answer_gen.py": ["ANSWER_GEN_PROMPT"],
    }
    if not (root / "src" / "prompts").is_dir():
        pytest.skip(f"ERB clone not present at {root}")
    for rel, names in files.items():
        namespace: dict = {}
        exec(compile(ast.parse((root / rel).read_text()), rel, "exec"), namespace)
        for name in names:
            assert getattr(erb_prompts, name) == namespace[name], name
    assert "MIT License" in erb_prompts.__doc__
    assert "DanswerAI" in erb_prompts.__doc__


def test_prompts_format_with_their_placeholders():
    erb_prompts.ANSWER_WHOLISTIC_EVALUATION_PROMPT.format(
        query="q", gold_answer="g", candidate_answer="c"
    )
    erb_prompts.INDIVIDUAL_FACT_VALIDATOR_PROMPT.format(answer="a", statement="s")
    erb_prompts.ANSWER_GEN_PROMPT.format(context_documents="d", question="q")
    CITED_ANSWER_PROMPT.format(context_documents="d", question="q")


def test_parse_aligned_follows_erb():
    assert parse_aligned('{"reason": "ok", "aligned": "yes"}') is True
    assert parse_aligned('```json\n{"reason": "x", "aligned": "No"}\n```') is False
    assert parse_aligned("not json") is None
    assert parse_aligned('{"aligned": 1}') is None


def test_parse_fact_valid_reads_only_the_first_line():
    assert parse_fact_valid("Yes\nbecause") is True
    assert parse_fact_valid("no\nyes") is False
    assert parse_fact_valid("") is False


def test_format_context_documents_matches_erb_layout():
    out = format_context_documents([(D1, "T1", "body one"), (D2, "T2", "body two")])
    assert out == (
        f"--- Document 1 (ID: {D1}) ---\nTitle: T1\n\nbody one\n\n"
        f"--- Document 2 (ID: {D2}) ---\nTitle: T2\n\nbody two"
    )


def test_strip_dsid_references():
    assert strip_dsid_references(f"See {D1} and  more") == "See and more"


def test_parse_and_gate_cited_answer():
    answer = (
        f'The pump alarm fired twice [{D1}: "The pump alarm fired twice."]. '
        f'Acme runs 47 pumps [{D2}: "Acme runs 47 pumps"][{D1}: "not in the doc"]. '
        f'It renews in June [{D2}: "renews in June"]. '
        "An uncited closing line."
    )
    sentences = parse_cited_answer(answer)
    assert [len(s.citations) for s in sentences] == [1, 2, 1, 0]
    docs = {D1: "The pump alarm fired twice.", D2: "Acme runs 47 pumps and renews in March."}
    result = gate_cited_answer(sentences, docs)
    assert [s.text for s in result.kept] == ["The pump alarm fired twice", "Acme runs 47 pumps"]
    assert result.quotes_total == 4 and result.quotes_supported == 2
    assert len(result.dropped) == 2
    assert result.text == "The pump alarm fired twice. Acme runs 47 pumps."


def test_gate_drops_a_citation_to_a_document_not_in_context():
    sentences = parse_cited_answer(f'Claim [{D2}: "anything"].')
    assert gate_cited_answer(sentences, {D1: "anything"}).kept == []


def test_sample_questions_is_fixed_and_sorted():
    qs = [{"question_id": f"qst_{i:04d}"} for i in range(50)]
    a = sample_questions(qs, 5)
    b = sample_questions(list(reversed(qs)), 5)
    assert a == b and len(a) == 5
    assert [q["question_id"] for q in a] == sorted(q["question_id"] for q in a)


def _chat(replies, calls, tmp_path=None, budget=5.0, meter=None):
    it = iter(replies)

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": next(it)}}],
                "usage": {"cost": 0.001},
            },
        )

    return OpenRouterChat(
        "sk-test",
        meter=meter,
        corpus="erb_exam",
        budget_usd=budget,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


def test_score_answer_text_judges_correctness_and_each_fact(tmp_path):
    calls: list = []
    meter = SpendMeter(tmp_path / "spend.json")
    chat = _chat(['{"reason": "r", "aligned": "yes"}', "yes", "no"], calls, meter=meter)
    q = {"question": "Q?", "gold_answer": "G", "answer_facts": ["f1", "f2"]}
    out = score_answer_text(chat, "judge/model", q, "an answer")
    assert out == {"answer_correct": True, "completeness_pct": 50.0, "facts_valid": [True, False]}
    assert len(calls) == 3
    assert all(c["temperature"] == 0 and c["model"] == "judge/model" for c in calls)
    assert meter.corpus_usd("erb_exam") == pytest.approx(0.003)
    assert summarize([out]) == {
        "count": 1,
        "average_correctness_pct": 100.0,
        "average_completeness_pct": 50.0,
        "combined_correctness_completeness_score": 50.0,
    }


def test_empty_answer_scores_zero_without_calls():
    calls: list = []
    chat = _chat([], calls)
    out = score_answer_text(chat, "m", {"question": "q", "answer_facts": ["f"]}, "  ")
    assert out["answer_correct"] is False and out["completeness_pct"] == 0.0
    assert calls == []


def test_chat_refuses_once_budget_is_spent(tmp_path):
    meter = SpendMeter(tmp_path / "spend.json")
    meter.record(model="m", corpus="erb_exam", amount_usd=5.0)
    calls: list = []
    chat = _chat(["x"], calls, meter=meter, budget=5.0)
    with pytest.raises(BudgetExceeded):
        chat.complete("m", "hi")
    assert calls == []
