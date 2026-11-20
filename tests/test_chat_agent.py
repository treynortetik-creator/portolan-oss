"""Chat's agent mode (`portolan.web.chat_agent`, 01-spec/chat-agent-2026-09-25.md):
the tool loop's limits, its fallback to the single-shot answer, and the rule
that only material a tool returned in this loop is citable. The model is a
scripted fake; notes, claims, chunks and links are hand-built rows."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import httpx
import psycopg
import pytest

from portolan.config import Settings
from portolan.llm.base import LLMError, ToolCall, ToolTurn
from portolan.llm.fake import FakeLLMClient
from portolan.llm.openrouter import OpenRouterClient
from portolan.metering import InMemorySink
from portolan.search.embed import FakeEmbedder
from portolan.search.query import SearchHit, SearchResult
from portolan.web.answer import NOTHING_RECORDED_MESSAGE, passage_id_for
from portolan.web.chat_agent import (
    ANSWER_TOOL,
    FORCE_ANSWER,
    AgentLimits,
    HistoryTurn,
    _history_block,
    answer_question_agent,
    run_chat_agent,
)
from portolan.web.routes.chat import agent_options

ACME_BODY = "## Synthesis\n\nAcme renewed its contract.\n"
BETA_TEXT = "In the March call Beta asked for an Epic integration before the 2027 budget cycle."
ACME_CLAIM = "Acme renewed for two years."
BETA_CLAIM = "Beta asked for an Epic integration."


def _note(conn: psycopg.Connection, slug: str, title: str, body: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md, updated) "
            "VALUES (%s, 'source', %s, 'd', %s, now())",
            (slug, title, body),
        )
    conn.commit()


def _claim(conn: psycopg.Connection, slug: str, source: str, text: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_type, s3_key, content_hash, source_date, "
            "extraction_status) VALUES (%s, 'call', %s, %s, '2026-08-01', 'done') "
            "ON CONFLICT (slug) DO NOTHING",
            (source, f"originals/call/{source}", f"hash-{source}"),
        )
        cur.execute(
            "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
            "quote, severity, sentiment, confidence) "
            "VALUES (%s, 'request', %s, %s, 'L1', %s, 'low', 0, 0.9) RETURNING id",
            (slug, text, source, text),
        )
        (claim_id,) = cur.fetchone()
    conn.commit()
    return claim_id


@pytest.fixture
def graph(db) -> dict[str, Any]:
    """Acme (the seed's only hit) links to Beta; Beta has a claim and a chunk
    the seed never shows."""
    _note(db, "acme", "Acme Health", ACME_BODY)
    _note(db, "beta", "Beta Clinic", "## Synthesis\n\nBeta is an account.\n")
    acme_claim = _claim(db, "acme", "call-acme", ACME_CLAIM)
    beta_claim = _claim(db, "beta", "call-beta", BETA_CLAIM)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO chunks (note_slug, heading, text) VALUES ('beta', 'Calls', %s)",
            (BETA_TEXT,),
        )
        cur.execute("INSERT INTO links (from_slug, to_slug, kind) VALUES ('acme', 'beta', 'body')")
    db.commit()
    return {
        "acme_claim": acme_claim,
        "beta_claim": beta_claim,
        "beta_passage": passage_id_for("beta", BETA_TEXT),
    }


def _hit(slug: str, title: str, chunk: str) -> SearchHit:
    return SearchHit(
        note_slug=slug,
        title=title,
        score=1.0,
        via=["fts"],
        best_chunk=chunk,
        heading=None,
        updated=date(2026, 8, 1),
        source_slugs=[],
    )


@pytest.fixture
def seeded(graph, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The seed search returns Acme; the `search` tool returns Beta, and
    counts its calls."""
    searches: list[str] = []

    def seed_search(conn, embedder, query, *, k, expand):
        return SearchResult(
            hits=[_hit("acme", "Acme Health", ACME_CLAIM)],
            control_ok=True,
            query_terms_unmatched=[],
        )

    def tool_run_search(conn, embedder, query, **kwargs):
        searches.append(query)
        return SearchResult(
            hits=[_hit("beta", "Beta Clinic", BETA_TEXT)], control_ok=True, query_terms_unmatched=[]
        )

    monkeypatch.setattr("portolan.web.chat_agent.search", seed_search)
    monkeypatch.setattr("portolan.mcp_server.tools.run_search", tool_run_search)
    return {**graph, "searches": searches}


def _turn(*calls: tuple[str, dict[str, Any]], cost: float = 0.01, text: str = "") -> ToolTurn:
    return ToolTurn(
        text=text,
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments=args, raw_arguments=json.dumps(args))
            for i, (name, args) in enumerate(calls)
        ],
        input_tokens=1000,
        output_tokens=100,
        cached_tokens=0,
        cost_usd=cost,
        model="anthropic/claude-sonnet-5",
    )


def _answer(*sentences: dict[str, Any], cost: float = 0.01) -> ToolTurn:
    return _turn((ANSWER_TOOL, {"sentences": list(sentences), "gaps": []}), cost=cost)


def _s(text: str, slug: str, *, claims=(), passages=()) -> dict[str, Any]:
    return {
        "text": text,
        "note_slugs": [slug],
        "claim_ids": list(claims),
        "passages": [{"passage_id": p, "quote": q} for p, q in passages],
    }


SINGLE_SHOT_TEXT = "The single-shot answer."


class ScriptedModel:
    """`complete_with_tools` plays `script` in order (the last step repeats); a
    step is a `ToolTurn`, an exception to raise, or a callable of the request.
    `complete` is the single-shot fallback: one canned sentence citing Acme."""

    def __init__(self, script: list[Any], acme_claim: int | None = None) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []
        single = {
            "sentences": [_s(SINGLE_SHOT_TEXT, "acme", claims=[acme_claim] if acme_claim else [])],
            "gaps": [],
        }
        self.single = FakeLLMClient([single])
        self.single_calls = 0

    def complete_with_tools(self, **kwargs: Any) -> ToolTurn:
        self.requests.append(kwargs)
        step = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(step, Exception):
            raise step
        return step(kwargs) if callable(step) else step

    def complete(self, **kwargs: Any):
        self.single_calls += 1
        return self.single.complete(**kwargs)


def _run(db, model: ScriptedModel, limits: AgentLimits | None = None, **kwargs: Any):
    return answer_question_agent(
        db,
        model,
        FakeEmbedder(),
        "Which accounts asked for Epic?",
        model="anthropic/claude-sonnet-5",
        fallback_model="anthropic/claude-sonnet-5",
        limits=limits or AgentLimits(),
        tenant="test",
        **kwargs,
    )


def _texts(answer) -> list[str]:
    return [s.text for s in answer.sentences]


# ---------------------------------------------------------------------
# The gate: only what a tool returned in this loop is citable
# ---------------------------------------------------------------------


def test_a_quote_or_claim_from_a_note_the_agent_never_retrieved_is_rejected(db, seeded):
    model = ScriptedModel(
        [
            _turn(("graph_neighbors", {"slug": "acme"})),
            _answer(
                _s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]]),
                _s(
                    "Beta asked for Epic.",
                    "beta",
                    passages=[(seeded["beta_passage"], "Beta asked for an Epic integration")],
                ),
                _s("Beta wants Epic.", "beta", claims=[seeded["beta_claim"]]),
            ),
        ]
    )
    answer = _run(db, model)

    # The neighbour list named Beta (titles only); nothing of Beta's was returned.
    assert "beta" in model.requests[1]["messages"][-1]["content"]
    assert _texts(answer) == ["Acme renewed for two years."]
    assert answer.uncited_dropped == 2
    assert answer.usage["chat_mode"] == "agent"


def test_the_same_citations_pass_once_the_agent_opened_the_note(db, seeded):
    model = ScriptedModel(
        [
            _turn(("open_note", {"slug": "beta"})),
            _answer(
                _s(
                    "Beta asked for Epic.",
                    "beta",
                    passages=[(seeded["beta_passage"], "Beta asked for an Epic integration")],
                ),
                _s("Beta wants Epic.", "beta", claims=[seeded["beta_claim"]]),
            ),
        ]
    )
    answer = _run(db, model)

    assert _texts(answer) == ["Beta asked for Epic.", "Beta wants Epic."]
    passage, claim = (s.citations[0] for s in answer.sentences)
    assert passage.passage_id == seeded["beta_passage"] and passage.note_slug == "beta"
    assert claim.claim_id == seeded["beta_claim"] and claim.source_slug == "call-beta"
    assert answer.usage["agent"]["by_tool"] == {"open_note": 1}


def test_a_search_hit_passage_is_citable_but_a_made_up_quote_from_it_is_not(db, seeded):
    model = ScriptedModel(
        [
            _turn(("search", {"query": "Epic integration"})),
            _answer(
                _s(
                    "Beta asked for Epic.",
                    "beta",
                    passages=[(seeded["beta_passage"], "Beta asked for an Epic integration")],
                ),
                _s(
                    "Beta signed.",
                    "beta",
                    passages=[(seeded["beta_passage"], "Beta signed a three-year deal")],
                ),
            ),
        ]
    )
    answer = _run(db, model)

    assert seeded["searches"] == ["Epic integration"]
    assert _texts(answer) == ["Beta asked for Epic."]
    assert answer.uncited_dropped == 1


# ---------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------


def test_the_tool_loop_stops_at_the_call_cap_and_forces_the_answer(db, seeded):
    def respond(request: dict[str, Any]) -> ToolTurn:
        if request["tool_choice"] == FORCE_ANSWER:
            return _answer(_s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]]))
        return _turn(
            ("search", {"query": "a"}), ("search", {"query": "b"}), ("search", {"query": "c"})
        )

    model = ScriptedModel([respond])
    answer = _run(db, model, AgentLimits(max_tool_calls=4))

    assert len(seeded["searches"]) == 4
    assert answer.usage["agent"]["tool_calls"] == 4
    assert [r["tool_choice"] for r in model.requests] == ["auto", "auto", FORCE_ANSWER]
    assert [t["function"]["name"] for t in model.requests[-1]["tools"]] == [ANSWER_TOOL]
    # The call past the cap was refused, not run, and the model was told so.
    refused = [m for m in model.requests[-1]["messages"] if m["role"] == "tool"][-1]
    assert refused["content"].startswith("No more research")
    assert _texts(answer) == ["Acme renewed for two years."]


def test_the_dollar_cap_turns_the_next_turn_into_the_answer(db, seeded):
    model = ScriptedModel(
        [
            _turn(("search", {"query": "a"}), cost=0.10),
            _answer(_s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]])),
        ]
    )
    answer = _run(db, model, AgentLimits(max_usd=0.15))

    assert [r["tool_choice"] for r in model.requests] == ["auto", FORCE_ANSWER]
    assert answer.usage["agent"]["tool_calls"] == 1
    assert answer.usage["chat_mode"] == "agent"


def test_the_dollar_cap_stops_the_loop_when_the_model_will_not_answer(db, seeded):
    model = ScriptedModel([_turn(("search", {"query": "a"}), cost=0.10)], seeded["acme_claim"])
    answer = _run(db, model, AgentLimits(max_usd=0.15))

    assert len(model.requests) == 2  # the second turn offered only `answer`; nothing ran
    assert seeded["searches"] == ["a"]
    agent = answer.usage["agent"]
    assert agent["fallback"] == "dollar_cap"
    assert agent["cost_usd"] == pytest.approx(0.20)
    assert _texts(answer) == [SINGLE_SHOT_TEXT]


def test_the_wall_clock_timeout_falls_back_and_bounds_the_request(db, seeded):
    ticks = iter(range(0, 1000, 30))
    model = ScriptedModel([_turn(("search", {"query": "a"}))], seeded["acme_claim"])
    answer = _run(db, model, AgentLimits(timeout_seconds=45), clock=lambda: next(ticks))

    assert model.requests[0]["timeout"] == 15  # 45 s from the start, asked at 30 s
    assert seeded["searches"] == []  # the tool call came back after the deadline
    assert answer.usage["agent"]["fallback"] == "timeout"
    assert _texts(answer) == [SINGLE_SHOT_TEXT]


# ---------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------


def test_a_model_error_falls_back_to_the_single_shot_answer_and_logs_why(db, seeded, caplog):
    model = ScriptedModel([LLMError("upstream 502")], seeded["acme_claim"])
    with caplog.at_level(logging.WARNING, logger="portolan.web.chat_agent"):
        answer = _run(db, model)

    assert model.single_calls == 1
    assert _texts(answer) == [SINGLE_SHOT_TEXT]
    assert answer.sentences[0].citations[0].claim_id == seeded["acme_claim"]
    assert answer.usage["chat_mode"] == "single"
    assert answer.usage["agent"]["fallback"] == "error: LLMError: upstream 502"
    assert "fell back to single-shot (error: LLMError: upstream 502)" in caplog.text


def test_sentences_sent_as_a_json_string_are_read_not_lost(db, seeded):
    sentences = [_s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]])]
    model = ScriptedModel(
        [_turn((ANSWER_TOOL, {"sentences": json.dumps(sentences), "gaps": "[]"}))]
    )
    answer = _run(db, model)

    assert _texts(answer) == ["Acme renewed for two years."]
    assert answer.usage["chat_mode"] == "agent"


def test_sentences_string_carrying_the_rest_of_the_object_is_read(db, seeded):
    # The shape seen on ERB: the array, then `, "gaps": []`, all inside one string.
    sentences = [_s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]])]
    model = ScriptedModel(
        [_turn((ANSWER_TOOL, {"sentences": json.dumps(sentences) + ', "gaps": []'}))]
    )
    answer = _run(db, model)

    assert _texts(answer) == ["Acme renewed for two years."]
    assert answer.usage["agent"]["fallback"] is None


def test_an_answer_whose_sentences_are_not_json_falls_back(db, seeded):
    model = ScriptedModel(
        [_turn((ANSWER_TOOL, {"sentences": "Acme renewed.", "gaps": []}))], seeded["acme_claim"]
    )
    answer = _run(db, model)

    assert _texts(answer) == [SINGLE_SHOT_TEXT]
    assert answer.usage["agent"]["fallback"].startswith("error: ValueError")


def test_an_empty_answer_before_any_search_gets_one_more_look(db, seeded):
    empty = _turn((ANSWER_TOOL, {"sentences": [], "gaps": ["Epic requests"]}))
    model = ScriptedModel(
        [
            empty,
            _turn(("search", {"query": "Epic integration"})),
            _answer(
                _s(
                    "Beta asked for Epic.",
                    "beta",
                    passages=[(seeded["beta_passage"], "Beta asked for an Epic integration")],
                )
            ),
        ]
    )
    answer = _run(db, model)

    assert seeded["searches"] == ["Epic integration"]
    assert _texts(answer) == ["Beta asked for Epic."]
    retry = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert retry[0]["content"].startswith("Not accepted yet")


def test_an_empty_answer_after_searching_stands(db, seeded):
    model = ScriptedModel(
        [
            _turn(("search", {"query": "Epic integration"})),
            _turn((ANSWER_TOOL, {"sentences": [], "gaps": ["an NPS score"]})),
        ]
    )
    answer = _run(db, model)

    assert len(model.requests) == 2
    assert _texts(answer) == ["Not in the graph yet: an NPS score."]


# The live miss of 2026-09-25 (q1): sentences citing a synthesis's
# `^[source @ locator]` markers as passage ids, all refused by the gate.
MARKER_CITED = _s(
    "Beta asked for Epic.", "acme", passages=[("call-beta @ L1", "Beta asked for an Epic")]
)


def test_an_answer_the_gate_would_empty_gets_one_look_with_the_reasons(db, seeded):
    model = ScriptedModel(
        [
            _answer(MARKER_CITED, _s("Gamma wants it too.", "gamma")),
            _turn(("open_note", {"slug": "beta"})),
            _answer(_s("Beta wants Epic.", "beta", claims=[seeded["beta_claim"]])),
        ]
    )
    answer = _run(db, model)

    (refusal,) = [
        m["content"]
        for m in model.requests[1]["messages"]
        if m["role"] == "tool" and m["content"].startswith("Not accepted yet")
    ]
    assert refusal.startswith("Not accepted yet: the citation check kept none")
    assert "- Sentence 1: passage_id 'call-beta @ L1' is not a passage id you were shown" in refusal
    assert "- Sentence 2: note gamma was not returned by a tool" in refusal
    assert _texts(answer) == ["Beta wants Epic."]
    assert answer.usage["agent"]["retries"] == ["nothing_kept"]


def test_a_second_answer_the_gate_empties_stands(db, seeded):
    model = ScriptedModel([_answer(MARKER_CITED)])
    answer = _run(db, model)

    assert len(model.requests) == 2
    assert _texts(answer) == [NOTHING_RECORDED_MESSAGE]
    assert answer.uncited_dropped == 1


def test_an_answer_the_gate_keeps_in_part_is_not_sent_back(db, seeded):
    model = ScriptedModel(
        [_answer(MARKER_CITED, _s("Acme renewed.", "acme", claims=[seeded["acme_claim"]]))]
    )
    answer = _run(db, model)

    assert len(model.requests) == 1
    assert _texts(answer) == ["Acme renewed."]
    assert answer.usage["agent"]["retries"] == []


def test_a_client_without_tool_calls_falls_back(db, seeded):
    single = {
        "sentences": [_s(SINGLE_SHOT_TEXT, "acme", claims=[seeded["acme_claim"]])],
        "gaps": [],
    }
    answer = _run(db, FakeLLMClient([single]))

    assert _texts(answer) == [SINGLE_SHOT_TEXT]
    assert answer.usage["agent"]["fallback"] == "client_has_no_tool_calls"


def test_progress_lines_name_what_the_agent_is_doing(db, seeded):
    model = ScriptedModel(
        [
            _turn(("search", {"query": "Epic integration"})),
            _turn(("open_note", {"slug": "beta"})),
            _turn(("graph_neighbors", {"slug": "beta"})),
            _answer(_s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]])),
        ]
    )
    steps = run_chat_agent(
        db,
        model,
        FakeEmbedder(),
        "Which accounts asked for Epic?",
        model="anthropic/claude-sonnet-5",
        fallback_model=None,
        limits=AgentLimits(),
        tenant="test",
    )
    lines = []
    while True:
        try:
            lines.append(next(steps))
        except StopIteration as done:
            answer = done.value
            break

    assert lines == [
        "Searching the graph…",
        "Searching for “Epic integration”…",
        "Reading Beta Clinic…",
        "Following links from Beta Clinic…",
        "Checking citations…",
    ]
    assert answer.usage["agent"]["tool_calls"] == 3


# ---------------------------------------------------------------------
# Settings and the route
# ---------------------------------------------------------------------


def test_chat_mode_defaults_to_agent(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "PORTOLAN_CHAT_MODE",
        "PORTOLAN_CHAT_AGENT_MODEL",
        "PORTOLAN_CHAT_AGENT_MAX_TOOL_CALLS",
        "PORTOLAN_CHAT_AGENT_MAX_USD",
        "PORTOLAN_CHAT_AGENT_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PORTOLAN_CHAT_MODE", "single")
    settings = Settings(_env_file=None)
    assert settings.chat_mode == "single"
    assert agent_options(settings) is None

    monkeypatch.delenv("PORTOLAN_CHAT_MODE")
    options = agent_options(Settings(_env_file=None))
    assert options == {
        "model": "anthropic/claude-sonnet-5",
        "limits": AgentLimits(max_tool_calls=6, max_usd=0.15, timeout_seconds=45.0),
    }


def test_the_chat_stream_sends_progress_then_the_gated_answer(db, db_url, seeded, tmp_path):
    from fastapi.testclient import TestClient

    from portolan.web.app import create_app
    from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep

    settings = Settings(
        _env_file=None,
        database_url=db_url,
        tenant="test",
        store_root=tmp_path,
        llm_provider="fake",
        embed_provider="fake",
        org_key="test-org-key-for-agent-tests",
        session_secret="test-session-secret-for-agent-tests-not-real",
        cookie_secure=False,
        chat_mode="agent",
    )
    db.autocommit = True
    model = ScriptedModel(
        [
            _turn(("open_note", {"slug": "beta"})),
            _answer(_s("Beta wants Epic.", "beta", claims=[seeded["beta_claim"]])),
        ]
    )
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: model
    with TestClient(app) as client:
        login = client.post(
            "/login",
            data={"org_key": "test-org-key-for-agent-tests", "editor": "Tester"},
            follow_redirects=False,
        )
        r = client.post(
            "/chat/new/stream", data={"question": "Who wants Epic?"}, cookies=login.cookies
        )
    assert r.status_code == 200
    events = [
        (block.split("\n")[0][len("event: ") :], json.loads(block.split("\n")[1][len("data: ") :]))
        for block in r.text.strip().split("\n\n")
    ]
    names = [name for name, _ in events]
    assert names == ["progress", "progress", "progress", "sentence", "done"]
    assert events[1][1]["text"] == "Reading Beta Clinic…"
    assert f"/notes/beta#claim-{seeded['beta_claim']}" in events[3][1]["html"]
    assert model.requests[0]["model"] == "anthropic/claude-sonnet-5"
    with db.cursor() as cur:
        cur.execute("SELECT usage FROM messages WHERE role = 'assistant'")
        (usage,) = cur.fetchone()
    assert usage["chat_mode"] == "agent"
    assert usage["agent"]["by_tool"] == {"open_note": 1}


# ---------------------------------------------------------------------
# Follow-up questions see the conversation
# ---------------------------------------------------------------------


def _record_seed_queries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []

    def seed_search(conn, embedder, query, *, k, expand):
        queries.append(query)
        return SearchResult(
            hits=[_hit("acme", "Acme Health", ACME_CLAIM)],
            control_ok=True,
            query_terms_unmatched=[],
        )

    monkeypatch.setattr("portolan.web.chat_agent.search", seed_search)
    return queries


def _first_user_text(request: dict[str, Any]) -> str:
    content = request["messages"][1]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def test_a_follow_up_sends_the_earlier_exchange_and_searches_on_both_questions(
    db, seeded, monkeypatch
):
    queries = _record_seed_queries(monkeypatch)
    model = ScriptedModel([_answer(_s("Acme renewed.", "acme", claims=[seeded["acme_claim"]]))])
    history = [HistoryTurn("Which accounts are at risk?", "Ashgrove is at risk.", ("beta",))]
    answer_question_agent(
        db,
        model,
        FakeEmbedder(),
        "Which of those renews soon?",
        model="anthropic/claude-sonnet-5",
        fallback_model="anthropic/claude-sonnet-5",
        limits=AgentLimits(),
        tenant="test",
        history=history,
    )
    sent = _first_user_text(model.requests[0])
    assert "Which accounts are at risk?" in sent
    assert "Ashgrove is at risk." in sent
    assert "Notes that answer cited: beta" in sent
    assert queries == ["Which accounts are at risk? Which of those renews soon?"]


def test_a_first_question_is_unchanged_by_the_history_code(db, seeded, monkeypatch):
    queries = _record_seed_queries(monkeypatch)
    model = ScriptedModel([_answer(_s("Acme renewed.", "acme", claims=[seeded["acme_claim"]]))])
    _run(db, model)
    sent = _first_user_text(model.requests[0])
    assert sent.startswith("Question: Which accounts asked for Epic?")
    assert "Earlier in this conversation" not in sent
    assert queries == ["Which accounts asked for Epic?"]


def test_history_is_trimmed_to_the_last_turns_and_a_length_budget():
    turns = [HistoryTurn(f"q{i}", "x" * 5000, ()) for i in range(6)]
    block = _history_block(turns)
    assert "Q: q5" in block and "Q: q3" in block and "Q: q2" not in block
    assert len(block) < 3 * 1700 + 600


def test_the_chat_route_passes_earlier_turns_to_the_agent(
    db, db_url, seeded, tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from portolan.web.app import create_app
    from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep

    queries = _record_seed_queries(monkeypatch)
    settings = Settings(
        _env_file=None,
        database_url=db_url,
        tenant="test",
        store_root=tmp_path,
        llm_provider="fake",
        embed_provider="fake",
        org_key="test-org-key-for-agent-tests",
        session_secret="test-session-secret-for-agent-tests-not-real",
        cookie_secure=False,
        chat_mode="agent",
    )
    db.autocommit = True
    reply = _answer(_s("Acme renewed for two years.", "acme", claims=[seeded["acme_claim"]]))
    model = ScriptedModel([reply, reply])
    app = create_app(settings)
    app.dependency_overrides[get_conn] = lambda: db
    app.dependency_overrides[get_embedder_dep] = lambda: FakeEmbedder()
    app.dependency_overrides[get_llm_dep] = lambda: model
    with TestClient(app) as client:
        login = client.post(
            "/login",
            data={"org_key": "test-org-key-for-agent-tests", "editor": "Tester"},
            follow_redirects=False,
        )
        client.post("/chat/new/stream", data={"question": "Who renewed?"}, cookies=login.cookies)
        with db.cursor() as cur:
            cur.execute("SELECT id FROM conversations")
            (conversation_id,) = cur.fetchone()
        client.post(
            f"/chat/{conversation_id}/stream",
            data={"question": "For how long?"},
            cookies=login.cookies,
        )
    first, second = (_first_user_text(r) for r in model.requests)
    assert "Earlier in this conversation" not in first
    assert "Q: Who renewed?" in second
    assert "A: Acme renewed for two years." in second
    assert "Notes that answer cited: acme" in second
    assert queries == ["Who renewed?", "Who renewed? For how long?"]


# ---------------------------------------------------------------------
# The OpenRouter client's tool turn
# ---------------------------------------------------------------------


def test_openrouter_complete_with_tools_sends_tools_and_reads_calls_and_cost():
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"query": "x"}'},
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": "{not json"},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "cost": 0.0027},
            },
        )

    sink = InMemorySink()
    client = OpenRouterClient(
        spend_key="test", api_key="or-test", sink=sink, transport=httpx.MockTransport(handler)
    )
    messages = [{"role": "user", "content": "q"}]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    turn = client.complete_with_tools(
        messages=messages,
        tools=tools,
        model="anthropic/claude-sonnet-5",
        surface="chat",
        tool_choice=FORCE_ANSWER,
    )

    assert sent[0]["messages"] == messages
    assert sent[0]["tools"] == tools
    assert sent[0]["tool_choice"] == FORCE_ANSWER
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("call_1", "search", {"query": "x"}),
        ("call_2", "search", None),
    ]
    assert turn.cost_usd == pytest.approx(0.0027)
    assert (turn.input_tokens, turn.output_tokens) == (1200, 30)
    assert [e.cost_estimate for e in sink.events] == [pytest.approx(0.0027)]
    assert sink.events[0].surface == "chat"
