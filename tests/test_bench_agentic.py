"""The agentic retrieval bench (`portolan.bench.agentic`): the answer gate,
the per-question tool budget and ledger, the agent loop with the model
stubbed, and the door wiring against a real migrated database. No model is
called: `claude_agent_sdk.query` is replaced with a fake that drives the
session's tools the way the CLI would."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from portolan.bench import agentic
from portolan.bench.agentic import (
    AGENT_TOOL_NAMES,
    BenchQuestion,
    DoorResult,
    DoorSession,
    NoteEvidence,
    door_caller,
    door_tool_specs,
    gate_answer,
    load_note_evidence,
    recall_pct,
    run_agent,
    slugs_in_result,
)

NOTE = "signal-pricing-cobalt-ridge"
OTHER = "cobalt-ridge"
QUOTE = "we need twelve percent off the next invoice because the pumps came late"


def _evidence() -> dict[str, NoteEvidence]:
    return {
        NOTE: NoteEvidence(
            slug=NOTE,
            text=f"Pricing at Cobalt Ridge\n\nThe buyer said: {QUOTE}.",
            synthesis="Cobalt Ridge wants a credit for late pumps.",
            claims={7: ("Cobalt Ridge wants a 12 percent credit on its next invoice.", QUOTE)},
        ),
        OTHER: NoteEvidence(
            slug=OTHER,
            text="Cobalt Ridge Medical Center\n\nAn account in Ohio with 40 pumps.",
            synthesis="An account.",
            claims={},
        ),
    }


# ---------------------------------------------------------------------
# gate_answer
# ---------------------------------------------------------------------


def test_gate_keeps_a_sentence_whose_quote_is_in_a_returned_note():
    gated = gate_answer(
        [
            {
                "text": "Cobalt Ridge wants a credit",
                "citations": [{"note_slug": NOTE, "quote": QUOTE}],
            }
        ],
        _evidence(),
        {NOTE},
    )
    assert [s["text"] for s in gated.kept] == ["Cobalt Ridge wants a credit"]
    assert gated.text == "Cobalt Ridge wants a credit."
    assert gated.cited_slugs == [NOTE]
    assert gated.summary()["quotes_supported"] == 1


@pytest.mark.parametrize(
    ("citation", "citable", "reason"),
    [
        ({"note_slug": NOTE, "quote": QUOTE}, set(), "note_not_returned"),
        ({"note_slug": "no-such-note", "quote": QUOTE}, {"no-such-note"}, "note_not_returned"),
        ({"note_slug": NOTE, "quote": "the pumps were early"}, {NOTE}, "quote_not_supported"),
        ({"note_slug": NOTE, "quote": QUOTE, "claim_id": 99}, {NOTE}, "claim_not_in_note"),
    ],
)
def test_gate_drops_a_sentence_with_no_valid_citation_and_says_why(citation, citable, reason):
    gated = gate_answer([{"text": "A claim.", "citations": [citation]}], _evidence(), citable)
    assert gated.kept == []
    assert [d["reason"] for d in gated.dropped] == [reason]


def test_gate_drops_a_sentence_that_is_not_an_object():
    gated = gate_answer(["Cobalt Ridge wants a credit."], _evidence(), {NOTE})
    assert gated.kept == [] and gated.dropped[0]["reason"] == "no_citation"


def test_gate_drops_an_uncited_sentence():
    gated = gate_answer([{"text": "Nothing on file.", "citations": []}], _evidence(), {NOTE})
    assert gated.kept == [] and gated.dropped[0]["reason"] == "no_citation"
    assert gated.text == ""


def test_gate_numbers_must_be_in_what_the_sentence_cites():
    ev = _evidence()
    by_claim = gate_answer(
        [
            {
                "text": "Cobalt Ridge wants 12 percent off.",
                "citations": [{"note_slug": NOTE, "quote": QUOTE, "claim_id": 7}],
            }
        ],
        ev,
        {NOTE},
    )
    assert len(by_claim.kept) == 1  # the claim text states 12 percent

    invented = gate_answer(
        [
            {
                "text": "Cobalt Ridge wants 15 percent off.",
                "citations": [{"note_slug": NOTE, "quote": QUOTE, "claim_id": 7}],
            }
        ],
        ev,
        {NOTE},
    )
    assert invented.kept == [] and invented.dropped[0]["reason"] == "numerals"
    assert invented.numerals_dropped == 1

    # 40 is in OTHER's body, not in what this sentence cites.
    elsewhere = gate_answer(
        [{"text": "It runs 40 pumps.", "citations": [{"note_slug": NOTE, "quote": QUOTE}]}],
        ev,
        {NOTE, OTHER},
    )
    assert elsewhere.kept == []


def test_gate_keeps_a_sentence_when_one_of_several_citations_is_valid():
    gated = gate_answer(
        [
            {
                "text": "Cobalt Ridge wants a credit.",
                "citations": [
                    {"note_slug": OTHER, "quote": "not in there"},
                    {"note_slug": NOTE, "quote": QUOTE},
                ],
            }
        ],
        _evidence(),
        {NOTE, OTHER},
    )
    assert len(gated.kept) == 1
    assert gated.kept[0]["citations"] == [{"note_slug": NOTE, "claim_id": None, "quote": QUOTE}]
    assert gated.summary()["quotes"] == 2 and gated.summary()["quotes_supported"] == 1


def test_recall_pct():
    assert recall_pct(["a", "b"], ["b", "c"]) == 50.0
    assert recall_pct([], ["a"]) == 0.0


# ---------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------


def test_slugs_in_result_reads_each_door_shape():
    assert slugs_in_result("search", {"hits": [{"note_slug": "a"}, {"note_slug": "b"}]}) == (
        ["a", "b"],
        ["a", "b"],
        [],
    )
    opened = {
        "slug": "a",
        "links": [{"to_slug": "x", "kind": "body"}, {"to_slug": "S-call-1", "kind": "provenance"}],
    }
    assert slugs_in_result("open_note", opened) == (["a", "x"], ["a"], ["a"])
    assert slugs_in_result("list_claims", {"slug": "a"}) == (["a"], ["a"], [])
    assert slugs_in_result("graph_neighbors", [{"slug": "n1"}, {"slug": "n2"}]) == (
        ["n1", "n2"],
        [],
        [],
    )
    assert slugs_in_result("open_source", {"notes_extracted": ["a", "b"]}) == (["a", "b"], [], [])
    assert slugs_in_result("open_note", None) == ([], [], [])


class _FakeDoor:
    """A door that answers `search` with `hits`, `open_note` with the note asked
    for, and `graph_neighbors` with OTHER, counting what it actually ran."""

    def __init__(self, hits=(NOTE, OTHER)) -> None:
        self.ran: list[str] = []
        self.hits = list(hits)

    async def __call__(self, name, arguments):
        self.ran.append(name)
        if name == "search":
            payload = {"hits": [{"note_slug": s} for s in self.hits]}
            return DoorResult([json.dumps(payload)], False, payload)
        if name == "open_note":
            payload = {"slug": arguments["slug"], "claims": []}
            return DoorResult([json.dumps(payload)], False, payload)
        if name == "graph_neighbors":
            payload = [{"slug": OTHER, "direction": "out"}]
            return DoorResult([json.dumps(p) for p in payload], False, payload)
        return DoorResult(["Unknown tool"], True)


def _session(door, **kw) -> DoorSession:
    specs = [{"name": n, "description": f"{n} tool", "input_schema": {}} for n in AGENT_TOOL_NAMES]
    specs.append({"name": "add_standing_question", "description": "writes", "input_schema": {}})
    return DoorSession(
        tool_specs=specs, call_door=door, evidence_for=lambda slugs: _evidence(), **kw
    )


def test_session_offers_the_read_tools_only():
    session = _session(_FakeDoor())
    assert [s["name"] for s in session.tool_specs] == list(AGENT_TOOL_NAMES)
    assert "mcp__portolan__add_standing_question" not in session.allowed_tools
    assert "mcp__portolan__submit_answer" in session.allowed_tools


def test_session_refuses_calls_past_the_budget_without_running_them():
    door = _FakeDoor()
    session = _session(door, max_tool_calls=2)

    async def go():
        await session.call("search", {"query": "q"})
        await session.call("open_note", {"slug": NOTE})
        return await session.call("search", {"query": "again"})

    refused = asyncio.run(go())
    assert door.ran == ["search", "open_note"]
    assert refused["is_error"] is True
    assert "budget" in refused["content"][0]["text"]
    assert session.research_calls == 2
    assert list(session.seen) == [NOTE, OTHER]
    assert session.citable == {NOTE, OTHER}
    assert list(session.opened) == [NOTE]


def test_submit_answer_gates_against_notes_the_session_returned():
    session = _session(_FakeDoor(), max_submits=2)
    cited_elsewhere = {
        "sentences": [
            {
                "text": "Cobalt Ridge wants a credit.",
                "citations": [{"note_slug": NOTE, "quote": QUOTE}],
            }
        ]
    }

    async def go():
        before = await session.call("submit_answer", cited_elsewhere)  # nothing returned yet
        await session.call("search", {"query": "cobalt"})
        after = await session.call("submit_answer", cited_elsewhere)
        third = await session.call("submit_answer", cited_elsewhere)
        return before, after, third

    before, after, third = asyncio.run(go())
    assert json.loads(before["content"][0]["text"])["status"] == "nothing_kept"
    assert json.loads(after["content"][0]["text"])["status"] == "accepted"
    assert third["is_error"] is True  # max_submits reached
    assert session.final is not None and session.final.text == "Cobalt Ridge wants a credit."
    assert len(session.submissions) == 2


# ---------------------------------------------------------------------
# The agent loop, model stubbed
# ---------------------------------------------------------------------


def _question() -> BenchQuestion:
    return BenchQuestion(
        qid="q1",
        question="Does Cobalt Ridge want money back?",
        gold_answer="Yes, a credit.",
        answer_facts=("Cobalt Ridge wants a credit.",),
        expected=(NOTE,),
    )


class _ScriptedQuery:
    """Stands in for `claude_agent_sdk.query`: calls the session's tools in a
    fixed order (as the CLI would on the model's behalf), yielding one
    `AssistantMessage` per tool use, then a `ResultMessage` with usage."""

    def __init__(self, session: DoorSession, script, *, hang: bool = False) -> None:
        self.session = session
        self.script = list(script)
        self.hang = hang
        self.closed = False
        self.options = None

    def __call__(self, *, prompt, options):
        self.options = options
        self.prompt = prompt
        return self._gen()

    async def _gen(self):
        from claude_agent_sdk import AssistantMessage, ResultMessage, ToolUseBlock

        try:
            for i, (name, args) in enumerate(self.script):
                await self.session.call(name, args)
                yield AssistantMessage(
                    content=[ToolUseBlock(id=str(i), name=f"mcp__portolan__{name}", input=args)],
                    model="claude-haiku-4-5",
                )
            if self.hang:
                await asyncio.sleep(3600)
            yield ResultMessage(
                subtype="success",
                duration_ms=10,
                duration_api_ms=5,
                is_error=False,
                num_turns=len(self.script) + 1,
                session_id="s",
                total_cost_usd=0.01,
                usage={"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500},
            )
        finally:
            self.closed = True


def test_run_agent_answers_from_its_last_submission(monkeypatch):
    import claude_agent_sdk

    session = _session(_FakeDoor())
    answer = {
        "sentences": [
            {
                "text": "Cobalt Ridge wants a credit.",
                "citations": [{"note_slug": NOTE, "quote": QUOTE}],
            }
        ]
    }
    fake = _ScriptedQuery(
        session,
        [
            ("search", {"query": "cobalt credit"}),
            ("open_note", {"slug": NOTE}),
            ("submit_answer", answer),
        ],
    )
    monkeypatch.setattr(claude_agent_sdk, "query", fake)
    monkeypatch.setattr(session, "sdk_server", lambda: object())

    run = asyncio.run(
        run_agent(_question(), session=session, model="claude-haiku-4-5", system_prompt="S")
    )

    assert run.answer_text == "Cobalt Ridge wants a credit."
    assert run.tool_calls == 2 and run.submissions == 1
    assert run.cited == [NOTE]
    assert run.stop == "ended"
    # Haiku 4.5 at $1 / $5: 1,000 uncached + 500 cache-read at 0.1x in, 200 out.
    assert run.cost_usd == pytest.approx((1000 + 50 + 200 * 5) / 1e6)
    row = run.as_row(_question())
    assert row["recall_retrieved_pct"] == 100.0 and row["recall_cited_pct"] == 100.0
    assert row["gold_first_seen_call"] == 1
    assert row["calls_by_tool"] == {"search": 1, "open_note": 1, "submit_answer": 1}
    assert row["calls_by_kind"] == {"search": 1, "land": 1, "hop": 0}
    assert row["cited_via_hop_only"] == [] and row["gold_via_hop_only"] == []
    assert fake.prompt == "Question: Does Cobalt Ridge want money back?"
    assert fake.options.tools == [] and fake.options.model == "claude-haiku-4-5"
    assert set(fake.options.allowed_tools) == set(session.allowed_tools)


def test_seeded_run_searches_the_question_first_and_hands_the_result_to_the_model(monkeypatch):
    import claude_agent_sdk

    door = _FakeDoor()
    session = _session(door)
    answer = {
        "sentences": [
            {
                "text": "Cobalt Ridge wants a credit.",
                "citations": [{"note_slug": NOTE, "quote": QUOTE}],
            }
        ]
    }
    fake = _ScriptedQuery(session, [("submit_answer", answer)])
    monkeypatch.setattr(claude_agent_sdk, "query", fake)
    monkeypatch.setattr(session, "sdk_server", lambda: object())

    run = asyncio.run(
        run_agent(
            _question(),
            session=session,
            model="claude-sonnet-5",
            system_prompt="S",
            seed_search=True,
        )
    )

    assert door.ran == ["search"]
    assert session.calls[0].arguments == {"query": _question().question}
    assert fake.prompt.startswith("Question: Does Cobalt Ridge want money back?")
    assert NOTE in fake.prompt  # the seeded hits are in the first message
    assert run.mode == "agent_seeded" and run.tool_calls == 1
    assert run.answer_text == "Cobalt Ridge wants a credit."


def test_run_agent_counts_hops_and_evidence_only_hopping_reached(monkeypatch):
    import claude_agent_sdk

    session = _session(_FakeDoor(hits=(NOTE,)))
    answer = {
        "sentences": [
            {
                "text": "Cobalt Ridge is an account in Ohio.",
                "citations": [{"note_slug": OTHER, "quote": "An account in Ohio"}],
            }
        ]
    }
    fake = _ScriptedQuery(
        session,
        [
            ("search", {"query": "cobalt"}),
            ("open_note", {"slug": NOTE}),
            ("graph_neighbors", {"slug": NOTE}),
            ("open_note", {"slug": OTHER}),
            ("submit_answer", answer),
        ],
    )
    monkeypatch.setattr(claude_agent_sdk, "query", fake)
    monkeypatch.setattr(session, "sdk_server", lambda: object())
    question = BenchQuestion("q2", "Where is Cobalt Ridge?", "Ohio.", ("Ohio",), (OTHER,))

    run = asyncio.run(
        run_agent(question, session=session, model="claude-sonnet-5", system_prompt="S")
    )
    row = run.as_row(question)
    assert row["calls_by_kind"] == {"search": 1, "land": 1, "hop": 2}
    assert row["recall_searched_pct"] == 0.0 and row["recall_retrieved_pct"] == 100.0
    assert row["gold_via_hop_only"] == [OTHER]
    assert row["cited_via_hop_only"] == [OTHER]
    assert run.answer_text == "Cobalt Ridge is an account in Ohio."


def test_run_agent_stops_at_the_wall_clock_and_keeps_what_was_submitted(monkeypatch):
    import claude_agent_sdk

    session = _session(_FakeDoor())
    answer = {
        "sentences": [
            {
                "text": "Cobalt Ridge wants a credit.",
                "citations": [{"note_slug": NOTE, "quote": QUOTE}],
            }
        ]
    }
    fake = _ScriptedQuery(
        session, [("search", {"query": "cobalt"}), ("submit_answer", answer)], hang=True
    )
    monkeypatch.setattr(claude_agent_sdk, "query", fake)
    monkeypatch.setattr(session, "sdk_server", lambda: object())

    run = asyncio.run(
        run_agent(
            _question(),
            session=session,
            model="claude-haiku-4-5",
            system_prompt="S",
            wall_seconds=0.5,
        )
    )
    assert run.stop == "wall_time"
    assert run.answer_text == "Cobalt Ridge wants a credit."
    assert fake.closed is True
    assert run.usage == {} and run.cost_usd is None


def test_run_agent_with_no_submission_is_an_empty_answer(monkeypatch):
    import claude_agent_sdk

    session = _session(_FakeDoor())
    fake = _ScriptedQuery(session, [("search", {"query": "cobalt"})])
    monkeypatch.setattr(claude_agent_sdk, "query", fake)
    monkeypatch.setattr(session, "sdk_server", lambda: object())
    run = asyncio.run(
        run_agent(_question(), session=session, model="claude-haiku-4-5", system_prompt="S")
    )
    assert run.stop == "no_submission" and run.answer_text == "" and run.cited == []


def test_sdk_server_carries_the_door_descriptions_and_schemas(monkeypatch):
    import claude_agent_sdk

    captured = {}

    def fake_create(*, name, tools):
        captured["name"], captured["tools"] = name, tools
        return "server"

    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", fake_create)
    session = _session(_FakeDoor())
    assert session.sdk_server() == "server"
    by_name = {t.name: t for t in captured["tools"]}
    assert set(by_name) == set(AGENT_TOOL_NAMES) | {"submit_answer"}
    assert by_name["search"].description == "search tool"
    assert by_name["submit_answer"].input_schema == agentic.ANSWER_SCHEMA
    assert by_name["search"].annotations.maxResultSizeChars == agentic.MAX_RESULT_SIZE_CHARS


# ---------------------------------------------------------------------
# Against the real door and a migrated database
# ---------------------------------------------------------------------


@pytest.fixture()
def small_graph(db):
    from portolan.mcp_server.access import set_ai_access
    from portolan.search.embed import FakeEmbedder
    from portolan.search.index import reindex_all

    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash, source_date,
                                 extraction_status, summary)
            VALUES ('S-call-1', 'call', 'originals/call/S-call-1', 'fake-hash-1', '2026-08-14',
                    'done', 'Pricing call.')
            """
        )
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md, updated, crm_id)
            VALUES (%s, 'signal', 'Pricing at Cobalt Ridge', 'Credit request.', %s, '2026-08-14',
                    NULL),
                   (%s, 'account', 'Cobalt Ridge Medical Center', 'An account.', %s, '2026-08-14',
                    'ACC-001')
            """,
            (NOTE, f"The buyer asked for a credit: {QUOTE}.", OTHER, "Ohio account, 40 pumps."),
        )
        cur.execute(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'body')", (NOTE, OTHER)
        )
        cur.execute(
            """
            INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, quote,
                                speaker, recorded)
            VALUES (%s, 'pricing', 'Cobalt Ridge wants a 12 percent credit.', 'S-call-1',
                    '@ 00:01:00', %s, 'Buyer', %s)
            RETURNING id
            """,
            (NOTE, QUOTE, datetime(2026, 8, 14, 9, tzinfo=UTC)),
        )
        claim_id = cur.fetchone()[0]
    reindex_all(db, FakeEmbedder())
    set_ai_access(db, "test", enabled=True, changed_by="test", reason="test fixture")
    return claim_id


def test_the_real_door_feeds_the_ledger_and_the_gate(small_graph, db, db_url, tmp_path):
    from portolan.config import Settings
    from portolan.mcp_server.server import create_server
    from portolan.search.embed import FakeEmbedder
    from portolan.store import LocalStore

    settings = Settings(
        database_url=db_url, tenant="test", embed_provider="fake", store_root=tmp_path
    )
    door = create_server(settings, embedder=FakeEmbedder(), store=LocalStore(tmp_path))
    specs = door_tool_specs(door)
    assert {s["name"] for s in specs} >= set(AGENT_TOOL_NAMES)

    def evidence_for(slugs):
        return load_note_evidence(db, slugs)

    session = DoorSession(tool_specs=specs, call_door=door_caller(door), evidence_for=evidence_for)
    claim_id = small_graph

    async def go():
        found = await session.call("search", {"query": "Cobalt Ridge credit"})
        opened = await session.call("open_note", {"slug": NOTE})
        missing = await session.call("open_note", {"slug": "no-such-note"})
        bad = await session.call("search", {"query": "x", "types": ["signals"]})
        neighbours = await session.call("graph_neighbors", {"slug": NOTE})
        submitted = await session.call(
            "submit_answer",
            {
                "sentences": [
                    {
                        "text": "Cobalt Ridge wants a 12 percent credit.",
                        "citations": [{"note_slug": NOTE, "quote": QUOTE, "claim_id": claim_id}],
                    },
                    {
                        "text": "It has 40 pumps.",
                        "citations": [{"note_slug": NOTE, "quote": QUOTE}],
                    },
                ]
            },
        )
        return found, opened, missing, bad, neighbours, submitted

    found, opened, missing, bad, neighbours, submitted = asyncio.run(go())
    hits = json.loads(found["content"][0]["text"])["hits"]
    assert NOTE in [h["note_slug"] for h in hits]
    note = json.loads(opened["content"][0]["text"])
    assert note["slug"] == NOTE and "_portolan_security" in note  # the door's sanitized shape
    assert missing["content"] == []  # a `None` result reaches a client as no content at all
    assert bad["is_error"] is True and "Unknown note type" in bad["content"][0]["text"]
    # A list result is one block per item; the ledger reads the structured payload.
    assert [json.loads(b["text"])["slug"] for b in neighbours["content"]] == [OTHER]
    assert OTHER in session.seen
    feedback = json.loads(submitted["content"][0]["text"])
    assert feedback["status"] == "partly_accepted"
    assert feedback["dropped"] == [{"sentence": 1, "reason": "numerals"}]
    assert NOTE in session.opened and NOTE in session.citable
    assert session.research_calls == 5
