"""The door's hop affordances and caller-facing errors, found by the agentic
retrieval bench (`01-spec/bench-agentic-retrieval-2026-09-23.md`): `open_note`
names each link's target and the notes linking in, `search(near=...)` scopes
to a note's neighbourhood, and a tool's own `ValueError` reaches an MCP client
as its message rather than "Error executing tool"."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from portolan.config import Settings
from portolan.mcp_server.access import set_ai_access
from portolan.mcp_server.server import create_server
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.store import LocalStore

TENANT = "test"
SIGNAL = "signal-request-cobalt-ridge"
ACCOUNT = "cobalt-ridge-medical-center"
SYSTEM = "cobalt-health"
PRODUCT = "tidewater-link"
OTHER_SIGNAL = "signal-request-elm-grove"


@pytest.fixture()
def graph(db):
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, s3_key, content_hash, source_date,
                                 extraction_status, summary)
            VALUES ('S-call-7', 'call', 'originals/call/S-call-7', 'h7', '2026-08-01', 'done',
                    'Integration call.')
            """
        )
        cur.execute(
            """
            INSERT INTO notes (slug, type, title, description, body_md, updated, crm_id)
            VALUES
              (%s, 'signal', 'Request - Cobalt Ridge', 'Requests.',
               'Cobalt Ridge asked for an Epic interface for Tidewater Link.', '2026-08-02', NULL),
              (%s, 'account', 'Cobalt Ridge Medical Center', 'A hospital.',
               'A hospital in the Cobalt Health system.', '2026-08-02', 'A-1'),
              (%s, 'account', 'Cobalt Health', 'A system.', 'A health system.', '2026-08-02',
               'A-2'),
              (%s, 'product', 'Tidewater Link', 'A product.', 'An interface engine.',
               '2026-08-02', 'P-1'),
              (%s, 'signal', 'Request - Elm Grove', 'Requests.',
               'Elm Grove asked for an Epic interface too.', '2026-08-02', NULL)
            """,
            (SIGNAL, ACCOUNT, SYSTEM, PRODUCT, OTHER_SIGNAL),
        )
        cur.executemany(
            "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, %s)",
            [
                (SIGNAL, ACCOUNT, "body"),
                (SIGNAL, PRODUCT, "body"),
                (SIGNAL, "S-call-7", "provenance"),
                (ACCOUNT, SIGNAL, "body"),
                (ACCOUNT, SYSTEM, "parent"),
                (OTHER_SIGNAL, PRODUCT, "body"),
            ],
        )
    reindex_all(db, FakeEmbedder())
    set_ai_access(db, TENANT, enabled=True, changed_by="test", reason="test fixture")


def _door(db_url, tmp_path):
    settings = Settings(
        database_url=db_url, tenant=TENANT, embed_provider="fake", store_root=tmp_path
    )
    return create_server(settings, embedder=FakeEmbedder(), store=LocalStore(tmp_path))


def _call(door, name, arguments):
    result = asyncio.run(door.call_tool(name, arguments))
    assert not result.is_error, result.content
    return json.loads(result.content[0].text) if result.content else None


def test_open_note_names_link_targets_and_the_notes_linking_in(graph, db_url, tmp_path):
    door = _door(db_url, tmp_path)

    account = _call(door, "open_note", {"slug": ACCOUNT})
    assert {
        (link["to_slug"], link["kind"], link["title"], link["type"]) for link in account["links"]
    } == {
        (SIGNAL, "body", "Request - Cobalt Ridge", "signal"),
        (SYSTEM, "parent", "Cobalt Health", "account"),
    }
    assert account["linked_from"] == [
        {"from_slug": SIGNAL, "kind": "body", "title": "Request - Cobalt Ridge", "type": "signal"}
    ]

    product = _call(door, "open_note", {"slug": PRODUCT})
    assert product["links"] == []
    assert {n["from_slug"] for n in product["linked_from"]} == {SIGNAL, OTHER_SIGNAL}

    signal = _call(door, "open_note", {"slug": SIGNAL})
    source = [link for link in signal["links"] if link["kind"] == "provenance"]
    assert source == [{"to_slug": "S-call-7", "kind": "provenance", "title": None, "type": "call"}]


def test_the_researcher_raw_open_note_shape_is_unchanged(graph, db):
    from portolan.mcp_server import tools

    note = tools.tool_open_note(db, tenant=TENANT, slug=ACCOUNT)
    assert "linked_from" not in note
    assert all(set(link) == {"to_slug", "kind"} for link in note["links"])


def test_search_near_looks_only_at_a_note_and_its_neighbours(graph, db_url, tmp_path):
    door = _door(db_url, tmp_path)

    everywhere = _call(door, "search", {"query": "Epic interface", "k": 10})
    assert OTHER_SIGNAL in {h["note_slug"] for h in everywhere["hits"]}

    near = _call(door, "search", {"query": "Epic interface", "k": 10, "near": [ACCOUNT]})
    slugs = {h["note_slug"] for h in near["hits"]}
    assert SIGNAL in slugs
    assert slugs <= {ACCOUNT, SIGNAL, SYSTEM}


def test_search_near_notes_that_do_not_exist_is_an_error_not_an_empty_answer(
    graph, db_url, tmp_path
):
    door = _door(db_url, tmp_path)
    with pytest.raises(ToolError, match="None of the `near` notes exist") as caught:
        asyncio.run(door.call_tool("search", {"query": "Epic", "near": ["no-such-note"]}))
    assert not isinstance(caught.value, UnexpectedToolError)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"query": "Epic", "types": ["signals"]}, "Unknown note type"),
        ({"query": "x" * 501}, "the limit is 500"),
    ],
)
def test_a_tools_own_error_reaches_the_client_as_its_message(
    graph, db_url, tmp_path, arguments, message
):
    """The MCP SDK sends a client `str(exc)`: for a `ToolError` that is the
    message; for anything else only "Error executing tool search"."""
    door = _door(db_url, tmp_path)
    with pytest.raises(ToolError, match=message) as caught:
        asyncio.run(door.call_tool("search", arguments))
    assert not isinstance(caught.value, UnexpectedToolError)
