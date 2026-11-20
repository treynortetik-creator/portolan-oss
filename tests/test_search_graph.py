"""Tests for portolan/search/graph.py: graph modes over a small hand-built graph.

DB-backed, FakeEmbedder (word overlap only). The graph, fictional names:

- `keystone-health` (system) with two hospitals, `harbor-north-hospital` and
  `harbor-south-hospital` (`parent` links), and a standalone `delta-clinic`.
- A late-delivery complaint signal on each of the three hospitals/clinic, worded alike
  so text alone cannot tell Keystone's hospitals from Delta; Delta also has a pricing
  signal and a competitor signal (RivalCo), sharing the account.
- Registry: the accounts (aliases "Keystone", "KH"), product `pump-x` (alias "PX"),
  competitor RivalCo, and a contact, Jo Park, at Harbor South with no note of her own.
- Claims dated so a week window picks one note.
"""

from __future__ import annotations

from datetime import date

import pytest

from portolan.bench.graph_search import graph_params
from portolan.search import graph as g
from portolan.search.embed import FakeEmbedder
from portolan.search.graph import (
    GraphTrace,
    _Graph,
    _neighbourhood,
    _ppr,
    changed_since,
    detect_entities,
    graph_search,
)
from portolan.search.index import reindex_all
from portolan.search.query import search

ACCOUNTS = [
    ("keystone-health", "Keystone Health", "K-1", None, "A health system."),
    ("harbor-north-hospital", "Harbor North Hospital", "K-2", "keystone-health", "A hospital."),
    ("harbor-south-hospital", "Harbor South Hospital", "K-3", "keystone-health", "A hospital."),
    ("delta-clinic", "Delta Clinic", "D-1", None, "A clinic."),
]
SIGNALS = [
    # slug, title, body, linked entities, claim date
    (
        "sig-complaint-harbor-north",
        "Complaint — Harbor North Hospital",
        "Consumable deliveries arrived four days late in August.",
        ["harbor-north-hospital", "pump-x"],
        "2026-08-10",
    ),
    (
        "sig-complaint-harbor-south",
        "Complaint — Harbor South Hospital",
        "Consumable deliveries arrived five days late in August.",
        ["harbor-south-hospital", "pump-x"],
        "2026-08-19",
    ),
    (
        "sig-complaint-delta",
        "Complaint — Delta Clinic",
        "Consumable deliveries arrived three days late in August.",
        ["delta-clinic", "pump-x"],
        "2026-08-12",
    ),
    (
        "sig-pricing-delta",
        "Pricing — Delta Clinic",
        "Asked for a discount on consumables at renewal.",
        ["delta-clinic"],
        "2026-07-01",
    ),
    (
        "sig-competitor-delta",
        "Competitor — Delta Clinic",
        "Received a proposal from RivalCo.",
        ["delta-clinic", "rivalco"],
        "2026-06-01",
    ),
]


@pytest.fixture
def graph_db(db):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO sources (slug, source_type, source_date, extraction_status) "
            "VALUES ('S-call-1', 'call', '2026-08-01', 'done')"
        )
        for slug, title, crm, parent, desc in ACCOUNTS:
            cur.execute(
                "INSERT INTO notes (slug, type, title, description, crm_id, parent_slug, body_md) "
                "VALUES (%s, 'account', %s, %s, %s, %s, %s)",
                (slug, title, desc, crm, parent, f"{title} is an account."),
            )
            if parent:
                cur.execute(
                    "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'parent')",
                    (slug, parent),
                )
        cur.execute(
            "INSERT INTO notes (slug, type, title, description, body_md) "
            "VALUES ('pump-x', 'product', 'Pump X', 'A product.', 'Pump X is a product.'),"
            " ('rivalco', 'competitor', 'RivalCo', 'A competitor.', 'RivalCo competes.')"
        )
        for slug, title, body, entities, day in SIGNALS:
            cur.execute(
                "INSERT INTO notes (slug, type, title, description, body_md) "
                "VALUES (%s, 'signal', %s, %s, %s)",
                (slug, title, title, body),
            )
            cur.execute(
                "INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator, "
                "recorded) VALUES (%s, 'other', %s, 'S-call-1', '00:00:01', %s)",
                (slug, body, f"{day} 12:00:00+00"),
            )
            for entity in entities:
                cur.execute(
                    "INSERT INTO links (from_slug, to_slug, kind) VALUES (%s, %s, 'body'), "
                    "(%s, %s, 'body')",
                    (slug, entity, entity, slug),
                )
        ids = {}
        for kind, name, slug, parent in [
            ("account", "Keystone Health", "keystone-health", None),
            ("account", "Harbor North Hospital", "harbor-north-hospital", "Keystone Health"),
            ("account", "Harbor South Hospital", "harbor-south-hospital", "Keystone Health"),
            ("account", "Delta Clinic", "delta-clinic", None),
            ("product", "Pump X", "pump-x", None),
            ("competitor", "RivalCo", "rivalco", None),
            ("contact", "Jo Park", None, "Harbor South Hospital"),
        ]:
            cur.execute(
                "INSERT INTO registry_entities (kind, canonical_name, note_slug, parent_id) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (kind, name, slug, ids.get(parent)),
            )
            ids[name] = cur.fetchone()[0]
        for alias, name in [
            ("Keystone", "Keystone Health"),
            ("KH", "Keystone Health"),
            ("PX", "Pump X"),
            ("the clinic", "Delta Clinic"),
        ]:
            cur.execute(
                "INSERT INTO registry_aliases (alias, entity_id, source) VALUES (%s, %s, 'crm')",
                (alias, ids[name]),
            )
    embedder = FakeEmbedder()
    reindex_all(db, embedder)
    return db, embedder


def _slugs(result) -> list[str]:
    return [h.note_slug for h in result.hits]


def test_detect_entities_reads_names_aliases_and_a_contact_through_its_account(graph_db):
    conn, _ = graph_db
    found = {m.name: m for m in detect_entities(conn, "What did Jo Park and KH's team say?")}
    assert set(found) == {"Jo Park", "Keystone Health"}
    assert found["Jo Park"].kind == "contact"
    assert found["Jo Park"].anchor_slugs == ("harbor-south-hospital",)
    assert found["Keystone Health"].anchor_slugs == ("keystone-health",)


def test_short_aliases_match_only_as_written_and_the_aliases_are_ignored(graph_db):
    conn, _ = graph_db
    assert [m.name for m in detect_entities(conn, "Any PX recalls?")] == ["Pump X"]
    assert detect_entities(conn, "any px recalls at kh?") == []
    # "the clinic" is stored as an alias but reads as ordinary words.
    assert detect_entities(conn, "What did the clinic say?") == []


def test_mode_off_is_plain_search(graph_db):
    conn, embedder = graph_db
    question = "late consumable deliveries"
    assert _slugs(graph_search(conn, embedder, question, mode="off", k=5)) == _slugs(
        search(conn, embedder, question, k=5)
    )


def test_a_graph_mode_with_nothing_to_walk_returns_the_search_result(graph_db):
    conn, embedder = graph_db
    question = "late consumable deliveries"  # names no registry entity
    assert _slugs(graph_search(conn, embedder, question, mode="entity", k=5)) == _slugs(
        search(conn, embedder, question, k=5)
    )


def test_unknown_mode_raises(graph_db):
    conn, embedder = graph_db
    with pytest.raises(ValueError, match="Unknown graph mode"):
        graph_search(conn, embedder, "anything", mode="bfs")  # type: ignore[arg-type]


def test_entity_mode_rolls_a_system_up_to_its_hospitals(graph_db):
    conn, embedder = graph_db
    question = "Which Keystone Health sites had late consumable deliveries?"
    plain = _slugs(search(conn, embedder, question, k=5))
    # Text alone ranks Delta's identical complaint among Keystone's.
    assert plain.index("sig-complaint-delta") < max(
        plain.index("sig-complaint-harbor-north"), plain.index("sig-complaint-harbor-south")
    )
    trace = GraphTrace(mode="entity")
    ranked = _slugs(graph_search(conn, embedder, question, mode="entity", k=5, trace=trace))
    assert [m.name for m in trace.entities] == ["Keystone Health"]
    assert set(ranked[:2]) == {"sig-complaint-harbor-north", "sig-complaint-harbor-south"}
    assert ranked.index("sig-complaint-delta") > 1


def test_entity_mode_finds_a_contacts_account_the_text_never_names(graph_db):
    conn, embedder = graph_db
    question = "What has Jo Park reported?"
    assert "sig-complaint-harbor-south" not in _slugs(search(conn, embedder, question, k=5))
    result = graph_search(conn, embedder, question, mode="entity", k=5)
    assert _slugs(result)[0] == "sig-complaint-harbor-south"
    assert result.hits[0].expanded_from == "harbor-south-hospital"


def test_expand_adds_a_shared_account_note_the_question_never_matches(graph_db):
    conn, embedder = graph_db
    question = "RivalCo proposal"
    assert "sig-pricing-delta" not in _slugs(search(conn, embedder, question, k=5))
    ranked = _slugs(graph_search(conn, embedder, question, mode="expand", k=5))
    assert ranked[0] == "sig-competitor-delta"
    assert "sig-pricing-delta" in ranked


def test_a_hub_entity_is_not_stepped_through_unless_named(graph_db):
    conn, _ = graph_db
    graph = _Graph(conn, ["living"])
    via_pump = _neighbourhood(graph, "sig-complaint-delta", 1, protected=set())
    assert "sig-complaint-harbor-north" in via_pump  # shares Pump X
    with graph_params(HUB_MAX_CONTENT=2):  # Pump X links 3 complaints
        graph = _Graph(conn, ["living"])
        capped = _neighbourhood(graph, "sig-complaint-delta", 1, protected=set())
        assert "sig-complaint-harbor-north" not in capped
        graph = _Graph(conn, ["living"])
        named = _neighbourhood(graph, "pump-x", 1, protected={"pump-x"})
        assert "sig-complaint-harbor-north" in named


def test_neighbourhood_weights_one_step_fully_and_bridges_at_half(graph_db):
    conn, _ = graph_db
    near = _neighbourhood(_Graph(conn, ["living"]), "keystone-health", 1, protected=set())
    assert near["sig-complaint-harbor-north"] == pytest.approx(1.0)
    # Hospitals under the system are what a rollup asks about: full weight.
    assert near["harbor-north-hospital"] == pytest.approx(1.0)
    assert "keystone-health" not in near
    far = _neighbourhood(_Graph(conn, ["living"]), "sig-competitor-delta", 1, protected=set())
    assert far["delta-clinic"] == pytest.approx(g.BRIDGE_WEIGHT)
    assert far["sig-pricing-delta"] == pytest.approx(1.0)


def test_ppr_is_a_distribution_that_favours_the_restart_side(graph_db):
    conn, _ = graph_db
    graph = _Graph(conn, ["living"])
    nodes = [s for s, *_ in SIGNALS] + [a for a, *_ in ACCOUNTS] + ["pump-x", "rivalco"]
    mass = _ppr(graph, nodes, {"harbor-south-hospital": 1.0})
    assert sum(mass.values()) == pytest.approx(1.0)
    assert mass["sig-complaint-harbor-south"] > mass["sig-complaint-delta"]


def test_ppr_mode_ranks_the_named_contacts_account_first(graph_db):
    conn, embedder = graph_db
    ranked = _slugs(graph_search(conn, embedder, "What has Jo Park reported?", mode="ppr", k=5))
    assert ranked[0] == "sig-complaint-harbor-south"


def test_window_keeps_only_notes_with_a_claim_recorded_that_week(graph_db):
    conn, embedder = graph_db
    question = "late consumable deliveries"
    week = (date(2026, 8, 17), date(2026, 8, 23))
    for mode in ("off", "entity"):
        ranked = _slugs(graph_search(conn, embedder, question, mode=mode, k=5, window=week))
        assert ranked == ["sig-complaint-harbor-south"]
    changed = changed_since(
        conn, embedder, "What changed for Keystone Health?", since=date(2026, 8, 1),
        until=date(2026, 8, 31),
    )  # fmt: skip
    assert set(_slugs(changed)) == {"sig-complaint-harbor-north", "sig-complaint-harbor-south"}
