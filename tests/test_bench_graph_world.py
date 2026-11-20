"""`portolan.bench.graph_world.write_corpus`: a simulated v2 world laid out as the
Halvard corpus, so the fake extractor plants every fact where the planner put it.

No database: seeding the world graph through the pipeline takes about half an hour on
the dev machine (`scripts/bench/graph_retrieval.py --graph world`)."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from portolan.bench.graph_world import build_world, write_corpus
from portolan.connectors.file_connector import FileConnector
from portolan.extract.schema import QUOTE_REQUIRED_TYPES
from portolan.fake_extractor import FakePlacements


@pytest.fixture(scope="module")
def world_corpus(tmp_path_factory):
    world, plan = build_world("medical_devices:test:7")
    out = write_corpus(world, plan, tmp_path_factory.mktemp("world") / "w")
    return world, plan, out


def test_every_planted_fact_is_placed_in_a_document_that_quotes_it(world_corpus):
    _world, plan, out = world_corpus
    placements = FakePlacements(out)
    connector = FileConnector(out)
    refs = {r.source_id: r for r in connector.list_sources()}
    planted = {f for d in plan.doc_plans for f in d.fact_ids}
    table = yaml.safe_load((out.parent / "truth-table.yaml").read_text(encoding="utf-8"))
    doc_facts = [f for f in table["facts"] if not f["id"].startswith("CF-")]
    assert {f["id"] for f in doc_facts} == planted
    for doc in plan.doc_plans:
        if not doc.fact_ids:
            continue
        assert placements.has_placements(doc.id)
        text = connector.fetch(refs[doc.id]).text
        result = placements.extraction_result_for(doc.id)
        assert len(result.facts) == len(doc.fact_ids)
        for fact in result.facts:
            if fact.type in QUOTE_REQUIRED_TYPES:
                assert fact.quote and fact.quote in " ".join(text.split())


def test_documents_carry_the_planned_dates_and_crm_rows_their_fields(world_corpus):
    world, plan, out = world_corpus
    manifest = {s["id"]: s for s in json.loads((out / "manifest.json").read_text())["sources"]}
    doc = next(d for d in plan.doc_plans if d.fact_ids)
    assert manifest[doc.id]["date"] == doc.date.isoformat()
    with (out / "crm" / "accounts.csv").open(encoding="utf-8") as fh:
        rows = {r["crm_id"]: r for r in csv.DictReader(fh)}
    child = next(a for a in world.accounts if a.parent)
    parent = next(a for a in world.accounts if a.id == child.parent)
    assert rows[child.crm_id]["parent_crm_id"] == parent.crm_id
    record = plan.crm_accounts[0]
    crm_ids = {f["fact_id"] for f in manifest[record.id]["facts"]}
    assert crm_ids == {ff.id for ff in record.field_facts if record.row.get(ff.fields[0])}
