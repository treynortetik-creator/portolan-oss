"""A v2 world's graph and questions for the graph retrieval bench (evaluation only).

`build_world("pack:split:seed")` simulates and plans a world with the v2 generator
(`corpus/generator/v2`: simulate and plan only, no writer, no model call).
`seed_world_graph` turns it into a Portolan graph through the real pipeline with perfect
extraction, the way the demo graph is built from Halvard: every planted fact becomes a
claim on the note the merge picks, cited to the document the planner put it in. The
documents are stand-ins holding their facts' claim sentences (retrieval reads notes, not
documents), plus the CRM rows the planner filed, each field rendered as one sentence.

`world_questions` asks about that graph from the world's own structure, with gold notes
read back from the seeded claims: single-hop lookups, multi-hop questions through the
account hierarchy, a person, a CRM field, a competitor, an arc of events or a contact's
account, and "what changed for account X in the week of D". The wording is code
templates; none of it is training data.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
import yaml

from portolan.bench.graph_search import MAX_REQUIREMENTS, GraphQuestion, window_questions

__all__ = ["build_world", "seed_world_graph", "world_questions", "write_corpus"]

_FILE_KIND = {"call": "call", "ticket": "ticket"}  # everything else is written as an email
_TYPE_PHRASE = {
    "complaint": "complaints",
    "request": "requests",
    "risk": "risks",
    "praise": "praise",
    "competitor_mention": "competitor mentions",
    "commitment": "commitments",
    "pricing": "pricing issues",
    "usage": "usage updates",
    "contact_change": "contact changes",
    "decision": "decisions",
    "question": "open questions",
    "other": "other updates",
}


def build_world(spec: str) -> tuple[Any, Any]:
    """`(World, WorldDocPlan)` for `pack:split:seed`, eight months, no name guard."""
    from corpus.generator.v2.models import Split
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.planner import plan_world
    from corpus.generator.v2.rounds import PACKS_DIR
    from corpus.generator.v2.simulator import simulate_world

    pack_name, split, seed = spec.split(":")
    pack = load_pack(PACKS_DIR / f"{pack_name}.yaml")
    world = simulate_world(pack, seed=int(seed), months=8, split=Split(split))
    return world, plan_world(world, pack)


def _crm_sentence(name: str, field: str, value: str, as_of: date) -> str:
    if field == "arr":
        return f"{name}'s annual recurring revenue is {int(float(value)):,} dollars as of {as_of}."
    if field == "renewal_date":
        return f"{name}'s contract renews on {value}."
    if field == "renewal_stage":
        return f"{name}'s renewal stage is {value.replace('_', ' ')} as of {as_of}."
    if field == "competitors":
        return f"{name} is also talking to {value}."
    return f"{name}'s CRM {field.replace('_', ' ')} is {value} as of {as_of}."


def write_corpus(world: Any, plan: Any, root: Path) -> Path:
    """`root/truth-table.yaml` and `root/out/` (calls, tickets, email, crm, manifest.json)
    in the Halvard corpus layout `FileConnector` and `FakePlacements` read. Returns the
    `out` directory."""
    out = root / "out"
    for sub in ("calls", "tickets", "email", "crm"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    accounts = {a.id: a for a in world.accounts}
    people = {p.id: p for p in world.people}
    facts = {f.id: f for f in world.facts}
    names = {**{a.id: a.name for a in world.accounts}, **{p.id: p.name for p in world.people}}
    competitor_ids = {c.name: c.id for c in world.vendor.competitors}

    table = world.to_truth_table()
    placed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest: list[dict[str, Any]] = []
    for doc in plan.doc_plans:
        doc_facts = [facts[f] for f in doc.fact_ids if f in facts]
        if not doc_facts:
            continue
        kind = _FILE_KIND.get(doc.kind.value, "email")
        entries = []
        lines = []
        for i, fact in enumerate(doc_facts, 1):
            locator = f"00:{i // 60:02d}:{i % 60:02d}" if kind == "call" else f"¶{i}"
            entries.append({"fact_id": fact.id, "locator": locator, "excerpt": fact.claim})
            placed[fact.id].append(
                {"id": doc.id, "type": kind, "speaker": fact.speaker, "internal_only": False}
            )
            lines.append((names.get(fact.speaker, "Customer"), locator, fact.claim))
        account_name = accounts[doc.account_id].name if doc.account_id in accounts else ""
        if kind == "call":
            body = [f"[00:00:00] {world.vendor.name}: Call with {account_name}."]
            body += [f"[{loc}] {who}: {text}" for who, loc, text in lines]
            path = out / "calls" / f"{doc.id}.txt"
            path.write_text("\n\n".join(body) + "\n", encoding="utf-8")
        elif kind == "ticket":
            ticket = {
                "id": doc.id,
                "requester": lines[0][0],
                "account": account_name,
                "created": doc.date.isoformat(),
                "subject": f"{account_name} ticket",
                "comments": [
                    {"number": n, "author": who, "body": text}
                    for n, (who, _loc, text) in enumerate(lines, 1)
                ],
            }
            path = out / "tickets" / f"{doc.id}.json"
            path.write_text(json.dumps(ticket, indent=2) + "\n", encoding="utf-8")
        else:
            sender = people.get(doc.participants[0]) if doc.participants else None
            headers = [
                f"Message-ID: <{doc.id}@{world.vendor.message_id_domain}>",
                f"From: {sender.name} <{sender.email}>" if sender else "From: Customer",
                f"Date: {doc.date.isoformat()} 09:00:00 -0700",
                f"Subject: {account_name} ({doc.kind.value})",
            ]
            text = "\n".join(headers) + "\n\n" + "\n\n".join(t for _w, _l, t in lines) + "\n"
            path = out / "email" / f"{doc.id}.txt"
            path.write_text(text, encoding="utf-8")
        manifest.append(
            {
                "id": doc.id,
                "type": kind,
                "path": str(path.relative_to(out)),
                "date": doc.date.isoformat(),
                "participants": list(doc.participants),
                "accounts": [doc.account_id] if doc.account_id else [],
                "facts": entries,
            }
        )

    crm_facts = []
    for record in plan.crm_accounts:
        name = accounts[record.account_id].name
        entries = []
        for n, ff in enumerate(record.field_facts, 1):
            value = "; ".join(str(record.row.get(f, "")) for f in ff.fields)
            if not value.strip("; "):
                continue
            claim = _crm_sentence(name, ff.fields[0], value, ff.date)
            entities = [record.account_id] + [
                competitor_ids[c.strip()]
                for c in value.split(";")
                if ff.fields[0] == "competitors" and c.strip() in competitor_ids
            ]
            crm_facts.append(
                {
                    "id": ff.id,
                    "type": ff.fact_type.value,
                    "claim": claim,
                    "entities": entities,
                    "date": ff.date.isoformat(),
                    "valid_from": None,
                    "valid_to": None,
                    "supersedes": None,
                    "sources": [{"id": record.id, "type": "crm", "speaker": None}],
                    "phrasing": None,
                }
            )
            entries.append(
                {
                    "fact_id": ff.id,
                    "locator": f"crm/accounts.csv#{record.crm_id}#{n}",
                    "excerpt": claim,
                }
            )
        dates = [ff.date for ff in record.field_facts]
        manifest.append(
            {
                "id": record.id,
                "type": "crm",
                "path": "crm/accounts.csv",
                "date": max(dates).isoformat() if dates else world.window_end.isoformat(),
                "participants": [],
                "accounts": [record.account_id],
                "facts": entries,
            }
        )

    for fact in table["facts"]:
        fact["sources"] = placed.get(fact["id"], [])
    table["facts"] = [f for f in table["facts"] if f["sources"]] + crm_facts
    (root / "truth-table.yaml").write_text(
        yaml.safe_dump(table, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps({"sources": manifest}, indent=1) + "\n", encoding="utf-8"
    )

    crm_by_account = {r.account_id: r for r in plan.crm_accounts}
    with (out / "crm" / "accounts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["crm_id", "name", "kind", "parent_crm_id", "domain", "arr", "region", "status",
             "owner_crm_id", "branch"]
        )  # fmt: skip
        for a in world.accounts:
            row = crm_by_account[a.id].row if a.id in crm_by_account else {}
            parent = accounts[a.parent].crm_id if a.parent in accounts else ""
            writer.writerow(
                [a.crm_id, a.name, a.kind.value, parent, a.domain, int(a.arr), a.region,
                 row.get("status", a.status.value), "", ""]
            )  # fmt: skip
    with (out / "crm" / "contacts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crm_id", "name", "title", "account_crm_id", "email", "status"])
        for c in table["contacts"]:
            account = accounts.get(c["account"])
            writer.writerow(
                [c["crm_id"], c["name"], c["title"], account.crm_id if account else "",
                 c["email"], c["status"]]
            )  # fmt: skip
    return out


def seed_world_graph(
    conn: psycopg.Connection, world: Any, plan: Any, workdir: Path, embedder: Any
) -> dict[str, int]:
    """The world's graph on an empty, migrated database: `pipeline.run` with the fake
    extractor over `write_corpus`'s stand-in documents, indexed with `embedder`.
    Returns row counts."""
    from portolan.config import Settings
    from portolan.connectors.file_connector import FileConnector
    from portolan.fake_extractor import FakePlacements, build_fake_llm_client
    from portolan.pipeline import run as pipeline_run
    from portolan.registry import Registry
    from portolan.store import LocalStore

    corpus = write_corpus(world, plan, workdir / "world")
    settings = Settings(
        database_url="postgresql://unused/portolan_test_graph_world",
        tenant="test",
        store_root=workdir / "store",
        llm_provider="fake",
        embed_provider="fake",
    )
    pipeline_run(
        conn,
        connector=FileConnector(corpus),
        llm=build_fake_llm_client(FakePlacements(corpus), tenant=settings.tenant),
        embedder=embedder,
        registry=Registry.load(conn),
        store=LocalStore(workdir / "store"),
        settings=settings,
    )
    counts = {}
    with conn.cursor() as cur:
        for name in ("notes", "claims", "links", "sources", "registry_entities"):
            cur.execute(f"SELECT count(*) FROM {name}")
            counts[name] = cur.fetchone()[0]
    return counts


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


@dataclass
class _GraphIndex:
    fact_notes: dict[str, set[str]]  # fact id -> notes holding its claim
    account_note: dict[str, str]  # world account id -> account note slug
    speaker_claims: dict[str, list[tuple[str, str]]]  # person id -> [(note, claim text)]


def _index(conn: psycopg.Connection, world: Any, plan: Any) -> _GraphIndex:
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug, claim_text, speaker FROM claims")
        claims = cur.fetchall()
        cur.execute(
            "SELECT crm_id, note_slug FROM registry_entities WHERE kind = 'account' "
            "AND note_slug IS NOT NULL"
        )
        by_crm = dict(cur.fetchall())
    by_text: dict[str, set[str]] = defaultdict(set)
    speaker_claims: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for note, text, speaker in claims:
        by_text[text.split(" ([[")[0].strip()].add(note)
        if speaker:
            speaker_claims[speaker].append((note, text))
    texts = {f.id: f.claim.strip() for f in world.facts}
    fact_notes = {fid: by_text.get(t, set()) for fid, t in texts.items()}
    for record in plan.crm_accounts:
        name = next(a.name for a in world.accounts if a.id == record.account_id)
        for ff in record.field_facts:
            value = "; ".join(str(record.row.get(f, "")) for f in ff.fields)
            fact_notes[ff.id] = by_text.get(
                _crm_sentence(name, ff.fields[0], value, ff.date), set()
            )
    account_note = {a.id: by_crm[a.crm_id] for a in world.accounts if a.crm_id in by_crm}
    return _GraphIndex(fact_notes, account_note, speaker_claims)


def _month(d: date) -> str:
    return d.strftime("%B %Y")


def world_questions(
    conn: psycopg.Connection, world: Any, plan: Any, *, seed: int = 0
) -> dict[str, tuple[GraphQuestion, ...]]:
    """{"single", "multi", "window"} question sets for the seeded world graph."""
    rng = random.Random(f"graph-bench:{world.id}:{seed}")
    ix = _index(conn, world, plan)
    accounts = {a.id: a for a in world.accounts}
    people = {p.id: p for p in world.people}
    products = {p.id: p.name for p in world.vendor.products}
    competitors = {c.id: c.name for c in world.vendor.competitors}
    facts = [f for f in world.facts if ix.fact_notes.get(f.id)]
    by_account_type: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in facts:
        by_account_type[(f.account_id, f.fact_type.value)] |= ix.fact_notes[f.id]

    # Single hop: one fact, the account and product named.
    single: list[GraphQuestion] = []
    by_type: dict[str, list[Any]] = defaultdict(list)
    for f in facts:
        by_type[f.fact_type.value].append(f)
    for t in sorted(by_type):
        for f in rng.sample(by_type[t], min(3, len(by_type[t]))):
            product = next((products[e] for e in f.entities if e in products), None)
            about = f" about {product}" if product else ""
            name = accounts[f.account_id].name
            single.append(
                GraphQuestion(
                    f"w_single_{len(single) + 1:02d}",
                    f"What {_TYPE_PHRASE[t]} did {name} raise{about} in {_month(f.date)}?",
                    t,
                    (tuple(sorted(ix.fact_notes[f.id])),),
                    "single",
                )
            )

    multi: list[GraphQuestion] = []

    def add(category: str, question: str, reqs: list[set[str]]) -> None:
        reqs = [r for r in reqs if r]
        if 2 <= len(reqs) <= MAX_REQUIREMENTS:
            multi.append(
                GraphQuestion(
                    f"w_multi_{len(multi) + 1:02d}",
                    question,
                    category,
                    tuple(tuple(sorted(r)) for r in reqs),
                )
            )

    # Hierarchy: a system's facilities, one fact type.
    children: dict[str, list[str]] = defaultdict(list)
    for a in world.accounts:
        if a.parent:
            children[a.parent].append(a.id)
    for parent, kids in sorted(children.items()):
        types = sorted({t for k in kids for (acc, t) in by_account_type if acc == k})
        for t in rng.sample(types, min(3, len(types))):
            hit_kids = [k for k in kids if by_account_type.get((k, t))]
            reqs = [{ix.account_note[k]} for k in hit_kids if k in ix.account_note]
            reqs += [by_account_type[(k, t)] for k in hit_kids]
            add(
                "hierarchy",
                f"Which of {accounts[parent].name}'s facilities have raised {_TYPE_PHRASE[t]}, "
                "and what were they?",
                reqs,
            )

    # Person: what one customer contact has said about one product.
    spoken: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in facts:
        person = people.get(f.speaker)
        if not person or person.side.value != "customer":
            continue
        for e in f.entities:
            if e in products:
                spoken[(f.speaker, e)] |= ix.fact_notes[f.id]
    pairs = sorted(k for k, v in spoken.items() if len(v) >= 2)
    for speaker, product in rng.sample(pairs, min(6, len(pairs))):
        add(
            "person",
            f"What has {people[speaker].name} raised with us about {products[product]}?",
            [{n} for n in sorted(spoken[(speaker, product)])],
        )

    # CRM field bridge: an account picked out by its renewal date or its ARR.
    renewals: dict[str, list[str]] = defaultdict(list)
    for a in world.accounts:
        renewals[_month(a.renewal_date)].append(a.id)
    for record in plan.crm_accounts:
        acc = record.account_id
        crm_notes = {ff.fields[0]: ix.fact_notes.get(ff.id, set()) for ff in record.field_facts}
        types = sorted(t for (a, t) in by_account_type if a == acc and t in ("complaint", "risk"))
        if not types:
            continue
        t = rng.choice(types)
        month = _month(accounts[acc].renewal_date)
        if len(renewals[month]) == 1 and crm_notes.get("renewal_date"):
            add(
                "crm_field",
                f"The account whose contract renews in {month}: what {_TYPE_PHRASE[t]} has it "
                "raised?",
                [crm_notes["renewal_date"], by_account_type[(acc, t)]],
            )

    # Competitor bridge: the one account that mentioned a competitor in a given month,
    # and its pricing issues.
    mentions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in facts:
        if f.fact_type.value == "competitor_mention":
            for e in f.entities:
                if e in competitors:
                    mentions[(e, _month(f.date))].add(f.account_id)
    unique = sorted(
        (cid, month, next(iter(accs)))
        for (cid, month), accs in mentions.items()
        if len(accs) == 1 and by_account_type.get((next(iter(accs)), "pricing"))
    )
    for cid, month, acc in rng.sample(unique, min(4, len(unique))):
        add(
            "competitor",
            f"The account that brought up {competitors[cid]} in {month}: what pricing issues "
            "has it raised?",
            [by_account_type[(acc, "competitor_mention")], by_account_type[(acc, "pricing")]],
        )

    # Arc: what followed an account's first event in a chain.
    events = {e.id: e for e in world.events}
    for arc in world.arcs:
        chain = [events[e] for e in arc.event_ids if e in events]
        if len(chain) < 2:
            continue
        first = chain[0]
        later = {n for f in facts if f.event_id in {e.id for e in chain[1:]}
                 for n in ix.fact_notes[f.id]}  # fmt: skip
        first_notes = {n for f in facts if f.event_id == first.id for n in ix.fact_notes[f.id]}
        if not later or not first_notes:
            continue
        add(
            "arc",
            f"After the {first.type.value.replace('_', ' ')} at {accounts[arc.account_id].name} "
            f"in {_month(first.date)}, what happened next?",
            [first_notes, *({n} for n in sorted(later - first_notes))],
        )

    # Person to account: the account is named only through a contact (the registry hop).
    contacts = sorted(
        p.id
        for p in world.people
        if p.side.value == "customer" and p.tenure.end is None
        and by_account_type.get((p.account_id, "risk"))
    )  # fmt: skip
    for pid in rng.sample(contacts, min(4, len(contacts))):
        person = people[pid]
        multi.append(
            GraphQuestion(
                f"w_multi_{len(multi) + 1:02d}",
                f"What risks have come up at the account where {person.name} works?",
                "person_account",
                (tuple(sorted(by_account_type[(person.account_id, "risk")])),),
            )
        )

    window = window_questions(conn, since=world.window_start, limit=12)
    return {"single": tuple(single), "multi": tuple(multi), "window": window}
