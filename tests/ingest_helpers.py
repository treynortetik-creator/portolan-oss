"""Shared fixtures for the ingest test suite: a FakeRegistry built from
`corpus/truth-table.yaml`, and loaders that turn real files under
`corpus/out/` into `RawSource` objects the way a (not-yet-built)
`FileConnector` eventually will.

Not a `test_*.py` module on purpose — pytest must not collect it as a
test file; it is imported by `test_ingest_*.py` files instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from portolan.extract.schema import EntityKind, SourceType
from portolan.ingest.models import Participant, RawSource

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = REPO_ROOT / "corpus" / "out"
TRUTH_TABLE_PATH = REPO_ROOT / "corpus" / "truth-table.yaml"


# --------------------------------------------------------------------
# FakeRegistry
# --------------------------------------------------------------------


@dataclass
class FakeEntity:
    """Structurally satisfies `portolan.ingest.filter.RegistryEntityLike`
    (and the `Entity` shape in build-contracts.md, trimmed to the fields
    this wave's tests need)."""

    id: str
    kind: EntityKind
    canonical_name: str
    email: str | None = None
    email_domain: str | None = None
    aliases: list[str] = field(default_factory=list)
    parent_id: str | None = None


@dataclass
class FakeMatch:
    """Structurally satisfies `RegistryMatchLike`."""

    entity_id: str
    canonical_name: str
    kind: EntityKind


class FakeRegistry:
    """Test double for `portolan.registry.Registry`, implementing only the
    four methods `portolan.ingest.filter` depends on: `by_email`,
    `by_domain`, `scan`, `ancestors`. `portolan.registry` itself is never
    imported (build instructions).
    """

    def __init__(self, entities: list[FakeEntity]) -> None:
        self._entities: dict[str, FakeEntity] = {e.id: e for e in entities}
        self._by_email = {e.email.lower(): e for e in entities if e.email}
        self._by_domain: dict[str, list[FakeEntity]] = {}
        for e in entities:
            if e.email_domain:
                self._by_domain.setdefault(e.email_domain.lower(), []).append(e)

    def by_email(self, email: str) -> FakeEntity | None:
        return self._by_email.get(email.lower())

    def by_domain(self, domain: str) -> list[FakeEntity]:
        return list(self._by_domain.get(domain.lower(), []))

    def scan(self, text: str) -> list[FakeMatch]:
        matches: list[FakeMatch] = []
        seen: set[str] = set()
        for entity in self._entities.values():
            if entity.id in seen:
                continue
            for name in (entity.canonical_name, *entity.aliases):
                if not name:
                    continue
                if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                    matches.append(
                        FakeMatch(
                            entity_id=entity.id,
                            canonical_name=entity.canonical_name,
                            kind=entity.kind,
                        )
                    )
                    seen.add(entity.id)
                    break
        return matches

    def ancestors(self, entity_id: str) -> list[str]:
        chain: list[str] = []
        current = self._entities.get(entity_id)
        while current is not None and current.parent_id:
            chain.append(current.parent_id)
            current = self._entities.get(current.parent_id)
        return chain

    def entity(self, entity_id: str) -> FakeEntity:
        return self._entities[entity_id]


def _domain_of(email: str) -> str:
    return email.rsplit("@", 1)[-1].lower()


def load_truth_table() -> dict:
    return yaml.safe_load(TRUTH_TABLE_PATH.read_text(encoding="utf-8"))


def build_fake_registry() -> FakeRegistry:
    """FakeRegistry seeded from the truth table's accounts, contacts,
    products, competitors, internal_people, aliases and domains.

    `maria-chen-personal` (the truth table's deliberately-unregistered
    contact, `note: not_in_crm`) is excluded on purpose: the whole point
    of that fixture is that the registry does NOT know her.

    One addition beyond the truth table's literal alias lists: "Tidewater"
    on `tidewater-t2`, standing in for the brand-level alias a real
    registry accumulates from usage over time (`registry_aliases.source =
    'extractor'`/`'tracker'`) — the per-product alias lists in the truth
    table (`T2`, `T3`, `Link`, ...) don't include the bare brand word, but
    fact F08 is planted as "mentions 'Tidewater consumables' once, so the
    registry scan should catch it", which requires *something* in the
    registry to match on "Tidewater" alone.
    """
    tt = load_truth_table()
    entities: list[FakeEntity] = []

    for account in tt["accounts"]:
        entities.append(
            FakeEntity(
                id=account["id"],
                kind=EntityKind.ACCOUNT,
                canonical_name=account["name"],
                email_domain=account.get("domain"),
                aliases=list(account.get("aliases", [])),
                parent_id=account.get("parent"),
            )
        )

    for contact in tt["contacts"]:
        if contact.get("note") == "not_in_crm":
            continue
        entities.append(
            FakeEntity(
                id=contact["id"],
                kind=EntityKind.CONTACT,
                canonical_name=contact["name"],
                email=contact.get("email"),
                email_domain=_domain_of(contact["email"]) if contact.get("email") else None,
            )
        )

    for product in tt["meta"]["products"]:
        aliases = list(product.get("aliases", []))
        if product["id"] == "tidewater-t2":
            aliases.append("Tidewater")
        entities.append(
            FakeEntity(
                id=product["id"],
                kind=EntityKind.PRODUCT,
                canonical_name=product["name"],
                aliases=aliases,
            )
        )

    for competitor in tt["meta"]["competitors"]:
        entities.append(
            FakeEntity(
                id=competitor["id"],
                kind=EntityKind.COMPETITOR,
                canonical_name=competitor["name"],
                aliases=list(competitor.get("aliases", [])),
            )
        )

    for person in tt["meta"]["internal_people"]:
        entities.append(
            FakeEntity(
                id=person["id"],
                kind=EntityKind.INTERNAL_PERSON,
                canonical_name=person["name"],
                email=person.get("email"),
                email_domain=_domain_of(person["email"]) if person.get("email") else None,
            )
        )

    return FakeRegistry(entities)


def people_by_id() -> dict[str, dict[str, str | None]]:
    """id -> {"name", "email"} for every contact (registered or not) and
    internal person, used to build realistic `Participant` lists for
    corpus-derived `RawSource` fixtures without hand-typing email
    addresses. Kept separate from `build_fake_registry` because
    `maria-chen-personal` belongs here (a connector still records who
    sent an email even when the registry can't place them) but not there.
    """
    tt = load_truth_table()
    people: dict[str, dict[str, str | None]] = {}
    for contact in tt["contacts"]:
        people[contact["id"]] = {"name": contact["name"], "email": contact.get("email")}
    for person in tt["meta"]["internal_people"]:
        people[person["id"]] = {"name": person["name"], "email": person.get("email")}
    return people


# --------------------------------------------------------------------
# Corpus loaders: manifest.json + corpus/out/** -> RawSource
# --------------------------------------------------------------------

_manifest_cache: dict | None = None


def _manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = json.loads((CORPUS_OUT / "manifest.json").read_text(encoding="utf-8"))
    return _manifest_cache


def _manifest_entry(source_id: str) -> dict:
    for entry in _manifest()["sources"]:
        if entry["id"] == source_id:
            return entry
    raise KeyError(f"no manifest entry for source id {source_id!r}")


def _participants_for(entry: dict) -> list[Participant]:
    people = people_by_id()
    participants: list[Participant] = []
    for person_id in entry.get("participants", []):
        info = people.get(person_id)
        if info is None:
            participants.append(Participant(name=person_id))
        else:
            participants.append(Participant(name=info["name"], email=info["email"]))
    return participants


_HEADER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z-]*):\s?(.*)$")
_ADDR_RE = re.compile(r"^(?P<name>.*?)\s*<(?P<email>[^>]+)>$")


def _parse_email_file(path: Path) -> tuple[dict[str, str], str]:
    lines = path.read_text(encoding="utf-8").split("\n")
    headers: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            break
        match = _HEADER_LINE_RE.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
        i += 1
    body = "\n".join(lines[i:])
    return headers, body


def _parse_address_list(value: str) -> list[Participant]:
    participants: list[Participant] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _ADDR_RE.match(chunk)
        if match:
            participants.append(
                Participant(name=match.group("name").strip(), email=match.group("email").strip())
            )
        else:
            participants.append(Participant(name=chunk))
    return participants


def load_email(source_id: str, *, account_hints: list[str] | None = None) -> RawSource:
    """Parses `corpus/out/email/<source_id>.txt` header block and body,
    the way a connector would: participants and Message-ID come from
    headers (not part of the extracted text), and only the body is
    `RawSource.text`.
    """
    path = CORPUS_OUT / "email" / f"{source_id}.txt"
    headers, body = _parse_email_file(path)

    from_participants = _parse_address_list(headers.get("From", ""))
    to_participants = _parse_address_list(headers.get("To", ""))
    date_token = headers.get("Date", "").split(" ")[0]
    source_date = date.fromisoformat(date_token) if date_token else date.today()
    message_id = headers.get("Message-ID", "").strip("<>") or source_id

    return RawSource(
        source_id=source_id,
        source_type=SourceType.EMAIL,
        source_date=source_date,
        participants=[*from_participants, *to_participants],
        account_hints=account_hints or [],
        text=body.strip("\n") + "\n",
        metadata={
            "message_id": message_id,
            "subject": headers.get("Subject", ""),
            "from_email": from_participants[0].email if from_participants else None,
        },
    )


def load_call(source_id: str, *, account_hints: list[str] | None = None) -> RawSource:
    entry = _manifest_entry(source_id)
    path = CORPUS_OUT / entry["path"]
    return RawSource(
        source_id=source_id,
        source_type=SourceType.CALL,
        source_date=date.fromisoformat(entry["date"]),
        participants=_participants_for(entry),
        account_hints=account_hints or [],
        text=path.read_text(encoding="utf-8"),
        metadata={},
    )


def load_ticket(source_id: str, *, account_hints: list[str] | None = None) -> RawSource:
    entry = _manifest_entry(source_id)
    path = CORPUS_OUT / entry["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    comments = payload["comments"]
    flat_text = "\n\n".join(f"{c['author']}: {c['body']}" for c in comments)
    return RawSource(
        source_id=source_id,
        source_type=SourceType.TICKET,
        source_date=date.fromisoformat(entry["date"]),
        participants=_participants_for(entry),
        account_hints=account_hints or [],
        text=flat_text,
        metadata={"comments": comments, "subject": payload.get("subject", "")},
    )


def load_rss(source_id: str, *, account_hints: list[str] | None = None) -> RawSource:
    entry = _manifest_entry(source_id)
    feed_path = CORPUS_OUT / entry["path"]
    items = json.loads(feed_path.read_text(encoding="utf-8"))
    item = next(i for i in items if i["id"] == source_id)
    text = f"{item['title']}\n\n{item.get('summary', '')}".strip() + "\n"
    return RawSource(
        source_id=source_id,
        source_type=SourceType.RSS,
        source_date=date.fromisoformat(item["date"]),
        participants=[],
        account_hints=account_hints or [],
        text=text,
        metadata={"link": item["link"]},
    )


def load_crm_row(*, object_type: str, crm_id: str, text: str, row_date: date) -> RawSource:
    """CRM rows have no single natural file-per-source shape in the
    corpus (`crm/accounts.csv`, `crm/opportunities.csv` each hold many
    rows); callers build the row's rendered text and pass it straight
    through, mirroring what a `FileConnector`'s CRM reader will eventually
    do per row.
    """
    return RawSource(
        source_id=f"S-crm-{crm_id}",
        source_type=SourceType.CRM,
        source_date=row_date,
        participants=[],
        account_hints=[],
        text=text,
        metadata={"object": object_type, "crm_id": crm_id},
    )
