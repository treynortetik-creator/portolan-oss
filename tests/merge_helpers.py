"""Shared fixtures/helpers for the `portolan.notes`/`portolan.merge` test
suite: the real `Registry` imported from `corpus/out/crm/` (per the build
instructions: "the real registry imported from corpus/out/crm via
Registry.import_crm with --meta equivalent"), and small builders for
`SourceRow`/`Fact`/`Action` fixtures drawn from `corpus/truth-table.yaml`.

Not a `test_*.py` module on purpose (mirrors `tests/ingest_helpers.py`) --
pytest must not collect it as a test file; it is imported by
`test_notes*.py`/`test_merge*.py` files instead.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from portolan.extract.schema import Action, Fact, FactType
from portolan.ingest.models import Participant, SourceRow
from portolan.registry import Registry
from portolan.registry.cli import _load_meta

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_CSV = REPO_ROOT / "corpus" / "out" / "crm" / "accounts.csv"
CONTACTS_CSV = REPO_ROOT / "corpus" / "out" / "crm" / "contacts.csv"
TRUTH_TABLE = REPO_ROOT / "corpus" / "truth-table.yaml"

# crm_id constants, named after the truth table's own slugs for
# readability in test failures (Entity.id == registry_entities.crm_id,
# real for accounts/contacts, the meta dict's own id for products/
# competitors -- see portolan/registry/registry.py's module docstring).
PINECREST = "A-1001"
LARKSPUR = "A-1002"
LARKSPUR_MEMORIAL = "A-1003"
LARKSPUR_NORTH = "A-1004"
ASHGROVE = "A-1010"
ASHGROVE_MEDICAL = "A-1011"
COBALT_RIDGE = "A-1020"
NORTHLAKE = "A-1030"
NORTHLAKE_GENERAL = "A-1031"
TIDEWATER_T2 = "tidewater-t2"
TIDEWATER_T3 = "tidewater-t3"
HALVARDCARE = "halvardcare"
CORVANE = "corvane"
SAM_OKAFOR = "sam-okafor"


@pytest.fixture(name="loaded_registry")
def loaded_registry_fixture(db) -> Registry:
    """A `Registry` imported from the real corpus CSVs plus the truth
    table's meta block (products, competitors, internal_people) and
    per-account aliases, then freshly `.load()`-ed. Mirrors
    `tests/test_registry.py`'s fixture of the same name/shape.

    Registered under the explicit name `loaded_registry` (rather than
    relying on the function's own name) so importing it into a test module
    doesn't collide with that module's own `loaded_registry` fixture
    *parameter* -- the importable Python symbol is `loaded_registry_fixture`,
    the fixture pytest resolves by name is `loaded_registry`.
    """
    products, competitors, internal_people, account_aliases = _load_meta(TRUTH_TABLE)
    Registry().import_crm(
        db,
        ACCOUNTS_CSV,
        CONTACTS_CSV,
        products=products,
        competitors=competitors,
        internal_people=internal_people,
        account_aliases=account_aliases,
    )
    return Registry.load(db)


def make_source_row(
    conn,
    slug: str,
    *,
    source_type: str = "call",
    source_date: date = date(2026, 8, 5),
    participants: list[Participant] | None = None,
) -> SourceRow:
    """Insert a minimal `sources` row and return the matching `SourceRow`
    (build-contracts.md's ingest model). Real code path is
    `portolan.ingest.intake.intake`; merge/notes tests only need a row to
    exist for the FK from `claims.source_slug`/`links` provenance edges."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sources (slug, source_type, source_date, extraction_status, s3_key)
            VALUES (%s, %s, %s, 'pending', %s)
            ON CONFLICT (slug) DO NOTHING
            """,
            (slug, source_type, source_date, f"originals/{slug}"),
        )
    conn.commit()
    return SourceRow(
        slug=slug,
        source_type=source_type,
        source_date=source_date,
        ingested_at=datetime.now(UTC),
        participants=participants or [],
    )


def build_fact(
    fact_id: str,
    fact_type: FactType,
    claim: str,
    entities: list[str],
    *,
    locator: str = "00:00:01",
    quote: str | None = None,
    speaker: str | None = None,
    valid_from: date | None = None,
    valid_to: date | None = None,
    severity: str | None = None,
    sentiment: int = 0,
    confidence: float = 0.9,
) -> Fact:
    """A `Fact` fixture built the way the build instructions ask ("raw
    fixtures built from corpus/truth-table.yaml facts"). `entities` here
    are plain-language *mentions* (canonical names or aliases), matching
    what a real extractor would write into `claim` and `entities` under
    fact rule 3 ("No pronouns... Name the entity every time") -- not
    truth-table slugs, which never appear in `registry_entities` (see
    `portolan/registry/registry.py`'s module docstring on the id scheme).
    """
    return Fact(
        id=fact_id,
        type=fact_type,
        claim=claim,
        entities=entities,
        locator=locator,
        quote=quote,
        speaker=speaker,
        valid_from=valid_from,
        valid_to=valid_to,
        severity=severity,
        sentiment=sentiment,
        confidence=confidence,
    )


def build_action(
    what: str,
    *,
    owner_mention: str | None = None,
    due: date | None = None,
    locator: str = "00:00:01",
) -> Action:
    return Action(what=what, owner_mention=owner_mention, due=due, locator=locator)
