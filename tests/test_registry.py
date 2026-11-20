"""Tests for `portolan/registry/` (wave A3), against the real CRM export
in `corpus/out/crm/` and the truth table in `corpus/truth-table.yaml`.

Id scheme under test: `Entity.id` is always `registry_entities.crm_id` —
the real CRM id for an account/contact, the meta dict's own `id` for a
product/competitor/internal person, or a synthesised slug for a dirty-CRM
row with no id at all. See `portolan/registry/registry.py`'s module
docstring for the full reasoning. Concretely: Larkspur Memorial Hospital
is `"A-1003"`, not `"larkspur-memorial"` — the truth table's own slugs are
never written to the database, only used (via `--meta`) to seed extra
account aliases.

Run with the module's own test database:
    createdb portolan_test_registry
    DATABASE_URL=postgresql://localhost/portolan_test_registry \
        uv run pytest tests/test_registry.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from portolan.registry.cli import _load_meta
from portolan.registry.registry import Registry, add_alias, queue_unknown

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_CSV = REPO_ROOT / "corpus" / "out" / "crm" / "accounts.csv"
CONTACTS_CSV = REPO_ROOT / "corpus" / "out" / "crm" / "contacts.csv"
TRUTH_TABLE = REPO_ROOT / "corpus" / "truth-table.yaml"
S_EMAIL_062 = REPO_ROOT / "corpus" / "out" / "email" / "S-email-062.txt"

# crm_id constants from corpus/out/crm/accounts.csv and contacts.csv,
# named after the truth table's own ids for readability in failures.
PINECREST = "A-1001"
LARKSPUR = "A-1002"
LARKSPUR_MEMORIAL = "A-1003"
ASHGROVE = "A-1010"
ASHGROVE_MEDICAL = "A-1011"
ASHGROVE_RIVERSIDE = "A-1012"
MARIA_CHEN = "C-2001"


@pytest.fixture()
def loaded_registry(db) -> Registry:
    """A Registry imported from the real corpus CSVs plus the truth
    table's meta block (products, competitors, internal_people) and
    per-account aliases, then freshly `.load()`-ed."""
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


def _insert_source(cur, slug: str) -> None:
    cur.execute(
        """
        INSERT INTO sources (slug, source_type, source_date, extraction_status)
        VALUES (%s, 'email', '2026-08-25', 'done')
        """,
        (slug,),
    )


# --------------------------------------------------------------------
# Import
# --------------------------------------------------------------------


def _csv_row_count(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def test_import_counts_match_the_real_crm_export(db):
    products, competitors, internal_people, account_aliases = _load_meta(TRUTH_TABLE)
    report = Registry().import_crm(
        db,
        ACCOUNTS_CSV,
        CONTACTS_CSV,
        products=products,
        competitors=competitors,
        internal_people=internal_people,
        account_aliases=account_aliases,
    )

    expected_accounts = _csv_row_count(ACCOUNTS_CSV)
    expected_contacts = _csv_row_count(CONTACTS_CSV)
    expected_products = len(products)
    expected_competitors = len(competitors)
    expected_internal_people = len(internal_people)

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'account'")
        assert cur.fetchone()[0] == expected_accounts
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'contact'")
        assert cur.fetchone()[0] == expected_contacts
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'product'")
        assert cur.fetchone()[0] == expected_products
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'competitor'")
        assert cur.fetchone()[0] == expected_competitors
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'internal_person'")
        assert cur.fetchone()[0] == expected_internal_people

    # The real CRM export is clean (no duplicate domains/emails to merge,
    # nothing ambiguous), so every row plus every meta entry is a fresh
    # create and nothing is queued for review.
    expected_created = (
        expected_accounts
        + expected_contacts
        + expected_products
        + expected_competitors
        + expected_internal_people
    )
    assert report.created == expected_created
    assert report.updated == 0
    assert report.merged == 0
    assert report.queued == 0


def test_reimport_is_idempotent_and_updates_not_duplicates(db):
    products, competitors, internal_people, account_aliases = _load_meta(TRUTH_TABLE)
    registry = Registry()
    registry.import_crm(
        db,
        ACCOUNTS_CSV,
        CONTACTS_CSV,
        products=products,
        competitors=competitors,
        internal_people=internal_people,
        account_aliases=account_aliases,
    )
    second = registry.import_crm(
        db,
        ACCOUNTS_CSV,
        CONTACTS_CSV,
        products=products,
        competitors=competitors,
        internal_people=internal_people,
        account_aliases=account_aliases,
    )

    expected_total = (
        _csv_row_count(ACCOUNTS_CSV)
        + _csv_row_count(CONTACTS_CSV)
        + len(products)
        + len(competitors)
        + len(internal_people)
    )

    assert second.created == 0
    assert second.updated == expected_total
    assert second.merged == 0
    assert second.queued == 0

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM registry_entities")
        assert cur.fetchone()[0] == expected_total


# --------------------------------------------------------------------
# ancestors / parent chain
# --------------------------------------------------------------------


def test_ancestors_returns_the_full_chain_nearest_first(loaded_registry):
    assert loaded_registry.ancestors(LARKSPUR_MEMORIAL) == [LARKSPUR, PINECREST]


def test_ancestors_of_a_root_account_is_empty(loaded_registry):
    assert loaded_registry.ancestors(PINECREST) == []


def test_ancestors_of_unknown_id_is_empty(loaded_registry):
    assert loaded_registry.ancestors("no-such-id") == []


# --------------------------------------------------------------------
# resolve()
# --------------------------------------------------------------------


def test_resolve_larkspur_is_an_alias_of_the_system(loaded_registry):
    # "Larkspur" is an explicit alias of the system (truth-table.yaml's
    # own accounts list), not a partial name of either facility, so this
    # is a clean, unambiguous alias-tier hit.
    resolution = loaded_registry.resolve("Larkspur")
    assert resolution.status == "resolved"
    assert resolution.entity_id == LARKSPUR


def test_resolve_memorial_finds_the_facility(loaded_registry):
    resolution = loaded_registry.resolve("Memorial")
    assert resolution.status == "resolved"
    assert resolution.entity_id == LARKSPUR_MEMORIAL


def test_resolve_riverside_finds_the_facility(loaded_registry):
    resolution = loaded_registry.resolve("Riverside")
    assert resolution.status == "resolved"
    assert resolution.entity_id == ASHGROVE_RIVERSIDE


def test_resolve_riverside_clinic_is_unknown(loaded_registry):
    # corpus/truth-table.yaml distractor D01: a different "Riverside
    # Clinic" in an industry newsletter, not our Ashgrove Riverside.
    # "Riverside Clinic" is not a canonical name, not an alias, and does
    # not normalise to anything registered -- never fuzzy-matched down to
    # "Riverside".
    resolution = loaded_registry.resolve("Riverside Clinic")
    assert resolution.status == "unknown"
    assert resolution.entity_id is None


def test_resolve_exact_canonical_name(loaded_registry):
    resolution = loaded_registry.resolve("Ashgrove Medical Center")
    assert resolution.status == "resolved"
    assert resolution.entity_id == ASHGROVE_MEDICAL


def test_resolve_normalised_tier_strips_suffix_and_punctuation(loaded_registry):
    # "St Aldric Health" (no period) should still normalise-match
    # "St. Aldric Health" once the "Health" suffix and punctuation are
    # stripped from both sides.
    resolution = loaded_registry.resolve("St Aldric Health")
    assert resolution.status == "resolved"


def test_resolve_auto_generated_acronym_matches_truth_table_alias(loaded_registry):
    # Validates the acronym rule (2-4 capitalised words): AMC, CRMC and
    # NGH are all listed by hand in truth-table.yaml, and this build
    # derives the identical acronym from the CRM name alone.
    assert loaded_registry.resolve("AMC").entity_id == ASHGROVE_MEDICAL
    assert loaded_registry.resolve("CRMC").entity_id == "A-1020"
    assert loaded_registry.resolve("NGH").entity_id == "A-1031"


def test_short_alias_requires_exact_case(loaded_registry):
    assert loaded_registry.resolve("AMC").status == "resolved"
    assert loaded_registry.resolve("amc").status == "unknown"


def test_resolve_unknown_mention(loaded_registry):
    resolution = loaded_registry.resolve("Some Hospital Nobody Has Heard Of")
    assert resolution.status == "unknown"
    assert resolution.entity_id is None
    assert resolution.candidates == []


def test_resolve_kind_hint_filters_candidates(loaded_registry):
    # "Maria Chen" is both a contact's exact canonical name and, in the
    # broader corpus, a mention of the same real person from a personal
    # address that never enters the registry at all (see
    # test_by_email_personal_address_not_in_registry). With kind_hint
    # supplied, resolve() must land on the CRM contact.
    resolution = loaded_registry.resolve("Maria Chen", kind_hint="contact")
    assert resolution.status == "resolved"
    assert resolution.entity_id == MARIA_CHEN


def test_resolve_context_ids_narrows_ambiguous_candidates(loaded_registry):
    # "Ashgrove" is ambiguous: an explicit alias of the system (A-1010)
    # and also the suffix-stripped name of the medical center (A-1011,
    # "Ashgrove Medical Center" minus "Medical Center"). With a context id
    # naming exactly one of the two candidates, resolve() takes it.
    plain = loaded_registry.resolve("Ashgrove")
    assert plain.status == "ambiguous"
    assert set(plain.candidates) == {ASHGROVE, ASHGROVE_MEDICAL}
    # Bug hunt 2026-09-18, finding 3: A-1011 (Ashgrove Medical Center) is
    # itself a CHILD of A-1010 (Ashgrove Health) -- not a sibling case,
    # so `parents` never agrees on one shared value ({None, "A-1010"}).
    # Before the fix this fell through to `attach_to=None` ("facts are
    # never dropped" notwithstanding); now it recognises A-1010 as an
    # ancestor of every OTHER candidate and attaches there.
    assert plain.attach_to == ASHGROVE

    narrowed = loaded_registry.resolve("Ashgrove", context_ids=[ASHGROVE_MEDICAL])
    assert narrowed.status == "resolved"
    assert narrowed.entity_id == ASHGROVE_MEDICAL


def test_resolve_ambiguous_with_shared_parent(db, loaded_registry):
    # Two facilities under the same system (Larkspur Memorial and
    # Larkspur North, both children of the Larkspur system) given a
    # shared alias by hand: resolve() must attach to their shared parent
    # rather than pick one arbitrarily.
    add_alias(db, LARKSPUR_MEMORIAL, "Larkspur Facility", source="human")
    add_alias(db, "A-1004", "Larkspur Facility", source="human")
    refreshed = Registry.load(db)
    resolution = refreshed.resolve("Larkspur Facility")
    assert resolution.status == "ambiguous"
    assert set(resolution.candidates) == {LARKSPUR_MEMORIAL, "A-1004"}
    assert resolution.attach_to == LARKSPUR


# --------------------------------------------------------------------
# by_email / by_domain
# --------------------------------------------------------------------


def test_by_email_finds_the_crm_contact(loaded_registry):
    entity = loaded_registry.by_email("mchen@ashgrovehealth.org")
    assert entity is not None
    assert entity.id == MARIA_CHEN
    assert entity.canonical_name == "Maria Chen"


def test_by_email_personal_address_not_in_registry(loaded_registry):
    # truth-table.yaml deliberately excludes maria-chen-personal from
    # contacts.csv to exercise the unknown-entity queue elsewhere in the
    # pipeline; the registry itself must not know this address.
    assert loaded_registry.by_email("maria.chen.ohio@gmail.com") is None


def test_by_email_is_case_insensitive(loaded_registry):
    entity = loaded_registry.by_email("MCHEN@AshgroveHealth.org")
    assert entity is not None
    assert entity.id == MARIA_CHEN


def test_by_domain_returns_every_entity_on_that_domain(loaded_registry):
    entities = loaded_registry.by_domain("ashgrovehealth.org")
    ids = {e.id for e in entities}
    assert {ASHGROVE, ASHGROVE_MEDICAL, ASHGROVE_RIVERSIDE}.issubset(ids)


# --------------------------------------------------------------------
# scan()
# --------------------------------------------------------------------


def test_scan_finds_a_product_match_in_s_email_062(loaded_registry):
    text = S_EMAIL_062.read_text()
    matches = loaded_registry.scan(text)
    product_matches = [m for m in matches if m.kind.value == "product"]
    assert product_matches, f"expected a product match, got: {matches}"


def test_scan_case_insensitive_for_long_aliases(loaded_registry):
    matches = loaded_registry.scan("we spoke with the larkspur team yesterday")
    assert any(m.entity_id == LARKSPUR for m in matches)


def test_scan_short_alias_requires_exact_case(loaded_registry):
    matches_upper = loaded_registry.scan("purchasing manages the AMC contract")
    matches_lower = loaded_registry.scan("purchasing manages the amc contract")
    assert any(m.entity_id == ASHGROVE_MEDICAL for m in matches_upper)
    assert not any(m.entity_id == ASHGROVE_MEDICAL for m in matches_lower)


def test_scan_respects_word_boundaries(loaded_registry):
    # "T2" (product alias) must not fire inside "T20" or "AT2".
    matches = loaded_registry.scan("the AT2 bracket and a T20 torx bit")
    assert not any(m.matched_text == "T2" for m in matches)


@pytest.mark.perf
def test_scan_is_fast_on_a_20k_character_text(loaded_registry):
    # Required timing check: scan() over a 20,000-character text, using
    # every pattern the real corpus import produces (accounts, contacts,
    # products, competitors, internal people, and every alias — see the
    # wave report for the exact pattern count and measured time).
    import time

    base_text = (
        S_EMAIL_062.read_text()
        + " "
        + (
            "A routine note about pumps, deliveries and renewals with nothing "
            "in particular planted in it for the scanner to find. "
        )
    )
    text = (base_text * ((20_000 // len(base_text)) + 1))[:20_000]
    assert len(text) == 20_000

    loaded_registry.scan(text)  # warm-up
    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        loaded_registry.scan(text)
    elapsed = (time.perf_counter() - start) / runs

    assert elapsed < 0.2, f"scan() took {elapsed * 1000:.1f} ms average over {runs} runs"


def test_scan_longest_match_wins(loaded_registry):
    # "Larkspur Memorial" (alias) should win over the shorter "Larkspur"
    # (alias of the parent system) when both could match at this position.
    matches = loaded_registry.scan("a delivery went to Larkspur Memorial Hospital yesterday")
    spans_at_larkspur = [m for m in matches if m.matched_text.startswith("Larkspur")]
    assert any(m.entity_id == LARKSPUR_MEMORIAL for m in spans_at_larkspur)
    assert not any(m.entity_id == LARKSPUR for m in spans_at_larkspur)


# --------------------------------------------------------------------
# add_alias / queue_unknown
# --------------------------------------------------------------------


def test_add_alias_then_resolve(db, loaded_registry):
    add_alias(db, LARKSPUR_MEMORIAL, "the Memorial site", source="human")
    refreshed = Registry.load(db)
    resolution = refreshed.resolve("the Memorial site")
    assert resolution.status == "resolved"
    assert resolution.entity_id == LARKSPUR_MEMORIAL


def test_add_alias_is_idempotent(db, loaded_registry):
    add_alias(db, LARKSPUR_MEMORIAL, "the Memorial site", source="human")
    add_alias(db, LARKSPUR_MEMORIAL, "the Memorial site", source="human")
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM registry_aliases WHERE alias = %s", ("the Memorial site",)
        )
        assert cur.fetchone()[0] == 1


def test_add_alias_unknown_entity_raises(db, loaded_registry):
    with pytest.raises(ValueError):
        add_alias(db, "no-such-entity", "whatever", source="human")


def test_queue_unknown_writes_a_review_queue_row(db, loaded_registry):
    with db.cursor() as cur:
        _insert_source(cur, "S-email-058-b")
    queue_unknown(db, "Renata Solis", "S-email-058-b", {"role": "new CFO"})
    with db.cursor() as cur:
        cur.execute(
            "SELECT payload FROM review_queue WHERE kind = 'unknown_entity' AND source_slug = %s",
            ("S-email-058-b",),
        )
        rows = cur.fetchall()
    assert len(rows) == 1
    payload = rows[0][0]
    assert payload["mention"] == "Renata Solis"
    assert payload["context"]["role"] == "new CFO"


def test_queue_unknown_dedups_by_mention_and_source(db, loaded_registry):
    with db.cursor() as cur:
        _insert_source(cur, "S-email-058-c")
    queue_unknown(db, "Renata Solis", "S-email-058-c", {"role": "new CFO"})
    queue_unknown(db, "Renata Solis", "S-email-058-c", {"role": "new CFO"})
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM review_queue WHERE kind = 'unknown_entity' "
            "AND source_slug = %s AND payload ->> 'mention' = %s",
            ("S-email-058-c", "Renata Solis"),
        )
        assert cur.fetchone()[0] == 1


def test_queue_unknown_keeps_a_second_source_on_the_same_row(db, loaded_registry):
    """One open row per name (2026-09-22 review-queue volume fix): the
    second source's sighting is kept on that row, not dropped."""
    with db.cursor() as cur:
        _insert_source(cur, "S-a")
        _insert_source(cur, "S-b")
    queue_unknown(db, "Renata Solis", "S-a", {})
    queue_unknown(db, "Renata Solis", "S-b", {"role": "new CFO"})
    with db.cursor() as cur:
        cur.execute(
            "SELECT source_slug, payload -> 'occurrences' FROM review_queue "
            "WHERE kind = 'unknown_entity' AND payload ->> 'mention' = %s",
            ("Renata Solis",),
        )
        rows = cur.fetchall()
    assert rows == [("S-a", [{"source_slug": "S-b", "context": {"role": "new CFO"}}])]


# --------------------------------------------------------------------
# Dedup rules (hand-made CSVs), portolan-spec-v2.md sec 3 /
# open-questions.md sec F "B2. Dirty CRMs"
# --------------------------------------------------------------------


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_dedup_same_domain_merges_same_different_domain_queues(db, tmp_path):
    accounts_path = tmp_path / "dirty_accounts.csv"
    contacts_path = tmp_path / "dirty_contacts.csv"

    _write_csv(
        accounts_path,
        ["crm_id", "name", "kind", "parent_crm_id", "domain", "arr", "region", "status"],
        [
            {
                "crm_id": "X-1",
                "name": "Ashgrove Health",
                "kind": "system",
                "parent_crm_id": "",
                "domain": "ashgrovehealth.org",
                "arr": "0",
                "region": "Midwest",
                "status": "",
            },
            {
                "crm_id": "X-2",
                "name": "Ashgrove Health",
                "kind": "system",
                "parent_crm_id": "",
                "domain": "ashgrovehealth.org",
                "arr": "0",
                "region": "Midwest",
                "status": "",
            },
            {
                "crm_id": "Y-1",
                "name": "Northlake Health System",
                "kind": "system",
                "parent_crm_id": "",
                "domain": "northlakehs.org",
                "arr": "0",
                "region": "Midwest",
                "status": "",
            },
            {
                "crm_id": "Y-2",
                "name": "Northlake Health System",
                "kind": "system",
                "parent_crm_id": "",
                "domain": "northlakehs-alt.org",
                "arr": "0",
                "region": "Midwest",
                "status": "",
            },
        ],
    )
    _write_csv(contacts_path, ["crm_id", "name", "title", "account_crm_id", "email", "status"], [])

    report = Registry().import_crm(db, accounts_path, contacts_path)

    assert report.created == 3  # X-1, Y-1, Y-2 (X-2 merges into X-1)
    assert report.merged == 1  # X-2 -> X-1
    assert report.queued == 1  # Y-2 vs Y-1, different domain

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'account'")
        assert cur.fetchone()[0] == 3
        cur.execute("SELECT count(*) FROM review_queue WHERE kind = 'ambiguous_entity'")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT payload FROM review_queue WHERE kind = 'ambiguous_entity'")
        payload = cur.fetchone()[0]
        assert set(payload["candidates"]) == {"Y-1", "Y-2"}


def test_dedup_contacts_same_email_merges(db, tmp_path):
    accounts_path = tmp_path / "accounts.csv"
    contacts_path = tmp_path / "contacts.csv"
    _write_csv(
        accounts_path,
        ["crm_id", "name", "kind", "parent_crm_id", "domain", "arr", "region", "status"],
        [
            {
                "crm_id": "X-1",
                "name": "Ashgrove Health",
                "kind": "system",
                "parent_crm_id": "",
                "domain": "ashgrovehealth.org",
                "arr": "0",
                "region": "",
                "status": "",
            }
        ],
    )
    _write_csv(
        contacts_path,
        ["crm_id", "name", "title", "account_crm_id", "email", "status"],
        [
            {
                "crm_id": "C-1",
                "name": "Pat Rivera",
                "title": "Buyer",
                "account_crm_id": "X-1",
                "email": "privera@ashgrovehealth.org",
                "status": "",
            },
            {
                "crm_id": "C-2",
                "name": "Patricia Rivera",
                "title": "Senior Buyer",
                "account_crm_id": "X-1",
                "email": "PRIVERA@ashgrovehealth.org",
                "status": "",
            },
        ],
    )

    report = Registry().import_crm(db, accounts_path, contacts_path)
    assert report.merged == 1

    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM registry_entities WHERE kind = 'contact'")
        assert cur.fetchone()[0] == 1


def test_dirty_crm_row_with_blank_crm_id_gets_a_synthesised_id(db, tmp_path):
    accounts_path = tmp_path / "accounts.csv"
    contacts_path = tmp_path / "contacts.csv"
    _write_csv(
        accounts_path,
        ["crm_id", "name", "kind", "parent_crm_id", "domain", "arr", "region", "status"],
        [
            {
                "crm_id": "",
                "name": "Unlisted Regional Hospital",
                "kind": "facility",
                "parent_crm_id": "",
                "domain": "unlistedregional.org",
                "arr": "0",
                "region": "",
                "status": "",
            }
        ],
    )
    _write_csv(contacts_path, ["crm_id", "name", "title", "account_crm_id", "email", "status"], [])

    report = Registry().import_crm(db, accounts_path, contacts_path)
    assert report.created == 1

    registry = Registry.load(db)
    resolution = registry.resolve("Unlisted Regional Hospital")
    assert resolution.status == "resolved"
    assert resolution.entity_id  # non-empty, synthesised from the name
    assert resolution.entity_id != ""


# --------------------------------------------------------------------
# Bug hunt 2026-09-18, finding 12: `registry_entities.active` (set
# `false` by the nightly CRM sync for a churned account, `registry.
# importer._is_inactive`) was loaded and indexed like any other row, so
# merge kept resolving fresh mentions to a churned account and attaching
# new claims to it. `corpus/out/crm/accounts.csv`'s own Kestrel Medical
# Group (A-1080, `status: churned_2026-02` in `truth-table.yaml`)
# already imports `active=false` -- no synthetic fixture needed.
# --------------------------------------------------------------------

KESTREL = "A-1080"


def test_resolve_excludes_an_inactive_entity(loaded_registry: Registry):
    entity = loaded_registry.get(KESTREL)
    assert entity is not None
    assert entity.active is False

    by_name = loaded_registry.resolve("Kestrel Medical Group")
    assert by_name.status == "unknown"

    by_alias = loaded_registry.resolve("Kestrel")
    assert by_alias.status == "unknown"


def test_get_and_ancestors_still_find_an_inactive_entity(loaded_registry: Registry):
    """Only fresh mention resolution is refused -- an already-known id
    (an existing note, a CRM re-import processing its own row, an owner
    resolved earlier in the same call) must keep working, so a churned
    account's own existing note stays reachable and the shipped corpus's
    own churn fixture (Kestrel, used elsewhere as a "gone quiet" scenario)
    keeps getting a note at all."""
    entity = loaded_registry.get(KESTREL)
    assert entity is not None
    assert entity.canonical_name == "Kestrel Medical Group"
    # Kestrel is a top-level account (no parent) in this corpus; the real
    # assertion is that `get()` didn't silently return None.
    assert loaded_registry.ancestors(KESTREL) == []
