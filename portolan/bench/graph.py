"""Score the graph database against the truth table's `expectations.merge`
and `expectations.filter` blocks (`corpus/truth-table.yaml`).

Every check is raw SQL against the tables `01-spec/portolan-spec-v2.md` and
`portolan/db/migrations/0001_init.sql` define (`claims`, `notes`, `links`,
`review_queue`, `sources`, `coverage_ledger`, `registry_entities`) — this
module does not import `portolan.registry`, `portolan.merge`, or
`portolan.ingest` (other waves' modules; see build-contracts.md's "own
only your listed paths" rule), only `portolan.db.connect`'s `psycopg.
Connection` shape.

Each expectation becomes one `ExpectationResult`: pass/fail plus the
evidence (row ids, counts, payload snippets) that produced the verdict, so
a failure is diagnosable from the report alone.
"""

from __future__ import annotations

from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, Field

from portolan.bench.locators import locator_matches
from portolan.bench.truth import Truth

Row = dict[str, Any]


class ExpectationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str  # "merge" | "filter"
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class GraphScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ExpectationResult] = Field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0

    def failures(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed]


def _rows(conn: psycopg.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def _guarded(category: str, name: str, fn) -> ExpectationResult:
    """Run one check, turning a DB error (missing table, e.g. against an
    unmigrated database) into a clean failure instead of an exception that
    would abort every other expectation."""
    try:
        return fn()
    except psycopg.Error as exc:
        return ExpectationResult(
            category=category, name=name, passed=False, detail=f"query failed: {exc}"
        )


# --------------------------------------------------------------------------
# expectations.merge
# --------------------------------------------------------------------------


def _matching_claims_for_fact(conn: psycopg.Connection, truth: Truth, fact_id: str) -> list[Row]:
    """Every `claims` row that corresponds to one of `fact_id`'s manifest
    placements: same fact_type, same source_slug, and a locator inside
    that placement's window."""
    fact_def = truth.fact(fact_id)
    matches: list[Row] = []
    for placement in [p for p in truth.all_planted() if p.fact_id == fact_id]:
        candidates = _rows(
            conn,
            "SELECT id, note_slug, source_slug, locator, claim_text, superseded_by, superseded_at "
            "FROM claims WHERE fact_type = %s AND source_slug = %s",
            (fact_def.type, placement.source_id),
        )
        for row in candidates:
            if any(
                locator_matches(placement.source_type, loc, row["locator"])
                for loc in placement.locators
            ):
                matches.append(row)
    return matches


def _check_cross_source_single_claim(
    conn: psycopg.Connection, truth: Truth, fact_id: str, expected_citations: int
) -> ExpectationResult:
    def run() -> ExpectationResult:
        matches = _matching_claims_for_fact(conn, truth, fact_id)
        note_slugs = {m["note_slug"] for m in matches}
        passed = len(matches) == expected_citations and len(note_slugs) == 1
        return ExpectationResult(
            category="merge",
            name=f"cross_source_single_claim:{fact_id}",
            passed=passed,
            detail=(
                f"expected {expected_citations} citation(s) on one note; found "
                f"{len(matches)} claim row(s) across {len(note_slugs)} note(s)"
            ),
            evidence={
                "claim_ids": [m["id"] for m in matches],
                "note_slugs": sorted(note_slugs),
                "expected_citations": expected_citations,
            },
        )

    return _guarded("merge", f"cross_source_single_claim:{fact_id}", run)


def _check_supersession(
    conn: psycopg.Connection, truth: Truth, old_fact_id: str, new_fact_id: str
) -> ExpectationResult:
    def run() -> ExpectationResult:
        old_matches = _matching_claims_for_fact(conn, truth, old_fact_id)
        new_matches = _matching_claims_for_fact(conn, truth, new_fact_id)
        new_ids = {m["id"] for m in new_matches}
        passed = (
            bool(old_matches)
            and bool(new_matches)
            and any(
                m["superseded_at"] is not None and m["superseded_by"] in new_ids
                for m in old_matches
            )
        )
        return ExpectationResult(
            category="merge",
            name=f"supersession:{old_fact_id}->{new_fact_id}",
            passed=passed,
            detail=(
                f"{old_fact_id} claim(s) {[m['id'] for m in old_matches]} should be "
                f"superseded by one of {new_fact_id}'s claim(s) {sorted(new_ids)}"
            ),
            evidence={
                "old_claim_ids": [m["id"] for m in old_matches],
                "old_superseded_by": [m["superseded_by"] for m in old_matches],
                "new_claim_ids": sorted(new_ids),
            },
        )

    return _guarded("merge", f"supersession:{old_fact_id}->{new_fact_id}", run)


def _search_terms_for(truth: Truth, item: str) -> list[str]:
    """`unknown_entity_queue_contains` / `ambiguous_resolution` entries are
    either a truth-table entity id (expand to its name/aliases) or a bare
    literal string (e.g. a brand-new hire's name that has no id yet, like
    `Renata Solis`) — used as-is."""
    variants = truth.entity_names(item)
    return list(variants) if variants else [item]


def _payload_contains(payload: Any, terms: list[str]) -> bool:
    text = str(payload).lower()
    return any(term.lower() in text for term in terms)


def _check_unknown_entity_queue_contains(
    conn: psycopg.Connection, truth: Truth, item: str
) -> ExpectationResult:
    def run() -> ExpectationResult:
        rows = _rows(
            conn,
            "SELECT id, payload, source_slug FROM review_queue WHERE kind = 'unknown_entity'",
        )
        terms = _search_terms_for(truth, item)
        hits = [r for r in rows if _payload_contains(r["payload"], terms)]
        return ExpectationResult(
            category="merge",
            name=f"unknown_entity_queue_contains:{item}",
            passed=bool(hits),
            detail=f"searched {len(rows)} unknown_entity row(s) for {terms!r}",
            evidence={"matching_ids": [r["id"] for r in hits]},
        )

    return _guarded("merge", f"unknown_entity_queue_contains:{item}", run)


def _check_ambiguous_resolution(
    conn: psycopg.Connection, truth: Truth, fact_id: str, expected_attach: str
) -> ExpectationResult:
    def run() -> ExpectationResult:
        placements = [p for p in truth.all_planted() if p.fact_id == fact_id]
        source_ids = [p.source_id for p in placements]
        rows = _rows(
            conn,
            "SELECT id, payload, source_slug FROM review_queue "
            "WHERE kind = 'ambiguous_entity' AND source_slug = ANY(%s)",
            (source_ids,),
        )
        terms = _search_terms_for(truth, expected_attach)
        hits = [r for r in rows if _payload_contains(r["payload"], terms)]
        return ExpectationResult(
            category="merge",
            name=f"ambiguous_resolution:{fact_id}",
            passed=bool(hits),
            detail=(
                f"expected an ambiguous_entity row on source(s) {source_ids} "
                f"attaching to {expected_attach!r}"
            ),
            evidence={"matching_ids": [r["id"] for r in hits], "sources_checked": source_ids},
        )

    return _guarded("merge", f"ambiguous_resolution:{fact_id}", run)


def _check_parent_chain(
    conn: psycopg.Connection, truth: Truth, chain: list[str]
) -> ExpectationResult:
    def run() -> ExpectationResult:
        crm_ids = []
        for account_id in chain:
            account = next((a for a in truth.table.accounts if a.id == account_id), None)
            crm_ids.append(account.crm_id if account else None)

        entities = {}
        for account_id, crm_id in zip(chain, crm_ids, strict=True):
            if crm_id is None:
                continue
            rows = _rows(
                conn,
                "SELECT id, parent_id, crm_id FROM registry_entities WHERE crm_id = %s",
                (crm_id,),
            )
            if rows:
                entities[account_id] = rows[0]

        ok = len(entities) == len(chain)
        broken_links: list[str] = []
        for child_id, parent_id in zip(chain[1:], chain[:-1], strict=True):
            child = entities.get(child_id)
            parent = entities.get(parent_id)
            if not child or not parent or child["parent_id"] != parent["id"]:
                ok = False
                broken_links.append(f"{child_id}->{parent_id}")

        return ExpectationResult(
            category="merge",
            name=f"parent_chain:{'->'.join(chain)}",
            passed=ok,
            detail=(
                "every account in the chain resolves to a registry_entities row, "
                "each pointing at the next via parent_id"
                if ok
                else f"missing or broken links: {broken_links or 'some accounts not found'}"
            ),
            evidence={
                "found": {k: v["id"] for k, v in entities.items()},
                "broken_links": broken_links,
            },
        )

    return _guarded("merge", f"parent_chain:{'->'.join(chain)}", run)


# --------------------------------------------------------------------------
# expectations.filter
# --------------------------------------------------------------------------


def _source_row(conn: psycopg.Connection, source_id: str) -> Row | None:
    rows = _rows(
        conn,
        "SELECT slug, extraction_status, filter_reason, source_date, source_type "
        "FROM sources WHERE slug = %s",
        (source_id,),
    )
    return rows[0] if rows else None


def _check_excluded_by_customer_filter(
    conn: psycopg.Connection, source_id: str
) -> ExpectationResult:
    def run() -> ExpectationResult:
        row = _source_row(conn, source_id)
        passed = bool(row) and row["extraction_status"] == "filtered"
        return ExpectationResult(
            category="filter",
            name=f"excluded_by_customer_filter:{source_id}",
            passed=passed,
            detail=f"sources.{source_id}.extraction_status == 'filtered'",
            evidence={"row": row},
        )

    return _guarded("filter", f"excluded_by_customer_filter:{source_id}", run)


def _check_included_by_registry_scan(conn: psycopg.Connection, source_id: str) -> ExpectationResult:
    def run() -> ExpectationResult:
        row = _source_row(conn, source_id)
        passed = bool(row) and row["extraction_status"] != "filtered"
        return ExpectationResult(
            category="filter",
            name=f"included_by_registry_scan:{source_id}",
            passed=passed,
            detail=f"sources.{source_id}.extraction_status != 'filtered'",
            evidence={"row": row},
        )

    return _guarded("filter", f"included_by_registry_scan:{source_id}", run)


def _check_junk_filtered(conn: psycopg.Connection, source_id: str) -> ExpectationResult:
    def run() -> ExpectationResult:
        row = _source_row(conn, source_id)
        reason = (row or {}).get("filter_reason") or ""
        passed = bool(row) and row["extraction_status"] == "filtered" and bool(reason.strip())
        return ExpectationResult(
            category="filter",
            name=f"junk_filtered:{source_id}",
            passed=passed,
            detail=f"sources.{source_id} filtered with a reason recorded",
            evidence={"row": row},
        )

    return _guarded("filter", f"junk_filtered:{source_id}", run)


def _check_sample_audit_should_find(
    conn: psycopg.Connection, truth: Truth, fact_id: str
) -> ExpectationResult:
    def run() -> ExpectationResult:
        placements = [p for p in truth.all_planted() if p.fact_id == fact_id]
        if not placements:
            return ExpectationResult(
                category="filter",
                name=f"sample_audit_should_find:{fact_id}",
                passed=False,
                detail=f"{fact_id} has no manifest placement",
            )
        placement = placements[0]
        source = truth.manifest.source(placement.source_id)
        rows = _rows(
            conn,
            "SELECT connector, day, sampled FROM coverage_ledger WHERE connector = %s AND day = %s",
            (source.type, source.date),
        )
        passed = bool(rows) and rows[0]["sampled"] >= 1
        return ExpectationResult(
            category="filter",
            name=f"sample_audit_should_find:{fact_id}",
            passed=passed,
            detail=(
                f"coverage_ledger[{source.type}, {source.date}].sampled >= 1 "
                f"(its excluded source {placement.source_id} should turn up in the sample)"
            ),
            evidence={"rows": rows},
        )

    return _guarded("filter", f"sample_audit_should_find:{fact_id}", run)


def _check_quoted_text_stripped(conn: psycopg.Connection, source_id: str) -> ExpectationResult:
    def run() -> ExpectationResult:
        rows = _rows(conn, "SELECT id FROM claims WHERE source_slug = %s", (source_id,))
        passed = len(rows) == 0
        return ExpectationResult(
            category="filter",
            name=f"quoted_text_stripped:{source_id}",
            passed=passed,
            detail=(
                f"a quoted-reply thread should yield zero of its own claims; found {len(rows)}"
            ),
            evidence={"claim_ids": [r["id"] for r in rows]},
        )

    return _guarded("filter", f"quoted_text_stripped:{source_id}", run)


# --------------------------------------------------------------------------
# score_graph
# --------------------------------------------------------------------------


def score_graph(conn: psycopg.Connection, truth: Truth) -> GraphScore:
    """Evaluate `truth.table.expectations['merge']` and `[...]['filter']`
    against the database on `conn`. Every expectation produces one
    `ExpectationResult`, pass or fail, with evidence — nothing raises for a
    single failed or errored check; `GraphScore.failures()` lists them."""
    expectations = truth.table.expectations
    merge = expectations.get("merge", {})
    filter_ = expectations.get("filter", {})

    results: list[ExpectationResult] = []

    for fact_id, expected_n in (merge.get("cross_source_single_claim") or {}).items():
        results.append(_check_cross_source_single_claim(conn, truth, fact_id, expected_n))

    for pair in merge.get("supersessions") or []:
        results.append(_check_supersession(conn, truth, pair["old"], pair["new"]))

    for item in merge.get("unknown_entity_queue_contains") or []:
        results.append(_check_unknown_entity_queue_contains(conn, truth, item))

    for fact_id, expected_attach in (merge.get("ambiguous_resolution") or {}).items():
        results.append(_check_ambiguous_resolution(conn, truth, fact_id, expected_attach))

    # `parent_chains` is a list of independent hierarchies (each its own
    # ordered child->parent list); `parent_chain` (singular) is kept as a
    # fallback for a truth table that only has one. v1.1 needs two
    # (Pinecrest/Larkspur/Larkspur Memorial, and the unrelated Wrenfield
    # Health/Wrenfield Medical Center root) -- see the truth table's own
    # comment on why these were split rather than flattened into one list.
    parent_chains = merge.get("parent_chains")
    if parent_chains is None:
        single = merge.get("parent_chain")
        parent_chains = [single] if single else []
    for chain in parent_chains:
        results.append(_check_parent_chain(conn, truth, list(chain)))

    for source_id in filter_.get("excluded_by_customer_filter") or []:
        results.append(_check_excluded_by_customer_filter(conn, source_id))

    for source_id in filter_.get("included_by_registry_scan") or []:
        results.append(_check_included_by_registry_scan(conn, source_id))

    for fact_id in filter_.get("sample_audit_should_find") or []:
        results.append(_check_sample_audit_should_find(conn, truth, fact_id))

    for source_id in filter_.get("junk_filtered") or []:
        results.append(_check_junk_filtered(conn, source_id))

    quoted_source = filter_.get("quoted_text_stripped")
    if quoted_source:
        results.append(_check_quoted_text_stripped(conn, quoted_source))

    return GraphScore(results=results)
