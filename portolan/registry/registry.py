"""The registry: `Registry.load`, `.resolve`, `.ancestors`, `.by_email`,
`.by_domain`, `.scan`, plus `add_alias` and `queue_unknown`
(`portolan-spec-v2.md` sec 3; `open-questions.md` sec F "B2. Dirty CRMs").

**Id scheme: `Entity.id` is `registry_entities.crm_id`**, not the table's
bigint primary key. For an account or contact that is the real CRM id
(`"A-1002"`, `"C-2001"`). `crm_id` is a plain nullable text column, so it
also holds:

- a product/competitor/internal-person's id from the meta YAML
  (`"tidewater-t2"`, `"corvane"`, `"sam-okafor"` in
  `corpus/truth-table.yaml`, used verbatim), and
- a synthesised `text.slugify(name)` for a dirty-CRM row that arrives
  with no id at all.

So every entity has one stable external string id (`import_crm` never
leaves `crm_id` null), and `ancestors`/`resolve`/`by_email`/`by_domain`/
`add_alias` all speak in it: Larkspur Memorial Hospital is `"A-1003"`,
not `"larkspur-memorial"` (`tests/test_registry.py`). A row with no
`crm_id` (made outside `import_crm`) loads as `row-<id>`.

**Resolution tiers** (`resolve()`): exact canonical name, then alias, then
normalised (suffix- and punctuation-stripped) — the first tier with any
candidate left after the active/kind filters wins. Below normalised,
nothing: never fuzzy-match, never guess.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from portolan.registry.importer import run_import_crm
from portolan.registry.models import Entity, EntityKind, ImportReport, Match, Resolution
from portolan.registry.scanner import (
    SHORT_PATTERN_MAX_LEN,
    CompiledScanner,
    PatternEntry,
    build_scanner,
    scan_text,
)
from portolan.registry.text import normalize_name

__all__ = ["Registry", "add_alias", "park_mention", "queue_unknown"]


class Registry:
    """In-memory index over `registry_entities`/`registry_aliases`, built
    by `Registry.load(conn)`. `import_crm` writes to the database directly
    (it does not require `.load()` first, and does not refresh `self`
    afterward — call `.load(conn)` again to see the new state)."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._canonical_ci: dict[str, list[str]] = {}
        self._aliases_ci: dict[str, list[str]] = {}
        self._aliases_cs: dict[str, list[str]] = {}
        self._normalized_index: dict[str, list[str]] = {}
        self._email_index: dict[str, str] = {}
        self._domain_index: dict[str, list[str]] = {}
        self._scanner: CompiledScanner = build_scanner([])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, conn) -> Registry:
        """Load every entity, inactive ones included.

        `active` (false for a churned account or departed contact,
        `importer._is_inactive`) is loaded onto each `Entity`, and only
        `resolve()` -- the free-text-mention path -- filters on it.
        `get()`/`by_email()`/`by_domain()`/`ancestors()`/`scan()` still see
        inactive entities, so anything that already knows an inactive
        entity's id (its own note, e.g. Kestrel Medical Group, `A-1080`)
        keeps working.
        """
        registry = cls()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, kind, canonical_name, crm_id, parent_id, account_kind,
                       email, email_domain, active, inactive_since
                FROM registry_entities
                """
            )
            rows = cur.fetchall()
            cur.execute("SELECT alias, entity_id FROM registry_aliases")
            alias_rows = cur.fetchall()

        dbid_to_extid: dict[int, str] = {}
        raw: dict[int, tuple] = {}
        for row in rows:
            dbid, parent_dbid = row[0], row[4]
            # Only a row created outside `import_crm` (e.g. by hand in a test) has no crm_id.
            extid = row[3] or f"row-{dbid}"
            dbid_to_extid[dbid] = extid
            raw[dbid] = row

        aliases_by_dbid: dict[int, list[str]] = {}
        for alias, entity_dbid in alias_rows:
            aliases_by_dbid.setdefault(entity_dbid, []).append(alias)

        # Row shape, matching the SELECT above:
        # (id, kind, canonical_name, crm_id, parent_id, account_kind, email,
        #  email_domain, active, inactive_since)
        entities: dict[str, Entity] = {}
        for dbid, row in raw.items():
            parent_dbid = row[4]
            extid = dbid_to_extid[dbid]
            parent_extid = dbid_to_extid.get(parent_dbid) if parent_dbid else None
            entities[extid] = Entity(
                id=extid,
                kind=EntityKind(row[1]),
                canonical_name=row[2],
                crm_id=row[3],
                parent_id=parent_extid,
                account_kind=row[5],
                email=row[6],
                email_domain=row[7],
                aliases=sorted(aliases_by_dbid.get(dbid, [])),
                active=row[8],
                inactive_since=row[9],
            )

        registry._entities = entities
        registry._build_indexes()
        return registry

    def _build_indexes(self) -> None:
        self._canonical_ci = {}
        self._aliases_ci = {}
        self._aliases_cs = {}
        self._normalized_index = {}
        self._email_index = {}
        self._domain_index = {}
        pattern_entries: list[PatternEntry] = []

        for extid, entity in self._entities.items():
            self._canonical_ci.setdefault(entity.canonical_name.lower(), []).append(extid)
            norm_key = normalize_name(entity.canonical_name)
            self._normalized_index.setdefault(norm_key, []).append(extid)
            if entity.email:
                self._email_index[entity.email.lower()] = extid
            if entity.email_domain:
                self._domain_index.setdefault(entity.email_domain.lower(), []).append(extid)

            pattern_entries.append(
                PatternEntry(entity.canonical_name, extid, entity.kind.value, entity.canonical_name)
            )
            for alias in entity.aliases:
                if len(alias) <= SHORT_PATTERN_MAX_LEN:
                    self._aliases_cs.setdefault(alias, []).append(extid)
                else:
                    self._aliases_ci.setdefault(alias.lower(), []).append(extid)
                pattern_entries.append(
                    PatternEntry(alias, extid, entity.kind.value, entity.canonical_name)
                )

        self._scanner = build_scanner(pattern_entries)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_crm(
        self,
        conn,
        accounts_csv: Path | str,
        contacts_csv: Path | str,
        *,
        products: list[dict[str, Any]] | None = None,
        competitors: list[dict[str, Any]] | None = None,
        internal_people: list[dict[str, Any]] | None = None,
        account_aliases: dict[str, list[str]] | None = None,
        supplier: dict[str, Any] | None = None,
        branches: list[dict[str, Any]] | None = None,
    ) -> ImportReport:
        """Upsert `registry_entities`/`registry_aliases` from a CRM export.

        See `importer.py` for the dedup rules, `account_aliases`, and the
        vendor's own `supplier`/`branches` entities.
        """
        return run_import_crm(
            conn,
            accounts_csv,
            contacts_csv,
            products=products,
            competitors=competitors,
            internal_people=internal_people,
            account_aliases=account_aliases,
            supplier=supplier,
            branches=branches,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def ancestors(self, entity_id: str) -> list[str]:
        """Parent chain, nearest first. `[]` for an unknown id or a root."""
        result: list[str] = []
        seen = {entity_id}
        entity = self._entities.get(entity_id)
        parent_id = entity.parent_id if entity else None
        while parent_id and parent_id not in seen:
            result.append(parent_id)
            seen.add(parent_id)
            parent = self._entities.get(parent_id)
            parent_id = parent.parent_id if parent else None
        return result

    def by_email(self, email: str) -> Entity | None:
        extid = self._email_index.get(email.strip().lower())
        return self._entities.get(extid) if extid else None

    def by_domain(self, domain: str) -> list[Entity]:
        extids = self._domain_index.get(domain.strip().lower(), [])
        return [self._entities[extid] for extid in extids]

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def scan(self, text: str) -> list[Match]:
        matches: list[Match] = []
        for start, end, matched_text, entries in scan_text(self._scanner, text):
            for entry in entries:
                matches.append(
                    Match(
                        entity_id=entry.entity_id,
                        kind=EntityKind(entry.kind),
                        canonical_name=entry.canonical_name,
                        matched_text=matched_text,
                        start=start,
                        end=end,
                    )
                )
        return matches

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(
        self,
        mention: str,
        *,
        kind_hint: EntityKind | str | None = None,
        context_ids: Sequence[str] = (),
        as_of: date | None = None,
    ) -> Resolution:
        """Resolve a free-text mention to one entity, an ambiguity, or `unknown`.

        An inactive candidate (`active = false`: churned account, departed
        contact) is filtered out of every tier, so a mention matching only
        inactive entities resolves `unknown` rather than attaching a new
        claim to an entity the CRM says is gone (see `load` for why this is
        the only method that filters).

        `as_of`: the date the fact was said. An inactive entity whose
        `inactive_since` is on or after it still resolves -- a call from
        March that names a contact who left in August is not a mention of
        someone gone. `None` keeps the plain active-only rule.
        """
        mention = mention.strip()
        if not mention:
            return Resolution(status="unknown")

        kind_val = EntityKind(kind_hint) if kind_hint is not None else None

        def _was_active(entity: Entity) -> bool:
            if entity.active:
                return True
            since = entity.inactive_since
            return as_of is not None and since is not None and as_of <= since

        def _filter_active_kind(ids: list[str]) -> list[str]:
            filtered = (i for i in ids if _was_active(self._entities[i]))
            if kind_val is not None:
                filtered = (i for i in filtered if self._entities[i].kind == kind_val)
            return list(filtered)

        # Tier 1: exact canonical name, case-insensitive.
        candidates = _filter_active_kind(self._canonical_ci.get(mention.lower(), []))

        # Tier 2: alias. Short aliases (<=SHORT_PATTERN_MAX_LEN chars)
        # require exact case; everything else is case-insensitive.
        if not candidates:
            if len(mention) <= SHORT_PATTERN_MAX_LEN:
                candidates = _filter_active_kind(self._aliases_cs.get(mention, []))
            else:
                candidates = _filter_active_kind(self._aliases_ci.get(mention.lower(), []))

        # Tier 3: normalised (suffix- and punctuation-stripped).
        if not candidates:
            candidates = _filter_active_kind(
                self._normalized_index.get(normalize_name(mention), [])
            )

        candidates = list(dict.fromkeys(candidates))  # de-dup, keep order

        if not candidates:
            return Resolution(status="unknown")

        if len(candidates) == 1:
            return Resolution(status="resolved", entity_id=candidates[0], candidates=candidates)

        narrowed = [c for c in candidates if c in context_ids]
        if len(narrowed) == 1:
            return Resolution(status="resolved", entity_id=narrowed[0], candidates=candidates)

        parents = {self._entities[c].parent_id for c in candidates}
        if len(parents) == 1:
            shared_parent = next(iter(parents))
            if shared_parent is not None:
                return Resolution(
                    status="ambiguous", candidates=candidates, attach_to=shared_parent
                )

        # One candidate may be the ancestor of all the others ("Ashgrove" names both
        # Ashgrove Health, A-1010, and its child A-1011): that candidate is the shared
        # parent spec sec 3 attaches to. Import refuses parent loops (`importer._would_loop`),
        # so at most one candidate qualifies and the result does not depend on the order.
        for candidate_id in candidates:
            others = [c for c in candidates if c != candidate_id]
            if others and all(candidate_id in self.ancestors(c) for c in others):
                return Resolution(status="ambiguous", candidates=candidates, attach_to=candidate_id)

        return Resolution(status="ambiguous", candidates=candidates, attach_to=None)


# ----------------------------------------------------------------------
# Module-level writers: standalone functions taking `conn`, for callers
# that don't need a loaded Registry.
# ----------------------------------------------------------------------


def add_alias(conn, entity_id: str, alias: str, source: str) -> None:
    """Add one alias to an existing entity, looked up by its external id
    (`registry_entities.crm_id` — see this module's docstring). A human
    maps a name once and it sticks (spec sec 3): idempotent via `ON
    CONFLICT DO NOTHING` on `(alias, entity_id)`."""
    alias = alias.strip()
    if not alias:
        return
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM registry_entities WHERE crm_id = %s", (entity_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"no registry entity with id {entity_id!r}")
        cur.execute(
            "INSERT INTO registry_aliases (alias, entity_id, source) VALUES (%s, %s, %s) "
            "ON CONFLICT (alias, entity_id) DO NOTHING",
            (alias, row[0], source),
        )
    conn.commit()


#: Distinct sources after which a name that blocks nothing is still worth
#: a person's answer (`park_mention`). Read from the environment per call.
PROMOTE_AFTER_SOURCES_ENV = "PORTOLAN_MENTION_PROMOTE_AFTER"
DEFAULT_PROMOTE_AFTER_SOURCES = 3


def _promote_after_sources() -> int:
    try:
        return max(1, int(os.environ.get(PROMOTE_AFTER_SOURCES_ENV, "")))
    except ValueError:
        return DEFAULT_PROMOTE_AFTER_SOURCES


def park_mention(
    conn,
    *,
    kind: str,
    mention: str,
    source_slug: str,
    payload: dict,
    occurrence: dict,
    fact: dict | None = None,
    blocking: bool = True,
) -> None:
    """One live `review_queue` row per `(kind, mention)`, compared trimmed
    and case-insensitively: a person answers "who is this name" once, and
    one alias covers every source. The first source writes `payload` as a
    new row; each later source is appended to that row's
    `payload.occurrences` as `{"source_slug": ..., **occurrence}`.

    `fact`: the whole fact that named the mention. Every such fact is kept
    on the row under `payload.facts` as `{"source_slug", "blocking",
    "fact"}`, once per `(source, locator, claim)`, so
    `portolan.merge.replay.replay_parked_facts` can merge it after a person
    maps the name. Without it a source's second fact under one unknown name
    had no record anywhere.

    `blocking`: the fact could not be filed without this name. A row is
    `open` (a person is asked) when the name blocks a fact, or once it has
    appeared in `PORTOLAN_MENTION_PROMOTE_AFTER` distinct sources (default
    3); until then it is `watching`, which changes nothing in the graph.

    A source already parked under this name on a row a person closed
    (resolved or dismissed) is a no-op. Never commits."""
    key = mention.strip().lower()
    occurrence_json = json.dumps([{"source_slug": source_slug}])
    fact_entry = (
        {"source_slug": source_slug, "blocking": blocking, "fact": fact}
        if fact is not None
        else None
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, status, source_slug, payload FROM review_queue
            WHERE kind = %s AND status IN ('open', 'watching')
              AND lower(btrim(payload ->> 'mention')) = %s
            ORDER BY id LIMIT 1
            FOR UPDATE
            """,
            (kind, key),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                SELECT 1 FROM review_queue
                WHERE kind = %s AND lower(btrim(payload ->> 'mention')) = %s
                  AND (source_slug = %s OR payload -> 'occurrences' @> %s::jsonb)
                LIMIT 1
                """,
                (kind, key, source_slug, occurrence_json),
            )
            if cur.fetchone() is not None:
                return
            new_payload = dict(payload)
            if fact_entry is not None:
                new_payload["facts"] = [fact_entry]
            status = "open" if blocking or _promote_after_sources() <= 1 else "watching"
            cur.execute(
                "INSERT INTO review_queue (kind, payload, source_slug, status) "
                "VALUES (%s, %s::jsonb, %s, %s)",
                (kind, json.dumps(new_payload), source_slug, status),
            )
            return

        row_id, status, row_source, row_payload = row
        row_payload = dict(row_payload or {})
        occurrences = list(row_payload.get("occurrences") or [])
        seen_sources = {row_source} | {o.get("source_slug") for o in occurrences}
        changed = False
        if source_slug not in seen_sources:
            occurrences.append({"source_slug": source_slug, **occurrence})
            row_payload["occurrences"] = occurrences
            seen_sources.add(source_slug)
            changed = True
        if fact_entry is not None:
            facts = list(row_payload.get("facts") or [])
            identity = (source_slug, fact.get("locator"), fact.get("claim"))
            known = {
                (
                    f.get("source_slug"),
                    (f.get("fact") or {}).get("locator"),
                    (f.get("fact") or {}).get("claim"),
                ): index
                for index, f in enumerate(facts)
            }
            if identity not in known:
                facts.append(fact_entry)
                row_payload["facts"] = facts
                changed = True
            elif blocking and not facts[known[identity]].get("blocking"):
                facts[known[identity]]["blocking"] = True
                row_payload["facts"] = facts
                changed = True
        new_status = status
        if status == "watching" and (
            blocking or len(seen_sources - {None}) >= _promote_after_sources()
        ):
            new_status = "open"
        if changed or new_status != status:
            cur.execute(
                "UPDATE review_queue SET payload = %s::jsonb, status = %s WHERE id = %s",
                (json.dumps(row_payload), new_status, row_id),
            )


def queue_unknown(
    conn,
    mention: str,
    source_slug: str,
    context: dict,
    *,
    commit: bool = True,
    fact: dict | None = None,
    blocking: bool = True,
) -> None:
    """Park an unresolved mention in `review_queue` (kind `unknown_entity`).
    Facts are never dropped, only flagged (spec sec 3). One live row per
    name (`park_mention`, which also explains `fact` and `blocking`): a
    second source naming it is added to that row; a second call for the
    same source and fact is a no-op.

    Pass `commit=False` from inside a `with conn.transaction():` block,
    where psycopg refuses a manual `conn.commit()`."""
    park_mention(
        conn,
        kind="unknown_entity",
        mention=mention,
        source_slug=source_slug,
        payload={"mention": mention, "context": context},
        occurrence={"context": context},
        fact=fact,
        blocking=blocking,
    )
    if commit:
        conn.commit()
