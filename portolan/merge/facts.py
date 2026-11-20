"""`merge_fact` (`01-spec/build-contracts.md` "Merge").

The pipeline, per fact:

1. Resolve entities (`portolan.merge.resolve.resolve_fact_entities`).
2. Pick a home note (`portolan.merge.resolve.select_home`); when every
   name resolved to the vendor's own side, the source's account
   or the fact named nobody, the source's account (`fallback_account_id`);
   no home -> `queued_unknown` (the whole fact is parked on each unknown
   mention's row, or via `_queue_no_home`).
3. Ensure the home note exists (an entity note from the registry, wrapped
   in a `signal-<fact_type>-<anchor>` note unless the home is a contact),
   and every other resolved entity's note too, so every mention can be
   wikilinked.
4. The candidate gate (`portolan.merge.gate.gate`) against the home note's
   live claims; below threshold -> `new`, no LLM call; otherwise one
   compare call.
5. Apply the verdict. `different` (the gate paired two unrelated claims) is
   filed as `new`. `new`/`extends` insert a claim. `same` never writes a
   second `claims` row: the sighting becomes a `claim_citations` row on the
   existing claim, whose own `source_slug`/`locator` stay the first
   sighting's (`01-spec/provenance-chain.md`). `contradicts` inserts and
   supersedes the old claim only when both carry an explicit world date and
   the new one is strictly later; otherwise it is downgraded to `extends`
   (both stay live) with a `review_queue` row, unless the pair is undated
   and from one source (`_undated_same_source`).

Claims are not rendered into the home note here; `merge_extraction`
re-renders each touched note once per source ("one version per source,
not per claim"), which is also what bumps `updated` on a `same` verdict.
"""

from __future__ import annotations

import json
from datetime import datetime

import psycopg

from portolan.extract.schema import EntityKind, Fact
from portolan.ingest.models import SourceRow
from portolan.llm.base import LLMClient
from portolan.merge.gate import COMPARE_SYSTEM, GATE_SKIP_THRESHOLD, build_compare_prompt, gate
from portolan.merge.models import CompareVerdict, MergeOutcome, compare_verdict_json_schema
from portolan.merge.resolve import (
    ResolvedMention,
    park_unresolved,
    resolve_fact_entities,
    select_home,
)
from portolan.merge.util import fact_payload, recorded_dt
from portolan.notes import (
    Claim,
    NoteRow,
    NoteStatus,
    NoteType,
    ensure_entity_note,
    link_mentions,
    link_related,
    note_exists,
    read_note,
    signal_note_slug,
    write_note,
)
from portolan.registry import Registry
from portolan.search.embed import Embedder
from portolan.util import clock

_NOTE_BEARING_KINDS = {
    EntityKind.ACCOUNT,
    EntityKind.CONTACT,
    EntityKind.PRODUCT,
    EntityKind.COMPETITOR,
}

# `HomeChoice.kind` values homed on a signal note rather than the entity's
# own note (`_ensure_signal_note`). A contact home is left unwrapped: it
# already gets an inbound edge from its account (`ensure_entity_note`).
_SIGNAL_WRAPPED_HOME_KINDS = ("signal", "entity", "account")


def _cached_note_slug(conn: psycopg.Connection, entity_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT note_slug FROM registry_entities WHERE crm_id = %s", (entity_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _ensure_and_track(
    conn: psycopg.Connection,
    registry: Registry,
    entity_id: str,
    created: list[str],
    *,
    commit: bool = True,
) -> NoteRow:
    existed = _cached_note_slug(conn, entity_id) is not None
    note = ensure_entity_note(conn, registry, entity_id, commit=commit)
    if not existed:
        created.append(note.slug)
    return note


def _ensure_signal_note(
    conn: psycopg.Connection,
    anchor_note: NoteRow,
    fact_type: str,
    *,
    created: list[str],
    touched: list[str],
    commit: bool = True,
) -> NoteRow:
    """A `signal-<fact_type>-<anchor_note.slug>` note, created idempotently
    and backlinked from `anchor_note`'s `## Related` list.

    `anchor_note` may be an account, product or competitor. Homing a fact on
    the signal note rather than the entity's own note keeps both out of the
    lint's orphan list: the signal note gets the backlink, and the entity's
    mention (no longer the home) is wikilinked from the claim text.
    """
    slug = signal_note_slug(fact_type, anchor_note.slug)
    if note_exists(conn, slug):
        note, _claims = read_note(conn, slug)
        return note

    now = clock.now()
    label = fact_type.replace("_", " ").title()
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title=f"{label} — {anchor_note.title}",
        description=f"{label} signals for [[{anchor_note.slug}]].",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        related=[anchor_note.slug],
    )
    write_note(
        conn,
        note,
        [],
        changed_by="merge",
        reason=f"signal note created for fact_type={fact_type} on {anchor_note.slug}",
        commit=commit,
    )
    created.append(slug)
    if link_related(
        conn,
        anchor_note.slug,
        slug,
        changed_by="merge",
        reason=f"new {fact_type} signal",
        commit=commit,
    ):
        touched.append(anchor_note.slug)
    return note


def _note_slug_for_mention(
    conn: psycopg.Connection,
    registry: Registry,
    mention: ResolvedMention,
    created: list[str],
    *,
    commit: bool = True,
) -> str | None:
    if mention.kind not in _NOTE_BEARING_KINDS:
        return None
    return _ensure_and_track(conn, registry, mention.entity_id, created, commit=commit).slug


def _explicit_supersession_date(fact: Fact, candidate: Claim):
    """The new claim's `valid_from` if this `contradicts` verdict should
    supersede `candidate`; `None` means downgrade to `extends` instead.

    Requires both claims to carry an explicit `valid_from` and the new one
    to be strictly later. There is deliberately no fallback to the source's
    date: a model can still call two unrelated claims that gate together
    `contradicts` (it has `different` for that, but nothing forces it), and
    source dates would let those retire each other.
    """
    if fact.valid_from is None or candidate.valid_from is None:
        return None
    if fact.valid_from > candidate.valid_from:
        return fact.valid_from
    return None


def _undated_same_source(fact: Fact, candidate: Claim, source: SourceRow) -> bool:
    """A `contradicts` pair the review queue cannot help with: both claims
    come from the same source and at least one carries no explicit date,
    so they were said at the same moment and no date can order them.
    Pinned by `tests/test_review_queue_volume.py`."""
    if fact.valid_from is not None and candidate.valid_from is not None:
        return False
    return candidate.source_slug == source.slug


def _build_claim(
    *,
    note_slug: str,
    fact: Fact,
    source: SourceRow,
    claim_text: str,
    company_context_updated_at: datetime | None = None,
) -> Claim:
    return Claim(
        note_slug=note_slug,
        fact_type=fact.type.value,
        claim_text=claim_text,
        source_slug=source.slug,
        locator=fact.locator,
        quote=fact.quote,
        speaker=fact.speaker,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        recorded=recorded_dt(source),
        severity=fact.severity,
        sentiment=fact.sentiment,
        confidence=fact.confidence,
        company_context_updated_at=company_context_updated_at,
    )


def _insert_claim(conn: psycopg.Connection, claim: Claim, *, commit: bool = True) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO claims (
                note_slug, fact_type, claim_text, source_slug, locator, quote,
                speaker, valid_from, valid_to, recorded, severity, sentiment, confidence,
                company_context_updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                claim.note_slug,
                claim.fact_type,
                claim.claim_text,
                claim.source_slug,
                claim.locator,
                claim.quote,
                claim.speaker,
                claim.valid_from,
                claim.valid_to,
                claim.recorded,
                claim.severity,
                claim.sentiment,
                claim.confidence,
                claim.company_context_updated_at,
            ),
        )
        new_id = cur.fetchone()[0]
    if commit:
        conn.commit()
    return new_id


def _supersede(
    conn: psycopg.Connection,
    old_claim_id: int,
    new_claim_id: int,
    valid_to,
    *,
    source_slug: str | None = None,
    commit: bool = True,
) -> bool:
    """Mark `old_claim_id` superseded by `new_claim_id` on both clocks:
    `superseded_at`/`superseded_by` (record time) and `valid_to` (world
    time, the new claim's own date; spec sec 2).

    A claim already superseded is never overwritten: the first correction
    stands, and this one goes to `review_queue` (tagged with `source_slug`)
    instead. Returns whether that happened. Not idempotent: each blocked
    call adds a review row.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE claims SET superseded_at = now(), superseded_by = %s, valid_to = %s "
            "WHERE id = %s AND superseded_by IS NULL",
            (new_claim_id, valid_to, old_claim_id),
        )
        blocked = cur.rowcount == 0
        if blocked:
            cur.execute(
                "INSERT INTO review_queue (kind, payload, source_slug) "
                "VALUES ('other', %s::jsonb, %s)",
                (
                    json.dumps(
                        {
                            "reason": ("contradicts verdict blocked: claim was already superseded"),
                            "old_claim_id": old_claim_id,
                            "attempted_new_claim_id": new_claim_id,
                        }
                    ),
                    source_slug,
                ),
            )
    if commit:
        conn.commit()
    return blocked


def _insert_claim_citation(
    conn: psycopg.Connection,
    *,
    claim_id: int,
    source_slug: str,
    locator: str,
    quote: str | None,
    commit: bool = True,
) -> bool:
    """A `same` verdict's corroboration: an extra citation on `claim_id`,
    never a second `claims` row. A repeat `(claim_id, source_slug, locator)`
    is a no-op. Returns whether a new row was written."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO claim_citations (claim_id, source_slug, locator, quote)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (claim_id, source_slug, locator) DO NOTHING
            RETURNING id
            """,
            (claim_id, source_slug, locator, quote),
        )
        written = cur.fetchone() is not None
    if commit:
        conn.commit()
    return written


def _queue_downgraded_contradiction(
    conn: psycopg.Connection,
    note_slug: str,
    old_id: int,
    new_id: int,
    *,
    old_valid_from,
    new_valid_from,
    source_slug: str | None = None,
    commit: bool = True,
) -> None:
    """Queue a `contradicts` verdict that was downgraded to `extends`
    (`_explicit_supersession_date`), so a human still sees the model's call.
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES ('other', %s::jsonb, %s)",
            (
                json.dumps(
                    {
                        "reason": (
                            "contradicts verdict downgraded to extends: not both "
                            "explicitly dated, or not strictly later"
                        ),
                        "note_slug": note_slug,
                        "old_claim_id": old_id,
                        "new_claim_id": new_id,
                        "old_valid_from": old_valid_from.isoformat() if old_valid_from else None,
                        "new_valid_from": new_valid_from.isoformat() if new_valid_from else None,
                    }
                ),
                source_slug,
            ),
        )
    if commit:
        conn.commit()


def _claim_from_an_earlier_version(
    conn: psycopg.Connection, fact: Fact, source: SourceRow
) -> int | None:
    """The claim an earlier version of this same document already recorded
    for this fact (same `sources.source_id`, another slug; same locator,
    type and quote), live, superseded, archived or human-edited alike. The
    gate only sees live claims, so without this a re-crawl would bring back
    a corrected fact or duplicate a live one (`tests/test_bughunt_wave2.py`)."""
    if not fact.quote:
        return None
    params = {
        "slug": source.slug,
        "locator": fact.locator,
        "fact_type": fact.type.value,
        "quote": fact.quote,
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH family AS (
                SELECT s.slug FROM sources s
                WHERE s.slug <> %(slug)s
                  AND s.source_id = (SELECT source_id FROM sources WHERE slug = %(slug)s)
            )
            SELECT c.id FROM claims c
            WHERE c.source_slug IN (SELECT slug FROM family)
              AND c.locator = %(locator)s AND c.fact_type = %(fact_type)s
              AND c.quote = %(quote)s
            UNION
            SELECT c.id FROM claim_citations cc JOIN claims c ON c.id = cc.claim_id
            WHERE cc.source_slug IN (SELECT slug FROM family)
              AND cc.locator = %(locator)s AND c.fact_type = %(fact_type)s
              AND cc.quote = %(quote)s
            ORDER BY 1
            LIMIT 1
            """,
            params,
        )
        row = cur.fetchone()
    return row[0] if row else None


def _queue_no_home(
    conn: psycopg.Connection,
    fact: Fact,
    source: SourceRow,
    resolved: list[ResolvedMention],
    *,
    commit: bool = True,
) -> bool:
    """Park a fact that names only the vendor's own side, or nobody, from a
    source with no account to file it under (`merge_fact`'s `fallback_account_id`).
    A repeat merge of the same source and claim is a no-op. Returns whether
    a row was written."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM review_queue WHERE kind = 'other' AND source_slug = %s "
            "AND payload ->> 'reason' = 'no_home_note' AND payload ->> 'claim' = %s",
            (source.slug, fact.claim),
        )
        if cur.fetchone() is not None:
            return False
        payload = {
            "reason": "no_home_note",
            "mentions": [r.mention for r in resolved],
            "fact_type": fact.type.value,
            "claim": fact.claim,
            "quote": fact.quote,
            "locator": fact.locator,
            "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
            "fact": fact_payload(fact),
        }
        cur.execute(
            "INSERT INTO review_queue (kind, payload, source_slug) VALUES ('other', %s::jsonb, %s)",
            (json.dumps(payload), source.slug),
        )
    if commit:
        conn.commit()
    return True


def merge_fact(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    registry: Registry,
    fact: Fact,
    source: SourceRow,
    *,
    vector_cache: dict[int, list[float]] | None = None,
    company_context_updated_at: datetime | None = None,
    fallback_account_id: str | None = None,
    commit: bool = True,
) -> MergeOutcome:
    """Merge one fact into the graph (module docstring).

    `commit=False` when the caller holds the whole source's writes in one
    open `conn.transaction()`, so a failure partway through rolls back every
    row the source wrote; `True` commits each write as it goes.

    `fallback_account_id`: the home for a fact with nothing to anchor a
    note and nothing waiting on a person: its names all resolved, but only
    to the vendor's own side (e.g. "Halvard Support will send the delivery
    details" on a customer's ticket), or it named nobody. With none, the
    fact is parked for a person (`_queue_no_home`), never dropped."""
    notes_created: list[str] = []
    notes_touched: list[str] = []
    ambiguous_mentions: list[str] = []

    earlier_claim_id = _claim_from_an_earlier_version(conn, fact, source)
    if earlier_claim_id is not None:
        return MergeOutcome(
            fact_id=fact.id,
            status="same",
            claim_id=earlier_claim_id,
            already_recorded=True,
            reason="already recorded from an earlier version of this source",
        )

    resolved, unresolved = resolve_fact_entities(
        registry, fact, source, ambiguous_mentions=ambiguous_mentions
    )

    home = select_home(fact, resolved)
    if home is None and not unresolved and fallback_account_id is not None:
        # Nothing in the fact can anchor a note and nothing is waiting on a
        # person: it named only the vendor's own side, or nobody at all.
        home = select_home(fact, [ResolvedMention("", fallback_account_id, EntityKind.ACCOUNT)])
    # Parked with the whole fact; `blocking` when these rows are the only
    # place the fact now lives.
    queue_rows_from_resolution = park_unresolved(
        conn, unresolved, fact, source, blocking=home is None, commit=commit
    )
    if home is None and not unresolved:
        parked = _queue_no_home(conn, fact, source, resolved, commit=commit)
        return MergeOutcome(
            fact_id=fact.id,
            status="queued_unknown",
            reason=("nothing in fact.entities can anchor a note and the source has no account"),
            queue_rows=1 if parked else 0,
        )
    if home is None:
        return MergeOutcome(
            fact_id=fact.id,
            status="queued_unknown",
            reason="the fact waits on a name the registry does not know",
            queue_rows=queue_rows_from_resolution,
        )

    if home.kind in _SIGNAL_WRAPPED_HOME_KINDS:
        anchor_note = _ensure_and_track(
            conn, registry, home.entity_id, notes_created, commit=commit
        )
        home_note = _ensure_signal_note(
            conn,
            anchor_note,
            fact.type.value,
            created=notes_created,
            touched=notes_touched,
            commit=commit,
        )
    else:  # "contact"
        home_note = _ensure_and_track(conn, registry, home.entity_id, notes_created, commit=commit)

    # Every resolved entity gets (or already has) a note, so it can be
    # wikilinked from the claim text -- including secondary mentions that
    # aren't the home entity (notes contract: "every entity a claim
    # mentions is written as [[slug]]").
    mention_slug_pairs: list[tuple[str, str]] = []
    related_links: list[str] = []
    for mention in resolved:
        slug = _note_slug_for_mention(conn, registry, mention, notes_created, commit=commit)
        # A claim never links the note it sits on.
        if slug is not None and slug != home_note.slug:
            mention_slug_pairs.append((mention.mention, slug))
        # "Signal notes link their account and product(s)": collected for
        # `merge_extraction`'s one end-of-source write, not written here.
        if (
            home.kind in _SIGNAL_WRAPPED_HOME_KINDS
            and mention.kind in (EntityKind.PRODUCT, EntityKind.COMPETITOR)
            and slug
        ):
            related_links.append(slug)

    claim_text = link_mentions(fact.claim, mention_slug_pairs)
    if ambiguous_mentions:
        claim_text = f"{claim_text} (ambiguous: {', '.join(ambiguous_mentions)})"

    home_note, existing_claims = read_note(conn, home_note.slug)
    candidate, score = gate(embedder, claim_text, existing_claims, vector_cache=vector_cache)

    compare_calls = 0
    compare_in = compare_out = 0
    compare_verdict: str | None = None
    compare_reason: str | None = None
    status: str

    if candidate is None or score < GATE_SKIP_THRESHOLD:
        status = "new"
    else:
        result = llm.complete(
            system=COMPARE_SYSTEM,
            user=build_compare_prompt(
                candidate=candidate, fact=fact, source_date=source.source_date
            ),
            json_schema=compare_verdict_json_schema(),
            surface="merge",
        )
        verdict = CompareVerdict.model_validate(result.data)
        compare_calls = 1
        compare_in = result.input_tokens
        compare_out = result.output_tokens
        compare_verdict = verdict.verdict
        compare_reason = verdict.reason
        # `different`: the gate paired two unrelated claims; the fact is new.
        status = "new" if verdict.verdict == "different" else verdict.verdict

    queue_rows = queue_rows_from_resolution
    claim_id: int | None = None
    claim_inserted = False
    citation_inserted = False

    # Claim rows only: `merge_extraction` re-renders the home note once per
    # source, so five facts on one note make one `note_versions` row, not five.
    if status == "new" or status == "extends":
        new_claim = _build_claim(
            note_slug=home_note.slug,
            fact=fact,
            source=source,
            claim_text=claim_text,
            company_context_updated_at=company_context_updated_at,
        )
        claim_id = _insert_claim(conn, new_claim, commit=commit)
        claim_inserted = True

    elif status == "same":
        # A second sighting is an extra citation; the claim's own first
        # citation (`claims.source_slug`/`locator`) is never touched.
        assert candidate is not None
        claim_id = candidate.id
        # The claim's own primary citation is not a second sighting.
        if (candidate.source_slug, candidate.locator) != (source.slug, fact.locator):
            citation_inserted = _insert_claim_citation(
                conn,
                claim_id=candidate.id,
                source_slug=source.slug,
                locator=fact.locator,
                quote=fact.quote,
                commit=commit,
            )

    elif status == "contradicts":
        assert candidate is not None
        new_claim = _build_claim(
            note_slug=home_note.slug,
            fact=fact,
            source=source,
            claim_text=claim_text,
            company_context_updated_at=company_context_updated_at,
        )
        claim_id = _insert_claim(conn, new_claim, commit=commit)
        claim_inserted = True
        supersede_date = _explicit_supersession_date(fact, candidate)
        if supersede_date is not None:
            blocked = _supersede(
                conn,
                candidate.id,
                claim_id,
                supersede_date,
                source_slug=source.slug,
                commit=commit,
            )
            if blocked:
                queue_rows += 1  # `_supersede` queued a review row
        else:
            # Downgraded: both claims stay live, and the status reports what
            # the graph did, not what the model said.
            if not _undated_same_source(fact, candidate, source):
                _queue_downgraded_contradiction(
                    conn,
                    home_note.slug,
                    candidate.id,
                    claim_id,
                    old_valid_from=candidate.valid_from,
                    new_valid_from=fact.valid_from,
                    source_slug=source.slug,
                    commit=commit,
                )
                queue_rows += 1
            status = "extends"

    created_set = set(notes_created)
    notes_updated = list(dict.fromkeys(s for s in notes_touched if s not in created_set))

    return MergeOutcome(
        fact_id=fact.id,
        status=status,  # type: ignore[arg-type]
        note_slug=home_note.slug,
        claim_id=claim_id,
        claim_inserted=claim_inserted,
        citation_inserted=citation_inserted,
        reason=compare_reason,
        compare_verdict=compare_verdict,
        compare_calls=compare_calls,
        compare_input_tokens=compare_in,
        compare_output_tokens=compare_out,
        notes_created=notes_created,
        notes_updated=notes_updated,
        queue_rows=queue_rows,
        related_links=list(dict.fromkeys(related_links)),
    )
