"""`merge_extraction` (`01-spec/build-contracts.md` "Merge"): loops
`merge_fact` over `result.facts` and `merge_action` over `result.actions`,
then updates `sources.summary` and the changelog, all in one transaction.

Never writes `sources.extraction_status`; the caller sets it (`done` or
`partial`) from the whole run's outcome.

A non-empty extraction that writes nothing to the graph (e.g. no entity
resolved) is legitimate but looks like a lost write from outside, so it gets
its own `merge_empty` changelog row with the counts.
"""

from __future__ import annotations

from datetime import datetime

import psycopg

from portolan.extract.schema import EntityKind, ExtractionResult
from portolan.ingest.models import SourceRow
from portolan.llm.base import LLMClient
from portolan.merge.actions import merge_action
from portolan.merge.facts import merge_fact
from portolan.merge.models import FactPlacement, MergeReport
from portolan.metering import InMemorySink
from portolan.notes import append_changelog, read_note, write_note
from portolan.registry import Registry
from portolan.search.embed import Embedder


def _primary_account_id(
    registry: Registry, source: SourceRow, result: ExtractionResult
) -> str | None:
    """The source's account: the first account among the extraction's
    top-level `entities`, else the first participant's account by email or
    email domain. Actions link to it, and facts that name only the vendor's
    own side are homed on it."""
    for mention in result.entities:
        if mention.kind != EntityKind.ACCOUNT:
            continue
        if mention.registry_id:
            entity = registry.get(mention.registry_id)
            if entity is not None:
                return entity.id
        resolution = registry.resolve(mention.mention)
        if resolution.status == "resolved" and resolution.entity_id:
            entity = registry.get(resolution.entity_id)
            if entity is not None and entity.kind == EntityKind.ACCOUNT:
                return entity.id

    for participant in source.participants:
        if not participant.email:
            continue
        entity = registry.by_email(participant.email)
        if entity is not None and entity.kind == EntityKind.ACCOUNT:
            return entity.id
        domain = participant.email.rsplit("@", 1)[-1] if "@" in participant.email else None
        if domain:
            for candidate in registry.by_domain(domain):
                if candidate.kind == EntityKind.ACCOUNT:
                    return candidate.id
    return None


def merge_extraction(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    registry: Registry,
    result: ExtractionResult,
    source: SourceRow,
    *,
    company_context_updated_at: datetime | None = None,
) -> MergeReport:
    """Merge one source's extraction result; atomic per source.

    `company_context_updated_at`: the company-context note's `updated` when
    the run loaded it, stamped on every inserted claim
    (`claims.company_context_updated_at`); `None` stamps nothing."""
    report = MergeReport(source_slug=source.slug)
    created: set[str] = set()
    updated: set[str] = set()
    claim_touched_notes: list[str] = []  # order-preserving; deduped below
    pending_related: dict[str, list[str]] = {}
    # One embedding per existing claim for this call, not one per fact.
    vector_cache: dict[int, list[float]] = {}

    # One transaction per source: every writer below gets `commit=False`,
    # so an exception rolls back everything this source wrote and the
    # caller's retry starts from a clean state, instead of leaving claims
    # that no note renders.
    #
    # Usage events are buffered and replayed only after the commit: a sink
    # that commits on this same `conn` (`portolan.metering.PostgresSink`)
    # would raise inside `conn.transaction()`. A source whose merge rolls
    # back loses the recorded cost of its compare calls.
    real_sink = getattr(llm, "sink", None)
    buffered_sink = InMemorySink()
    if real_sink is not None:
        llm.sink = buffered_sink  # type: ignore[attr-defined]
    try:
        with conn.transaction():
            primary_account_id = _primary_account_id(registry, source, result)
            for fact in result.facts:
                outcome = merge_fact(
                    conn,
                    llm,
                    embedder,
                    registry,
                    fact,
                    source,
                    vector_cache=vector_cache,
                    company_context_updated_at=company_context_updated_at,
                    fallback_account_id=primary_account_id,
                    commit=False,
                )
                report.outcome_counts[outcome.status] = (
                    report.outcome_counts.get(outcome.status, 0) + 1
                )
                report.compare_calls += outcome.compare_calls
                report.compare_input_tokens += outcome.compare_input_tokens
                report.compare_output_tokens += outcome.compare_output_tokens
                report.queue_rows += outcome.queue_rows
                if outcome.compare_verdict is not None:
                    report.compare_verdicts[outcome.compare_verdict] = (
                        report.compare_verdicts.get(outcome.compare_verdict, 0) + 1
                    )
                if outcome.claim_inserted:
                    report.claims_inserted += 1
                if outcome.citation_inserted:
                    report.citations_inserted += 1
                if outcome.already_recorded:
                    report.facts_already_recorded += 1
                created.update(outcome.notes_created)
                updated.update(outcome.notes_updated)
                if outcome.note_slug is not None:
                    claim_touched_notes.append(outcome.note_slug)
                    if outcome.related_links:
                        pending_related.setdefault(outcome.note_slug, []).extend(
                            outcome.related_links
                        )
                report.placements.append(
                    FactPlacement(
                        fact_id=outcome.fact_id,
                        claim_id=outcome.claim_id,
                        note_slug=outcome.note_slug,
                    )
                )

            # Re-render every note that received a claim, once per source
            # ("one version per source, not per claim"), with any `## Related`
            # links its facts asked for.
            for note_slug in dict.fromkeys(claim_touched_notes):
                note, claims = read_note(conn, note_slug)
                for related_slug in pending_related.get(note_slug, []):
                    if related_slug not in note.related:
                        note.related.append(related_slug)
                write_note(
                    conn,
                    note,
                    claims,
                    changed_by="merge",
                    reason=f"source {source.slug}",
                    embedder=embedder,
                    commit=False,
                )
                updated.add(note_slug)

            for action in result.actions:
                action_outcome = merge_action(
                    conn,
                    registry,
                    action,
                    source,
                    embedder=embedder,
                    entity_id=primary_account_id,
                    commit=False,
                )
                if action_outcome.status == "new":
                    report.actions_new += 1
                else:
                    report.actions_deduped += 1
                created.update(action_outcome.notes_created)
                updated.update(action_outcome.notes_updated)

            updated -= created
            report.notes_created = len(created)
            report.notes_updated = len(updated)

            # `summary` only; never `extraction_status` (module docstring).
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sources SET summary = %s WHERE slug = %s",
                    (result.summary, source.slug),
                )

            facts_extracted = len(result.facts)
            actions_extracted = len(result.actions)
            merge_wrote_nothing = (
                report.notes_created == 0
                and report.notes_updated == 0
                and report.claims_inserted == 0
                and report.citations_inserted == 0
                and report.actions_new == 0
                and report.actions_deduped == 0
            )
            # A re-crawled version whose facts were all already recorded
            # from an earlier version legitimately writes nothing.
            facts_not_already_recorded = facts_extracted - report.facts_already_recorded
            if merge_wrote_nothing and (facts_not_already_recorded > 0 or actions_extracted > 0):
                append_changelog(
                    conn,
                    "merge_empty",
                    f"merge for {source.slug} wrote zero notes/claims despite "
                    f"{facts_extracted} fact(s) and {actions_extracted} action(s) extracted "
                    f"(outcome_counts={report.outcome_counts})",
                    {
                        "source_slug": source.slug,
                        "facts_extracted": facts_extracted,
                        "actions_extracted": actions_extracted,
                        "outcome_counts": report.outcome_counts,
                    },
                    commit=False,
                )

            append_changelog(
                conn,
                "merge",
                f"merged {source.slug}: {report.outcome_counts}, "
                f"{report.notes_created} notes created, {report.notes_updated} updated, "
                f"{report.claims_inserted} claims inserted, {report.compare_calls} compare calls",
                report.model_dump(mode="json"),
                commit=False,
            )
    finally:
        if real_sink is not None:
            llm.sink = real_sink  # type: ignore[attr-defined]

    # Reached only after the transaction committed.
    if real_sink is not None:
        for event in buffered_sink.events:
            real_sink.record(event)

    return report
