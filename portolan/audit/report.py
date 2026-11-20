"""`run_audit` and `render_markdown` (`01-spec/quality-loop.md` "The
production audit", `01-spec/build-contracts.md`).

`run_audit` is the one entry point: sample, second-pass, diff, judge,
roll up, write the report note + changelog row, return the `AuditReport`.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import psycopg

from portolan.audit.diff import diff_facts
from portolan.audit.judge import JudgeClient, judge
from portolan.audit.models import (
    AuditReport,
    FactDiff,
    Judgement,
    TypeRate,
    UsageTotals,
    WorstSource,
)
from portolan.audit.sample import population_counts, sample_sources
from portolan.audit.second_pass import second_pass
from portolan.extract.backends import Backend
from portolan.notes.models import Claim, NoteRow, NoteStatus, NoteType
from portolan.notes.persist import (
    _citations_for_claims,
    append_changelog,
    note_exists,
    read_note,
    write_note,
)
from portolan.registry import Registry
from portolan.store import Store
from portolan.util import clock

__all__ = ["WORST_SOURCES_LIMIT", "render_markdown", "run_audit"]

WORST_SOURCES_LIMIT = 10


# --------------------------------------------------------------------------
# claims already on file for one source
# --------------------------------------------------------------------------


def _claims_for_source(conn: psycopg.Connection, source_slug: str) -> list[Claim]:
    """Every unarchived claim primarily cited to `source_slug`, with its
    extra citations (review finding G4's `claim_citations`, so `diff_
    facts` can report a corroboration count) attached.

    Review finding G20: `superseded_at IS NULL` -- a superseded claim
    (G17's second clock) is dead, not a candidate invention when the
    second pass doesn't happen to restate it too. `diff_facts` filters
    this defensively as well; both places matter since `diff_facts` is a
    pure function and should not silently depend on every caller having
    already pre-filtered.

    Excludes claims on `action`-type notes (found while fixing G23):
    `merge_action` (`portolan/merge/actions.py`) writes its own claim row
    for an action, always `fact_type='other'`, alongside `merge_fact`'s
    ordinary claim for the same underlying fact on its signal/account
    note -- a structured echo of a fact already captured, not an
    independently-extracted one. `second_pass`'s `ExtractionResult.facts`
    has no notion of "actions" at all, so this echo can never match
    anything there; before G23 tightened `_entities_overlap`'s
    "nothing to check" escape hatch, it silently auto-matched instead,
    masking the mismatch rather than correctly reporting it as neither a
    miss nor an invention.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.note_slug, c.fact_type, c.claim_text, c.source_slug, c.locator,
                   c.quote, c.speaker, c.valid_from, c.valid_to, c.recorded, c.superseded_at,
                   c.severity, c.sentiment, c.confidence, c.superseded_by
            FROM claims c
            JOIN notes n ON n.slug = c.note_slug
            WHERE c.source_slug = %s AND c.archived = false AND c.superseded_at IS NULL
              AND n.type != 'action'
            ORDER BY c.recorded, c.id
            """,
            (source_slug,),
        )
        claim_rows = cur.fetchall()
        citations_by_claim = _citations_for_claims(cur, [r[0] for r in claim_rows])
        return [
            Claim(
                id=r[0],
                note_slug=r[1],
                fact_type=r[2],
                claim_text=r[3],
                source_slug=r[4],
                locator=r[5],
                quote=r[6],
                speaker=r[7],
                valid_from=r[8],
                valid_to=r[9],
                recorded=r[10],
                superseded_at=r[11],
                severity=r[12],
                sentiment=r[13],
                confidence=r[14],
                superseded_by=r[15],
                citations=citations_by_claim.get(r[0], []),
            )
            for r in claim_rows
        ]


def _usage_totals(events) -> UsageTotals:
    totals = UsageTotals()
    for event in events:
        totals.calls += 1
        totals.input_tokens += event.input_tokens
        totals.output_tokens += event.output_tokens
        totals.cached_tokens += event.cached_tokens
        totals.cost_estimate += event.cost_estimate
    return totals


def _iso_week_slug(today: date) -> str:
    iso = today.isocalendar()
    return f"audit-{iso.year}-w{iso.week:02d}"


# --------------------------------------------------------------------------
# run_audit
# --------------------------------------------------------------------------


def _score_misses(
    diffs: list[FactDiff], judgements: list[Judgement]
) -> tuple[dict[str, TypeRate], dict[str, TypeRate], int, int, dict[str, int]]:
    """The per-`FactDiff` miss-scoring pass, factored out of `run_audit`
    for direct unit testing (review finding 12, test gap 13) with no DB,
    LLM, or pipeline needed -- just `DiffCandidate`/`Judgement` objects.

    Maps `judgements` back to the candidates in `diffs` by `id(candidate)`
    -- each `Judgement`'s own `.candidate` (`audit/models.py`), the SAME
    object `judge()` was handed via `all_miss_candidates` -- rather than
    by position. Before this fix, the mapping was positional
    (`enumerate`), correct only while `judge()` returns exactly one
    `Judgement` per candidate, in order; a `judgements` list shorter than
    `sum(len(d.misses) for d in diffs)` (a batched call, a retry, a
    filtered response) would then silently attach every verdict after the
    gap to the WRONG candidate, and `dict.get(idx, True)`'s default
    counted an unjudged candidate as a genuine miss -- inflating the
    headline number the whole audit exists to report. A candidate with no
    matching `Judgement` at all is now counted in the returned
    `unjudged_misses_total` instead.

    Returns `(miss_by_fact_type, miss_by_source_type, genuine_misses_
    total, unjudged_misses_total, source_genuine_by_slug)` -- the last a
    per-source count of genuine misses, for `run_audit`'s own `WorstSource`
    rows.
    """
    judgement_by_candidate_id: dict[int, Judgement] = {id(j.candidate): j for j in judgements}

    miss_by_fact_type: dict[str, TypeRate] = {}
    miss_by_source_type: dict[str, TypeRate] = {}
    genuine_misses_total = 0
    unjudged_misses_total = 0
    source_genuine_by_slug: dict[str, int] = {}

    for diff in diffs:
        # Denominators first: every second-pass fact of a type/source_type
        # counts, matched or missed, so a type with zero misses still
        # shows "0 of N" rather than being absent from the table.
        miss_by_source_type.setdefault(diff.source_type, TypeRate())
        miss_by_source_type[diff.source_type].total += diff.second_pass_count
        for fact_type, count in diff.fact_type_totals.items():
            miss_by_fact_type.setdefault(fact_type, TypeRate())
            miss_by_fact_type[fact_type].total += count

        source_genuine = 0
        for candidate in diff.misses:
            judgement = judgement_by_candidate_id.get(id(candidate))
            if judgement is None:
                unjudged_misses_total += 1
                continue
            if judgement.verdict == "genuine_miss":
                miss_by_fact_type[candidate.fact_type].misses += 1
                miss_by_source_type[diff.source_type].misses += 1
                genuine_misses_total += 1
                source_genuine += 1
        source_genuine_by_slug[diff.source_slug] = source_genuine

    return (
        miss_by_fact_type,
        miss_by_source_type,
        genuine_misses_total,
        unjudged_misses_total,
        source_genuine_by_slug,
    )


def run_audit(
    conn: psycopg.Connection,
    store: Store,
    llm: JudgeClient,
    strong_backend: Backend,
    *,
    since: date,
    rate: float = 0.02,
    seed: str | int = "audit",
    today: date | None = None,
    registry: Registry | None = None,
    connector=None,
) -> AuditReport:
    # Review finding 13: stdlib `date.today()` is the process host's own
    # local zone -- the exact wall-clock leak `portolan.util.clock` was
    # written to remove everywhere in this build's owned files (its own
    # docstring names `report/**`, which `portolan.audit.report` is not,
    # but the leak is the same one regardless of which module it's in).
    # An audit run near midnight used to be able to file its own note
    # under the wrong ISO week.
    resolved_today = today if today is not None else clock.today()
    note_slug = _iso_week_slug(resolved_today)
    registry = registry if registry is not None else Registry.load(conn)

    # Review finding G10: `sample_sources` now draws from `done` AND
    # `partial` sources by default (the likeliest place for a miss to
    # hide); `population_counts` is the same eligible population, used
    # below to weight the headline rate by how much of the real traffic
    # each stratum actually is.
    sources = sample_sources(conn, since=since, rate=rate, seed=seed)
    population_by_source_type = population_counts(conn, since=since)
    sp_results = second_pass(
        conn, store, strong_backend, sources, registry=registry, connector=connector
    )

    diffs = []
    corroborations_total = 0
    for source in sources:
        result = sp_results.get(source.slug)
        if result is None:
            continue
        first_claims = _claims_for_source(conn, source.slug)
        diff = diff_facts(first_claims, result, source.source_type, registry)
        diffs.append(diff)
        corroborations_total += diff.corroborations_count

    all_miss_candidates = [c for d in diffs for c in d.misses]
    judgements: list[Judgement] = judge(llm, all_miss_candidates)

    (
        miss_by_fact_type,
        miss_by_source_type,
        genuine_misses_total,
        unjudged_misses_total,
        source_genuine_by_slug,
    ) = _score_misses(diffs, judgements)

    inventions_total = 0
    first_pass_total = 0
    second_pass_total = 0
    worst: list[WorstSource] = []

    for diff in diffs:
        inventions_total += len(diff.inventions)
        first_pass_total += diff.first_pass_count
        second_pass_total += diff.second_pass_count

        worst.append(
            WorstSource(
                source_slug=diff.source_slug,
                source_type=diff.source_type,
                genuine_misses=source_genuine_by_slug.get(diff.source_slug, 0),
                inventions=len(diff.inventions),
                # Review finding G22: plain text, never `[[wikilink]]` --
                # `sources` rows are not notes (spec sec 2 lists `source`
                # among the ten note types, but nothing in this codebase
                # creates one yet), so a `[[source-slug]]` token here was
                # a `links` row (`kind='body'`) the lint's `broken_link`
                # check could never resolve, one error per worst source.
                # Whether source notes should exist at all is a decision
                # for whoever next touches the note-type list, not this
                # fix wave -- noted in the wave report.
                link=diff.source_slug,
            )
        )

    worst.sort(key=lambda w: w.genuine_misses + w.inventions, reverse=True)
    worst = worst[:WORST_SOURCES_LIMIT]

    # `judge()` calls go through `llm` directly, whose own sink (if any) is
    # the caller's concern to inspect separately; the usage this report
    # tallies is the second-pass extraction spend from `sp_results.sink`.
    usage_events = list(getattr(sp_results.sink, "events", []))
    usage = _usage_totals(usage_events)

    miss_rate_overall = genuine_misses_total / second_pass_total if second_pass_total else None
    invention_rate_overall = inventions_total / first_pass_total if first_pass_total else None

    # Review finding G10, sampling bias #2: weight each source type's own
    # miss rate by its real population share, not by how many of its
    # sources happened to land in the sample -- a stratum with only 3
    # sources sampled at ~100% and one with 10,000 sampled at 2% must not
    # count equally toward the headline number.
    weighted_num = 0.0
    weighted_den = 0.0
    for source_type, type_rate in miss_by_source_type.items():
        if type_rate.total == 0 or type_rate.miss_rate is None:
            continue
        weight = population_by_source_type.get(source_type, 0)
        weighted_num += type_rate.miss_rate * weight
        weighted_den += weight
    miss_rate_weighted = weighted_num / weighted_den if weighted_den else None

    report = AuditReport(
        note_slug=note_slug,
        since=since,
        rate=rate,
        seed=str(seed),
        sources_sampled=len(sources),
        sources_second_passed=len(sp_results),
        first_pass_claims_total=first_pass_total,
        second_pass_facts_total=second_pass_total,
        genuine_misses_total=genuine_misses_total,
        inventions_total=inventions_total,
        unjudged_misses_total=unjudged_misses_total,
        corroborations_total=corroborations_total,
        miss_rate_overall=miss_rate_overall,
        invention_rate_overall=invention_rate_overall,
        miss_rate_weighted=miss_rate_weighted,
        population_by_source_type=population_by_source_type,
        miss_by_fact_type=miss_by_fact_type,
        miss_by_source_type=miss_by_source_type,
        worst_sources=worst,
        judgements=judgements,
        skipped_sources=sorted(sp_results.failures),
        tokens_input=usage.input_tokens,
        tokens_output=usage.output_tokens,
        tokens_cached=usage.cached_tokens,
        cost_estimate=usage.cost_estimate,
    )

    _write_report_note(conn, report, today=resolved_today)
    append_changelog(
        conn,
        "audit",
        f"production audit {note_slug}: {genuine_misses_total} genuine miss(es) of "
        f"{second_pass_total} second-pass facts, {inventions_total} candidate invention(s)",
        {
            "note_slug": note_slug,
            "since": since.isoformat(),
            "rate": rate,
            "sources_sampled": len(sources),
        },
    )

    return report


def _write_report_note(conn: psycopg.Connection, report: AuditReport, *, today: date) -> None:
    # Review finding 13: stamp in the deployment's own zone via
    # `portolan.util.clock` (`PORTOLAN_TIMEZONE`), not literal UTC.
    # `portolan.notes.persist` no longer stamps UTC either -- `e5c92b3`
    # changed it to `clock.now()` -- so the old comment here ("every other
    # writer in this codebase stamps UTC explicitly") argued for the
    # opposite of what the codebase actually does, which is worse than no
    # comment at all. Finding 1's fix is exactly why the specific zone
    # still matters: whatever later reads this `created`/`updated` value
    # back and wants "which calendar day" must cast through the SAME zone
    # (`portolan.util.clock.world_date_sql`), not rely on the reading
    # session's own `TimeZone`.
    now = datetime.combine(today, datetime.min.time(), tzinfo=ZoneInfo(clock.timezone_name()))
    if note_exists(conn, report.note_slug):
        note, _claims = read_note(conn, report.note_slug)
    else:
        note = NoteRow(
            slug=report.note_slug,
            type=NoteType.REPORT,
            title=f"Production audit — {report.note_slug.removeprefix('audit-')}",
            description=(
                f"Weekly production audit: miss rate "
                f"{_fmt_pct(report.miss_rate_overall)}, "
                f"{report.inventions_total} candidate invention(s)."
            ),
            status=NoteStatus.LIVING,
            created=now,
            updated=now,
        )
    note.synthesis = render_markdown(report)
    write_note(conn, note, [], changed_by="audit", reason="weekly production audit")


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


# --------------------------------------------------------------------------
# render_markdown
# --------------------------------------------------------------------------


def render_markdown(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append(f"# Production audit — {report.note_slug}")
    lines.append("")
    lines.append(
        f"Since {report.since.isoformat()}, sampled {report.sources_sampled} source(s) "
        f"at rate {report.rate:.0%} (seed `{report.seed}`); "
        f"{report.sources_second_passed} re-extracted successfully."
    )
    lines.append("")
    lines.append(
        f"**Miss rate (population-weighted)**: {_fmt_pct(report.miss_rate_weighted)} -- "
        f"the headline number (review finding G10): each source type's own rate weighted "
        f"by its real share of eligible sources, not by how many landed in the sample."
    )
    lines.append(
        f"**Miss rate (pooled, unweighted)**: {_fmt_pct(report.miss_rate_overall)} "
        f"({report.genuine_misses_total} of {report.second_pass_facts_total} "
        f"second-pass facts) -- over-represents a rare, fully-sampled source type; "
        f"kept for comparison only."
    )
    lines.append(
        f"**Candidate inventions (not judged)**: {_fmt_pct(report.invention_rate_overall)} "
        f"({report.inventions_total} of {report.first_pass_claims_total} "
        f"first-pass claims) -- claims on file the second pass did not restate, matched "
        f"by wording alone. Nothing has read them against the source, so this is a list "
        f"to check, not a rate of invented facts."
    )
    if report.unjudged_misses_total:
        # Review finding 12: a genuine gap in the judge call (never a
        # silent "count it as a genuine miss") -- surfaced here rather
        # than folded into either rate above.
        lines.append(
            f"**Unjudged miss candidates**: {report.unjudged_misses_total} -- the judge call "
            f"returned no verdict for these; excluded from the miss rate above rather than "
            f"assumed genuine."
        )
    lines.append(
        f"**Corroborations**: {report.corroborations_total} (review finding G4/G20 -- a fact "
        f"seen again by a second source or a second mention, never counted as a miss or an "
        f"invention)."
    )
    lines.append("")

    if report.miss_by_fact_type:
        lines.append("## Miss rate by fact type")
        lines.append("")
        lines.append("| Fact type | Misses | Total | Rate |")
        lines.append("|---|---:|---:|---:|")
        for fact_type in sorted(report.miss_by_fact_type):
            rate = report.miss_by_fact_type[fact_type]
            lines.append(
                f"| {fact_type} | {rate.misses} | {rate.total} | {_fmt_pct(rate.miss_rate)} |"
            )
        lines.append("")

    if report.miss_by_source_type:
        lines.append("## Miss rate by source type")
        lines.append("")
        lines.append("| Source type | Misses | Sampled | Rate | Population (eligible) |")
        lines.append("|---|---:|---:|---:|---:|")
        for source_type in sorted(report.miss_by_source_type):
            rate = report.miss_by_source_type[source_type]
            population = report.population_by_source_type.get(source_type, 0)
            lines.append(
                f"| {source_type} | {rate.misses} | {rate.total} | {_fmt_pct(rate.miss_rate)} "
                f"| {population} |"
            )
        lines.append("")

    if report.worst_sources:
        lines.append("## Worst sources")
        lines.append("")
        lines.append("| Source | Type | Genuine misses | Inventions |")
        lines.append("|---|---|---:|---:|")
        for w in report.worst_sources:
            lines.append(f"| `{w.link}` | {w.source_type} | {w.genuine_misses} | {w.inventions} |")
        lines.append("")

    if report.skipped_sources:
        lines.append("## Skipped (could not re-read/re-extract)")
        lines.append("")
        for slug in report.skipped_sources:
            lines.append(f"- {slug}")
        lines.append("")

    lines.append(
        f"Tokens: {report.tokens_input} in / {report.tokens_output} out "
        f"({report.tokens_cached} cached). Estimated cost: ${report.cost_estimate:.4f}."
    )

    return "\n".join(lines).rstrip() + "\n"
