"""`sample_sources`: a deterministic, stratified sample of `done`/`partial`
sources for the production audit (`01-spec/quality-loop.md` "The
production audit": "each week, sample one to two percent of sources
ingested that week, stratified by source type").

Determinism matters here more than for most sampling code: the same
`(since, rate, seed)` must always name the same sources, so a re-run
after a crash, or a second look at last week's audit, inspects the same
evidence rather than a fresh random draw. `random.Random(seed)` (a
per-process, not per-interpreter-global, PRNG) gives that.

**Review finding G10 (fix wave 2026-09-12), sampling bias #1: `partial`
sources were excluded.** A source that only partially extracted (one
segment failed) is exactly the likeliest place for a miss to hide, and
the old query only ever sampled `extraction_status = 'done'`. Both
statuses are eligible now; a caller that genuinely wants `done`-only can
still pass `statuses=("done",)`.

**Sampling bias #2, reported here for `run_audit` to use: population vs
sample size.** `sample_sources` guarantees at least `minimum` source per
non-empty stratum, so a rare source type (3 RSS items a week) is fully
sampled while a common one (10,000 calls) gets its nominal 1-2% -- correct
for *coverage*, but pooling the resulting miss counts with equal weight
per candidate then over-represents the rare stratum by orders of
magnitude in the headline rate. `population_counts` gives `run_audit` the
per-stratum population sizes it needs to weight the other way (see
`audit.report.run_audit`'s `miss_rate_weighted`).

**Slop list item, closed polish wave 2026-09-12**: this module used to
carry its own third copy of `portolan.ingest.intake`'s column list and
row mapper -- one that never even selected `metadata`, so every sampled
`SourceRow.metadata` was silently `{}` regardless of what `intake` stored.
It now imports the public `SOURCE_COLUMNS`/`row_to_source_row` instead, so
a sampled source carries its real `metadata` (including migration
`0009`'s `raw` key, `portolan.audit.second_pass._reconstruct_raw_source`
needs).
"""

from __future__ import annotations

import math
import random
from datetime import date

import psycopg

from portolan.ingest.intake import SOURCE_COLUMNS, row_to_source_row
from portolan.ingest.models import SourceRow

__all__ = ["population_counts", "sample_sources"]

# G10: `partial` sources (one segment failed) are the likeliest place for
# a miss to hide and must be eligible for the sample, not just `done`.
_ELIGIBLE_STATUSES: tuple[str, ...] = ("done", "partial")


def _eligible_sources_since(
    conn: psycopg.Connection, since: date, statuses: tuple[str, ...]
) -> list[SourceRow]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {", ".join(SOURCE_COLUMNS)} FROM sources
            WHERE extraction_status = ANY(%s)
              AND COALESCE(source_date, ingested_at::date) >= %s
            ORDER BY slug
            """,
            (list(statuses), since),
        )
        return [row_to_source_row(row) for row in cur.fetchall()]


def _strata(rows: list[SourceRow], stratify_by: str) -> dict[str, list[SourceRow]]:
    strata: dict[str, list[SourceRow]] = {}
    for row in rows:
        key = str(getattr(row, stratify_by))
        strata.setdefault(key, []).append(row)
    return strata


def sample_sources(
    conn: psycopg.Connection,
    *,
    since: date,
    rate: float = 0.02,
    seed: str | int,
    stratify_by: str = "source_type",
    minimum: int = 1,
    statuses: tuple[str, ...] = _ELIGIBLE_STATUSES,
) -> list[SourceRow]:
    """A deterministic sample of `rate` (1-2% by default) of `done`/
    `partial` sources (review finding G10; pass `statuses=("done",)` for
    the old, narrower behaviour) with `source_date` (falling back to
    `ingested_at`'s date) on or after `since`, stratified by
    `stratify_by` (a `SourceRow` attribute, `"source_type"` by default) so
    a rare source type is not starved out by a common one.

    Within each stratum, at least `minimum` source is sampled whenever the
    stratum is non-empty (so a stratum with only a handful of sources
    still gets audited at all), else `ceil(len(stratum) * rate)`, capped
    at the stratum's size. The per-stratum draw order is shuffled with a
    `random.Random` seeded from `f"{seed}:{stratum_key}"`, so the same
    `(since, rate, seed)` always names the same sources -- re-running the
    audit, or inspecting last week's sample again, sees identical evidence
    (module docstring).
    """
    rows = _eligible_sources_since(conn, since, statuses)
    strata = _strata(rows, stratify_by)

    sampled: list[SourceRow] = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: r.slug)
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(bucket)
        n = min(len(bucket), max(minimum, math.ceil(len(bucket) * rate)))
        sampled.extend(bucket[:n])

    sampled.sort(key=lambda r: r.slug)
    return sampled


def population_counts(
    conn: psycopg.Connection,
    *,
    since: date,
    stratify_by: str = "source_type",
    statuses: tuple[str, ...] = _ELIGIBLE_STATUSES,
) -> dict[str, int]:
    """`{stratum_key: population_size}` -- every eligible source in the
    window, not just the sampled ones (review finding G10, sampling bias
    #2: this is what lets `run_audit` weight a per-stratum miss rate by
    how much of the real traffic that stratum actually is, instead of
    pooling raw counts sampled at wildly different rates -- a small
    stratum sampled near 100% and a large one sampled at 2% must not
    count equally toward the headline number)."""
    rows = _eligible_sources_since(conn, since, statuses)
    strata = _strata(rows, stratify_by)
    return {key: len(bucket) for key, bucket in strata.items()}
