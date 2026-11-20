"""One external-signals scan: check every active target, judge what is new, write
what matters (`01-spec/external-signals-2026-09-23.md` "Schedule").

    feeds -> new items (by GUID)          \\
                                            -> judge (cap, budget) -> source + claim + note
    pages -> snapshot -> diff -> change   /

Refuses to run unless `PORTOLAN_EXTERNAL_SIGNALS` is on. Every fetch error,
robots refusal and judge failure is counted on the report, the target's
`last_error`, the coverage ledger (connectors `signals_feed` and
`signals_web`) and one `external_signals_scan` changelog row. A judge call that
fails leaves the item `error` and it is retried on the next scan, three times
at most.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import psycopg

from portolan.config import Settings
from portolan.context import load_company_context
from portolan.ingest.ledger import record as record_coverage
from portolan.llm.base import LLMClient
from portolan.notes import append_changelog
from portolan.registry import EntityKind, Registry
from portolan.search.embed import Embedder
from portolan.signals.feeds import DEFAULT_FEED_DAYS, collect_feed_items
from portolan.signals.fetch import PoliteFetcher
from portolan.signals.judge import judge_item
from portolan.signals.targets import Target, load_targets
from portolan.signals.web import BrowserFetch, snapshot_page
from portolan.signals.write import SignalItem, load_item, write_public_signal
from portolan.store import Store
from portolan.util import clock

__all__ = ["MAX_JUDGE_ATTEMPTS", "ScanReport", "SignalsDisabled", "judge_pending", "run_scan"]

MAX_JUDGE_ATTEMPTS = 3
DEFAULT_JUDGE_LIMIT = 200


class SignalsDisabled(RuntimeError):
    MESSAGE = (
        "external signals are off: set PORTOLAN_EXTERNAL_SIGNALS=true to scan "
        "(01-spec/external-signals-2026-09-23.md)"
    )


@dataclass
class ScanReport:
    targets: int = 0
    feeds_checked: int = 0
    pages_checked: int = 0
    feed_items_new: int = 0
    feed_items_too_old: int = 0
    page_baselines: int = 0
    page_changes: int = 0
    needs_browser: int = 0
    robots_disallowed: int = 0
    personal_profile_refused: int = 0
    fetch_errors: int = 0
    unknown_accounts: int = 0
    judged: int = 0
    relevant: int = 0
    not_relevant: int = 0
    rejected: int = 0
    judge_errors: int = 0
    claims_written: int = 0
    findings_dropped: int = 0
    judge_cost_usd: float = 0.0
    budget_hit: bool = False
    errors: list[str] = field(default_factory=list)


def _set_target_status(conn: psycopg.Connection, target: Target, error: str | None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE signal_targets SET last_checked_at = now(), last_error = %s WHERE id = %s",
            (error, target.id),
        )
    conn.commit()


def _sink_events(llm: LLMClient) -> list | None:
    """The metered events list of `llm`'s sink (`InMemorySink` and
    `PostgresSink` both keep one), or `None` when it keeps none."""
    events = getattr(getattr(llm, "sink", None), "events", None)
    return events if isinstance(events, list) else None


def _cost_since(llm: LLMClient, start: int) -> float:
    events = _sink_events(llm)
    if events is None:
        return 0.0
    return float(sum(e.cost_estimate or 0.0 for e in events[start:]))


def _pending_ids(conn: psycopg.Connection, limit: int) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM signal_items
            WHERE status = 'pending' OR (status = 'error' AND attempts < %s)
            ORDER BY observed_at, id
            LIMIT %s
            """,
            (MAX_JUDGE_ATTEMPTS, limit),
        )
        return [r[0] for r in cur.fetchall()]


def _dated_line(item: SignalItem) -> str:
    if item.published_at is not None:
        return f"Published: {item.published_at:%Y-%m-%d}"
    return f"Observed: {item.observed_at:%Y-%m-%d}"


def _mark(conn: psycopg.Connection, item_id: int, status: str, **fields) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE signal_items SET status = %s, category = %s, judge_reason = %s,
                   judge_model = %s, judged_at = now(), attempts = attempts + 1
            WHERE id = %s
            """,
            (
                status,
                fields.get("category"),
                fields.get("reason"),
                fields.get("model"),
                item_id,
            ),
        )
    conn.commit()


def judge_pending(
    conn: psycopg.Connection,
    *,
    store: Store,
    llm: LLMClient,
    registry: Registry,
    model: str,
    report: ScanReport,
    limit: int = DEFAULT_JUDGE_LIMIT,
    budget_usd: float | None = None,
) -> None:
    """Judge up to `limit` pending items, oldest first, stopping early once
    this run's judge spend reaches `budget_usd`. Writes are not indexed here;
    the caller indexes once (`run_scan`)."""
    context = load_company_context(conn)
    vendor_context = context.text if context else None
    events = _sink_events(llm)
    start = len(events) if events is not None else 0
    if budget_usd is not None and events is None:
        raise ValueError("budget_usd needs an LLM client whose sink keeps its events")
    for item_id in _pending_ids(conn, limit):
        if budget_usd is not None and _cost_since(llm, start) >= budget_usd:
            report.budget_hit = True
            break
        item = load_item(conn, item_id)
        entity = registry.get(item.account_crm_id)
        if entity is None:
            _mark(conn, item_id, "error", reason="account not in registry")
            report.judge_errors += 1
            continue
        report.judged += 1
        try:
            verdict = judge_item(
                llm,
                model=model,
                account_name=entity.canonical_name,
                aliases=[a for a in entity.aliases if a != entity.canonical_name][:5],
                item_kind=item.item_kind,
                publisher=item.publisher or "a public source",
                url=item.url,
                dated=_dated_line(item),
                body=item.body,
                vendor_context=vendor_context,
            )
        except Exception as exc:
            _mark(conn, item_id, "error", reason=f"{type(exc).__name__}: {exc}"[:300], model=model)
            report.judge_errors += 1
            continue

        if verdict.status == "relevant":
            _source, _claim_ids, new_claims = write_public_signal(
                conn, store, registry, item, verdict, judge_model=model
            )
            report.relevant += 1
            report.claims_written += new_claims
            report.findings_dropped += len(verdict.dropped)
        else:
            _mark(
                conn,
                item_id,
                verdict.status,
                category=verdict.category,
                reason=verdict.reason,
                model=model,
            )
            if verdict.status == "rejected":
                report.rejected += 1
            else:
                report.not_relevant += 1
    report.judge_cost_usd = round(_cost_since(llm, start), 6)


def _record_coverage(conn: psycopg.Connection, report: ScanReport, per_kind: dict) -> None:
    day = clock.today()
    for connector, counts in per_kind.items():
        if not any(counts.values()):
            continue
        record_coverage(
            conn,
            connector,
            day,
            seen=counts["seen"],
            extracted=counts["extracted"],
            failed=counts["failed"],
            filtered={k: v for k, v in counts["filtered"].items() if v},
        )


def _verdict_counts(conn: psycopg.Connection, since: datetime) -> dict[str, dict[str, int]]:
    """Per item kind, how the items judged in this run came out."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT item_kind, status, count(*) FROM signal_items
            WHERE judged_at >= %s GROUP BY item_kind, status
            """,
            (since,),
        )
        rows = cur.fetchall()
    out: dict[str, dict[str, int]] = {}
    for kind, status, n in rows:
        out.setdefault(kind, {})[status] = n
    return out


def run_scan(
    conn: psycopg.Connection,
    *,
    settings: Settings,
    store: Store,
    llm: LLMClient,
    fetcher: PoliteFetcher,
    model: str | None = None,
    only: str | None = None,
    account: str | None = None,
    feed_days: int = DEFAULT_FEED_DAYS,
    judge_limit: int = DEFAULT_JUDGE_LIMIT,
    budget_usd: float | None = None,
    skip_judge: bool = False,
    browser_fetch: BrowserFetch | None = None,
    embedder: Embedder | None = None,
    now: datetime | None = None,
) -> ScanReport:
    """Run one scan (module docstring). `only` is `"feeds"` or `"pages"`;
    `account` limits the scan to one account's targets."""
    if not settings.external_signals_enabled:
        raise SignalsDisabled(SignalsDisabled.MESSAGE)
    started = clock.now()
    now = now or started
    report = ScanReport()
    registry = Registry.load(conn)
    judge_model = model or settings.external_signals_model

    targets = load_targets(conn)
    if account:
        targets = [t for t in targets if t.account_crm_id == account]
    if only == "feeds":
        targets = [t for t in targets if t.kind == "feed"]
    elif only == "pages":
        targets = [t for t in targets if t.kind == "page"]
    report.targets = len(targets)

    per_kind = {
        "signals_feed": {"seen": 0, "extracted": 0, "failed": 0, "filtered": {}},
        "signals_web": {"seen": 0, "extracted": 0, "failed": 0, "filtered": {}},
    }

    for target in targets:
        connector = "signals_feed" if target.kind == "feed" else "signals_web"
        entity = registry.get(target.account_crm_id)
        if entity is None or entity.kind != EntityKind.ACCOUNT:
            report.unknown_accounts += 1
            _set_target_status(conn, target, "account not in registry")
            continue
        if target.kind == "feed":
            report.feeds_checked += 1
            outcome = collect_feed_items(conn, target, fetcher, now=now, feed_days=feed_days)
            report.feed_items_new += outcome.new
            report.feed_items_too_old += outcome.too_old
            per_kind[connector]["seen"] += outcome.new
            error = outcome.error
            robots = bool(error and error.startswith("RobotsDisallowed"))
        else:
            report.pages_checked += 1
            page = snapshot_page(conn, store, target, fetcher, now=now, browser_fetch=browser_fetch)
            report.page_baselines += int(page.baseline)
            report.page_changes += int(page.item_id is not None)
            report.needs_browser += int(page.needs_browser)
            per_kind[connector]["seen"] += int(page.item_id is not None)
            if page.needs_browser:
                filtered = per_kind[connector]["filtered"]
                filtered["needs_browser"] = filtered.get("needs_browser", 0) + 1
            error = page.error
            robots = page.robots_disallowed
        if error:
            filtered = per_kind[connector]["filtered"]
            if robots:
                report.robots_disallowed += 1
                filtered["robots_disallowed"] = filtered.get("robots_disallowed", 0) + 1
            elif error.startswith("PersonalProfileRefused"):
                report.personal_profile_refused += 1
                filtered["personal_profile"] = filtered.get("personal_profile", 0) + 1
            else:
                report.fetch_errors += 1
                per_kind[connector]["failed"] += 1
            report.errors.append(f"target {target.id} ({target.url}): {error}"[:300])
        _set_target_status(conn, target, error)

    if not skip_judge:
        judge_pending(
            conn,
            store=store,
            llm=llm,
            registry=registry,
            model=judge_model,
            report=report,
            limit=judge_limit,
            budget_usd=budget_usd,
        )
        verdicts = _verdict_counts(conn, started)
        for kind, connector in (("feed_item", "signals_feed"), ("page_change", "signals_web")):
            counts = verdicts.get(kind, {})
            per_kind[connector]["extracted"] += counts.get("relevant", 0)
            per_kind[connector]["failed"] += counts.get("error", 0)
            filtered = per_kind[connector]["filtered"]
            for status in ("not_relevant", "rejected"):
                if counts.get(status):
                    filtered[status] = filtered.get(status, 0) + counts[status]

    _record_coverage(conn, report, per_kind)
    if embedder is not None:
        from portolan.search.index import index_pending

        index_pending(conn, embedder)
    append_changelog(
        conn,
        "external_signals_scan",
        (
            f"external signals scan: {report.targets} targets, {report.feed_items_new} new "
            f"feed items, {report.page_changes} page changes, {report.claims_written} claims"
        ),
        asdict(report),
    )
    return report
