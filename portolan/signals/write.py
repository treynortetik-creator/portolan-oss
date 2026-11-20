"""A relevant item becomes a source, one claim per accepted finding, and lines
on the account's public-signals note.

- **Source**: an `rss`-type source (the provenance chain's "RSS / web" row),
  whose stored original is a JSON record of exactly what the judge read: the
  feed item, or the page change with both snapshots' times and store keys. It
  is built only from the stored `signal_items` row, so a retry intakes the same
  bytes and `intake` returns the existing source.
- **Locator**: `url:<link> <date>`, the format `portolan.ingest.prepare`
  already gives RSS items, so "Open the original" lands on the block.
- **Claim**: fact type `public_signal` on `signal-public_signal-<account>`,
  a `signal` note linked from the account. The claim text says it is public,
  second-hand, and who published it: "Leadership change (public news, Becker's
  Hospital Review): ...". `speaker` is the publisher. A second item whose judge
  sentence and category match a live claim word for word (the same release in
  two languages, a syndicated copy) adds a `claim_citations` row instead of a
  second claim.

Public-signal claims are kept out of themes, waves and outliers (those are the
customer's own voice); the weekly report gives them their own section
(`portolan.report.public_signals`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime

import psycopg

from portolan.extract.schema import SourceType
from portolan.ingest.intake import intake
from portolan.ingest.models import RawSource
from portolan.merge.util import recorded_dt
from portolan.notes import (
    Claim,
    NoteRow,
    NoteStatus,
    NoteType,
    append_changelog,
    ensure_entity_note,
    link_related,
    note_exists,
    read_note,
    signal_note_slug,
    write_note,
)
from portolan.registry import Registry
from portolan.signals import PUBLIC_SIGNAL_FACT_TYPE
from portolan.signals.judge import CATEGORY_LABELS, Verdict
from portolan.signals.web import utc_stamp
from portolan.store import Store
from portolan.util import clock

__all__ = ["SignalItem", "load_item", "public_signal_note_slug", "write_public_signal"]


@dataclass(frozen=True)
class SignalItem:
    id: int
    target_id: int
    account_crm_id: str
    item_kind: str
    item_key: str
    url: str
    title: str | None
    publisher: str | None
    published_at: datetime | None
    observed_at: datetime
    body: str
    details: dict
    feed_kind: str | None


_ITEM_COLUMNS = (
    "si.id, si.target_id, si.account_crm_id, si.item_kind, si.item_key, si.url, si.title, "
    "si.publisher, si.published_at, si.observed_at, si.body, si.details, st.feed_kind"
)


def load_item(conn: psycopg.Connection, item_id: int) -> SignalItem:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {_ITEM_COLUMNS} FROM signal_items si "
            "JOIN signal_targets st ON st.id = si.target_id WHERE si.id = %s",
            (item_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"no signal item {item_id}")
    return SignalItem(*row[:11], details=row[11] or {}, feed_kind=row[12])


def public_signal_note_slug(account_slug: str) -> str:
    return signal_note_slug(PUBLIC_SIGNAL_FACT_TYPE, account_slug)


def _source_date(item: SignalItem) -> date:
    return clock.world_date(item.published_at or item.observed_at)


def _short_hash(value: str, n: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def _kind_label(item: SignalItem) -> str:
    if item.item_kind == "page_change":
        return "website change"
    if item.feed_kind == "linkedin":
        return "LinkedIn company post"
    if item.feed_kind == "news":
        return "news"
    return "feed item"


def _raw_source(item: SignalItem, account_slug: str) -> RawSource:
    source_date = _source_date(item)
    if item.item_kind == "page_change":
        stamp = utc_stamp(item.observed_at)
        source_id = f"signal-web-{account_slug}-{_short_hash(item.url, 8)}-{stamp}"
        record = {
            "signal_kind": "page_change",
            "title": f"{item.title or item.url}: website change",
            "link": item.url,
            "summary": item.body,
            "observed": item.observed_at.isoformat(),
            "snapshot": item.details.get("snapshot"),
            "previous_snapshot": item.details.get("previous_snapshot"),
            "changes": item.details.get("changes"),
        }
    else:
        source_id = f"signal-feed-{account_slug}-{_short_hash(item.item_key, 12)}"
        record = {
            "signal_kind": "feed_item",
            "title": item.title,
            "link": item.url,
            "summary": item.body,
            "publisher": item.publisher,
            "published": item.published_at.isoformat() if item.published_at else None,
            "observed": item.observed_at.isoformat(),
            "guid": item.item_key,
            "feed_url": item.details.get("feed_url"),
            "feed_kind": item.feed_kind,
        }
    return RawSource(
        source_id=source_id,
        source_type=SourceType.RSS,
        source_date=source_date,
        account_hints=[item.account_crm_id],
        text=item.body,
        original_bytes=json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
        original_content_type="application/json",
        metadata={
            "link": item.url,
            "title": item.title,
            "signal_kind": item.item_kind,
            "publisher": item.publisher,
            "signal_item_id": item.id,
        },
    )


def _ensure_public_signal_note(
    conn: psycopg.Connection, account: NoteRow, *, embedder=None
) -> NoteRow:
    slug = public_signal_note_slug(account.slug)
    if note_exists(conn, slug):
        note, _claims = read_note(conn, slug)
        return note
    now = clock.now()
    blurb = (
        f"Public, second-hand information about [[{account.slug}]]: news, posts on its "
        "LinkedIn company page (through RSS.app) and changes on its own website, each "
        "checked by a relevance judge. Not something the customer told us."
    )
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title=f"Public signals — {account.title}",
        description=f"Public signals for [[{account.slug}]].",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=blurb,
        related=[account.slug],
    )
    write_note(
        conn,
        note,
        [],
        changed_by="signals",
        reason=f"public signals note created for {account.slug}",
        embedder=embedder,
    )
    link_related(
        conn,
        account.slug,
        slug,
        changed_by="signals",
        reason="public signals note",
        embedder=embedder,
    )
    return note


def write_public_signal(
    conn: psycopg.Connection,
    store: Store,
    registry: Registry,
    item: SignalItem,
    verdict: Verdict,
    *,
    judge_model: str,
    embedder=None,
) -> tuple[str, list[int], int]:
    """Write `item`'s source and one claim per accepted finding; mark the item
    `relevant`. Returns `(source_slug, claim_ids, new_claims)`: a finding that
    repeats a live claim adds a citation to it instead of a new claim."""
    assert verdict.status == "relevant" and verdict.findings
    account = ensure_entity_note(conn, registry, item.account_crm_id)
    source = intake(_raw_source(item, account.slug), conn=conn, store=store)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sources SET extraction_status = 'done', summary = %s, "
            "extraction_model = %s WHERE slug = %s",
            (verdict.findings[0].claim[:500], judge_model, source.slug),
        )
    conn.commit()

    note = _ensure_public_signal_note(conn, account, embedder=embedder)
    publisher = item.publisher or "public source"
    locator = f"url:{item.url} {source.source_date.isoformat()}"
    claim_ids: list[int] = []
    new_claims = 0
    with conn.cursor() as cur:
        for finding in verdict.findings:
            duplicate_of = _same_claim(conn, note.slug, finding.category, finding.claim)
            if duplicate_of is not None:
                # The same event from a second item (a translated or syndicated copy):
                # one claim, one more citation, as merge does for a `same` verdict.
                claim_id = duplicate_of
                cur.execute(
                    """
                    INSERT INTO claim_citations (claim_id, source_slug, locator, quote)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (claim_id, source_slug, locator) DO NOTHING
                    """,
                    (claim_id, source.slug, locator, finding.quote),
                )
            else:
                claim_id = _insert_claim(
                    cur,
                    Claim(
                        note_slug=note.slug,
                        fact_type=PUBLIC_SIGNAL_FACT_TYPE,
                        claim_text=(
                            f"{CATEGORY_LABELS[finding.category]} "
                            f"(public {_kind_label(item)}, {publisher}): {finding.claim}"
                        ),
                        source_slug=source.slug,
                        locator=locator,
                        quote=finding.quote,
                        speaker=publisher[:200],
                        recorded=recorded_dt(source),
                        confidence=round(finding.confidence, 3),
                    ),
                )
                new_claims += 1
            if claim_id not in claim_ids:
                claim_ids.append(claim_id)
                cur.execute(
                    "INSERT INTO signal_item_claims (item_id, claim_id, category) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (item.id, claim_id, finding.category),
                )
        cur.execute(
            """
            UPDATE signal_items SET status = 'relevant', category = %s, judge_reason = %s,
                   judge_model = %s, judged_at = now(), source_slug = %s,
                   attempts = attempts + 1
            WHERE id = %s
            """,
            (verdict.category, verdict.reason, judge_model, source.slug, item.id),
        )
    conn.commit()

    fresh, claims = read_note(conn, note.slug)
    write_note(
        conn,
        fresh,
        claims,
        changed_by="signals",
        reason=f"public signals from item {item.id}",
        embedder=embedder,
    )
    append_changelog(
        conn,
        "public_signal",
        f"public signals on {account.slug}: "
        + ", ".join(CATEGORY_LABELS[f.category] for f in verdict.findings),
        {
            "claim_ids": claim_ids,
            "new_claims": new_claims,
            "source_slug": source.slug,
            "signal_item_id": item.id,
            "categories": [f.category for f in verdict.findings],
            "dropped_findings": verdict.dropped,
            "url": item.url,
        },
    )
    return source.slug, claim_ids, new_claims


def _insert_claim(cur: psycopg.Cursor, claim: Claim) -> int:
    cur.execute(
        """
        INSERT INTO claims (note_slug, fact_type, claim_text, source_slug, locator,
                            quote, speaker, recorded, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            claim.recorded,
            claim.confidence,
        ),
    )
    return cur.fetchone()[0]


_WORDS_RE = re.compile(r"[a-z0-9]+")


def _sentence_key(sentence: str) -> str:
    return " ".join(_WORDS_RE.findall(sentence.lower()))


def _same_claim(
    conn: psycopg.Connection, note_slug: str, category: str, sentence: str
) -> int | None:
    """A live claim on `note_slug` with the same category label and, word for
    word, the same judge sentence; `None` if there is none."""
    prefix = f"{CATEGORY_LABELS[category]} (public "
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, claim_text FROM claims WHERE note_slug = %s AND fact_type = %s "
            "AND superseded_at IS NULL AND archived = false AND claim_text LIKE %s ORDER BY id",
            (note_slug, PUBLIC_SIGNAL_FACT_TYPE, prefix.replace("%", r"\%") + "%"),
        )
        rows = cur.fetchall()
    wanted = _sentence_key(sentence)
    for claim_id, claim_text in rows:
        _label, _sep, existing = claim_text.partition("): ")
        if _sentence_key(existing) == wanted:
            return claim_id
    return None
