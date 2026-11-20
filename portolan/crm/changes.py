"""CRM changes become claims through code, not a model
(`01-spec/red-team-2026-09-24.md` sec 1; `01-spec/monday-email-2026-09-25.md`).

`compute_changes` compares each CRM record's snapshots (`portolan.crm.
snapshots`): for every tracked field, a snapshot's value against the most
recent earlier snapshot that carries that field (an export without the
column says nothing about it). It also finds opportunities and contacts that
first appear after an earlier export of their kind, and contacts that drop out
of the next contacts export for an account still in it.

`refresh_crm_changes` writes each change it has not written before as one
`crm_change` claim on the account's `signal-crm_change-<account>` note: the
sentence comes from a template, every number in it from this module, the
primary citation is the newer CRM row and a second citation is the older one.
The change itself, with its old and new values, goes in `crm_changes`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import psycopg
from psycopg.types.json import Jsonb

from portolan.crm.snapshots import Snapshot, load_snapshots
from portolan.merge.facts import _insert_claim, _insert_claim_citation
from portolan.notes import (
    Claim,
    NoteRow,
    NoteStatus,
    NoteType,
    ensure_entity_note,
    link_related,
    note_exists,
    read_note,
    signal_note_slug,
    write_note,
)
from portolan.registry import Registry
from portolan.search.embed import Embedder
from portolan.util import clock

__all__ = [
    "CRM_CHANGE_FACT_TYPE",
    "ChangeRow",
    "ChangeSpec",
    "FieldRule",
    "Names",
    "RefreshReport",
    "change_sentence",
    "compute_changes",
    "load_changes",
    "refresh_crm_changes",
]

CRM_CHANGE_FACT_TYPE = "crm_change"


@dataclass(frozen=True)
class FieldRule:
    column: str
    label: str
    kind: str  # stage | money | date | owner | health | forecast | text
    points: int


# The columns compared, per object, and how much a change to each weighs in
# the Monday email's ranking. A column missing from an export is skipped.
RULES: dict[str, tuple[FieldRule, ...]] = {
    "opportunity": (
        FieldRule("stage", "Stage", "stage", 5),
        FieldRule("amount", "Amount", "money", 4),
        FieldRule("close_date", "Close date", "date", 3),
        FieldRule("owner_crm_id", "Owner", "owner", 3),
        FieldRule("health", "Health", "health", 4),
        FieldRule("forecast_category", "Forecast category", "forecast", 2),
        FieldRule("next_step", "Next step", "text", 1),
    ),
    "account": (
        FieldRule("owner_crm_id", "Owner", "owner", 3),
        FieldRule("health", "Health", "health", 4),
        FieldRule("arr", "ARR", "money", 3),
        FieldRule("status", "Status", "text", 2),
    ),
    "contact": (
        FieldRule("title", "Title", "text", 2),
        FieldRule("status", "Status", "text", 1),
    ),
}
ADDED_POINTS = {"opportunity": 4, "contact": 2}
REMOVED_POINTS = {"contact": 2}
RECORD_FIELD = "__record__"

# Ordered from furthest from a close to a win; `closed_lost` is scored apart.
STAGE_ORDER = (
    "prospecting",
    "qualification",
    "discovery",
    "proposal",
    "negotiation",
    "commit",
    "closed_won",
)
HEALTH_ORDER = ("red", "yellow", "green")
HEALTH_ALIASES = {"amber": "yellow", "at_risk": "red"}
FORECAST_ORDER = ("omitted", "pipeline", "best_case", "commit", "closed")

_MONEY_JUNK_RE = re.compile(r"[,$\s]")


def _key(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip().lower())


def _money(value: str) -> Decimal | None:
    cleaned = _MONEY_JUNK_RE.sub("", value or "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _day(value: str) -> date | None:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def _same(kind: str, old: str, new: str) -> bool:
    if kind == "money":
        a, b = _money(old), _money(new)
        if a is not None and b is not None:
            return a == b
    if kind in ("stage", "health", "forecast", "owner"):
        return _key(old) == _key(new)
    return " ".join(old.split()) == " ".join(new.split())


def _rank(order: tuple[str, ...], value: str, aliases: dict[str, str] | None = None) -> int | None:
    k = _key(value)
    if aliases:
        k = aliases.get(k, k)
    return order.index(k) if k in order else None


@dataclass(frozen=True)
class ChangeSpec:
    object: str
    crm_id: str
    account_crm_id: str
    field: str
    kind: str  # changed | added | removed
    old_value: str | None
    new_value: str | None
    as_of: date
    prev_as_of: date | None
    snapshot: Snapshot
    prev: Snapshot | None
    direction: int
    points: int
    details: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    value_kind: str = "text"

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (self.object, self.crm_id, self.field, self.kind, self.snapshot.id)


def _field_change(
    rule: FieldRule, snap: Snapshot, prev: Snapshot, old: str, new: str
) -> ChangeSpec:
    direction = 0
    points = rule.points
    details: dict[str, Any] = {}
    if rule.kind == "stage":
        if _key(new) == "closed_lost":
            direction, points = -1, points + 3
        else:
            a, b = _rank(STAGE_ORDER, old), _rank(STAGE_ORDER, new)
            if a is not None and b is not None and a != b:
                direction = 1 if b > a else -1
                if direction < 0:
                    points += 2
    elif rule.kind == "money":
        a, b = _money(old), _money(new)
        if a is not None and b is not None:
            delta = b - a
            direction = (delta > 0) - (delta < 0)
            details = {"old": str(a), "new": str(b), "delta": str(delta)}
            if a != 0:
                pct = (abs(delta) / abs(a) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                details["pct"] = str(pct)
                points += 2 if pct >= 20 else 1 if pct >= 10 else 0
    elif rule.kind == "date":
        a, b = _day(old), _day(new)
        if a is not None and b is not None:
            days = (b - a).days
            details = {"days": days}
            # A later close date is a slip; an earlier one is pulled in.
            direction = -1 if days > 0 else 1 if days < 0 else 0
            if abs(days) > 30:
                points += 2
    elif rule.kind == "health":
        a, b = _rank(HEALTH_ORDER, old, HEALTH_ALIASES), _rank(HEALTH_ORDER, new, HEALTH_ALIASES)
        if a is not None and b is not None and a != b:
            direction = 1 if b > a else -1
        if b == 0:
            points += 2
    elif rule.kind == "forecast":
        a, b = _rank(FORECAST_ORDER, old), _rank(FORECAST_ORDER, new)
        if a is not None and b is not None and a != b:
            direction = 1 if b > a else -1
    return ChangeSpec(
        object=snap.object,
        crm_id=snap.crm_id,
        account_crm_id=snap.account_crm_id or prev.account_crm_id or "",
        field=rule.column,
        kind="changed",
        old_value=old,
        new_value=new,
        as_of=snap.as_of,
        prev_as_of=prev.as_of,
        snapshot=snap,
        prev=prev,
        direction=direction,
        points=points,
        details=details,
        label=rule.label,
        value_kind=rule.kind,
    )


def compute_changes(snapshots: list[Snapshot]) -> list[ChangeSpec]:
    """Every change the snapshots show (module docstring), oldest first.
    Pure: no database."""
    ordered = sorted(snapshots, key=lambda s: (s.object, s.crm_id, s.as_of, s.id))
    changes: list[ChangeSpec] = []

    by_record: dict[tuple[str, str], list[Snapshot]] = defaultdict(list)
    for snap in ordered:
        by_record[(snap.object, snap.crm_id)].append(snap)

    for (obj, _crm_id), snaps in by_record.items():
        rules = RULES.get(obj, ())
        last_seen: dict[str, tuple[str, Snapshot]] = {}
        for snap in snaps:
            for rule in rules:
                if rule.column not in snap.fields:
                    continue
                new = snap.fields.get(rule.column) or ""
                if rule.column in last_seen:
                    old, prev = last_seen[rule.column]
                    if not _same(rule.kind, old, new) and (old.strip() or new.strip()):
                        changes.append(_field_change(rule, snap, prev, old, new))
                last_seen[rule.column] = (new, snap)

    exports_by_object: dict[str, dict[date, list[Snapshot]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for snap in ordered:
        exports_by_object[snap.object][snap.as_of].append(snap)

    for obj, points in ADDED_POINTS.items():
        dates = sorted(exports_by_object.get(obj, {}))
        if not dates:
            continue
        first_export = dates[0]
        for (record_obj, _crm_id), snaps in by_record.items():
            if record_obj != obj:
                continue
            first = snaps[0]
            if first.as_of > first_export and first.account_crm_id:
                changes.append(
                    ChangeSpec(
                        object=obj,
                        crm_id=first.crm_id,
                        account_crm_id=first.account_crm_id,
                        field=RECORD_FIELD,
                        kind="added",
                        old_value=None,
                        new_value=first.crm_id,
                        as_of=first.as_of,
                        prev_as_of=None,
                        snapshot=first,
                        prev=None,
                        direction=1 if obj == "opportunity" else 0,
                        points=points,
                        label="Record",
                    )
                )

    for obj, points in REMOVED_POINTS.items():
        exports = exports_by_object.get(obj, {})
        dates = sorted(exports)
        for prev_date, cur_date in pairwise(dates):
            current_ids = {s.crm_id for s in exports[cur_date]}
            current_accounts = {s.account_crm_id for s in exports[cur_date]}
            for prev in exports[prev_date]:
                # Only an account the newer export still covers: a partial
                # export is not a removal.
                if prev.crm_id in current_ids or prev.account_crm_id not in current_accounts:
                    continue
                if not prev.account_crm_id:
                    continue
                changes.append(
                    ChangeSpec(
                        object=obj,
                        crm_id=prev.crm_id,
                        account_crm_id=prev.account_crm_id,
                        field=RECORD_FIELD,
                        kind="removed",
                        old_value=prev.crm_id,
                        new_value=None,
                        as_of=cur_date,
                        prev_as_of=prev_date,
                        snapshot=prev,
                        prev=None,
                        direction=-1,
                        points=points,
                        label="Record",
                    )
                )

    changes.sort(key=lambda c: (c.as_of, c.account_crm_id, c.object, c.crm_id, c.field))
    return changes


# --------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------


@dataclass
class Names:
    """Display names by CRM id, from the registry (accounts, contacts,
    internal people). An id with no registry row shows as itself."""

    by_crm_id: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, conn: psycopg.Connection) -> Names:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT crm_id, canonical_name FROM registry_entities WHERE crm_id IS NOT NULL"
            )
            return cls(by_crm_id={r[0]: r[1] for r in cur.fetchall()})

    def name(self, crm_id: str | None) -> str:
        if not crm_id:
            return ""
        return self.by_crm_id.get(crm_id, crm_id)


def pretty(value: str) -> str:
    """`closed_lost` -> `Closed Lost`: a CRM picklist value as a reader sees it."""
    words = re.split(r"[_\s]+", value.strip())
    return " ".join(w[:1].upper() + w[1:] for w in words if w)


def money_text(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def _display(kind: str, value: str, names: Names) -> str:
    if kind in ("stage", "health", "forecast"):
        return pretty(value)
    if kind == "money":
        amount = _money(value)
        return money_text(amount) if amount is not None else value
    if kind == "owner":
        return names.name(value)
    if kind == "text":
        return f'"{value.strip()}"'
    return value.strip()


def _subject(spec: ChangeSpec, names: Names) -> str:
    fields = spec.snapshot.fields
    if spec.object == "opportunity":
        return fields.get("name") or spec.crm_id
    if spec.object == "contact":
        who = fields.get("name") or names.name(spec.crm_id)
        return f"{who} at {names.name(spec.account_crm_id)}"
    return names.name(spec.crm_id) or fields.get("name") or spec.crm_id


def change_sentence(spec: ChangeSpec, names: Names) -> str:
    """The claim text for one change. Every number in it is computed here."""
    stamp = f"(CRM, {spec.as_of.isoformat()})"
    account = names.name(spec.account_crm_id)
    fields = spec.snapshot.fields

    if spec.kind == "added" and spec.object == "opportunity":
        parts = [f"{account}: new opportunity {fields.get('name') or spec.crm_id}"]
        if fields.get("stage"):
            parts.append(f"at stage {pretty(fields['stage'])}")
        amount = _money(fields.get("amount") or "")
        if amount is not None:
            parts.append(f"amount {money_text(amount)}")
        if fields.get("close_date"):
            parts.append(f"close date {fields['close_date'].strip()}")
        return ", ".join(parts) + f" {stamp}."
    if spec.kind == "added":
        who = ", ".join(x for x in (fields.get("name"), fields.get("title")) if x)
        return f"{account}: contact added, {who or spec.crm_id} {stamp}."
    if spec.kind == "removed":
        who = ", ".join(x for x in (fields.get("name"), fields.get("title")) if x)
        return (
            f"{account}: contact removed, {who or spec.crm_id}; not in the "
            f"{spec.as_of.isoformat()} contacts export {stamp}."
        )

    subject = _subject(spec, names)
    label = spec.label.lower() if spec.label not in ("ARR",) else spec.label
    old = spec.old_value or ""
    new = spec.new_value or ""
    if not old.strip():
        return f"{subject}: {label} set to {_display(spec.value_kind, new, names)} {stamp}."
    if not new.strip():
        return f"{subject}: {label} cleared, was {_display(spec.value_kind, old, names)} {stamp}."
    old_text = _display(spec.value_kind, old, names)
    new_text = _display(spec.value_kind, new, names)
    if spec.value_kind == "money" and "delta" in spec.details:
        delta = Decimal(spec.details["delta"])
        verb, word = ("rose", "up") if delta > 0 else ("fell", "down")
        pct = f" or {spec.details['pct']}%" if "pct" in spec.details else ""
        return (
            f"{subject}: {label} {verb} from {old_text} to {new_text}, {word} "
            f"{money_text(abs(delta))}{pct} {stamp}."
        )
    if spec.value_kind == "date" and "days" in spec.details:
        days = spec.details["days"]
        when = "later" if days > 0 else "earlier"
        return (
            f"{subject}: {label} moved from {old_text} to {new_text}, {abs(days)} days {when} "
            f"{stamp}."
        )
    if spec.value_kind in ("stage", "health", "forecast"):
        return f"{subject}: {label} moved from {old_text} to {new_text} {stamp}."
    return f"{subject}: {label} changed from {old_text} to {new_text} {stamp}."


def _quote(spec: ChangeSpec, *, previous: bool) -> str:
    """A `column=value` span that appears on the cited row's original page
    (`portolan.web.routes.originals._crm_view` shows one `key=value` per field)."""
    if spec.field == RECORD_FIELD:
        id_column = "opportunity_id" if spec.object == "opportunity" else "crm_id"
        return f"{id_column}={spec.crm_id}"
    value = spec.old_value if previous else spec.new_value
    return f"{spec.field}={value or ''}"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@dataclass
class RefreshReport:
    changes_written: int = 0
    already_written: int = 0
    unknown_accounts: list[str] = field(default_factory=list)
    notes_touched: list[str] = field(default_factory=list)


def _severity(points: int) -> str:
    return "high" if points >= 6 else "medium" if points >= 3 else "low"


def _recorded(as_of: date) -> datetime:
    # Same anchor as `portolan.merge.util.recorded_dt`: midnight in the deployment zone.
    return datetime.combine(as_of, time.min, tzinfo=ZoneInfo(clock.timezone_name()))


def _existing_keys(conn: psycopg.Connection) -> set[tuple[str, str, str, str, int]]:
    with conn.cursor() as cur:
        cur.execute("SELECT object, crm_id, field, kind, snapshot_id FROM crm_changes")
        return {tuple(r) for r in cur.fetchall()}  # type: ignore[misc]


def _ensure_crm_note(
    conn: psycopg.Connection, account_note: NoteRow, touched: list[str]
) -> NoteRow:
    slug = signal_note_slug(CRM_CHANGE_FACT_TYPE, account_note.slug)
    if note_exists(conn, slug):
        note, _claims = read_note(conn, slug)
        return note
    now = clock.now()
    note = NoteRow(
        slug=slug,
        type=NoteType.SIGNAL,
        title=f"CRM changes — {account_note.title}",
        description=f"Field changes between CRM exports for [[{account_note.slug}]].",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        related=[account_note.slug],
    )
    write_note(conn, note, [], changed_by="crm", reason="CRM change note created", commit=False)
    if link_related(
        conn, account_note.slug, slug, changed_by="crm", reason="CRM changes", commit=False
    ):
        touched.append(account_note.slug)
    return note


def refresh_crm_changes(
    conn: psycopg.Connection,
    registry: Registry,
    *,
    embedder: Embedder | None = None,
) -> RefreshReport:
    """Write every change `compute_changes` finds that `crm_changes` does not
    hold yet (module docstring). An account with no registry entry is skipped
    and named in the report. One transaction per account; commits."""
    report = RefreshReport()
    specs = compute_changes(load_snapshots(conn))
    done = _existing_keys(conn)
    names = Names.load(conn)
    # The reads above open a transaction on a non-autocommit connection, which
    # would turn each `conn.transaction()` below into a savepoint.
    conn.commit()
    fresh = [s for s in specs if s.key not in done]
    report.already_written = len(specs) - len(fresh)
    if not fresh:
        return report

    by_account: dict[str, list[ChangeSpec]] = defaultdict(list)
    for spec in fresh:
        by_account[spec.account_crm_id].append(spec)

    for account_crm_id, account_specs in sorted(by_account.items()):
        if not account_crm_id or registry.get(account_crm_id) is None:
            report.unknown_accounts.append(account_crm_id or "(none)")
            continue
        touched: list[str] = []
        with conn.transaction():
            account_note = ensure_entity_note(conn, registry, account_crm_id, commit=False)
            crm_note = _ensure_crm_note(conn, account_note, touched)
            for spec in account_specs:
                primary = spec.snapshot
                claim = Claim(
                    note_slug=crm_note.slug,
                    fact_type=CRM_CHANGE_FACT_TYPE,
                    claim_text=change_sentence(spec, names),
                    source_slug=primary.source_slug,
                    locator=primary.locator,
                    quote=_quote(spec, previous=spec.kind == "removed"),
                    valid_from=spec.as_of,
                    recorded=_recorded(spec.as_of),
                    severity=_severity(spec.points),
                    confidence=1.0,
                )
                claim_id = _insert_claim(conn, claim, commit=False)
                if spec.prev is not None and (
                    spec.prev.source_slug,
                    spec.prev.locator,
                ) != (primary.source_slug, primary.locator):
                    _insert_claim_citation(
                        conn,
                        claim_id=claim_id,
                        source_slug=spec.prev.source_slug,
                        locator=spec.prev.locator,
                        quote=_quote(spec, previous=True),
                        commit=False,
                    )
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO crm_changes
                            (object, crm_id, account_crm_id, field, kind, old_value, new_value,
                             as_of, prev_as_of, snapshot_id, prev_snapshot_id, direction, points,
                             details, claim_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            spec.object,
                            spec.crm_id,
                            spec.account_crm_id,
                            spec.field,
                            spec.kind,
                            spec.old_value,
                            spec.new_value,
                            spec.as_of,
                            spec.prev_as_of,
                            spec.snapshot.id,
                            spec.prev.id if spec.prev is not None else None,
                            spec.direction,
                            spec.points,
                            Jsonb(spec.details),
                            claim_id,
                        ),
                    )
                report.changes_written += 1
            note, claims = read_note(conn, crm_note.slug)
            write_note(
                conn,
                note,
                claims,
                changed_by="crm",
                reason=f"{len(account_specs)} CRM change(s)",
                embedder=embedder,
                commit=False,
            )
        report.notes_touched.extend([crm_note.slug, *touched])
    conn.commit()
    return report


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeRow:
    """One written change with its claim and both citations, for the email."""

    id: int
    object: str
    crm_id: str
    account_crm_id: str
    field: str
    kind: str
    as_of: date
    prev_as_of: date | None
    direction: int
    points: int
    claim_id: int
    claim_text: str
    source_slug: str
    locator: str
    quote: str | None
    prev_source_slug: str | None
    prev_locator: str | None
    prev_quote: str | None


def load_changes(conn: psycopg.Connection, *, start: date, end: date) -> list[ChangeRow]:
    """Changes whose newer export date falls in `[start, end]`, oldest first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ch.id, ch.object, ch.crm_id, ch.account_crm_id, ch.field, ch.kind, ch.as_of,
                   ch.prev_as_of, ch.direction, ch.points, c.id, c.claim_text, c.source_slug,
                   c.locator, c.quote, prev.source_slug, prev.locator, cc.quote
            FROM crm_changes ch
            JOIN claims c ON c.id = ch.claim_id
            LEFT JOIN crm_snapshots prev ON prev.id = ch.prev_snapshot_id
            LEFT JOIN claim_citations cc ON cc.claim_id = c.id
                AND cc.source_slug = prev.source_slug AND cc.locator = prev.locator
            WHERE ch.as_of BETWEEN %s AND %s AND NOT c.archived
            ORDER BY ch.as_of, ch.id
            """,
            (start, end),
        )
        rows = cur.fetchall()
    return [ChangeRow(*r) for r in rows]
