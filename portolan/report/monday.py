"""The Monday email: what changed this week on one person's top accounts, from
CRM snapshots and call transcripts, every sentence linked to the row or the
call moment it came from (`01-spec/monday-email-2026-09-25.md`;
`01-spec/red-team-2026-09-24.md` sec 4, change 2).

Per account it shows only the week's changes: CRM changes (`portolan.crm.
changes`), CRM-vs-call checks (`portolan.crm.truth`), new call claims, and
actions that went overdue or were opened. Accounts are ranked by a points
score computed here (`CALL_POINTS` and friends, plus each CRM change's own
points); accounts with nothing are one line at the bottom. An optional
two-sentence summary per account comes from a model, kept only when every
sentence cites lines it was given and states no number they don't hold.

CLI: `python -m portolan.report monday --owner <person> --week YYYY-Www
--out DIR [--top N | --accounts A,B] [--llm openrouter --max-usd 1] [--send
--to ADDRESS]`. Writes `monday-<owner>-<week>.html`, `.txt` and `.json`;
sends nothing unless `--send` and `PORTOLAN_SMTP_URL` are both given.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import smtplib
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import psycopg

from portolan.crm.changes import (
    RECORD_FIELD,
    RULES,
    ChangeRow,
    compute_changes,
    load_changes,
    money_text,
)
from portolan.crm.snapshots import Snapshot, load_snapshots
from portolan.crm.truth import (
    LOOKBACK_DAYS,
    CallClaim,
    TruthFlag,
    account_families,
    call_claims,
    truth_flags,
)
from portolan.llm.base import LLMClient
from portolan.notes.render import WIKILINK_RE
from portolan.report.numbers import NumberFacts, check_numerals
from portolan.report.util import citation_markers_in, validate_citations, week_bounds
from portolan.util import clock

__all__ = [
    "DEFAULT_TOP_N",
    "SPEND_KEY",
    "AccountBlock",
    "Cite",
    "MondayEmail",
    "SendRefused",
    "Sentence",
    "add_parser",
    "brand_tokens",
    "build_monday",
    "render_html",
    "render_text",
    "run_cli",
    "send_email",
    "to_json",
]

DEFAULT_TOP_N = 25
# Points per item, for ranking accounts by the size of the week's change.
CALL_POINTS = {"risk": 4, "complaint": 2, "competitor_mention": 2, "pricing": 2}
CALL_POINTS_DEFAULT = 1
TRUTH_POINTS = 6
OVERDUE_POINTS = 3
NEW_ACTION_POINTS = 1
SUMMARY_MAX_SENTENCES = 2

BRAND_TOKENS_CSS = Path(__file__).resolve().parents[2] / "04-brand" / "brand-tokens.css"


@dataclass(frozen=True)
class Cite:
    source_slug: str
    locator: str
    label: str
    # The span the cited original holds, for a link check (`scripts/verify.py` `quote_match`).
    quote: str | None = None

    @property
    def marker(self) -> str:
        return f"^[{self.source_slug} @ {self.locator}]"

    def href(self, base_url: str) -> str:
        return (
            f"{base_url.rstrip('/')}/originals/{quote(self.source_slug, safe='')}"
            f"?at={quote(self.locator, safe='')}"
        )


@dataclass
class Sentence:
    text: str
    cites: list[Cite]


@dataclass
class Item:
    kind: str  # truth | crm | call | action
    tone: str  # risk | forward | ""
    sentences: list[Sentence]


@dataclass
class AccountBlock:
    crm_id: str
    name: str
    note_slug: str | None
    amount: Decimal
    amount_basis: str  # "open pipeline" | "ARR" | ""
    score: int = 0
    items: list[Item] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


@dataclass
class MondayEmail:
    owner_crm_id: str | None
    owner_name: str
    week: str
    start: date
    end: date
    base_url: str
    accounts: list[AccountBlock]
    quiet: list[AccountBlock]
    unwritten_crm_changes: int = 0
    summary_notes: list[str] = field(default_factory=list)

    @property
    def subject(self) -> str:
        return f"What changed on your accounts, week of {self.start.isoformat()} ({self.week})"

    def overview(self) -> str:
        """The count line at the top; every count is of the lines below it."""
        c = self.counts()
        total = len(self.accounts) + len(self.quiet)
        return (
            f"{len(self.accounts)} of {total} accounts changed: {_n(c['crm'], 'CRM change')}, "
            f"{_n(c['truth'], 'CRM-versus-call check')}, {_n(c['call'], 'call moment')}, "
            f"{_n(c['action'], 'action')}."
        )

    def counts(self) -> dict[str, int]:
        counts = {"truth": 0, "crm": 0, "call": 0, "action": 0}
        for block in self.accounts:
            for item in block.items:
                counts[item.kind] += 1
        return counts


def _n(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


# --------------------------------------------------------------------------
# Reading the graph
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Account:
    crm_id: str
    name: str
    parent_crm_id: str | None
    note_slug: str | None
    registry_owner: str | None


def _accounts(conn: psycopg.Connection) -> dict[str, _Account]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.crm_id, a.canonical_name, p.crm_id, a.note_slug, o.crm_id
            FROM registry_entities a
            LEFT JOIN registry_entities p ON p.id = a.parent_id
            LEFT JOIN registry_entities o ON o.id = a.owner_entity_id
            WHERE a.kind = 'account' AND a.crm_id IS NOT NULL AND a.active
            """
        )
        return {r[0]: _Account(*r) for r in cur.fetchall()}


def _people(conn: psycopg.Connection) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, canonical_name FROM registry_entities "
            "WHERE kind = 'internal_person' AND crm_id IS NOT NULL"
        )
        return {r[0]: r[1] for r in cur.fetchall()}


def resolve_owner(conn: psycopg.Connection, owner: str) -> tuple[str, str]:
    """`(crm_id, display name)` for an internal person named by id or name."""
    people = _people(conn)
    wanted = owner.strip().lower()
    for crm_id, name in people.items():
        if wanted in (crm_id.lower(), name.lower()):
            return crm_id, name
    known = ", ".join(sorted(people)) or "none"
    raise ValueError(f"no internal person {owner!r} in the registry (known: {known})")


def _latest(snapshots: list[Snapshot]) -> dict[tuple[str, str], Snapshot]:
    latest: dict[tuple[str, str], Snapshot] = {}
    for snap in snapshots:  # ordered by as_of, id
        latest[(snap.object, snap.crm_id)] = snap
    return latest


def _owner_by_account(accounts: dict[str, _Account], latest: dict) -> dict[str, str | None]:
    """The CRM's owner as of the week's end, else the registry's."""
    owners: dict[str, str | None] = {}
    for crm_id, account in accounts.items():
        snap = latest.get(("account", crm_id))
        from_crm = (snap.fields.get("owner_crm_id") or "").strip() if snap else ""
        owners[crm_id] = from_crm or account.registry_owner
    return owners


def _amounts(accounts: dict[str, _Account], latest: dict) -> dict[str, tuple[Decimal, str]]:
    """Open pipeline (sum of open opportunity amounts) as of the week's end,
    else the account's ARR from its latest CRM snapshot."""
    pipeline: dict[str, Decimal] = defaultdict(Decimal)
    for (obj, _crm_id), snap in latest.items():
        if obj != "opportunity" or not snap.account_crm_id:
            continue
        if (snap.fields.get("stage") or "").strip().lower().startswith("closed"):
            continue
        amount = _parse_money(snap.fields.get("amount") or "")
        if amount:
            pipeline[snap.account_crm_id] += amount
    out: dict[str, tuple[Decimal, str]] = {}
    for crm_id in accounts:
        if pipeline.get(crm_id):
            out[crm_id] = (pipeline[crm_id], "open pipeline")
            continue
        snap = latest.get(("account", crm_id))
        arr = _parse_money(snap.fields.get("arr") or "") if snap else None
        out[crm_id] = (arr, "ARR") if arr else (Decimal(0), "")
    return out


def _parse_money(value: str) -> Decimal | None:
    cleaned = re.sub(r"[,$\s]", "", value)
    try:
        return Decimal(cleaned) if cleaned else None
    except ArithmeticError:
        return None


@dataclass(frozen=True)
class _Action:
    id: int
    note_slug: str
    what: str
    due: date | None
    status: str
    source_slug: str
    locator: str | None
    source_type: str | None
    source_date: date | None
    created: date | None
    account_slug: str | None


def _actions(conn: psycopg.Connection) -> list[_Action]:
    day_sql = clock.world_date_sql("c.recorded")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT a.id, a.note_slug, a.what, a.due, a.status, a.source_slug, a.locator,
                   s.source_type, s.source_date,
                   (SELECT min({day_sql}) FROM claims c WHERE c.note_slug = a.note_slug),
                   (SELECT l.to_slug FROM links l JOIN notes t ON t.slug = l.to_slug
                    WHERE l.from_slug = a.note_slug AND t.type = 'account'
                    ORDER BY l.to_slug LIMIT 1)
            FROM actions a LEFT JOIN sources s ON s.slug = a.source_slug
            WHERE a.status IN ('open', 'overdue')
            ORDER BY a.id
            """
        )
        return [_Action(*r) for r in cur.fetchall()]


def _source_label(source_type: str | None, source_date: date | None, locator: str | None) -> str:
    when = source_date.isoformat() if source_date else ""
    if source_type == "call":
        return f"call {when} at {locator}".strip()
    if source_type == "crm":
        return f"CRM {when}".strip()
    return f"{source_type or 'source'} {when}".strip()


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _crm_item(change: ChangeRow) -> Item:
    cites = [_change_cite(change)]
    if change.prev_source_slug and change.prev_locator and change.prev_as_of:
        label = f"was {change.prev_as_of.isoformat()}"
        cites.append(Cite(change.prev_source_slug, change.prev_locator, label, change.prev_quote))
    tone = "risk" if change.direction < 0 else "forward" if change.direction > 0 else ""
    return Item("crm", tone, [Sentence(change.claim_text, cites)])


def _change_cite(change: ChangeRow) -> Cite:
    return Cite(change.source_slug, change.locator, f"CRM {change.as_of.isoformat()}", change.quote)


def _call_cite(call: CallClaim) -> Cite:
    label = f"call {call.day.isoformat()} at {call.locator}"
    return Cite(call.source_slug, call.locator, label, call.quote)


_FIELD_WORDS = {
    **{rule.column: rule.label.lower() for rules in RULES.values() for rule in rules},
    RECORD_FIELD: "new record",
    "arr": "ARR",
}


def _truth_items(flags: list[TruthFlag], titles: dict[str, str]) -> list[Item]:
    """One item per contradicting call claim: which fields the CRM moved
    forward (the full changes are listed under "CRM"), then the call."""
    grouped: dict[int, list[TruthFlag]] = defaultdict(list)
    for flag in flags:
        grouped[flag.call.claim_id].append(flag)
    items = []
    for group in grouped.values():
        call = group[0].call
        by_subject: dict[str, list[str]] = defaultdict(list)
        cites: dict[tuple[str, str], Cite] = {}
        for flag in group:
            change = flag.change
            subject = change.claim_text.split(": ", 1)[0]
            word = _FIELD_WORDS.get(change.field, change.field)
            if word not in by_subject[subject]:
                by_subject[subject].append(word)
            cites.setdefault((change.source_slug, change.locator), _change_cite(change))
        moved = "; ".join(
            f"{subject} ({', '.join(words)})" for subject, words in by_subject.items()
        )
        items.append(
            Item(
                "truth",
                "risk",
                [
                    Sentence(f"The CRM moved forward this week: {moved}.", list(cites.values())),
                    Sentence(
                        f"But a call on {call.day.isoformat()} said: "
                        f"{readable(call.claim_text, titles)}",
                        [_call_cite(call)],
                    ),
                ],
            )
        )
    return items


_WIKILINK_GROUP_RE = re.compile(r"\s*\((?:\s*\[\[[a-z0-9][a-z0-9_-]*\]\])+\s*\)")


def readable(text: str, titles: dict[str, str]) -> str:
    """Claim text for a reader: the trailing `([[a]] [[b]])` mention list
    dropped, any other `[[slug]]` shown as its note's title."""
    text = _WIKILINK_GROUP_RE.sub("", text)
    return WIKILINK_RE.sub(lambda m: titles.get(m.group(1), m.group(1)), text).strip()


def _note_titles(conn: psycopg.Connection) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT slug, title FROM notes WHERE type <> 'source'")
        return dict(cur.fetchall())


def _rollup(crm_id: str | None, selected: set[str], accounts: dict[str, _Account]) -> str | None:
    seen: set[str] = set()
    while crm_id and crm_id not in selected and crm_id not in seen:
        seen.add(crm_id)
        account = accounts.get(crm_id)
        crm_id = account.parent_crm_id if account else None
    return crm_id if crm_id in selected else None


def build_monday(
    conn: psycopg.Connection,
    *,
    week: str,
    owner: str | None = None,
    account_ids: list[str] | None = None,
    top_n: int = DEFAULT_TOP_N,
    llm: LLMClient | None = None,
    prose_model: str | None = None,
    base_url: str = "",
    may_call_model: Callable[[], bool] | None = None,
) -> MondayEmail:
    """The Monday email for `owner`'s top `top_n` accounts by amount, or for
    `account_ids` (CRM ids or names). Reads only; the model, when given, is
    called once per changed account while `may_call_model()` allows."""
    start, end = week_bounds(week)
    accounts = _accounts(conn)
    snapshots = load_snapshots(conn, until=end)
    latest = _latest(snapshots)
    owners = _owner_by_account(accounts, latest)
    amounts = _amounts(accounts, latest)

    owner_crm_id, owner_name = (None, "")
    if owner:
        owner_crm_id, owner_name = resolve_owner(conn, owner)
    if account_ids:
        by_name = {a.name.lower(): a.crm_id for a in accounts.values()}
        chosen = []
        for raw in account_ids:
            crm_id = raw if raw in accounts else by_name.get(raw.strip().lower())
            if crm_id is None:
                raise ValueError(f"no account {raw!r} in the registry")
            chosen.append(crm_id)
    else:
        if owner_crm_id is None:
            raise ValueError("give --owner or --accounts")
        owned = [c for c in accounts if owners.get(c) == owner_crm_id]
        owned.sort(key=lambda c: (-amounts[c][0], accounts[c].name))
        chosen = owned[:top_n]
    selected = set(chosen)

    blocks = {
        c: AccountBlock(
            crm_id=c,
            name=accounts[c].name,
            note_slug=accounts[c].note_slug,
            amount=amounts[c][0],
            amount_basis=amounts[c][1],
        )
        for c in chosen
    }

    titles = _note_titles(conn)
    changes = load_changes(conn, start=start, end=end)
    calls = call_claims(conn, start=start - timedelta(days=LOOKBACK_DAYS), end=end)
    flags = truth_flags(changes, calls, account_families(conn), window_end=end)

    flags_by_account: dict[str, list[TruthFlag]] = defaultdict(list)
    for flag in flags:
        target = _rollup(flag.change.account_crm_id, selected, accounts)
        if target:
            flags_by_account[target].append(flag)
    for target, group in flags_by_account.items():
        for item in _truth_items(group, titles):
            blocks[target].items.append(item)
            blocks[target].score += TRUTH_POINTS

    for change in changes:
        target = _rollup(change.account_crm_id, selected, accounts)
        if target:
            blocks[target].items.append(_crm_item(change))
            blocks[target].score += change.points

    slug_to_crm = {a.note_slug: c for c, a in accounts.items() if a.note_slug}
    action_cites: set[tuple[str, str]] = set()
    for action in _actions(conn):
        target = _rollup(slug_to_crm.get(action.account_slug or ""), selected, accounts)
        if not target or not action.locator:
            continue
        overdue = action.status == "overdue" or (action.due is not None and action.due <= end)
        new = action.created is not None and start <= action.created <= end
        if not (overdue or new):
            continue
        due = f"due {action.due.isoformat()}" if action.due else "no due date"
        label = "Overdue" if overdue else "New action"
        cite = Cite(
            action.source_slug,
            action.locator,
            _source_label(action.source_type, action.source_date, action.locator),
        )
        action_cites.add((cite.source_slug, cite.locator))
        sentence = Sentence(f"{label} ({due}): {readable(action.what, titles)}", [cite])
        blocks[target].items.append(Item("action", "risk" if overdue else "", [sentence]))
        blocks[target].score += OVERDUE_POINTS if overdue else NEW_ACTION_POINTS

    for call in calls:
        if not (start <= call.day <= end) or call.note_type == "action":
            continue
        # The same moment already shown as an action is not repeated.
        if (call.source_slug, call.locator) in action_cites:
            continue
        target = _rollup(slug_to_crm.get(call.account_slug or ""), selected, accounts)
        if target:
            tone = "risk" if call.fact_type == "risk" else ""
            sentence = Sentence(readable(call.claim_text, titles), [_call_cite(call)])
            blocks[target].items.append(Item("call", tone, [sentence]))
            blocks[target].score += CALL_POINTS.get(call.fact_type, CALL_POINTS_DEFAULT)

    order = {"truth": 0, "crm": 1, "call": 2, "action": 3}
    for block in blocks.values():
        block.items.sort(key=lambda i: order[i.kind])

    changed = [b for b in blocks.values() if b.items]
    changed.sort(key=lambda b: (-b.score, -b.amount, b.name))
    quiet = sorted((b for b in blocks.values() if not b.items), key=lambda b: (-b.amount, b.name))

    email = MondayEmail(
        owner_crm_id=owner_crm_id,
        owner_name=owner_name,
        week=week,
        start=start,
        end=end,
        base_url=base_url,
        accounts=changed,
        quiet=quiet,
    )
    written = _written_keys(conn)
    email.unwritten_crm_changes = sum(
        1 for spec in compute_changes(snapshots) if spec.key not in written
    )
    if llm is not None:
        for block in changed:
            if may_call_model is not None and not may_call_model():
                email.summary_notes.append(f"{block.name}: no summary, the spend cap was reached")
                continue
            block.summary = _summarize(
                llm, block, prose_model=prose_model, notes=email.summary_notes
            )
    return email


def _written_keys(conn: psycopg.Connection) -> set[tuple[str, str, str, str, int]]:
    with conn.cursor() as cur:
        cur.execute("SELECT object, crm_id, field, kind, snapshot_id FROM crm_changes")
        return {tuple(r) for r in cur.fetchall()}  # type: ignore[misc]


# --------------------------------------------------------------------------
# The optional summary
# --------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You summarize what changed this week on one customer account for the sales rep who owns "
    "it. Use only the lines you are given. Write at most two plain sentences. End every sentence "
    "with the citation marker or markers, copied exactly as ^[source @ locator], of the lines it "
    "relies on. State no number, date or amount that is not written in those lines. Do not "
    'recommend, predict or speculate. Return JSON: {"sentences": ["..."]}.'
)


def _summary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": SUMMARY_MAX_SENTENCES,
            }
        },
        "required": ["sentences"],
        "additionalProperties": False,
    }


def _summary_lines(block: AccountBlock) -> tuple[list[str], NumberFacts]:
    lines: list[str] = []
    facts = NumberFacts()
    for item in block.items:
        for sentence in item.sentences:
            markers = "".join(c.marker for c in sentence.cites)
            line = f"{sentence.text} {markers}"
            lines.append(line)
            first = sentence.cites[0]
            facts.harvest(sentence.text, marker=(first.source_slug, first.locator))
    return lines, facts


def _summarize(
    llm: LLMClient, block: AccountBlock, *, prose_model: str | None, notes: list[str]
) -> list[str]:
    """Kept sentences only: each must cite at least one given line, cite
    nothing else, and state only numbers those lines hold."""
    lines, facts = _summary_lines(block)
    allowed = citation_markers_in("\n".join(lines))
    payload = {"account": block.name, "lines": lines}
    try:
        result = llm.complete(
            system=SUMMARY_SYSTEM,
            user=json.dumps(payload),
            json_schema=_summary_schema(),
            model=prose_model,
            surface="report",
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001 -- one failed summary must not fail the email
        notes.append(f"{block.name}: no summary, the model call failed ({type(exc).__name__})")
        return []
    raw = (result.data or {}).get("sentences") or []
    kept: list[str] = []
    dropped = 0
    for sentence in raw[:SUMMARY_MAX_SENTENCES]:
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        sentence = sentence.strip()
        if (
            not citation_markers_in(sentence)
            or validate_citations(sentence, allowed) is not None
            or check_numerals(sentence, facts)
        ):
            dropped += 1
            continue
        kept.append(sentence)
    if dropped:
        notes.append(f"{block.name}: {dropped} summary sentence(s) dropped by the citation checks")
    return kept


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def brand_tokens(path: Path = BRAND_TOKENS_CSS) -> dict[str, str]:
    """The light-mode `--pl-*` values from the brand tokens file: email
    clients drop CSS variables, so the values are inlined."""
    text = path.read_text(encoding="utf-8")
    root = text.split("@media", 1)[0]
    tokens = {}
    for name, value in re.findall(r"--pl-([a-z]+):\s*([^;]+);", root):
        tokens[name] = value.strip().replace('"', "'")
    return tokens


_MARKER_RE = re.compile(r"\s*\^\[\s*([^\]@]+?)\s*@\s*([^\]]*)\]")


def _summary_html(sentence: str, email: MondayEmail, t: dict[str, str]) -> str:
    parts = []
    pos = 0
    for match in _MARKER_RE.finditer(sentence):
        parts.append(html.escape(sentence[pos : match.start()]))
        cite = Cite(match.group(1).strip(), match.group(2).strip(), "source")
        parts.append(_cite_html(cite, email, t))
        pos = match.end()
    parts.append(html.escape(sentence[pos:]))
    return "".join(parts)


def _cite_html(cite: Cite, email: MondayEmail, t: dict[str, str]) -> str:
    return (
        f' <a href="{html.escape(cite.href(email.base_url))}" style="color:{t["accent"]};'
        f'font-family:{t["mono"]};font-size:12px;text-decoration:underline;white-space:nowrap">'
        f"[{html.escape(cite.label)}]</a>"
    )


_KIND_LABEL = {
    "truth": "Does the CRM match the calls?",
    "crm": "CRM",
    "call": "From calls",
    "action": "Actions",
}


def render_html(email: MondayEmail, tokens: dict[str, str] | None = None) -> str:
    t = tokens or brand_tokens()
    esc = html.escape
    out = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(email.subject)}</title></head>",
        f'<body style="margin:0;padding:0;background:{t["ground"]};color:{t["ink"]};'
        f'font-family:{t["body"]}">',
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:{t["ground"]}"><tr><td align="center" style="padding:24px 12px">',
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="max-width:640px">',
        f'<tr><td style="font-family:{t["display"]};font-size:13px;letter-spacing:.06em;'
        f'text-transform:uppercase;color:{t["ink"]};padding-bottom:4px">Portolan</td></tr>',
        f'<tr><td style="font-family:{t["display"]};font-size:22px;font-weight:600;'
        f'padding-bottom:4px">What changed on your accounts</td></tr>',
        f'<tr><td style="font-family:{t["mono"]};font-size:12px;color:{t["muted"]};'
        f'padding-bottom:16px;border-bottom:1px solid {t["line"]}">'
        f"{esc(email.owner_name or 'Selected accounts')} &middot; week of "
        f"{email.start.isoformat()} to {email.end.isoformat()} ({esc(email.week)})</td></tr>",
        f'<tr><td style="font-size:15px;padding:16px 0">'
        f"{esc(email.overview())} Every line links to the CRM "
        f"row or the moment in the call it came from.</td></tr>",
    ]
    for rank, block in enumerate(email.accounts, start=1):
        amount = f"{block.amount_basis} {money_text(block.amount)}" if block.amount_basis else ""
        out.append(
            f'<tr><td style="padding:8px 0 20px 0;border-top:1px solid {t["line"]}">'
            f'<div style="font-family:{t["display"]};font-size:17px;font-weight:600;'
            f'padding-top:12px">{rank}. {esc(block.name)}</div>'
            f'<div style="font-family:{t["mono"]};font-size:12px;color:{t["muted"]};'
            f'padding-bottom:6px">{esc(amount)}{" &middot; " if amount else ""}change score '
            f"{block.score}</div>"
        )
        if block.summary:
            out.append(
                f'<p style="margin:6px 0 10px 0;font-size:15px;line-height:1.5">'
                f"{' '.join(_summary_html(s, email, t) for s in block.summary)}</p>"
            )
        current_kind = None
        for item in block.items:
            if item.kind != current_kind:
                if current_kind is not None:
                    out.append("</ul>")
                color = t["accent"] if item.kind == "truth" else t["muted"]
                out.append(
                    f'<div style="font-family:{t["display"]};font-size:11px;letter-spacing:.06em;'
                    f'text-transform:uppercase;color:{color};padding-top:8px">'
                    f"{esc(_KIND_LABEL[item.kind])}</div>"
                    '<ul style="margin:4px 0 0 0;padding-left:18px">'
                )
                current_kind = item.kind
            marker = {"risk": t["accent"], "forward": t["secondary"]}.get(item.tone, t["ink"])
            body = " ".join(
                esc(s.text) + "".join(_cite_html(c, email, t) for c in s.cites)
                for s in item.sentences
            )
            out.append(
                f'<li style="margin:0 0 6px 0;font-size:14px;line-height:1.45;color:{t["ink"]}">'
                f'<span style="color:{marker}">&#9632;</span> {body}</li>'
            )
        if current_kind is not None:
            out.append("</ul>")
        out.append("</td></tr>")
    if email.quiet:
        names = ", ".join(esc(b.name) for b in email.quiet)
        out.append(
            f'<tr><td style="padding:12px 0;border-top:1px solid {t["line"]};font-size:14px;'
            f'color:{t["muted"]}">No changes this week: {names}.</td></tr>'
        )
    footer = (
        "Made by code from CRM exports and call transcripts. Account summaries, where shown, "
        "were written by a model and kept only when every sentence cites the lines above."
    )
    if email.unwritten_crm_changes:
        footer += (
            f" {email.unwritten_crm_changes} CRM changes are not written yet; run "
            "python -m portolan.crm changes."
        )
    out.append(
        f'<tr><td style="padding-top:16px;border-top:1px solid {t["line"]};'
        f'font-family:{t["mono"]};font-size:11px;color:{t["muted"]}">{esc(footer)}</td></tr>'
    )
    out.append("</table></td></tr></table></body></html>")
    return "\n".join(out) + "\n"


def render_text(email: MondayEmail) -> str:
    refs: list[str] = []
    ref_index: dict[str, int] = {}

    def ref(cite: Cite) -> str:
        href = cite.href(email.base_url)
        if href not in ref_index:
            refs.append(href)
            ref_index[href] = len(refs)
        return f"[{ref_index[href]}]"

    def with_refs(sentence: str) -> str:
        return _MARKER_RE.sub(
            lambda m: " " + ref(Cite(m.group(1).strip(), m.group(2).strip(), "")), sentence
        )

    lines = [
        "PORTOLAN - What changed on your accounts",
        f"{email.owner_name or 'Selected accounts'}, week of {email.start.isoformat()} to "
        f"{email.end.isoformat()} ({email.week})",
        "",
        email.overview(),
    ]
    for rank, block in enumerate(email.accounts, start=1):
        amount = f", {block.amount_basis} {money_text(block.amount)}" if block.amount_basis else ""
        lines += ["", f"{rank}. {block.name}{amount} (change score {block.score})"]
        if block.summary:
            lines.append("   " + " ".join(with_refs(s) for s in block.summary))
        current_kind = None
        for item in block.items:
            if item.kind != current_kind:
                lines.append(f"   {_KIND_LABEL[item.kind]}")
                current_kind = item.kind
            text = " ".join(
                s.text + " " + " ".join(ref(c) for c in s.cites) for s in item.sentences
            )
            lines.append(f"   - {text}")
    if email.quiet:
        lines += ["", "No changes this week: " + ", ".join(b.name for b in email.quiet) + "."]
    lines += ["", "Sources"]
    lines += [f"[{i}] {href}" for i, href in enumerate(refs, start=1)]
    return "\n".join(lines) + "\n"


def to_json(email: MondayEmail) -> dict[str, Any]:
    data = asdict(email)
    data["subject"] = email.subject
    data["links"] = sorted(
        {
            cite.href(email.base_url)
            for block in email.accounts
            for item in block.items
            for sentence in item.sentences
            for cite in sentence.cites
        }
    )
    return json.loads(json.dumps(data, default=str))


# --------------------------------------------------------------------------
# Sending
# --------------------------------------------------------------------------


class SendRefused(RuntimeError):
    pass


def send_email(settings: Any, *, to: str, subject: str, html_body: str, text_body: str) -> None:
    """Send through `PORTOLAN_SMTP_URL`. Refuses (`SendRefused`) when it or
    `--to` is unset, so nothing leaves without a configured server."""
    from portolan.config import secret_value

    url = secret_value(settings.smtp_url)
    if not url:
        raise SendRefused("PORTOLAN_SMTP_URL is not set; the email was written to files only")
    if not to:
        raise SendRefused("--send needs --to")
    parts = urlsplit(url)
    if parts.scheme not in ("smtp", "smtps") or not parts.hostname:
        raise SendRefused("PORTOLAN_SMTP_URL must look like smtp://user:password@host:port")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from or (parts.username or "portolan")
    message["To"] = to
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    smtp_cls = smtplib.SMTP_SSL if parts.scheme == "smtps" else smtplib.SMTP
    with smtp_cls(
        parts.hostname, parts.port or (465 if parts.scheme == "smtps" else 587)
    ) as server:
        if parts.scheme == "smtp":
            server.starttls()
        if parts.username:
            server.login(parts.username, parts.password or "")
        server.send_message(message)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

SPEND_KEY = "report"
_REPO_VAR = Path(__file__).resolve().parents[2] / "var"


def add_parser(sub: Any) -> None:
    parser = sub.add_parser("monday", help="the Monday email for one person's accounts")
    parser.add_argument("--owner", default=None, help="internal person: CRM id or name")
    parser.add_argument("--accounts", default=None, help="comma-separated CRM ids or names")
    parser.add_argument("--week", required=True, help="ISO week the email covers, YYYY-Www")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--out", type=Path, default=Path("var/reports/monday"))
    parser.add_argument("--base-url", default=None, help="default: PORTOLAN_PUBLIC_URL")
    parser.add_argument(
        "--llm",
        choices=["none", "openrouter", "agent_sdk", "anthropic", "bedrock"],
        default="none",
        help="model for the per-account summaries; none (default) writes no summaries",
    )
    parser.add_argument("--prose-model", default=None)
    parser.add_argument(
        "--max-usd",
        type=float,
        default=1.0,
        help="stop asking for summaries once this run's metered spend reaches this",
    )
    parser.add_argument("--send", action="store_true", help="send through PORTOLAN_SMTP_URL")
    parser.add_argument("--to", default=None)


def _build_llm(args: argparse.Namespace, settings: Any, conn: psycopg.Connection):
    """`(llm, may_call_model, spent_usd)` for `--llm`."""
    if args.llm == "none":
        return None, None, lambda: 0.0
    from portolan.metering import PostgresSink

    if args.llm == "openrouter":
        from portolan.config import secret_value
        from portolan.llm.openrouter import OpenRouterClient
        from portolan.training.spend import SpendMeter, default_spend_path

        api_key = secret_value(settings.openrouter_api_key)
        if not api_key:
            raise SystemExit("--llm openrouter needs OPENROUTER_API_KEY set")
        meter = SpendMeter(default_spend_path(_REPO_VAR))
        before = meter.corpus_usd(SPEND_KEY)
        llm = OpenRouterClient(
            spend_key=SPEND_KEY,
            spend_meter=meter,
            api_key=api_key,
            default_model=settings.strong_model or settings.extraction_model,
            provider_order=settings.openrouter_provider_order,
            sink=PostgresSink(conn),
            tenant=settings.tenant,
        )

        def spent() -> float:
            return meter.corpus_usd(SPEND_KEY) - before

        return llm, (lambda: spent() < args.max_usd), spent
    from portolan.llm.factory import get_llm

    llm = get_llm(settings, provider=args.llm)
    return llm, None, lambda: 0.0


def run_cli(args: argparse.Namespace) -> int:
    from portolan import db
    from portolan.config import get_settings

    settings = get_settings()
    base_url = args.base_url or settings.public_base_url or ""
    account_ids = (
        [a.strip() for a in args.accounts.split(",") if a.strip()] if args.accounts else None
    )
    conn = db.connect(autocommit=True)
    try:
        db.apply_migrations(conn)
        llm, may_call, spent = _build_llm(args, settings, conn)
        try:
            email = build_monday(
                conn,
                week=args.week,
                owner=args.owner,
                account_ids=account_ids,
                top_n=args.top,
                llm=llm,
                prose_model=args.prose_model,
                base_url=base_url,
                may_call_model=may_call,
            )
        except ValueError as exc:
            print(f"monday: {exc}", file=sys.stderr)
            return 2
    finally:
        conn.close()

    who = email.owner_crm_id or "accounts"
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"monday-{who}-{email.week}"
    html_body = render_html(email)
    text_body = render_text(email)
    stem.with_suffix(".html").write_text(html_body, encoding="utf-8")
    stem.with_suffix(".txt").write_text(text_body, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(to_json(email), indent=2), encoding="utf-8")
    counts = email.counts()
    print(f"wrote {stem}.html, .txt, .json")
    print(
        f"accounts changed {len(email.accounts)}, quiet {len(email.quiet)}; crm {counts['crm']}, "
        f"truth checks {counts['truth']}, calls {counts['call']}, actions {counts['action']}; "
        f"summaries {sum(1 for b in email.accounts if b.summary)}; spend ${spent():.4f}"
    )
    for note in email.summary_notes:
        print(f"  {note}")
    if args.send:
        try:
            send_email(
                settings,
                to=args.to or "",
                subject=email.subject,
                html_body=html_body,
                text_body=text_body,
            )
        except SendRefused as exc:
            print(f"--send refused: {exc}", file=sys.stderr)
            return 3
        print(f"sent to {args.to}")
    return 0
