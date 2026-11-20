"""Structured, no-model outputs: CRM CSVs, RSS JSON, email headers, ticket
skeletons, and the locator-format helpers every other module shares.

Nothing in this module calls a model. CRM rows, RSS item envelopes, email
headers, and ticket metadata are deterministic transformations of the truth
table. Where a source needs natural-language body text (an email body, a
ticket comment, a call line), this module builds the *container* and
`prose.py` (or a hand/subagent-written source in manual mode) supplies the
words that go inside it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MESSAGE_ID_DOMAIN = "halvardmedical.com"


# --------------------------------------------------------------------------
# Locator formats — the single source of truth for every format string used
# elsewhere (markers.py records these, manifest.py validates against them).
# --------------------------------------------------------------------------


def call_locator(hh_mm_ss: str) -> str:
    """A call locator IS the timestamp, e.g. '00:14:32'."""
    return hh_mm_ss


def ticket_locator(ticket_id: str, comment_number: int) -> str:
    return f"{ticket_id}#{comment_number}"


def email_message_id(source_id: str) -> str:
    """`Message-ID: <S-email-052@halvardmedical.com>` — the id without angle
    brackets, used both in the header and as the locator's source half."""
    return f"{source_id}@{MESSAGE_ID_DOMAIN}"


def email_locator(source_id: str, paragraph_index: int) -> str:
    return f"{email_message_id(source_id)}#p{paragraph_index}"


def rss_locator(link: str, date: str) -> str:
    return f"url:{link} {date}"


def crm_locator(relative_path: str, crm_id: str) -> str:
    return f"{relative_path}#{crm_id}"


def seconds_to_hhmmss(total_seconds: int) -> str:
    h, rem = divmod(max(0, total_seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# Chat (Slack / Teams) locators — added 2026-09-19 for the synthetic-data
# pipeline (`corpus/generator/worlds.py`, `cli.py synth`). A chat locator
# points at one message inside its channel/day file: the file path (relative
# to `corpus/out`-equivalent) plus a 1-based index into that file's JSON
# array, matching the 1-based convention `ticket_locator`/`email_locator`
# already use for comment numbers/paragraph indices.
# `manifest.LOCATOR_PATTERNS["slack"]`/`["teams"]` enforce this format.
# --------------------------------------------------------------------------


def slack_channel_path(channel: str, date: str) -> str:
    return f"slack/{channel}/{date}.json"


def slack_locator(channel: str, date: str, message_index: int) -> str:
    return f"{slack_channel_path(channel, date)}#m{message_index}"


def teams_channel_path(channel: str, date: str) -> str:
    return f"teams/{channel}/{date}.json"


def teams_locator(channel: str, date: str, message_index: int) -> str:
    return f"{teams_channel_path(channel, date)}#m{message_index}"


def slack_ts(date: str, seconds_into_day: int) -> str:
    """A plausible Slack `ts` value: `<unix-seconds>.<6-digit-fraction>`.
    Derived deterministically from `date` + an offset so two calls with the
    same inputs always produce the same `ts` (no wall-clock dependency,
    same discipline as the rest of this module)."""
    import datetime as _dt

    day = _dt.date.fromisoformat(date)
    epoch_seconds = int(_dt.datetime(day.year, day.month, day.day, tzinfo=_dt.UTC).timestamp())
    return f"{epoch_seconds + seconds_into_day}.000100"


def teams_created_datetime(date: str, seconds_into_day: int) -> str:
    """A plausible Teams `createdDateTime`: ISO-8601 UTC with a `Z` suffix."""
    import datetime as _dt

    day = _dt.date.fromisoformat(date)
    moment = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.UTC) + _dt.timedelta(
        seconds=seconds_into_day
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Email headers
# --------------------------------------------------------------------------


def email_header_block(
    source_id: str,
    date: str,
    from_addr: str,
    to_addrs: list[str],
    subject: str,
    in_reply_to: str | None = None,
) -> str:
    """The `.eml`-style header block. `date` is an ISO date; a fixed
    09:00 local time is used since the truth table does not specify times
    for emails."""
    lines = [
        f"Message-ID: <{email_message_id(source_id)}>",
        f"From: {from_addr}",
        f"To: {', '.join(to_addrs)}",
        f"Date: {date} 09:00:00 -0700",
        f"Subject: {subject}",
    ]
    if in_reply_to:
        lines.append(f"In-Reply-To: <{email_message_id(in_reply_to)}>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CRM CSVs
# --------------------------------------------------------------------------


def write_accounts_csv(truth: dict[str, Any], out_path: Path) -> int:
    fields = ["crm_id", "name", "kind", "parent_crm_id", "domain", "arr", "region", "status"]
    accounts_by_id = {a["id"]: a for a in truth["accounts"]}
    rows = []
    for a in truth["accounts"]:
        parent_crm_id = accounts_by_id[a["parent"]]["crm_id"] if a.get("parent") else ""
        rows.append(
            {
                "crm_id": a["crm_id"],
                "name": a["name"],
                "kind": a["kind"],
                "parent_crm_id": parent_crm_id,
                "domain": a["domain"],
                "arr": a["arr"],
                "region": a["region"],
                "status": a.get("status", ""),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_contacts_csv(truth: dict[str, Any], out_path: Path) -> int:
    """Every contact EXCEPT any with crm_id null (maria-chen-personal is not
    in the CRM on purpose — that is the point of F30's unknown-entity test)."""
    fields = ["crm_id", "name", "title", "account_crm_id", "email", "status"]
    accounts_by_id = {a["id"]: a for a in truth["accounts"]}
    rows = []
    for c in truth["contacts"]:
        if not c.get("crm_id"):
            continue
        rows.append(
            {
                "crm_id": c["crm_id"],
                "name": c["name"],
                "title": c.get("title", ""),
                "account_crm_id": accounts_by_id[c["account"]]["crm_id"],
                "email": c.get("email", ""),
                "status": c.get("status", ""),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# Opportunity rows are not fully modeled in the truth table beyond the two
# renewal facts (F11 / F11b) and Kestrel's closed-lost history (F34). This
# builds one opportunity per account that has a HalvardCare-relevant renewal
# date or a closed status, which is exactly the set the facts touch plus a
# few plausible neighbors so the CSV does not look suspiciously sparse.
def _opportunity_rows(truth: dict[str, Any], as_of: str) -> list[dict[str, Any]]:
    accounts_by_id = {a["id"]: a for a in truth["accounts"]}
    rows: list[dict[str, Any]] = []

    def row(
        account_id: str, name: str, stage: str, close_date: str, amount: int, opp_id: str
    ) -> dict[str, Any]:
        a = accounts_by_id[account_id]
        return {
            "opportunity_id": opp_id,
            "account_crm_id": a["crm_id"],
            "name": name,
            "stage": stage,
            "close_date": close_date,
            "amount": amount,
        }

    # Ashgrove HalvardCare renewal: F11 (2026-04-14 snapshot) says 2027-03-31.
    # The 2026-08-21 snapshot (F11b) supersedes it with 2027-06-30.
    renewal_date = "2027-06-30" if as_of >= "2026-08-21" else "2027-03-31"
    rows.append(
        row(
            "ashgrove",
            "Ashgrove Health — HalvardCare renewal",
            "negotiation",
            renewal_date,
            accounts_by_id["ashgrove"]["arr"],
            "O-3001",
        )
    )
    # Larkspur consumables renewal, in play for the volume-discount ask (F12).
    rows.append(
        row(
            "larkspur",
            "Larkspur Regional Health — Tidewater T2 consumables renewal",
            "negotiation",
            "2027-06-30",
            accounts_by_id["larkspur"]["arr"],
            "O-3002",
        )
    )
    # Northlake QBR-driven expansion (context for F32/F15b), open throughout.
    rows.append(
        row(
            "northlake",
            "Northlake Health System — Tidewater T3 tower expansion",
            "qualification",
            "2027-01-31",
            300000,
            "O-3003",
        )
    )
    # Kestrel: closed-lost after moving to Brightline (F34).
    rows.append(
        row(
            "kestrel",
            "Kestrel Medical Group — HalvardCare renewal",
            "closed_lost",
            "2026-02-15",
            0,
            "O-3004",
        )
    )
    return rows


def write_opportunities_csv(truth: dict[str, Any], out_path: Path, export_date: str) -> int:
    fields = [
        "opportunity_id",
        "account_crm_id",
        "name",
        "stage",
        "close_date",
        "amount",
        "export_date",
    ]
    rows = _opportunity_rows(truth, export_date)
    for r in rows:
        r["export_date"] = export_date
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_crm(truth: dict[str, Any], out_dir: Path) -> dict[str, int]:
    counts = {
        "accounts": write_accounts_csv(truth, out_dir / "accounts.csv"),
        "contacts": write_contacts_csv(truth, out_dir / "contacts.csv"),
        "opportunities": write_opportunities_csv(
            truth, out_dir / "opportunities.csv", "2026-04-15"
        ),
        "opportunities_v2": write_opportunities_csv(
            truth, out_dir / "opportunities-2026-08-21.csv", "2026-08-21"
        ),
    }
    return counts


# --------------------------------------------------------------------------
# RSS
# --------------------------------------------------------------------------


def rss_link(item_id: str) -> str:
    """A fictional but stable link, per item id."""
    return f"https://feeds.example-newswire.test/portolan-corpus/{item_id.lower()}"


def write_rss_feed(items: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, sort_keys=False)
        fh.write("\n")


# --------------------------------------------------------------------------
# Tickets
# --------------------------------------------------------------------------


def write_ticket(
    ticket_id: str,
    requester_name: str,
    account_name: str,
    created: str,
    subject: str,
    comments: list[dict[str, Any]],
    out_path: Path,
) -> None:
    payload = {
        "id": ticket_id,
        "requester": requester_name,
        "account": account_name,
        "created": created,
        "subject": subject,
        "comments": comments,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


# --------------------------------------------------------------------------
# Slack export shape — added 2026-09-19.
#
# `channels.json`: [{"id", "name", "is_private", "account"}]
# `users.json`:    [{"id", "name", "real_name", "is_bot"}]
# `<channel>/<YYYY-MM-DD>.json`: [{"type", "user", "text", "ts", "thread_ts"}]
#   `type` is always "message" here (no channel_join/purpose-change noise —
#   not worth modeling for extraction-training purposes). `thread_ts` is
#   `None` for a root message and the root's own `ts` for a reply, mirroring
#   Slack's real conversations.history/replies shape.
# --------------------------------------------------------------------------


def write_slack_channels(channels: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(channels, fh, indent=2, sort_keys=False)
        fh.write("\n")


def write_slack_users(users: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2, sort_keys=False)
        fh.write("\n")


def write_slack_day(messages: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(messages, fh, indent=2, sort_keys=False)
        fh.write("\n")


# --------------------------------------------------------------------------
# Microsoft Teams export shape — added 2026-09-19.
#
# `<channel>/<YYYY-MM-DD>.json`: [{"id", "from": {"user": {"displayName"}},
# "body": {"content"}, "createdDateTime", "replyToId"}] — the Graph API
# `chatMessage` resource's shape, trimmed to the fields this corpus needs.
# `replyToId` is `None` for a root message and the root message's own `id`
# for a reply.
# --------------------------------------------------------------------------


def write_teams_day(messages: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(messages, fh, indent=2, sort_keys=False)
        fh.write("\n")


# --------------------------------------------------------------------------
# Richer CRM exports — added 2026-09-19. Opportunity/case rows follow the
# Salesforce field names `portolan/connectors/salesforce.py` reads
# (`Name`/`CloseDate`/`StageName`/`Amount`/`AccountId` for opportunities;
# Salesforce's standard `Case` object for cases, which that connector does
# not yet ingest but whose field names this export still mirrors so a
# future connector can). Activity notes are free text (a case comment / a
# logged call note) and carry planted facts, so they get one small JSON
# file per note — the same "container is structured, body is prose"
# split `write_ticket` already uses — rather than a CSV cell, so
# `markers.py`'s marker pass has plain text to scan.
# --------------------------------------------------------------------------


def write_synth_opportunities_csv(rows: list[dict[str, Any]], out_path: Path) -> int:
    fields = [
        "opportunity_id",
        "account_crm_id",
        "name",
        "stage",
        "close_date",
        "amount",
        "export_date",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_synth_cases_csv(rows: list[dict[str, Any]], out_path: Path) -> int:
    fields = [
        "case_id",
        "account_crm_id",
        "contact_crm_id",
        "subject",
        "status",
        "priority",
        "created_date",
        "closed_date",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def activity_note_path(out_dir: Path, note_id: str) -> Path:
    return out_dir / "crm" / "activity_notes" / f"{note_id}.json"


def activity_note_relpath(note_id: str) -> str:
    return f"crm/activity_notes/{note_id}.json"


def write_activity_note(
    note_id: str,
    account_crm_id: str,
    contact_crm_id: str | None,
    author: str,
    created_date: str,
    subject: str,
    body: str,
    out_path: Path,
) -> None:
    payload = {
        "id": note_id,
        "account_crm_id": account_crm_id,
        "contact_crm_id": contact_crm_id,
        "author": author,
        "created_date": created_date,
        "subject": subject,
        "body": body,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
