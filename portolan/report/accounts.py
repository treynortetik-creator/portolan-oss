"""Account-ARR, region and branch lookups, and "which account does this
claim belong to", for `portolan.report`.

ARR, region and branch are not stored in the graph (no registry or notes
column carries them), so they are read at report time from the CRM accounts
export, keyed by `crm_id`. The `*_via_connector` variants ask the connector's
`crm_paths()` for the file; the plain loaders take a path.

`home_account_slug` reconstructs a claim's account from what merge wrote:
the home note itself, a `signal-<fact_type>-<account>` slug, or the first
`[[account]]` wikilink in the claim text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol

import psycopg

from portolan.notes.render import WIKILINK_RE

__all__ = [
    "account_arr_by_note_slug",
    "account_branch_by_note_slug",
    "account_region_by_note_slug",
    "home_account_slug",
    "load_account_arr",
    "load_account_arr_via_connector",
    "load_account_branches",
    "load_account_branches_via_connector",
    "load_account_regions",
    "load_account_regions_via_connector",
]


class _CrmPathsLike(Protocol):
    accounts: Path


class _ConnectorLike(Protocol):
    def crm_paths(self) -> _CrmPathsLike | None: ...


def _queue_unparseable_arr(conn: psycopg.Connection, *, crm_id: str, raw_arr: str) -> None:
    """Queue a present-but-non-numeric `arr` cell (a dirty export, not a
    zero) once per open (crm_id, value), since every report run re-reads it."""
    payload = {
        "reason": "arr_not_parseable",
        "crm_id": crm_id,
        "detail": f"CRM accounts export: arr={raw_arr!r} for {crm_id} is not a number",
    }
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM review_queue WHERE kind = 'other' AND status = 'open' "
            "AND payload ->> 'reason' = 'arr_not_parseable' "
            "AND payload ->> 'crm_id' = %s AND payload ->> 'detail' = %s",
            (crm_id, payload["detail"]),
        )
        if cur.fetchone() is not None:
            return
        cur.execute(
            "INSERT INTO review_queue (kind, payload) VALUES ('other', %s::jsonb)",
            (json.dumps(payload),),
        )
    conn.commit()


def load_account_arr(
    accounts_csv: Path, conn: psycopg.Connection | None = None
) -> dict[str, float]:
    """`{crm_id: arr}` from the CRM accounts export's `arr` column.

    A blank cell reads as `0.0` (no revenue on file). An unparseable cell
    (`"N/A"`, a stray comma) is left out of the dict, so callers can treat
    it as unknown rather than `$0`; with `conn` given, each one also gets a
    `review_queue` row.
    """
    arr_by_crm_id: dict[str, float] = {}
    with accounts_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crm_id = (row.get("crm_id") or "").strip()
            if not crm_id:
                continue
            raw_arr = (row.get("arr") or "").strip()
            if not raw_arr:
                arr_by_crm_id[crm_id] = 0.0
                continue
            try:
                arr_by_crm_id[crm_id] = float(raw_arr)
            except ValueError:
                if conn is not None:
                    _queue_unparseable_arr(conn, crm_id=crm_id, raw_arr=raw_arr)
    return arr_by_crm_id


def load_account_arr_via_connector(
    connector: _ConnectorLike, conn: psycopg.Connection | None = None
) -> dict[str, float]:
    """`load_account_arr` for the file `connector.crm_paths()` names, or `{}`
    when the connector has no `crm_paths` or it returns `None`."""
    crm_paths_fn = getattr(connector, "crm_paths", None)
    if crm_paths_fn is None:
        return {}
    paths = crm_paths_fn()
    if paths is None:
        return {}
    return load_account_arr(paths.accounts, conn)


def account_arr_by_note_slug(
    conn: psycopg.Connection, arr_by_crm_id: dict[str, float]
) -> dict[str, float]:
    """`{account_note_slug: arr}`, joining `arr_by_crm_id` through
    `registry_entities.note_slug`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, note_slug FROM registry_entities "
            "WHERE kind = 'account' AND note_slug IS NOT NULL AND crm_id IS NOT NULL"
        )
        rows = cur.fetchall()
    return {
        note_slug: arr_by_crm_id[crm_id] for crm_id, note_slug in rows if crm_id in arr_by_crm_id
    }


def load_account_regions(accounts_csv: Path) -> dict[str, str]:
    """`{crm_id: region}` from the CRM accounts export's `region` column.
    A blank or missing cell is left out of the dict (unknown), never `""`."""
    region_by_crm_id: dict[str, str] = {}
    with accounts_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crm_id = (row.get("crm_id") or "").strip()
            if not crm_id:
                continue
            region = (row.get("region") or "").strip()
            if region:
                region_by_crm_id[crm_id] = region
    return region_by_crm_id


def load_account_regions_via_connector(connector: _ConnectorLike) -> dict[str, str]:
    """`load_account_regions` for the file `connector.crm_paths()` names,
    or `{}` when there is none."""
    crm_paths_fn = getattr(connector, "crm_paths", None)
    if crm_paths_fn is None:
        return {}
    paths = crm_paths_fn()
    if paths is None:
        return {}
    return load_account_regions(paths.accounts)


def account_region_by_note_slug(
    conn: psycopg.Connection, region_by_crm_id: dict[str, str]
) -> dict[str, str]:
    """`{account_note_slug: region}`, joined through `registry_entities.note_slug`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, note_slug FROM registry_entities "
            "WHERE kind = 'account' AND note_slug IS NOT NULL AND crm_id IS NOT NULL"
        )
        rows = cur.fetchall()
    return {
        note_slug: region_by_crm_id[crm_id]
        for crm_id, note_slug in rows
        if crm_id in region_by_crm_id
    }


def load_account_branches(accounts_csv: Path) -> dict[str, str]:
    """`{crm_id: branch}` from the CRM accounts export's `branch` column.
    A blank or missing cell is left out of the dict, never `""`."""
    branch_by_crm_id: dict[str, str] = {}
    with accounts_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crm_id = (row.get("crm_id") or "").strip()
            if not crm_id:
                continue
            branch = (row.get("branch") or "").strip()
            if branch:
                branch_by_crm_id[crm_id] = branch
    return branch_by_crm_id


def load_account_branches_via_connector(connector: _ConnectorLike) -> dict[str, str]:
    """`load_account_branches` for the file `connector.crm_paths()` names,
    or `{}` when there is none."""
    crm_paths_fn = getattr(connector, "crm_paths", None)
    if crm_paths_fn is None:
        return {}
    paths = crm_paths_fn()
    if paths is None:
        return {}
    return load_account_branches(paths.accounts)


def account_branch_by_note_slug(
    conn: psycopg.Connection, branch_by_crm_id: dict[str, str]
) -> dict[str, str]:
    """`{account_note_slug: branch}`, joined through `registry_entities.note_slug`."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT crm_id, note_slug FROM registry_entities "
            "WHERE kind = 'account' AND note_slug IS NOT NULL AND crm_id IS NOT NULL"
        )
        rows = cur.fetchall()
    return {
        note_slug: branch_by_crm_id[crm_id]
        for crm_id, note_slug in rows
        if crm_id in branch_by_crm_id
    }


def _account_slug_from_signal(note_slug: str, fact_type: str) -> str | None:
    """Invert `signal_note_slug` (`signal-{fact_type}-{account_slug}`) using
    the claim's known `fact_type`, so hyphenated slugs need no guesswork."""
    prefix = f"signal-{fact_type}-"
    if note_slug.startswith(prefix):
        return note_slug[len(prefix) :]
    return None


def home_account_slug(
    conn: psycopg.Connection,
    *,
    note_slug: str,
    fact_type: str,
    claim_text: str,
) -> str | None:
    """The account a claim belongs to:

    1. The claim's home note is an account note -> that account.
    2. The home note is `signal-<fact_type>-<account>` and that account
       note exists -> that account.
    3. Otherwise -> the first `[[slug]]` in `claim_text` that is an account
       note (any status; there is no `living` filter).

    `None` when none of these resolves.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT type FROM notes WHERE slug = %s", (note_slug,))
        row = cur.fetchone()
    note_type = row[0] if row else None

    if note_type == "account":
        return note_slug

    if note_type == "signal":
        guess = _account_slug_from_signal(note_slug, fact_type)
        if guess is not None:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM notes WHERE slug = %s AND type = 'account'", (guess,))
                if cur.fetchone() is not None:
                    return guess

    mentioned = [m.group(1) for m in WIKILINK_RE.finditer(claim_text)]
    if not mentioned:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT slug FROM notes WHERE type = 'account' AND slug = ANY(%s)",
            (mentioned,),
        )
        accounts_in_text = {r[0] for r in cur.fetchall()}
    for slug in mentioned:
        if slug in accounts_in_text:
            return slug
    return None
