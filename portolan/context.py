"""Company context and account context (owner decision 2026-09-17,
`01-spec/portolan-spec-v2.md` "Company and account context"): the
tenant's own short profile note, pulled into every extraction, judge, and
chat prompt so the model knows who the company is, what it sells, who it
competes with, and why a fact matters. Alongside it, per-source account
context: a short blurb about the account a source is about, so the model
understands the relationship, not just the bare account name.

Kept out of `portolan.extract.prompt` on purpose -- that module's own
docstring says it is "pure rendering: strings in, strings out... so it
can be unit-tested without touching a model." Loading the context note
(and an account's own note) is the one piece of database I/O every
caller needs before it can hand `prompt.py`, `portolan.report.themes`, or
`portolan.web.answer` a plain string. `render_company_context_block`/
`render_account_context_block` live here too, next to the loaders, so
every call site renders identical block text.

**Missing context note.** Every loader returns `None`/`{}` and logs one
warning; nothing here ever raises (build instructions: "Missing context
note: prompts render without the block and log one warning; nothing
fails").

**The one tenant this build ships for.** Per `CLAUDE.md`, the mock CRM
owner is the fictional vendor Halvard Medical (infusion pumps, Tidewater
product line; hospitals are its customers) -- `HALVARD_MEDICAL_CONTEXT_BODY`
below is that tenant's context note body, written from `corpus/README.md`,
`corpus/GENERATOR.md` and `corpus/truth-table.yaml`. No real company. The
extraction prompt's own examples name a different fictional vendor on
purpose, so this note is the only thing that names Halvard in an extraction
system prompt.
"""

from __future__ import annotations

import logging
from datetime import datetime

import psycopg
from pydantic import BaseModel, ConfigDict

from portolan.extract.schema import EntityKind, RegistryContextEntry
from portolan.notes.models import NoteRow, NoteStatus, NoteType
from portolan.notes.persist import note_exists, read_note, write_note
from portolan.util import clock

__all__ = [
    "ACCOUNT_CONTEXT_PARAGRAPH_CAP",
    "COMPANY_CONTEXT_BODY_CHAR_CAP",
    "COMPANY_CONTEXT_SLUG",
    "COMPANY_CONTEXT_TITLE",
    "HALVARD_MEDICAL_CONTEXT_BODY",
    "CompanyContext",
    "build_account_context",
    "first_paragraph",
    "load_company_context",
    "render_account_context_block",
    "render_company_context_block",
    "seed_company_context",
]

logger = logging.getLogger(__name__)

# One context note per tenant, at this fixed slug (build instructions
# item 1). `portolan.web.routes.notes`/`edit_note.html` need no changes
# to make it editable -- it is a `NoteType.CONTEXT` note like any other,
# and its whole fixed-heading body lives in `note.synthesis` (the same
# field `record_edit`'s no-`claim_id` branch already edits).
COMPANY_CONTEXT_SLUG = "company-context"
COMPANY_CONTEXT_TITLE = "Company context"

# Hard length cap on the note's body (build instructions item 1): the
# editor rejects longer with a clear message, the lint reports a note
# that is somehow over it anyway, and this is the single constant both
# read from.
COMPANY_CONTEXT_BODY_CHAR_CAP = 2500

# Per-account cap on the synthesis paragraph folded into the "Account
# context" block (build instructions item 2).
ACCOUNT_CONTEXT_PARAGRAPH_CAP = 400


class CompanyContext(BaseModel):
    """What `load_company_context` hands every caller: the note's body
    text and the `updated` timestamp stamped onto extracted facts'
    provenance (`claims.company_context_updated_at`, migration
    `0021_claims_context_stamp.sql`) so a later reader can tell which
    version of the context a fact was extracted under."""

    model_config = ConfigDict(extra="forbid")

    text: str
    updated: datetime


def load_company_context(conn: psycopg.Connection) -> CompanyContext | None:
    """The company context note's body plus its `updated` timestamp, or
    `None` (with one warning logged) when the note is missing or empty.
    Never raises -- a missing or empty note is not this build's problem
    to fail loudly over; every prompt just renders without the block."""
    if not note_exists(conn, COMPANY_CONTEXT_SLUG):
        logger.warning(
            "company context note %r not found; prompts render without the Company context block",
            COMPANY_CONTEXT_SLUG,
        )
        return None
    note, _claims = read_note(conn, COMPANY_CONTEXT_SLUG)
    text = (note.synthesis or "").strip()
    if not text:
        logger.warning(
            "company context note %r has no body; prompts render without the Company context block",
            COMPANY_CONTEXT_SLUG,
        )
        return None
    return CompanyContext(text=text, updated=note.updated)


def render_company_context_block(text: str) -> str:
    """The "Company context" block every prompt renders before its
    vocabulary block (build instructions item 2)."""
    return "## Company context\n" + text.strip()


def first_paragraph(text: str, cap: int) -> str:
    """The first blank-line-separated paragraph of `text`, capped to
    `cap` characters (build instructions item 2: "cap 400 characters per
    account"). Truncation lands on a whole-character boundary and is
    marked with a trailing ellipsis, not silently cut mid-word forever."""
    if not text:
        return ""
    para = text.strip().split("\n\n", 1)[0].strip()
    if len(para) > cap:
        para = para[:cap].rstrip() + "…"
    return para


def _account_note_slug(conn: psycopg.Connection, registry_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT note_slug FROM registry_entities WHERE crm_id = %s AND kind = 'account'",
            (registry_id,),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def build_account_context(
    conn: psycopg.Connection, registry_context: list[RegistryContextEntry]
) -> dict[str, str]:
    """`{registry_id: text}` for every ACCOUNT entry in `registry_context`
    (build instructions item 2): the registry entry line, plus the first
    paragraph (capped) of that account's note synthesis when the note
    already exists. It may not: extraction runs before the merge stage
    that creates entity notes (`portolan.notes.persist.ensure_entity_
    note`), so a brand-new account gets just the registry line."""
    out: dict[str, str] = {}
    for entry in registry_context:
        if entry.kind != EntityKind.ACCOUNT:
            continue
        alias_note = f", aliases: {', '.join(entry.aliases)}" if entry.aliases else ""
        line = f"{entry.name} (account, id={entry.registry_id}){alias_note}"
        note_slug = _account_note_slug(conn, entry.registry_id)
        if note_slug is not None:
            try:
                note, _claims = read_note(conn, note_slug)
            except ValueError:
                note = None
            if note is not None and note.synthesis:
                para = first_paragraph(note.synthesis, ACCOUNT_CONTEXT_PARAGRAPH_CAP)
                if para:
                    line = f"{line}\n{para}"
        out[entry.registry_id] = line
    return out


def render_account_context_block(entries: dict[str, str]) -> str:
    """The "Account context" block (build instructions item 2), one entry
    per account, blank-line separated. Empty string -- callers skip
    rendering entirely rather than emit a bare heading -- when there are
    no account entries."""
    if not entries:
        return ""
    return "## Account context\n\n" + "\n\n".join(entries.values())


# ---------------------------------------------------------------------
# Seed: the synthetic Halvard Medical company-context note.
# ---------------------------------------------------------------------

HALVARD_MEDICAL_CONTEXT_BODY = (
    "## Who we are\n"
    "Halvard Medical (halvardmedical.com) sells infusion pumps and "
    "integration modules to hospitals and health systems, mostly in the "
    "Midwest and Northeast. Sam Okafor and Priya Natarajan carry accounts "
    "as reps; Dana Whitlock leads customer success; Luis Ferreira runs "
    "operations (distribution branches: Dayton, Columbus, Hartford); "
    "Kenji Mori is the product manager for the Tidewater line.\n"
    "\n"
    "## What we sell (products and aliases)\n"
    "- Tidewater -- the product family (aka Tidewater consumables, "
    "Tidewater pumps, Tidewater line)\n"
    "- Tidewater T2 -- the current-generation pump (aka T2, the T2, "
    "Tidewater pump)\n"
    "- Tidewater T3 -- the newer pump (aka T3, the new pump, T3 pump)\n"
    "- Tidewater Link -- the EHR integration module (aka Link, the Link "
    "module, EHR bridge)\n"
    "- HalvardCare -- the service contract (aka HalvardCare, the service "
    "plan)\n"
    "\n"
    "## Who we compete with\n"
    "- Corvane Systems (aka Corvane) -- the competitor most often named "
    "when an account is shopping around or comparing pumps.\n"
    "- Brightline Medical (aka Brightline) -- the other named competitor; "
    "usually surfaces on price or support-quality complaints.\n"
    "\n"
    "## What matters to us\n"
    "A signal is worth recording when it touches: a delivery or product "
    "problem (a late consumable shipment, a pump or alarm malfunction); a "
    "risk of losing the account (a competitor mention, a contract nearing "
    "renewal, a complaint left unresolved); a commitment a rep made (a "
    "callback, a fix, a training session) with a date attached; a "
    "decision the customer made (switching products, expanding a "
    "rollout, cancelling); a pricing or contract change; or a contact "
    "change on the account (a champion leaving, a new signer). ARR and "
    "renewal timing matter because they say how much is actually at "
    "stake.\n"
    "\n"
    "## What not to ingest\n"
    "Scheduling logistics (finding a meeting time); small talk (weather, "
    "personal news, a conference someone attended for reasons unrelated "
    "to Halvard); internal Halvard process discussion that never touches "
    "a customer; and general health-system industry news that doesn't "
    "name Halvard, a competitor, or an account."
)

assert len(HALVARD_MEDICAL_CONTEXT_BODY) <= COMPANY_CONTEXT_BODY_CHAR_CAP


def seed_company_context(
    conn: psycopg.Connection, *, body: str = HALVARD_MEDICAL_CONTEXT_BODY
) -> bool:
    """Create the tenant's company-context note if it doesn't exist yet
    (build instructions item 1: "created by a seed step"). Idempotent --
    a second call is a no-op -- and never overwrites an existing note, so
    a human's edit (or a later re-seed with different content) is never
    clobbered by a redeploy. Returns whether it actually created one.

    Called from `portolan.web.app`'s lifespan, after `db.apply_
    migrations`, so a fresh deploy gets the note without a manual step.
    """
    if note_exists(conn, COMPANY_CONTEXT_SLUG):
        return False
    if len(body) > COMPANY_CONTEXT_BODY_CHAR_CAP:
        raise ValueError(
            f"seed_company_context: body is {len(body)} chars, over the "
            f"{COMPANY_CONTEXT_BODY_CHAR_CAP}-char cap"
        )
    now = clock.now()
    note = NoteRow(
        slug=COMPANY_CONTEXT_SLUG,
        type=NoteType.CONTEXT,
        title=COMPANY_CONTEXT_TITLE,
        description="Who Halvard Medical is, what it sells, and what counts as a signal.",
        status=NoteStatus.LIVING,
        created=now,
        updated=now,
        synthesis=body,
    )
    write_note(conn, note, [], changed_by="seed", reason="company context note seeded")
    return True
