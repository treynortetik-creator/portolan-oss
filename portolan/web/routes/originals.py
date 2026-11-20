"""`/originals/{slug}`: the raw-document viewer with seek-to-locator
(provenance-chain.md link 4). Bytes come from `Store.get(s3_key)`; every
open is logged (portolan-spec-v2.md sec 12).

Rather than re-implement each source type's locator convention, this
rebuilds a `RawSource` from the stored bytes and calls
`portolan.ingest.prepare.prepare_text`, rendering its locator spans as
anchors, so a `?at=` link matches a span by construction. A CRM row's
locator also needs the connector's metadata (the export's path and the
row's id), read from `sources.metadata["raw"]`. PDFs, chat and survey
sources render without per-block anchors.
"""

from __future__ import annotations

import json
import re
from datetime import date
from email import message_from_string
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from portolan.activity import record_activity
from portolan.config import Settings
from portolan.extract.schema import SourceType
from portolan.ingest.models import RawSource
from portolan.ingest.prepare import prepare_text
from portolan.store import Store
from portolan.web.auth import Session, log_usage_event, require_session
from portolan.web.deps import get_conn, get_settings_dep, get_store_dep
from portolan.web.tracking import track_page_view

router = APIRouter()

# Doc locators (`prepare.py` `_prepare_doc`) are `p<page>¶<para>` or
# `L<line>`; only the page form gives a PDF page to jump to.
_PDF_PAGE_LOCATOR_RE = re.compile(r"^p(\d+)")

# A call line starts with its own `[HH:MM:SS]`, which is also its locator;
# the template's gutter already prints the locator, so it is stripped when
# it matches this block's locator exactly.
_LEADING_TIMESTAMP_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*")


def _load_source_row(conn: psycopg.Connection, slug: str) -> dict[str, Any] | None:
    # Postgres text cannot hold NUL, so a `%00` slug would raise DataError;
    # no real slug contains one.
    if "\x00" in slug:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT slug, source_id, source_type, s3_key, source_date, participants, summary,
                   metadata
            FROM sources WHERE slug = %s
            """,
            (slug,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "slug": row[0],
        # The connector's unversioned id (migration 0022); `slug` if null.
        "source_id": row[1] or row[0],
        "source_type": row[2],
        "s3_key": row[3],
        "source_date": row[4],
        "participants": row[5],
        "summary": row[6],
        "metadata": row[7] or {},
    }


def _split_email(text: str) -> tuple[dict[str, str], str]:
    """The RFC 2822 header block as a dict, plus the body after it, parsed
    with the stdlib `email` package."""
    message = message_from_string(text)
    headers = {str(key): str(value) for key, value in message.items()}
    if not headers:
        # No header block: the whole text is body.
        return {}, text
    body = message.get_payload()
    return headers, body if isinstance(body, str) else text.partition("\n\n")[2]


def _pdf_page_from_locator(at: str | None) -> int | None:
    """The page number a `p<page>¶<para>` locator names, or `None` when `at`
    is unset or a line locator (`L<n>`). The template turns it into
    `#page=N`, the fragment browser PDF viewers honour.
    """
    if not at:
        return None
    match = _PDF_PAGE_LOCATOR_RE.match(at)
    return int(match.group(1)) if match else None


def _blocks_for(raw: RawSource) -> list[dict[str, str]]:
    prepared = prepare_text(raw)
    blocks = []
    for span in prepared.locators:
        text = prepared.text[span.start : span.end].strip()
        match = _LEADING_TIMESTAMP_RE.match(text)
        if match and match.group(1) == span.locator:
            text = text[match.end() :]
        else:
            # `prepare_text` prefixes ticket, email, doc and RSS blocks with
            # `[<locator>] `; drop exactly this block's marker.
            marker = f"[{span.locator}]"
            if text.startswith(marker):
                text = text[len(marker) :].lstrip()
        blocks.append({"locator": span.locator, "text": text})
    return blocks


def _crm_view(source_row: dict[str, Any], source_date: date, text: str) -> dict[str, Any]:
    """One CRM row: its fields as `key=value` lines (the form the file
    connector's text gave the extractor) in one block at the row's locator.

    The stored row is JSON with sorted keys, so the fields are in key order,
    not the export's column order the extractor read: a quote spanning
    several fields is on the page field by field, not as one string
    (`scripts/verify.py` `row_quote_match`).
    """
    try:
        row = json.loads(text)
    except json.JSONDecodeError:
        row = None
    if not isinstance(row, dict):
        return {"kind": "generic", "text": text}
    fields = [
        (str(key), value if isinstance(value, str) else json.dumps(value, sort_keys=True))
        for key, value in row.items()
    ]
    connector_metadata = (source_row.get("metadata") or {}).get("raw") or {}
    raw = RawSource(
        source_id=source_row["source_id"],
        source_type=SourceType.CRM,
        source_date=source_date,
        text=", ".join(f"{key}={value}" for key, value in fields),
        metadata=connector_metadata,
    )
    [span] = prepare_text(raw).locators
    return {
        "kind": "crm",
        "locator": span.locator,
        "object": connector_metadata.get("object"),
        "fields": fields,
    }


def _build_view(source_row: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
    # Rebuild from the unversioned `source_id`, not a `--v<n>` slug:
    # extraction saw that id, and ticket locators (and an email's, when it
    # has no Message-ID) embed it.
    source_id = source_row["source_id"]
    source_type = SourceType(source_row["source_type"])
    source_date = source_row["source_date"] or date.today()
    text = raw_bytes.decode("utf-8", errors="replace")

    if source_type == SourceType.CALL:
        raw = RawSource(
            source_id=source_id, source_type=source_type, source_date=source_date, text=text
        )
        return {"kind": "call", "blocks": _blocks_for(raw)}

    if source_type == SourceType.EMAIL:
        headers, body = _split_email(text)
        message_id = headers.get("Message-ID", source_id).strip("<>")
        raw = RawSource(
            source_id=source_id,
            source_type=source_type,
            source_date=source_date,
            text=body,
            metadata={"message_id": message_id},
        )
        return {"kind": "email", "headers": headers, "blocks": _blocks_for(raw)}

    if source_type == SourceType.TICKET:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        comments = data.get("comments") or []
        raw = RawSource(
            source_id=source_id,
            source_type=source_type,
            source_date=source_date,
            text=text,
            metadata={"comments": comments},
        )
        return {
            "kind": "ticket",
            "subject": data.get("subject"),
            "requester": data.get("requester"),
            "account": data.get("account"),
            "blocks": _blocks_for(raw),
        }

    if source_type == SourceType.RSS:
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            item = {}
        raw = RawSource(
            source_id=source_id,
            source_type=source_type,
            source_date=source_date,
            text=item.get("summary") or item.get("title") or "",
            metadata={"link": item.get("link", "")},
        )
        return {"kind": "rss", "item": item, "blocks": _blocks_for(raw)}

    if source_type == SourceType.DOC:
        if raw_bytes[:5] == b"%PDF-":
            return {"kind": "pdf"}
        raw = RawSource(
            source_id=source_id, source_type=source_type, source_date=source_date, text=text
        )
        return {"kind": "doc", "blocks": _blocks_for(raw)}

    if source_type == SourceType.CRM:
        return _crm_view(source_row, source_date, text)

    # chat and survey: the raw text as one block. A `source_type` outside
    # `SourceType` (the schema also allows 'edit') raises ValueError above.
    return {"kind": "generic", "text": text}


def context_snippet(
    conn: psycopg.Connection, store: Store, slug: str, locator: str
) -> dict[str, str | None] | None:
    """The block at `locator` plus the blocks before and after it, for a
    review card; the same blocks `original_detail` renders.

    `None` when there is no such source, no stored original, no per-block
    view (CRM, PDF, chat, survey), or no block at `locator`. Not a
    catch-all: an error from `_build_view` (an unknown `source_type`, a
    `prepare_text` failure) propagates to the caller.
    """
    source_row = _load_source_row(conn, slug)
    if source_row is None or not source_row["s3_key"]:
        return None
    try:
        raw_bytes = store.get(source_row["s3_key"])
    except OSError:
        return None
    view = _build_view(source_row, raw_bytes)
    blocks = view.get("blocks")
    if not blocks:
        return None
    idx = next((i for i, b in enumerate(blocks) if b["locator"] == locator), None)
    if idx is None:
        return None
    return {
        "before": blocks[idx - 1]["text"] if idx > 0 else None,
        "at": blocks[idx]["text"],
        "after": blocks[idx + 1]["text"] if idx + 1 < len(blocks) else None,
    }


@router.get("/originals/{slug}", dependencies=[Depends(track_page_view)])
def original_detail(
    request: Request,
    slug: str,
    at: str | None = None,
    from_: str | None = Query(None, alias="from"),
    m: str | None = None,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    store: Store = Depends(get_store_dep),
    settings: Settings = Depends(get_settings_dep),
) -> object:
    """`?from=chat&m=<message id>` marks a click from a chat answer's
    "Open the original" chip (`_chat_sentence.html`); it is recorded as a
    `citation_click` for the pilot dashboard."""
    templates = request.app.state.templates
    source_row = _load_source_row(conn, slug)
    if source_row is None:
        raise HTTPException(status_code=404, detail=f"no such source: {slug}")
    if not source_row["s3_key"]:
        raise HTTPException(status_code=404, detail=f"no original stored for {slug}")

    try:
        raw_bytes = store.get(source_row["s3_key"])
    except OSError as exc:
        raise HTTPException(status_code=404, detail="original not found in store") from exc

    view = _build_view(source_row, raw_bytes)

    # Access log on every original opened (spec sec 12), zero tokens spent
    # -- never the note/claim/document text itself, only the slug.
    log_usage_event(
        conn,
        tenant=settings.tenant,
        surface="chat",
        model_key="original_view",
        extra={"opened": slug},
    )
    if from_ == "chat":
        message_id = int(m) if m and m.isdigit() else None
        record_activity(
            conn,
            kind="citation_click",
            user_name=session.editor,
            route="/originals/{slug}",
            target=slug,
            detail={"message_id": message_id},
        )

    if view["kind"] == "pdf":
        return templates.TemplateResponse(
            request,
            "original_pdf.html",
            {"slug": slug, "at": at, "page": _pdf_page_from_locator(at)},
        )

    return templates.TemplateResponse(
        request,
        "original.html",
        {"slug": slug, "source": source_row, "view": view, "at": at},
    )


@router.get("/originals/{slug}/raw")
def original_raw_bytes(
    slug: str,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
    store: Store = Depends(get_store_dep),
) -> Response:
    """The stored original's bytes, for the PDF `<iframe>` (`#page=N` needs
    a real URL). A missing stored file is a 404. `nosniff` because the
    content type is guessed from the first five bytes.
    """
    source_row = _load_source_row(conn, slug)
    if source_row is None or not source_row["s3_key"]:
        raise HTTPException(status_code=404, detail=f"no such source: {slug}")
    try:
        raw_bytes = store.get(source_row["s3_key"])
    except OSError as exc:
        raise HTTPException(status_code=404, detail="original not found in store") from exc
    media_type = "application/pdf" if raw_bytes[:5] == b"%PDF-" else "application/octet-stream"
    return Response(
        content=raw_bytes,
        media_type=media_type,
        headers={"X-Content-Type-Options": "nosniff"},
    )


__all__ = ["context_snippet", "router"]
