"""Markdown rendering for note bodies, the `.md` download, and chat
sentences.

Pages render a note body through `render_note_body`, so it looks the same
everywhere. A `[[slug]]` pre-pass turns each wikilink into a Markdown link
(`[title](/notes/slug)`) *before* markdown-it-py runs, so there is no second
HTML-building path.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import quote as urlquote

import psycopg
from markdown_it import MarkdownIt
from markupsafe import Markup, escape

from portolan.notes.models import Claim, NoteRow
from portolan.notes.render import CITATION_RE
from portolan.util import clock as clock_module
from portolan.web.queries import note_summaries

__all__ = [
    "SIGNALS_FOLD_RE",
    "WIKILINK_RE",
    "linkify_wikilinks",
    "render_chat_sentence_text",
    "render_download_markdown",
    "render_note_body",
    "render_report_body",
]

# Kebab-case-or-underscore slugs. Looser than `portolan.notes.render.
# WIKILINK_RE`, which also requires the first character to be alphanumeric.
WIKILINK_RE = re.compile(r"\[\[([a-z0-9_-]+)\]\]")

# `html: False` is passed explicitly: claim text comes verbatim from source
# documents and reader edits, and the `commonmark` preset defaults to
# `html: True` (stored XSS). The `default` preset adds tables and
# strikethrough; its `linkify` and `typographer` options stay off.
_MD = MarkdownIt("default", {"html": False})


# A linked note's title can be arbitrary CRM/vendor text, and the `.md`
# download ships the markdown unrendered. Backslash-escaping `[ ] < >` keeps
# a title from breaking the link or becoming raw HTML in any CommonMark viewer.
_LINK_TEXT_ESCAPE_RE = re.compile(r"[\[\]<>]")


def _escape_link_text(text: str) -> str:
    return _LINK_TEXT_ESCAPE_RE.sub(lambda m: "\\" + m.group(0), text)


def _wikilink_pass(text: str, note_titles: dict[str, str], *, base_url: str = "") -> str:
    def repl(match: re.Match[str]) -> str:
        slug = match.group(1)
        title = note_titles.get(slug, slug)
        return f"[{_escape_link_text(title)}]({base_url}/notes/{slug})"

    return WIKILINK_RE.sub(repl, text)


# `portolan.notes.render.link_mentions` appends a mention it can't find
# verbatim as `([[slug]])`, so a `- [[acct]]: ...` bullet can end with a
# second link to the same slug. The backreference strips only that exact
# repeat, never an unrelated mention.
_SELF_REFERENTIAL_MENTION_RE = re.compile(
    r"^(- \[\[([a-z0-9_-]+)\]\]: .*?) \(\[\[\2\]\]\)(?=\s*(?:\^\[|$))",
    re.MULTILINE,
)


def _dedupe_self_referential_mention(text: str) -> str:
    return _SELF_REFERENTIAL_MENTION_RE.sub(r"\1", text)


def _dedupe_repeated_title(text: str, note_titles: dict[str, str]) -> str:
    """Strip a note title repeated right after its own `[[slug]]` (whose
    rendered link text is already that title), as stored reports can carry:
    `[[slug]] "title"` in Themes and `[[slug]]: title (owner:` in Actions.

    Narrow on purpose: only text byte-for-byte equal to the title resolved
    for that exact slug, immediately after the token, is removed.
    """
    for slug, title in note_titles.items():
        if not title:
            continue
        token = f"[[{slug}]]"
        if token not in text:
            continue
        esc_token = re.escape(token)
        esc_title = re.escape(title)
        # Themes: `[[slug]] "title"` -> `[[slug]]`.
        text = re.sub(rf'{esc_token} "{esc_title}"', token, text)
        # Actions: `[[slug]]: title (owner:` -> `[[slug]] (owner:` -- the
        # `(owner:` anchor keeps this from ever matching an unrelated
        # sentence that happens to start with the same words as a note's
        # title (Outliers/Contact changes never have an "(owner:" clause).
        text = re.sub(rf"{esc_token}: {esc_title} \(owner:", f"{token} (owner:", text)
    return text


# ---------------------------------------------------------------------
# Signals folds. A report body (`portolan.report.weekly._signals_fold_block`)
# wraps a block of signal bullets in `<!--signals:LABEL-->` /
# `<!--/signals-->` sentinel lines, which `_MD` (`html: False`) never passes
# through. The HTML view turns the block into a `<details
# class="signals-fold">` (`_extract_signals_folds`); the `.md` download
# drops the two sentinel lines and keeps the nested list
# (`_strip_signals_fold_markers`). A body with no such block is unchanged.
# ---------------------------------------------------------------------

SIGNALS_FOLD_RE = re.compile(
    r"<!--signals:(?P<label>[^\n]*?)-->\n(?P<body>.*?)\n<!--/signals-->",
    re.DOTALL,
)

# A top-level signal bullet in a fold (`_signals_fold_block`'s default
# `indent="  "`); quote sub-bullets are indented deeper, so the preview
# built from these lines skips them.
_TOP_SIGNAL_LINE_PREFIX = "  - "

# Wrapped in private-use-area characters (U+E000), which real text doesn't
# contain and markdown passes through inertly, so `str.replace()` finds only it.
_FOLD_TOKEN_TEMPLATE = "SIGNALSFOLD{index}"


def _strip_signals_fold_markers(text: str) -> str:
    """For the `.md` download: drop the two sentinel lines and keep the
    nested bullet list between them where it was."""
    return SIGNALS_FOLD_RE.sub(lambda m: m.group("body"), text)


# A hand edit can leave one half of a sentinel pair, which `_MD` would show
# escaped and the download would ship raw. Run after the paired pass, so any
# sentinel line left is unpaired and is dropped.
_UNPAIRED_SIGNALS_SENTINEL_RE = re.compile(
    r"^[ \t]*<!--/?signals(?::[^\n]*)?-->[ \t]*\n?", re.MULTILINE
)


def _strip_unpaired_signals_sentinels(text: str) -> str:
    return _UNPAIRED_SIGNALS_SENTINEL_RE.sub("", text)


def _extract_signals_folds(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Replace every signals block in `text` with one `  - TOKEN` line, so
    it still parses as a child item of the bullet above it.

    Returns `(text_with_placeholders, folds)`, `folds` being
    `[(token, label, raw_inner_markdown), ...]` in order; the caller
    renders each separately and substitutes the HTML for its token.
    """
    folds: list[tuple[str, str, str]] = []

    def repl(match: re.Match[str]) -> str:
        index = len(folds)
        token = _FOLD_TOKEN_TEMPLATE.format(index=index)
        folds.append((token, match.group("label").strip(), match.group("body")))
        return f"  - {token}"

    new_text = SIGNALS_FOLD_RE.sub(repl, text)
    return new_text, folds


def _fold_preview_text(raw_body: str, *, max_items: int = 3, max_len: int = 140) -> str:
    """Plain text for the fold's `title=` hover tooltip: up to `max_items`
    signal one-liners with bold, wikilink brackets and citation markers
    stripped, each truncated to `max_len` characters."""
    items: list[str] = []
    for line in raw_body.splitlines():
        if not line.startswith(_TOP_SIGNAL_LINE_PREFIX):
            continue
        text = line[len(_TOP_SIGNAL_LINE_PREFIX) :]
        text = CITATION_RE.sub("", text)
        text = WIKILINK_RE.sub(lambda m: m.group(1), text)
        text = text.replace("**", "").replace("·", "")
        text = " ".join(text.split())
        if len(text) > max_len:
            text = text[: max_len - 1].rstrip() + "…"
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return "\n".join(items)


def _render_signals_fold(
    label: str,
    raw_body: str,
    note_titles: dict[str, str],
    source_dates: dict[str, date],
    *,
    base_url: str = "",
) -> str:
    """One `<details class="signals-fold">` fragment: `label` (e.g. "6
    signals") as a `<summary class="badge">` with a plain-text hover
    preview, and the inner bullets rendered through the same citation,
    wikilink and markdown passes as the rest of the body."""
    preview = _fold_preview_text(raw_body)
    inner = _citation_pass(raw_body, source_dates, base_url=base_url)
    inner = _wikilink_pass(inner, note_titles, base_url=base_url)
    inner_html = _MD.render(inner)
    label_html = str(escape(label))
    title_attr = f' title="{escape(preview)}"' if preview else ""
    return (
        f'<details class="signals-fold"><summary class="badge"{title_attr}>'
        f"{label_html} ▾</summary>{inner_html}</details>"
    )


def _source_dates(conn: psycopg.Connection, source_slugs: list[str]) -> dict[str, date]:
    """`sources.source_date` for every slug in `source_slugs`, in one query
    (the visible text of a `_citation_pass` link)."""
    if not source_slugs:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT slug, source_date FROM sources WHERE slug = ANY(%s)", (source_slugs,))
        return dict(cur.fetchall())


def _citation_pass(text: str, source_dates: dict[str, date], *, base_url: str = "") -> str:
    """Every `^[source-slug @ locator]` marker (`portolan.notes.render.
    CITATION_RE`) as a Markdown link to `/originals/{source-slug}?at=
    {locator}`. The link text is the source's `source_date` when known,
    else the slug -- never the locator, which can be a long URL.

    `base_url` is prefixed onto the href (the `.md` download needs absolute
    links); `""` keeps it relative."""

    def repl(match: re.Match[str]) -> str:
        source_slug = match.group(1).strip()
        locator = match.group(2).strip()
        source_date = source_dates.get(source_slug)
        label = f"{source_date:%Y-%m-%d}" if source_date else source_slug
        href = f"{base_url}/originals/{source_slug}?at={urlquote(locator, safe='')}"
        return f"[{label}]({href})"

    return CITATION_RE.sub(repl, text)


def render_report_body(
    body_md: str, note_titles: dict[str, str], source_dates: dict[str, date] | None = None
) -> Markup:
    """`body_md` as safe HTML: citation markers become `/originals/...`
    links, `[[slug]]` becomes a `/notes/slug` link labelled with the note's
    title (a title repeated right after it is stripped), and the rest goes
    through markdown-it-py, which escapes text nodes.
    """
    # Dedupe runs before the citation pass: the self-referential-mention
    # lookahead expects the raw `^[...]` marker, not the rendered link.
    # Folds are pulled out first so the outer passes never touch their markers.
    pre = _dedupe_repeated_title(body_md or "", note_titles)
    pre = _dedupe_self_referential_mention(pre)
    pre, folds = _extract_signals_folds(pre)
    pre = _strip_unpaired_signals_sentinels(pre)
    pre = _citation_pass(pre, source_dates or {})
    pre = _wikilink_pass(pre, note_titles)
    html = _MD.render(pre)
    for token, label, raw_body in folds:
        fold_html = _render_signals_fold(label, raw_body, note_titles, source_dates or {})
        html = html.replace(token, fold_html)
    return Markup(html)


def render_note_body(conn: psycopg.Connection, body_md: str) -> Markup:
    """`render_report_body`, with titles and source dates looked up for
    exactly the `[[slug]]`s and citation markers in `body_md` (not
    `note.related`, which comes only from a "## Related" section).
    """
    body_md = body_md or ""
    referenced_slugs = WIKILINK_RE.findall(body_md)
    titles = {slug: info["title"] for slug, info in note_summaries(conn, referenced_slugs).items()}
    source_slugs = sorted({m.group(1).strip() for m in CITATION_RE.finditer(body_md)})
    source_dates = _source_dates(conn, source_slugs)
    return render_report_body(body_md, titles, source_dates)


def _readable_citation(source_slug: str, locator: str, *, base_url: str) -> str:
    """A readable `[source: S-call-005 @ 00:08:54](url)` link for the
    download's claims, evidence and done sections. Unlike `_citation_pass`
    (a bare date, for markers inside prose), it shows slug and locator,
    since these lists double as the provenance record."""
    href = f"{base_url}/originals/{source_slug}?at={urlquote(locator, safe='')}"
    return f"[source: {source_slug} @ {locator}]({href})"


def _render_claims_section(
    claims: list[Claim], note_titles: dict[str, str], *, base_url: str
) -> str:
    """A `## Claims` section for the download: date, fact type, claim text
    (wikilinks resolved), every citation as a readable link, quote, and
    superseded status. `""` for no claims."""
    if not claims:
        return ""
    lines = ["", "## Claims", ""]
    for claim in claims:
        text = _wikilink_pass(claim.claim_text, note_titles, base_url=base_url)
        citation = _readable_citation(claim.source_slug, claim.locator, base_url=base_url)
        extra = "".join(
            " " + _readable_citation(c.source_slug, c.locator, base_url=base_url)
            for c in claim.citations
        )
        quote = f' (quote: "{claim.quote}")' if claim.quote else ""
        line = (
            f"- **{claim.recorded:%Y-%m-%d}** | {claim.fact_type} | {text} {citation}{extra}{quote}"
        )
        if claim.superseded_at is not None:
            line = f"~~{line}~~ -- superseded {claim.superseded_at:%Y-%m-%d}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _action_why_markdown(
    why: dict[str, Any], action: dict[str, Any] | None, *, base_url: str
) -> str:
    """A card-confirmed action's download body: the same Action facts,
    Why, Evidence, What the system checked and proposed/confirmed line
    `_action_why.html` shows. `why` is `report.action_why.build_why_context`;
    `action` is the action's `queries.actions_feed` row."""
    lines: list[str] = []
    if action is not None:
        owner = action.get("owner_name") or "Unassigned"
        if action.get("owner_note_slug"):
            owner = f"[[{action['owner_note_slug']}]]"
        account = f"[[{action['account_slug']}]]" if action.get("account_slug") else "none"
        due = f"{action['due']:%Y-%m-%d}" if action.get("due") else "none"
        lines += [
            "## Action",
            "",
            f"- **What:** {action['what']}",
            f"- **Owner:** {owner}",
            f"- **Account:** {account}",
            f"- **Due:** {due}",
            f"- **Status:** {action.get('status') or 'open'}",
            "",
        ]
    lines += ["## Why", ""]
    for sentence in why.get("inference") or []:
        lines += [sentence.get("sentence") or "", ""]
    if not why.get("inference"):
        lines += ["The researcher drew no additional inference beyond the evidence below.", ""]
    lines += ["## Evidence", ""]
    for ev in why.get("evidence") or []:
        citation = _readable_citation(ev["source_slug"], ev["locator"], base_url=base_url)
        quote = f' (quote: "{ev["quote"]}")' if ev.get("quote") else ""
        lines.append(f"- {ev.get('text') or ''} {citation}{quote}")
    lines += ["", "## What the system checked", ""]
    for finding in why.get("findings") or []:
        lines.append(f"- {finding.get('plain_text')}")
    if not why.get("findings"):
        lines.append("No derived findings on this card.")
    proposed = f" on {why['proposed_at']:%Y-%m-%d}" if why.get("proposed_at") else ""
    confirmed = f" on {why['confirmed_at']:%Y-%m-%d}" if why.get("confirmed_at") else ""
    lines += [
        "",
        f"Proposed by the researcher{proposed}, confirmed by {why.get('confirmed_by')}"
        f"{confirmed}, confidence {why.get('confidence')}.",
    ]
    return "\n".join(lines)


def _action_done_markdown(done: dict[str, Any], *, base_url: str) -> str:
    """An action closed from later evidence: the same who, when and
    answering claim the note page's Done card shows
    (`report.action_done.build_done_context`)."""
    ev = done.get("evidence") or {}
    confirmed = f" on {done['confirmed_at']:%Y-%m-%d}" if done.get("confirmed_at") else ""
    said = ""
    if ev.get("speaker"):
        said = f", from what {ev['speaker']} recorded"
        if ev.get("claim_date"):
            said += f" on {ev['claim_date']}"
    citation = _readable_citation(
        ev.get("source_slug") or "", ev.get("locator") or "", base_url=base_url
    )
    quote = f' (quote: "{ev["quote"]}")' if ev.get("quote") else ""
    lines = [
        "## Done",
        "",
        f"Marked done by {done.get('confirmed_by')}{confirmed}{said}:",
        "",
        f"- {ev.get('text') or ''} {citation}{quote}",
    ]
    if done.get("finding_text"):
        lines += ["", f"What the system checked: {done['finding_text']}"]
    return "\n".join(lines)


def render_download_markdown(
    conn: psycopg.Connection,
    note: NoteRow,
    claims: list[Claim] | None = None,
    *,
    base_url: str,
    why: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    done: dict[str, Any] | None = None,
) -> str:
    """The `.md` download for one note or report: front matter (title,
    `created`, `updated`, `generated-at`), then the synthesis with every
    wikilink and citation rewritten to an absolute URL under `base_url`, so
    the file still resolves outside the app, with the same dedupe passes
    as the HTML view.

    `claims`: appended as a `## Claims` section.

    `why`/`action`: a card-confirmed action note downloads what its page
    shows (`_action_why_markdown`) instead of its synthesis, and its Claims
    section lists `why["claims"]`.

    `done`: adds a `## Done` section with the answering claim.
    """
    base_url = base_url.rstrip("/")
    if why is not None:
        body_md = _action_why_markdown(why, action, base_url=base_url)
        claims = list(why.get("claims") or [])
    else:
        body_md = note.synthesis or note.description or note.title or ""
    if done is not None:
        body_md = body_md.rstrip() + "\n\n" + _action_done_markdown(done, base_url=base_url)
    claims = claims or []
    referenced_slugs = WIKILINK_RE.findall(body_md) + [
        slug for claim in claims for slug in WIKILINK_RE.findall(claim.claim_text)
    ]
    titles = {slug: info["title"] for slug, info in note_summaries(conn, referenced_slugs).items()}
    source_slugs = sorted({m.group(1).strip() for m in CITATION_RE.finditer(body_md)})
    source_dates = _source_dates(conn, source_slugs)

    # Same order as `render_report_body`: dedupe needs the raw `^[...]` marker.
    pre = _dedupe_repeated_title(body_md, titles)
    pre = _dedupe_self_referential_mention(pre)
    pre = _strip_signals_fold_markers(pre)
    pre = _strip_unpaired_signals_sentinels(pre)
    pre = _citation_pass(pre, source_dates, base_url=base_url)
    pre = _wikilink_pass(pre, titles, base_url=base_url)
    pre = pre.rstrip() + "\n" + _render_claims_section(claims, titles, base_url=base_url)

    def _yaml_str(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    frontmatter = [
        "---",
        f"title: {_yaml_str(note.title)}",
        f"created: {note.created:%Y-%m-%d}",
        f"updated: {note.updated:%Y-%m-%d}",
        f"generated-at: {clock_module.now():%Y-%m-%d}",
        "---",
        "",
    ]
    return "\n".join(frontmatter) + pre.rstrip() + "\n"


def render_chat_sentence_text(text: str) -> Markup:
    """One chat sentence as safe, inline-only HTML (bold, italic, code
    spans, links; no `<p>` or other blocks), since each sentence sits
    inside a `<span>`. Uses the shared `_MD` (`html: False`), so literal
    HTML in model output is escaped.
    """
    return Markup(_MD.renderInline(text or ""))


def linkify_wikilinks(text: str, existing: set[str] | None = None) -> Markup:
    """A one-line, non-markdown sibling of `render_report_body` for text
    that carries literal `[[slug]]` markers (claim text stores them). The
    text is HTML-escaped first and only the `<a>` tags built here are
    unescaped, so the result is safe to use as `|safe`.

    `existing`: when given, a slug not in this set renders as plain text
    instead of a link that would 404. `None` always links.
    """
    escaped = str(escape(text or ""))

    def repl(match: re.Match[str]) -> str:
        slug = match.group(1)
        if existing is not None and slug not in existing:
            return slug
        return f'<a href="/notes/{slug}">{slug}</a>'

    return Markup(WIKILINK_RE.sub(repl, escaped))
