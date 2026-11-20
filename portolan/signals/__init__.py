"""External signals: public information about each customer account, folded into
the graph and the weekly diff (spec sec 14 build item 7;
`01-spec/external-signals-2026-09-23.md`).

Three kinds of public source, all configured per account in `signal_targets`:

- **Feeds** (`feeds.py`): any RSS/Atom URL. In practice an RSS.app feed of the
  account's LinkedIn company page (set up by the operator in RSS.app; Portolan
  never touches LinkedIn itself) or a Google News search feed. New items are
  keyed by GUID, so a re-scan is a no-op.
- **Pages** (`web.py`): public pages on the account's own website, fetched
  politely (robots.txt, one request per host every few seconds), split into
  sections (`html_text.py`), and diffed against the last snapshot section by
  section with dates and counters ignored (`diff.py`).
- **Relevance judge** (`judge.py`): one cheap model call per new item or page
  change decides whether it matters to the vendor's account team and, if so,
  writes one sentence plus a verbatim quote. Code checks the quote and every
  number against the item before anything is written (`write.py`): a `rss`
  source whose original is the item or change record, and a `public_signal`
  claim on the account's `signal-public_signal-<account>` note, cited
  `^[source @ url:<link> <date>]`.

`scan.py` runs all of it; `python -m portolan.signals` is the CLI. Everything is
off unless `PORTOLAN_EXTERNAL_SIGNALS=true`.
"""

from __future__ import annotations

PUBLIC_SIGNAL_FACT_TYPE = "public_signal"

__all__ = ["PUBLIC_SIGNAL_FACT_TYPE"]
