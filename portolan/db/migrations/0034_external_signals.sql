-- Portolan schema, migration 0034: external signals (spec sec 14 build item 7,
-- "scheduled re-crawl of external sources with per-node diff";
-- 01-spec/external-signals-2026-09-23.md).
--
-- signal_targets: what to watch per account. A `feed` is any RSS/Atom URL
-- (an RSS.app feed of the account's LinkedIn company page, an RSS.app news
-- feed, the account's own newsroom feed); a `page` is one public web page
-- scanned and diffed against its last snapshot. account_crm_id is the
-- registry's external id (registry_entities.crm_id), the same key every other
-- CRM-sourced row uses.
--
-- LinkedIn is company pages only (the maintainer, 2026-09-23): a `linkedin` feed must
-- name the company page it follows in source_page_url, and the CHECK refuses
-- anything but https://[www.]linkedin.com/company/<name>. Personal profiles are
-- never monitored.
--
-- page_snapshots: one row per page scan. sections holds the extracted,
-- sectioned text the diff compares; html_key names the raw HTML in the
-- originals store. content_hash is over the normalised text, so a page whose
-- only change is a date or a counter hashes the same.
--
-- signal_items: one row per unit the relevance judge reads: a new feed item
-- (item_key = its GUID) or a page change between two snapshots. The unique
-- key makes a re-scan idempotent. A relevant item points at the source it
-- became and (signal_item_claims) the claims it produced; a dropped one keeps
-- its verdict, so nothing is judged twice and nothing is dropped silently.
--
-- IDEMPOTENT: IF NOT EXISTS throughout. Additive only: four new tables.

CREATE TABLE IF NOT EXISTS signal_targets (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    account_crm_id   text NOT NULL,
    kind             text NOT NULL CHECK (kind IN ('feed', 'page')),
    url              text NOT NULL,
    label            text,
    feed_kind        text CHECK (feed_kind IN ('linkedin', 'news', 'other')),
    source_page_url  text,
    fetch_mode       text NOT NULL DEFAULT 'http' CHECK (fetch_mode IN ('http', 'browser')),
    active           boolean NOT NULL DEFAULT true,
    created          timestamptz NOT NULL DEFAULT now(),
    last_checked_at  timestamptz,
    last_error       text,
    CONSTRAINT signal_targets_account_kind_url_uq UNIQUE (account_crm_id, kind, url),
    CONSTRAINT signal_targets_linkedin_company_only CHECK (
        feed_kind IS DISTINCT FROM 'linkedin'
        OR source_page_url ~ '^https://(www\.)?linkedin\.com/company/[^/?#]+/?'
    )
);

COMMENT ON TABLE signal_targets IS
    'Public sources watched per account (external signals): RSS/Atom feeds and '
    'web pages. account_crm_id is registry_entities.crm_id.';

CREATE TABLE IF NOT EXISTS page_snapshots (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    target_id     bigint NOT NULL REFERENCES signal_targets (id) ON DELETE RESTRICT,
    url           text NOT NULL,
    fetched_at    timestamptz NOT NULL,
    http_status   integer,
    fetch_method  text NOT NULL CHECK (fetch_method IN ('http', 'browser')),
    content_hash  text NOT NULL,
    html_key      text,
    title         text,
    sections      jsonb NOT NULL DEFAULT '[]'::jsonb,
    needs_browser boolean NOT NULL DEFAULT false
);

COMMENT ON TABLE page_snapshots IS
    'One scan of one watched page: extracted sections (what the diff compares), '
    'a hash over their normalised text, and the store key of the raw HTML.';

CREATE INDEX IF NOT EXISTS idx_page_snapshots_target_fetched
    ON page_snapshots (target_id, fetched_at DESC);

CREATE TABLE IF NOT EXISTS signal_items (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    target_id             bigint NOT NULL REFERENCES signal_targets (id) ON DELETE RESTRICT,
    account_crm_id        text NOT NULL,
    item_kind             text NOT NULL CHECK (item_kind IN ('feed_item', 'page_change')),
    item_key              text NOT NULL,
    url                   text NOT NULL,
    title                 text,
    publisher             text,
    published_at          timestamptz,
    observed_at           timestamptz NOT NULL,
    body                  text NOT NULL,
    details               jsonb NOT NULL DEFAULT '{}'::jsonb,
    snapshot_id           bigint REFERENCES page_snapshots (id) ON DELETE RESTRICT,
    previous_snapshot_id  bigint REFERENCES page_snapshots (id) ON DELETE RESTRICT,
    status                text NOT NULL DEFAULT 'pending' CHECK (
                              status IN ('pending', 'relevant', 'not_relevant',
                                         'rejected', 'error')
                          ),
    category              text,
    judge_reason          text,
    judge_model           text,
    judged_at             timestamptz,
    attempts              integer NOT NULL DEFAULT 0,
    source_slug           text REFERENCES sources (slug) ON DELETE RESTRICT,
    created               timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT signal_items_account_kind_key_uq UNIQUE (account_crm_id, item_kind, item_key)
);

COMMENT ON TABLE signal_items IS
    'Each new feed item or page change the relevance judge read, with its '
    'verdict; a relevant one links to the source it became (and, through '
    'signal_item_claims, the claims it produced).';

CREATE INDEX IF NOT EXISTS idx_signal_items_status ON signal_items (status);
CREATE INDEX IF NOT EXISTS idx_signal_items_judged_at ON signal_items (judged_at);

-- signal_item_claims: the claims one relevant item produced. A page change can
-- carry several events (two new press releases); the same event from two
-- items (a release and its translation) is one claim linked from both.
CREATE TABLE IF NOT EXISTS signal_item_claims (
    item_id   bigint NOT NULL REFERENCES signal_items (id) ON DELETE RESTRICT,
    claim_id  bigint NOT NULL REFERENCES claims (id) ON DELETE RESTRICT,
    category  text NOT NULL,
    PRIMARY KEY (item_id, claim_id)
);

CREATE INDEX IF NOT EXISTS idx_signal_item_claims_claim_id ON signal_item_claims (claim_id);
