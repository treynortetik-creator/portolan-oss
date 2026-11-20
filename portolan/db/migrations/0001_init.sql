-- Portolan schema, migration 0001: init
--
-- Spec references: 01-spec/portolan-spec-v2.md sections 2 (notes, claims,
-- citations), 3 (registry), 4 (sources and provenance), 6.2 (coverage
-- ledger), 8 (storage), 11 (metering); 01-spec/extraction-schema-v0.md;
-- 01-spec/provenance-chain.md.
--
-- ONE DATABASE PER CUSTOMER (spec sec 8). Domain tables carry no tenant
-- column. The single exception is usage_events.tenant, because the
-- metering wrapper (spec sec 11) is shared code that may, in some
-- deployments, write usage rows for more than one logical tenant into the
-- same database (e.g. a shared demo/bench database). Every other table
-- assumes "this database belongs to one customer."
--
-- TWO CLOCKS ON CLAIMS (spec sec 2). World time (valid_from/valid_to) is
-- when the fact was true out in the world. Record time (recorded/
-- superseded_at) is when Portolan learned it and, later, when a newer
-- claim replaced it. The two are independent: a claim can be recorded
-- today about something true last quarter, and a claim recorded last
-- month can be superseded today without ever having been wrong about the
-- world. This is what makes "what did we believe on 1 September" answerable
-- from recorded/superseded_at alone, regardless of valid_from/valid_to.
--
-- NOTHING IS HARD-DELETED (spec sec 8, sec 11). Corrections are new claims
-- that supersede old ones; notes move to status 'superseded' or 'archived'
-- rather than being dropped; source originals are versioned in S3, never
-- overwritten. This migration reflects that: every foreign key that could
-- otherwise cascade into removing a claim or a source uses ON DELETE
-- RESTRICT, so a destructive delete fails loudly instead of silently
-- eating provenance. The only way rows disappear is a deliberate,
-- application-level erasure path (spec sec 8), not a DB cascade.
--
-- IDEMPOTENT: every statement uses IF NOT EXISTS / OR REPLACE so this file
-- can be re-applied safely; portolan/db/__init__.py additionally tracks
-- applied migrations in schema_migrations so it only runs once per database
-- in the normal path.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- sources: one row per original (call, email, ticket, doc, CRM export,
-- chat, survey, RSS item, or a human edit) stored in S3. The source note
-- of spec sec 4. Slug is the stable id used everywhere else (claims,
-- links, coverage) to point back at "the call, email or file."
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sources (
    slug              text PRIMARY KEY,
    source_type       text NOT NULL CHECK (
                          source_type IN ('call', 'email', 'ticket', 'doc',
                                          'crm', 'chat', 'survey', 'rss', 'edit')
                      ),
    s3_key            text,
    content_hash      text,
    source_date       date,
    ingested_at       timestamptz NOT NULL DEFAULT now(),
    extraction_status text NOT NULL DEFAULT 'pending' CHECK (
                          extraction_status IN ('pending', 'filtered', 'done',
                                                 'partial', 'failed')
                      ),
    filter_reason     text,
    schema_version    text,
    attempts          integer NOT NULL DEFAULT 0,
    last_error        text,
    summary           text,
    participants      jsonb NOT NULL DEFAULT '[]'::jsonb
);

COMMENT ON TABLE sources IS
    'One row per original document (transcript, email, ticket, doc, CRM '
    'export, RSS item, or human edit) stored in S3. Link 3 of the '
    'provenance chain (provenance-chain.md): source note -> original.';

CREATE INDEX IF NOT EXISTS idx_sources_extraction_status
    ON sources (extraction_status);
CREATE INDEX IF NOT EXISTS idx_sources_source_type_date
    ON sources (source_type, source_date);

-- ---------------------------------------------------------------------
-- notes: the small, linked, cited markdown notes that are the unit of the
-- graph (spec sec 2). type is the closed list from sec 2. crm_id is
-- required on account/contact notes; parent_slug is only meaningful on
-- account notes (mirrors the CRM hierarchy, spec sec 3).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS notes (
    slug            text PRIMARY KEY,
    type            text NOT NULL CHECK (
                        type IN ('account', 'contact', 'product', 'competitor',
                                 'signal', 'theme', 'decision', 'action',
                                 'source', 'report')
                    ),
    title           text NOT NULL,
    description     text,
    status          text NOT NULL DEFAULT 'living' CHECK (
                        status IN ('living', 'superseded', 'archived')
                    ),
    created         timestamptz NOT NULL DEFAULT now(),
    updated         timestamptz NOT NULL DEFAULT now(),
    -- Absolute cutoff, not a duration: the lint (spec sec 6.1) flags a note
    -- once now() passes stale_after. Computed at write time from the
    -- per-type default (spec sec 2) unless overridden.
    stale_after     timestamptz,
    schema_version  text,
    crm_id          text,
    parent_slug     text REFERENCES notes (slug) ON DELETE RESTRICT,
    superseded_by   text REFERENCES notes (slug) ON DELETE RESTRICT,
    body_md         text NOT NULL DEFAULT '',
    version         integer NOT NULL DEFAULT 1,
    search          tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(body_md, '')), 'C')
                    ) STORED,
    CONSTRAINT notes_crm_id_required_for_account_contact CHECK (
        type NOT IN ('account', 'contact') OR crm_id IS NOT NULL
    ),
    CONSTRAINT notes_parent_only_on_account CHECK (
        parent_slug IS NULL OR type = 'account'
    )
);

COMMENT ON TABLE notes IS
    'One idea, one note, plain markdown body. type is the closed list from '
    'spec sec 2. search is a generated full-text column (title weighted '
    'above description weighted above body) backing the GIN index below.';
COMMENT ON COLUMN notes.stale_after IS
    'Absolute timestamp after which the lint flags this note as stale. '
    'Not a duration; computed from the per-type default at write time.';

CREATE INDEX IF NOT EXISTS idx_notes_search ON notes USING gin (search);
CREATE INDEX IF NOT EXISTS idx_notes_type ON notes (type);
CREATE INDEX IF NOT EXISTS idx_notes_status ON notes (status);
CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes (updated);
CREATE INDEX IF NOT EXISTS idx_notes_parent_slug ON notes (parent_slug);

-- Auto-touch updated on any write that doesn't explicitly move it itself,
-- so "updated:" stays the recency signal search relies on (spec sec 9)
-- without every call site having to remember to set it.
CREATE OR REPLACE FUNCTION portolan_touch_notes_updated_at()
RETURNS trigger AS $$
BEGIN
    IF NEW.updated IS NOT DISTINCT FROM OLD.updated THEN
        NEW.updated := now();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notes_touch_updated_at ON notes;
CREATE TRIGGER trg_notes_touch_updated_at
    BEFORE UPDATE ON notes
    FOR EACH ROW
    EXECUTE FUNCTION portolan_touch_notes_updated_at();

-- ---------------------------------------------------------------------
-- note_versions: every prior version of a note's body, kept because
-- note history lives in the database, not git (spec sec 8), and nothing
-- is hard-deleted.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS note_versions (
    slug        text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    version     integer NOT NULL,
    body_md     text NOT NULL,
    changed_at  timestamptz NOT NULL DEFAULT now(),
    changed_by  text,
    reason      text,
    PRIMARY KEY (slug, version)
);

COMMENT ON TABLE note_versions IS
    'Snapshot of a note body at every prior version. Never pruned.';

-- ---------------------------------------------------------------------
-- claims: the atomic, cited facts underneath a note's synthesis (spec
-- sec 2). fact_type is the closed signal-type list from
-- extraction-schema-v0.md. Two clocks: valid_from/valid_to (world time)
-- and recorded/superseded_at (record time) -- see header comment.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_slug       text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    fact_type       text NOT NULL CHECK (
                        fact_type IN ('complaint', 'request', 'praise', 'risk',
                                      'competitor_mention', 'commitment',
                                      'decision', 'contact_change', 'pricing',
                                      'usage', 'question', 'other')
                    ),
    claim_text      text NOT NULL,
    source_slug     text NOT NULL REFERENCES sources (slug) ON DELETE RESTRICT,
    locator         text NOT NULL,
    quote           text,
    speaker         text,
    valid_from      date,
    valid_to        date,
    recorded        timestamptz NOT NULL DEFAULT now(),
    superseded_at   timestamptz,
    superseded_by   bigint REFERENCES claims (id) ON DELETE RESTRICT,
    severity        text,
    sentiment       smallint CHECK (sentiment BETWEEN -2 AND 2),
    confidence      numeric(4, 3) CHECK (confidence BETWEEN 0 AND 1),
    archived        boolean NOT NULL DEFAULT false,
    -- FK to edit_records added after that table exists (see below); the
    -- two tables reference each other (an edit can touch a claim, and a
    -- claim can carry the edit that last touched it).
    human_edit_id   bigint
);

COMMENT ON TABLE claims IS
    'Atomic, cited facts. fact_type is the closed signal-type list from '
    'extraction-schema-v0.md. See header comment for the two clocks.';
COMMENT ON COLUMN claims.locator IS
    'Required (extraction-schema-v0.md rule 4): a claim with no locator is '
    'rejected at merge and never reaches this table.';

CREATE INDEX IF NOT EXISTS idx_claims_note_slug ON claims (note_slug);
CREATE INDEX IF NOT EXISTS idx_claims_source_slug ON claims (source_slug);
CREATE INDEX IF NOT EXISTS idx_claims_fact_type ON claims (fact_type);
CREATE INDEX IF NOT EXISTS idx_claims_archived ON claims (archived);

-- ---------------------------------------------------------------------
-- registry_entities / registry_aliases: the list of things that exist
-- (spec sec 3), built from the CRM and re-synced nightly. note_slug links
-- an entity to its account/contact note once one has been written.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS registry_entities (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind            text NOT NULL CHECK (
                        kind IN ('account', 'contact', 'product', 'competitor',
                                 'internal_person', 'other')
                    ),
    canonical_name  text NOT NULL,
    crm_id          text,
    parent_id       bigint REFERENCES registry_entities (id) ON DELETE RESTRICT,
    account_kind    text,
    email           text,
    email_domain    text,
    note_slug       text REFERENCES notes (slug) ON DELETE RESTRICT,
    active          boolean NOT NULL DEFAULT true,
    synced_at       timestamptz
);

COMMENT ON TABLE registry_entities IS
    'Accounts, contacts, products, competitors and internal people. Source '
    'of truth is the CRM (spec sec 3); parent_id mirrors the CRM hierarchy.';

CREATE INDEX IF NOT EXISTS idx_registry_entities_kind ON registry_entities (kind);
CREATE INDEX IF NOT EXISTS idx_registry_entities_crm_id ON registry_entities (crm_id);
CREATE INDEX IF NOT EXISTS idx_registry_entities_email_domain
    ON registry_entities (email_domain);
CREATE INDEX IF NOT EXISTS idx_registry_entities_parent_id
    ON registry_entities (parent_id);

CREATE TABLE IF NOT EXISTS registry_aliases (
    alias      text NOT NULL,
    entity_id  bigint NOT NULL REFERENCES registry_entities (id) ON DELETE RESTRICT,
    source     text NOT NULL CHECK (source IN ('crm', 'tracker', 'human', 'extractor')),
    created    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (alias, entity_id)
);

COMMENT ON TABLE registry_aliases IS
    'Alternate names for a registry entity: CRM alt names, Gong/Clari '
    'tracker vocabulary, human-mapped names, or names the extractor '
    'proposed. A human maps a name once and it sticks (spec sec 3).';

CREATE INDEX IF NOT EXISTS idx_registry_aliases_alias ON registry_aliases (alias);

-- ---------------------------------------------------------------------
-- review_queue: unknown entities, ambiguous mentions, failed
-- consolidations and anything else that needs a human (spec sec 3, 6.3).
-- Facts are never dropped; they park here instead.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_queue (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind         text NOT NULL CHECK (
                     kind IN ('unknown_entity', 'ambiguous_entity',
                              'failed_consolidation', 'other')
                 ),
    payload      jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_slug  text REFERENCES sources (slug) ON DELETE RESTRICT,
    status       text NOT NULL DEFAULT 'open' CHECK (
                     status IN ('open', 'resolved', 'dismissed')
                 ),
    created      timestamptz NOT NULL DEFAULT now(),
    resolved_at  timestamptz,
    resolved_by  text
);

COMMENT ON TABLE review_queue IS
    'Human-in-the-loop queue for unknown/ambiguous entities and failed '
    'consolidations (spec sec 3, sec 6.3). Nothing here is auto-resolved.';

CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue (status);

-- ---------------------------------------------------------------------
-- links: the graph's edges (spec sec 2). from_slug is always a note; the
-- provenance edge kind points at a source (a different slug namespace),
-- so to_slug intentionally carries no single-table foreign key -- see
-- comment below.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS links (
    from_slug  text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    to_slug    text NOT NULL,
    kind       text NOT NULL CHECK (kind IN ('body', 'provenance', 'supersession', 'parent')),
    PRIMARY KEY (from_slug, to_slug, kind)
);

COMMENT ON TABLE links IS
    'Graph edges (spec sec 2): body = explicit [[wikilink]] in the note '
    'text (to_slug is a note); provenance = claim -> source note (to_slug '
    'is a sources.slug); supersession = superseded_by; parent = account '
    'hierarchy edge (to_slug is a note in both of the latter two cases). '
    'to_slug deliberately has no single FK because its target table '
    'depends on kind; the lint (spec sec 6.1) is what checks it resolves.';

CREATE INDEX IF NOT EXISTS idx_links_to_slug ON links (to_slug);
CREATE INDEX IF NOT EXISTS idx_links_kind ON links (kind);

-- ---------------------------------------------------------------------
-- chunks: retrieval units for fused keyword + vector search (spec sec 9).
-- embedding dimension matches Titan Text Embeddings V2 configured at 1024
-- (spec sec 5.1, sec 8).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_slug    text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    heading      text,
    text         text NOT NULL,
    token_count  integer,
    embedding    vector(1024),
    search       tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
);

COMMENT ON TABLE chunks IS
    'Chunked note text plus its embedding, re-embedded incrementally by '
    'content hash (spec sec 6.1) whenever the owning note changes.';

CREATE INDEX IF NOT EXISTS idx_chunks_note_slug ON chunks (note_slug);
CREATE INDEX IF NOT EXISTS idx_chunks_search ON chunks USING gin (search);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- coverage_ledger: per connector, per day (spec sec 6.2). Feeds the
-- weekly report and the "is the filter lying to us" sample.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coverage_ledger (
    connector  text NOT NULL,
    day        date NOT NULL,
    seen       integer NOT NULL DEFAULT 0,
    filtered   jsonb NOT NULL DEFAULT '{}'::jsonb,
    extracted  integer NOT NULL DEFAULT 0,
    partial    integer NOT NULL DEFAULT 0,
    failed     integer NOT NULL DEFAULT 0,
    sampled    integer NOT NULL DEFAULT 0,
    PRIMARY KEY (connector, day)
);

COMMENT ON TABLE coverage_ledger IS
    'Daily per-connector counts: seen, filtered (reason -> count), '
    'extracted, partial, failed, sampled for audit (spec sec 6.2).';

-- ---------------------------------------------------------------------
-- usage_events: the metering wrapper's write path (spec sec 11). This is
-- the one table with a tenant column, because the metering code is
-- shared across customer databases even though the rest of the schema is
-- one-database-per-customer.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS usage_events (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant         text NOT NULL,
    surface        text NOT NULL CHECK (
                       surface IN ('extract', 'merge', 'consolidate', 'audit',
                                   'chat', 'report', 'embed')
                   ),
    model_key      text NOT NULL,
    input_tokens   integer NOT NULL DEFAULT 0,
    output_tokens  integer NOT NULL DEFAULT 0,
    cached_tokens  integer NOT NULL DEFAULT 0,
    cost_estimate  numeric(12, 6),
    request_id     text,
    created        timestamptz NOT NULL DEFAULT now(),
    extra          jsonb NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE usage_events IS
    'Every model call, written by the one shared metering wrapper (spec '
    'sec 11): tenant, surface, model, tokens, cost. The only table with a '
    'tenant column in this otherwise one-database-per-customer schema.';

CREATE INDEX IF NOT EXISTS idx_usage_events_tenant_created
    ON usage_events (tenant, created);
CREATE INDEX IF NOT EXISTS idx_usage_events_surface ON usage_events (surface);

-- ---------------------------------------------------------------------
-- edit_records: human edits are sources (spec sec 4). Rendered in-note as
-- ^[edit:name date]; reapplied on rebuild.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS edit_records (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_slug   text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    claim_id    bigint REFERENCES claims (id) ON DELETE RESTRICT,
    editor      text NOT NULL,
    edited_at   timestamptz NOT NULL DEFAULT now(),
    before_text text,
    after_text  text,
    reason      text
);

COMMENT ON TABLE edit_records IS
    'A human edit to a note or claim in the UI: who, when, before, after, '
    'why (spec sec 4). Reapplied on rebuild; the lint flags an edit later '
    'contradicted by a fresh extraction.';

CREATE INDEX IF NOT EXISTS idx_edit_records_note_slug ON edit_records (note_slug);

-- Review finding F7: a bare ALTER TABLE ... ADD CONSTRAINT is not
-- idempotent (`ADD CONSTRAINT` has no `IF NOT EXISTS` form in Postgres),
-- contradicting this file's own "IDEMPOTENT" header claim above -- verified
-- by applying this file twice: the second run raised
-- `constraint "claims_human_edit_id_fkey" for relation "claims" already
-- exists`. Guarded the same way Postgres recommends for anything without a
-- native IF NOT EXISTS: check pg_constraint first.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'claims_human_edit_id_fkey'
    ) THEN
        ALTER TABLE claims
            ADD CONSTRAINT claims_human_edit_id_fkey
            FOREIGN KEY (human_edit_id) REFERENCES edit_records (id) ON DELETE RESTRICT;
    END IF;
END $$;

-- ---------------------------------------------------------------------
-- actions: an action note's structured half (extraction-schema-v0.md).
-- Not a fact; survives merge only if it dedups against open actions on
-- the same entity.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS actions (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    note_slug        text NOT NULL REFERENCES notes (slug) ON DELETE RESTRICT,
    what             text NOT NULL,
    owner_entity_id  bigint REFERENCES registry_entities (id) ON DELETE RESTRICT,
    due              date,
    status           text NOT NULL DEFAULT 'open' CHECK (
                         status IN ('open', 'done', 'overdue', 'dropped')
                     ),
    source_slug      text NOT NULL REFERENCES sources (slug) ON DELETE RESTRICT,
    locator          text
);

COMMENT ON TABLE actions IS
    'Structured fields for an action note (extraction-schema-v0.md): what, '
    'owner, due date, status, and the source it came from.';

CREATE INDEX IF NOT EXISTS idx_actions_note_slug ON actions (note_slug);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions (status);

-- ---------------------------------------------------------------------
-- changelog: the dated, mechanical record of what happened to the graph
-- (spec sec 5.1 write step, sec 6.1 lint, sec 7 weekly report feeds).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS changelog (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at timestamptz NOT NULL DEFAULT now(),
    kind        text NOT NULL,
    summary     text NOT NULL,
    details     jsonb NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE changelog IS
    'Append-only, dated record of graph events. Feeds the weekly report''s '
    'diff feed (spec sec 7) and the lint report (spec sec 6.1).';

CREATE INDEX IF NOT EXISTS idx_changelog_occurred_at ON changelog (occurred_at);
