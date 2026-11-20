-- Portolan schema, migration 0021: stamp the company context version a
-- claim was extracted under.
--
-- Owner decision 2026-09-17 ("Company context page"): the company
-- context note (migration 0020) can change over time -- a human edits
-- it, or it's re-seeded -- so a claim extracted under one version of it
-- should say which version that was, the same way `claims.schema_version`
-- (well, `notes.schema_version` -- claims carry no schema_version of
-- their own today) already lets a reader tell which extraction schema
-- produced a fact. `company_context_updated_at` mirrors the context
-- note's own `notes.updated` at the moment the pipeline run that
-- extracted this claim loaded it (`portolan.context.load_company_
-- context`); NULL for every claim extracted before this feature existed,
-- or when no context note existed at extraction time.
--
-- Nullable, no default, no backfill: this is provenance metadata, not a
-- fact about the world -- an absent value means "unknown", not "no
-- context existed."

ALTER TABLE claims ADD COLUMN IF NOT EXISTS company_context_updated_at timestamptz;

COMMENT ON COLUMN claims.company_context_updated_at IS
    'The company-context note''s notes.updated at the moment this claim '
    'was extracted (portolan.context.load_company_context) -- NULL when '
    'no context note existed yet, or for a claim extracted before this '
    'column existed. Lets a later reader tell which version of the '
    'company context a fact was extracted under.';
