-- Portolan schema, migration 0023: remember which extractor model and
-- prompt variant a source was actually extracted with.
--
-- Bug hunt 2026-09-18, finding 3 (P1): re-running the pipeline over an
-- unchanged corpus re-extracted (and re-billed) every source, every time
-- -- `intake()` correctly returned the existing row for unchanged
-- content, but nothing checked whether it was already `done`/`partial`
-- before `apply_filter_decision` reset it back to `pending` and
-- extraction ran again. A source is now skipped when its stored content
-- hash (implicit: `intake()` only returns an existing row for byte-
-- identical content), extractor model, and prompt variant all match what
-- it was already extracted with (`--force` re-extracts regardless).
--
-- Nullable, no default, no backfill: NULL means "extracted before this
-- column existed, or never extracted" -- either way, unknown, not "no
-- signature applies" -- so a source last extracted before this migration
-- is re-extracted (and gets a real signature) exactly once, the same
-- conservative default every other bi-temporal/provenance column in this
-- schema uses (see migration 0021's claims.company_context_updated_at).
--
-- IDEMPOTENT: IF NOT EXISTS throughout, safe to re-apply.

ALTER TABLE sources ADD COLUMN IF NOT EXISTS extraction_model text;
ALTER TABLE sources ADD COLUMN IF NOT EXISTS extraction_prompt_variant text;

COMMENT ON COLUMN sources.extraction_model IS
    'settings.extraction_model as it stood the last time this source '
    'reached done/partial. Compared against the current run''s model on '
    're-crawl (portolan.pipeline._prepare_one) to decide whether an '
    'unchanged source can be skipped rather than re-extracted.';

COMMENT ON COLUMN sources.extraction_prompt_variant IS
    'The portolan.extract.prompt.build_system_prompt variant used the '
    'last time this source reached done/partial. Same skip check as '
    'extraction_model, above.';
