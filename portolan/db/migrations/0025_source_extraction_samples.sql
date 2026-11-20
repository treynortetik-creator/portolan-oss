-- Portolan schema, migration 0025: remember how many independent
-- extraction passes (`--samples N`) a source was extracted with,
-- alongside the extractor model and prompt variant migration 0023
-- already tracks.
--
-- Extraction accuracy campaign closing table (01-spec/HANDOFF-2026-09-18.
-- md): the production pick is Nova Lite, five merged samples, the
-- v6-type-precision prompt. `portolan.pipeline`'s `--samples N` (2026-09-19)
-- runs a source's whole (possibly segmented) extraction N independent
-- times and unions the results with `portolan.extract.segment.
-- merge_segment_results` -- the same dedup rule `portolan.bench.
-- extraction` already used to prove the pick on the bench. A sample-count
-- change is a real extraction-signature change: a source already done/
-- partial at samples=1 must be re-extracted, not skipped, the first time
-- a run over the same unchanged content asks for samples=5.
--
-- NOT NULL DEFAULT 1 (unlike extraction_model/extraction_prompt_variant,
-- migration 0023's nullable, no-default columns): every source that
-- already reached done/partial before this column existed really was
-- extracted with exactly one pass -- 1 is the true historical value here,
-- not an "unknown, re-extract once more" placeholder, so backfilling
-- every existing row to the column default is correct.
--
-- IDEMPOTENT: IF NOT EXISTS, safe to re-apply.

ALTER TABLE sources ADD COLUMN IF NOT EXISTS extraction_samples integer NOT NULL DEFAULT 1;

COMMENT ON COLUMN sources.extraction_samples IS
    'Number of independent extraction passes (--samples N) unioned '
    'together (portolan.extract.segment.merge_segment_results) the last '
    'time this source reached done/partial. Compared against the current '
    'run''s --samples by portolan.pipeline._prepare_one, alongside '
    'extraction_model/extraction_prompt_variant, to decide whether an '
    'unchanged source can be skipped rather than re-extracted.';
