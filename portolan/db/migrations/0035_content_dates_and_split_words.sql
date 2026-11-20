-- Portolan schema, migration 0035: a note's content date, and keyword search
-- that splits slash- and dot-joined words (2026-09-25).
--
-- 1. Content date (01-spec/gbrain-borrow-knowledge-2026-09-23.md, rank 2;
--    01-spec/bench-graph-retrieval-2026-09-23.md on feature/graph-retrieval).
--    `search(since=)` and the recency weight read `notes.updated`, a write
--    time: on a graph loaded in one sitting every note looks new.
--    - notes.content_date_claims: the newest `recorded` (the source's own
--      date, `portolan.merge.util.recorded_dt`) among the note's live claims
--      (not superseded, not archived); NULL when it has none. Kept current by
--      trg_claims_content_date on every claims write path.
--    - notes.content_date: content_date_claims, else `updated` (the fallback).
--    - notes.content_date_source: 'claims_recorded' or 'note_updated', so a
--      guessed date can be counted.
--    A content-date refresh is not a write: portolan_touch_notes_updated_at
--    now leaves `updated` alone when only content_date_claims changed.
--
-- 2. Keyword words. The `english` parser reads "Link/Epic" as one file-path
--    token (lexeme `link/epic`) and "Link.Epic" as one host token, so a note
--    never matched "epic". portolan_search_text() turns `/`, `\` and a dot
--    between two letters into spaces before to_tsvector, on both generated
--    columns and on the query (`portolan.search.query.QUERY_TSVECTOR_SQL`).
--    Hyphens need nothing: the parser already emits the whole word and its
--    parts. Decimals ("v2.1", "3.5") keep their dot.
--
-- destructive-ok: the DROPs remove only derived data re-created in the same
-- statement (the generated notes.search/chunks.search columns, recomputed
-- from title/description/body_md/text) and a trigger re-created below; the
-- web startup's pre-migration pg_dump (db.migrate_with_backup) keeps the rest.
--
-- Rebuild: re-adding notes.search and chunks.search recomputes every row
-- (a table rewrite, indexes included), and the UPDATE below fills
-- content_date_claims, all inside this migration. Small at today's size; on
-- a database the size of portolan_erb the chunks rewrite rebuilds its HNSW
-- index too.

CREATE OR REPLACE FUNCTION portolan_search_text(t text)
RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT pg_catalog.regexp_replace(
        t, '[/\\]|(?<=[[:alpha:]])\.(?=[[:alpha:]])', ' ', 'g'
    )
$$;

COMMENT ON FUNCTION portolan_search_text(text) IS
    'Text as the keyword index reads it: / and \ and a dot between two '
    'letters become spaces, so slash- and dot-joined words index separately. '
    'Used by notes.search, chunks.search and the query side of search().';

ALTER TABLE notes
    DROP COLUMN IF EXISTS search,
    ADD COLUMN IF NOT EXISTS search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', portolan_search_text(coalesce(title, ''))), 'A') ||
        setweight(to_tsvector('english', portolan_search_text(coalesce(description, ''))), 'B') ||
        setweight(to_tsvector('english', portolan_search_text(coalesce(body_md, ''))), 'C')
    ) STORED,
    ADD COLUMN IF NOT EXISTS content_date_claims timestamptz,
    ADD COLUMN IF NOT EXISTS content_date timestamptz GENERATED ALWAYS AS (
        coalesce(content_date_claims, updated)
    ) STORED,
    ADD COLUMN IF NOT EXISTS content_date_source text GENERATED ALWAYS AS (
        CASE WHEN content_date_claims IS NULL THEN 'note_updated' ELSE 'claims_recorded' END
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_notes_search ON notes USING gin (search);
CREATE INDEX IF NOT EXISTS idx_notes_content_date ON notes (content_date);

COMMENT ON COLUMN notes.content_date_claims IS
    'Newest claims.recorded among this note''s live claims (superseded_at IS NULL, '
    'not archived); NULL when it has none. Maintained by trg_claims_content_date.';
COMMENT ON COLUMN notes.content_date IS
    'When the note''s content was true: content_date_claims, else updated. '
    'search(since=) and the recency weight read this.';
COMMENT ON COLUMN notes.content_date_source IS
    'claims_recorded, or note_updated when the date is the write-time fallback.';

ALTER TABLE chunks
    DROP COLUMN IF EXISTS search,
    ADD COLUMN IF NOT EXISTS search tsvector GENERATED ALWAYS AS (
        to_tsvector('english', portolan_search_text(coalesce(text, '')))
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_search ON chunks USING gin (search);

-- A refresh of content_date_claims alone is not a write of the note.
CREATE OR REPLACE FUNCTION portolan_touch_notes_updated_at()
RETURNS trigger AS $$
BEGIN
    IF NEW.updated IS NOT DISTINCT FROM OLD.updated THEN
        IF NEW.content_date_claims IS DISTINCT FROM OLD.content_date_claims
           AND (to_jsonb(NEW) - ARRAY['content_date_claims', 'content_date',
                                      'content_date_source', 'search'])
               = (to_jsonb(OLD) - ARRAY['content_date_claims', 'content_date',
                                        'content_date_source', 'search']) THEN
            RETURN NEW;
        END IF;
        NEW.updated := now();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Recompute content_date_claims for the given notes (every note when NULL);
-- returns how many rows changed, so a second call reporting 0 shows the
-- column is current.
CREATE OR REPLACE FUNCTION portolan_refresh_content_dates(p_slugs text[] DEFAULT NULL)
RETURNS integer AS $$
DECLARE
    changed integer;
BEGIN
    UPDATE notes n
    SET content_date_claims = fresh.newest
    FROM (
        SELECT s.slug, (
            SELECT max(c.recorded) FROM claims c
            WHERE c.note_slug = s.slug AND c.superseded_at IS NULL AND NOT c.archived
        ) AS newest
        FROM notes s
        WHERE p_slugs IS NULL OR s.slug = ANY (p_slugs)
    ) fresh
    WHERE n.slug = fresh.slug
      AND n.content_date_claims IS DISTINCT FROM fresh.newest;
    GET DIAGNOSTICS changed = ROW_COUNT;
    RETURN changed;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION portolan_claims_content_date()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM portolan_refresh_content_dates(ARRAY[NEW.note_slug]);
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM portolan_refresh_content_dates(ARRAY[OLD.note_slug]);
    ELSE
        PERFORM portolan_refresh_content_dates(ARRAY[NEW.note_slug, OLD.note_slug]);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_claims_content_date ON claims;
CREATE TRIGGER trg_claims_content_date
    AFTER INSERT OR DELETE OR UPDATE OF note_slug, recorded, superseded_at, archived
    ON claims
    FOR EACH ROW
    EXECUTE FUNCTION portolan_claims_content_date();

SELECT portolan_refresh_content_dates();
