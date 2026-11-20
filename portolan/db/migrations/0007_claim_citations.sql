-- Portolan schema, migration 0007: claim_citations.
--
-- Owned by the fix wave 2026-09-12 (portolan/merge/**). Review finding G4
-- (01-spec/reviews/review-wave-bc-graph-2026-09-12.md): a `same` verdict
-- used to write a SECOND claims row carrying the *candidate*'s wording
-- (`candidate.claim_text`) but *this* source's citation (`source=source,
-- locator=fact.locator, quote=fact.quote`) -- so the rendered claim line
-- asserted one source's sentence while citing a different source at a
-- locator that sentence never appears at. 01-spec/provenance-chain.md
-- exists specifically to prevent a citation pointing at a source that does
-- not contain the claim; this was that failure.
--
-- The fix (01-spec/quality-loop.md's own framing: "one fact, one claim"):
-- a corroboration -- the same underlying fact seen again, whether from a
-- second source or a second mention within the same source (the F04
-- within-source-duplicate case) -- is a second CITATION on the existing
-- claim, never a second claims row. `claims.source_slug`/`claims.locator`
-- stay the primary citation (unchanged, first-writer-wins); every
-- additional sighting of the same claim lands here instead, rendered
-- after the primary citation on the same claim line
-- (portolan/notes/render.py).
--
-- IDEMPOTENT, like every migration here: IF NOT EXISTS throughout, safe to
-- re-apply (tests/test_db_schema.py applies every migration file twice).

CREATE TABLE IF NOT EXISTS claim_citations (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    claim_id     bigint NOT NULL REFERENCES claims (id) ON DELETE RESTRICT,
    source_slug  text NOT NULL REFERENCES sources (slug) ON DELETE RESTRICT,
    locator      text NOT NULL,
    quote        text,
    recorded     timestamptz NOT NULL DEFAULT now(),
    -- A second sighting of the exact same (source, locator) is a no-op
    -- re-run, not a third citation -- ON CONFLICT DO NOTHING at the write
    -- site relies on this.
    UNIQUE (claim_id, source_slug, locator)
);

COMMENT ON TABLE claim_citations IS
    'Extra citations on an existing claim (provenance-chain.md, review '
    'finding G4): a "same" merge verdict records a corroborating source '
    'here instead of writing a duplicate claims row. claims.source_slug/'
    'locator remain the primary citation; these are rendered after it.';

CREATE INDEX IF NOT EXISTS idx_claim_citations_claim_id ON claim_citations (claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_citations_source_slug ON claim_citations (source_slug);
