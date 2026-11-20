-- Portolan schema, migration 0029: action researcher, revised contract
-- (01-spec/actions-and-owners-2026-09-19.md, "Phase 2, revised contract").
--
-- researcher_findings: one row per finding-tool call a researcher agent
-- session makes (absence_check, count_by, timeline, renewal) -- written by
-- the TOOL ITSELF, never by the model (the model can only cite a
-- finding_id it was actually handed back in a tool result). `run_id`
-- groups every finding + every submit_card attempt from one single
-- `Researcher.run()` call; `candidate_slug` is redundant with `run_id` but
-- kept as its own indexed column since "every finding for this candidate"
-- (the review page's own "rerun" link, an operator spot-check) is a
-- common lookup this build has no other reason to keep a run-id table
-- for.
--
-- support_judge_verdicts: the typed judge's own cache, mirroring
-- theme_judge_verdicts' pattern (migrations 0017/0018) -- cached by the
-- sentence's own hash, the hash of every referenced evidence item's
-- claim_id/text/quote and every referenced finding's id/result (in
-- order, so the same references in a different order still hit), and a
-- judge_version hash (the judge's system prompt + model id). A claim
-- whose text changed, or a run under a tightened judge prompt, therefore
-- never reuses a stale verdict.

CREATE TABLE IF NOT EXISTS researcher_findings (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id         text NOT NULL,
    candidate_slug text NOT NULL,
    tool           text NOT NULL,
    arguments      jsonb NOT NULL,
    result         jsonb NOT NULL,
    created        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_researcher_findings_run ON researcher_findings (run_id);
CREATE INDEX IF NOT EXISTS idx_researcher_findings_candidate
    ON researcher_findings (candidate_slug);

CREATE TABLE IF NOT EXISTS support_judge_verdicts (
    sentence_hash  text NOT NULL,
    reference_hash text NOT NULL,
    judge_version  text NOT NULL,
    sentence       text NOT NULL,
    verdict        text NOT NULL CHECK (verdict IN ('supported', 'not_supported')),
    judged_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (sentence_hash, reference_hash, judge_version)
);
