-- Portolan schema, migration 0019: persist the resolution judge's own
-- run-health counters onto the `reports` row.
--
-- Owned by review-theme-wave-5c.md should-fix 7. The "Run health"
-- section in the rendered report body (`portolan/report/weekly.py`)
-- already surfaces `judge_calls`/`judge_failures`/`judge_cap_hit` to a
-- reader of THAT WEEK's report -- but nothing durable let an operator
-- ask "how has the judge been doing" after the fact, across weeks,
-- without re-reading report bodies or grepping `changelog`. `reports.
-- judge_health` is a small jsonb snapshot written once, when the
-- `reports` row itself is written (`build_weekly`), carrying exactly the
-- three counters plus a derived `outage` flag (`judge_failures ==
-- judge_calls AND judge_calls > 0` -- every call this run failed, not
-- merely some).
--
-- IDEMPOTENT: `ADD COLUMN IF NOT EXISTS`, safe to re-apply
-- (tests/test_db_schema.py applies every migration file twice).

ALTER TABLE reports
    ADD COLUMN IF NOT EXISTS judge_health jsonb;

COMMENT ON COLUMN reports.judge_health IS
    'Snapshot of the resolution judge''s own run-health counters for '
    'this report''s run: {"judge_calls": int, "judge_failures": int, '
    '"judge_cap_hit": bool, "outage": bool}. outage is true only when '
    'judge_failures == judge_calls AND judge_calls > 0 -- every call '
    'this run failed, not merely some. NULL for a report written before '
    'this column existed. Never read as evidence of anything about the '
    'GRAPH by itself -- purely an operational signal about the judge.';
