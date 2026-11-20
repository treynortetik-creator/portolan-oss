-- Portolan schema, migration 0037: pilot activity, answer feedback, and
-- candidate eval cases (2026-09-25).
--
-- The 30-day pilot in 01-spec/red-team-2026-09-24.md needs to know whether
-- people use the app. `usage_events` (0001) is the model-call meter, keyed
-- by tenant and surface; people's actions go in a separate table so the
-- meter's shape and its restore-drill counts stay as they are.
--
-- activity_events: one row per thing a signed-in person did (a page view, a
--   chat question, a click from an answer through to an original, a review
--   decision, an action confirm) or an email open/click. `user_name` is the
--   display name from the login (`portolan.web.auth.Session.editor`), or the
--   recipient label an email link was signed with. No IP address, no user
--   agent, no query string. `detail` holds small fields per kind
--   (`portolan.activity.record_activity` lists them); a chat question keeps
--   its text so the dashboard can list the questions that got no answer.
--   Rows older than Settings.activity_retention_days are deleted at startup
--   and by `python -m portolan.activity purge`.
--
-- answer_feedback: one thumbs up or down per person per assistant message,
--   with an optional one-line reason. Re-rating updates the row.
--
-- eval_candidates: a thumbs-down question and its answer, copied at the
--   time, waiting for a person to write and confirm the gold answer. Nothing
--   is automated past saving it (01-spec/red-team-2026-09-24.md sec 3, the
--   "failed chat to eval case" row).
--
-- Additive only: three new tables, no change to an existing one.

CREATE TABLE IF NOT EXISTS activity_events (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at  timestamptz NOT NULL DEFAULT now(),
    user_name    text,
    kind         text NOT NULL CHECK (kind IN (
                     'page_view', 'chat_question', 'citation_click',
                     'review_decision', 'action_confirm', 'feedback',
                     'email_open', 'email_click'
                 )),
    route        text,
    target       text,
    detail       jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_activity_events_occurred
    ON activity_events (occurred_at);
CREATE INDEX IF NOT EXISTS idx_activity_events_kind_occurred
    ON activity_events (kind, occurred_at);
CREATE INDEX IF NOT EXISTS idx_activity_events_user
    ON activity_events (user_name);

CREATE TABLE IF NOT EXISTS answer_feedback (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- SET NULL: a regenerated or deleted answer keeps its rating; the
    -- eval candidate below carries its own copy of the text.
    message_id   bigint REFERENCES messages (id) ON DELETE SET NULL,
    user_name    text NOT NULL,
    rating       text NOT NULL CHECK (rating IN ('up', 'down')),
    reason       text CHECK (reason IS NULL OR char_length(reason) <= 200),
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (message_id, user_name)
);

CREATE INDEX IF NOT EXISTS idx_answer_feedback_user ON answer_feedback (user_name);

CREATE TABLE IF NOT EXISTS eval_candidates (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    origin        text NOT NULL DEFAULT 'thumbs_down' CHECK (origin IN ('thumbs_down')),
    feedback_id   bigint UNIQUE REFERENCES answer_feedback (id) ON DELETE SET NULL,
    message_id    bigint,
    user_name     text,
    question      text NOT NULL,
    answer_md     text NOT NULL,
    citations     jsonb NOT NULL DEFAULT '[]'::jsonb,
    reason        text,
    status        text NOT NULL DEFAULT 'candidate'
                  CHECK (status IN ('candidate', 'confirmed', 'rejected')),
    gold_answer   text,
    confirmed_by  text,
    confirmed_at  timestamptz
);

CREATE INDEX IF NOT EXISTS idx_eval_candidates_status ON eval_candidates (status, created_at);
