-- Portolan schema, migration 0038: people who signed in to the web app through
-- WorkOS AuthKit, and their role (2026-09-25, 01-spec/pilot-sign-in-2026-09-25.md).
--
-- One row per verified email, written at each sign-in (`portolan.web.people`).
-- `role` is 'editor' (the default: may edit notes, decide review items and
-- confirm actions) or 'viewer' (reads and asks only). Only an admin
-- (Settings.admin_emails) changes it, on /admin/people; `role_changed_by` names
-- who did. Reading is never restricted by role: everyone in the org sees
-- everything in v1 (CLAUDE.md, no RBAC).
--
-- Additive only: one new table.

CREATE TABLE IF NOT EXISTS web_users (
    email            text PRIMARY KEY CHECK (email = lower(email) AND length(email) BETWEEN 3 AND 320),
    name             text,
    workos_user_id   text,
    role             text NOT NULL DEFAULT 'editor' CHECK (role IN ('editor', 'viewer')),
    first_sign_in    timestamptz NOT NULL DEFAULT now(),
    last_sign_in     timestamptz NOT NULL DEFAULT now(),
    sign_in_count    integer NOT NULL DEFAULT 1,
    role_changed_by  text,
    role_changed_at  timestamptz
);

COMMENT ON TABLE web_users IS
    'People who signed in to the web app with WorkOS AuthKit, and their editor/viewer '
    'role (portolan.web.people). Admins come from PORTOLAN_ADMIN_EMAILS, not this table.';
