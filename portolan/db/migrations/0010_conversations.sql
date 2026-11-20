-- Portolan schema, migration 0010: chat conversations and messages
--
-- Owned by the UI wave 1 build (01-spec/build-contracts.md "Migration
-- ledger": "you own 0010"). Backs the two-pane chat surface
-- (portolan/web/routes/chat.py): a left sidebar of the signed-in reader's
-- own conversations, a right pane of messages, streamed answers over SSE.
--
-- UNLIKE the rest of this schema (0001_init.sql's "NOTHING IS HARD-
-- DELETED"), conversations and messages ARE hard-deletable. They are a
-- reader's own scratch space over the graph, not provenance data -- the
-- graph's notes/claims/sources/links carry the record that must never be
-- destroyed; a chat transcript referencing them is disposable the way an
-- email client lets you delete a thread without deleting the emails it
-- quoted. `messages` cascades on its parent conversation for exactly this
-- reason (ON DELETE CASCADE, the one cascade in this codebase that isn't
-- ON DELETE RESTRICT).
--
-- IDEMPOTENT, like every migration before it: IF NOT EXISTS throughout.

CREATE TABLE IF NOT EXISTS conversations (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner     text NOT NULL,
    title     text NOT NULL DEFAULT 'New chat',
    created   timestamptz NOT NULL DEFAULT now(),
    updated   timestamptz NOT NULL DEFAULT now(),
    archived  boolean NOT NULL DEFAULT false
);

COMMENT ON TABLE conversations IS
    'One chat thread, owned by the display name a reader logged in with '
    '(portolan.web.auth.Session.editor). title is auto-generated from the '
    'first question''s opening 60 characters -- no model call.';

CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated
    ON conversations (owner, updated DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_owner_archived
    ON conversations (owner, archived);

CREATE TABLE IF NOT EXISTS messages (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id  bigint NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant')),
    content_md       text NOT NULL,
    -- For an assistant message: the validated `Answer.sentences` shape
    -- (portolan/web/answer.py), one entry per sentence -- {"text":
    -- ..., "citations": [{"note_slug", "claim_id", "source_slug",
    -- "locator", "is_note_level"}, ...]} -- exactly what answer_question()
    -- already validated before this row was written (build instructions:
    -- "validate, then stream the validated sentences"; nothing here is
    -- re-derived from the model's raw output). '[]' for a user message.
    citations        jsonb NOT NULL DEFAULT '[]'::jsonb,
    created          timestamptz NOT NULL DEFAULT now(),
    -- {model, input_tokens, output_tokens, cached_tokens} for the assistant
    -- message that triggered a real LLM call; '{}' for a user message or an
    -- answer that never reached the model (the "nothing recorded"/"index
    -- broken" short-circuits in answer.py).
    usage            jsonb NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE messages IS
    'One turn of a conversation. citations carries the already-validated '
    'per-sentence citation list for an assistant message (provenance-'
    'chain.md link 1: answer -> note) -- never re-validated at render '
    'time, only re-rendered.';

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_created
    ON messages (conversation_id, created);

-- Auto-touch conversations.updated whenever a message is appended, so the
-- sidebar's "newest first" ordering reflects the last turn, not just the
-- conversation's own creation time.
CREATE OR REPLACE FUNCTION portolan_touch_conversation_on_message()
RETURNS trigger AS $$
BEGIN
    UPDATE conversations SET updated = now() WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_messages_touch_conversation ON messages;
CREATE TRIGGER trg_messages_touch_conversation
    AFTER INSERT ON messages
    FOR EACH ROW
    EXECUTE FUNCTION portolan_touch_conversation_on_message();
