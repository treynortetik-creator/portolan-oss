"""Conversation and message persistence for the chat surface (migration
`0010_conversations.sql`): create, rename, archive and hard-delete a
conversation, and read and append its messages.

Every function that reads or changes an existing conversation or message
takes `owner` and scopes its `WHERE` to it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from portolan.web.answer import Answer, Sentence

__all__ = [
    "TITLE_MAX_CHARS",
    "TITLE_WORD_COUNT",
    "Conversation",
    "Message",
    "add_message",
    "conversation_messages",
    "count_conversations",
    "create_conversation",
    "delete_conversation",
    "delete_message",
    "get_conversation",
    "last_message",
    "last_user_message",
    "list_conversations",
    "rename_conversation",
    "sentence_to_dict",
    "sentences_to_content_md",
    "set_archived",
    "title_from_question",
    "top_account_names",
]

TITLE_MAX_CHARS = 60

# Auto-titles take whole words, never cutting one in half; TITLE_MAX_CHARS
# still caps a single very long "word" such as a pasted URL.
TITLE_WORD_COUNT = 6


class Conversation:
    def __init__(
        self,
        *,
        id: int,
        owner: str,
        title: str,
        created: datetime,
        updated: datetime,
        archived: bool,
    ) -> None:
        self.id = id
        self.owner = owner
        self.title = title
        self.created = created
        self.updated = updated
        self.archived = archived


class Message:
    def __init__(
        self,
        *,
        id: int,
        conversation_id: int,
        role: Literal["user", "assistant"],
        content_md: str,
        citations: list[dict[str, Any]],
        created: datetime,
        usage: dict[str, Any],
    ) -> None:
        self.id = id
        self.conversation_id = conversation_id
        self.role = role
        self.content_md = content_md
        self.citations = citations
        self.created = created
        self.usage = usage


def _row_to_conversation(row: tuple) -> Conversation:
    return Conversation(
        id=row[0], owner=row[1], title=row[2], created=row[3], updated=row[4], archived=row[5]
    )


def _row_to_message(row: tuple) -> Message:
    return Message(
        id=row[0],
        conversation_id=row[1],
        role=row[2],
        content_md=row[3],
        citations=row[4] or [],
        created=row[5],
        usage=row[6] or {},
    )


def sentence_to_dict(sentence: Sentence) -> dict[str, Any]:
    """A `Sentence` as the plain dict stored in `messages.citations`,
    including `is_note_level` and `is_passage`, computed properties
    `model_dump()` would omit."""
    return {
        "text": sentence.text,
        "citations": [
            {
                "note_slug": c.note_slug,
                "claim_id": c.claim_id,
                "source_slug": c.source_slug,
                "locator": c.locator,
                "is_note_level": c.is_note_level,
                "passage_id": c.passage_id,
                "quote": c.quote,
                "is_passage": c.is_passage,
            }
            for c in sentence.citations
        ],
    }


def sentences_to_content_md(answer: Answer) -> str:
    """The assistant message's stored body: every sentence's text, in
    order, separated by blank lines. Citations live in the `citations`
    column (`sentence_to_dict`), not in this text."""
    return "\n\n".join(s.text for s in answer.sentences)


def count_conversations(conn: psycopg.Connection, *, owner: str) -> int:
    """How many conversations `owner` has, archived included (the per-owner
    cap check)."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM conversations WHERE owner = %s", (owner,))
        (count,) = cur.fetchone()
    return count


def create_conversation(
    conn: psycopg.Connection, *, owner: str, title: str = "New chat"
) -> Conversation:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (owner, title)
            VALUES (%s, %s)
            RETURNING id, owner, title, created, updated, archived
            """,
            (owner, title),
        )
        row = cur.fetchone()
    conn.commit()
    return _row_to_conversation(row)


def get_conversation(
    conn: psycopg.Connection, conversation_id: int, *, owner: str
) -> Conversation | None:
    """`None` both when the id doesn't exist and when it belongs to a
    different owner -- a caller has no way to distinguish "not found" from
    "not yours," which is the point: nothing here confirms another
    reader's conversation exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, owner, title, created, updated, archived
            FROM conversations WHERE id = %s AND owner = %s
            """,
            (conversation_id, owner),
        )
        row = cur.fetchone()
    return _row_to_conversation(row) if row else None


def list_conversations(
    conn: psycopg.Connection, *, owner: str, include_archived: bool = False
) -> list[Conversation]:
    with conn.cursor() as cur:
        if include_archived:
            cur.execute(
                """
                SELECT id, owner, title, created, updated, archived
                FROM conversations WHERE owner = %s
                ORDER BY archived, updated DESC
                """,
                (owner,),
            )
        else:
            cur.execute(
                """
                SELECT id, owner, title, created, updated, archived
                FROM conversations WHERE owner = %s AND NOT archived
                ORDER BY updated DESC
                """,
                (owner,),
            )
        rows = cur.fetchall()
    return [_row_to_conversation(r) for r in rows]


def rename_conversation(
    conn: psycopg.Connection, conversation_id: int, *, owner: str, title: str
) -> Conversation | None:
    title = title.strip()[:TITLE_MAX_CHARS] or "New chat"
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversations SET title = %s
            WHERE id = %s AND owner = %s
            RETURNING id, owner, title, created, updated, archived
            """,
            (title, conversation_id, owner),
        )
        row = cur.fetchone()
    conn.commit()
    return _row_to_conversation(row) if row else None


def set_archived(
    conn: psycopg.Connection, conversation_id: int, *, owner: str, archived: bool
) -> Conversation | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE conversations SET archived = %s
            WHERE id = %s AND owner = %s
            RETURNING id, owner, title, created, updated, archived
            """,
            (archived, conversation_id, owner),
        )
        row = cur.fetchone()
    conn.commit()
    return _row_to_conversation(row) if row else None


def delete_conversation(conn: psycopg.Connection, conversation_id: int, *, owner: str) -> bool:
    """Hard delete; `messages` cascades (conversations are not provenance,
    migration 0010). Returns whether a row was owned and removed."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE id = %s AND owner = %s RETURNING id",
            (conversation_id, owner),
        )
        row = cur.fetchone()
    conn.commit()
    return row is not None


def conversation_messages(
    conn: psycopg.Connection, conversation_id: int, *, owner: str
) -> list[Message]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.conversation_id, m.role, m.content_md, m.citations, m.created, m.usage
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.conversation_id = %s AND c.owner = %s
            ORDER BY m.id
            """,
            (conversation_id, owner),
        )
        rows = cur.fetchall()
    return [_row_to_message(r) for r in rows]


def add_message(
    conn: psycopg.Connection,
    *,
    conversation_id: int,
    owner: str,
    role: Literal["user", "assistant"],
    content_md: str,
    citations: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> Message:
    """Raises `LookupError`, writing nothing, when the conversation does not
    exist or is not `owner`'s."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, content_md, citations, usage)
            SELECT c.id, %s, %s, %s, %s FROM conversations c WHERE c.id = %s AND c.owner = %s
            RETURNING id, conversation_id, role, content_md, citations, created, usage
            """,
            (
                role,
                content_md,
                Jsonb(citations or []),
                Jsonb(usage or {}),
                conversation_id,
                owner,
            ),
        )
        row = cur.fetchone()
    conn.commit()
    if row is None:
        raise LookupError("no such conversation")
    return _row_to_message(row)


def last_user_message(
    conn: psycopg.Connection, conversation_id: int, *, owner: str
) -> Message | None:
    """The most recent `role='user'` message -- what `regenerate` re-asks."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.conversation_id, m.role, m.content_md, m.citations, m.created, m.usage
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.conversation_id = %s AND m.role = 'user' AND c.owner = %s
            ORDER BY m.id DESC LIMIT 1
            """,
            (conversation_id, owner),
        )
        row = cur.fetchone()
    return _row_to_message(row) if row else None


def last_message(conn: psycopg.Connection, conversation_id: int, *, owner: str) -> Message | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.conversation_id, m.role, m.content_md, m.citations, m.created, m.usage
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.conversation_id = %s AND c.owner = %s
            ORDER BY m.id DESC LIMIT 1
            """,
            (conversation_id, owner),
        )
        row = cur.fetchone()
    return _row_to_message(row) if row else None


def delete_message(conn: psycopg.Connection, message_id: int, *, owner: str) -> None:
    """Scoped to `owner` through the message's conversation (`messages`
    has no `owner` column)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM messages
            WHERE id = %s AND conversation_id IN (SELECT id FROM conversations WHERE owner = %s)
            """,
            (message_id, owner),
        )
    conn.commit()


def title_from_question(question: str) -> str:
    """Auto-title: the first `TITLE_WORD_COUNT` words of the question, no
    model call; an ellipsis marks a longer question, and the result is
    capped at `TITLE_MAX_CHARS` like `rename_conversation`."""
    words = question.split()
    if not words:
        return "New chat"
    title = " ".join(words[:TITLE_WORD_COUNT])
    if len(words) > TITLE_WORD_COUNT:
        title += "…"
    return title[:TITLE_MAX_CHARS] or "New chat"


def top_account_names(conn: psycopg.Connection, limit: int = 3) -> list[str]:
    """The most recently updated account names, for the empty chat
    state's example questions."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT title FROM notes WHERE type = 'account' ORDER BY updated DESC LIMIT %s",
            (limit,),
        )
        return [row[0] for row in cur.fetchall()]
