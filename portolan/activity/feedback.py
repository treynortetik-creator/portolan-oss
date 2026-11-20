"""Thumbs up or down on a chat answer, and the eval candidate a thumbs-down
leaves behind (01-spec/red-team-2026-09-24.md sec 3: a failed chat becomes
an eval case once a person confirms the gold answer).

A thumbs-down copies the question and the answer, as they stand, into
`eval_candidates` with status `candidate`. Nothing past that is automated.
Changing the rating to up removes a candidate nobody has reviewed yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb

from portolan.activity import record_activity

__all__ = [
    "REASON_MAX_CHARS",
    "Feedback",
    "clean_reason",
    "feedback_for_messages",
    "save_feedback",
]

# Matches the CHECK on answer_feedback.reason (0037_pilot_activity.sql).
REASON_MAX_CHARS = 200


@dataclass(frozen=True)
class Feedback:
    message_id: int
    conversation_id: int
    rating: str
    reason: str | None


def clean_reason(raw: str | None) -> str | None:
    """One line, at most `REASON_MAX_CHARS`; empty becomes None."""
    if raw is None:
        return None
    text = " ".join(raw.split())[:REASON_MAX_CHARS].strip()
    return text or None


def save_feedback(
    conn: psycopg.Connection,
    *,
    message_id: int,
    user_name: str,
    rating: str,
    reason: str | None,
) -> Feedback | None:
    """Rate an assistant message in a conversation `user_name` owns. None
    when there is no such message for them."""
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    reason = clean_reason(reason)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.conversation_id, m.content_md, m.citations
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE m.id = %s AND m.role = 'assistant' AND c.owner = %s
            """,
            (message_id, user_name),
        )
        row = cur.fetchone()
        if row is None:
            return None
        conversation_id, answer_md, citations = row
        cur.execute(
            """
            INSERT INTO answer_feedback (message_id, user_name, rating, reason)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (message_id, user_name) DO UPDATE
                SET rating = EXCLUDED.rating, reason = EXCLUDED.reason, updated_at = now()
            RETURNING id
            """,
            (message_id, user_name, rating, reason),
        )
        feedback_id = cur.fetchone()[0]
        if rating == "down":
            cur.execute(
                """
                SELECT content_md FROM messages
                WHERE conversation_id = %s AND role = 'user' AND id < %s
                ORDER BY id DESC LIMIT 1
                """,
                (conversation_id, message_id),
            )
            question_row = cur.fetchone()
            cur.execute(
                """
                INSERT INTO eval_candidates
                    (feedback_id, message_id, user_name, question, answer_md, citations, reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feedback_id) DO UPDATE SET reason = EXCLUDED.reason
                """,
                (
                    feedback_id,
                    message_id,
                    user_name,
                    question_row[0] if question_row else "",
                    answer_md,
                    Jsonb(citations),
                    reason,
                ),
            )
        else:
            cur.execute(
                "DELETE FROM eval_candidates WHERE feedback_id = %s AND status = 'candidate'",
                (feedback_id,),
            )
    record_activity(
        conn,
        kind="feedback",
        user_name=user_name,
        target=str(message_id),
        detail={"rating": rating, "message_id": message_id},
    )
    return Feedback(
        message_id=message_id, conversation_id=conversation_id, rating=rating, reason=reason
    )


def feedback_for_messages(
    conn: psycopg.Connection, message_ids: list[int], *, user_name: str
) -> dict[int, dict[str, str | None]]:
    """`{message_id: {"rating", "reason"}}` for the ones `user_name` rated."""
    if not message_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT message_id, rating, reason FROM answer_feedback
            WHERE user_name = %s AND message_id = ANY(%s)
            """,
            (user_name, list(message_ids)),
        )
        return {mid: {"rating": rating, "reason": reason} for mid, rating, reason in cur}
