"""People who signed in through WorkOS, and their editor/viewer role
(`web_users`, migration 0038). Admins are `Settings.admin_emails`, not a row here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psycopg

__all__ = [
    "ROLES",
    "Person",
    "get_role",
    "list_people",
    "parse_email_list",
    "record_sign_in",
    "set_role",
]

ROLES = ("editor", "viewer")


@dataclass(frozen=True)
class Person:
    email: str
    name: str | None
    role: str
    first_sign_in: datetime
    last_sign_in: datetime
    sign_in_count: int
    role_changed_by: str | None
    role_changed_at: datetime | None


def parse_email_list(value: str) -> frozenset[str]:
    """Comma- or whitespace-separated entries, lowercased, blanks dropped."""
    return frozenset(part.strip().lower() for part in value.replace(",", " ").split() if part)


def record_sign_in(
    conn: psycopg.Connection, *, email: str, name: str | None, workos_user_id: str | None
) -> None:
    """Insert the person on first sign-in, else refresh their name and sign-in time.
    The role is never changed here."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO web_users (email, name, workos_user_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET
                name = EXCLUDED.name,
                workos_user_id = EXCLUDED.workos_user_id,
                last_sign_in = now(),
                sign_in_count = web_users.sign_in_count + 1
            """,
            (email.lower(), name, workos_user_id),
        )
    conn.commit()


def get_role(conn: psycopg.Connection, email: str) -> str:
    """'editor' for anyone without a row (the default role)."""
    with conn.cursor() as cur:
        cur.execute("SELECT role FROM web_users WHERE email = %s", (email.lower(),))
        row = cur.fetchone()
    return row[0] if row else "editor"


def set_role(conn: psycopg.Connection, *, email: str, role: str, changed_by: str) -> bool:
    """False when no such person has signed in."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE web_users
               SET role = %s, role_changed_by = %s, role_changed_at = now()
             WHERE email = %s
            """,
            (role, changed_by, email.lower()),
        )
        updated = cur.rowcount == 1
    conn.commit()
    return updated


def list_people(conn: psycopg.Connection) -> list[Person]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT email, name, role, first_sign_in, last_sign_in, sign_in_count,
                   role_changed_by, role_changed_at
              FROM web_users
             ORDER BY last_sign_in DESC, email
            """
        )
        return [Person(*row) for row in cur.fetchall()]
