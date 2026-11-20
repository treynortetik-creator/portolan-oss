"""Input bounds for the web routes: each raises `HTTPException(400)` with a
plain reason a reader can act on, instead of letting an oversized value
reach a framework 500 or an unbounded model call.
"""

from __future__ import annotations

from fastapi import HTTPException

# Default only: the chat route passes `Settings.chat_question_max_chars`.
# Long enough for a pasted email; still a ceiling on billed model context.
MAX_QUESTION_CHARS = 32_000


def require_question_length(question: str, *, max_chars: int = MAX_QUESTION_CHARS) -> str:
    """The chat question, or a 400 when it is empty or over `max_chars`.
    Pass it already stripped; the composer's own empty check is
    client-side only."""
    if not question:
        raise HTTPException(
            status_code=400,
            detail="That question is empty. Type something to ask.",
        )
    if len(question) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=(
                f"That question is {len(question):,} characters long. Portolan reads the "
                f"first {max_chars:,} characters of a chat question -- ask something "
                "shorter, or split it into two questions."
            ),
        )
    return question


def bounded_int(value: int, *, name: str, minimum: int, maximum: int) -> int:
    """`value`, or a 400 naming the valid range."""
    if value < minimum or value > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {minimum} and {maximum} (got {value}).",
        )
    return value


def bounded_str(value: str, *, name: str, max_chars: int) -> str:
    """`value`, or a 400 naming the length limit."""
    if len(value) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"{name} is {len(value):,} characters long; the limit is {max_chars:,}.",
        )
    return value
