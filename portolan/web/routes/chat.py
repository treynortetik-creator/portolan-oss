"""`/` and `/c/{id}` (the two-pane chat page), `/conversations` (CRUD),
and `/chat/{id}/stream` + `/chat/{id}/regenerate` (the SSE answer stream).
portolan-spec-v2.md sec 10: "Chat, first."

Every conversation is owned by the display name the reader logged in with
(`Session.editor`).

The stream endpoints call `answer_question`, or `run_chat_agent` when
`settings.chat_mode` is "agent" (both validate every sentence's citations
with the same gate), and only then send the validated sentences over SSE,
rendered through `_chat_sentence.html` -- the same partial a page reload
uses, so a streamed sentence and the reloaded one are byte-identical. The
model's raw output is never streamed; agent mode streams only its progress
lines ("Reading <note title>…") before the answer.

SSE runs over POST with a `fetch().body.getReader()` parser in chat.html,
not `EventSource`: a question can be `settings.chat_question_max_chars`
long, too long for a query string, and `EventSource` cannot POST.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from portolan.activity import record_activity
from portolan.activity.feedback import feedback_for_messages
from portolan.llm.base import LLMClient
from portolan.pricing import UnknownModelError, estimate_cost
from portolan.search.embed import Embedder
from portolan.web.answer import Answer, answer_question
from portolan.web.auth import Session, log_usage_event, require_session, require_session_api
from portolan.web.bounds import require_question_length
from portolan.web.chat_agent import AgentLimits, history_from_messages, run_chat_agent
from portolan.web.conversations import (
    Conversation,
    add_message,
    conversation_messages,
    count_conversations,
    create_conversation,
    delete_conversation,
    delete_message,
    get_conversation,
    last_message,
    last_user_message,
    list_conversations,
    rename_conversation,
    sentence_to_dict,
    sentences_to_content_md,
    set_archived,
    title_from_question,
    top_account_names,
)
from portolan.web.deps import get_conn, get_embedder_dep, get_llm_dep, get_settings_dep
from portolan.web.queries import note_summaries
from portolan.web.tracking import track_page_view

router = APIRouter()
_logger = logging.getLogger(__name__)

# Sent to the client instead of `str(exc)`, which can carry SQL, connection
# details or a provider's raw error body; the exception is logged server-side.
GENERIC_ANSWER_ERROR = "The answer could not be generated right now -- try again in a moment."

# Bounds the sidebar and the table, not real use.
MAX_CONVERSATIONS_PER_OWNER = 500

# The Agent SDK client talks only to Claude, so an OpenRouter-style
# "vendor/model" `CHAT_MODEL` falls back to this under `agent_sdk`.
DEFAULT_AGENT_SDK_CHAT_MODEL = "claude-sonnet-5"


def resolve_chat_model(settings) -> str:  # noqa: ANN001 - portolan.config.Settings
    """The model chat requests: `settings.chat_model`, else
    `settings.strong_model`. Under `llm_provider="agent_sdk"`, a missing or
    non-`"claude-..."` `chat_model` gives `DEFAULT_AGENT_SDK_CHAT_MODEL`.
    """
    model = settings.chat_model
    if settings.llm_provider == "agent_sdk":
        if not model or not model.startswith("claude-"):
            return DEFAULT_AGENT_SDK_CHAT_MODEL
        return model
    return model or settings.strong_model


def agent_options(settings) -> dict[str, Any] | None:  # noqa: ANN001 - portolan.config.Settings
    """`run_chat_agent`'s model and limits when `settings.chat_mode` is
    "agent"; `None` (single mode) otherwise."""
    if settings.chat_mode != "agent":
        return None
    return {
        "model": settings.chat_agent_model,
        "limits": AgentLimits.from_settings(settings),
    }


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # nginx buffers proxied responses by default, which would deliver the
    # whole stream as one chunk at the end.
    "X-Accel-Buffering": "no",
}


def _wants_fragment(request: Request) -> bool:
    """True for an HTMX request (the sidebar's rename/archive/delete
    forms), which gets a fragment or an `HX-Refresh` back; a plain form
    POST gets a redirect."""
    return request.headers.get("hx-request", "").lower() == "true"


def _example_questions(conn: psycopg.Connection) -> list[str]:
    return [f"What has {name} said recently?" for name in top_account_names(conn, 3)]


def _citation_slugs_in_messages(messages: list) -> list[str]:
    slugs: list[str] = []
    for m in messages:
        if m.role != "assistant":
            continue
        for sentence in m.citations:
            for c in sentence.get("citations", []) or []:
                slug = c.get("note_slug")
                if slug:
                    slugs.append(slug)
    return slugs


def _render_chat_page(
    request: Request,
    conn: psycopg.Connection,
    session: Session,
    *,
    conversation: Conversation | None,
) -> object:
    templates = request.app.state.templates
    conversations = list_conversations(conn, owner=session.editor)
    all_conversations = list_conversations(conn, owner=session.editor, include_archived=True)
    archived_conversations = [c for c in all_conversations if c.archived]

    messages = (
        conversation_messages(conn, conversation.id, owner=session.editor) if conversation else []
    )
    summaries = note_summaries(conn, _citation_slugs_in_messages(messages))
    examples = _example_questions(conn) if not messages else []
    message_feedback = feedback_for_messages(
        conn, [m.id for m in messages if m.role == "assistant"], user_name=session.editor
    )

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "conversations": conversations,
            "archived_conversations": archived_conversations,
            "conversation": conversation,
            "messages": messages,
            "note_summaries": summaries,
            "examples": examples,
            "message_feedback": message_feedback,
        },
    )


@router.get("/", dependencies=[Depends(track_page_view)])
def chat_home(
    request: Request,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    return _render_chat_page(request, conn, session, conversation=None)


@router.get("/c/{conversation_id}", dependencies=[Depends(track_page_view)])
def chat_conversation_page(
    request: Request,
    conversation_id: int,
    session: Session = Depends(require_session),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    conversation = get_conversation(conn, conversation_id, owner=session.editor)
    if conversation is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return _render_chat_page(request, conn, session, conversation=conversation)


# --------------------------------------------------------------------
# Conversation CRUD
# --------------------------------------------------------------------


@router.post("/conversations")
def conversations_create(
    session: Session = Depends(require_session_api), conn: psycopg.Connection = Depends(get_conn)
) -> object:
    if count_conversations(conn, owner=session.editor) >= MAX_CONVERSATIONS_PER_OWNER:
        raise HTTPException(
            status_code=400,
            detail=(
                f"You have {MAX_CONVERSATIONS_PER_OWNER:,} conversations already -- "
                "delete or archive one before starting another."
            ),
        )
    conversation = create_conversation(conn, owner=session.editor)
    return RedirectResponse(url=f"/c/{conversation.id}", status_code=303)


def _parse_active_id(value: str) -> int | None:
    """The `current_conversation_id` hidden field, so a swapped-in row keeps
    its "active" highlight. Empty or malformed means no active conversation,
    never a 400: a display detail is not worth failing a mutation over."""
    try:
        return int(value) if value.strip() else None
    except ValueError:
        return None


@router.post("/conversations/{conversation_id}/rename")
def conversations_rename(
    request: Request,
    conversation_id: int,
    title: str = Form(...),
    current_conversation_id: str = Form(""),
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    conversation = rename_conversation(conn, conversation_id, owner=session.editor, title=title)
    if conversation is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    if _wants_fragment(request):
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "_conversation_item.html",
            {"c": conversation, "active_id": _parse_active_id(current_conversation_id)},
        )
    return RedirectResponse(url=f"/c/{conversation_id}", status_code=303)


def _set_archived_route(
    request: Request,
    conversation_id: int,
    session: Session,
    conn: psycopg.Connection,
    *,
    archived: bool,
    current_conversation_id: str = "",
) -> object:
    conversation = set_archived(conn, conversation_id, owner=session.editor, archived=archived)
    if conversation is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    if _wants_fragment(request):
        # Archiving moves a row between two lists (and their counts), so a
        # full reload is used instead of swapping one `<li>` in place.
        response = HTMLResponse("")
        response.headers["HX-Refresh"] = "true"
        return response
    return RedirectResponse(url="/", status_code=303)


@router.post("/conversations/{conversation_id}/archive")
def conversations_archive(
    request: Request,
    conversation_id: int,
    current_conversation_id: str = Form(""),
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    return _set_archived_route(
        request,
        conversation_id,
        session,
        conn,
        archived=True,
        current_conversation_id=current_conversation_id,
    )


@router.post("/conversations/{conversation_id}/unarchive")
def conversations_unarchive(
    request: Request,
    conversation_id: int,
    current_conversation_id: str = Form(""),
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    return _set_archived_route(
        request,
        conversation_id,
        session,
        conn,
        archived=False,
        current_conversation_id=current_conversation_id,
    )


@router.post("/conversations/{conversation_id}/delete")
def conversations_delete(
    request: Request,
    conversation_id: int,
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
) -> object:
    deleted = delete_conversation(conn, conversation_id, owner=session.editor)
    if not deleted:
        raise HTTPException(status_code=404, detail="no such conversation")
    if _wants_fragment(request):
        return HTMLResponse("")
    return RedirectResponse(url="/", status_code=303)


# --------------------------------------------------------------------
# Streaming answers
# --------------------------------------------------------------------


def _resolve_or_create_conversation(
    conn: psycopg.Connection, conversation_id: str, *, owner: str, question: str
) -> Conversation:
    if conversation_id == "new":
        return create_conversation(conn, owner=owner, title=title_from_question(question))
    try:
        cid = int(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="no such conversation") from exc
    conversation = get_conversation(conn, cid, owner=owner)
    if conversation is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return conversation


def _log_answer_usage(
    conn: psycopg.Connection,
    *,
    tenant: str,
    answer: Answer,
    question: str,
    editor: str,
    conversation_id: int,
) -> None:
    extra: dict[str, Any] = {
        "question_chars": len(question),
        "editor": editor,
        "conversation_id": conversation_id,
    }
    agent = answer.usage.get("agent")
    if agent:
        extra["agent"] = {k: v for k, v in agent.items() if k != "context_slugs"}
    log_usage_event(
        conn,
        tenant=tenant,
        surface="chat",
        model_key=answer.usage.get("model", "no-call"),
        input_tokens=answer.usage.get("input_tokens", 0),
        output_tokens=answer.usage.get("output_tokens", 0),
        cached_tokens=answer.usage.get("cached_tokens", 0),
        extra=extra,
    )


def _answer_mode(answer: Answer) -> str:
    """How the answer ended, for the pilot log: `answered` when a sentence
    carries a citation, else which "no answer" message `answer_question`
    chose."""
    if any(s.citations for s in answer.sentences):
        return "answered"
    if not answer.control_ok:
        return "index_broken"
    if answer.gaps:
        return "gaps"
    return "nothing_recorded"


def _answer_cost(usage: dict[str, Any]) -> float | None:
    model = usage.get("model")
    if not model:
        return 0.0 if not usage else None
    try:
        return estimate_cost(
            model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cached_tokens", 0),
        )
    except UnknownModelError:
        return None


def _record_question(
    conn: psycopg.Connection,
    *,
    editor: str,
    question: str,
    conversation_id: int,
    latency_ms: int,
    regenerate: bool,
    answer: Answer | None = None,
    message_id: int | None = None,
    content_md: str = "",
) -> None:
    detail: dict[str, Any] = {
        "question": question,
        "latency_ms": latency_ms,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "regenerate": regenerate,
    }
    if answer is None:
        detail["mode"] = "error"
    else:
        detail.update(
            {
                "mode": _answer_mode(answer),
                "answer_chars": len(content_md),
                "cited_sentences": sum(1 for s in answer.sentences if s.citations),
                "citations": sum(len(s.citations) for s in answer.sentences),
                "withheld": answer.numerals_dropped + answer.uncited_dropped,
                "cost_usd": _answer_cost(answer.usage),
                "model": answer.usage.get("model"),
            }
        )
    record_activity(
        conn,
        kind="chat_question",
        user_name=editor,
        route="/chat/{conversation_id}/stream",
        target=str(message_id) if message_id is not None else None,
        detail=detail,
    )


def _history_before_question(conn: psycopg.Connection, conversation: Conversation, editor: str):
    """Answered exchanges before the question being answered (the last user
    message; on regenerate the trailing old answer follows it and is left out)."""
    messages = conversation_messages(conn, conversation.id, owner=editor)
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=0)
    return history_from_messages(messages[:last_user])


def _stream_answer(
    request: Request,
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    *,
    tenant: str,
    editor: str,
    conversation: Conversation,
    question: str,
    model: str,
    replace_message_id: int | None = None,
    is_new_conversation: bool = False,
    agent: dict[str, Any] | None = None,
) -> Generator[str]:
    """Answer, persist, then stream the persisted sentences one at a time.

    The assistant message is written before the first `sentence` event, so
    a client disconnect (`GeneratorExit` at a `yield`) after the answer
    exists cannot skip saving it. In agent mode (`agent`: `run_chat_agent`'s
    `model` and `limits`) `progress` events come first; a disconnect during
    them stops the loop before its next step, and nothing is saved.

    `replace_message_id` (regenerate's old trailing answer) is deleted only
    after the new message is written, and never if `answer_question`
    raises, so a failed regenerate keeps the previous answer.

    `is_new_conversation` (first turn of a conversation `chat_stream`
    created): the `done` event carries the rendered `_conversation_item.html`
    row so the client can add it to the sidebar.
    """
    templates = request.app.state.templates
    sentence_template = templates.env.get_template("_chat_sentence.html")

    started = time.monotonic()
    try:
        if agent is None:
            answer = answer_question(conn, llm, embedder, question, k=8, expand=1, model=model)
        else:
            steps = run_chat_agent(
                conn,
                llm,
                embedder,
                question,
                fallback_model=model,
                tenant=tenant,
                history=_history_before_question(conn, conversation, editor),
                **agent,
            )
            while True:
                try:
                    line = next(steps)
                except StopIteration as finished:
                    answer = finished.value
                    break
                yield f"event: progress\ndata: {json.dumps({'text': line})}\n\n"
    except Exception:  # noqa: BLE001 - logged here, never surfaced to the client
        _logger.exception(
            "answer_question failed for conversation %s (editor=%s)", conversation.id, editor
        )
        _record_question(
            conn,
            editor=editor,
            question=question,
            conversation_id=conversation.id,
            latency_ms=round((time.monotonic() - started) * 1000),
            regenerate=replace_message_id is not None,
        )
        yield f"event: error\ndata: {json.dumps({'detail': GENERIC_ANSWER_ERROR})}\n\n"
        return

    _log_answer_usage(
        conn,
        tenant=tenant,
        answer=answer,
        question=question,
        editor=editor,
        conversation_id=conversation.id,
    )

    sentence_dicts = [sentence_to_dict(s) for s in answer.sentences]
    slugs = [c["note_slug"] for sd in sentence_dicts for c in sd["citations"]]
    summaries = note_summaries(conn, slugs)

    latency_ms = round((time.monotonic() - started) * 1000)
    content_md = sentences_to_content_md(answer)
    message = add_message(
        conn,
        conversation_id=conversation.id,
        owner=editor,
        role="assistant",
        content_md=content_md,
        citations=sentence_dicts,
        usage=answer.usage,
    )
    _record_question(
        conn,
        editor=editor,
        question=question,
        conversation_id=conversation.id,
        latency_ms=latency_ms,
        regenerate=replace_message_id is not None,
        answer=answer,
        message_id=message.id,
        content_md=content_md,
    )

    if replace_message_id is not None:
        delete_message(conn, replace_message_id, owner=editor)

    for sentence in sentence_dicts:
        html = sentence_template.render(
            sentence=sentence, note_summaries=summaries, message_id=message.id
        )
        yield f"event: sentence\ndata: {json.dumps({'html': html})}\n\n"

    done_payload: dict[str, Any] = {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "message_id": message.id,
        # The live footnote reads these; a reload reads the same values from
        # the stored `messages.usage`, so the two cannot disagree.
        "model": answer.usage.get("model", ""),
        "numerals_dropped": answer.usage.get("numerals_dropped", 0),
        "uncited_dropped": answer.usage.get("uncited_dropped", 0),
    }
    done_payload["feedback_html"] = templates.env.get_template("_answer_feedback.html").render(
        message_id=message.id, feedback=None
    )
    if is_new_conversation:
        conv_item_template = templates.env.get_template("_conversation_item.html")
        done_payload["conversation_item_html"] = conv_item_template.render(
            c=conversation, active_id=conversation.id
        )
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


@router.post("/chat/{conversation_id}/stream")
def chat_stream(
    request: Request,
    conversation_id: str,
    question: str = Form(...),
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
    llm: LLMClient = Depends(get_llm_dep),
    embedder: Embedder = Depends(get_embedder_dep),
    settings=Depends(get_settings_dep),
) -> StreamingResponse:
    question = require_question_length(question.strip(), max_chars=settings.chat_question_max_chars)
    is_new_conversation = conversation_id == "new"
    conversation = _resolve_or_create_conversation(
        conn, conversation_id, owner=session.editor, question=question
    )

    add_message(
        conn,
        conversation_id=conversation.id,
        owner=session.editor,
        role="user",
        content_md=question,
    )

    generator = _stream_answer(
        request,
        conn,
        llm,
        embedder,
        tenant=settings.tenant,
        editor=session.editor,
        conversation=conversation,
        question=question,
        model=resolve_chat_model(settings),
        is_new_conversation=is_new_conversation,
        agent=agent_options(settings),
    )
    return StreamingResponse(generator, media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/chat/{conversation_id}/regenerate")
def chat_regenerate(
    request: Request,
    conversation_id: int,
    session: Session = Depends(require_session_api),
    conn: psycopg.Connection = Depends(get_conn),
    llm: LLMClient = Depends(get_llm_dep),
    embedder: Embedder = Depends(get_embedder_dep),
    settings=Depends(get_settings_dep),
) -> StreamingResponse:
    """Re-answers the last user question and replaces the trailing
    assistant message: one new `messages` row and one `usage_events` row.
    The old message is deleted only after the new one is written
    (`_stream_answer`).
    """
    conversation = get_conversation(conn, conversation_id, owner=session.editor)
    if conversation is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    user_message = last_user_message(conn, conversation_id, owner=session.editor)
    if user_message is None:
        raise HTTPException(
            status_code=400, detail="nothing to regenerate yet -- ask a question first"
        )

    trailing = last_message(conn, conversation_id, owner=session.editor)
    replace_message_id = (
        trailing.id if trailing is not None and trailing.role == "assistant" else None
    )

    generator = _stream_answer(
        request,
        conn,
        llm,
        embedder,
        tenant=settings.tenant,
        editor=session.editor,
        conversation=conversation,
        question=user_message.content_md,
        replace_message_id=replace_message_id,
        model=resolve_chat_model(settings),
        agent=agent_options(settings),
    )
    return StreamingResponse(generator, media_type="text/event-stream", headers=_SSE_HEADERS)


__all__ = ["DEFAULT_AGENT_SDK_CHAT_MODEL", "agent_options", "resolve_chat_model", "router"]
