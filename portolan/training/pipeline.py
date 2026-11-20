"""Per-corpus silver-labeling run (silver-labels build instructions items
3-5): read documents from `var/data/converted/<corpus>/`, build the same
`SourceEnvelope`s production would (`portolan.pipeline.build_envelopes`,
which chunks any source over ~4000 estimated tokens exactly the way a
real pipeline run does), extract with two open-weight free teachers per
envelope, gate their output through `portolan.training.agreement`, and
write `TrainingRecord`s.

Five of the six corpora (`fda_maude`, `ami`, `enron`, `cfpb`,
`saas_sales`) sit in a `corpus/out`-shaped directory already --
`calls/*.txt`, `email/*.txt`, `tickets/*.json` -- so their documents come
straight from the real `portolan.connectors.file_connector.FileConnector`,
not a bespoke reader (`01-spec/training-data-2026-09-19.md`: every one of
those converted files "round-tripped through the real FileConnector").
`ubuntu_irc` has no connector yet (same doc: "Ubuntu IRC has no connector
yet"), so `_iter_ubuntu_irc_documents` below builds `RawSource`s directly
from its Slack-shaped daily export, windowed into chat-sized chunks a
`SourceType.CHAT` source can actually be extracted from one call at a
time.

**Hold-out (build instructions item 5).** `assert_not_holdout` is called
on every root path this module ever opens a document from. `corpus/out/`
is the hand-built Halvard Medical corpus with a known answer key
(`corpus/truth-table.yaml`) -- the eval set a future fine-tune gets
scored against, never a training input. Silver labeling reads only
`var/data/converted/` and `var/data/converted/.../slack_export/`.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from corpus.generator.prose import TokenBucketLimiter
from portolan.extract.backends import Backend
from portolan.extract.prompt import PROMPT_VARIANTS, system_prompt_sha256
from portolan.extract.quote_gate import quote_is_supported
from portolan.extract.schema import Participant as ExtractParticipant
from portolan.extract.schema import SourceType, Vocabulary
from portolan.ingest.models import Participant as RawParticipant
from portolan.ingest.models import RawSource
from portolan.ingest.prepare import prepare_text
from portolan.metering import UsageSink
from portolan.pipeline import build_envelopes
from portolan.training.agreement import evaluate_document
from portolan.training.free_tier import FreeTierBudget
from portolan.training.local_teacher import is_local_teacher
from portolan.training.overlap_gate import overlap_gate_reason
from portolan.training.quote_span import resolve_quote_span
from portolan.training.records import (
    DroppedFact,
    TrainingRecord,
    append_dropped,
    append_record,
    read_records,
    teacher_answer_fields,
    teacher_fact_fields,
)
from portolan.training.rotation import (
    BudgetExhausted,
    CongestionEvent,
    HardFailure,
    ModelCooldownTracker,
    RotationCursor,
    select_two_teachers,
)
from portolan.training.spend import SpendMeter
from portolan.training.typed_judge import JevClient, ask_typed_questions, fact_type_criteria
from portolan.util import clock

__all__ = [
    "CORPUS_LICENSES",
    "CORPUS_KINDS",
    "Document",
    "document_envelopes",
    "SilverRunStats",
    "assert_not_holdout",
    "documents_for_corpus",
    "load_progress",
    "save_progress",
    "run_corpus",
    "JudgedRunStats",
    "run_corpus_judged",
    "SingleRunStats",
    "run_corpus_single",
]

CORPUS_LICENSES: dict[str, str] = {
    "fda_maude": "public domain (US government data, open.fda.gov)",
    "ami": "CC BY 4.0",
    "enron": "public record (FERC-compelled disclosure, 2003)",
    "ubuntu_irc": "public domain (Ubuntu IRC terms of service)",
    "cfpb": "public domain (US government data, consumerfinance.gov)",
    "saas_sales": "Apache 2.0 (DeepMostInnovations/saas-sales-conversations, Hugging Face)",
}

CORPUS_KINDS: dict[str, str] = {
    "fda_maude": "ticket",
    "ami": "call",
    "enron": "email",
    "ubuntu_irc": "chat",
    "cfpb": "ticket",
    "saas_sales": "call",
}

DEFAULT_CHAT_WINDOW_SIZE = 100


# --------------------------------------------------------------------------
# Hold-out guard
# --------------------------------------------------------------------------


def assert_not_holdout(path: Path) -> None:
    """Raise if `path` resolves anywhere under a `corpus/out/` directory
    -- the hand-built Halvard Medical exam corpus (build instructions item
    5: "Never read `corpus/out/`... Assert this in code"). Checked against
    the resolved path's own components, not a fixed repo-root prefix, so
    it catches `corpus/out` reached via a symlink or a relative path
    equally."""
    parts = path.resolve().parts
    for i in range(len(parts) - 1):
        if parts[i] == "corpus" and parts[i + 1] == "out":
            raise RuntimeError(
                f"hold-out violation: refusing to read {path} -- corpus/out/ is the "
                "held-out exam corpus, never a silver-labeling input "
                "(silver-labels build instructions item 5)"
            )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@dataclass
class Document:
    """One unit `run_corpus` extracts from: a real `RawSource` plus the
    kind/corpus bookkeeping `TrainingRecord` wants."""

    source_id: str
    raw: RawSource
    kind: str


def _to_extract_participants(raw_participants: list[RawParticipant]) -> list[ExtractParticipant]:
    return [
        ExtractParticipant(name=p.name, role=p.role, registry_id=None) for p in raw_participants
    ]


def document_envelopes(document: Document) -> list[Any]:
    """Build the exact `SourceEnvelope`s every `run_corpus*` variant below
    builds for `document` -- the same `prepare_text` -> `build_envelopes`
    steps, factored out (fix/overlap-gate-paraphrase) so a caller outside
    a live run can reproduce identical chunking offline.
    `scripts/data/regate_dropped.py` is the reason this needed to be
    public: it re-derives which envelope a `DroppedFact` (which only
    carries the document's bare `source_id`, never a `#segN` suffix) came
    from by rebuilding every envelope for that source THIS way and
    checking which one's text actually supports the dropped quote."""
    prepared = prepare_text(document.raw)
    return build_envelopes(
        document.raw,
        prepared.text,
        participants=_to_extract_participants(document.raw.participants),
        registry_context=[],
        vocabulary=Vocabulary(),
    )


def _iter_file_connector_documents(root: Path, kind: str) -> Iterator[Document]:
    assert_not_holdout(root)
    from portolan.connectors.file_connector import FileConnector

    connector = FileConnector(root)
    for ref in connector.list_sources():
        raw = connector.fetch(ref)
        yield Document(source_id=raw.source_id, raw=raw, kind=kind)


def _load_ubuntu_users(users_path: Path) -> dict[str, str]:
    data = json.loads(users_path.read_text(encoding="utf-8"))
    users: dict[str, str] = {}
    for entry in data:
        user_id = entry.get("id")
        if not user_id:
            continue
        profile = entry.get("profile") or {}
        users[user_id] = profile.get("real_name") or entry.get("name") or user_id
    return users


def _iter_ubuntu_irc_documents(
    channel_dir: Path,
    users_path: Path,
    *,
    window_size: int = DEFAULT_CHAT_WINDOW_SIZE,
    channel_name: str = "ubuntu",
) -> Iterator[Document]:
    """One `Document` per `window_size`-message window of one day's
    channel export (build instructions item 1: "chunked the way
    production chunks a source" -- windowing here keeps each document
    well under `build_envelopes`' own ~4000-estimated-token segmentation
    threshold, so a window is normally one `SourceEnvelope`/one model
    call, the same granularity a real Slack connector's own per-thread or
    per-session source would give the pipeline).

    Locators: `SourceType.CHAT` has no dedicated `prepare_text` handler
    (`portolan.ingest.prepare`'s own module docstring: "chat... fall[s]
    back to the doc behaviour (line numbers)") -- one `[Ln]` locator per
    message line, exactly as a real chat source would get today, not a
    Slack-specific locator this pipeline invents."""
    assert_not_holdout(channel_dir)
    assert_not_holdout(users_path)
    users = _load_ubuntu_users(users_path)

    for day_path in sorted(channel_dir.glob("*.json")):
        try:
            source_date = date.fromisoformat(day_path.stem)
        except ValueError:
            continue
        raw_messages = json.loads(day_path.read_text(encoding="utf-8"))
        messages = [
            m for m in raw_messages if m.get("type") == "message" and (m.get("text") or "").strip()
        ]
        for window_index, start in enumerate(range(0, len(messages), window_size), start=1):
            window = messages[start : start + window_size]
            lines: list[str] = []
            speakers: dict[str, None] = {}
            for message in window:
                user_id = message.get("user", "")
                username = users.get(user_id, user_id or "unknown")
                lines.append(f"{username}: {message.get('text', '')}")
                speakers.setdefault(username, None)
            source_id = f"ubuntu-{day_path.stem}-w{window_index:03d}"
            raw = RawSource(
                source_id=source_id,
                source_type=SourceType.CHAT,
                source_date=source_date,
                participants=[RawParticipant(name=name) for name in speakers],
                text="\n".join(lines),
                metadata={"channel": channel_name},
            )
            yield Document(source_id=source_id, raw=raw, kind="chat")


def documents_for_corpus(
    corpus: str, converted_root: Path, *, window_size: int = DEFAULT_CHAT_WINDOW_SIZE
) -> Iterator[Document]:
    """`converted_root` is `var/data/converted/<corpus>` -- the caller's
    job (`scripts/data/silver_label.py`) is only to point this at the
    right directory; this function knows each corpus's own layout inside
    it."""
    if corpus in ("fda_maude", "ami", "enron", "cfpb", "saas_sales"):
        yield from _iter_file_connector_documents(converted_root, CORPUS_KINDS[corpus])
    elif corpus == "ubuntu_irc":
        slack_export = converted_root / "slack_export"
        yield from _iter_ubuntu_irc_documents(
            slack_export / "ubuntu", slack_export / "users.json", window_size=window_size
        )
    else:
        raise ValueError(f"documents_for_corpus: unknown corpus {corpus!r}")


# --------------------------------------------------------------------------
# Progress / resumability
# --------------------------------------------------------------------------


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("processed", []))


def save_progress(path: Path, processed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"processed": sorted(processed)}, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Run stats
# --------------------------------------------------------------------------


@dataclass
class SilverRunStats:
    corpus: str
    documents_seen: int = 0
    documents_processed: int = 0
    documents_skipped_resumed: int = 0
    envelopes_processed: int = 0
    facts_kept: int = 0
    facts_dropped_disagreement: int = 0
    facts_dropped_quote_gate: int = 0
    records_written: int = 0
    negatives_written: int = 0
    negatives_capped: int = 0
    failures: int = 0
    # Coordinator course-correction 2026-09-21: a congestion pause (every
    # model eligible for a slot was cooling down at once) is its own
    # counter, separate from `failures` -- it is NOT a document the run
    # gave up on, only a wait, logged once to `failures.jsonl` as a
    # diagnostic line (`kind: "congestion"`) before retrying.
    congestion_pauses: int = 0
    teacher_pairs_used: Counter = field(default_factory=Counter)
    # model id -> count of 429/5xx/timeout responses that model produced
    # this run -- the per-model yield-table column the coordinator asked
    # for.
    rate_limit_events: Counter = field(default_factory=Counter)
    requests_used_today: int | None = None
    stopped_reason: str | None = None


def _pair_key(a: str, b: str) -> str:
    return " + ".join(sorted((a, b)))


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _check_prompt_variant(prompt_variant: str) -> None:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"unknown prompt variant {prompt_variant!r}; choose one of {list(PROMPT_VARIANTS)}"
        )


def _record_provenance(
    system_prompt: str, prompt_variant: str, envelope: Any, result: Any
) -> dict[str, Any]:
    """What every `TrainingRecord` written here carries besides its facts:
    the prompt stamp and the teacher's own answer, enough for
    `portolan.training.sft_format` to rebuild the exact example."""
    return {
        "prompt_variant": prompt_variant,
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "user_message_inputs": envelope.model_dump(mode="json", exclude={"text"}),
        "teacher_answer": teacher_answer_fields(result),
    }


def run_corpus(
    *,
    corpus: str,
    kind: str,
    license_text: str,
    documents: Iterable[Document],
    system_prompt: str,
    prompt_variant: str,
    models: list[str],
    backend_factory: Callable[[str], Backend],
    limiter: TokenBucketLimiter,
    sink: UsageSink,
    tenant: str,
    records_file: Path,
    failures_file: Path,
    progress_file: Path,
    max_docs: int | None = None,
    negative_cap_ratio: float = 0.30,
    budget: FreeTierBudget | None = None,
    cooldown: ModelCooldownTracker | None = None,
    cursor: RotationCursor | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = lambda: clock.now().isoformat(),
    monotonic: Callable[[], float] = time.monotonic,
    on_summary: Callable[[SilverRunStats], None] | None = None,
    summary_every: int = 25,
) -> SilverRunStats:
    """Process `documents` for one corpus, writing `records_file` and
    `failures_file` and updating `progress_file` as it goes -- resumable
    (a source already in `progress_file` is skipped without spending any
    teacher calls on it) and negative-capped (build instructions item 2:
    "cap negatives at 30 percent of records", enforced against counts
    seeded from whatever `records_file` already holds, so a resumed run
    respects the same cap a fresh one would).

    Teacher selection (coordinator course-correction 2026-09-21):
    `models` (every discovered open-weight `:free` id, not a fixed pair)
    plus `cooldown`/`cursor` (`portolan.training.rotation`, fresh
    instances built when omitted) drive `select_two_teachers` per
    envelope -- a 429/5xx rotates to a different model immediately rather
    than retrying the same one; a document is only paused (never
    abandoned) when every eligible model is cooling down at once. A
    non-congestion (`HardFailure`) error still gets exactly one retry of
    the whole pair-selection (the original "retry once on transient
    failure" -- silver-labels build instructions item 4) before being
    recorded to `failures.jsonl` and moved past.

    `budget` (`portolan.training.free_tier.FreeTierBudget`), when given,
    is checked before every real call and incremented by the number of
    real HTTP requests each call actually made (read off `sink.events`,
    since `LLMClientBackend.extract()` can make more than one real
    request per call -- a strict attempt plus a validation retry). Once
    exhausted, the run stops cleanly (`stopped_reason = "daily_budget"`)
    without spending another request.

    `prompt_variant` names the variant `system_prompt` was built with;
    every record written carries it (`_record_provenance`).
    """
    _check_prompt_variant(prompt_variant)
    stats = SilverRunStats(corpus=corpus)
    processed_ids = load_progress(progress_file)
    cooldown = cooldown or ModelCooldownTracker(now=monotonic)
    cursor = cursor or RotationCursor()

    total_written = 0
    negatives_written = 0
    for existing in read_records(records_file):
        total_written += 1
        if existing.is_negative:
            negatives_written += 1

    for document in documents:
        stats.documents_seen += 1

        if document.source_id in processed_ids:
            stats.documents_skipped_resumed += 1
            continue

        if max_docs is not None and stats.documents_processed >= max_docs:
            stats.stopped_reason = "max_docs"
            break

        envelopes = document_envelopes(document)

        for envelope in envelopes:

            def call_model(model: str, _envelope: Any = envelope) -> Any:
                limiter.acquire()
                if budget is not None and budget.is_exhausted():
                    raise BudgetExhausted(
                        f"daily free-tier budget exhausted ({budget.daily_limit} requests/day)"
                    )
                events_before = len(getattr(sink, "events", []))
                try:
                    return backend_factory(model).extract(_envelope, system_prompt, sink, tenant)
                finally:
                    if budget is not None:
                        delta = len(getattr(sink, "events", [])) - events_before
                        if delta > 0:
                            budget.record_requests(delta)

            def on_congestion(
                event: CongestionEvent, _envelope: Any = envelope, _document: Document = document
            ) -> None:
                stats.congestion_pauses += 1
                _write_jsonl(
                    failures_file,
                    {
                        "source_id": _document.source_id,
                        "corpus": corpus,
                        "kind": "congestion",
                        "envelope_index": _envelope.segment.index if _envelope.segment else 1,
                        "candidates": event.candidates,
                        "wait_seconds": round(event.wait_seconds, 1),
                        "timestamp": now(),
                    },
                )

            def on_model_failure(model: str, exc: Exception) -> None:
                stats.rate_limit_events[model] += 1

            try:
                try:
                    result_a, result_b, teacher_a, teacher_b = select_two_teachers(
                        models=models,
                        cooldown=cooldown,
                        cursor=cursor,
                        call_model=call_model,
                        sleep=sleep,
                        on_congestion=on_congestion,
                        on_model_failure=on_model_failure,
                    )
                except HardFailure:
                    # Silver-labels build instructions item 4: one retry
                    # on transient failure, for a non-congestion error
                    # only (congestion already got its "retry" via
                    # rotation to a different model, inside
                    # select_two_teachers itself).
                    sleep(2.0)
                    result_a, result_b, teacher_a, teacher_b = select_two_teachers(
                        models=models,
                        cooldown=cooldown,
                        cursor=cursor,
                        call_model=call_model,
                        sleep=sleep,
                        on_congestion=on_congestion,
                        on_model_failure=on_model_failure,
                    )
            except BudgetExhausted:
                stats.stopped_reason = "daily_budget"
                if budget is not None:
                    stats.requests_used_today = budget.used_today()
                save_progress(progress_file, processed_ids)
                return stats
            except HardFailure as exc:
                _write_jsonl(
                    failures_file,
                    {
                        "source_id": document.source_id,
                        "corpus": corpus,
                        "kind": "hard_failure",
                        "envelope_index": envelope.segment.index if envelope.segment else 1,
                        "teachers_tried": exc.teachers_tried,
                        "error": str(exc),
                        "timestamp": now(),
                    },
                )
                stats.failures += 1
                continue

            stats.envelopes_processed += 1
            stats.teacher_pairs_used[_pair_key(teacher_a, teacher_b)] += 1

            outcome = evaluate_document(list(result_a.facts), list(result_b.facts), envelope.text)
            stats.facts_kept += len(outcome.kept)
            stats.facts_dropped_disagreement += outcome.dropped_disagreement
            stats.facts_dropped_quote_gate += outcome.dropped_quote_gate

            is_negative = outcome.is_negative
            if is_negative:
                projected_ratio = (negatives_written + 1) / (total_written + 1)
                if total_written > 0 and projected_ratio > negative_cap_ratio:
                    stats.negatives_capped += 1
                    continue

            record_id = document.source_id
            if envelope.segment is not None:
                seg = envelope.segment
                record_id = f"{document.source_id}#seg{seg.index}of{seg.total}"

            record = TrainingRecord(
                record_id=record_id,
                corpus=corpus,
                source_id=document.source_id,
                kind=kind,
                license=license_text,
                text=envelope.text,
                facts=outcome.kept,
                teachers=[teacher_a, teacher_b],
                agreement=outcome.agreement,
                created=now(),
                # kept facts are teacher A's (`agreement.evaluate_document`)
                **_record_provenance(system_prompt, prompt_variant, envelope, result_a),
            )
            append_record(records_file, record)
            total_written += 1
            stats.records_written += 1
            if is_negative:
                negatives_written += 1
                stats.negatives_written += 1

        processed_ids.add(document.source_id)
        save_progress(progress_file, processed_ids)
        stats.documents_processed += 1
        if on_summary and stats.documents_processed % summary_every == 0:
            on_summary(stats)

    if on_summary and (stats.documents_processed % summary_every != 0):
        on_summary(stats)

    if budget is not None:
        stats.requests_used_today = budget.used_today()

    return stats


# --------------------------------------------------------------------------
# Paid-teacher-plus-typed-judge run (coordinator course-correction,
# 2026-09-21): teacher one is a cheap paid open-weight model
# (`portolan.training.paid_teacher`), not a second free-teacher pass --
# every fact it emits that clears the mechanical quote gate goes to a
# typed judge (`portolan.training.typed_judge`, Jev and/or Kev) for the
# keep decision instead of a second extraction being diffed against the
# first. The original two-free-teacher `run_corpus` above stays exactly
# as it was -- it is what `--judge none` runs.
# --------------------------------------------------------------------------


@dataclass
class JudgedRunStats:
    corpus: str
    documents_seen: int = 0
    documents_processed: int = 0
    documents_skipped_resumed: int = 0
    envelopes_processed: int = 0
    facts_emitted: int = 0
    facts_dropped_quote_gate: int = 0
    facts_sent_to_judge: int = 0
    facts_kept: int = 0
    facts_dropped_judge: int = 0
    records_written: int = 0
    negatives_written: int = 0
    negatives_capped: int = 0
    failures: int = 0
    extractor_spend_usd: float = 0.0
    judge_spend_usd: float = 0.0
    total_spend_usd: float = 0.0
    stopped_reason: str | None = None


def _extract_paid_with_retry(
    backend: Backend,
    envelope: Any,
    system_prompt: str,
    sink: UsageSink,
    tenant: str,
    *,
    sleep: Callable[[float], None],
    retry_delay: float = 2.0,
) -> Any:
    """One real attempt against the single paid teacher; on any
    exception, sleep briefly and try exactly once more (silver-labels
    build instructions item 4, unchanged by the coordinator's teacher-
    design correction: "Retries once on transient failure"). There is no
    rotation target here -- one paid model, chosen for price, is the
    whole point -- so a second failure propagates to the caller."""
    try:
        return backend.extract(envelope, system_prompt, sink, tenant)
    except Exception:  # noqa: BLE001 - broad on purpose, see docstring
        sleep(retry_delay)
        return backend.extract(envelope, system_prompt, sink, tenant)


def _sink_cost_delta(sink: UsageSink, events_before: int) -> float:
    events = getattr(sink, "events", [])
    return sum((e.cost_estimate or 0.0) for e in events[events_before:])


def run_corpus_judged(
    *,
    corpus: str,
    kind: str,
    license_text: str,
    documents: Iterable[Document],
    system_prompt: str,
    prompt_variant: str,
    paid_model: str,
    backend_factory: Callable[[str], Backend],
    limiter: TokenBucketLimiter,
    sink: UsageSink,
    tenant: str,
    records_file: Path,
    failures_file: Path,
    progress_file: Path,
    judges: dict[str, JevClient],
    judge_mode: str,
    spend: SpendMeter,
    budget_usd: float,
    max_docs: int | None = None,
    negative_cap_ratio: float = 0.30,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = lambda: clock.now().isoformat(),
    on_summary: Callable[[JudgedRunStats], None] | None = None,
    summary_every: int = 25,
) -> JudgedRunStats:
    """Process `documents` with the paid-teacher-plus-typed-judge design.

    `judges`: `{"jev": JevClient(...)}` or `{"jev": ..., "kev": ...}` --
    every configured judge is queried and recorded on every quote-gate-
    passing fact (coordinator: "Record BOTH verdicts ... when both
    endpoints are configured"); `judge_mode` (`"jev"`, `"kev"`, or
    `"both"`) selects which verdict(s) drive the keep decision -- `"both"`
    requires every configured judge named in `judges` to say keep.

    `spend` is one shared meter for BOTH the paid extractor's real
    OpenRouter cost (read off `sink.events[i].cost_estimate`, the same
    per-request cost OpenRouter itself reported) and each Jev/Kev call's
    cost (`FactVerdict.cost_usd()`, billed input tokens only) --
    "Cap Jev spend with the same meter" (coordinator item 2). The run
    stops cleanly (`stopped_reason = "budget_usd"`) the moment `spend.
    total_usd() >= budget_usd`, checked before every real call, never
    mid-call.

    `prompt_variant`: as in `run_corpus`.
    """
    _check_prompt_variant(prompt_variant)
    stats = JudgedRunStats(corpus=corpus)
    processed_ids = load_progress(progress_file)
    criteria = fact_type_criteria()

    total_written = 0
    negatives_written = 0
    for existing in read_records(records_file):
        total_written += 1
        if existing.is_negative:
            negatives_written += 1

    def over_budget() -> bool:
        return spend.is_over_budget(budget_usd)

    for document in documents:
        stats.documents_seen += 1

        if document.source_id in processed_ids:
            stats.documents_skipped_resumed += 1
            continue

        if max_docs is not None and stats.documents_processed >= max_docs:
            stats.stopped_reason = "max_docs"
            break

        if over_budget():
            stats.stopped_reason = "budget_usd"
            break

        envelopes = document_envelopes(document)

        budget_stopped = False
        for envelope in envelopes:
            if over_budget():
                stats.stopped_reason = "budget_usd"
                budget_stopped = True
                break

            limiter.acquire()
            events_before = len(getattr(sink, "events", []))
            try:
                result = _extract_paid_with_retry(
                    backend_factory(paid_model), envelope, system_prompt, sink, tenant, sleep=sleep
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not re-raised
                _write_jsonl(
                    failures_file,
                    {
                        "source_id": document.source_id,
                        "corpus": corpus,
                        "kind": "hard_failure",
                        "envelope_index": envelope.segment.index if envelope.segment else 1,
                        "model": paid_model,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp": now(),
                    },
                )
                stats.failures += 1
                continue
            finally:
                extractor_cost = _sink_cost_delta(sink, events_before)
                if extractor_cost > 0:
                    spend.record(model=paid_model, corpus=corpus, amount_usd=extractor_cost)
                    stats.extractor_spend_usd += extractor_cost

            stats.envelopes_processed += 1
            stats.facts_emitted += len(result.facts)

            kept_facts: list[dict[str, Any]] = []
            sent_to_judge = 0
            for fact in result.facts:
                if not quote_is_supported(fact.quote, envelope.text):
                    stats.facts_dropped_quote_gate += 1
                    continue
                if over_budget():
                    stats.stopped_reason = "budget_usd"
                    budget_stopped = True
                    break

                sent_to_judge += 1
                stats.facts_sent_to_judge += 1
                verdicts: dict[str, Any] = {}
                keep_by_judge: dict[str, bool] = {}
                for judge_name, client in judges.items():
                    verdict = ask_typed_questions(
                        client, fact.claim, fact.quote or "", criteria=criteria
                    )
                    verdicts[judge_name] = verdict
                    keep_by_judge[judge_name] = verdict.keep(
                        fact.type.value if hasattr(fact.type, "value") else str(fact.type)
                    )
                    judge_cost = verdict.cost_usd()
                    if judge_cost > 0:
                        spend.record(
                            model=f"judge:{judge_name}", corpus=corpus, amount_usd=judge_cost
                        )
                        stats.judge_spend_usd += judge_cost

                if judge_mode == "both":
                    keep = all(keep_by_judge.get(name, False) for name in judges)
                else:
                    keep = keep_by_judge.get(judge_mode, False)

                if keep:
                    stats.facts_kept += 1
                    kept_facts.append(
                        {
                            "fact_type": (
                                fact.type.value if hasattr(fact.type, "value") else str(fact.type)
                            ),
                            "claim": fact.claim,
                            "quote": fact.quote,
                            "locator": fact.locator,
                            "entities": list(fact.entities),
                            "judge": {
                                name: verdict.to_record_dict() for name, verdict in verdicts.items()
                            },
                            **teacher_fact_fields(fact),
                        }
                    )
                else:
                    stats.facts_dropped_judge += 1

            is_negative = len(kept_facts) == 0
            if is_negative:
                projected_ratio = (negatives_written + 1) / (total_written + 1)
                if total_written > 0 and projected_ratio > negative_cap_ratio:
                    stats.negatives_capped += 1
                    if budget_stopped:
                        break
                    continue

            record_id = document.source_id
            if envelope.segment is not None:
                seg = envelope.segment
                record_id = f"{document.source_id}#seg{seg.index}of{seg.total}"

            agreement = (len(kept_facts) / sent_to_judge) if sent_to_judge > 0 else 1.0

            record = TrainingRecord(
                record_id=record_id,
                corpus=corpus,
                source_id=document.source_id,
                kind=kind,
                license=license_text,
                text=envelope.text,
                facts=kept_facts,
                teachers=[paid_model],
                agreement=agreement,
                created=now(),
                mode="judge",
                **_record_provenance(system_prompt, prompt_variant, envelope, result),
            )
            append_record(records_file, record)
            total_written += 1
            stats.records_written += 1
            if is_negative:
                negatives_written += 1
                stats.negatives_written += 1

            if budget_stopped:
                break

        processed_ids.add(document.source_id)
        save_progress(progress_file, processed_ids)
        stats.documents_processed += 1
        stats.total_spend_usd = spend.total_usd()
        if on_summary and stats.documents_processed % summary_every == 0:
            on_summary(stats)

        if budget_stopped:
            break

    if on_summary and (stats.documents_processed % summary_every != 0):
        on_summary(stats)

    stats.total_spend_usd = spend.total_usd()
    return stats


# --------------------------------------------------------------------------
# Single-teacher, mechanical-overlap-gate run (coordinator course-
# correction, 2026-09-21, final design): one extractor, no cross-
# extractor agreement, no typed-judge gate -- neither reliably gated a
# kept fact in real runs (see `scripts/data/judge_calibration.py`'s
# output and this task's own report). The keep decision is
# `portolan.training.overlap_gate.passes_overlap_gate`, calibrated
# against the Halvard gold set (`scripts/data/overlap_calibration.py`).
# A configured judge (normally Kev only, to save Jev's cost) still runs
# on every kept fact and its verdict is recorded, purely
# informationally -- it never gates here either.
# --------------------------------------------------------------------------


@dataclass
class SingleRunStats:
    corpus: str
    documents_seen: int = 0
    documents_processed: int = 0
    documents_skipped_resumed: int = 0
    envelopes_processed: int = 0
    facts_emitted: int = 0
    facts_dropped_quote_gate: int = 0
    facts_dropped_overlap: int = 0
    # feature/labeling-exact-quotes: a fact that clears the overlap gate
    # but whose quote cannot be resolved to an exact span of the record
    # text (`portolan.training.quote_span.resolve_quote_span`) -- dropped
    # rather than kept with a paraphrased/stitched quote a student model
    # could not learn to copy.
    facts_dropped_quote_not_exact: int = 0
    facts_kept: int = 0
    facts_kept_by_type: Counter = field(default_factory=Counter)
    records_written: int = 0
    negatives_written: int = 0
    negatives_capped: int = 0
    failures: int = 0
    extractor_spend_usd: float = 0.0
    judge_spend_usd: float = 0.0
    total_spend_usd: float = 0.0
    stopped_reason: str | None = None


def run_corpus_single(
    *,
    corpus: str,
    kind: str,
    license_text: str,
    documents: Iterable[Document],
    system_prompt: str,
    prompt_variant: str,
    teacher_model: str,
    overlap_threshold: float,
    backend_factory: Callable[[str], Backend],
    limiter: TokenBucketLimiter,
    sink: UsageSink,
    tenant: str,
    records_file: Path,
    failures_file: Path,
    progress_file: Path,
    dropped_file: Path,
    spend: SpendMeter,
    budget_usd: float,
    judges: dict[str, JevClient] | None = None,
    max_docs: int | None = None,
    negative_cap_ratio: float = 0.30,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = lambda: clock.now().isoformat(),
    on_summary: Callable[[SingleRunStats], None] | None = None,
    summary_every: int = 25,
    timings_file: Path | None = None,
    clock_seconds: Callable[[], float] = time.perf_counter,
) -> SingleRunStats:
    """One `teacher_model` call per envelope; a fact is kept only when it
    clears the overlap gate (`portolan.training.overlap_gate.
    overlap_gate_reason`, threshold `overlap_threshold`) AND its quote
    resolves to an exact span of `envelope.text`
    (`portolan.training.quote_span.resolve_quote_span`,
    feature/labeling-exact-quotes) -- no second extractor, no typed-judge
    gate. A kept fact's `quote` is overwritten with the resolved exact
    substring and carries `quote_span`; every dropped fact (quote gate,
    overlap gate, or unresolvable quote) is appended to `dropped_file`
    (`portolan.training.records.DroppedFact`). Every configured `judges`
    entry (informational, `judge_mode` doesn't apply here: ALL configured
    judges are recorded on every kept fact) still runs and its verdict is
    attached to `facts[].judge`, spend-tracked into the same `spend`
    meter, but never changes what was kept.

    `prompt_variant`: as in `run_corpus`; dropped facts carry it too.

    A local teacher (`local:<repo id>`, `portolan.training.local_teacher`)
    costs nothing, so each of its calls writes a $0 spend line naming the
    model. `timings_file`, when given, gets one JSON line per processed
    document with its wall time (`clock_seconds`) and envelope count."""
    _check_prompt_variant(prompt_variant)
    prompt_sha = system_prompt_sha256(system_prompt)
    stats = SingleRunStats(corpus=corpus)
    local_teacher = is_local_teacher(teacher_model)
    processed_ids = load_progress(progress_file)
    criteria = fact_type_criteria() if judges else None

    total_written = 0
    negatives_written = 0
    for existing in read_records(records_file):
        total_written += 1
        if existing.is_negative:
            negatives_written += 1

    def over_budget() -> bool:
        return spend.is_over_budget(budget_usd)

    for document in documents:
        stats.documents_seen += 1

        if document.source_id in processed_ids:
            stats.documents_skipped_resumed += 1
            continue

        if max_docs is not None and stats.documents_processed >= max_docs:
            stats.stopped_reason = "max_docs"
            break

        if over_budget():
            stats.stopped_reason = "budget_usd"
            break

        document_started = clock_seconds()
        envelopes = document_envelopes(document)

        budget_stopped = False
        for envelope in envelopes:
            if over_budget():
                stats.stopped_reason = "budget_usd"
                budget_stopped = True
                break

            limiter.acquire()
            events_before = len(getattr(sink, "events", []))
            try:
                result = _extract_paid_with_retry(
                    backend_factory(teacher_model),
                    envelope,
                    system_prompt,
                    sink,
                    tenant,
                    sleep=sleep,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not re-raised
                _write_jsonl(
                    failures_file,
                    {
                        "source_id": document.source_id,
                        "corpus": corpus,
                        "kind": "hard_failure",
                        "envelope_index": envelope.segment.index if envelope.segment else 1,
                        "model": teacher_model,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp": now(),
                    },
                )
                stats.failures += 1
                continue
            finally:
                extractor_cost = _sink_cost_delta(sink, events_before)
                if extractor_cost > 0:
                    spend.record(model=teacher_model, corpus=corpus, amount_usd=extractor_cost)
                    stats.extractor_spend_usd += extractor_cost
                elif local_teacher:
                    spend.record(
                        model=teacher_model, corpus=corpus, amount_usd=0.0, allow_zero=True
                    )

            stats.envelopes_processed += 1
            stats.facts_emitted += len(result.facts)

            kept_facts: list[dict[str, Any]] = []
            for fact in result.facts:
                fact_type = fact.type.value if hasattr(fact.type, "value") else str(fact.type)
                raw_fact = {
                    "fact_type": fact_type,
                    "claim": fact.claim,
                    "quote": fact.quote,
                    "locator": fact.locator,
                    "entities": list(fact.entities),
                    **teacher_fact_fields(fact),
                }

                reason = overlap_gate_reason(
                    claim=fact.claim,
                    quote=fact.quote,
                    doc_text=envelope.text,
                    threshold=overlap_threshold,
                )
                if reason is not None:
                    if reason == "quote_gate":
                        stats.facts_dropped_quote_gate += 1
                    else:
                        stats.facts_dropped_overlap += 1
                    append_dropped(
                        dropped_file,
                        DroppedFact(
                            source_id=document.source_id,
                            reason=reason,
                            fact=raw_fact,
                            teacher=teacher_model,
                            created=now(),
                            prompt_variant=prompt_variant,
                            system_prompt_sha256=prompt_sha,
                        ),
                    )
                    continue

                resolved = resolve_quote_span(fact.quote, envelope.text)
                if resolved is None:
                    stats.facts_dropped_quote_not_exact += 1
                    append_dropped(
                        dropped_file,
                        DroppedFact(
                            source_id=document.source_id,
                            reason="quote_not_exact",
                            fact=raw_fact,
                            teacher=teacher_model,
                            created=now(),
                            prompt_variant=prompt_variant,
                            system_prompt_sha256=prompt_sha,
                        ),
                    )
                    continue
                assert envelope.text[resolved.span[0] : resolved.span[1]] == resolved.quote

                if over_budget():
                    stats.stopped_reason = "budget_usd"
                    budget_stopped = True
                    break

                judge_verdicts: dict[str, dict[str, Any]] = {}
                for judge_name, client in (judges or {}).items():
                    verdict = ask_typed_questions(
                        client, fact.claim, fact.quote or "", criteria=criteria
                    )
                    judge_verdicts[judge_name] = verdict.to_record_dict()
                    judge_cost = verdict.cost_usd()
                    if judge_cost > 0:
                        spend.record(
                            model=f"judge:{judge_name}", corpus=corpus, amount_usd=judge_cost
                        )
                        stats.judge_spend_usd += judge_cost

                stats.facts_kept += 1
                stats.facts_kept_by_type[fact_type] += 1
                kept_facts.append(
                    {
                        "fact_type": fact_type,
                        "claim": fact.claim,
                        "quote": resolved.quote,
                        "quote_span": list(resolved.span),
                        "locator": fact.locator,
                        "entities": list(fact.entities),
                        "judge": judge_verdicts,
                        **teacher_fact_fields(fact),
                    }
                )

            is_negative = len(kept_facts) == 0
            if is_negative:
                projected_ratio = (negatives_written + 1) / (total_written + 1)
                if total_written > 0 and projected_ratio > negative_cap_ratio:
                    stats.negatives_capped += 1
                    if budget_stopped:
                        break
                    continue

            record_id = document.source_id
            if envelope.segment is not None:
                seg = envelope.segment
                record_id = f"{document.source_id}#seg{seg.index}of{seg.total}"

            record = TrainingRecord(
                record_id=record_id,
                corpus=corpus,
                source_id=document.source_id,
                kind=kind,
                license=license_text,
                text=envelope.text,
                facts=kept_facts,
                teachers=[teacher_model],
                agreement=(len(kept_facts) / len(result.facts) if result.facts else 1.0),
                created=now(),
                mode="single",
                **_record_provenance(system_prompt, prompt_variant, envelope, result),
            )
            append_record(records_file, record)
            total_written += 1
            stats.records_written += 1
            if is_negative:
                negatives_written += 1
                stats.negatives_written += 1

            if budget_stopped:
                break

        processed_ids.add(document.source_id)
        save_progress(progress_file, processed_ids)
        stats.documents_processed += 1
        stats.total_spend_usd = spend.total_usd()
        if timings_file is not None:
            _write_jsonl(
                timings_file,
                {
                    "source_id": document.source_id,
                    "corpus": corpus,
                    "teacher": teacher_model,
                    "envelopes": len(envelopes),
                    "seconds": round(clock_seconds() - document_started, 3),
                    "timestamp": now(),
                },
            )
        if on_summary and stats.documents_processed % summary_every == 0:
            on_summary(stats)

        if budget_stopped:
            break

    if on_summary and (stats.documents_processed % summary_every != 0):
        on_summary(stats)

    stats.total_spend_usd = spend.total_usd()
    return stats
