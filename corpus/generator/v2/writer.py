"""WP4: the writer (`01-spec/world-generator-v2-2026-09-22.md` sections 1 to
5 and 8; section 9's WP4 row: "format specs, prompt assembly, markers with
decoys, the eight verification checks, targeted repair, regeneration,
rotation, 25% cap, spend lines, resume").

For one `RenderedDoc` (WP3's skeleton) plus its `DocPlan`, the `World` and
the pack, `write_document`:

1. assembles the writer prompt (`writer_prompt`) from the per-kind format
   spec (`formats/<kind>.md`), the world slice at the document's date, the
   cast's personas, the slot script, and the event-cluster or topic
   context;
2. calls one open-weight model from `ALLOWED_WRITER_MODELS` through
   OpenRouter (never another id; Nemotron and `:free` ids refused);
3. parses the reply's ⟦F…⟧ fact and ⟦D…⟧ decoy markers into a finished
   document with code-rendered structure (`assembly`);
4. runs the spec's eight checks on every slot (`verify`), measured against
   the text `prepare_text` actually produces;
5. on a failing fact, makes ONE targeted repair call (same writer) that
   rewrites only the named units; if a fact still fails, regenerates the
   whole document ONCE on a different writer; after that a failing fact is
   dropped from the labels and logged. A failing decoy is dropped without
   any extra call (spec section 3.2: "a missing decoy is simply dropped").

**Provenance (binding).** Every word in a filled document or a label comes
from the open-weight writer. Code inserts, strips and normalizes marker
glyphs and renders structure (timestamps, JSON, headers, ids, quote
blocks). It never writes fallback prose: a document whose final attempt is
still structurally broken is dropped, never filled by code.

`run_world` drives a whole world (or a chosen subset): date order, the 25%
per-writer cap, spend lines on the shared `SpendMeter` under corpus key
`synthetic_v2` with a hard per-run dollar cap checked before every call, a
per-document JSON checkpoint for resume, and a run report.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from corpus.generator import prose
from corpus.generator.v2.assembly import (
    BLOCK_KINDS,
    LINE_KINDS,
    SPEAKER_KINDS,
    AssembledDoc,
    Draft,
    DraftUnit,
    PeopleIndex,
    PreparedSource,
    assemble,
    parse_draft,
    prepare_sources,
    skeleton_requests_subject,
)
from corpus.generator.v2.models import EventType, Person, PersonSide, World
from corpus.generator.v2.packs.schema import VerticalPack
from corpus.generator.v2.planner import DocKind, DocPlan, DocRole, PlannedDecoy, WorldDocPlan
from corpus.generator.v2.renderers import RenderedDoc, SlotKind, render
from corpus.generator.v2.verify import (
    CHECKS_VERSION,
    DecoySpan,
    DocVerification,
    SlotResult,
    SlotSpec,
    WriterLabel,
    build_slot_specs,
    copied_elsewhere,
    is_claim_copy,
    make_labels,
    pasted_sentence,
    verify_document,
)
from corpus.generator.v2.writer_prompt import (
    PROMPT_VERSION,
    PromptContext,
    RepairTarget,
    build_initial_prompt,
    build_repair_prompt,
    fact_instruction,
    length_target,
)
from portolan.training.local_teacher import (
    is_local_teacher,
    make_local_client,
    permitted_local_teacher_ids,
)
from portolan.training.spend import SpendMeter, default_spend_path

__all__ = [
    "ALLOWED_WRITER_MODELS",
    "CORPUS_KEY",
    "DEFAULT_ROTATION",
    "LocalWriterBackend",
    "OpenRouterWriterBackend",
    "RoutingWriterBackend",
    "RunBudget",
    "WriterBackend",
    "WriterBudgetExceeded",
    "WriterConfig",
    "WriterPool",
    "WriterReply",
    "WriterSession",
    "WrittenDoc",
    "run_world",
    "write_document",
]

# Spend-meter corpus key for every writer call (`portolan.training.spend`).
CORPUS_KEY = "synthetic_v2"


# --------------------------------------------------------------------------
# Writer allowlist (open-weight only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterModelInfo:
    family: str
    prompt_usd_per_m: float
    completion_usd_per_m: float
    proven: bool  # used by the paid v1 synth runs (`var/ops/synth_paid.sh`)


# Spec section 4, "Writer rotation": eight open-weight models from five
# families. Prices read from OpenRouter `GET /api/v1/models` on 2026-09-22
# at 12:12 MST; used only for the pre-call worst-case budget check (the
# real cost recorded is OpenRouter's own `usage.cost`). Cohere, Gemma and
# Llama stay out (spec section 4); Nemotron 3.5 Lightning hangs and is
# refused by `_check_allowed` whatever list a caller passes. Local MLX models
# (`local:<repo id>`, free) join below from `portolan.training.local_teacher`,
# which lists each with the license that lets its outputs train a model
# (Gemma 4 is Apache 2.0, unlike the Gemma terms the spec excludes).
ALLOWED_WRITER_MODELS: dict[str, WriterModelInfo] = {
    "qwen/qwen3-30b-a3b-instruct-2507": WriterModelInfo("qwen", 0.048, 0.193, True),
    "deepseek/deepseek-v4-flash": WriterModelInfo("deepseek", 0.049, 0.098, True),
    "mistralai/mistral-small-3.2-24b-instruct": WriterModelInfo("mistral", 0.094, 0.250, True),
    "qwen/qwen3-32b": WriterModelInfo("qwen", 0.080, 0.280, True),
    "qwen/qwen3-235b-a22b-2507": WriterModelInfo("qwen", 0.087, 0.350, False),
    "z-ai/glm-4.7-flash": WriterModelInfo("glm", 0.061, 0.400, False),
    "mistralai/mistral-small-2603": WriterModelInfo("mistral", 0.150, 0.600, False),
    "openai/gpt-oss-120b": WriterModelInfo("gpt-oss", 0.150, 0.600, False),
    **{
        teacher: WriterModelInfo("local", 0.0, 0.0, False)
        for teacher in sorted(permitted_local_teacher_ids())
    },
}

# The four ids proven in the paid v1 synth runs first, then two more so the
# 25% cap always has slack (with only four writers, a 25% cap would force
# exact balance). `z-ai/glm-4.7-flash` stays allowed but out of the default
# rotation: in the 2026-09-22 WP4 smoke it looped one sentence for 4,700
# tokens on a call.
DEFAULT_ROTATION: tuple[str, ...] = (
    "qwen/qwen3-30b-a3b-instruct-2507",
    "deepseek/deepseek-v4-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "qwen/qwen3-235b-a22b-2507",
    "mistralai/mistral-small-2603",
)

_REFUSED_SUBSTRINGS = ("nemotron",)
_FALLBACK_PRICE = WriterModelInfo("unknown", 1.0, 3.0, False)


def _check_allowed(model: str) -> None:
    low = model.lower()
    if model.endswith(":free") or any(s in low for s in _REFUSED_SUBSTRINGS):
        raise ValueError(f"writer model {model!r} is refused (free tier or Nemotron)")
    if model not in ALLOWED_WRITER_MODELS:
        raise ValueError(f"writer model {model!r} is not in ALLOWED_WRITER_MODELS")


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterReply:
    text: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class WriterBackend(Protocol):
    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float | None
    ) -> WriterReply: ...


class WriterCallError(RuntimeError):
    """The backend could not get a reply from this model (HTTP error,
    timeout, empty body). The caller cools the model and tries another."""


class OpenRouterWriterBackend:
    """`portolan.llm.openrouter.OpenRouterClient` with an in-memory usage
    sink (no database), a short internal retry budget so a stuck model
    surfaces quickly to the rotation, and the API key read from the
    environment by code (never printed)."""

    def __init__(self, *, read_timeout_s: float = 150.0, max_retries: int = 1) -> None:
        import httpx

        from portolan.llm.openrouter import OpenRouterClient
        from portolan.metering import InMemorySink

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in the environment")
        self._client = OpenRouterClient(
            spend_key=CORPUS_KEY,
            api_key=api_key,
            default_model=DEFAULT_ROTATION[0],
            sink=InMemorySink(),
            tenant="corpus-generator-v2",
            timeout=httpx.Timeout(30.0, read=read_timeout_s),
            max_retries=max_retries,
        )

    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float | None
    ) -> WriterReply:
        from portolan.llm.base import LLMError

        _check_allowed(model)
        sink = self._client.sink
        before = len(sink.events)
        try:
            result = self._client.complete(
                system=system,
                user=user,
                model=model,
                surface="chat",
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMError as exc:
            raise WriterCallError(str(exc)) from exc
        finally:
            spent_events = sink.events[before:]
        cost = sum(e.cost_estimate or 0.0 for e in spent_events)
        if cost <= 0:
            info = ALLOWED_WRITER_MODELS.get(model, _FALLBACK_PRICE)
            cost = (
                result.input_tokens * info.prompt_usd_per_m
                + result.output_tokens * info.completion_usd_per_m
            ) / 1e6
        return WriterReply(result.text or "", cost, result.input_tokens, result.output_tokens)


class LocalWriterBackend:
    """A `local:<repo id>` writer run in-process by MLX
    (`portolan.training.local_teacher`). Holds one model at a time: a call
    for another local model unloads the current one first. Calls are free;
    with a `meter`, each writes a $0 line naming the model under
    `CORPUS_KEY`."""

    def __init__(self, *, meter: SpendMeter | None = None, client_factory: Any = None) -> None:
        self.meter = meter
        self._factory = client_factory or (lambda model: make_local_client(model))
        self._model: str | None = None
        self._client: Any = None

    def _client_for(self, model: str) -> Any:
        if model != self._model:
            if self._client is not None:
                self._client.unload()
            self._client = self._factory(model)
            self._model = model
        return self._client

    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float | None
    ) -> WriterReply:
        _check_allowed(model)
        if not is_local_teacher(model):
            raise WriterCallError(f"{model!r} is not a local model")
        client = self._client_for(model)
        try:
            result = client.complete(
                system=system,
                user=user,
                model=client.model_id,
                surface="chat",
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 - any local failure cools this writer
            raise WriterCallError(f"{model}: {type(exc).__name__}: {exc}") from exc
        if self.meter is not None:
            self.meter.record(model=model, corpus=CORPUS_KEY, amount_usd=0.0, allow_zero=True)
        return WriterReply(result.text or "", 0.0, result.input_tokens, result.output_tokens)


class RoutingWriterBackend:
    """`local:` ids to a `LocalWriterBackend`, every other id to OpenRouter
    (created on first use, so an all-local run needs no key)."""

    def __init__(self, local: WriterBackend, remote_factory: Any = None) -> None:
        self.local = local
        self._remote_factory = remote_factory or OpenRouterWriterBackend
        self._remote: WriterBackend | None = None

    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float | None
    ) -> WriterReply:
        if is_local_teacher(model):
            backend = self.local
        else:
            if self._remote is None:
                self._remote = self._remote_factory()
            backend = self._remote
        return backend.complete(
            model=model, system=system, user=user, max_tokens=max_tokens, temperature=temperature
        )


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


class WriterBudgetExceeded(prose.PaidBudgetExceeded):
    """Raised before a call that could take this run past its dollar cap.
    Carries `prose.PAID_BUDGET_STOP_MARKER` like every paid-budget stop."""


@dataclass
class RunBudget:
    """This run's own hard dollar cap, checked BEFORE every call against a
    worst-case projection (a token per three prompt characters, the full
    `max_tokens` completion, at the model's list price times
    `price_safety_factor`), plus an optional ceiling on the whole `synthetic_v2`
    corpus line of the shared meter. Every real cost goes to the meter."""

    cap_usd: float
    meter: SpendMeter | None = None
    corpus_key: str = CORPUS_KEY
    corpus_cap_usd: float | None = None
    # OpenRouter's listed price is one provider's; routing can land on a
    # dearer one for the same model id. The projection is multiplied by
    # this so the cap still holds.
    price_safety_factor: float = 3.0
    spent_usd: float = 0.0
    calls: int = 0

    def worst_case_usd(self, model: str, prompt_chars: int, max_tokens: int) -> float:
        info = ALLOWED_WRITER_MODELS.get(model, _FALLBACK_PRICE)
        list_price = (
            math.ceil(prompt_chars / 3.0) * info.prompt_usd_per_m
            + max_tokens * info.completion_usd_per_m
        ) / 1e6
        return list_price * self.price_safety_factor

    def precheck(self, model: str, prompt_chars: int, max_tokens: int) -> None:
        projected = self.spent_usd + self.worst_case_usd(model, prompt_chars, max_tokens)
        if projected > self.cap_usd:
            raise WriterBudgetExceeded(
                f"{prose.PAID_BUDGET_STOP_MARKER}: run spend ${self.spent_usd:.4f} plus this call's "
                f"worst case would pass the ${self.cap_usd:.2f} cap"
            )
        if (
            self.meter is not None
            and self.corpus_cap_usd is not None
            and self.meter.is_corpus_over_budget(self.corpus_key, self.corpus_cap_usd)
        ):
            raise WriterBudgetExceeded(
                f"{prose.PAID_BUDGET_STOP_MARKER}: {self.corpus_key} spend "
                f"${self.meter.corpus_usd(self.corpus_key):.4f} >= ${self.corpus_cap_usd:.2f}"
            )

    def record(self, model: str, cost_usd: float) -> None:
        self.calls += 1
        self.spent_usd = round(self.spent_usd + max(0.0, cost_usd), 8)
        if self.meter is not None and cost_usd > 0:
            self.meter.record(model=model, corpus=self.corpus_key, amount_usd=cost_usd)


# --------------------------------------------------------------------------
# Rotation with the 25% cap
# --------------------------------------------------------------------------


class WriterCapExhausted(RuntimeError):
    """Every allowed writer is excluded, capped or cooling past the wait
    budget -- the document cannot be written without breaking the cap."""


@dataclass
class WriterPool:
    """Round-robin over the rotation (spec section 4: "No model writes more
    than 25% of a world, and each document records its writer"), reusing
    v1's `prose.TeacherRotation` for per-model cooldowns after a failed
    call. `cap` is `floor(cap_share * world_doc_total)` (at least 1);
    `doc_counts` counts FINISHED documents by their final writer."""

    models: tuple[str, ...]
    world_doc_total: int
    cap_share: float = 0.25
    clock: Callable[[], float] | None = None
    sleep: Callable[[float], None] | None = None
    doc_counts: Counter = field(default_factory=Counter)
    _index: int = 0

    def __post_init__(self) -> None:
        for m in self.models:
            _check_allowed(m)
        kwargs: dict[str, Any] = {}
        if self.clock is not None:
            kwargs["clock"] = self.clock
        if self.sleep is not None:
            kwargs["sleep"] = self.sleep
        self.rotation = prose.TeacherRotation(list(self.models), **kwargs)

    @property
    def cap(self) -> int:
        return max(1, math.floor(self.cap_share * self.world_doc_total))

    def eligible(self, exclude: set[str]) -> list[str]:
        return [m for m in self.models if m not in exclude and self.doc_counts[m] < self.cap]

    def pick(self, exclude: set[str] | None = None, *, max_wait_s: float = 900.0) -> str:
        exclude = exclude or set()
        waited = 0.0
        while True:
            eligible = self.eligible(exclude)
            if not eligible:
                raise WriterCapExhausted(
                    f"no writer left under the {self.cap}-document cap (excluded {sorted(exclude)})"
                )
            n = len(self.models)
            for offset in range(n):
                candidate = self.models[(self._index + offset) % n]
                if candidate in eligible and not self.rotation.is_cooling(candidate):
                    self._index = (self._index + offset + 1) % n
                    self.rotation.request_counts[candidate] += 1
                    return candidate
            wait = min(self.rotation.cooldown_remaining(m) for m in eligible)
            if waited + wait > max_wait_s:
                raise WriterCapExhausted("every eligible writer is cooling down")
            self.rotation.sleep(wait)
            waited += wait

    def cool(self, model: str) -> None:
        self.rotation.record_rate_limit(model)

    def ok(self, model: str) -> None:
        self.rotation.mark_success(model)

    def assign(self, model: str) -> None:
        self.doc_counts[model] += 1

    def shares(self) -> dict[str, float]:
        total = sum(self.doc_counts.values()) or 1
        return {m: self.doc_counts[m] / total for m in self.models}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    call_kind: Literal["initial", "repair", "regenerate"]
    model: str
    prompt_sha256: str
    prompt_chars: int
    max_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    raw_output: str = ""
    at: str


class SlotOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    marker_id: str
    slot_kind: Literal["fact", "decoy"]
    status: Literal["verified", "repaired", "regenerated", "dropped"]
    first_failures: list[str] = Field(default_factory=list)
    final_failures: list[str] = Field(default_factory=list)
    dropped_text: str | None = None  # the failing span's text, for WP5 triage


class PreparedSourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: str
    path: str
    text: str


class WrittenDoc(BaseModel):
    """One document's WP4 outcome, persisted as `docs/<plan_id>.json`."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    kind: DocKind
    status: Literal["written", "dropped", "skipped", "error"]
    reason: str | None = None
    input_fingerprint: str
    writer_model: str | None = None
    final_call_kind: str | None = None
    subject: str | None = None
    word_count: int = 0
    files: dict[str, str] = Field(default_factory=dict)
    prepared_sources: list[PreparedSourceRecord] = Field(default_factory=list)
    labels: list[WriterLabel] = Field(default_factory=list)
    decoy_spans: list[DecoySpan] = Field(default_factory=list)
    slot_outcomes: list[SlotOutcome] = Field(default_factory=list)
    calls: list[CallRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    fatal: list[str] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def counts(self) -> dict[str, int]:
        facts = [o for o in self.slot_outcomes if o.slot_kind == "fact"]
        decoys = [o for o in self.slot_outcomes if o.slot_kind == "decoy"]
        return {
            "facts_planted": len(facts),
            "facts_verified_first_pass": sum(1 for o in facts if not o.first_failures),
            "facts_verified": sum(1 for o in facts if o.status != "dropped"),
            "facts_repaired": sum(1 for o in facts if o.status == "repaired"),
            "facts_regenerated": sum(1 for o in facts if o.status == "regenerated"),
            "facts_dropped": sum(1 for o in facts if o.status == "dropped"),
            "decoys_planted": len(decoys),
            "decoys_verified": sum(1 for o in decoys if o.status != "dropped"),
            "decoys_dropped": sum(1 for o in decoys if o.status == "dropped"),
            # What actually ships: zero for a document that was dropped
            # even when some of its slots had verified.
            "facts_labeled": len(self.labels),
            "decoys_labeled": len(self.decoy_spans),
        }


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


@dataclass
class WriterConfig:
    models: tuple[str, ...] = DEFAULT_ROTATION
    cap_share: float = 0.25
    temperature: float = 0.8
    repair_temperature: float = 0.4
    max_model_switches: int = 3
    # Near-duplicates are WP6's writer calls (spec section 9: "near-duplicate
    # writer calls" belong to the noise stage).
    skip_roles: frozenset[DocRole] = frozenset({DocRole.NEAR_DUPLICATE})
    max_story_spans: int = 6
    max_sibling_titles: int = 5


@dataclass
class WriterSession:
    backend: WriterBackend
    pool: WriterPool
    budget: RunBudget
    config: WriterConfig = field(default_factory=WriterConfig)
    call_log: Path | None = None
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def log(self, record: CallRecord) -> None:
        if self.call_log is None:
            return
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")


@dataclass
class DocContext:
    """Cross-document context (spec section 2, "Two tiers"): verified span
    texts of earlier documents in the same event cluster, and titles of
    earlier routine documents on the same topic leaf."""

    story_so_far: list[str] = field(default_factory=list)
    sibling_titles: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Cast
# --------------------------------------------------------------------------


def _pick(people: list[Person], *key: object) -> Person | None:
    if not people:
        return None
    digest = int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:12], 16)
    return people[digest % len(people)]


def skeleton_size(rendered: RenderedDoc) -> tuple[int, int] | None:
    """`(total target words, unit count)` of WP3's skeleton: the length the
    planner gave this document (spec section 2's realism targets)."""
    body = [s for s in rendered.slots if s.kind != SlotKind.SUBJECT]
    words = sum(s.target_words or 0 for s in body)
    units = len({s.locator for s in body if s.locator})
    return (words, units) if words and units else None


def structural_text(rendered: RenderedDoc) -> str:
    """Code-rendered header text that reaches the extractor (a call's or a
    doc's Account and Attendees lines): names in it need no mention."""
    for path, content in rendered.files.items():
        if path.startswith(("calls/", "docs/", "meeting_notes/")):
            head = content.split("[SLOT", 1)[0]
            return re.sub(r"^\[\d{2}:\d{2}:\d{2}\][^\n]*$", "", head, flags=re.MULTILINE)
    return ""


def departed_before(world: World, on: date) -> dict[str, date]:
    """Person id -> the date a contact-change fact says they left, for
    departures before `on`. WP1 keeps a departed champion on the roster
    (WP3 review, not-fixed 1), so a later document could still give them a
    line; the writer never does."""
    people = world.person_by_id()
    out: dict[str, date] = {}
    for f in world.facts:
        if f.fact_type.value != "contact_change" or f.date >= on:
            continue
        for e in f.entities:
            if e in people and people[e].side == PersonSide.CUSTOMER and e != f.speaker:
                out[e] = min(out.get(e, f.date), f.date)
    return out


def slot_specs_for(
    rendered: RenderedDoc,
    plan: DocPlan,
    world: World,
    decoys_by_id: dict[str, PlannedDecoy],
) -> list[SlotSpec]:
    """The slot specs the writer works from: `verify.build_slot_specs`,
    with a speaker who already left replaced by a colleague still there."""
    departed = departed_before(world, plan.date)
    return [
        _replace_departed_speaker(s, plan, world, departed)
        for s in build_slot_specs(rendered, plan, world, decoys_by_id)
    ]


def _mentioned(specs: list[SlotSpec], world: World) -> set[str]:
    facts = {f.id: f for f in world.facts}
    return {
        e for s in specs if s.is_fact and s.marker_id in facts for e in facts[s.marker_id].entities
    }


def _replace_departed_speaker(
    spec: SlotSpec, plan: DocPlan, world: World, departed: dict[str, date]
) -> SlotSpec:
    """A slot whose planned speaker already left is said by another person
    on the same side who is still there (the label records who said it)."""
    if spec.speaker_person_id not in departed:
        return spec
    people = world.person_by_id()
    gone = people[spec.speaker_person_id]
    candidates = [
        p
        for p in account_contacts(world, gone.account_id)
        if p.id not in departed
        and p.tenure.start <= plan.date
        and (p.tenure.end is None or p.tenure.end > plan.date)
    ]
    if not candidates:
        return dataclasses.replace(spec, speaker_person_id=None)
    choice = _pick(candidates, "departed-speaker", plan.id, spec.slot_id)
    assert choice is not None
    return dataclasses.replace(spec, speaker_person_id=choice.id)


def account_contacts(world: World, account_id: str | None) -> list[Person]:
    """Customer people who can speak for `account_id`: its own contacts, or
    when it has none (a system account in WP1's worlds) those of its
    facilities, then of its parent."""
    if not account_id:
        return []
    customers = [p for p in world.people if p.side == PersonSide.CUSTOMER]
    own = [p for p in customers if p.account_id == account_id]
    if own:
        return own
    children = {a.id for a in world.accounts if a.parent == account_id}
    via_children = [p for p in customers if p.account_id in children]
    if via_children:
        return via_children
    parent = world.account_by_id().get(account_id)
    return [p for p in customers if parent is not None and p.account_id == parent.parent]


def build_cast(
    plan: DocPlan, world: World, specs: list[SlotSpec]
) -> tuple[list[Person], Person | None, PeopleIndex]:
    """`(cast, author, people_index)`. The cast is the plan's participants
    plus every slot speaker; a speaking kind always gets both sides (a
    vendor person and, when the document has an account, one of its
    contacts) so a routine one-participant plan still reads as a real
    exchange. Prose kinds get an author (a vendor person)."""
    by_id = world.person_by_id()
    vendor_staff = [p for p in world.people if p.side == PersonSide.VENDOR]
    contacts = account_contacts(world, plan.account_id)
    ids = list(
        dict.fromkeys(
            [*plan.participants, *(s.speaker_person_id for s in specs if s.speaker_person_id)]
        )
    )
    cast = [by_id[i] for i in ids if i in by_id]
    if plan.role == DocRole.OFF_TOPIC:
        cast = [p for p in cast if p.side == PersonSide.VENDOR]
        while len(cast) < 3 and len(cast) < len(vendor_staff):
            extra = _pick(
                [p for p in vendor_staff if p not in cast], "offtopic", plan.id, len(cast)
            )
            if extra is None:
                break
            cast.append(extra)
    elif plan.kind in LINE_KINDS | BLOCK_KINDS:
        if not any(p.side == PersonSide.VENDOR for p in cast):
            extra = _pick(vendor_staff, "cast-vendor", plan.id)
            if extra is not None:
                cast.append(extra)
        if plan.account_id and not any(p.side == PersonSide.CUSTOMER for p in cast):
            extra = _pick(contacts, "cast-customer", plan.id)
            if extra is not None:
                cast.append(extra)
    author: Person | None = None
    if plan.kind not in LINE_KINDS | BLOCK_KINDS:
        author = next((p for p in cast if p.side == PersonSide.VENDOR), None) or _pick(
            vendor_staff, "author", plan.id
        )
        if author is not None and author not in cast:
            cast.insert(0, author)
    others = [p for p in [*vendor_staff, *contacts] if p not in cast]
    return cast, author, PeopleIndex(preferred=cast, others=others)


# --------------------------------------------------------------------------
# One attempt
# --------------------------------------------------------------------------


@dataclass
class Attempt:
    call_kind: str
    model: str
    draft: Draft
    assembled: AssembledDoc | None
    prepared: list[PreparedSource]
    verification: DocVerification


def _evaluate(
    call_kind: str,
    model: str,
    draft: Draft,
    *,
    plan: DocPlan,
    rendered: RenderedDoc,
    world: World,
    people: PeopleIndex,
    specs: list[SlotSpec],
    min_words: int = 0,
) -> Attempt:
    try:
        assembled = assemble(plan=plan, rendered=rendered, world=world, draft=draft, people=people)
        prepared = prepare_sources(assembled)
    except Exception as exc:  # noqa: BLE001 -- a broken draft is a fatal attempt, not a crash
        verification = DocVerification(
            [SlotResult(s, ["c0:document_not_assembled"]) for s in specs],
            [*draft.fatal, f"could not assemble: {exc}"],
            list(draft.notes),
        )
        return Attempt(call_kind, model, draft, None, [], verification)
    extra = list(draft.fatal)
    words = len(assembled.writer_text().split())
    if words < min_words:
        # A reply cut off mid-document (live 2026-09-22: a 35-word email
        # ending "We have been hearing from our clinical teams that").
        extra.append(
            f"document far shorter than planned ({words} of {min_words}+ words): truncated"
        )
    verification = verify_document(assembled, prepared, specs, world, extra_fatal=extra)
    return Attempt(call_kind, model, draft, assembled, prepared, verification)


# A document under this share of its planned length is a truncated reply.
MIN_LENGTH_SHARE = 0.3


def _max_tokens(plan: DocPlan, n_slots: int, skeleton: tuple[int, int] | None = None) -> int:
    words, _units, _noun = length_target(plan, n_slots, skeleton)
    return max(600, min(6000, int(words * 1.4 * 1.5) + 300 + 60 * n_slots))


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\x00\n{user}".encode()).hexdigest()


def _call(
    session: WriterSession,
    *,
    plan: DocPlan,
    call_kind: Literal["initial", "repair", "regenerate"],
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    calls: list[CallRecord],
) -> str:
    """One budget-checked, logged call. Raises `WriterCallError` (after
    logging it) when the model gave no usable reply."""
    session.budget.precheck(model, len(system) + len(user), max_tokens)
    record = CallRecord(
        plan_id=plan.id,
        call_kind=call_kind,
        model=model,
        prompt_sha256=_prompt_hash(system, user),
        prompt_chars=len(system) + len(user),
        max_tokens=max_tokens,
        at=session.now().isoformat(),
    )
    try:
        reply = session.backend.complete(
            model=model, system=system, user=user, max_tokens=max_tokens, temperature=temperature
        )
    except WriterCallError as exc:
        record.error = str(exc)[:500]
        calls.append(record)
        session.log(record)
        raise
    record.prompt_tokens = reply.prompt_tokens
    record.completion_tokens = reply.completion_tokens
    record.cost_usd = reply.cost_usd
    record.raw_output = reply.text
    if not reply.text.strip():
        record.error = "empty reply"
    session.budget.record(model, reply.cost_usd)
    calls.append(record)
    session.log(record)
    if record.error:
        raise WriterCallError(f"{model}: {record.error}")
    return reply.text


def _generate(
    session: WriterSession,
    ctx: PromptContext,
    *,
    call_kind: Literal["initial", "regenerate"],
    exclude: set[str],
    calls: list[CallRecord],
    evaluate: Callable[[str, str, Draft], Attempt],
    people: PeopleIndex,
    want_subject: bool,
) -> Attempt | None:
    """A full-document call on the next eligible writer, switching models
    (with a cooldown on the one that failed) up to `max_model_switches`
    times. `None` when no model produced a reply."""
    system, user = build_initial_prompt(ctx)
    max_tokens = _max_tokens(ctx.plan, len(ctx.specs), ctx.skeleton)
    tried: set[str] = set(exclude)
    for _ in range(session.config.max_model_switches + 1):
        try:
            model = session.pool.pick(tried)
        except WriterCapExhausted:
            return None
        tried.add(model)
        try:
            raw = _call(
                session,
                plan=ctx.plan,
                call_kind=call_kind,
                model=model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                temperature=session.config.temperature,
                calls=calls,
            )
        except WriterCallError:
            session.pool.cool(model)
            continue
        session.pool.ok(model)
        draft = parse_draft(ctx.plan.kind, raw, people, want_subject=want_subject)
        return evaluate(call_kind, model, draft)
    return None


# --------------------------------------------------------------------------
# Targeted repair
# --------------------------------------------------------------------------

_REPAIR_LABEL_RE = re.compile(r"^\s*=+\s*(NEW\s+AFTER\s+U\d+|U\d+)\s*=+.*$", re.IGNORECASE)


def _numbered_units(attempt: Attempt) -> list[str]:
    kind = attempt.assembled.kind if attempt.assembled else None
    out = []
    for i, u in enumerate(attempt.draft.units, start=1):
        if kind in LINE_KINDS:
            out.append(f"[U{i}] {u.author_name}: {u.text}")
        elif kind in BLOCK_KINDS:
            out.append(f"[U{i}] ### {u.author_name}\n{u.text}")
        else:
            out.append(f"[U{i}] {u.text}")
    return out


def _best_unit(claim: str, units: list[tuple[int, str]]) -> int | None:
    best, best_ratio = None, -1.0
    for index, text in units:
        ratio = difflib.SequenceMatcher(None, claim.lower(), text.lower()).ratio()
        if ratio > best_ratio:
            best, best_ratio = index, ratio
    return best


def _removal(marker_id: str) -> str:
    return (
        f"remove the ⟦{marker_id}⟧ markers from this unit and no longer state that fact here "
        "(the speaker may react to it instead)"
    )


def plan_repairs(attempt: Attempt, ctx: PromptContext) -> list[RepairTarget]:
    """Name the units one repair call must rewrite for every failing FACT
    slot (decoys are never repaired). A fact whose marker sits in one unit
    by the right speaker gets that unit rewritten; otherwise the best-
    matching unmarked unit by the right speaker is rewritten to carry it
    (or a new unit by that speaker is inserted), and any wrong unit that
    carries its marker is told to drop it. With no planned speaker, the
    right speaker is anyone on the fact's side (check 4's speaker side)."""
    assert attempt.assembled is not None
    world = ctx.world
    kind = attempt.assembled.kind
    units = attempt.assembled.units
    people = world.person_by_id()
    targets: dict[tuple[int, bool], RepairTarget] = {}

    def _target(index: int, insert: bool, author: str | None) -> RepairTarget:
        key = (index, insert)
        if key not in targets:
            targets[key] = RepairTarget(index, author, [], insert_after=insert)
        return targets[key]

    for r in attempt.verification.results:
        if r.passed or not r.spec.is_fact:
            continue
        spec = r.spec
        if spec.claim:
            for u in units:
                if u.index not in r.marker_units and is_claim_copy(
                    u.clean_text, spec.claim, spec.entity_words
                ):
                    _target(u.index, False, u.author_name).instructions.append(
                        f"this unit repeats the wording of the script note for {spec.marker_id}; "
                        "rewrite it so it no longer states that fact (it is stated elsewhere)"
                    )
        for ui in r.repeat_units:
            if ui not in r.marker_units:
                _target(ui, False, units[ui - 1].author_name).instructions.append(
                    f"this unit says the fact of {spec.marker_id} again; rewrite it so it no "
                    "longer states that fact (it is stated once, elsewhere)"
                )
        speaker_id = spec.speaker_person_id if kind in SPEAKER_KINDS else None
        side = spec.expected_side if kind in SPEAKER_KINDS else None

        def _right(person_id: str | None, speaker_id: str | None = speaker_id, side=side) -> bool:
            if speaker_id is not None:
                return person_id == speaker_id
            if side is None:
                return True
            person = people.get(person_id or "")
            return person is not None and person.side.value == side

        speaker_name = people[speaker_id].name if speaker_id in people else None
        if speaker_name is None and side is not None:
            speaker_name = next((u.author_name for u in units if _right(u.author_person_id)), None)
        problems = [f.split(":", 1)[1] for f in r.failures]
        c6_failed = any(f.startswith("c6:") for f in r.failures)

        def instruction(unit_index: int | None, spec=spec, problems=problems, c6=c6_failed) -> str:
            # A rewrite must keep a party name the unit already carries: it
            # may be the source's only mention (check 6).
            text = units[unit_index - 1].clean_text if unit_index else ""
            keeps = any(t.lower() in text.lower() for t in spec.party_terms)
            return fact_instruction(spec, ctx, problems=problems, name_parties=c6 or keeps)

        right = [ui for ui in r.marker_units if _right(units[ui - 1].author_person_id)]
        if right:
            keep = right[0]
            _target(keep, False, units[keep - 1].author_name).instructions.append(instruction(keep))
            for ui in r.marker_units:
                if ui != keep:
                    _target(ui, False, units[ui - 1].author_name).instructions.append(
                        _removal(spec.marker_id)
                    )
            continue
        free = [
            (u.index, u.clean_text)
            for u in units
            if _right(u.author_person_id)
            and not u.spans
            and not u.unbalanced
            and (u.index, False) not in targets
        ]
        chosen = _best_unit(spec.claim or "", free)
        if chosen is not None:
            _target(
                chosen, False, units[chosen - 1].author_name or speaker_name
            ).instructions.append(instruction(chosen))
        else:
            base = r.marker_units[0] if r.marker_units else len(units)
            _target(base, True, speaker_name).instructions.append(instruction(None))
        for ui in r.marker_units:
            if ui != chosen:
                _target(ui, False, units[ui - 1].author_name).instructions.append(
                    _removal(spec.marker_id)
                )
    return sorted(targets.values(), key=lambda t: (t.unit_index, t.insert_after))


def parse_repair_reply(raw: str) -> dict[str, str]:
    replies: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        m = _REPAIR_LABEL_RE.match(line)
        if m:
            if current is not None:
                replies[current] = "\n".join(buf).strip()
            current = re.sub(r"\s+", " ", m.group(1).upper())
            buf = []
            continue
        if current is not None and not line.strip().startswith("```"):
            buf.append(line)
    if current is not None:
        replies[current] = "\n".join(buf).strip()
    return {k: v for k, v in replies.items() if v}


def apply_repairs(
    draft: Draft, targets: list[RepairTarget], replies: dict[str, str], kind: DocKind, world: World
) -> tuple[Draft, list[str]]:
    """Splice the writer's rewritten units into a copy of `draft`. The
    unit's author is fixed by code; a leading `Name:` the writer repeated is
    removed; a line kind's reply is joined onto one line. A target with no
    reply keeps its old text."""
    people_by_name = {p.name: p for p in world.people}
    units = [DraftUnit(u.text, u.person_id, u.author_name) for u in draft.units]
    notes: list[str] = []
    inserts: list[tuple[int, DraftUnit]] = []
    for t in targets:
        text = replies.get(t.label.upper())
        if text is None:
            notes.append(f"repair reply had no text for {t.label}")
            continue
        if kind in LINE_KINDS:
            text = " ".join(text.split())
            prefix = re.match(r"^\s*\**\s*([^:]{1,60}?)\s*\**\s*:\s*\**\s*", text)
            if prefix and prefix.group(1).strip() in people_by_name:
                text = text[prefix.end() :]
        if t.author_name:
            text = re.sub(rf"^\s*\**\s*{re.escape(t.author_name)}\s*\**\s*:\s*\**\s*", "", text)
            text = re.sub(rf"^\s*#+\s*{re.escape(t.author_name)}\s*\n", "", text)
        others = [u.text for i, u in enumerate(units, start=1) if i != t.unit_index]
        pasted = pasted_sentence(text, others if not t.insert_after else [u.text for u in units])
        if pasted is not None:
            notes.append(
                f"repair reply for {t.label} pasted another unit's sentence; kept old text"
            )
            continue
        if t.insert_after:
            person = people_by_name.get(t.author_name or "")
            inserts.append(
                (t.unit_index, DraftUnit(text, person.id if person else None, t.author_name))
            )
        elif 1 <= t.unit_index <= len(units):
            units[t.unit_index - 1].text = text
    for index, unit in sorted(inserts, key=lambda p: p[0], reverse=True):
        units.insert(min(index, len(units)), unit)
    return Draft(draft.subject, units, [*draft.notes, *notes], list(draft.fatal)), notes


def _repair(
    session: WriterSession,
    ctx: PromptContext,
    attempt: Attempt,
    *,
    calls: list[CallRecord],
    evaluate: Callable[[str, str, Draft], Attempt],
) -> Attempt | None:
    targets = plan_repairs(attempt, ctx)
    if not targets:
        return None
    system, user = build_repair_prompt(ctx, _numbered_units(attempt), targets)
    max_tokens = min(4000, 400 + 350 * len(targets))
    try:
        raw = _call(
            session,
            plan=ctx.plan,
            call_kind="repair",
            model=attempt.model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=session.config.repair_temperature,
            calls=calls,
        )
    except WriterCallError:
        session.pool.cool(attempt.model)
        return None
    replies = parse_repair_reply(raw)
    draft, _notes = apply_repairs(attempt.draft, targets, replies, ctx.plan.kind, ctx.world)
    return evaluate("repair", attempt.model, draft)


# --------------------------------------------------------------------------
# write_document
# --------------------------------------------------------------------------


def input_fingerprint(
    rendered: RenderedDoc, plan: DocPlan, world: World, decoys_by_id: dict[str, PlannedDecoy]
) -> str:
    facts = {f.id: f for f in world.facts}
    payload = {
        "prompt_version": PROMPT_VERSION,
        "checks_version": CHECKS_VERSION,
        "rendered": rendered.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json", exclude={"writer_model"}),
        "facts": [facts[s.fact_id].model_dump(mode="json") for s in rendered.slots if s.fact_id],
        "decoys": [
            decoys_by_id[s.decoy_id].model_dump(mode="json")
            for s in rendered.slots
            if s.decoy_id and s.decoy_id in decoys_by_id
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _event_type(plan: DocPlan, world: World) -> EventType | None:
    if not plan.event_ids:
        return None
    events = {e.id: e for e in world.events}
    event = events.get(plan.event_ids[0])
    return event.type if event else None


def write_document(
    rendered: RenderedDoc,
    plan: DocPlan,
    world: World,
    pack: VerticalPack,
    *,
    decoys_by_id: dict[str, PlannedDecoy],
    session: WriterSession,
    context: DocContext | None = None,
    do_not_state: list[str] | None = None,
    avoid_models: set[str] | None = None,
) -> WrittenDoc:
    """Write one document end to end (module docstring, steps 1 to 5).
    Raises `WriterBudgetExceeded` (nothing half-recorded) when the next
    call would pass the run's cap. `do_not_state` and `avoid_models` are
    WP5's regeneration inputs (spec section 3.4): unscripted sentences an
    earlier draft stated, and the writers that must not write this one."""
    fingerprint = input_fingerprint(rendered, plan, world, decoys_by_id)
    if plan.role in session.config.skip_roles:
        return WrittenDoc(
            plan_id=plan.id,
            kind=plan.kind,
            status="skipped",
            reason=f"{plan.role.value} documents are written by WP6's noise stage",
            input_fingerprint=fingerprint,
        )
    context = context or DocContext()
    departed = departed_before(world, plan.date)
    specs = slot_specs_for(rendered, plan, world, decoys_by_id)
    cast, author, people = build_cast(plan, world, specs)
    cast = [p for p in cast if p.id not in departed or p.id in _mentioned(specs, world)]
    want_subject = skeleton_requests_subject(rendered)
    ctx = PromptContext(
        plan=plan,
        world=world,
        pack=pack,
        specs=specs,
        cast=cast,
        author=author,
        want_subject=want_subject,
        event_type=_event_type(plan, world),
        story_so_far=context.story_so_far[: session.config.max_story_spans],
        sibling_titles=context.sibling_titles[: session.config.max_sibling_titles],
        skeleton=skeleton_size(rendered),
        structural_text=structural_text(rendered),
        departed=frozenset(departed),
        do_not_state=list(do_not_state or []),
    )
    avoid = set(avoid_models or ())
    calls: list[CallRecord] = []

    planned_words, _units, _noun = length_target(plan, len(specs), ctx.skeleton)
    min_words = int(MIN_LENGTH_SHARE * planned_words)

    def evaluate(call_kind: str, model: str, draft: Draft) -> Attempt:
        return _evaluate(
            call_kind,
            model,
            draft,
            plan=plan,
            rendered=rendered,
            world=world,
            people=people,
            specs=specs,
            min_words=min_words,
        )

    first = _generate(
        session,
        ctx,
        call_kind="initial",
        exclude=set(avoid),
        calls=calls,
        evaluate=evaluate,
        people=people,
        want_subject=want_subject,
    )
    if first is None:
        return WrittenDoc(
            plan_id=plan.id,
            kind=plan.kind,
            status="error",
            reason="no writer produced a reply",
            input_fingerprint=fingerprint,
            calls=calls,
        )
    best = first

    def facts_failing(a: Attempt) -> bool:
        return any(r.spec.is_fact and not r.passed for r in a.verification.results)

    if not first.verification.fatal and facts_failing(first):
        repaired = _repair(session, ctx, first, calls=calls, evaluate=evaluate)
        if repaired is not None and repaired.verification.score() > first.verification.score():
            best = repaired

    limit = unscripted_limit(plan)
    if best.verification.fatal or facts_failing(best) or len(best.verification.unscripted) > limit:
        feedback = (
            [
                f"{r.spec.marker_id}: " + "; ".join(f.split(":", 1)[1] for f in r.failures)
                for r in best.verification.results
                if r.spec.is_fact and not r.passed
            ]
            + [f"document: {f}" for f in best.verification.fatal]
            + [
                "document: this sentence states something nobody scripted (a promise, a "
                f"figure, a date or a name); do not state it or anything like it: {u.split(': ', 1)[1]}"
                for u in best.verification.unscripted
            ]
        )
        regen_ctx = dataclasses.replace(ctx, feedback=feedback)
        regen = _generate(
            session,
            regen_ctx,
            call_kind="regenerate",
            exclude={first.model, *avoid},
            calls=calls,
            evaluate=evaluate,
            people=people,
            want_subject=want_subject,
        )
        # A tie goes to the regenerated document (spec section 3.2: "the
        # whole document is regenerated once on a different writer").
        if regen is not None and regen.verification.score() >= best.verification.score():
            best = regen

    return _finish(best, first, plan, world, decoys_by_id, fingerprint, calls)


# Check 14 limit per document (WP5's target, 2026-09-22: under 2 unscripted
# facts per call, under 1 elsewhere). Over it, the document is regenerated
# once with the sentences as "do not state"; it is never dropped for this.
UNSCRIPTED_LIMIT_CALL = 1
UNSCRIPTED_LIMIT_OTHER = 0


def unscripted_limit(plan: DocPlan) -> int:
    return UNSCRIPTED_LIMIT_CALL if plan.kind == DocKind.CALL else UNSCRIPTED_LIMIT_OTHER


def _finish(
    best: Attempt,
    first: Attempt,
    plan: DocPlan,
    world: World,
    decoys_by_id: dict[str, PlannedDecoy],
    fingerprint: str,
    calls: list[CallRecord],
) -> WrittenDoc:
    outcomes: list[SlotOutcome] = []
    for r in best.verification.results:
        r0 = first.verification.result_for(r.spec.slot_id)
        if not r.passed:
            status = "dropped"
        elif best.call_kind == "regenerate":
            status = "regenerated"
        elif best.call_kind == "repair" and not r0.passed:
            status = "repaired"
        else:
            status = "verified"
        outcomes.append(
            SlotOutcome(
                slot_id=r.spec.slot_id,
                marker_id=r.spec.marker_id,
                slot_kind="fact" if r.spec.is_fact else "decoy",
                status=status,
                first_failures=list(r0.failures),
                final_failures=list(r.failures),
                dropped_text=r.found.text if (not r.passed and r.found is not None) else None,
            )
        )
    base = {
        "plan_id": plan.id,
        "kind": plan.kind,
        "input_fingerprint": fingerprint,
        "writer_model": best.model,
        "final_call_kind": best.call_kind,
        "slot_outcomes": outcomes,
        "calls": calls,
        "notes": [
            *best.verification.notes,
            *(f"unscripted {u}" for u in best.verification.unscripted),
        ],
        "fatal": list(best.verification.fatal),
    }
    if best.verification.fatal or best.assembled is None:
        return WrittenDoc(
            status="dropped",
            reason="final attempt still structurally broken: " + "; ".join(best.verification.fatal),
            **base,
        )
    # An unmarked unit that repeats a claim's wording ships, with a note:
    # the templates are open-weight text since WP2 (spec Build notes), and
    # the repeat is a restatement WP5 labels. The marked span itself must
    # not copy (check 5), so no label is claim-shaped.
    copied = copied_elsewhere(best.verification, best.assembled)
    if copied:
        base["notes"] = [*base["notes"], f"unmarked text repeats the script wording of {copied}"]
    labels, decoy_spans = make_labels(
        best.verification,
        plan=plan,
        world=world,
        decoys_by_id=decoys_by_id,
        writer_model=best.model,
    )
    return WrittenDoc(
        status="written",
        subject=best.assembled.subject,
        word_count=len(best.assembled.writer_text().split()),
        files=best.assembled.files,
        prepared_sources=[
            PreparedSourceRecord(
                source_id=p.source_id, source_type=p.source_type, path=p.path, text=p.text
            )
            for p in best.prepared
        ],
        labels=labels,
        decoy_spans=decoy_spans,
        **base,
    )


# --------------------------------------------------------------------------
# run_world: ordering, cap, resume, persistence, report
# --------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_finished(path: Path, fingerprint: str) -> WrittenDoc | None:
    if not path.exists():
        return None
    try:
        doc = WrittenDoc.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if doc.input_fingerprint != fingerprint or doc.status == "error":
        return None
    return doc


def _prior_files(path: Path) -> set[str]:
    """Files an earlier result for this document wrote, whatever its
    fingerprint: a rewrite that no longer ships them must remove them."""
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("files") or {})
    except (OSError, ValueError, AttributeError):
        return set()


def _context_for(
    plan: DocPlan, finished: dict[str, WrittenDoc], plans: dict[str, DocPlan]
) -> DocContext:
    story: list[str] = []
    for other_id in plan.links.get("same_event", []):
        other = finished.get(other_id)
        other_plan = plans.get(other_id)
        if other is None or other_plan is None or other.status != "written":
            continue
        if (other_plan.date, other_plan.id) >= (plan.date, plan.id):
            continue
        story.extend(label.text[:220] for label in other.labels[:3])
    siblings: list[str] = []
    if plan.topic_leaf and not plan.event_ids:
        for other in finished.values():
            other_plan = plans.get(other.plan_id)
            if (
                other.status == "written"
                and other_plan is not None
                and other_plan.topic_leaf == plan.topic_leaf
                and other.plan_id != plan.id
            ):
                title = other.subject or " ".join(
                    (other.prepared_sources[0].text if other.prepared_sources else "").split()[:14]
                )
                siblings.append(f"{other_plan.kind.value}: {title}")
    return DocContext(story_so_far=story, sibling_titles=siblings)


def writer_doc_plans(world_plan: WorldDocPlan, config: WriterConfig) -> list[DocPlan]:
    """Every plan the writer is responsible for, in date order (spec
    section 2: event-cluster documents "are written in date order")."""
    return sorted(
        (d for d in world_plan.doc_plans if d.role not in config.skip_roles),
        key=lambda d: (d.date, d.id),
    )


def run_world(
    world: World,
    world_plan: WorldDocPlan,
    pack: VerticalPack,
    *,
    session: WriterSession,
    out_dir: Path,
    doc_ids: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write every writer-owned document of `world_plan` (or just
    `doc_ids`), resuming finished ones from `out_dir/docs/`. Returns (and
    writes) the run report."""
    from portolan.training.pipeline import assert_not_holdout

    assert_not_holdout(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decoys_by_id = {d.id: d for d in world_plan.decoys}
    plans_by_id = world_plan.doc_by_id()
    ordered = writer_doc_plans(world_plan, session.config)
    if doc_ids is not None:
        wanted = set(doc_ids)
        ordered = [p for p in ordered if p.id in wanted]
    finished: dict[str, WrittenDoc] = {}
    # Resume pass over every writer-owned document (not just this run's
    # subset) so the cap counts and the cluster context see earlier work.
    for plan in writer_doc_plans(world_plan, session.config):
        rendered = render(plan, world, world_plan, pack)
        fp = input_fingerprint(rendered, plan, world, decoys_by_id)
        prior = _load_finished(out_dir / "docs" / f"{plan.id}.json", fp)
        if prior is not None:
            finished[plan.id] = prior
            if prior.status == "written" and prior.writer_model:
                session.pool.assign(prior.writer_model)

    stop_reason: str | None = None
    this_run: list[str] = []
    resumed: list[str] = []
    for plan in ordered:
        if plan.id in finished:
            resumed.append(plan.id)
            continue
        rendered = render(plan, world, world_plan, pack)
        try:
            doc = write_document(
                rendered,
                plan,
                world,
                pack,
                decoys_by_id=decoys_by_id,
                session=session,
                context=_context_for(plan, finished, plans_by_id),
            )
        except WriterBudgetExceeded as exc:
            stop_reason = str(exc)
            break
        doc_path = out_dir / "docs" / f"{plan.id}.json"
        stale = _prior_files(doc_path) - set(doc.files)
        _atomic_write(doc_path, doc.model_dump_json(indent=2))
        for rel in stale:
            (out_dir / "files" / rel).unlink(missing_ok=True)
        if doc.status != "error":
            finished[plan.id] = doc
        if doc.status == "written":
            assert doc.writer_model is not None
            session.pool.assign(doc.writer_model)
            for rel, content in doc.files.items():
                _atomic_write(out_dir / "files" / rel, content)
        this_run.append(plan.id)
        if progress is not None:
            c = doc.counts()
            progress(
                f"{plan.id} {doc.status} writer={doc.writer_model} facts "
                f"{c['facts_verified']}/{c['facts_planted']} decoys {c['decoys_verified']}/"
                f"{c['decoys_planted']} cost=${doc.cost_usd:.4f} run=${session.budget.spent_usd:.4f}"
            )

    report = build_report(
        finished, session, this_run=this_run, resumed=resumed, stop_reason=stop_reason
    )
    selected = [finished[i] for i in [*resumed, *this_run] if i in finished]
    _atomic_write(
        out_dir / "labels.jsonl",
        "".join(label.model_dump_json() + "\n" for d in selected for label in d.labels),
    )
    _atomic_write(
        out_dir / "decoys.jsonl",
        "".join(span.model_dump_json() + "\n" for d in selected for span in d.decoy_spans),
    )
    _atomic_write(out_dir / "run_report.json", json.dumps(report, indent=2))
    return report


def build_report(
    finished: dict[str, WrittenDoc],
    session: WriterSession,
    *,
    this_run: list[str],
    resumed: list[str],
    stop_reason: str | None,
) -> dict[str, Any]:
    docs = [finished[i] for i in [*resumed, *this_run] if i in finished]
    totals: Counter = Counter()
    first_checks: Counter = Counter()
    final_checks: Counter = Counter()
    for d in docs:
        totals.update(d.counts())
        for o in d.slot_outcomes:
            first_checks.update({f.split(":", 1)[0] for f in o.first_failures})
            final_checks.update({f.split(":", 1)[0] for f in o.final_failures})
    cost = sum(d.cost_usd for d in docs)
    written = [d for d in docs if d.status == "written"]
    planted = totals["facts_planted"] or 1
    return {
        "generated_at": session.now().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "checks_version": CHECKS_VERSION,
        "corpus_key": session.budget.corpus_key,
        "stop_reason": stop_reason,
        "documents": {
            "this_run": len(this_run),
            "resumed": len(resumed),
            "by_status": dict(Counter(d.status for d in docs)),
        },
        "cost_usd": round(cost, 6),
        "run_spend_usd": round(session.budget.spent_usd, 6),
        "calls": sum(len(d.calls) for d in docs),
        "cost_per_document_usd": round(cost / len(docs), 6) if docs else None,
        "cost_per_labeled_fact_usd": round(cost / totals["facts_labeled"], 6)
        if totals["facts_labeled"]
        else None,
        "totals": dict(totals),
        "pass_rates": {
            "facts_first_pass": round(totals["facts_verified_first_pass"] / planted, 4),
            "facts_final": round(totals["facts_verified"] / planted, 4),
            "facts_labeled": round(totals["facts_labeled"] / planted, 4),
            "decoys_final": round(totals["decoys_verified"] / (totals["decoys_planted"] or 1), 4),
        },
        "failures_by_check_first_attempt": dict(sorted(first_checks.items())),
        "failures_by_check_final": dict(sorted(final_checks.items())),
        "writer_docs": dict(Counter(d.writer_model for d in written)),
        # The pool caps each writer at 25% of the world's planned documents;
        # dropped documents make the share of what ships higher than that.
        "writer_max_share_of_written": round(
            max(Counter(d.writer_model for d in written).values(), default=0) / (len(written) or 1),
            4,
        ),
        "writer_cap_documents": session.pool.cap,
        "per_document": [
            {
                "plan_id": d.plan_id,
                "kind": d.kind.value,
                "status": d.status,
                "writer_model": d.writer_model,
                "final_call_kind": d.final_call_kind,
                "calls": len(d.calls),
                "cost_usd": round(d.cost_usd, 6),
                "words": d.word_count,
                **d.counts(),
            }
            for d in docs
        ],
    }


# --------------------------------------------------------------------------
# CLI: the WP4 live smoke (`python -m corpus.generator.v2.writer smoke`)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
# One document per surface kind that carries facts, and one decoys-only
# document (a hard negative renders as an ordinary kind).
_SMOKE_GROUPS: tuple[tuple[tuple[DocKind, ...], DocRole], ...] = (
    ((DocKind.CALL,), DocRole.EVENT),
    ((DocKind.EMAIL,), DocRole.EVENT),
    ((DocKind.TICKET,), DocRole.EVENT),
    ((DocKind.CRM_NOTE,), DocRole.EVENT),
    ((DocKind.SLACK, DocKind.TEAMS), DocRole.EVENT),
    ((DocKind.MEETING_NOTES,), DocRole.EVENT),
    ((DocKind.DOC,), DocRole.EVENT),
    (tuple(DocKind), DocRole.HARD_NEGATIVE),
)


def pick_smoke_docs(world_plan: WorldDocPlan, groups: int = len(_SMOKE_GROUPS)) -> list[str]:
    """One document per smoke group, preferring one that carries a decoy
    and more than one fact, earliest first."""
    picked: list[str] = []
    for kinds, role in _SMOKE_GROUPS[:groups]:
        candidates = sorted(
            (
                d
                for d in world_plan.doc_plans
                if d.kind in kinds and d.role == role and (d.fact_ids or d.decoy_ids)
            ),
            key=lambda d: (-(bool(d.decoy_ids)), -min(len(d.fact_ids), 3), d.date, d.id),
        )
        if candidates:
            picked.append(candidates[0].id)
    return picked


def main(argv: list[str] | None = None) -> int:
    from corpus.generator.v2.models import Split
    from corpus.generator.v2.packs.schema import load_pack
    from corpus.generator.v2.planner import plan_world
    from corpus.generator.v2.simulator import simulate_world

    parser = argparse.ArgumentParser(prog="python -m corpus.generator.v2.writer")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="fill a few documents of one world with real calls")
    smoke.add_argument("--pack", default="medical_devices")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--months", type=int, default=8)
    smoke.add_argument("--split", default="train")
    smoke.add_argument(
        "--doc-ids", default="", help="comma-separated plan ids (default: auto pick 5)"
    )
    smoke.add_argument("--budget-usd", type=float, default=1.00)
    smoke.add_argument(
        "--out", type=Path, default=REPO_ROOT / "var" / "data" / "synth_v2" / "smoke"
    )
    smoke.add_argument(
        "--spend-path",
        type=Path,
        default=default_spend_path(REPO_ROOT / "var"),
        help="SpendMeter file",
    )
    smoke.add_argument("--models", default=",".join(DEFAULT_ROTATION))
    args = parser.parse_args(argv)

    pack = load_pack(REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / f"{args.pack}.yaml")
    world = simulate_world(pack, seed=args.seed, months=args.months, split=Split(args.split))
    world_plan = plan_world(world, pack)
    doc_ids = [d for d in args.doc_ids.split(",") if d] or pick_smoke_docs(world_plan)
    config = WriterConfig(models=tuple(m for m in args.models.split(",") if m))
    pool = WriterPool(config.models, world_doc_total=len(writer_doc_plans(world_plan, config)))
    budget = RunBudget(cap_usd=args.budget_usd, meter=SpendMeter(args.spend_path))
    session = WriterSession(
        backend=RoutingWriterBackend(LocalWriterBackend(meter=SpendMeter(args.spend_path))),
        pool=pool,
        budget=budget,
        config=config,
        call_log=args.out / "calls.jsonl",
    )
    print(f"world {world.id}: writing {len(doc_ids)} documents: {doc_ids}", flush=True)
    report = run_world(
        world,
        world_plan,
        pack,
        session=session,
        out_dir=args.out,
        doc_ids=doc_ids,
        progress=lambda line: print(line, flush=True),
    )
    print(
        json.dumps(
            {k: report[k] for k in ("stop_reason", "cost_usd", "pass_rates", "totals")}, indent=2
        )
    )
    return 4 if report["stop_reason"] else 0


if __name__ == "__main__":
    sys.exit(main())
