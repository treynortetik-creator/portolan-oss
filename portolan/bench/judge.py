"""Layered evidence judge -- the disputed middle the deterministic
classifier in `portolan.bench.extraction` cannot decide.

`01-spec/extraction-accuracy-campaign-2026-09-18.md` wave 3A, owner
decision 2026-09-18 08:27: deterministic tiers (`extraction._evidence_
basis`) decide the clear cases; a model judge decides ONLY the disputed
middle -- a produced fact the deterministic classifier calls "invented"
because its evidence is genuinely spread across the source too widely for
a fragment/window check to certify (`var/bench/2026-09-18-sanity/
opus-invented-audit.md`'s `S-call-n03/f1`, "stretched/busy winter", split
across two turns ~70 sentences apart) -- and must return the supporting
sentence(s) so its verdict is string-checkable, never just trusted. Every
verdict is cached by `(source content hash, normalised claim)` in
`corpus/verdicts.yaml` (committed, human-readable) so a re-run is
deterministic and free; humans calibrate on a sample and the labels
accumulate into a golden set that overrides the judge. The same three
layers (deterministic / cached judge / live judge) grade near-miss
matching on the recall side.

Deliberately does NOT touch `extraction._evidence_basis` itself -- that
function stays the pure, already-tested deterministic classifier every
existing test calls directly. This module is a POST-PROCESSING pass over
an already-scored `ExtractionScore`: `apply_evidence_judge` looks at every
`UnplantedFact` the deterministic pass already called "none" (invented)
and, per `--judge` mode, resolves some of them to `judge_supported` /
`judge_unsupported` / `judge_unverifiable` (`UnplantedFact.evidence_basis`,
widened by these three terminal values -- see `extraction.
EVIDENCE_BASIS_VALUES`). `apply_match_judge` does the parallel thing for
`--judge-matches`: every planted fact the STRICT matcher missed but the
LENIENT matcher credited (`ExtractionScore.near_miss_credited_fact_ids`)
gets one judge call asking whether the closest produced candidate
(`UnmatchedFact.best_candidate`) states the same fact.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import yaml

from portolan.bench.corpus_loader import DEFAULT_CORPUS_ROOT, load_prepared_source
from portolan.bench.extraction import (
    _EMPTY_EVIDENCE_CONTEXT,
    _SENTENCE_SPLIT_RE,
    EvidenceContext,
    ExtractionScore,
    UnplantedFact,
    _claim_content_words,
    _normalize_claim_tokens,
    _salient_claim_fragments,
    evidence_context_for_source,
)
from portolan.bench.truth import Manifest, Truth
from portolan.llm.agent_sdk import AgentSDKClient
from portolan.llm.base import LLMError

JudgeMode = Literal["off", "cached", "live"]

DEFAULT_VERDICTS_PATH = Path("corpus/verdicts.yaml")
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

# How much of a claim's own content words the sentences the judge cites
# must together contain for a `supported` verdict to be accepted outright
# -- the "string-checkable" requirement. Same threshold FAMILY as
# extraction.py's own fragment/window votes (0.6): the judge's citation is
# held to the same bar the deterministic window fallback already enforces
# on itself, not a looser one.
SENTENCE_CHECK_THRESHOLD = 0.6

# A source text longer than this gets windowed down to the sentences most
# likely to hold the claim's evidence rather than sent whole. Generous for
# this corpus on purpose -- only a handful of call transcripts exceed
# 20,000 characters and even the largest (`S-call-015`, ~75,000 chars) is
# a few thousand tokens, nowhere near Sonnet's real context budget -- but
# a real threshold exists so a pathological source can't blow the prompt.
MAX_SOURCE_CHARS = 100_000


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _normalized_claim(claim: str) -> str:
    return " ".join(_normalize_claim_tokens(claim))


def evidence_cache_key(source_text: str, claim: str) -> str:
    """`sha256(source_text)[:16] + ":" + normalised_claim_hash` -- the
    exact key shape the build instructions specify. Two claims that
    normalise to the same token sequence (e.g. differing only in
    whitespace or number-word spelling) share a cache entry, which is the
    point: the judge's answer to "does this source support this claim"
    does not depend on incidental phrasing."""
    return f"{_sha16(source_text)}:{_sha16(_normalized_claim(claim))}"


def match_cache_key(source_text: str, claim_a: str, claim_b: str) -> str:
    """The match-judge's cache key, same shape as `evidence_cache_key`:
    the source hash plus a hash of BOTH normalised claims (planted,
    produced), order-preserving -- "does X state the same fact as Y" is
    not symmetric in how it's asked, so the two orderings are kept as
    distinct cache entries rather than sorted."""
    combined = f"{_normalized_claim(claim_a)}||{_normalized_claim(claim_b)}"
    return f"{_sha16(source_text)}:{_sha16(combined)}"


def split_sentences(text: str) -> list[str]:
    """Same sentence split `extraction._window_supported` uses, exposed
    here so a judge verdict's `sentence_indices` and the deterministic
    window fallback's own numbering agree."""
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _today() -> str:
    return _dt.date.today().isoformat()


# --------------------------------------------------------------------------
# corpus/verdicts.yaml -- the cache + golden set
# --------------------------------------------------------------------------


class VerdictStore:
    """`corpus/verdicts.yaml`: committed, human-readable. Each entry:
    `kind` ("evidence" or "match"), `source_id`, `claim` (the claim text,
    or "planted || produced" for a match entry, kept for a human reading
    the file -- never used to recompute the key), `verdict`
    ("supported"/"unsupported"/"unverifiable" for evidence,
    "same"/"not_same" for match), `sentence_indices` (evidence only),
    `reason`, `judge_model`, `date`, and `human` (`null` / `"supported"` /
    `"fabricated"` -- a hand label that OVERRIDES the judge's own verdict,
    per `effective_verdict`, and is never clobbered by a fresh model call,
    per `put`)."""

    def __init__(self, path: Path | str = DEFAULT_VERDICTS_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self._data, f, sort_keys=True, default_flow_style=False, allow_unicode=True
                )

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get(key)
            return dict(entry) if entry is not None else None

    def put(self, key: str, entry: dict[str, Any]) -> None:
        """Store `entry` under `key`. If `entry` itself doesn't specify a
        hand label (`human` is `None` -- true of every entry a judge re-
        run produces, since a model call never sets `human`) AND an
        existing entry there already has one, that hand label is carried
        forward rather than dropped: a re-run with `--judge live` (a model
        re-answering the same question) must never silently erase a
        human's calibration read. An `entry` that DOES specify a `human`
        value (a deliberate hand-label write, `human: supported` /
        `"fabricated"`) is stored as given -- including changing or
        clearing a prior hand label -- since that's an explicit human
        decision, not a model call that should be kept ignorant of it."""
        with self._lock:
            existing = self._data.get(key)
            if (
                entry.get("human") is None
                and existing is not None
                and existing.get("human") is not None
            ):
                entry = dict(entry)
                entry["human"] = existing["human"]
            self._data[key] = entry

    def effective_verdict(self, key: str) -> dict[str, Any] | None:
        """The stored entry, with `human` applied as an override on top
        of the judge's own `verdict` when a hand label is present --
        `human overrides the judge`, per the build instructions."""
        entry = self.get(key)
        if entry is None or entry.get("human") is None:
            return entry
        overridden = dict(entry)
        if entry.get("kind") == "match":
            overridden["verdict"] = "same" if entry["human"] == "supported" else "not_same"
        else:
            overridden["verdict"] = "supported" if entry["human"] == "supported" else "unsupported"
        return overridden

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------
# windowing -- select which sentences of a (possibly long) source to show
# the judge, preserving each sentence's ORIGINAL index so a verdict's
# `sentence_indices` stay checkable against the full sentence list.
# --------------------------------------------------------------------------


def select_sentence_window(
    sentences: list[str], claim: str, max_chars: int = MAX_SOURCE_CHARS
) -> list[int]:
    """Indices into `sentences` to show the judge. Every sentence, in
    order, when the source fits under `max_chars` whole (true for all but
    a handful of this corpus's longest call transcripts, see
    `MAX_SOURCE_CHARS`'s docstring) -- otherwise the sentences most likely
    to hold the claim's own evidence: any sentence containing one of the
    claim's salient fragments (numbers, proper nouns -- `_salient_claim_
    fragments`) or content words (`_claim_content_words`) gets priority,
    each expanded one sentence on either side for surrounding context,
    truncated to fit the budget -- a DIRECT hit is never bumped by a
    lower-index filler sentence that happens to come first; only once
    every direct hit that fits is in does a neighbor get added. Falls
    back to the leading sentences that fit if nothing in the claim
    matches anything (should not happen for a real claim; only exercised
    by a constructed test). Returned in original sentence order."""
    if sum(len(s) for s in sentences) <= max_chars:
        return list(range(len(sentences)))

    fragments = {f.lower() for f in _salient_claim_fragments(claim)}
    words = _claim_content_words(claim)
    direct_hits = [
        i
        for i, s in enumerate(sentences)
        if any(f in s.lower() for f in fragments) or any(w in s.lower() for w in words)
    ]
    if not direct_hits:
        selected: list[int] = []
        used = 0
        for i, s in enumerate(sentences):
            if used + len(s) > max_chars:
                break
            selected.append(i)
            used += len(s)
        return selected

    # Priority order: every direct hit first (by index), THEN their
    # immediate neighbors -- so a budget too tight for everything drops
    # neighbors before it drops a sentence that actually matched.
    priority: list[int] = list(direct_hits)
    for i in direct_hits:
        priority.append(max(0, i - 1))
        priority.append(min(len(sentences) - 1, i + 1))

    selected_set: set[int] = set()
    used = 0
    seen: set[int] = set()
    for i in priority:
        if i in seen:
            continue
        seen.add(i)
        if used + len(sentences[i]) > max_chars:
            continue  # doesn't fit -- skip it, keep trying the rest (never abandon the scan)
        selected_set.add(i)
        used += len(sentences[i])
    return sorted(selected_set)


# --------------------------------------------------------------------------
# prompts + JSON schemas
# --------------------------------------------------------------------------

EVIDENCE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "sentence_indices": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["supported", "sentence_indices", "reason"],
    "additionalProperties": False,
}

MATCH_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["same", "reason"],
    "additionalProperties": False,
}

EVIDENCE_SYSTEM = (
    "You are an evidence judge for an extraction accuracy benchmark. You will be given a "
    "CLAIM an extractor produced (optionally a QUOTE it cited), the CONTEXT the extractor "
    "was given up front (account/contact/supplier/product names -- citing one of these "
    "alone is not evidence for the rest of the claim, only for that one name), and the "
    "SOURCE as numbered sentences (numbering may skip -- the source may be windowed). "
    "Decide whether the source genuinely supports the claim, even if the supporting words "
    "are spread across more than one sentence, reworded, or given as numerals vs. spelled "
    "out. Cite ONLY sentence numbers that actually appear in the SOURCE list below and that "
    "actually contain words supporting the claim -- do not guess a number that isn't shown. "
    "Reply with JSON only: "
    '{"supported": true|false, "sentence_indices": [...], "reason": "<one line>"}. '
    "If unsupported, set sentence_indices to []."
)

MATCH_SYSTEM = (
    "You are grading a near-miss in an extraction accuracy benchmark. You will be given a "
    "PLANTED fact (the ground truth) and a PRODUCED fact an extractor actually wrote from "
    "the same source. Decide whether the produced fact states the SAME underlying fact as "
    "the planted one -- same claim, same entities and numbers, allowing for paraphrase, "
    "reordering, or a different fact-type label. Reply with JSON only: "
    '{"same": true|false, "reason": "<one line>"}.'
)


def _format_context(context: EvidenceContext) -> str:
    names = ", ".join(sorted(context.context_strings)) or "(none)"
    dates = ", ".join(sorted(context.date_tokens)) or "(none)"
    return (
        f"CONTEXT names/aliases given up front: {names}\n"
        f"CONTEXT source-envelope date parts: {dates}"
    )


def build_evidence_prompt(
    claim: str,
    quote: str | None,
    numbered_sentences: list[tuple[int, str]],
    context: EvidenceContext,
) -> str:
    lines = [f"CLAIM: {claim}"]
    if quote and quote.strip():
        lines.append(f"QUOTE the extractor cited: {quote}")
    lines.append("")
    lines.append(_format_context(context))
    lines.append("")
    lines.append("SOURCE (numbered sentences):")
    for i, s in numbered_sentences:
        lines.append(f"[{i}] {s}")
    return "\n".join(lines)


def build_match_prompt(planted_claim: str, produced_claim: str) -> str:
    return f"PLANTED: {planted_claim}\nPRODUCED: {produced_claim}"


# --------------------------------------------------------------------------
# verdict resolution -- pure functions, no model call, fully unit-testable
# --------------------------------------------------------------------------


def sentence_check(
    claim: str,
    sentences: list[str],
    indices: list[int],
    context: EvidenceContext = _EMPTY_EVIDENCE_CONTEXT,
) -> bool:
    """`True` iff at least one cited index is real AND the cited
    sentences, joined, contain at least `SENTENCE_CHECK_THRESHOLD` of the
    claim's own content words -- the string check a `supported` verdict
    must pass before it's trusted (build instructions: "a true verdict is
    accepted only if the cited sentences exist and together contain at
    least 60 percent of the claim's content words").

    Calibration finding (2026-09-18, reading the first live run's
    "unverifiable" bucket by hand): a claim's own account/supplier/
    product name is exactly the kind of word the deterministic classifier
    ALREADY treats as grounded via `EvidenceContext` alone (`extraction.
    _fragment_grounding`'s `_CONTEXT_GROUNDED` signal) because a call
    transcript routinely never says the account's own name aloud -- the
    same root cause `01-spec/extraction-miss-analysis-2026-09-18.md` and
    the sanity audit both already found. A content-word count that treats
    "Cobalt Ridge Medical Center" as four required words the cited
    sentences must contain re-creates that exact false-positive one layer
    up: the judge correctly cited the sentences that support everything
    ELSE in the claim, and got downgraded anyway because the name (never
    spoken, by design) dragged coverage under 60%. Claim words that also
    appear in `context.context_strings` are excluded from the required
    set for exactly this reason -- UNLESS excluding them would empty the
    set entirely, in which case they're put back: a claim that is
    NOTHING BUT a context name (no other content at all) is precisely the
    vacuous case the deterministic layer's own follow-up guard exists to
    reject, and must still fail here rather than pass on zero real
    checking."""
    valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(sentences)]
    if not valid:
        return False
    claim_words = _claim_content_words(claim)
    if not claim_words:
        return False
    context_words: set[str] = set()
    for s in context.context_strings:
        context_words.update(_claim_content_words(s))
    required = claim_words - context_words
    if not required:
        required = claim_words
    cited_words = _claim_content_words(" ".join(sentences[i] for i in valid))
    hits = sum(1 for w in required if w in cited_words)
    return (hits / len(required)) >= SENTENCE_CHECK_THRESHOLD


def resolve_evidence_verdict(
    claim: str,
    sentences: list[str],
    raw: dict[str, Any],
    context: EvidenceContext = _EMPTY_EVIDENCE_CONTEXT,
) -> dict[str, Any]:
    """The judge's raw `{"supported", "sentence_indices", "reason"}` ->
    `{"verdict": "supported"|"unsupported"|"unverifiable", "sentence_indices", "reason"}`.
    A `supported` raw verdict is downgraded to `unverifiable` when
    `sentence_check` fails -- the cited sentences don't exist, or don't
    actually cover the claim."""
    indices = [i for i in (raw.get("sentence_indices") or []) if isinstance(i, int)]
    reason = str(raw.get("reason") or "")
    if not raw.get("supported"):
        return {"verdict": "unsupported", "sentence_indices": indices, "reason": reason}
    if sentence_check(claim, sentences, indices, context):
        return {"verdict": "supported", "sentence_indices": indices, "reason": reason}
    return {
        "verdict": "unverifiable",
        "sentence_indices": indices,
        "reason": reason or "judge said supported but its cited sentences do not cover the claim",
    }


def resolve_match_verdict(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "same" if raw.get("same") else "not_same",
        "reason": str(raw.get("reason") or ""),
    }


# --------------------------------------------------------------------------
# Judge -- the model call
# --------------------------------------------------------------------------


class Judge:
    """Wraps `AgentSDKClient` (the local Claude Code CLI login, subscription,
    no per-call API charge -- `portolan/llm/agent_sdk.py`) at a fixed
    model, used ONLY for the disputed-middle calls this module makes.
    `AgentSDKClient.complete(json_schema=...)` already retries once on
    invalid JSON on its own (`portolan.llm.agent_sdk`'s own retry path) --
    this class does not add a second retry loop. `AgentSDKClient` has no
    `temperature` parameter at all (not exposed by `ClaudeAgentOptions`,
    per that module's own docstring) so none is set here either; every
    call runs at whatever the CLI's own default is."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        log_path: Path | str | None = None,
    ) -> None:
        self.model = model
        self.log_path = Path(log_path) if log_path else None
        self.calls_made = 0
        self._lock = threading.Lock()

    def _log(self, *, kind: str, source_id: str, result: Any | None, error: str | None) -> None:
        record = {
            "kind": kind,
            "source_id": source_id,
            "model": self.model,
            "input_tokens": getattr(result, "input_tokens", 0) if result else 0,
            "output_tokens": getattr(result, "output_tokens", 0) if result else 0,
            "cached_tokens": getattr(result, "cached_tokens", 0) if result else 0,
            "error": error,
            "date": _today(),
        }
        print(f"judge call: {record}")
        if self.log_path is None:
            return
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def _call(
        self,
        *,
        kind: str,
        source_id: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any] | None:
        client = AgentSDKClient(default_model=self.model)
        try:
            result = client.complete(
                system=system,
                user=user,
                json_schema=schema,
                model=self.model,
                surface="audit",
                max_tokens=max_tokens,
            )
            error = None
            data = result.data
        except LLMError as exc:  # invalid JSON after the client's own retry, or a call failure
            result = None
            error = str(exc)
            data = None
        with self._lock:
            self.calls_made += 1
        self._log(kind=kind, source_id=source_id, result=result, error=error)
        return data

    def judge_evidence_raw(
        self,
        *,
        source_id: str,
        claim: str,
        quote: str | None,
        numbered_sentences: list[tuple[int, str]],
        context: EvidenceContext,
    ) -> dict[str, Any]:
        user = build_evidence_prompt(claim, quote, numbered_sentences, context)
        data = self._call(
            kind="evidence",
            source_id=source_id,
            system=EVIDENCE_SYSTEM,
            user=user,
            schema=EVIDENCE_JSON_SCHEMA,
            max_tokens=600,
        )
        return (
            data
            if data is not None
            else {"supported": False, "sentence_indices": [], "reason": "judge_error"}
        )

    def judge_match_raw(
        self, *, source_id: str, planted_claim: str, produced_claim: str
    ) -> dict[str, Any]:
        user = build_match_prompt(planted_claim, produced_claim)
        data = self._call(
            kind="match",
            source_id=source_id,
            system=MATCH_SYSTEM,
            user=user,
            schema=MATCH_JSON_SCHEMA,
            max_tokens=200,
        )
        return data if data is not None else {"same": False, "reason": "judge_error"}


# --------------------------------------------------------------------------
# apply_evidence_judge / apply_match_judge -- the orchestration
# --------------------------------------------------------------------------


def _apply_evidence_verdict(u: UnplantedFact, verdict: str) -> UnplantedFact:
    if verdict == "supported":
        return u.model_copy(update={"evidence_basis": "judge_supported", "in_source_text": True})
    if verdict == "unverifiable":
        return u.model_copy(
            update={"evidence_basis": "judge_unverifiable", "is_unverifiable": True}
        )
    return u.model_copy(update={"evidence_basis": "judge_unsupported"})


def apply_evidence_judge(
    score: ExtractionScore,
    manifest: Manifest,
    truth: Truth,
    *,
    mode: JudgeMode = "cached",
    store: VerdictStore | None = None,
    judge: Judge | None = None,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    max_workers: int = 8,
) -> ExtractionScore:
    """Resolve the disputed middle of `score.unplanted` (every entry with
    `evidence_basis == "none"`, i.e. what the deterministic classifier
    alone calls "invented") via the cached/live judge tier.

    `mode="off"`: no cache read, no call -- `invented_deterministic` and
    `invented_adjudicated` are set equal (the judge tier never ran).
    `mode="cached"` (default): consult `store`, never call a model -- a
    cache miss stays "none" (deterministically invented) until a `live`
    run resolves it. `mode="live"`: call for cache misses (and cache the
    result), so a re-run afterwards is deterministic and free. Live calls
    for DIFFERENT facts are independent (a fresh `AgentSDKClient` per
    call, same "one client per call" rule `Judge._call` already follows)
    and are fanned out over `max_workers` threads -- calibrating a whole
    saved-run directory is dozens of calls, each a real CLI subprocess
    round trip, and this module must return within one foreground command
    (no polling, no backgrounding).
    """
    if mode == "off":
        deterministic = sum(
            1 for u in score.unplanted if not u.is_distractor_leak and u.evidence_basis == "none"
        )
        return score.model_copy(
            update={"invented_deterministic": deterministic, "invented_adjudicated": deterministic}
        )

    if store is None:
        store = VerdictStore()
    source_ids = {u.source_id for u in score.unplanted}
    texts: dict[str, str] = {}
    sentences_by_source: dict[str, list[str]] = {}
    context_by_source: dict[str, EvidenceContext] = {}
    for source_id in source_ids:
        try:
            manifest_source = manifest.source(source_id)
            text = load_prepared_source(manifest_source, corpus_root=corpus_root).text
        except (KeyError, FileNotFoundError, ValueError):
            text = ""
        texts[source_id] = text
        sentences_by_source[source_id] = split_sentences(text)
        context_by_source[source_id] = evidence_context_for_source(source_id, truth)

    def _quote_for(u: UnplantedFact) -> str | None:
        for rec in score.produced_by_source.get(u.source_id, []):
            if rec.id == u.produced_id:
                return rec.quote
        return None

    deterministic_invented = 0
    calls = 0
    cache_hits = 0

    # `updated[i]` is filled in immediately for a passthrough/cache-hit
    # entry; `None` is a placeholder for one that needs a live call,
    # resolved after the concurrent dispatch below.
    updated: list[UnplantedFact | None] = []
    pending: list[
        tuple[int, UnplantedFact, str, list[str]]
    ] = []  # (index, fact, cache_key, sentences)

    for u in score.unplanted:
        if u.is_distractor_leak or u.evidence_basis != "none":
            updated.append(u)
            continue
        deterministic_invented += 1

        text = texts.get(u.source_id, "")
        sentences = sentences_by_source.get(u.source_id, [])
        if not text:
            updated.append(u)
            continue

        key = evidence_cache_key(text, u.claim)
        cached = store.effective_verdict(key)
        if cached is not None:
            cache_hits += 1
            updated.append(_apply_evidence_verdict(u, cached["verdict"]))
            continue
        if mode != "live" or judge is None:
            updated.append(u)
            continue
        updated.append(None)
        pending.append((len(updated) - 1, u, key, sentences))

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    judge.judge_evidence_raw,
                    source_id=u.source_id,
                    claim=u.claim,
                    quote=_quote_for(u),
                    numbered_sentences=[
                        (i, sentences[i]) for i in select_sentence_window(sentences, u.claim)
                    ],
                    context=context_by_source.get(u.source_id, _EMPTY_EVIDENCE_CONTEXT),
                ): (idx, u, key, sentences)
                for idx, u, key, sentences in pending
            }
            for fut in as_completed(futures):
                idx, u, key, sentences = futures[fut]
                raw = fut.result()
                resolved = resolve_evidence_verdict(
                    u.claim,
                    sentences,
                    raw,
                    context_by_source.get(u.source_id, _EMPTY_EVIDENCE_CONTEXT),
                )
                entry = {
                    "kind": "evidence",
                    "source_id": u.source_id,
                    "claim": u.claim,
                    "verdict": resolved["verdict"],
                    "sentence_indices": resolved["sentence_indices"],
                    "reason": resolved["reason"],
                    "judge_model": judge.model,
                    "date": _today(),
                    "human": None,
                }
                store.put(key, entry)
                calls += 1
                cached = store.effective_verdict(key)
                updated[idx] = _apply_evidence_verdict(
                    u, cached["verdict"] if cached else "unsupported"
                )

    if calls:
        store.save()

    final: list[UnplantedFact] = [u for u in updated if u is not None]
    unverifiable = sum(1 for u in final if u.is_unverifiable)
    adjudicated_invented = sum(
        1
        for u in final
        if not u.is_distractor_leak and u.evidence_basis in ("none", "judge_unsupported")
    )

    return score.model_copy(
        update={
            "unplanted": final,
            "invented_deterministic": deterministic_invented,
            "invented_adjudicated": adjudicated_invented,
            "unverifiable_count": unverifiable,
            "judge_calls": calls,
            "judge_cache_hits": cache_hits,
        }
    )


def apply_match_judge(
    score: ExtractionScore,
    manifest: Manifest,
    truth: Truth,
    *,
    mode: JudgeMode = "off",
    store: VerdictStore | None = None,
    judge: Judge | None = None,
    corpus_root: Path | str = DEFAULT_CORPUS_ROOT,
    max_workers: int = 8,
) -> ExtractionScore:
    """For every planted fact_id in `score.near_miss_credited_fact_ids`
    (strict miss, lenient hit), ask the judge whether the closest produced
    candidate (`UnmatchedFact.best_candidate`, the same record the report
    already shows) states the same fact as the planted one. `adjudicated_
    recall` is STRICT `recall` plus one full credit per fact_id the judge
    (or a human override) confirms -- never a replacement for `recall` or
    `lenient_recall`, always reported beside both. Live calls are fanned
    out over `max_workers` threads, same reasoning as `apply_evidence_
    judge`."""
    if mode == "off" or not score.near_miss_credited_fact_ids:
        return score.model_copy(update={"adjudicated_recall": score.recall})

    if store is None:
        store = VerdictStore()
    unmatched_by_id = {u.fact_id: u for u in score.unmatched}
    source_texts: dict[str, str] = {}

    def _text_for(source_id: str) -> str:
        if source_id not in source_texts:
            try:
                manifest_source = manifest.source(source_id)
                source_texts[source_id] = load_prepared_source(
                    manifest_source, corpus_root=corpus_root
                ).text
            except (KeyError, FileNotFoundError, ValueError):
                source_texts[source_id] = ""
        return source_texts[source_id]

    calls = 0
    cache_hits = 0
    confirmed: dict[str, bool] = {}
    pending: list[tuple[str, Any, str]] = []  # (fact_id, unmatched_entry, cache_key)

    for fact_id in score.near_miss_credited_fact_ids:
        entry = unmatched_by_id.get(fact_id)
        if entry is None or entry.best_candidate is None or not entry.claim:
            continue
        text = _text_for(entry.source_id)
        key = match_cache_key(text, entry.claim, entry.best_candidate.claim)
        cached = store.effective_verdict(key)
        if cached is not None:
            cache_hits += 1
            confirmed[fact_id] = cached["verdict"] == "same"
            continue
        if mode != "live" or judge is None:
            continue
        pending.append((fact_id, entry, key))

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    judge.judge_match_raw,
                    source_id=entry.source_id,
                    planted_claim=entry.claim,
                    produced_claim=entry.best_candidate.claim,
                ): (fact_id, entry, key)
                for fact_id, entry, key in pending
            }
            for fut in as_completed(futures):
                fact_id, entry, key = futures[fut]
                raw = fut.result()
                resolved = resolve_match_verdict(raw)
                new_entry = {
                    "kind": "match",
                    "source_id": entry.source_id,
                    "claim": f"{entry.claim} || {entry.best_candidate.claim}",
                    "verdict": resolved["verdict"],
                    "sentence_indices": None,
                    "reason": resolved["reason"],
                    "judge_model": judge.model,
                    "date": _today(),
                    "human": None,
                }
                store.put(key, new_entry)
                calls += 1
                cached = store.effective_verdict(key)
                confirmed[fact_id] = bool(cached and cached["verdict"] == "same")

    if calls:
        store.save()

    confirmed_same = sorted(fid for fid, same in confirmed.items() if same)
    planted_total = score.planted_total or 0
    bonus = (len(confirmed_same) / planted_total) if planted_total else 0.0
    return score.model_copy(
        update={
            "adjudicated_recall": score.recall + bonus,
            "adjudicated_matched_fact_ids": confirmed_same,
            "judge_matches_calls": calls,
            "judge_matches_cache_hits": cache_hits,
        }
    )
