"""ERB answer evaluation on a sample, with Portolan's citation gate
(checklist item 28, first half: the ported prompts, the per-fact
completeness judge and the holistic correctness judge; the three-judge
gold correction is deliberately not built yet).

The judge prompts are ERB's own, verbatim, in `portolan.bench.erb_prompts`
(MIT, Copyright (c) 2026 DanswerAI, Inc.; licence reproduced there). The
parsing rules are ported from ERB too, each cited where it is used:

- correctness: `src/scripts/answer_evaluation/metrics_based_eval.py`
  `evaluate_answer_correctness` -- JSON with an `aligned` string; aligned
  iff it matches `\\byes\\b` case-insensitively.
- completeness: `src/utils/eval_utils.py` `validate_single_fact` -- each
  `answer_facts` entry judged in isolation; valid iff the first line of
  the reply matches `\\byes\\b`. Completeness = validated / facts x 100.
- context formatting: `src/utils/retrieval.py` `format_context_documents`.
- dsid stripping: `src/utils/eval_utils.py` `_strip_dsid_references`.

Two answer modes per question, over the same top-10 fused documents:

- `erb_plain` -- ERB's `ANSWER_GEN_PROMPT`, unchanged. This is the mode
  whose numbers line up with the paper's.
- `portolan_cited` -- the same documents, but every sentence must end
  with `[dsid_...: "verbatim quote"]`. Portolan's citation gate then drops
  any sentence whose quote does not pass `quote_is_supported` against the
  document it cites (or that cites a document not in context), exactly
  the chat path's "nothing is shown without a citation that resolves".
  Citations are then stripped deterministically (we control the format,
  so ERB's LLM stripping pass is not needed) and the judged text is what
  survives the gate.

The gate (`gate_cited_answer`) gives every segment a reason: kept, or
uncited, a document not in context, a quote not found, a number its quotes
do not state (the chat path's numbers guard, same function), or, with a
support judge, quotes that do not entail it. `run_cited` optionally sends
the dropped segments back once (`REPAIR_PROMPT`) for a cited restatement
or `NOT_IN_SOURCES`; every restatement goes through the same gate again.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from portolan.bench.erb_prompts import (
    ANSWER_GEN_PROMPT,
    ANSWER_WHOLISTIC_EVALUATION_PROMPT,
    INDIVIDUAL_FACT_VALIDATOR_PROMPT,
)
from portolan.extract.quote_gate import quote_is_supported
from portolan.report.numbers import unsupported_cited_numerals
from portolan.report.support_judge import ANSWER_SUPPORT_SYSTEM, SupportJudgeFn
from portolan.training.spend import SpendMeter

__all__ = [
    "ANSWER_MODES",
    "CITED_ANSWER_PROMPT",
    "CITED_ANSWER_PROMPT_V2",
    "DROP_REASONS",
    "NOT_IN_SOURCES",
    "REPAIR_PROMPT",
    "REPAIR_PROMPT_V2",
    "CitedRun",
    "CitedSentence",
    "GateResult",
    "OpenRouterChat",
    "estimate_evaluation_usd",
    "estimate_generation_usd",
    "evaluate_question",
    "evaluation_item",
    "format_context_documents",
    "gate_cited_answer",
    "generate_answers",
    "make_chat_support_judge",
    "parse_aligned",
    "parse_cited_answer",
    "parse_fact_valid",
    "parse_numbered",
    "parse_support_verdict",
    "run_cited",
    "sample_questions",
    "strip_dsid_references",
    "support_reference",
]

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_DSID_PATTERN = re.compile(r"dsid_[a-f0-9]{20,32}")  # ERB eval_utils._DSID_PATTERN

# Portolan's own prompt, not ERB's: ERB's ANSWER_GEN_PROMPT plus the
# citation contract the chat path enforces.
CITED_ANSWER_PROMPT = """
You are a helpful and precise assistant that generates answers based on the provided documents. The documents came from a retrieval system which is imperfect. \
Base your answer purely on the documents and do not make up any information. Many of the documents provided are likely to be irrelevant. \
Be concise and only provide information directly relevant to the query.

Every sentence of your answer must end with one or more citations of the form [DOCUMENT_ID: "exact quote"], where DOCUMENT_ID is the ID shown in the document header (it starts with dsid_) and the quote is copied character for character from that document and supports the sentence. A sentence without a supporting verbatim quote will be deleted.

## Context Documents
{context_documents}

## Question
{question}

## Answer
Output your answer below, do not include any additional text or formatting:
"""  # noqa: E501

NOT_IN_SOURCES = "NOT_IN_SOURCES"

# The v1 contract plus what the 2026-09-23 diagnosis found missing: a bare
# value with no sentence or citation, a cited answer that leaves out details
# the question asks for, and a free-text abstention the gate cannot keep.
CITED_ANSWER_PROMPT_V2 = """
You are a helpful and precise assistant that generates answers based on the provided documents. The documents came from a retrieval system which is imperfect. \
Base your answer purely on the documents and do not make up any information. Many of the documents provided are likely to be irrelevant. \
Be concise and only provide information directly relevant to the query.

Write complete sentences, never a bare value. Give every detail the documents state that the question asks for (names, dates, numbers, conditions, next steps). \
Every sentence of your answer must end with one or more citations of the form [DOCUMENT_ID: "exact quote"], where DOCUMENT_ID is the ID shown in the document header (it starts with dsid_) and the quote is copied character for character from that document and states what the sentence says. \
When a sentence uses values from more than one place, cite a quote for each value. A sentence without a supporting verbatim quote will be deleted.

If no document answers the question, output exactly NOT_IN_SOURCES and nothing else.

## Context Documents
{context_documents}

## Question
{question}

## Answer
Output your answer below, do not include any additional text or formatting:
"""  # noqa: E501

REPAIR_PROMPT = """
You answered the question below from the context documents, and a check removed some of your sentences because it could not tie them to a document. \
For each numbered removed sentence, either rewrite it so that it ends with one or more citations of the form [DOCUMENT_ID: "exact quote"] (DOCUMENT_ID as shown in the document header, the quote copied character for character from that document and stating what the rewritten sentence says), or, if no document states it, write exactly NOT_IN_SOURCES. \
You may leave out any part no quote states. Do not add anything the documents do not say, and do not repeat the kept sentences.

## Context Documents
{context_documents}

## Question
{question}

## Kept sentences
{kept}

## Removed sentences
{removed}

## Output
One line per removed sentence, numbered as above, and nothing else:
"""  # noqa: E501

# Repair v1's replies abstained whenever part of a sentence was uncitable, and
# one cited "DOCUMENT 6" instead of the ID (01-spec/bench-cited-answer-repair-2026-09-23.md).
REPAIR_PROMPT_V2 = """
You answered the question below from the context documents, and a check removed some of your sentences because it could not tie them to a document. \
For each numbered removed sentence, rewrite it so that every part of it is stated by a quote, and end it with one or more citations of the form [DOCUMENT_ID: "exact quote"]. DOCUMENT_ID is the ID in the document header that starts with dsid_, never the document number; the quote is copied character for character from that document. \
If a quote states only part of the sentence, keep only that part: give the values the quotes state rather than a value worked out from them (a count of days, a date, a total). \
Write exactly NOT_IN_SOURCES only when no quote states any part of it. Do not add anything the documents do not say, and do not repeat the kept sentences.

## Context Documents
{context_documents}

## Question
{question}

## Kept sentences
{kept}

## Removed sentences
{removed}

## Output
One line per removed sentence, numbered as above, and nothing else:
"""  # noqa: E501

# Why the gate dropped a segment, in the words the repair prompt uses.
DROP_REASONS: dict[str, str] = {
    "uncited": "it carried no citation",
    "doc_not_in_context": "it cited a document that is not among the context documents",
    "quote_not_found": "its quote is not copied exactly from the document it cites",
    "number_not_in_quote": "it states a number its quotes do not state",
    "not_supported": "its quotes do not state what it says",
}


def format_context_documents(docs: Sequence[tuple[str, str, str]]) -> str:
    """ERB `src/utils/retrieval.py` `format_context_documents`, over
    already-loaded (dsid, title, content) triples."""
    parts = []
    for i, (dsid, title, content) in enumerate(docs, 1):
        parts.append(f"--- Document {i} (ID: {dsid}) ---\nTitle: {title}\n\n{content}")
    return "\n\n".join(parts)


def strip_dsid_references(text: str) -> str:
    """ERB `_strip_dsid_references`: remove dsid ids, collapse double spaces."""
    cleaned = _DSID_PATTERN.sub("", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def parse_aligned(response: str) -> bool | None:
    """ERB `evaluate_answer_correctness`'s parse: a JSON object whose
    `aligned` string matches `\\byes\\b`; None when unparseable (ERB then
    retries, and scores a final failure as not correct)."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("aligned"), str):
        return None
    return re.search(r"\byes\b", parsed["aligned"], re.IGNORECASE) is not None


def parse_fact_valid(response: str) -> bool:
    """ERB `validate_single_fact`'s parse: `\\byes\\b` on the first line."""
    stripped = response.strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    return re.search(r"\byes\b", first_line, re.IGNORECASE) is not None


_CITATION_RE = re.compile(r"\[(dsid_[a-f0-9]{20,32})\s*:\s*[\"“](.+?)[\"”]\s*\]", re.DOTALL)


@dataclass
class CitedSentence:
    text: str
    citations: list[tuple[str, str]] = field(default_factory=list)


def parse_cited_answer(answer: str) -> list[CitedSentence]:
    """Split a `CITED_ANSWER_PROMPT` answer into sentences, each with the
    citations that follow it. A sentence is the prose between the end of
    one citation run and the next citation run; trailing prose with no
    citation becomes an uncited sentence (which the gate drops)."""
    out: list[CitedSentence] = []
    pos = 0
    current: CitedSentence | None = None
    for m in _CITATION_RE.finditer(answer):
        prose = answer[pos : m.start()].strip().lstrip(".,;: ").strip()
        if prose or current is None:
            current = CitedSentence(text=prose)
            out.append(current)
        current.citations.append((m.group(1), m.group(2)))
        pos = m.end()
    tail = answer[pos:].strip().lstrip(".,;: ").strip()
    if tail:
        out.append(CitedSentence(text=tail))
    return [s for s in out if s.text or s.citations]


def _joined_text(segments: Sequence[CitedSentence]) -> str:
    return " ".join(
        s.text if s.text.endswith((".", "!", "?")) else s.text + "." for s in segments
    ).strip()


def _is_abstention(s: CitedSentence) -> bool:
    return not s.citations and s.text.strip().rstrip(".").strip() == NOT_IN_SOURCES


@dataclass
class GateResult:
    """One verdict per segment, in answer order: `None` keeps it; anything
    else is a `DROP_REASONS` key, `"no_text"` (citations with no prose) or
    `"abstained"` (a bare `NOT_IN_SOURCES`, never shown)."""

    segments: list[CitedSentence]
    reasons: list[str | None]
    quotes_total: int = 0
    quotes_supported: int = 0
    # Verified quotes per segment: what the numbers guard and the support
    # judge read. Empty for a segment no quote of which passed.
    evidence: list[list[tuple[str, str]]] = field(default_factory=list)
    support_verdicts: list[str | None] = field(default_factory=list)

    @property
    def kept(self) -> list[CitedSentence]:
        return [s for s, r in zip(self.segments, self.reasons, strict=True) if r is None]

    @property
    def dropped(self) -> list[CitedSentence]:
        return [
            s
            for s, r in zip(self.segments, self.reasons, strict=True)
            if r is not None and r != "abstained"
        ]

    @property
    def abstained(self) -> int:
        return sum(1 for r in self.reasons if r == "abstained")

    @property
    def text(self) -> str:
        """The kept sentences, citations stripped, each ending in a stop."""
        return _joined_text(self.kept)


def _locate(quote: str, text: str) -> tuple[int, int] | None:
    """Where `quote` sits in `text`: exact, else whitespace- and case-folded
    (mapped back to `text`'s offsets); None when neither finds it."""
    at = text.find(quote)
    if at >= 0:
        return at, at + len(quote)
    folded: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            if folded and folded[-1] == " ":
                continue
            ch = " "
        folded.append(ch.lower())
        offsets.append(i)
    needle = " ".join(quote.split()).lower()
    at = "".join(folded).find(needle) if needle else -1
    if at < 0:
        return None
    return offsets[at], offsets[at + len(needle) - 1] + 1


def support_reference(
    evidence: Sequence[tuple[str, str]],
    titles: dict[str, str] | None,
    *,
    doc_texts: dict[str, str] | None = None,
    context_chars: int = 0,
) -> str:
    """The support judge's reference text: each verified quote with the title
    of the document it came from, so "the customer" can be checked against
    whose document it is. With `context_chars`, each quote is shown marked
    ⟦...⟧ inside that many characters of its passage on either side."""
    lines = []
    for dsid, quote in evidence:
        title = (titles or {}).get(dsid)
        shown = f'"{quote}"'
        text = (doc_texts or {}).get(dsid)
        span = _locate(quote, text) if context_chars and text else None
        if span is not None and text is not None:
            start, end = span
            before = " ".join(text[max(0, start - context_chars) : start].split())
            after = " ".join(text[end : end + context_chars].split())
            shown = f"...{before} ⟦{' '.join(text[start:end].split())}⟧ {after}..."
        lines.append(f'- from "{title}": {shown}' if title else f"- {shown}")
    return "\n".join(lines)


def gate_cited_answer(
    sentences: Sequence[CitedSentence],
    doc_texts: dict[str, str],
    *,
    numbers: bool = True,
    support: SupportJudgeFn | None = None,
    titles: dict[str, str] | None = None,
    support_context_chars: int = 0,
    support_workers: int = 4,
) -> GateResult:
    """Keep a segment only if at least one of its quotes passes
    `quote_is_supported` against a context document it cites; then, with
    `numbers`, only if every number it states is in those verified quotes
    (`unsupported_cited_numerals`, the chat path's guard); then, with a
    `support` judge, only if the judge says the verified quotes entail it
    (a judge error counts as not supported; `support_context_chars` shows
    the judge each quote inside its passage). Counts every quote."""
    reasons: list[str | None] = []
    evidence: list[list[tuple[str, str]]] = []
    total = supported = 0
    for s in sentences:
        verified: list[tuple[str, str]] = []
        in_context = False
        for dsid, quote in s.citations:
            total += 1
            text = doc_texts.get(dsid)
            in_context = in_context or text is not None
            if text is not None and quote_is_supported(quote, text):
                supported += 1
                verified.append((dsid, quote))
        evidence.append(verified)
        if _is_abstention(s):
            reasons.append("abstained")
        elif not s.text:
            reasons.append("no_text")
        elif not s.citations:
            reasons.append("uncited")
        elif not in_context:
            reasons.append("doc_not_in_context")
        elif not verified:
            reasons.append("quote_not_found")
        elif numbers and unsupported_cited_numerals(s.text, [q for _d, q in verified]):
            reasons.append("number_not_in_quote")
        else:
            reasons.append(None)

    verdicts: list[str | None] = [None] * len(reasons)
    if support is not None:
        todo = [i for i, r in enumerate(reasons) if r is None]

        def _judge(i: int) -> str | None:
            return support(
                sentence=sentences[i].text,
                reference_text=support_reference(
                    evidence[i],
                    titles,
                    doc_texts=doc_texts,
                    context_chars=support_context_chars,
                ),
            )

        with ThreadPoolExecutor(max_workers=max(1, support_workers)) as pool:
            for i, verdict in zip(todo, pool.map(_judge, todo), strict=True):
                verdicts[i] = verdict
                if verdict != "supported":
                    reasons[i] = "not_supported"
    return GateResult(
        segments=list(sentences),
        reasons=reasons,
        quotes_total=total,
        quotes_supported=supported,
        evidence=evidence,
        support_verdicts=verdicts,
    )


_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")


def parse_numbered(text: str) -> dict[int, str]:
    """`1. ...` / `2) ...` lines into {number: text}; a line that does not
    start a number continues the previous entry."""
    out: dict[int, str] = {}
    current: int | None = None
    for line in text.splitlines():
        m = _NUMBERED_RE.match(line)
        if m:
            current = int(m.group(1))
            out[current] = m.group(2).strip()
        elif current is not None and line.strip():
            out[current] = f"{out[current]} {line.strip()}".strip()
    return out


@dataclass
class CitedRun:
    """One cited answer through the gate and, optionally, one repair pass.
    `final` is in answer order: first-pass keeps in place, each dropped
    segment replaced by whatever of its restatement passed the same gate."""

    raw: str
    first: GateResult
    final: list[CitedSentence]
    repair_raw: str | None = None
    repair_attempted: int = 0  # dropped segments sent back
    repaired: int = 0  # of those, how many came back with a kept restatement
    repaired_segments: list[CitedSentence] = field(default_factory=list)
    repair_reasons: list[str | None] = field(default_factory=list)
    abstained: int = 0
    dropped_final: int = 0
    gen_s: float = 0.0
    gate_s: float = 0.0
    repair_s: float = 0.0
    usd: float = 0.0
    calls: int = 0

    @property
    def text(self) -> str:
        return _joined_text(self.final)

    @property
    def latency_s(self) -> float:
        return self.gen_s + self.gate_s + self.repair_s


def run_cited(
    chat: OpenRouterChat,
    gen_model: str,
    question: str,
    context: str,
    doc_texts: dict[str, str],
    *,
    prompt: str = CITED_ANSWER_PROMPT,
    raw: str | None = None,
    raw_gen_s: float = 0.0,
    repair: bool = False,
    repair_prompt: str = REPAIR_PROMPT,
    numbers: bool = True,
    support: SupportJudgeFn | None = None,
    support_context_chars: int = 0,
    titles: dict[str, str] | None = None,
    tally: dict[str, float] | None = None,
) -> CitedRun:
    """Generate (or reuse `raw`), gate, and with `repair` send the dropped
    segments back once. The repair call uses the generator and the same
    context; its restatements pass through the same gate (numbers guard and
    support judge included) before anything is kept."""
    tally = tally if tally is not None else {"usd": 0.0, "calls": 0}
    gen_s = raw_gen_s
    if raw is None:
        t0 = time.perf_counter()
        raw = chat.complete(
            gen_model,
            prompt.format(context_documents=context, question=question),
            max_tokens=1500,
            tally=tally,
        ).strip()
        gen_s = time.perf_counter() - t0

    def _gate(segments: Sequence[CitedSentence]) -> GateResult:
        return gate_cited_answer(
            segments,
            doc_texts,
            numbers=numbers,
            support=support,
            titles=titles,
            support_context_chars=support_context_chars,
        )

    t0 = time.perf_counter()
    first = _gate(parse_cited_answer(raw))
    gate_s = time.perf_counter() - t0

    to_repair = [i for i, r in enumerate(first.reasons) if r is not None and r in DROP_REASONS]
    replacements: dict[int, GateResult] = {}
    repair_raw = None
    repair_s = 0.0
    if repair and to_repair:
        t0 = time.perf_counter()
        kept_lines = "\n".join(f"- {s.text}" for s in first.kept) or "(none)"
        removed_lines = "\n".join(
            f"{n}. {first.segments[i].text or '(no text)'} -- removed because "
            f"{DROP_REASONS[first.reasons[i]]}"
            for n, i in enumerate(to_repair, 1)
        )
        repair_raw = chat.complete(
            gen_model,
            repair_prompt.format(
                context_documents=context,
                question=question,
                kept=kept_lines,
                removed=removed_lines,
            ),
            max_tokens=1500,
            tally=tally,
        ).strip()
        restated = parse_numbered(repair_raw)
        for n, i in enumerate(to_repair, 1):
            text = restated.get(n, "").strip()
            if text:
                replacements[i] = _gate(parse_cited_answer(text))
        repair_s = time.perf_counter() - t0

    final: list[CitedSentence] = []
    repaired_segments: list[CitedSentence] = []
    repair_reasons: list[str | None] = []
    repaired = dropped_final = 0
    abstained = first.abstained
    for i, (segment, reason) in enumerate(zip(first.segments, first.reasons, strict=True)):
        if reason is None:
            final.append(segment)
            continue
        if reason not in DROP_REASONS:
            continue
        again = replacements.get(i)
        if again is None:
            dropped_final += 1
            continue
        repair_reasons.extend(again.reasons)
        abstained += again.abstained
        if again.kept:
            repaired += 1
            final.extend(again.kept)
            repaired_segments.extend(again.kept)
        elif not again.abstained:
            dropped_final += 1
    return CitedRun(
        raw=raw,
        first=first,
        final=final,
        repair_raw=repair_raw,
        repair_attempted=len(to_repair) if repair else 0,
        repaired=repaired,
        repaired_segments=repaired_segments,
        repair_reasons=repair_reasons,
        abstained=abstained,
        dropped_final=dropped_final,
        gen_s=gen_s,
        gate_s=gate_s,
        repair_s=repair_s,
        usd=float(tally["usd"]),
        calls=int(tally["calls"]),
    )


_VERDICT_RE = re.compile(r"\{.*\}", re.DOTALL)
# A reason with an unescaped quote breaks the JSON; the verdict field alone
# is still unambiguous.
_VERDICT_FIELD_RE = re.compile(r'"verdict"\s*:\s*"(supported|not_supported)"')
_SUPPORT_MAX_TOKENS = 150  # a verdict and a short reason


def parse_support_verdict(raw: str) -> Literal["supported", "not_supported"] | None:
    """The judge's JSON `verdict`, else a bare `"verdict": "..."` field;
    None when neither is there."""
    value: Any = None
    m = _VERDICT_RE.search(raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            value = parsed.get("verdict") if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            value = None
    if value not in ("supported", "not_supported"):
        field_match = _VERDICT_FIELD_RE.search(raw)
        value = field_match.group(1) if field_match else None
    return value


def make_chat_support_judge(
    chat: OpenRouterChat,
    model: str,
    *,
    system: str = ANSWER_SUPPORT_SYSTEM,
    tally: dict[str, float] | None = None,
    cache: dict[tuple[str, str], str | None] | None = None,
    unparsed: list[str] | None = None,
) -> SupportJudgeFn:
    """`system` (default `ANSWER_SUPPORT_SYSTEM`) over `chat` (temperature
    0), as a `SupportJudgeFn`. An unparseable reply returns `None`, which the
    gate treats as not supported, and is appended to `unparsed`. `cache`
    shares verdicts for the same sentence and reference within one
    question."""

    def _judge(
        *, sentence: str, reference_text: str
    ) -> Literal["supported", "not_supported"] | None:
        key = (sentence, reference_text)
        if cache is not None and key in cache:
            return cache[key]  # type: ignore[return-value]
        user = f"Evidence this sentence cites:\n{reference_text}\n\nSentence to grade:\n{sentence}"
        raw = chat.complete(model, user, system=system, max_tokens=_SUPPORT_MAX_TOKENS, tally=tally)
        verdict = parse_support_verdict(raw)
        if verdict is None and unparsed is not None:
            unparsed.append(raw[:400])
        if cache is not None:
            cache[key] = verdict
        return verdict

    return _judge


def sample_questions(
    questions: Sequence[dict[str, Any]], n: int = 20, seed: int = 20260922
) -> list[dict[str, Any]]:
    """A fixed, seeded sample, returned in question-id order."""
    ordered = sorted(questions, key=lambda q: q["question_id"])
    picked = random.Random(seed).sample(ordered, min(n, len(ordered)))
    return sorted(picked, key=lambda q: q["question_id"])


class OpenRouterChat:
    """Minimal chat-completions caller for the sample: temperature 0,
    reasoning disabled (same as `portolan.llm.openrouter`), every
    response's `usage.cost` recorded in `meter` under `corpus`, and no
    request once that corpus's spend has reached `budget_usd` or this
    client's own spend has reached `run_cap_usd`."""

    def __init__(
        self,
        api_key: str,
        *,
        meter: SpendMeter | None,
        corpus: str,
        budget_usd: float,
        run_cap_usd: float | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 180.0,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.meter = meter
        self.corpus = corpus
        self.budget_usd = budget_usd
        self.run_cap_usd = run_cap_usd
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            transport=transport, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )
        self._lock = threading.Lock()
        self.calls = 0
        self.spent_usd = 0.0
        self.per_model_usd: dict[str, float] = {}

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 1200,
        system: str | None = None,
        tally: dict[str, float] | None = None,
    ) -> str:
        """`tally`, when given, gets this call's cost added to `"usd"` and one
        to `"calls"` (a per-answer cost next to the run's totals)."""
        from portolan.bench.erb_search import BudgetExceeded

        spent = self.meter.corpus_usd(self.corpus) if self.meter else self.spent_usd
        if spent >= self.budget_usd:
            raise BudgetExceeded(f"{self.corpus} spend ${spent:.4f} reached ${self.budget_usd}")
        if self.run_cap_usd is not None and self.spent_usd >= self.run_cap_usd:
            raise BudgetExceeded(
                f"this run's spend ${self.spent_usd:.4f} reached its cap ${self.run_cap_usd}"
            )
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning": {"enabled": False},
        }
        last = "no attempt"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(_OPENROUTER_CHAT_URL, json=body)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                if resp.status_code == 200:
                    payload = resp.json()
                    choices = payload.get("choices") or []
                    usage = payload.get("usage") or {}
                    cost = float(usage.get("cost") or 0.0)
                    with self._lock:
                        self.calls += 1
                        self.spent_usd += cost
                        self.per_model_usd[model] = self.per_model_usd.get(model, 0.0) + cost
                        if tally is not None:
                            tally["usd"] = tally.get("usd", 0.0) + cost
                            tally["calls"] = tally.get("calls", 0) + 1
                    if self.meter is not None:
                        self.meter.record(model=model, corpus=self.corpus, amount_usd=cost)
                    if choices:
                        return (choices[0].get("message") or {}).get("content") or ""
                    last = "no choices"
                elif resp.status_code == 429 or resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}"
                elif (
                    resp.status_code == 400
                    and "reasoning" in body
                    and "reasoning" in resp.text.lower()
                    and ("mandatory" in resp.text.lower() or "cannot be disabled" in resp.text)
                ):
                    # A model whose reasoning cannot be switched off (same
                    # rejection portolan.llm.openrouter learns from): resend
                    # without the field, immediately (no backoff).
                    body.pop("reasoning")
                    last = "reasoning mandatory; resent without the disable"
                    continue
                else:
                    raise RuntimeError(
                        f"OpenRouter chat HTTP {resp.status_code}: {resp.text[:300]}"
                    )
            if attempt < self.max_retries:
                self._sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"OpenRouter chat failed after retries: {last}")


def judge_correctness(chat: OpenRouterChat, model: str, question: str, gold: str, answer: str):
    """(aligned, raw) -- ERB retries an unparseable reply up to 3 times."""
    prompt = ANSWER_WHOLISTIC_EVALUATION_PROMPT.format(
        query=question, gold_answer=gold, candidate_answer=answer
    )
    raw = ""
    for _ in range(3):
        raw = chat.complete(model, prompt, max_tokens=300)
        verdict = parse_aligned(raw)
        if verdict is not None:
            return verdict, raw
    return False, raw


def judge_fact(chat: OpenRouterChat, model: str, answer: str, statement: str) -> bool:
    prompt = INDIVIDUAL_FACT_VALIDATOR_PROMPT.format(answer=answer, statement=statement)
    return parse_fact_valid(chat.complete(model, prompt, max_tokens=20))


def generate_answer(
    chat: OpenRouterChat, model: str, question: str, context: str, *, cited: bool
) -> str:
    template = CITED_ANSWER_PROMPT if cited else ANSWER_GEN_PROMPT
    prompt = template.format(context_documents=context, question=question)
    return chat.complete(model, prompt, max_tokens=1500).strip()


def score_answer_text(
    chat: OpenRouterChat, judge_model: str, question: dict[str, Any], answer: str
) -> dict[str, Any]:
    """ERB `score_answer`'s answer half: correctness and completeness,
    judged independently. An empty answer scores 0 / not correct, as ERB's
    `if answer_text:` guard leaves it."""
    facts = question.get("answer_facts") or []
    if not answer.strip():
        return {"answer_correct": False, "completeness_pct": 0.0, "facts_valid": []}
    correct, _raw = judge_correctness(
        chat, judge_model, question["question"], question.get("gold_answer", ""), answer
    )
    valid = [judge_fact(chat, judge_model, answer, f) for f in facts]
    completeness = (sum(valid) / len(facts) * 100) if facts else 100.0
    return {
        "answer_correct": bool(correct),
        "completeness_pct": round(completeness, 2),
        "facts_valid": valid,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """ERB `compute_stats_for_group`'s answer half."""
    n = len(rows)
    if not n:
        return {"count": 0}
    return {
        "count": n,
        "average_correctness_pct": round(sum(1 for r in rows if r["answer_correct"]) / n * 100, 2),
        "average_completeness_pct": round(sum(r["completeness_pct"] for r in rows) / n, 2),
        "combined_correctness_completeness_score": round(
            sum(r["completeness_pct"] if r["answer_correct"] else 0.0 for r in rows) / n, 2
        ),
    }


# ---------------------------------------------------------------------
# The 2026-09-23 repair measurement: every mode per question, one run.
# ---------------------------------------------------------------------

#: Answer modes, in report order. `cited_before` is the gate as the
#: 2026-09-22 sample ran it (plus the numbers guard); `cited_repair` is the
#: same raw answer with one repair pass; `cited_v2` the v2 prompt with the
#: repair pass; `cited_v2_support` the same v2 raw answer with the support
#: judge in the gate too. `recorded_*` replay the 2026-09-22 raw answers.
ANSWER_MODES = (
    "erb_plain",
    "cited_before",
    "cited_repair",
    "cited_v2",
    "cited_v2_support",
    "recorded_before",
    "recorded_repair",
)


def verified_evidence(s: CitedSentence, doc_texts: dict[str, str]) -> list[tuple[str, str]]:
    """The citations of `s` whose quote passes the quote gate."""
    return [
        (dsid, quote)
        for dsid, quote in s.citations
        if dsid in doc_texts and quote_is_supported(quote, doc_texts[dsid])
    ]


def generate_answers(
    chat: OpenRouterChat, gen_model: str, question: str, context: str
) -> dict[str, dict[str, Any]]:
    """The three raw answers the modes share: ERB's prompt (`plain`), the
    v1 citation contract (`v1`) and v2 (`v2`), each with its seconds and
    dollars."""
    out: dict[str, dict[str, Any]] = {}
    for key, template in (
        ("plain", ANSWER_GEN_PROMPT),
        ("v1", CITED_ANSWER_PROMPT),
        ("v2", CITED_ANSWER_PROMPT_V2),
    ):
        tally: dict[str, float] = {"usd": 0.0, "calls": 0}
        t0 = time.perf_counter()
        text = chat.complete(
            gen_model,
            template.format(context_documents=context, question=question),
            max_tokens=1500,
            tally=tally,
        ).strip()
        out[key] = {"text": text, "s": time.perf_counter() - t0, "usd": float(tally["usd"])}
    return out


def _run_summary(run: CitedRun, *, gen_usd: float) -> dict[str, Any]:
    first_reasons: dict[str, int] = {}
    for r in run.first.reasons:
        if r is not None:
            first_reasons[r] = first_reasons.get(r, 0) + 1
    return {
        "segments": len(run.first.segments),
        "first_pass_kept": len(run.first.kept),
        "first_pass_dropped": len(run.first.dropped),
        "first_pass_reasons": first_reasons,
        "quotes": run.first.quotes_total,
        "quotes_supported": run.first.quotes_supported,
        "repair_attempted": run.repair_attempted,
        "repaired": run.repaired,
        "repaired_segments": len(run.repaired_segments),
        "abstained": run.abstained,
        "dropped_final": run.dropped_final,
        "shipped": len(run.final),
        "latency_s": round(run.latency_s, 3),
        "usd": round(gen_usd + run.usd, 6),
        "calls": run.calls,
    }


def _support_diag(
    judge: SupportJudgeFn,
    segments: Sequence[CitedSentence],
    doc_texts: dict[str, str],
    titles: dict[str, str],
    context_chars: int = 0,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Measurement only: the support judge's verdict on each shipped
    segment. Returns the counts and the flagged segments (for local
    reading; they carry document text)."""
    counts = {"shipped": len(segments), "supported": 0, "not_supported": 0, "judge_error": 0}
    flagged: list[dict[str, Any]] = []
    for s in segments:
        evidence = verified_evidence(s, doc_texts)
        reference = support_reference(
            evidence, titles, doc_texts=doc_texts, context_chars=context_chars
        )
        verdict = judge(sentence=s.text, reference_text=reference)
        if verdict == "supported":
            counts["supported"] += 1
            continue
        counts["not_supported" if verdict == "not_supported" else "judge_error"] += 1
        flagged.append({"sentence": s.text, "evidence": evidence, "verdict": verdict})
    return counts, flagged


def evaluate_question(
    chat: OpenRouterChat,
    *,
    gen_model: str,
    judge_model: str,
    question: dict[str, Any],
    docs: Sequence[tuple[str, str, str]],
    gens: dict[str, dict[str, Any]],
    recorded_raw: str | None = None,
    recorded_cited_score: dict[str, Any] | None = None,
    seed_scores: dict[str, dict[str, Any]] | None = None,
    repair_prompt: str = REPAIR_PROMPT,
    support_system: str = ANSWER_SUPPORT_SYSTEM,
    support_context_chars: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every mode for one question over already-generated `gens`: gate,
    repair, support judge, then ERB's judges once per distinct answer text.
    `recorded_cited_score` seeds the judge cache for the recorded run's own
    gated text, and `seed_scores` (judged text -> score, from an earlier run
    over the same generations) for any text already judged, so unchanged
    text is not re-judged. `repair_prompt`, `support_system` and
    `support_context_chars` pick the repair and support-judge variant; the
    measurement-only support verdicts use the same variant.

    Returns (row with ids and numbers only, local record with texts)."""
    qtext = question["question"]
    context = format_context_documents(docs)
    doc_texts = {d: f"{t}\n\n{c}" for d, t, c in docs}
    titles = {d: t for d, t, _c in docs}
    gens = dict(gens)
    if recorded_raw is not None:
        gens["recorded"] = {"text": recorded_raw, "s": 0.0, "usd": 0.0}

    runs: dict[str, CitedRun] = {}
    gen_usd: dict[str, float] = {}

    def _cited(mode: str, raw_key: str, **kw: Any) -> None:
        raw_info = gens[raw_key]
        runs[mode] = run_cited(
            chat,
            gen_model,
            qtext,
            context,
            doc_texts,
            raw=raw_info["text"],
            raw_gen_s=raw_info["s"],
            titles=titles,
            repair_prompt=repair_prompt,
            support_context_chars=support_context_chars,
            **kw,
        )
        gen_usd[mode] = raw_info["usd"]

    unparsed: list[str] = []
    _cited("cited_before", "v1")
    _cited("cited_repair", "v1", repair=True)
    _cited("cited_v2", "v2", repair=True)
    support_tally: dict[str, float] = {"usd": 0.0, "calls": 0}
    gate_cache: dict[tuple[str, str], str | None] = {}
    gate_judge = make_chat_support_judge(
        chat,
        judge_model,
        system=support_system,
        tally=support_tally,
        cache=gate_cache,
        unparsed=unparsed,
    )
    _cited("cited_v2_support", "v2", repair=True, support=gate_judge, tally=support_tally)
    if recorded_raw is not None:
        _cited("recorded_before", "recorded")
        _cited("recorded_repair", "recorded", repair=True)

    # Measurement-only verdicts start from the gate's cache, so a segment the
    # gate already judged is not paid for twice.
    diag_tally: dict[str, float] = {"usd": 0.0, "calls": 0}
    diag_judge = make_chat_support_judge(
        chat,
        judge_model,
        system=support_system,
        tally=diag_tally,
        cache=dict(gate_cache),
        unparsed=unparsed,
    )

    judged: dict[str, dict[str, Any]] = dict(seed_scores or {})
    if recorded_raw is not None and recorded_cited_score is not None:
        judged[strip_dsid_references(runs["recorded_before"].text)] = recorded_cited_score

    def _score(text: str) -> dict[str, Any]:
        key = strip_dsid_references(text)
        if key not in judged:
            judged[key] = score_answer_text(chat, judge_model, question, key)
        return judged[key]

    row: dict[str, Any] = {
        "question_id": question["question_id"],
        "question_type": question.get("question_type"),
        "erb_plain": {
            **_score(gens["plain"]["text"]),
            "latency_s": round(gens["plain"]["s"], 3),
            "usd": round(gens["plain"]["usd"], 6),
        },
    }
    record: dict[str, Any] = {
        "question_id": question["question_id"],
        "raw": {k: v["text"] for k, v in gens.items()},
        "gen": {k: {"s": v["s"], "usd": v["usd"]} for k, v in gens.items()},
        "modes": {},
        "support_unparsed": unparsed,
    }
    for mode, run in runs.items():
        entry = {**_score(run.text), **_run_summary(run, gen_usd=gen_usd[mode])}
        mode_record: dict[str, Any] = {
            "final_text": run.text,
            "first_pass_reasons": run.first.reasons,
            "repair_raw": run.repair_raw,
            "repair_reasons": run.repair_reasons,
        }
        if mode == "cited_v2_support":
            withheld = [
                {"sentence": s.text, "evidence": ev, "verdict": v}
                for s, r, ev, v in zip(
                    run.first.segments,
                    run.first.reasons,
                    run.first.evidence,
                    run.first.support_verdicts,
                    strict=True,
                )
                if r == "not_supported"
            ]
            entry["support_gate"] = {
                "judged": sum(1 for r in run.first.reasons if r in (None, "not_supported")),
                "not_supported_first_pass": len(withheld),
                "judge_error_first_pass": sum(1 for w in withheld if w["verdict"] is None),
                "repair_not_supported": run.repair_reasons.count("not_supported"),
            }
            mode_record["support_withheld"] = withheld
        else:
            counts, flagged = _support_diag(
                diag_judge, run.final, doc_texts, titles, support_context_chars
            )
            entry["support_diag"] = counts
            mode_record["support_flagged"] = flagged
        entry["gen_s"] = round(run.gen_s, 3)
        entry["repair_s"] = round(run.repair_s, 3)
        entry["gate_s"] = round(run.gate_s, 3)
        row[mode] = entry
        record["modes"][mode] = mode_record
    row["support_diag_usd"] = round(float(diag_tally["usd"]), 6)
    return row, record


def summarize_modes(rows: Sequence[dict[str, Any]], modes: Sequence[str]) -> dict[str, Any]:
    """Per mode: ERB's answer stats plus the gate, repair, support, cost and
    latency totals over the rows that have the mode."""
    out: dict[str, Any] = {}
    for mode in modes:
        entries = [r[mode] for r in rows if mode in r]
        if not entries:
            continue
        stats = summarize(entries)
        lat = sorted(e["latency_s"] for e in entries if "latency_s" in e)
        usd = [e["usd"] for e in entries if "usd" in e]
        if lat:
            stats["latency_s_mean"] = round(sum(lat) / len(lat), 3)
            stats["latency_s_median"] = round(lat[len(lat) // 2], 3)
            stats["latency_s_max"] = round(lat[-1], 3)
        if usd:
            stats["usd_per_answer_mean"] = round(sum(usd) / len(usd), 6)
        for key in (
            "segments",
            "first_pass_kept",
            "first_pass_dropped",
            "repair_attempted",
            "repaired",
            "repaired_segments",
            "abstained",
            "dropped_final",
            "shipped",
            "quotes",
            "quotes_supported",
        ):
            if any(key in e for e in entries):
                stats[key] = sum(e.get(key, 0) for e in entries)
        reasons: dict[str, int] = {}
        for e in entries:
            for r, n in (e.get("first_pass_reasons") or {}).items():
                reasons[r] = reasons.get(r, 0) + n
        if reasons:
            stats["first_pass_reasons"] = reasons
        for block in ("support_diag", "support_gate"):
            if any(block in e for e in entries):
                total: dict[str, int] = {}
                for e in entries:
                    for k, v in (e.get(block) or {}).items():
                        total[k] = total.get(k, 0) + v
                stats[block] = total
        out[mode] = stats
    return out


def _usd(model: str, tokens_in: float, tokens_out: float) -> float:
    from portolan.pricing import PRICING

    price = PRICING[model]
    return (tokens_in * price.input_per_million + tokens_out * price.output_per_million) / 1e6


def estimate_generation_usd(context_chars: Sequence[int], gen_model: str) -> float:
    """Upper bound for `generate_answers` over the sample: three prompts per
    question at chars/3 input tokens and the full 1,500-token output."""
    return sum(3 * _usd(gen_model, (c + 3000) / 3, 1500) for c in context_chars)


#: A repair restatement is bounded at this many segments per dropped
#: sentence for the estimate (the client's run cap is the hard stop).
REPAIR_SEGMENTS_PER_DROP = 3
_REPAIR_TEXT_CHARS = 6000  # 1,500 output tokens


def evaluation_item(
    *,
    question: dict[str, Any],
    context_chars: int,
    doc_texts: dict[str, str],
    gens: dict[str, dict[str, Any]],
    recorded_raw: str | None,
    already_judged: Sequence[str] = (),
    support_context_chars: int = 0,
) -> dict[str, Any]:
    """What `estimate_evaluation_usd` needs for one question, from the
    already-generated answers: the free mechanical gate says which modes
    will call repair, which segments the support judge will read, and how
    long each judged text can be. `already_judged` names answers ("plain",
    "v1") whose scores are reused and so cost nothing."""
    first = {
        key: gate_cited_answer(parse_cited_answer(gens[key]["text"]), doc_texts)
        for key in ("v1", "v2")
    }
    if recorded_raw is not None:
        first["recorded"] = gate_cited_answer(parse_cited_answer(recorded_raw), doc_texts)

    def _len(key: str) -> int:
        return len(gens[key]["text"]) if key in gens else len(recorded_raw or "")

    repair_calls = 1 if first["v2"].segments else 0  # v2 + support: any segment may fail
    repair_drops = len(first["v2"].segments)
    judged = [
        chars
        for key, chars in (("plain", len(gens["plain"]["text"])), ("v1", _len("v1")))
        if key not in already_judged
    ]
    judged += [_len("v2") + _REPAIR_TEXT_CHARS, _len("v2") + _REPAIR_TEXT_CHARS]
    for key, result in first.items():
        if result.dropped:
            repair_calls += 1
            repair_drops += len(result.dropped)
            if key != "v2":
                judged.append(_len(key) + _REPAIR_TEXT_CHARS)
    support_chars = [
        len(s.text) + sum(len(q) + 120 + 2 * support_context_chars for _d, q in ev)
        for result in first.values()
        for s, r, ev in zip(result.segments, result.reasons, result.evidence, strict=True)
        if r is None
    ]
    return {
        "context_chars": context_chars,
        "facts": len(question.get("answer_facts") or []),
        "qa_chars": len(question.get("question", "")) + len(question.get("gold_answer", "")),
        "repair_calls": repair_calls,
        "repair_segments": REPAIR_SEGMENTS_PER_DROP * repair_drops,
        "support_chars": support_chars,
        "judged_chars": judged,
    }


def estimate_evaluation_usd(
    items: Sequence[dict[str, Any]], gen_model: str, judge_model: str
) -> float:
    """Upper bound for `evaluate_question` over the sample (see
    `evaluation_item`): each repair call at full context and 1,500 output
    tokens; one support call per first-pass keep plus
    `REPAIR_SEGMENTS_PER_DROP` per dropped sentence, each at the full
    150-token output; every judged text at its bound length, with the full
    judge outputs."""
    system = len(ANSWER_SUPPORT_SYSTEM) + 80
    total = 0.0
    for it in items:
        total += it["repair_calls"] * _usd(gen_model, (it["context_chars"] + 3000) / 3, 1500)
        for chars in it["support_chars"]:
            total += _usd(judge_model, (system + chars) / 3, _SUPPORT_MAX_TOKENS)
        total += it["repair_segments"] * _usd(judge_model, (system + 1500) / 3, _SUPPORT_MAX_TOKENS)
        for chars in it["judged_chars"]:
            total += _usd(judge_model, (2300 + it["qa_chars"] + chars) / 3, 300)
            total += it["facts"] * _usd(judge_model, (950 + 400 + chars) / 3, 20)
    return total
