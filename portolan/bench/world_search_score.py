"""WP10 scoring: a retrieval pass and an answer pass over a v2 world's
search-evaluation questions (`portolan.bench.world_questions`).

ERB's own definitions (`portolan.bench.erb`, from ERB's
`metrics_based_eval.py` `score_answer`):

- **Document recall at 10**: share of a question's `expected_doc_ids` in the
  first 10 retrieved, averaged over questions.
- **Invalid extra documents**: retrieved documents that are neither expected
  nor `valid_doc_ids` (sources the WP10 judge found also answer), averaged.
  A recency question's superseded source is not valid: retrieving the older
  state is an invalid extra.
- **Answer-fact completeness**: an answer model reads the retrieved sources
  and answers (ERB's `ANSWER_GEN_PROMPT`); ERB's per-fact judge
  (`INDIVIDUAL_FACT_VALIDATOR_PROMPT`) then rules on each answer fact alone,
  "must not" facts included, and the holistic correctness judge on the whole
  (`portolan.bench.erb_answers.score_answer_text`).

Ours, labelled as such: hit@k and MRR@10 (`erb.aggregate`), and **fact
coverage at 10**, the share of a question's positive answer facts with at
least one source that states it among the first 10 (a multi-document
question can reach full recall only by finding every source, but needs only
one source per fact).

Retrieval results come from `bm25_retrieve` (the world's own BM25 baseline)
or any system's output in `{"question_id", "retrieved": [...]}` rows.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from portolan.bench.erb import (
    TOP_K,
    Bm25Index,
    aggregate,
    document_recall_pct,
    first_gold_rank,
    invalid_extra_docs,
)
from portolan.bench.erb_answers import (
    format_context_documents,
    generate_answer,
    score_answer_text,
    summarize,
)

__all__ = [
    "answer_questions",
    "bm25_retrieve",
    "load_questions",
    "load_retrieved",
    "score_answers",
    "score_retrieval",
]


def load_questions(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_retrieved(path: Path | str) -> dict[str, list[str]]:
    """`{"question_id": ..., "retrieved": [doc ids in rank order]}` rows."""
    out: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["question_id"]] = [str(d) for d in row.get("retrieved") or []]
    return out


def bm25_retrieve(
    index: Bm25Index, questions: Sequence[dict[str, Any]], k: int = TOP_K
) -> dict[str, list[str]]:
    return {q["question_id"]: index.search(q["question"], k) for q in questions}


def _fact_coverage(question: dict[str, Any], top: Sequence[str]) -> float | None:
    kinds = question.get("answer_fact_kinds") or []
    docs = question.get("answer_fact_doc_ids") or []
    positive = [set(d) for kind, d in zip(kinds, docs, strict=False) if kind != "must_not" and d]
    if not positive:
        return None
    found = sum(1 for d in positive if d & set(top))
    return round(found / len(positive) * 100, 2)


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    return {name: _summary(rs) for name, rs in sorted(groups.items())}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = aggregate(rows)
    cov = [r["fact_coverage_pct"] for r in rows if r.get("fact_coverage_pct") is not None]
    out["ours_fact_coverage_pct"] = round(sum(cov) / len(cov), 2) if cov else None
    out["ours_superseded_retrieved"] = sum(r.get("superseded_retrieved", 0) for r in rows)
    return out


def score_retrieval(
    questions: Sequence[dict[str, Any]], retrieved: dict[str, list[str]], k: int = TOP_K
) -> dict[str, Any]:
    """Per-question rows and summaries overall, by kind, by sub-kind and by
    measured paraphrase level. A question with no retrieval row scores as
    an empty list (recall 0), never skipped."""
    rows = []
    for q in questions:
        top = list(dict.fromkeys(retrieved.get(q["question_id"], [])))[:k]
        expected = q.get("expected_doc_ids") or []
        rows.append(
            {
                "question_id": q["question_id"],
                "kind": q.get("kind"),
                "subkind": q.get("subkind"),
                "paraphrase_level": q.get("paraphrase_level"),
                "retrieved": top,
                "document_recall_pct": document_recall_pct(top, expected),
                "invalid_extra_docs": invalid_extra_docs(
                    top, expected, q.get("valid_doc_ids") or []
                ),
                "first_gold_rank": first_gold_rank(top, expected),
                "fact_coverage_pct": _fact_coverage(q, top),
                "superseded_retrieved": len(set(top) & set(q.get("superseded_doc_ids") or [])),
            }
        )
    return {
        "k": k,
        "overall": _summary(rows),
        "by_kind": _group(rows, "kind"),
        "by_subkind": _group(rows, "subkind"),
        "by_paraphrase_level": _group(rows, "paraphrase_level"),
        "rows": rows,
    }


def answer_questions(
    chat: Any,
    answer_model: str,
    questions: Sequence[dict[str, Any]],
    retrieved: dict[str, list[str]],
    texts: dict[str, tuple[str, str]],
    k: int = TOP_K,
) -> dict[str, str]:
    """ERB's answer prompt over the first k retrieved sources, laid out as
    ERB's `format_context_documents` (title = kind and date)."""
    answers = {}
    for q in questions:
        top = list(dict.fromkeys(retrieved.get(q["question_id"], [])))[:k]
        docs = [(d, texts[d][0], texts[d][1]) for d in top if d in texts]
        context = format_context_documents(docs)
        answers[q["question_id"]] = generate_answer(
            chat, answer_model, q["question"], context, cited=False
        )
    return answers


def score_answers(
    chat: Any,
    judge_model: str,
    questions: Sequence[dict[str, Any]],
    answers: dict[str, str],
) -> dict[str, Any]:
    """ERB's correctness and per-fact completeness judges on each answer,
    summarized overall and by kind (`erb_answers.summarize`)."""
    rows = []
    for q in questions:
        scored = score_answer_text(chat, judge_model, q, answers.get(q["question_id"], ""))
        rows.append({"question_id": q["question_id"], "kind": q.get("kind"), **scored})
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_kind[str(r["kind"])].append(r)
    return {
        "overall": summarize(rows),
        "by_kind": {k: summarize(v) for k, v in sorted(by_kind.items())},
        "rows": rows,
    }
