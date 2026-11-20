"""WP10: search-eval questions from a v2 world's state
(`portolan.bench.world_questions`, `world_search_score`, the `questions` CLI).

A hand-built world directory with one arc, one near-duplicate supersession,
one CRM record linking one call, and a system/facility pair sharing a short
name. Every model call is stubbed; the one test of the live wrapper uses an
`httpx.MockTransport`, so nothing leaves the process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from portolan.bench import world_questions as wq
from portolan.bench.world_questions import (
    BudgetExceeded,
    BuildConfig,
    CallLog,
    ModelReply,
    SearchEvalBudget,
    build_intents,
    build_questions,
    check_wording,
    judge_decision,
    lexical_overlap,
    load_question_world,
    numeric_tokens,
    openrouter_model_call,
    paraphrase_level,
    write_build_outputs,
)
from portolan.bench.world_search_score import score_answers, score_retrieval
from portolan.training.spend import SpendMeter
from tests._paths import needs_repo_path

# `model_allowed` reads the teacher allowlist from this script (`label_judge.judge_model_allowed`).
needs_allowlist = needs_repo_path("scripts", "data", "build_training_set.py")

VENDOR = "Tidewell Devices"

F1 = "Marrow General reported that Pumpline T2 infusions stopped for 3 days."
F2 = "Marrow General escalated the Pumpline T2 outage to its chief financial officer."
F3 = "Tidewell Devices committed to ship replacement pumps to Marrow General by 2026-03-20."
F3S = "Tidewell Devices committed to ship replacement pumps to Marrow General by 2026-03-27."
F4 = "Marrow General decided to keep Pumpline T2 after the repair."
F5 = "Birchfield Clinic was quoted 42,000 dollars for its renewal."

SPAN1 = "our Pumpline T2 infusions just stopped, three days straight"
SPAN2 = "I have taken the pump outage to our CFO"
SPAN3 = "we will ship the replacement pumps by March 20"
SPAN3S = "we will ship the replacement pumps by March 27"
SPAN4 = "we are keeping the Pumpline T2 now that it is repaired"
SPAN5 = "the renewal quote came in at 42,000 dollars"

CRM_TEXT = (
    "[crm/accounts.csv#A-4001] account: crm_id=A-4001, name=Marrow General, kind=standalone, "
    "arr=120000, renewal_date=2027-01-31, competitors=Corvane Systems"
)

SOURCES: dict[str, dict[str, Any]] = {
    "S-call-00001": {
        "kind": "call",
        "date": "2026-03-02",
        "account": "acct-marrow",
        "text": f"Account: Marrow General\n[09:00:01] Dana Reyes: Morning. Honestly, {SPAN1}.\n"
        "[09:00:09] Sam Ortiz: Sorry to hear that, let me look.",
    },
    "S-email-00001-m1": {
        "kind": "email",
        "plan": "S-email-00001",
        "date": "2026-03-10",
        "account": "acct-marrow",
        "text": f"[S-email-00001-m1#p1] Sam, {SPAN2} at Marrow General this morning.",
    },
    "S-ticket-00001": {
        "kind": "ticket",
        "date": "2026-03-15",
        "account": "acct-marrow",
        "text": f"[S-ticket-00001#1] Sam Ortiz: Marrow General pumps: {SPAN3}.",
    },
    "S-ticket-00002": {
        "kind": "ticket",
        "date": "2026-03-22",
        "account": "acct-marrow",
        "role": "near_duplicate",
        "text": f"[S-ticket-00002#1] Sam Ortiz: Marrow General pumps: {SPAN3S}.",
    },
    "S-slack-00001": {
        "kind": "slack",
        "date": "2026-04-01",
        "account": "acct-marrow",
        "text": f"Dana Reyes: update from Marrow General, {SPAN4}",
    },
    "S-call-00002": {
        "kind": "call",
        "date": "2026-02-01",
        "account": "acct-birch",
        "text": f"Account: Birchfield Clinic\n[10:00:00] Lee Park: So {SPAN5}.",
    },
    "S-crm-A4001": {
        "kind": "crm_account",
        "plan": "crm/accounts.csv",
        "date": "2026-09-25",
        "account": "acct-marrow",
        "text": CRM_TEXT,
    },
    "S-email-00002-m1": {
        "kind": "email",
        "plan": "S-email-00002",
        "date": "2026-02-11",
        "account": "acct-oak-north",
        "text": "[S-email-00002-m1#p1] Oakridge North Hospital asked about parking for training.",
    },
}

LABELS = [
    ("F1", "complaint", "S-call-00001", SPAN1, {}),
    ("F2", "risk", "S-email-00001-m1", SPAN2, {}),
    ("F3", "commitment", "S-ticket-00001", SPAN3, {}),
    ("F3s", "commitment", "S-ticket-00002", SPAN3S, {"supersedes": "F3"}),
    ("F4", "decision", "S-slack-00001", SPAN4, {}),
    ("F5", "pricing", "S-call-00002", SPAN5, {}),
    ("CF-acct-marrow-arr-01", "other", "S-crm-A4001", "120000", {}),
    ("CF-acct-marrow-renewal_date-02", "decision", "S-crm-A4001", "2027-01-31", {}),
]


def _fact(fid, ftype, claim, account, day, event, **extra):
    return {
        "id": fid,
        "fact_type": ftype,
        "claim": claim,
        "entities": [account, "product-pumpline-t2"],
        "speaker": "person-dana-reyes",
        "date": day,
        "account_id": account,
        "event_id": event,
        "valid_from": None,
        "valid_to": None,
        "supersedes": None,
        "anchors": [],
        "directive": None,
        "reference_mode": "explicit",
        **extra,
    }


def make_world(root: Path, *, crm_link: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    accounts = [
        ("acct-marrow", "Marrow General", ["Marrow"], "A-4001"),
        ("acct-birch", "Birchfield Clinic", ["Birchfield"], "A-4002"),
        ("acct-oak", "Oakridge Health System", ["Oakridge"], "A-4003"),
        ("acct-oak-north", "Oakridge North Hospital", ["Oakridge"], "A-4004"),
    ]
    world = {
        "id": "world-test-1",
        "vertical": "medical_devices",
        "vendor": {
            "name": VENDOR,
            "vertical": "medical_devices",
            "products": [{"id": "product-pumpline-t2", "name": "Pumpline T2", "aliases": ["T2"]}],
            "competitors": [{"id": "competitor-corvane", "name": "Corvane Systems"}],
        },
        "accounts": [
            {"id": a, "name": n, "aliases": al, "crm_id": c, "parent": None}
            for a, n, al, c in accounts
        ],
        "people": [
            {
                "id": "person-dana-reyes",
                "name": "Dana Reyes",
                "title": "Director of Nursing",
                "side": "customer",
                "account_id": "acct-marrow",
            },
            {
                "id": "person-sam-ortiz",
                "name": "Sam Ortiz",
                "title": "Account Rep",
                "side": "vendor",
            },
        ],
        "events": [
            {"id": e, "date": d, "type": t, "account_id": a}
            for e, d, t, a in [
                ("E1", "2026-03-02", "incident", "acct-marrow"),
                ("E2", "2026-03-10", "escalation", "acct-marrow"),
                ("E3", "2026-03-15", "commitment_made", "acct-marrow"),
                ("E4", "2026-04-01", "decision", "acct-marrow"),
                ("E5", "2026-02-01", "price_negotiation", "acct-birch"),
            ]
        ],
        "facts": [
            _fact("F1", "complaint", F1, "acct-marrow", "2026-03-02", "E1"),
            _fact("F2", "risk", F2, "acct-marrow", "2026-03-10", "E2"),
            _fact("F3", "commitment", F3, "acct-marrow", "2026-03-15", "E3"),
            _fact("F4", "decision", F4, "acct-marrow", "2026-04-01", "E4"),
            _fact("F5", "pricing", F5, "acct-birch", "2026-02-01", "E5"),
        ],
        "arcs": [
            {
                "id": "ARC0001",
                "arc_type": "incident_escalation_exec",
                "account_id": "acct-marrow",
                "event_ids": ["E1", "E2", "E3", "E4"],
            }
        ],
    }
    (root / "world.yaml").write_text(yaml.safe_dump(world, sort_keys=False), encoding="utf-8")
    with (root / "sources.jsonl").open("w", encoding="utf-8") as fh:
        for sid, s in SOURCES.items():
            fh.write(
                json.dumps(
                    {
                        "source_id": sid,
                        "source_type": {"crm_account": "crm", "slack": "chat"}.get(
                            s["kind"], s["kind"]
                        ),
                        "kind": s["kind"],
                        "date": s["date"],
                        "text": s["text"],
                        "plan_id": s.get("plan", sid),
                        "account_id": s["account"],
                        "true_account_id": s["account"],
                        "role": s.get("role", "event"),
                    }
                )
                + "\n"
            )
    with (root / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for fid, ftype, doc, span, extra in LABELS:
            start = SOURCES[doc]["text"].index(span)
            fh.write(
                json.dumps(
                    {
                        "fact_id": fid,
                        "fact_type": ftype,
                        "doc_id": doc,
                        "char_start": start,
                        "char_end": start + len(span),
                        "text": span,
                        **extra,
                    }
                )
                + "\n"
            )
    row = {
        "crm_id": "A-4001",
        "name": "Marrow General",
        "arr": "120000",
        "renewal_date": "2027-01-31",
        "competitors": "Corvane Systems",
    }
    plan = {
        "world_id": "world-test-1",
        "crm_accounts": [
            {
                "id": "S-crm-A4001",
                "account_id": "acct-marrow",
                "crm_id": "A-4001",
                "row": row,
                "field_facts": [
                    {"id": "CF-acct-marrow-arr-01", "fields": ["arr"]},
                    {"id": "CF-acct-marrow-renewal_date-02", "fields": ["renewal_date"]},
                ],
                "linked_calls": ["S-call-00001"],
            }
        ],
    }
    (root / "world_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "noise").mkdir(exist_ok=True)
    noise = {
        "supersessions": [
            {
                "old_fact_id": "F3",
                "new_fact_id": "F3s",
                "near_duplicate_id": "S-ticket-00002",
                "shadow_id": "S-ticket-00001",
            }
        ],
        "new_facts": [
            _fact("F3s", "commitment", F3S, "acct-marrow", "2026-03-22", "E3", supersedes="F3")
        ],
        # WP6 link noise: the true link to the call survives only when crm_link.
        "crm_linked_overrides": {
            "S-crm-A4001": {"linked_calls": ["S-call-00001"] if crm_link else ["S-call-00002"]}
        },
    }
    (root / "noise" / "noise_manifest.json").write_text(json.dumps(noise), encoding="utf-8")
    return root


@pytest.fixture
def world_dir(tmp_path: Path) -> Path:
    return make_world(tmp_path / "world-test-1")


ALL = {"single_fact": 10, "multi_doc": 5, "recency": 5, "cross_source": 5}


# ---------------------------------------------------------------------
# Loading and intents
# ---------------------------------------------------------------------


def test_load_maps_facts_to_their_sources_and_anchors(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    assert w.fact_docs["F1"] == {"S-call-00001"}
    assert w.fact_docs["F3s"] == {"S-ticket-00002"}
    assert w.facts["F3s"]["claim"] == F3S
    # The shared short name "Oakridge" names neither the system nor the facility.
    assert w.anchor_names["acct-oak-north"] == ["Oakridge North Hospital"]
    assert w.anchor_names["acct-marrow"] == ["Marrow General", "Marrow"]
    assert w.crm_records[0]["effective_linked_calls"] == ["S-call-00001"]


def test_a_stale_crm_link_gives_no_cross_source_question(tmp_path: Path) -> None:
    w = load_question_world(make_world(tmp_path / "w", crm_link=False))
    assert w.crm_records[0]["effective_linked_calls"] == []
    _intents, supply = build_intents(w, ALL)
    assert supply["cross_source"] == 0


def test_intents_have_exact_gold_by_kind(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    intents, supply = build_intents(w, ALL)
    by_kind: dict[str, list] = {}
    for i in intents:
        by_kind.setdefault(i.kind, []).append(i)

    singles = {i.fact_ids[0]: i for i in by_kind["single_fact"]}
    # F3 was superseded and F3s is the superseding fact: neither is a lookup.
    assert set(singles) == {"F1", "F2", "F4", "F5"}
    assert singles["F5"].gold_doc_ids == ["S-call-00002"]
    assert singles["F5"].answer[0].text == F5

    (multi,) = by_kind["multi_doc"]
    assert multi.fact_ids == ["F1", "F2", "F4"]
    assert multi.gold_doc_ids == ["S-call-00001", "S-email-00001-m1", "S-slack-00001"]
    assert multi.arc_id == "ARC0001"

    recency = {i.subkind: i for i in by_kind["recency"]}
    sup = recency["supersession"]
    assert sup.gold_doc_ids == ["S-ticket-00002"]
    assert sup.superseded_doc_ids == ["S-ticket-00001"]
    assert sup.fact_ids == ["F3s"]
    must_not = [a for a in sup.answer if a.negative]
    assert must_not[0].values["old"] == "2026-03-20"
    latest = recency["arc_latest"]
    assert latest.gold_doc_ids == ["S-slack-00001"] and latest.background == [F1]
    assert latest.latest_date == "2026-04-01"

    (cross,) = by_kind["cross_source"]
    assert cross.gold_doc_ids == ["S-call-00001", "S-crm-A4001"]
    assert cross.answer[0].kind == "crm_field" and cross.answer[1].text == F1
    assert supply == {"single_fact": 4, "multi_doc": 1, "recency": 2, "cross_source": 1}


def test_intents_are_deterministic_and_spread(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    a, _ = build_intents(w, ALL, seed=3)
    b, _ = build_intents(load_question_world(world_dir), ALL, seed=3)
    assert [(i.intent_id, i.gold_doc_ids, i.paraphrase_requested) for i in a] == [
        (i.intent_id, i.gold_doc_ids, i.paraphrase_requested) for i in b
    ]
    two, _ = build_intents(w, {"single_fact": 2})
    types = {w.facts[i.fact_ids[0]]["fact_type"] for i in two}
    assert len(types) == 2  # round-robin over fact types, not two of one type


def test_answer_values_are_forbidden_in_the_question(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    intents, _ = build_intents(w, ALL)
    sup = next(i for i in intents if i.subkind == "supersession")
    assert "2026-03-27" in sup.forbidden_numbers
    assert "2026-03-20" in sup.forbidden_numbers  # the old value too
    assert "March 27" in sup.forbidden_phrases
    single_f1 = next(i for i in intents if i.kind == "single_fact" and i.fact_ids == ["F1"])
    assert "3" in single_f1.forbidden_numbers and "three" in single_f1.forbidden_phrases
    # "Pumpline T2" is a name: its digit is not a number.
    assert "2" not in single_f1.forbidden_numbers


# ---------------------------------------------------------------------
# Code checks on wording
# ---------------------------------------------------------------------


def test_numeric_tokens_normalize() -> None:
    assert numeric_tokens("ARR $775,773 and 12% by 2026-03-20, the 9th, Meralight T3") == {
        "775773",
        "12",
        "2026-03-20",
        "9",
    }


def test_check_wording_flags_each_problem(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    intents, _ = build_intents(w, ALL)
    names = w.names()
    f1 = next(i for i in intents if i.kind == "single_fact" and i.fact_ids == ["F1"])
    good = "What went wrong with the infusion pumps at Marrow General in March?"
    assert check_wording(f1, good, {}, names) == []
    assert "no_question_mark" in check_wording(f1, good.rstrip("?"), {}, names)
    assert "account_not_named" in check_wording(f1, "What went wrong with the pumps?", {}, names)
    leak = check_wording(f1, "Did Marrow General's pumps stop for three days?", {}, names)
    assert any(p.startswith("answer_value_in_question") for p in leak)
    copied = "Why has Marrow General reported that Pumpline T2 infusions stopped for days?"
    assert any(p.startswith("copies_answer") for p in check_wording(f1, copied, {}, names))

    oak = next(i for i in intents if i.kind == "single_fact")
    oak.anchor_names = ["Oakridge North Hospital"]
    assert "account_not_named" in check_wording(oak, "What did Oakridge ask for?", {}, names)


def test_check_wording_checks_statement_values(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    intents, _ = build_intents(w, ALL)
    names = w.names()
    cross = next(i for i in intents if i.kind == "cross_source")
    field, value = next(iter(cross.answer[0].values.items()))
    q = "What does the CRM say about Marrow General, and what broke on their March call?"
    good = {"A1": f"The CRM record for Marrow General lists {field} as {value}."}
    assert check_wording(cross, q, good, names) == []
    wrong = {"A1": "The CRM record for Marrow General lists something else."}
    assert "statement_value:A1" in check_wording(cross, q, wrong, names)
    assert "statement_missing:A1" in check_wording(cross, q, {}, names)

    sup = next(i for i in intents if i.subkind == "supersession")
    q2 = "When will Marrow General now get its replacement pumps?"
    ok = {"N1": "The answer must not give the 2026-03-20 ship date."}
    assert check_wording(sup, q2, ok, names) == []
    assert "statement_not_negative:N1" in check_wording(
        sup, q2, {"N1": "The ship date was 2026-03-20."}, names
    )


def test_lexical_overlap_and_level() -> None:
    assert lexical_overlap("Which pumps stopped at Marrow?", ["the pumps stopped at Marrow"]) == 1.0
    assert paraphrase_level(0.8) == "close"
    assert paraphrase_level(0.6) == "moderate"
    assert paraphrase_level(0.2) == "heavy"


# ---------------------------------------------------------------------
# The judge's verdict, decided by code
# ---------------------------------------------------------------------


def _intent(world_dir: Path, subkind: str):
    w = load_question_world(world_dir)
    intents, _ = build_intents(w, ALL)
    return next(i for i in intents if i.subkind == subkind)


def _reply(facts, docs, clear="yes"):
    return {
        "facts": [{"fact": n, "doc_ids": d} for n, d in facts],
        "documents": [{"doc_id": d, "answers": a, "conflict": c} for d, a, c in docs],
        "question_clear": clear,
        "reason": "r",
    }


def test_judge_decision_keeps_records_valid_and_drops(world_dir: Path) -> None:
    intent = _intent(world_dir, "lookup")
    gold = intent.gold_doc_ids[0]
    shown = [gold, "S-email-00002-m1", "S-ticket-00001"]
    dates = dict.fromkeys(shown, "2026-03-01")
    ok = judge_decision(
        intent,
        _reply(
            [(1, [gold, "S-ticket-00001"])],
            [
                (gold, "full", "no"),
                ("S-ticket-00001", "partial", "no"),
                ("S-email-00002-m1", "none", "no"),
            ],
        ),
        shown,
        dates,
        1,
    )
    assert ok.keep and ok.valid_doc_ids == ["S-ticket-00001"]

    unsupported = judge_decision(intent, _reply([(1, ["S-ticket-00001"])], []), shown, dates, 1)
    assert unsupported.reason == "judge_fact_not_in_gold:1"

    conflict = judge_decision(
        intent, _reply([(1, [gold])], [("S-ticket-00001", "full", "yes")]), shown, dates, 1
    )
    assert conflict.reason == "judge_conflicting_document"

    unclear = judge_decision(intent, _reply([(1, [gold])], [], clear="no"), shown, dates, 1)
    assert unclear.reason == "judge_question_unclear"
    assert judge_decision(intent, None, shown, dates, 1).reason == "judge_no_verdict"


def test_judge_decision_recency_older_source_is_superseded(world_dir: Path) -> None:
    intent = _intent(world_dir, "supersession")
    shown = ["S-ticket-00001", "S-ticket-00002"]
    dates = {"S-ticket-00001": "2026-03-15", "S-ticket-00002": "2026-03-22"}
    out = judge_decision(
        intent,
        _reply([(1, ["S-ticket-00002"])], [("S-ticket-00001", "full", "yes")]),
        shown,
        dates,
        1,
    )
    assert out.keep and out.superseded_doc_ids == ["S-ticket-00001"]


# ---------------------------------------------------------------------
# The run, with stubbed models
# ---------------------------------------------------------------------


class StubModels:
    """Wording by intent account, judge by the labels; records every call."""

    def __init__(self, world, questions: dict[str, list[str]], judge=None) -> None:
        self.world = world
        self.questions = questions
        self.calls: list[dict[str, Any]] = []
        self.judge = judge or wq.fixture_judge_call(world)

    def __call__(self, **kwargs: Any) -> ModelReply:
        self.calls.append(kwargs)
        if kwargs["json_schema"] is not wq.WORDING_SCHEMA:
            return self.judge(**kwargs)
        brief = json.loads(kwargs["user"])
        key = next(a["fact"] for a in brief["ANSWER"] if a.get("fact") in self.questions)
        queue = self.questions[key]
        question = queue.pop(0) if len(queue) > 1 else queue[0]
        statements = []
        for a in brief["ANSWER"]:
            if "crm_record" in a:
                ((field, value),) = a["crm_record"].items()
                statements.append(
                    {"id": a["id"], "text": f"Marrow General's CRM {field} is {value}."}
                )
        if "OLD" in brief:
            statements.append({"id": "N1", "text": "The answer must not give 2026-03-20."})
        data = {"question": question, "statements": statements}
        return ModelReply(data=data, text=json.dumps(data), cost_usd=0.001)


def _questions_for() -> dict[str, list[str]]:
    return {
        F1: ["What happened to the infusion pumps at Marrow General?"],
        F2: ["Who did Marrow General take the pump outage to?"],
        F4: [
            "Did Marrow General decide to keep Pumpline T2 after the repair?",  # copies: retried
            "What did Marrow General settle on for its pumps after the repair work?",
        ],
        F5: ["What renewal quote did Birchfield Clinic get?"],
        F3S: ["When will Marrow General now get its replacement pumps?"],
    }


def _cfg(**kw: Any) -> BuildConfig:
    base = {
        "counts": ALL,
        "wording_models": ("qwen/qwen3-32b", "deepseek/deepseek-v4-flash"),
        "concurrency": 1,
    }
    base.update(kw)
    return BuildConfig(**base)


@needs_allowlist
def test_build_questions_end_to_end(world_dir: Path, tmp_path: Path) -> None:
    w = load_question_world(world_dir)
    stub = StubModels(w, _questions_for())
    budget = SearchEvalBudget(1.0)
    result = build_questions(w, _cfg(), call=stub, budget=budget, log=CallLog(None))
    kept = {q["question_id"]: q for q in result.questions}
    assert result.report()["planned"] == 8
    assert result.dropped == [] and len(kept) == 8
    f4 = next(q for q in kept.values() if q["fact_ids"] == ["F4"] and q["kind"] == "single_fact")
    assert f4["wording_attempts"] == 2  # the first wording copied the claim
    assert f4["wording_model"] in ("qwen/qwen3-32b", "deepseek/deepseek-v4-flash")
    sup = next(q for q in kept.values() if q["subkind"] == "supersession")
    assert sup["expected_doc_ids"] == ["S-ticket-00002"]
    assert sup["answer_facts"] == [F3S, "The answer must not give 2026-03-20."]
    assert sup["answer_fact_sources"] == ["world_claim", "open_weight_model"]
    assert sup["gold_answer"] == F3S
    cross = next(q for q in kept.values() if q["kind"] == "cross_source")
    assert cross["answer_fact_kinds"] == ["crm_field", "claim"]
    assert set(cross["expected_doc_ids"]) == {"S-crm-A4001", "S-call-00001"}
    for q in kept.values():
        assert q["bm25_first_gold_rank"] is not None
        assert q["paraphrase_requested"] in wq.PARAPHRASE_LEVELS
        assert 0.0 <= q["lexical_overlap"] <= 1.0
    wording_calls = [c for c in stub.calls if c["json_schema"] is wq.WORDING_SCHEMA]
    assert len(wording_calls) == 9  # eight questions, one retried
    assert budget.spent() == pytest.approx(0.001 * len(wording_calls))

    out = tmp_path / "out"
    write_build_outputs(out, result)
    rows = [json.loads(line) for line in (out / "questions.jsonl").read_text().splitlines()]
    assert len(rows) == len(result.questions)
    report = json.loads((out / "report.json").read_text())
    assert report["kept"] + len(result.dropped) == report["planned"]
    assert not (out / "DRY_RUN").exists()


@needs_allowlist
def test_bm25_miss_and_judge_rejection_are_logged(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    questions = _questions_for()
    # Ranks the ticket that says "replacement pumps" above F2's email.
    questions[F2] = ["Did Marrow General ship replacement pumps, pumps, pumps?"]

    def judge(**kwargs: Any) -> ModelReply:
        payload = json.loads(kwargs["user"])
        shown = [d["doc_id"] for d in payload["documents"]]
        data = _reply([(1, [])], [(d, "none", "no") for d in shown])
        return ModelReply(data=data, text="", cost_usd=0.0)

    stub = StubModels(w, questions, judge=judge)
    cfg = _cfg(counts={"single_fact": 4}, bm25_depth=1)
    result = build_questions(w, cfg, call=stub, budget=SearchEvalBudget(1.0), log=CallLog(None))
    reasons = {d["reason"].split(":")[0] for d in result.dropped}
    assert "no_gold_in_bm25_top_1" in reasons
    assert "judge_fact_not_in_gold" in reasons  # the judge found no fact in any source
    assert result.questions == []
    assert all(d["stage"] in ("bm25", "judge") for d in result.dropped)


@needs_allowlist
def test_budget_stops_the_run_before_the_cap(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    stub = StubModels(w, _questions_for())
    budget = SearchEvalBudget(1e-9)
    result = build_questions(w, _cfg(), call=stub, budget=budget, log=CallLog(None))
    assert stub.calls == []
    assert result.budget_stopped
    assert {d["reason"] for d in result.dropped} == {"budget_exhausted"}


def test_budget_reserve_counts_calls_in_flight() -> None:
    budget = SearchEvalBudget(0.01)
    budget.reserve(0.006)
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.006)
    budget.release(0.006, 0.002)
    budget.reserve(0.006)
    assert budget.spent() == pytest.approx(0.002)


@needs_allowlist
def test_live_models_must_be_allowlisted_open_weight(world_dir: Path) -> None:
    w = load_question_world(world_dir)
    with pytest.raises(ValueError, match="allowlist"):
        build_questions(
            w,
            _cfg(judge_model="anthropic/claude-haiku-4.5"),
            call=StubModels(w, {}),
            budget=SearchEvalBudget(1.0),
            log=CallLog(None),
        )
    assert wq.model_allowed(wq.DEFAULT_JUDGE_MODEL)
    assert all(wq.model_allowed(m) for m in wq.DEFAULT_WORDING_MODELS)


def test_openrouter_call_is_metered_under_v2_search_eval(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen/qwen3-32b"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"question": "Q?", "statements": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0042},
            },
        )

    meter = SpendMeter(tmp_path / "spend.json")
    call = openrouter_model_call("k", meter, transport=httpx.MockTransport(handler))
    reply = call(
        model="qwen/qwen3-32b",
        system="s",
        user="u",
        max_tokens=50,
        json_schema=wq.WORDING_SCHEMA,
        temperature=0.7,
    )
    assert reply.data == {"question": "Q?", "statements": []}
    assert reply.cost_usd == pytest.approx(0.0042)
    assert meter.corpus_usd("v2_search_eval") == pytest.approx(0.0042)


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------


def _qrow(**kw: Any) -> dict[str, Any]:
    row = {
        "question_id": "q1",
        "kind": "multi_doc",
        "subkind": "arc",
        "question": "What happened at Marrow General?",
        "expected_doc_ids": ["a", "b"],
        "valid_doc_ids": ["v"],
        "superseded_doc_ids": ["old"],
        "answer_facts": ["fact one", "fact two", "The answer must not give 5."],
        "answer_fact_kinds": ["claim", "claim", "must_not"],
        "answer_fact_doc_ids": [["a"], ["b", "c"], []],
        "gold_answer": "fact one fact two",
        "paraphrase_level": "moderate",
    }
    row.update(kw)
    return row


def test_score_retrieval_uses_erb_definitions() -> None:
    q = _qrow()
    out = score_retrieval([q], {"q1": ["a", "v", "old", "x", "c"]}, k=10)
    row = out["rows"][0]
    assert row["document_recall_pct"] == 50.0  # a of {a, b}
    assert row["invalid_extra_docs"] == 3  # old, x, c: v is valid
    assert row["first_gold_rank"] == 1
    assert row["fact_coverage_pct"] == 100.0  # c states fact two
    assert row["superseded_retrieved"] == 1
    assert out["overall"]["average_recall_pct"] == 50.0
    assert out["by_kind"]["multi_doc"]["count"] == 1

    missing = score_retrieval([q], {}, k=10)["rows"][0]
    assert missing["document_recall_pct"] == 0.0 and missing["fact_coverage_pct"] == 0.0


def test_score_answers_judges_each_fact_alone() -> None:
    seen: list[str] = []

    class Chat:
        def complete(self, model: str, prompt: str, *, max_tokens: int = 1200) -> str:
            seen.append(prompt)
            if "## Gold Answer" in prompt:
                return '{"reason": "r", "aligned": "yes"}'
            statement = prompt.split("## Statement", 1)[1]
            return "yes" if "fact one" in statement or "must not" in statement else "no"

    out = score_answers(Chat(), "judge/model", [_qrow()], {"q1": "fact one, and more"})
    row = out["rows"][0]
    assert row["facts_valid"] == [True, False, True]
    assert row["completeness_pct"] == pytest.approx(66.67)
    assert out["overall"]["average_correctness_pct"] == 100.0
    assert sum("## Statement" in p for p in seen) == 3


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def test_cli_plan_only_makes_no_call(world_dir: Path, tmp_path: Path, monkeypatch) -> None:
    from portolan.bench import world_questions_cli
    from portolan.bench.__main__ import main

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("plan-only must not build a model client")

    monkeypatch.setattr(world_questions_cli, "openrouter_model_call", refuse)
    out = tmp_path / "plan"
    args = ["questions", "build", "--world", str(world_dir), "--out", str(out), "--plan-only"]
    assert main(args) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["supply"] == {"single_fact": 4, "multi_doc": 1, "recency": 2, "cross_source": 1}
    assert plan["estimate"]["expected_usd"] > 0
    assert len((out / "intents.jsonl").read_text().splitlines()) == 8
    assert not (out / "questions.jsonl").exists()


def test_cli_dry_run_build_then_score(world_dir: Path, tmp_path: Path) -> None:
    from portolan.bench.__main__ import main

    out = tmp_path / "dry"
    args = ["questions", "build", "--world", str(world_dir), "--out", str(out), "--dry-run"]
    assert main(args) == 0
    assert (out / "DRY_RUN").exists()
    rows = [json.loads(line) for line in (out / "questions.jsonl").read_text().splitlines()]
    assert rows and all(r["dry_run"] for r in rows)
    report = json.loads((out / "report.json").read_text())
    assert report["kept"] == len(rows) and report["spent_usd"] == 0.0

    score_path = tmp_path / "score.json"
    args = ["questions", "score", "--world", str(world_dir), "--questions"]
    args += [str(out / "questions.jsonl"), "--answers", "--dry-run", "--out", str(score_path)]
    assert main(args) == 0
    score = json.loads(score_path.read_text())
    assert score["retrieval"] == "bm25"
    assert score["search"]["overall"]["count"] == len(rows)
    assert len(score["answers"]["rows"]) == len(rows)


def test_a_dry_run_world_is_never_worded_for_real(world_dir: Path, tmp_path: Path) -> None:
    from portolan.bench.__main__ import main

    (world_dir / "manifest.json").write_text(json.dumps({"dry_run": True}), encoding="utf-8")
    w = load_question_world(world_dir)
    assert w.dry_run_world
    with pytest.raises(ValueError, match="dry-run world"):
        build_questions(
            w, _cfg(), call=StubModels(w, {}), budget=SearchEvalBudget(1.0), log=CallLog(None)
        )
    args = ["questions", "build", "--world", str(world_dir), "--out", str(tmp_path / "x")]
    assert main(args) == 2
    assert main([*args, "--dry-run"]) == 0


@needs_allowlist
def test_cli_refuses_a_closed_model(world_dir: Path, tmp_path: Path) -> None:
    from portolan.bench.__main__ import main

    args = ["questions", "build", "--world", str(world_dir), "--out", str(tmp_path / "x")]
    args += ["--judge-model", "anthropic/claude-haiku-4.5"]
    assert main(args) == 2
    assert not (tmp_path / "x" / "calls.jsonl").exists()
