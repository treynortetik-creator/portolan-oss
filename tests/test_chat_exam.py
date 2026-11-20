"""The chat exam (`portolan.bench.chat_exam*`, 01-spec/chat-exam-2026-09-29.md): the split,
the lock on the test half, the grader's scoring and blindness, the statistics, the
acceptance rule, and the cases and checks against a seeded demo graph. No model call."""

from __future__ import annotations

import dataclasses
import gzip
import json
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from portolan.bench import chat_exam as ex
from portolan.bench.chat_exam_cases import (
    CASE_TYPES,
    ExamCase,
    GoldClaim,
    demo_cases,
    load_split,
    split_cases,
    write_exam,
)
from portolan.llm.base import LLMResult
from tests._paths import needs_repo_path


def _case(cid: str, kind: str = "single_fact", public: bool = False, n_gold: int = 2) -> ExamCase:
    gold = (
        ()
        if kind == "abstain"
        else tuple(
            GoldClaim(f"fact {cid} {i}", (f"note-{cid}",), (i,), f"F{cid}{i}")
            for i in range(n_gold)
        )
    )
    return ExamCase(cid, "demo", kind, f"Question {cid}?", gold, "test", public)


def _transcript(sentences: list[dict[str, Any]] | None = None, **kw: Any) -> dict[str, Any]:
    sentences = sentences or []
    return {
        "id": "x",
        "question": "q",
        "sentences": sentences,
        "abstained": not any(s["citations"] for s in sentences),
        "usage": {"model": "anthropic/claude-sonnet-5", "chat_mode": "agent"},
        "error": None,
        **kw,
    }


OK = {"ok": True, "citations": 1, "unresolved": [], "not_verbatim": []}


# --- split and lock ---------------------------------------------------------------------


def test_split_pins_public_cases_to_train_and_stratifies_the_rest() -> None:
    cases = [_case(f"p{i}", "multi_doc", public=True) for i in range(5)]
    for kind in CASE_TYPES:
        cases += [_case(f"{kind}-{i}", kind) for i in range(8)]
    split = split_cases(cases, seed=7, test_fraction=0.5)
    assert set(split["train"]) | set(split["test"]) == {c.id for c in cases}
    assert not set(split["train"]) & set(split["test"])
    assert {f"p{i}" for i in range(5)} <= set(split["train"])
    for kind in CASE_TYPES:
        assert sum(1 for i in split["test"] if i.startswith(f"{kind}-")) == 4
    assert split_cases(cases, seed=7, test_fraction=0.5) == split
    assert split_cases(cases, seed=8, test_fraction=0.5) != split


# write_exam reads scripts/demo_questions.txt, which the image does not ship.
@needs_repo_path("scripts", "demo_questions.txt")
def test_the_test_half_is_written_compressed_and_only_under_the_locked_path(
    tmp_path: Path,
) -> None:
    gone = dataclasses.replace(_case("c1", "cross_account"), dropped="gold incomplete")
    cases = [_case("a1"), _case("a2"), _case("b1", "abstain"), gone]
    split = {"train": ["a1", "c1"], "test": ["a2", "b1"]}
    exam_dir, locked = tmp_path / "evals" / "chat-exam", tmp_path / "evals" / "locked" / "t.json.gz"
    counts = write_exam(cases, split, exam_dir=exam_dir, locked_path=locked)
    assert counts == {"train": {"single_fact": 1}, "test": {"single_fact": 1, "abstain": 1}}
    assert json.loads((exam_dir / "split.json").read_text())["dropped"] == {"c1": "gold incomplete"}
    readable = "".join(p.read_text() for p in exam_dir.iterdir())
    assert "Question a2?" not in readable and "fact a2" not in readable
    assert b"Question a2?" not in locked.read_bytes()
    assert "Question a2?" in gzip.decompress(locked.read_bytes()).decode()
    test = load_split("test", exam_dir=exam_dir, locked_path=locked)
    assert [c.id for c in test] == ["a2", "b1"] and test[0].gold[0].claim_ids == (0,)
    assert [c.id for c in load_split("train", exam_dir=exam_dir)] == ["a1"]
    meta = json.loads((exam_dir / "split.json").read_text())
    assert meta["test"] == ["a2", "b1"] and meta["seed"]


def test_the_committed_exam_matches_its_split() -> None:
    split = json.loads((ex.REPO_ROOT / "evals" / "chat-exam" / "split.json").read_text())
    train = load_split("train")
    test = load_split("test")
    assert [c.id for c in train] == split["train"]
    assert [c.id for c in test] == split["test"]
    assert len(train) + len(test) >= 60
    assert not any(c.public for c in test)
    for c in test:
        assert c.expects_abstain or c.gold, c.id
        assert all(g.notes for g in c.gold), c.id
    refs = [g.ref for c in [*train, *test] for g in c.gold if g.ref]
    assert len(refs) == len(set(refs)), "a gold fact answers two cases"
    smoke = load_split("smoke")
    assert len(smoke) == 10 and not {c.question for c in smoke} & {c.question for c in train}


# --- grading ----------------------------------------------------------------------------


def test_parse_verdict_needs_every_item() -> None:
    good = {
        "checklist": [{"id": "G1", "stated": "yes"}, {"id": "g2", "stated": "No"}],
        "sentences": [{"id": "S1", "supported": "yes"}],
    }
    assert ex.parse_verdict(good, 2, 1) == {"stated": [True, False], "supported": [True]}
    assert ex.parse_verdict(good, 3, 1) is None
    assert ex.parse_verdict({"checklist": good["checklist"]}, 2, 1) is None
    bad = {**good, "sentences": [{"id": "S1", "supported": "maybe"}]}
    assert ex.parse_verdict(bad, 2, 1) is None


def test_score_case_outcomes() -> None:
    cited = [{"text": "A.", "citations": [{"note_slug": "n", "claim_id": 1}]}]
    uncited = [{"text": "Not in the graph yet: x.", "citations": []}]
    answerable = _case("c", n_gold=2)
    abstain = _case("d", "abstain")
    v = {"stated": [True, False], "supported": [True]}
    s = ex.score_case(answerable, _transcript(cited), OK, v)
    assert (s["score"], s["pass"], s["recall"]) == (0.5, False, 0.5)
    v2 = {"stated": [True, True], "supported": [True, False]}
    assert ex.score_case(answerable, _transcript(cited * 2), OK, v2)["score"] == 0.5
    full = {"stated": [True, True], "supported": [True]}
    assert ex.score_case(answerable, _transcript(cited), OK, full)["pass"] is True
    assert ex.score_case(answerable, _transcript(uncited), OK, None)["score"] == 0.0
    assert ex.score_case(abstain, _transcript(uncited), OK, None)["score"] == 1.0
    assert ex.score_case(abstain, _transcript(cited), OK, None)["score"] == 0.0
    failed = {**OK, "ok": False}
    assert ex.score_case(answerable, _transcript(cited), failed, full)["outcome"] == "check_failed"
    judged_out = ex.score_case(answerable, _transcript(cited), OK, None)
    assert judged_out["score"] is None and judged_out["outcome"] == "judge_failed"
    err = ex.score_case(answerable, _transcript(cited, error="LLMError: x"), OK, full)
    assert err["score"] == 0.0


class _FakeJudge:
    def __init__(self, replies: list[dict[str, Any] | None]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, **kw: Any) -> LLMResult:
        self.prompts.append(kw["system"] + "\n" + kw["user"])
        data = self.replies.pop(0)
        return LLMResult(
            text=json.dumps(data) if data else "no json", data=data, input_tokens=1,
            output_tokens=1, cached_tokens=0, model=kw["model"],
        )  # fmt: skip


def test_the_judge_is_blind_to_the_config_and_retries_a_malformed_reply() -> None:
    case = _case("c", n_gold=1)
    sentences = [{"text": "Fact c 0.", "citations": [{"note_slug": "n", "claim_id": 1}]}]
    t = _transcript(sentences)
    good = {
        "checklist": [{"id": "G1", "stated": "yes"}],
        "sentences": [{"id": "S1", "supported": "yes"}],
    }
    judge = _FakeJudge([None, good])
    verdict = ex.judge_case(judge, "judge/model", case, t, [["fact c 0"]])
    assert verdict == {"stated": [True], "supported": [True]}
    assert len(judge.prompts) == 2
    prompt = judge.prompts[0].lower()
    assert "fact c 0" in prompt and "question c?" in prompt
    for leaked in ("sonnet", "anthropic", "agent", "single", "baseline", "chat_mode"):
        assert leaked not in prompt


# --- statistics and the acceptance rule -------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_is_repeatable() -> None:
    values = [1.0] * 30 + [0.0] * 10
    lo, hi = ex.bootstrap_ci(values)
    assert lo < 0.75 < hi and 0.55 < lo and hi < 0.9
    assert ex.bootstrap_ci(values) == (lo, hi)


def test_min_difference_scales_with_noise_and_never_drops_below_one_case() -> None:
    assert ex.min_difference(0.0, 40) == pytest.approx(1 / 40)
    assert ex.min_difference(0.2, 40) == pytest.approx(1.96 * 2**0.5 * 0.2 / 40**0.5)
    assert ex.min_difference(0.2, 160) < ex.min_difference(0.2, 40)
    assert ex.per_case_noise_sd([1, 0, 1, 1], [1, 0, 1, 1]) == 0.0
    # chi-square(7) 10th percentile is 2.833: eight pairs put the SD's 90% bound at 1.57x.
    assert ex.sd_upper_bound(0.1, 8) == pytest.approx(0.1 * (7 / 2.833) ** 0.5, rel=0.01)
    assert ex.sd_upper_bound(0.1, 200) < ex.sd_upper_bound(0.1, 8)
    stored = {"per_case_sd": 0.1, "answer_sd": 0.1, "repeated": 8}
    assert ex.noise_sd_for_rule(stored) == ex.sd_upper_bound(0.1, 8)
    assert ex.noise_sd_for_rule({"per_case_sd_upper90": 0.2, **stored}) == 0.2
    assert ex.noise_sd_for_rule(None) is None


def _results(train: list[float], test: list[float], sd: float = 0.08, judge: str = "j") -> dict:
    rows = [{"id": f"tr{i}", "split": "train", "score": s} for i, s in enumerate(train)]
    rows += [{"id": f"te{i}", "split": "test", "score": s} for i, s in enumerate(test)]
    return {
        "config": {"judge_model": judge},
        "noise": {"per_case_sd_upper90": sd},
        "rows": rows,
    }


def test_compare_keeps_only_a_change_that_lifts_both_halves_beyond_the_noise() -> None:
    base = _results([0.0] * 10 + [1.0] * 10, [0.0] * 10 + [1.0] * 10)
    both = _results([0.0] * 5 + [1.0] * 15, [0.0] * 5 + [1.0] * 15)
    assert ex.compare_results(base, both)["decision"] == "KEEP"
    train_only = _results([0.0] * 5 + [1.0] * 15, [0.0] * 10 + [1.0] * 10)
    assert ex.compare_results(base, train_only)["decision"] == "REVERT"
    test_down = _results([0.0] * 5 + [1.0] * 15, [0.0] * 15 + [1.0] * 5)
    assert ex.compare_results(base, test_down)["decision"] == "REVERT"
    tiny = _results([0.0] * 9 + [1.0] * 11, [0.0] * 10 + [1.0] * 10, sd=0.16)
    assert (
        ex.compare_results(_results(*[[0.0] * 10 + [1.0] * 10] * 2, sd=0.16), tiny)["decision"]
        == "NOISE"
    )
    with pytest.raises(ValueError):
        ex.compare_results(base, _results([1.0] * 20, [1.0] * 20, judge="other"))


def test_budget_holds_a_reserve_per_unit_in_flight(tmp_path: Path) -> None:
    from portolan.training.spend import SpendMeter

    meter = SpendMeter(tmp_path / "spend.json")
    meter.record(model="m", corpus=ex.SPEND_KEY, amount_usd=2.8)
    budget = ex.Budget(meter, max_usd=1.0)
    assert budget.cap_usd == pytest.approx(ex.HARD_CAP_USD)
    assert budget.acquire(0.15)
    assert not budget.acquire(0.25)  # never fits, even with nothing in flight
    got: list[bool] = []
    waiter = threading.Thread(target=lambda: got.append(budget.acquire(0.1)))
    waiter.start()
    waiter.join(timeout=0.3)
    assert waiter.is_alive() and not got  # fits only once the first unit is done
    budget.release(0.15)
    waiter.join(timeout=5)
    assert got == [True]


# --- cases and checks on the demo graph -------------------------------------------------


@pytest.fixture()
def demo_graph(db):
    from portolan.bench.demo_search import seed_demo_graph
    from portolan.search.embed import FakeEmbedder

    seed_demo_graph(db, Path(tempfile.mkdtemp()), FakeEmbedder())
    return db


def test_demo_cases_and_the_citation_checks(demo_graph) -> None:
    conn = demo_graph
    cases = demo_cases(conn)
    by_type = {t: [c for c in cases if c.type == t] for t in CASE_TYPES}
    assert all(by_type[t] for t in CASE_TYPES), {t: len(v) for t, v in by_type.items()}
    assert sum(1 for c in cases if c.origin == "demo:multihop") == 20
    with conn.cursor() as cur:
        cur.execute("SELECT slug FROM notes")
        slugs = {r[0] for r in cur.fetchall()}
    for c in cases:
        assert all(set(g.notes) <= slugs for g in c.gold), c.id
    public_notes = {n for c in cases if c.public for g in c.gold for n in g.notes}
    for c in cases:
        if not c.public:
            assert not {n for g in c.gold for n in g.notes} & public_notes, c.id

    gold = next(c for c in cases if c.id == "d-cc-01").gold[0]
    claim_id, note = gold.claim_ids[0], gold.notes[0]
    with conn.cursor() as cur:
        cur.execute("SELECT text FROM chunks WHERE note_slug = %s ORDER BY id LIMIT 1", (note,))
        chunk = cur.fetchone()[0]
    words = [w for w in chunk.replace("\n", " ").split(" ") if w and "[[" not in w and "^" not in w]
    quote = " ".join(words[-4:])
    ok = _transcript(
        [
            {"text": "Promoted.", "citations": [{"note_slug": note, "claim_id": claim_id}]},
            {
                "text": "Quoted.",
                "citations": [{"note_slug": note, "passage_id": "p-1", "quote": quote}],
            },
        ]
    )
    assert ex.check_answer(conn, ok)["ok"], ex.check_answer(conn, ok)
    evidence = ex.evidence_for(conn, ok)
    assert evidence[0][0].startswith(gold.text + " (recorded 2026-") and "source " in evidence[0][0]
    assert evidence[1][0].startswith(f'"{quote}" (quoted from the note')
    wrong_claim = _transcript(
        [{"text": "x", "citations": [{"note_slug": "ashgrove-health", "claim_id": claim_id}]}]
    )
    assert ex.check_answer(conn, wrong_claim)["unresolved"]
    missing = _transcript([{"text": "x", "citations": [{"note_slug": "no-such-note"}]}])
    assert ex.check_answer(conn, missing)["unresolved"] == ["note no-such-note"]
    made_up = _transcript(
        [
            {
                "text": "x",
                "citations": [{"note_slug": note, "passage_id": "p-1", "quote": "he was fired"}],
            }
        ]
    )
    assert ex.check_answer(conn, made_up)["not_verbatim"]


def test_a_run_writes_test_answers_only_to_the_locked_side_and_prints_only_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from portolan.bench import chat_exam_cli as cli
    from portolan.training.spend import SpendMeter

    train, test = _case("tr1", n_gold=1), _case("te1", n_gold=1)
    cited = [{"text": "SECRET ANSWER TEXT.", "citations": [{"note_slug": "n", "claim_id": 1}]}]

    def t(cid: str) -> dict[str, Any]:
        return {
            **_transcript(cited),
            "id": cid,
            "checks": OK,
            "cost_usd": 0.01,
            "wall_s": 1.0,
            "truncated": False,
            "fallback": None,
            "evidence": [["x"]],
        }

    by_id = {"tr1": t("tr1"), "te1": t("te1")}
    verdicts = {c: {"stated": [True], "supported": [True]} for c in by_id}
    paths = ex.RunPaths(tmp_path / "run", tmp_path / "evals" / "locked" / "run")
    cfg = ex.RunConfig("agent", "m", "j", {})
    budget = ex.Budget(SpendMeter(tmp_path / "spend.json"), 1.0)
    sink = type("Sink", (), {"events": []})()
    selected = [("train", train), ("test", test)]
    cli._finish(selected, by_id, None, verdicts, {}, cfg, paths, "both", budget, sink)

    printed = capsys.readouterr().out
    readable = "".join(p.read_text() for p in paths.run_dir.iterdir())
    for text in ("Question te1?", "fact te1 0"):
        assert text not in printed and text not in readable
    assert "Question tr1?" in readable and "fact tr1 0" in readable
    assert "test passed: te1" in printed
    assert "SECRET ANSWER TEXT." in readable
    locked = ex.read_locked(paths.locked_dir / "test-transcripts.json.gz")
    assert locked["transcripts"]["te1"]["sentences"][0]["text"] == "SECRET ANSWER TEXT."
    train_t = json.loads((paths.run_dir / "train-transcripts.json").read_text())
    assert set(train_t) == {"tr1"}
    results = json.loads((paths.run_dir / "results.json").read_text())
    test_row = next(r for r in results["rows"] if r["id"] == "te1")
    assert set(test_row) <= {
        "id", "split", "type", "graph", "score", "pass", "outcome", "cost_usd", "wall_s", "infra",
    }  # fmt: skip
