"""search()'s rerank stage and query routing (`portolan/search/rerank.py`,
`portolan/search/query.py` `_rank`, `_route_weights`, `_fts_leg`'s coverage).

No model is downloaded here: the search tests use stand-in rerankers, and the
one test that runs the real ONNX models skips unless they are already in the
local cache (`allow_download=False`; the network guard would refuse a
download anyway).
"""

from __future__ import annotations

from datetime import date

import pydantic
import pytest

from portolan import config
from portolan.config import Settings
from portolan.merge import gate as gate_module
from portolan.search import query as q
from portolan.search import rerank
from portolan.search.embed import FakeEmbedder
from portolan.search.index import reindex_all
from portolan.search.query import search

NOTES = [
    ("pump-alarm-alpha", "Pump alarm alpha", "The alpha pump alarm sounded twice at night."),
    ("pump-alarm-bravo", "Pump alarm bravo", "Bravo ward reported one pump alarm."),
    ("pump-alarm-charlie", "Pump alarm charlie", "Charlie saw a pump alarm and a recall notice."),
    ("invoice-dispute", "Invoice dispute", "A duplicate charge on the March invoice."),
    ("recall-notice", "Recall notice", "Corvane pulled its pump from the market."),
    ("renewal-date", "Renewal date", "The service contract renews in October."),
]


@pytest.fixture()
def graph(db):
    with db.cursor() as cur:
        for slug, title, body in NOTES:
            cur.execute(
                "INSERT INTO notes (slug, type, title, body_md, updated) "
                "VALUES (%s, 'signal', %s, %s, %s)",
                (slug, title, body, date(2026, 9, 1)),
            )
    reindex_all(db, FakeEmbedder())
    return db


class _Fixed:
    """Scores a passage by the first key of `by_word` found in it."""

    name = "fixed"

    def __init__(self, by_word: dict[str, float]) -> None:
        self.by_word = by_word
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        out = []
        for p in passages:
            low = p.lower()
            out.append(next((v for w, v in self.by_word.items() if w in low), 0.01))
        return out


def test_the_reranker_orders_the_fused_pool(graph):
    base = search(graph, FakeEmbedder(), "pump alarm", k=3, reranker=False)
    assert base.hits[0].note_slug != "pump-alarm-charlie"

    everything = search(graph, FakeEmbedder(), "pump alarm", k=10, reranker=False).hits
    assert len(everything) == 4

    fixed = _Fixed({"charlie": 0.9, "bravo": 0.5})
    result = search(graph, FakeEmbedder(), "pump alarm", k=3, reranker=fixed, rerank_pool=3)

    assert [h.note_slug for h in result.hits[:2]] == ["pump-alarm-charlie", "pump-alarm-bravo"]
    # relevance x recency (the notes are weeks old, so just under 1)
    assert 0.85 * 0.9 < result.hits[0].score < 0.9
    [(query, passages)] = fixed.calls
    assert query == "pump alarm"
    assert len(passages) == 3  # the pool, in fused order
    # the title chunk, then the body (the cited chunk here is the title chunk)
    assert "Pump alarm charlie\nCharlie saw a pump alarm and a recall notice." in passages

    fixed = _Fixed({})
    search(graph, FakeEmbedder(), "pump alarm", k=3, reranker=fixed, rerank_pool=50)
    assert len(fixed.calls[0][1]) == 4  # every fused candidate, never more


def test_the_pool_is_never_smaller_than_k(graph):
    fixed = _Fixed({})
    search(graph, FakeEmbedder(), "pump alarm", k=4, reranker=fixed, rerank_pool=2)
    assert len(fixed.calls[0][1]) == 4


def test_no_reranker_by_default(graph, monkeypatch):
    assert config.settings.search_rerank == "off"
    calls = []
    real = rerank.get_reranker

    def _spy(settings):
        calls.append(settings.search_rerank)
        return real(settings)

    monkeypatch.setattr(rerank, "get_reranker", _spy)
    default = search(graph, FakeEmbedder(), "pump alarm recall", k=4)
    off = search(graph, FakeEmbedder(), "pump alarm recall", k=4, reranker=False)
    assert calls == ["off"]
    assert [h.note_slug for h in default.hits] == [h.note_slug for h in off.hits]
    assert [h.score for h in default.hits] == [h.score for h in off.hits]


def test_the_configured_model_is_used_when_the_setting_names_one(graph, monkeypatch):
    fixed = _Fixed({"charlie": 0.99})
    monkeypatch.setattr(rerank, "get_reranker", lambda settings: fixed)
    result = search(graph, FakeEmbedder(), "pump alarm", k=3)
    assert fixed.calls
    assert result.hits[0].note_slug == "pump-alarm-charlie"


def test_get_reranker_is_none_when_off_and_one_instance_per_model(tmp_path):
    off = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert rerank.get_reranker(off) is None
    on = Settings(
        database_url="postgresql://localhost/x",
        search_rerank="ms-marco-minilm-l6",
        fastembed_cache_path=str(tmp_path),
        _env_file=None,
    )
    first = rerank.get_reranker(on)
    assert isinstance(first, rerank.OnnxReranker)
    assert first.cache_dir == tmp_path
    assert rerank.get_reranker(on) is first  # constructed only; nothing loads until score()


def test_settings_for_the_reranker_and_routing(monkeypatch):
    s = Settings(database_url="postgresql://localhost/x", _env_file=None)
    assert (s.search_rerank, s.search_rerank_pool, s.search_fusion_routing) == ("off", 30, True)
    monkeypatch.setenv("PORTOLAN_SEARCH_RERANK", "bge-reranker-v2-m3")
    monkeypatch.setenv("PORTOLAN_SEARCH_RERANK_POOL", "50")
    monkeypatch.setenv("PORTOLAN_SEARCH_FUSION_ROUTING", "false")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/data/fastembed")
    s = config.get_settings()
    assert (s.search_rerank, s.search_rerank_pool, s.search_fusion_routing) == (
        "bge-reranker-v2-m3",
        50,
        False,
    )
    assert rerank.default_cache_dir(s).as_posix() == "/data/fastembed"
    assert set(rerank.RERANK_MODELS) == set(
        Settings.model_fields["search_rerank"].annotation.__args__
    ) - {"off"}  # type: ignore[union-attr]
    with pytest.raises(pydantic.ValidationError):
        Settings(database_url="postgresql://localhost/x", search_rerank="bogus", _env_file=None)  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        Settings(database_url="postgresql://localhost/x", search_rerank_pool=0, _env_file=None)


def test_the_merge_gate_never_reranks(monkeypatch):
    seen = {}

    def _search(conn, embedder, query, **kwargs):
        seen.update(kwargs)
        return q.SearchResult(hits=[], control_ok=True, query_terms_unmatched=[])

    monkeypatch.setattr(gate_module, "search", _search)
    gate_module.gate(FakeEmbedder(), "a claim", [], conn=object(), entity_slugs=["acct"])
    assert seen["reranker"] is False


def test_coverage_is_the_best_notes_share_of_the_query_idf(graph):
    _r, _c, full = q._fts_leg(graph, "Corvane pulled its pump", 10)
    assert full == pytest.approx(1.0)
    _r, _c, part = q._fts_leg(graph, "Corvane pump zanzibar", 10)
    assert 0.0 < part < 1.0
    _r, _c, none = q._fts_leg(graph, "zanzibar quokka", 10)
    assert none == 0.0


def test_routing_picks_the_profile_by_coverage(graph, monkeypatch):
    monkeypatch.setattr(q, "ROUTE_COVERAGE_THRESHOLD", 0.5)
    monkeypatch.setattr(q, "ROUTE_KEYWORD_WEIGHTS", (1.0, 0.2))
    monkeypatch.setattr(q, "ROUTE_SEMANTIC_WEIGHTS", (1.0, 3.0))
    assert q._route_weights(0.5) == (1.0, 0.2)
    assert q._route_weights(0.49) == (1.0, 3.0)

    def legs(coverage: float) -> q.Legs:
        return q.Legs(
            query="x",
            fts_rank=["invoice-dispute", "renewal-date"],
            fts_chunks={},
            vector_rank=["recall-notice", "pump-alarm-alpha"],
            vector_chunks={},
            coverage=coverage,
        )

    def order(coverage: float, routing: bool) -> list[str]:
        result = q._rank(graph, legs(coverage), k=4, fusion_routing=routing, reranker=False)
        return [h.note_slug for h in result.hits]

    assert order(0.9, True)[0] == "invoice-dispute"  # keyword-led
    assert order(0.1, True)[0] == "recall-notice"  # vector weight 3
    # routing off: the fixed equal weights, rank ties broken by slug
    assert order(0.1, False)[:2] == ["invoice-dispute", "recall-notice"]


def test_search_reads_the_routing_setting(graph, monkeypatch):
    seen = []
    monkeypatch.setattr(q, "_route_weights", lambda c: seen.append(c) or (1.0, 1.0))
    search(graph, FakeEmbedder(), "pump alarm", k=3, reranker=False, fusion_routing=False)
    assert seen == []
    search(graph, FakeEmbedder(), "pump alarm", k=3, reranker=False)
    assert len(seen) == 1


def test_fake_reranker_scores_word_overlap():
    fake = rerank.FakeReranker()
    assert fake.score("pump recall", ["the pump recall", "a pump", "nothing"]) == [1.0, 0.5, 0.0]


def test_model_dir_never_downloads_when_told_not_to(tmp_path):
    with pytest.raises(FileNotFoundError, match="not cached"):
        rerank.model_dir(rerank.RERANK_MODELS["ms-marco-minilm-l6"], tmp_path, allow_download=False)


def _cached(key: str) -> bool:
    try:
        rerank.model_dir(
            rerank.RERANK_MODELS[key], rerank.default_cache_dir(), allow_download=False
        )
    except Exception:
        return False
    return True


@pytest.mark.parametrize("key", sorted(rerank.RERANK_MODELS))
def test_real_model_prefers_the_answering_passage(key):
    if not _cached(key):
        pytest.skip(f"{key} is not in the local model cache (no download in tests)")
    model = rerank.OnnxReranker(rerank.RERANK_MODELS[key], allow_download=False)
    question = "Which competitor had to recall one of its pumps?"
    answer = (
        "Corvane Systems. The biomed lead said Corvane had to pull its infusion pump from "
        "the market after a recall notice in July."
    )
    other = "Sam promised a written report on the late consumables deliveries by month end."
    passages = [other, answer, other + " " + other]
    scores = model.score(question, passages)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[1] > 0.5 > max(scores[0], scores[2])
    # batching (fp32 files) or not (int8, Qwen), a pair's score is its own
    assert model.score(question, [answer])[0] == pytest.approx(scores[1], abs=1e-5)
