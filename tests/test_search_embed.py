"""Tests for portolan/search/embed.py.

No network access, per the build instructions: `FakeEmbedder` covers every
ranking-relevant test; `TitanEmbedder` and `FastEmbedEmbedder` are only
tested for construction/factory wiring (never `.embed()`, which would hit a
model download or AWS). The one exception is
`test_fastembed_real_model_has_1024_dims`, opt-in via
`PORTOLAN_REAL_EMBED=1`.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest

from portolan.search.embed import (
    FakeEmbedder,
    FastEmbedEmbedder,
    TitanEmbedder,
    get_embedder,
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def test_fake_embedder_dim_is_1024():
    fe = FakeEmbedder()
    assert fe.dim == 1024
    [vec] = fe.embed(["hello world"])
    assert len(vec) == 1024


def test_fake_embedder_vectors_are_l2_normalized():
    fe = FakeEmbedder()
    [vec] = fe.embed(["Ashgrove Medical Center had a late Tidewater delivery."])
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_fake_embedder_is_deterministic():
    fe = FakeEmbedder()
    a = fe.embed(["late deliveries in the Midwest"])[0]
    b = fe.embed(["late deliveries in the Midwest"])[0]
    assert a == b


def test_fake_embedder_empty_text_is_the_zero_vector():
    fe = FakeEmbedder()
    [vec] = fe.embed([""])
    assert vec == [0.0] * 1024


def test_fake_embedder_shared_vocabulary_is_closer_than_unrelated_text():
    fe = FakeEmbedder()
    a, b, c = fe.embed(
        [
            "Ashgrove Medical Center reported late Tidewater deliveries this month.",
            "Late Tidewater deliveries were reported by Ashgrove Medical Center again.",
            "The quarterly board meeting scheduled a review of unrelated topics.",
        ]
    )
    assert _cosine(a, b) > _cosine(a, c)


def test_fake_embedder_disjoint_vocabulary_is_exactly_orthogonal():
    fe = FakeEmbedder()
    a, b = fe.embed(["zephyr crystalline saxophone", "portal password change request"])
    assert _cosine(a, b) == pytest.approx(0.0, abs=1e-9)


def test_titan_embedder_constructs_without_aws_credentials_or_network():
    # Construction-only: importing boto3 must succeed, but no client or
    # network call happens until embed() runs (never exercised here).
    embedder = TitanEmbedder(region="us-east-1")
    assert embedder.dim == 1024
    assert embedder.model_id == "amazon.titan-embed-text-v2:0"
    assert embedder._client is None


def test_fastembed_embedder_constructs_without_loading_the_model():
    embedder = FastEmbedEmbedder()
    assert embedder.dim == 1024
    assert embedder.model_name == "BAAI/bge-large-en-v1.5"
    assert embedder._model is None


def test_get_embedder_factory_fake():
    settings = SimpleNamespace(embed_provider="fake", aws_region="us-east-1")
    embedder = get_embedder(settings)
    assert isinstance(embedder, FakeEmbedder)


def test_get_embedder_factory_titan():
    settings = SimpleNamespace(embed_provider="titan", aws_region="us-west-2")
    embedder = get_embedder(settings)
    assert isinstance(embedder, TitanEmbedder)
    assert embedder.region == "us-west-2"


def test_get_embedder_factory_fastembed():
    settings = SimpleNamespace(embed_provider="fastembed", aws_region="us-east-1")
    embedder = get_embedder(settings)
    assert isinstance(embedder, FastEmbedEmbedder)


def test_get_embedder_factory_rejects_unknown_provider():
    settings = SimpleNamespace(embed_provider="not-a-real-provider", aws_region="us-east-1")
    with pytest.raises(ValueError):
        get_embedder(settings)


@pytest.mark.skipif(
    os.environ.get("PORTOLAN_REAL_EMBED") != "1", reason="set PORTOLAN_REAL_EMBED=1 to run"
)
def test_fastembed_real_model_has_1024_dims():
    embedder = FastEmbedEmbedder()
    [vec] = embedder.embed(["late deliveries in the Midwest"])
    assert len(vec) == 1024
    assert embedder.dim == 1024


# --- lazy model load is thread-safe (security review 2026-09-19, S4) ----
#
# `_LazyStateProxy` (`portolan.web.app`) shares one `FastEmbedEmbedder`
# instance between the web app's own routes and every MCP tool call so
# the ~1.2GB ONNX model loads once per process -- but MCP tools run via
# the SDK's own `anyio.to_thread.run_sync` while FastAPI's sync routes
# run in their own threadpool, so the FIRST concurrent burst from both
# families races on the same instance's lazy `_get_model()`. Reproduced
# without the real (slow, network-fetching) model: monkeypatch
# `fastembed.TextEmbedding` with a fake that counts constructions and
# widens the race window with a short sleep before returning.


def test_fastembed_lazy_model_load_builds_the_model_exactly_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import fastembed

    construction_count = 0
    count_lock = threading.Lock()
    start_barrier = threading.Barrier(8)

    class _FakeVector:
        def tolist(self) -> list[float]:
            return [0.0]

    class _FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            nonlocal construction_count
            del model_name
            with count_lock:
                construction_count += 1
            # Widen the window during which a second, unsynchronized
            # caller would still see `self._model is None` -- without the
            # fix, this is exactly the gap several threads race through.
            time.sleep(0.05)

        def embed(self, texts: list[str]) -> list[_FakeVector]:
            return [_FakeVector() for _ in texts]

    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)

    embedder = FastEmbedEmbedder()

    def _call() -> None:
        # All 8 threads reach this line before any of them proceeds into
        # `embed()`/`_get_model()`, maximizing overlap at the lazy-load
        # check -- the race this test exists to catch.
        start_barrier.wait(timeout=5)
        embedder.embed(["hello"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_call) for _ in range(8)]
        for future in futures:
            future.result(timeout=5)

    assert construction_count == 1
