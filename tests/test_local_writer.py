"""A local MLX model as a v2 writer (`corpus.generator.v2.writer`
`LocalWriterBackend`, `RoutingWriterBackend`): its `local:` id passes the
writer allowlist only when `portolan.training.local_teacher` permits it, a
document written by it goes through the unchanged WP4 checks, and its calls
cost $0 with a spend line naming the model. The model is a stub that answers
like the WP4 tests' obedient writer; nothing is downloaded."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus.generator.v2.models import Split
from corpus.generator.v2.packs.schema import load_pack
from corpus.generator.v2.planner import DocKind, DocRole, plan_world
from corpus.generator.v2.renderers import render
from corpus.generator.v2.simulator import simulate_world
from corpus.generator.v2.writer import (
    ALLOWED_WRITER_MODELS,
    CORPUS_KEY,
    LocalWriterBackend,
    RoutingWriterBackend,
    RunBudget,
    WriterCallError,
    WriterConfig,
    WriterPool,
    WriterReply,
    WriterSession,
    _check_allowed,
    write_document,
)
from portolan.training.local_teacher import permitted_local_teacher_ids
from portolan.training.spend import SpendMeter
from tests.v2_writer_fakes import obedient

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / "corpus" / "generator" / "v2" / "packs" / "medical_devices.yaml"
GEMMA = "local:mlx-community/gemma-4-e4b-it-4bit"
QWEN = "local:mlx-community/Qwen3.5-4B-MLX-4bit"


class _StubClient:
    """Stands in for `MLXClient`: answers each call with `reply`."""

    def __init__(self, model: str, reply=obedient, fail: bool = False) -> None:
        self.model_id = model.split(":", 1)[1]
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []
        self.unloaded = False

    def complete(self, *, system, user, model, surface, max_tokens, temperature):  # noqa: ANN001
        self.calls.append({"model": model, "surface": surface, "temperature": temperature})
        if self.fail:
            raise RuntimeError("metal out of memory")
        text = self.reply(model, system, user, len(self.calls) - 1)
        return SimpleNamespace(text=text, input_tokens=len(system + user) // 4, output_tokens=9)

    def unload(self) -> None:
        self.unloaded = True


def test_permitted_local_models_join_the_writer_allowlist_at_zero_price() -> None:
    for teacher in permitted_local_teacher_ids():
        _check_allowed(teacher)
        info = ALLOWED_WRITER_MODELS[teacher]
        assert (info.prompt_usd_per_m, info.completion_usd_per_m) == (0.0, 0.0)
    with pytest.raises(ValueError, match="not in ALLOWED_WRITER_MODELS"):
        _check_allowed("local:someone/unlisted-4b")


def test_local_backend_answers_free_and_names_the_model_on_the_meter(tmp_path: Path) -> None:
    clients: list[_StubClient] = []

    def factory(model: str) -> _StubClient:
        clients.append(_StubClient(model, reply=lambda *_a: "draft text"))
        return clients[-1]

    meter = SpendMeter(path=tmp_path / "spend.json")
    backend = LocalWriterBackend(meter=meter, client_factory=factory)
    reply = backend.complete(model=QWEN, system="S", user="U", max_tokens=50, temperature=0.8)
    assert reply == WriterReply("draft text", 0.0, 0, 9)
    assert clients[0].calls == [
        {"model": "mlx-community/Qwen3.5-4B-MLX-4bit", "surface": "chat", "temperature": 0.8}
    ]
    backend.complete(model=GEMMA, system="S", user="U", max_tokens=50, temperature=0.8)
    assert clients[0].unloaded and len(clients) == 2  # one model in memory at a time
    snap = meter.snapshot()
    assert snap["per_model_usd"] == {QWEN: 0.0, GEMMA: 0.0}
    assert snap["per_corpus_usd"] == {CORPUS_KEY: 0.0}


def test_local_backend_turns_a_failure_into_a_writer_call_error() -> None:
    backend = LocalWriterBackend(client_factory=lambda m: _StubClient(m, fail=True))
    with pytest.raises(WriterCallError, match="metal out of memory"):
        backend.complete(model=QWEN, system="S", user="U", max_tokens=5, temperature=None)
    with pytest.raises(ValueError, match="not in ALLOWED_WRITER_MODELS"):
        backend.complete(model="local:x/y", system="S", user="U", max_tokens=5, temperature=None)


def test_routing_backend_sends_only_local_ids_to_the_local_backend() -> None:
    seen: list[str] = []

    class _Remote:
        def complete(self, *, model, **_kw):  # noqa: ANN001
            seen.append(f"remote:{model}")
            return WriterReply("r", 0.01, 1, 1)

    local = LocalWriterBackend(client_factory=lambda m: _StubClient(m, reply=lambda *_a: "l"))
    routing = RoutingWriterBackend(local, remote_factory=_Remote)
    assert (
        routing.complete(model=QWEN, system="", user="", max_tokens=1, temperature=None).text == "l"
    )
    assert seen == []
    routing.complete(model="qwen/qwen3-32b", system="", user="", max_tokens=1, temperature=None)
    assert seen == ["remote:qwen/qwen3-32b"]


def test_a_local_writer_document_passes_the_unchanged_wp4_checks(tmp_path: Path) -> None:
    pack = load_pack(PACK_PATH)
    world = simulate_world(pack, seed=42, months=8, split=Split.TRAIN)
    world_plan = plan_world(world, pack)
    plan = next(
        d
        for d in sorted(world_plan.doc_plans, key=lambda d: (d.date, d.id))
        if d.kind == DocKind.EMAIL and d.role == DocRole.EVENT and len(d.fact_ids) >= 2
    )
    config = WriterConfig(models=(QWEN,))
    meter = SpendMeter(path=tmp_path / "spend.json")
    session = WriterSession(
        backend=LocalWriterBackend(meter=meter, client_factory=lambda m: _StubClient(m)),
        pool=WriterPool(config.models, world_doc_total=300),
        budget=RunBudget(cap_usd=0.01, meter=meter),
        config=config,
    )
    doc = write_document(
        render(plan, world, world_plan, pack),
        plan,
        world,
        pack,
        decoys_by_id={d.id: d for d in world_plan.decoys},
        session=session,
    )
    assert doc.status == "written", doc.fatal
    assert doc.writer_model == QWEN
    assert doc.cost_usd == 0.0
    assert len(doc.labels) == len(plan.fact_ids)
    assert all(label.writer_model == QWEN for label in doc.labels)
    assert meter.snapshot()["per_model_usd"] == {QWEN: 0.0}
