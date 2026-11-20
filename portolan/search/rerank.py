"""Cross-encoder reranking of `search()`'s fused candidates, run locally.

A cross-encoder reads the question and one candidate passage together and
scores how well the passage answers it: slower than the two retrieval legs,
but it sees both texts at once. `search()` reranks its fused top
`search_rerank_pool` notes (`query._rerank_passages`) when the setting
`search_rerank` names a model (01-spec/bench-erb-search-2026-09-22.md,
"Reranker and query routing").

Four models, all Apache-2.0, all ONNX run by onnxruntime on CPU (no
torch), each pinned to a Hugging Face commit:

- `bge-reranker-v2-m3`: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa, 568M), int8,
  as exported by onnx-community. A sequence-pair classifier: one logit.
- `qwen3-reranker-0.6b`: Qwen/Qwen3-Reranker-0.6B (a causal LM that answers
  "yes" or "no"), int8, as exported by n24q02m with the output cut to the
  two logits `[no, yes]` at the last position. Prompt as in the model card.
- `ms-marco-minilm-l6` (cross-encoder/ms-marco-MiniLM-L6-v2, 23M) and
  `jina-reranker-v1-turbo-en` (38M): small pair classifiers from their
  authors' own ONNX files, fp32; the sizes a CPU can run per query.

Model files are cached under `FASTEMBED_CACHE_PATH` (fastembed's default
directory when unset), the directory the embedding model already uses, so a
deployment's volume keeps them across restarts. `Settings` is imported for
type checking only, so importing this module loads no settings.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from portolan.config import Settings


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs; higher is more relevant, in [0, 1]."""

    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankModel:
    setting: str  # its `search_rerank` value
    upstream: str  # the model the ONNX file was exported from
    license: str
    repo: str
    revision: str  # a commit sha: the files never change under a deployment
    onnx_file: str
    kind: Literal["pair_classifier", "qwen3_yes_no"]
    max_length: int = 512
    # Pairs per onnxruntime call, padded on the right. 1 for the int8 files:
    # dynamic quantization scales activations over the whole batch, so a
    # pair's score would change with its batch (by up to 0.02 for bge).
    batch: int = 1

    @property
    def files(self) -> tuple[str, ...]:
        return (self.onnx_file, "tokenizer.json", "tokenizer_config.json", "config.json")


RERANK_MODELS: dict[str, RerankModel] = {
    m.setting: m
    for m in (
        RerankModel(
            setting="bge-reranker-v2-m3",
            upstream="BAAI/bge-reranker-v2-m3",
            license="apache-2.0",
            repo="onnx-community/bge-reranker-v2-m3-ONNX",
            revision="6f5ff65298512715a1e669753bc754d2bc8f367b",
            onnx_file="onnx/model_int8.onnx",
            kind="pair_classifier",
        ),
        RerankModel(
            setting="qwen3-reranker-0.6b",
            upstream="Qwen/Qwen3-Reranker-0.6B",
            license="apache-2.0",
            repo="n24q02m/Qwen3-Reranker-0.6B-ONNX",
            revision="342a8a54de23bba03eb8cce5f5bef503edf18315",
            onnx_file="onnx/model_yesno_quantized.onnx",
            kind="qwen3_yes_no",
        ),
        RerankModel(
            setting="ms-marco-minilm-l6",
            upstream="cross-encoder/ms-marco-MiniLM-L6-v2",
            license="apache-2.0",
            repo="cross-encoder/ms-marco-MiniLM-L6-v2",
            revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
            onnx_file="onnx/model.onnx",
            kind="pair_classifier",
            batch=8,
        ),
        RerankModel(
            setting="jina-reranker-v1-turbo-en",
            upstream="jinaai/jina-reranker-v1-turbo-en",
            license="apache-2.0",
            repo="jinaai/jina-reranker-v1-turbo-en",
            revision="b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
            onnx_file="onnx/model.onnx",
            kind="pair_classifier",
            batch=8,
        ),
    )
}

# Qwen/Qwen3-Reranker-0.6B model card, "Usage": system prompt, the default
# instruction, and the empty think block the assistant turn starts with.
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
# Chat-control tokens in a passage would end the user turn early.
_QWEN3_CONTROL = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")


def default_cache_dir(settings: Settings | None = None) -> Path:
    """`FASTEMBED_CACHE_PATH`, else fastembed's own default (`<tmp>/fastembed_cache`)."""
    configured = getattr(settings, "fastembed_cache_path", None) if settings else None
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "fastembed_cache"


def model_dir(model: RerankModel, cache_dir: Path, *, allow_download: bool = True) -> Path:
    """The pinned snapshot's directory under `cache_dir`, downloading only the
    files `model.files` names when they are not cached yet."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "revision": model.revision,
        "allow_patterns": list(model.files),
        "cache_dir": str(cache_dir),
    }
    try:
        path = snapshot_download(model.repo, local_files_only=True, **kwargs)
    except Exception:
        if not allow_download:
            raise FileNotFoundError(
                f"{model.repo}@{model.revision[:12]} is not cached under {cache_dir}"
            ) from None
        path = snapshot_download(model.repo, **kwargs)
    root = Path(path)
    missing = [f for f in model.files if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{model.repo}: missing {', '.join(missing)} under {root}")
    return root


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class OnnxReranker:
    """One of `RERANK_MODELS` on onnxruntime's CPU provider. The model loads on
    the first `score()`, not at construction; the lock makes concurrent first
    callers load it once.

    Pair classifiers run `model.batch` pairs per call, sorted by length and
    padded on the right (exact for fp32 files). Qwen3 pairs run one at a
    time: its export takes the last position's logits and numbers positions
    0..n-1 itself, so padding would change a pair's score with its batch.
    """

    def __init__(
        self,
        model: RerankModel,
        *,
        cache_dir: Path | None = None,
        threads: int | None = None,
        allow_download: bool = True,
    ) -> None:
        self.model = model
        self.name = model.setting
        self.cache_dir = cache_dir or default_cache_dir()
        self.threads = threads
        self.allow_download = allow_download
        self._session: Any = None
        self._tokenizer: Any = None
        self._affixes: tuple[list[int], list[int]] = ([], [])
        self._pad_id = 0
        self._type_ids = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from tokenizers import Tokenizer

            root = model_dir(self.model, self.cache_dir, allow_download=self.allow_download)
            options = ort.SessionOptions()
            if self.threads:
                options.intra_op_num_threads = self.threads
            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            tokenizer.no_padding()
            if self.model.kind == "pair_classifier":
                tokenizer.enable_truncation(
                    max_length=self.model.max_length, strategy="longest_first"
                )
                pad = json.loads((root / "tokenizer_config.json").read_text())["pad_token"]
                pad = pad["content"] if isinstance(pad, dict) else pad
                self._pad_id = tokenizer.token_to_id(pad)
            else:
                tokenizer.no_truncation()
                prefix = tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False).ids
                suffix = tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False).ids
                self._affixes = (prefix, suffix)
            self._tokenizer = tokenizer
            session = ort.InferenceSession(
                str(root / self.model.onnx_file),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._type_ids = "token_type_ids" in {i.name for i in session.get_inputs()}
            self._session = session

    def token_counts(self, query: str, passages: list[str]) -> list[int]:
        """Tokens the model reads per pair, after truncation: its cost."""
        self._load()
        if self.model.kind == "pair_classifier":
            return [len(self._tokenizer.encode(query, p).ids) for p in passages]
        return [len(self._qwen3_ids(query, p)) for p in passages]

    def _qwen3_ids(self, query: str, passage: str) -> list[int]:
        prefix, suffix = self._affixes
        body = (
            f"<Instruct>: {_QWEN3_INSTRUCTION}\n<Query>: {_QWEN3_CONTROL.sub(' ', query)}\n"
            f"<Document>: {_QWEN3_CONTROL.sub(' ', passage)}"
        )
        ids = self._tokenizer.encode(body, add_special_tokens=False).ids
        return prefix + ids[: self.model.max_length - len(prefix) - len(suffix)] + suffix

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        self._load()
        import numpy as np

        if self.model.kind == "qwen3_yes_no":
            out: list[float] = []
            for passage in passages:
                ids = np.array([self._qwen3_ids(query, passage)], dtype=np.int64)
                [logits] = self._session.run(
                    ["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)}
                )
                out.append(_sigmoid(float(logits[0][1]) - float(logits[0][0])))  # P(yes)
            return out

        encodings = [self._tokenizer.encode(query, p) for p in passages]
        order = sorted(range(len(encodings)), key=lambda i: len(encodings[i].ids))
        scores = [0.0] * len(encodings)
        step = self.model.batch
        for start in range(0, len(order), step):
            batch = [encodings[i] for i in order[start : start + step]]
            width = max(len(e.ids) for e in batch)
            ids = np.full((len(batch), width), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            types = np.zeros((len(batch), width), dtype=np.int64)
            for row, e in enumerate(batch):
                ids[row, : len(e.ids)] = e.ids
                mask[row, : len(e.ids)] = 1
                types[row, : len(e.ids)] = e.type_ids
            feed = {"input_ids": ids, "attention_mask": mask}
            if self._type_ids:
                feed["token_type_ids"] = types
            [logits] = self._session.run(["logits"], feed)
            for row, i in enumerate(order[start : start + step]):
                scores[i] = _sigmoid(float(logits[row][0]))
        return scores


_WORD_RE = re.compile(r"[a-z0-9]+")


class FakeReranker:
    """Deterministic stand-in for tests: the share of the query's words (three
    letters or more) found in the passage. No model, no network."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        terms = {w for w in _WORD_RE.findall(query.lower()) if len(w) > 2}
        if not terms:
            return [0.0 for _ in passages]
        return [len(terms & set(_WORD_RE.findall(p.lower()))) / len(terms) for p in passages]


_instances: dict[tuple[str, Path], OnnxReranker] = {}
_instances_lock = threading.Lock()


def get_reranker(settings: Settings) -> Reranker | None:
    """The configured reranker, one instance per process and model (the ONNX
    session is large and thread-safe); None when `search_rerank` is "off"."""
    key = settings.search_rerank
    if key == "off":
        return None
    if key not in RERANK_MODELS:
        raise ValueError(f"Unknown search_rerank: {key!r}")
    cache_dir = default_cache_dir(settings)
    with _instances_lock:
        instance = _instances.get((key, cache_dir))
        if instance is None:
            instance = OnnxReranker(RERANK_MODELS[key], cache_dir=cache_dir)
            _instances[(key, cache_dir)] = instance
    return instance
