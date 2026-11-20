"""Embedding backends for Portolan's search index.

Spec: 01-spec/portolan-spec-v2.md sec 9 (vector similarity over
`chunks.embedding`, dim 1024).

Three implementations, one Protocol:
  - `FakeEmbedder` -- deterministic, no model, no network; for tests.
  - `FastEmbedEmbedder` -- local ONNX model via `fastembed`.
  - `TitanEmbedder` -- Amazon Bedrock Titan Text Embeddings V2, when
    `Settings.embed_provider == "titan"`.

`Settings` is imported for type checking only, so importing this module
loads no settings.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from portolan.config import Settings


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-dimension vectors."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------
# FakeEmbedder -- deterministic hash-based bag-of-words vectors.
# ---------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords are dropped before hashing: a word in nearly every document would
# amplify any accidental bucket collision into spurious similarity.
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "it",
    "its",
    "this",
    "that",
    "with",
    "by",
    "at",
    "as",
    "be",
    "been",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "not",
    "no",
    "so",
    "we",
    "they",
    "he",
    "she",
    "you",
    "i",
}


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class FakeEmbedder:
    """Deterministic embedder for tests: no model download, no network.

    Each text is tokenized (lowercased `[a-z0-9]+` runs, stopwords dropped)
    and hashed into a 1024-dim vector with the signed hashing trick
    (Weinberger et al. 2009): token `t` adds +1 or -1 (one more hash bit) to
    bucket `sha256(t) % dim`, then the vector is L2-normalized. The sign makes
    an accidental collision between two different tokens net zero in
    expectation, while a shared token adds constructively; texts with no
    shared tokens score near 0 (not exactly 0: collisions still happen).
    Literal token overlap only, no meaning.
    """

    dim = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = (t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return _l2_normalize(vec)


# ---------------------------------------------------------------------
# FastEmbedEmbedder -- local ONNX model, BAAI/bge-large-en-v1.5, 1024-dim.
# ---------------------------------------------------------------------


class FastEmbedEmbedder:
    """Local embedding via `fastembed`'s ONNX runtime. The large model is
    downloaded/loaded on the first `embed()`, not at construction, so
    `get_embedder` at process start never blocks.

    No `query: `/`passage: ` prefixes: fastembed's model description for
    `BAAI/bge-large-en-v1.5` says "Prefixes for queries/documents: not so
    necessary". A model that needs asymmetric prefixes would need a second
    Protocol method (e.g. `embed_query`).
    """

    dim = 1024
    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self._model: Any = None
        # One instance can be shared across threads; the lock makes concurrent
        # first callers load the model once, not once each.
        self._model_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                # Double-checked: a thread that lost the race finds the model built.
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        # fastembed batches internally; `.embed()` is a lazy generator.
        return [vec.tolist() for vec in model.embed(list(texts))]


# ---------------------------------------------------------------------
# TitanEmbedder -- Amazon Bedrock Titan Text Embeddings V2.
# ---------------------------------------------------------------------


class TitanEmbedder:
    """Amazon Bedrock `amazon.titan-embed-text-v2:0` at 1024 dims (sent as
    `dimensions`, normalized). Constructible without AWS credentials or
    network; the `boto3` client is created on the first `embed()`. Tests
    cover construction only.

    Titan's `invoke_model` takes one `inputText` per call, so `embed()`
    makes one Bedrock call per text.
    """

    dim = 1024
    MODEL_ID = "amazon.titan-embed-text-v2:0"

    def __init__(self, region: str = "us-east-1", model_id: str | None = None) -> None:
        # Imported here so the module imports without boto3; the client is
        # built in _get_client().
        import boto3

        self._boto3 = boto3
        self.region = region
        self.model_id = model_id or self.MODEL_ID
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json

        client = self._get_client()
        out: list[list[float]] = []
        for text in texts:
            response = client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True}),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            out.append(payload["embedding"])
        return out


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def get_embedder(settings: Settings) -> Embedder:
    """Build the configured `Embedder` from `settings.embed_provider`
    (`"fastembed"`, `"titan"` or `"fake"`); raises `ValueError` otherwise.
    """
    provider = settings.embed_provider
    if provider == "fake":
        return FakeEmbedder()
    if provider == "fastembed":
        return FastEmbedEmbedder()
    if provider == "titan":
        return TitanEmbedder(region=getattr(settings, "aws_region", "us-east-1"))
    raise ValueError(f"Unknown embed_provider: {provider!r}")
