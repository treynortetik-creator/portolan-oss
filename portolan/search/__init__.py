"""Embeddings, indexing, and fused search for the Portolan knowledge graph.

Spec: 01-spec/portolan-spec-v2.md sec 9. Contract:
01-spec/build-contracts.md "Embeddings, index, search". Submodules:

- `embed`  -- `Embedder` protocol, `FakeEmbedder`, `FastEmbedEmbedder`,
  `TitanEmbedder`, `get_embedder(settings)`.
- `chunk`  -- `chunk_note(body_md, max_chars=1200)` heading/paragraph
  chunking, with title+description as chunk 0.
- `index`  -- `index_note`, `reindex_all`, `index_pending`: chunk+embed a
  note into `chunks`, keyed off a `body_md` content hash
  (`note_index_state`, migration 0002_index_state.sql).
- `query`  -- `search`, `SearchHit`, `SearchResult`, `explain`: RRF-fused
  BM25 keyword + vector search with title boost, recency, `expand`, and the
  zero-result control-query diagnosis.

Reads `notes`, `chunks` and `links`; writes only `chunks` and
`note_index_state`.
"""

from portolan.search.chunk import Chunk, chunk_note
from portolan.search.embed import (
    Embedder,
    FakeEmbedder,
    FastEmbedEmbedder,
    TitanEmbedder,
    get_embedder,
)
from portolan.search.index import IndexReport, index_note, index_pending, reindex_all
from portolan.search.query import SearchHit, SearchResult, explain, search

__all__ = [
    "Chunk",
    "Embedder",
    "FakeEmbedder",
    "FastEmbedEmbedder",
    "IndexReport",
    "SearchHit",
    "SearchResult",
    "TitanEmbedder",
    "chunk_note",
    "explain",
    "get_embedder",
    "index_note",
    "index_pending",
    "reindex_all",
    "search",
]
