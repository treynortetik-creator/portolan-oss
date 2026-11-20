"""Within-set deduplication for the fine-tuning set builder
(`scripts/data/build_training_set.py`, review
`01-spec/reviews/review-training-data-2026-09-22.md`).

Two tools, both pure functions over strings:

- `normalize_for_dedup` and `pair_key`/`quote_key`: normalized-text keys
  for exact repeats (the same claim and quote in two records, the same
  quote in twenty SaaS calls).
- `similarity_groups`: near-duplicate documents by 5-word-shingle Jaccard,
  found with MinHash (128 hash functions) and LSH banding (32 bands of 4
  rows), then confirmed by the exact Jaccard of the shingle sets, so the
  returned pairs never depend on the estimate alone. 32 x 4 puts the LSH
  S-curve midpoint near Jaccard 0.42, so pairs at the 0.5 split-grouping
  line are found with probability about 0.87 and pairs at the 0.8 dedup
  line with probability above 0.99.

This module compares training records with each other only. Comparing
training records with the exam corpora (Halvard, EnterpriseRAG-Bench)
is the contamination checker's job (world generator v2 spec, section 6,
WP8), not this module's.

No third-party dependency: the hash functions are salted BLAKE2b digests
from the standard library, so a signature is identical on every machine
and every run.
"""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "NUM_PERM",
    "LSH_BANDS",
    "LSH_ROWS",
    "SHINGLE_WORDS",
    "normalize_for_dedup",
    "pair_key",
    "quote_key",
    "shingle_hashes",
    "minhash_signature",
    "jaccard",
    "SimilarityGroups",
    "similarity_groups",
]

SHINGLE_WORDS = 5
NUM_PERM = 128
LSH_BANDS = 32
LSH_ROWS = 4
_DIGEST_BYTES = 64  # one BLAKE2b digest yields 16 unsigned 32-bit values
_VALUES_PER_DIGEST = _DIGEST_BYTES // 4
_SALTS: tuple[bytes, ...] = tuple(
    f"portolan-mh-{i:02d}".encode()[:16] for i in range(NUM_PERM // _VALUES_PER_DIGEST)
)
_UNPACK = struct.Struct(f"<{_VALUES_PER_DIGEST}I")

_GLYPHS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})
# Keep letters, digits and "$" (so "$299" and "299" stay different keys);
# everything else is a separator.
_NON_KEY_CHARS = re.compile(r"[^0-9a-z$]+")


def normalize_for_dedup(text: str | None) -> str:
    """NFKC, glyph fold, casefold, then every run of characters other
    than letters, digits and "$" becomes one space. "It’s $299/month!"
    and "it's $299 / month" normalize to the same string."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).translate(_GLYPHS).casefold()
    return _NON_KEY_CHARS.sub(" ", folded).strip()


def pair_key(claim: str, quote: str | None) -> tuple[str, str]:
    """Key for "the same fact twice": normalized claim plus normalized quote."""
    return normalize_for_dedup(claim), normalize_for_dedup(quote)


def quote_key(quote: str | None) -> str:
    """Key for "the same evidence span twice", regardless of the claim."""
    return normalize_for_dedup(quote)


def shingle_hashes(text: str, k: int = SHINGLE_WORDS) -> frozenset[int]:
    """64-bit hashes of every run of `k` consecutive normalized words. A
    text shorter than `k` words is one shingle; an empty text has none."""
    words = normalize_for_dedup(text).split()
    if not words:
        return frozenset()
    if len(words) < k:
        grams = [" ".join(words)]
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    return frozenset(
        int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")
        for g in grams
    )


def minhash_signature(shingles: frozenset[int]) -> tuple[int, ...] | None:
    """128 minima, one per salted hash function; `None` for no shingles."""
    if not shingles:
        return None
    rows = []
    for h in shingles:
        raw = h.to_bytes(8, "little")
        values: list[int] = []
        for salt in _SALTS:
            values.extend(
                _UNPACK.unpack(hashlib.blake2b(raw, digest_size=_DIGEST_BYTES, salt=salt).digest())
            )
        rows.append(values)
    return tuple(min(column) for column in zip(*rows, strict=True))


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return [sorted(g) for g in out.values()]


@dataclass(frozen=True)
class SimilarityGroups:
    """`duplicate_groups`: groups of indices whose texts are the same
    after normalization or have shingle Jaccard >= the dedup threshold
    (single-member groups included). `split_groups`: coarser groups at the
    lower split threshold, used to keep near-siblings in one split.
    `pairs`: every confirmed `(i, j, jaccard)` at or above the lower
    threshold, `i < j`, for reporting."""

    duplicate_groups: list[list[int]]
    split_groups: list[list[int]]
    pairs: list[tuple[int, int, float]]


def similarity_groups(
    texts: Sequence[str],
    *,
    dedup_threshold: float = 0.8,
    split_threshold: float = 0.5,
) -> SimilarityGroups:
    """Group `texts` twice: at `dedup_threshold` (near-duplicates, to keep
    one of) and at `split_threshold` (near-siblings, to keep on one side
    of the train/validation split). Exact normalized-text matches always
    join both. Candidates come from LSH over MinHash signatures; each
    candidate pair is confirmed by exact Jaccard before it counts."""
    if not 0.0 < split_threshold <= dedup_threshold <= 1.0:
        raise ValueError("need 0 < split_threshold <= dedup_threshold <= 1")
    n = len(texts)
    dup = _UnionFind(n)
    split = _UnionFind(n)

    by_exact: dict[str, int] = {}
    for i, t in enumerate(texts):
        key = hashlib.sha256(normalize_for_dedup(t).encode("utf-8")).hexdigest()
        if key in by_exact:
            dup.union(by_exact[key], i)
            split.union(by_exact[key], i)
        else:
            by_exact[key] = i

    shingles = [shingle_hashes(t) for t in texts]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i, s in enumerate(shingles):
        sig = minhash_signature(s)
        if sig is None:
            continue
        for band in range(LSH_BANDS):
            buckets[(band, sig[band * LSH_ROWS : (band + 1) * LSH_ROWS])].append(i)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_idx in range(len(members)):
            for b_idx in range(a_idx + 1, len(members)):
                a, b = members[a_idx], members[b_idx]
                candidates.add((a, b) if a < b else (b, a))

    pairs: list[tuple[int, int, float]] = []
    for a, b in sorted(candidates):
        score = jaccard(shingles[a], shingles[b])
        if score >= split_threshold:
            pairs.append((a, b, score))
            split.union(a, b)
            if score >= dedup_threshold:
                dup.union(a, b)

    return SimilarityGroups(duplicate_groups=dup.groups(), split_groups=split.groups(), pairs=pairs)
