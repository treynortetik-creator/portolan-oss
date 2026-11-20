"""Build the contamination reference signatures under `var/contamination/`
(`01-spec/world-generator-v2-2026-09-22.md` section 6, work package
WP8, build item 1).

Reads (once, streamed one file at a time, never all at once):
- every Halvard exam document under `corpus/out/` (calls/*.txt,
  email/*.txt, tickets/*.json, rss/*.json, crm/*.csv), and
- EnterpriseRAG-Bench's HubSpot, Fireflies, Jira and Gmail slices at
  `~/src/EnterpriseRAG-Bench/generated_data/sources/<slice>/` (plain JSON
  on disk in the clone this was built against -- not zipped, but this
  script still reads and discards one file at a time rather than loading
  a slice into memory, so it costs nothing extra if a future clone ships
  the slices zipped).

Writes, under `--out` (default `var/contamination/`):
- `minhash.npy` -- MinHash signatures, one row per reference document.
- `ngram13.npy` -- every reference document's 13-word-window hashes,
  deduplicated and sorted.
- `doc_meta.json` -- `{"id", "source"}` per `minhash.npy` row.
- `name_blocklist.json` -- hashed proper names, by source and word length.
- `manifest.json` -- signature parameters and per-source document counts.

**Never writes reference document TEXT anywhere** -- `doc_meta.json`
carries only each document's id (a path or ERB's own `dataset_doc_uuid`)
and source label, never its body. This is the ONE script in the repo
allowed to read `corpus/out/` for a purpose other than scoring against
it -- it deliberately does NOT call
`portolan.training.pipeline.assert_not_holdout` (unlike every reader in
the silver-labeling pipeline), because fingerprinting the exam corpus so
NOTHING trains on a lookalike of it is exactly this script's job.

Usage:
    uv run python scripts/data/build_contamination_signatures.py
    uv run python scripts/data/build_contamination_signatures.py \
        --erb-slices hubspot fireflies  # a smaller run, e.g. for a dry run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import yaml

from portolan.training import contamination as C

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HALVARD_ROOT = REPO_ROOT / "corpus" / "out"
DEFAULT_TRUTH_TABLE = REPO_ROOT / "corpus" / "truth-table.yaml"
DEFAULT_ERB_ROOT = Path.home() / "src" / "EnterpriseRAG-Bench"
DEFAULT_OUT = REPO_ROOT / "var" / "contamination"
DEFAULT_CONVERTED_ROOT = REPO_ROOT / "var" / "data" / "converted"
ERB_SLICES = ("hubspot", "fireflies", "jira", "gmail")


def build_background_doc_frequency(
    root: Path, phrases: Iterable[tuple[str, ...]] = ()
) -> tuple[dict[str, int], int]:
    """`contamination.background_frequency` over every file under `root`
    (default `var/data/converted/`: the six real public corpora -- ami,
    cfpb, enron, fda_maude, saas_sales, ubuntu_irc; 10,948 documents as
    of this build): document frequency of every word token and of each
    multi-word candidate name in `phrases`. Reads every file as plain
    text (JSON files included, unparsed): JSON key names count too, and a
    broader background only drops more entries. Built fresh on every run;
    it takes seconds, against the ~20-minute signature pass."""

    def texts() -> Iterator[str]:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                yield path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

    return C.background_frequency(texts(), phrases)


def candidate_phrases(
    name_entries: list[tuple[str, str]], params: C.ContaminationParams
) -> set[tuple[str, ...]]:
    """Every multi-word name the blocklist could hold, for the background phrase count."""
    phrases = set()
    for _source, raw_name in name_entries:
        words = C.normalize_name_words(raw_name)
        if 1 < len(words) <= params.max_name_words:
            phrases.add(words)
    return phrases


# Per-slice scalar fields that hold one proper name (WP8 build item 1:
# "a hashed blocklist of their proper names ... extracted mechanically
# ... from the ERB document metadata"). Chosen from the fields each
# slice's own JSON schema already labels as a company, person, or
# assignee -- never from free-text prose fields.
ERB_NAME_SCALAR_FIELDS: dict[str, tuple[str, ...]] = {
    "hubspot": ("company_name", "owner", "se_assigned", "csm_assigned"),
    "fireflies": ("customer_company", "redwood_owner"),
    "jira": ("customer_company", "reporter", "assignee"),
    "gmail": ("mailbox_owner", "related_account"),
}
# Per-slice list fields whose entries are proper names (people, possibly
# with a "(Role)" suffix or an "<email>" address to strip -- see
# `_clean_person_or_company`) or company names.
ERB_NAME_LIST_FIELDS: dict[str, tuple[str, ...]] = {
    "hubspot": ("competitors",),
    "fireflies": ("redwood_attendees", "customer_attendees", "competitors_mentioned"),
    "jira": (),
    "gmail": ("participants_internal", "participants_external"),
}

_ROLE_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")
_EMAIL_ANGLE_RE = re.compile(r"\s*<[^>]*>\s*$")
_PRODUCT_HEADER_RE = re.compile(r"^#{2,3}\s*\d+\)\s*(.+?)(?:\s*\(.*)?$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _clean_person_or_company(raw: str) -> str:
    """Strips a trailing `<email@domain>` or `(Role/Title)` -- ERB's
    attendee and participant lists both format entries this way (e.g.
    `"Maya Patel (AE)"`, `"Alyssa Chen <alyssa.chen@cascadefg.com>"`)."""
    cleaned = _EMAIL_ANGLE_RE.sub("", raw)
    cleaned = _ROLE_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------
# Halvard (corpus/out/)
# --------------------------------------------------------------------------


def _flatten_json_strings(obj: object) -> list[str]:
    """Every string value in a JSON-like structure, in a stable
    (key-sorted) order -- Halvard's tickets/*.json and rss/*.json carry
    no ERB-style `title_field_name`/`content_field_names` convention
    telling us which fields are prose, so this is the generic fallback."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for key in sorted(obj):
            out.extend(_flatten_json_strings(obj[key]))
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_flatten_json_strings(item))
        return out
    return []


def iter_halvard_documents(root: Path) -> Iterator[tuple[str, str]]:
    """Yields `(doc_id, text)` for every Halvard exam document under
    `root` (`corpus/out/`). `manifest.json` is the generator's own run
    record, not a document, and is skipped."""
    for kind in ("calls", "email"):
        directory = root / kind
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.txt")):
            yield f"{kind}/{path.name}", path.read_text(encoding="utf-8")

    tickets_dir = root / "tickets"
    if tickets_dir.exists():
        for path in sorted(tickets_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            yield f"tickets/{path.name}", " ".join(_flatten_json_strings(data))

    rss_dir = root / "rss"
    if rss_dir.exists():
        for path in sorted(rss_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else [data]
            for i, item in enumerate(items):
                item_id = item.get("id", str(i)) if isinstance(item, dict) else str(i)
                yield f"rss/{path.name}#{item_id}", " ".join(_flatten_json_strings(item))

    crm_dir = root / "crm"
    if crm_dir.exists():
        for path in sorted(crm_dir.glob("*.csv")):
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row_id = row.get("crm_id") or row.get("opportunity_id") or ""
                    text = " ".join(str(v) for v in row.values() if v)
                    yield f"crm/{path.name}#{row_id}", text


def extract_halvard_names(truth_table_path: Path) -> list[str]:
    """Mechanically pulls every proper name out of the Halvard truth
    table (`corpus/truth-table.yaml`) -- supplier, products, competitors,
    internal staff, branches, accounts and contacts, plus every listed
    alias -- WP8 build item 1's "extracted mechanically from the Halvard
    truth table"."""
    data = yaml.safe_load(truth_table_path.read_text(encoding="utf-8"))
    names: list[str] = []

    meta = data.get("meta", {})
    supplier = meta.get("supplier", {})
    if supplier.get("name"):
        names.append(supplier["name"])
    for section in ("products", "competitors", "internal_people", "branches"):
        for entry in meta.get(section, []) or []:
            if entry.get("name"):
                names.append(entry["name"])
            names.extend(entry.get("aliases", []) or [])

    for section in ("accounts", "contacts"):
        for entry in data.get(section, []) or []:
            if entry.get("name"):
                names.append(entry["name"])
            names.extend(entry.get("aliases", []) or [])

    return names


def resolve_halvard_name_hash(
    digest_hex: str, truth_table_path: Path, params: C.ContaminationParams
) -> str | None:
    """Reporting-only helper (WP8 build item 4: "if the synthetic v1
    worlds show hits against Halvard names, report which names"). The
    on-disk blocklist is hash-only (module docstring, "never raw text"),
    so this re-derives the same hashes fresh from `corpus/truth-table.
    yaml` -- a file already checked into this repo in the clear, not a
    reference corpus being protected -- and returns the first plaintext
    name that produces `digest_hex`, or `None` if none does."""
    target = int(digest_hex, 16)
    for name in extract_halvard_names(truth_table_path):
        words = C.normalize_name_words(name)
        if words and C.word_sequence_hash(words) == target:
            return name
    return None


# --------------------------------------------------------------------------
# ERB (~/src/EnterpriseRAG-Bench)
# --------------------------------------------------------------------------


def _erb_document_text(data: dict) -> str:
    """ERB's own `title_field_name`/`content_field_names` convention
    (present on every document across all four slices) tells us exactly
    which fields are prose -- falls back to flattening every string in
    the document if a future ERB export ever drops that convention."""
    title_field = data.get("title_field_name")
    content_fields = data.get("content_field_names")
    if title_field and content_fields:
        parts: list[str] = []
        title_val = data.get(title_field)
        if title_val:
            parts.append(str(title_val))
        for field_name in content_fields:
            value = data.get(field_name)
            if isinstance(value, list):
                parts.append(" ".join(str(v) for v in value))
            elif value:
                parts.append(str(value))
        return " ".join(parts)
    return " ".join(_flatten_json_strings(data))


def _erb_document_names(slice_name: str, data: dict) -> list[str]:
    names: list[str] = []
    for field_name in ERB_NAME_SCALAR_FIELDS.get(slice_name, ()):
        value = data.get(field_name)
        if isinstance(value, str) and value:
            names.append(_clean_person_or_company(value))
    for field_name in ERB_NAME_LIST_FIELDS.get(slice_name, ()):
        for value in data.get(field_name) or []:
            if isinstance(value, str) and value:
                names.append(_clean_person_or_company(value))
    return names


def iter_erb_slice_documents(
    erb_root: Path, slice_name: str
) -> Iterator[tuple[str, str, list[str]]]:
    """Yields `(doc_id, text, names)` for every document in one ERB
    slice, reading and discarding one file at a time -- `text` and
    `names` are the only things a caller keeps; the parsed JSON dict
    itself is dropped at the end of each loop iteration."""
    slice_dir = erb_root / "generated_data" / "sources" / slice_name
    if not slice_dir.exists():
        print(f"warning: ERB slice not found, skipping: {slice_dir}", file=sys.stderr)
        return
    for dirpath, _dirnames, filenames in os.walk(slice_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".json"):
                continue
            path = Path(dirpath) / fname
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            doc_id = data.get("dataset_doc_uuid") or str(path.relative_to(slice_dir))
            yield doc_id, _erb_document_text(data), _erb_document_names(slice_name, data)


def extract_erb_static_names(erb_root: Path) -> list[str]:
    """Every employee name (`employee_directory.yaml`) plus the company
    name and product names (`company_overview.md`) -- extracted the same
    mechanical way as the Halvard truth table, not from any document's
    prose body."""
    names: list[str] = []

    employees_path = erb_root / "generated_data" / "employee_directory.yaml"
    if employees_path.exists():
        directory = yaml.safe_load(employees_path.read_text(encoding="utf-8")) or {}
        for people in (directory.get("departments") or {}).values():
            for person in people or []:
                if person.get("name"):
                    names.append(person["name"])

    overview_path = erb_root / "generated_data" / "company_overview.md"
    if overview_path.exists():
        text = overview_path.read_text(encoding="utf-8")
        bold = _BOLD_RE.search(text)
        if bold:
            names.append(bold.group(1))
        names.extend(m.strip() for m in _PRODUCT_HEADER_RE.findall(text))

    return names


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def collect_name_entries(
    *,
    halvard_root: Path,
    truth_table_path: Path,
    erb_root: Path,
    erb_slices: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Every `(source_label, raw_name)` pair this build's four name
    sources produce -- Halvard's truth table, each ERB document's own
    metadata fields, and ERB's employee directory / company overview --
    without computing a single MinHash signature or 13-gram hash. This
    is `--names-only`'s fast path: re-walks the same documents
    `build_signatures` does, but only reads the small metadata fields a
    name can come from, not the full body text, so it costs a fraction
    of the full build (no `minhash_signature`/`ngram_window_hash_array`
    calls at all) -- useful for re-running just the distinctiveness
    filter or the background-frequency table without repeating the
    ~20-minute reference-signature pass over ERB's Gmail slice."""
    name_entries: list[tuple[str, str]] = []
    name_entries.extend(("halvard", name) for name in extract_halvard_names(truth_table_path))
    for slice_name in erb_slices:
        source = f"erb_{slice_name}"
        for _doc_id, _text, names in iter_erb_slice_documents(erb_root, slice_name):
            name_entries.extend((source, name) for name in names)
    if erb_slices:
        name_entries.extend(("erb", name) for name in extract_erb_static_names(erb_root))
    return name_entries


def signature_sources(
    *, halvard_root: Path, truth_table_path: Path, erb_root: Path, erb_slices: tuple[str, ...]
) -> dict[str, Path]:
    """Every path the signatures are built from, by name, for `C.fingerprint_sources`."""
    sources = {"halvard": halvard_root, "halvard_truth_table": truth_table_path}
    generated = erb_root / "generated_data"
    for slice_name in erb_slices:
        sources[f"erb_{slice_name}"] = generated / "sources" / slice_name
    if erb_slices:
        sources["erb_employee_directory"] = generated / "employee_directory.yaml"
        sources["erb_company_overview"] = generated / "company_overview.md"
    return sources


def build_signatures(
    *,
    halvard_root: Path,
    truth_table_path: Path,
    erb_root: Path,
    erb_slices: tuple[str, ...],
    params: C.ContaminationParams,
) -> tuple[np.ndarray, list[dict], np.ndarray, list[tuple[str, str]], dict[str, int]]:
    """The full build, in one pass per source (never reading a document
    twice). Returns everything `save_references` needs, EXCEPT the name
    blocklist itself: `name_entries` (raw `(source, name)` pairs) comes
    back unfiltered so `main()` can apply `contamination.
    build_name_blocklist`'s distinctiveness filter in exactly one place,
    whether the entries came from this full pass or from `--names-only`'s
    `collect_name_entries`.

    The 13-gram hash array is folded (`C.fold_ngram_chunks`) every
    `C._NGRAM_FLUSH_EVERY` documents rather than concatenated once at
    the end -- on the real ~150k-document ERB build, holding every raw
    (pre-dedup) window hash simultaneously (high hundreds of millions of
    them) drove this process past 10GB of RSS and the machine into heavy
    swap before it was killed. Folding periodically bounds peak memory
    to roughly one batch plus the running deduplicated set."""
    a, b = C.permutation_coefficients(params.minhash_permutations, params.minhash_seed)
    signatures: list[np.ndarray] = []
    doc_meta: list[dict] = []
    ngram_chunks: list[np.ndarray] = []
    ngram_sorted = np.array([], dtype=np.uint64)
    name_entries: list[tuple[str, str]] = []
    source_counts: Counter[str] = Counter()
    doc_count = 0

    def _absorb(doc_id: str, source: str, text: str) -> None:
        nonlocal ngram_sorted, doc_count
        words = C.normalize_words(text)
        shingles = C.word_shingles(words, params.shingle_size)
        signatures.append(C.minhash_signature(shingles, a, b))
        doc_meta.append({"id": doc_id, "source": source})
        ngrams = C.ngram_window_hash_array(words, params.ngram_window)
        if ngrams.size:
            ngram_chunks.append(ngrams)
        source_counts[source] += 1
        doc_count += 1
        if doc_count % C._NGRAM_FLUSH_EVERY == 0:
            ngram_sorted = C.fold_ngram_chunks(ngram_sorted, ngram_chunks)
            ngram_chunks.clear()
        if doc_count % 2000 == 0:
            print(f"  ... {doc_count} documents ({sum(source_counts.values())} total)", flush=True)

    for doc_id, text in iter_halvard_documents(halvard_root):
        _absorb(doc_id, "halvard", text)
    name_entries.extend(("halvard", name) for name in extract_halvard_names(truth_table_path))
    print(f"halvard done: {source_counts.get('halvard', 0)} documents", flush=True)

    for slice_name in erb_slices:
        source = f"erb_{slice_name}"
        for doc_id, text, names in iter_erb_slice_documents(erb_root, slice_name):
            _absorb(doc_id, source, text)
            name_entries.extend((source, name) for name in names)
        print(f"{source} done: {source_counts.get(source, 0)} documents", flush=True)
    if erb_slices:
        name_entries.extend(("erb", name) for name in extract_erb_static_names(erb_root))

    ngram_sorted = C.fold_ngram_chunks(ngram_sorted, ngram_chunks)
    matrix = (
        np.stack(signatures)
        if signatures
        else np.zeros((0, params.minhash_permutations), dtype=np.uint64)
    )
    return matrix, doc_meta, ngram_sorted, name_entries, dict(source_counts)


def _legacy_survivor_count(
    name_entries: list[tuple[str, str]], params: C.ContaminationParams
) -> int:
    """How many distinct `(source, normalized-words)` entries would have
    survived the PRE-review rules -- length and max-word-count only, no
    distinctiveness filter -- for the manifest's before/after comparison
    (coordinator review, 2026-09-22: "Report how many of the 173,615
    entries survive")."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for source, raw_name in name_entries:
        words = C.normalize_name_words(raw_name)
        if not words or len(words) > params.max_name_words:
            continue
        if len(words) == 1 and len(words[0]) < 3:
            continue
        seen.add((source, words))
    return len(seen)


def _drop_reason_counts(
    name_entries: list[tuple[str, str]],
    params: C.ContaminationParams,
    background_doc_freq: dict[str, int],
    background_doc_count: int,
) -> dict[str, int]:
    """Distinct `(source, normalized-words)` entries left out of the blocklist, by
    `C.name_drop_reason` (counts only: the manifest never holds a name)."""
    reasons: dict[tuple[str, tuple[str, ...]], str] = {}
    for source, raw_name in name_entries:
        words = C.normalize_name_words(raw_name)
        if not words or len(words) > params.max_name_words:
            continue
        reason = C.name_drop_reason(raw_name, background_doc_freq, background_doc_count)
        if reason is not None:
            reasons.setdefault((source, words), reason)
    return dict(sorted(Counter(reasons.values()).items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--halvard-root", type=Path, default=DEFAULT_HALVARD_ROOT)
    parser.add_argument("--truth-table", type=Path, default=DEFAULT_TRUTH_TABLE)
    parser.add_argument("--erb-root", type=Path, default=DEFAULT_ERB_ROOT)
    parser.add_argument(
        "--converted-root",
        type=Path,
        default=DEFAULT_CONVERTED_ROOT,
        help="background corpus for the name-blocklist distinctiveness filter "
        "(default: var/data/converted/, the six real public corpora)",
    )
    parser.add_argument(
        "--erb-slices",
        nargs="*",
        default=list(ERB_SLICES),
        help="ERB slices to fingerprint (default: hubspot fireflies jira gmail); pass "
        "--erb-slices with no values to skip ERB entirely (Halvard-only signatures)",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="skip the MinHash/13-gram pass and re-use the minhash.npy/ngram13.npy/doc_meta.json "
        "already at --out (which must exist) -- only re-walks documents for names and rebuilds "
        "name_blocklist.json + manifest.json. Fast re-run after a distinctiveness-filter or "
        "background-corpus change; does not touch the reference signatures.",
    )
    args = parser.parse_args(argv)

    params = C.ContaminationParams()
    print(f"halvard root:    {args.halvard_root}")
    print(f"truth table:     {args.truth_table}")
    print(f"erb root:        {args.erb_root}")
    print(f"erb slices:      {', '.join(args.erb_slices) or '(none)'}")
    print(f"converted root:  {args.converted_root}")
    print(f"out:             {args.out}")
    print(f"names-only:      {args.names_only}", flush=True)

    if args.names_only:
        if not C.references_exist(args.out):
            print(
                f"error: --names-only requires existing signatures at {args.out} "
                "(run without --names-only first)",
                file=sys.stderr,
            )
            return 1
        existing = C.load_references(args.out)
        matrix, doc_meta, ngram_sorted = (
            existing.minhash_matrix,
            existing.doc_meta,
            existing.ngram_sorted,
        )
        existing_manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
        source_counts = existing_manifest["source_counts"]
        # The MinHash and 13-gram arrays are reused, so their sources' fingerprints are too:
        # if a source changed since, the training-set builder refuses these signatures.
        source_fingerprints = existing_manifest.get("source_fingerprints") or {}
        name_entries = collect_name_entries(
            halvard_root=args.halvard_root,
            truth_table_path=args.truth_table,
            erb_root=args.erb_root,
            erb_slices=tuple(args.erb_slices),
        )
    else:
        source_fingerprints = C.fingerprint_sources(
            signature_sources(
                halvard_root=args.halvard_root,
                truth_table_path=args.truth_table,
                erb_root=args.erb_root,
                erb_slices=tuple(args.erb_slices),
            )
        )
        matrix, doc_meta, ngram_sorted, name_entries, source_counts = build_signatures(
            halvard_root=args.halvard_root,
            truth_table_path=args.truth_table,
            erb_root=args.erb_root,
            erb_slices=tuple(args.erb_slices),
            params=params,
        )

    phrases = candidate_phrases(name_entries, params)
    background_doc_freq, background_doc_count = build_background_doc_frequency(
        args.converted_root, phrases
    )
    print(
        f"background corpus: {background_doc_count} documents, "
        f"{len(background_doc_freq)} distinct tokens and candidate phrases",
        flush=True,
    )

    legacy_count = _legacy_survivor_count(name_entries, params)
    drop_reasons = _drop_reason_counts(
        name_entries, params, background_doc_freq, background_doc_count
    )
    name_blocklist = C.build_name_blocklist(
        name_entries,
        params,
        background_doc_freq=background_doc_freq,
        background_doc_count=background_doc_count,
    )
    distinctive_count = sum(
        len(hashes) for by_length in name_blocklist.values() for hashes in by_length.values()
    )
    print(
        f"name blocklist: {legacy_count} candidate entries before the distinctiveness filter, "
        f"{distinctive_count} survive ({distinctive_count / legacy_count:.1%})"
        if legacy_count
        else "name blocklist: 0 candidate entries",
        flush=True,
    )

    manifest = C.save_references(
        args.out,
        matrix,
        doc_meta,
        ngram_sorted,
        name_blocklist,
        params,
        source_counts,
        source_fingerprints=source_fingerprints,
    )
    manifest["name_blocklist_legacy_count"] = legacy_count
    manifest["name_blocklist_distinctive_count"] = distinctive_count
    manifest["name_blocklist_drop_reasons"] = drop_reasons
    manifest["background_corpus"] = {
        "root": str(args.converted_root),
        "documents": background_doc_count,
        "distinct_keys": len(background_doc_freq),
        "candidate_phrases": len(phrases),
        "common_word_doc_share": C.COMMON_WORD_DOC_SHARE,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"reference documents: {manifest['reference_docs_total']}")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:<14} {count}")
    print(f"unique 13-grams: {manifest['unique_13grams']}")
    print("name blocklist (post-distinctiveness):")
    for source, by_length in manifest["name_blocklist_counts"].items():
        total = sum(by_length.values())
        print(f"  {source:<14} {total}  ({by_length})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
