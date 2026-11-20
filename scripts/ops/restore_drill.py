"""Restore drill: back up a seeded database the way a deploy does, restore it, prove it identical.

    uv run python scripts/ops/restore_drill.py [--scale 1] [--app-check all|none] [--keep]

Runs on the local Postgres only, on two scratch databases whose names start with
`portolan_test_` (default `portolan_test_restore_src` and `portolan_test_restore_dst`), and drops
both at the end unless `--keep`:

1. Seed the source: every migration, the `tests/verify_seed.py` graph, then extra rows
   (`--scale`; 1 is above the live graph's size in sources, notes and claims) that fill every
   table the seed leaves empty and carry awkward text, jsonb, vectors, circular foreign keys and
   a sequence ahead of its table.
2. Back it up through `portolan.db.migrate_with_backup`, the function every deploy calls, into a
   scratch directory standing in for the volume (the dump and its `.sha256`). A drill-only
   destructive migration is pending, so the backup is taken for the same reason a deploy takes
   one.
3. Restore that dump with `python -m portolan.db restore` into a freshly created target.
4. Compare source-at-backup with the restored target (`portolan.db.fingerprint`): tables, row
   counts, a checksum of every table's rows in primary-key order, sequences, the migration list,
   extensions and schema objects. The source after the bad migration must differ (the
   comparison can see a change). Then serve the app on the restored database and run
   `scripts/verify.py all` against it.
5. Restore eight broken files onto the restored target: five damaged copies beside the
   checksum recorded at backup time (truncated twice, flipped bytes, a corrupted schema
   statement, not a dump), an intact copy with no checksum file, and two damaged copies
   with `--no-verify-checksum`. Each must fail at the layer named for it and leave the
   target unchanged.

Result: `var/proof/<UTC stamp>-restore-drill.json`, plus the tables on stdout. The live
counterpart, run in the Railway web container, is `python -m portolan.db.rehearse`
(`01-spec/restore-drill-2026-09-22.md`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psycopg import sql  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from portolan import db as db_module  # noqa: E402
from portolan.db.fingerprint import compare, equality_table, fingerprint  # noqa: E402
from portolan.store import LocalStore  # noqa: E402
from portolan.util.redact import redact_dsn  # noqa: E402

DEFAULT_SERVER_URL = "postgresql://localhost/postgres"
DEFAULT_SRC_DB = "portolan_test_restore_src"
DEFAULT_DST_DB = "portolan_test_restore_dst"
DEFAULT_OUT_DIR = REPO_ROOT / "var" / "proof"
TEST_DB_PREFIX = "portolan_test_"
PAID_MODEL_KEYS = ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "ANTHROPIC_API_KEY")
BAD_MIGRATION = "9999_restore_drill_bad_migration.sql"
# Destroys data the drill can then see missing from the source and present in the restore.
BAD_MIGRATION_SQL = """\
-- Restore drill only (scripts/ops/restore_drill.py): destroys data, so the backup taken
-- before it is the only copy of what it destroyed.
DELETE FROM usage_events;
UPDATE claims SET claim_text = 'lost in a bad migration';
"""
# Live graph size, 2026-09-19 reseed (`01-spec/HANDOFF-2026-09-19.md`, "Live reseed").
LIVE_GRAPH_ROWS = {"sources": 248, "notes": 135, "claims": 456}
# Extra rows per unit of --scale.
EXTRA_ROWS_PER_SCALE = {
    "sources": 50,
    "notes": 40,
    "claims": 400,
    "chunks": 300,
    "links": 100,
    "changelog": 300,
    "usage_events": 300,
}
EMBEDDING_DIM = 1024
AWKWARD_TEXT = (
    "plain ascii",
    "quotes ' \" and a backslash \\ and a tab\tand a newline\nand a carriage return\r end",
    "unicode: café, naïve, 港口, Ω≈ç√, emoji 🧭📦",
    "combining e\u0301 and a zero-width\u200bspace",
    "sql-ish: '); DROP TABLE claims; --",
    "   leading and trailing spaces   ",
    "",
)


class DrillRefused(RuntimeError):
    """The drill was pointed somewhere it must not go; nothing was touched."""


# ---------------------------------------------------------------------
# Databases
# ---------------------------------------------------------------------


def url_for(server_url: str, dbname: str) -> str:
    """`server_url` with its database name replaced, credentials and query kept."""
    parts = urlsplit(server_url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{dbname}", parts.query, parts.fragment))


def refuse_unless_local_test(server_url: str, *dbnames: str) -> None:
    """Refuse (before any connection) unless every database is a local `portolan_test_*` one
    and the process is not running under `railway run`. Same rule as `tests/guards.py`."""
    from tests import guards

    railway = sorted(name for name in os.environ if name.startswith("RAILWAY_"))
    if railway:
        raise DrillRefused(
            "refusing to run the drill with RAILWAY_* variables set (a `railway run` shell): "
            "the drill is local-only"
        )
    container = guards.in_container()
    for dbname in dbnames:
        if not dbname.startswith(TEST_DB_PREFIX):
            raise DrillRefused(
                f"refusing database {dbname!r}: the drill only uses databases whose name "
                f"starts with {TEST_DB_PREFIX!r}"
            )
        reason = guards.database_refusal(
            url_for(server_url, dbname), os.environ, container=container, allow_maintenance=False
        )
        if reason:
            raise DrillRefused(f"refusing database {dbname!r}: its connection string {reason}")


def recreate_database(server_url: str, dbname: str) -> None:
    conn = db_module.connect(server_url, autocommit=True)
    try:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
        )
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    finally:
        conn.close()


def drop_database(server_url: str, dbname: str) -> None:
    conn = db_module.connect(server_url, autocommit=True)
    try:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
        )
    finally:
        conn.close()


def database_exists(server_url: str, dbname: str) -> bool:
    conn = db_module.connect(server_url, autocommit=True)
    try:
        row = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
    finally:
        conn.close()
    return row is not None


def database_size(url: str) -> int:
    conn = db_module.connect(url)
    try:
        return int(conn.execute("SELECT pg_database_size(current_database())").fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------


def seed_source(src_url: str, store_root: Path, *, seed: str) -> dict[str, int]:
    """`verify`: the `tests/verify_seed.py` graph (pipeline over `corpus/out`, a weekly report,
    action cards). `minimal`: every migration and nothing else, for the in-suite drill."""
    if seed == "verify":
        from tests.verify_seed import seed_verify_graph

        return seed_verify_graph(src_url, store_root)
    if seed == "minimal":
        conn = db_module.connect(src_url)
        try:
            db_module.apply_migrations(conn)
        finally:
            conn.close()
        return {}
    raise ValueError(f"unknown seed {seed!r} (expected 'verify' or 'minimal')")


def _embedding(rng: random.Random, centroids: list[list[float]]) -> str:
    """A unit vector near one of a few topic centroids, as real embeddings cluster. Uniformly
    random 1024-d vectors slowed HNSW inserts to about one a second past 5,000 rows."""
    centroid = centroids[rng.randrange(len(centroids))]
    vector = [c + rng.gauss(0, 0.15 / EMBEDDING_DIM**0.5) for c in centroid]
    norm = sum(v * v for v in vector) ** 0.5
    return "[" + ",".join(f"{v / norm:.6f}" for v in vector) + "]"


def _centroids(rng: random.Random, n: int = 16) -> list[list[float]]:
    return [[rng.gauss(0, 1) / EMBEDDING_DIM**0.5 for _ in range(EMBEDDING_DIM)] for _ in range(n)]


def _awkward(i: int) -> str:
    return AWKWARD_TEXT[i % len(AWKWARD_TEXT)]


def add_extra_rows(url: str, store_root: Path, *, scale: float) -> dict[str, int]:
    """Deterministic extra rows (the same on every run at the same scale). Returns how many
    rows went into each table."""
    rng = random.Random(20260922)
    centroids = _centroids(rng)
    counts = {name: max(5, round(per * scale)) for name, per in EXTRA_ROWS_PER_SCALE.items()}
    base = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    store = LocalStore(store_root)
    source_types = ("call", "email", "ticket", "doc", "crm", "chat", "survey", "rss")
    conn = db_module.connect(url)
    added: dict[str, int] = {}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT fact_type FROM claims ORDER BY 1")
            fact_types = [row[0] for row in cur.fetchall()] or ["complaint", "request", "other"]
            cur.execute("SELECT slug FROM notes WHERE type = 'account' ORDER BY slug")
            accounts = [row[0] for row in cur.fetchall()]

            sources = [f"drill-source-{i:05d}" for i in range(counts["sources"])]
            with cur.copy(
                "COPY sources (slug, source_type, s3_key, content_hash, source_date, ingested_at,"
                " extraction_status, summary, participants, metadata, source_id,"
                " extraction_model, extraction_samples) FROM STDIN"
            ) as copy:
                for i, slug in enumerate(sources):
                    kind = source_types[i % len(source_types)]
                    key = f"originals/drill/{slug}.txt"
                    body = f"Drill original {i}.\n{_awkward(i)}\n".encode()
                    store.put(key, body, "text/plain")
                    copy.write_row(
                        (
                            slug,
                            kind,
                            key,
                            hashlib.sha256(body).hexdigest(),
                            date(2026, 1, 1) + timedelta(days=i % 250),
                            base + timedelta(minutes=i, microseconds=i),
                            "done",
                            _awkward(i + 1) if i % 3 else None,
                            Jsonb(
                                [{"name": "Zoë Ångström", "role": "buyer"}, {"name": _awkward(i)}]
                            ),
                            Jsonb({"drill": True, "i": i, "nested": {"list": [1, 2.5, None, "x"]}}),
                            slug,
                            "drill-model",
                            1 + i % 5,
                        )
                    )
            added["sources"] = len(sources)

            notes = [f"drill-note-{i:05d}" for i in range(counts["notes"])]
            with cur.copy(
                "COPY notes (slug, type, title, description, status, created, updated,"
                " stale_after, schema_version, body_md, version) FROM STDIN"
            ) as copy:
                for i, slug in enumerate(notes):
                    if i == 0:
                        body = "Long body, stored out of line.\n" + ("lorem ipsum 港口 " * 12000)
                    else:
                        body = f"# Drill note {i}\n\n{_awkward(i)}\n\n[[{notes[i - 1]}]]\n"
                    copy.write_row(
                        (
                            slug,
                            "decision" if i % 5 == 0 else "signal",
                            f"Drill note {i}: {_awkward(i + 2)}",
                            _awkward(i + 3) if i % 2 else None,
                            "living",
                            base + timedelta(hours=i),
                            base + timedelta(hours=i, seconds=i),
                            "infinity" if i == 1 else base + timedelta(days=90 + i),
                            "0.1",
                            body,
                            1 + i % 3,
                        )
                    )
            added["notes"] = len(notes)

            with cur.copy(
                "COPY claims (note_slug, fact_type, claim_text, source_slug, locator, quote,"
                " speaker, valid_from, valid_to, recorded, severity, sentiment, confidence,"
                " archived) FROM STDIN"
            ) as copy:
                for i in range(counts["claims"]):
                    copy.write_row(
                        (
                            notes[i % len(notes)],
                            fact_types[i % len(fact_types)],
                            f"Drill claim {i}: {_awkward(i)}",
                            sources[i % len(sources)],
                            f"00:{i % 60:02d}:{(i * 7) % 60:02d}",
                            _awkward(i + 4) if i % 4 else None,
                            "zoë-ångström" if i % 2 else None,
                            date(1970, 1, 1)
                            if i == 0
                            else date(2026, 1, 1) + timedelta(days=i % 200),
                            date(2999, 12, 31) if i == 1 else None,
                            base + timedelta(seconds=i * 13, microseconds=i * 7),
                            "high" if i % 11 == 0 else None,
                            (i % 5) - 2,
                            f"{(i % 1001) / 1000:.3f}",
                            i % 17 == 0,
                        )
                    )
            cur.execute(
                "SELECT id FROM claims WHERE note_slug LIKE 'drill-note-%' ORDER BY id LIMIT 4"
            )
            first_claims = [row[0] for row in cur.fetchall()]
            # A superseded claim points at a later one: a self-referencing foreign key.
            cur.execute(
                "UPDATE claims SET superseded_at = %s, superseded_by = %s WHERE id = %s",
                (base, first_claims[1], first_claims[0]),
            )
            added["claims"] = counts["claims"]

            with cur.copy(
                "COPY chunks (note_slug, heading, text, token_count, embedding) FROM STDIN"
            ) as copy:
                for i in range(counts["chunks"]):
                    copy.write_row(
                        (
                            notes[i % len(notes)],
                            f"Section {i}" if i % 3 else None,
                            f"Drill chunk {i} about deliveries and pricing. {_awkward(i)}",
                            20 + i % 400,
                            _embedding(rng, centroids),
                        )
                    )
            added["chunks"] = counts["chunks"]

            link_rows = set()
            for i in range(counts["links"]):
                target = (
                    accounts[i % len(accounts)]
                    if accounts and i % 2
                    else notes[(i + 1) % len(notes)]
                )
                link_rows.add((notes[i % len(notes)], target, "body" if i % 3 else "provenance"))
            with cur.copy("COPY links (from_slug, to_slug, kind) FROM STDIN") as copy:
                for row in sorted(link_rows):
                    copy.write_row(row)
            added["links"] = len(link_rows)

            with cur.copy(
                "COPY changelog (occurred_at, kind, summary, details) FROM STDIN"
            ) as copy:
                for i in range(counts["changelog"]):
                    copy.write_row(
                        (
                            base + timedelta(seconds=i * 3, microseconds=i),
                            "drill_event",
                            f"Drill event {i}: {_awkward(i)}",
                            Jsonb({"i": i, "text": _awkward(i), "ratio": i / 7}),
                        )
                    )
            added["changelog"] = counts["changelog"]

            with cur.copy(
                "COPY usage_events (tenant, surface, model_key, input_tokens, output_tokens,"
                " cached_tokens, cost_estimate, request_id, created, extra) FROM STDIN"
            ) as copy:
                surfaces = ("extract", "merge", "chat", "report", "embed", "mcp", "researcher")
                for i in range(counts["usage_events"]):
                    copy.write_row(
                        (
                            "drill",
                            surfaces[i % len(surfaces)],
                            "drill/model",
                            100 + i,
                            10 + i % 50,
                            i % 3,
                            f"{i / 997:.6f}",
                            f"req-{i:06d}",
                            base + timedelta(seconds=i),
                            Jsonb({"attempt": 1 + i % 2}),
                        )
                    )
            added["usage_events"] = counts["usage_events"]
            # A sequence ahead of its table's max id, as after rolled-back inserts.
            cur.execute(
                "SELECT setval(pg_get_serial_sequence('usage_events', 'id'),"
                " (SELECT max(id) FROM usage_events) + 500)"
            )

            added.update(_fill_small_tables(cur, notes, first_claims, base))
        conn.commit()
    finally:
        conn.close()
    return added


def _fill_small_tables(cur, notes: list[str], claims: list[int], base: datetime) -> dict[str, int]:
    """A few rows in each table the verify seed leaves empty, so every table's checksum covers
    real rows. Timestamps are explicit, so an insert trigger firing on restore would show."""
    cur.execute(
        "INSERT INTO conversations (owner, title, created, updated) VALUES"
        " ('drill-user', 'Drill chat: 港口 🧭', %s, %s), ('drill-user', 'Second', %s, %s)"
        " RETURNING id",
        (base, base, base, base),
    )
    conversation_ids = [row[0] for row in cur.fetchall()]
    for n, conversation_id in enumerate(conversation_ids):
        for m, role in enumerate(("user", "assistant", "user")):
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content_md, citations, created, usage)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    conversation_id,
                    role,
                    f"Message {n}.{m}: {_awkward(n + m)}",
                    Jsonb([{"note": notes[m], "claim_id": claims[m % len(claims)]}]),
                    base - timedelta(days=1, seconds=m),
                    Jsonb({"input_tokens": 10 * m}),
                ),
            )
    cur.execute(
        "INSERT INTO standing_questions (reader, question, created, active) VALUES"
        " ('drill-reader', 'Which accounts mention late deliveries?', %s, true),"
        " ('drill-reader', 'Retired question', %s, false) RETURNING id",
        (base, base),
    )
    question_ids = [row[0] for row in cur.fetchall()]
    for week in ("2026-W35", "2026-W36"):
        cur.execute(
            "INSERT INTO standing_question_answers (question_id, week, answer_md, citations, created)"
            " VALUES (%s, %s, %s, %s, %s)",
            (question_ids[0], week, f"Answer for {week}: {_awkward(2)}", Jsonb([]), base),
        )
    cur.execute(
        "INSERT INTO dead_letter (source_id, stage, attempt, error, payload, created, redriven_at)"
        " VALUES ('drill-source-00000', 'extract', 3, %s, %s, %s, NULL),"
        " ('drill-source-00001', 'merge', 1, 'timeout', '{}', %s, %s)",
        (_awkward(1), Jsonb({"why": _awkward(2)}), base, base, base + timedelta(hours=1)),
    )
    cur.execute(
        "INSERT INTO edit_records (note_slug, claim_id, editor, edited_at, before_text,"
        " after_text, reason) VALUES (%s, %s, 'drill-editor', %s, 'before', %s, 'typo')"
        " RETURNING id",
        (notes[2], claims[2], base, _awkward(2)),
    )
    edit_id = cur.fetchone()[0]
    # claims -> edit_records -> claims: a circular pair of foreign keys.
    cur.execute("UPDATE claims SET human_edit_id = %s WHERE id = %s", (edit_id, claims[2]))
    for i in range(3):
        cur.execute(
            "INSERT INTO researcher_findings (run_id, candidate_slug, tool, arguments, result,"
            " created) VALUES ('drill-run', %s, 'search', %s, %s, %s)",
            (notes[i], Jsonb({"q": _awkward(i)}), Jsonb({"hits": [i, i + 1]}), base),
        )
    for i, verdict in enumerate(("supported", "not_supported", "supported")):
        cur.execute(
            "INSERT INTO support_judge_verdicts (sentence_hash, reference_hash, judge_version,"
            " sentence, verdict, judged_at) VALUES (%s, %s, 'drill-v1', %s, %s, %s)",
            (f"s{i:03d}", f"r{i:03d}", _awkward(i), verdict, base),
        )
    cur.execute(
        "INSERT INTO theme_judge_unverified (theme_slug, candidate_claim_id, prompt_state_hash,"
        " retry_count, updated_at) VALUES (%s, %s, 'drill-hash', 2, %s)",
        (notes[0], claims[3], base),
    )
    cur.execute(
        "INSERT INTO theme_resolutions (theme_slug, resolution_claim_id, resolved_reason,"
        " resolved_week, created, status, evidence) VALUES (%s, %s, %s, %s, %s, 'proposed', %s)",
        (notes[0], claims[3], _awkward(3), date(2026, 8, 31), base, _awkward(4)),
    )
    return {
        "conversations": 2,
        "messages": 6,
        "standing_questions": 2,
        "standing_question_answers": 2,
        "dead_letter": 2,
        "edit_records": 1,
        "researcher_findings": 3,
        "support_judge_verdicts": 3,
        "theme_judge_unverified": 1,
        "theme_resolutions": 1,
    }


# ---------------------------------------------------------------------
# Drill steps
# ---------------------------------------------------------------------


@dataclass
class DrillConfig:
    server_url: str = DEFAULT_SERVER_URL
    src_db: str = DEFAULT_SRC_DB
    dst_db: str = DEFAULT_DST_DB
    workdir: Path | None = None
    out_dir: Path = DEFAULT_OUT_DIR
    scale: float = 1.0
    seed: str = "verify"
    # A scripts/verify.py target run against the app on the restored database, or None.
    app_check: str | None = "all"
    keep: bool = False


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - started, 3)


def deploy_backup(src_url: str, store_root: Path, workdir: Path) -> tuple[list[str], Path]:
    """What a deploy does: `migrate_with_backup` with a migration pending. The migrations
    directory is the real one plus the drill's bad migration."""
    migrations = workdir / "migrations"
    migrations.mkdir(parents=True, exist_ok=True)
    for path in sorted(db_module.MIGRATIONS_DIR.glob("*.sql")):
        shutil.copy2(path, migrations / path.name)
    (migrations / BAD_MIGRATION).write_text(BAD_MIGRATION_SQL, encoding="utf-8")

    backups = store_root / "backups"
    existing = set(backups.glob("*.dump")) if backups.exists() else set()
    conn = db_module.connect(src_url)
    try:
        applied = db_module.migrate_with_backup(
            conn, database_url=src_url, store_root=store_root, migrations_dir=migrations
        )
    finally:
        conn.close()
    new = sorted(set(backups.glob("*.dump")) - existing)
    if len(new) != 1:
        raise RuntimeError(f"expected one new backup under {backups}, found {len(new)}")
    return applied, new[0]


def restore_cli(dump: Path, database_url: str, *extra: str) -> dict[str, Any]:
    """`python -m portolan.db restore <dump>` in a child process, the command the runbook runs."""
    env = {k: v for k, v in os.environ.items() if k not in PAID_MODEL_KEYS}
    env["DATABASE_URL"] = database_url
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "portolan.db", "restore", str(dump), *extra],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    output = (proc.stdout + proc.stderr).strip()
    lines = output.splitlines()
    errors = [line for line in lines if line.startswith("ERROR")]
    return {
        "exit_code": proc.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "output_tail": [redact_dsn(line)[:400] for line in (errors or lines)[-2:]],
        "stopped_by": _stopped_by(output) if proc.returncode else None,
    }


# Phrases from the error messages of `portolan.db.verify_checksum`, `verify_dump` and
# `restore_database`, in the order those layers run.
STOP_PHRASES = (
    ("has no checksum file", "missing checksum"),
    ("does not match its checksum", "checksum"),
    ("is unreadable", "pre-read"),
    ("rolled back, target unchanged", "rollback"),
)
BEFORE_RESTORE = frozenset({"missing checksum", "checksum", "pre-read"})


def _stopped_by(output: str) -> str | None:
    for phrase, layer in STOP_PHRASES:
        if phrase in output:
            return layer
    return None


@dataclass
class BrokenDump:
    case: str
    path: Path
    extra: tuple[str, ...]
    expected: str


def bad_dumps(dump: Path, out: Path) -> list[BrokenDump]:
    """Broken copies of a good dump. The first five sit beside the checksum recorded for the
    good dump (damage after the backup), so the checksum stops each one; the sixth is intact
    with no checksum file; the last two are restored with `--no-verify-checksum`, where the
    read-through and the single transaction are what protect the target."""
    out.mkdir(parents=True, exist_ok=True)
    data = dump.read_bytes()
    recorded = db_module.checksum_path(dump).read_text(encoding="utf-8").split()[0]
    marker = b"CREATE TABLE public.claims"
    at = data.find(marker)
    if at < 0:
        raise RuntimeError("the dump holds no 'CREATE TABLE public.claims' statement to corrupt")
    toc = bytearray(data)
    toc[at : at + len(marker)] = b"CREATE TABLX public.claims"
    middle = int(len(data) * 0.6)
    flipped = bytearray(data)
    flipped[middle : middle + 64] = bytes(b ^ 0xFF for b in data[middle : middle + 64])
    no_check = ("--no-verify-checksum",)
    variants = [
        ("truncated to half", data[: len(data) // 2], True, (), "checksum"),
        ("last 512 bytes cut", data[:-512], True, (), "checksum"),
        ("64 bytes flipped mid-file", bytes(flipped), True, (), "checksum"),
        ("schema statement corrupted", bytes(toc), True, (), "checksum"),
        ("not a dump", random.Random(7).randbytes(4096), True, (), "checksum"),
        ("intact, checksum file missing", data, False, (), "missing checksum"),
        (
            "truncated to half, --no-verify-checksum",
            data[: len(data) // 2],
            False,
            no_check,
            "pre-read",
        ),
        (
            "schema statement corrupted, --no-verify-checksum",
            bytes(toc),
            False,
            no_check,
            "rollback",
        ),
    ]
    broken = []
    for n, (case, payload, with_recorded, extra, expected) in enumerate(variants):
        path = out / f"bad-{n}.dump"
        path.write_bytes(payload)
        if with_recorded:
            db_module.write_checksum(path, recorded)
        broken.append(BrokenDump(case, path, extra, expected))
    return broken


def run_app_check(dst_url: str, store_root: Path, workdir: Path, target: str) -> dict[str, Any]:
    """Serve the app on the restored database (its startup runs the deploy's migration path)
    and run `scripts/verify.py <target>` against it over HTTP."""
    import secrets as secrets_module

    from tests.verify_seed import app_settings, serve, write_keys

    org_key, mcp_key, org_path, mcp_path = write_keys(workdir / "keys")
    settings = app_settings(
        dst_url,
        store_root,
        org_key=org_key,
        session_secret=secrets_module.token_hex(32),
        mcp_org_key=mcp_key,
    )
    proof_dir = workdir / "verify-proof"
    with serve(settings) as base_url:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "verify.py"),
                target,
                "--base-url",
                base_url,
                "--org-key-file",
                str(org_path),
                "--mcp-key-file",
                str(mcp_path),
                "--out-dir",
                str(proof_dir),
            ],
            cwd=REPO_ROOT,
            env={k: v for k, v in os.environ.items() if k not in PAID_MODEL_KEYS},
            capture_output=True,
            text=True,
            timeout=600,
        )
    results = sorted(proof_dir.glob("*.json"))
    verify = json.loads(results[-1].read_text(encoding="utf-8")) if results else {}
    return {
        "target": target,
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0 and bool(verify.get("ok")),
        "passed": verify.get("passed"),
        "total": verify.get("total"),
        "summary": proc.stdout.strip().splitlines()[:1],
        "failed_checks": [
            f"{c['target']}: {c['name']}" for c in verify.get("checks", []) if not c["ok"]
        ],
    }


def _headline_counts(fp: dict[str, Any]) -> dict[str, int]:
    return {
        name: fp["tables"].get(name, {}).get("rows", 0)
        for name in ("sources", "notes", "claims", "chunks")
    }


def run_drill(cfg: DrillConfig) -> dict[str, Any]:
    refuse_unless_local_test(cfg.server_url, cfg.src_db, cfg.dst_db)
    src_url = url_for(cfg.server_url, cfg.src_db)
    dst_url = url_for(cfg.server_url, cfg.dst_db)
    own_workdir = cfg.workdir is None
    workdir = (
        Path(tempfile.mkdtemp(prefix="portolan-restore-drill-")) if own_workdir else cfg.workdir
    )
    workdir.mkdir(parents=True, exist_ok=True)
    store_root = workdir / "volume" / "store"
    timer = Timer()
    result: dict[str, Any] = {
        "tool": "scripts/ops/restore_drill.py",
        "started_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "src_db": cfg.src_db,
            "dst_db": cfg.dst_db,
            "seed": cfg.seed,
            "scale": cfg.scale,
            "app_check": cfg.app_check,
        },
    }
    try:
        with timer.step("create source database"):
            recreate_database(cfg.server_url, cfg.src_db)
        with timer.step(f"seed ({cfg.seed})"):
            result["seed_counts"] = seed_source(src_url, store_root, seed=cfg.seed)
        with timer.step("extra rows"):
            result["extra_rows"] = add_extra_rows(src_url, store_root, scale=cfg.scale)
        result["source_database_bytes"] = database_size(src_url)

        with timer.step("fingerprint source at backup time"):
            before = fingerprint(src_url)
        result["headline_rows"] = _headline_counts(before)
        result["live_graph_rows"] = LIVE_GRAPH_ROWS
        result["drill_at_or_above_live"] = all(
            result["headline_rows"][k] >= v for k, v in LIVE_GRAPH_ROWS.items()
        )

        with timer.step("deploy: migrate_with_backup (pg_dump + bad migration)"):
            applied, dump = deploy_backup(src_url, store_root, workdir)
        result["deploy"] = {
            "applied": applied,
            "dump": str(dump.relative_to(workdir)),
            "dump_bytes": dump.stat().st_size,
            "dump_magic_ok": dump.read_bytes()[:5] == b"PGDMP",
            "checksum_file": db_module.checksum_path(dump).name,
            "checksum_written": db_module.checksum_path(dump).is_file(),
        }
        after_deploy = fingerprint(src_url)
        control = compare(before, after_deploy)
        result["negative_control"] = {
            "source_changed_by_bad_migration": not control["equal"],
            "differs": sorted({t for r in control["rows"] for t in r["differs"]}),
        }

        with timer.step("verify checksum (sha256 of the dump)"):
            db_module.verify_checksum(dump)
        with timer.step("verify dump (read every block, no database)"):
            server_major = db_module._server_major_version(src_url)
            pg_restore = db_module._resolve_pg_binary("pg_restore", server_major)
            db_module.verify_dump(dump, pg_restore)

        with timer.step("create target database"):
            recreate_database(cfg.server_url, cfg.dst_db)
        with timer.step("restore (python -m portolan.db restore)"):
            result["restore"] = restore_cli(dump, dst_url)
        if result["restore"]["exit_code"] != 0:
            raise RuntimeError(f"restore failed: {result['restore']['output_tail']}")
        result["target_database_bytes"] = database_size(dst_url)

        with timer.step("fingerprint restored target"):
            restored = fingerprint(dst_url)
        result["equality"] = compare(before, restored)
        conn = db_module.connect(dst_url)
        try:
            result["pending_migrations_on_target"] = db_module.pending_migrations(conn)
        finally:
            conn.close()

        if cfg.app_check:
            with timer.step(f"app check (scripts/verify.py {cfg.app_check})"):
                result["app_check"] = run_app_check(dst_url, store_root, workdir, cfg.app_check)

        with timer.step("broken dumps refused"):
            baseline = fingerprint(dst_url)
            cases = []
            for broken in bad_dumps(dump, workdir / "bad"):
                outcome = restore_cli(broken.path, dst_url, *broken.extra)
                unchanged = compare(baseline, fingerprint(dst_url))
                cases.append(
                    {
                        "case": broken.case,
                        "bytes": broken.path.stat().st_size,
                        "exit_code": outcome["exit_code"],
                        "stopped_by": outcome["stopped_by"],
                        "expected_stop": broken.expected,
                        "refused_before_restore": outcome["stopped_by"] in BEFORE_RESTORE,
                        "rolled_back": outcome["stopped_by"] == "rollback",
                        "target_unchanged": unchanged["equal"],
                        "differs": sorted({t for r in unchanged["rows"] for t in r["differs"]}),
                        "output_tail": outcome["output_tail"],
                    }
                )
        result["broken_dumps"] = cases
    finally:
        result["timings_seconds"] = timer.seconds
        if not cfg.keep:
            for dbname in (cfg.src_db, cfg.dst_db):
                drop_database(cfg.server_url, dbname)
            if own_workdir:
                shutil.rmtree(workdir, ignore_errors=True)
        result["databases_dropped"] = not cfg.keep and not any(
            database_exists(cfg.server_url, name) for name in (cfg.src_db, cfg.dst_db)
        )
        result["workdir"] = str(workdir) if (cfg.keep or not own_workdir) else None

    result["throughput"] = _throughput(result)
    result["ok"] = _verdict(result)
    result["finished_at"] = datetime.now(UTC).isoformat()
    return result


def _throughput(result: dict[str, Any]) -> dict[str, float]:
    t = result["timings_seconds"]
    mib = 1024 * 1024
    backup_s = t["deploy: migrate_with_backup (pg_dump + bad migration)"]
    restore_s = result["restore"]["seconds"]
    return {
        "source_database_mib": round(result["source_database_bytes"] / mib, 2),
        "dump_mib": round(result["deploy"]["dump_bytes"] / mib, 2),
        "dump_to_database_ratio": round(
            result["deploy"]["dump_bytes"] / result["source_database_bytes"], 3
        ),
        "backup_database_mib_per_second": round(
            result["source_database_bytes"] / mib / backup_s, 2
        ),
        "restore_database_mib_per_second": round(
            result["source_database_bytes"] / mib / restore_s, 2
        ),
    }


def _verdict(result: dict[str, Any]) -> bool:
    app = result.get("app_check")
    return (
        result["equality"]["equal"]
        and result["negative_control"]["source_changed_by_bad_migration"]
        and result["deploy"]["applied"] == [BAD_MIGRATION]
        and result["deploy"]["dump_magic_ok"]
        and result["deploy"]["checksum_written"]
        and result["pending_migrations_on_target"] == []
        and (app is None or app["ok"])
        and len(result["broken_dumps"]) == 8
        and all(
            c["exit_code"] != 0 and c["target_unchanged"] and c["stopped_by"] == c["expected_stop"]
            for c in result["broken_dumps"]
        )
    )


def _git_sha() -> str | None:
    """HEAD, with `+dirty` when tracked files differ from it (the run is then not that commit)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------


def report(result: dict[str, Any], path: Path) -> str:
    lines = [equality_table(result["equality"], "Source at backup", "Restored"), ""]
    app = result.get("app_check")
    if app:
        lines.append(
            f"App check: scripts/verify.py {app['target']} on the restored database: "
            f"{'PASS' if app['ok'] else 'FAIL'} {app['passed']}/{app['total']}"
            + (f" (failed: {'; '.join(app['failed_checks'][:5])})" if app["failed_checks"] else "")
        )
    control = result["negative_control"]
    lines.append(
        "Negative control: source after the bad migration differs from the backup in "
        + (", ".join(control["differs"]) or "nothing (the comparison is blind)")
    )
    lines.append("")
    lines.append("| Broken dump | Exit | Stopped by | Target unchanged |")
    lines.append("|---|---|---|---|")
    for case in result["broken_dumps"]:
        stopped = case["stopped_by"] or "nothing"
        if stopped != case["expected_stop"]:
            stopped += f" (expected {case['expected_stop']})"
        lines.append(
            f"| {case['case']} | {case['exit_code']} | {stopped} | "
            f"{'yes' if case['target_unchanged'] else 'NO: ' + ', '.join(case['differs'])} |"
        )
    lines.append("")
    lines.append("| Step | Seconds |")
    lines.append("|---|---|")
    for step, seconds in result["timings_seconds"].items():
        lines.append(f"| {step} | {seconds} |")
    tp = result["throughput"]
    lines.append("")
    lines.append(
        f"Sizes: database {tp['source_database_mib']} MiB, dump {tp['dump_mib']} MiB "
        f"({result['deploy']['dump_bytes']} bytes); rows {result['headline_rows']} vs live "
        f"{result['live_graph_rows']} (drill at or above live: {result['drill_at_or_above_live']})"
    )
    verdict = "PASS" if result["ok"] else "FAIL"
    lines.append(f"{verdict} restore drill -> {path}")
    return "\n".join(lines)


def write_result(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}-restore-drill.json"
    n = 1
    while path.exists():
        path = out_dir / f"{stamp}-restore-drill-{n}.json"
        n += 1
    path.write_text(
        json.dumps(result, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="restore_drill.py", description=__doc__.splitlines()[0])
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="local maintenance db")
    parser.add_argument("--src-db", default=DEFAULT_SRC_DB)
    parser.add_argument("--dst-db", default=DEFAULT_DST_DB)
    parser.add_argument("--scale", type=float, default=1.0, help="extra rows; 1 is live size")
    parser.add_argument("--seed", choices=("verify", "minimal"), default="verify")
    parser.add_argument("--app-check", default="all", help="scripts/verify.py target, or 'none'")
    parser.add_argument("--workdir", type=Path, default=None, help="volume stand-in (temp)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--keep", action="store_true", help="keep both databases and the workdir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = DrillConfig(
        server_url=args.server_url,
        src_db=args.src_db,
        dst_db=args.dst_db,
        workdir=args.workdir,
        out_dir=args.out_dir,
        scale=args.scale,
        seed=args.seed,
        app_check=None if args.app_check == "none" else args.app_check,
        keep=args.keep,
    )
    try:
        result = run_drill(cfg)
    except DrillRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    path = write_result(result, cfg.out_dir)
    print(report(result, path))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
