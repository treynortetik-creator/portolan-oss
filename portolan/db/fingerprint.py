"""Fingerprint a database and compare two fingerprints: the evidence behind a restore.

`fingerprint` reads, in one read-only repeatable-read snapshot: every public table's row
count and an md5 over its rows in primary-key order, every sequence's position, the
applied migrations, the extensions and a digest of the schema objects. `compare` turns two
fingerprints into the equality table `python -m portolan.db.rehearse` and
`scripts/ops/restore_drill.py` print. Text sorts use the C collation, so servers with
different locales agree.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from psycopg import sql

from portolan import db as db_module

# The fingerprint session pins every setting that changes how a row prints as text.
FINGERPRINT_SETTINGS = (
    "SET TIME ZONE 'UTC'",
    "SET DateStyle = 'ISO, MDY'",
    "SET IntervalStyle = 'postgres'",
    "SET extra_float_digits = 1",
    "SET bytea_output = 'hex'",
)

_SCHEMA_QUERIES = {
    "columns": """
        SELECT table_name,
               -- Position among live columns: a dropped column leaves a gap in
               -- ordinal_position that a restore closes.
               row_number() OVER (PARTITION BY table_name ORDER BY ordinal_position),
               column_name, data_type, udt_name, is_nullable,
               column_default, is_generated, generation_expression, is_identity,
               identity_generation, character_maximum_length, numeric_precision, numeric_scale
        FROM information_schema.columns WHERE table_schema = 'public'
    """,
    "constraints": """
        SELECT conrelid::regclass::text, conname, contype, pg_get_constraintdef(oid)
        FROM pg_constraint WHERE connamespace = 'public'::regnamespace
    """,
    "indexes": "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'",
    "triggers": """
        SELECT t.tgrelid::regclass::text, t.tgname, pg_get_triggerdef(t.oid)
        FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
        WHERE NOT t.tgisinternal AND c.relnamespace = 'public'::regnamespace
    """,
    "functions": """
        SELECT p.oid::regprocedure::text, pg_get_functiondef(p.oid)
        FROM pg_proc p
        WHERE p.pronamespace = 'public'::regnamespace AND p.prokind IN ('f', 'p')
          AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_proc'::regclass
                          AND d.objid = p.oid AND d.deptype = 'e')
    """,
    "views": "SELECT viewname, definition FROM pg_views WHERE schemaname = 'public'",
    "sequence definitions": """
        SELECT sequencename, data_type::text, start_value, min_value, max_value, increment_by,
               cycle, cache_size
        FROM pg_sequences WHERE schemaname = 'public'
    """,
    "comments": """
        SELECT c.relname, coalesce(a.attname, ''), d.description
        FROM pg_description d
        JOIN pg_class c ON c.oid = d.objoid AND d.classoid = 'pg_class'::regclass
        LEFT JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.objsubid
        WHERE c.relnamespace = 'public'::regnamespace
    """,
}


def fingerprint(url: str) -> dict[str, Any]:
    """Everything `compare` checks, read in one read-only, repeatable-read snapshot."""
    conn = db_module.connect(url)
    try:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        for statement in FINGERPRINT_SETTINGS:
            conn.execute(statement)
        result: dict[str, Any] = {
            "database": conn.execute("SELECT current_database()").fetchone()[0],
            "server_version": conn.execute("SHOW server_version").fetchone()[0],
            "taken_at": datetime.now(UTC).isoformat(),
        }
        tables = conn.execute(
            """
            SELECT c.oid, c.relname FROM pg_class c
            WHERE c.relnamespace = 'public'::regnamespace AND c.relkind IN ('r', 'p')
            ORDER BY c.relname COLLATE "C"
            """
        ).fetchall()
        result["tables"] = {name: _table_fingerprint(conn, oid, name) for oid, name in tables}

        sequences = conn.execute(
            """
            SELECT c.relname FROM pg_class c
            WHERE c.relnamespace = 'public'::regnamespace AND c.relkind = 'S'
            ORDER BY c.relname COLLATE "C"
            """
        ).fetchall()
        result["sequences"] = {}
        for (name,) in sequences:
            last_value, is_called = conn.execute(
                sql.SQL("SELECT last_value, is_called FROM public.{}").format(sql.Identifier(name))
            ).fetchone()
            result["sequences"][name] = {"last_value": int(last_value), "is_called": is_called}

        has_migrations = "schema_migrations" in result["tables"]
        result["migrations"] = (
            [
                row[0]
                for row in conn.execute(
                    'SELECT filename FROM schema_migrations ORDER BY filename COLLATE "C"'
                ).fetchall()
            ]
            if has_migrations
            else []
        )
        result["extensions"] = dict(
            conn.execute(
                'SELECT extname, extversion FROM pg_extension ORDER BY extname COLLATE "C"'
            ).fetchall()
        )
        result["schema"] = {}
        for kind, query in _SCHEMA_QUERIES.items():
            items = sorted(
                json.dumps([str(v) if v is not None else None for v in row], ensure_ascii=False)
                for row in conn.execute(query).fetchall()
            )
            digest = hashlib.sha256("\n".join(items).encode()).hexdigest()
            result["schema"][kind] = {"count": len(items), "digest": digest}
        conn.rollback()
        return result
    finally:
        conn.close()


def _table_fingerprint(conn, oid: int, name: str) -> dict[str, Any]:
    pk = conn.execute(
        """
        SELECT a.attname, a.attcollation <> 0
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = %s::oid AND i.indisprimary
        ORDER BY array_position(i.indkey::int2[], a.attnum)
        """,
        (oid,),
    ).fetchall()
    order = [
        sql.SQL('t.{} COLLATE "C"' if collatable else "t.{}").format(sql.Identifier(column))
        for column, collatable in pk
    ] or [sql.SQL('t::text COLLATE "C"')]
    rows, checksum = conn.execute(
        sql.SQL(
            "SELECT count(*), coalesce(md5(string_agg(md5(t::text), '' ORDER BY {})), '')"
            " FROM public.{} t"
        ).format(sql.SQL(", ").join(order), sql.Identifier(name))
    ).fetchone()
    return {
        "rows": int(rows),
        "checksum": checksum,
        "ordered_by": [column for column, _ in pk] or ["row text"],
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The equality table: one row per check, plus the names of whatever differs."""
    rows: list[dict[str, Any]] = []

    def add(check: str, left: Any, right: Any, equal: bool, differs: list[str] | None = None):
        rows.append(
            {
                "check": check,
                "before": left,
                "after": right,
                "equal": equal,
                "differs": differs or [],
            }
        )

    bt, at = before["tables"], after["tables"]
    add("tables", len(bt), len(at), sorted(bt) == sorted(at), sorted(set(bt) ^ set(at)))
    shared = sorted(set(bt) & set(at))
    count_diff = [t for t in shared if bt[t]["rows"] != at[t]["rows"]]
    add(
        "row counts",
        sum(v["rows"] for v in bt.values()),
        sum(v["rows"] for v in at.values()),
        not count_diff and sorted(bt) == sorted(at),
        count_diff,
    )
    sum_diff = [t for t in shared if bt[t]["checksum"] != at[t]["checksum"]]
    add(
        "row checksums (primary-key order)",
        f"{len(bt)} tables",
        f"{len(shared) - len(sum_diff)} of {len(bt)} match",
        not sum_diff and sorted(bt) == sorted(at),
        sum_diff,
    )
    bs, as_ = before["sequences"], after["sequences"]
    seq_diff = sorted(n for n in set(bs) | set(as_) if bs.get(n) != as_.get(n))
    add("sequence values", len(bs), len(as_), not seq_diff, seq_diff)
    bm, am = before["migrations"], after["migrations"]
    add("migrations applied", len(bm), len(am), bm == am, sorted(set(bm) ^ set(am)))
    be, ae = before["extensions"], after["extensions"]
    add(
        "extensions",
        ", ".join(f"{k} {v}" for k, v in be.items()),
        ", ".join(f"{k} {v}" for k, v in ae.items()),
        be == ae,
        sorted(k for k in set(be) | set(ae) if be.get(k) != ae.get(k)),
    )
    bsch, asch = before["schema"], after["schema"]
    schema_diff = sorted(k for k in set(bsch) | set(asch) if bsch.get(k) != asch.get(k))
    add(
        "schema objects",
        sum(v["count"] for v in bsch.values()),
        sum(v["count"] for v in asch.values()),
        not schema_diff,
        schema_diff,
    )
    return {"equal": all(r["equal"] for r in rows), "rows": rows}


def equality_table(comparison: dict[str, Any], left: str, right: str) -> str:
    lines = [f"| Check | {left} | {right} | Equal |", "|---|---|---|---|"]
    for row in comparison["rows"]:
        mark = "yes" if row["equal"] else "NO: " + ", ".join(row["differs"][:6])
        lines.append(f"| {row['check']} | {row['before']} | {row['after']} | {mark} |")
    return "\n".join(lines)
