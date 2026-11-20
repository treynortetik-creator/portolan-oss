"""`scripts/ops/restore_drill.py` (checklist item 15): a small drill inside the suite.

The drill backs a seeded database up through `migrate_with_backup` (the deploy path), restores
it with `python -m portolan.db restore` into a second database, compares the two, serves the
app on the restore, and feeds five broken dumps to the restore. Here it runs on the minimal
seed at a small scale with the `health` app check; the full drill (the verify seed and
`scripts/verify.py all`) is run by hand, results in `01-spec/restore-drill-2026-09-22.md`.

`scripts/` is not in the container image, so this module skips there.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import pytest

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "ops", "restore_drill.py")


def _load():
    spec = importlib.util.spec_from_file_location("restore_drill", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the module through sys.modules.
    sys.modules["restore_drill"] = module
    spec.loader.exec_module(module)
    return module


drill = _load()


def _session_db_name(db_url: str) -> str:
    return urlsplit(db_url).path.rsplit("/", 1)[-1].split("?", 1)[0]


@pytest.fixture()
def drill_config(db_url: str, tmp_path: Path):
    try:
        psycopg.connect(db_url).close()
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable: {exc}")
    name = _session_db_name(db_url)
    return drill.DrillConfig(
        server_url=drill.url_for(db_url, "postgres"),
        src_db=f"{name}_drill_src",
        dst_db=f"{name}_drill_dst",
        workdir=tmp_path / "work",
        out_dir=tmp_path / "proof",
        scale=0.05,
        seed="minimal",
        app_check="health",
    )


def test_small_drill_restores_an_identical_database_and_refuses_broken_dumps(drill_config):
    result = drill.run_drill(drill_config)

    rows = {row["check"]: row for row in result["equality"]["rows"]}
    assert set(rows) == {
        "tables",
        "row counts",
        "row checksums (primary-key order)",
        "sequence values",
        "migrations applied",
        "extensions",
        "schema objects",
    }
    unequal = {name: row["differs"] for name, row in rows.items() if not row["equal"]}
    assert unequal == {}
    assert rows["row counts"]["before"] == rows["row counts"]["after"] > 0
    assert rows["migrations applied"]["after"] > 0

    # The backup came from the deploy path, taken because a migration was pending.
    assert result["deploy"]["applied"] == [drill.BAD_MIGRATION]
    assert result["deploy"]["dump_magic_ok"]
    assert result["restore"]["exit_code"] == 0
    assert result["pending_migrations_on_target"] == []
    # The comparison sees what the bad migration destroyed in the source.
    assert {"claims", "usage_events"} <= set(result["negative_control"]["differs"])
    assert result["app_check"]["ok"], result["app_check"]

    cases = {case["case"]: case for case in result["broken_dumps"]}
    assert len(cases) == 8
    for name, case in cases.items():
        assert case["exit_code"] != 0, name
        assert case["target_unchanged"], (name, case["differs"])
    assert cases["truncated to half"]["refused_before_restore"]
    assert cases["64 bytes flipped mid-file"]["refused_before_restore"]
    # Damage after the backup is stopped by the recorded checksum, wherever it lands.
    for name in (
        "truncated to half",
        "last 512 bytes cut",
        "64 bytes flipped mid-file",
        "schema statement corrupted",
        "not a dump",
    ):
        assert cases[name]["stopped_by"] == "checksum", (name, cases[name]["output_tail"])
    assert cases["intact, checksum file missing"]["stopped_by"] == "missing checksum"
    # Without the checksum, the read-through still stops a truncated file before the target...
    assert cases["truncated to half, --no-verify-checksum"]["stopped_by"] == "pre-read"
    # ...and a file that reads to the end but fails a statement is rolled back whole.
    no_check = cases["schema statement corrupted, --no-verify-checksum"]
    assert not no_check["refused_before_restore"]
    assert no_check["rolled_back"]

    assert result["ok"]
    assert result["databases_dropped"]


def test_fingerprint_sees_one_changed_row_and_one_moved_sequence(db, db_url, tmp_path: Path):
    drill.add_extra_rows(db_url, tmp_path / "store", scale=0.05)
    first = drill.fingerprint(db_url)
    assert drill.compare(first, drill.fingerprint(db_url))["equal"]

    with db.cursor() as cur:
        cur.execute(
            "UPDATE claims SET claim_text = claim_text || '.' "
            "WHERE id = (SELECT min(id) FROM claims)"
        )
    second = drill.fingerprint(db_url)
    rows = {row["check"]: row for row in drill.compare(first, second)["rows"]}
    assert rows["row counts"]["equal"]
    assert rows["row checksums (primary-key order)"]["differs"] == ["claims"]
    assert [name for name, row in rows.items() if not row["equal"]] == [
        "row checksums (primary-key order)"
    ]

    with db.cursor() as cur:
        cur.execute("SELECT nextval(pg_get_serial_sequence('changelog', 'id'))")
    third = drill.fingerprint(db_url)
    rows = {row["check"]: row for row in drill.compare(second, third)["rows"]}
    assert [name for name, row in rows.items() if not row["equal"]] == ["sequence values"]
    assert rows["sequence values"]["differs"] == ["changelog_id_seq"]


def test_drill_refuses_anything_but_local_test_databases(
    drill_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(**changes) -> str:
        cfg = drill.DrillConfig(**{**drill_config.__dict__, **changes})
        with pytest.raises(drill.DrillRefused) as excinfo:
            drill.run_drill(cfg)
        return str(excinfo.value)

    assert "portolan_test_" in refused(dst_db="portolan")
    assert "not local" in refused(server_url="postgresql://db.example.com/postgres")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    assert "railway run" in refused()
    # Refused before any database was created.
    assert not (drill_config.workdir / "volume").exists()
