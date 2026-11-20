"""Applied migrations are never edited (CLAUDE.md hard rule; guardrails plan 2026-09-22,
rank 4). `portolan/db/migrations/MANIFEST.sha256` records the sha256 of every migration that
exists; this test fails when one of those files changes, disappears, or has no manifest line.
A new migration passes once its line is added.

Why a hash and not review: `apply_migrations` records a migration by filename only, so an
edit to an applied file never re-runs on a database that already has it -- production and a
fresh test database silently diverge, and nothing fails until a deploy does.

Regenerating the whole manifest after an edit would pass this test; that is caught at merge
instead: `scripts/check_test_edits.py` reports any changed or deleted line in this manifest
(appending a line is not reported)."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pytest

from portolan.db import MIGRATIONS_DIR

MANIFEST_NAME = "MANIFEST.sha256"

HOW_TO = (
    "Applied migrations are never edited: put the change in the next numbered file instead. "
    "To register a NEW migration, run from the repo root: "
    "`cd portolan/db/migrations && shasum -a 256 00NN_name.sql >> MANIFEST.sha256` "
    "(or `sha256sum`), and commit the manifest line with the file."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    """`<hex>  <name>` lines, as `shasum -a 256` writes them. A path in the name column
    (`portolan/db/migrations/0031_x.sql`) is reduced to its file name."""
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        entries[Path(name).name] = digest.lower()
    return entries


def manifest_problems(directory: Path) -> list[str]:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return [f"{manifest_path} is missing"]
    manifest = read_manifest(manifest_path)
    on_disk = {p.name: p for p in sorted(directory.glob("*.sql"))}
    problems = []
    for name, path in on_disk.items():
        if name not in manifest:
            problems.append(f"{name}: no line in {MANIFEST_NAME}")
        elif manifest[name] != _sha256(path):
            problems.append(
                f"{name}: contents changed since it was recorded (an applied migration was edited)"
            )
    for name in manifest:
        if name not in on_disk:
            problems.append(
                f"{name}: listed in {MANIFEST_NAME} but the file is gone "
                "(an applied migration was deleted or renamed)"
            )
    return problems


def test_every_migration_matches_the_manifest() -> None:
    problems = manifest_problems(MIGRATIONS_DIR)
    assert not problems, "\n".join(problems) + "\n\n" + HOW_TO


# --- order and destructive statements (guardrails review 2026-09-22, rows 24-25) -------
#
# `apply_migrations` applies files in name order and records them by name, so a new file
# that sorts before an applied one runs last on production and mid-sequence on a fresh
# database: the two schemas diverge silently. And spec section 11 says migrations are
# additive; only the pre-migration dump stands between a DROP and lost data.

MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
# 0001-0032 were applied before the destructive-statement rule existed.
FIRST_DESTRUCTIVE_CHECKED = 33
DESTRUCTIVE = re.compile(r"\b(?:DROP|TRUNCATE)\b|\bDELETE\s+FROM\b", re.IGNORECASE)
DESTRUCTIVE_OK = re.compile(r"^[ \t]*--[ \t]*destructive-ok:[ \t]*\S", re.MULTILINE)

ORDER_HOW_TO = (
    "A new migration takes the next number (`NNNN_snake_case.sql`, one above the highest), "
    "and its manifest line goes at the end of MANIFEST.sha256."
)
DESTRUCTIVE_HOW_TO = (
    "Migrations are additive (spec section 11). A migration that must DROP, DELETE or "
    "TRUNCATE carries a line `-- destructive-ok: <why, and what the backup covers>`."
)


def _manifest_names_in_file_order(path: Path) -> list[str]:
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.append(Path(line.partition(" ")[2].strip().lstrip("*")).name)
    return names


def ordering_problems(directory: Path) -> list[str]:
    problems = []
    names = sorted(p.name for p in directory.glob("*.sql"))
    numbers = []
    for name in names:
        match = MIGRATION_NAME.match(name)
        if match is None:
            problems.append(f"{name}: not named NNNN_snake_case.sql")
        else:
            numbers.append(int(match.group(1)))
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(
            f"migration numbers are not 1..{len(numbers)} with no gap or repeat: {numbers}"
        )
    manifest_order = _manifest_names_in_file_order(directory / MANIFEST_NAME)
    if manifest_order != sorted(manifest_order):
        problems.append(f"{MANIFEST_NAME} lines are not in migration order")
    last_listed = max(manifest_order, default="")
    for name in names:
        if name not in manifest_order and name < last_listed:
            problems.append(f"{name}: sorts before {last_listed}, which is already listed")
    return problems


def _sql_without_comments_or_literals(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def destructive_problems(directory: Path) -> list[str]:
    problems = []
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_NAME.match(path.name)
        if match is None or int(match.group(1)) < FIRST_DESTRUCTIVE_CHECKED:
            continue
        sql = path.read_text(encoding="utf-8")
        found = DESTRUCTIVE.findall(_sql_without_comments_or_literals(sql))
        if found and not DESTRUCTIVE_OK.search(sql):
            shown = ", ".join(sorted({f.split()[0].upper() for f in found}))
            problems.append(f"{path.name}: {shown} with no `-- destructive-ok: <reason>` line")
    return problems


def test_migrations_are_numbered_in_order() -> None:
    problems = ordering_problems(MIGRATIONS_DIR)
    assert not problems, "\n".join(problems) + "\n\n" + ORDER_HOW_TO


def test_new_migrations_are_additive_or_say_why() -> None:
    problems = destructive_problems(MIGRATIONS_DIR)
    assert not problems, "\n".join(problems) + "\n\n" + DESTRUCTIVE_HOW_TO


def _next_number(directory: Path) -> int:
    return len(list(directory.glob("*.sql"))) + 1


def _add_migration(directory: Path, name: str, sql: str) -> Path:
    path = directory / name
    path.write_text(sql, encoding="utf-8")
    with (directory / MANIFEST_NAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{_sha256(path)}  {name}\n")
    return path


@pytest.fixture()
def copy_of_migrations(tmp_path: Path) -> Path:
    dest = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, dest)
    return dest


def test_check_catches_an_edited_migration(copy_of_migrations: Path) -> None:
    target = copy_of_migrations / "0001_init.sql"
    target.write_text(target.read_text(encoding="utf-8") + "\n-- sneaky edit\n", encoding="utf-8")
    problems = manifest_problems(copy_of_migrations)
    assert problems == [
        "0001_init.sql: contents changed since it was recorded (an applied migration was edited)"
    ]


def test_check_catches_a_deleted_migration(copy_of_migrations: Path) -> None:
    (copy_of_migrations / "0002_index_state.sql").unlink()
    problems = manifest_problems(copy_of_migrations)
    assert len(problems) == 1 and problems[0].startswith("0002_index_state.sql: listed in")


def test_new_migration_fails_until_its_line_is_added(copy_of_migrations: Path) -> None:
    new = copy_of_migrations / "0099_new_table.sql"
    new.write_text("CREATE TABLE example (id int);\n", encoding="utf-8")
    assert manifest_problems(copy_of_migrations) == [
        f"0099_new_table.sql: no line in {MANIFEST_NAME}"
    ]

    # the how-to's own command shape: `shasum -a 256 <file> >> MANIFEST.sha256`
    with (copy_of_migrations / MANIFEST_NAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{_sha256(new)}  0099_new_table.sql\n")
    assert manifest_problems(copy_of_migrations) == []


def test_manifest_accepts_a_path_in_the_name_column(copy_of_migrations: Path) -> None:
    # `shasum -a 256 portolan/db/migrations/0099_x.sql >> ...` run from the repo root.
    new = copy_of_migrations / "0099_new_table.sql"
    new.write_text("SELECT 1;\n", encoding="utf-8")
    with (copy_of_migrations / MANIFEST_NAME).open("a", encoding="utf-8") as fh:
        fh.write(f"{_sha256(new)}  portolan/db/migrations/0099_new_table.sql\n")
    assert manifest_problems(copy_of_migrations) == []


def test_the_next_number_passes_every_check(copy_of_migrations: Path) -> None:
    n = _next_number(copy_of_migrations)
    _add_migration(copy_of_migrations, f"{n:04d}_new_table.sql", "CREATE TABLE example (id int);\n")
    assert manifest_problems(copy_of_migrations) == []
    assert ordering_problems(copy_of_migrations) == []
    assert destructive_problems(copy_of_migrations) == []


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("0015a_late_fix.sql", "not named"),
        ("0015_second_fifteen.sql", "no gap or repeat"),
        ("0099_far_ahead.sql", "no gap or repeat"),
    ],
)
def test_an_out_of_order_new_migration_is_caught(
    copy_of_migrations: Path, name: str, fragment: str
) -> None:
    _add_migration(copy_of_migrations, name, "CREATE TABLE example (id int);\n")
    assert manifest_problems(copy_of_migrations) == []  # the hash check alone lets it through
    problems = ordering_problems(copy_of_migrations)
    assert any(fragment in p for p in problems), problems


def test_a_new_file_sorting_before_a_listed_one_is_caught(copy_of_migrations: Path) -> None:
    (copy_of_migrations / "0015a_late_fix.sql").write_text("SELECT 1;\n", encoding="utf-8")
    problems = ordering_problems(copy_of_migrations)
    assert any("sorts before" in p for p in problems), problems


def test_a_manifest_line_inserted_mid_file_is_caught(copy_of_migrations: Path) -> None:
    manifest = copy_of_migrations / MANIFEST_NAME
    lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
    manifest.write_text("".join([lines[-1], *lines[:-1]]), encoding="utf-8")
    assert f"{MANIFEST_NAME} lines are not in migration order" in ordering_problems(
        copy_of_migrations
    )


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE notes DROP COLUMN body;\n",
        "DROP TABLE IF EXISTS old_things;\n",
        "DELETE FROM claims WHERE fact_type = 'other';\n",
        "truncate usage_events;\n",
        "-- destructive-ok:\nDROP INDEX notes_body_idx;\n",  # a marker with no reason
    ],
)
def test_a_destructive_new_migration_needs_a_reason(copy_of_migrations: Path, sql: str) -> None:
    n = _next_number(copy_of_migrations)
    _add_migration(copy_of_migrations, f"{n:04d}_change.sql", sql)
    problems = destructive_problems(copy_of_migrations)
    assert len(problems) == 1 and "destructive-ok" in problems[0], problems


@pytest.mark.parametrize(
    "sql",
    [
        "-- destructive-ok: column unused since 0030; pre-migration dump keeps it\n"
        "ALTER TABLE notes DROP COLUMN body;\n",
        "ALTER TABLE claims ADD CONSTRAINT c_fk FOREIGN KEY (note_id) REFERENCES notes (id) "
        "ON DELETE CASCADE;\n",
        "-- nothing here drops or truncates anything\nCREATE TABLE t (id int);\n",
        "INSERT INTO review_kinds (name) VALUES ('drop the stale ones');\n",
    ],
)
def test_additive_or_explained_migrations_pass(copy_of_migrations: Path, sql: str) -> None:
    n = _next_number(copy_of_migrations)
    _add_migration(copy_of_migrations, f"{n:04d}_change.sql", sql)
    assert destructive_problems(copy_of_migrations) == []
