"""Tests for `portolan.ingest.ledger` (spec sec 6.2, build item 6)."""

from __future__ import annotations

import threading
from datetime import date

from portolan import db as db_module
from portolan.ingest import ledger


def _row(db, connector: str, day: date) -> tuple:
    with db.cursor() as cur:
        cur.execute(
            "SELECT seen, filtered, extracted, partial, failed, sampled "
            "FROM coverage_ledger WHERE connector = %s AND day = %s",
            (connector, day),
        )
        return cur.fetchone()


def test_record_inserts_a_new_row(db):
    day = date(2026, 9, 1)
    ledger.record(db, "email", day, seen=1, extracted=1)

    row = _row(db, "email", day)
    assert row == (1, {}, 1, 0, 0, 0)


def test_record_accumulates_across_calls(db):
    day = date(2026, 9, 2)
    ledger.record(db, "email", day, seen=1, extracted=1)
    ledger.record(db, "email", day, seen=1, extracted=1)
    ledger.record(db, "email", day, seen=1, filtered={"excluded": 1})

    row = _row(db, "email", day)
    assert row == (3, {"excluded": 1}, 2, 0, 0, 0)


def test_record_merges_filtered_reasons_by_key(db):
    day = date(2026, 9, 3)
    ledger.record(db, "email", day, seen=1, filtered={"junk:auto_reply": 1})
    ledger.record(db, "email", day, seen=1, filtered={"junk:auto_reply": 1})
    ledger.record(db, "email", day, seen=1, filtered={"excluded": 1})
    ledger.record(db, "email", day, seen=1, filtered={"excluded_sampled": 1}, sampled=1)

    row = _row(db, "email", day)
    seen, filtered, _extracted, _partial, _failed, sampled = row
    assert seen == 4
    assert filtered == {"junk:auto_reply": 2, "excluded": 1, "excluded_sampled": 1}
    assert sampled == 1


def test_record_keeps_connectors_and_days_separate(db):
    ledger.record(db, "email", date(2026, 9, 4), seen=5)
    ledger.record(db, "call", date(2026, 9, 4), seen=3)
    ledger.record(db, "email", date(2026, 9, 5), seen=2)

    assert _row(db, "email", date(2026, 9, 4))[0] == 5
    assert _row(db, "call", date(2026, 9, 4))[0] == 3
    assert _row(db, "email", date(2026, 9, 5))[0] == 2


def test_ledger_counts_add_up_across_a_small_batch(db):
    # Simulate a small day's batch: 10 sources seen, 4 filtered (2 junk,
    # 1 excluded, 1 excluded_sampled), 5 extracted, 1 failed.
    day = date(2026, 9, 6)
    batch = [
        {"seen": 1, "filtered": {"junk:auto_reply": 1}},
        {"seen": 1, "filtered": {"junk:auto_reply": 1}},
        {"seen": 1, "filtered": {"excluded": 1}},
        {"seen": 1, "filtered": {"excluded_sampled": 1}, "sampled": 1},
        {"seen": 1, "extracted": 1},
        {"seen": 1, "extracted": 1},
        {"seen": 1, "extracted": 1},
        {"seen": 1, "extracted": 1},
        {"seen": 1, "extracted": 1},
        {"seen": 1, "failed": 1},
    ]
    for counts in batch:
        ledger.record(db, "email", day, **counts)

    row = _row(db, "email", day)
    seen, filtered, extracted, _partial, failed, sampled = row
    assert seen == 10
    assert extracted == 5
    assert failed == 1
    assert sampled == 1
    assert filtered == {"junk:auto_reply": 2, "excluded": 1, "excluded_sampled": 1}
    assert filtered["junk:auto_reply"] + filtered["excluded"] + filtered["excluded_sampled"] == 4


def test_summary_returns_rows_since_a_date_ordered(db):
    ledger.record(db, "email", date(2026, 8, 30), seen=1)
    ledger.record(db, "call", date(2026, 9, 1), seen=2)
    ledger.record(db, "email", date(2026, 9, 2), seen=3)

    rows = ledger.summary(db, since=date(2026, 9, 1))

    assert [(r["connector"], r["day"]) for r in rows] == [
        ("call", date(2026, 9, 1)),
        ("email", date(2026, 9, 2)),
    ]
    assert rows[0]["seen"] == 2


def test_summary_excludes_rows_before_since(db):
    ledger.record(db, "email", date(2026, 8, 1), seen=1)
    rows = ledger.summary(db, since=date(2026, 9, 1))
    assert all(r["day"] >= date(2026, 9, 1) for r in rows)


def test_record_concurrent_first_writes_both_land(db, db_url):
    # F5: two concurrent first-writes for a brand-new (connector, day) used
    # to both read no row (SELECT ... FOR UPDATE locks nothing that
    # doesn't exist yet), both INSERT, and the second's plain
    # `ON CONFLICT DO UPDATE SET seen = EXCLUDED.seen` overwrote the
    # first's count instead of adding to it -- one call's counts vanished.
    # Real concurrent connections against the fixed single-statement
    # upsert must both land.
    day = date(2026, 9, 20)
    connector = "email-concurrent"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _write(seen: int) -> None:
        conn = db_module.connect(db_url)
        try:
            conn.autocommit = True
            barrier.wait(timeout=5)
            ledger.record(conn, connector, day, seen=seen, filtered={"junk:auto_reply": seen})
        except BaseException as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=_write, args=(seen,)) for seen in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    row = _row(db, connector, day)
    seen, filtered, *_rest = row
    assert seen == 3  # 1 + 2, neither call's count lost
    assert filtered == {"junk:auto_reply": 3}
