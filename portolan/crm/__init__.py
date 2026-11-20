"""CRM snapshots and the changes between them, found by code
(`01-spec/monday-email-2026-09-25.md`).

- `portolan.crm.snapshots`: `record_snapshot`, `load_snapshots`, `snapshot_date`.
- `portolan.crm.changes`: `compute_changes`, `refresh_crm_changes`, `load_changes`.
- `portolan.crm.truth`: the CRM truth check, `truth_flags`.
- `portolan.crm.ingest`: `ingest_crm_snapshots`, an export taken in without a model.

`python -m portolan.crm ingest <dir>` and `python -m portolan.crm changes` run them.
"""
