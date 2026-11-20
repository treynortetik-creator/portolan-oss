"""Deploy-safety scripts, run in CI on every release.

Spec: `01-spec/portolan-spec-v2.md` §11 ("Deployment and operations"):
"Snapshot every database before deploy. Deploy to the demo tenant first,
then a canary customer, then the rest. Post-deploy smoke checks: control
query passes, note and source counts did not fall, coverage ledger
consistent. Automatic rollback on a failed check. No destructive operation
runs without a two-step confirm and an audit row."

Each script here is one piece of that: `snapshot.py`, `smoke.py`,
`canary.py`, `rollback.py`. Every one takes `--dry-run` and prints its plan
instead of touching AWS or a database when passed. None of them are
exercised against a real AWS account by this wave's tests — there is no
account yet — but their argument parsing and pure logic (customer
ordering, count-drop detection, tag selection) are.
"""
