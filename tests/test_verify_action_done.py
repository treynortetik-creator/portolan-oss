"""`scripts/verify.py action-done` against a real app over HTTP: it fails
while `/review` has no "Proposed done" card and passes once the presence
check (`portolan.report.action_done`, stub judge) has queued one. The
graph is `tests/test_action_done.py`'s fixture: one open action and Sam
Okafor's follow-up email the next day.
"""

from __future__ import annotations

import importlib.util
import json
import secrets
import sys
from pathlib import Path

from portolan.registry import Registry
from portolan.web.review_presenter import category_label_for
from tests._paths import require_repo_path
from tests.merge_helpers import loaded_registry_fixture  # noqa: F401
from tests.test_action_done import StubJudge, _follow_up, _open_action, _propose
from tests.verify_seed import app_settings, serve, write_keys

_VERIFY = require_repo_path("scripts", "verify.py")


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_action_done", _VERIFY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_action_done"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verify()


def _run(base_url: str, keys: tuple[Path, Path], out_dir: Path) -> tuple[int, dict]:
    code = verify.main(
        [
            "action-done",
            "--base-url",
            base_url,
            "--org-key-file",
            str(keys[0]),
            "--mcp-key-file",
            str(keys[1]),
            "--out-dir",
            str(out_dir),
        ]
    )
    [path] = out_dir.glob("*-action-done.json")
    return code, json.loads(path.read_text(encoding="utf-8"))


def _markers(result: dict) -> dict[str, int]:
    check = next(c for c in result["checks"] if c["name"] == "proposed-done cards on /review")
    return {m["name"]: m["found"] for m in check["markers"]}


def test_the_category_matches_the_presenter_and_all_skips_the_target() -> None:
    assert verify.ACTION_DONE_CATEGORY == category_label_for("action_done", None)
    assert "action-done" in verify.OPT_IN_TARGETS
    assert "action-done" not in verify.TARGETS


def test_action_done_fails_without_a_proposal_and_passes_with_one(
    db, db_url: str, loaded_registry: Registry, tmp_path: Path
) -> None:
    _open_action(db, loaded_registry)
    _follow_up(db, loaded_registry)
    org_key, mcp_key, org_path, mcp_path = write_keys(tmp_path / "keys")
    store_root = tmp_path / "store"
    store_root.mkdir()
    settings = app_settings(
        db_url,
        store_root,
        org_key=org_key,
        session_secret=secrets.token_hex(32),
        mcp_org_key=mcp_key,
    )

    with serve(settings) as base_url:
        before_code, before = _run(base_url, (org_path, mcp_path), tmp_path / "before")
        [outcome] = _propose(db, StubJudge())
        assert outcome.status == "proposed"
        after_code, after = _run(base_url, (org_path, mcp_path), tmp_path / "after")

    assert before_code == 1
    assert _markers(before)["proposed-done cards"] == 0
    assert after_code == 0, [c for c in after["checks"] if not c["ok"]]
    assert _markers(after) == {
        "proposed-done cards": 1,
        "cards with the answering claim and its original": 1,
        "cards linking the action they would close": 1,
    }
