"""Tests for the pure logic in `infra/deploy/*.py` (ownership: `infra/**`
and `tests/test_infra*.py`, `01-spec/build-contracts.md` "Infra").

These exercise customer ordering, count-drop detection, and tag selection
without touching AWS, a database, or a subprocess -- the parts of the
deploy-safety scripts that are meaningfully unit-testable without an
account to run them against.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INFRA_DIR = _REPO_ROOT / "infra"
_DEPLOY_DIR = _INFRA_DIR / "deploy"


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    """Load a module by its absolute file path rather than `sys.path` +
    a bare `import <name>`.

    Review finding F7: this file used to insert `infra/deploy` onto
    `sys.path` and then do `from app import CustomerSpec`, even though
    `app.py` lives in `infra/`, not `infra/deploy/` -- a bug that only
    ever passed when some *other* test module happened to import
    `canary` first, because `canary.py`'s own top-of-file
    `sys.path.insert` puts the *right* directory on the path as a side
    effect (`canary.py:38-41`). That made this file's own pass depend on
    pytest's collection order, not on anything this file did correctly.
    Loading by absolute path sidesteps `sys.path` and any other test
    module's import order entirely.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# infra/app.py, loaded first and by its own absolute path -- not via
# sys.path + `import app`. Registered in sys.modules under its own bare
# name too: canary.py does `from app import ...` at its own module scope
# (relying on `infra/` being on sys.path, which it arranges itself), and
# Python's import system checks sys.modules before touching sys.path at
# all, so this is enough for that import to resolve correctly regardless
# of what any other test module has already put on sys.path.
_app = _load_module_from_path("app", _INFRA_DIR / "app.py")
sys.modules.setdefault("app", _app)
if str(_INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(_INFRA_DIR))

_canary = _load_module_from_path("infra_deploy_canary", _DEPLOY_DIR / "canary.py")
_smoke = _load_module_from_path("infra_deploy_smoke", _DEPLOY_DIR / "smoke.py")

CustomerSpec = _app.CustomerSpec
load_customers = _app.load_customers
build_app = _app.build_app
_env_var_for = _canary._env_var_for
deploy_order = _canary.deploy_order
Counts = _smoke.Counts
compare = _smoke.compare

# -- canary.py: deploy_order ---------------------------------------------


def test_deploy_order_puts_demo_first() -> None:
    customers = [
        CustomerSpec(name="acme", tier="small", region="us-east-1"),
        CustomerSpec(name="demo", tier="small", region="us-east-1"),
        CustomerSpec(name="beta-corp", tier="large", region="us-west-2"),
    ]
    order = deploy_order(customers)
    assert order[0].name == "demo"


def test_deploy_order_second_is_first_remaining_customer_by_default() -> None:
    customers = [
        CustomerSpec(name="demo", tier="small", region="us-east-1"),
        CustomerSpec(name="acme", tier="small", region="us-east-1"),
        CustomerSpec(name="beta-corp", tier="large", region="us-west-2"),
    ]
    order = deploy_order(customers)
    assert [c.name for c in order] == ["demo", "acme", "beta-corp"]


def test_deploy_order_respects_canary_override() -> None:
    customers = [
        CustomerSpec(name="demo", tier="small", region="us-east-1"),
        CustomerSpec(name="acme", tier="small", region="us-east-1"),
        CustomerSpec(name="beta-corp", tier="large", region="us-west-2"),
    ]
    order = deploy_order(customers, canary_override="beta-corp")
    assert [c.name for c in order] == ["demo", "beta-corp", "acme"]


def test_deploy_order_without_a_demo_tenant() -> None:
    customers = [
        CustomerSpec(name="acme", tier="small", region="us-east-1"),
        CustomerSpec(name="beta-corp", tier="large", region="us-west-2"),
    ]
    order = deploy_order(customers)
    assert [c.name for c in order] == ["acme", "beta-corp"]


def test_deploy_order_every_customer_exactly_once() -> None:
    customers = [
        CustomerSpec(name="demo", tier="small", region="us-east-1"),
        CustomerSpec(name="acme", tier="small", region="us-east-1"),
        CustomerSpec(name="beta-corp", tier="large", region="us-west-2"),
    ]
    order = deploy_order(customers, canary_override="acme")
    assert sorted(c.name for c in order) == sorted(c.name for c in customers)
    assert len(order) == len(customers)


def test_env_var_for_slugifies_hyphens() -> None:
    assert _env_var_for("beta-corp", "PORTOLAN_DATABASE_URL") == "PORTOLAN_DATABASE_URL_BETA_CORP"


# -- app.py: load_customers parses an optional domain_name (F13) --------


def test_load_customers_parses_domain_name(tmp_path: Path) -> None:
    customers_file = tmp_path / "customers.yaml"
    customers_file.write_text(
        "- name: acme\n"
        "  tier: medium\n"
        "  region: us-east-1\n"
        "  domain_name: acme.customers.portolan.example\n"
    )
    specs = load_customers(customers_file)
    assert len(specs) == 1
    assert specs[0].domain_name == "acme.customers.portolan.example"


def test_load_customers_domain_name_defaults_to_none(tmp_path: Path) -> None:
    customers_file = tmp_path / "customers.yaml"
    customers_file.write_text("- name: demo\n  tier: dev\n  region: us-east-1\n")
    specs = load_customers(customers_file)
    assert specs[0].domain_name is None


# -- app.py: a real deploy can't reuse the immutable "latest" tag (F18) --


def test_build_app_refuses_latest_image_tag_outside_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding F18: the shared ECR repo's tags are immutable
    (`platform_stack.py`'s `image_tag_mutability=IMMUTABLE`), so a real
    deploy reusing the literal "latest" would fail at `docker push` after
    everything else already ran -- catching it here, at synth time,
    is much cheaper."""
    monkeypatch.delenv("PORTOLAN_IMAGE_TAG", raising=False)
    monkeypatch.setenv("PORTOLAN_PLATFORM_ACCOUNT_ID", "111111111111")

    customers_file = tmp_path / "customers.yaml"
    customers_file.write_text("- name: acme\n  tier: dev\n  region: us-east-1\n")

    with pytest.raises(RuntimeError, match="PORTOLAN_IMAGE_TAG"):
        build_app(dry=False, customers_file=customers_file)


def test_build_app_dry_run_tolerates_the_default_latest_tag(tmp_path: Path) -> None:
    customers_file = tmp_path / "customers.yaml"
    customers_file.write_text("- name: acme\n  tier: dev\n  region: us-east-1\n")
    app = build_app(dry=True, customers_file=customers_file)
    assert app is not None


# -- smoke.py: Counts / compare ------------------------------------------


def test_compare_passes_when_counts_hold_steady() -> None:
    before = Counts(control_ok=True, note_count=10, source_count=5)
    after = Counts(control_ok=True, note_count=10, source_count=5)
    assert compare(before, after) == []


def test_compare_passes_when_counts_grow() -> None:
    before = Counts(control_ok=True, note_count=10, source_count=5)
    after = Counts(control_ok=True, note_count=12, source_count=9)
    assert compare(before, after) == []


def test_compare_flags_a_note_count_drop() -> None:
    before = Counts(control_ok=True, note_count=10, source_count=5)
    after = Counts(control_ok=True, note_count=9, source_count=5)
    problems = compare(before, after)
    assert len(problems) == 1
    assert "note count dropped" in problems[0]


def test_compare_flags_a_source_count_drop() -> None:
    before = Counts(control_ok=True, note_count=10, source_count=5)
    after = Counts(control_ok=True, note_count=10, source_count=4)
    problems = compare(before, after)
    assert len(problems) == 1
    assert "source count dropped" in problems[0]


def test_compare_flags_a_broken_control_query() -> None:
    before = Counts(control_ok=True, note_count=10, source_count=5)
    after = Counts(control_ok=False, note_count=10, source_count=5)
    problems = compare(before, after)
    assert any("control query failed" in problem for problem in problems)


def test_compare_can_report_multiple_problems_at_once() -> None:
    before = Counts(control_ok=True, note_count=10, source_count=5)
    after = Counts(control_ok=False, note_count=1, source_count=1)
    assert len(compare(before, after)) == 3


def test_counts_json_round_trip() -> None:
    counts = Counts(control_ok=True, note_count=10, source_count=5)
    assert Counts.from_json(counts.to_json()) == counts


# -- canary.py: automatic rollback on a failed smoke check (review F14) --


def test_main_calls_rollback_automatically_on_failed_smoke_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Spec sec 11: "Automatic rollback on a failed check." This used to
    print a pointer at rollback.py and stop, requiring a human to run it
    by hand -- canary.main() now calls rollback.main() itself."""
    import types

    customers_file = tmp_path / "customers.yaml"
    customers_file.write_text("- name: acme\n  tier: small\n  region: us-east-1\n")

    def fake_run_stage(customer, *, image_tag, dry_run):
        return _canary.StageResult(customer=customer.name, deployed=True, smoke_passed=False)

    rollback_calls: list[list[str]] = []

    def fake_rollback_main(argv: list[str]) -> int:
        rollback_calls.append(argv)
        return 0

    fake_rollback_module = types.ModuleType("rollback")
    fake_rollback_module.main = fake_rollback_main

    monkeypatch.setattr(_canary, "run_stage", fake_run_stage)
    monkeypatch.setitem(sys.modules, "rollback", fake_rollback_module)

    returncode = _canary.main(
        ["--image-tag", "sha-bad", "--customers", str(customers_file), "--dry-run"]
    )

    assert returncode == 1
    assert rollback_calls == [["--customer", "acme", "--bad-tag", "sha-bad", "--dry-run"]]
