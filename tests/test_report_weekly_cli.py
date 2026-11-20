"""Tests for `portolan.report.weekly`'s CLI plumbing (e2e-run follow-up
wave): the `--llm` provider dispatch (item 1) and the `--clock` flag's
argparse wiring (item 4). Construction only, same discipline as
`tests/test_llm_factory.py` -- no `.complete()` call, so no network for
any provider, and no `db` fixture needed (building a client touches
neither Postgres nor the network; only calling it would).
"""

from __future__ import annotations

from pathlib import Path

from portolan.config import Settings
from portolan.llm.anthropic_api import AnthropicAPIClient
from portolan.llm.fake import FakeLLMClient
from portolan.llm.openrouter import OpenRouterClient
from portolan.report.weekly import _build_arg_parser, _build_llm

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "corpus" / "out"


def _settings(tmp_path: Path, **overrides) -> Settings:
    """Same helper as `test_llm_factory.py`'s: `_env_file=None` so this
    deployment's real `.env` never leaks into a test, only the keyword
    overrides given."""
    defaults = {"database_url": "postgresql://localhost/x", "store_root": tmp_path}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


# --------------------------------------------------------------------
# Item 1: `--llm` must accept openrouter/anthropic, and omitting it must
# mean "use this deployment's own LLM_PROVIDER", not silently fall back
# to `fake`.
# --------------------------------------------------------------------


def test_llm_flag_accepts_openrouter_and_anthropic() -> None:
    for provider in ("openrouter", "anthropic", "bedrock", "agent_sdk", "fake"):
        args = _build_arg_parser().parse_args(
            ["weekly", "--as-of", "2026-09-04", "--llm", provider]
        )
        assert args.llm == provider


def test_llm_flag_rejects_an_unknown_provider() -> None:
    import pytest

    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(
            ["weekly", "--as-of", "2026-09-04", "--llm", "not-a-real-provider"]
        )


def test_llm_flag_defaults_to_none_not_fake() -> None:
    args = _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04"])
    assert args.llm is None


def test_build_llm_none_goes_through_get_llm_with_settings_provider(tmp_path: Path) -> None:
    cfg = _settings(
        tmp_path,
        llm_provider="anthropic",
        extraction_model="claude-haiku-4-5",
        anthropic_api_key="sk-ant-test",
        tenant="acme",
    )
    client = _build_llm(None, CORPUS_ROOT, cfg)
    assert isinstance(client, AnthropicAPIClient)
    assert client.default_model == "claude-haiku-4-5"
    assert client.tenant == "acme"


def test_build_llm_explicit_openrouter_overrides_just_the_provider(monkeypatch, tmp_path: Path):
    """`--llm openrouter` must build an `OpenRouterClient` even when this
    deployment's own environment says a different provider -- while every
    OTHER setting (tenant, that provider's own API key, model id) still
    comes from the real environment, not a blank slate (`_build_llm`'s
    docstring)."""
    monkeypatch.setenv("LLM_PROVIDER", "agent_sdk")  # the deployment default, overridden below
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("PORTOLAN_TENANT", "acme")
    monkeypatch.setenv("EXTRACTION_MODEL", "some/model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("STORE_ROOT", str(tmp_path))

    from portolan.config import get_settings

    settings = get_settings()
    assert settings.llm_provider == "agent_sdk"  # sanity: the "omit --llm" case

    client = _build_llm("openrouter", CORPUS_ROOT, settings)
    assert isinstance(client, OpenRouterClient)
    assert client.default_model == "some/model"
    assert client.tenant == "acme"


def test_build_llm_fake_never_goes_through_get_llm(tmp_path: Path) -> None:
    """`fake` stays a CLI-only special case regardless of `settings.
    llm_provider` -- it replays the corpus's own planted facts, a concept
    `get_llm`/`Settings.llm_provider` has no notion of."""
    cfg = _settings(tmp_path, llm_provider="openrouter", tenant="acme")
    client = _build_llm("fake", CORPUS_ROOT, cfg)
    assert isinstance(client, FakeLLMClient)


# --------------------------------------------------------------------
# Item 4: `--clock record|world`.
# --------------------------------------------------------------------


def test_clock_flag_accepts_record_and_world() -> None:
    for value in ("record", "world"):
        args = _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04", "--clock", value])
        assert args.clock == value


def test_clock_flag_defaults_to_record() -> None:
    args = _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04"])
    assert args.clock == "record"


def test_clock_flag_rejects_an_unknown_value() -> None:
    import pytest

    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04", "--clock", "yesterday"])


# --------------------------------------------------------------------
# Nit 20: exactly one of --as-of/--week, not both possible but one
# silently discarded.
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Weekly-report wave 2: `--prose-model` (overrides STRONG_MODEL for
# every prose call this report makes) and the "provider/model print
# once" requirement.
# --------------------------------------------------------------------


def test_prose_model_flag_defaults_to_none() -> None:
    args = _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04"])
    assert args.prose_model is None


def test_prose_model_flag_is_accepted_verbatim() -> None:
    args = _build_arg_parser().parse_args(
        [
            "weekly",
            "--as-of",
            "2026-09-04",
            "--llm",
            "agent_sdk",
            "--prose-model",
            "claude-sonnet-5",
        ]
    )
    assert args.prose_model == "claude-sonnet-5"
    assert args.llm == "agent_sdk"


def test_week_alone_is_accepted_without_as_of() -> None:
    args = _build_arg_parser().parse_args(["weekly", "--week", "2026-W37"])
    assert args.week == "2026-W37"
    assert args.as_of is None


def test_as_of_alone_is_accepted_without_week() -> None:
    args = _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04"])
    assert args.as_of == "2026-09-04"
    assert args.week is None


def test_neither_as_of_nor_week_is_rejected() -> None:
    import pytest

    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["weekly"])


def test_both_as_of_and_week_is_rejected() -> None:
    import pytest

    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["weekly", "--as-of", "2026-09-04", "--week", "2026-W37"])


# --------------------------------------------------------------------
# Should-fix 11 (`01-spec/reviews/review-theme-fix-wave-3.md` finding
# 11): nothing pinned `weekly.main`'s `conn.autocommit = True` -- it was
# trivial to delete without a single test noticing, since every
# durability/atomicity test runs on the `db_conn` fixture, which sets
# autocommit itself.
# --------------------------------------------------------------------


class _StopAfterConnect(Exception):
    """Raised by the `db.connect` stub below so `weekly.main` never gets
    past opening its own connection -- this test is only about WHICH
    keyword argument reached `db.connect`, not about running a report."""


def test_weekly_main_opens_its_connection_with_autocommit(monkeypatch, tmp_path: Path) -> None:
    """Calls the real `weekly.main`, with `portolan.db.connect` replaced
    by a stub that records its own keyword arguments and then aborts --
    fails loudly if `main` ever goes back to the two-statement shape
    (`conn = db.connect(); conn.autocommit = True`) or otherwise stops
    asking for `autocommit=True`."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/portolan_test_autocommit_probe")
    monkeypatch.setenv("STORE_ROOT", str(tmp_path))

    calls: list[dict] = []

    def _stub_connect(url=None, **kwargs):
        calls.append(kwargs)
        raise _StopAfterConnect

    monkeypatch.setattr("portolan.db.connect", _stub_connect)

    import pytest

    from portolan.report import weekly

    with pytest.raises(_StopAfterConnect):
        weekly.main(["weekly", "--as-of", "2026-09-04", "--llm", "fake"])

    assert len(calls) == 1
    assert calls[0].get("autocommit") is True


def test_weekly_main_prints_provider_and_prose_model_before_connecting(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Task ("the provider and model print once"): `main` prints a
    `Provider: ... Prose model: ...` line before it ever opens a
    connection -- so a run that fails partway through still told the
    operator what it was using. `--prose-model` must win over `.env`'s
    own `STRONG_MODEL`/`EXTRACTION_MODEL` fallback."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/portolan_test_provider_probe")
    monkeypatch.setenv("STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("STRONG_MODEL", "some-other-strong-model")

    def _stub_connect(url=None, **kwargs):
        raise _StopAfterConnect

    monkeypatch.setattr("portolan.db.connect", _stub_connect)

    import pytest

    from portolan.report import weekly

    with pytest.raises(_StopAfterConnect):
        weekly.main(
            [
                "weekly",
                "--as-of",
                "2026-09-04",
                "--llm",
                "agent_sdk",
                "--prose-model",
                "claude-sonnet-5",
            ]
        )

    out = capsys.readouterr().out
    assert "Provider: agent_sdk" in out
    assert "Prose model: claude-sonnet-5" in out


def test_weekly_main_prose_model_falls_back_to_strong_model_when_flag_omitted(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/portolan_test_provider_probe")
    monkeypatch.setenv("STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("STRONG_MODEL", "the-deployments-strong-model")

    def _stub_connect(url=None, **kwargs):
        raise _StopAfterConnect

    monkeypatch.setattr("portolan.db.connect", _stub_connect)

    import pytest

    from portolan.report import weekly

    with pytest.raises(_StopAfterConnect):
        weekly.main(["weekly", "--as-of", "2026-09-04", "--llm", "openrouter"])

    out = capsys.readouterr().out
    assert "Provider: openrouter" in out
    assert "Prose model: the-deployments-strong-model" in out
