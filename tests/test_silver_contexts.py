"""Tests for `scripts/data/silver_contexts.py` (silver-labels build
instructions item 1: "write one short neutral context per corpus").

`scripts/` isn't a Python package (no `__init__.py`) and is excluded from
the container image, so this loads the module by file path and skips when
absent -- the same pattern `tests/test_set_owners.py` already uses for
`scripts/set_owners.py`.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from tests._paths import require_repo_path

_SCRIPT = require_repo_path("scripts", "data", "silver_contexts.py")


def _load_silver_contexts() -> ModuleType:
    spec = importlib.util.spec_from_file_location("silver_contexts_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_corpus_has_a_context() -> None:
    module = _load_silver_contexts()
    for corpus in ("fda_maude", "ami", "enron", "ubuntu_irc", "cfpb", "saas_sales"):
        assert corpus in module.CORPUS_CONTEXTS
        body = module.CORPUS_CONTEXTS[corpus]
        assert isinstance(body, str)
        assert len(body.strip()) > 0
        # not the production Halvard Medical body -- these corpora aren't
        # Halvard's own data
        assert "Halvard" not in body


def test_context_for_corpus_returns_body() -> None:
    module = _load_silver_contexts()
    assert module.context_for_corpus("ami") == module.CORPUS_CONTEXTS["ami"]


def test_context_for_corpus_unknown_raises_key_error() -> None:
    module = _load_silver_contexts()
    with pytest.raises(KeyError):
        module.context_for_corpus("not-a-real-corpus")


def test_context_bodies_stay_under_company_context_char_cap() -> None:
    """`portolan.context.COMPANY_CONTEXT_BODY_CHAR_CAP` caps a real
    tenant's own context note; these bodies aren't stored through that
    path, but staying under the same cap keeps them "short" the way the
    build instructions ask, not a second multi-page document."""
    from portolan.context import COMPANY_CONTEXT_BODY_CHAR_CAP

    module = _load_silver_contexts()
    for corpus, body in module.CORPUS_CONTEXTS.items():
        assert len(body) <= COMPANY_CONTEXT_BODY_CHAR_CAP, corpus
