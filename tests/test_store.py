"""Tests for `portolan.store` (build-contracts.md "Store").

`LocalStore` is exercised fully (put/get/exists/url, atomic writes, key
safety). `S3Store` is, per the build instructions, "never exercised in
tests beyond construction" — these tests confirm the boto3 client is
genuinely lazy (never touched by `__init__`) and that key prefixing is
correct, without making any network call.
"""

from __future__ import annotations

import pytest

from portolan.store import LocalStore, S3Store, Store


def test_local_store_satisfies_store_protocol(tmp_path):
    store = LocalStore(tmp_path)
    assert isinstance(store, Store)


def test_local_store_put_get_roundtrip(tmp_path):
    store = LocalStore(tmp_path)
    store.put("originals/email/S-email-001", b"hello world", "text/plain")

    assert store.exists("originals/email/S-email-001") is True
    assert store.get("originals/email/S-email-001") == b"hello world"


def test_local_store_exists_false_for_missing_key(tmp_path):
    store = LocalStore(tmp_path)
    assert store.exists("originals/email/does-not-exist") is False


def test_local_store_get_missing_key_raises(tmp_path):
    store = LocalStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("nope")


def test_local_store_url_is_a_file_path_under_root(tmp_path):
    store = LocalStore(tmp_path)
    store.put("originals/call/S-call-001", b"transcript", "text/plain")

    url = store.url("originals/call/S-call-001")

    assert url == str(tmp_path / "originals" / "call" / "S-call-001")


def test_local_store_creates_parent_directories(tmp_path):
    store = LocalStore(tmp_path)
    store.put("a/b/c/d.txt", b"nested", "text/plain")
    assert (tmp_path / "a" / "b" / "c" / "d.txt").read_bytes() == b"nested"


def test_local_store_put_overwrites_cleanly(tmp_path):
    store = LocalStore(tmp_path)
    store.put("k", b"first", "text/plain")
    store.put("k", b"second version, longer than the first", "text/plain")
    assert store.get("k") == b"second version, longer than the first"


def test_local_store_put_leaves_no_temp_files_behind(tmp_path):
    store = LocalStore(tmp_path)
    store.put("originals/email/S-email-001", b"payload", "text/plain")

    leftovers = [p for p in tmp_path.rglob(".*") if p.is_file()]
    assert leftovers == []


@pytest.mark.parametrize(
    "bad_key",
    [
        "/etc/passwd",
        "../../etc/passwd",
        "originals/../../../etc/passwd",
        "",
    ],
)
def test_local_store_rejects_unsafe_keys(tmp_path, bad_key):
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError):
        store.put(bad_key, b"data", "text/plain")


def test_s3_store_satisfies_store_protocol():
    store = S3Store(bucket="portolan-dev", prefix="tenant-a")
    assert isinstance(store, Store)


def test_s3_store_construction_never_touches_boto3(monkeypatch):
    def _fail_if_called(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("boto3.client should not be called at construction time")

    monkeypatch.setattr("boto3.client", _fail_if_called)

    store = S3Store(bucket="portolan-dev", prefix="tenant-a")

    assert store.bucket == "portolan-dev"
    assert store.prefix == "tenant-a"
    assert store._client is None


def test_s3_store_strips_slashes_from_prefix():
    store = S3Store(bucket="portolan-dev", prefix="/tenant-a/")
    assert store.prefix == "tenant-a"


def test_s3_store_full_key_includes_prefix():
    store = S3Store(bucket="portolan-dev", prefix="tenant-a")
    assert store._full_key("originals/email/S-email-001") == "tenant-a/originals/email/S-email-001"


def test_s3_store_full_key_with_no_prefix():
    store = S3Store(bucket="portolan-dev")
    assert store._full_key("originals/email/S-email-001") == "originals/email/S-email-001"


def test_s3_store_client_is_lazily_constructed(monkeypatch):
    calls = []

    class _FakeClient:
        pass

    def _fake_client(service_name):
        calls.append(service_name)
        return _FakeClient()

    monkeypatch.setattr("boto3.client", _fake_client)

    store = S3Store(bucket="portolan-dev")
    assert calls == []  # not called yet

    client = store.client
    assert calls == ["s3"]
    assert isinstance(client, _FakeClient)

    # Second access reuses the same client rather than reconstructing it.
    assert store.client is client
    assert calls == ["s3"]
