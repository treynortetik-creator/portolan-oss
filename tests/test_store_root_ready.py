"""Tests for `portolan.store.assert_store_root_ready` (bug hunt
2026-09-18, P1 finding 5): a missing or mistyped volume path used to be
silent -- `LocalStore` just `mkdir(parents=True)`'d whatever path it was
given, so a volume that failed to mount on Railway meant originals landed
on the container's ephemeral disk with no error, no warning, and a clean
404 on every future lookup indistinguishable from normal missing data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from portolan.store import StoreRootError, assert_store_root_ready


def test_creates_a_missing_directory_outside_railway(tmp_path: Path) -> None:
    """Local dev / CI / tests: no RAILWAY_VOLUME_MOUNT_PATH means no
    volume is expected at all -- the lenient path, matching what every
    write already did implicitly before this finding's fix."""
    target = tmp_path / "not-yet-created" / "store"
    assert not target.exists()
    assert_store_root_ready(target)
    assert target.is_dir()


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root ignores directory permissions, so an unwritable root cannot be staged",
)
def test_raises_when_store_root_is_not_writable(tmp_path: Path) -> None:
    target = tmp_path / "readonly-store"
    target.mkdir()
    target.chmod(0o500)
    try:
        with pytest.raises(StoreRootError, match="not writable"):
            assert_store_root_ready(target)
    finally:
        target.chmod(0o700)  # so tmp_path cleanup can remove it


def test_raises_when_the_railway_volume_mount_path_does_not_exist(tmp_path: Path) -> None:
    """The exact silent-data-loss scenario the finding reproduced: the
    volume never mounted (RAILWAY_VOLUME_MOUNT_PATH points nowhere real),
    but the app used to boot anyway."""
    missing_mount = tmp_path / "data"
    store_root = missing_mount / "store"
    with pytest.raises(StoreRootError, match="did not mount"):
        assert_store_root_ready(store_root, railway_volume_mount_path=str(missing_mount))
    assert not store_root.exists()


def test_raises_when_store_root_is_outside_the_mounted_volume(tmp_path: Path) -> None:
    mount = tmp_path / "data"
    mount.mkdir()
    store_root_elsewhere = tmp_path / "ephemeral" / "store"
    with pytest.raises(StoreRootError, match="not on the mounted volume"):
        assert_store_root_ready(store_root_elsewhere, railway_volume_mount_path=str(mount))
    assert not store_root_elsewhere.exists()


def test_passes_when_store_root_is_under_the_mounted_volume(tmp_path: Path) -> None:
    mount = tmp_path / "data"
    mount.mkdir()
    store_root = mount / "store"
    assert_store_root_ready(store_root, railway_volume_mount_path=str(mount))
    assert store_root.is_dir()


def test_passes_when_store_root_is_exactly_the_mounted_volume(tmp_path: Path) -> None:
    mount = tmp_path / "data"
    mount.mkdir()
    assert_store_root_ready(mount, railway_volume_mount_path=str(mount))
