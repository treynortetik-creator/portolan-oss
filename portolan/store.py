"""Original-document storage (portolan-spec-v2.md sec 8, build-contracts.md).

Every original a connector hands to `intake()` (a call transcript, a logged
email, a ticket export, a CRM row, an RSS item) is stored here under a
stable key, versioned nowhere yet (S3 versioning is a deploy-time bucket
setting, not something this class does) but never overwritten in place —
`intake()` is the only writer and it only writes once per content hash.

Two implementations, one `Store` protocol:

- `LocalStore` — a directory on disk, standing in for S3 in local dev and
  tests. Keys map directly to files under `root`. Writes are atomic (write
  to a temp file in the same directory, then `os.replace`) so a crash
  mid-write never leaves a half-written original behind. `url()` returns
  the absolute file path.
- `S3Store` — the real thing. The boto3 client is constructed lazily (only
  on first use) so importing this module, or constructing an `S3Store`,
  never requires AWS credentials or network access — the build
  instructions call for this explicitly: "never exercised in tests beyond
  construction." `url()` returns a presigned GET URL.

`portolan/lint/_store.py` already depends on this exact shape (it carries
a structural fallback Protocol until this module exists); nothing else
should redefine `Store`.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class StoreRootError(RuntimeError):
    """Raised by `assert_store_root_ready` when `store_root` cannot be
    trusted to persist originals -- startup should abort rather than
    silently write to ephemeral disk (bug hunt 2026-09-18, P1 finding 5).
    """


def assert_store_root_ready(
    store_root: Path | str, *, railway_volume_mount_path: str | None = None
) -> None:
    """Fail loudly, before the app accepts a single request, if
    `store_root` cannot be trusted to persist originals.

    Bug hunt 2026-09-18, P1 finding 5: `LocalStore` used to `mkdir
    (parents=True)` whatever path it was given, with no check at
    startup -- if `/data` ever failed to mount on Railway, the app
    booted green, new originals landed on the container's ephemeral
    disk, and every one of them vanished on the next deploy, with every
    provenance link returning an indistinguishable "not found."

    Outside Railway (`railway_volume_mount_path` is `None` -- local dev,
    CI, tests) this only ensures the directory exists and is writable,
    creating it if needed: the same thing every write path already did
    implicitly (`LocalStore.put`, `get_llm()`'s usage sink), just done
    once, up front, with a clear error instead of a write failing deep
    inside a request.

    On Railway (`RAILWAY_VOLUME_MOUNT_PATH` set -- Railway's own
    convention, read from the environment by the caller) `store_root`
    must already exist AND sit on or under that mount path. A missing
    mount path means the volume failed to mount at all; a `store_root`
    outside it means originals would land on the container's own disk
    even though a volume is attached -- both are the exact silent
    data-loss failure this finding reproduced.
    """
    root = Path(store_root).resolve()

    if railway_volume_mount_path:
        mount = Path(railway_volume_mount_path).resolve()
        if not mount.is_dir():
            raise StoreRootError(
                f"RAILWAY_VOLUME_MOUNT_PATH={mount} is set but does not exist or is not a "
                "directory -- the volume did not mount. Refusing to start: originals "
                "written now would land on ephemeral container disk and vanish on the "
                "next deploy."
            )
        if root != mount and mount not in root.parents:
            raise StoreRootError(
                f"STORE_ROOT={root} is not on the mounted volume "
                f"(RAILWAY_VOLUME_MOUNT_PATH={mount}). Originals written here would not "
                "survive the next deploy -- set STORE_ROOT to a path under the volume mount."
            )

    root.mkdir(parents=True, exist_ok=True)
    probe = root / f".portolan-store-write-check.{os.getpid()}"
    try:
        probe.write_bytes(b"")
    except OSError as exc:
        raise StoreRootError(f"STORE_ROOT={root} is not writable: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)


@runtime_checkable
class Store(Protocol):
    """What every store implementation offers. See build-contracts.md
    "Store: portolan/store.py (wave A2)"."""

    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def url(self, key: str) -> str:
        """A local path (`LocalStore`) or a signed URL (`S3Store`)."""
        ...


def _validate_key(key: str) -> None:
    """Reject keys that could escape the store root or an S3 prefix.

    An absolute key, or one with a `..` path segment, would let a caller
    read or write outside the intended namespace (`LocalStore.root`, or
    the wrong customer's originals under a shared `S3Store` prefix). Every
    real key in this codebase looks like `originals/email/S-email-061`, so
    this is a defensive check, not a feature anyone should rely on.
    """
    if not key or key.startswith("/") or key.startswith("\\"):
        raise ValueError(f"invalid store key (must be relative): {key!r}")
    if ".." in Path(key).parts:
        raise ValueError(f"invalid store key (must not contain '..'): {key!r}")


class LocalStore:
    """Files under `root`, standing in for S3 in local dev and tests
    (build-contracts.md: "local file store standing in for S3", also
    `Settings.store_root` in the shared config contract).
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        _validate_key(key)
        return self.root / key

    def put(self, key: str, data: bytes, content_type: str) -> None:
        # content_type is part of the Store protocol (S3 needs it as an
        # object header) but a plain filesystem has nowhere natural to put
        # it without a sidecar file nobody reads yet, so it is accepted
        # and intentionally unused here.
        del content_type
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_name)
            raise

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def url(self, key: str) -> str:
        return str(self._path(key))


class S3Store:
    """The real store. boto3 client is lazy: constructing an `S3Store`, or
    importing this module, never touches the network or requires
    credentials — only calling `put`/`get`/`exists`/`url` does.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _full_key(self, key: str) -> str:
        _validate_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=data,
            ContentType=content_type,
        )

    def get(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full_key(key))
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def url(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._full_key(key)},
            ExpiresIn=3600,
        )
