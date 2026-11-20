"""The one way a test reaches a repo path the docker image does not ship.

`.dockerignore` keeps `scripts/`, `.claude/` and `.github/` out of the image, and CI's `docker`
job runs this suite inside it (`PORTOLAN_ROLE=test`). A test module that loads one of those paths
unguarded fails there at COLLECTION, which stops the whole job and every deploy behind it.

    from tests._paths import require_repo_path

    _SCRIPT = require_repo_path("scripts", "verify.py")  # module level: skips the module

`require_repo_path` returns the path, or skips the calling module at collection when the path is
absent (inside a test it skips that test). `needs_repo_path` is the marker form, for a module
where only some tests need the path. `tests/test_repo_paths_guarded.py` fails any test module
that builds a path into a directory `.dockerignore` excludes any other way, so the mistake is
caught in the local suite, before CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Used where .dockerignore or .gitignore cannot be read (inside the image). A checkout test
# (`test_the_fallback_matches_the_list_read_from_dockerignore`) keeps it equal to the list
# computed from the two files, so it cannot drift.
_FALLBACK_EXCLUDED_DIRS = (
    "02-reference-implementation",
    "03-talos-agent-kit",
    ".claude",
    "infra",
    "scripts",
    ".github",
)


def dockerignored_dirs(text: str) -> tuple[str, ...]:
    """Directory entries of a .dockerignore (lines ending in `/`), without the slash: the
    directories the image does not have. Comments, negations and file patterns are skipped."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")) or not line.endswith("/"):
            continue
        name = line.strip("/")
        if name and not any(ch in name for ch in "*?["):
            out.append(name)
    return tuple(dict.fromkeys(out))


def _read(name: str) -> str | None:
    try:
        return (REPO_ROOT / name).read_text(encoding="utf-8")
    except OSError:
        return None


def image_only_missing_dirs(dockerignore: str, gitignore: str) -> tuple[str, ...]:
    """Directories a git checkout has and the image does not: .dockerignore's directory
    entries, less the ones .gitignore also ignores (a checkout lacks those too, so a test that
    needs one fails everywhere, not only in the image) and `.git` itself."""
    ignored_by_git = set(dockerignored_dirs(gitignore))
    return tuple(
        d for d in dockerignored_dirs(dockerignore) if d not in ignored_by_git and d != ".git"
    )


def _image_excluded_dirs() -> tuple[str, ...]:
    # Inside the image .gitignore is absent (.dockerignore lists it): use the fallback there.
    dockerignore, gitignore = _read(".dockerignore"), _read(".gitignore")
    if dockerignore is None or gitignore is None:
        return _FALLBACK_EXCLUDED_DIRS
    return image_only_missing_dirs(dockerignore, gitignore) or _FALLBACK_EXCLUDED_DIRS


# Read from .dockerignore, so the list cannot drift from what the image leaves out.
IMAGE_EXCLUDED_DIRS = _image_excluded_dirs()


def _reason(parts: tuple[str, ...]) -> str:
    return (
        f"{'/'.join(parts)} is not in this checkout: .dockerignore keeps "
        f"{parts[0] if parts else ''}/ out of the image"
    )


def require_repo_path(*parts: str) -> Path:
    """`REPO_ROOT/<parts>`, or skip. Called at module level it skips the whole module during
    collection, before any module-level load can raise."""
    path = REPO_ROOT.joinpath(*parts)
    if not path.exists():
        pytest.skip(_reason(parts), allow_module_level=True)
    return path


def needs_repo_path(*parts: str) -> pytest.MarkDecorator:
    """A `skipif` marker for tests that need `REPO_ROOT/<parts>`."""
    return pytest.mark.skipif(not REPO_ROOT.joinpath(*parts).exists(), reason=_reason(parts))
